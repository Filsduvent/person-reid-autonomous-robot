from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

from edge_reid_runtime.core.types import RunConfig
from edge_reid_runtime.detectors import NullDetector, YoloV8Config, YoloV8PersonDetector
from edge_reid_runtime.sources.factory import create_source
from edge_reid_runtime.utils.log import setup_logger, JsonlWriter
from edge_reid_runtime.utils.profiler import StageProfiler
from edge_reid_runtime.utils.video_io import VideoWriter
from edge_reid_runtime.visualization import draw_detections

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


VALID_SOURCES = ("webcam", "video", "robot")
VALID_DEVICES = ("cpu", "cuda", "auto")
VALID_DETECTORS = ("yolov8", "null")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="edge_reid",
        description="EDGE ReID real-time pipeline (Phase 1+2 scaffold).",
    )
    p.add_argument("--source", required=True, choices=VALID_SOURCES,
                   help="Input source type.")
    p.add_argument("--device", default="auto", choices=VALID_DEVICES,
                   help="Compute device selection (placeholder for later).")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write logs/artifacts.")
    p.add_argument("--video_path", default=None,
                   help="Path to video file (required if --source=video).")
    p.add_argument("--webcam_index", type=int, default=0,
                   help="Webcam index (if --source=webcam).")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited).")
    p.add_argument("--print_every", type=int, default=10,
                   help="Print stats every N frames.")
    p.add_argument("--reid_backbone", default=None,
                   help="ReID backbone name (e.g., osnet_x0_25, mobilenetv3).")
    p.add_argument("--weights", default=None,
                   help="Path or URL to model weights (optional for now).")
    p.add_argument("--detector", default="yolov8", choices=VALID_DETECTORS,
                   help="Detector backend (yolo or null).")
    p.add_argument("--yolo_model", default="yolov8n.pt",
                   help="Ultralytics YOLO model path/name (e.g., yolov8n.pt).")
    p.add_argument("--det_conf", type=float, default=0.35,
                   help="Detector confidence threshold.")
    p.add_argument("--det_iou", type=float, default=0.70,
                   help="Detector NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO inference image size.")
    p.add_argument("--max_det", type=int, default=100,
                   help="Maximum detections per frame.")
    p.add_argument("--save_video", action="store_true",
                   help="Save visualization video to output_dir.")
    p.add_argument("--display", action="store_true",
                   help="If set, show a live window (requires OpenCV). Press 'q' to quit.")
    p.add_argument("--output_video", default=None,
                   help="Optional output video path (default: output_dir/detections.mp4).")
    return p.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    # Ensure output dir is writable
    out.mkdir(parents=True, exist_ok=True)
    testfile = out / ".write_test"
    try:
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(f"output_dir not writable: {out}. Error: {e}")

    if args.source == "video" and not args.video_path:
        raise ValueError("--video_path is required when --source=video")
    if args.source != "video" and args.video_path:
        print("Warning: --video_path is ignored unless --source=video", file=sys.stderr)


def build_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        source=args.source,
        device=args.device,
        output_dir=Path(args.output_dir),
        video_path=args.video_path if args.video_path else None,
        webcam_index=args.webcam_index,
        max_frames=args.max_frames,
        print_every=args.print_every,
        reid_backbone=args.reid_backbone,
        weights=args.weights if args.weights else None,
        detector=args.detector,
        yolo_model=args.yolo_model,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        save_video=args.save_video,
        display=args.display,
        output_video=Path(args.output_video) if args.output_video else None,
    )


def resolve_device(device_flag: str) -> str:
    """
    Returns device string for Ultralytics:
    - "cpu"
    - "cuda" (or "0" etc.)
    """
    if device_flag == "cpu":
        return "cpu"
    if device_flag == "cuda":
        return "cuda"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_minimal_loop(cfg: RunConfig) -> int:
    logger = setup_logger("edge_reid", cfg.output_dir)
    logger.info(
        "Starting EDGE pipeline scaffold (Phase 3.2+3.3: null detector/tracker + stage profiling)."
    )
    logger.info(
        f"source={cfg.source} device={cfg.device} detector={cfg.detector} output_dir={cfg.output_dir}"
    )

    profiler = StageProfiler(fps_window=30)
    device_resolved = resolve_device(cfg.device)
    logger.info(f"resolved_device={device_resolved}")

    detector_name = getattr(cfg, "detector", "yolov8")
    if detector_name == "null":
        detector = NullDetector(mode="empty")
        cls_name_fn = None
    else:
        try:
            if "/" in cfg.yolo_model or cfg.yolo_model.endswith(".pt"):
                model_path = Path(cfg.yolo_model)
                if not model_path.exists():
                    logger.warning(
                        f"YOLO model not found at {model_path}; Ultralytics may download it."
                    )
            ycfg = YoloV8Config(
                model=cfg.yolo_model,
                conf=cfg.det_conf,
                iou=cfg.det_iou,
                imgsz=cfg.imgsz,
                max_det=cfg.max_det,
                half=(device_resolved != "cpu"),
            )
            detector = YoloV8PersonDetector(device=device_resolved, cfg=ycfg)
            cls_name_fn = detector.cls_name
        except Exception as e:
            logger.error(f"Failed to create detector: {e}")
            return 2

    src = None
    jsonl = None
    writer = None
    try:
        src = create_source(cfg)
    except Exception as e:
        logger.error(f"Failed to create source: {e}")
        return 2

    jsonl = JsonlWriter(cfg.output_dir / "frames.jsonl")
    output_path = cfg.output_video or (cfg.output_dir / "detections.mp4")

    frames = 0
    t_start = time.time()

    try:
        it = iter(src)
        while True:
            profiler.on_frame_start()
            try:
                frame = next(it)
            except StopIteration:
                break

            with profiler.stage("detector"):
                detections = detector.detect(frame)

            annotated = frame.image
            if annotated is not None and (cfg.save_video or cfg.display):
                annotated = annotated.copy()
                draw_detections(annotated, detections, cls_name_fn=cls_name_fn)

            with profiler.stage("video_write"):
                if cfg.save_video and annotated is not None:
                    if writer is None:
                        fps = frame.meta.get("fps", 30.0) if isinstance(frame.meta, dict) else 30.0
                        writer = VideoWriter(output_path, fps=float(fps))
                    writer.write(annotated)

            if cfg.display and annotated is not None:
                if cv2 is None:
                    logger.warning("--display requested but OpenCV not installed.")
                else:
                    cv2.imshow("EDGE-ReID: detections", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            stats = profiler.on_frame_end(
                frame_id=frame.frame_id,
                timestamp_s=frame.timestamp_s,
            )

            record: Dict[str, Any] = {
                "frame_id": stats.frame_id,
                "timestamp_s": stats.timestamp_s,
                "dt_ms": stats.dt_ms,
                "fps_rolling": stats.fps_rolling,
                "rss_mb": stats.rss_mb,
                "source": frame.meta.get("source"),
                "num_det": len(detections),
                "stages_ms": stats.stages_ms,
            }
            jsonl.write(record)

            frames += 1
            if cfg.print_every > 0 and (frames % cfg.print_every == 0):
                st = stats.stages_ms
                msg = (
                    f"frame={stats.frame_id} total={stats.dt_ms:.2f}ms "
                    f"det={st.get('detector', 0.0):.2f}ms "
                    f"fps(roll)={stats.fps_rolling:.2f} dets={len(detections)}"
                )
                if stats.rss_mb is not None:
                    msg += f" rss={stats.rss_mb:.1f}MB"
                logger.info(msg)

        t_total = time.time() - t_start
        fps_avg = frames / t_total if t_total > 0 else 0.0
        logger.info(f"Finished. frames={frames} total_s={t_total:.2f} fps_avg={fps_avg:.2f}")
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        return 130
    except Exception as e:
        logger.exception(f"Runtime error: {e}")
        return 1
    finally:
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if cfg.display and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if jsonl is not None:
            try:
                jsonl.close()
            except Exception:
                pass


def smoke_test_no_op() -> int:
    """
    Tiny smoke-test with robot stub (no OpenCV dependency).
    """
    tmp = Path("outputs") / "smoke_test"
    args = ["--source", "robot", "--device", "cpu", "--output_dir", str(tmp), "--max_frames", "30", "--print_every", "10"]
    ns = parse_args(args)
    validate_args(ns)
    cfg = build_config(ns)
    return run_minimal_loop(cfg)


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)
    cfg = build_config(args)
    code = run_minimal_loop(cfg)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
