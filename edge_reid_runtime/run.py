from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from edge_reid_runtime.core.types import RunConfig
from edge_reid_runtime.core.interfaces import TrackEmbedding
from edge_reid_runtime.detectors import NullDetector, YoloV8Config, YoloV8PersonDetector
from edge_reid_runtime.embedders import create_embedder, EmbedderConfig
from edge_reid_runtime.embedders.cropper_extraction import extract_track_crops
from edge_reid_runtime.gallery import GalleryConfig, GalleryManager, AssignerConfig, IdentityAssigner
from edge_reid_runtime.sources.factory import create_source
from edge_reid_runtime.trackers import DeepSortConfig, DeepSortRealtimeTracker, NullTracker
from edge_reid_runtime.utils.log import setup_logger, JsonlWriter
from edge_reid_runtime.utils.profiler import StageProfiler
from edge_reid_runtime.utils.video_io import VideoWriter
from edge_reid_runtime.visualization import draw_detections
from edge_reid_runtime.visualization.draw_tracks import draw_tracks

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


VALID_SOURCES = ("webcam", "video", "robot")
VALID_DEVICES = ("cpu", "cuda", "auto")
VALID_DETECTORS = ("yolov8", "null")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edge_reid",
        description="EDGE ReID real-time pipeline (Phase 1+2 scaffold).",
    )
    p.add_argument("--config", default=None,
                   help="Path to YAML config file.")
    p.add_argument("--source", default=None, choices=VALID_SOURCES,
                   help="Input source type.")
    p.add_argument("--device", default="auto", choices=VALID_DEVICES,
                   help="Compute device selection (placeholder for later).")
    p.add_argument("--output_dir", default=None,
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
                   help="ReID backbone (e.g., osnet_x0_25, osnet_x0_5, mobilenetv3_small, "
                        "mobilenetv3_large, shufflenetv2_x0_5, shufflenetv2_x1_0, efficientnet_lite0).")
    p.add_argument("--weights", default=None,
                   help="Path or URL to model weights (optional for now).")
    p.add_argument("--embedder_backend", default="torch",
                   choices=("torch", "onnx"),
                   help="Embedder backend: torch or onnx.")
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
    p.add_argument("--max_age", type=int, default=30,
                   help="DeepSORT: frames to keep lost tracks before deletion.")
    p.add_argument("--n_init", type=int, default=3,
                   help="DeepSORT: hits before a track is confirmed.")
    p.add_argument("--max_iou_distance", type=float, default=0.7,
                   help="DeepSORT: association gate (IoU distance).")
    p.add_argument("--known_threshold", type=float, default=0.55,
                   help="Gallery: similarity >= known_threshold => Known.")
    p.add_argument("--unknown_threshold", type=float, default=0.45,
                   help="Gallery: below this similarity => Unknown.")
    p.add_argument("--max_identities", type=int, default=500,
                   help="Gallery: maximum identities to store.")
    p.add_argument("--id_prefix", type=str, default="person_",
                   help="Gallery: identity id prefix.")
    p.add_argument("--id_width", type=int, default=4,
                   help="Gallery: identity id zero-pad width.")
    p.add_argument("--update_threshold", type=float, default=0.65,
                   help="Gallery: minimum similarity to update prototype.")
    p.add_argument("--margin_threshold", type=float, default=0.15,
                   help="Gallery: minimum top1-top2 gap to update.")
    p.add_argument("--stable_age", type=int, default=10,
                   help="Gallery: minimum track age for enrollment/update.")
    p.add_argument("--stable_hits", type=int, default=5,
                   help="Gallery: minimum hits for enrollment/update.")
    p.add_argument("--min_det_conf", type=float, default=0.35,
                   help="Gallery: minimum detection confidence to update.")
    p.add_argument("--area_drop_ratio", type=float, default=0.5,
                   help="Gallery: area drop ratio to block updates (occlusion).")
    p.add_argument("--aspect_ratio_min", type=float, default=0.2,
                   help="Gallery: min aspect ratio to allow update.")
    p.add_argument("--aspect_ratio_max", type=float, default=0.9,
                   help="Gallery: max aspect ratio to allow update.")
    p.add_argument("--ema_alpha", type=float, default=0.1,
                   help="Gallery: EMA update rate for prototype.")
    p.add_argument("--reacquire_cooldown_frames", type=int, default=15,
                   help="Gallery: cooldown frames after track reacquired.")
    p.add_argument("--gallery_path", type=str, default=None,
                   help="Path to save/load gallery state (JSON).")
    p.add_argument("--reset_gallery", action="store_true",
                   help="If set, ignore any existing gallery file and start fresh.")
    p.add_argument("--save_video", action="store_true",
                   help="Save visualization video to output_dir.")
    p.add_argument("--display", action="store_true",
                   help="If set, show a live window (requires OpenCV). Press 'q' to quit.")
    p.add_argument("--output_video", default=None,
                   help="Optional output video path (default: output_dir/detections.mp4).")
    return p


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is not installed. Install with: pip install pyyaml")
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file must be a YAML mapping at the top level.")
    return data


def _apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    for key, value in cfg.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.config:
        cfg = _load_yaml_config(args.config)
        _apply_config_defaults(args, parser, cfg)
    if args.source is None:
        parser.error("--source is required (or provide it in --config)")
    if args.output_dir is None:
        parser.error("--output_dir is required (or provide it in --config)")
    return args


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
        embedder_backend=args.embedder_backend,
        detector=args.detector,
        yolo_model=args.yolo_model,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=args.max_iou_distance,
        save_video=args.save_video,
        display=args.display,
        output_video=Path(args.output_video) if args.output_video else None,
        unknown_threshold=args.unknown_threshold,
        known_threshold=args.known_threshold,
        max_identities=args.max_identities,
        id_prefix=args.id_prefix,
        id_width=args.id_width,
        update_threshold=args.update_threshold,
        margin_threshold=args.margin_threshold,
        stable_age=args.stable_age,
        stable_hits=args.stable_hits,
        min_det_conf=args.min_det_conf,
        area_drop_ratio=args.area_drop_ratio,
        aspect_ratio_min=args.aspect_ratio_min,
        aspect_ratio_max=args.aspect_ratio_max,
        ema_alpha=args.ema_alpha,
        reacquire_cooldown_frames=args.reacquire_cooldown_frames,
        gallery_path=Path(args.gallery_path) if args.gallery_path else None,
        reset_gallery=args.reset_gallery,
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
        "Starting EDGE pipeline scaffold (Phase 3.2+3.3: detector/tracker + stage profiling)."
    )
    logger.info(
        f"source={cfg.source} device={cfg.device} detector={cfg.detector} output_dir={cfg.output_dir}"
    )

    # Save the resolved config for reproducibility
    try:
        if yaml is not None:
            cfg_dict = asdict(cfg)
            for k, v in list(cfg_dict.items()):
                if isinstance(v, Path):
                    cfg_dict[k] = str(v)
            (cfg.output_dir / "config_used.yaml").write_text(
                yaml.safe_dump(cfg_dict, sort_keys=False),
                encoding="utf-8",
            )
    except Exception as e:
        logger.warning(f"Failed to write config_used.yaml: {e}")

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

    try:
        ts_cfg = DeepSortConfig(
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            max_iou_distance=cfg.max_iou_distance,
            nms_max_overlap=1.0,
            embedder="mobilenet",
            embedder_gpu=False,
        )
        tracker = DeepSortRealtimeTracker(cfg=ts_cfg)
        tracker_name = "deepsort"
    except Exception as e:
        logger.warning(f"Failed to init DeepSortRealtimeTracker: {e}; falling back to NullTracker.")
        tracker = NullTracker()
        tracker_name = "null"
    logger.info(
        f"tracker={tracker_name} max_age={cfg.max_age} n_init={cfg.n_init} max_iou_distance={cfg.max_iou_distance}"
    )

    embedder = None
    embedder_name = None
    if cfg.reid_backbone:
        try:
            embedder_cfg = EmbedderConfig(
                backbone=cfg.reid_backbone,
                device=device_resolved,
                weights=str(cfg.weights) if cfg.weights else None,
                backend=cfg.embedder_backend,
            )
            embedder = create_embedder(embedder_cfg)
            embedder_name = cfg.reid_backbone
            logger.info(
                f"embedder={embedder_name} input_size={embedder.cfg.input_size} dim={embedder.embedding_dim}"
            )
        except Exception as e:
            logger.warning(f"Failed to init embedder '{cfg.reid_backbone}': {e}. Embedding disabled.")

    gallery_cfg = GalleryConfig(
        known_threshold=cfg.known_threshold,
        unknown_threshold=cfg.unknown_threshold,
        ema_alpha=cfg.ema_alpha,
        max_identities=cfg.max_identities,
        id_prefix=cfg.id_prefix,
        id_width=cfg.id_width,
    )
    gallery = GalleryManager(cfg=gallery_cfg)
    if cfg.gallery_path and not cfg.reset_gallery:
        try:
            gallery.load(cfg.gallery_path)
            logger.info(f"gallery_loaded path={cfg.gallery_path} size={len(gallery)}")
        except Exception as e:
            logger.warning(f"Failed to load gallery from {cfg.gallery_path}: {e}")
    assigner_cfg = AssignerConfig(
        stable_age=cfg.stable_age,
        stable_hits=cfg.stable_hits,
        unknown_threshold=cfg.unknown_threshold,
        known_threshold=cfg.known_threshold,
        update_threshold=cfg.update_threshold,
        margin_threshold=cfg.margin_threshold,
        min_det_conf=cfg.min_det_conf,
        area_drop_ratio=cfg.area_drop_ratio,
        aspect_ratio_min=cfg.aspect_ratio_min,
        aspect_ratio_max=cfg.aspect_ratio_max,
        reacquire_cooldown_frames=cfg.reacquire_cooldown_frames,
    )
    assigner = IdentityAssigner(gallery, cfg=assigner_cfg)

    src = None
    jsonl = None
    emb_jsonl = None
    id_jsonl = None
    writer = None
    try:
        src = create_source(cfg)
    except Exception as e:
        logger.error(f"Failed to create source: {e}")
        return 2

    jsonl = JsonlWriter(cfg.output_dir / "frames.jsonl")
    emb_jsonl = JsonlWriter(cfg.output_dir / "embeddings.jsonl")
    id_jsonl = JsonlWriter(cfg.output_dir / "identities.jsonl")
    output_path = cfg.output_video or (cfg.output_dir / "detections.mp4")

    frames = 0
    prev_ids = set()
    t_start = time.time()

    try:
        it = iter(src)
        while True:
            profiler.on_frame_start()
            try:
                with profiler.stage("input"):
                    frame = next(it)
            except StopIteration:
                break

            with profiler.stage("detector"):
                detections = detector.detect(frame)

            with profiler.stage("tracker"):
                tracks = tracker.update(frame, detections)

            active_ids = {t.track_id for t in tracks}
            created_ids = sorted(list(active_ids - prev_ids))
            deleted_ids = sorted(list(prev_ids - active_ids))
            prev_ids = active_ids
            assigner.mark_missed_tracks(frame.frame_id, active_ids)

            embeddings: list[TrackEmbedding] = []
            num_crops = 0
            embed_dim = 0
            if embedder is not None and frame.image is not None and tracks:
                with profiler.stage("embedder"):
                    crops, crop_ids = extract_track_crops(frame.image, tracks)
                    num_crops = len(crops)
                    if crops:
                        feats = embedder.embed(crops)
                        embed_dim = int(feats.shape[1]) if feats.ndim == 2 else 0
                        embeddings = [
                            TrackEmbedding(
                                track_id=int(crop_ids[i]),
                                embedding=feats[i],
                                embedding_dim=embed_dim,
                                quality=None,
                            )
                            for i in range(len(crop_ids))
                        ]

            assignments = []
            identity_info: Dict[int, Dict[str, object]] = {}
            if embeddings:
                with profiler.stage("gallery"):
                    emb_map = {emb.track_id: emb for emb in embeddings}
                    for tr in tracks:
                        emb = emb_map.get(int(tr.track_id))
                        if emb is None:
                            continue
                        assign = assigner.assign(frame.frame_id, tr, emb.embedding, frame.timestamp_s)
                        assignments.append(assign)
                        status = "ENROLL" if assign.enrolled else ("UPDATE" if assign.updated else "")
                        if assign.skip_reason:
                            status = f"NO-UPDATE:{assign.skip_reason}"
                        identity_info[int(tr.track_id)] = {
                            "identity_id": assign.identity_id,
                            "score": assign.score,
                            "status": status,
                        }

            annotated = frame.image
            if annotated is not None and (cfg.save_video or cfg.display):
                with profiler.stage("visualization"):
                    annotated = annotated.copy()
                    draw_detections(annotated, detections, cls_name_fn=cls_name_fn)
                    draw_tracks(annotated, tracks, identities=identity_info)
                    if embedder is not None and cv2 is not None:
                        label = f"embeddings: {num_crops} | dim: {embed_dim}"
                        cv2.putText(
                            annotated,
                            label,
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

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
                "vram_mb": stats.vram_mb,
                "source": frame.meta.get("source"),
                "num_det": len(detections),
                "num_tracks": len(tracks),
                "tracks_active_ids": sorted(list(active_ids)),
                "tracks_created_ids": created_ids,
                "tracks_deleted_ids": deleted_ids,
                "num_crops": num_crops,
                "embed_dim": embed_dim,
                "num_assignments": len(assignments),
                "gallery_size": len(gallery),
                "num_known": sum(1 for a in assignments if a.label == "Known" and a.identity_id != "unknown"),
                "num_unknown": sum(1 for a in assignments if a.identity_id == "unknown"),
                "stages_ms": stats.stages_ms,
            }
            jsonl.write(record)
            if embeddings:
                for emb in embeddings:
                    emb_jsonl.write(
                        {
                            "frame_id": stats.frame_id,
                            "timestamp_s": stats.timestamp_s,
                            "track_id": emb.track_id,
                            "embedding": emb.embedding.tolist(),
                            "embedding_dim": emb.embedding_dim,
                            "quality": emb.quality,
                        }
                    )
            if assignments:
                for assign in assignments:
                    id_jsonl.write(
                        {
                            "frame_id": stats.frame_id,
                            "timestamp_s": stats.timestamp_s,
                            "track_id": assign.track_id,
                            "identity_id": assign.identity_id,
                            "label": assign.label,
                            "score": assign.score,
                            "margin": assign.margin,
                            "enrolled": assign.enrolled,
                            "updated": assign.updated,
                            "skip_reason": assign.skip_reason,
                        }
                    )

            frames += 1
            if cfg.print_every > 0 and (frames % cfg.print_every == 0):
                st = stats.stages_ms
                msg = (
                    f"frame={stats.frame_id} total={stats.dt_ms:.2f}ms "
                    f"det={st.get('detector', 0.0):.2f}ms "
                    f"trk={st.get('tracker', 0.0):.2f}ms "
                    f"emb={st.get('embedder', 0.0):.2f}ms "
                    f"gal={st.get('gallery', 0.0):.2f}ms "
                    f"fps(roll)={stats.fps_rolling:.2f} dets={len(detections)} tracks={len(tracks)} "
                    f"crops={num_crops} assigns={len(assignments)}"
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
        if emb_jsonl is not None:
            try:
                emb_jsonl.close()
            except Exception:
                pass
        if cfg.gallery_path:
            try:
                gallery.save(cfg.gallery_path)
                logger.info(f"gallery_saved path={cfg.gallery_path} size={len(gallery)}")
            except Exception as e:
                logger.warning(f"Failed to save gallery to {cfg.gallery_path}: {e}")
        if id_jsonl is not None:
            try:
                id_jsonl.close()
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
