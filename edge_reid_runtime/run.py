from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

from edge_reid_runtime.core.types import RunConfig
from edge_reid_runtime.sources.factory import create_source
from edge_reid_runtime.utils.log import setup_logger, JsonlWriter
from edge_reid_runtime.utils.profiler import SimpleProfiler


VALID_SOURCES = ("webcam", "video", "robot")
VALID_DEVICES = ("cpu", "cuda", "auto")


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
    )


def run_minimal_loop(cfg: RunConfig) -> int:
    logger = setup_logger("edge_reid", cfg.output_dir)
    logger.info("Starting EDGE pipeline scaffold.")
    logger.info(f"source={cfg.source} device={cfg.device} output_dir={cfg.output_dir}")

    profiler = SimpleProfiler(fps_window=30)

    # Source
    src = None
    jsonl = None
    try:
        src = create_source(cfg)
    except Exception as e:
        logger.error(f"Failed to create source: {e}")
        return 2

    # JSONL per-frame logs
    jsonl = JsonlWriter(cfg.output_dir / "frames.jsonl")

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

            # Phase 2: No detection/tracking yet. This loop is the timing scaffold.
            # Later you will add: detections = detector.detect(frame), tracks = tracker.update(...), etc.

            stats = profiler.on_frame_end(frame_id=frame.frame_id, timestamp_s=frame.timestamp_s)

            record: Dict[str, Any] = {
                "frame_id": stats.frame_id,
                "timestamp_s": stats.timestamp_s,
                "dt_ms": stats.dt_ms,
                "fps_rolling": stats.fps_rolling,
                "rss_mb": stats.rss_mb,
                "source": frame.meta.get("source"),
            }
            jsonl.write(record)

            frames += 1
            if cfg.print_every > 0 and (frames % cfg.print_every == 0):
                if stats.rss_mb is None:
                    logger.info(f"frame={stats.frame_id} dt={stats.dt_ms:.2f}ms fps(roll)={stats.fps_rolling:.2f}")
                else:
                    logger.info(
                        f"frame={stats.frame_id} dt={stats.dt_ms:.2f}ms fps(roll)={stats.fps_rolling:.2f} rss={stats.rss_mb:.1f}MB"
                    )

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
