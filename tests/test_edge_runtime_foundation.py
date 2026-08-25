import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from edge_reid_runtime.core.interfaces import Track
from edge_reid_runtime.core.types import RunConfig, validate_run_config
from edge_reid_runtime.gallery.assigner import IdentityAssigner
from edge_reid_runtime.gallery.manager import GalleryConfig, GalleryManager
from edge_reid_runtime.utils.log import JsonlWriter
from edge_reid_runtime.run import allocate_run_directory, build_config, parse_args, write_run_metadata
from edge_reid_runtime.utils.profiler import FrameStats, StageProfiler


def test_runtime_requires_embedder_for_experiment(tmp_path):
    with pytest.raises(ValueError, match="reid_backbone"):
        validate_run_config(RunConfig(source="robot", device="cpu", output_dir=tmp_path))


def test_gallery_threshold_regions_are_explicit():
    gallery = GalleryManager(GalleryConfig(known_threshold=0.9, unknown_threshold=0.5, model_id="x", embedding_dim=2))
    gallery.add("person_0001", np.array([1.0, 0.0]), 0.0)
    assert gallery.match(np.array([1.0, 0.0])).status == "known"
    assert gallery.match(np.array([0.7, 0.7141428])).status == "uncertain"
    assert gallery.match(np.array([-1.0, 0.0])).status == "unknown"


def test_gallery_persistence_rejects_model_switch(tmp_path):
    path = tmp_path / "gallery.json"
    gallery = GalleryManager(GalleryConfig(model_id="torch:osnet", embedding_dim=2))
    gallery.add(None, np.array([1.0, 0.0]), 0.0)
    gallery.save(path)
    with pytest.raises(RuntimeError, match="incompatible"):
        GalleryManager(GalleryConfig(model_id="onnx:osnet", embedding_dim=2)).load(path)


def test_jsonl_artifacts_cannot_append(tmp_path):
    path = tmp_path / "frames.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"frame_id": 1})
    with pytest.raises(FileExistsError):
        JsonlWriter(path)


def test_assigner_forgets_deleted_track_state():
    assigner = IdentityAssigner(GalleryManager(GalleryConfig(model_id="x", embedding_dim=2)))
    track = Track(track_id=7, bbox_xyxy=(0, 0, 10, 20), conf=1.0)
    assigner.assign(1, track, None, 0.0)
    assert 7 in assigner._track_state
    assigner.forget_tracks({7})
    assert 7 not in assigner._track_state


def test_profiler_summary_retains_active_stages():
    profiler = StageProfiler(collect_history=True)
    profiler.on_frame_start()
    with profiler.stage("detector"):
        pass
    profiler.on_frame_end(frame_id=0, timestamp_s=0.0)
    assert "detector" in profiler.summarize()


def test_warmup_excludes_first_thirty_frames_and_aggregates_rss():
    profiler = StageProfiler(collect_history=True)
    profiler._history = [
        FrameStats(
            frame_id=index,
            timestamp_s=float(index),
            dt_ms=float(index),
            fps_rolling=1.0,
            rss_mb=float(100 + index),
            vram_mb=None,
            stages_ms={"detector": float(index)},
        )
        for index in range(31)
    ]
    assert [stat.frame_id for stat in profiler.measured_history(30)] == [30]
    assert profiler.summarize(30)["total"]["mean"] == 30.0
    assert profiler.rss_summary(30) == {"rss_mean_mb": 130.0, "rss_peak_mb": 130.0}


def test_reset_gallery_is_isolated_inside_each_run_directory(tmp_path):
    cfg = RunConfig(source="robot", device="cpu", output_dir=tmp_path, reid_backbone="osnet_x0_25", reset_gallery=True)
    first = allocate_run_directory(cfg)
    second = allocate_run_directory(cfg)
    assert first.output_dir != second.output_dir
    assert first.gallery_path == first.output_dir / "gallery.json"
    assert second.gallery_path == second.output_dir / "gallery.json"


def test_video_metadata_includes_sha256(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"qualification-video")
    cfg = RunConfig(source="video", device="cpu", output_dir=tmp_path, video_path=video, reid_backbone="osnet_x0_25")
    write_run_metadata(cfg, "cpu", 512)
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["input_video"]["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert metadata["input_video"]["size_bytes"] == len(b"qualification-video")


def test_qualification_configs_have_identical_locked_conditions():
    paths = sorted(Path("configs").glob("runtime_qualification_*_*.yaml"))
    configs = [build_config(parse_args(["--config", str(path)])) for path in paths]
    assert len(configs) == 6
    locked = {
        (cfg.source, str(cfg.video_path), cfg.device, cfg.max_frames, cfg.warmup_frames,
         cfg.detector, cfg.yolo_model, cfg.det_conf, cfg.det_iou, cfg.imgsz, cfg.max_det,
         cfg.max_age, cfg.n_init, cfg.max_iou_distance, cfg.reset_gallery, cfg.gallery_path)
        for cfg in configs
    }
    assert locked == {
        ("video", "data/qualification_sequences/qualificationsequence1.mp4", "cpu", 1227, 30,
         "yolo26", "edge_reid_runtime/weights/yolo26/yolo26n.pt", 0.35, 0.70, 640, 100,
         30, 3, 0.70, True, None)
    }
