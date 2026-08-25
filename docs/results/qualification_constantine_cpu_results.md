# Qualification Runtime Results — Constantine CPU

Snapshot generated: 2026-08-25 21:59:39 UTC.

Generated from existing benchmark artifacts. No benchmark was rerun to produce this document.

## 1. Experiment Scope

These results correspond to the controlled qualification runtime benchmark executed on Constantine using CPU execution. The comparison evaluates the selected MSMT17-trained ResNet50, OSNet x0.25, and OSNet x0.5 with Torch and ONNX Runtime backends. They serve as the Constantine CPU reference before reproducing the same protocol on the local PC and Raspberry Pi 4.

## 2. Locked Experimental Conditions

| Condition | Verified value |
|---|---|
| Platform | Constantine |
| Device | CPU |
| Input video | `data/qualification_sequences/qualificationsequence1.mp4` |
| Video SHA-256 | `f86d05e023982b8c76c2e852e9419b53ef1c67fc6ab3b89af160176e5e66197b` |
| Resolution | 1920x1080 |
| Input FPS | 29.99221066194164 |
| Total frames | 1227 |
| Warm-up frames | 30 |
| Measured frames | 1197 |
| Detector | yolo26 |
| Detector weights | `edge_reid_runtime/weights/yolo26/yolo26n.pt` |
| Tracker | deepsort |
| Visualization policy | Disabled (`save_video: false`, `display: false`) |
| Gallery initialization policy | Reset to an empty run-local `gallery.json` for each run |
| Number of configurations | 6 |

## 3. Benchmark Configurations

| Experiment | ReID model | Training/source information | Backend | Embedding dimension | ReID weights/checkpoint | Run directory |
|---|---|---|---|---:|---|---|
| resnet50_selected_torch | Selected MSMT17 ResNet50 | Selected MSMT17 checkpoint (`msmt17_no_label_smoothing`) | torch | 2048 | `exp/msmt17_no_label_smoothing/checkpoints/ckpt_best.pth` | `exp/qualification_runtime/resnet50_selected_torch/20260825T204032Z` |
| resnet50_selected_onnx | Selected MSMT17 ResNet50 | Selected MSMT17 checkpoint (`msmt17_no_label_smoothing`) | onnx | 2048 | `edge_reid_runtime/weights/onnx/resnet50_selected.onnx` | `exp/qualification_runtime/resnet50_selected_onnx/20260825T210132Z` |
| osnet_x0_25_torch | OSNet x0.25 | Configured OSNet MSMT17 weight file | torch | 512 | `edge_reid_runtime/weights/models--kaiyangzhou--osnet/snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth` | `exp/qualification_runtime/osnet_x0_25_torch/20260825T211257Z` |
| osnet_x0_25_onnx | OSNet x0.25 | Configured OSNet MSMT17 weight file | onnx | 512 | `edge_reid_runtime/weights/onnx/osnet_x0_25.onnx` | `exp/qualification_runtime/osnet_x0_25_onnx/20260825T211946Z` |
| osnet_x0_5_torch | OSNet x0.5 | Configured OSNet MSMT17 weight file | torch | 512 | `edge_reid_runtime/weights/models--kaiyangzhou--osnet/snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth` | `exp/qualification_runtime/osnet_x0_5_torch/20260825T212357Z` |
| osnet_x0_5_onnx | OSNet x0.5 | Configured OSNet MSMT17 weight file | onnx | 512 | `edge_reid_runtime/weights/onnx/osnet_x0_5.onnx` | `exp/qualification_runtime/osnet_x0_5_onnx/20260825T212837Z` |

## 4. Complete Runtime Results

| Model | Backend | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | FPS | Detector (ms) | Tracker (ms) | ReID (ms) | Gallery (ms) | Mean RSS (MB) | Peak RSS (MB) | Final Gallery Size | Measured Frames |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected MSMT17 ResNet50 | torch | 883.98 | 871.66 | 2106.26 | 1.13 | 100.61 | 26.22 | 753.24 | 0.23 | 1247.47 | 1351.48 | 28 | 1197 |
| Selected MSMT17 ResNet50 | onnx | 222.53 | 231.47 | 350.65 | 4.49 | 113.39 | 27.90 | 77.00 | 0.26 | 1146.48 | 1187.75 | 28 | 1197 |
| OSNet x0.25 | torch | 177.51 | 187.70 | 252.24 | 5.63 | 100.70 | 26.19 | 46.78 | 0.14 | 1539.61 | 1590.73 | 5 | 1197 |
| OSNet x0.25 | onnx | **154.61** | 156.01 | 211.28 | **6.47** | 113.69 | 25.58 | **11.12** | 0.15 | 1094.81 | 1155.25 | 5 | 1197 |
| OSNet x0.5 | torch | 192.20 | 206.45 | 272.20 | 5.20 | 100.27 | 25.72 | 62.32 | 0.15 | 1606.47 | 1663.93 | 7 | 1197 |
| OSNet x0.5 | onnx | 173.06 | 179.54 | 256.27 | 5.78 | 115.52 | 27.17 | 25.85 | 0.20 | 1094.85 | 1136.34 | 7 | 1197 |

## 5. Torch vs ONNX Comparison

Latency reduction (%) = `(Torch latency - ONNX latency) / Torch latency * 100`. Speedup = `Torch latency / ONNX latency`. FPS factor = `ONNX FPS / Torch FPS`.

| Model | Mean latency reduction (%) | Mean-latency speedup | FPS factor | ReID latency reduction (%) | ReID speedup | Mean RSS difference (Torch - ONNX, MB) | Peak RSS difference (Torch - ONNX, MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected ResNet50 | 74.83 | 3.97x | 3.97x | 89.78 | 9.78x | 100.99 | 163.73 |
| OSNet x0.25 | 12.90 | 1.15x | 1.15x | 76.23 | 4.21x | 444.80 | 435.47 |
| OSNet x0.5 | 9.96 | 1.11x | 1.11x | 58.51 | 2.41x | 511.62 | 527.58 |

## 6. ReID Model Comparison

| Backend | Model | Mean total latency (ms) | FPS | ReID latency (ms) | Mean RSS (MB) |
|---|---|---:|---:|---:|---:|
| torch | Selected MSMT17 ResNet50 | 883.98 | 1.13 | 753.24 | 1247.47 |
| torch | OSNet x0.25 | 177.51 | 5.63 | 46.78 | 1539.61 |
| torch | OSNet x0.5 | 192.20 | 5.20 | 62.32 | 1606.47 |
| onnx | Selected MSMT17 ResNet50 | 222.53 | 4.49 | 77.00 | 1146.48 |
| onnx | OSNet x0.25 | 154.61 | 6.47 | 11.12 | 1094.81 |
| onnx | OSNet x0.5 | 173.06 | 5.78 | 25.85 | 1094.85 |

The lowest measured total latency was obtained by OSNet x0.25 ONNX (154.61 ms). The highest measured FPS was obtained by OSNet x0.25 ONNX (6.47 FPS). The lowest measured ReID latency was obtained by OSNet x0.25 ONNX (11.12 ms).

## 7. Pipeline Stage Analysis

| Model | Backend | Detector mean (ms) | Tracker mean (ms) | ReID mean (ms) | Gallery mean (ms) |
|---|---|---:|---:|---:|---:|
| Selected MSMT17 ResNet50 | torch | 100.61 | 26.22 | 753.24 | 0.23 |
| Selected MSMT17 ResNet50 | onnx | 113.39 | 27.90 | 77.00 | 0.26 |
| OSNet x0.25 | torch | 100.70 | 26.19 | 46.78 | 0.14 |
| OSNet x0.25 | onnx | 113.69 | 25.58 | 11.12 | 0.15 |
| OSNet x0.5 | torch | 100.27 | 25.72 | 62.32 | 0.15 |
| OSNet x0.5 | onnx | 115.52 | 27.17 | 25.85 | 0.20 |

The measured timings show that after ONNX optimization, detector latency becomes a major component of total latency for the OSNet configurations. This is a timing observation only.

## 8. Gallery Observations

| Model | Torch final gallery size | ONNX final gallery size |
|---|---:|---:|
| Selected MSMT17 ResNet50 | 28 | 28 |
| OSNet x0.25 | 5 | 5 |
| OSNet x0.5 | 7 | 7 |

Gallery size is an operational observation, not an identity-accuracy metric, because this benchmark sequence does not provide independent identity ground truth for the assignments.

## 9. Key Preliminary Observations

- OSNet x0.25 ONNX obtained the highest measured FPS (6.47 FPS) and the lowest mean latency (154.61 ms).
- ONNX reduced mean latency for all three matched models in this single-run CPU result set.
- Selected ResNet50 showed the largest Torch-to-ONNX mean-latency reduction (74.83%).
- The measured ReID stage accounts for most of the Torch-to-ONNX timing difference for the selected ResNet50.
- For the ONNX OSNet configurations, detector latency is a major component of the measured total latency.

## 10. Limitations of This Result Set

- One fixed video was used.
- One CPU platform was used.
- One run per configuration was performed.
- Repeated-run statistical variance is not available yet.
- Gallery assignments are not external ground truth.
- These results should not yet be generalized to the local PC or Raspberry Pi.

## 11. Artifact Traceability

| Experiment | Run directory | summary.json | run_metadata.json | resolved_config.yaml | gallery.json |
|---|---|---|---|---|---|
| resnet50_selected_torch | `exp/qualification_runtime/resnet50_selected_torch/20260825T204032Z` | `exp/qualification_runtime/resnet50_selected_torch/20260825T204032Z/summary.json` | `exp/qualification_runtime/resnet50_selected_torch/20260825T204032Z/run_metadata.json` | `exp/qualification_runtime/resnet50_selected_torch/20260825T204032Z/resolved_config.yaml` | `exp/qualification_runtime/resnet50_selected_torch/20260825T204032Z/gallery.json` |
| resnet50_selected_onnx | `exp/qualification_runtime/resnet50_selected_onnx/20260825T210132Z` | `exp/qualification_runtime/resnet50_selected_onnx/20260825T210132Z/summary.json` | `exp/qualification_runtime/resnet50_selected_onnx/20260825T210132Z/run_metadata.json` | `exp/qualification_runtime/resnet50_selected_onnx/20260825T210132Z/resolved_config.yaml` | `exp/qualification_runtime/resnet50_selected_onnx/20260825T210132Z/gallery.json` |
| osnet_x0_25_torch | `exp/qualification_runtime/osnet_x0_25_torch/20260825T211257Z` | `exp/qualification_runtime/osnet_x0_25_torch/20260825T211257Z/summary.json` | `exp/qualification_runtime/osnet_x0_25_torch/20260825T211257Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_25_torch/20260825T211257Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_25_torch/20260825T211257Z/gallery.json` |
| osnet_x0_25_onnx | `exp/qualification_runtime/osnet_x0_25_onnx/20260825T211946Z` | `exp/qualification_runtime/osnet_x0_25_onnx/20260825T211946Z/summary.json` | `exp/qualification_runtime/osnet_x0_25_onnx/20260825T211946Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_25_onnx/20260825T211946Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_25_onnx/20260825T211946Z/gallery.json` |
| osnet_x0_5_torch | `exp/qualification_runtime/osnet_x0_5_torch/20260825T212357Z` | `exp/qualification_runtime/osnet_x0_5_torch/20260825T212357Z/summary.json` | `exp/qualification_runtime/osnet_x0_5_torch/20260825T212357Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_5_torch/20260825T212357Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_5_torch/20260825T212357Z/gallery.json` |
| osnet_x0_5_onnx | `exp/qualification_runtime/osnet_x0_5_onnx/20260825T212837Z` | `exp/qualification_runtime/osnet_x0_5_onnx/20260825T212837Z/summary.json` | `exp/qualification_runtime/osnet_x0_5_onnx/20260825T212837Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_5_onnx/20260825T212837Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_5_onnx/20260825T212837Z/gallery.json` |

All metadata recorded git commit `12138be9dccac461ff64351c1c4612718775e851` and detector weights `edge_reid_runtime/weights/yolo26/yolo26n.pt`. A detector weight checksum was not recorded in the source metadata. The locked video SHA-256 was `f86d05e023982b8c76c2e852e9419b53ef1c67fc6ab3b89af160176e5e66197b`.

## 12. Raw Frozen Values

The following values are reproduced from the source `summary.json` files without rounding.

```text
{"experiment": "resnet50_selected_torch", "mean_total_ms": 883.9798954929363, "median_total_ms": 871.6635960154235, "p95_total_ms": 2106.2596291303635, "benchmark_fps": 1.131247447027477, "detector_mean_ms": 100.60909847513389, "tracker_mean_ms": 26.224588197652086, "reid_mean_ms": 753.2445253048064, "gallery_mean_ms": 0.23260971518823179, "rss_mean_mb": 1247.466596177945, "rss_peak_mb": 1351.48046875, "gallery_size": 28, "processed_frames_total": 1227, "warmup_frames": 30, "measured_frames": 1197}
{"experiment": "resnet50_selected_onnx", "mean_total_ms": 222.52918323214067, "median_total_ms": 231.47025611251593, "p95_total_ms": 350.652021355927, "benchmark_fps": 4.4937926139638416, "detector_mean_ms": 113.39208643195673, "tracker_mean_ms": 27.89512738926257, "reid_mean_ms": 77.00305857322954, "gallery_mean_ms": 0.25510842427175645, "rss_mean_mb": 1146.4758771929824, "rss_peak_mb": 1187.74609375, "gallery_size": 28, "processed_frames_total": 1227, "warmup_frames": 30, "measured_frames": 1197}
{"experiment": "osnet_x0_25_torch", "mean_total_ms": 177.51356286020854, "median_total_ms": 187.69732816144824, "p95_total_ms": 252.24350625649095, "benchmark_fps": 5.633372368214463, "detector_mean_ms": 100.70103019399689, "tracker_mean_ms": 26.18960087221044, "reid_mean_ms": 46.78120683262447, "gallery_mean_ms": 0.1355454651677768, "rss_mean_mb": 1539.6096328059732, "rss_peak_mb": 1590.7265625, "gallery_size": 5, "processed_frames_total": 1227, "warmup_frames": 30, "measured_frames": 1197}
{"experiment": "osnet_x0_25_onnx", "mean_total_ms": 154.61163278070023, "median_total_ms": 156.01275768131018, "p95_total_ms": 211.28263184800744, "benchmark_fps": 6.467818637025787, "detector_mean_ms": 113.68835936810078, "tracker_mean_ms": 25.58225734858037, "reid_mean_ms": 11.117819703278212, "gallery_mean_ms": 0.15238558938890173, "rss_mean_mb": 1094.8064464546783, "rss_peak_mb": 1155.25390625, "gallery_size": 5, "processed_frames_total": 1227, "warmup_frames": 30, "measured_frames": 1197}
{"experiment": "osnet_x0_5_torch", "mean_total_ms": 192.20164101788697, "median_total_ms": 206.45107328891754, "p95_total_ms": 272.1991017460823, "benchmark_fps": 5.202869209149657, "detector_mean_ms": 100.2664194561373, "tracker_mean_ms": 25.720707916576064, "reid_mean_ms": 62.31622362265063, "gallery_mean_ms": 0.14841002034787446, "rss_mean_mb": 1606.472277699457, "rss_peak_mb": 1663.92578125, "gallery_size": 7, "processed_frames_total": 1227, "warmup_frames": 30, "measured_frames": 1197}
{"experiment": "osnet_x0_5_onnx", "mean_total_ms": 173.05744011593094, "median_total_ms": 179.53660432249308, "p95_total_ms": 256.2655182555318, "benchmark_fps": 5.778428245154333, "detector_mean_ms": 115.52014713678992, "tracker_mean_ms": 27.167465231235862, "reid_mean_ms": 25.853221675843223, "gallery_mean_ms": 0.19728384504317242, "rss_mean_mb": 1094.850766238513, "rss_peak_mb": 1136.34375, "gallery_size": 7, "processed_frames_total": 1227, "warmup_frames": 30, "measured_frames": 1197}
```
