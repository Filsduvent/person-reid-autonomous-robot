# Qualification Runtime Results — Local PC CPU

Generated from existing benchmark artifacts. No benchmark was rerun to produce this document.

Markdown snapshot generated: 2026-08-25 21:42:27 -03:00.

## 1. Experiment Scope

These results correspond to the controlled qualification runtime benchmark performed on the local PC using CPU execution. The experiment compares the selected MSMT17-trained ResNet50, OSNet x0.25, and OSNet x0.5 using Torch and ONNX Runtime.

The same fixed benchmark video and runtime protocol used on Constantine were preserved for this local-PC experiment. This document reports local-PC results only and makes no cross-platform comparison.

## 2. Local Hardware and Software Environment

| Item | Recorded value |
|---|---|
| Hostname | `Irakoze` |
| OS | Linux `7.0.0-30-generic`, x86-64 |
| CPU | Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz |
| Physical / logical cores | 4 / 8 |
| RAM | 16,648,306,688 bytes (15.50 GiB) |
| Python | 3.12.3 |
| PyTorch | `2.7.1+cpu` |
| Ultralytics | `8.4.128` |
| Torchreid | `0.2.5` |
| deep-sort-realtime | `1.3.2` |
| ONNX | `1.22.0` |
| ONNX Runtime | `1.29.0` |
| ONNX execution provider | `CPUExecutionProvider` |
| CUDA available | No |

The Python, PyTorch, Ultralytics, and ONNX Runtime versions are also recorded in the selected run metadata. Package versions not stored there were read from the current local `Reid` environment.

## 3. Locked Experimental Conditions

| Condition | Verified value |
|---|---|
| Platform | Local PC (`Irakoze`) |
| Device | CPU |
| Input video | `data/qualification_sequences/qualificationsequence1.mp4` |
| Video SHA-256 | `f86d05e023982b8c76c2e852e9419b53ef1c67fc6ab3b89af160176e5e66197b` |
| Resolution | 1920x1080 |
| Input FPS | 29.99221066194164 |
| Total frames | 1227 |
| Warm-up frames | 30 |
| Measured frames | 1197 |
| Detector | Official Ultralytics YOLO26n (`yolo26`) |
| Detector weights | `edge_reid_runtime/weights/yolo26/yolo26n.pt` |
| Tracker | DeepSORT (`max_age=30`, `n_init=3`, `max_iou_distance=0.7`) |
| Visualization policy | Disabled (`save_video: false`, `display: false`) |
| Gallery initialization policy | Reset to an empty run-local `gallery.json` for every run |
| Number of configurations | 6 |

## 4. Benchmark Configurations

| Experiment | ReID model | Backend | Embedding dimension | ReID weights/checkpoint | Run directory |
|---|---|---|---:|---|---|
| `resnet50_selected_torch` | Selected MSMT17-trained ResNet50, no label smoothing | Torch | 2048 | `exp/msmt17_no_label_smoothing/checkpoints/ckpt_best.pth` | `exp/qualification_runtime/resnet50_selected_torch/20260825T234319Z` |
| `resnet50_selected_onnx` | Selected MSMT17-trained ResNet50, no label smoothing | ONNX Runtime | 2048 | `edge_reid_runtime/weights/onnx/resnet50_selected.onnx` | `exp/qualification_runtime/resnet50_selected_onnx/20260826T000726Z` |
| `osnet_x0_25_torch` | OSNet x0.25 | Torch | 512 | `edge_reid_runtime/weights/models--kaiyangzhou--osnet/snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth` | `exp/qualification_runtime/osnet_x0_25_torch/20260826T001746Z` |
| `osnet_x0_25_onnx` | OSNet x0.25 | ONNX Runtime | 512 | `edge_reid_runtime/weights/onnx/osnet_x0_25.onnx` | `exp/qualification_runtime/osnet_x0_25_onnx/20260826T002201Z` |
| `osnet_x0_5_torch` | OSNet x0.5 | Torch | 512 | `edge_reid_runtime/weights/models--kaiyangzhou--osnet/snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth` | `exp/qualification_runtime/osnet_x0_5_torch/20260826T002747Z` |
| `osnet_x0_5_onnx` | OSNet x0.5 | ONNX Runtime | 512 | `edge_reid_runtime/weights/onnx/osnet_x0_5.onnx` | `exp/qualification_runtime/osnet_x0_5_onnx/20260826T003228Z` |

## 5. Complete Runtime Results

| Model | Backend | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | FPS | Detector (ms) | Tracker (ms) | ReID (ms) | Gallery (ms) | Mean RSS (MB) | Peak RSS (MB) | Final Gallery Size | Measured Frames |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected MSMT17 ResNet50 | Torch | 1109.47 | 1039.20 | 2546.13 | 0.90 | 112.60 | 31.54 | 961.76 | 0.18 | 801.96 | 809.07 | 28 | 1197 |
| Selected MSMT17 ResNet50 | ONNX Runtime | 432.36 | 436.36 | 820.52 | 2.31 | 156.31 | 28.73 | 244.11 | 0.16 | 796.04 | 837.02 | 28 | 1197 |
| OSNet x0.25 | Torch | **162.96** | **158.96** | **310.48** | **6.14** | 99.28 | 26.95 | 33.44 | 0.10 | 1205.81 | 1231.03 | 5 | 1197 |
| OSNet x0.25 | ONNX Runtime | 217.41 | 222.39 | 346.79 | 4.60 | 167.32 | 29.75 | **16.86** | 0.10 | **697.46** | **738.66** | 5 | 1197 |
| OSNet x0.5 | Torch | 187.91 | 179.29 | 338.58 | 5.32 | 102.16 | 31.25 | 50.94 | 0.11 | 1207.09 | 1221.34 | 7 | 1197 |
| OSNet x0.5 | ONNX Runtime | 227.60 | 235.55 | 374.88 | 4.39 | 157.88 | 28.82 | 37.69 | 0.11 | 704.04 | 748.55 | 7 | 1197 |

Displayed values are rounded to two decimal places. Section 13 preserves the source values at full precision.

## 6. Torch vs ONNX Comparison

Latency reduction (%) = `(Torch latency - ONNX latency) / Torch latency * 100`. Speedup = `Torch latency / ONNX latency`. FPS factor = `ONNX FPS / Torch FPS`. Negative latency reduction and factors below 1 indicate that ONNX measured slower than Torch for that pair. RSS differences are reported as Torch minus ONNX.

### Selected ResNet50

| Metric | Frozen-value comparison |
|---|---:|
| Total mean latency reduction | 61.03% |
| Total latency speedup | 2.57x |
| FPS increase factor | 2.57x |
| ReID latency reduction | 74.62% |
| ReID latency speedup | 3.94x |
| Mean RSS difference (Torch - ONNX) | 5.93 MB |
| Peak RSS difference (Torch - ONNX) | -27.95 MB |

### OSNet x0.25

| Metric | Frozen-value comparison |
|---|---:|
| Total mean latency reduction | -33.42% |
| Total latency speedup | 0.75x |
| FPS increase factor | 0.75x |
| ReID latency reduction | 49.59% |
| ReID latency speedup | 1.98x |
| Mean RSS difference (Torch - ONNX) | 508.35 MB |
| Peak RSS difference (Torch - ONNX) | 492.36 MB |

### OSNet x0.5

| Metric | Frozen-value comparison |
|---|---:|
| Total mean latency reduction | -21.12% |
| Total latency speedup | 0.83x |
| FPS increase factor | 0.83x |
| ReID latency reduction | 26.01% |
| ReID latency speedup | 1.35x |
| Mean RSS difference (Torch - ONNX) | 503.05 MB |
| Peak RSS difference (Torch - ONNX) | 472.79 MB |

## 7. ReID Model Comparison

### Torch

| Model | Mean total latency (ms) | FPS | ReID latency (ms) | Mean RSS (MB) |
|---|---:|---:|---:|---:|
| Selected MSMT17 ResNet50 | 1109.47 | 0.90 | 961.76 | 801.96 |
| OSNet x0.25 | 162.96 | 6.14 | 33.44 | 1205.81 |
| OSNet x0.5 | 187.91 | 5.32 | 50.94 | 1207.09 |

### ONNX Runtime

| Model | Mean total latency (ms) | FPS | ReID latency (ms) | Mean RSS (MB) |
|---|---:|---:|---:|---:|
| Selected MSMT17 ResNet50 | 432.36 | 2.31 | 244.11 | 796.04 |
| OSNet x0.25 | 217.41 | 4.60 | 16.86 | 697.46 |
| OSNet x0.5 | 227.60 | 4.39 | 37.69 | 704.04 |

Across all six configurations, OSNet x0.25 Torch obtained the lowest measured total latency (162.96 ms) and highest measured FPS (6.14 FPS). OSNet x0.25 ONNX obtained the lowest measured ReID-stage latency (16.86 ms).

## 8. Pipeline Stage Analysis

| Model | Backend | Detector mean (ms) | Tracker mean (ms) | ReID mean (ms) | Gallery mean (ms) |
|---|---|---:|---:|---:|---:|
| Selected MSMT17 ResNet50 | Torch | 112.60 | 31.54 | 961.76 | 0.18 |
| Selected MSMT17 ResNet50 | ONNX Runtime | 156.31 | 28.73 | 244.11 | 0.16 |
| OSNet x0.25 | Torch | 99.28 | 26.95 | 33.44 | 0.10 |
| OSNet x0.25 | ONNX Runtime | 167.32 | 29.75 | 16.86 | 0.10 |
| OSNet x0.5 | Torch | 102.16 | 31.25 | 50.94 | 0.11 |
| OSNet x0.5 | ONNX Runtime | 157.88 | 28.82 | 37.69 | 0.11 |

ONNX reduced the measured ReID-stage mean latency for every model. For selected ResNet50, the ReID share of total mean latency changed from 86.69% with Torch to 56.46% with ONNX Runtime, and the total mean latency also decreased.

For OSNet x0.25, the ReID share changed from 20.52% to 7.75%; for OSNet x0.5, it changed from 27.11% to 16.56%. In both ONNX OSNet runs, detector mean latency was higher than in the paired Torch run and represented most of the measured total mean latency (76.96% for x0.25 and 69.36% for x0.5). These paired OSNet runs therefore measured lower ReID latency but higher total latency with ONNX.

Gallery mean latency remained between 0.10 ms and 0.18 ms across the six runs.

## 9. Gallery Observations

| Model | Torch final gallery size | ONNX final gallery size |
|---|---:|---:|
| Selected MSMT17 ResNet50 | 28 | 28 |
| OSNet x0.25 | 5 | 5 |
| OSNet x0.5 | 7 | 7 |

Gallery size is an operational observation, not an identity-accuracy metric, because there is no independent ground truth for gallery assignments in this benchmark sequence. A larger or smaller final gallery is not interpreted as better.

## 10. Key Preliminary Observations

- OSNet x0.25 Torch obtained the highest measured FPS (6.14 FPS) and lowest mean total latency (162.96 ms).
- ONNX reduced total mean latency for selected ResNet50, but not for OSNet x0.25 or OSNet x0.5 in these local single runs.
- Selected ResNet50 showed the largest Torch-to-ONNX total mean-latency improvement: 61.03%.
- ONNX reduced the measured ReID-stage mean latency for all three models; the selected ResNet50 ReID stage showed the largest reduction at 74.62%.
- The ReID stage accounts for most of the selected ResNet50 Torch-to-ONNX total-latency improvement.
- Detector latency became relatively dominant in both ONNX OSNet runs after the measured ReID-stage latency decreased.
- Both ONNX OSNet runs used approximately 500 MB less mean RSS than their paired Torch runs.

## 11. Limitations

- The benchmark used one fixed input video.
- The results come from one local CPU platform.
- The artifact set contains one completed qualification run per configuration.
- Repeated-run variance is not available.
- Gallery assignments are not external identity ground truth.
- Runtime values alone do not support identity-accuracy claims.
- These results should not yet be generalized to Raspberry Pi.

## 12. Artifact Traceability

| Experiment | Exact timestamped run directory | `summary.json` | `run_metadata.json` | `resolved_config.yaml` | `gallery.json` |
|---|---|---|---|---|---|
| `resnet50_selected_torch` | `exp/qualification_runtime/resnet50_selected_torch/20260825T234319Z` | `exp/qualification_runtime/resnet50_selected_torch/20260825T234319Z/summary.json` | `exp/qualification_runtime/resnet50_selected_torch/20260825T234319Z/run_metadata.json` | `exp/qualification_runtime/resnet50_selected_torch/20260825T234319Z/resolved_config.yaml` | `exp/qualification_runtime/resnet50_selected_torch/20260825T234319Z/gallery.json` |
| `resnet50_selected_onnx` | `exp/qualification_runtime/resnet50_selected_onnx/20260826T000726Z` | `exp/qualification_runtime/resnet50_selected_onnx/20260826T000726Z/summary.json` | `exp/qualification_runtime/resnet50_selected_onnx/20260826T000726Z/run_metadata.json` | `exp/qualification_runtime/resnet50_selected_onnx/20260826T000726Z/resolved_config.yaml` | `exp/qualification_runtime/resnet50_selected_onnx/20260826T000726Z/gallery.json` |
| `osnet_x0_25_torch` | `exp/qualification_runtime/osnet_x0_25_torch/20260826T001746Z` | `exp/qualification_runtime/osnet_x0_25_torch/20260826T001746Z/summary.json` | `exp/qualification_runtime/osnet_x0_25_torch/20260826T001746Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_25_torch/20260826T001746Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_25_torch/20260826T001746Z/gallery.json` |
| `osnet_x0_25_onnx` | `exp/qualification_runtime/osnet_x0_25_onnx/20260826T002201Z` | `exp/qualification_runtime/osnet_x0_25_onnx/20260826T002201Z/summary.json` | `exp/qualification_runtime/osnet_x0_25_onnx/20260826T002201Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_25_onnx/20260826T002201Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_25_onnx/20260826T002201Z/gallery.json` |
| `osnet_x0_5_torch` | `exp/qualification_runtime/osnet_x0_5_torch/20260826T002747Z` | `exp/qualification_runtime/osnet_x0_5_torch/20260826T002747Z/summary.json` | `exp/qualification_runtime/osnet_x0_5_torch/20260826T002747Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_5_torch/20260826T002747Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_5_torch/20260826T002747Z/gallery.json` |
| `osnet_x0_5_onnx` | `exp/qualification_runtime/osnet_x0_5_onnx/20260826T003228Z` | `exp/qualification_runtime/osnet_x0_5_onnx/20260826T003228Z/summary.json` | `exp/qualification_runtime/osnet_x0_5_onnx/20260826T003228Z/run_metadata.json` | `exp/qualification_runtime/osnet_x0_5_onnx/20260826T003228Z/resolved_config.yaml` | `exp/qualification_runtime/osnet_x0_5_onnx/20260826T003228Z/gallery.json` |

All selected metadata records git commit `fca2f27279e23bf7ddd12ea16f202f92786ca84b`, input-video SHA-256 `f86d05e023982b8c76c2e852e9419b53ef1c67fc6ab3b89af160176e5e66197b`, and detector weight path `edge_reid_runtime/weights/yolo26/yolo26n.pt`.

The checkpoint checksums were calculated during snapshot generation; they were not fields in the source run metadata:

| Artifact | SHA-256 |
|---|---|
| `edge_reid_runtime/weights/yolo26/yolo26n.pt` | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `exp/msmt17_no_label_smoothing/checkpoints/ckpt_best.pth` | `5c751a6a2f3d2b19456684d66c72b3f86ad8d8a7cf7ab061aacd2071720bfc3f` |
| `edge_reid_runtime/weights/onnx/resnet50_selected.onnx` | `d783736fc8d4040264151a497e3104bb7e99f458dec89c58fe71b1df51e60de9` |
| `edge_reid_runtime/weights/models--kaiyangzhou--osnet/snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth` | `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18` |
| `edge_reid_runtime/weights/onnx/osnet_x0_25.onnx` | `8f4e44f7053b8973914b850d994483df6654109d931ff8005dc4a25e26bf6a32` |
| `edge_reid_runtime/weights/models--kaiyangzhou--osnet/snapshots/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth` | `e96cbd20ee9cc3c6dcc0e8f4fbba8c8069d47647a5a96a59ce381fb785c54f68` |
| `edge_reid_runtime/weights/onnx/osnet_x0_5.onnx` | `bc78a6df12ccc5f30adab52ae16d88b530325ec5cd637217e97a86382fd691be` |

## 13. Raw Frozen Values

The following complete numerical values are reproduced from each selected source `summary.json` without rounding.

```json
{"experiment":"resnet50_selected_torch","processed_frames_total":1227,"warmup_frames":30,"measured_frames":1197,"elapsed_s":1334.4263606071472,"measured_elapsed_s":1328.0368884708441,"benchmark_fps":0.9013303850153399,"stages":{"total":{"mean":1109.4710847709641,"median":1039.2027639900334,"p95":2546.127418987453},"detector":{"mean":112.60322253253337,"median":98.26066499226727,"p95":189.27866299054585},"embedder":{"mean":961.7588847601868,"median":896.0413180175237,"p95":2414.1002550022677},"gallery":{"mean":0.18329271020051854,"median":0.1945659751072526,"p95":0.4036249883938581},"input":{"mean":3.346119324526724,"median":2.726426988374442,"p95":5.885328020667657},"tracker":{"mean":31.53714398080199,"median":37.71271600271575,"p95":85.81106801284477},"video_write":{"mean":0.0008562538514509735,"median":0.0006819900590926409,"p95":0.0010259973350912333}},"gallery_size":28,"rss_mean_mb":801.9639724310777,"rss_peak_mb":809.06640625}
{"experiment":"resnet50_selected_onnx","processed_frames_total":1227,"warmup_frames":30,"measured_frames":1197,"elapsed_s":525.3900623321533,"measured_elapsed_s":517.5407029687485,"benchmark_fps":2.312861564575879,"stages":{"total":{"mean":432.3648312186704,"median":436.3601289805956,"p95":820.5215990019497},"detector":{"mean":156.3114400649324,"median":154.53197399619967,"p95":230.5418070172891},"embedder":{"mean":244.10527941498827,"median":230.1263300178107,"p95":610.3847890044563},"gallery":{"mean":0.16149439972571705,"median":0.17337099416181445,"p95":0.370627996744588},"input":{"mean":3.0171294658358847,"median":2.766494988463819,"p95":4.561203997582197},"tracker":{"mean":28.72982957119701,"median":39.09284100518562,"p95":67.7912890096195},"video_write":{"mean":0.0007396580712868208,"median":0.000679021468386054,"p95":0.0011569936759769917}},"gallery_size":28,"rss_mean_mb":796.0368368838764,"rss_peak_mb":837.015625}
{"experiment":"osnet_x0_25_torch","processed_frames_total":1227,"warmup_frames":30,"measured_frames":1197,"elapsed_s":199.82929515838623,"measured_elapsed_s":195.0608032840537,"benchmark_fps":6.136548090888823,"stages":{"total":{"mean":162.95806456479005,"median":158.9578320272267,"p95":310.476121987449},"detector":{"mean":99.28123966697176,"median":90.67965298891068,"p95":128.17321298643947},"embedder":{"mean":33.43724121936576,"median":28.024152998114005,"p95":96.93338797660545},"gallery":{"mean":0.10030246525010873,"median":0.11041600373573601,"p95":0.19888300448656082},"input":{"mean":3.127054955825675,"median":2.6611700013745576,"p95":5.580502009252086},"tracker":{"mean":26.946709064893977,"median":39.98041898012161,"p95":65.23921800544485},"video_write":{"mean":0.0006408461903703517,"median":0.0006039917934685946,"p95":0.0009010254871100187}},"gallery_size":5,"rss_mean_mb":1205.811677631579,"rss_peak_mb":1231.02734375}
{"experiment":"osnet_x0_25_onnx","processed_frames_total":1227,"warmup_frames":30,"measured_frames":1197,"elapsed_s":265.50077080726624,"measured_elapsed_s":260.2451695173222,"benchmark_fps":4.59950900229995,"stages":{"total":{"mean":217.4145108749559,"median":222.39371101022698,"p95":346.7889739840757},"detector":{"mean":167.32472974553963,"median":163.29308997956105,"p95":252.36757198581472},"embedder":{"mean":16.856712942981684,"median":18.068545003188774,"p95":32.57123701041564},"gallery":{"mean":0.10314563327895301,"median":0.11747801909223199,"p95":0.2057619858533144},"input":{"mean":3.316344443136599,"median":2.8386259800754488,"p95":5.930369981797412},"tracker":{"mean":29.746725261742412,"median":40.663369989488274,"p95":80.16047699493356},"video_write":{"mean":0.0007316105339268445,"median":0.0006979971658438444,"p95":0.0010379881132394075}},"gallery_size":5,"rss_mean_mb":697.4587934680451,"rss_peak_mb":738.6640625}
{"experiment":"osnet_x0_5_torch","processed_frames_total":1227,"warmup_frames":30,"measured_frames":1197,"elapsed_s":232.18391704559326,"measured_elapsed_s":224.931457518338,"benchmark_fps":5.32162114275373,"stages":{"total":{"mean":187.91266292258814,"median":179.2930729861837,"p95":338.5768969892524},"detector":{"mean":102.161904345958,"median":94.21048700460233,"p95":133.81823900272138},"embedder":{"mean":50.94239521724403,"median":44.577664986718446,"p95":152.21506101079285},"gallery":{"mean":0.10748080222236434,"median":0.11469601304270327,"p95":0.21335098426789045},"input":{"mean":3.3730706631819856,"median":2.687892992980778,"p95":5.791228992165998},"tracker":{"mean":31.245511988266983,"median":40.05138698266819,"p95":75.24817099329084},"video_write":{"mean":0.0006905272799491907,"median":0.0006399932317435741,"p95":0.0009830109775066376}},"gallery_size":7,"rss_mean_mb":1207.0948823882622,"rss_peak_mb":1221.33984375}
{"experiment":"osnet_x0_5_onnx","processed_frames_total":1227,"warmup_frames":30,"measured_frames":1197,"elapsed_s":276.70169281959534,"measured_elapsed_s":272.43846073249006,"benchmark_fps":4.393652778619043,"stages":{"total":{"mean":227.60105324351719,"median":235.55261801811866,"p95":374.8784539930057},"detector":{"mean":157.87501084347315,"median":156.68930299580097,"p95":231.28185697714798},"embedder":{"mean":37.691661977626765,"median":37.06701900227927,"p95":82.36035500885919},"gallery":{"mean":0.106628460489811,"median":0.11314501171000302,"p95":0.19898300524801016},"input":{"mean":3.067406345860614,"median":2.743613993516192,"p95":5.625291989417747},"tracker":{"mean":28.82180394623378,"median":39.14025198901072,"p95":71.1432879907079},"video_write":{"mean":0.0006816499794964852,"median":0.0006399932317435741,"p95":0.0009739887900650501}},"gallery_size":7,"rss_mean_mb":704.0411412646199,"rss_peak_mb":748.55078125}
```
