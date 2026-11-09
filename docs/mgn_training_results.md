# MGN within-domain training results

Registered and artifact-verified on **2026-11-05**. **All four source trainings completed 120 epochs and their trainer-side final evaluations.** No common training process was found during host inspection. MSMT17, the last source in the sequence, finished its final evaluation on **2026-10-30 at 03:34:12 São Paulo time (UTC−03:00)**. Formal shell exit-code receipts were not saved; completion is established by checkpoint/scheduler states, all epoch markers, periodic metrics and final-evaluation artifacts.

Full-precision metrics, all eight checkpoint SHA-256 values, commands, configuration and artifact identities, periodic results and checks are in [mgn_training_results.yaml](mgn_training_results.yaml). The [eight-row CSV](../results/mgn_training_results.csv) keeps selected-best and final-epoch results separate. Tables below use percentages.

## Selected-best checkpoint results

| Dataset | Checkpoint epoch | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market1501 | 70 | 88.18 | 94.83 | 96.64 | 71.00 | 35.00 |
| DukeMTMC-ReID | 40 | 81.64 | 90.75 | 93.18 | 66.87 | 27.26 |
| CUHK03 detected / processed_partition | 120 | 46.07 | 66.00 | 75.43 | 45.86 | 34.45 |
| MSMT17 | 70 | 43.13 | 59.40 | 66.34 | 19.67 | 1.66 |

These are the saved `metrics/final_test.json` results from re-evaluating `ckpt_best.pth`. Best means the first strict maximum of periodic **source-test mAP**, evaluated every 10 epochs. There is no independent validation split; these are trainer-produced within-domain results, not newly executed independent benchmark evaluations.

## Final training state, epoch120

| Dataset | Checkpoint epoch | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market1501 | 120 | 88.03 | 94.74 | 96.56 | 70.74 | 34.70 |
| DukeMTMC-ReID | 120 | 81.37 | 90.35 | 93.00 | 66.03 | 25.95 |
| CUHK03 detected / processed_partition | 120 | 46.07 | 66.00 | 75.43 | 45.86 | 34.45 |
| MSMT17 | 120 | 42.49 | 58.86 | 66.09 | 19.21 | 1.62 |

These are the saved `metrics/final_epoch_test.json` results for `ckpt_last.pth`. Both terminal checkpoint and scheduler record 120 completed epochs for every source. CUHK03 selected-best and final-state metrics coincide because its selected epoch is120; the two checkpoint files remain separately identified.

## Configuration and provenance

All four runs record launch commit `9ac343ff3a36fcd858f270ef0688233bb2b4ffc0`. Its 57 runtime-file hashes match the active [frozen manifest](mgn_experiment_manifest.json), SHA-256 `b289ea8a1ee1ee03d2a2b41267e0286168241dcda3b07817d864df0f1d8f286e`. The underlying implementation anchor is `569e8d533c1a3b863c2681ad5898160c546364bc`; the later launch commit also includes documentation. The prior freeze/history remains intact. Each preset, resolved run configuration and saved checkpoint configuration agrees after normalizing only the output path and operational resume value. Current runtime and six qualification-receipt hashes were also checked. No scientific configuration was revised during registration.

The fixed project recipe is MGN `paper_v3_global2048_independent_v1`, three residual branches, max pooling, eight independent 256-D reductions and 2048-D concatenated retrieval descriptor; three global 2048-D and five local 256-D bias-free CE heads; mean eight-head CE plus mean three-feature batch-hard Triplet, margin 1.2 and family weights 1/1. SGD LR 0.01/momentum 0.9/weight-decay 0.0005, 120 epochs, no warmup, exact completed-epoch decays after 60/90, P16/K4/B64, FP32 and seed 42. Training uses RGB 384×128 bicubic resize, ImageNet normalization, horizontal flip, and no random erasing. Retrieval uses raw original/flip mean followed by global L2 normalization, Euclidean distance, no reranking. Details and scientific caveats are retained in the [protocol](mgn_experiment_protocol.md).

| Dataset | Experiment / YAML | Source identities | Train images | Training partition | Final-evaluation log time (São Paulo) |
| --- | --- | ---: | ---: | --- | --- |
| Market1501 | `mgn_market1501` / [market1501.yaml](../configs/mgn/market1501.yaml) | 751 | 12936 | processed / trainval | 2026-10-28 13:32:42,238 |
| DukeMTMC-ReID | `mgn_duke` / [duke.yaml](../configs/mgn/duke.yaml) | 702 | 16522 | processed / trainval | 2026-10-28 21:50:09,725 |
| CUHK03 detected / processed_partition | `mgn_cuhk03` / [cuhk03.yaml](../configs/mgn/cuhk03.yaml) | 767 | 7365 | detected processed_partition / trainval | 2026-10-29 00:39:41,560 |
| MSMT17 | `mgn_msmt17` / [msmt17.yaml](../configs/mgn/msmt17.yaml) | 1041 | 30248 | raw list_train.txt only | 2026-10-30 03:34:12,617 |

The launch receipts identify the same qualified TITAN Xp UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, Reid Python3.11.2 / Torch2.7.1+cu126. Launch commands and full saved environment-file hashes are in the YAML receipt. Dataset identities are bound to launch-time admission metadata, identity maps, inventories and consumed-smoke bytes; this registration does not rehash or decode the full image corpora.

### Recorded launch commands

- Market1501: `python scripts/train.py --config configs/mgn/market1501.yaml --opts train.save.resume=exp/mgn/market1501/checkpoints/ckpt_last.pth`.
- DukeMTMC-ReID: `python scripts/train.py --config configs/mgn/duke.yaml`.
- CUHK03 detected / processed_partition: `python scripts/train.py --config configs/mgn/cuhk03.yaml`.
- MSMT17: `python scripts/train.py --config configs/mgn/msmt17.yaml`.

For Market1501 this is the continuation command; its original fresh command was `python scripts/train.py --config configs/mgn/market1501.yaml` and remains in the preserved archive.

## Market1501 interruption and continuation

Market1501 completed epoch 86, then its first launch stopped during epoch 87 (last logged batch100/182) without a saved traceback. The cause is unverified. The next launch resumed the epoch 86 checkpoint and completed epochs 87–120. Original checkpoints and launch/config/log/metric records (23 files total) were preserved and checksum-verified in [the resume archive](../exp/mgn/market1501/artifacts/resume_archive/20261028T141741Z/archive_manifest.json); every archived file was rechecked during this registration. Best epoch 70 was preserved through continuation.

Resume restores model, optimizer, epoch-based scheduler and historical selection, but not sampler/RNG trajectory. The repeated partial epoch 87 is additional exposure. Do not claim a bitwise uninterrupted run or reuse the fresh index-only update forecast as its measured total updates. Ordinary checkpoints do not provide an authoritative successful-update counter; completed epochs are verified separately from optimization-step counts.

## Checkpoint artifact identities

| Source | Role | Epoch | SHA-256 |
| --- | --- | ---: | --- |
| Market1501 | [ckpt_best.pth](../exp/mgn/market1501/checkpoints/ckpt_best.pth) | 70 | `5067cb70fb2e3fe7609ca7338487c44f62699596088852abe060c9672ebe8024` |
| Market1501 | [ckpt_last.pth](../exp/mgn/market1501/checkpoints/ckpt_last.pth) | 120 | `f6d1a766ed51cb69ce668cee49b68f4bb9cf5c05896dce7204401fc5e3810343` |
| DukeMTMC-ReID | [ckpt_best.pth](../exp/mgn/duke/checkpoints/ckpt_best.pth) | 40 | `97b7b674d85806f32f14d76d1a3136b05b7083bc4833eb3579db453139df1641` |
| DukeMTMC-ReID | [ckpt_last.pth](../exp/mgn/duke/checkpoints/ckpt_last.pth) | 120 | `58f1abca506ae273b7e067ee0775c556f9929b61129fe3a19b3656bbcea9b090` |
| CUHK03 detected / processed_partition | [ckpt_best.pth](../exp/mgn/cuhk03/checkpoints/ckpt_best.pth) | 120 | `a936f3c48224225aa76a3216c0a2cc2c13609c0a51c14e12ad5893c44ad9b23a` |
| CUHK03 detected / processed_partition | [ckpt_last.pth](../exp/mgn/cuhk03/checkpoints/ckpt_last.pth) | 120 | `8167c18ce5f3ddc2ab19e04d8cbc83b0c676261dcd5bd54327f63c8768f1e8bf` |
| MSMT17 | [ckpt_best.pth](../exp/mgn/msmt17/checkpoints/ckpt_best.pth) | 70 | `01c96982d0aff8ada9c2a1286d8177a534e1800a44b213f1a6c5eb57ef72a345` |
| MSMT17 | [ckpt_last.pth](../exp/mgn/msmt17/checkpoints/ckpt_last.pth) | 120 | `00f5dd1c35720c549fa6cf3bcdd1b948cab7f3ab8f18b16392db1288c0608b8b` |

## Verification and limits

- Verified eight checkpoints on CPU with `weights_only=True` and memory mapping; no model initialization or inference. Source class counts and reconstruction metadata match the source presets; optimizer state is present, scheduler cadence/milestones/epoch and selection history agree.
- Verified all 120 completed-epoch log markers and all 12 periodic evaluations for each source (48 total), source/model/split/checkpoint labels, final-evaluation log markers and final metric files. Every metric is finite/in-range and Rank-1≤Rank-5≤Rank-10.
- Compared checkpoint scores, periodic best/final metrics and the final selected re-evaluation with tolerance 1e-5 in fractional units. Maximum observed difference: `2.980232238769531e-07` (Market1501 mINP); no material metric disagreement.
- CUHK03 is detected `processed_partition`, split_id=null. Semantic new/classic correspondence remains **unverified**; this report does not relabel it or transfer historical PCB split claims.
- MSMT17 training uses the official train list only; the separate validation list is not silently included. Last epoch average total loss is 1.4026; the final logged running Triplet term is 1.1919 against margin 1.2. This is an observed training diagnostic, not proof of a particular failure mode. No diagnostic rerun, retuning or recipe change was performed.
- Fixed 120 epochs do not equalize update budgets across datasets. These are single configured seed 42 runs; no uncertainty intervals, matched-budget architecture ranking or paper benchmark reproduction is claimed.
- Cross-domain, independent within-domain evaluation and LODO registration are separate from these trainer-produced results. MGN LODO remains mandatory for the offline dissertation study; none of those later workloads was executed here.
- The prior qualification receipts describe bounded three-update smoke tests; they are not overwritten or substituted for these official training results. No training, GPU experiment, benchmark evaluation, code/configuration/weight change, commit or push was performed during registration.

Work stops after documenting and presenting these observed results, as requested.
