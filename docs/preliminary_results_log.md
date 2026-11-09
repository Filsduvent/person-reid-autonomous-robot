# Preliminary Results Log

This file is the working registry for the preliminary results chapter.

It has two goals:

1. keep one clean record of experiments already completed
2. give us one place to append the next results as new training runs finish

Date of consolidation: 2026-08-18

## Current Experimental Stage

All four PCB source trainings (Market-1501, DukeMTMC-ReID, CUHK03 detected/new protocol and MSMT17) completed120 epochs and are registered below. All record commit3b015c9; the original exact-commit freeze exception remains under review. Independent evaluation is complete and registered: all16 source-target pairs are verified (four diagonals and twelve cross-domain cells). Previous ResNet50 records remain below as historical context.

## PCB selected-checkpoint evaluation matrix

All four source rows are complete and verified: **16/16 cells**, including4/4 independent self-evaluations and12/12 cross-domain evaluations. See [the evaluation matrix](pcb_evaluation_matrix.md) and [full-precision CSV](../results/pcb_selected_matrix/matrix.csv). Market, Duke and MSMT17 use their epoch50 selected checkpoints; CUHK03 detected/new uses epoch80. Every source uses the same checkpoint across its row, without target-dependent selection.

MSMT17 mAP on Market/Duke/CUHK03/MSMT17 is0.26933884620666504/0.3446248173713684/0.11661786586046219/0.43399664759635925. Its independent diagonal differs from training by only5.960464477539063e-8 mAP; mINP and ranks match exactly. All five metrics, result-file hashes and source checkpoint hashes are preserved in the matrix and source-row verification YAMLs. Final checks confirmed16 unique cells, unchanged earlier result files, and all diagonals within2e-6 fractional units of their training evaluations. Registration followed the user's source-by-source order. The original commit-provenance exception remains documented; phase acceptance and Phase24 comparison are separate from this completed registration.

## PCB Market-1501 — Phase 21A completed training (registered 2026-10-08)

Experiment `pcb_market1501`, artifacts `exp/pcb/market1501/`. The user reported completion and requested result registration. Read-only verification confirms 120 epochs, 202 updates/epoch (24240 total), all12 periodic evaluations at epochs10–120, finite logged losses and checkpoint model/momentum tensors, and final selected-best evaluation. The complete hash/metric receipt is [pcb_market1501_results.yaml](pcb_market1501_results.yaml). At training registration, independent Phase22 evaluation had not yet been run; it is now registered in the completed matrix above.

| State | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected best | `exp/pcb/market1501/metrics/final_test.json` | 50 | 0.79541075 | 0.42386705 | 0.91923988 | 0.97209024 | 0.98277909 |
| Final training state | `exp/pcb/market1501/metrics/final_epoch_test.json` | 120 | 0.78556681 | 0.41416246 | 0.92072445 | 0.97001189 | 0.98099762 |

Metrics above are fractions. Epoch50 is the first and unique highest periodic mAP (0.7954107522964478); its saved score equals `val_epoch_050.json` and selected-best `final_test.json`. Epoch120's last checkpoint agrees with `latest_val.json`, `val_epoch_120.json` and `final_epoch_test.json`. Use **epoch50 `ckpt_best.pth`** for later within/cross-domain evaluation, not the epoch120 checkpoint.

- `exp/pcb/market1501/checkpoints/ckpt_best.pth`: epoch50, SHA256 `8ad7c820a31bb8adc9cebafb48d5e25cb6c042401d44fdbe8084ee587317fd38`, 222,993,291 bytes; scheduler at 10100 updates.
- `exp/pcb/market1501/checkpoints/ckpt_last.pth`: epoch120, SHA256 `c82b325b7a7d5f0735b45da0c2d374f4cbd1b92cc8c8e01f42705b2ee1d0b9d6`, 222,993,291 bytes; scheduler at 24240 updates.

Command recorded: `scripts/train.py --config configs/pcb/market1501.yaml` (no overrides, resume empty). Saved config exactly matches the Phase20 resolved configuration: independent-part PCB, C=751,1536-D embedding, historical ImageNet initialization,384×128, random batch64, FP32, seed42, six summed unsmoothed identity losses, SGD, backbone/new-layer rates0.01/0.1 then0.001/0.01 after epoch40,120 epochs. Logs confirm the LR boundary and TITAN Xp cuda:0. Start/end log timestamps are `2026-10-07 16:41:38,704` / `2026-10-07 19:52:55,338`, elapsed3h11m16.634s; timezone is not recorded in these log lines.

**Commit exception pending review:** saved `git_commit.txt` records `3b015c964b99bfe8ad4b7842bd174d3bc58af607`, whereas Phase20 froze `66ea88d06ba0f79c298b19abe1652ae892d69fdd`. Git comparison shows only the three Phase20 documents and `plan.md` differ; committed production code and presets are identical. Do not rewrite the actual run provenance or claim exact-SHA compliance. The saved environment also lists both CUDA devices, rather than UUID-only visibility; actual training device is confirmed by the log. Shell exports and launch-time dirty state are not recorded. A requested decision on documenting this exception and using3b015c9 for the remaining sources is pending; no next training was launched by the registration agent.

**Contextual comparison, not a matched architecture ablation:** re-read `exp/baseline_r50_triplet_market1501/metrics/final_test.json` (selected best epoch110): mAP0.85789943, mINP0.59063596, Rank-1 0.94388360, Rank-5 0.98248219, Rank-10 0.98871732. PCB differs by −6.249 percentage points mAP, −16.677 mINP and −2.464 Rank-1. Both use the processed Market trainval/test protocol, but the reference uses256×128, PK sampling, augmentation including erasing, ID+triplet with label smoothing, BNNeck and Adam. This comparison does not isolate architecture. These freshly read values take precedence for this comparison over older rounded anchor rows elsewhere in this registry. No exactly matched ResNet50 control is established here.

Retain the common methodological limitation: periodic model selection uses the test query/gallery protocol. The measured decline after epoch50 supports preserving the selected checkpoint; it does not justify changing the frozen recipe mid-campaign. At Market registration, Duke was the next dataset and its output was absent; Duke has since completed and is registered below. The exact manual command remains in the experiment protocol, but commit-exception review and a clean published documentation state are needed before following the guarded handoff. No training/evaluation was executed for this registration.

## PCB DukeMTMC-ReID — Phase 21B completed training (registered 2026-10-08)

Experiment `pcb_duke`, artifacts `exp/pcb/duke/`. User reported completion and requested registration. Read-only verification confirms all 120 epochs, 258 updates/epoch (30960 total), all 12 periodic evaluations at epochs 10–120, finite logged epoch losses and checkpoint model/momentum tensors, and the final selected-best evaluation. Complete metrics, periodic history, hashes and provenance: [pcb_duke_results.yaml](pcb_duke_results.yaml). At training registration, independent Phase22 evaluation was pending; it is now registered in the completed matrix above.

| State | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected best | `exp/pcb/duke/metrics/final_test.json` | 50 | 0.70122021 | 0.29280707 | 0.84739679 | 0.92010772 | 0.93940753 |
| Final training state | `exp/pcb/duke/metrics/final_epoch_test.json` | 120 | 0.69410342 | 0.28643936 | 0.84201080 | 0.91876119 | 0.93895870 |

Metrics are fractions. Epoch50 is the earliest maximum of the 12 periodic mAP values; checkpoint scores, `val_epoch_050.json` and `final_test.json` agree exactly. Epoch120 checkpoint scores match `val_epoch_120.json`, `latest_val.json` and `final_epoch_test.json`. Use **epoch50 `ckpt_best.pth`** for subsequent within-domain and cross-domain evaluation, retaining the final-state checkpoint separately.

- `exp/pcb/duke/checkpoints/ckpt_best.pth`: epoch 50, SHA256 `53df33756029ad8a0213852de53afb28758322657d2781083cb43731249e57c3`, 222,388,875 bytes; scheduler at 12900 updates.
- `exp/pcb/duke/checkpoints/ckpt_last.pth`: epoch 120, SHA256 `b040a2decba55d99d538d1a6e0a772bcaf912a5c3ff5383db31db17c073750f0`, 222,388,875 bytes; scheduler at 30960 updates.

Recorded command: `scripts/train.py --config configs/pcb/duke.yaml`; no overrides, empty resume. Saved config exactly matches the frozen resolved config: independent-part PCB, six classifiers with C=702,1536-D retrieval,384×128 input, random batch64, FP32, seed42, historical ImageNet initialization, six summed unsmoothed identity losses and SGD. Logs confirm backbone/new-layer LR0.01/0.1 through epoch40 and0.001/0.01 from epoch41; actual execution on TITAN Xp cuda:0. Recorded start/end: `2026-10-08 15:04:05,714` / `2026-10-08 19:02:38,545`; elapsed3h58m32.831s. The log does not record a timezone. No training, forward evaluation or independent Phase22 evaluation was performed during registration.

**Provenance:** actual saved commit is `3b015c964b99bfe8ad4b7842bd174d3bc58af607`, the same as Market. Original frozen commit remains `66ea88d06ba0f79c298b19abe1652ae892d69fdd`. The committed-tree difference is confined to Phase20 documents and `plan.md`; production code and presets are identical. The previously requested exception decision remains unresolved; reporting completion does not silently revise the freeze. Two GPUs were visible, while the log confirms TITAN Xp cuda:0 was used. Shell exports and launch-time dirty state were not captured. Current Duke and CUHK preset/data-provenance hashes and historical initialization hash match the freeze. This does not certify the historical contents of every image.

**Contextual comparison:** freshly read `exp/baseline_r50_triplet_duke/metrics/final_test.json`, best epoch120: mAP0.76968551, mINP0.41036296, Rank-1 0.86983842, Rank-5 0.94524235, Rank-10 0.96588868. PCB is lower by6.847 percentage points mAP,11.756 mINP and2.244 Rank-1. The dataset uses the same processed trainval/test protocol, but reference training differs in resolution (256×128), PK sampling, padding/crop/erasing, ID+triplet with label smoothing, BNNeck, Adam and schedule. This is not a matched architecture-only comparison; no causal architecture conclusion follows.

Checkpoint selection still uses test query/gallery mAP rather than independent validation. Preserve the selected-best model and common recipe. At Duke registration, the next source was **CUHK03 detected/new protocol** and its output was absent; CUHK03 has since completed and is registered below. For historical launch instructions, the existing guarded command is in [the protocol](pcb_experiment_protocol.md#cuhk03--detectednew-protocol), subject to resolving the documented commit choice and committing pending records so its clean-checkout guard can pass. No next run was started.

## PCB CUHK03 detected/new protocol — Phase 21C completed training (registered 2026-10-09)

Experiment `pcb_cuhk03`, artifacts `exp/pcb/cuhk03/`. User reported completion and requested registration. Read-only verification confirms 120 epochs,115 updates/epoch (13800 total), all12 periodic evaluations at epochs10–120, finite logged epoch losses and checkpoint model/momentum tensors, and final selected-best evaluation. Full metrics, periodic history, hashes and provenance: [pcb_cuhk03_results.yaml](pcb_cuhk03_results.yaml). At training registration, independent Phase22 evaluation was pending; it is now registered in the completed matrix above.

| State | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected best | `exp/pcb/cuhk03/metrics/final_test.json` | 80 | 0.59661323 | 0.46582380 | 0.63357145 | 0.78642857 | 0.85285717 |
| Final training state | `exp/pcb/cuhk03/metrics/final_epoch_test.json` | 120 | 0.59481984 | 0.46014640 | 0.64285713 | 0.78214288 | 0.84928572 |

Metrics are fractions. Epoch80 is the earliest maximum of the12 periodic mAP values. Its checkpoint scores equal `val_epoch_080.json` and `final_test.json`; epoch120 scores equal `val_epoch_120.json`, `latest_val.json` and `final_epoch_test.json`. Use **epoch80 `ckpt_best.pth`** for subsequent within/cross-domain evaluation. Epoch120 has higher Rank-1 but lower mAP; selection remains by the frozen mAP rule.

- `exp/pcb/cuhk03/checkpoints/ckpt_best.pth`: epoch80, SHA256 `05d33f32793568313ce22d404b213620009b72a17ed0025fb34222173186078a`, 223,190,859 bytes; scheduler at 9200 updates.
- `exp/pcb/cuhk03/checkpoints/ckpt_last.pth`: epoch120, SHA256 `2ceb172362dc8ee671689f4ba0409c6a24cd29e6455da3c4fdd97921c94d311a`, 223,190,859 bytes; scheduler at 13800 updates.

Recorded command: `scripts/train.py --config configs/pcb/cuhk03.yaml`, no overrides, resume empty. Saved config exactly matches the Phase20 resolved config. Protocol is **detected images, new protocol**, using the processed_partition parser and null split_id. Phase16 provenance (MAT, split pickle, JSON, detected source data) remains the scientific identity; relevant metadata hashes were refreshed, without repeating the full image audit. Counts from the accepted freeze:767 train IDs/7365 train images,1400 query/5332 gallery.

Independent-part PCB, six C=767 classifiers,1536-D embedding, historical ImageNet initialization,384×128, random batch64, FP32, seed42, summed unsmoothed identity loss and SGD are unchanged. Logs verify backbone/new-layer LR0.01/0.1 through epoch40 and0.001/0.01 from epoch41; actual TITAN Xp cuda:0. Recorded start/end: `2026-10-08 21:59:53,821` / `2026-10-08 23:45:12,627`, elapsed1h45m18.806s; timezone is not in the log. No training or independent evaluation was run during registration.

**Provenance:** saved commit `3b015c964b99bfe8ad4b7842bd174d3bc58af607` matches Market/Duke; original freeze specifies `66ea88d06ba0f79c298b19abe1652ae892d69fdd`. Git comparison confirms documentation-only differences, with identical committed production code/presets. Existing exception decision remains unresolved; the freeze was not silently revised. Saved environment lists two visible GPUs, actual TITAN Xp is confirmed in the log; launch-time dirty state and shell exports cannot be retrospectively certified. Current CUHK/MSMT preset/provenance hashes and historical weight hash match.

**Contextual comparison:** freshly read `exp/baseline_r50_triplet_cuhk03_detected/metrics/final_test.json`, selected best epoch120: mAP0.56104255, mINP0.43791610, Rank-1 0.57571429, Rank-5 0.76928574, Rank-10 0.84714288. PCB improves by3.557 percentage points mAP,2.791 mINP and5.786 Rank-1 relative to this registry anchor. The reference shares the detected processed protocol but differs in resolution, PK sampling, padding/crop/erasing, ID+triplet/label smoothing, BNNeck, Adam and schedule. This is not an otherwise matched architecture ablation or a claim to outperform all ResNet50 variants.

Retain the methodological limitation: periodic checkpoint selection uses the test query/gallery protocol. Preserve selected-best and final-state checkpoints separately. At CUHK03 registration, MSMT17 was the remaining source: its existing log had reached epoch2, with no final metrics then. MSMT17 has since completed and is registered below. This is evidence of a started run, not certified completion or process liveness; do not launch another fresh run into that directory. Monitor using `tail -F exp/pcb/msmt17/train.log`; register its results after completion. No MSMT17 restart, resume or training was executed here.

## PCB MSMT17 — Phase 21D completed training (registered 2026-10-11)

Experiment `pcb_msmt17`, artifacts `exp/pcb/msmt17/`. User reported completion and requested registration. Read-only verification confirms all120 epochs,472 updates/epoch (56640 total), all12 periodic evaluations at epochs10–120, finite logged epoch losses and checkpoint model/momentum tensors, and final selected-best evaluation. Complete metrics, periodic history, hashes and provenance: [pcb_msmt17_results.yaml](pcb_msmt17_results.yaml). At training registration, independent Phase22 evaluation was pending; it is now registered in the completed matrix above.

| State | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected best | `exp/pcb/msmt17/metrics/final_test.json` | 50 | 0.43399659 | 0.06835141 | 0.70563513 | 0.82622868 | 0.86705548 |
| Final training state | `exp/pcb/msmt17/metrics/final_epoch_test.json` | 120 | 0.41395065 | 0.06173285 | 0.68968177 | 0.81919545 | 0.85950768 |

Metrics are fractions. Epoch50 is the earliest maximum of the12 periodic mAP values. Its checkpoint scores equal `val_epoch_050.json` and `final_test.json`. Epoch120 scores equal `val_epoch_120.json`, `latest_val.json` and `final_epoch_test.json`. Use **epoch50 `ckpt_best.pth`** for later within/cross-domain evaluation; retain `ckpt_last.pth` separately as the final training state.

- `exp/pcb/msmt17/checkpoints/ckpt_best.pth`: epoch50, SHA256 `5657281a3750b63f1335d63e1bdde387a82c6d8bc5708c60d9ef20a81efc6914`, 226,571,403 bytes; scheduler at 23600 updates.
- `exp/pcb/msmt17/checkpoints/ckpt_last.pth`: epoch120, SHA256 `d9ce09debf72e99072d7e9a15d1f68e70c13ac39ba0bdaa8f1f6232ab27cda91`, 226,571,403 bytes; scheduler at 56640 updates.

Recorded command: `scripts/train.py --config configs/pcb/msmt17.yaml`, no overrides, resume empty. Saved configuration exactly matches the Phase20 resolved config. MSMT17_V2 uses raw official `list_train.txt` only; `list_val.txt` is excluded. Frozen counts:1041 train IDs/30248 train images,11659 query/82161 gallery. The current preset, all four official list hashes and historical ImageNet weight hash match the freeze. This is a **project extension of the selected Huang recipe**, not reproduction of a Huang-reported MSMT17 experiment.

Independent-part PCB, six C=1041 classifiers,1536-D embedding,384×128 input, random batch64, FP32, seed42, summed unsmoothed identity loss and SGD are unchanged. Logs confirm backbone/new-layer LR0.01/0.1 through epoch40 and0.001/0.01 from epoch41; actual TITAN Xp cuda:0. Recorded start/end: `2026-10-09 07:14:44,177` / `2026-10-09 17:54:09,280`, elapsed10h39m25.103s; log timezone unspecified. No training or independent evaluation was performed during registration.

**Provenance:** saved commit `3b015c964b99bfe8ad4b7842bd174d3bc58af607` now matches all four runs. Original freeze specifies `66ea88d06ba0f79c298b19abe1652ae892d69fdd`; only Phase20 documentation and plan differ between committed trees, with production code/presets identical. The previously documented exception remains unresolved; four matching actual commits do not retrospectively change the original freeze. Environment records two visible GPUs; the log confirms TITAN Xp cuda:0, but shell exports and launch-time dirty state were not captured. No full image-content audit was repeated.

**Contextual comparison:** freshly read `exp/baseline_r50_triplet_msmt17/metrics/final_test.json`, selected best epoch120: mAP0.52377164, mINP0.11697485, Rank-1 0.75598252, Rank-5 0.86199504, Rank-10 0.89475942. PCB is lower by8.978 percentage points mAP,4.862 mINP and5.035 Rank-1. Both configurations use raw MSMT17 train/test and120 epochs, but the reference differs in256×128 resolution, PK sampling, augmentation, ID+triplet/label smoothing, BNNeck, Adam and schedule. This is not an otherwise matched architecture-only comparison; no causal architecture conclusion follows.

Periodic checkpoint selection uses test query/gallery mAP, not independent validation. All four source trainings are now registered; the next planned step is Phase22 independent selected-best evaluation after review and resolution of the recorded provenance exception. No further training, evaluation, commit or push was executed here.

## PCB source-training summary

These are trainer-produced selected-best evaluations, not the independently rerun Phase22 results. All four runs completed120 epochs and recorded commit3b015c9; selected epochs differ. Values are fractions.

| Source | Best epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Receipt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Market-1501 | 50 | 0.79541075 | 0.42386705 | 0.91923988 | 0.97209024 | 0.98277909 | [receipt](pcb_market1501_results.yaml) |
| DukeMTMC-ReID | 50 | 0.70122021 | 0.29280707 | 0.84739679 | 0.92010772 | 0.93940753 | [receipt](pcb_duke_results.yaml) |
| CUHK03 detected/new | 80 | 0.59661323 | 0.46582380 | 0.63357145 | 0.78642857 | 0.85285717 | [receipt](pcb_cuhk03_results.yaml) |
| MSMT17 | 50 | 0.43399659 | 0.06835141 | 0.70563513 | 0.82622868 | 0.86705548 | [receipt](pcb_msmt17_results.yaml) |

## Cross-Dataset Baseline Anchors

These are the strongest completed baseline anchors already present in `exp/`.
All mINP values in this registry use the corrected last-valid-positive-rank
calculation and were verified against their listed metric artifact.

| Dataset | Experiment | Artifact | mAP | mINP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Market1501 | `baseline_r50_triplet_market1501` | `exp/baseline_r50_triplet_market1501/metrics/final_test.json` | 0.8578 | 0.5906 | 0.9439 | 0.9813 | 0.9884 |
| DukeMTMC-ReID | `baseline_r50_triplet_duke` | `exp/baseline_r50_triplet_duke/metrics/final_test.json` | 0.7697 | 0.4104 | 0.8698 | 0.9452 | 0.9659 |
| CUHK03 detected | `baseline_r50_triplet_cuhk03_detected` | `exp/baseline_r50_triplet_cuhk03_detected/metrics/final_test.json` | 0.5610 | 0.4379 | 0.5757 | 0.7693 | 0.8471 |
| MSMT17 | `baseline_r50_triplet_msmt17` | `exp/baseline_r50_triplet_msmt17/metrics/final_test.json` | 0.5238 | 0.1170 | 0.7560 | 0.8620 | 0.8948 |

## Market1501 Loss-Mode Sweep Results

These are the main completed `Market1501` comparison runs already available.

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Smoke | `smoke_market1501_resnet50` | `exp/smoke_market1501_resnet50/metrics/final_test.json` | 1 | 0.0347 | 0.0053 | 0.0766 | 0.1876 | 0.2604 | one-epoch smoke only |
| One-epoch manual | `manual_market1501_one_epoch` | `exp/manual_market1501_one_epoch/metrics/final_test.json` | 1 | 0.0349 | 0.0055 | 0.0811 | 0.1915 | 0.2613 | one-epoch manual sanity run |
| Triplet only | `market1501_r50_triplet_only_official_experimental_settings` | `exp/market1501_r50_triplet_only_official_experimental_settings/metrics/final_test.json` | 30 | 0.0012 | 0.0010 | 0.0006 | 0.0048 | 0.0086 | clearly non-viable in current setup |
| ID only | `market1501_r50_id_only_official_experimental_settings` | `exp/market1501_r50_id_only_official_experimental_settings/metrics/final_test.json` | 120 | 0.8420 | 0.5601 | 0.9385 | 0.9783 | 0.9852 | strong classification-only baseline |
| Triplet + ID | `market1501_r50_triplet_id_official_experimental_settings` | `exp/market1501_r50_triplet_id_official_experimental_settings/metrics/final_test.json` | 120 | 0.8590 | 0.5916 | 0.9448 | 0.9837 | 0.9896 | main strong-baseline family |
| Triplet + ID + Center | `market1501_r50_triplet_id_center_official_experimental_settings` | `exp/market1501_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json` | 120 | 0.8604 | 0.5954 | 0.9436 | 0.9843 | 0.9902 | best completed base result in final-test artifacts |
| Triplet + ID + Center + rerank eval | `market1501_r50_triplet_id_center_rerank` | `exp/market1501_r50_triplet_id_center_rerank/metrics/latest_val.json` | 120 | 0.8626 | 0.9625 | 0.9463 | 0.9843 | 0.9887 | rerank mAP 0.9426, rerank Rank-1 0.9561 |

## Market1501 Phase 2 Results

These runs start the structure/ablation comparison around the strong baseline.

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `market1501_full_bag_of_tricks` | `exp/market1501_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.8583 | 0.5895 | 0.9427 | 0.9828 | 0.9893 | first Phase 2 reference run; slightly below the best center-loss variants |
| BNNeck off ablation | `market1501_bnneck_off` | `exp/market1501_bnneck_off/metrics/final_test.json` | 120 | 0.0009 | 0.0011 | 0.0003 | 0.0024 | 0.0033 | catastrophic collapse; BNNeck is essential in this setup |
| Stride-2 ablation | `market1501_stride2` | `exp/market1501_stride2/metrics/final_test.json` | 120 | 0.8457 | 0.5636 | 0.9391 | 0.9828 | 0.9896 | clearly worse than stride-1 reference; keep stride 1 in the canonical recipe |
| No random erasing ablation | `market1501_no_random_erasing` | `exp/market1501_no_random_erasing/metrics/final_test.json` | 120 | 0.8372 | 0.5579 | 0.9474 | 0.9798 | 0.9881 | lower mAP than reference; random erasing still helps retrieval quality overall |
| No label smoothing ablation | `market1501_no_label_smoothing` | `exp/market1501_no_label_smoothing/metrics/final_test.json` | 120 | 0.8600 | 0.5948 | 0.9397 | 0.9819 | 0.9881 | slightly higher mAP than reference, but lower Rank-1 and lower than center-loss winner on mAP |
| Full Bag-of-Tricks + Center | `market1501_r50_triplet_id_center_official_experimental_settings` | `exp/market1501_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json` | 120 | 0.8604 | 0.5954 | 0.9436 | 0.9843 | 0.9902 | equivalent coverage of the Phase 2 center-enabled variant already exists in repo |

## Additional Results Worth Keeping Separate

These are useful anchors but should not be mixed directly with the main final-test
table without noting their status.

| Experiment | Artifact | Status | Key Result |
| --- | --- | --- | --- |
| `market1501_r50_triplet_id` | `exp/market1501_r50_triplet_id/metrics/latest_val.json` | partial/intermediate | epoch 60, mAP 0.8360, Rank-1 0.9376 |
| `market1501_r50_triplet_id_center` | `exp/market1501_r50_triplet_id_center/metrics/latest_val.json` | latest-val anchor | epoch 120, mAP 0.8609, Rank-1 0.9463 |
| `msmt17_r50_triplet_id_center_official_experimental_settings` | `exp/msmt17_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json` | additive ablation | mAP 0.5300, Rank-1 0.7575 |

## Current Interpretation

- `Market1501` Phase 1 is effectively already populated with usable evidence.
- `triplet_only` is not competitive in this framework.
- `id_only` is strong, but both `triplet + id` and `triplet + id + center` are better.
- the strongest completed `Market1501` base run so far is `triplet + id + center`.
- the new `market1501_full_bag_of_tricks` reference is essentially aligned with the plain `triplet + id` family, but does not beat the center-loss variants.
- removing `BNNeck` is decisively invalid for this baseline; it should stay in the canonical recipe.
- switching to `last_conv_stride=2` degrades `mAP` and `Rank-1` versus the stride-1 reference, so stride 1 remains the correct baseline choice.
- removing random erasing improves `Rank-1` slightly but hurts `mAP` materially, so random erasing should stay in the canonical recipe unless final priorities shift toward top-1 only.
- removing label smoothing gives a tiny `mAP` increase over the plain reference but loses `Rank-1` and still does not beat the center-loss variant overall.
- the strongest completed `Market1501` recipe in the current record is still the center-enabled strong baseline.
- reranking provides a very large retrieval boost, but it should remain analysis-only.

## Market1501 Freeze Decision

`Market1501` is sufficiently covered to move on.

Current canonical winner to carry forward:

- base training recipe: `Triplet + ID + BNNeck + stride 1 + random erasing`
- strongest completed additive variant: `center.enabled=true`
- winning artifact anchor:
  `exp/market1501_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json`

## Next Planned Dataset

The next dataset should be `DukeMTMC-ReID`, reproducing the same controlled
recipe family used on `Market1501`.

## DukeMTMC-ReID Phase 2 Results

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `duke_full_bag_of_tricks` | `exp/duke_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.7707 | 0.4130 | 0.8667 | 0.9443 | 0.9641 | Duke reference run; essentially matches the older Duke baseline anchor |
| BNNeck off ablation | `duke_bnneck_off` | `exp/duke_bnneck_off/metrics/final_test.json` | 120 | 0.0009 | 0.0009 | 0.0000 | 0.0036 | 0.0036 | catastrophic collapse; BNNeck is essential on Duke too |
| Stride-2 ablation | `duke_stride2` | `exp/duke_stride2/metrics/final_test.json` | 120 | 0.7574 | 0.3939 | 0.8546 | 0.9327 | 0.9556 | clearly worse than the stride-1 Duke reference; keep stride 1 |
| No random erasing ablation | `duke_no_random_erasing` | `exp/duke_no_random_erasing/metrics/final_test.json` | 120 | 0.7206 | 0.3634 | 0.8469 | 0.9237 | 0.9479 | materially worse than reference across all reported retrieval metrics; retain random erasing |
| No label smoothing ablation | `duke_no_label_smoothing` | `exp/duke_no_label_smoothing/metrics/final_test.json` | 120 | 0.7741 | 0.4045 | 0.8797 | 0.9448 | 0.9605 | improves mAP, mINP, and Rank-1 over the reference; label smoothing is not beneficial in this Duke run |
| Full Bag-of-Tricks + Center | `duke_full_bag_of_tricks_center` | `exp/duke_full_bag_of_tricks_center/metrics/final_test.json` | 120 | 0.7688 | 0.4154 | 0.8654 | 0.9443 | 0.9618 | slightly below the no-center Duke reference; center loss is not retained for the Duke recipe |

## DukeMTMC-ReID Current Interpretation

- the Duke full strong-baseline reference is now registered and stable.
- its result is effectively aligned with the earlier `baseline_r50_triplet_duke` anchor.
- removing `BNNeck` is decisively invalid on Duke as well.
- switching to `last_conv_stride=2` degrades Duke on both `mAP` and `Rank-1`, so stride 1 stays locked here as well.
- removing random erasing substantially degrades every reported Duke metric, so random erasing stays in the Duke recipe.
- removing label smoothing improves `mAP` by 0.0034 and `Rank-1` by 0.0130 over the plain Duke reference; this is the current strongest completed Duke variant.
- adding center loss produces mAP 0.7688 and Rank-1 0.8654, slightly below the no-center reference; it is not beneficial on Duke in this run.
- the planned Duke sweep is complete. The current Duke winner is `duke_no_label_smoothing`.

## CUHK03 Detected Phase 2 Results

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `cuhk03_full_bag_of_tricks` | `exp/cuhk03_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.5602 | 0.4409 | 0.5657 | 0.7714 | 0.8500 | matched no-center reference; broadly aligned with the prior CUHK03 detected baseline anchor |
| Full Bag-of-Tricks + Center | `cuhk03_full_bag_of_tricks_center` | `exp/cuhk03_full_bag_of_tricks_center/metrics/final_test.json` | 120 | 0.5698 | 0.4499 | 0.5814 | 0.7657 | 0.8457 | first completed CUHK03 Phase 2 run; exceeds the prior CUHK03 detected baseline anchor on mAP and Rank-1 |
| No label smoothing ablation | `cuhk03_no_label_smoothing` | `exp/cuhk03_no_label_smoothing/metrics/final_test.json` | 120 | 0.6352 | 0.5119 | 0.6571 | 0.8271 | 0.8914 | strongest completed CUHK03 run; substantially improves every reported metric over the matched reference |
| No label smoothing replication (seed 1337) | `cuhk03_no_label_smoothing_seed1337` | `exp/cuhk03_no_label_smoothing_seed1337/metrics/final_test.json` | 120 | 0.6439 | 0.5136 | 0.6757 | 0.8386 | 0.9029 | independently seeded replication; confirms the high no-label-smoothing result and is the strongest CUHK03 run so far |
| No random erasing ablation | `cuhk03_no_random_erasing` | `exp/cuhk03_no_random_erasing/metrics/final_test.json` | 120 | 0.5141 | 0.4016 | 0.5193 | 0.7221 | 0.8029 | worse than the reference across every reported metric; retain random erasing |
| Stride-2 ablation | `cuhk03_stride2` | `exp/cuhk03_stride2/metrics/final_test.json` | 120 | 0.5194 | 0.4003 | 0.5243 | 0.7336 | 0.8236 | worse than the stride-1 reference; keep stride 1 |

## CUHK03 Detected Current Interpretation

- `cuhk03_full_bag_of_tricks_center` completed successfully using `ckpt_best.pth` at epoch 120.
- `cuhk03_full_bag_of_tricks` completed successfully using `ckpt_best.pth` at epoch 120.
- in the matched comparison, center loss improves mAP from 0.5602 to 0.5698 and Rank-1 from 0.5657 to 0.5814.
- removing label smoothing produces a strong seed-42 result: mAP improves by 0.0750 and Rank-1 by 0.0914 versus the matched reference.
- the seed-1337 no-label-smoothing replication is even stronger (mAP 0.6439, Rank-1 0.6757), so the high result is reproducible across two no-label-smoothing seeds. To attribute the seed-1337 gain specifically to removing label smoothing, a seed-1337 label-smoothed reference is still needed as the matched control.
- removing random erasing reduces mAP by 0.0461 and Rank-1 by 0.0464 versus the matched reference; random erasing stays in the CUHK03 recipe.
- switching to stride 2 reduces mAP by 0.0409 and Rank-1 by 0.0414 versus the matched reference; stride 1 stays in the CUHK03 recipe.
- the CUHK03 recipe sweep is practically complete. A seed-1337 label-smoothed control remains an optional verification run, not a blocker for opening MSMT17.

## MSMT17 Phase 2 Results

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `msmt17_full_bag_of_tricks` | `exp/msmt17_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.5214 | 0.1177 | 0.7531 | 0.8607 | 0.8931 | fresh matched MSMT17 reference; aligned with the earlier MSMT17 baseline anchor |
| No label smoothing ablation | `msmt17_no_label_smoothing` | `exp/msmt17_no_label_smoothing/metrics/final_test.json` | 120 | 0.5432 | 0.1260 | 0.7669 | 0.8741 | 0.9049 | improves every reported metric over the matched reference; current strongest completed MSMT17 variant |

## MSMT17 Current Interpretation

- both initial MSMT17 Phase 2 runs completed successfully using `ckpt_best.pth` at epoch 120.
- removing label smoothing improves mAP by 0.0218 and Rank-1 by 0.0138 versus the matched reference, matching the positive direction observed on Duke and CUHK03.
- the remaining planned MSMT17 comparisons are random erasing, stride, and the matched center-loss variant.

## Update Rule For New Runs

After each new experiment finishes, append:

- experiment name
- exact config or override command
- dataset
- checkpoint used for reporting
- epoch
- mAP
- mINP
- Rank-1
- Rank-5
- Rank-10
- rerank metrics if applicable
- one short interpretation note


## MGN within-domain training registration — 2026-11-05

All four official MGN source trainings are **complete through epoch120 and trainer-side final evaluation**, verified from both checkpoints, scheduler states, all120 epoch-completion log markers per source, final/periodic metrics, resolved configuration and launch provenance. No training process remained active. This is registration of existing trainer-produced results; no independent evaluation was executed. Existing PCB/ResNet50/history above is preserved. See [MGN training report](mgn_training_results.md), [full-precision verification receipt](mgn_training_results.yaml) and [selected/final CSV](../results/mgn_training_results.csv).

### Selected-best source-test results (percentages)

| Dataset | Checkpoint epoch | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market1501 | 70 | 88.18 | 94.83 | 96.64 | 71.00 | 35.00 |
| DukeMTMC-ReID | 40 | 81.64 | 90.75 | 93.18 | 66.87 | 27.26 |
| CUHK03 detected / processed_partition | 120 | 46.07 | 66.00 | 75.43 | 45.86 | 34.45 |
| MSMT17 | 70 | 43.13 | 59.40 | 66.34 | 19.67 | 1.66 |

### Final-epoch120 results (percentages)

| Dataset | Checkpoint epoch | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market1501 | 120 | 88.03 | 94.74 | 96.56 | 70.74 | 34.70 |
| DukeMTMC-ReID | 120 | 81.37 | 90.35 | 93.00 | 66.03 | 25.95 |
| CUHK03 detected / processed_partition | 120 | 46.07 | 66.00 | 75.43 | 45.86 | 34.45 |
| MSMT17 | 120 | 42.49 | 58.86 | 66.09 | 19.21 | 1.62 |

All runs record commit9ac343ff3a36fcd858f270ef0688233bb2b4ffc0 and the frozen MGN FP32/SGD120 recipe, with exact epoch60/90 decays, P16/K4, original/flip mean before L2, Euclidean distance and no reranking. Best is selected by periodic source-test mAP, not an independent validation split. Exact YAMLs/commands, eight checkpoint hashes, data/config/environment evidence, full48 periodic metrics and final log times are in the linked receipt.

Market1501 resumed from completed epoch86 at epoch87 after a partial interrupted epoch; its original23 archived files/checkpoints were reverified, best epoch70 is preserved, and uninterrupted sampler/RNG recovery or exact total update parity is not claimed. CUHK03 remains detected processed_partition with semantic new/classic correspondence unverified. MSMT17 uses list_train.txt only; its final average loss1.4026 and last running Triplet term1.1919 are recorded without diagnosing or changing the recipe. Different datasets have different optimization exposure under120 epochs.

No existing checkpoint, experiment artifact, frozen configuration, qualification receipt or historical record was modified. Cross-domain and LODO results are separate and are not fabricated here. No further training/evaluation was launched; stop and await the user.

## MGN LODO implementation/qualification publication — 2026-11-09

MGN LODO v2 implementation readiness and the bounded Oneiros qualification are complete, documented in docs/mgn_lodo_phase20/ and docs/mgn_lodo_phase21/. Phase21 passed8 cases with24 physical optimizer entries; these short fixture checks are not official training or new retrieval results. The four official85,200-update folds remain user-operated. The user requested publishing the accumulated work to distribute folds across Oneiros and Constantine; Constantine qualification is separate. Historical MGN/PCB source and evaluation numbers above remain unchanged.
