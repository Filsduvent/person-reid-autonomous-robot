# ResNet50 and PCB: combined evaluation matrices

Prepared on 2026-10-26 from the saved independent evaluation artifacts: **16 ResNet50 cells + 16 PCB cells**.

**How to read:** rows are the training source; columns are the evaluation target. Each cell is **ResNet50 / PCB**, in **percent (%)**. Diagonal cells are within-domain results; the other cells are cross-domain results. Higher is better for every metric shown. CUHK03 means detected images, new protocol.

These tables use the selected ResNet50 matrix, not the earlier generic baseline anchors in the preliminary-results log. Each source uses one fixed `ckpt_best.pth` across all four targets. The values come from `cross_dataset.json`, not epoch120 metrics or training-time evaluations. Displayed values are rounded to two decimals; calculations use the original full-precision fractions.

## mAP

| Training source ↓ / evaluation target → | Market-1501 | DukeMTMC-ReID | CUHK03 detected/new | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | 86.08 / 79.54 | 18.42 / 22.63 | 4.32 / 7.12 | 3.34 / 3.32 |
| DukeMTMC-ReID | 22.04 / 27.54 | 77.41 / 70.12 | 6.36 / 6.46 | 5.28 / 5.28 |
| CUHK03 detected/new | 24.09 / 25.46 | 15.53 / 15.81 | 63.52 / 59.66 | 3.19 / 2.37 |
| MSMT17 | 28.96 / 26.93 | 33.77 / 34.46 | 13.20 / 11.66 | 54.32 / 43.40 |

## Rank-1

| Training source ↓ / evaluation target → | Market-1501 | DukeMTMC-ReID | CUHK03 detected/new | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | 94.36 / 91.92 | 32.50 / 39.54 | 4.07 / 6.36 | 10.21 / 10.70 |
| DukeMTMC-ReID | 48.66 / 55.94 | 87.97 / 84.74 | 6.14 / 6.36 | 16.27 / 16.85 |
| CUHK03 detected/new | 46.88 / 49.79 | 28.95 / 29.08 | 65.71 / 63.36 | 10.33 / 8.29 |
| MSMT17 | 55.29 / 52.67 | 50.76 / 54.08 | 12.93 / 11.64 | 76.70 / 70.56 |

## mINP

| Training source ↓ / evaluation target → | Market-1501 | DukeMTMC-ReID | CUHK03 detected/new | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | 59.54 / 42.39 | 2.49 / 3.33 | 2.02 / 3.50 | 0.15 / 0.16 |
| DukeMTMC-ReID | 2.41 / 3.37 | 40.45 / 29.28 | 2.99 / 2.70 | 0.16 / 0.16 |
| CUHK03 detected/new | 3.44 / 3.50 | 1.80 / 1.64 | 51.19 / 46.58 | 0.12 / 0.08 |
| MSMT17 | 4.91 / 4.23 | 6.22 / 6.23 | 7.97 / 6.40 | 12.60 / 6.84 |

## Rank-5

| Training source ↓ / evaluation target → | Market-1501 | DukeMTMC-ReID | CUHK03 detected/new | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | 98.46 / 97.21 | 46.50 / 54.80 | 9.00 / 15.93 | 16.84 / 17.75 |
| DukeMTMC-ReID | 64.58 / 71.53 | 94.48 / 92.01 | 12.93 / 15.36 | 25.53 / 27.09 |
| CUHK03 detected/new | 62.95 / 66.15 | 43.99 / 42.24 | 82.71 / 78.64 | 18.05 / 14.26 |
| MSMT17 | 71.64 / 71.11 | 67.55 / 69.21 | 24.21 / 22.00 | 87.41 / 82.62 |

## Rank-10

| Training source ↓ / evaluation target → | Market-1501 | DukeMTMC-ReID | CUHK03 detected/new | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| Market-1501 | 98.99 / 98.28 | 52.29 / 61.09 | 12.50 / 21.50 | 20.63 / 21.78 |
| DukeMTMC-ReID | 71.02 / 78.65 | 96.05 / 93.94 | 19.07 / 20.79 | 30.64 / 32.11 |
| CUHK03 detected/new | 68.88 / 72.21 | 51.08 / 47.94 | 89.14 / 85.29 | 22.50 / 17.58 |
| MSMT17 | 77.46 / 77.26 | 72.98 / 74.78 | 33.00 / 28.50 | 90.49 / 86.71 |

## Descriptive averages

Unweighted means over the **4 diagonal cells** or the **12 off-diagonal cells**, respectively. These are descriptive summaries, not significance tests or an architecture-only ranking.

| Evaluation group | Model | mAP (%) | Rank-1 (%) | mINP (%) | Rank-5 (%) | Rank-10 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Within-domain (4 cells) | ResNet50 | 70.33 | 81.19 | 40.94 | 90.76 | 93.67 |
| Within-domain (4 cells) | PCB | 63.18 | 77.65 | 31.27 | 87.62 | 91.05 |
| Cross-domain (12 cells) | ResNet50 | 14.87 | 26.92 | 2.89 | 38.65 | 44.34 |
| Cross-domain (12 cells) | PCB | 15.75 | 28.44 | 2.94 | 40.62 | 46.18 |

## Selected checkpoints

The ResNet50 sources use dataset-specific selected recipes. PCB uses the frozen independent-part recipe. These are the actual checkpoint paths named by each row's artifacts.

| Source | ResNet50 selected run | PCB selected checkpoint |
| --- | --- | --- |
| Market-1501 | [`exp/market1501_r50_triplet_id_center_official_experimental_settings/checkpoints/ckpt_best.pth`](../exp/market1501_r50_triplet_id_center_official_experimental_settings/checkpoints/ckpt_best.pth) | [`exp/pcb/market1501/checkpoints/ckpt_best.pth`](../exp/pcb/market1501/checkpoints/ckpt_best.pth) |
| DukeMTMC-ReID | [`exp/duke_no_label_smoothing/checkpoints/ckpt_best.pth`](../exp/duke_no_label_smoothing/checkpoints/ckpt_best.pth) | [`exp/pcb/duke/checkpoints/ckpt_best.pth`](../exp/pcb/duke/checkpoints/ckpt_best.pth) |
| CUHK03 detected/new | [`exp/cuhk03_no_label_smoothing/checkpoints/ckpt_best.pth`](../exp/cuhk03_no_label_smoothing/checkpoints/ckpt_best.pth) | [`exp/pcb/cuhk03/checkpoints/ckpt_best.pth`](../exp/pcb/cuhk03/checkpoints/ckpt_best.pth) |
| MSMT17 | [`exp/msmt17_no_label_smoothing/checkpoints/ckpt_best.pth`](../exp/msmt17_no_label_smoothing/checkpoints/ckpt_best.pth) | [`exp/pcb/msmt17/checkpoints/ckpt_best.pth`](../exp/pcb/msmt17/checkpoints/ckpt_best.pth) |

## Interpretation and provenance

- All32 records use normalized features, Euclidean distance and no re-ranking. Target dataset selectors match between model families for each pair.
- This is a comparison of **completed model recipes**, not a controlled architecture-only ablation. ResNet50 uses256×128 input and dataset-specific selected loss/augmentation recipes; PCB uses384×128, six summed identity heads, random image sampling and SGD. Optimizer, losses, sampling and preprocessing differences prevent attributing a score difference solely to architecture.
- Checkpoint selection uses periodic test query/gallery mAP rather than an independent validation split; retain this limitation in the dissertation. Cross-domain target scores did not determine the checkpoint selected for each row.
- PCB's recorded training commit is `3b015c9`; the original Phase20 freeze specified `66ea88d`. Their committed production code/presets are identical and the documentation-only exception remains recorded. PCB evaluations used the supplied isolated `3b015c9` snapshot. The original evaluator does not embed execution-time code/checkpoint hashes; the PCB row-verification records provide post-run hash checks.
- This file consolidates existing results. No training, evaluation, checkpoint selection change or statistical significance test was performed to create it.

## Source artifacts

Each link below points to the actual result JSON used for that pair.

| Source → target | ResNet50 record | PCB record |
| --- | --- | --- |
| Market-1501 → Market-1501 | [JSON](../results/resnet50_selected_matrix/market1501_to_market1501/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/market1501_to_market1501/cross_dataset.json) |
| Market-1501 → DukeMTMC-ReID | [JSON](../results/resnet50_selected_matrix/market1501_to_duke/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/market1501_to_duke/cross_dataset.json) |
| Market-1501 → CUHK03 detected/new | [JSON](../results/resnet50_selected_matrix/market1501_to_cuhk03/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/market1501_to_cuhk03/cross_dataset.json) |
| Market-1501 → MSMT17 | [JSON](../results/resnet50_selected_matrix/market1501_to_msmt17/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/market1501_to_msmt17/cross_dataset.json) |
| DukeMTMC-ReID → Market-1501 | [JSON](../results/resnet50_selected_matrix/duke_to_market1501/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/duke_to_market1501/cross_dataset.json) |
| DukeMTMC-ReID → DukeMTMC-ReID | [JSON](../results/resnet50_selected_matrix/duke_to_duke/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/duke_to_duke/cross_dataset.json) |
| DukeMTMC-ReID → CUHK03 detected/new | [JSON](../results/resnet50_selected_matrix/duke_to_cuhk03/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/duke_to_cuhk03/cross_dataset.json) |
| DukeMTMC-ReID → MSMT17 | [JSON](../results/resnet50_selected_matrix/duke_to_msmt17/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/duke_to_msmt17/cross_dataset.json) |
| CUHK03 detected/new → Market-1501 | [JSON](../results/resnet50_selected_matrix/cuhk03_to_market1501/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/cuhk03_to_market1501/cross_dataset.json) |
| CUHK03 detected/new → DukeMTMC-ReID | [JSON](../results/resnet50_selected_matrix/cuhk03_to_duke/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/cuhk03_to_duke/cross_dataset.json) |
| CUHK03 detected/new → CUHK03 detected/new | [JSON](../results/resnet50_selected_matrix/cuhk03_to_cuhk03/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/cuhk03_to_cuhk03/cross_dataset.json) |
| CUHK03 detected/new → MSMT17 | [JSON](../results/resnet50_selected_matrix/cuhk03_to_msmt17/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/cuhk03_to_msmt17/cross_dataset.json) |
| MSMT17 → Market-1501 | [JSON](../results/resnet50_selected_matrix/msmt17_to_market1501/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/msmt17_to_market1501/cross_dataset.json) |
| MSMT17 → DukeMTMC-ReID | [JSON](../results/resnet50_selected_matrix/msmt17_to_duke/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/msmt17_to_duke/cross_dataset.json) |
| MSMT17 → CUHK03 detected/new | [JSON](../results/resnet50_selected_matrix/msmt17_to_cuhk03/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/msmt17_to_cuhk03/cross_dataset.json) |
| MSMT17 → MSMT17 | [JSON](../results/resnet50_selected_matrix/msmt17_to_msmt17/cross_dataset.json) | [JSON](../results/pcb_selected_matrix/msmt17_to_msmt17/cross_dataset.json) |

PCB full-precision table and hash evidence: [matrix CSV](../results/pcb_selected_matrix/matrix.csv), [PCB matrix report](pcb_evaluation_matrix.md).
