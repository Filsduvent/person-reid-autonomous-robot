# MGN independent within-domain and cross-domain evaluation

Status: **COMPLETE_16_CELLS_ARTIFACT_VERIFIED_REVIEW_PENDING**; 16/16 cells recorded. Metrics below are percentages.

Original four selected-best checkpoints only; one fixed source checkpoint across every target in its row. The separate40/60 schedule trial is excluded. GPU FP32 original/width-flip feature extraction, raw mean then global L2, source-owned RGB384x128 bicubic normalization, Euclidean distance, no reranking. Common CPU distance/ranking/metrics remain unchanged.

## mAP

| Source → target | Market1501 | Duke | CUHK03 | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| market1501 | 71.00 | 21.35 | 7.47 | 2.81 |
| duke | 26.24 | 66.87 | 6.12 | 4.58 |
| cuhk03 | 18.92 | 11.55 | 45.85 | 1.71 |
| msmt17 | 19.01 | 22.18 | 9.10 | 19.67 |

## Rank-1

| Source → target | Market1501 | Duke | CUHK03 | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| market1501 | 88.18 | 37.84 | 6.79 | 8.87 |
| duke | 53.41 | 81.69 | 5.93 | 15.10 |
| cuhk03 | 42.13 | 21.36 | 46.00 | 5.84 |
| msmt17 | 42.22 | 40.31 | 9.07 | 43.16 |

Full-precision Rank-1/5/10, mAP and mINP, checkpoint identities, commands, logs, GPU observations and source/target provenance are in [the registry](mgn_evaluation_results.yaml) and `results/mgn_selected_matrix/oneiros_20261106_attempt2/`. Diagonals use standalone `scripts/evaluate.py`; off-diagonals use `scripts/evaluate_cross_domain.py`.

Periodic source-test mAP originally selected each checkpoint; standalone evaluation does not create an independent validation split. Single-seed results compare complete recipes. CUHK03 is detected/processed_partition; new/classic semantic correspondence remains unverified. Original Market1501 resumed from86 after partial87; no bitwise uninterrupted trajectory is claimed. No target-based checkpoint selection, tuning or training was performed. LODO remains a separate mandatory pending setting.

## Additional metrics (%)

### Rank5

| Source → target | Market1501 | Duke | CUHK03 | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| market1501 | 94.83 | 52.38 | 14.86 | 15.44 |
| duke | 71.11 | 90.71 | 14.14 | 23.96 |
| cuhk03 | 59.00 | 35.86 | 66.00 | 10.78 |
| msmt17 | 59.92 | 55.12 | 17.50 | 59.41 |

### Rank10

| Source → target | Market1501 | Duke | CUHK03 | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| market1501 | 96.64 | 58.35 | 20.29 | 19.48 |
| duke | 77.32 | 93.13 | 19.36 | 28.78 |
| cuhk03 | 65.29 | 41.70 | 75.50 | 13.51 |
| msmt17 | 66.92 | 62.12 | 22.64 | 66.34 |

### mINP

| Source → target | Market1501 | Duke | CUHK03 | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| market1501 | 34.99 | 3.16 | 3.87 | 0.17 |
| duke | 3.65 | 27.26 | 2.59 | 0.18 |
| cuhk03 | 2.10 | 1.33 | 34.45 | 0.07 |
| msmt17 | 2.75 | 3.08 | 5.22 | 1.66 |

## Verification and interpretation

All16 unique cells and4 fixed source rows passed artifact/configuration/GPU-placement verification; all evaluator processes exited0. Row receipts are stored beside the matrix CSV. Four standalone diagonals retain numerical differences beyond the strict1e-5 threshold; the user explicitly authorized Phase18 continuation while preserving those flags. No threshold was widened and no hardware cause was proven. FP32 parameters/inputs and AMP=false use qualified matmul TF32=false/cuDNN TF32=true controls; effective convolution arithmetic differs in capability from the previous Pascal host.

Full verification: [matrix-integrity-check.yaml](mgn_evaluation_20261106/matrix-integrity-check.yaml). Descriptive averages: [descriptive-summary.yaml](mgn_evaluation_20261106/descriptive-summary.yaml). Bounded historical recipe comparison: [MGN versus PCB](mgn_historical_pcb_comparison.md). Human/advisor review remains separate from observed execution and artifact integrity. The next roadmap phase is the versioned MGN LODO scientific decision; no LODO implementation or training is started here.
