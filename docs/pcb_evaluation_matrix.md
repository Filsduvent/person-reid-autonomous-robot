# PCB selected-checkpoint evaluation matrix

All four source rows verified on 2026-10-26: **16/16 cells complete** (four independent diagonals and twelve cross-domain cells). No missing or duplicate pairs. The user chose source-by-source evaluation: each source on itself and the other three targets, then registration before the next source. This supersedes the earlier all-diagonals-first execution order; source checkpoint and evaluation semantics remain unchanged.

Evaluation uses the isolated snapshot at commit `3b015c964b99bfe8ad4b7842bd174d3bc58af607`, with source-owned384×128 preprocessing, normalized1536-D features, Euclidean distance, no flip/re-ranking and source checkpoint fixed across the row. CUHK03 means detected/new protocol; MSMT17 uses the official V2 query/gallery lists.

Full-precision metrics: [matrix.csv](../results/pcb_selected_matrix/matrix.csv). Source-row evidence: [market1501_row_verification.yaml](../results/pcb_selected_matrix/market1501_row_verification.yaml). Duke-row evidence: [duke_row_verification.yaml](../results/pcb_selected_matrix/duke_row_verification.yaml). CUHK03-row evidence: [cuhk03_row_verification.yaml](../results/pcb_selected_matrix/cuhk03_row_verification.yaml). MSMT17-row evidence: [msmt17_row_verification.yaml](../results/pcb_selected_matrix/msmt17_row_verification.yaml). Table values below are percentages.

| Source → target | mAP | mINP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Market → market1501 | 79.54 | 42.39 | 91.92 | 97.21 | 98.28 |
| Market → duke | 22.63 | 3.33 | 39.54 | 54.80 | 61.09 |
| Market → cuhk03 | 7.12 | 3.50 | 6.36 | 15.93 | 21.50 |
| Market → msmt17 | 3.32 | 0.16 | 10.70 | 17.75 | 21.78 |
| Duke → market1501 | 27.54 | 3.37 | 55.94 | 71.53 | 78.65 |
| Duke → duke | 70.12 | 29.28 | 84.74 | 92.01 | 93.94 |
| Duke → cuhk03 | 6.46 | 2.70 | 6.36 | 15.36 | 20.79 |
| Duke → msmt17 | 5.28 | 0.16 | 16.85 | 27.09 | 32.11 |
| CUHK03 detected/new → market1501 | 25.46 | 3.50 | 49.79 | 66.15 | 72.21 |
| CUHK03 detected/new → duke | 15.81 | 1.64 | 29.08 | 42.24 | 47.94 |
| CUHK03 detected/new → cuhk03 | 59.66 | 46.58 | 63.36 | 78.64 | 85.29 |
| CUHK03 detected/new → msmt17 | 2.37 | 0.08 | 8.29 | 14.26 | 17.58 |
| MSMT17 → market1501 | 26.93 | 4.23 | 52.67 | 71.11 | 77.26 |
| MSMT17 → duke | 34.46 | 6.23 | 54.08 | 69.21 | 74.78 |
| MSMT17 → cuhk03 | 11.66 | 6.40 | 11.64 | 22.00 | 28.50 |
| MSMT17 → msmt17 | 43.40 | 6.84 | 70.56 | 82.62 | 86.71 |

| Source / target (mAP %) | Market | Duke | CUHK03 detected/new | MSMT17 |
| --- | ---: | ---: | ---: | ---: |
| Market | 79.54 | 22.63 | 7.12 | 3.32 |
| Duke | 27.54 | 70.12 | 6.46 | 5.28 |
| CUHK03 detected/new | 25.46 | 15.81 | 59.66 | 2.37 |
| MSMT17 | 26.93 | 34.46 | 11.66 | 43.40 |

All four Market cells use epoch50 `exp/pcb/market1501/checkpoints/ckpt_best.pth`, SHA256 `8ad7c820a31bb8adc9cebafb48d5e25cb6c042401d44fdbe8084ee587317fd38`. Source identity, target selectors, saved configurations, preprocessing and finite metrics were checked. The independent diagonal agrees with the trainer-produced result to within1.073e-6 in fractional metrics; all rank metrics match exactly. This small numerical variation does not indicate a meaningful performance change.

Cross-domain scores are substantially lower than the self-evaluation; these measurements describe transfer under the fixed source recipe, not adaptation or a reason to select another checkpoint using target results. All16 cells are registered; architecture comparison and broader interpretation remain a separate review task.

The original evaluator does not embed execution-time code/checkpoint hashes; this registration verifies the current checkpoint and supplied code snapshot, and preserves result-file hashes. The existing original-freeze versus actual-training-commit exception remains documented. No training or evaluation was executed during registration.

All source rows are now complete. The next planned step is review of the completed matrix and the recorded provenance exception, followed by separately scoped Phase24 comparison. No further evaluation is needed to fill this matrix.

Duke row: all four cells use epoch50 `exp/pcb/duke/checkpoints/ckpt_best.pth`, SHA256 `53df33756029ad8a0213852de53afb28758322657d2781083cb43731249e57c3`. Target selectors, source-owned preprocessing, saved configurations and finite metrics verified. Independent self-evaluation differs from training by at most1.908e-6 in fractional metrics; all rank metrics match exactly. The small mAP/mINP differences are consistent with numerical variation; preserve the measured values. No evaluation was launched during registration.

CUHK03 detected/new row: all four cells use epoch80 `exp/pcb/cuhk03/checkpoints/ckpt_best.pth`, SHA256 `05d33f32793568313ce22d404b213620009b72a17ed0025fb34222173186078a`. Source identity, target selectors, saved configurations, source preprocessing and finite metrics verified. All five independent self-evaluation metrics match the training-time selected-best result exactly. No evaluation was executed during registration.

MSMT17 row: all four cells use epoch50 `exp/pcb/msmt17/checkpoints/ckpt_best.pth`, SHA256 `5657281a3750b63f1335d63e1bdde387a82c6d8bc5708c60d9ef20a81efc6914`. Source/target identities, source-owned preprocessing, saved configurations and finite metrics verified. Independent self-evaluation differs from training only in mAP by5.960464477539063e-8 (fractional units); mINP and all ranks match exactly.

Final registration checks covered16 unique pairs,4 registered checkpoint hashes, identical checkpoint identity within each source row, all five metrics, and agreement of all four independent diagonals with their training-time results within2e-6 fractional units. Previously registered result-file hashes remain unchanged. No training, evaluation, commit or push was executed during registration.
