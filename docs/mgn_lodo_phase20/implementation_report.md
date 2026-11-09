# Phase20 implementation readiness

The approved `mgn_lodo_v2` protocol is implemented through the existing framework on `feature/mgn-lodo-v2`. This is implementation readiness, not experimental validation, Oneiros qualification, or approval of retrieval results. No GPU execution, official training, real benchmark evaluation, ONNX/runtime campaign, push or merge occurred in Phase20.

## Implementation

Four versioned YAMLs retain canonical source partitions/order/maps and fresh pinned ImageNet initialization. The common trainer, loader, global PK sampler, MGN architecture/losses, SGD, successful-update controller and checkpoint recovery machinery are reused. The exact 85,200 accepted-update workload has no epoch cap. Tests exercise updates42,600/42,601/63,900/63,901/85,200 and no optimizer call or batch access after completion. A common-trainer fixture runs121 one-batch fixture epochs to detect an inherited120-epoch cap; this is not real training or convergence evidence.

Scientific identities distinguish the new protocol; full checkpoint identities additionally bind actual source C, maps, inventories and numerical policy. Terminal reconstruction rejects schema/head/counter/scheduler/source mismatches before target execution. Strict completed-epoch recovery retains two generations. Work accounting records accepted source presentations, attempted phases, confirmed discarded work/replay and unknown physical work after unobserved termination or fallback. Inconsistent or unobserved work is not reported as an exact physical total.

Float32 tensors and disabled AMP remain distinct from full-FP32 arithmetic. Convolution TF32 permission remains enabled, matmul TF32 disabled. Runtime evidence records actual compute capability and potential hardware TF32 support without claiming measured kernel arithmetic. Qualification binds CUDA/cuDNN builds and the actual NVML driver/name/compute capability as well as software versions; querying this identity does not initialize CUDA.

Training remains source-only with no target checkpoint choice, recipe adjustment or BN-statistics adaptation. Held-out evaluation and12 supplementary no-flip references have separate explicit request/admission paths. Their reconstruction/extraction/ranking use shared framework code. All3 compatible references and their mean are primary; best-of-three summaries are explicitly retrospective target oracles. The four original source checkpoints, original matrix and ordinary ownership guards are preserved. The prepared reference plan pins the original checkpoints and target inventories but authorizes no evaluation.

The existing single-command interface remains:

```bash
python scripts/train.py --config configs/lodo/mgn_lodo_v2/mgn_lodo-1_seed42.yaml
```

All four presets currently stop at the missing Phase21 qualification/execution receipt before CUDA/model/data iteration/output creation. Direct common-trainer entry is also guarded. No qualification or execution approval receipt was fabricated. See [operational record contract](operational_records.md).

## Evidence and limits

[Validation receipt](validation.yaml) records exact commands, CPU-only results and log hashes. [Preservation receipt](preservation_validation.yaml) checks the578-file pre-task snapshot, eight historical v1 scientific identities and four common-CLI refusals. Historical runtime behavior remains covered by regressions; shared-code changes intentionally invalidate old runtime-byte admission receipts rather than silently updating them. Existing unrelated dirty documentation and historical results remain outside these commits.

[Real metadata validation](real_metadata_validation.yaml) constructs the four real source metadata loaders without reading image batches, creating models or initializing CUDA. Actual C is2510/2559/2494/2220 and historical source-map bytes match. Source checkpoint header/hash validation prepares the12 additional no-flip request cells; none were executed. Synthetic terminal fixtures use fabricated completed counters to test reconstruction and are not85,200-step training evidence.

Earlier Phase19 focused counts reconcile as4 tests in one command plus2 in another:6 focused cases across2 commands. Eight legacy vectors and eight frozen config identities are separate assertions, not extra pytest cases. Phase20 failed attempts are retained in the validation receipt; corrected passing commands replace them as implementation evidence.

The [Phase21 matrix](phase21_matrix.md) prepares eight bounded cases, at most3 optimizer calls aggregated across all children per case/attempt, initial24 calls, and the existing short smoke limits. It remains unexecuted and subject to resource scope and bounded-recovery equivalence review. A bounded fixture epoch does not prove full real-epoch CUDA recovery. Official LODO training remains user-operated after corresponding fold qualification. Single-seed and CUHK03 provenance limitations remain unchanged; no matched-compute, convergence or retrieval-quality claim is made.
