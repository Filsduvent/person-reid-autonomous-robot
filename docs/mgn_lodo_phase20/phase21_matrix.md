# Phase21 bounded qualification matrix — PREPARED, NOT EXECUTED

Science is approved; GPU execution and successful qualification are not supplied by this plan. Phase20 CPU checks do not qualify MGN on Oneiros. Use the actual Phase20 committed runtime/configuration release and fresh resource approval. The prior Oneiros UUID `GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2` is a candidate identity, not a current availability/reservation claim; do not reuse TITAN permissions/runtime or old v1 bindings.

| Fold | Ordered sources | Held-out target (metadata only during training cases) | Actual C to verify | Cases |
| --- | --- | --- | ---: | --- |
| LODO-1 | Duke/CUHK03/MSMT17 | Market1501 |2510 |F1-forward-update;F1-rollback |
| LODO-2 | Market1501/CUHK03/MSMT17 | Duke |2559 |F2-forward-update;F2-rollback |
| LODO-3 | Market1501/Duke/MSMT17 | CUHK03 |2494 |F3-forward-update;F3-rollback |
| LODO-4 | Market1501/Duke/CUHK03 | MSMT17 |2220 |F4-forward-update;F4-rollback |

Every case is isolated, sequential, non-authoritative and immutable: protocol/fold/case, actual GPU UUID, precision policy, scientific config, code/release/data/map identities and attempt number. Qualification outputs use `exp/mgn_lodo_qualification/mgn_lodo_v2/...`, never official LODO run leaves. No case trains a complete benchmark epoch or executes the85,200-update budget. The official scientific budget is not silently changed to make qualification pass; fixture execution bounds are explicit harness metadata and cannot become an official terminal.

## Bounds and required observations

- At most3 optimizer calls **aggregated across all children per case/attempt**; max600 seconds per supervised worker. Four forward/update cases plus four recovery cases give at most24 calls in their initial attempts. At most initial+2 documented implementation-correction attempts per fixed case; no namespace reset, capacity retry, batch/class/P/K/precision/worker change or automatic fallback. Preserve first failure/OOM/timeout/interruption and correction evidence.
- Fresh metadata/host inventory/partition/ordered-map/sidecar checks, pinned ImageNet SHA, committed source membership, current runtime/driver/UUID ownership and uncontended memory audit before CUDA discovery/model placement. Zero workers, P16/K4/B64, actual combined C,384×128 bicubic RGB MGN pixels/augmentation and real common virtual-source loader. No target loader, image, loss, metric, recipe decision or BN adaptation in training cases.
- Forward/update case: actual8 `[64,C]` logits,3 metric features,2048-D descriptor, independently checked mean losses, finite gradients/model/momentum and changed parameters, exactly3 successful primary calls, no auxiliary calls, scheduler/counters/exposure consistency. Actual consumed PK source identities/images and source pixels are recorded separately from scheduled work.
- Record float32 parameters/input tensors and AMP=false separately from numerical arithmetic; retain matmul TF32=false/convolution TF32 permission=true, strict deterministic controls and workspace-before-import. Record actual device name/UUID/compute capability and TF32 hardware capability; do not infer that all kernels used full-FP32 or that TF32 kernels were measured. Cold and steady allocated/reserved/free/total memory include model/first momentum/workspace and instrumentation overhead, not a Part2 benchmark.
- Recovery case:3 calls aggregate, e.g.1 call reaching a **declared one-batch fixture epoch boundary**,1 uninterrupted continuation and1 restored replay from the same immutable boundary. Compare model/BN buffers, SGD momentum, scheduler-used/next LR, RNG/sampler state, source C/map and accepted exposure. Exercise two validated generations, corruption/fallback negatives on CPU, recorded discarded work and terminal no-op plumbing using synthetic metadata fixtures. No arbitrary mid-epoch resume claim.
- A fixture epoch is not an actual full combined-dataset epoch. Pair bounded CUDA state checks with CPU/index/RNG tests over real metadata epoch boundaries; keep equivalence assumptions explicit. Qualification publication requires review of the bounded recovery equivalence and links/hashes for preserved evidence. A short PASS alone cannot certify long-run bitwise recovery. Any required stronger validation outside these bounds needs its own approved scope.
- Synthetic held-out plumbing tests use synthetic targets and no optimizer calls. They prove source C/strict reconstruction/no-download/no-flip/L2/filter/metric/eval-BN behavior, not real target accuracy. Four real benchmark held-out results remain Phase23 after four user-operated Phase22 terminals.

## Admission and readiness

The four YAMLs remain implementation-ready only. A future `MGN_LODO_FOLD_QUALIFIED` receipt must bind fold/science, reviewed committed source membership, source/target inventories, exact environment/UUID, preserved evidence hashes and an explicit bounded-fixture recovery-equivalence review reference. It is not generated in Phase20. Each later operator execution declaration additionally pins the config/effective arguments/output/resume, actual release and qualification receipt hash. Training permission is distinct from held-out and supplementary evaluation permission.

A failed fold blocks its own official launch; one successful fold does not qualify other C/maps or devices. No official run follows automatically. After accepted corresponding qualification, prepare the single-command YAML handoff for the user, who launches official training. Phase21 completion and acceptance remain pending.
