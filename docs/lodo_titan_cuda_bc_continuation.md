# Phase 9R.1 — Preserved TITAN A with independent B/C continuations

**INCOMPLETE — NUMERICAL INVESTIGATION REQUIRED**

The user-requested B and C continuations both restored the preserved A update-2 checkpoint exactly and completed through update 5. The controlled experiment completed; **numerical qualification FAILS** under unchanged tolerances. All three pairs pass at update 3 and fail at updates 4 and 5. Failing saved-state elements are exclusively Adam first moments. Restoration, input/RNG/sampling/accounting and Adam integer steps are exact; model/buffers, Center/Center optimizer, Adam second moments and losses remain within their original criteria.

Original campaign: `phase9r1-abc-titan-20261009T214240-original`. New output directory: `exp/lodo_smoke/phase9r1-abc-titan-20261009T214240-original/continuations/bc-20261009T224416/`. A was never rerun or replaced by a restored reference. The historical failed B attempt and STOPPED report remain unchanged. Operational child outcome PASS means execution/restoration succeeded; the continuation `final-summary.json` explicitly retains **qualification FAIL**.

## A. Repository/worktree preservation and changed files

LODO worktree `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`, canonical `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo`, branch `feature/lodo-v1`. Original checkout remains `main`. Both HEADs: `3b015c964b99bfe8ad4b7842bd174d3bc58af607`.

Only these files changed in this continuation task:

- `tests/lodo_cuda_abc_diagnostic.py`: optional B/C continuation directory; original A reference stays read-only, C compares against the new B. Default diagnostic behavior is retained.
- `tests/test_lodo_cuda_abc_diagnostic.py`: six CPU cases for invalid/aliased/nonexistent continuation directories, B/C-only selection, preserved A/failed-B bytes, reference routing and collision rejection.
- New `tests/lodo_cuda_titan_bc_continuation.py`: sequential exact-UUID B/C driver using the existing harness and unchanged comparison criteria, with immutable original-file hash verification before/after execution.
- New this report.

Fresh baseline: 174 original source files/1,777 original artifact entries and 210 preexisting LODO source files/1,108 old artifact entries. All original source hashes/status/HEAD and prior artifacts match. Only the two diagnostic files above changed among preexisting LODO sources; all other 208 source hashes match. Complete statuses and old-artifact comparisons are retained in the closeout evidence. Existing uncommitted production work predates this task and was preserved. Production admission/recovery/engine/model/loss/sampler/optimizer/scheduler/config/evaluation files were not changed.

All **405 existing files in the original TITAN campaign** were SHA256-frozen before GPU execution and verified unchanged afterward, including A tensors/trace, common/terminal checkpoints, old failed-B files, manifests and prior reports/receipts. Other historical LODO/GTX/PCB artifacts were checked by size/mtime/inode; this is not a full rehash of all image or historical artifact content. PCB checkpoint SHA256 was independently rechecked. No dataset, official output, original checkout, checkpoint or historical log was modified; no reset/stash/clean/commit/push occurred.

## B. Authorization, fresh ownership and PCB completion

The prior exact-UUID allocation confirmation remains valid. The user explicitly requested this continuation: preserve A, recheck TITAN, run B/C from the same update-2 checkpoint and compare the original tolerances. No redundant allocation confirmation was inferred or required.

Read-only audit at **2026-10-09T22:43:40.363647Z**: TITAN physical index 0, UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, total 12,288 MiB, used 5 MiB, free **12,179 MiB**, utilization 0%, Default compute mode, MIG N/A, driver 570.211.01; no compute processes. GTX UUID `GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5` was excluded. Historical PCB PID 2370601 remained absent; no active Python `scripts/train.py` process was found.

PCB completion still agrees across epoch-120 final training/evaluation, selected-best final evaluation at epoch 50, metrics and CPU-loaded checkpoint metadata. Unchanged SHA256: best `5657281a3750b63f1335d63e1bdde387a82c6d8bc5708c60d9ef20a81efc6914`, last `d9ce09debf72e99072d7e9a15d1f68e70c13ac39ba0bdaa8f1f6232ab27cda91`. No reevaluation or registry write occurred.

Final read-only inventory at **2026-10-09T22:53:35.461106Z** again shows TITAN idle, 12,179 MiB free and no compute processes. Allocation is supported by the explicit user statement; NVML inventory alone cannot prove absence of administrative reservations. No job was killed, paused or modified.

## C. Admission and resource checks

Parent and child fresh admissions passed before B and independently before C. B child initial check 22:44:23 UTC, device selection 22:45:37 UTC; C child initial check 22:48:47 UTC, device selection 22:50:03 UTC. Each fresh sample showed 12,179 MiB free, zero utilization and no compute process. Production selection verified exact physical UUID -> logical `cuda:0`; no fallback occurred. Parent journal has 2 ACCEPT records; B and C each have 10 ACCEPT records; **0 resource rejections**.

Every journal retains the exact guard query, numeric memory/utilization, before/after inventories, PID/owner/command, threshold and decision. Own-PID-only checks continued through all updates. Existing fresh threshold **7,500 MiB**, ownership and activity checks were unchanged. C started after B exited and a fixed pre-admission settling interval. No automatic admission retry, threshold relaxation or other GPU use occurred.

## D. Preserved common checkpoint and identical origin

Common artifact is unchanged:

`resnet50/LODO-1/producer/checkpoints/ckpt_epoch_000001_u000000002_g0542df6441e9.pth`

SHA256 **`562a0ed8fb673dbbfd3101fa4514fe41b2b80956a3ef664c1a4362a8ba6222ce`**; identity digest `396ce52acb7543a3e6bf4501e9d60638b128cc28467dee943586991c9725e60c`. A's original terminal SHA256 remains `25113e41bf394f9d4b22af9c5bd4c0dcd096f3e3e59062263c2a7e45cbbe6e86`.

Before execution, the real checkpoint reader validated completed count 2, boundary/manifest/scientific identity and TITAN UUID. All 352 saved A event states match their recorded semantic digests, and A losses/Adam implementation observations match the retained previous closeout evidence. **No earlier whole-file SHA for A/trace.json had been recorded**: historical consistency was checked through event digests and retained evidence, then whole-file SHA256 was frozen for this continuation. The new raw-file baseline proves preservation through B/C, not byte-level identity to an unrecorded earlier trace digest.

A's producing process originally continued 2->5 without restore. B/C independently read the exact same immutable update-2 bytes and validated payload compatibility on CPU before CUDA setup. The actual GPU restore copied into existing model/Center Parameters, maintained optimizer Parameter associations, then restored scheduler and RNG. All four initial stages—checkpoint identity, modules, optimizers/ownership, scheduler/RNG—are **exact for A/B, A/C and B/C**. No different checkpoint or A recreation was used.

Source map SHA256 `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`; LODO-1 54,135 images, 2,510 classifier/Center classes. Duke/CUHK03 detected processed trainval and MSMT17 training list only; Market target remained metadata-only. Source descriptors/map/partition identity match on restoration. Cached pretrained SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`; no download.

## E. All nine pairwise comparisons

Earlier path is actual, later path is reference, matching the original comparison direction. State/gradient criterion **rtol=0.001, atol=0.0001**; loss criterion **rtol=0.001, atol=0.001**. Integer/structural equality remains exact. Failed elements below are counts, not failed tensors.

| Pair | Update | Numerical gate | Gradient elements outside | Adam first-moment elements outside | First-moment max absolute error |
|---|---:|---|---:|---:|---:|
| A/B | 3 | PASS | 0 | 0 | 2.16066837e-07 |
| A/B | 4 | FAIL | 79,384 | 258 | 0.000424724072 |
| A/B | 5 | FAIL | 67,673 | 509 | 0.000542901456 |
| A/C | 3 | PASS | 0 | 0 | 1.8812716e-07 |
| A/C | 4 | FAIL | 63,333 | 202 | 0.000318735838 |
| A/C | 5 | FAIL | 106,718 | 772 | 0.000477397814 |
| B/C | 3 | PASS | 0 | 0 | 2.53785402e-07 |
| B/C | 4 | FAIL | 14,167 | 64 | 0.000193614513 |
| B/C | 5 | FAIL | 133,363 | 666 | 0.000575982034 |

Every pair/update has exact source indices, labels, transformed-input hashes, RNG, LR trace, scheduler, exposure and update accounting. All 161 Adam scalar steps are exact. All model/buffer, Adam second-moment and Center/Center-optimizer comparisons have zero outside elements; losses pass the original loss criterion. All captured floating states are finite. Non-bitwise differences are retained separately from tolerance failures.

| Update | A total loss | B total loss | C total loss |
|---|---:|---:|---:|
| 3 | 10.190587997437 | 10.190587997437 | 10.190587997437 |
| 4 | 10.046455383301 | 10.046456336975 | 10.046455383301 |
| 5 | 10.118157386780 | 10.118158340454 | 10.118157386780 |

Each path has 352 events. Updates 3–5 each capture complete forward outputs, individual ID/Triplet/Center loss terms, named pre-scaling gradients, post-scaling Center gradient, pre-Adam state/LR, complete post-Adam model/buffers/moments/steps, post-Center state and accounting. Convolution/BatchNorm full activation hashes and bounded 512-element samples are retained. Full per-component/parameter statistics are in `measurements.json` and each trace, including all 13 historically prioritized parameters and all newly failing moments.

Both B/C add three primary and three Center steps to restored counters 2->5; cumulative counters are 5/5 with 320 accepted images, including 128 from the common checkpoint and 192 newly consumed images per continuation. Successful source exposures match A exactly: Duke 80, CUHK03 116, MSMT17 124; identity slots 20/29/31. Adam LR-used at updates 3–5: 0.000004188, 0.000004782, 0.000005376 across 161 groups; Center LR 0.5. Original PK16x4, transforms, resolution 256x128 and recipe remain unchanged.

At update 5, first-convolution first moments dominate the failures:

| Pair | backbone.0.weight failed elements | Max absolute error | Relative L2 | All failing first-moment tensors |
|---|---:|---:|---:|---:|
| A/B | 392 / 9,408 | 0.000542901456 | 0.00318444473 | 18 |
| A/C | 505 / 9,408 | 0.000477397814 | 0.00369803263 | 17 |
| B/C | 378 / 9,408 | 0.000575982034 | 0.0035750914 | 18 |

Relative L2 is a tensor-wide norm ratio. Large elementwise relative errors near zero are not substituted for that ratio; original absolute/relative tolerance determines failures. Targeted A/B first-convolution audit separately retains quantiles, zero-reference changes and maximum nonzero relative error. For A/B update 5: relative L2 0.00318444, maximum elementwise ratio 44.0385, zero-reference changes 0. This does not convert the original FAIL to PASS.

B's full named terminal-state observations are retained as `.pth`; C compares its complete live terminal tensors against A/B and retains bounded statistics/full-state fingerprints rather than duplicate full tensor files. A's authoritative diagnostic terminal checkpoint was not rewritten. B/C did not save into A's producer/checkpoint directory.

## F. First observed divergence and remaining localization gap

| Pair | First exact difference | Max abs at first difference | First tolerance failure | Failed gradient elements then | Gradient max abs then |
|---|---|---:|---|---:|---:|
| A/B | 3 backward | 2.14576721e-06 | 4 backward | 79,384 | 0.00424695015 |
| A/C | 3 backward | 1.8812716e-06 | 4 backward | 63,333 | 0.00318700075 |
| B/C | 3 backward | 2.53692269e-06 | 4 backward | 14,167 | 0.0019367421 |

At update 3, restored state, batch input/labels/RNG, every captured convolution/BatchNorm full-output hash, final forward outputs/buffers and losses match exactly across all pairs. The first observed difference appears **inside `Tensor.backward()`**, at the immediately-after-backward gradient capture, before Center inverse-weight scaling or either optimizer step. These small update-3 differences pass tolerance, then propagate through the ensuing updates. The first original-criterion failure is the update-4 backward result; first-moment failure appears after that update's Adam step. Therefore the first observed discrepancy precedes Adam; attributing its origin to optimizer restoration or Adam alone is unsupported.

**Sub-operation/kernel attribution remains unresolved.** Existing hooks bound the first differing operation to backward but do not capture incoming/outgoing gradients of each autograd node or actual cuDNN algorithm IDs. Several convolutions/BatchNorm branches and Center scatter accumulation are present. No specific kernel, reduction, atomic path, warm-cache choice or ReLU boundary is proven responsible. B/C's independent divergence after exact restoration demonstrates variability between the two restored executions, not a byte-corrupt checkpoint or a proven production recovery defect. Deeper CUDA instrumentation would be needed before causal attribution.

CPU-only supplementary analysis reused saved A/B raw features and labels with the original triplet implementation. FP32 and FP64 recomputations select identical hard-positive/hard-negative indices and hinge branches at all three updates. CPU triplet-feature gradients are identical at update 3 and remain within tolerance at updates 4–5 (max abs 3.74392e-7 and 2.93367e-7). This analysis does not reproduce GPU reduction/kernel execution and cannot rule out GPU-specific behavior or prove the cause. No independent training implementation or new GPU diagnostic was launched.

Bounded activation samples cannot establish full-activation tolerance counts even though hashes establish exact differences. Instrumentation and hashing synchronize to CPU and may affect timing/autotuning. These limitations remain explicit; no guessed cause is treated as established.

## G. Original numerical policy retained

A/B/C record PyTorch 2.7.1+cu126, CUDA build 12.6, cuDNN 90501, benchmark true, cuDNN deterministic false, deterministic algorithms false, matmul TF32 false, cuDNN TF32 true, cuBLAS workspace unset. Actual `_multi_tensor_adam` execution is observed 483 calls over updates 3–5 in each path. No optimizer/backend flag, tolerance, loss, sampler, LR recipe or production policy changed. No separate deterministic-mode experiment ran, and no TITAN result is generalized to GTX.

## H. CPU tests and executed CUDA commands

Working directory is the isolated LODO worktree. CPU commands:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_admission_diagnostic.py tests/test_lodo_cuda_abc_diagnostic.py tests/test_lodo_device.py tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_recovery_smoke_contract.py tests/test_lodo_cuda_numerical_diagnostic.py -q --junitxml=/tmp/lodo-titan-bc-focused.xml

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_checkpoint.py tests/test_checkpoint_reconstruction.py tests/test_train_loop_optim.py -q --junitxml=/tmp/lodo-titan-bc-stagea.xml
```

Focused: **144 passed, 0 failed, 0 skipped, 30.00 s**. Stage A: **46 passed, 0 failed, 3 skipped, 60.51 s**. Disjoint total **190 passed, 3 skipped**. Three existing CUDA-dependent Stage A tests are intentionally skipped, not counted as GPU evidence. Focused worker cases ran outside the sandbox. Syntax/whitespace and final `git diff --check` pass. No broad-suite rerun is claimed.

Executed parent:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_titan_bc_continuation.py --allocation-receipt exp/lodo_smoke/phase9r1-abc-titan-20261009T214240-original/allocation-confirmation.json --origin-integrity /tmp/lodo-titan-bc-origin-integrity.json --policy original
```

Each sequential child runs the original harness with `--campaign phase9r1-abc-titan-20261009T214240-original --path B` (then `--path C`), `--policy original --gpu-uuid GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c --continuation-id bc-20261009T224416`. Both `CUDA_VISIBLE_DEVICES` and `LODO_AUTHORIZED_GPU_UUID` equal that full UUID. Exact commands/environments are retained in `final-summary.json`; cached initialization and engine paths are unchanged. Parent never initializes CUDA.

| Path | Child wall s | Harness elapsed s | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|---:|
| B | 261.65 | 253.64 | 9,308,494,848 | 9,915,334,656 |
| C | 278.95 | 270.95 | 9,308,494,848 | 9,915,334,656 |

These include metadata checks, synchronized diagnostics/comparisons and disk I/O; they are not training throughput measurements. Both children returned 0 after successful structural restoration/execution; parent scientific qualification is FAIL. Source hashes and executed diagnostic files are archived separately from prior A source provenance.

Supplementary CPU scripts are archived in `cpu-closeout/`: `lodo_titan_bc_integrity.py`, `lodo_titan_bc_loss_forensics.py`, `lodo_titan_bc_firstconv.py` (each invoked with the same OMP/MKL/bytecode/CUDA-hidden/PYTHONPATH environment and Reid interpreter `-B`), and `lodo_titan_bc_measurements.py` (standard-library `python3`). Their outputs retain metadata, bounded statistics and hashes, not image dumps. They are artifact checks/analysis, not additional pytest counts or GPU experiments.

## I. Qualification and protected boundaries

**INCOMPLETE — NUMERICAL INVESTIGATION REQUIRED**. The common-origin confound is resolved: A was preserved, B/C restore identical bytes, and initial structural state is exact. Nevertheless the original numerical gate fails for all three pairs at updates 4 and 5. Neither bitwise nor tolerance-qualified TITAN recovery is established. No long-run, arbitrary mid-epoch, cross-device or GTX guarantee follows from this diagnostic.

PCB recovery qualification remains gated. No official LODO training/evaluation, PCB recovery test, optional GPU campaign, Phase 10, production correction, commit or push occurred. Official 85,200-update configuration and CUDA recovery boundary policy remain unchanged. Original recipe precision/losses/Center optimizer/PK sampler and target exclusion were preserved.

## J. Next action and stop

Retain this original-criterion FAIL and all historical failures. Review the completed B/C evidence and the backward localization gap. Any further CUDA investigation should capture per-node backward gradients and available algorithm-selection evidence around the update-3 discrepancy, with no tolerance or scientific-policy changes and fresh resource checks. It requires a separately reviewed next diagnostic scope; it has not been launched here. Do not advance to PCB recovery or official LODO execution.

The device is idle and no diagnostic remains running. Dataset semantic-identity/content-duplicate limits and CUHK03 official-new provenance remain unchanged. No root-cause claim exceeds the measured evidence.
