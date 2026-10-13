# Phase 9R.1 — Authorized TITAN Xp A/B/C attempt

**INCOMPLETE — NUMERICAL INVESTIGATION REQUIRED**

A completed five successful updates, preserving a common update-2 checkpoint. B passed GPU admission but failed in CPU configuration preflight before source-loader construction and restoration: `ValueError: Fresh producer collision`. C was not launched. This is an identified diagnostic harness path-validation defect, not an observed numerical divergence or recovery-state defect. The harness was corrected and CPU-tested afterward. No automatic retry or second GPU campaign occurred. Numerical qualification remains unexecuted.

Evidence namespace: `exp/lodo_smoke/phase9r1-abc-titan-20261009T214240-original/`. The earlier authorization-deferred report and all historical GTX/preflight results remain intact. The campaign's `final-summary.json` retains STOPPED, with B FAIL and no numerical qualification result; the closeout classification above does not relabel it PASS.

## A. Repository/worktree preservation and changes

Isolated implementation worktree `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo` (canonical `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo`), branch `feature/lodo-v1`. Original checkout branch `main`. Both HEADs: `3b015c964b99bfe8ad4b7842bd174d3bc58af607`.

A fresh pre-execution snapshot records both complete Git statuses, source hashes and old artifact inventories. Original main has 174 source files and 1,777 artifact entries; LODO has 209 preexisting source files and 703 old artifact entries. All original source hashes/status/HEAD and old artifacts remain unchanged. Of LODO's preexisting source files only `tests/lodo_cuda_abc_diagnostic.py` and `tests/test_lodo_cuda_abc_diagnostic.py` changed after the failure. This document is new. All other preexisting LODO source hashes and historical artifact size/mtime/inode match the snapshot. Full baseline/final statuses are retained in `cpu-closeout/`; Git shows preexisting uncommitted implementation work, which was preserved.

The two executed diagnostic source files were copied into `executed-source/` before correction and verified against the campaign's original source-hash manifest. Post-fix source hashes are separate. Production guard, data loaders, engine, recovery, model, losses, samplers, scheduler, scientific configuration, Stage A and evaluation code were not changed. No dataset or PCB output was written. No reset/stash/clean/commit/push occurred. Artifact preservation is a size/mtime/inode comparison, not a full content rehash of every image/checkpoint; original PCB checkpoints and the new common/terminal checkpoints have independent SHA256 verification.

## B. Timestamped ownership, PCB completion and authorization

The user explicitly confirmed: “I confirm that the NVIDIA TITAN Xp with UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` is allocated to me and explicitly authorized for the short Phase 9R.1 ResNet50 A/B/C CUDA recovery diagnostics.” This actual confirmation is recorded in `allocation-confirmation.json`; no inference from idle VRAM was used.

Fresh read-only host audit at **2026-10-09T21:40:59.964008Z** found TITAN physical index 0, 12,288 MiB total, 5 MiB used, **12,179 MiB free**, utilization 0%, Default compute mode, MIG N/A, driver 570.211.01, no compute processes. GTX index 1 was excluded. Historical PCB PID 2370601 was absent; no active Python `scripts/train.py` process was observed.

PCB completion was independently rechecked read-only: epoch-120 training/evaluation followed by selected-best final evaluation at epoch 50; metric/checkpoint epoch metadata agree. CPU checkpoint inspection left CUDA uninitialized. PCB SHA256 remains:

- Selected best: `5657281a3750b63f1335d63e1bdde387a82c6d8bc5708c60d9ef20a81efc6914`.
- Final epoch 120: `d9ce09debf72e99072d7e9a15d1f68e70c13ac39ba0bdaa8f1f6232ab27cda91`.

During A a read-only audit showed only owned diagnostic PID 3176700 (user addirakoze), exact A command on TITAN; GTX had no compute workload. Final post-stop inventory at **2026-10-09T21:49:23.749488Z** again showed 0% utilization, 12,179 MiB free and no compute processes. Administrative reservation cannot be disproved from NVML inventory; the explicit allocation confirmation supplies that gate. No existing job was signaled or modified.

## C. Admission logging and resources

Parent pre-A admission: 21:42:40 UTC; child A initial admission: 21:42:46 UTC. Parent pre-B admission and child B initial admission occurred after A exited; child B admission: 21:46:50 UTC. All fresh admissions observed 12,179 MiB free and no processes. A update checks admitted only its own PID.

Retained journals: parent **2 ACCEPT**, A **12 ACCEPT**, B **1 ACCEPT**, **0 resource rejections**. C has no admission/run. Each numbered immutable JSON retains the exact production-guard query, before/after inventories, timestamps, visibility/authorization UUID, allowed PID/ownership, resource values, threshold and decision. The unchanged 7,500 MiB fresh threshold, zero-utilization/no-process requirement and ownership checks passed. The campaign stopped on B's later harness exception, not on resource admission. There was no admission retry or fallback.

## D. New common checkpoint and origin proof

Common artifact:

`resnet50/LODO-1/producer/checkpoints/ckpt_epoch_000001_u000000002_g0542df6441e9.pth`

SHA256 **`562a0ed8fb673dbbfd3101fa4514fe41b2b80956a3ef664c1a4362a8ba6222ce`**.

Identity digest `396ce52acb7543a3e6bf4501e9d60638b128cc28467dee943586991c9725e60c`. The checkpoint reader validated payload/manifest and completed-epoch boundary: exactly two completed updates, next epoch 2, cursor 0. A's live model, Center, main/Center optimizer, Parameter ownership, scheduler and RNG exactly matched serialized state at this boundary. A continued in the producer process without invoking restore and completed update 5; `producer_never_restored=true`.

LODO-1 source map SHA256 `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`; 54,135 source images, 2,510 classifier/Center classes. Sources: Duke processed trainval, detected CUHK03 processed trainval, MSMT17 raw training list only. Market-1501 target was restricted to metadata preflight; target-loader and evaluation guards remained active. No cached pretrained download occurred; pretrained SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.

B successfully read/validated the exact common manifest, hash, identity, update count and GPU UUID before its loader call, but never restored it. C did not run. Identical initial restored state for all three paths is **NOT ESTABLISHED**.

A terminal checkpoint SHA256 **`25113e41bf394f9d4b22af9c5bd4c0dcd096f3e3e59062263c2a7e45cbbe6e86`**, update 5. A separate CPU-only audit exactly matched its model/buffers, named Adam state, Center, Center optimizer, scheduler and RNG against A's recorded final-update observations. The common checkpoint remains unchanged. B/C have no terminal states or checkpoints.

## E. Per-update comparisons and recorded quantities

| Update | A total loss | A ID | A Triplet | A Center (unweighted) | A/B | A/C | B/C |
|---|---:|---:|---:|---:|---|---|---|
| 3 | 10.190587997437 | 7.830750465393 | 1.835417866707 | 1048.839233398438 | NOT RUN | NOT RUN | NOT RUN |
| 4 | 10.046455383301 | 7.818824768066 | 1.693269968033 | 1068.721923828125 | NOT RUN | NOT RUN | NOT RUN |
| 5 | 10.118157386780 | 7.819169044495 | 1.766859531403 | 1064.255859375000 | NOT RUN | NOT RUN | NOT RUN |

A recorded 352 finite events: indices/expected and actual labels, full transformed-input hashes, RNG, convolution/BatchNorm activation hashes and bounded samples, forward outputs, individual losses, full named gradients before Center scaling, scaled Center gradient, pre-Adam state fingerprints/first-convolution state/LR, complete post-Adam model and moments/steps, post-Center state, scheduler/accounting/RNG/source exposure.

A completed five primary and five Center updates, accepting 320 images. Successful source image exposures were Duke 80 (25%), CUHK03 116 (36.25%), MSMT17 124 (38.75%); identity slots were 20/29/31. All five recorded LR-used values were respectively 3e-06, 3.594e-06, 4.188e-06, 4.782e-06, 5.376e-06 across all 161 Adam groups; Center LR remained 0.5. Source exposure reconciles with actual consumed indices. The retained `A/u005-12_accounting.pth` and CPU terminal audit include exact LR traces, scheduler state and exposure counts. No B/C observations exist at updates 3–5. Therefore no pairwise loss, gradient, model, Center, first/second Adam moment, LR, input, RNG or exposure equivalence result is claimed. Original state `rtol=.001, atol=.0001`, loss `rtol=.001, atol=.001`, and integer/structural equality criteria remain unchanged.

## F. First failure and causal uncertainty

The first observed campaign failure is deterministic CPU control flow:

`B loader(cfg) -> build_train_loader -> validate_lodo_config -> diagnostic output_path -> Fresh producer collision`.

The diagnostic validator correctly rejected collisions for A but incorrectly applied that fresh-run rule to B/C, whose configuration necessarily references A's existing producer directory. It failed before constructing source datasets/model/optimizer or executing checkpoint restoration/CUDA selection. This identifies a harness preflight defect, not a CUDA numerical operation or a production restoration defect. No first numerical divergence is established, and no causal claim about CUDA kernels is supported.

Post-stop correction is diagnostic-only: an existing producer requires nonempty resume, the unchanged production `validate_resume_path` compatibility check, canonical same-campaign `read_common` proof, and exact requested/selected common update-2 generation. Fresh collisions, missing/incompatible checkpoints, unrelated generations, wrong UUID/identity/hash/path and symlink/traversal remain rejected. No checkpoint is selected by fallback or rewritten. Five focused cases were added. CPU verification on the actual preserved update-2 checkpoint passes the corrected path validator with CUDA uninitialized. This fix has **NOT been GPU-tested** and does not retroactively repair B's recorded failure.

All parameters, including `backbone.0.weight` and the 13 historically implicated parameters, are captured in A's gradient/moment states. Without a restored comparison, no new absolute/relative error or failing-element counts can be inferred. Bounded activation samples cannot prove full-activation tolerance counts; instrumentation synchronization may affect timing/autotuning. A future numerical failure may need narrower backward instrumentation before kernel attribution.

## G. Determinism policy

Only original policy ran. Recorded A environment: PyTorch 2.7.1+cu126, CUDA build 12.6, cuDNN 90501, benchmark true, cuDNN deterministic false, deterministic algorithms false, matmul TF32 false, cuDNN TF32 true, cuBLAS workspace unset. Actual Adam execution hook observed `_multi_tensor_adam` 483 calls over updates 3–5 (161 parameter groups per update). No optional deterministic campaign ran, no backend flags/tolerances/recipe changed, and no GTX continuation ran.

## H. CPU commands/results and GPU execution

All CPU commands use isolated worktree cwd, OMP/MKL 1, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=.:tests`, interpreter `/home/addirakoze/environments/Reid/bin/python -B`, pytest `-p no:cacheprovider`. Worker-bearing focused tests ran outside the sandbox.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_admission_diagnostic.py tests/test_lodo_cuda_abc_diagnostic.py tests/test_lodo_device.py tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_recovery_smoke_contract.py tests/test_lodo_cuda_numerical_diagnostic.py -q --junitxml=/tmp/lodo-titan-authorized-focused.xml

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_checkpoint.py tests/test_checkpoint_reconstruction.py tests/test_train_loop_optim.py -q --junitxml=/tmp/lodo-titan-authorized-stagea.xml
```

The post-fix command repeats the focused command exactly with JUnit destination `/tmp/lodo-titan-authorized-postfix.xml`.

| Invocation | Passed | Failed/errors | Skipped | Pytest duration |
|---|---:|---:|---:|---:|
| Pre-GPU focused | 133 | 0 | 0 | 31.59 s |
| Stage A regressions | 46 | 0 | 3 | 60.42 s |
| Post-fix focused | 138 | 0 | 0 | 31.44 s |

Pre-GPU total 179 passed/3 skipped. Final disjoint post-fix focused plus unchanged Stage A: **184 passed/3 skipped**, no failures. Focused reruns overlap and are not cumulatively added. Three skips are existing CUDA-dependent Stage A tests, not evidence of GPU success. `git diff --check` passes. No broad full-suite rerun is claimed. The failed B diagnostic child is separately retained and not hidden by passing CPU tests.

Executed parent command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_titan_abc_campaign.py --allocation-receipt /tmp/lodo-titan-allocation-confirmation.json --policy original
```

Executed child commands (A then B; no C):

```bash
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_abc_diagnostic.py --campaign phase9r1-abc-titan-20261009T214240-original --path A --policy original --gpu-uuid GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_abc_diagnostic.py --campaign phase9r1-abc-titan-20261009T214240-original --path B --policy original --gpu-uuid GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
```

Both child environments bind `CUDA_VISIBLE_DEVICES` and `LODO_AUTHORIZED_GPU_UUID` to the exact TITAN UUID. A's production selector verifies physical UUID -> logical `cuda:0`; B never reached device selection. Exact environments/commands and timestamps are in `campaign-summary.json`.

A wall time **241.49 s**, harness measured **233.94 s**, peak allocated **9,074,793,472 bytes**, reserved **9,672,065,024 bytes**. B wall **9.64 s**, FAIL before CUDA selection/restoration; no B peak GPU measurement. C NOT RUN. Runtime differences include metadata/preflight, instrumentation and disk I/O; they are not training throughput benchmarks.

## I. Qualification decision and PCB gate

**INCOMPLETE — NUMERICAL INVESTIGATION REQUIRED**. Authorization and resource gates passed. A's checkpoint/trajectory completed, but required exact B/C restoration invariants and all nine per-update pairwise comparisons remain missing due to the harness defect. Neither numerical PASS nor numerical FAIL is established. The original STOPPED campaign and B FAIL remain immutable evidence.

ResNet50 recovery is unqualified for TITAN under the original criteria. No bitwise, long-run, mid-epoch, cross-device or GTX equivalence claim is made. PCB CUDA recovery remains a separately authorized next gate. Official 85,200-update recipe/validators, zero-worker CUDA recovery restriction and scientific protocol remain unchanged. No official training, benchmark evaluation, PCB recovery test, Phase 10 or commit/push occurred.

## J. Remaining risks and recommendation

Review the preserved failed attempt and CPU-tested path-validator fix. A fresh controlled original-policy A/B/C campaign would still be required to qualify recovery, with fresh ownership/headroom checks and a new namespace. No retry has been performed and no diagnostic remains running. Follow the requested stop/review gate before further GPU tests; do not advance to PCB recovery or official LODO training.

Keep original and post-fix source provenance separate. The corrected harness's GPU execution is untested. Independent reservations remain an operator-level gate, and metadata-only target checks do not prove absence of duplicate content or semantic identities across datasets; CUHK03 official-new provenance remains unverified. No tolerance relaxation, sampler adjustment, loss/model/optimizer change or production recovery correction is supported by this failed preflight.
