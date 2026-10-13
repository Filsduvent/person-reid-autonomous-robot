# Phase 9R.3 — deterministic CUDA recovery qualification

**COMPLETE — READY FOR PHASE 10 REVIEW.** Production-path ResNet50 and independently gated PCB passed on the explicitly allocated TITAN Xp. This is a bounded five-update, zero-worker LODO-1 qualification, not approval for official training or Phase 10 execution.

Evidence namespace: `exp/lodo_smoke/phase9r3-titan-20261010T005917/`. The earlier CPU-only report is preserved as `prior-cpu-policy-report.md`; historical original-policy numerical failures remain unchanged.

## A. Worktree safety and files changed

Implementation worktree: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`, `feature/lodo-v1`. Original Stage A: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot`, `main`. Both HEADs remain `3b015c964b99bfe8ad4b7842bd174d3bc58af607`; canonical roots start `/mnt/hd-data/addirakoze/Projects/`. Both were already dirty; no reset/stash/clean/branch switch, commit or push occurred.

This authorized continuation adds seven test/diagnostic files: `tests/lodo_cuda_policy_qualification.py`, `lodo_cuda_policy_campaign.py`, `lodo_cuda_policy_artifacts.py`, `lodo_cuda_policy_performance.py`, and `test_lodo_cuda_policy_qualification.py`, `test_lodo_cuda_policy_artifacts.py`, `test_lodo_cuda_policy_performance.py`. Only this report, `lodo_recovery_contract.md` and `lodo_decision_register.md` change among preexisting files. Production code remains byte-identical to the completed CPU policy implementation; its earlier changes are documented in the archived CPU report.

Fresh preservation baseline covers 174 original and 226 preexisting LODO source files, 1,777 original plus 2,392 historical LODO artifact file stats (size/mtime/inode), and **4,038 matching full SHA256 hashes** (1,655 original + 2,383 LODO). All old LODO checkpoints and small files, plus protected PCB MSMT17 best/last checkpoints, are included. Earlier Phase 9R.3 CPU evidence also matches its immutable manifest. This is bounded hashing, not every Stage A artifact or every dataset image. Original HEAD/status/source bytes are unchanged. New artifacts are confined to this fresh non-authoritative namespace.

## B. GPU authorization, inventory and PCB protection

The current user explicitly allocated and authorized NVIDIA TITAN Xp `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` for Phase 9R.3. Initial live inventory: 12,288 MiB total, 5 MiB used, 12,179 MiB free, utilization 0%, no compute processes. GTX `GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5` was excluded. No idle-memory inference was used as allocation authority.

Every CUDA child exposes only the full authorized UUID in both `CUDA_VISIBLE_DEVICES` and `LODO_AUTHORIZED_GPU_UUID`, using logical `cuda:0`. Controller and child admission journals retain raw numeric inventories, process ownership, exact UUID mapping, time and ACCEPT/REJECT decisions; fresh minimum headroom is the unchanged 7,500 MiB threshold, with no competing process and fresh utilization 0%. Production guards recheck ownership during execution. Only each newly spawned diagnostic child is deliberately terminated. Completed PCB MSMT17 and all historical diagnostic artifacts remain unchanged by hash/stat checks.

## C. Deterministic production policy and initialization

`reid/utils/lodo_cuda_policy.py` defines explicit immutable `strict_cuda_deterministic_v1`; omission/`original_cuda_v1` retain historical defaults. `repro.cuda_policy` must explicitly select strict Stage B CUDA with deterministic=true and benchmark=false. The pure policy module is imported before Torch; launch must export `CUBLAS_WORKSPACE_CONFIG=:4096:8`. `scripts/train.py:run_lodo_training` activates and verifies the policy before device discovery, seeding, loaders or output creation; direct device admission enforces the same pre-context rule. Missing/late workspace, preinitialized CUDA, unknown policy or drift fails closed.

Observed bundle: cuDNN benchmark=false, cuDNN deterministic=true, deterministic algorithms=true, warn-only=false, workspace `:4096:8`; approved matmul TF32=false and cuDNN TF32=true are preserved. Actual environments and optimizer implementation call counts are saved per child. Stage A defaults remain unchanged.

The frozen fold/descriptors/map, seed42, FP32, original PK16×4 and shuffled random64/drop_last, ResNet50 256×128 ID+Triplet+Center conventions, PCB 384×128 six-head summed ID, Adam/Center SGD versus PCB SGD, LR groups/milestones and official 85,200/94,200 budgets are unchanged. All production validators reject shortened test configurations before and after execution.

## D. Checkpoint schema, production path and recovery scope

Strict scientific identity and payload/identity/manifest are v2; original/CPU remain v1. No migration or historical checkpoint rewriting occurs. Compatibility binds complete policy, same UUID/driver/host/Python/Torch/torchvision/NumPy/CUDA/cuDNN/TF32 controls, fold/map/class dimensions, provenance, optimizer ownership/options, scheduler, Center state and worker contract before tensor mutation. Hardware/library mismatch blocks exact recovery and requires separate qualification. The runtime tuple is recorded below, not a fingerprint of every native library.

Every A/B/C and recovery case calls actual `scripts.train.run_lodo_training` and the unchanged shared engine; no manual continuation engine loop is substituted. A saves update2 and continues without restoring to5. The controller archives the completed branch and reconstructs the **same canonical diagnostic run directory** from an immutable boundary snapshot. Checkpoint pairs are hard-linked, so B/C read the same bytes and inode; index/events metadata is copied separately. `validate_resume_path`, `select_generation`, checkpoint state/identity checks and terminal precedence remain unmodified. Branch archives are evidence, not a newly supported production fork/resume API.

Only fixture budget5, fresh smoke output validation and a two-batch prefix defining each test epoch are substituted. Original sampler decisions are unchanged within those prefixes. Source inventories remain real: LODO-1 Duke/CUHK03 detected processed trainval/MSMT17 raw train, 54,135 images and 2,510 classes; Market1501 is excluded. This does not demonstrate full real-dataset epoch replay. CUDA capability is FP32, zero ordered/nonpersistent workers and completed persisted epoch rollback; CPU retains separately tested 0/2-worker recovery. Mid-epoch exact cursor continuation, arbitrary workers/hardware/library versions and long-run bitwise reproducibility are not claimed.

Observed runtime: {"cuda_build": "12.6", "cudnn_version": 90501, "driver": "570.211.01", "host": "constantine", "numpy": "2.1.3", "python": "3.11.2", "torch": "2.7.1+cu126", "torchvision": "0.22.1", "uuid": "GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c"}

## E. ResNet50 production-path A/B/C

| Pair | Update 3 | Update 4 | Update 5 | First divergence |
|---|---|---|---|---|
| A/B | EXACT / PASS | EXACT / PASS | EXACT / PASS | None |
| A/C | EXACT / PASS | EXACT / PASS | EXACT / PASS | None |
| B/C | EXACT / PASS | EXACT / PASS | EXACT / PASS | None |

Common update-2 SHA256: `a5b12a57311cc33c0913d70475db2ca2e6276841ca34dbf6f2664ad35df25541`. **84 pairwise full-state event comparisons**, zero outside tolerance; maximum absolute difference 0. Exact pre-step model/buffers, main optimizer, scheduler, RNG and parameter ownership were asserted. Mapping: `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`.

Traces compare input indices/transformed input hashes, forward outputs/buffers, loss components, full model/Center gradients before and after Center rescaling, model state, named Adam first/second moments and step counters, Center tensor/SGD state, scheduler LR-used, RNG and exposure. Center shape is 2,510×2,048; every completed branch has five primary and five auxiliary updates.

Original numerical rules are unchanged: floating state/gradients rtol=.001, atol=.0001; losses rtol=.001, atol=.001; integer/structural fields and optimizer step counters exact. Exactness is recorded separately. No divergent operation was observed, so no new backward localization rerun was needed.
## F. Independent PCB production-path A/B/C

| Pair | Update 3 | Update 4 | Update 5 | First divergence |
|---|---|---|---|---|
| A/B | EXACT / PASS | EXACT / PASS | EXACT / PASS | None |
| A/C | EXACT / PASS | EXACT / PASS | EXACT / PASS | None |
| B/C | EXACT / PASS | EXACT / PASS | EXACT / PASS | None |

Common update-2 SHA256: `fedd58564a1e48cbdf93e35ab788d56f546580091ef9a0b5f9adf2a8de146f4f`. **66 pairwise full-state event comparisons**, zero outside tolerance; maximum absolute difference 0. Exact pre-step model/buffers, main optimizer, scheduler, RNG and parameter ownership were asserted. Mapping: `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`.

PCB started only after ResNet50 A/B/C, actual interruption/replay and terminal no-op passed. It uses the actual six heads at C=2,510, 384×128 random batch64, original SGD groups/momentum and zero Center auxiliary updates. Full backbone/head gradients, model/buffers, momentum, LRs/RNG/exposure are compared. Independent CPU recomputation from saved actual logits/labels confirms the sum of six CE losses at updates3–5 within the unchanged loss tolerance.

Original numerical rules are unchanged: floating state/gradients rtol=.001, atol=.0001; losses rtol=.001, atol=.001; integer/structural fields and optimizer step counters exact. Exactness is recorded separately. No divergent operation was observed, so no new backward localization rerun was needed.

## G. Actual interruption, rollback, integrity and failure records

For **each architecture**, a separate child restored the common update2 boundary, completed update3 consistently (192 accepted presentations in memory), flushed an external diagnostic marker and terminated itself with SIGTERM. The controller observed return code -15. Persisted recovery remained at2; no terminal or incomplete checkpoint was published. A new process resumed through production CPU-first validation, restored2, replayed3 and completed5 with **320 accepted images**, not384; one discarded update/64 images was rolled back. Interrupted update3 and replayed3–5 match A, including LR-used, RNG, indices and source exposure.

SIGTERM does not promise a Python exception-handler event. The external marker and immutable checkpoint prove the discarded interval; resumed production events record the rollback source/SHA and restart epoch. No unrelated process was interrupted. Terminal reopening from a stale latest-generation pointer is a CPU-only no-op with throwing device/target guards, zero further updates and no CUDA initialization.

Both architectures have two actual immutable completed generations (updates2/4) and a separate terminal5. CPU-only inspection of **actual v2 CUDA-produced artifacts**, under throwing CUDA-init and state-mutation guards, passes 11 gates per architecture: valid terminal; same-C/different contiguous map assignment; wrong fold/config/policy/runtime; indexed SHA256 window; corrupt/missing newest fallback; explicit corrupt rejection; and interrupted index publication retaining the prior index/rejecting uncommitted orphan promotion. Destructive fixtures are ordinary copies in temporary directories. The publication fault injects a same-completed-boundary new generation nonce, not a claim of a new GPU update.

One initial CPU artifact-inspection attempt failed at its smoke namespace fixture **before state mutation/CUDA initialization**. The source, failure receipt and log remain preserved. A scoped read-only archived-run namespace fixture corrected it; focused CPU tests reran and actual artifact inspection passed. No numerical tolerance, production validator or GPU recovery harness changed. Historical original-policy numerical FAILs remain FAILs.

## H. Measured VRAM, timing, checkpoint cost and limits

| Architecture / case | Peak allocated GiB | Peak reserved GiB | Terminal MiB | Save seconds (A, all snapshots) | Restore seconds (B: CPU check / actual load) |
|---|---:|---:|---:|---|---|
| resnet50 / A | 4.360 | 4.801 | 348.16 | 8.249, 14.117, 17.394, 10.688 | 8.025 / 8.224 |
| resnet50 / B | 4.362 | 4.988 | 348.16 | see JSON | see above |
| pcb / A | 6.358 | 6.938 | 233.54 | 8.296, 11.931, 14.133, 9.412 | 7.072 / 7.458 |
| pcb / B | 6.356 | 7.166 | 233.54 | see JSON | see above |

Full-state qualification traces add CPU serialization/hash/comparison overhead; their step durations are not throughput benchmarks. Save timing includes the actual serialization, verification, manifest/index commit and retention. Actual-load timing excludes subsequent diagnostic full-state assertions; CPU-first validation is reported separately. Device-used/free snapshots are retained per outcome; they are completion samples, not device-wide peak measurements. Allocator peaks do not isolate every native allocation.

| Matched timing | Original seconds/step | Strict seconds/step | Strict images/s | Strict step-time change |
|---|---:|---:|---:|---:|
| resnet50 | 0.445605 | 0.619177 | 103.36 | +38.95% |
| pcb | 0.533117 | 0.786235 | 81.40 | +47.48% |

Matched timing uses separate fresh original/strict processes, same GPU, cached pretrained SHA, real source map/classes, exact five input-index/transform hashes and the same frozen recipe except the execution-policy bundle. Steps1/2 warm up; the median of3/4/5 measures model entry through completed optimizer/scheduler/exposure accounting. Input transfer/hash, loader/preflight, GPU audit and checkpoint I/O are excluded. No state trace serialization runs. Original policy is used only for non-authoritative timing, never promoted to recovery-qualified or official.

This is one sequential pair per architecture with three timed observations: process/order/thermal/cache effects and short-run variance limit overhead inference. It does not establish end-to-end deployment throughput or long-run performance. Earlier original-policy smoke measurements have different instrumentation/hardware/conditions; they are historical context and were not reclassified. No OOM/kernel workaround, batch/resolution reduction, AMP change or deterministic fallback was applied.


| Matched VRAM | Original allocated/reserved GiB | Strict allocated/reserved GiB | Strict device used/free MiB (completion sample) | Approx device-used minus current Torch-reserved MiB |
|---|---:|---:|---|---:|
| resnet50 | 8.452 / 9.008 | 4.360 / 4.801 | 5155 / 7029 | 239.00 |
| pcb | 9.670 / 10.076 | 6.358 / 6.938 | 7343 / 4841 | 239.00 |

Allocator maxima cover all five updates, including warmup and any original-policy cuDNN autotuning; they are not steady-state VRAM estimates, and the peak operation was not localized. The last column is an approximate completion-sample difference including context/driver/display/native allocations, not native peak attribution. Actual CUDA free/total and current allocator measurements are also retained in the timing JSON.

## I. Commands, test counts and regression evidence

All commands execute in the isolated worktree with `/home/addirakoze/environments/Reid/bin/python -B`. CPU prefix: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests CUBLAS_WORKSPACE_CONFIG=:4096:8`. Pytest always uses `-p no:cacheprovider`.

```bash
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_policy_qualification.py tests/test_lodo_cuda_policy.py tests/test_lodo_cuda_recovery_smoke_contract.py tests/test_lodo_cuda_admission_diagnostic.py -q --junitxml=/tmp/lodo9r3-gpu-harness.xml
timeout 180 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_policy_artifacts.py tests/test_lodo_cuda_policy_qualification.py -q --junitxml=/tmp/lodo9r3-artifact-contract.xml
timeout 180 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_policy_performance.py tests/test_lodo_cuda_policy_artifacts.py tests/test_lodo_cuda_policy_qualification.py -q --junitxml=/tmp/lodo9r3-new-harness-final.xml
timeout 1500 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests -q --junitxml=/tmp/lodo9r3-gpu-full-regression.xml > /tmp/lodo9r3-gpu-full-regression.log 2>&1
timeout 180 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_policy_performance.py tests/test_lodo_cuda_policy_artifacts.py tests/test_lodo_cuda_policy_qualification.py -q --junitxml=/tmp/lodo9r3-harness-revised.xml
```

| CPU invocation | Passed | Failed/errors | Skips | JUnit seconds |
|---|---:|---:|---:|---:|
| initial harness gate | 102 | 0 | 0 | 18.205 |
| artifact contracts | 22 | 0 | 0 | 13.330 |
| new harnesses | 30 | 0 | 0 | 7.933 |
| full regression | 1357 | 0 | 4 | 532.408 |
| after read-only fixture correction | 30 | 0 | 0 | 9.077 |

Counts overlap and must not be summed. Full stdout reports **1,357 passed, 4 skipped in 532.45s**. Skips are the existing CUDA model-forward case and three CUDA loss-combination training cases with CUDA hidden; these skips are not GPU evidence. Broad Stage A coverage includes config/datasets/samplers/collator, models/losses/optimizers/schedulers, training, legacy checkpoint reconstruction and evaluators. Final production code is unchanged throughout this continuation. The read-only inspector fixture was corrected after full collection; focused tests and real-artifact integration cover the correction.

CUDA prefix additionally sets the exact TITAN UUID in both visibility/authorization variables. Strict children have launch-time workspace `:4096:8`; original-only timing children omit it. Controller command:

```bash
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_policy_campaign.py phase9r3-titan-20261010T005917
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_policy_qualification.py --campaign phase9r3-titan-20261010T005917 --architecture <resnet50|pcb> --case <A|B|C|interrupt|resume|terminal>
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_policy_performance.py phase9r3-titan-20261010T005917 <resnet50|pcb> <original|strict>
```

The exact expanded child commands/environments/PIDs/return codes/durations are in `campaign-summary.json` and `matched-performance.json`. Ten strict CUDA recovery children completed (including two expected SIGTERM exits); two additional terminal children were CPU-only. Four matched timing CUDA children completed. No automatic numerical rerun occurred. Controllers never allocate CUDA. Actual artifact and semantic audits use CPU tensors only; their visibility declaration may retain UUID solely for strict compatibility checks. Host execution was required because sandbox shell creation failed; observed host results, not unobserved launch assumptions, support all claims.

## J. Remaining scientific and operational review

The measured runtime and representative LODO-1 five-update/zero-worker scope are bounded. No guarantee is made for full real-dataset epochs, other workers/folds/hardware/library versions or long runs. Official cadence/retention/storage, full code/data/map/pretrained/command/environment provenance freeze, complete experiment matrix and a dedicated official GPU allocation require review and explicit launch authorization. CUHK03 remains detected `processed_partition`, split_id=null; official detected/new provenance remains unverified.

ResNet50/PCB loss/sampler/model/optimizer/scheduler formulas and all Stage A data/evaluation interfaces are unchanged by this continuation. No real held-out image processing, embedding extraction or benchmark evaluation occurred. No official training, Phase 10, commits or pushes occurred. Historical failures and completed PCB artifacts remain preserved.

## K. Readiness and stop

**COMPLETE — READY FOR PHASE 10 REVIEW.** Explicit production policy, CPU-first schema/runtime compatibility, both independent same-checkpoint A/B/C qualifications, actual process rollback/replay, immutable generation/terminal behavior, real-artifact negative checks, measured resource/paired policy timing, fresh CPU regressions and historical preservation gates passed. Stop after this report; Phase 10 and official training still need explicit authorization.
