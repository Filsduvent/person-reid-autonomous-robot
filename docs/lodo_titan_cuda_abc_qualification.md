# Phase 9R.1 TITAN Xp A/B/C — CPU preparation and authorization stop

**GPU DIAGNOSTICS DEFERRED — AUTHORIZATION/RESOURCE GATE**

No TITAN CUDA workload ran. The user prompt explicitly requires a separate allocation confirmation and states that the prompt itself is not proof. That confirmation was requested and has not been received. CPU diagnostic preparation, regression checks, read-only PCB completion checks and repository preservation are complete. Numerical qualification remains unexecuted. Evidence: `exp/lodo_smoke/phase9r1-abc-titan-preflight-20261009T213142/` (CPU/read-only only).

## A. Repository/worktree preservation and files changed

Implementation worktree: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`, canonical `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo`, branch `feature/lodo-v1`. Original checkout remains `main`. Both HEADs remain `3b015c964b99bfe8ad4b7842bd174d3bc58af607`.

Only these six source/document files change in this task:

- New `tests/lodo_cuda_admission_diagnostic.py`: diagnostic-only admission journal and safe final-status publication.
- New `tests/test_lodo_cuda_admission_diagnostic.py`: 22 CPU cases covering logging, missing metrics, ownership/UUID, atomic publication, failure status and explicit receipt requirements.
- New `tests/lodo_cuda_titan_abc_campaign.py`: separately gated TITAN-only sequential campaign driver.
- Modified `tests/lodo_cuda_abc_diagnostic.py`: opt-in full UUID selection, TITAN admission instrumentation, canonical same-checkpoint/environment validation and atomic trace/outcome/common-manifest publication.
- Modified `tests/test_lodo_cuda_abc_diagnostic.py`: six additional provenance/duplicate-event checks, 20 cases total.
- New this document.

Baseline `/tmp/lodo-titan-before.json` records both statuses, source SHA256 and artifact size/mtime/inode. Original 174 source files and exact Git status remain unchanged. Of 205 preexisting LODO source files, only the two diagnostic harness/test files above change. All other preexisting source hashes remain identical. All 1,777 original and 687 LODO artifact inventory entries remain unchanged by size/mtime/inode, including historical GTX checkpoints, traces and failure records. This is not a full content rehash of every historical artifact or dataset image. PCB checkpoint hashes were separately verified below. Existing uncommitted changes are preserved; tracked production modifications shown by Git already predated this task.

All nine requested documents were read, with historical readiness labels interpreted in their phase context. Actual current admission, A/B/C harness, forensic analysis and focused tests were inspected against the existing recovery/shared-engine contracts. No production module, dataset, architecture/loss/optimizer/sampler/scheduler/configuration, Stage A/PCB recipe, official output, legacy checkpoint API or evaluation interface changed. No commits/pushes/reset/stash/clean occurred.

## B. TITAN ownership, PCB completion and explicit authorization

Target: **TITAN Xp `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`**. GTX 1080 `GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5` is excluded from the new driver. Neither device was initialized for this task.

Read-only host audit recorded at **2026-10-09T21:27:26.523487Z**:

| Device | Physical index | Total MiB | Used MiB | Free MiB | Utilization | Compute processes |
|---|---:|---:|---:|---:|---:|---|
| TITAN Xp | 0 | 12,288 | 5 | 12,179 | 0% | None |
| GTX 1080, excluded | 1 | 8,192 | 5 | 8,103 | 0% | None |

Driver `570.211.01`, default compute mode, MIG N/A. Historical PID 2370601 is absent; the host process query found no current `python ... scripts/train.py` job. Absence/idle status alone was not treated as evidence of completion or allocation.

Independent completion evidence from original `exp/pcb/msmt17/`, inspected read-only on CPU:

- `train.log` reaches completed epoch 120, epoch-120 evaluation and `[Final Eval] checkpoint=ckpt_best.pth epoch=50`.
- `metrics/final_epoch_test.json` names epoch 120/`ckpt_last.pth`; `metrics/final_test.json` names epoch 50/`ckpt_best.pth`.
- CPU-loaded checkpoint metadata independently matches epochs 120 and 50. Both files are 226,571,403 bytes.
- Last checkpoint SHA256: `d9ce09debf72e99072d7e9a15d1f68e70c13ac39ba0bdaa8f1f6232ab27cda91`.
- Selected-best checkpoint SHA256: `5657281a3750b63f1335d63e1bdde387a82c6d8bc5708c60d9ef20a81efc6914`.

These observations support completed training/final evaluation; no evaluation was rerun and no result was registered or modified. Selected-best and final-epoch artifacts remain distinct. No existing process was signaled or changed.

**Explicit allocation authorization: MISSING.** The separate exact-UUID confirmation question remains pending. The latest prompt says: “This prompt is an instruction to check authorization, not itself proof of allocation.” Earlier GTX authorizations do not apply to TITAN. Idle resources do not establish an administrative reservation; no reservation lock or independent allocation record is available. Execution must wait for explicit user/admin confirmation and a renewed resource audit.

## C. Admission logging and resource checks

`AdmissionJournal` wraps the **unchanged production `audit_gpu`** only within the diagnostic harness. It records the exact query responses consumed by that guard, not merely adjacent snapshots. It retains timestamps, target/authorized UUID, visibility, allowed own PID, full GPU/process inventories, target total/used/free MiB and utilization, target process owners/commands, before/after snapshots, required threshold, observed guard evidence, accept/reject decision, rejecting check and exact reason. Raw malformed/missing metrics are retained when available; query/owner errors fail closed.

Fresh launch still requires zero utilization and no compute process, plus the same **7,500 MiB diagnostic free-memory threshold** used by the previous campaign driver. UUID, compute/MIG, ownership and production policy checks are preserved. Update checks permit only the admitted process's own PID; another process aborts execution. Missing after-inventory evidence also rejects. No fallback, relaxation or automatic admission retry exists.

Each check publishes an exclusive immutable numbered JSON through same-filesystem temporary write/fsync, atomic hard-link publication and directory fsync. Existing files/symlinks are not replaced. Status finalization catches an optional final read-only audit error and still publishes STOPPED with the primary error retained. The driver uses atomic live-summary updates and an exclusive final summary. A fixed three-second settling interval occurs **before** the first admission for B/C after the preceding child exits; it is not a retry after rejection.

CPU tests exercise accepted/rejected samples, nonnumeric metrics, occupancy with owners, wrong UUID/visibility/declaration, self-only update checks, no overwrite, final-audit exceptions and missing/invalid explicit allocation receipts. A final real read-only inventory at **2026-10-09T21:31:42.016029Z** also exercises the parser without attempting admission or initializing CUDA; TITAN remained idle with 12,179 MiB free and no compute process. Actual TITAN GPU admission/child logging remains **UNTESTED ON GPU**. No rejected TITAN workload-admission sample exists in this task because the human authorization gate was never passed; synthetic rejected samples are unit-test evidence only.

## D. New common update-2 checkpoint and origin proof

**NOT CREATED.** A, B and C are all NOT RUN. No checkpoint/hash from the GTX campaign is used as a TITAN baseline.

The prepared driver creates a new exclusive `phase9r1-abc-titan-<UTC>-original` namespace only after validating a separately supplied allocation-confirmation receipt. It invokes the existing shared-engine harness sequentially. A creates the validated update-2 boundary and continues in its producing process without restore; B/C independently reference that same file. The new `read_common` check validates canonical same-campaign checkpoint location, SHA256 and identity digest, policy, completed count 2 and recorded GPU UUID before component construction/restoration. Existing CPU-first compatibility and exact live state/Parameter ownership checks remain in place.

The receipt is an operational record of actual user/admin confirmation, not a mechanism for granting allocation from idle VRAM. No authorization receipt was fabricated or written in this task. New TITAN checkpoint integrity, classifier/Center dimensions, source-map/partition identities, B/C exact GPU restoration and terminal-state checks still require execution; prior CPU/GTX results do not replace these gates.

## E. Per-update pairwise comparisons

| Update | A/B | A/C | B/C | Loss/gradient/model/Center/Adam/input/LR/RNG/exposure evidence |
|---|---|---|---|---|
| 3 | NOT RUN | NOT RUN | NOT RUN | No TITAN measurements |
| 4 | NOT RUN | NOT RUN | NOT RUN | No TITAN measurements |
| 5 | NOT RUN | NOT RUN | NOT RUN | No TITAN measurements |

Existing hooks preserve all requested restoration, batch, forward/loss, pre/post Center scaling, pre/post Adam, post-Center and scheduler/accounting stages. Named parameter/moment statistics include the first convolution and all parameters, encompassing the 13 historically implicated parameters. Full-state comparisons retain earlier path as actual and later as reference. Integer structure/steps are exact; original state `.001/.0001` and loss `.001/.001` tolerances remain unchanged. B/C do not force batches or bypass original sampler/transform RNG.

No zero-error, bitwise, moment or Center comparison result is inferred from CPU tests or the previous GTX A-only run. Final checkpoint-component/trajectory comparisons and source exposure reconciliation remain unobserved on TITAN.

## F. First divergence and causal uncertainty

**NOT DETERMINED.** No new CUDA operation ran, so neither recovery correctness nor a numerical defect/variability cause is established. The old two-fresh-run comparison remains confounded by pre-restoration divergence. No threshold is relaxed and no production correction is justified.

The harness instruments individual convolution/BatchNorm forward outputs and complete named parameter/gradient/moment states. Bounded activation samples cannot establish full-activation error counts; hashes establish exact differences only. Kernel attribution may require narrower backward instrumentation if forward traces match. Synchronization, hashing and checkpoint work can affect timing/autotuning; future comparisons must disclose this and use targeted follow-up only if authorized/necessary.

## G. Separate diagnostic determinism

**NOT RUN.** Production FP32, seed 42, cuDNN benchmark/deterministic policy, TF32, sampler, losses, model and optimizers are unchanged. The optional deterministic fixture remains separate and CPU-tested; it is not evidence of TITAN CUDA behavior. Any later deterministic campaign must follow the original-policy comparison, use a fresh namespace and set `CUBLAS_WORKSPACE_CONFIG` before GPU initialization. Unsupported-operation errors must be retained without substitution. Deterministic success cannot qualify the original policy.

## H. CPU tests, commands and GPU measurements

Working directory is the isolated LODO worktree. Bytecode/cache writes are disabled and CPU threads limited. Two-worker recovery cases run outside the managed sandbox; no sandbox worker stall was incurred in this task.

Final focused command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_admission_diagnostic.py tests/test_lodo_cuda_abc_diagnostic.py tests/test_lodo_device.py tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_recovery_smoke_contract.py tests/test_lodo_cuda_numerical_diagnostic.py -q --junitxml=/tmp/lodo-titan-focused.xml
```

Stage A command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_checkpoint.py tests/test_checkpoint_reconstruction.py tests/test_train_loop_optim.py -q --junitxml=/tmp/lodo-titan-stagea.xml
```

| Invocation | Passed | Failed | Skipped | Pytest seconds |
|---|---:|---:|---:|---:|
| Initial staged admission checks | 15 | 0 | 0 | 3.06 |
| Staged admission/harness | 35 | 0 | 0 | 7.28 |
| Staged final receipt/admission/harness | 42 | 0 | 0 | 7.56 |
| Final installed focused suite | 133 | 0 | 0 | 29.29 |
| Relevant Stage A regressions | 46 | 0 | 3 | 60.34 |

Final disjoint suites total **179 passed, 0 failed, 3 skipped**; staged runs overlap and are not added. The skips are existing CUDA-only Stage A tests. No broad-suite rerun, baseline failure, GPU test success or GPU skip-derived qualification is claimed. The last minor driver publication/documentation adjustments were covered by the installed suite's receipt/status-helper tests; its GPU orchestration itself remains unexecuted.

Prepared parent command, **NOT EXECUTED** (the confirmation receipt does not yet exist):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_titan_abc_campaign.py --allocation-receipt /tmp/lodo-titan-allocation-confirmation.json --policy original
```

After real confirmation, each child would be bound to only `CUDA_VISIBLE_DEVICES=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` and matching `LODO_AUTHORIZED_GPU_UUID`, using `--gpu-uuid` with that exact full UUID. Production selection verifies physical UUID → logical `cuda:0`; no numeric-device fallback exists. Child commands and environment are persisted by the driver.

TITAN GPU runs: **0**. Update counts, CUDA backend/kernel flags, actual Adam execution path, timing, peak allocated/reserved VRAM and numerical errors: **NOT MEASURED**. Historical GTX measurements are not transferred to TITAN. Read-only `nvidia-smi` and CPU checkpoint inspection are not GPU diagnostics.

## I. Scientific/recovery qualification and PCB gate

**GPU DIAGNOSTICS DEFERRED — AUTHORIZATION/RESOURCE GATE.** CPU logging/harness regression gates pass, and observed PCB completion/resource conditions support proceeding to allocation review. Explicit TITAN allocation confirmation is still absent. No ResNet50 TITAN numerical gate has passed.

The provisional `epoch_boundary_rollback_cuda_v1` capability, zero-worker restriction, original 85,200-update budget/validators, target isolation and existing rollback/terminal policies remain unchanged. CPU tests cannot establish short or long-run TITAN continuation. No arbitrary mid-epoch, cross-device, bitwise or official-length equivalence is claimed. PCB CUDA continuation remains a separately authorized gate, even if a future TITAN ResNet comparison passes. No Phase 10, official LODO job, benchmark evaluation or PCB recovery test occurred.

## J. Remaining risks and recommendation

Receive explicit user/admin confirmation for this exact TITAN UUID, record the actual confirmation without inference, and re-audit ownership/headroom immediately before each child. Then run an entirely fresh internally controlled TITAN A/B/C campaign, complete original-criterion comparisons at updates 3–5 and final checkpoint state, localize any first divergence and retain all failures. The prepared driver and logging changes require observed GPU validation; their CPU tests do not establish operational success.

No reservation/lock is provided by NVML checks, and hidden administrative reservations cannot be disproved from process inventory. Dataset semantic identity/duplicate-content and CUHK03 official-new provenance limitations remain unchanged. No revised criterion or production correction is proposed from missing measurements. Stop here pending the requested separate allocation confirmation; no GPU retry or background diagnostic is running.
