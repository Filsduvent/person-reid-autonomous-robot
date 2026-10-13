# Phase 9 continuation: actual CUDA smoke validation

All outputs and synthetic metrics in this document are **NON-AUTHORITATIVE**.
No official LODO training, real benchmark evaluation, Phase 10 implementation,
commit or push occurred. This supersedes the initial GPU-deferred status in
`lodo_gpu_readiness.md`; the original observations remain historical evidence.

## A. Repository safety

Original checkout `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot`:
branch `main`. Isolated checkout
`/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`:
branch `feature/lodo-v1`. Both HEADs remain
`3b015c964b99bfe8ad4b7842bd174d3bc58af607`; both retain their pre-existing dirty work.
Canonical paths begin `/mnt/hd-data/addirakoze/Projects/`.

This continuation adds only `tests/lodo_cuda_smoke.py`,
`tests/test_lodo_cuda_smoke_contract.py` and this document, plus a status link in
`docs/lodo_gpu_readiness.md`. No production Python, Stage A config, source loader,
model, sampler, loss, optimizer, scheduler, collator or evaluator file changes.
The 174 original source files/status are checked against the continuation's
pre-edit SHA256 snapshot. Existing authoritative artifacts are untouched by this
task; the independently running PCB job continues writing only its own outputs.
All new smoke artifacts are under the isolated worktree's `exp/lodo_smoke/`.
Exact statuses/source deltas and artifact-prefix checks are retained in
`/tmp/lodo-phase9-continuation-final-audit.json`; initial hashes are in
`/tmp/lodo-phase9-continuation-before.json`.

### Pre-continuation status: person-reid-autonomous-robot

```text
 M docs/preliminary_results_log.md
 M plan.md
?? docs/lodo_decision_register.md
?? docs/lodo_experiment_protocol.md
?? docs/lodo_implementation_plan.md
?? docs/pcb_cuhk03_results.yaml
?? docs/pcb_duke_results.yaml
?? docs/pcb_market1501_results.yaml
```

### Pre-continuation status: person-reid-autonomous-robot-lodo

```text
 M .gitignore
 M reid/data/build.py
 M reid/engine/train_loop.py
 M reid/optim/build.py
 M reid/optim/lr_scheduler.py
 M reid/utils/config.py
 M reid/utils/config_schema.py
 M scripts/evaluate.py
 M scripts/train.py
?? docs/lodo_cpu_integration_validation.md
?? docs/lodo_decision_register.md
?? docs/lodo_evaluation_contract.md
?? docs/lodo_experiment_protocol.md
?? docs/lodo_gpu_readiness.md
?? docs/lodo_implementation_plan.md
?? docs/lodo_recovery_contract.md
?? reid/data/index_observation.py
?? reid/data/lodo_preflight.py
?? reid/data/multi_source.py
?? reid/engine/update_control.py
?? reid/utils/lodo_checkpoint.py
?? reid/utils/lodo_comparison.py
?? reid/utils/lodo_device.py
?? reid/utils/lodo_evaluation.py
?? reid/utils/lodo_protocol.py
?? tests/test_lodo_checkpoint.py
?? tests/test_lodo_config.py
?? tests/test_lodo_device.py
?? tests/test_lodo_evaluation.py
?? tests/test_lodo_integration.py
?? tests/test_lodo_updates.py
?? tests/test_multi_source_dataset.py
```

The continuation delta is the four files named above. All other status entries
are preserved, including earlier uncommitted LODO phase implementation.

The five previously read protocol/planning/recovery/evaluation documents were
SHA256-verified unchanged against the earlier complete reading. The current
GPU-readiness report and Phase 3–9 device/orchestration/checkpoint/engine/tests
were inspected. Historical planning-time implementation statements do not
redefine current capabilities or frozen science.

## B. GPU ownership audit

Host `constantine`, renewed audit **2026-10-09 15:55:03 UTC**, driver **570.211.01**.
GPU 0, TITAN Xp, UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`:
12288 MiB total, 4827 MiB free, 98% utilization; PID **2370601**, owner
`addirakoze`, command `python scripts/train.py --config configs/pcb/msmt17.yaml`,
7350 MiB compute allocation. It was excluded from every smoke process.

GPU 1, GTX 1080, UUID **`GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5`**:
8192 MiB total, 8103 MiB free and 0% utilization, no observed compute process.
Both reported default compute mode and MIG N/A. Initial visibility and scheduler
allocation variables were unset. Idle status alone did not authorize execution.
The user then explicitly confirmed that this exact UUID was independently
allocated and authorized. Gate 9A changed to **SAFE DEDICATED GPU VERIFIED**.

Every child requires `CUDA_VISIBLE_DEVICES` and `LODO_AUTHORIZED_GPU_UUID` to
contain only that full UUID. Physical GPU 1 maps to logical `cuda:0`; actual
Torch UUID and compiled architecture are checked, with no fallback. NVML is
rechecked before each case and update. Own compute PID is the only allowed PID
on the selected device. A live audit observed PCB on GPU 0 and this task's
PID 2785002 on GPU 1, confirming physical separation during the first case.
Per-case `intent.json` and `fresh/cuda_admission.json` retain timestamped evidence.

Environment: Python 3.11.2, Torch 2.7.1+cu126, torchvision 0.22.1+cu126,
CUDA build 12.6, GPU capability 6.1 supported through compiled sm_60. Initial
read-only inspection did not initialize CUDA; actual allocations began only
after explicit authorization and CPU gates. Final host inventory after the last smoke process exits,
**2026-10-09T17:19:04.169363+00:00**:

```text
index, name, uuid, memory.total [MiB], memory.used [MiB], memory.free [MiB], utilization.gpu [%], compute_mode, driver_version
0, NVIDIA TITAN Xp, GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, 12288 MiB, 7357 MiB, 4827 MiB, 98 %, Default, 570.211.01
1, NVIDIA GeForce GTX 1080, GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5, 8192 MiB, 5 MiB, 8103 MiB, 0 %, Default, 570.211.01
```

Compute-process inventory:

```text
gpu_uuid, pid, process_name, used_gpu_memory [MiB]
GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, 2370601, python, 7350 MiB
```

Original PCB process inspection:

```text
PID USER     STAT     ELAPSED COMMAND
2370601 addirak+ Rl+     07:04:41 python scripts/train.py --config configs/pcb/msmt17.yaml
```

No smoke compute process remains on the authorized GTX 1080. The task never
signaled or changed the original PCB process.

## C. CUDA readiness and isolated harness

The production LODO admission, shared engine and checkpoint code are unchanged
in this continuation. The new harness invokes actual `run_lodo_training`,
original virtual source datasets and image transforms, original PK16×4/random64,
real ResNet50/PCB, real losses, main/Center optimizers and update scheduler.
C derives from live validated sources and is asserted against the requested fold
matrix. Real source images are read without modification or copying.

FP32, batch64, dimensions 256×128/384×128, smoothing/loss/Center conventions,
optimizer recipes, exact counters and LR policy are preserved. Runtime workers
are explicitly **0 for smoke qualification**, within the validated loader contract;
this does not freeze a future official worker policy. No AMP/TF32/determinism or
benchmark policy is changed. Cached pretrained files are required; downloads are
prohibited. Observed weight hashes:

- ResNet50 V2: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
- PCB historical: `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`.

Short budgets are a bounded **test-only** in-process `_expect` substitution and
smoke-only namespace validator. Production rejects the shortened config before
and after fixture scope. A checked AST copy substitutes only the two official
budget literals in the terminal gate; all other statements remain intact.
Production budgets 85200/94200 and validators are unchanged. No official launcher
or runnable official configuration is added.

CPU validation commands, from the isolated worktree, use:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=''
timeout 180 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider --junitxml=/tmp/lodo-phase9-continuation-focused.xml tests/test_lodo_device.py
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider --junitxml=/tmp/lodo-phase9-continuation-cpu.xml tests/test_lodo_cuda_smoke_contract.py tests/test_lodo_device.py tests/test_lodo_integration.py tests/test_train_loop_optim.py
timeout 120 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider --junitxml=/tmp/lodo-phase9-continuation-harness.xml tests/test_lodo_cuda_smoke_contract.py
timeout 180 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9-continuation-fixture.xml tests/test_lodo_cuda_smoke_contract.py tests/test_msmt17_dataset.py
```

| Invocation | Passed | Failed | Skipped | pytest seconds |
|---|---:|---:|---:|---:|
| Admission recheck | 41 | 0 | 0 | 16.46 |
| Relevant LODO/Stage A group | 85 | 0 | 3 | 52.92 |
| Harness namespace/budget corrections | 18 | 0 | 0 | 6.78 |
| Final harness plus original MSMT parser regressions | 34 | 0 | 0 | 26.07 |

Counts overlap and must not be summed as unique tests. Three CUDA-only skips in
the CPU group are not GPU proof. The earlier full 1123-passed/4-skipped suite is
historical Phase 9 evidence, not a fresh continuation full-suite run. Production
bytes remain unchanged. Review caught and corrected the harness validator's
`project_root` keyword support before GPU execution; cached-weight checks were
also added before launch. One CPU command named a nonexistent
`test_msmt17_parsers.py` (exit 4, zero tests); the corrected command above passed.

## D. Eight-case actual CUDA matrix

Every passing row uses GTX 1080 UUID
`GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5`. VRAM is allocated/reserved **GiB**;
time is first/mean of subsequent two **compute steps**, seconds; checkpoint size
is **MiB**. These are observations from three successful updates per case.

| Architecture | Fold | C | Logical GPU (UUID above) | Updates | Peak VRAM GiB | Warm/steady seconds | Images/s | Terminal MiB | SHA/reconstruct | Outcome |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| resnet50 | LODO-1 | 2510 | cuda:0 | 3 | 6.524/6.699 | 13.283/0.562 | 113.8 | 348.14 | PASS | PASS |
| resnet50 | LODO-2 | 2559 | cuda:0 | 3 | 6.524/6.699 | 10.817/0.564 | 113.5 | 349.68 | PASS | PASS |
| resnet50 | LODO-3 | 2494 | cuda:0 | 3 | 6.524/6.699 | 10.741/0.625 | 102.5 | 347.64 | PASS | PASS |
| resnet50 | LODO-4 | 2220 | cuda:0 | 3 | 6.515/6.693 | 11.205/0.610 | 104.9 | 339.07 | PASS | PASS |
| pcb | LODO-1 | 2510 | cuda:0 | 3 | 7.334/7.689 | 16.076/1.098 | 58.3 | 233.52 | PASS | PASS |
| pcb | LODO-2 | 2559 | cuda:0 | 3 | 7.335/7.516 | 15.706/0.793 | 80.7 | 234.10 | PASS | PASS |
| pcb | LODO-3 | 2494 | cuda:0 | 3 | 7.334/7.516 | 18.911/0.847 | 75.6 | 233.34 | PASS | PASS |
| pcb | LODO-4 | 2220 | cuda:0 | 3 | 7.333/7.516 | 15.772/0.747 | 85.7 | 230.10 | PASS | PASS |

Initial ResNet folds 1–3 remain under `exp/lodo_smoke/phase9-20261009T1600/`.
ResNet fold 4 retry and PCB cases are under
`exp/lodo_smoke/phase9-20261009T1610/`. Each case retains its intent, outcome,
checkpoints/manifests, synthetic held-out fixture and applicable console log.
`phase9-20261009T1610/campaign-summary.json` links current matrix/recovery results
and preserves the prior attempt history. Nothing is promoted into official output.

The first ResNet LODO-4 attempt completed training and terminal publication but
failed in synthetic MSMT17 target setup: gallery minimum camera 2 prevented the
original parser from inferring the camera base. The driver stopped before PCB.
Only the synthetic fixture was corrected to include a camera-1 negative and a
cross-camera positive; all four original target-loader CPU checks were added.
An additional read-only diagnostic was attempted before the restart checkpoint
manifest existed; the checkpoint reader rejected the incomplete pair. Final
state assessment runs only after all child processes complete.
The failed attempt remains unchanged, and a fresh namespace rerun supplied the
passing fold-4 row. No production parser or real dataset metadata was repaired.

Observed GPU commands used the following exact environment prefix:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5 LODO_AUTHORIZED_GPU_UUID=GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5
```

The prefix preceded these commands (each outside the managed sandbox):

```bash
timeout 360 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_smoke.py --architecture resnet50 --fold LODO-1 --campaign phase9-20261009T1600
timeout 1800 /home/addirakoze/environments/Reid/bin/python -B /tmp/lodo_phase9_gpu_driver.py
timeout 360 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_smoke.py --architecture resnet50 --fold LODO-4 --campaign phase9-20261009T1610
timeout 1800 /home/addirakoze/environments/Reid/bin/python -B /tmp/lodo_phase9_gpu_driver_retry.py
```

Driver command arrays record every sequential child invocation, including
`--interrupt` cases. Durable driver copies are in their campaign directories.
No OOM, occupied-device fallback or recipe reduction was used.

## E. Runtime validation

Hooks independently verify actual CUDA input/label placement, float32 input,
original input shape, combined C, finite losses and main gradients, Center device
and local half-mean formula, actual optimizer LR-used trace, main/auxiliary counts
and canonical Center ownership. The shared engine also checks finite/rescaled
auxiliary gradients, finite parameters and synchronized complete steps.

For every passing row: exactly 3 primary updates, 3 Center steps for ResNet/0 for
PCB; 192 actual successful image presentations reconciled against consumed
sidecar indices and per-source/global-identity counters. ResNet LR-used begins
.000003/.000003594/.000004188, Center .5; PCB retains backbone/new .01/.1.
Distant official LR boundaries remain covered by cheap CPU integration fixtures.
No update beyond the smoke budget executes.

Throwing guards prohibit target loader/factory/evaluator and legacy output setup
throughout training. Terminal payload/manifest SHA256 is read on CPU, strict
source-C model reconstruction then executes on the authorized CUDA device.
Only after completion, the original excluded-target loader/evaluator processes
11 synthetic query/gallery images with source-owned preprocessing. Target labels
and cameras remain raw. Tiny fixture metrics are non-authoritative and never used
for tuning or selection. No real held-out images are decoded or benchmark matrix
is evaluated. The full provenance-bound Phase 7 publication entry point remains
CPU-qualified; this GPU harness validates terminal/reconstruction and original
retrieval components without weakening real-target metadata checks.

## F. CUDA recovery and interruption assessment

Capability remains **fresh_cuda_no_resume_v1**. CPU retains
**epoch_boundary_rollback_cpu_v1**. Actual CUDA model reconstruction succeeds;
main optimizer, canonical Center/auxiliary state, scheduler, CUDA/CPU RNG,
original sampler ownership, position and exposure are captured in checked
snapshots. Applying optimizer/Center/scheduler/RNG state for CUDA training
continuation remains unsupported and unvalidated. No completed-epoch or bitwise
CUDA recovery claim follows from fresh-model success.

CUDA resume is explicitly rejected before device discovery/source construction,
including requests against these actual GPU snapshots. Intentional interruption
is injected before update 2 in a new smoke process after one successful complete
main/auxiliary step. Its event requires a fresh restart, retains the initial
snapshot, records one discarded successful update and produces no terminal.
Independent CPU reads of actual interruption/restart payloads validate matching
scientific identities, zero initial versus three terminal counters, trained main
optimizer state, scheduler position, one visible CUDA RNG stream, and changed
model/Center tensors without initializing CUDA. Evidence is retained in
`/tmp/lodo-phase9-continuation-state-assessment.json`.
A separate new namespace restarts from seed/pretrained initialization and finishes
three successful updates. Both architectures are assessed independently:

| Architecture | Assessment | Successful updates | Checkpoint MiB |
|---|---|---:|---:|
| resnet50 | EXPECTED_INTERRUPTION | 1 | 129.41 |
| resnet50 | PASS | 3 | 348.14 |
| pcb | EXPECTED_INTERRUPTION | 1 | 116.99 |
| pcb | PASS | 3 | 233.52 |

Review policy: preserve failed artifacts/events; reject resume; allocate a new
unique namespace; revalidate source/map/config/weights/device; restart with zero
counters and initial schedule. This is a tested **fresh restart**, not recovered
continuation. Physical work from the interrupted run is discarded; a long-run
interruption can lose the entire run's completed work. Official acceptance of
that risk, or separately validated CUDA recovery, is a Phase 10 review gate.

## G. Measured resources and limited extrapolation

Per-case warm/steady timings, allocator peaks, host RSS, checkpoint size and UUID
are stored in outcomes. Timing synchronizes CUDA around model/gradient/main/Center
compute; optimizer-only timings are recorded separately. Data loading, admission
query, preflight and checkpoint overhead are excluded from the compute-step rate.
Whole training-call wall times include startup/preflight/publication and are
retained separately. Two steady steps cannot establish sustained thermal, I/O,
allocator or long-run throughput behavior.

Compute-only budget projection = observed mean steady seconds × official updates
/ 3600. The sample range projects the two observed steps independently; it is
not a confidence interval or full uncertainty bound. End-to-end allocation-hours
need additional loader/monitor/checkpoint overhead measurements.

| Architecture | Fold | Compute-only hours | Two-step sample range, hours |
|---|---|---:|---:|
| resnet50 | LODO-1 | 13.31 | 13.25–13.37 |
| resnet50 | LODO-2 | 13.35 | 13.23–13.46 |
| resnet50 | LODO-3 | 14.78 | 13.13–16.44 |
| resnet50 | LODO-4 | 14.45 | 12.95–15.94 |
| pcb | LODO-1 | 28.73 | 26.72–30.74 |
| pcb | LODO-2 | 20.75 | 19.49–22.02 |
| pcb | LODO-3 | 22.16 | 19.35–24.96 |
| pcb | LODO-4 | 19.54 | 19.48–19.60 |

Eight measured terminal files total **2.261 GiB**.
Current smoke artifacts, including retained failed/interrupted/restart runs, total
**4.721 GiB** at the resource snapshot.
Future-review storage options: terminal + one fully populated boundary generation
approximately **4.523 GiB** across eight runs;
terminal + two such generations approximately
**6.784 GiB**.
These estimates use trained terminal sizes, because initial empty-optimizer
snapshots understate trained recovery storage. Add filesystem/metadata growth and
logging space. Cadence/retention are proposals only; no policy is deployed.
Without CUDA recovery, additional snapshots do not provide a validated way to
avoid a fresh restart. Interruption cost can approach the full budget projection
plus unmeasured overhead.

Allocator headroom is 8 GiB minus observed reserved peak, before CUDA context/
driver/nonallocator allocations. PCB's measured peak near device capacity makes
memory-margin review necessary; three updates do not guarantee long-run capacity.
No batch/input/sampler/model change was made to improve feasibility.

## H. Remaining Phase 10 gates

CUHK03 remains detected `processed_partition`, null split ID, with official-new
provenance unverified. Review current source/map/target/pretrained manifests,
code/dirty-state identity and frozen eight configurations; approve worker policy,
CUDA fresh-restart risk or a separately qualified recovery capability, memory
margin, resource reservation duration and checkpoint retention/cadence. Official
launch requires explicit approval. Synthetic smoke metrics are not dissertation
benchmark results or evidence of accuracy/convergence.

## I. Readiness

**COMPLETE — READY FOR PHASE 10 REVIEW**.
All eight actual CUDA cases and the two architecture-specific interruption/new-namespace restart pairs passed. CPU regressions passed and the authorized GPU was isolated from PCB. This is readiness for review, not authorization for official training.
Stop after Phase 9. No Phase 10 task or official training is started.
