# Phase 9R: CUDA boundary continuation qualification

**NON-AUTHORITATIVE. INCOMPLETE — FIXES REQUIRED.**
The implementation and bounded ResNet restoration checks are complete, but the
ResNet optimizer-trajectory comparison failed and PCB GPU testing was stopped; it does not authorize official training,
full benchmark evaluation, Phase 10 implementation, commits or pushes.

## A. Repository and preservation

Implementation: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`,
`feature/lodo-v1`; original: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot`,
`main`. Both HEADs: `3b015c964b99bfe8ad4b7842bd174d3bc58af607`.
Canonical project paths begin `/mnt/hd-data/addirakoze/Projects/`.
Pre-edit source SHA256/status and artifact size/mtime/inode snapshots are in
`/tmp/lodo-phase9r-before.json`; final deltas are in `/tmp/lodo-phase9r-final-audit.json`.
Existing uncommitted work is preserved. Main source/status/configs and all inactive
artifacts are unchanged. Only the independently running official PCB job writes
its own original `exp/pcb/msmt17/`; this task never writes there or signals it.

Phase 9R changes four existing Stage B files: `reid/utils/lodo_checkpoint.py`,
`reid/utils/lodo_device.py`, `reid/utils/lodo_protocol.py`, `scripts/train.py`.
New files: `reid/utils/lodo_recovery.py`, `tests/test_lodo_cuda_recovery.py`,
`tests/lodo_cuda_recovery_smoke.py`, `tests/test_lodo_cuda_recovery_smoke_contract.py`,
and this report. `docs/lodo_recovery_contract.md` gains a current-policy pointer.
All model/loss/source loader/sampler/collator/optimizer/scheduler/evaluator and
Stage A code paths/configurations remain unchanged. The shared training engine
is reused. No images are copied or modified. The seven required documents were
read completely before editing, along with the actual implementation/tests.
Final source/status and preservation evidence is recorded alongside the campaign
in `audit-evidence/`, including CPU JUnit reports and the failed numerical comparison.

### Pre-edit person-reid-autonomous-robot

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

### Pre-edit person-reid-autonomous-robot-lodo

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
?? docs/lodo_gpu_smoke_validation.md
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
?? tests/lodo_cuda_smoke.py
?? tests/test_lodo_checkpoint.py
?? tests/test_lodo_config.py
?? tests/test_lodo_cuda_smoke_contract.py
?? tests/test_lodo_device.py
?? tests/test_lodo_evaluation.py
?? tests/test_lodo_integration.py
?? tests/test_lodo_updates.py
?? tests/test_multi_source_dataset.py
```

## B. Capability and supported boundary

New capability: **`epoch_boundary_rollback_cuda_v1`**, versioned independently of
legacy CPU `epoch_boundary_rollback_cpu_v1` and fresh CUDA `fresh_cuda_no_resume_v1`.
Omitting the recovery policy preserves the previous defaults; CUDA resume without
the explicit opt-in remains rejected. Checkpoint/identity schema stays version 1;
recovery metadata carries capability, worker contract, policy version and execution
environment. Old fresh-CUDA snapshots are not upgraded or made resumable.

Explicit operational configuration (illustrative only, no runnable official config):

```yaml
train:
  recovery:
    version: 1
    policy: completed_epoch_boundary
    every_epochs: 1
    keep_generations: 2
```

CUDA supports FP32, **zero workers**, nonpersistent/in-order original PK16×4 or
shuffled random64/drop_last. CPU retains tested zero/two ordered nonpersistent
workers with default prefetch. Actual loader worker count must match requested
configuration. Unsupported workers, prefetch, timeout, policy/version/keys fail.
CUDA resume requires the same full UUID, driver, host, Python/Torch/torchvision/
NumPy/CUDA-build versions, TF32 flags and deterministic/benchmark policy recorded
at save. No different-device compatibility or arbitrary mid-epoch replay is promised.
The record is a bounded environment contract, not a fingerprint of every native
library, OS setting or external machine condition; Phase 10 still freezes the full
environment. No AMP/TF32/determinism/scientific recipe is changed.

A completed boundary has zero cursor, a completed-epoch marker, contiguous epoch
records, no partial/failed epoch, matching epoch/update totals, and consistent main/
auxiliary/exposure state. Initialization is an explicit zero-update boundary.
Terminal is distinct, budget-complete, and can only be verified without updates.

## C. State, publication and fail-closed flow

Saved/restored: model parameters/buffers, canonical learned Center tensor exactly
once, main Adam/SGD state, separate Center SGD, update scheduler, actual next LR,
main/auxiliary/attempt/forward/backward counts, epoch records, source/identity
exposure, Python/NumPy/Torch CPU/single-visible-CUDA RNG, PK-owned Python/NumPy RNG,
loader/sampler generators and supported boundary position. Restoring copies into
existing Center Parameters owned by its optimizer. Torch optimizer loading places
parameter-shaped state on each parameter device; explicit placement verification
checks actual Adam/momentum/Center tensors afterward. Adam scalar step may remain
CPU. Scheduler loading consumes no update; RNG restoration is last.

Resume resolves canonical same-run paths and reads SHA256 sidecars/payloads on CPU.
The same original source inventories/map and fresh CPU components are constructed;
fold/C/map/data/partition/config/model/optimizer/scheduler/Center/worker/environment,
progress, tensor and RNG compatibility is checked before CUDA runtime discovery.
Only then does unchanged UUID admission allocate on logical cuda:0 and restore.
Actual CUDA smoke asserts CPU validation finishes while CUDA is uninitialized and
compares live restored model/optimizer/Center/scheduler exactly to saved CPU tensors
before the first resumed update. Corrupt/incompatible/partial/legacy state cannot
silently initialize a new training trajectory. A verified terminal reopen returns
without CUDA discovery, training or artifact rewrite; actual GPU-generated terminal
artifacts were tested this way with throwing CUDA/engine guards.

Opt-in boundaries use immutable `ckpt_epoch_<epoch>_u<count>_g<nonce>.pth` generations
and SHA256 JSON sidecars. Nonces use OS randomness, never training RNG. Each pair
is serialized/read back, fsynced and verified before atomic `recovery_index.json`
publication. Only afterward may owned, verified older generations be pruned to two.
The terminal remains separate. A crash before index publication leaves the prior
index/last good generation intact; an orphan is never auto-promoted. Corrupt bytes
are preserved for audit outside the validated retention window.

`train.save.resume` may name a retained immutable generation, or the same-run
`checkpoints/ckpt_last.pth` **logical pointer** resolved through the index. It is
not a copied/overwritten last payload under this opt-in. Explicit bad generations
fail. The logical latest request may fall back to its prior verified generation;
resume events record failed candidates, selected hash, boundary count, timestamp,
UUID/environment and admission evidence. Missing/unusable index or all invalid
pairs fail without repair. The pair/index are separate atomic publications, not
a filesystem transaction; only verified indexed pairs are committed recovery state.
SHA256 is integrity evidence for trusted local artifacts, not authentication.

Mid-epoch interruption discards the interval after the last durable boundary.
Default proposed cadence one means the unfinished epoch; larger configured cadence
can also discard completed-but-unsaved epochs. Resume restores saved exposure and
replays that interval, counting it once in the accepted trajectory. Events retain
observed discarded physical work separately. Abrupt process death may leave no
observable count for discarded work; no counts are invented. Partial main/Center
steps are never saved as resumable. Full-run budgets remain 85200/94200.

## D. CPU validation

Python `/home/addirakoze/environments/Reid/bin/python`, threads 1, CUDA hidden,
bytecode and pytest cache disabled. From the isolated worktree:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=''
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9r-focused.xml tests/test_lodo_cuda_recovery.py
timeout 1200 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider --junitxml=/tmp/lodo-phase9r-full.xml tests
timeout 120 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9r-harness-final.xml tests/test_lodo_cuda_recovery_smoke_contract.py
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9r-focused-final.xml tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_recovery_smoke_contract.py
```

| Invocation | Passed | Failed | Skipped | seconds |
|---|---:|---:|---:|---:|
| Initial new recovery tests | 29 | 0 | 0 | 19.31 |
| Broad CPU regression | 1174 | 0 | 4 | 533.13 |
| Final harness guards | 6 | 0 | 0 | 6.02 |
| Final strengthened recovery/harness tests | 40 | 0 | 0 | 29.49 |

Counts overlap. Broad collection preceded five added cadence/missing-pair cases and
six added harness cases; the final focused run covers all 40 current cases. No
production code changed after the broad suite. The four skips are existing CUDA
cases, not GPU proof. Full coverage includes Stage A config/data/sampler/collator,
training/loss/optimizer/scheduler, checkpoint/model reconstruction and evaluation;
CPU LR restoration immediately before/after warmup and all decays; Center/optimizer
exact restoration; ordered zero/two-worker augmentations/dropout; target throwing
guards; rollback and terminal no-op; corrupt/missing pairs; index interruption and
fallback; same-C wrong-map/fold/config/C/environment/state failures before mutation.

An earlier sandbox-only command (`timeout 180 ... pytest -xq ... tests/test_lodo_device.py
tests/test_lodo_config.py tests/test_lodo_checkpoint.py`, JUnit target
`/tmp/lodo-phase9r-initial.xml`) timed out in restricted multiprocessing (exit 124),
without an observed assertion failure. It is not a passing run. Outside-sandbox
CPU regressions above supersede it. An earlier harness-only run also passed 6
cases in 6.23 seconds. No failed assertions were hidden or recipes altered.

## E. GPU allocation and PCB protection

The user explicitly confirmed **Phase 9R** authorization for
**`GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5`**, independently of Phase 9 permission.
Before allocation, renewed read-only audit at **2026-10-09T19:37:29.274075+00:00**:

```text
index, name, uuid, memory.total [MiB], memory.used [MiB], memory.free [MiB], utilization.gpu [%], compute_mode, driver_version
0, NVIDIA TITAN Xp, GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, 12288 MiB, 7357 MiB, 4827 MiB, 0 %, Default, 570.211.01
1, NVIDIA GeForce GTX 1080, GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5, 8192 MiB, 5 MiB, 8103 MiB, 0 %, Default, 570.211.01
gpu_uuid, pid, process_name, used_gpu_memory [MiB]
GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, 2370601, python, 7350 MiB
PID USER     STAT     ELAPSED COMMAND
2370601 addirak+ Rl+     09:23:06 python scripts/train.py --config configs/pcb/msmt17.yaml
```

TITAN Xp UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` remained excluded.
The driver runs one child at a time; each child audits exact UUID/resource/process
ownership before setup, the production selector checks actual Torch UUID/visibility,
and the unchanged guard rechecks before updates/publication. No fallback or recipe
reduction. Final read-only inventory at **2026-10-09T20:00:50.193136+00:00**:

```text
index, name, uuid, memory.total [MiB], memory.used [MiB], memory.free [MiB], utilization.gpu [%], compute_mode, driver_version
0, NVIDIA TITAN Xp, GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, 12288 MiB, 7357 MiB, 4827 MiB, 98 %, Default, 570.211.01
1, NVIDIA GeForce GTX 1080, GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5, 8192 MiB, 5 MiB, 8103 MiB, 0 %, Default, 570.211.01
gpu_uuid, pid, process_name, used_gpu_memory [MiB]
GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, 2370601, python, 7350 MiB
PID USER     STAT     ELAPSED COMMAND
2370601 addirak+ Rl+     09:46:27 python scripts/train.py --config configs/pcb/msmt17.yaml
```

## F. Actual CUDA interruption/continuation

Executed cases: real ResNet50; PCB is deferred. Configured recipes remain actual LODO-1 sources/C=2510/map, FP32 batch64,
256×128 PK16×4 ID+Triplet+Center/Adam or 384×128 original random/six-head ID/SGD.
Pretrained cache is required, hashed and never downloaded. Target factories,
evaluator and legacy output setup throw throughout fresh/resumed training. No
target image or benchmark evaluation occurs.

Short **test-only budget 5 and two-batch test epochs** are in-process harness
substitutions, not supported production options. The observed loader exhausts
a two-batch prefix of the original sampler without changing its algorithm/RNG
ownership/selected indices; its public PK/random sampler remains original. CPU
fixtures exercise complete original small-dataset epochs, including two workers.
GPU tests qualify boundary/device restoration with these bounded test epochs;
**they do not traverse a full real-dataset epoch or establish long-run replay**.
Real source metadata/mapping and classifier dimensions are retained. Production
rejects short budgets/output namespaces before and after fixture scope. Official
LR boundaries and budget validators are unchanged.

| Architecture | Durable boundary | Before interruption | Discarded | Resumed count → terminal | Final Center steps | Accepted image presentations | Outcome |
|---|---:|---:|---:|---|---:|---:|---|
| resnet50 | 2 | 3 | 1 | 2 → 5 | 5 | 320 | RESTORE PASS; TRAJECTORY FAIL |

ResNet50 used three separate processes: uninterrupted, fresh/interrupted,
and resumed. Resumed processes execute three primary updates starting from count
two; they never re-execute committed updates 1–2. Physical combined work is six
updates/384 presentations, of which one/64 is explicitly discarded; final accepted
trajectory is five/320. Sampler indices, transformed input SHA256s, LR traces,
source/identity exposure and identities match uninterrupted execution. Accepted
source presentations: Duke 80, CUHK03 116, MSMT17 124 (total 320). The discarded
64 presentations comprise Duke 16, CUHK03 44 and MSMT17 4. Class count is 2510;
mapping SHA256 is `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`.
The resumed boundary SHA256 is
`31ca39b70ce4f69ccbbe4363c7f8158051b70a5b8867cc267c1cf09b4cde78a8`. Immediate
restoration is tensor-exact. Final losses and model/Center differences satisfy the predeclared tolerance:
loss rtol=.001/atol=.001; float state rtol=.001/atol=.0001. **Adam moments fail
the state tolerance. GPU execution stopped before PCB, as required by section 10.**
No threshold was relaxed and no audit was rerun to hide this failure.

| Compared terminal state | Max absolute difference | Elements beyond state tolerance |
|---|---:|---:|
| Model/buffers | 0.0000457763671875 | 0 / 28,709,824 |
| Center | 0.000000536441802979 | 0 / 5,140,480 |
| Adam state | 0.000580295920372 | 521 / 57,301,281 |
| Center optimizer / scheduler | exact | 0 |

First-convolution Adam first-moment relative L2 difference is 0.32381%.
Loss differences for updates 3–5 are 0.000000953674, 0.000003814697 and
0.000003814697. All restored tensors matched the checkpoint exactly *before*
the first resumed step; the observed optimizer discrepancy occurs subsequently.
This is consistent with nondeterministic CUDA arithmetic, an inference rather
than an established root cause. No near-identical optimizer replay is claimed.
The user has been asked whether structural continuation qualification is
acceptable; no response changing this criterion has been received. PCB's actual
interruption/resume gate is **NOT RUN**, and CPU coverage cannot replace it.

- **resnet50** uninterrupted terminal: `8c987316147599b8e800088b331b083b371a956922b60002920c970bcb09fcc0`; resumed terminal: `2107589ae0374eb624bc546ca131dacc5c5d5eaa76b2fc024c58375435bbd8b7`. Final state differences: see the measured numerical comparison above and `/tmp/lodo-phase9r-numerical-assessment.json`.

ResNet LR used updates 1–5: `.000003, .000003594, .000004188, .000004782,
.000005376`; Center LR .5. PCB backbone/new LR .01/.1 remains unchanged in code but was not exercised on
GPU in Phase 9R. ResNet scheduler/main/Center counters all end at 5. State payload SHA256s differ because run paths,
operational config/provenance/time/state are serialized; tensor equality is assessed
separately. Observed equality does not guarantee bitwise nondeterministic CUDA replay.

Additional CPU checks on **actual GPU-generated artifacts** validate terminal,
coherently rehashed wrong-map rejection, corrupt-copy rejection, model nonmutation
and terminal no-op while throwing guards forbid CUDA initialization. No authoritative
checkpoint is altered. `cpu-actual-artifact-checks.json` records results, live source
counts/map and original sampler epoch lengths. Negative copies stay in smoke output.

Exact execution environment prefix:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5 LODO_AUTHORIZED_GPU_UUID=GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5
```

It preceded `timeout 1800 /home/addirakoze/environments/Reid/bin/python -B
/tmp/lodo_phase9r_gpu_driver.py`. The driver command arrays preserve these three child invocations (same prefix):

```bash
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_recovery_smoke.py --architecture resnet50 --campaign phase9r-20261009T193908 --mode continuous
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_recovery_smoke.py --architecture resnet50 --campaign phase9r-20261009T193908 --mode interrupt
/home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_recovery_smoke.py --architecture resnet50 --campaign phase9r-20261009T193908 --mode resume
```
 Same prefix was
used for CPU-only actual-artifact inspection:
`timeout 600 /home/addirakoze/environments/Reid/bin/python -B
/tmp/lodo_phase9r_actual_negatives.py /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo/exp/lodo_smoke/phase9r-20261009T193908 resnet50`. Driver/source/outcomes/logs/events/
checkpoints are under `exp/lodo_smoke/phase9r-20261009T193908/`; all are non-authoritative.

## G. Measured resources and operational proposals

| Architecture | Peak allocated/reserved GiB | Resumed terminal MiB | Trained checkpoint write seconds | Preliminary audit → restored event seconds | Median post-first compute step seconds |
|---|---:|---:|---:|---:|---:|
| resnet50 | 6.741/6.926 | 348.16 | 7.98–15.27 | 115.98 | 0.567 |

Write timings include validation, serialization/readback, fsync/hash, generation
index publication/retention work; not just raw disk transfer. The 115.98-second restart interval starts at the harness preliminary inventory
audit and includes CPU provenance/component validation and restoration before the
event. Production CUDA admission itself occurred at 19:46:16.608442 UTC and the
restored event at 19:46:27.162266 UTC (10.55 seconds apart). Neither interval
is a pure tensor-upload measurement. Per-child wall times and individual compute-step times
remain in outcomes. Compute timing excludes loading/admission/checkpoint overhead;
few steps, including cold/post-boundary outliers, do not establish sustained rates.
Available disk at checkpoint snapshots: at least
**3.089 TiB**, not a reservation.
Measured campaign files including negative copies total **2.380 GiB**.

Original sampler reference lengths/upper bounds and illustrative rollback cost:

- resnet50: initial original epoch 764 batches; metadata-derived upper bound 782 updates per epoch. Upper-bound update count × observed median compute step ≈ 7.39 minutes of compute-only replay, plus measured restart and unmeasured data/guard/I/O overhead.

PK epoch length varies with original sampler state. These are estimates from short
measurements, not end-to-end recovery guarantees or reservation durations. Cadence
one loses at most work since the preceding durable epoch boundary; cadence k can
lose up to k epochs of work (plus observations unavailable after abrupt death).
Recommend every completed epoch and two validated generations for review.
Retention terminal + two trained generations costs approximately three trained
checkpoint sizes per run, plus progress growth, manifests/logs and retained failed
artifacts. During publication, extra temporary/unindexed bytes may coexist.
Official cadence/retention/resource policy still requires Phase 10 approval.

## H. Limits and approvals

No arbitrary mid-epoch CUDA continuation, multiworker CUDA, persistent/unordered
workers, changed GPU/recorded environment, AMP, bitwise replay, full real-epoch GPU
validation or official-length convergence/thermal/I/O qualification is claimed.
Nondeterministic CUDA settings are preserved. CPU fixtures show exact stochastic
continuation in their tested configurations; GPU evidence is bounded above.
CUHK03 remains detected processed_partition/null split ID with official-new
provenance unverified. Phase 10 must review code/config/data/map/pretrained/full
environment manifests, worker policy, cadence/retention, memory margin and resource
reservation before separate official launch approval. No Phase 10 work begins here.

## I. Readiness

**INCOMPLETE — FIXES REQUIRED.** Implementation, CPU regressions, exact actual
ResNet50 checkpoint restoration, actual-artifact CPU negative checks and preservation
gates passed. ResNet50 post-resume Adam trajectory exceeded the documented tolerance;
PCB's actual GPU continuation pair was consequently not started. Resolve the numerical
qualification criterion or demonstrate correctness within the retained criterion, then
complete PCB before claiming Phase 9R ready. No Phase 10 or official run is authorized.
