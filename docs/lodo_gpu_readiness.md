# Phase 9 LODO GPU readiness and CUDA admission

Continuation status and actual CUDA measurements are recorded in
[lodo_gpu_smoke_validation.md](lodo_gpu_smoke_validation.md). The report below
preserves the initial GPU-deferred audit and CPU implementation evidence.

This is Phase 9 implementation evidence, not an official launch authorization.
Stage A remains frozen. No Phase 10 implementation, official LODO training,
benchmark evaluation, GPU smoke workload, commit or push occurred.

## A. Repository state and preservation

Original checkout: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot`,
branch `main`. Implementation checkout:
`/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`,
branch `feature/lodo-v1`. Both HEADs are
`3b015c964b99bfe8ad4b7842bd174d3bc58af607`. Canonical roots replace
`/home/addirakoze/nobackup/Projects` with `/mnt/hd-data/addirakoze/Projects`.
Both worktrees already contained uncommitted changes from earlier work.
Pre-edit source SHA256/status and artifact size/mtime/inode snapshots are in
`/tmp/lodo-phase9-before.json`; final deltas are in
`/tmp/lodo-phase9-final-audit.json`. These are temporary review evidence.

Phase 9 changes exactly:

- New `reid/utils/lodo_device.py`: explicit UUID admission, read-only inventory,
  bounded process/resource rechecks, device/precision/optimizer validation.
- Modified `scripts/train.py`: LODO-only admission dispatch, operational evidence,
  update guard and explicit fresh-restart interruption record for CUDA.
- Modified `reid/utils/lodo_checkpoint.py`: device-consistent identity and distinct
  fresh-CUDA capability; unsupported recovery fails before tensor mutation.
- Modified `reid/engine/train_loop.py`: optional update-mode admission callback;
  the Stage A default has no callback.
- New `tests/test_lodo_device.py`: CPU-only admission/placement/checkpoint tests.
- New this report. No official/smoke runnable configuration is installed.

The original source/status is preserved. The active PCB job owns
`<original>/exp/pcb/msmt17/` (including checkpoints, logs, metrics and TensorBoard)
and independently continues writing there. Those live files cannot honestly be
reported as byte-unchanged. This task never writes them or controls that process.
LODO configuration/experiment artifacts remain unchanged; no LODO experiment
output directory is created. Dataset, model, loss, sampler, optimizer, scheduler,
collator, evaluator and Stage A device utility files are unchanged in Phase 9.

## B. Gate 9A resource audit

Host `constantine`, audit beginning **2026-10-09 15:05:01 UTC**. Managed sandbox
`nvidia-smi` could not communicate with the driver; the authorized read-only
outside-sandbox audit succeeded. Driver **570.211.01**, driver CUDA ceiling **12.8**,
Python **3.11.2**, Torch **2.7.1+cu126**, torchvision **0.22.1+cu126**,
built CUDA **12.6**. Torch reported CUDA available and compiled architectures
`sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90`; **CUDA initialized: false**.
Both devices report compute capability 6.1, default compute mode and MIG N/A.
These are driver/build observations, not successful CUDA kernel validation.

| Physical GPU | UUID | Device | Total MiB | Free MiB at initial UUID query | GPU utilization | Observed compute owner |
|---|---|---|---:|---:|---:|---|
| 0 | GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c | NVIDIA TITAN Xp | 12288 | 4827 | 99% | PID 2370601, addirakoze, official PCB |
| 1 | GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5 | NVIDIA GeForce GTX 1080 | 8192 | 8103 | 0% | No compute process observed |

PCB command: `python scripts/train.py --config configs/pcb/msmt17.yaml`.
Its `/proc/2370601/cwd` is the canonical original checkout. GPU UUID/process
query establishes that PCB uses physical GPU 0. Free memory there grants no
sharing permission. The initial overview and UUID snapshots differ slightly in
memory/utilization because the active job continues running.

No `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `SLURM_JOB_ID`, `SLURM_JOB_GPUS`,
`SLURM_STEP_GPUS` or `PBS_JOBID` was set. `squeue` and `qstat` were unavailable.
No independent allocation/reservation evidence was found for GPU 1. User
allocation authorization was requested; an idle-looking GPU is insufficient.
Gate 9A: **OWNERSHIP UNCERTAIN**. Gate 9C is deferred without verified allocation.
A second read-only parser snapshot is in `/tmp/lodo-phase9-gpu-audit.json`.
Neither audit selects a CUDA device or allocates a tensor/model.
A final host NVML snapshot at 15:13:42 UTC still reports PID 2370601 holding
7350 MiB on GPU 0 (instantaneous utilization 0%); GPU 1 remains idle-looking.
A sandbox-only `ps` temporarily hid that host PID: this was corrected against
the host GPU query, and no process-exit or completed-job claim is made.

RAM initially: 31 GiB total, 24 GiB available; swap 59 GiB total, 3.6 GiB used.
Project filesystem: 11 TiB total, 3.1 TiB available; `/tmp` filesystem: 916 GiB
total, 781 GiB available. Capacity is an audit snapshot, not a reservation.

## C. Gate 9B implementation and CPU qualification

LODO supports explicit `cpu` or opt-in fresh `cuda`; `auto` is rejected. The
operator must independently establish authorized allocation BEFORE setting
`LODO_AUTHORIZED_GPU_UUID`. This environment declaration is an explicit operator
assertion, not independently verifiable reservation evidence. The code does not
grant ownership from utilization or free VRAM.

For CUDA, `CUDA_VISIBLE_DEVICES` must be exactly the same full UUID, set before
starting Python, and `system.gpu_id` must be integer **0**: the single visible
logical device, even if its physical NVML index is 1. Numeric, abbreviated,
multiple, missing and mismatched visibility are rejected rather than guessed.
CUDA availability, exactly one visible device, its actual Torch UUID and a
compatible compiled architecture must match. No fallback occurs. NVML admission
rejects occupied/active devices, incomplete inventories, unsupported compute/MIG
modes and failed/timed-out queries. Other GPUs are never selected automatically.

Admission is rechecked after runtime discovery, before each epoch/update and
before terminal publication. After admission, only this process's own compute
PID is allowed. An observed new competing process aborts this job; other jobs
are never terminated or modified. These snapshots do not establish a lock:
independent resource reservation remains mandatory, and contention arising
inside a running kernel cannot be prevented by periodic inspection. Each query
has a ten-second timeout. Per-update query overhead remains unmeasured.

The shared engine retains FP32, batch64, original architecture preprocessing,
losses, original PK/random sampling, exact successful-update control, LR-used
semantics, exposure accounting and Center synchronization. No AMP, TF32,
determinism/benchmark, workers or official budget setting is silently changed.
Admission evidence records existing TF32/cuDNN policies; it does not change them.

Model and criterion move to the selected device before optimizers are built.
Identity checks validate model/Center parameters and buffers, floating dtype,
optimizer parameter devices and parameter-shaped optimizer tensor placement.
Adam's non-capturable scalar `step` counter may remain CPU. Engine input/label
transfer already uses the selected device. Checkpoint integrity/reconstruction
reads explicitly remain `map_location='cpu'`, avoiding incidental CUDA allocation
when inspecting or evaluating a GPU-produced artifact. CPU recovery loads into
existing parameters and optimizer ownership through the existing implementation.
Real GPU serialization/reconstruction and allocation still need Gate 9C.

Fresh CUDA snapshots declare `fresh_cuda_no_resume_v1`, distinct from
`epoch_boundary_rollback_cpu_v1`. CUDA resume requests fail before GPU discovery,
source-loader construction or output creation, including already-terminal reopen.
The restore API also rejects fresh-CUDA artifacts for CPU continuation before
model/Center mutation. Initial/full-epoch CUDA `ckpt_last` files are snapshots,
not a validated recovery promise. Interrupted CUDA records specify a fresh
restart and no `recover_from` path. Output collision validation remains unchanged.
No short-budget switch is added to production validators.

Stage A continues using its original device helper/default fallback and epoch
path. Target-free LODO dispatch remains ahead of legacy target/artifact setup.
CPU tests mock every CUDA call: they are not evidence for CUDA execution.

## D. Gate 9C eight-case matrix

No GPU workload ran. A dash means **not measured**, not zero or a passing check.
Class dimensions below are frozen requested dimensions, not GPU-verified counts.

| Architecture | Fold | C | GPU UUID | Updates | Peak allocated/reserved VRAM | Warm/steady step time | Terminal/hash/reconstruction | Outcome |
|---|---|---:|---|---|---|---|---|---|
| ResNet50 | LODO-1 | 2510 | — | — | — | — | — | DEFERRED |
| ResNet50 | LODO-2 | 2559 | — | — | — | — | — | DEFERRED |
| ResNet50 | LODO-3 | 2494 | — | — | — | — | — | DEFERRED |
| ResNet50 | LODO-4 | 2220 | — | — | — | — | — | DEFERRED |
| PCB | LODO-1 | 2510 | — | — | — | — | — | DEFERRED |
| PCB | LODO-2 | 2559 | — | — | — | — | — | DEFERRED |
| PCB | LODO-3 | 2494 | — | — | — | — | — | DEFERRED |
| PCB | LODO-4 | 2220 | — | — | — | — | — | DEFERRED |

Actual architecture/batch64 CUDA forward/backward, finite losses/gradients, exact
2–6-step stopping, Center updates, initial LR/exposure, target isolation, terminal
SHA256/reconstruction, throughput, host RAM and peak VRAM remain unvalidated.
Distant official LR boundary checks continue in CPU integration/update tests.

## E. Recovery and fresh restart

CPU completed-epoch rollback remains validated for ordered nonpersistent 0/2
workers; mid-epoch replay rolls back the entire unfinished epoch. CUDA fresh
execution does not imply CUDA recovery. No bitwise GPU continuation, RNG/order
replay or epoch-boundary CUDA recovery is advertised. A CUDA interruption
requires a new namespace and fresh run; actual CUDA fresh-restart behavior is
pending GPU validation. Unsupported resume rejection is CPU-tested.

## F. Resource planning

Measured here: hardware capacity and idle/occupied snapshots only. There are no
LODO GPU step-time, peak-memory, checkpoint-size/write-time or throughput
measurements. Full-run GPU-hours, memory margin and storage demand therefore
remain **unavailable**, rather than extrapolated from synthetic CPU doubles or
the single-source PCB job. Pending measurements: warmup vs steady timings,
per-update admission overhead, allocator reserved/allocated peaks, actual C/head/
Center/optimizer checkpoint sizes and memory margin at batch64 for all eight cases.

After authorized smoke measurements, use observed seconds/update × 85,200 or
94,200 / 3600, with warmup/checkpoint/guard overhead and uncertainty stated per
architecture/fold. If CUDA resume stays unsupported, an interruption loses the
entire run's completed physical work; worst-case restart cost approaches the full
budget. Do not describe saved CUDA snapshots as reducing that cost.

Later-review options: retain terminal plus one prior validated boundary generation
or two generations; assess full-epoch versus update-interval snapshot cadence from
measured write latency/storage. CUDA recovery validation and policy approval must
precede describing any cadence as recoverable. No cadence/retention policy is
changed or deployed in this phase.

## G. CPU test evidence

Environment: verified Reid interpreter; Python 3.11.2, Torch 2.7.1+cu126,
torchvision 0.22.1+cu126, NumPy 2.1.3, pytest 9.0.3. All tests hide CUDA, disable
bytecode/cache writes and limit CPU threads. Outputs use temporary directories.

From the isolated worktree:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=''
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9-focused.xml tests/test_lodo_device.py
timeout 900 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider --junitxml=/tmp/lodo-phase9-full.xml tests
```

Focused: **41 passed, 0 failed, 0 skipped, 11.59 seconds**. Full suite:
**1123 passed, 0 failed, 4 skipped, 513.98 seconds**, across all 45 test modules.
The four unchanged CUDA skips are the ResNet last-stride CUDA forward case and
the three CUDA loss-combination training-loop cases. All 1082 Phase 8 CPU cases
plus the 41 new admission cases passed; no baseline failure or regression was
observed. Exact JUnit counts/skip names are also saved in
`/tmp/lodo-phase9-validation.json`. Full regression runs outside the managed sandbox
for existing worker tests. No fresh pre-edit full baseline was rerun in Phase 9;
Phase 8's recorded 1082 passed/4 skipped is historical comparison evidence, and
current full tests verify the current state. Skipped CUDA tests are not GPU proof.
No initial focused failure or production workaround occurred. The initial
snapshot command used an unavailable `python` alias; it was rerun successfully
with the verified Reid interpreter before modifications.

## H. Remaining Phase 10 review gates

Independently allocated GPU/explicit ownership; all eight actual CUDA smoke cases;
measured memory/throughput/storage feasibility; reviewed CUDA recovery or accepted
fresh-restart risk; CUHK03 official-new provenance (still unverified; only detected
processed-partition/null split ID supported); current dataset/map/pretrained/code
manifests and frozen eight configurations; reviewed checkpoint cadence/retention;
and explicit official launch authorization. No official readiness claim follows
from the CPU admission tests.

## I. Classification and stop

**PARTIAL — GPU SMOKE DEFERRED**: Gate 9B implementation and CPU regressions pass;
Gate 9A has no verified independently allocated GPU and Gate 9C has no executed
cases. Stop after this Phase 9 report; Phase 10 requires separate authorization.
