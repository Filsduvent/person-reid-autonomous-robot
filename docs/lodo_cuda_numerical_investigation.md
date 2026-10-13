# Phase 9R.1 CUDA numerical divergence investigation

**NON-AUTHORITATIVE — GPU DIAGNOSTICS DEFERRED — AUTHORIZATION REQUIRED.**
Phase 9R remains INCOMPLETE. No CUDA workload ran in Phase 9R.1. The original
numerical criterion remains unchanged. This report records completed CPU/read-only
forensics and explicitly incomplete CUDA investigation/qualification gates.

## A. Repository preservation

Original `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot` (`main`);
implementation `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`
(`feature/lodo-v1`). Both HEADs are `3b015c964b99bfe8ad4b7842bd174d3bc58af607`.
Canonical roots are under `/mnt/hd-data/addirakoze/Projects/`.

Only three files are added in Phase 9R.1: this report,
`tests/lodo_cuda_numerical_diagnostic.py` and
`tests/test_lodo_cuda_numerical_diagnostic.py`. No production file, previous report,
scientific recipe, optimizer, sampler, transform, seed, dataset or checkpoint is
modified. There is no restoration fix without demonstrated evidence of a defect.
The CPU analyzer reads trusted existing SHA256-validated artifacts, reports compact
statistics/parameter names, and refuses to overwrite its output. It never allocates
CUDA or copies images. Previous failed artifacts/history are preserved.

Source hashes, status and artifact size/mtime/inode baselines are in
`/tmp/lodo-phase9r1-before.json`. Final preservation evidence and small CPU reports
are retained in the new isolated `exp/lodo_smoke/phase9r1-cpu-20261009/` namespace.
The independent official PCB job may update its own `exp/pcb/msmt17/`; those live
artifacts are not claimed byte-unchanged. This task never controls that process or
writes to its outputs. No commits, pushes, reset, stash, clean, official training,
real benchmark evaluation or Phase 10 work occurred.

### person-reid-autonomous-robot (before Phase 9R.1)

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

### person-reid-autonomous-robot-lodo (before Phase 9R.1)

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
?? docs/lodo_cuda_recovery_qualification.md
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
?? reid/utils/lodo_recovery.py
?? tests/lodo_cuda_recovery_smoke.py
?? tests/lodo_cuda_smoke.py
?? tests/test_lodo_checkpoint.py
?? tests/test_lodo_config.py
?? tests/test_lodo_cuda_recovery.py
?? tests/test_lodo_cuda_recovery_smoke_contract.py
?? tests/test_lodo_cuda_smoke_contract.py
?? tests/test_lodo_device.py
?? tests/test_lodo_evaluation.py
?? tests/test_lodo_integration.py
?? tests/test_lodo_updates.py
?? tests/test_multi_source_dataset.py
```

## B. Authorization and device audit

Read-only initial host audit: **2026-10-09 17:26:13.319338 America/Sao_Paulo**
(**2026-10-09T20:26:13.319338+00:00**). GTX 1080 UUID
`GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5`: 8192 MiB total, 8103 MiB free,
0% utilization, no compute process. Driver 570.211.01. TITAN Xp UUID
`GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`: PID 2370601, official
`python scripts/train.py --config configs/pcb/msmt17.yaml`, 7350 MiB compute
allocation. It is excluded; no GPU fallback is permitted.

Phase 9R.1 section 2 says **“Do not infer ongoing authorization for this new
Phase 9R.1 campaign.”** A new exact-UUID allocation/authorization question was
sent. No answer authorizing this campaign has been received. Idle VRAM and old
Phase 9/9R permission are not authorization. Consequently no GPU allocation, A/B/C run, deterministic-mode experiment or
PCB test occurred. CUDA availability checks in CPU pytest collection are not
GPU execution evidence.
All current Python diagnostic/test commands hide CUDA. The read-only inventory
uses nvidia-smi and ps only, not CUDA initialization. Final audit at
2026-10-09 17:38:48.155366 America/Sao_Paulo (20:38:48.155366 UTC) again
showed no GTX compute process and PCB PID 2370601 active on TITAN (7384 MiB).

## C. Original criterion and complete 521-element analysis

The original campaign driver is preserved at
`exp/lodo_smoke/phase9r-20261009T193908/driver.py`; its STOPPED summary is unchanged.
Its actual comparison statements are:

```python
assert a.shape==b.shape and a.dtype==b.dtype
torch.testing.assert_close(a,b,rtol=.001,atol=.0001)
# Nonfloating tensors:
assert torch.equal(a,b)
# Loss trajectory:
torch.testing.assert_close(torch.tensor(continuous['losses'][2:]),
                           torch.tensor(resumed['losses']),rtol=.001,atol=.001)
```

Recursive dictionaries require identical keys, sequences identical lengths, and
other values exact equality. Original state `a` is continuous and `b` is resumed:
`abs(continuous - resumed) <= 0.0001 + 0.001 * abs(resumed)` elementwise. Both
absolute and relative tolerances apply additively, not as independent pass gates.
Shapes/dtypes are checked; all inspected tensors here are CPU loads of GPU artifacts.
The forensic utility uses torch.isclose in that same direction, reports errors in
float64, and does not adopt any new acceptance threshold. Loss tolerance is
`rtol=0.001, atol=0.001`. Indices, transformed-input hashes, identities, LR and
exposure must match exactly. Structural pre-step restoration equality is a separate
requirement and was asserted exactly by the prior actual CUDA harness.

Terminal hashes:

- Continuous: `8c987316147599b8e800088b331b083b371a956922b60002920c970bcb09fcc0`.
- Resumed: `2107589ae0374eb624bc546ca131dacc5c5d5eaa76b2fc024c58375435bbd8b7`.

All **521** failed elements are in first moments (`exp_avg`) across 13 parameters:

| Parameter | State/group ID | Failed elements | Max abs error | Max relative error over nonzero reference | Relative L2 | Max reference magnitude |
|---|---:|---:|---:|---:|---:|---:|
| backbone.0.weight | 0 | 435 | 0.000580295920372 | 12.4954 | 0.0032381 | 0.153036 |
| backbone.4.0.conv1.weight | 3 | 24 | 0.000191625207663 | 35.4106 | 0.00289866 | 0.0585792 |
| backbone.4.1.conv1.weight | 15 | 1 | 0.000112631358206 | 24.8467 | 0.0027544 | 0.0525282 |
| backbone.4.1.conv2.weight | 18 | 14 | 0.000220335554332 | 1177.69 | 0.00303875 | 0.0499709 |
| backbone.4.1.conv3.weight | 21 | 4 | 0.000237527769059 | 32.9051 | 0.00275161 | 0.0525082 |
| backbone.4.2.conv2.weight | 27 | 4 | 0.000169468577951 | 89.9705 | 0.00298018 | 0.0474339 |
| backbone.4.2.conv3.weight | 30 | 12 | 0.000200367067009 | 33.4809 | 0.0027612 | 0.0577028 |
| backbone.5.2.conv2.weight | 57 | 3 | 0.000253138248809 | 400.257 | 0.00221425 | 0.0381292 |
| backbone.5.3.conv1.weight | 63 | 1 | 0.000113575777505 | 331.348 | 0.00277879 | 0.0133121 |
| backbone.5.3.conv2.weight | 66 | 14 | 0.00019295696984 | 676.558 | 0.00299775 | 0.0175989 |
| backbone.7.0.downsample.0.weight | 138 | 1 | 0.000107486150227 | 2062.95 | 0.00185759 | 0.0232066 |
| backbone.7.1.conv1.weight | 141 | 1 | 0.000120248645544 | 3142.17 | 0.00171594 | 0.0201303 |
| backbone.7.2.conv2.weight | 153 | 7 | 0.000341000457411 | 540.529 | 0.00123617 | 0.0756601 |

All 161 first-moment tensors are non-bitwise-equal; 13 exceed the tolerance.
All 161 second-moment tensors are also non-bitwise-equal but none exceed it
(max absolute error **0.00000558281317353**). All 161 scalar step tensors are
exactly equal at 5. All inspected optimizer tensors are finite. Param-group IDs
map one-to-one to the names above through saved group `param_name` and independent
identity-contract `parameters`; both payload mappings match. No swapped optimizer
ownership is demonstrated. This CPU serialization audit does not replace a live
CUDA Parameter-object identity assertion in the pending experiment.

For the first convolution, absolute-error quantiles p50/p90/p99/max are
0.0000195011962 / 0.0000767898746 / 0.000221787859 / 0.000580295920.
Its first-moment relative L2 error is **0.3238103%**. 435/521 failed elements are
there. Large elementwise relative ratios elsewhere are dominated by near-zero
reference entries; they are not an overall relative error measure. Zero-reference
changes are separately counted, rather than hidden behind an epsilon. The complete
bounded JSON contains per-tensor shapes, dtype, devices, finite/exact status,
absolute quantiles, relative errors, magnitudes and failure ranges for every state.
No full tensors or dataset records are dumped.

## D. Three-path comparison and discovered confound

**New finding:** the old reference was a separate fresh run, not the same update-2
checkpoint from which the interrupted run resumed. Their SHA256-validated,
index-committed completed-boundary snapshots already differ **before resume**:

| Update-2 state | Nonexact tensors | Max absolute difference | Elements beyond original tolerance |
|---|---:|---:|---:|
| Model/buffers | 250 / 324 | 0.00000597164034843 | 0 |
| Center | 1 / 1 | 0.000000238418579102 | 0 |
| Main Adam | 322 / 483 | 0.000164346653037 | **15** |
| Center optimizer / scheduler | exact | 0 | 0 |

Continuous boundary SHA256:
`05ed9d5a48db82ef3525d4d57154ee85dd6ffc7fad0a909dcf6b4a1281f012fa`.
Interrupted boundary SHA256:
`31ca39b70ce4f69ccbbe4363c7f8158051b70a5b8867cc267c1cf09b4cde78a8`.

At this boundary, two failed first-moment elements belong to `backbone.0.weight`
and 13 to `backbone.7.1.conv1.weight` (state ID 141). All saved global and
sampler/loader RNG fields are equal: Python, NumPy, Torch CPU, single CUDA stream,
PK Python/NumPy, loader generator and sampler generator. Scientific identity,
parameter groups, LR and source map match. Transformed input hashes match for
updates 1–3. Separate fresh-run losses are:

| Update | Continuous | Independent fresh/interrupted | Absolute difference |
|---|---:|---:|---:|
| 1 | 10.261112213134766 | 10.261112213134766 | 0 |
| 2 | 10.09913158416748 | 10.099130630493164 | 0.000000953674316406 |
| 3 | 10.19058609008789 | 10.190585136413574 | 0.000000953674316406 |

Thus numerical variation exists between two fresh executions before checkpoint
restoration. The old final comparison cannot isolate restoration as its cause.
This is a demonstrated qualification-design confound, not proof that recovery is
correct or that a particular nondeterministic kernel is responsible.

Required new controlled comparison remains **NOT RUN**:

| Path | Required setup | Current evidence |
|---|---|---|
| A | Continue in the producing process from one newly saved validated update-2 boundary | NOT RUN — new GPU authorization pending |
| B | Fresh process restores that exact A checkpoint, advances 2→5 | NOT RUN |
| C | Second independent fresh process restores the exact same checkpoint, advances 2→5 | NOT RUN |
| A/B, A/C, B/C | Compare every instrumented update and final state | NOT RUN |
| Optional D | Reload the same checkpoint inside the producing process | NOT RUN |

A retained old checkpoint cannot recreate its historical uninterrupted process or
cuDNN cache. A valid new A must be recorded as that process continues; simply
loading the old checkpoint three times would give three restored paths, not A.
All inputs/indices/transforms must be independently reconciled against A, without
silently forcing batches or changing sampler/RNG behavior.

## E. First-divergence localization

The earliest difference in the *available loss logs* is update 2 of independent
fresh execution. State divergence is proven by the completed update-2 boundary.
The actual first differing tensor/operation may be in update-1 forward, backward
or optimizer execution; scalar loss equality at update 1 does not prove equality
of features or gradients. No intermediate tensors exist to answer that question.
**Mandatory operation-level localization is INCOMPLETE.** No attribution to Adam
arithmetic, Center, cuDNN or cuBLAS is presented as established fact.

Pending diagnostic-only instrumentation must record all twelve requested stages:
checkpoint read; restored model/Center; restored optimizer and Parameter ownership;
scheduler/RNG; selected/transformed batch; forward and separate losses; backward
before Center scaling; after Center scaling; before main step; after main step;
after Center step; after scheduler/accounting. Use the original shared engine,
hooks/wrappers only in the harness, and bounded CPU summaries/digests plus targeted
first-convolution tensor copies. Retain per-update A/B, A/C and B/C exactness/error
statistics. If the first discrepancy is within a large forward block, narrow
instrumentation to its constituent operations before claiming localization.
Timing/memory effects from instrumentation require disclosure and controlled
repeatability; observation can change autotuning conditions.

## F. CUDA determinism, optimizer and RNG analysis

Prior checkpoint environment: host constantine, Python 3.11.2, Torch 2.7.1+cu126,
torchvision 0.22.1, NumPy 2.1.3, CUDA build 12.6, driver 570.211.01, exact GTX UUID
above. Stored policy: FP32, cuDNN benchmark true, deterministic false, matmul TF32
false, cuDNN allow_tf32 true. No current experiment changes those defaults.
The old bounded environment record does not establish the actual selected cuDNN
algorithm, cuBLAS workspace environment, deterministic-algorithms global switch,
cuDNN library version or execution cache. Flag equality is not kernel equality.

PyTorch documents that cuDNN benchmark noise can select different algorithms in
separate runs and that disabling benchmark alone does not guarantee deterministic
operations. Restoring RNG streams is separate from eliminating numerical
nondeterminism. See [PyTorch 2.7 reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html).

Actual serialized Adam flags: foreach null, fused null, capturable false,
differentiable false, amsgrad false; betas (.9,.999), eps 1e-8, decay .0005.
Inspected installed Torch 2.7.1 source selects `_default_to_fused_or_foreach` with
`use_fused=False`, then `_multi_tensor_adam` or `_single_tensor_adam` according to
actual parameters. CUDA normally selects foreach when supported; the executed
kernel path in prior runs was not logged and is not asserted as observed.
See [version-pinned Adam source](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adam.py).

Original engine order is zero_grad(set_to_none=True), forward/loss, backward,
finite-gradient checks, Center inverse-weight scaling, main Adam step, Center SGD
step, counter/exposure update, then scheduler. Center optimizer references canonical
learned Center Parameters, not a replacement module. Triplet contains matrix
multiplication/reductions; Center uses indexed class selection with repeated PK
labels; backbone uses CUDA convolution/BN. These are investigation locations, not
identified culprits. No loss reduction, optimizer group or gradient order change
is justified. Captured RNG equality at update 2 is verified, but later B/C streams
and live associations remain to be tested.

A diagnostic-only deterministic-mode experiment is **NOT RUN**. If newly authorized,
record torch.use_deterministic_algorithms, cuDNN policy, CUBLAS_WORKSPACE_CONFIG,
TF32/backend versions and optimizer path before execution. Any deliberately different
settings must be separately labelled and restricted to a new diagnostic process;
never edit the production recipe or rehash an old checkpoint to masquerade as an
original-policy run. Deterministic success cannot qualify original nondeterministic
execution. No alternative threshold has been adopted.

## G. Corrections and regression evidence

No production defect is demonstrated; **no production correction was made**.
The new CPU utility corrects analysis scope by checking both retained pre-resume
boundaries and preserving the original comparison direction. Its ten focused
checks cover tolerance agreement with torch.assert_close, zero-reference reporting,
integer equality, stable param-name/ID mapping with ambiguous ownership rejection,
and no RNG draw. Existing recovery tests remain unchanged.

## H. ResNet50 qualification

**FAIL under original Phase 9R criterion remains recorded.** Exact live restoration
before the resumed step passed historically, while the terminal Adam comparison
failed. The new pre-boundary evidence shows that comparison does not isolate a
resume defect. It does not turn FAIL into PASS.

**PROPOSED — REQUIRES APPROVAL** only if controlled A/B/C evidence establishes a
scientifically defensible basis: separately assess exact restoration/invariants and
post-step numerical continuation against measured original-recipe repeatability.
No new numerical cutoff is proposed from terminal maxima alone; no threshold or
policy is changed. First complete authorized causal diagnostics. Unknown cause
keeps provisional capability unqualified for official readiness.

## I. PCB qualification

**NOT RUN — GATED.** ResNet50 must pass the original criterion, or a separately
reviewed explicitly approved replacement, before PCB CUDA continuation. A revised
acceptance policy has not been approved. No PCB workload is started in this phase.

## J. Recovery invariants

Both actual retained update-2 pairs pass SHA256, completion-boundary and committed
index selection checks on CPU. Source class count is 2510; map SHA256 is
`e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`.
Original schema/capability/cadence/retention and CPU-first compatibility code remain
unchanged. The 40 current recovery/harness cases recheck exact CPU stochastic
continuation with 0/2 ordered nonpersistent workers, model/Center/main/aux/scheduler
restoration, cadence 1/2 rollback/exposure, retained generations, corruption/missing
pairs, atomic-index interruption, same-C wrong map/fold/environment/state rejection,
terminal no-op and target guards. CUDA metadata doubles are CPU tests, not actual
GPU execution. Previous actual ResNet 2→5 continuation/320 accepted presentations
and actual-terminal CPU rejection/no-op results remain historical Phase 9R evidence,
not rerun Phase 9R.1 successes. Distant warmup/decay correctness remains covered by
existing cheap fixtures; no scheduler or training budget is altered.

## K. Tests, exact commands and resources

Working directory: isolated LODO worktree. CPU environment for all commands:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=''
```

Executed commands:

```bash
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9r1-existing-focused.xml tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_recovery_smoke_contract.py
timeout 300 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_numerical_diagnostic.py --analyze-prior /tmp/lodo-phase9r1-forensic-analysis-final.json
timeout 120 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9r1-forensic-tests.xml tests/test_lodo_cuda_numerical_diagnostic.py
timeout 120 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_numerical_diagnostic.py --audit-boundaries /tmp/lodo-phase9r1-boundary-audit-final.json
timeout 120 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase9r1-forensic-tests-final.xml tests/test_lodo_cuda_numerical_diagnostic.py
timeout 180 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider --junitxml=/tmp/lodo-phase9r1-stage-a.xml tests/test_checkpoint.py tests/test_checkpoint_reconstruction.py tests/test_train_loop_optim.py
```

The existing-focused and Stage A commands used PYTHONPATH=. (tests resolve sibling
imports through pytest); analyzer/forensic tests used PYTHONPATH=.:tests. The first
command ran outside the sandbox for two-worker fixtures. Others ran with CUDA
hidden; bytecode/cache writes disabled. JUnit evidence:

| Invocation | Passed | Failed/errors | Skipped | JUnit seconds |
|---|---:|---:|---:|---:|
| existing-focused | 40 | 0/0 | 0 | 31.036 |
| forensic-tests-final | 10 | 0/0 | 0 | 6.797 |
| stage-a | 46 | 0/0 | 3 | 64.557 |

An earlier forensic utility run passed 9 cases in 7.719 JUnit seconds before the
additional nested RNG-array test. It overlaps the final ten and is not added to
them. All three skips are original CUDA-only Stage A cases and provide no GPU
qualification. No failing assertion or baseline failure is hidden. The prior broad
1174-passed/4-skipped Phase 9R regression is historical; no broad suite is claimed
rerun in this CPU-forensic-only phase. Production hashes remain identical to the
new baseline. CPU analyzer completed within its 300-second timeout; no precise
analyzer wall-time or peak RSS measurement was collected. Diagnostic JSON is about
389 KiB; no huge tensors or source image data are retained. GPU VRAM/runtime for
Phase 9R.1 are **NOT MEASURED**, rather than inferred from old smoke results.

## L. Readiness and stop

**GPU DIAGNOSTICS DEFERRED — AUTHORIZATION REQUIRED.** CPU forensics establishes
fresh-run divergence before restoration and identifies every failed terminal state
tensor. It does not establish the first divergent CUDA operation or scientific
validity of the provisional continuation capability. A/B/C comparisons, diagnostic
operation traces, controlled repeatability and subsequent gated PCB qualification
remain unexecuted. Phase 9R remains INCOMPLETE; no official/Phase 10 readiness is
claimed. Await the exact-UUID Phase 9R.1 allocation authorization before CUDA work.
