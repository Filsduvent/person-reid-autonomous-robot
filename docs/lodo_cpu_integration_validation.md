# Phase 8 CPU integration validation

This validation concerns implementation plumbing and Stage A compatibility.
Synthetic outputs/metrics are **NON-AUTHORITATIVE**. It does not authorize Phase 9,
GPU work, official training/evaluation or campaign launch. The Phase 2 documents
retain their historical planning-time readiness statements; the Phase 6/7
contracts and observed tests describe subsequent implementation capability.
No scientific setting is changed to reconcile those historical statements.

## Test inventory and execution

The baseline includes every existing `tests/test_*.py` module: configuration and
legacy YAML, all four dataset parsers, transforms/public samples/collation,
sampling, real ResNet/PCB models and losses, optimizer groups, epoch/update
schedulers, training/orchestration, Center/checkpoint/recovery, evaluation/ranking,
cross-domain matrix/selection, artifacts, smoke CLI plumbing and edge foundations.
The final suite additionally includes `tests/test_lodo_integration.py`.
No test module is excluded. Existing CUDA availability guards skip GPU tests;
those skips are not evidence for GPU execution or recovery.

Environment: `/home/addirakoze/environments/Reid/bin/python`, CPU only,
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.` and
`CUDA_VISIBLE_DEVICES=''`, Python `-B`, pytest `-p no:cacheprovider`. Worker tests
run outside the managed sandbox with a command timeout; outputs use pytest temp
paths. Exact commands/counts/durations are recorded in validation evidence.

## Eight integration workflows

The new parametrized test exercises four frozen folds × both architectures,
using actual `scripts.train.main` and `scripts.evaluate.main`, resolved temporary
YAML, original fold/config/preflight, virtual adapter and deterministic mapping,
original source image loaders, public samples/collator, unchanged architecture
transforms (256×128 or 384×128), original PK16×4 or random64/drop_last, real losses,
main/Center optimizers, update LR, shared engine, recovery and terminal publication,
original target loader/evaluator, Stage B output and comparison utilities.

Each source has 8 synthetic identities with 12 generated images each: combined
C=24 and 288 virtual training images. No authoritative dataset is opened for
training. A complete first epoch precedes an interruption in the next epoch.
Two successful updates remain after restoring the persisted boundary. Final
budget is `first_epoch_updates + 2`, ending inside the following epoch. Tests
reconcile every consumed global index against source/identity exposure, prove
repeated draws, partial-epoch mean denominators and no extra terminal update.

Throwing guards cover target factories/builders/evaluator and legacy output
setup throughout fresh, interrupted, resumed and already-terminal training.
Reference/resumed model tensors, Center tensors, optimizer/scheduler states,
RNG states, loss sequence, transformed inputs, consumed indices and exposure
are compared. Recovery events account separately for one discarded successful
update and two attempts. Terminal reopening performs zero forward/update work.

Only after training finishes does the synthetic target loader run. Each workflow
publishes an explicitly incomplete comparison, then a complete comparison using
three synthetic compatible Stage A cross-domain records. Original Stage A
selection/output code is not called. Terminal bytes remain unchanged.

## Explicit test doubles and limits

The expensive ResNet convolution/PCB part computation is replaced with a small
linear compute double retaining combined-source heads, 2048/1536 widths, dropout,
and distinct ResNet pre-BN/post-BN routing. Real model forward/reconstruction and
stripe constraints are covered separately by the complete regression suite.
The double does not establish convolutional training throughput, memory feasibility,
pretrained initialization quality or official retrieval performance.

The existing Phase 6 fixture substitutes only the frozen budget expectation and
redirects output validation to the pytest temporary project. For post-training
integration a checked AST clone of the existing Phase 7 terminal validator replaces
exactly its two architecture budget literals with the same short budget. EVERY
other gate statement is retained. This is test-only code, not a production short
budget escape hatch. The original production gate is first required to reject the
short terminal artifact; another test verifies production scientific validation
rejects short budgets after fixture teardown. No counters are fabricated or
promoted to 85,200/94,200. No executable official/smoke configuration is shipped.

Workflow orchestration uses zero workers. Existing component/recovery tests cover
both zero and two ordered nonpersistent workers, PK replacement and variable epoch
lengths, unchanged RNG/draws, unused prefetch exclusion, rejected attempts,
nonfinite/partial optimizer failure and unsupported modes. This distinction must
remain explicit; eight workflows are not eight two-worker qualification runs.

## Independent numerical and integrity checks

New cheap fixtures execute optimizer steps with actual parameter gradients and
pre-step LR hooks at ResNet updates 1/500/501/28400/28401/49700/49701/85200 and PCB
1/31400/31401/94200. Existing full 85,200/94,200 cheap traces and restore tests
remain part of the suite. Loader length never multiplies Stage B milestones.

Integration hooks independently check local Center `0.5 * mean(sum(diff²))`, raw
2048-D routing, coefficient .0005, actual inverse-weight-rescaled Center gradient,
SGD LR .5 and ownership of the same C×2048 parameter. PCB checks exact six-head
summed ordinary CE, no smoothing/Triplet/Center and preserved group LR ratio.

Existing Phase 3–7 negatives cover fold/map/config/data/C/D/hash/schema compatibility,
leakage/path aliases, unsupported recovery modes, incomplete terminals, corruption,
metrics/filtering equivalence, comparison means/envelopes/tie-break/percentage
points and Stage A isolation. An added publication-interruption fixture rejects a
result without its final hash proof and rejects collision reuse without repair.

## Phase 9 resource and recovery blockers

No LODO CUDA memory/throughput or CUDA recovery capability is verified. Recovery
remains `epoch_boundary_rollback_cpu_v1`; production training currently rejects
GPU/auto execution. Merely setting a CUDA device is not a supported launch path.
Review actual device capacity, combined C/Center/head memory and optimizer state,
activation/workspace peaks, throughput, checkpoint size/write latency/retention,
worker/prefetch settings and CUDA stochastic/numerical continuation before a
separately approved GPU capability. Protect the pre-existing PCB jobs; dedicated
resource ownership, run order and collision-free outputs require review.

CUHK03 official-new provenance remains unverified. Historical Stage A references
may lack sufficient target metadata/filter evidence. Path/hash checks do not prove
absence of duplicate content or the same people across datasets. Strategy-level
comparisons do not isolate domain diversity. These limits survive passing tests.

## Observed Phase 8 results — 2026-10-09

| Gate | Passed | Failed | Skipped | pytest seconds |
|---|---:|---:|---:|---:|
| Full pre-edit baseline, 43 modules | 1060 | 0 | 4 | 472.79 |
| New focused integration/numerical/integrity cases | 22 | 0 | 0 | 44.69 |
| Full final suite, 44 modules | 1082 | 0 | 4 | 508.76 |

All eight workflows passed: four complete first-epoch updates, then two updates
in a partial terminal epoch (six primary total; six Center auxiliary for ResNet,
zero auxiliary for PCB). Reference/resumed CPU states matched bitwise under the
explicit compute double. Retained exposure was 384 presentations; distinct image
indices were 269 for PK and 267 for random. In canonical source order, PK source
presentations were 128/120/136; random source presentations were 128/128/128 in
this seed/fixture, without any balancing algorithm. Interruption records separate
two discarded attempts and one discarded successful update. Both complete and
incomplete comparison paths passed for every workflow.

The initial new test failed once (12.55 seconds) because it assumed evaluation
datasets require `num_classes`. The existing contract uses raw `pids`, cameras
and marks. Only that new assertion was corrected to inspect distinct raw PIDs;
no loader or production behavior changed. No production defect or baseline
failure was found. All pre-existing Phase 3–7/source/config bytes remained
unchanged; Phase 8 adds only this document and the new test module.

Exact commands, from the isolated LODO worktree:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=''
timeout 600 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider tests
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -xq -p no:cacheprovider --junitxml=/tmp/lodo-phase8-focused.xml tests/test_lodo_integration.py
timeout 900 /home/addirakoze/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider --junitxml=/tmp/lodo-phase8-full.xml tests
```

The full baseline/final commands ran outside the managed sandbox for existing
worker tests; focused workflows used zero workers. Environment: Python 3.11.2,
Torch 2.7.1+cu126, torchvision 0.22.1+cu126, NumPy 2.1.3, pytest 9.0.3. The CUDA
package suffix does not imply CUDA execution. Four existing CUDA-only cases
skipped; no modules were excluded. No GPU validation or official experiment ran.

Review evidence is saved in temporary files: `/tmp/lodo-phase8-validation.json`,
`/tmp/lodo-phase8-workflows.json`, `/tmp/lodo-phase8-results.json`,
`/tmp/lodo-phase8-final-audit.json` and the JUnit XML above. Temporary evidence is
not an authoritative experiment result or a durable campaign archive. The
original checkout's source/status remained unchanged; its pre-existing PCB
MSMT17 job independently continued updating its own artifacts. LODO worktree
experiment/config artifacts remained unchanged. No commits, pushes or Phase 9
work occurred. Phase 8 is complete for CPU review; Phase 9 blockers above remain.
