# Stage B checkpoint and recovery contract — Phase 6

## Phase 9R.3 — explicit strict CUDA policy (CPU and bounded TITAN qualification verified)

The current opt-in policy implementation is described in
[lodo_cuda_policy_integration.md](lodo_cuda_policy_integration.md). Earlier text
below remains the historical Phase 6/9R contract. No historical numerical failure
has been reclassified, and no official launch is authorized.

Stage B may explicitly select `repro.cuda_policy: strict_cuda_deterministic_v1`,
with `system.device: cuda`, `repro.deterministic: true`, and `repro.benchmark: false`.
Export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts. Selection without
Stage B opt-in, unknown versions, incompatible declarations, missing/late workspace,
already-initialized CUDA, or incompatible approved TF32 controls fails closed.
The actual `scripts/train.py` LODO path activates the policy before device discovery,
seeding, loaders and output creation. It enables deterministic algorithms with
warn-only disabled, disables cuDNN benchmarking and enables cuDNN determinism.
Matmul TF32 remains false and cuDNN TF32 true. Runtime guards detect policy/control
drift. Absent selection, Stage A and historical CPU/original-CUDA behavior remain
unchanged; explicit `original_cuda_v1` has the same scientific identity as omission.

Strict scientific identity and checkpoint payload/identity/manifest use schema v2.
Original identity/checkpoints remain v1; the CPU-first reader accepts v1/v2 and
rejects inconsistent versions/policies. No migration, promotion or rewriting of
old artifacts occurs. The complete strict definition is scientific identity;
recovery also binds actual deterministic/warn-only/workspace controls and cuDNN
version alongside the existing exact UUID/driver/host/Python/Torch/torchvision/
NumPy/CUDA-build/TF32 contract. Optimizer group/implementation settings, Center
Parameter ownership, scheduler, counters, RNG and map checks remain intact.
Hardware or recorded-library changes block exact recovery and require separate
qualification. This bounded contract does not fingerprint all native libraries.

CUDA recovery remains opt-in `completed_epoch_boundary`, FP32, zero ordered,
nonpersistent workers, original PK/random samplers, two immutable generations and
a separate terminal. Mid-epoch work rolls back to the last validated completed
persisted epoch, including exposure/counters; it is not arbitrary cursor resume.
Terminal reopening validates on CPU and does no CUDA discovery or update. CPU
retains its existing tested 0/2-worker recovery. Schema-v2 terminal inspection is
CPU-compatible; legacy Stage A model-only readers are unchanged.

Production-path ResNet50 and independently gated PCB A/B/C are now verified on
the explicitly allocated TITAN Xp, exact UUID and runtime recorded in
[lodo_cuda_policy_integration.md](lodo_cuda_policy_integration.md). Both pass all
nine pair/update checks3–5 bitwise exactly, using one immutable update2 checkpoint
per architecture. Actual process SIGTERM after successful3 and production rollback
to2/replay through5 preserve320 accepted presentations, LR-used, RNG and exposure.
Two immutable generations, separate terminal no-op and CPU negative checks on
actual v2 CUDA-produced artifacts pass. Source artifact namespace:
`exp/lodo_smoke/phase9r3-titan-20261010T005917/`.

This tests LODO-1, FP32, zero workers, budget5 and a fixture-defined two-batch test
epoch using prefixes of the original samplers. Every branch calls actual LODO
orchestration; a controlled same-canonical-run boundary snapshot view preserves
production same-run validators and unchanged checkpoint bytes. It does not add a
production fork API or prove full real-dataset epoch/long-run replay. Paired short
original/strict timing and memory measurements are documented with limits. The
prior CPU-only report and initial inspector namespace failure remain archived;
historical original-policy numerical failures remain failures. Official GPU
allocation, cadence/retention, provenance freeze and launch still require review.


Phase 9R adds a separately opt-in CUDA completed-epoch policy and retained
generations; see [CUDA recovery qualification](lodo_cuda_recovery_qualification.md).
The Phase 6 text below records the legacy CPU/default contract and its historical
GPU restriction. Existing fresh-CUDA snapshots are not upgraded.

This document describes implementation capability, not authorization for official
training. Scientific protocol, recipes, budgets, source splits and samplers are
unchanged. Phase 7 evaluation and Phase 9 GPU evidence remain separate gates.

Checkpoint schema: `lodo_schema_version: 1`.
Recovery capability: `epoch_boundary_rollback_cpu_v1`.

Supported execution is FP32 CPU with the original PK or shuffled random sampler,
0 or 2 in-order nonpersistent workers, and the original public sample/collator.
Other worker counts, persistent workers, unordered loading, concurrent iterators
and GPU recovery are rejected. Worker count is never changed automatically.
Review this restricted capability before official launch; changing runtime
configuration or claiming GPU continuation requires new evidence.

Recovery `checkpoints/ckpt_last.pth` is saved at initialization and after each
fully exhausted epoch. Its next-epoch cursor is zero; no active iterator/worker
state survives at this boundary. Python, NumPy, Torch CPU, initialized CUDA RNG,
PK-owned Python/NumPy RNG, and explicit loader/sampler generator states are
stored. CUDA state storage is not a GPU recovery guarantee. Worker processes
are freshly seeded by the original DataLoader on each next epoch. Deterministic
CPU fixtures exercise Python/NumPy/Torch augmentations, model dropout, original
PK and random sampling, and explicit generator ownership with 0 and 2 workers.

There is no exact mid-epoch checkpoint/replay claim. A mid-epoch interruption
retains the prior checkpoint; the entire unfinished epoch is discarded and
replayed after restoring its boundary. `recovery_events.json` records interruption,
discarded attempts/successes and per-source presentations, then the rollback
restart. Replayed presentations replace discarded training counters rather than
being added twice. Exposure is training-trajectory accounting; discarded physical
work remains separate audit evidence. Abrupt process death cannot produce an
in-memory interruption event; restarting still records the persisted rollback
boundary, but cannot invent counts for unobserved work.

The checkpoint saves the canonical `criterion.center_loss.state_dict()` exactly
once plus its separate SGD state for ResNet50. No whole LossBundle state is saved
because `center` aliases `center_loss`. Loading copies tensors into existing
parameters referenced by the Center optimizer. PCB saves neither Center state.
Main optimizer and scheduler state, successful/main/auxiliary counters, exposure,
epoch results, cursor, RNG and recoverable mapping metadata accompany the model.

Identity validation covers architecture/variant, fold, ordered sources/target,
source class count, actual classifier heads, feature width, raw Center branch,
coefficient, local half-mean reduction, optimizer groups, schedule, FP32,
configuration identity, map/sidecar/data provenance digests and recovery mode.
The frozen scientific identity reuses Phase 3 canonical serialization. Full cfg
is embedded; operational resume paths remain outside scientific identity.
Actual worker/generator ownership and source provenance must still match.
Validation precedes model/Center tensor mutation. Legacy checkpoints without
canonical learned centers cannot exact-resume through this interface; legacy
Stage A model-only evaluation and historical APIs remain unchanged.

Restoration order: construct fresh model/criterion/optimizers/scheduler; validate
checkpoint and sidecar; load model and canonical centers into existing parameters;
load main and auxiliary optimizers; restore scheduler count and next LR WITHOUT
stepping; restore RNG last. Recovery cannot continue failed/inconsistent progress.

Target-free orchestration is an explicit branch in `scripts/train.py` before
legacy output setup and target-loader/evaluator construction. Source metadata
preflight may inspect target inventories; target image processing/evaluation is
never part of Stage B training. No `ckpt_best` is selected.

`checkpoints/ckpt_terminal.pth` and its JSON manifest are published only after the
exact successful primary budget, synchronized auxiliary count and consistent
state validate. Its manifest records `budget_completed`, planned/actual main and
auxiliary counts, identities and checkpoint SHA256. A partial terminal cursor is
recorded but is never used for further training. Reopening a verified terminal
performs zero updates, even when the supplied resume path points to stale last.

Writes use a same-filesystem temporary file, verification, fsync and atomic
replace. JSON integrity/completion metadata is published separately afterward.
These two replaces are not a filesystem transaction: interruption between them
can leave an uncommitted payload or mismatched pair. Readers fail closed; a
filename alone never proves completion. No automatic repair or promotion occurs.
A prior checkpoint remains intact on failures before payload replacement; failure
between the two publications may require an independently retained validated
checkpoint. Durable multi-generation retention/cadence needs operational review.
SHA256 detects accidental corruption, not malicious replacement of both trusted
local files. Torch payloads must come from a trusted source.

Existing run reuse requires canonical same-run `ckpt_last.pth` or
`ckpt_terminal.pth`, verified hash/schema/config, then complete current
source/model/recovery compatibility validation before updates. Fresh output paths
retain Phase 3 namespace and symlink/collision protections. Terminal files are
never overwritten. No existing Stage A artifacts are converted or rewritten.
