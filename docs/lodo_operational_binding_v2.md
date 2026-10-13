# Unified Stage B operational binding v2

This is the operational contract for future runs from the unified checkout.
Scientific protocol `lodo_v1`, all eight historical YAMLs/manifests, checkpoint
schemas 1/2, model/loss/sampling/scheduling and retrieval semantics remain unchanged.
This uncommitted implementation is for review. It grants no execution authority.

## Historical compatibility

The scientific freeze `4a03fb0d37cec2e41af92b8ff6e681c91ff5c387` and external
launcher release `7b72587577d087bbe88b4dd1b2de4ac2e4f1443d` continue to identify
the completed ResNet50 LODO-1 run in the original LODO worktree. Historical
manifests retain their original prefreeze source snapshot, root, proposed output
and NOT_APPROVED/NOT_ALLOCATED fields. Those historical fields are never rewritten
as declarations about this unified implementation.

Existing checkpoint/result readers remain compatible for read-only inspection.
The unified launcher accepts future binding v2 only; it does not dispatch a
historical v1 receipt or silently resume/relocate an original checkpoint. Original
same-run recovery remains available in the frozen original implementation under
its existing scope. The new wrapper always executes its own absolute
`scripts/train.py`, using the same shared engine.

## Freeze and binding

After review and an explicitly approved commit, create one external, operator-owned
0600 binding per candidate at:

`.local/lodo_operational/v2/bindings/<configuration-stem>.json`

`reid.utils.lodo_operational.make_binding(config, root=..., commit=...,
run_id=..., policy_path=...)` prepares the object. It requires the actual full,
clean unified HEAD. It does not write it, approve it, or invent a commit. Use
`write_private` for exclusive publication only after review. No real binding or
execution declaration is created during Phase 1. Do not use synthetic test SHAs
or records outside temporary tests.

The binding contains exactly:

- Integer schema version 2 and `lodo_unified_operational_binding` record type.
- Canonical actual checkout, full reviewed commit and Git source-tree identity.
- Historical scientific and launcher commit references.
- Exact candidate config path/file SHA256, historical manifest path/file SHA256,
  immutable digest and scientific configuration digest.
- Full frozen source dataset provenance, canonical root, map/sidecar/data hashes,
  class and image dimensions, recoverable mapping references and exclusion proof.
- Exact cached pretrained artifact identity, strict CUDA execution/runtime and
  checkpoint recovery contracts.
- Designated full TITAN UUID, zero workers, interpreter path/hash and approved
  standing-allocation policy path/hash.
- A future canonical run leaf beginning `unified-`, under the unchanged logical
  `exp/lodo/v1/<architecture>/<fold>/seed42` namespace.

Validation compares the complete object, including types, against current
independently reconstructed evidence. Git status alone is insufficient: every
tracked file is compared directly to its Git blob and executable mode; ignored
or untracked executable/config additions fail membership checks. Candidate files
and manifests must equal their historical Git blobs. New code is verified against
the actual unified commit, not the obsolete v1 code snapshot. Config/map/pretrained
science remains independently verified against the unchanged historical manifest.

The historical YAML is not edited to choose a future run. The launcher supplies
only bound `experiment.output_dir` and, on continuation, the selected immutable
`train.save.resume` generation as operational overrides. Scientific configuration
identity excludes the run leaf. An old checkpoint cannot be moved to this leaf
and silently accepted by the original same-run validators.

## Scoped operator declaration

`.local/lodo_operational/v2/approval.json` must be an actual-UID-owned private
regular file, read using a single descriptor with O_NOFOLLOW. It has these exact
keys:

| Key | Value |
| --- | --- |
| schema_version | integer 2 |
| record_type | local_operator_execution_declaration |
| root / commit | exact canonical unified checkout / actual reviewed full HEAD |
| approved_by | UID-derived operator, not USER or LOGNAME |
| approval_reference | actual explicit execution approval text or genuine reference |
| training | config-relative path → `{binding_sha256, allow_resume}` |
| evaluation | request-relative path → `{request_sha256, binding_sha256, terminal_sha256}` |

Empty training or evaluation scope grants no authority for that action. Each
training binding is validated, including other declared folds. Training scope
never authorizes target evaluation. Continuation additionally requires its exact
Boolean `allow_resume`. Evaluation requires a separately approved exact request
and completed terminal hash. No declaration automatically approves all eight
candidates, other architectures/folds, new output leaves or changed source/data.

This record is a local audit guard. The same Linux account can create or modify
it. SHA256 proves byte identity, not authenticated intent; no independent security
boundary, signature, institutional approval or reservation identifier is claimed.

## Launch and direct-entry admission

After the future reviewed freeze, binding and separately scoped execution approval:

```bash
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml --resume
```

`--check` is read-only and needs a real reviewed binding but no execution
declaration. Missing binding/freeze must refuse, including in the current review
worktree. It creates no directories, logs, tmux sessions or CUDA context.

Before dispatch and again inside detached tmux, the wrapper validates binding,
source/science/config/data/map/sidecar/pretrained identities, exact runtime,
standing policy, GPU UUID/process contention, compatible compiled CUDA capability,
10,000 MiB free VRAM, 64 GiB free disk and output/recovery namespace. The source-only
metadata guard throws on image decoding, target-loader construction and CUDA
initialization. Every selected-device compute PID is refused, including another
operator-owned job. No process is killed and no alternate device is selected.

The standing TITAN research allocation is reused subject to laboratory sharing
rules. It is distinct from execution approval and fresh technical availability.
Changed allocation or a new reservation requirement needs clarification before
execution. No reservation identifier is invented.

The detached guard holds an advisory GPU lock across exec and repeats admission.
Unique external receipts/stdout/dependency records live under
`.local/lodo_launch/receipts/launch-*`. Only controlled environment variables are
saved; shell secrets are excluded. tmux receives the exact UUID, strict workspace,
Python path, thread and no-bytecode environment explicitly. The trainer creates
its own official directory only after admission.

`execution-admission.json` is private, schema 2, status `ADMITTED_TRAINING`. It
pins binding and operator-record bytes, actual config/run/root/commit, full resolved
config digest, selected checkpoint/hash and controlled launch environment.
It also binds PID, process start ticks and boot identity across exec, rejecting
stale receipt replay by a different process or a reused PID. The
launcher forwards its exact path and SHA256. `scripts/train.py` validates these
against the actual loaded config, root, scope, namespace and environment before
loaders/device/output setup. A Boolean alone, copied receipt, modified config,
foreign checkpoint, stale approval or changed environment cannot admit Stage B.
The ordinary Stage A path needs none of these records.

## Recovery and evaluation

The worker reuses the complete CPU-first checkpoint restoration validator without
loading pretrained weights on resume. It validates model, learned Center and its
optimizer, main optimizer, successful-update scheduler, RNG, exposure, provenance
and position. The index-selected retained generation and hash are pinned and
rechecked inside tmux. Terminal payload/sidecar, missing run/index, incompatible
state or collisions refuse; no implicit fresh restart, orphan promotion or repair.
The shared trainer performs the actual restoration and completed-epoch rollback.

Stage B evaluation uses the unchanged CLI:

```bash
python scripts/evaluate.py --config <separately-approved-future-evaluation-request.yaml>
```

Future requests should live under the ignored external
`.local/lodo_operational/v2/evaluation_requests/` directory so their post-training
publication does not dirty the scientific freeze.

It rejects arbitrary overrides and `--weight`, requires the bound future run's
exact-budget terminal, exact request bytes, separate evaluation scope and zero
workers. It reuses `lodo_evaluation.evaluate_lodo` and the original CPU FP32 retrieval
evaluator, target parsing/filtering, source preprocessing and isolated result
namespace. Admission/completion records are external under
`.local/lodo_evaluation/v2/evaluation-*`. Existing results refuse republication.
There is no target evaluation during training or target-based model selection.

## Scientific limits retained

CUHK03 official-new correspondence remains unverified. Historical Stage A target
hash/filter provenance is incomplete and PCB cross-domain references are missing.
The completed official evaluation retains its documented approval-trail limitation
and inherited mark-2 feature extraction, excluded before ranking. Nothing here
repairs or reclassifies those historical facts. Recovery qualification is bounded
to tested deterministic policy/runtime/epoch boundaries. One seed and fixed budgets
do not establish statistical variability or convergence optimality. Stage A/B
comparisons concern complete strategies, not domain diversity alone.
