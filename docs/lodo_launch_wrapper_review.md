# External one-command LODO launcher — review proposal

Status: local operator audit model approved by the user on 2026-10-12;
operational launcher release authorized. This release authorizes no scientific
execution. The original scientific implementation remains frozen at
`4a03fb0d37cec2e41af92b8ff6e681c91ff5c387`.

## Accepted operator audit model

The user explicitly approved Option 2: the operator record is an execution
declaration and audit guard. It is not cryptographically authenticated and the
same Linux account can create or modify it. Its purpose is reproducibility and
prevention of accidental launches, not an independent security boundary against
the account owner. SHA256 binds recorded bytes; it does not authenticate intent.
No signing key, institutional reservation identifier or approval identifier is
invented. The earlier authenticity blocker is resolved by this explicit choice.

The release commit applies only to this external launcher, its tests and this
document on `review/lodo-launch-wrapper`. The actual full release SHA is obtained
with `git rev-parse HEAD` and recorded after committing in the ignored external
receipt `.local/lodo_launcher_release/release-<actual-commit>.json`; it cannot be
embedded in its own tracked commit. The scientific checkout, eight candidate
configurations/manifests, Stage A and historical artifacts stay unchanged.

Approval of this audit model and launcher release does not declare the first
scientific execution approved. The real execution declaration remains absent
until the user explicitly authorizes experiment scope and continuation. Read-only
`--check` remains available without a declaration and cannot start training.

## Reviewed launch boundaries

The wrapper uses an absolute path to the canonical frozen trainer and imports
only frozen repository modules in its metadata worker. Internal preflight
requests are bound to that checkout, candidate and namespace before repository
imports. Internal CLI modes cannot be combined with public or read-only flags.
Declaration files must be owned by the actual UID, regular, private (0600) and
read through one non-symlink descriptor. The approved wrapper executable is
compared directly with its Git blob, even if Git status flags hide a modification.
Declaration and output admission are rechecked after long metadata validation,
and the declaration is rechecked immediately before trainer exec.

All original 72 focused CPU test cases are retained. Additional tests enforce
schema-v2 declaration bindings to the exact scientific commit, config bytes,
manifest path/full bytes/immutable digest, designated GPU and experiment scope.
Changing the fold, architecture, ordered sources, target, output namespace,
configuration hash or manifest hash invalidates the declaration.

Read-only first-run resume correctly refuses because there is no official run or
checkpoint yet. Successful resume is covered by synthetic CPU checkpoints and
the frozen full recovery validator; no official checkpoint is invented for tests.

## Interfaces

From this separate wrapper worktree, with the existing Reid environment active:

```bash
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml --resume
```

The same interface accepts each of the eight configurations in the original
frozen candidate index. Configuration paths resolve against the original LODO
worktree, not against this review worktree. No arbitrary overrides are accepted.

An additional read-only review command needs no execution approval:

```bash
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml --check
```

`--check` does not create directories, receipts, tmux sessions or CUDA contexts.
It is a metadata/runtime/resource check, not launch authorization. With `--resume`
it additionally validates the retained same-run recovery generation on CPU.

## Two independent freezes

The launcher runs the existing `scripts/train.py` in the original canonical
worktree `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo`.
That checkout must remain clean on `feature/lodo-v1` at the exact original SHA.
It imports the original verifier and reconstruction/checkpoint APIs from that
worktree in a fresh CPU-only process. The review checkout's copied training code
is never selected by the wrapper.

This is deliberate: adding a launcher to the original `scripts/` directory would
invalidate both its clean-HEAD check and the frozen source membership check.
The external launcher therefore needs a separately approved release commit; it
does not require rewriting the original scientific commit, configurations,
manifests, output namespaces, historical evidence or recovery state.

Do not copy this wrapper into the original checkout. Integrating it there later
would require a separately approved implementation freeze revision, new source
inventory validation and truthful revised operational provenance. This proposal
performs none of those operations.

## Checks and execution

Before dispatch, and again inside the detached tmux process immediately before
trainer execution, the wrapper checks:

- Exact original commit, branch, clean worktree and frozen candidate index.
- Candidate membership and YAML/manifest/index identities.
- Existing frozen verifier: scientific recipe, splits, metadata hashes, images
  and identity counts, mapping, sidecar, recoverable map and pretrained SHA256.
- Exact approved standing-policy document and SHA256, without reconfirming the
  unchanged allocation or fabricating a laboratory reservation number.
- Matching qualified interpreter/runtime/driver/cuDNN, compiled CUDA support,
  exact TITAN UUID, zero competing compute PIDs, zero utilization, at least
  10,000 MiB VRAM and at least 64 GiB free on the output filesystem.
- Strict deterministic controls activated before CUDA initialization; metadata
  guards forbid CUDA initialization, image decoding and target loaders.
- Fresh output collision rejection, or full CPU-first same-run recovery
  compatibility through the existing restoration validator.
- Separately approved, clean wrapper release and actual scoped execution
  permission. Training scope and continuation permission are independently
  checked; support for eight configurations is not campaign approval.

GPU process admission rejects every existing compute PID on the selected GPU,
including another job owned by the operator. Ownership uncertainty cannot grant
admission. No other user's process is signalled or changed. No fallback GPU or
CPU is selected. Laboratory sharing/reservation rules remain binding; if they
change or require a reservation, resolve that requirement before execution.

The detached guard retains an advisory GPU lock across `exec` into the original
trainer. It serializes launches through this wrapper installation; external
jobs remain subject to the frozen trainer's repeated NVML contention checks.
The trainer alone creates the official run directory and runs training.

External operational receipts, dependency snapshots and unique stdout/stderr
logs live under this wrapper worktree's ignored
`.local/lodo_launch/receipts/launch-*` directories. Receipts contain both freeze
identities, actual execution approval, policy path/hash, configuration and
manifest hashes, metadata/map evidence, interpreter hash, runtime/resource
checks, command and selected resume generation. Only controlled environment
variables are recorded; unrelated shell secrets are excluded.

`launch-receipt.json` records dispatch intent. `execution-admission.json` records
fresh detached admission before `exec`; neither alone proves a successful
optimizer update. Inspect stdout and the original trainer's artifacts to verify
startup/progress. Failures preserve their audit directory and diagnostic record.
The wrapper prints session, log and run paths plus tail/attach/GPU commands.

## Recovery

`--resume` requires an existing original namespace, a committed recovery index
and no terminal payload or terminal sidecar. It selects the latest integrity-
validated retained generation, allowing only the frozen index fallback rules.
It reconstructs on CPU without loading pretrained weights or iterating images,
then uses `restore_lodo_checkpoint(..., cpu_inspection=True, validate_only=True)`
to validate model, both optimizers, Center, scheduler, RNG and exposure.

The exact selected immutable generation path is passed through the original
trainer's sole `train.save.resume` override. The detached recheck refuses a
changed selection or hash. The original trainer repeats validation and restores
all state. Incomplete epochs roll back and replay; there is no mid-epoch recovery
claim, progress reset, silent fresh restart, checkpoint repair or orphan
promotion. Completed terminal checkpoints require inspection instead of resume.

## Audit-record contract and release procedure

The release is local only: do not push, merge, copy the wrapper into the frozen
checkout or revise the original scientific freeze. After passing CPU validation,
commit only the wrapper, tests and documentation. Record the full new SHA and
verify a clean wrapper worktree and unchanged scientific worktree.

A future explicit execution approval must identify configuration(s) and whether
same-run recovery is covered. The standing GPU allocation is reused; fresh
technical admission still runs before each execution. Resolve any newly required
laboratory reservation before execution. Approval of the launcher release alone
must never be used as scientific execution approval.

After actual scoped execution approval, record it in the ignored, operator-owned
`.local/lodo_launch/approval.json` (mode 0600), with exactly these keys:

| Key | Required value |
| --- | --- |
| `schema_version` | integer `2` |
| `record_type` | `local_operator_execution_declaration` |
| `implementation_commit` | original full scientific SHA above |
| `wrapper_commit` | actual separately reviewed launcher release SHA |
| `gpu_uuid` | `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` |
| `approved_configs` | nonempty unique list of exact authorized candidate paths |
| `experiment_bindings` | dictionary with exactly the approved config paths as keys |
| `allow_resume` | boolean reflecting actual continuation approval |
| `approved_by` | actual UID-derived local operator username |
| `approval_reference` | actual user execution approval text or genuine reference |

Each `experiment_bindings[config]` must equal `experiment_binding(config)` from
the released wrapper, containing exactly:

- `config_sha256`: frozen configuration file SHA256.
- `manifest`: frozen candidate-relative manifest path.
- `manifest_sha256`: SHA256 of the complete frozen manifest file.
- `immutable_sha256`: immutable candidate digest in the frozen index.
- `architecture`, `fold`, ordered `sources`, excluded `target`.
- `run_dir`: original manifest's exact canonical output namespace.

Every declared binding is validated, including bindings for other experiments
within the declared scope. Missing, extra, duplicate or unknown scope entries,
legacy schema, wrong GPU, changed hashes or changed experiment identity fail
closed. Existing science/map/data/pretrained/recovery checks remain mandatory.

The declaration SHA256 and exact contents are retained in operational launch
receipts. Candidate JSONs keep their historical `NOT_APPROVED` fields unchanged;
the declaration supplements those records. It is an audit guard, not a signature,
a fabricated laboratory reservation, or proof against the Linux account owner.

Once actual scoped approval is recorded, activate the existing Reid environment,
enter this external wrapper worktree and use the one-command start/resume
interfaces above. Environment setup, admission, receipts, tmux and logs are
automatic. No repeated GPU allocation confirmation is requested.

The execution declaration is intentionally absent at release. Official start
and resume therefore remain closed until actual scientific execution is approved
and recorded. No training is launched during release validation.

## CPU validation evidence (2026-10-12)

Focused wrapper suite from this review worktree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.:scripts:tests \
/home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider \
  tests/test_lodo_launch.py -q
```

Result: **57 passed, 0 failed, 0 skipped in 8.05 seconds**. Coverage includes all
eight candidate command preparations, release/scope rejection, namespace and
symlink collisions, disk headroom, validated-generation resume, frozen full
CPU-first recovery validation (including invalid Center/RNG/counters), exact
forwarded environment, shell quoting, worker output framing, receipt secrecy,
failed dispatch preservation and mocked detached admission/exec. Real tmux and
trainer execution are mocked; no CUDA context or official outputs are created.

Existing CPU regression suite from the original frozen canonical worktree:

```bash
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.:scripts:tests \
/home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider \
  tests/test_lodo_official_preparation.py tests/test_lodo_checkpoint.py \
  tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_policy.py \
  tests/test_lodo_device.py tests/test_lodo_config.py -q
```

Result: **419 passed, 0 failed, 0 skipped in 62.55 seconds**. Running that same
suite from the relocated review checkout first produced 413 passes and six
failures: the frozen manifests intentionally bind absolute original-worktree
output paths, so verifier tests failed earlier at output-directory mismatch.
No frozen tests/manifests were edited; the complete unchanged suite passed from
its correct canonical location. This also substantiates the choice to execute
the original verifier and trainer there.

A real ResNet50 LODO-1 `--check` returned `CHECK_ONLY_NOT_AUTHORIZED` with
`PASS_CANDIDATE_ONLY`, 54,135 images, 2,510 identities, the original mapping
SHA256, no CUDA initialization, TITAN Xp with 12,179 MiB free/0% utilization/no
compute processes, matching runtime and strict deterministic controls. No
receipts or output directories were written. An early worker framing issue
from ordinary frozen builder diagnostics was fixed by routing diagnostics to
stderr; a focused regression test now protects its JSON protocol.

The ordinary launch interface was also exercised with the approval record
absent: it refused before CPU worker dispatch, tmux, receipts or trainer
execution. No execution approval was manufactured.

Final preservation comparisons matched both original worktrees' HEAD/status,
489 tracked/relevant-untracked source-file hashes and 4,629 existing experiment
artifact size/mtime/inode records against the prior read-only snapshot. The
original LODO checkout remains clean; original Stage A changes remain intact.
This is not a content rehash of every historical checkpoint or image. Only the
three review files are added, and there is no `.local/lodo_launch` directory or
official output directory created by this implementation/validation task.

## Approved local-operator release validation (2026-10-12)

The user approved Option 2 and the local operational release. The declaration
is explicitly an audit guard that the same Linux account can create or modify;
it is not cryptographically authenticated. This release approval does not
approve any scientific execution, and no execution declaration is created.

The focused command above now passes **96 tests, 0 failed, 0 skipped in 7.95
seconds**. All original 72 hardened-wrapper cases are retained; 24 additional
cases validate schema-2 GPU, configuration, manifest, scientific identity,
ordered source, target and output-namespace bindings. The unchanged six-module
frozen regression command above passes **419 tests, 0 failed, 0 skipped in
60.61 seconds** from the canonical frozen worktree.

Release scope is exactly `scripts/lodo_launch.py`, `tests/test_lodo_launch.py`
and this document on `review/lodo-launch-wrapper`. The scientific commit,
configurations, manifests, original worktrees and historical artifacts remain
preserved. The final release commit SHA and subsequent read-only start check
are recorded separately under `.local/lodo_launcher_release/`, so recording
release evidence does not change the committed source tree.
