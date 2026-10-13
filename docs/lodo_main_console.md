# Stage B from the main terminal

The main checkout provides the terminal interface. Scientific execution delegates
to the clean `person-reid-autonomous-robot-lodo-console-execution` checkout at
the same reviewed release commit as main. The exact full SHA is pinned in the
external release receipt, not embedded self-referentially in source. Training
engine code is unchanged. No second training engine is introduced.
The original scientific freeze `4a03fb0d37cec2e41af92b8ff6e681c91ff5c387`, eight
configurations/manifests and historical experiment directories stay unchanged.

The operational release requires a private external receipt identifying both
actual checkouts at the same reviewed commit. It refuses execution without that receipt and
the maintained runner's separate v2 experiment binding and scoped operator
declaration. It does not create approvals, allocate a GPU, or migrate historical
outputs automatically. Existing historical execution declarations do not grant
authority to launch a new maintained run.

## Commands after reviewed release and experiment preparation

From `~/nobackup/Projects/person-reid-autonomous-robot`, with the Reid environment
active, use the approved configuration:

```bash
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml --resume
```

These are alternatives: start a new approved run or resume that same run. The
generic interface accepts all eight frozen configurations; execution scope
must cover the specific experiment. The ResNet50 LODO-1 command is an interface
example, not approval to repeat the completed historical experiment.

The launcher displays live unbuffered stdout/stderr in your terminal by default.
The actual trainer and completion watcher run in detached tmux sessions. Ctrl+C
stops the viewer only; it does not interrupt the trainer. SSH disconnection also
leaves the detached processes running. Run the same command with `--monitor` to
follow the newest matching attempt, without launching or resuming anything:

```bash
python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml --monitor
```

Use `--no-follow` to dispatch without viewing. Each attempt prints its exact run,
trainer session, audit directory and log path. Logs belong to attempts; following
an earlier attempt's log cannot show a later resume. A missing watcher produces
an explicit error rather than silently leaving the viewer waiting forever.
Inspect the printed audit/supervisor log and trainer process before resuming.
The console never restarts a partial run from zero.

## Scientific and operational admission

Every start/resume invokes the maintained runner's original `prepare` and
`dispatch`, followed by its fresh detached `execute` admission. That runner
retains complete frozen source/blob/mode checks, original candidate YAML/manifest
identity, dataset inventories and mapping, pretrained hash, policy hash, runtime
and interpreter identity, exact TITAN Xp UUID, ownership/contention, GPU memory,
disk headroom, output namespace, and CPU-first recovery compatibility checks.
The strict CUDA environment is set before the scientific subprocess starts.
No GPU fallback, weakened tolerance, sampling change or training override is added.

The standing research allocation applies to
`GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, subject to laboratory sharing rules.
It requires no repeated allocation confirmation. Changed allocation or a genuine
reservation requirement needs clarification; occupied/unsafe devices refuse
execution. Allocation remains separate from scientific execution permission.

The terminal interface may coexist with exactly these six preserved user
documents: `plan.md`, `docs/preliminary_results_log.md` and the four
`docs/pcb_*_results.yaml` files. It rejects staged changes and every other dirty
source path, checks tracked code bytes/modes even behind Git index flags, and
rejects ignored/untracked executable/config additions. Ten pre-existing ignored
local Stage A YAML paths listed explicitly in `LOCAL_STAGE_A_CONFIGS` are excluded
from the main executable inventory: they are never accepted as Stage B candidates
or forwarded to the scientific runner. No other local config exception is made;
aliases still refuse. These document/config exceptions do not apply to the scientific execution checkout, which stays
fully clean. Stage A's original training/evaluation interfaces are unchanged.

`--check` delegates the maintained runner's read-only preparation, writes no run,
receipt or tmux session, and does not initialize CUDA. A reviewed console release
and maintained v2 binding are still required; no execution declaration is
consumed in check mode. Technical NVML/process/disk queries are part of admission.

## Results and separately approved evaluation

After the trainer tmux session ends, the detached watcher requires an actual
matching frozen training-admission receipt and fully validates the exact-budget
terminal checkpoint on CPU before registering completion. A stopped session or
checkpoint filename alone is insufficient. Interrupted, inconsistent, missing
or incompatible terminals produce an explicit failure status and no completion
record. The frozen engine retains its original recovery, optimizer, Center Loss,
scheduler, RNG and source-exposure persistence.

Training completion does not trigger target evaluation. After a separate
held-out evaluation declaration and canonical maintained-root request exist:

```bash
python scripts/lodo_evaluate.py --config .local/lodo_operational/v2/evaluation_requests/<approved-request>.yaml
```

Relative evaluation requests resolve under the maintained execution checkout.
The console executes its unchanged `scripts/evaluate.py` on CPU FP32, shows logs,
and registers metrics only after a matching actual evaluation-completion receipt,
result proof, terminal identity, representation, target metadata and comparison
hash all validate. It does not select a checkpoint using target metrics. Training
scope cannot authorize this command. `--check`, `--monitor` and `--no-follow` work
as above; evaluation has no training-resume flag.

Verified records live in main's ignored `.local/lodo_results/`, separately from
tracked scientific source and the Stage A registry:

```bash
python scripts/lodo_results.py
python scripts/lodo_results.py --verify
```

`README.md` and `index.json` are derived views. Private immutable
`records/<canonical-record-SHA256>.json` files separately describe completion
(updates/exposure, no target metrics) and evaluation (mAP, mINP, Rank-1/5/10,
terminal/result/comparison hashes and caveats). Publication uses a file lock,
exclusive record creation and atomic view replacement. Repeated registration
of unchanged artifacts is idempotent. A corrupt existing record refuses further
publication. Reading the view is not a live artifact revalidation; use `--verify`.

The user-approved historical ResNet50 LODO-1 import is limited to its existing
original namespace and frozen source. It preserves the original result and
terminal byte hashes, incomplete Stage A comparison, unverified CUHK03 official
split correspondence, and historical evaluation approval-trail limitation. It
does not rewrite an approval, authorize future execution or re-evaluate images.

Operational records are same-account audit guards. Their SHA256 values detect
accidental inconsistency; they are not cryptographic authorization against the
Linux account owner.

## One-time operational release review

Review only the console entry points, tests and this document; preserve existing
uncommitted user documents. After explicit commit approval, record the actual
reviewed frontend commit in a private (0600), ignored
`.local/lodo_console/release.json` with exactly:

```json
{
  "schema_version": 1,
  "record_type": "lodo_main_console_release",
  "frontend_root": "/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot",
  "frontend_commit": "<actual-full-reviewed-operational-commit>",
  "execution_root": "/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo-console-execution",
  "execution_commit": "<same-actual-full-reviewed-operational-commit>"
}
```

The receipt is external to the commit it identifies. Do not fabricate a future
commit or create the receipt while source is unreviewed. Prepare the maintained
runner's v2 binding/policy/experiment declaration using its existing documented
mechanism and a new `unified-` namespace. Historical approvals/outputs remain
immutable. Both checkouts must identify the same actual release commit; the old integrated
checkout at 17afc5eb6477cb6b93a0553408133d6dbc6185c6 remains preserved.
Neither step revises the historical scientific freeze. Validate `--check` after
preparation, and obtain experiment-specific execution authority before training.
Release preparation creates no training session or CUDA context. The user
executes the approved experiment command after successful technical preflight.
