# Stage B official LODO guarded launch runbook — Phase 10 candidate

## Phase 10F current user sign-off — 2026-10-11

The [final sign-off](lodo_phase10_final_signoff.md) records the explicit later user instruction. It accepts the exact hashed CUHK03 detected `processed_partition`/null split with official-new correspondence **UNVERIFIED**, approves Stage B-only strict deterministic CUDA/FP32/zero workers and completed-epoch/every-epoch/two-validated-generation recovery plus distinct terminal, and accepts incomplete historical Stage A comparison eligibility. Scientific recipes, splits and budgets are unchanged. Earlier dated implementation/approval statements below remain historical. Advisor/institutional approval is not supplied or inferred. One local feature-branch freeze is conditionally authorized after all gates; official training, resume, target evaluation, GPU work, allocation, merge and push remain unauthorized. Actual commit identity is recorded in a separate ignored operational receipt, never self-referentially in these manifests.


**DO NOT EXECUTE LAUNCH, INTERRUPTION, RESUME OR EVALUATION COMMANDS BEFORE THE RELEVANT EXPLICIT APPROVAL.** Phase 10 prepares these commands; none has been run. Candidate manifests pass CPU/read-only checks but confer neither a Git freeze nor resource allocation. One fold/process at a time. Original Stage A/PCB and all diagnostic namespaces are protected.

## Approval and immutable provenance gates

Review `lodo_phase10_readiness.md`, `lodo_official_experiment_protocol.md`, all eight YAMLs/manifests and `lodo_phase10_evidence/implementation-review.patch`. Accept or resolve CUHK03 detected processed-partition provenance and the restricted recovery scope. Approve the exact commit contents, zero workers, strict CUDA policy, every-epoch/two-generation policy, storage plan and dedicated full-UUID allocation separately from launch.

Proposed freeze sequence, **not executed**:

1. Review `lodo_phase10_evidence/repository-delta.json` and `pre-edit-state.json`. The base HEAD is not the implementation revision. Keep main's dirty files and all `exp/`/results/datasets/checkpoints out of staging. Review all earlier opt-in code and tests, not only the Phase 10 additions. No resets/stashes/clean/branch switches.
2. After explicit Git approval, stage only the approved entries in `proposed-freeze-files.txt`, using explicit paths. The existing global `*.json` ignore rule also hides candidate manifests/maps/evidence. Those precise small preparation JSON paths are explicitly included in the proposal: after approval only, add them individually with `git add -f -- <approved-exact-path>`; never force-add a directory, historical outputs or data. Local `.log` files stay preserved and ignored; their hashes are recorded in the evidence manifest. Inspect `git diff --cached --name-status` and `git diff --cached --check`; reject any path under `exp/`, external data, pretrained cache, Stage A result registries or generated diagnostic tensor files. CPU report/manifests/maps and source patches are review evidence, not official results. Review the entire staged patch, including untracked production modules.
3. Commit once on `feature/lodo-v1` with a reviewed message such as `Add isolated LODO protocol and deterministic recovery with official candidates`. No automatic commit/merge/push is authorized. Record the actual full new SHA; require a clean intended worktree. Any excluded/unexplained dirty files block this gate. Do not substitute the old base SHA.
4. Candidate manifests retain their truthful prefreeze base/dirty-snapshot identity. Record the newly approved code commit and each candidate immutable digest in a **separate signed-off operational launch receipt**, outside the commit being described. The receipt must also contain approval/allocation identifiers, interpreter/dependency hashes, exact command, full UUID and fresh audit. Never insert a self-referential commit SHA or relabel a candidate as a frozen commit. `--frozen-commit` below checks that the actual clean HEAD contains the exact candidate implementation snapshot; it does not authorize launching.
5. If source/config/data/weights/runtime changes, withdraw commands and produce a new explicitly reviewed proposal. Do not refresh hashes silently. Do not reuse historical smoke checkpoints for official continuation.

## Safe read-only candidate verification after the authorized freeze

From the isolated canonical checkout:

```bash
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:scripts CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES='' /home/addirakoze/environments/Reid/bin/python -B \
  scripts/lodo_official_preparation.py \
  --manifest docs/lodo_phase10_evidence/manifests/resnet50_lodo-1.json \
  --frozen-commit "${FROZEN_LODO_COMMIT:?Read the actual full SHA from the separate Phase10F receipt}"
```

After the commit, always supply its actual full SHA via `--frozen-commit`; the original prefreeze base-HEAD check deliberately fails after HEAD changes. The separate freeze receipt is `.local/lodo_phase10f/operational-freeze-receipt.json`. This CPU verification does not require or confer launch permission. Repeat with each of the eight manifest filenames in `candidate-index.json`. This parses actual production YAML/schema, verifies scientific/file/source/weight/runtime identities, constructs original source loaders without iteration, checks live inventories/maps/sidecars and future evaluation descriptors. Guards throw before image decoding, target-loader construction or CUDA initialization. It does not instantiate the training engine or reserve output directories. A stale proposal, output collision, same-C wrong mapping, incompatible runtime or unknown provenance fails. The verifier deliberately has no launch/evaluation subcommand.

## DO NOT EXECUTE BEFORE APPROVAL — final admission and one official launch

The following concrete example is **ResNet50 LODO-1 only**, with proposed TITAN UUID. It remains unexecuted. Explicit lab allocation and official-launch approval must exist **before** setting operator assertions. Prior short diagnostic authorization does not carry over.

```bash
set -euo pipefail
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo
export REID_PYTHON=/home/addirakoze/environments/Reid/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:scripts CUBLAS_WORKSPACE_CONFIG=:4096:8
# Fill these only from the accepted freeze/allocation/launch review:
: "${FROZEN_LODO_COMMIT:?Set the approved full implementation commit SHA}"
: "${LODO_RESOURCE_ALLOCATION_ID:?Dedicated lab allocation is required}"
: "${LODO_LAUNCH_APPROVAL_ID:?Explicit official-launch approval is required}"
export FROZEN_LODO_COMMIT LODO_RESOURCE_ALLOCATION_ID LODO_LAUNCH_APPROVAL_ID
export CUDA_VISIBLE_DEVICES=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
export LODO_AUTHORIZED_GPU_UUID=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
export CONFIG=configs/lodo/v1/resnet50_lodo-1_seed42.yaml
export MANIFEST=docs/lodo_phase10_evidence/manifests/resnet50_lodo-1.json
"$REID_PYTHON" -B scripts/lodo_official_preparation.py \
  --manifest "$MANIFEST" --frozen-commit "$FROZEN_LODO_COMMIT"
"$REID_PYTHON" -B - <<'PY'
import os, shutil
from reid.utils.lodo_device import audit_gpu
import torch
assert not torch.cuda.is_initialized()
audit = audit_gpu(os.environ['LODO_AUTHORIZED_GPU_UUID'])
assert audit['name'] == 'NVIDIA TITAN Xp' and audit['free_mib'] >= 10000
assert shutil.disk_usage('.').free >= 64 * 1024**3
assert not torch.cuda.is_initialized()
print(audit)
PY
# STOP on any failure; execute later lines only after reviewing success.
# These audit directories are created only in the APPROVED launch procedure.
mkdir -p exp/lodo_launch_audit
AUDIT_DIR=$(mktemp -d "$PWD/exp/lodo_launch_audit/resnet50-LODO-1.XXXXXXXX")
export AUDIT_DIR
"$REID_PYTHON" -B - <<'PY'
import os, json, subprocess, datetime
from pathlib import Path
from reid.utils.lodo_checkpoint import file_digest
from reid.utils.lodo_device import audit_gpu
p=Path(os.environ['AUDIT_DIR'])
manifest=Path(os.environ['MANIFEST']);m=json.loads(manifest.read_text())
r=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       frozen_commit=os.environ['FROZEN_LODO_COMMIT'],
       allocation_id=os.environ['LODO_RESOURCE_ALLOCATION_ID'],
       launch_approval_id=os.environ['LODO_LAUNCH_APPROVAL_ID'],
       manifest=str(manifest),manifest_sha256=file_digest(manifest),
       immutable_candidate_sha256=m['immutable_sha256'],
       config=os.environ['CONFIG'],config_sha256=file_digest(os.environ['CONFIG']),
       interpreter=os.environ['REID_PYTHON'],interpreter_sha256=file_digest(os.environ['REID_PYTHON']),
       cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
       workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
       command=[os.environ['REID_PYTHON'],'-u','scripts/train.py','--config',os.environ['CONFIG']],
       admission=audit_gpu(os.environ['LODO_AUTHORIZED_GPU_UUID']))
with (p/'pip-freeze.txt').open('x') as f:
    subprocess.run([os.environ['REID_PYTHON'],'-m','pip','freeze','--all'],stdout=f,check=True)
r['dependency_freeze_sha256']=file_digest(p/'pip-freeze.txt')
with (p/'launch-receipt.json').open('x') as f:json.dump(r,f,indent=2)
PY
printf -v LODO_TMUX_COMMAND 'exec %q -u scripts/train.py --config %q > %q 2>&1' \
  "$REID_PYTHON" "$CONFIG" "$AUDIT_DIR/stdout.log"
tmux new-session -d -s lodo-resnet50-lodo1 -c "$PWD" \
  -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  -e "LODO_AUTHORIZED_GPU_UUID=$LODO_AUTHORIZED_GPU_UUID" \
  -e "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG" \
  -e "PYTHONPATH=$PYTHONPATH" -e "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
  -e "MKL_NUM_THREADS=$MKL_NUM_THREADS" -e "PYTHONDONTWRITEBYTECODE=1" \
  "$LODO_TMUX_COMMAND"
tmux list-sessions
# Confirm process PID, /proc/PID/cwd, full UUID and launch log before claiming success.
tail -f "$AUDIT_DIR/stdout.log"
```

The installed tmux3.3a supports the shown `new-session -e` options; explicit environment forwarding avoids stale tmux-server CUDA/workspace settings. The shell assertions are operator declarations, not reservation proof. Commands are deliberately sequential: **do not paste past an error**. Production independently validates full UUID→logical `cuda:0`, strict setup before CUDA initialization, own-process-only admission and target isolation. There is no automatic GPU fallback. The proposed 10,000-MiB fresh headroom and 64-GiB disk gate are additional reviewable operator conditions, not changes to production recipes or Phase 9 numerical thresholds. Original Stage A output paths are never used.

After each accepted terminal completion and separate authorization, select the next pair of config/manifest from the eight-run index and a new tmux/audit name. Candidate order: ResNet50 LODO-1→2→3→4, then PCB LODO-1→2→3→4; any alternative requires review. Do not launch concurrently or change the UUID without separately qualified/allocated resources and a revised proposal.

## Monitoring, alerts and stop conditions

Use read-only NVML and process inspection:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
ps -u "$USER" -o pid,ppid,etime,args
watch -n 30 df -h /mnt/hd-data/addirakoze/Projects
```

Monitor only the selected own PID; verify its cwd/config/log. Watch stdout iteration/loss/LR and elapsed/speed. The current LODO path does **not** call Stage A's log-file/TensorBoard setup, even if its flags are true; candidates set them false and capture unbuffered stdout externally. It also does not expose live source/Center counters in every console line. Exact durable primary/auxiliary, LR-used/next, source/identity exposure and epoch results are read from newly committed CPU checkpoints, not inferred from utilization or assumed source shares.

After `checkpoint_metrics.json` changes, inspect its newest entry and `checkpoints/recovery_index.json`. Verify the index-selected checkpoint once on CPU; avoid repeatedly hashing/loading large files every second. For a reviewed same-run `RUN_DIR`, this read-only inspection neither restores training tensors nor touches target images:

```bash
# Set RUN_DIR to the canonical approved run directory from its manifest.
RUN_DIR=/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo/exp/lodo/v1/resnet50/LODO-1/seed42/official-candidate-20261011
export RUN_DIR
CUDA_VISIBLE_DEVICES='' "$REID_PYTHON" -B - <<'PY'
import os,json
from pathlib import Path
from reid.utils.lodo_checkpoint import read_checkpoint
p=Path(os.environ['RUN_DIR']);index=json.loads((p/'checkpoints/recovery_index.json').read_text())
payload,proof=read_checkpoint(p/'checkpoints'/index['generations'][-1]['name'])
state=payload['progress']
print(json.dumps(dict(checkpoint=proof,primary=state['completed_primary_updates'],
                     auxiliary=state['auxiliary_steps'],lr_trace=state['lr_trace'],
                     exposure={phase:dict(counts) for phase,counts in state['exposure']['presentations'].items()},
                     next_epoch=payload['position']['next_epoch']),indent=2))
PY
```

Treat nonfinite loss/gradients, partial main/Center failure, wrong target access, hash/config/map/runtime mismatch, CUDA/OOM/kernel errors, unexpected competitors or invalid generation/index as immediate stop-and-review events. Production aborts its own run on observed contention/nonfinite states; never kill other users' processes. Alert on less than 1 GiB device free or sustained allocator growth (inspection alone cannot prevent OOM), less than 16 GiB filesystem free, abnormal checkpoint-write stalls or orphan/corrupt-file growth. Operator thresholds require acceptance, not automatic recipe changes. No AMP/batch/resolution/P/K/loss reduction is allowed. Preserve failure receipts and all bytes; no manual pruning or repair.

## DO NOT EXECUTE BEFORE APPROVAL — interruption and same-run resume

Prefer allowing a completed epoch checkpoint to commit, then interrupt **only the verified own process**. A foreground `Ctrl-C`/SIGINT triggers Python rollback logging when delivered outside a fatal kernel condition. SIGTERM/power loss may leave no exception event; absence of an event does not imply no discarded work. Do not interrupt an active pair/index publication deliberately. There is no mid-epoch save command.

Check for a verified terminal first. Otherwise use the canonical same-run **logical** `checkpoints/ckpt_last.pth` path: under retained recovery it need not physically exist. It resolves only through `recovery_index.json`, never directory scanning. Do not copy/move a run or a checkpoint to another output namespace.

```bash
# DO NOT EXECUTE BEFORE EXPLICIT RESUME/ALLOCATION APPROVAL.
RESUME_PATH="$RUN_DIR/checkpoints/ckpt_last.pth"
CUDA_VISIBLE_DEVICES='' "$REID_PYTHON" -B scripts/lodo_official_preparation.py \
  --manifest "$MANIFEST" --frozen-commit "$FROZEN_LODO_COMMIT" --resume "$RESUME_PATH"
# Repeat approved full-UUID GPU/disk admission and create a NEW operational receipt/log.
# Preserve the prior stdout log and record resume path plus selected generation/SHA.
# The following is the production command to place in a new guarded tmux session:
"$REID_PYTHON" -u scripts/train.py --config "$CONFIG" \
  --opts "train.save.resume=$RESUME_PATH"
```

The candidate verifier reads integrity/config/same-run namespace through the existing output validator and verifies live provenance; it does not itself compare all saved tensors against fresh model/optimizer objects. Production performs that full CPU-only compatibility validation before device admission. The final line is a reviewable command, **not an instruction to execute now**. Resume exports the same approved full UUID/workspace/runtime as fresh launch; the CUDA-hidden verifier is only a CPU preflight. Production validates the selected checkpoint and current source/model/worker/runtime contracts on CPU before CUDA discovery, restores model/Center/optimizers/scheduler, and restores RNG last. Lost work after the durable boundary is replayed; accepted counters/exposure restore rather than double-count. Corrupt/missing newest indexed pair may fall back to the preceding validated generation and record its reason/SHA. Explicit corrupt generation requests fail without fallback. Missing/unusable index, both bad pairs or semantic identity mismatches fail; preserve them and review. Never promote orphan payloads or fabricate sidecars/indexes.

If a terminal pair exists, validate it explicitly and use `train.save.resume=$RUN_DIR/checkpoints/ckpt_terminal.pth` for a CPU-only completed-run no-op. A partial/incomplete terminal cannot be treated as complete or overwritten. Predetermined terminal may occur inside the final epoch; no resumable mid-epoch cursor is supported.

## DO NOT EXECUTE BEFORE POST-TRAINING APPROVAL — completion and held-out evaluation

Validate only after the official budget completes. A five-update smoke terminal must fail this official gate; none is being substituted:

```bash
REQUEST=docs/lodo_phase10_evidence/evaluation_requests/resnet50_lodo-1.yaml
export REQUEST
CUDA_VISIBLE_DEVICES='' "$REID_PYTHON" -B - <<'PY'
import os,json
from reid.utils.config import load_config
from reid.utils.lodo_evaluation import validate_terminal,target_metadata
request=load_config(os.environ['REQUEST'])
payload,manifest,path=validate_terminal(request)
target_metadata(payload['cfg'],payload['identity'],request['data_root'])
print(json.dumps(manifest,indent=2))
PY
# Only after separate real held-out evaluation approval:
CUDA_VISIBLE_DEVICES='' "$REID_PYTHON" -u scripts/evaluate.py --config "$REQUEST"
```

Expected terminal counts: ResNet50 85,200 primary +85,200 Center, 5,452,800 successful presentations; PCB94,200 primary +0 Center, 6,028,800 presentations. Require budget_completed, consistent counters/exposure, correct map/classes/config/data, finite tensors, schema2 and verified payload/manifest SHA. A checkpoint name or stdout exit alone is insufficient. Separate evaluation is CPU FP32 and opens only the excluded target's original query/gallery; no training feedback, reranking/flip or target-based selection.

Requests currently have an empty `stage_a_records` list: standalone Stage B evaluation can publish an explicitly incomplete comparison. Before complete comparison reporting, separately qualify exactly three historical source→target JSON cells with original target hashes/filter evidence and matching architecture/representation. Do not enrich historical files using today's hashes, average fewer than three, or substitute within-domain metrics. Comparison is between complete strategies, not an isolated causal test of dataset diversity.

## Mandatory stop

This runbook grants no permission. Git freeze, resource allocation, official launch, resume and real held-out evaluation all remain separately gated. No Phase 11 begins in Phase 10. Preserve all old artifacts and original Stage A checkout.
