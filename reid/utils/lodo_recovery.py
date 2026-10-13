"""Opt-in boundary continuation and immutable, two-generation publication.

CUDA policy is intentionally stricter than fresh CUDA: same UUID/environment,
zero workers, FP32, completed epochs only. No arbitrary cursor or bitwise claim.
"""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import uuid

import torch

CUDA_BOUNDARY_CAPABILITY = 'epoch_boundary_rollback_cuda_v1'
_GENERATION = re.compile(r'ckpt_epoch_([0-9]{6,})_u([0-9]{9,})_g[0-9a-f]{12}\.pth')


def recovery_policy(cfg):
    value = cfg.get('train', {}).get('recovery')
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {'version', 'policy', 'every_epochs', 'keep_generations'}:
        raise ValueError('Recovery requires explicit version/policy/cadence/retention keys')
    if type(value['version']) is not int or value['version'] != 1 or value['policy'] != 'completed_epoch_boundary':
        raise ValueError('Unsupported recovery version/policy')
    if type(value['every_epochs']) is not int or value['every_epochs'] < 1:
        raise ValueError('Recovery cadence must be a positive completed-epoch interval')
    if type(value['keep_generations']) is not int or value['keep_generations'] != 2:
        raise ValueError('Recovery requires two validated generations')
    if cfg['system']['device'] == 'cuda' and cfg['data'].get('num_workers') != 0:
        raise ValueError('CUDA boundary recovery supports zero workers only')
    return dict(value)


def cuda_boundary(cfg):
    return cfg['system']['device'] == 'cuda' and recovery_policy(cfg) is not None


def execution_environment(cfg):
    """Read-only CPU-side contract; no CUDA runtime discovery or allocation."""
    from reid.utils.lodo_device import authorization_uuid, _query
    uuid = authorization_uuid(cfg)
    matches = [r for r in _query('uuid,driver_version') if len(r) == 2 and r[0] == uuid]
    if len(matches) != 1:
        raise ValueError('Cannot establish same-GPU driver environment')
    from reid.utils.lodo_cuda_policy import policy_contract, validate_runtime, runtime_controls
    contract = policy_contract(cfg)
    if contract is not None:
        validate_runtime(cfg)
    return dict(**(dict(execution_policy=contract, cudnn_version=torch.backends.cudnn.version(),
                        runtime_controls=runtime_controls()) if contract is not None else {}),
                uuid=uuid, driver=matches[0][1], host=platform.node(),
                python=platform.python_version(), torch=str(torch.__version__),
                torchvision=version('torchvision'), numpy=version('numpy'),
                cuda_build=torch.version.cuda, precision='fp32',
                deterministic=cfg['repro']['deterministic'], benchmark=cfg['repro']['benchmark'],
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def validate_boundary(payload):
    """Cross-check the position against completed, consistent epoch records."""
    progress = payload['progress']; position = payload['position']
    count = progress['completed_primary_updates']; epoch = position['next_epoch']
    if type(epoch) is not int or epoch < 1:
        raise ValueError('Invalid next epoch')
    if payload['kind'] == 'terminal':
        if count != progress['max_updates']:
            raise ValueError('Invalid completed terminal count')
        return
    if position.get('boundary') != 'completed_epoch' or position['consumed_cursor'] != 0 or position['epoch'] != epoch-1:
        raise ValueError('Recovery requires a marked completed epoch boundary')
    results = progress['epoch_results']
    if epoch == 1:
        if count or results:
            raise ValueError('Initialization boundary must have zero updates')
    elif (len(results) != epoch-1 or [r['epoch'] for r in results] != list(range(1, epoch))
          or any(r['stop_reason'] != 'epoch_exhausted' or r['partial_epoch'] for r in results)
          or sum(r['successful_updates'] for r in results) != count
          or position['last_epoch_consumed_batches'] != results[-1]['attempted_batches']):
        raise ValueError('Checkpoint does not describe fully completed epochs')


def generation_name(next_epoch, completed):
    return f'ckpt_epoch_{next_epoch-1:06d}_u{completed:09d}_g{uuid.uuid4().hex[:12]}.pth'


def _read_index(directory):
    path = directory/'recovery_index.json'
    if path.is_symlink():
        raise ValueError('Recovery index alias prohibited')
    value = json.loads(path.read_text())
    if set(value) != {'version','generations'} or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Invalid recovery index')
    rows = value['generations']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2 or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Invalid recovery generation window')
    for row in rows:
        if set(row) != {'name','checkpoint_sha256','identity_digest'} or not _GENERATION.fullmatch(row['name']):
            raise ValueError('Invalid recovery generation reference')
    return rows


def select_generation(cfg, run_dir):
    """Explicit latest-pointer fallback; never repair, scan or promote orphans."""
    from reid.utils.lodo_checkpoint import read_checkpoint, digest
    directory = Path(run_dir)/'checkpoints'; raw = Path(cfg['train']['save']['resume'])
    if raw.parent != directory or raw.resolve() != raw:
        raise ValueError('Resume requires a canonical same-run checkpoint')
    rows = _read_index(directory)
    candidates = list(reversed(rows)) if raw.name == 'ckpt_last.pth' else [r for r in rows if r['name'] == raw.name]
    if not candidates:
        raise ValueError('Requested generation is not retained/committed')
    failures = []
    for row in candidates:
        path = directory/row['name']
        try:
            payload, manifest = read_checkpoint(path)
            if manifest['checkpoint_sha256'] != row['checkpoint_sha256'] or digest(payload['identity']) != row['identity_digest']:
                raise ValueError('Recovery index proof mismatch')
            validate_boundary(payload)
            match = _GENERATION.fullmatch(path.name)
            if int(match[1]) != payload['position']['epoch'] or int(match[2]) != payload['progress']['completed_primary_updates']:
                raise ValueError('Generation name/position mismatch')
        except ValueError as error:
            failures.append(dict(checkpoint=str(path),reason=str(error)))
            if raw.name != 'ckpt_last.pth': raise
            continue
        return path, failures
    raise ValueError('No retained validated checkpoint generation: '+str(failures))


def publish_generation(path, manifest):
    """Commit verified successor before pruning owned older generations."""
    from reid.utils.lodo_checkpoint import atomic_write, canonical_json, read_checkpoint
    path = Path(path); directory = path.parent
    payload, _ = read_checkpoint(path); validate_boundary(payload)
    index = directory/'recovery_index.json'
    rows = _read_index(directory) if index.exists() else []
    if any(r['name'] == path.name for r in rows):
        raise ValueError('Generation is already committed')
    if any(r['identity_digest'] != manifest['identity_digest'] for r in rows):
        raise ValueError('Refusing unrelated generation index')
    valid = []
    for row in rows:
        try:
            _, proof = read_checkpoint(directory/row['name'])
            if proof['checkpoint_sha256'] != row['checkpoint_sha256']:
                raise ValueError('Prior generation integrity mismatch')
        except ValueError:
            continue  # retain corrupt bytes for audit, exclude from the validated window
        valid.append(row)
    rows = valid
    rows.append(dict(name=path.name,checkpoint_sha256=manifest['checkpoint_sha256'],identity_digest=manifest['identity_digest']))
    retained = rows[-2:]
    atomic_write(index,lambda stream:stream.write(canonical_json(dict(version=1,generations=retained))))
    assert _read_index(directory) == retained
    for old in rows[:-2]:
        candidate = directory/old['name']
        # Never delete an unverified/unrelated artifact, even during cleanup.
        try:
            old_payload, old_manifest = read_checkpoint(candidate)
        except ValueError:
            continue  # preserve corrupt forensic artifacts outside the retained index
        if old_manifest['checkpoint_sha256'] != old['checkpoint_sha256'] or old_payload['identity'] != payload['identity']:
            raise ValueError('Refusing to prune unverified generation')
        candidate.unlink();candidate.with_suffix('.json').unlink()


def recovery_event_environment(cfg):
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                environment=execution_environment(cfg) if cuda_boundary(cfg) else dict(device='cpu'))
