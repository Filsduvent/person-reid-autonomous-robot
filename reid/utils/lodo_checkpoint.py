"""Stage B v1 checkpoints; trusted local torch payloads, SHA256 sidecar integrity.

Recovery is exact at completed epoch boundaries for tested CPU 0/2-worker,
nonpersistent/in-order loaders. A mid-epoch failure rolls back the ENTIRE epoch;
worker prefetch state is never presented as resumable. Terminal snapshots may
have a partial cursor but may only be reopened for zero further updates.
No bitwise GPU-continuation guarantee. Legacy Stage A readers are unchanged.
"""
import copy
from collections import Counter, deque
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from reid.data.index_observation import ObservedTrainLoader
from reid.data.multi_source import canonical_json
from reid.data.samplers import PKBatchSampler
from reid.engine.update_control import UpdateProgress, UpdateEpochResult, validate_update_mode
from reid.utils.lodo_protocol import configuration_identity, scientific_configuration

SCHEMA_VERSION = 1  # historical original/CPU payloads remain v1
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
from reid.utils.lodo_cuda_policy import checkpoint_schema, policy_contract, validate_runtime
RECOVERY_CAPABILITY = 'epoch_boundary_rollback_cpu_v1'


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def loader_contract(loader):
    if not isinstance(loader, ObservedTrainLoader):
        raise ValueError('Recovery requires the observed original sampler')
    if loader.num_workers not in (0, 2) or loader.persistent_workers or not getattr(loader, 'in_order', True):
        raise ValueError('Recovery supports only tested 0/2 nonpersistent ordered workers')
    sampler = loader.batch_sampler
    if loader.sampler_mode == 'pk' and not isinstance(sampler, PKBatchSampler):
        raise ValueError('Recovery requires the original PK sampler')
    if loader.sampler_mode == 'random' and not isinstance(loader.sampler, torch.utils.data.RandomSampler):
        raise ValueError('Recovery requires the original shuffled random sampler')
    worker = loader.worker_init_fn
    return dict(capability=RECOVERY_CAPABILITY, num_workers=loader.num_workers,
                persistent_workers=False, in_order=True, sampler=loader.sampler_mode,
                P=getattr(sampler, 'P', None), K=getattr(sampler, 'K', None),
                batch_size=loader.effective_batch_size,
                prefetch_factor=loader.prefetch_factor,
                worker_init=None if worker is None else f'{worker.__module__}.{worker.__qualname__}',
                multiprocessing_context=None if loader.multiprocessing_context is None else
                    loader.multiprocessing_context.get_start_method())


def optimizer_contract(optimizer, model):
    names = {id(p): name for name, p in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        values = {k: copy.deepcopy(v) for k, v in group.items() if k not in ('params', 'lr', 'initial_lr')}
        values['parameters'] = [names[id(p)] for p in group['params']]
        groups.append(values)
    return dict(type=f'{type(optimizer).__module__}.{type(optimizer).__qualname__}', groups=groups)


def center_module(criterion, auxiliary):
    module = getattr(criterion, 'center_loss', None)
    if module is None:
        if auxiliary is not None:
            raise ValueError('Unexpected Center optimizer')
        return None
    if auxiliary is None or getattr(criterion, 'center', module) is not module:
        raise ValueError('Canonical Center module/optimizer missing or ambiguous')
    expected = [p for p in module.parameters() if p.requires_grad]
    actual = [p for group in auxiliary.param_groups for p in group['params']]
    if len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
        raise ValueError('Center optimizer does not reference canonical Center parameters')
    return module


def make_identity(cfg, loader, model, criterion, optimizer, auxiliary, scheduler, *, cpu_inspection=False):
    """Validate effective scientific state before saving OR loading model tensors."""
    scientific = scientific_configuration(cfg)
    validate_runtime(cfg)
    ds = loader.dataset
    if tuple(ds.sources) != tuple(row['name'] for row in scientific['sources']) or ds.target != scientific['target']['name']:
        raise ValueError('Dataset sources/target disagree with frozen fold')
    mapping = ds.mapping_metadata()
    if digest(mapping) != ds.mapping_digest:
        raise ValueError('Current mapping digest does not match recoverable mapping')
    sidecar = json.loads(ds.serialize_sidecar())
    inventories = ds.preflight['inventories']
    provenance = {key: dict(metadata_hashes=value['metadata_hashes'],
                            descriptor_metadata_hashes=value['descriptor_metadata_hashes'])
                  for key, value in inventories.items()}
    module = center_module(criterion, auxiliary)
    architecture = cfg['protocol']['architecture']
    expected_center = architecture == 'resnet50'
    if (module is not None) != expected_center:
        raise ValueError('Center presence disagrees with architecture')
    declaration = getattr(model, 'checkpoint_metadata', None)
    if declaration is None or declaration['num_classes'] != ds.num_classes:
        raise ValueError('Model must declare validated source classifier dimensions')
    dim = scientific['architecture_contract']['embedding_dim']
    if declaration['embedding_dim'] != dim or declaration['model_name'] != cfg['model']['name'] or declaration['variant'] != cfg['model'].get('variant'):
        raise ValueError('Model architecture/feature dimension disagrees with protocol')
    heads = [model.classifier] if architecture == 'resnet50' else list(model.classifiers.fc_list)
    if len(heads) != scientific['architecture_contract']['classifier_heads'] or any(head.out_features != ds.num_classes for head in heads):
        raise ValueError('Actual classifier dimensions disagree with source class count')
    from reid.utils.lodo_device import validate_placement
    placement_cfg = copy.deepcopy(cfg) if cpu_inspection else cfg
    if cpu_inspection:
        placement_cfg['system']['device'] = 'cpu'
    device = validate_placement(placement_cfg, model, criterion, optimizer, auxiliary)
    recovery = loader_contract(loader)
    from reid.utils.lodo_recovery import recovery_policy, cuda_boundary, execution_environment, CUDA_BOUNDARY_CAPABILITY
    policy = recovery_policy(cfg)
    if policy is not None:
        if loader.num_workers != cfg['data'].get('num_workers'):
            raise ValueError('Loader workers differ from the requested recovery configuration')
        recovery['checkpoint_policy'] = policy
        if loader.prefetch_factor not in (None, 2) or loader.timeout != 0:
            raise ValueError('Boundary recovery requires default prefetch and zero loader timeout')
    if cfg['system']['device'] == 'cuda':
        from reid.utils.lodo_device import CUDA_FRESH_CAPABILITY, authorization_uuid
        authorization_uuid(cfg)
        recovery['capability'] = CUDA_BOUNDARY_CAPABILITY if cuda_boundary(cfg) else CUDA_FRESH_CAPABILITY
        if cuda_boundary(cfg):
            if loader.num_workers != 0:
                raise ValueError('CUDA boundary recovery supports zero workers only')
            recovery['environment'] = execution_environment(cfg)
    center = None
    if module is not None:
        if (module.num_classes != ds.num_classes or module.feat_dim != 2048
                or tuple(module.centers.shape) != (ds.num_classes, 2048)
                or criterion.metric_feat_key != 'feat_raw' or criterion.w_center != 0.0005):
            raise ValueError('Center C/D/branch/weight disagrees with protocol')
        center = dict(num_classes=ds.num_classes, feature_dim=2048, feature_branch='feat_raw',
                      weight=0.0005, reduction='half_mean_squared_distance',
                      module_type=f'{type(module).__module__}.{type(module).__qualname__}',
                      optimizer=optimizer_contract(auxiliary, module))
        if not isinstance(auxiliary, torch.optim.SGD) or any(g['lr'] != .5 or g['momentum'] != 0 or g['weight_decay'] != 0 for g in auxiliary.param_groups):
            raise ValueError('Center optimizer convention mismatch')
    schedule = scheduler.state_dict(); schedule.pop('completed_updates')
    return dict(schema_version=checkpoint_schema(cfg), protocol=cfg['protocol'],
                sources=list(ds.sources), target=ds.target, num_classes=ds.num_classes,
                feature_dim=dim, reconstruction=declaration, map_digest=ds.mapping_digest,
                mapping=mapping, sidecar_digest=digest(sidecar), data_digest=digest(provenance),
                provenance=provenance, config_digest=configuration_identity(cfg),
                scientific_config=scientific, precision='fp32', center=center,
                optimizer=optimizer_contract(optimizer, model), scheduler=schedule,
                recovery=recovery)


def rng_state(loader):
    sampler = loader.batch_sampler
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                pk_python=sampler.rng.getstate() if isinstance(sampler, PKBatchSampler) else None,
                pk_numpy=copy.deepcopy(sampler.np_rng.bit_generator.state) if isinstance(sampler, PKBatchSampler) else None,
                loader_generator=None if loader.generator is None else loader.generator.get_state(),
                sampler_generator=None if getattr(loader.sampler, 'generator', None) is None else loader.sampler.generator.get_state())


def restore_rng(state, loader):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_initialized():
            raise ValueError('CUDA RNG restoration requires an initialized CUDA context')
        torch.cuda.set_rng_state_all(state['cuda'])
    if isinstance(loader.batch_sampler, PKBatchSampler):
        loader.batch_sampler.rng.setstate(state['pk_python'])
        loader.batch_sampler.np_rng.bit_generator.state = state['pk_numpy']
    for generator, value in ((loader.generator, state['loader_generator']),
                             (getattr(loader.sampler, 'generator', None), state['sampler_generator'])):
        if (generator is None) != (value is None):
            raise ValueError('Generator ownership differs from checkpoint')
        if generator is not None:
            generator.set_state(value)


def progress_state(progress):
    state = {f.name: copy.deepcopy(getattr(progress, f.name)) for f in fields(progress)
             if f.name not in ('exposure', 'epoch_results')}
    state.update(index_history=list(progress.index_history), lr_trace=list(progress.lr_trace),
                 epoch_results=[asdict(row) for row in progress.epoch_results])
    exposure = progress.exposure
    state['exposure'] = {key: copy.deepcopy(getattr(exposure, key)) for key in
                         ('presentations', 'identity_presentations', 'identity_slots', 'batch_counts', 'distinct', 'lookup_seconds')}
    return state


def restore_progress(state, loader):
    p = UpdateProgress(state['max_updates'], auxiliary=state['auxiliary'], history_limit=state['history_limit'])
    for f in fields(p):
        if f.name not in ('exposure', 'epoch_results'):
            setattr(p, f.name, copy.deepcopy(state[f.name]))
    p.bind(loader)
    for key, value in state['exposure'].items():
        setattr(p.exposure, key, copy.deepcopy(value))
    p.index_history = deque(state['index_history'], maxlen=p.history_limit)
    p.lr_trace = deque(state['lr_trace'], maxlen=p.history_limit)
    p.epoch_results = [UpdateEpochResult(**row) for row in state['epoch_results']]
    validate_progress(p)
    return p


def validate_progress(p):
    if p.failed or not p.consistent:
        raise ValueError('Failed/partial state cannot be checkpointed or resumed')
    p.check_totals()
    count = p.completed_primary_updates
    if type(count) is not int or not 0 <= count <= p.max_updates:
        raise ValueError('Invalid successful-update count')
    if p.attempted_batches != count or p.forward_batches != count or p.backward_batches != count:
        raise ValueError('Checkpoint requires a complete successful-update boundary')
    if p.unattributed_attempted_images or p.unattributed_attempted_batches:
        raise ValueError('Unattributed failed attempts cannot be resumed')
    e = p.exposure
    if p.attempted_images != count * 64:
        raise ValueError('Checkpoint presentations disagree with frozen batch size')
    for phase in e.phases:
        if sum(e.presentations[phase].values()) != count * 64:
            raise ValueError('Checkpoint exposure disagrees with successful full batches')
        if any(type(v) is not int or v < 0 for v in e.presentations[phase].values()):
            raise ValueError('Invalid exposure count')
        expected = Counter()
        for label, amount in e.identity_presentations[phase].items():
            if label not in e.identities or type(amount) is not int or amount < 0:
                raise ValueError('Invalid identity exposure')
            expected[e.identities[label]['source_key']] += amount
        if expected != e.presentations[phase]:
            raise ValueError('Source and identity exposure disagree')
        if e.distinct[phase] != {g for g, n in e.identity_presentations[phase].items() if n}:
            raise ValueError('Distinct identities disagree with exposure')
        if phase != 'successful' and (e.presentations[phase] != e.presentations['successful'] or e.identity_presentations[phase] != e.identity_presentations['successful']):
            raise ValueError('Failed-attempt exposure cannot be exact-resumed')
        if e.identity_slots is not None and e.identity_slots[phase] != Counter({s: n // e.K for s, n in expected.items()}):
            raise ValueError('PK slots disagree with presentations')


def atomic_write(path, writer):
    """Same-filesystem temporary file, fsync, replace, then directory fsync."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w+b') as stream:
            writer(stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Checkpoint symlink aliases are prohibited')
    try:
        manifest = json.loads(path.with_suffix('.json').read_text())
        if type(manifest['schema_version']) is not int or manifest['schema_version'] not in SUPPORTED_SCHEMA_VERSIONS or manifest['checkpoint_sha256'] != file_digest(path):
            raise ValueError('Checkpoint integrity/hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if type(payload.get('lodo_schema_version')) is not int or payload.get('lodo_schema_version') not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError('Legacy/missing LODO Center state cannot be exact-resumed')
        if manifest['identity_digest'] != digest(payload['identity']):
            raise ValueError('Checkpoint identity/manifest mismatch')
        identity, progress = payload['identity'], payload['progress']
        schema = checkpoint_schema(payload['cfg'])
        if payload['lodo_schema_version'] != schema or identity['schema_version'] != schema:
            raise ValueError('Checkpoint schema/execution policy mismatch')
        if schema == 2 and identity['scientific_config'].get('execution_policy') != policy_contract(payload['cfg']):
            raise ValueError('Checkpoint strict execution policy identity mismatch')
        expected = dict(schema_version=schema, kind=payload['kind'],
                        completion_reason='budget_completed' if payload['kind']=='terminal' else 'recovery_boundary',
                        planned_primary_updates=progress['max_updates'],
                        planned_auxiliary_updates=progress['max_updates'] if progress['auxiliary'] else 0,
                        actual_primary_updates=progress['completed_primary_updates'],
                        actual_auxiliary_updates=progress['auxiliary_steps'],
                        fold=identity['protocol']['fold'], map_digest=identity['map_digest'],
                        config_digest=identity['config_digest'], data_digest=identity['data_digest'],
                        identity_digest=digest(identity), checkpoint_sha256=manifest['checkpoint_sha256'])
        if manifest != expected or payload['kind'] not in ('recovery','terminal'):
            raise ValueError('Checkpoint completion manifest mismatch')
        return payload, manifest
    except Exception as error:
        raise ValueError(f'Invalid/incomplete LODO checkpoint: {path}: {error}') from error


def validate_resume_path(cfg, run_dir):
    """Read-only collision exception, only for verified same-run checkpoints."""
    run_dir = Path(run_dir)
    raw = Path(cfg['train']['save'].get('resume', ''))
    from reid.utils.lodo_recovery import recovery_policy, _GENERATION
    valid_name = raw.name in ('ckpt_last.pth', 'ckpt_terminal.pth') or (recovery_policy(cfg) is not None and _GENERATION.fullmatch(raw.name))
    if not valid_name or raw.parent != run_dir / 'checkpoints' or raw.resolve() != raw:
        raise ValueError('Resume must reference a canonical checkpoint in this same LODO run')
    from reid.utils.lodo_recovery import recovery_policy, select_generation
    if recovery_policy(cfg) is not None and raw.name != 'ckpt_terminal.pth':
        raw, _ = select_generation(cfg, run_dir)
    payload, _ = read_checkpoint(raw)
    if payload['identity']['config_digest'] != configuration_identity(cfg):
        raise ValueError('Resume scientific config identity mismatch')
    if payload['run_dir'] != str(run_dir):
        raise ValueError('Checkpoint belongs to a different run directory')
    return raw


def save_lodo_checkpoint(path, *, cfg, loader, model, criterion, optimizer, auxiliary,
                         scheduler, progress, next_epoch, cursor=0, kind='recovery'):
    path = Path(path)
    from reid.utils.lodo_recovery import recovery_policy, generation_name, publish_generation, validate_boundary
    retained = recovery_policy(cfg) is not None and kind == 'recovery'
    if retained:
        path = path.with_name(generation_name(next_epoch, progress.completed_primary_updates))
    identity = make_identity(cfg, loader, model, criterion, optimizer, auxiliary, scheduler)
    validate_update_mode(progress, loader, model, criterion, optimizer, auxiliary, scheduler, False)
    validate_progress(progress)
    if progress.max_updates != cfg['train']['max_updates']:
        raise ValueError('Progress budget differs from frozen configuration')
    for module_to_check in (model, criterion):
        if any(not torch.isfinite(p).all() for p in module_to_check.parameters()):
            raise ValueError('Nonfinite checkpoint parameters')
    if loader._active:
        raise ValueError('Cannot save while an iterator/prefetch is active')
    if kind not in ('recovery', 'terminal') or (not retained and path.name != ('ckpt_terminal.pth' if kind == 'terminal' else 'ckpt_last.pth')):
        raise ValueError('Invalid checkpoint kind/name')
    if type(next_epoch) is not int or next_epoch < 1 or type(cursor) is not int or cursor < 0:
        raise ValueError('Invalid recovery position')
    if kind == 'recovery' and cursor != 0:
        raise ValueError('Exact mid-epoch recovery is unsupported; roll back to an epoch boundary')
    if kind == 'terminal' and not progress.budget_completed:
        raise ValueError('Terminal checkpoint requires exact budget completion')
    module = center_module(criterion, auxiliary)
    payload = dict(lodo_schema_version=checkpoint_schema(cfg), kind=kind, identity=identity,
                   run_dir=str(path.parent.parent.resolve()), cfg=copy.deepcopy(cfg),
                   reconstruction=identity['reconstruction'], model=model.state_dict(),
                   optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   progress=progress_state(progress), rng=rng_state(loader),
                   position=dict(next_epoch=next_epoch, consumed_cursor=cursor,
                                 epoch=next_epoch if cursor else next_epoch-1,
                                 last_epoch_consumed_batches=progress.epoch_results[-1].attempted_batches if progress.epoch_results else 0,
                                 remaining_order=None),
                   center=module.state_dict() if module is not None else None,
                   center_optimizer=auxiliary.state_dict() if auxiliary is not None else None)
    if recovery_policy(cfg) is not None:
        payload['position']['boundary'] = 'terminal' if kind == 'terminal' else 'completed_epoch'
        validate_boundary(payload)
    if retained and (path.exists() or path.with_suffix('.json').exists()):
        raise ValueError('Immutable checkpoint generation already exists')
    if path.exists() or path.with_suffix('.json').exists():
        old, _ = read_checkpoint(path)
        if old['identity'] != identity or old['run_dir'] != payload['run_dir'] or kind == 'terminal':
            raise ValueError('Refusing to overwrite an unrelated or terminal checkpoint')
    # Stage B run orchestration owns directory creation and exclusive reservation.
    if not path.parent.is_dir() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Checkpoint directory unavailable or aliased')
    for opt in (optimizer, auxiliary):
        if opt is not None:
            validate_optimizer_tensors(opt, opt.state_dict())
    def write(stream):
        torch.save(payload, stream)
        stream.flush(); stream.seek(0)
        staged = torch.load(stream, map_location='cpu', weights_only=False)
        if staged['identity'] != identity or staged['progress']['completed_primary_updates'] != progress.completed_primary_updates:
            raise ValueError('Temporary checkpoint verification failed')
    atomic_write(path, write)
    # Verify torch serialization before publishing completion metadata.
    verified = torch.load(path, map_location='cpu', weights_only=False)
    if verified['identity'] != identity or verified['progress']['completed_primary_updates'] != progress.completed_primary_updates:
        raise ValueError('Serialized checkpoint verification failed')
    manifest = dict(schema_version=checkpoint_schema(cfg), kind=kind,
                    completion_reason='budget_completed' if kind == 'terminal' else 'recovery_boundary',
                    planned_primary_updates=progress.max_updates,
                    planned_auxiliary_updates=progress.max_updates if progress.auxiliary else 0,
                    actual_primary_updates=progress.completed_primary_updates,
                    actual_auxiliary_updates=progress.auxiliary_steps,
                    fold=cfg['protocol']['fold'], map_digest=identity['map_digest'],
                    config_digest=identity['config_digest'], data_digest=identity['data_digest'],
                    identity_digest=digest(identity), checkpoint_sha256=file_digest(path))
    atomic_write(path.with_suffix('.json'), lambda stream: stream.write(canonical_json(manifest)))
    read_checkpoint(path)
    if retained:
        publish_generation(path, manifest)
    return manifest


def restore_lodo_checkpoint(path, *, cfg, loader, model, criterion, optimizer, auxiliary, scheduler,
                            cpu_inspection=False, validate_only=False):
    from reid.utils.lodo_recovery import cuda_boundary, recovery_policy, validate_boundary, CUDA_BOUNDARY_CAPABILITY
    if cfg['system']['device'] != 'cpu' and not cuda_boundary(cfg):
        raise ValueError('CUDA resume is unsupported; fresh restart requires a new namespace')
    payload, manifest = read_checkpoint(path)
    capability = CUDA_BOUNDARY_CAPABILITY if cuda_boundary(cfg) else RECOVERY_CAPABILITY
    if payload['identity']['recovery']['capability'] != capability:
        raise ValueError('Checkpoint has no validated CPU recovery capability; CUDA resume is unsupported')
    expected = make_identity(cfg, loader, model, criterion, optimizer, auxiliary, scheduler, cpu_inspection=cpu_inspection)
    if configuration_identity(payload['cfg']) != expected['config_digest']:
        raise ValueError('Embedded frozen config differs from checkpoint identity')
    if payload['identity'] != expected:
        raise ValueError('Incompatible LODO fold/map/C/D/config/Center/optimizer/scheduler/recovery identity')
    if recovery_policy(cfg) is not None:
        validate_boundary(payload)
    if payload['kind'] not in ('terminal', 'recovery'):
        raise ValueError('Unsupported recovery mode')
    if payload['kind'] == 'recovery' and payload['position']['consumed_cursor'] != 0:
        raise ValueError('Exact mid-epoch recovery is unsupported')
    p = restore_progress(payload['progress'], loader)
    if p.max_updates != cfg['train']['max_updates']:
        raise ValueError('Checkpoint budget disagrees with frozen config')
    if payload['kind'] == 'terminal' and (not p.budget_completed or manifest['completion_reason'] != 'budget_completed'):
        raise ValueError('Invalid terminal completion')
    if payload['scheduler']['completed_updates'] != p.completed_primary_updates:
        raise ValueError('Scheduler/counter mismatch')
    if any(payload['scheduler'].get(k) != v for k, v in scheduler.state_dict().items() if k != 'completed_updates'):
        raise ValueError('Incompatible scheduler state')
    module = center_module(criterion, auxiliary)
    if (payload['center'] is None) != (module is None) or (payload['center_optimizer'] is None) != (auxiliary is None):
        raise ValueError('Missing/unexpected canonical Center state')
    # Validate tensor keys/shapes and optimizer group topology BEFORE mutation.
    for target, state in ((model, payload['model']), (module, payload['center'])):
        if target is None:
            continue
        current = target.state_dict()
        if current.keys() != state.keys() or any(current[k].shape != state[k].shape or current[k].dtype != state[k].dtype or not torch.isfinite(state[k]).all() for k in current):
            raise ValueError('Checkpoint tensor keys/shapes/dtypes/finite values mismatch')
    for opt, state in ((optimizer, payload['optimizer']), (auxiliary, payload['center_optimizer'])):
        if opt is not None:
            current = opt.state_dict()['param_groups']; saved = state['param_groups']
            if len(current) != len(saved) or any(len(a['params']) != len(b['params']) or
                {k:v for k,v in a.items() if k not in ('params','lr','initial_lr')} !=
                {k:v for k,v in b.items() if k not in ('params','lr','initial_lr')}
                for a,b in zip(current,saved)):
                raise ValueError('Checkpoint optimizer group mismatch')
    rates = scheduler.rates_after(p.completed_primary_updates)
    if tuple(g['lr'] for g in payload['optimizer']['param_groups']) != rates:
        raise ValueError('Saved LR disagrees with successful-update count')
    if auxiliary is not None and any(g['lr'] != .5 for g in payload['center_optimizer']['param_groups']):
        raise ValueError('Saved Center LR mismatch')
    validate_rng(payload['rng'], loader, cuda_expected=cuda_boundary(cfg), cpu_inspection=cpu_inspection)
    for opt, state in ((optimizer, payload['optimizer']), (auxiliary, payload['center_optimizer'])):
        if opt is not None:
            validate_optimizer_tensors(opt, state)
    if validate_only:
        return p, payload['position'], payload['kind']
    if cpu_inspection:
        raise ValueError('CPU inspection may validate only; it cannot restore CUDA state')
    model.load_state_dict(payload['model'], strict=True)
    if module is not None:
        module.load_state_dict(payload['center'], strict=True)  # copy into SAME Parameters
    optimizer.load_state_dict(payload['optimizer'])
    if auxiliary is not None:
        auxiliary.load_state_dict(payload['center_optimizer'])
    scheduler.load_state_dict(payload['scheduler'])  # after optimizer, no step
    center_module(criterion, auxiliary)
    from reid.utils.lodo_device import validate_placement
    validate_placement(cfg, model, criterion, optimizer, auxiliary)
    restore_rng(payload['rng'], loader)  # LAST, after reconstruction/loading
    validate_update_mode(p, loader, model, criterion, optimizer, auxiliary, scheduler, False)
    return p, payload['position'], payload['kind']


def validate_rng(state, loader, *, cuda_expected=False, cpu_inspection=False):
    expected = rng_state(loader)
    if state.keys() != expected.keys():
        raise ValueError('Unsupported RNG state fields')
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state(state['numpy'])
    torch.Generator().set_state(state['torch'])
    if cuda_expected:
        streams = state['cuda']
        if (not isinstance(streams, list) or len(streams) != 1 or
                any(not torch.is_tensor(t) or t.device.type != 'cpu' or t.dtype != torch.uint8 or t.ndim != 1 or not t.numel() for t in streams)):
            raise ValueError('Invalid single-device CUDA RNG state')
        if not cpu_inspection:
            if expected['cuda'] is None or len(expected['cuda']) != 1 or streams[0].shape != expected['cuda'][0].shape:
                raise ValueError('CUDA RNG capability differs')
            torch.Generator(device='cuda:0').set_state(streams[0])
    elif (state['cuda'] is None) != (expected['cuda'] is None):
        raise ValueError('CUDA RNG capability differs')
    for key in ('loader_generator', 'sampler_generator', 'pk_python', 'pk_numpy'):
        if (state[key] is None) != (expected[key] is None):
            raise ValueError('Sampler/generator ownership differs')
    if state['pk_python'] is not None:
        random.Random().setstate(state['pk_python'])
        generator = np.random.default_rng(0)
        generator.bit_generator.state = state['pk_numpy']
    for key in ('loader_generator', 'sampler_generator'):
        if state[key] is not None:
            torch.Generator().set_state(state[key])


def validate_optimizer_tensors(optimizer, state):
    saved_ids = [identifier for group in state['param_groups'] for identifier in group['params']]
    params = [p for group in optimizer.param_groups for p in group['params']]
    if len(set(saved_ids)) != len(saved_ids) or set(state['state']) - set(saved_ids):
        raise ValueError('Invalid optimizer parameter state IDs')
    lookup = dict(zip(saved_ids, params))
    allowed = {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'} if isinstance(optimizer, torch.optim.Adam) else {'momentum_buffer'}
    for identifier, values in state['state'].items():
        if set(values) - allowed:
            raise ValueError('Incompatible optimizer tensor state')
        for key, value in values.items():
            if torch.is_tensor(value) and (not torch.isfinite(value).all() or
                    (key != 'step' and value.shape != lookup[identifier].shape)):
                raise ValueError('Invalid optimizer tensor shape/finite values')
