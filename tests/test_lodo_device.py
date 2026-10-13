"""Phase 9 CPU-only admission tests. All CUDA calls are doubles, not GPU evidence."""
import copy
import os
import subprocess
from types import SimpleNamespace

import pytest
import torch

from test_lodo_checkpoint import (synthetic, cfg_for, parts, epoch, save, rewrite,
                                  target_guards, assert_equal)
from reid.engine.train_loop import train_one_epoch
from reid.engine.update_control import UpdateProgress, UpdateFailure
from reid.utils import lodo_checkpoint as ck, lodo_device as dev
from reid.utils import lodo_protocol as protocol
from scripts import train

UUID = 'GPU-11111111-2222-3333-4444-555555555555'
OTHER = 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'


@pytest.fixture
def authorized(monkeypatch):
    cfg = cfg_for()
    cfg['system'].update(device='cuda', gpu_id=0)
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID', UUID)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID)
    return cfg


@pytest.fixture
def inventory(monkeypatch):
    data = dict(gpu=[['1', UUID, 'Test GPU', '570', '8192', '8100', '0', 'Default', '[N/A]']], apps=[])
    monkeypatch.setattr(dev, '_query', lambda fields, kind='gpu': data['apps' if kind == 'compute-apps' else 'gpu'])
    return data


@pytest.fixture
def cuda_double(monkeypatch):
    # Never call real CUDA, even on a machine where it is available.
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda i: SimpleNamespace(uuid=UUID[4:], major=6, minor=1))
    monkeypatch.setattr(torch.cuda, 'get_arch_list', lambda: ['sm_50', 'sm_60', 'sm_70'])
    monkeypatch.setattr(torch.cuda, 'set_device', lambda i: calls.append(i))
    return calls


def test_explicit_uuid_maps_physical_one_to_logical_zero(authorized, inventory, cuda_double):
    device, guard, evidence = dev.select_lodo_device(authorized)
    assert device == torch.device('cuda:0') and cuda_double == [0]
    assert evidence['physical_index'] == 1 and evidence['logical_index'] == 0
    assert evidence['uuid'] == UUID and evidence['capability'] == dev.CUDA_FRESH_CAPABILITY
    # Our own context is permitted on later checks, another job is never permitted.
    inventory['apps'] = [[UUID, str(os.getpid()), 'this-test', '100']]
    guard()
    inventory['apps'].append([UUID, '987654', 'other-job', '100'])
    with pytest.raises(ValueError, match='occupied'): guard()
    assert cuda_double == [0]


@pytest.mark.parametrize('variable,value', [
    ('LODO_AUTHORIZED_GPU_UUID', ''), ('LODO_AUTHORIZED_GPU_UUID', 'GPU-111'),
    ('CUDA_VISIBLE_DEVICES', ''), ('CUDA_VISIBLE_DEVICES', '1'),
    ('CUDA_VISIBLE_DEVICES', UUID+','+OTHER), ('CUDA_VISIBLE_DEVICES', OTHER),
])
def test_authorization_and_visibility_fail_before_cuda_or_images(authorized, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(dev, 'audit_gpu', lambda *a, **k: pytest.fail('Audit before authorization'))
    monkeypatch.setattr(train, 'build_train_loader', lambda *a: pytest.fail('Image loader accessed'))
    target_guards(monkeypatch)
    with pytest.raises(ValueError): train.run_lodo_training(authorized)


@pytest.mark.parametrize('index', [1, -1, True, '0', None])
def test_physical_or_invalid_index_is_never_fallback(authorized, index):
    authorized['system']['gpu_id'] = index
    with pytest.raises(ValueError, match='logical'): dev.authorization_uuid(authorized)


def test_auto_and_amp_and_resume_fail_closed(authorized, monkeypatch):
    for key, value, match in [('device', 'auto', 'auto'), ('amp', True, 'FP32')]:
        cfg = copy.deepcopy(authorized); cfg['system'][key] = value
        with pytest.raises(ValueError, match=match): dev.authorization_uuid(cfg)
    authorized['train']['save']['resume'] = '/not/opened/ckpt_last.pth'
    monkeypatch.setattr(dev, 'audit_gpu', lambda *a, **k: pytest.fail('CUDA resume queried GPU'))
    with pytest.raises(ValueError, match='CUDA resume'): train.run_lodo_training(authorized)
    with pytest.raises(ValueError, match='CUDA resume'): ck.restore_lodo_checkpoint('/not/read', cfg=authorized,
        loader=None, model=None, criterion=None, optimizer=None, auxiliary=None, scheduler=None)


@pytest.mark.parametrize('row', [
    [], ['bad'], ['1', OTHER, 'GPU', '570', '8192', '8100', '0', 'Default', '[N/A]'],
    ['1', UUID, 'GPU', '570', '8192', '8100', '0', 'Prohibited', '[N/A]'],
    ['1', UUID, 'GPU', '570', '8192', '8100', '0', 'Default', 'Enabled'],
    ['1', UUID, 'GPU', '570', '8192', '8100', '10', 'Default', '[N/A]'],
    ['1', UUID, 'GPU', '570', '8192', '0', '0', 'Default', '[N/A]'],
    ['1', UUID, 'GPU', '570', '8192', '[N/A]', '0', 'Default', '[N/A]'],
])
def test_invalid_inventory_precedes_cuda_setup(authorized, inventory, monkeypatch, row):
    inventory['gpu'] = [row] if row else []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('Unsafe GPU queried CUDA'))
    with pytest.raises(ValueError): dev.select_lodo_device(authorized)


def test_duplicate_uuid_and_occupied_self_at_initial_admission_rejected(inventory):
    inventory['gpu'] *= 2
    with pytest.raises(ValueError, match='ambiguous'): dev.audit_gpu(UUID)
    inventory['gpu'] = inventory['gpu'][:1]
    inventory['apps'] = [[UUID, str(os.getpid()), 'already-in-use', '100']]
    with pytest.raises(ValueError, match='occupied'): dev.audit_gpu(UUID)
    inventory['apps'] = [[OTHER, '987654', 'PCB', '7350']]
    assert dev.audit_gpu(UUID)['processes'] == []
    inventory['apps'] = [[UUID, 'unavailable', 'job', '100']]
    with pytest.raises(ValueError): dev.audit_gpu(UUID)


@pytest.mark.parametrize('failure', ['unavailable', 'count', 'uuid', 'arch'])
def test_no_fallback_on_runtime_mismatch(authorized, inventory, cuda_double, monkeypatch, failure):
    if failure == 'unavailable': monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    elif failure == 'count': monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    elif failure == 'uuid': monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda i: SimpleNamespace(uuid=OTHER, major=6, minor=1))
    else: monkeypatch.setattr(torch.cuda, 'get_arch_list', lambda: ['sm_70'])
    with pytest.raises(ValueError): dev.select_lodo_device(authorized)
    assert cuda_double == []


def test_guard_detects_revoked_authorization(authorized, inventory, cuda_double, monkeypatch):
    _, guard, _ = dev.select_lodo_device(authorized)
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID', OTHER)
    with pytest.raises(ValueError): guard()


@pytest.mark.parametrize('error', [FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 10),
                                  subprocess.CalledProcessError(1, 'nvidia-smi')])
def test_audit_unavailable_fails_closed(monkeypatch, error):
    def fail(*a, **k): raise error
    monkeypatch.setattr(dev.subprocess, 'run', fail)
    with pytest.raises(ValueError, match='ownership'): dev._query('uuid')


def test_query_is_bounded_read_only_csv(monkeypatch):
    def run(args, **kwargs):
        assert args == ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits']
        assert kwargs == dict(check=True, capture_output=True, text=True, timeout=10)
        return SimpleNamespace(stdout=UUID+'\n')
    monkeypatch.setattr(dev.subprocess, 'run', run)
    assert dev._query('uuid') == [[UUID]]


def test_cpu_never_audits_gpu_or_changes_stage_a_device_behavior(monkeypatch):
    cfg = cfg_for()
    monkeypatch.setattr(dev, 'audit_gpu', lambda *a, **k: pytest.fail('CPU queried GPU'))
    device, guard, evidence = dev.select_lodo_device(cfg)
    assert device.type == 'cpu' and guard is None and evidence is None
    from reid.utils.device import select_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert select_device('cuda', 1)[0].type == 'cpu'  # historical Stage A fallback
    assert select_device('auto', 1)[0].type == 'cpu'


@pytest.mark.parametrize('architecture', ['resnet50', 'pcb'])
def test_cpu_checkpoint_mapping_and_recovery_identity_unchanged(synthetic, monkeypatch, architecture):
    cfg = cfg_for(architecture, budget=1)
    objects = parts(cfg); progress = UpdateProgress(1, auxiliary=architecture=='resnet50')
    epoch(objects, progress)
    loads = []; original = torch.load
    def load(*a, **kw):
        loads.append(kw['map_location']); return original(*a, **kw)
    monkeypatch.setattr(torch, 'load', load)
    path = synthetic / 'mapping/checkpoints/ckpt_terminal.pth'
    save(objects, progress, path, kind='terminal', cursor=1)
    payload, _ = ck.read_checkpoint(path)
    assert loads and set(loads) == {'cpu'}
    assert payload['identity']['recovery']['capability'] == ck.RECOVERY_CAPABILITY
    assert all(t.device.type == 'cpu' for t in payload['model'].values())
    assert dev.validate_placement(**{key: objects[key] for key in ('cfg','model','criterion','optimizer','auxiliary')}).type == 'cpu'


@pytest.mark.parametrize('architecture', ['resnet50', 'pcb'])
def test_cuda_capability_is_distinct_and_cannot_be_cpu_resumed(synthetic, authorized, monkeypatch, architecture):
    # Metadata-only double: model/tensors remain CPU. This proves no GPU execution.
    cfg = cfg_for(architecture, budget=1); cfg['system'].update(device='cuda', gpu_id=0)
    objects = parts(cfg); progress = UpdateProgress(1, auxiliary=architecture=='resnet50')
    epoch(objects, progress)
    monkeypatch.setattr(dev, 'validate_placement', lambda *a: torch.device('cuda:0'))
    path = synthetic/'cuda-tag/checkpoints/ckpt_terminal.pth'
    save(objects, progress, path, kind='terminal', cursor=1)
    payload, _ = ck.read_checkpoint(path)
    assert payload['identity']['recovery']['capability'] == dev.CUDA_FRESH_CAPABILITY
    objects['cfg']['system']['device'] = 'cpu'
    before = copy.deepcopy(objects['model'].state_dict())
    with pytest.raises(ValueError, match='CUDA resume'): ck.restore_lodo_checkpoint(path, **objects)
    assert_equal(before, objects['model'].state_dict())


@pytest.mark.parametrize('which', ['model', 'center', 'optimizer'])
def test_placement_precision_and_state_mismatch_rejected(synthetic, which):
    objects = parts(cfg_for()); args = {key: objects[key] for key in ('cfg','model','criterion','optimizer','auxiliary')}
    if which == 'model': objects['model'].double()
    elif which == 'center': objects['criterion'].center_loss.double()
    else:
        parameter = next(objects['model'].parameters())
        objects['optimizer'].state[parameter]['exp_avg'] = torch.empty_like(parameter, device='meta')
    with pytest.raises(ValueError): dev.validate_placement(**args)


def test_contested_guard_stops_before_transfer_forward_or_exposure(synthetic):
    objects = parts(cfg_for()); progress = UpdateProgress(5, auxiliary=True)
    def contested(): raise ValueError('GPU became occupied')
    with pytest.raises(UpdateFailure, match='occupied'):
        train_one_epoch(objects['model'], objects['loader'], objects['criterion'], objects['optimizer'],
                        objects['auxiliary'], torch.device('cpu'), False, 1000,
                        scheduler=objects['scheduler'], update_progress=progress, before_update=contested)
    assert objects['model'].inputs == []
    assert progress.attempted_batches == progress.forward_batches == progress.completed_primary_updates == 0
    assert progress.failed and progress.consistent and not objects['loader']._active


def test_gpu_runtime_overrides_do_not_change_science_or_allow_short_budget(authorized):
    cfg = copy.deepcopy(authorized); cfg['train']['max_updates'] = 85200
    cpu = copy.deepcopy(cfg); cpu['system']['device'] = 'cpu'
    assert protocol.configuration_identity(cpu) == protocol.configuration_identity(cfg)
    cfg['train']['max_updates'] = 2
    with pytest.raises(ValueError, match='max_updates'): protocol.scientific_configuration(cfg)
