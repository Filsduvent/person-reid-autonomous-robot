"""Frozen presets, qualification equivalence and read-only admission contracts."""
import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from scripts import mgn_preflight as gate
from scripts import mgn_gpu_smoke as smoke
from reid.utils.config import load_config,validate_reid_config
from reid.utils.config_schema import validate_config
from reid.optim.build import build_optimizer,build_scheduler
from reid.optim.lr_scheduler import step_completed_epoch

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source',smoke.SOURCES)
def test_frozen_presets_match_qualified_recipe_except_isolated_official_output(source):
    cfg=load_config(str(ROOT/f'configs/mgn/{source}.yaml'))
    expected=smoke.configuration(source);expected['experiment']['output_dir']=f'exp/mgn/{source}'
    assert cfg==expected
    validate_config(cfg);validate_reid_config(cfg)
    manifest=json.loads((ROOT/'docs/mgn_experiment_manifest.json').read_text())
    entry=manifest['sources'][source]
    assert entry['config_sha256']==smoke.file_hash(ROOT/f'configs/mgn/{source}.yaml')
    assert entry['resolved_configuration']==cfg
    assert entry['training']['num_classes'] in [751,702,767,1041]
    assert cfg['model'].get('num_classes') is None
    assert entry['planned_fresh_budget']['total_updates']==sum(entry['planned_fresh_budget']['batches_by_epoch'])
    assert len(entry['planned_fresh_budget']['batches_by_epoch'])==120
    assert cfg['eval']['flip_aggregation']=='mean' and cfg['eval']['rerank']['enabled'] is False
    assert cfg['train']['save']['require_selection_state_on_resume'] is True


@pytest.mark.parametrize('source',smoke.SOURCES)
def test_every_preset_has_uniform_sgd_and_exact_120_epoch_trace(source):
    cfg=load_config(str(ROOT/f'configs/mgn/{source}.yaml'))
    model=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3))
    opt=build_optimizer(cfg,model);sched=build_scheduler(cfg,opt,steps_per_epoch=12345)
    for group in opt.param_groups:
        assert group['lr']==.01 and group['weight_decay']==.0005 and group['momentum']==.9
        assert not group['nesterov'] and 'lr_mult' not in group
    for epoch in range(1,121):
        expected=.01 if epoch<=60 else .001 if epoch<=90 else .0001
        assert all(group['lr']==pytest.approx(expected) for group in opt.param_groups)
        step_completed_epoch(sched,epoch)
    assert sched.completed_epochs==120


def test_preflight_refuses_changed_code_config_or_weights_without_writes(tmp_path):
    for name in ['source.py','preset.yaml','weights.pth']:
        p=tmp_path/name;p.write_bytes(b'approved')
        before={str(f):f.read_bytes() for f in tmp_path.iterdir()}
        gate.verify_hashes({name:smoke.file_hash(p)},root=tmp_path)
        p.write_bytes(b'changed')
        changed=p.read_bytes()
        with pytest.raises(ValueError,match='hash mismatch'):
            gate.verify_hashes({name:__import__('hashlib').sha256(b'approved').hexdigest()},root=tmp_path)
        assert p.read_bytes()==changed


def test_preflight_fresh_collision_wrong_namespace_and_symlink_guards(tmp_path):
    cfg={'experiment':{'output_dir':'exp/mgn/market1501'}}
    target=gate.output_guard('market1501',cfg,root=tmp_path)
    assert not target.exists()
    target.mkdir(parents=True)
    with pytest.raises(ValueError,match='existing output'):
        gate.output_guard('market1501',cfg,root=tmp_path)
    with pytest.raises(ValueError,match='last epoch-boundary'):
        gate.output_guard('market1501',cfg,root=tmp_path,resume=True)
    with pytest.raises(ValueError,match='isolated'):
        gate.output_guard('duke',cfg,root=tmp_path)
    cfg2={'experiment':{'output_dir':'exp/mgn/duke'}}
    link=tmp_path/'exp/mgn/duke';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        gate.output_guard('duke',cfg2,root=tmp_path)


def test_wrong_gpu_authority_and_visibility_refuse_before_device_queries(monkeypatch):
    uuid='GPU-11111111-2222-3333-4444-555555555555'
    monkeypatch.delenv('MGN_AUTHORIZED_GPU_UUID',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA reached before authority'))
    with pytest.raises(ValueError,match='MGN_AUTHORIZED'):
        gate.gpu_guard(uuid,{},0)
    monkeypatch.setenv('MGN_AUTHORIZED_GPU_UUID',uuid)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(ValueError,match='Visibility'):
        gate.gpu_guard(uuid,{},0)


def test_readonly_preflight_has_no_training_launch_mode(monkeypatch):
    monkeypatch.setattr(sys,'argv',['mgn_preflight.py','--source','market1501','--launch'])
    monkeypatch.setattr(gate,'preflight',lambda *a,**k:pytest.fail('Unknown launch flag must not execute preflight or training'))
    with pytest.raises(SystemExit) as caught:gate.main()
    assert caught.value.code==2


def test_resume_normalization_allows_only_operational_path_and_resume_value():
    cfg={'experiment':{'output_dir':'exp/mgn/market1501'},'train':{'save':{'resume':''}},'optim':{'lr':.01}}
    changed=copy.deepcopy(cfg);changed['experiment']['output_dir']=str(gate.ROOT/'exp/mgn/market1501')
    changed['train']['save']['resume']='original/checkpoints/ckpt_last.pth'
    assert gate.normalized_training_cfg(changed)==gate.normalized_training_cfg(cfg)
    changed['optim']['lr']=.1
    assert gate.normalized_training_cfg(changed)!=gate.normalized_training_cfg(cfg)


def test_changed_source_data_is_refused_before_gpu_or_output_creation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    uuid='GPU-11111111-2222-3333-4444-555555555555'
    cfg=smoke.configuration('market1501');cfg['experiment']['output_dir']='exp/mgn/market1501'
    frozen=dict(status='FROZEN',gpu_uuid=uuid,tested_implementation_commit='test',runtime_source_sha256={},
        qualification_receipt_sha256={},environment={'test':True},weights_sha256='test',
        sources={'market1501':dict(config_path='unused.yaml',config_sha256='test',resolved_configuration=cfg,
            training={'metadata_sha256':{}},evaluation={'metadata_sha256':{}})})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(frozen));before=path.read_bytes()
    monkeypatch.setattr(gate,'MANIFEST',path)
    monkeypatch.setenv('MGN_AUTHORIZED_GPU_UUID',uuid);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',uuid)
    monkeypatch.setattr(gate.subprocess,'check_output',lambda *a,**k:'')
    monkeypatch.setattr(gate.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(gate,'verify_hashes',lambda *a,**k:None)
    monkeypatch.setattr(gate,'environment',lambda:{'test':True})
    monkeypatch.setattr(gate,'output_guard',lambda *a,**k:tmp_path/'never-created')
    monkeypatch.setattr('reid.utils.config.load_config',lambda *a,**k:copy.deepcopy(cfg))
    monkeypatch.setattr('reid.data.build.build_train_loader',lambda *a:(SimpleNamespace(dataset=object()),751))
    monkeypatch.setattr('reid.data.build.build_test_loader',lambda *a:object())
    monkeypatch.setattr(gate,'source_provenance',lambda *a:{'changed':'identity map'})
    monkeypatch.setattr(gate,'gpu_guard',lambda *a:pytest.fail('GPU reached after data mismatch'))
    with pytest.raises(ValueError,match='Training data/map'):
        gate.preflight('market1501')
    assert path.read_bytes()==before and not (tmp_path/'never-created').exists()


@pytest.mark.parametrize('damage',['epoch0','epoch120','recipe','classes'])
def test_resume_preflight_refuses_unqualified_epoch_recipe_or_source(tmp_path,monkeypatch,damage):
    cfg={'experiment':{'output_dir':'exp/mgn/market1501'},'train':{'save':{'resume':''}},'optim':{'lr':.01}}
    payload=dict(epoch=2,cfg=copy.deepcopy(cfg),reconstruction=dict(model_name='mgn',num_classes=751,embedding_dim=2048))
    if damage=='epoch0':payload['epoch']=0
    elif damage=='epoch120':payload['epoch']=120
    elif damage=='recipe':payload['cfg']['optim']['lr']=.1
    else:payload['reconstruction']['num_classes']=999
    monkeypatch.setattr(torch,'load',lambda *a,**k:payload)
    monkeypatch.setattr('reid.utils.checkpoint.reconstruct_model',lambda *a:pytest.fail('Invalid continuation reached model reconstruction'))
    with pytest.raises(ValueError):gate.resume_guard(cfg,tmp_path,751)


def test_live_uuid_mismatch_refuses_before_any_model_allocation(monkeypatch):
    from types import SimpleNamespace
    uuid='GPU-11111111-2222-3333-4444-555555555555'
    monkeypatch.setenv('MGN_AUTHORIZED_GPU_UUID',uuid);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',uuid)
    monkeypatch.setattr('reid.utils.lodo_device.audit_gpu',lambda *a,**k:{'driver':'test'})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(uuid='different',name='test'))
    with pytest.raises(ValueError,match='Live GPU differs'):gate.gpu_guard(uuid,{'gpu_name':'test'},0)
