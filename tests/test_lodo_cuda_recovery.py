"""Phase 9R CPU qualification; CUDA metadata doubles never allocate a GPU."""
import copy
import json
from pathlib import Path

import pytest
import torch

from test_lodo_checkpoint import (synthetic, cfg_for, parts, epoch, save, seed,
                                 assert_equal, FixtureModel, StochasticDataset, target_guards)
from test_lodo_updates import loader_for
from reid.data.index_observation import ObservedTrainLoader
from reid.engine.update_control import UpdateProgress, UpdateFailure
from reid.utils import lodo_checkpoint as ck, lodo_recovery as recovery, lodo_device as dev
from reid.utils.lodo_protocol import scientific_configuration
from scripts import train

POLICY=dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2)
UUID='GPU-11111111-2222-3333-4444-555555555555'


def configured(arch='resnet50',budget=8,run='recovery'):
    cfg=cfg_for(arch,budget=budget,run=run);cfg['train']['recovery']=copy.deepcopy(POLICY)
    return cfg


def pointer(cfg,run):
    request=copy.deepcopy(cfg);request['train']['save']['resume']=str(run/'checkpoints/ckpt_last.pth')
    return recovery.select_generation(request,run)


def write_boundary(objects,p,run,next_epoch):
    ck.save_lodo_checkpoint(run/'checkpoints/ckpt_last.pth',progress=p,next_epoch=next_epoch,**objects)
    return pointer(objects['cfg'],run)[0]


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('workers',[0,2])
@pytest.mark.parametrize('cadence',[1,2])
def test_retained_orchestration_rollback_no_double_count_and_terminal_noop(synthetic,monkeypatch,architecture,workers,cadence):
    target_guards(monkeypatch);built=[]
    def loader(cfg):
        ds=StochasticDataset();mode='pk' if architecture=='resnet50' else 'random'
        base=loader_for(ds,mode,workers);base.effective_batch_size=64
        return ObservedTrainLoader(base,sampler_mode=mode),ds.num_classes
    def model(cfg,num_classes,**kwargs):
        obj=FixtureModel(cfg,num_classes);built.append(obj);return obj
    monkeypatch.setattr(train,'build_train_loader',loader);monkeypatch.setattr(train,'build_model',model)
    continuous=configured(architecture,run='reference');continuous['data']['num_workers']=workers;continuous['train']['recovery']['every_epochs']=cadence
    expected=train.run_lodo_training(continuous);expected_model=built[-1]
    cfg=configured(architecture,run='interrupted');cfg['data']['num_workers']=workers;cfg['train']['recovery']['every_epochs']=cadence
    def failing(*a,**k):
        obj=model(*a,**k);obj.fail_at=5;return obj
    monkeypatch.setattr(train,'build_model',failing)
    with pytest.raises(UpdateFailure):train.run_lodo_training(cfg)
    run=synthetic/cfg['experiment']['output_dir'];boundary,_=pointer(cfg,run)
    committed=3 if cadence==1 else 0
    assert ck.read_checkpoint(boundary)[0]['progress']['completed_primary_updates']==committed
    cfg['train']['save']['resume']=str(run/'checkpoints/ckpt_last.pth')
    monkeypatch.setattr(train,'build_model',model)
    resumed=train.run_lodo_training(cfg)
    assert_equal(expected_model.state_dict(),built[-1].state_dict())
    assert_equal(expected_model.inputs[committed:],built[-1].inputs)
    assert_equal(list(expected.index_history),list(resumed.index_history))
    assert_equal(expected.exposure.presentations,resumed.exposure.presentations)
    terminal=run/'checkpoints/ckpt_terminal.pth';actual,_=ck.read_checkpoint(terminal)
    ref,_=ck.read_checkpoint(synthetic/continuous['experiment']['output_dir']/'checkpoints/ckpt_terminal.pth')
    for key in ('model','center','optimizer','center_optimizer','scheduler'):
        assert_equal(ref[key],actual[key])
    assert actual['progress']['completed_primary_updates']==8
    events=json.loads((run/'recovery_events.json').read_text())
    assert events[0]['discarded_successful_updates']==4-committed and events[1]['restored_primary_updates']==committed
    assert events[0]['recover_from']==str(boundary)
    assert len(list((run/'checkpoints').glob('ckpt_epoch_*.pth')))==2
    assert train.run_lodo_training(cfg).completed_primary_updates==8 and built[-1].inputs==[]
    assert all(row['write_seconds']>0 and row['bytes']>0 for row in json.loads((run/'checkpoint_metrics.json').read_text()))


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_retention_corrupt_latest_fallback_and_immutable_successor(synthetic,architecture):
    cfg=configured(architecture,budget=20);o=parts(cfg);p=UpdateProgress(20,auxiliary=architecture=='resnet50')
    run=synthetic/cfg['experiment']['output_dir'];(run/'checkpoints').mkdir(parents=True)
    p.bind(o['loader']);first=write_boundary(o,p,run,1)
    epoch(o,p);second=write_boundary(o,p,run,2)
    epoch(o,p,2);third=write_boundary(o,p,run,3)
    assert not first.exists() and second.exists() and third.exists()
    third.write_bytes(b'corrupt smoke generation')
    selected,failures=pointer(cfg,run);assert selected==second and len(failures)==1
    bad=copy.deepcopy(cfg);bad['train']['save']['resume']=str(third)
    with pytest.raises(ValueError):recovery.select_generation(bad,run)
    restored,position,_=ck.restore_lodo_checkpoint(selected,**o)
    epoch(o,restored,position['next_epoch']);replacement=write_boundary(o,restored,run,3)
    assert replacement!=third and third.read_bytes()==b'corrupt smoke generation'
    rows=recovery._read_index(run/'checkpoints')
    assert [r['name'] for r in rows]==[second.name,replacement.name]


def test_index_publication_failure_keeps_last_valid_generation(synthetic,monkeypatch):
    cfg=configured(budget=20);o=parts(cfg);p=UpdateProgress(20,auxiliary=True);p.bind(o['loader'])
    run=synthetic/cfg['experiment']['output_dir'];(run/'checkpoints').mkdir(parents=True)
    first=write_boundary(o,p,run,1);epoch(o,p)
    replace=ck.os.replace
    def fail(src,dst):
        if str(dst).endswith('recovery_index.json'):raise OSError('index publication interrupted')
        return replace(src,dst)
    with monkeypatch.context() as patch:
        patch.setattr(ck.os,'replace',fail)
        with pytest.raises(OSError):write_boundary(o,p,run,2)
    assert pointer(cfg,run)[0]==first
    assert len(list((run/'checkpoints').glob('ckpt_epoch_*.pth')))==2 # orphan never promoted
    replacement=write_boundary(o,p,run,2);assert replacement!=first
    assert len(recovery._read_index(run/'checkpoints'))==2


@pytest.mark.parametrize('field,value',[('version',2),('version',True),('policy','mid_epoch'),('every_epochs',0),('every_epochs',True),('keep_generations',1),('unknown',1)])
def test_unsupported_recovery_policy_rejected(synthetic,field,value):
    cfg=configured();cfg['train']['recovery'][field]=value
    with pytest.raises(ValueError):scientific_configuration(cfg)


@pytest.mark.parametrize('workers',[1,2,4])
def test_cuda_worker_modes_rejected_before_cuda(synthetic,monkeypatch,workers):
    cfg=configured();cfg['system']['device']='cuda';cfg['data']['num_workers']=workers
    monkeypatch.setattr(dev,'audit_gpu',lambda *a,**k:pytest.fail('GPU access'))
    with pytest.raises(ValueError,match='zero workers'):train.run_lodo_training(cfg)


def test_nondefault_prefetch_rejected_before_save(synthetic):
    cfg=configured();cfg['data']['num_workers']=2;o=parts(cfg,workers=2);o['loader']._loader.prefetch_factor=3
    with pytest.raises(ValueError,match='prefetch'):ck.make_identity(**o)


def fake_cuda_generation(synthetic,monkeypatch):
    cfg=configured();o=parts(cfg);p=UpdateProgress(8,auxiliary=True);epoch(o,p)
    run=synthetic/cfg['experiment']['output_dir'];(run/'checkpoints').mkdir(parents=True)
    path=write_boundary(o,p,run,2)
    cfg['system'].update(device='cuda',gpu_id=0);cfg['train']['save']['resume']=str(run/'checkpoints/ckpt_last.pth')
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID',UUID);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID)
    monkeypatch.setattr(recovery,'execution_environment',lambda cfg:dict(uuid=UUID,torch='test-only'))
    o['cfg']=cfg;payload,manifest=ck.read_checkpoint(path)
    payload['identity']=ck.make_identity(**o,cpu_inspection=True);payload['cfg']=copy.deepcopy(cfg)
    payload['rng']['cuda']=[torch.zeros(16,dtype=torch.uint8)]
    def publish():
        torch.save(payload,path);manifest['checkpoint_sha256']=ck.file_digest(path);manifest['identity_digest']=ck.digest(payload['identity'])
        manifest.update(fold=payload['identity']['protocol']['fold'],map_digest=payload['identity']['map_digest'],
                        config_digest=payload['identity']['config_digest'],data_digest=payload['identity']['data_digest'])
        path.with_suffix('.json').write_bytes(ck.canonical_json(manifest))
        index=dict(version=1,generations=[dict(name=path.name,checkpoint_sha256=manifest['checkpoint_sha256'],identity_digest=manifest['identity_digest'])])
        (run/'checkpoints/recovery_index.json').write_bytes(ck.canonical_json(index))
    publish();return cfg,o,path,payload,publish


@pytest.mark.parametrize('mutation',[
 lambda x:x['identity'].update(map_digest='0'*64),
 lambda x:x['identity']['protocol'].update(fold='LODO-2'),
 lambda x:x['identity']['recovery']['environment'].update(torch='other'),
 lambda x:x['position'].update(boundary='mid_epoch'),
 lambda x:x['progress'].update(consistent=False),
 lambda x:x['scheduler'].update(completed_updates=0),
 lambda x:x.update(center=None),
 lambda x:x['optimizer']['param_groups'][0].update(weight_decay=1),
 lambda x:x['rng'].update(cuda=[]),
])
def test_bad_cuda_resume_rejected_on_cpu_before_discovery(synthetic,monkeypatch,mutation):
    cfg,o,path,payload,publish=fake_cuda_generation(synthetic,monkeypatch)
    mutation(payload);publish();target_guards(monkeypatch)
    monkeypatch.setattr(dev,'select_lodo_device',lambda *a:pytest.fail('CUDA allocated before validation'))
    monkeypatch.setattr(train,'build_train_loader',lambda cfg:(o['loader'],o['loader'].dataset.num_classes))
    monkeypatch.setattr(train,'build_model',lambda cfg,num_classes,**k:FixtureModel(cfg,num_classes))
    with pytest.raises(ValueError):train.run_lodo_training(cfg)
    assert not torch.cuda.is_initialized()


def test_valid_cuda_payload_cpu_inspection_is_nonmutating(synthetic,monkeypatch):
    cfg,o,path,_,_=fake_cuda_generation(synthetic,monkeypatch)
    before=copy.deepcopy(o['model'].state_dict())
    progress,position,kind=ck.restore_lodo_checkpoint(path,**o,cpu_inspection=True,validate_only=True)
    assert progress.completed_primary_updates==3 and position['next_epoch']==2 and kind=='recovery'
    assert_equal(before,o['model'].state_dict());assert not torch.cuda.is_initialized()


def test_optin_does_not_grant_uuid_authority_or_change_scientific_identity(synthetic,monkeypatch):
    cfg=configured();plain=copy.deepcopy(cfg);plain['train'].pop('recovery')
    assert ck.configuration_identity(cfg)==ck.configuration_identity(plain)
    cfg['system'].update(device='cuda',gpu_id=0)
    monkeypatch.delenv('LODO_AUTHORIZED_GPU_UUID',raising=False)
    with pytest.raises(ValueError,match='UUID'):dev.authorization_uuid(cfg)


def test_incomplete_latest_pair_falls_back_without_repair(synthetic):
    cfg=configured(budget=20);o=parts(cfg);p=UpdateProgress(20,auxiliary=True);p.bind(o['loader'])
    run=synthetic/cfg['experiment']['output_dir'];(run/'checkpoints').mkdir(parents=True)
    first=write_boundary(o,p,run,1);epoch(o,p);second=write_boundary(o,p,run,2)
    second.with_suffix('.json').unlink()
    selected,failures=pointer(cfg,run)
    assert selected==first and len(failures)==1 and second.exists()
    assert not second.with_suffix('.json').exists()
