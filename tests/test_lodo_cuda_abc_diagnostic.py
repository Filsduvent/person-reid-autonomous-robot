"""CPU contracts for the same-checkpoint diagnostic, without CUDA initialization."""
from contextlib import ExitStack
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lodo_cuda_abc_diagnostic as abc
from lodo_cuda_recovery_smoke import ShortTestEpoch
from test_lodo_checkpoint import synthetic, cfg_for, parts, epoch, save, seed
from reid.engine.update_control import UpdateProgress
from reid.engine.train_loop import train_one_epoch
from reid.utils import lodo_checkpoint as ck


@pytest.mark.parametrize('a,b', [([.0002,1.002],[0.,1.]), ([1001.05],[1000.]), ([0.],[0.])])
def test_original_comparison_direction(a,b):
    a=torch.tensor(a);b=torch.tensor(b)
    assert abc.difference(a,b)['outside']==int((~torch.isclose(a,b,rtol=.001,atol=.0001)).sum())


def test_integer_shape_dtype_and_nonfinite_gates():
    assert abc.difference(torch.tensor([5]),torch.tensor([6]))['outside']==1
    with pytest.raises(AssertionError):abc.difference(torch.zeros(2),torch.zeros(3))
    with pytest.raises(AssertionError):abc.difference(torch.zeros(2),torch.zeros(2,dtype=torch.float64))
    assert not abc.difference(torch.tensor([float('nan')]),torch.tensor([0.]))['finite']
    with pytest.raises(AssertionError):abc.compare({'a':torch.zeros(1)},{'b':torch.zeros(1)})
    assert not abc.compare({'a':torch.zeros(1),'counter':2},{'a':torch.zeros(1),'counter':3})['non_tensor_equal']


def test_abc_uses_earlier_as_actual_and_draws_no_rng(tmp_path):
    before=(torch.get_rng_state().clone(),np.random.get_state())
    for name,value in [('A',.0002),('B',0.),('C',.0001)]:
        rec=abc.Recorder(tmp_path,name)
        rec.capture('01_checkpoint',{'weight':torch.tensor([value])},step=3)
        if name=='B':assert rec.comparisons[0]['outside']==1
        if name=='C':assert [r['pair'] for r in rec.comparisons]==['A/C','B/C']
    assert torch.equal(before[0],torch.get_rng_state())
    assert np.array_equal(before[1][1],np.random.get_state()[1])
    assert not torch.cuda.is_initialized()
    with pytest.raises(FileExistsError):abc.Recorder(tmp_path,'A')


def test_capture_rejects_nonfinite_and_bounds_updates(tmp_path):
    rec=abc.Recorder(tmp_path,'A')
    rec.capture('ignored',torch.zeros(1),step=2)
    assert not rec.events
    with pytest.raises(AssertionError):rec.capture('bad',torch.tensor([float('inf')]),step=3)


@pytest.mark.parametrize('suffix',['../escape','producer/extra','other','producer/../producer'])
def test_namespace_fail_closed(tmp_path,monkeypatch,suffix):
    monkeypatch.setattr(abc,'ROOT',tmp_path)
    cfg=cfg_for();cfg['experiment']['output_dir']=str(tmp_path/'exp/lodo_smoke/phase9r1-abc-test/resnet50/LODO-1'/suffix)
    with pytest.raises(ValueError):abc.output_path(cfg)


def test_namespace_collision_and_root_rebinding(tmp_path,monkeypatch):
    monkeypatch.setattr(abc,'ROOT',tmp_path)
    cfg=cfg_for();p=tmp_path/'exp/lodo_smoke/phase9r1-abc-test/resnet50/LODO-1/producer'
    cfg['experiment']['output_dir']=str(p)
    assert abc.output_path(cfg)==p
    with pytest.raises(ValueError):abc.output_path(cfg,project_root=Path('/different'))
    p.mkdir(parents=True)
    with pytest.raises(ValueError):abc.output_path(cfg)


def test_restore_captures_actual_order_and_preserves_objects(synthetic):
    cfg=cfg_for();cfg['train']['recovery']=dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2)
    seed();o=parts(cfg);o['loader']._observer.original=ShortTestEpoch(o['loader'].batch_sampler)
    p=UpdateProgress(5,auxiliary=True);epoch(o,p)
    assert p.completed_primary_updates==2
    path=synthetic/'checkpoints/ckpt_last.pth';save(o,p,path)
    from reid.utils import lodo_recovery as recovery
    request=copy.deepcopy(cfg);request['train']['save']['resume']=str(path)
    path,_=recovery.select_generation(request,synthetic)
    a=abc.Recorder(synthetic,'A');a.objects=o;a.progress=p
    abc.boundary_capture(a,path,o)
    resumed=parts(cfg);b=abc.Recorder(synthetic,'B');b.objects=resumed
    center_parameter=resumed['criterion'].center_loss.centers
    with ExitStack() as stack:
        abc.observe_restore(stack,b,path)
        restored,_,_=ck.restore_lodo_checkpoint(path,**resumed)
    b.progress=restored;abc.boundary_capture(b,path,resumed,capture=False)
    assert [e['stage'] for e in b.events]==['01_checkpoint','02_restored_modules','03_restored_optimizers','04_restored_scheduler_rng']
    assert all(r['outside']==0 and r['non_tensor_equal'] and not r['nonexact_tensors'] for r in b.comparisons)
    assert resumed['auxiliary'].param_groups[0]['params'][0] is center_parameter
    assert not torch.cuda.is_initialized()


def test_trace_targets_original_engine_despite_module_wrapper(synthetic,monkeypatch):
    cfg=cfg_for();seed();o=parts(cfg);p=UpdateProgress(5,auxiliary=True)
    calls=[]
    rec=SimpleNamespace(objects=o,current=3,progress=p,implementations={},
        capture=lambda stage,value:calls.append(stage),gradients=lambda:{'x':torch.zeros(1)},
        accounting=lambda:calls.append('12_accounting'))
    monkeypatch.setattr(abc.train,'train_one_epoch',lambda *a,**k:None)
    with ExitStack() as stack:
        abc.install_engine_observation(stack,rec,train_one_epoch)
        epoch(o,p)
    assert calls.count('07_backward')==3 and calls.count('08_center_scaled')==3
    assert calls.count('12_accounting')==1 and rec.implementations
    assert not torch.cuda.is_initialized()


def test_deterministic_fixture_is_local_and_production_flags_remain_frozen():
    from test_lodo_config import lodo_config
    cfg=lodo_config();cfg['train']['max_updates']=5
    cfg['repro'].update(deterministic=True,benchmark=False)
    original=abc.protocol._expect
    with pytest.raises(ValueError):abc.protocol.scientific_configuration(cfg)
    with abc.diagnostic_budget('deterministic'):
        identity=abc.protocol.scientific_configuration(cfg)
        assert identity['deterministic'] and not identity['benchmark']
    assert abc.protocol._expect is original
    cfg['train']['max_updates']=85200
    with pytest.raises(ValueError):abc.protocol.scientific_configuration(cfg)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation',['sha','identity','gpu','path','updates'])
def test_common_checkpoint_provenance_rejected_before_components(tmp_path,monkeypatch,mutation):
    import json
    base=tmp_path.resolve();checkpoint=base/'resnet50/LODO-1/producer/checkpoints/common.pth'
    common=dict(checkpoint=str(checkpoint),sha256='a'*64,identity_digest='b'*64,policy='original',gpu_uuid=abc.TITAN_UUID)
    payload=dict(progress=dict(completed_primary_updates=2),identity=dict(recovery=dict(environment=dict(uuid=abc.TITAN_UUID))))
    proof=dict(checkpoint_sha256='a'*64,identity_digest='b'*64)
    monkeypatch.setattr(abc.ck,'read_checkpoint',lambda path:(payload,proof))
    monkeypatch.setattr(abc.recovery,'validate_boundary',lambda payload:None)
    if mutation=='sha':common['sha256']='c'*64
    if mutation=='identity':common['identity_digest']='c'*64
    if mutation=='gpu':payload['identity']['recovery']['environment']['uuid']=abc.UUID
    if mutation=='path':common['checkpoint']=str(base/'other/common.pth')
    if mutation=='updates':payload['progress']['completed_primary_updates']=3
    (base/'common.json').write_text(json.dumps(common))
    with pytest.raises(AssertionError):abc.read_common(base,'original',abc.TITAN_UUID)
    assert not torch.cuda.is_initialized()


def test_duplicate_event_does_not_replace_evidence(tmp_path):
    rec=abc.Recorder(tmp_path,'A');rec.capture('01_test',torch.ones(1),step=3)
    with pytest.raises(ValueError,match='Duplicate'):rec.capture('01_test',torch.zeros(1),step=3)
    assert torch.equal(torch.load(tmp_path/'A/u003-01_test.pth',weights_only=False),torch.ones(1))


@pytest.mark.parametrize('mismatch', [None, 'selected', 'requested', 'integrity'])
def test_existing_producer_requires_verified_exact_common_resume(tmp_path,monkeypatch,mismatch):
    monkeypatch.setattr(abc,'ROOT',tmp_path)
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID',abc.TITAN_UUID)
    producer=tmp_path/'exp/lodo_smoke/phase9r1-abc-titan-test/resnet50/LODO-1/producer'
    producer.mkdir(parents=True)
    checkpoint=producer/'checkpoints/common.pth'
    cfg=cfg_for();cfg['experiment']['output_dir']=str(producer)
    cfg['train']['save']['resume']=str(checkpoint)
    calls=[]
    def validate(actual,path):
        calls.append(('resume',path))
        return producer/'checkpoints/other.pth' if mismatch=='selected' else checkpoint
    def common(base,policy,uuid):
        calls.append(('common',base,policy,uuid))
        if mismatch=='integrity':raise ValueError('Checkpoint integrity mismatch')
        return {},checkpoint,{},{}
    monkeypatch.setattr(abc.ck,'validate_resume_path',validate)
    monkeypatch.setattr(abc,'read_common',common)
    if mismatch=='requested':cfg['train']['save']['resume']=str(producer/'checkpoints/other.pth')
    if mismatch is None:
        assert abc.output_path(cfg)==producer
        assert calls==[('resume',producer),('common',producer.parents[2],'original',abc.TITAN_UUID)]
    else:
        with pytest.raises(ValueError):abc.output_path(cfg)
    assert not torch.cuda.is_initialized()


def test_existing_producer_resume_rejection_propagates_before_common_read(tmp_path,monkeypatch):
    monkeypatch.setattr(abc,'ROOT',tmp_path)
    producer=tmp_path/'exp/lodo_smoke/phase9r1-abc-titan-test/resnet50/LODO-1/producer'
    producer.mkdir(parents=True)
    cfg=cfg_for();cfg['experiment']['output_dir']=str(producer)
    cfg['train']['save']['resume']=str(producer/'checkpoints/missing.pth')
    def reject(*args):raise ValueError('Missing or incompatible same-run checkpoint')
    monkeypatch.setattr(abc.ck,'validate_resume_path',reject)
    monkeypatch.setattr(abc,'read_common',lambda *args:pytest.fail('Invalid resume reached common read'))
    with pytest.raises(ValueError,match='Missing or incompatible'):abc.output_path(cfg)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('path,token',[('A','valid'),('B','../escape'),('C',''),('C','/absolute')])
def test_continuation_namespace_rejects_invalid(tmp_path,path,token):
    with pytest.raises(ValueError):abc.continuation_artifacts(tmp_path,path,token)


def test_continuation_namespace_requires_canonical_existing_directory(tmp_path):
    with pytest.raises(ValueError):abc.continuation_artifacts(tmp_path,'B','fresh')
    directory=tmp_path/'continuations/fresh';directory.mkdir(parents=True)
    assert abc.continuation_artifacts(tmp_path,'C','fresh')==directory
    (tmp_path/'continuations/alias').symlink_to(directory,target_is_directory=True)
    with pytest.raises(ValueError):abc.continuation_artifacts(tmp_path,'B','alias')


def test_continuation_preserves_original_A_and_failed_B(tmp_path):
    origin=tmp_path/'origin';origin.mkdir()
    a=abc.Recorder(origin,'A');a.capture('01_test',torch.ones(1),step=3);a.flush()
    failed=origin/'B';failed.mkdir();(failed/'outcome.json').write_text('original failure')
    before={str(p.relative_to(origin)):p.read_bytes() for p in origin.rglob('*') if p.is_file()}
    continuation=origin/'continuations/new';continuation.mkdir(parents=True)
    b=abc.Recorder(continuation,'B',reference_a=origin/'A')
    b.capture('01_test',torch.ones(1),step=3)
    c=abc.Recorder(continuation,'C',reference_a=origin/'A')
    c.capture('01_test',torch.ones(1),step=3)
    assert [r['pair'] for r in b.comparisons]==['A/B']
    assert [r['pair'] for r in c.comparisons]==['A/C','B/C']
    assert all(r['outside']==0 and r['non_tensor_equal'] for r in c.comparisons)
    for name,data in before.items():assert (origin/name).read_bytes()==data
    with pytest.raises(FileExistsError):abc.Recorder(continuation,'B',reference_a=origin/'A')
    assert not torch.cuda.is_initialized()
