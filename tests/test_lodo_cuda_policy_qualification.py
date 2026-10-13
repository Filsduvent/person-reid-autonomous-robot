"""CPU-only contract checks for the new production orchestration diagnostic."""
from pathlib import Path
import pytest
import torch
import lodo_cuda_policy_qualification as q
from test_lodo_config import lodo_config

@pytest.mark.parametrize('loss,delta,outside',[(False,.0002,1),(True,.0002,0),(True,.002,1),(False,0.,0)])
def test_original_tolerances(loss,delta,outside):
    r=q.compare_values({'v':torch.tensor([delta])},{'v':torch.tensor([0.])},loss=loss)
    assert r['outside']==outside

@pytest.mark.parametrize('kind',['counter','keys','shape','dtype','nan'])
def test_fail_closed_comparison(kind):
    a={'v':torch.zeros(1),'counter':2};b={'v':torch.zeros(1),'counter':2}
    if kind=='counter':b['counter']=3;assert not q.compare_values(a,b)['non_tensor_equal']
    if kind=='nan':b['v'][0]=float('nan');assert not q.compare_values(a,b)['finite']
    if kind=='keys':b['other']=b.pop('v')
    if kind=='shape':b['v']=torch.zeros(2)
    if kind=='dtype':b['v']=torch.zeros(1,dtype=torch.float64)
    if kind in ('keys','shape','dtype'):
        with pytest.raises(ValueError):q.compare_values(a,b)

@pytest.mark.parametrize('suffix',['active/../active','active/extra','wrong','../escape'])
def test_namespace_rejects_alias(tmp_path,monkeypatch,suffix):
    monkeypatch.setattr(q,'SMOKE',tmp_path);cfg=lodo_config()
    cfg['experiment']['output_dir']=str(tmp_path/'phase9r3-test/resnet50/LODO-1'/suffix)
    with pytest.raises(ValueError):q.output_path(cfg)

def test_namespace_preserves_same_run_validator(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'SMOKE',tmp_path);cfg=lodo_config()
    p=tmp_path/'phase9r3-test/resnet50/LODO-1/active';cfg['experiment']['output_dir']=str(p)
    assert q.output_path(cfg)==p
    p.mkdir(parents=True)
    with pytest.raises(ValueError):q.output_path(cfg)
    cfg['train']['save']['resume']=str(p/'checkpoints/ckpt_last.pth')
    def rejected(*a,**k):raise ValueError('actual same-run validator')
    monkeypatch.setattr(q.ck,'validate_resume_path',rejected)
    with pytest.raises(ValueError,match='actual same-run'):q.output_path(cfg)

def test_clone_preserves_exact_checkpoint_inode_and_isolates_metadata(tmp_path):
    source=tmp_path/'boundary';(source/'checkpoints').mkdir(parents=True)
    ck=source/'checkpoints/test.pth';ck.write_bytes(b'unchanged checkpoint bytes')
    index=source/'checkpoints/recovery_index.json';index.write_text('index')
    target=tmp_path/'active';q.clone_boundary(source,target)
    assert ck.stat().st_ino==(target/'checkpoints/test.pth').stat().st_ino
    assert index.stat().st_ino!=(target/'checkpoints/recovery_index.json').stat().st_ino
    with pytest.raises(ValueError):q.clone_boundary(source,target)
    (source/'checkpoints/ckpt_terminal.pth').write_bytes(b'completed')
    with pytest.raises(ValueError):q.clone_boundary(source,tmp_path/'other')

def test_clone_rejects_symlink_before_copy(tmp_path):
    source=tmp_path/'boundary';source.mkdir();(source/'alias').symlink_to(tmp_path/'other')
    with pytest.raises(ValueError):q.clone_boundary(source,tmp_path/'active')

@pytest.mark.parametrize('case',['A','B','C'])
def test_recorder_bounds_and_does_not_draw_rng(tmp_path,case):
    before=torch.get_rng_state().clone()
    for name in ('A','B','C'):
        rec=q.Recorder(tmp_path,name,'pcb')
        rec.capture('ignored',torch.ones(1),step=2)
        assert not rec.events
        rec.capture('12_accounting',{'weight':torch.ones(1),'count':3},step=3)
        if name!='A':assert all(r['outside']==0 and r['non_tensor_equal'] for r in rec.comparisons)
        with pytest.raises(ValueError):rec.capture('12_accounting',torch.ones(1),step=3)
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()

def test_harness_uses_actual_orchestration_and_no_manual_restore_loop():
    source=Path(q.__file__).read_text()
    assert 'progress=train.run_lodo_training(cfg)' in source
    assert 'while not progress.budget_completed' not in source
    assert "os.kill(os.getpid(),signal.SIGTERM)" in source
    assert "patch.object(ck,'validate_resume_path'" not in source
    assert "patch.object(recovery,'select_generation'" not in source
