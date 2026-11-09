"""CPU scientific/engine/identity checks; no CUDA or real benchmark execution."""
import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from torch import nn
from reid.utils.config import load_config, validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils import mgn_lodo_protocol as v2
from reid.utils import lodo_protocol as v1
from reid.utils.lodo_dispatch import configuration_identity
from reid.optim.build import build_optimizer, build_scheduler
from reid.losses.build import build_criterion
from reid.engine.train_loop import train_one_epoch
from reid.engine.update_control import UpdateProgress
from reid.utils.checkpoint import make_reconstruction_metadata
from reid.data.index_observation import ObservedTrainLoader
from test_lodo_updates import loader_for
from test_lodo_checkpoint import StochasticDataset, assert_equal, epoch
from reid.utils import lodo_checkpoint as ck

ROOT=Path(__file__).resolve().parents[1]


def config(fold=1,root=None):
    cfg=load_config(str(ROOT/f'configs/lodo/mgn_lodo_v2/mgn_lodo-{fold}_seed42.yaml'))
    if root is not None:
        cfg['experiment']['output_dir']=str(root/'exp/lodo/mgn_lodo_v2/mgn'/f'LODO-{fold}'/'seed42/test')
    return cfg


@pytest.mark.parametrize('fold',range(1,5))
def test_approved_presets_common_validation_and_v1_rejection(fold,tmp_path):
    cfg=config(fold,tmp_path);validate_config(cfg,project_root=tmp_path);validate_reid_config(cfg,project_root=tmp_path)
    assert 'epochs' not in cfg['train']
    assert v2.contract(cfg)[1]['name']==v1.FOLDS[f'LODO-{fold}'][1]
    with pytest.raises(ValueError):v1.validate_lodo_config(cfg,project_root=tmp_path)


@pytest.mark.parametrize('path,value',[
 ('protocol.version',1),('protocol.version',3),('protocol.architecture','pcb'),
 ('train.epochs',120),('train.max_updates',85201),('train.evaluate_during_training',True),
 ('train.checkpoint_selection','best'),('train.save.save_best',True),
 ('train.recovery.every_epochs',2),('data.num_workers',4),('system.device','cpu'),('system.amp',True),
 ('repro.cuda_policy','original_cuda_v1'),('sched.milestones',[42601,63900]),('sched.unit','epoch'),
 ('sched.warmup_iters',1),('loss.id.head_aggregation','sum'),('loss.triplet.margin',.3),
 ('loss.triplet.feature_source','legacy'),('loss.center.enabled',True),('optim.lr',.1),
 ('optim.param_groups',[{'prefix':'backbone.','lr_mult':.1}]),('data.train.batch.P',8),
 ('data.train.images.interpolation','bilinear'),('data.test.aug.mirror','random'),('eval.flip_aggregation','mean'),
 ('eval.rerank.enabled',True),('model.variant','other'),('model.head.classifier',False),
 ('data.train.dataset.sources',[{'name':'market1501'}]),('data.test.dataset.name','msmt17')])
def test_scientific_drift_rejected(path,value,tmp_path):
    cfg=config(root=tmp_path);item=cfg
    parts=path.split('.')
    for key in parts[:-1]:item=item[key]
    item[parts[-1]]=value
    with pytest.raises(ValueError):validate_config(cfg,project_root=tmp_path)


def test_identity_is_versioned_and_run_root_independent(tmp_path):
    cfg=config(root=tmp_path);before=configuration_identity(cfg)
    other=copy.deepcopy(cfg);other['experiment']['output_dir']=str(tmp_path/'elsewhere')
    other['model']['weights_path']='/absent-but-same-pinned-artifact'
    assert configuration_identity(other)==before
    other['protocol']['fold']='LODO-2'
    with pytest.raises(ValueError):configuration_identity(other)
    cfg['data']['train']['dataset']['sources'].reverse()
    with pytest.raises(ValueError):configuration_identity(cfg)


@pytest.mark.parametrize('name',['../escape','bad name','bad/child'])
def test_namespace_alias_and_collision(name,tmp_path):
    cfg=config(root=tmp_path);cfg['experiment']['output_dir']=str(tmp_path/'exp/lodo/mgn_lodo_v2/mgn/LODO-1/seed42'/name)
    with pytest.raises(ValueError):v2.validate_lodo_config(cfg,project_root=tmp_path)


def test_missing_execution_scope_refuses_before_gpu_data_or_output(monkeypatch,tmp_path):
    from reid.utils.mgn_lodo_operational import validate_training_admission
    monkeypatch.delenv('MGN_LODO_ADMISSION_RECEIPT',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried before admission'))
    cfg=config(root=tmp_path)
    with pytest.raises(ValueError,match='Phase21'):validate_training_admission('not-required-before-scope',cfg,root=tmp_path)
    assert not (tmp_path/'exp').exists()


class TinyMGN(nn.Module):
    """Transport/head fixture, not a replacement for the separately tested real MGN."""
    def __init__(self,cfg,classes):
        super().__init__();self.backbone=nn.Linear(2,2048);self.bn=nn.BatchNorm1d(2048)
        self.classifiers=nn.Module();self.classifiers.heads=nn.ModuleList(nn.Linear(2048 if i<3 else 256,classes,bias=False) for i in range(8))
        self.checkpoint_metadata=make_reconstruction_metadata(cfg,classes,2048);self.forward_calls=0
    def forward(self,x):
        self.forward_calls+=1
        if x.ndim==4:x=x.mean((2,3))[:,:2]
        z=self.bn(self.backbone(x/100))
        return dict(emb=torch.cat([z[:,:256]]*8,1),logits=tuple(h(z if i<3 else z[:,:256]) for i,h in enumerate(self.classifiers.heads)),metric_features=(z[:,:256],z[:,256:512],z[:,512:768]))


def parts(cfg,n=194):
    ds=StochasticDataset(n);loader=ObservedTrainLoader(loader_for(ds,'pk',0),sampler_mode='pk')
    model=TinyMGN(cfg,ds.num_classes);criterion=build_criterion(cfg,ds.num_classes,None)
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer)
    return dict(cfg=cfg,loader=loader,model=model,criterion=criterion,optimizer=optimizer,auxiliary=None,scheduler=scheduler)


def seed_progress(objects,n):
    """Consistent synthetic prefix counters: these are not claimed observed updates."""
    p=UpdateProgress(85200);p.bind(objects['loader']);e=p.exposure
    indices=tuple(range(64));labels=torch.tensor(objects['loader'].dataset.labels[:64]);records=e.resolve(indices,labels)
    for phase in e.phases:
        e.record(phase,records);e.batch_counts[phase]=n
        for counts in (e.presentations[phase],e.identity_presentations[phase],e.identity_slots[phase]):
            for key in counts:counts[key]*=n
        if not n:e.distinct[phase].clear()
    for key in ('completed_primary_updates','main_steps_returned','attempted_batches','forward_batches','backward_batches'):setattr(p,key,n)
    p.attempted_images=n*64;objects['scheduler'].prepare(n);p.check_totals();return p


@pytest.mark.parametrize('n,expected',[ (42599,.01),(42600,.001),(63899,.001),(63900,.0001),(85199,.0001)])
def test_boundary_used_lr_one_batch_one_step_and_no_terminal_extra(n,expected):
    objects=parts(config());p=seed_progress(objects,n)
    opt=objects['optimizer'];original=opt.step;calls=[]
    def step(*a,**k):calls.append(tuple(g['lr'] for g in opt.param_groups));return original(*a,**k)
    opt.step=step
    # Stop after one consumed batch at the exact requested boundary.
    p.max_updates=n+1
    result=epoch(objects,p,number=169)
    assert result.stop_reason=='budget_completed' and p.completed_primary_updates==n+1
    assert len(calls)==1 and all(rate==pytest.approx(expected) for rate in calls[0])
    assert p.main_steps_returned==n+1 and p.auxiliary_steps==0
    assert objects['scheduler'].completed_updates==n+1
    reads=len(objects['loader'].dataset.reads);forward=objects['model'].forward_calls
    state=copy.deepcopy(objects['model'].state_dict());epoch(objects,p,number=170)
    assert len(calls)==1 and len(objects['loader'].dataset.reads)==reads and objects['model'].forward_calls==forward
    assert_equal(state,objects['model'].state_dict())


def test_scheduler_entire_declared_trace_no_epoch_conversion():
    objects=parts(config());s=objects['scheduler']
    for update in range(1,85201):
        expected=.01 if update<=42600 else .001 if update<=63900 else .0001
        assert all(rate==pytest.approx(expected) for rate in s.rates_after(update-1))
    assert s.state_dict()['completed_updates']==0  # pure oracle, no optimizer calls


def test_no_optimizer_step_on_failed_forward():
    objects=parts(config());p=UpdateProgress(1)
    objects['optimizer'].step=Mock(side_effect=AssertionError('step after failed forward'))
    objects['model'].forward=Mock(side_effect=RuntimeError('source failure'))
    with pytest.raises(RuntimeError):epoch(objects,p,number=1)
    assert p.completed_primary_updates==0 and objects['scheduler'].completed_updates==0
    objects['optimizer'].step.assert_not_called()


@pytest.fixture
def cpu_checkpoint_scope(monkeypatch,tmp_path):
    """CPU-only contract double; no production CPU/budget escape is added."""
    expect=v2._expect
    def local(cfg,path,value):
        if path=='system.device':assert cfg['system']['device']=='cpu';return
        if path=='repro.cuda_policy':assert cfg['repro']['cuda_policy']=='original_cuda_v1';return
        if path=='train.max_updates':assert 0<cfg['train']['max_updates']<=121;return
        return expect(cfg,path,value)
    monkeypatch.setattr(v2,'_expect',local)
    output=v2.validate_output_path
    monkeypatch.setattr(v2,'validate_output_path',lambda cfg,project_root=None:output(cfg,project_root=project_root or tmp_path))
    return tmp_path


def test_mgn_source_heads_identity_recovery_and_terminal_noop(cpu_checkpoint_scope):
    cfg=config(root=cpu_checkpoint_scope);cfg['system']['device']='cpu';cfg['repro']['cuda_policy']='original_cuda_v1';cfg['train']['max_updates']=3
    objects=parts(cfg);p=UpdateProgress(3)
    # Use the real shared identity builder on the fixture's actual8 heads/source C.
    identity=ck.make_identity(**objects)
    assert identity['scientific_config']['scientific_protocol']=='mgn_lodo_v2'
    assert identity['num_classes']==objects['loader'].dataset.num_classes and identity['feature_dim']==2048
    assert len(objects['model'].classifiers.heads)==8 and identity['center'] is None
    checkpoint=Path(cfg['experiment']['output_dir'])/'checkpoints/ckpt_last.pth';checkpoint.parent.mkdir(parents=True)
    ck.save_lodo_checkpoint(checkpoint,progress=p,next_epoch=1,**objects)
    from reid.utils.lodo_recovery import select_generation
    resume=copy.deepcopy(cfg);resume['train']['save']['resume']=str(checkpoint)
    selected,_=select_generation(resume,checkpoint.parent.parent)
    first=epoch(objects,p,number=121)
    while not p.budget_completed:epoch(objects,p,number=122)
    baseline=copy.deepcopy(objects['model'].state_dict())
    restored,position,kind=ck.restore_lodo_checkpoint(selected,**objects)
    assert restored.completed_primary_updates==0 and kind=='recovery'
    while not restored.budget_completed:epoch(objects,restored,number=121)
    assert_equal(baseline,objects['model'].state_dict())
    terminal=checkpoint.with_name('ckpt_terminal.pth')
    ck.save_lodo_checkpoint(terminal,progress=restored,next_epoch=121,cursor=restored.epoch_results[-1].attempted_batches,kind='terminal',**objects)
    again,_,kind=ck.restore_lodo_checkpoint(terminal,**objects)
    before=objects['model'].forward_calls;epoch(objects,again,number=200)
    assert kind=='terminal' and objects['model'].forward_calls==before and again.completed_primary_updates==3


def test_bad_head_source_c_rejected_before_checkpoint(cpu_checkpoint_scope):
    cfg=config(root=cpu_checkpoint_scope);cfg['system']['device']='cpu';cfg['repro']['cuda_policy']='original_cuda_v1';cfg['train']['max_updates']=3
    objects=parts(cfg);objects['model'].classifiers.heads[0]=nn.Linear(2048,7,bias=False)
    with pytest.raises(ValueError,match='classifier dimensions'):ck.make_identity(**objects)


def test_bn_frozen_target_evaluation_and_adaptation_rejected(monkeypatch):
    from reid.utils.mgn_lodo_evaluation import evaluate_frozen_bn
    from reid.utils import lodo_evaluation
    model=nn.Sequential(nn.BatchNorm1d(2),nn.Linear(2,2))
    before=copy.deepcopy(model.state_dict())
    def evaluate(cfg,model,loader,device,**kwargs):
        kwargs['before_batch']();assert not torch.is_grad_enabled() and not model.training
        model(torch.ones(3,2));return {'mAP':.5}
    monkeypatch.setattr(lodo_evaluation,'evaluate_reid',evaluate)
    evaluate_frozen_bn({},model,None,torch.device('cpu'));assert_equal(before,model.state_dict())
    def adapt(cfg,model,loader,device,**kwargs):
        model[0].running_mean.add_(1);return {'mAP':.5}
    monkeypatch.setattr(lodo_evaluation,'evaluate_reid',adapt)
    with pytest.raises(ValueError,match='BatchNorm'):evaluate_frozen_bn({},model,None,torch.device('cpu'))


def test_real_mgn_identity_uses_all_actual_source_heads(cpu_checkpoint_scope):
    from reid.models.build import build_model
    cfg=config(root=cpu_checkpoint_scope);cfg['system']['device']='cpu';cfg['repro']['cuda_policy']='original_cuda_v1';cfg['train']['max_updates']=3
    objects=parts(cfg)
    objects['model']=build_model(cfg,objects['loader'].dataset.num_classes,initialize_pretrained=False)
    objects['optimizer']=build_optimizer(cfg,objects['model']);objects['scheduler']=build_scheduler(cfg,objects['optimizer'])
    identity=ck.make_identity(**objects)
    assert identity['num_classes']==49 and identity['feature_dim']==2048
    assert [h.in_features for h in objects['model'].classifiers.heads]==[2048]*3+[256]*5
    assert identity['numerical_policy']['full_fp32_arithmetic_guaranteed'] is False


def test_common_orchestrator_exceeds120_fixture_epochs_and_terminal_reopens_noop(cpu_checkpoint_scope,monkeypatch):
    """121 tiny vector-batch CPU epochs, not benchmark training or GPU qualification."""
    from scripts import train
    import reid.utils.lodo_device as device
    import reid.utils.mgn_lodo_operational as admission
    cfg=config(root=cpu_checkpoint_scope);cfg['system']['device']='cpu';cfg['repro']['cuda_policy']='original_cuda_v1';cfg['train']['max_updates']=121
    objects=parts(cfg,n=64);loader=objects['loader'];calls=[]
    monkeypatch.setattr(admission,'validate_training_session',lambda c:None)  # test-only CPU scope
    monkeypatch.setattr(device,'select_lodo_device',lambda c:(torch.device('cpu'),None,None))
    monkeypatch.setattr(train,'build_train_loader',lambda c:(loader,loader.dataset.num_classes))
    monkeypatch.setattr(train,'build_test_loader',lambda *a,**k:pytest.fail('target loader in training'))
    monkeypatch.setattr(train,'build_model',lambda c,num_classes,**kwargs:TinyMGN(c,num_classes))
    original=train.build_optimizer
    def optimizer(c,model):
        value=original(c,model);step=value.step
        def counted(*a,**k):calls.append(1);return step(*a,**k)
        value.step=counted;return value
    monkeypatch.setattr(train,'build_optimizer',optimizer)
    progress=train.run_lodo_training(cfg)
    assert progress.completed_primary_updates==121 and len(calls)==121
    assert progress.epoch_results[-1].epoch==121
    assert sum(progress.exposure.presentations['successful'].values())==121*64
    count=len(loader.dataset.reads)
    again=copy.deepcopy(cfg);again['train']['save']['resume']=str(Path(cfg['experiment']['output_dir'])/'checkpoints/ckpt_terminal.pth')
    restored=train.run_lodo_training(again)
    assert restored.budget_completed and len(calls)==121 and len(loader.dataset.reads)==count


@pytest.mark.parametrize('lost_journal',[False,True])
def test_common_interruption_discarded_work_and_unknown_abrupt_work_are_explicit(cpu_checkpoint_scope,monkeypatch,lost_journal):
    import json
    from scripts import train
    import reid.utils.lodo_device as device
    import reid.utils.mgn_lodo_operational as admission
    cfg=config(root=cpu_checkpoint_scope);cfg['system']['device']='cpu';cfg['repro']['cuda_policy']='original_cuda_v1';cfg['train']['max_updates']=3
    objects=parts(cfg,n=64);loader=objects['loader'];created=[]
    monkeypatch.setattr(admission,'validate_training_session',lambda c:None)
    monkeypatch.setattr(device,'select_lodo_device',lambda c:(torch.device('cpu'),None,None))
    monkeypatch.setattr(train,'build_train_loader',lambda c:(loader,loader.dataset.num_classes))
    monkeypatch.setattr(train,'build_test_loader',lambda *a,**k:pytest.fail('target loader in training'))
    def factory(c,num_classes,**kwargs):
        m=TinyMGN(c,num_classes)
        if not created:
            forward=m.forward
            def failing(x):
                if m.forward_calls==1:raise RuntimeError('source interruption')
                return forward(x)
            m.forward=failing
        created.append(m);return m
    monkeypatch.setattr(train,'build_model',factory)
    with pytest.raises(RuntimeError,match='source interruption'):train.run_lodo_training(cfg)
    root=Path(cfg['experiment']['output_dir']);events=root/'recovery_events.json'
    interrupted=json.loads((root/'work_accounting.json').read_text())
    assert interrupted['status']=='interrupted' and not interrupted['physical_work_totals_complete']
    if lost_journal:events.write_text('[]')  # explicit CPU fixture for an unobserved abrupt exit
    cfg['train']['save']['resume']=str(root/'checkpoints/ckpt_last.pth')
    progress=train.run_lodo_training(cfg);assert progress.completed_primary_updates==3
    summary=json.loads((root/'work_accounting.json').read_text())
    if lost_journal:
        assert summary['unknown_discarded_work_resume_events'] and not summary['physical_work_totals_complete']
        assert summary['known_discarded_source_presentations']['attempted']=={}
    else:
        assert summary['physical_work_totals_complete']
        assert sum(summary['known_discarded_source_presentations']['attempted'].values())==64
        assert summary['known_discarded_completed_primary_updates']==0
    assert sum(summary['current_in_memory_exposure']['successful']['per_source_images'].values())==3*64
