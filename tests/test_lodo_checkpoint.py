"""Phase 6 CPU fixtures; tiny budgets are TEST-ONLY in-memory substitutions.

Production protocol/budget validation stays frozen. No official datasets,
pretrained weights, GPU work, evaluator calls, or executable launch configs.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from test_lodo_config import lodo_config
from test_lodo_updates import IndexedToyDataset, loader_for
from test_multi_source_dataset import tree, config
from reid.data.collate import train_collate_fn
from reid.data.index_observation import ObservedTrainLoader
from reid.data.multi_source import canonical_json
from reid.data.protocol import ReIDTrainSample
from reid.engine.train_loop import train_one_epoch
from reid.engine.update_control import UpdateProgress, UpdateFailure
from reid.losses.build import build_criterion
from reid.optim.build import build_optimizer, build_center_optimizer, build_scheduler
from reid.optim.lr_scheduler import SuccessfulUpdateLR
from reid.utils.checkpoint import make_reconstruction_metadata
from reid.utils import lodo_checkpoint as ck
from reid.utils import lodo_protocol as protocol
from scripts import train


class StochasticDataset(IndexedToyDataset):
    def __init__(self, n=194):
        super().__init__(n)
        self.mapping_digest=ck.digest(self.mapping_metadata())
        self.target='market1501'
        self.preflight=dict(inventories={key:dict(metadata_hashes={f'{key}/partition':'a'*64},
                            descriptor_metadata_hashes=dict(source='a'*64,target='b'*64)) for key in (*self.sources,self.target)})
    def serialize_sidecar(self):
        return canonical_json(dict(schema_version=1,mapping_digest=self.mapping_digest,
                                   samples=[asdict(row) for row in self.sidecar]))
    def __getitem__(self,i):
        # Verify Python, NumPy AND torch augmentation RNG, including workers.
        value=torch.tensor([float(i),float(i%5)])+random.random()+float(np.random.rand())+torch.rand(())
        return ReIDTrainSample(value,self.labels[i])


class FixtureModel(nn.Module):
    def __init__(self,cfg,classes):
        super().__init__()
        self.arch=cfg['protocol']['architecture']
        dim=2048 if self.arch=='resnet50' else 1536
        self.feat_dim=dim
        self.backbone=nn.Linear(2,dim)
        self.dropout=nn.Dropout(.2)
        if self.arch=='resnet50': self.classifier=nn.Linear(dim,classes,bias=False)
        else:
            self.classifiers=nn.Module()
            self.classifiers.fc_list=nn.ModuleList(nn.Linear(256,classes) for _ in range(6))
        self.checkpoint_metadata=make_reconstruction_metadata(cfg,classes,dim)
        self.inputs=[]
        self.fail_at=None
    def forward(self,x):
        if x.ndim==4: x=x.mean((2,3))[:,:2]
        self.inputs.append(x.detach().clone())
        if len(self.inputs)==self.fail_at: raise RuntimeError('synthetic interruption')
        feat=self.dropout(self.backbone(x/100))
        if self.arch=='resnet50': return dict(emb=feat,feat_raw=feat,feat_bn=feat,logits=self.classifier(feat))
        return dict(emb=feat,logits=tuple(head(part) for head,part in zip(self.classifiers.fc_list,feat.chunk(6,dim=1))))


@pytest.fixture
def synthetic(monkeypatch,tmp_path):
    # Tiny budgets ONLY inside this fixture; all other recipe checks still run.
    original_expect=protocol._expect
    def expect(cfg,path,value):
        if path=='train.max_updates':
            assert type(cfg['train']['max_updates']) is int and 0<cfg['train']['max_updates']<=94200
        else: original_expect(cfg,path,value)
    monkeypatch.setattr(protocol,'_expect',expect)
    original_output=protocol.validate_output_path
    monkeypatch.setattr(protocol,'validate_output_path',lambda cfg,project_root=None: original_output(cfg,project_root=project_root or tmp_path))
    monkeypatch.setattr(train,'REPO_ROOT',tmp_path)
    return tmp_path


def cfg_for(architecture='resnet50',budget=5,run='test'):
    cfg=lodo_config(architecture)
    cfg['train']['max_updates']=budget
    cfg['system'].update(device='cpu',log_interval=1000)
    cfg['data'].update(num_workers=0,pin_memory=False)
    cfg['experiment']['output_dir']=cfg['experiment']['output_dir'].replace('run-1',run)
    return cfg


def seed(value=42):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def parts(cfg,workers=0,*,generator=False):
    ds=StochasticDataset()
    mode='pk' if cfg['protocol']['architecture']=='resnet50' else 'random'
    if generator and mode=='random':
        original=DataLoader(ds,batch_size=64,shuffle=True,drop_last=True,collate_fn=train_collate_fn,
                            num_workers=workers,generator=torch.Generator().manual_seed(197))
    else: original=loader_for(ds,mode,workers)
    original.effective_batch_size=64
    loader=ObservedTrainLoader(original,sampler_mode=mode)
    model=FixtureModel(cfg,ds.num_classes)
    criterion=build_criterion(cfg,ds.num_classes,model.feat_dim)
    criterion.loss_trace=[]
    criterion.register_forward_hook(lambda module,inputs,output: module.loss_trace.append(float(output[0].detach())))
    opt=build_optimizer(cfg,model);aux=build_center_optimizer(cfg,criterion);sched=build_scheduler(cfg,opt)
    return dict(cfg=cfg,loader=loader,model=model,criterion=criterion,optimizer=opt,auxiliary=aux,scheduler=sched)


def epoch(objects,progress,number=1):
    return train_one_epoch(objects['model'],objects['loader'],objects['criterion'],objects['optimizer'],objects['auxiliary'],
                           torch.device('cpu'),False,1000,scheduler=objects['scheduler'],epoch=number,update_progress=progress)


def assert_equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_equal(x,y)
    else: assert a==b


def save(objects,p,path,next_epoch=2,**kwargs):
    path.parent.mkdir(parents=True,exist_ok=True)
    return ck.save_lodo_checkpoint(path,progress=p,next_epoch=next_epoch,**objects,**kwargs)


def rewrite(path,mutate):
    payload,manifest=ck.read_checkpoint(path);mutate(payload)
    torch.save(payload,path)
    manifest['checkpoint_sha256']=ck.file_digest(path)
    manifest['identity_digest']=ck.digest(payload['identity'])
    path.with_suffix('.json').write_bytes(canonical_json(manifest))


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('workers',[0,2])
@pytest.mark.parametrize('generator',[False,True])
def test_epoch_boundary_next_and_multistep_equivalence(synthetic,architecture,workers,generator):
    cfg=cfg_for(architecture);seed();original=parts(cfg,workers,generator=generator)
    p=UpdateProgress(5,auxiliary=architecture=='resnet50');epoch(original,p)
    path=synthetic/'checkpoint-test/checkpoints/ckpt_last.pth'
    save(original,p,path)
    saved_center=copy.deepcopy(original['criterion'].center_loss.state_dict()) if original['auxiliary'] else None
    result=epoch(original,p,2)
    seed(991);resumed=parts(cfg,workers,generator=generator)
    center_ids=[id(x) for group in resumed['auxiliary'].param_groups for x in group['params']] if resumed['auxiliary'] else []
    restored,position,kind=ck.restore_lodo_checkpoint(path,**resumed)
    if saved_center:
        assert_equal(saved_center,resumed['criterion'].center_loss.state_dict())
        assert any(t.abs().sum()>0 for t in saved_center.values())
        assert center_ids==[id(x) for x in resumed['criterion'].center_loss.parameters()]
    resumed_result=epoch(resumed,restored,position['next_epoch'])
    assert original['criterion'].loss_trace[3:]==resumed['criterion'].loss_trace
    assert result.mean_loss==resumed_result.mean_loss
    assert result.mean_loss_terms==resumed_result.mean_loss_terms
    for key in ('model','optimizer','scheduler'):
        assert_equal(original[key].state_dict(),resumed[key].state_dict())
    if original['auxiliary']:
        assert_equal(original['criterion'].center_loss.state_dict(),resumed['criterion'].center_loss.state_dict())
        assert_equal(original['auxiliary'].state_dict(),resumed['auxiliary'].state_dict())
    assert_equal(original['model'].inputs[3:],resumed['model'].inputs)
    assert_equal(list(p.index_history),list(restored.index_history))
    for name in ('presentations','identity_presentations','identity_slots','batch_counts','distinct'):
        assert_equal(getattr(p.exposure,name),getattr(restored.exposure,name))
    assert restored.completed_primary_updates==5 and restored.auxiliary_steps==(5 if architecture=='resnet50' else 0)
    assert position['consumed_cursor']==0 and kind=='recovery'


@pytest.mark.parametrize('mutation',[
    lambda p:p['identity'].update(map_digest='0'*64),
    lambda p:p['identity'].update(num_classes=50),
    lambda p:p['identity'].update(feature_dim=2047),
    lambda p:p['identity']['protocol'].update(fold='LODO-2'),
    lambda p:p['identity'].update(config_digest='0'*64),
    lambda p:p['identity']['center'].update(feature_branch='feat_bn'),
    lambda p:p['identity']['center'].update(reduction='sum'),
    lambda p:p['identity']['center'].update(weight=1),
    lambda p:p['identity']['recovery'].update(capability='mid_epoch_exact'),
    lambda p:p.update(center=None),
    lambda p:p['scheduler'].update(warmup_iters=499),
    lambda p:p['optimizer']['param_groups'][0].update(weight_decay=1),
    lambda p:p['center_optimizer']['param_groups'][0].update(lr=.1),
    lambda p:p['position'].update(consumed_cursor=1),
    lambda p:p['progress'].update(consistent=False),
    lambda p:p['progress']['exposure']['presentations']['successful'].update(duke=999),
    lambda p:p['rng'].update(loader_generator=torch.Generator().get_state()),
])
def test_incompatible_payload_rejected_before_model_or_center_mutation(synthetic,mutation):
    cfg=cfg_for();seed();original=parts(cfg);p=UpdateProgress(5,auxiliary=True);epoch(original,p)
    path=synthetic/'negative/checkpoints/ckpt_last.pth';save(original,p,path);rewrite(path,mutation)
    fresh=parts(cfg);model=copy.deepcopy(fresh['model'].state_dict());center=copy.deepcopy(fresh['criterion'].center_loss.state_dict())
    with pytest.raises(ValueError): ck.restore_lodo_checkpoint(path,**fresh)
    assert_equal(model,fresh['model'].state_dict());assert_equal(center,fresh['criterion'].center_loss.state_dict())


@pytest.mark.parametrize('architecture,completed',[
    ('resnet50',n) for n in (0,1,499,500,501,28400,28401,49700,49701,85200)
]+[('pcb',n) for n in (0,31400,31401,94200)])
def test_scheduler_restore_actual_pre_step_lr(synthetic,architecture,completed):
    cfg=cfg_for(architecture);model=FixtureModel(cfg,20);opt=build_optimizer(cfg,model);sched=build_scheduler(cfg,opt)
    sched.prepare(completed);opt_state=copy.deepcopy(opt.state_dict());sched_state=sched.state_dict()
    fresh_model=FixtureModel(cfg,20);fresh_opt=build_optimizer(cfg,fresh_model);fresh_sched=build_scheduler(cfg,fresh_opt)
    fresh_opt.load_state_dict(opt_state);fresh_sched.load_state_dict(sched_state)
    called=[];fresh_opt.register_step_pre_hook(lambda optimizer,a,k: called.append(tuple(g['lr'] for g in optimizer.param_groups)))
    fresh_opt.step()
    if architecture=='resnet50':
        expected=.0003*(.01+.99*completed/500 if completed<500 else 1)*(.1**sum(completed>=x for x in (28400,49700)))
        assert called[0]==pytest.approx([expected]*len(called[0]),rel=1e-14)
    else:
        assert called[0][0]==pytest.approx(.001 if completed>=31400 else .01)
        assert max(called[0])/min(called[0])==pytest.approx(10)
    assert fresh_sched.completed_updates==completed
    assert_equal(sched_state,fresh_sched.state_dict())


def target_guards(monkeypatch):
    def prohibited(*a,**kw): pytest.fail('Target evaluation access during LODO training')
    monkeypatch.setattr(train,'build_test_loader',prohibited)
    import reid.engine.evaluator as evaluator
    from reid.data import build
    monkeypatch.setattr(evaluator,'evaluate_reid',prohibited)
    monkeypatch.setattr(build,'build_test_loader',prohibited)
    for name in ('market1501','duke','cuhk03','msmt17'):
        monkeypatch.setattr(build,f'_build_{name}_test_dataset',prohibited)


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('workers',[0,2])
def test_target_free_mid_epoch_rollback_and_terminal_zero_update(synthetic,monkeypatch,architecture,workers):
    target_guards(monkeypatch)
    models=[]
    def build_loader(cfg):
        ds=StochasticDataset();mode='pk' if architecture=='resnet50' else 'random'
        base=loader_for(ds,mode,workers);base.effective_batch_size=64
        return ObservedTrainLoader(base,sampler_mode=mode),ds.num_classes
    def model_builder(cfg,num_classes,**kwargs):
        m=FixtureModel(cfg,num_classes);models.append(m);return m
    monkeypatch.setattr(train,'build_train_loader',build_loader)
    monkeypatch.setattr(train,'build_model',model_builder)
    reference_cfg=cfg_for(architecture,run='reference');reference_cfg['data']['num_workers']=workers
    reference=train.run_lodo_training(reference_cfg);reference_model=models[-1]
    cfg=cfg_for(architecture,run='interrupted');cfg['data']['num_workers']=workers
    def interrupted_builder(cfg,num_classes,**kwargs):
        m=model_builder(cfg,num_classes,**kwargs);m.fail_at=5;return m
    monkeypatch.setattr(train,'build_model',interrupted_builder)
    with pytest.raises(UpdateFailure): train.run_lodo_training(cfg)
    run=synthetic/cfg['experiment']['output_dir'];last=run/'checkpoints/ckpt_last.pth'
    assert ck.read_checkpoint(last)[0]['progress']['completed_primary_updates']==3
    assert not (run/'checkpoints/ckpt_terminal.pth').exists()
    cfg['train']['save']['resume']=str(last)
    monkeypatch.setattr(train,'build_model',model_builder)
    recovered=train.run_lodo_training(cfg);recovered_model=models[-1]
    assert_equal(reference_model.state_dict(),recovered_model.state_dict())
    assert_equal(reference_model.inputs[3:],recovered_model.inputs)
    assert_equal(list(reference.index_history),list(recovered.index_history))
    assert_equal(reference.exposure.presentations,recovered.exposure.presentations)
    terminal=run/'checkpoints/ckpt_terminal.pth';payload,manifest=ck.read_checkpoint(terminal)
    assert manifest['completion_reason']=='budget_completed' and manifest['actual_primary_updates']==5
    assert manifest['actual_auxiliary_updates']==(5 if architecture=='resnet50' else 0)
    assert manifest['checkpoint_sha256']==ck.file_digest(terminal)
    assert payload['position']['consumed_cursor']==2 and payload['kind']=='terminal'
    events=json.loads((run/'recovery_events.json').read_text())
    assert events[0]['discarded_attempted_batches']==2 and events[0]['discarded_successful_updates']==1
    assert sum(events[0]['discarded_source_presentations']['attempted'].values())==128
    assert sum(events[0]['discarded_source_presentations']['successful'].values())==64
    assert events[1]['event']=='rollback_resume'
    before=terminal.read_bytes();already=train.run_lodo_training(cfg)
    assert already.completed_primary_updates==5 and models[-1].inputs==[] and terminal.read_bytes()==before
    assert not (run/'checkpoints/ckpt_best.pth').exists()


def test_atomic_failure_preserves_previous_checkpoint_and_corruption_rejected(synthetic,monkeypatch):
    cfg=cfg_for();seed();o=parts(cfg);p=UpdateProgress(5,auxiliary=True);epoch(o,p)
    path=synthetic/'atomic/checkpoints/ckpt_last.pth';save(o,p,path);before=path.read_bytes()
    def failed(*a): raise OSError('simulated interrupted rename')
    with monkeypatch.context() as patch:
        patch.setattr(ck.os,'replace',failed)
        with pytest.raises(OSError): save(o,p,path)
    assert path.read_bytes()==before;ck.read_checkpoint(path)
    assert not list(path.parent.glob('*.tmp')) and not list(path.parent.glob('.*.tmp'))
    path.write_bytes(before[:-31])
    with pytest.raises(ValueError): ck.restore_lodo_checkpoint(path,**o)


def test_partial_main_center_failure_cannot_checkpoint(synthetic):
    cfg=cfg_for();o=parts(cfg);p=UpdateProgress(5,auxiliary=True)
    o['auxiliary'].step=lambda: (_ for _ in ()).throw(RuntimeError('Center failed'))
    with pytest.raises(UpdateFailure): epoch(o,p)
    assert not p.consistent and p.completed_primary_updates==0 and o['scheduler'].completed_updates==0
    with pytest.raises(ValueError): save(o,p,synthetic/'bad/checkpoints/ckpt_last.pth')
    with pytest.raises(ValueError): save(o,p,synthetic/'bad/checkpoints/ckpt_terminal.pth',kind='terminal')


def test_unsupported_workers_mid_epoch_and_legacy_resume_rejected(synthetic):
    cfg=cfg_for();o=parts(cfg);p=UpdateProgress(5,auxiliary=True);epoch(o,p)
    with pytest.raises(ValueError,match='mid-epoch'): save(o,p,synthetic/'mid/checkpoints/ckpt_last.pth',cursor=1)
    o['loader']._loader.num_workers=1
    with pytest.raises(ValueError,match='0/2'): ck.loader_contract(o['loader'])
    legacy=synthetic/'legacy.pth';torch.save(dict(model=o['model'].state_dict(),center_optimizer=o['auxiliary'].state_dict()),legacy)
    with pytest.raises(ValueError): ck.restore_lodo_checkpoint(legacy,**o)


def test_same_run_collision_exception_requires_integrity_and_config(synthetic):
    cfg=cfg_for();seed();o=parts(cfg);p=UpdateProgress(5,auxiliary=True);p.bind(o['loader'])
    run=synthetic/cfg['experiment']['output_dir'];path=run/'checkpoints/ckpt_last.pth';save(o,p,path,next_epoch=1)
    with pytest.raises(ValueError): protocol.validate_output_path(cfg)
    cfg['train']['save']['resume']=str(path)
    assert protocol.validate_output_path(cfg)==run
    cfg['protocol']['fold']='LODO-2'
    with pytest.raises(ValueError): protocol.validate_output_path(cfg)


def test_script_dispatches_before_target_or_legacy_artifact_setup(synthetic,monkeypatch):
    cfg=cfg_for();target_guards(monkeypatch)
    monkeypatch.setattr('reid.utils.lodo_operational.validate_training_admission', lambda *a, **k: None)
    monkeypatch.setattr(train,'parse_args',lambda:SimpleNamespace(config='unused',opts=[]))
    monkeypatch.setattr(train,'load_config',lambda *a,**kw:cfg)
    monkeypatch.setattr(train,'create_experiment_dirs',lambda *a:pytest.fail('Legacy output setup called'))
    sentinel=object();monkeypatch.setattr(train,'run_lodo_training',lambda actual:sentinel)
    assert train.main() is sentinel


def test_phase4_adapter_checkpoint_integration(tree,synthetic,monkeypatch):
    from reid.data import build
    from torchvision.transforms import ToTensor
    monkeypatch.setattr(build,'build_train_tf',lambda **kw:ToTensor())
    cfg=config(tree);cfg['train']['max_updates']=2
    loader,classes=build.build_train_loader(cfg)
    model=FixtureModel(cfg,classes);criterion=build_criterion(cfg,classes,2048)
    opt=build_optimizer(cfg,model);aux=build_center_optimizer(cfg,criterion);sched=build_scheduler(cfg,opt)
    o=dict(cfg=cfg,loader=loader,model=model,criterion=criterion,optimizer=opt,auxiliary=aux,scheduler=sched)
    p=UpdateProgress(2,auxiliary=True);epoch(o,p)
    path=synthetic/'actual-adapter/checkpoints/ckpt_last.pth';save(o,p,path)
    restored,_,_=ck.restore_lodo_checkpoint(path,**o)
    assert restored.completed_primary_updates==1 and restored.exposure.mapping_digest==loader.dataset.mapping_digest


@pytest.mark.parametrize('field,value',[('completion_reason','interrupted'),('actual_auxiliary_updates',0),
    ('planned_primary_updates',999),('map_digest','0'*64),('config_digest','0'*64),('data_digest','0'*64)])
def test_manifest_fields_cannot_fabricate_completion(synthetic,field,value):
    cfg=cfg_for();o=parts(cfg);p=UpdateProgress(5,auxiliary=True);epoch(o,p)
    path=synthetic/'manifest/checkpoints/ckpt_last.pth';save(o,p,path)
    manifest=json.loads(path.with_suffix('.json').read_text());manifest[field]=value
    path.with_suffix('.json').write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError): ck.read_checkpoint(path)


def test_two_file_publication_interruption_fails_closed(synthetic,monkeypatch):
    cfg=cfg_for(budget=1);o=parts(cfg);p=UpdateProgress(1,auxiliary=True);epoch(o,p)
    path=synthetic/'unpublished/checkpoints/ckpt_terminal.pth'
    original_replace=ck.os.replace
    def interrupted_sidecar(src,dst):
        if str(dst).endswith('.json'): raise OSError('synthetic interruption before sidecar commit')
        return original_replace(src,dst)
    monkeypatch.setattr(ck.os,'replace',interrupted_sidecar)
    with pytest.raises(OSError): save(o,p,path,kind='terminal',cursor=1)
    # A payload alone is uncommitted; no reader may treat its filename as completion.
    assert not path.with_suffix('.json').exists()
    with pytest.raises(ValueError): ck.read_checkpoint(path)


def test_center_parameter_ownership_and_effective_branch_checked(synthetic):
    cfg=cfg_for();o=parts(cfg)
    o['criterion'].metric_feat_key='feat_bn'
    with pytest.raises(ValueError): ck.make_identity(**o)
    o['criterion'].metric_feat_key='feat_raw'
    o['auxiliary'].param_groups[0]['params']=[nn.Parameter(torch.zeros(49,2048))]
    with pytest.raises(ValueError): ck.make_identity(**o)


def test_missing_center_optimizer_and_nonfinite_optimizer_state_rejected(synthetic):
    cfg=cfg_for();o=parts(cfg);p=UpdateProgress(5,auxiliary=True);epoch(o,p)
    path=synthetic/'missing/checkpoints/ckpt_last.pth';save(o,p,path)
    rewrite(path,lambda payload:payload.update(center_optimizer=None))
    with pytest.raises(ValueError): ck.restore_lodo_checkpoint(path,**parts(cfg))
    path2=synthetic/'nonfinite/checkpoints/ckpt_last.pth';save(o,p,path2)
    def poison(payload):
        next(iter(payload['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
    rewrite(path2,poison)
    fresh=parts(cfg);before=copy.deepcopy(fresh['model'].state_dict())
    with pytest.raises(ValueError): ck.restore_lodo_checkpoint(path2,**fresh)
    assert_equal(before,fresh['model'].state_dict())


def test_terminal_requires_exact_budget_and_unrelated_path_is_preserved(synthetic):
    cfg=cfg_for();o=parts(cfg);p=UpdateProgress(5,auxiliary=True);epoch(o,p)
    terminal=synthetic/'early/checkpoints/ckpt_terminal.pth'
    with pytest.raises(ValueError,match='exact budget'): save(o,p,terminal,kind='terminal')
    assert not terminal.exists()
    last=synthetic/'unrelated/checkpoints/ckpt_last.pth';last.parent.mkdir(parents=True);last.write_bytes(b'Stage A artifact')
    with pytest.raises(ValueError): save(o,p,last)
    assert last.read_bytes()==b'Stage A artifact'


def test_active_prefetch_iterator_cannot_be_checkpointed(synthetic):
    cfg=cfg_for();o=parts(cfg,workers=2);p=UpdateProgress(5,auxiliary=True);p.bind(o['loader'])
    iterator=iter(o['loader']);next(iterator)
    try:
        with pytest.raises(ValueError,match='active'): save(o,p,synthetic/'active/checkpoints/ckpt_last.pth',next_epoch=1)
    finally: iterator.close()


def test_stage_a_main_still_builds_target_and_selects_metric_best(synthetic,monkeypatch):
    from reid.utils.config import load_config
    from test_lodo_config import ROOT
    cfg=load_config(str(ROOT/'configs/baseline_market1501_resnet50_triplet.yaml'))
    cfg['system'].update(device='cpu',amp=False)
    cfg['logging'].update(tensorboard=False)
    cfg['experiment']['output_dir']=str(synthetic/'stage-a-fixture')
    cfg['train'].update(epochs=2,eval_interval=1)
    cfg['train']['save'].update(save_best=True,save_last=True,metric='mAP',resume='')
    class Loader(list): pass
    loader=Loader([None,None]);loader.dataset=SimpleNamespace(num_classes=4);loader.effective_batch_size=64
    model=nn.Linear(2,3);model.feat_dim=3
    monkeypatch.setattr(train,'parse_args',lambda:SimpleNamespace(config='unused',opts=[]))
    monkeypatch.setattr(train,'load_config',lambda *a,**k:copy.deepcopy(cfg))
    monkeypatch.setattr(train,'build_train_loader',lambda cfg:(loader,4))
    targets=[];target=object()
    monkeypatch.setattr(train,'build_test_loader',lambda cfg:targets.append(cfg['data']['test']['dataset']['name']) or target)
    monkeypatch.setattr(train,'build_model',lambda *a,**k:model)
    monkeypatch.setattr(train,'train_one_epoch',lambda **k:.25)
    monkeypatch.setattr(train,'save_run_artifacts',lambda *a,**k:dict(command='synthetic',environment='synthetic'))
    monkeypatch.setattr(train,'save_training_plots',lambda **k:None)
    evaluations=[]
    def evaluate(cfg,model,loader,device,**kwargs):
        assert loader is target
        value=[.8,.3,.8][len(evaluations)];evaluations.append(value)
        return dict(mAP=value,mINP=.2,Rank1=.4,Rank5=.6,Rank10=.7,cmc=[.4,.6,.7])
    import reid.engine.evaluator as evaluator
    monkeypatch.setattr(evaluator,'evaluate_reid',evaluate)
    train.main()
    run=Path(cfg['experiment']['output_dir'])
    best=torch.load(run/'checkpoints/ckpt_best.pth',map_location='cpu',weights_only=False)
    last=torch.load(run/'checkpoints/ckpt_last.pth',map_location='cpu',weights_only=False)
    assert targets==['market1501'] and evaluations==[.8,.3,.8]
    assert best['epoch']==1 and best['scores']['mAP']==.8 and last['epoch']==2
    assert not (run/'checkpoints/ckpt_terminal.pth').exists()


def test_unsupported_gpu_or_auto_mode_rejected_before_device_setup(synthetic,monkeypatch):
    monkeypatch.setattr(train,'select_device',lambda *a,**k:pytest.fail('Unsupported device setup accessed'))
    cfg=cfg_for()
    for value in ('cuda','auto'):
        cfg['system']['device']=value
        with pytest.raises(ValueError,match='CPU-validated'): train.run_lodo_training(cfg)
