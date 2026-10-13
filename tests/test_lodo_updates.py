"""Phase 5 synthetic CPU engine traces; no launch script or real-image training."""
import copy
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torchvision.transforms import ToTensor

from test_lodo_config import lodo_config
from test_multi_source_dataset import tree, config
from reid.data.build import build_train_loader
from reid.data.collate import train_collate_fn
from reid.data.index_observation import ObservedTrainLoader
from reid.data.multi_source import build_multi_source_dataset, SourceIndex
from reid.data.protocol import ReIDTrainSample
from reid.data.samplers import PKBatchSampler
from reid.engine.train_loop import train_one_epoch
from reid.engine.update_control import UpdateProgress, UpdateFailure, ExposureAccounting
from reid.losses.build import build_criterion
from reid.optim.build import build_optimizer, build_center_optimizer, build_scheduler


class IndexedToyDataset:
    """Synthetic sidecar test double; real Phase 4 adapters tested separately."""
    def __init__(self, n=130, *, stochastic=False, few=False):
        self.stochastic = stochastic
        self.labels = [i//4 for i in range(n)] if not few else [0]*2 + [i//4+1 for i in range(n-2)]
        self.num_classes = max(self.labels)+1
        self.mapping_digest = 'synthetic'
        self.sidecar = []
        self.sources = {}
        self._identities = []
        for global_label in sorted(set(self.labels)):
            source_key = ('duke','cuhk03','msmt17')[min(2,global_label*3//self.num_classes)]
            source = self.sources.setdefault(source_key, SimpleNamespace(labels=[],pids=[],cams=[],im_names=[],im_dir='/synthetic'))
            local_label = sum(r['source_key']==source_key for r in self._identities)
            self._identities.append(dict(global_label=global_label, source_key=source_key,
                                         source_exposed_pid=global_label,local_label=local_label))
        identity={r['global_label']:r for r in self._identities}
        for i,label in enumerate(self.labels):
            r=identity[label];source=self.sources[r['source_key']];local=len(source.labels)
            source.labels.append(r['local_label']);source.pids.append(label);source.cams.append(1);source.im_names.append(str(i))
            self.sidecar.append(SourceIndex(i,r['source_key'],local,label,r['local_label'],label,1,str(i),f'/synthetic/{i}'))
        # SimpleNamespace doesn't have __len__; source indexing validation uses it.
        self.sources={k:ToySource(v) for k,v in self.sources.items()}
        self.sidecar=tuple(self.sidecar)
        self.reads=[]

    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        self.reads.append(i)
        noise=torch.rand(()) if self.stochastic else torch.tensor(0.)
        return ReIDTrainSample(torch.tensor([float(i),float(i%5)])+noise,self.labels[i])
    def resolve_index(self,i): return self.sidecar[i]
    def mapping_metadata(self): return dict(identities=copy.deepcopy(self._identities))


class ToySource:
    def __init__(self, source): self.__dict__.update(source.__dict__)
    def __len__(self): return len(self.labels)


class TinyModel(nn.Module):
    def __init__(self, classes=33):
        super().__init__();self.backbone=nn.Linear(2,3);self.head=nn.Linear(3,classes)
        self.forward_calls=0
    def forward(self,x):
        self.forward_calls+=1
        if x.ndim==4: x=x.mean(dim=(2,3))[:,:2]
        z=self.backbone(x/100)
        return dict(emb=z,feat_raw=z,feat_bn=z,logits=self.head(z))


def loader_for(ds,mode='random',workers=0,batch_size=64):
    kwargs=dict(num_workers=workers,collate_fn=train_collate_fn)
    if workers: kwargs['prefetch_factor']=2
    if mode=='pk':
        return DataLoader(ds,batch_sampler=PKBatchSampler(ds.labels,P=16,K=4,seed=42),**kwargs)
    return DataLoader(ds,batch_size=batch_size,shuffle=True,drop_last=True,**kwargs)


def setup(*, budget=1,center=False,mode='random',n=130,batch_size=64,workers=0,ds=None):
    ds=IndexedToyDataset(n) if ds is None else ds
    loader=ObservedTrainLoader(loader_for(ds,mode,workers,batch_size),sampler_mode=mode)
    model=TinyModel(ds.num_classes)
    cfg=lodo_config('resnet50' if center or mode=='pk' else 'pcb')
    criterion=build_criterion(cfg,num_classes=ds.num_classes,feat_dim=3)
    # Tests use the approved optimizer/loss recipe on a deliberately tiny model.
    optimizer=build_optimizer(cfg,model)
    aux=build_center_optimizer(cfg,criterion)
    scheduler=build_scheduler(cfg,optimizer,steps_per_epoch=999)
    progress=UpdateProgress(budget,auxiliary=aux is not None)
    return model,loader,criterion,optimizer,aux,scheduler,progress


def run_epoch(parts, epoch=1, **kwargs):
    model,loader,criterion,opt,aux,scheduler,progress=parts
    return train_one_epoch(model,loader,criterion,opt,aux,torch.device('cpu'),False,1000,
                           scheduler=scheduler,epoch=epoch,update_progress=progress,**kwargs)


@pytest.mark.parametrize('mode',['pk','random'])
@pytest.mark.parametrize('workers',[0,2])
def test_observation_preserves_full_index_transform_and_rng_sequences(mode,workers):
    ds=IndexedToyDataset(194,stochastic=True,few=mode=='pk')
    torch.manual_seed(101)
    base=loader_for(ds,mode,workers)
    plain=list(base);plain_rng=torch.get_rng_state().clone()
    torch.manual_seed(101)
    original=loader_for(ds,mode,workers)
    pk_python=copy.deepcopy(original.batch_sampler.rng.getstate()) if mode=='pk' else None
    wrapped=ObservedTrainLoader(original,sampler_mode=mode)
    if mode=='pk': assert wrapped.batch_sampler.rng.getstate()==pk_python
    actual=[];indices=[]
    for batch in wrapped:
        actual.append(batch);indices.append(wrapped.latest_indices)
    assert len(actual)==len(plain)
    for (a,la),(b,lb),idx in zip(actual,plain,indices):
        assert torch.equal(a,b) and torch.equal(la,lb)
        # First feature identifies each consumed index despite small transform noise.
        assert tuple(a[:,0].floor().long().tolist())==idx
    assert torch.equal(torch.get_rng_state(),plain_rng)
    assert wrapped.observation()['unconsumed_scheduled_images']==0
    if mode=='random': assert len(actual)*64==192  # tail dropped unchanged
    else: assert all(sorted(Counter(ds.labels[i] for i in batch).values())==[4]*16 for batch in indices)


def test_pk_variable_epochs_replacement_and_no_observation_rng_draws():
    labels=[0]*2+sum(([pid]*count for pid,count in enumerate([4,8,12,16,20],start=1)),[])
    plain=PKBatchSampler(labels,P=3,K=4,seed=4)
    wrapped=copy.deepcopy(plain)
    from reid.data.index_observation import _ObservedBatchSampler
    observer=_ObservedBatchSampler(wrapped)
    lengths=[];repeated=False
    for _ in range(12):
        expected=list(plain);actual=list(observer)
        assert actual==expected
        assert wrapped.rng.getstate()==plain.rng.getstate()
        assert wrapped.np_rng.bit_generator.state==plain.np_rng.bit_generator.state
        lengths.append(len(actual))
        repeated |= any(len(set(b))<len(b) for b in actual)
        observer.pending.clear()
    assert len(set(lengths))>1 and repeated


@pytest.mark.parametrize('budget',[1,2,3,5])
@pytest.mark.parametrize('center',[False,True])
def test_exact_budgets_across_partial_and_full_epochs(budget,center):
    parts=setup(budget=budget,center=center,mode='pk' if center else 'random')
    model,loader,criterion,opt,aux,scheduler,progress=parts
    calls=[];opt.register_step_pre_hook(lambda optimizer,a,k:calls.append(tuple(g['lr'] for g in optimizer.param_groups)))
    results=[];epoch=1
    while not progress.budget_completed:
        results.append(run_epoch(parts,epoch));epoch+=1
    assert len(calls)==budget==progress.completed_primary_updates==model.forward_calls
    assert progress.auxiliary_steps==(budget if center else 0)
    assert scheduler.completed_updates==budget
    assert results[-1].stop_reason=='budget_completed'
    assert progress.attempted_batches==progress.forward_batches==progress.backward_batches==budget
    assert sum(progress.exposure.presentations['successful'].values())==budget*64
    assert sum(r.successful_images for r in results)==budget*64
    if not center:
        assert [r.stop_reason for r in results[:-1]]==['epoch_exhausted']*(len(results)-1)
        assert results[-1].partial_epoch==(budget%2!=0)
    old_reads=list(loader.dataset.reads)
    result=run_epoch(parts,epoch)
    assert result.successful_updates==0 and result.stop_reason=='budget_completed'
    assert loader.dataset.reads==old_reads and len(calls)==budget
    progress.check_totals()


@pytest.mark.parametrize('budget',[0,-1,True,1.5])
def test_invalid_update_budgets_rejected(budget):
    with pytest.raises(ValueError): UpdateProgress(budget)


def test_partial_loss_uses_processed_batches_and_reports_raw_weighted_terms():
    parts=setup(budget=1,center=True,mode='pk',n=194)
    result=run_epoch(parts)
    assert result.partial_epoch and result.loss_batches==result.forward_batches==1
    assert result.mean_loss==pytest.approx(sum(result.mean_weighted_loss_terms.values()))
    assert result.mean_weighted_loss_terms['loss/center']==pytest.approx(.0005*result.mean_loss_terms['loss/center'])
    assert result.mean_weighted_loss_terms['loss/id']==result.mean_loss_terms['loss/id']


@pytest.mark.parametrize('failure',['loss','gradient','aux_rescale','main_step','center_step','post_step_nonfinite'])
def test_fail_closed_counts_and_scheduler(failure):
    parts=list(setup(center=True,mode='pk'))
    model,loader,criterion,opt,aux,scheduler,progress=parts
    if failure=='loss':
        original=criterion.forward
        criterion.forward=lambda outputs,labels: (original(outputs,labels)[0]*float('inf'),{})
    elif failure=='gradient':
        next(model.parameters()).register_hook(lambda grad: grad*float('inf'))
    elif failure=='aux_rescale':
        criterion.prepare_auxiliary_optimizer_step=lambda: (next(criterion.center_loss.parameters()).grad.fill_(float('inf')) is not None)
    elif failure=='main_step':
        opt.step=lambda: (_ for _ in ()).throw(RuntimeError('main failed'))
    elif failure=='center_step':
        aux.step=lambda: (_ for _ in ()).throw(RuntimeError('center failed'))
    else:
        original=opt.step
        def invalid():
            original()
            with torch.no_grad(): next(model.parameters()).fill_(float('inf'))
        opt.step=invalid
    with pytest.raises(UpdateFailure) as error: run_epoch(parts)
    assert error.value.progress is progress
    assert progress.failed and not progress.budget_completed
    assert progress.completed_primary_updates==0 and progress.auxiliary_steps==0
    assert scheduler.completed_updates==0
    assert progress.attempted_batches==progress.forward_batches==1
    assert progress.backward_batches==(0 if failure=='loss' else 1)
    assert sum(progress.exposure.presentations['attempted'].values())==64
    assert sum(progress.exposure.presentations['successful'].values())==0
    assert progress.consistent==(failure in ['loss','gradient','aux_rescale'])
    assert progress.epoch_results[-1].stop_reason=='failed'
    with pytest.raises(ValueError,match='cannot continue'): run_epoch(parts)


def test_failed_second_batch_separates_attempted_successful_exposure():
    parts=setup(budget=2);model,loader,criterion,opt,aux,scheduler,progress=parts
    original=criterion.forward;count=0
    def second_bad(outputs,labels):
        nonlocal count
        count+=1
        loss,logs=original(outputs,labels)
        return (loss*float('inf') if count==2 else loss),logs
    criterion.forward=second_bad
    with pytest.raises(UpdateFailure): run_epoch(parts)
    assert progress.completed_primary_updates==scheduler.completed_updates==1
    summary=progress.exposure.summary()
    assert summary['attempted']['images']==summary['forward']['images']==128
    assert summary['backward']['images']==summary['successful']['images']==64
    progress.check_totals()


@pytest.mark.parametrize('workers',[0,2])
def test_unused_prefetch_is_not_model_processed(workers):
    parts=setup(budget=1,workers=workers,n=514)
    result=run_epoch(parts);loader=parts[1];progress=parts[-1]
    obs=result.observation
    assert progress.forward_batches==progress.completed_primary_updates==1
    assert progress.exposure.summary()['forward']['images']==64
    assert obs['consumed_images']==64
    assert obs['scheduled_images']==obs['consumed_images']+obs['unconsumed_scheduled_images']
    if workers: assert obs['unconsumed_scheduled_images']>0
    else: assert obs['scheduled_images']==64 and len(loader.dataset.reads)==64


@pytest.mark.parametrize('field,value',[('global_index',999),('source_local_index',999),('global_label',999),('source_exposed_pid',999),('image_reference','bad'),('canonical_path','/wrong/image')])
def test_corrupted_sidecar_fails_before_forward(field,value):
    parts=setup();ds=parts[1].dataset
    ds.sidecar=tuple(replace(row,**{field:value}) for row in ds.sidecar)
    with pytest.raises(UpdateFailure): run_epoch(parts)
    progress=parts[-1]
    assert progress.forward_batches==parts[0].forward_calls==0
    assert progress.completed_primary_updates==0 and progress.unattributed_attempted_batches==1
    progress.check_totals()


def test_missing_consumed_indices_fails_closed():
    parts=setup();loader=parts[1]
    original=loader.__class__.__iter__
    # Test double clears only observation; public batch still has the same fields.
    def missing(self):
        for batch in original(self):
            self.latest_indices=None
            yield batch
    from unittest.mock import patch
    with patch.object(type(loader),'__iter__',missing):
        with pytest.raises(UpdateFailure): run_epoch(parts)
    assert parts[-1].forward_batches==0


def test_per_source_slots_distinct_identities_and_actual_small_batch_size():
    parts=setup(mode='pk',center=True,n=194)
    run_epoch(parts);summary=parts[-1].exposure.summary()['successful']
    assert sum(summary['per_source_identity_slots'].values())==16
    assert len(summary['distinct_global_identities'])==16
    assert sum(summary['per_source_images'].values())==64
    assert sum(summary['per_source_percent'].values())==pytest.approx(100)
    for source,slots in summary['per_source_identity_slots'].items(): assert summary['per_source_images'][source]==4*slots
    small=setup(n=26,batch_size=7)
    run_epoch(small)
    assert small[-1].exposure.summary()['successful']['images']==7
    assert small[-1].exposure.summary()['successful']['per_source_identity_slots'] is None


@pytest.mark.parametrize('architecture,budget,points',[('resnet50',85200,[1,2,499,500,501,28400,28401,49700,49701,85200]),('pcb',94200,[1,31400,31401,94200])])
def test_complete_actual_step_used_lr_trace(architecture,budget,points):
    cfg=lodo_config(architecture)
    model=TinyModel(24);optimizer=build_optimizer(cfg,model)
    scheduler=build_scheduler(cfg,optimizer,steps_per_epoch=1)
    used=[];stored=[]
    optimizer.register_step_pre_hook(lambda opt,a,k:used.append(tuple(g['lr'] for g in opt.param_groups)))
    for _ in range(budget):
        optimizer.step();scheduler.step();stored.append(tuple(g['lr'] for g in optimizer.param_groups))
    assert len(used)==scheduler.completed_updates==budget
    for update,actual in enumerate(used,start=1):
        if architecture=='resnet50':
            expected=.0003*(.01+.99*(update-1)/500) if update<=500 else .0003 if update<=28400 else .00003 if update<=49700 else .000003
            assert actual==pytest.approx([expected]*len(actual),rel=1e-14,abs=0)
        else:
            wanted=[(.01 if update<=31400 else .001) if g['param_name'].startswith('backbone.') else (.1 if update<=31400 else .01) for g in optimizer.param_groups]
            assert actual==pytest.approx(wanted,rel=1e-14,abs=0)
            assert actual[-1]/actual[0]==pytest.approx(10)
    for update in points: assert used[update-1]==pytest.approx(scheduler.rates_after(update-1),rel=1e-14,abs=0)
    for completed in [0,499,500,28400,49700] if architecture=='resnet50' else [0,31399,31400]:
        other=build_optimizer(cfg,TinyModel(24));continuation=build_scheduler(cfg,other,steps_per_epoch=9999)
        continuation.prepare(completed)
        captured=[];other.register_step_pre_hook(lambda opt,a,k:captured.append(tuple(g['lr'] for g in opt.param_groups)))
        other.step()
        assert captured[0]==used[completed]
    if architecture=='resnet50':
        assert used[499][0]==pytest.approx(.000299406)
        assert stored[499][0]==pytest.approx(.0003)
        assert stored[28399][0]==pytest.approx(.00003)
    else: assert stored[31399][0]==pytest.approx(.001)


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('steps',[None,0,1,997])
def test_update_schedule_independent_of_loader_length(architecture,steps):
    cfg=lodo_config(architecture);optimizer=build_optimizer(cfg,TinyModel())
    scheduler=build_scheduler(cfg,optimizer,steps_per_epoch=steps)
    assert scheduler.milestones==tuple(cfg['sched']['milestones'])
    assert scheduler.completed_updates==0


def test_center_actual_step_lr_and_noninvasive_weighted_gradients():
    parts=setup(center=True,mode='pk',budget=2)
    aux=parts[4];lrs=[]
    aux.register_step_pre_hook(lambda opt,a,k:lrs.append(tuple(g['lr'] for g in opt.param_groups)))
    while not parts[-1].budget_completed: run_epoch(parts)
    assert lrs==[(.5,),(.5,)]
    assert parts[-1].auxiliary_steps==parts[-1].completed_primary_updates==2


@pytest.mark.parametrize('fold',['LODO-1','LODO-2','LODO-3','LODO-4'])
@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_phase4_original_adapter_factory_integration(tree,fold,architecture,monkeypatch):
    cfg=config(tree,architecture,fold)
    # Use original image files and source loaders, with a cheap existing tensor transform.
    monkeypatch.setattr('reid.data.build.build_train_tf',lambda **kw:ToTensor())
    loader,classes=build_train_loader(cfg)
    model=TinyModel(classes);criterion=build_criterion(cfg,classes,3)
    opt=build_optimizer(cfg,model);aux=build_center_optimizer(cfg,criterion);scheduler=build_scheduler(cfg,opt,len(loader))
    progress=UpdateProgress(1,auxiliary=aux is not None)
    digest=loader.dataset.mapping_digest
    result=train_one_epoch(model,loader,criterion,opt,aux,torch.device('cpu'),False,1000,scheduler=scheduler,update_progress=progress)
    assert result.stop_reason=='budget_completed' and result.successful_images==64
    assert loader.dataset.mapping_digest==digest
    assert set(progress.exposure.summary()['successful']['per_source_images'])<=set(loader.dataset.sources)


def test_stage_a_return_contract_and_new_mode_rejects_amp():
    parts=setup();model,loader,criterion,opt,aux,scheduler,progress=parts
    plain=loader_for(loader.dataset)
    legacy_cfg={k:copy.deepcopy(lodo_config('pcb')[k]) for k in ('optim','sched')};legacy_cfg['sched'].pop('unit')
    legacy=build_scheduler(legacy_cfg,opt,steps_per_epoch=len(plain))
    value=train_one_epoch(model,plain,criterion,opt,aux,torch.device('cpu'),False,1000,scheduler=legacy)
    assert isinstance(value,float)
    fresh=setup()
    with pytest.raises(ValueError,match='FP32'):
        train_one_epoch(*fresh[:5],device=torch.device('cpu'),amp=True,log_interval=1,scheduler=fresh[5],update_progress=fresh[6])



def test_same_identity_image_index_reassignment_fails_closed():
    parts=setup();ds=parts[1].dataset
    # Reassign a global index to another image of the SAME identity, so labels
    # alone would miss this corruption. Original concatenation offsets reject it.
    changed=[]
    for row in ds.sidecar:
        source=ds.sources[row.source_key]
        other=(row.source_local_index+1) % len(source)
        if source.pids[other]==row.source_exposed_pid:
            changed.append(replace(row,source_local_index=other,image_reference=source.im_names[other],
                                   canonical_path='/synthetic/'+source.im_names[other]))
        else: changed.append(replace(row,global_index=999))
    ds.sidecar=tuple(changed)
    with pytest.raises(UpdateFailure): run_epoch(parts)
    assert parts[0].forward_calls==0


def test_step_used_and_next_rates_are_explicit_in_opt_in_logging():
    parts=setup(center=True,mode='pk')
    captured={}
    class Writer:
        def add_scalar(self,tag,value,global_step): captured[tag,global_step]=value
    model,loader,criterion,opt,aux,scheduler,progress=parts
    result=train_one_epoch(model,loader,criterion,opt,aux,torch.device('cpu'),False,1,
                           scheduler=scheduler,tb_writer=Writer(),update_progress=progress)
    assert captured['lr/used/group_0',1]==pytest.approx(.000003)
    assert captured['lr/next/group_0',1]==pytest.approx(.000003594)
    assert result.lr_used[0]==captured['lr/used/group_0',1]



def test_empty_epoch_fails_instead_of_looping_without_progress():
    parts=setup(n=2)
    with pytest.raises(UpdateFailure,match='no batches'): run_epoch(parts)
    assert parts[-1].failed and parts[-1].completed_primary_updates==0
    assert parts[-1].epoch_results[-1].stop_reason=='failed'


def test_opt_in_center_step_matches_original_manual_fp32_arithmetic():
    parts=setup(center=True,mode='pk')
    model,loader,criterion,opt,aux,scheduler,progress=parts
    manual_model=copy.deepcopy(model);manual_criterion=copy.deepcopy(criterion)
    cfg=lodo_config('resnet50')
    manual_opt=build_optimizer(cfg,manual_model);manual_aux=build_center_optimizer(cfg,manual_criterion)
    build_scheduler(cfg,manual_opt)
    images,labels=next(iter(loader_for(loader.dataset,'pk')))
    manual_opt.zero_grad(set_to_none=True);manual_aux.zero_grad(set_to_none=True)
    loss,_=manual_criterion(manual_model(images),labels);loss.backward()
    assert manual_criterion.prepare_auxiliary_optimizer_step()
    manual_opt.step();manual_aux.step()
    result=run_epoch(parts)
    assert result.mean_loss==float(loss.detach())
    for actual,expected in zip(model.parameters(),manual_model.parameters()): assert torch.equal(actual,expected)
    for actual,expected in zip(criterion.center_loss.parameters(),manual_criterion.center_loss.parameters()): assert torch.equal(actual,expected)
