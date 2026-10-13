"""NON-AUTHORITATIVE Phase 9R separate-process CUDA continuation harness.

Real source inventories, architectures, transforms, original PK/random decisions,
losses and optimizers. Test-only epochs exhaust a two-batch prefix of the original
sampler; only the test budget/output contracts are substituted. Production cannot
accept these short checkpoints. No real target access or official launch.
"""
import argparse
from contextlib import ExitStack,contextmanager
import copy
import gc
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

import torch

from test_lodo_config import lodo_config
from lodo_cuda_smoke import fixture_budget, require_production_rejection
from reid.utils import lodo_checkpoint as ck, lodo_protocol as protocol, lodo_recovery as recovery
from reid.utils.lodo_device import authorization_uuid,audit_gpu,validate_placement
from reid.utils import lodo_device as device_api
from reid.utils.checkpoint import reconstruct_model
from reid.data import build as data_build
from reid.engine import evaluator
from scripts import train

ROOT=Path(__file__).resolve().parents[1]
SMOKE=ROOT/'exp/lodo_smoke'
BUDGET=5
EPOCH_BATCHES=2


class ShortTestEpoch:
    """Exhaust a bounded test epoch without changing original sampler decisions."""
    def __init__(self,original):self.original=original
    def __iter__(self):yield from islice(self.original,EPOCH_BATCHES)
    def __len__(self):return EPOCH_BATCHES


def smoke_output(cfg,*,project_root=None):
    if project_root is not None and Path(project_root).resolve()!=ROOT:raise ValueError('Smoke root rebound')
    path=Path(cfg['experiment']['output_dir'])
    if path.resolve()!=path or SMOKE.resolve()!=SMOKE or '..' in path.parts:raise ValueError('Smoke alias/escape')
    try:
        parts=path.relative_to(SMOKE).parts
        assert len(parts)==4 and parts[0].startswith('phase9r-') and parts[1:3]==(cfg['protocol']['architecture'],cfg['protocol']['fold'])
        assert parts[3] in ('continuous','interrupted')
        assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',x) for x in parts)
    except (AssertionError,ValueError) as error:raise ValueError('Invalid recovery smoke namespace') from error
    if cfg['train']['save'].get('resume'):
        ck.validate_resume_path(cfg,path)
    elif path.exists():raise ValueError('Smoke namespace collision')
    return path


def equal_saved(actual,expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual.detach().cpu(),expected)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for k in expected:equal_saved(actual[k],expected[k])
    elif isinstance(expected,(tuple,list)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):equal_saved(a,b)
    else:assert actual==expected


def run(architecture,campaign,mode):
    if not re.fullmatch(r'phase9r-[A-Za-z0-9._-]+',campaign):raise ValueError('Invalid campaign')
    cfg=lodo_config(architecture,'LODO-1');cfg['system'].update(device='cuda',gpu_id=0,amp=False,log_interval=1000)
    cfg['data'].update(root='/home/addirakoze/nobackup/Projects/Dataset',num_workers=0)
    cfg['train']['recovery']=dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2)
    kind='continuous' if mode=='continuous' else 'interrupted'
    path=SMOKE/campaign/architecture/'LODO-1'/kind
    cfg['experiment'].update(name='NON-AUTHORITATIVE-CUDA-RECOVERY-SMOKE',output_dir=str(path))
    protocol.scientific_configuration(cfg)
    cfg['train']['max_updates']=BUDGET
    require_production_rejection(cfg)
    if mode=='resume':cfg['train']['save']['resume']=str(path/'checkpoints/ckpt_last.pth')
    uuid=authorization_uuid(cfg);admission=audit_gpu(uuid)
    with fixture_budget(BUDGET),patch.object(protocol,'validate_output_path',smoke_output):smoke_output(cfg)
    weights=Path(torch.hub.get_dir())/'checkpoints'/('resnet50-11ad3fa6.pth' if architecture=='resnet50' else 'resnet50-19c8e357.pth')
    if not weights.is_file():raise ValueError('Pretrained cache missing; no downloads permitted')
    intent=dict(non_authoritative=True,architecture=architecture,fold='LODO-1',mode=mode,budget=BUDGET,
                epoch_batches=EPOCH_BATCHES,uuid=uuid,admission=admission,pretrained_sha256=ck.file_digest(weights))
    objects={};losses=[];inputs=[];steps=[];counts=[];restore_checks=[];inspection_checks=[];pending={}
    original_loader=train.build_train_loader;original_model=train.build_model;original_criterion=train.build_criterion
    original_epoch=train.train_one_epoch;original_restore=ck.restore_lodo_checkpoint
    def loader(actual):
        value,classes=original_loader(actual);assert classes==2510 and value.num_workers==0 and len(value)>=EPOCH_BATCHES
        value._observer.original=ShortTestEpoch(value.batch_sampler)
        objects['loader']=value;return value,classes
    def model(actual,num_classes,**kwargs):
        value=original_model(actual,num_classes,**kwargs);objects['model']=value
        def before(module,args):
            x=args[0];assert x.device==torch.device('cuda:0') and x.dtype==torch.float32
            assert tuple(x.shape)==(64,3,*actual['data']['train']['images']['size'])
            inputs.append(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest())
            torch.cuda.synchronize();pending['start']=time.perf_counter()
        value.register_forward_pre_hook(before);return value
    def criterion(actual,num_classes,feat_dim):
        value=original_criterion(actual,num_classes,feat_dim);objects['criterion']=value
        def observe(module,args,result):
            outputs,labels=args;loss,logs=result
            assert labels.device==torch.device('cuda:0') and torch.isfinite(loss).item()
            losses.append(float(loss.detach().cpu()))
            if architecture=='resnet50':assert tuple(module.center_loss.centers.shape)==(2510,2048)
        value.register_forward_hook(observe);return value
    def restored(checkpoint,**kwargs):
        result=original_restore(checkpoint,**kwargs)
        if kwargs.get('validate_only'):
            assert not torch.cuda.is_initialized()
            assert all(p.device.type=='cpu' for p in kwargs['model'].parameters())
            inspection_checks.append(dict(cpu_first=True,cuda_initialized=False,updates=result[0].completed_primary_updates))
        if not kwargs.get('validate_only'):
            payload,_=ck.read_checkpoint(checkpoint);p,position,kind=result
            assert p.completed_primary_updates==2 and position['next_epoch']==2
            equal_saved(kwargs['model'].state_dict(),payload['model'])
            equal_saved(kwargs['optimizer'].state_dict(),payload['optimizer'])
            equal_saved(kwargs['scheduler'].state_dict(),payload['scheduler'])
            if architecture=='resnet50':
                equal_saved(kwargs['criterion'].center_loss.state_dict(),payload['center'])
                equal_saved(kwargs['auxiliary'].state_dict(),payload['center_optimizer'])
            validate_placement(**{k:kwargs[k] for k in ('cfg','model','criterion','optimizer','auxiliary')})
            restore_checks.append(dict(update_count=2,next_epoch=2,model_exact=True,optimizer_exact=True,
                                       scheduler_exact=True,center_exact=architecture=='resnet50',
                                       next_lr=[g['lr'] for g in kwargs['optimizer'].param_groups]))
        return result
    def epoch(*args,**kwargs):
        p=kwargs['update_progress'];guard=kwargs['before_update']
        if not counts:torch.cuda.reset_peak_memory_stats(0)
        def checked():
            guard()
            if mode=='interrupt' and p.completed_primary_updates==3:raise RuntimeError('Phase 9R deliberate interruption before update 4')
        kwargs['before_update']=checked
        result=original_epoch(*args,**kwargs)
        counts.append(p.completed_primary_updates);return result
    original_opt=train.build_optimizer;original_aux=train.build_center_optimizer
    def optimizer(actual,value):
        opt=original_opt(actual,value);objects['optimizer']=opt
        def step_after(opt,args,kwargs):
            if architecture=='pcb':torch.cuda.synchronize();steps.append(time.perf_counter()-pending['start'])
        opt.register_step_post_hook(step_after);return opt
    def auxiliary(actual,value):
        opt=original_aux(actual,value);objects['auxiliary']=opt
        if opt:
            def step_after(opt,args,kwargs):torch.cuda.synchronize();steps.append(time.perf_counter()-pending['start'])
            opt.register_step_post_hook(step_after)
        return opt
    original_select=device_api.select_lodo_device
    def selected(actual):
        assert not torch.cuda.is_initialized()
        if mode=='resume':assert inspection_checks and inspection_checks[-1]['cpu_first']
        return original_select(actual)
    def prohibited(*a,**k):raise AssertionError('Target/legacy access during recovery smoke')
    outcome=dict(intent,status='RUNNING');start=time.perf_counter()
    try:
        with fixture_budget(BUDGET),ExitStack() as stack:
            stack.enter_context(patch.object(protocol,'validate_output_path',smoke_output))
            for name,fn in [('build_train_loader',loader),('build_model',model),('build_criterion',criterion),
                            ('build_optimizer',optimizer),('build_center_optimizer',auxiliary),('train_one_epoch',epoch),
                            ('build_test_loader',prohibited),('create_experiment_dirs',prohibited)]:
                stack.enter_context(patch.object(train,name,fn))
            stack.enter_context(patch.object(ck,'restore_lodo_checkpoint',restored))
            stack.enter_context(patch.object(device_api,'select_lodo_device',selected))
            stack.enter_context(patch.object(data_build,'build_test_loader',prohibited))
            stack.enter_context(patch.object(evaluator,'evaluate_reid',prohibited))
            for key in protocol.DATASET_ORDER:stack.enter_context(patch.object(data_build,f'_build_{key}_test_dataset',prohibited))
            try:progress=train.run_lodo_training(cfg)
            except Exception as error:
                if mode!='interrupt' or 'Phase 9R deliberate interruption' not in str(error):raise
                events=json.loads((path/'recovery_events.json').read_text());event=events[-1]
                assert event['event']=='interruption_rollback_required' and event['discarded_successful_updates']==1
                assert event['source_boundary_updates']==2 and event['recover_from'] is not None
                selected,_=recovery.select_generation(dict(cfg,train=dict(cfg['train'],save=dict(cfg['train']['save'],resume=str(path/'checkpoints/ckpt_last.pth')))),path)
                payload,_=ck.read_checkpoint(selected);assert payload['progress']['completed_primary_updates']==2
                assert not (path/'checkpoints/ckpt_terminal.pth').exists()
                outcome.update(status='EXPECTED_INTERRUPTION',successful_updates_before_failure=3,boundary_updates=2,interruption=event)
            else:
                assert mode!='interrupt' and progress.completed_primary_updates==BUDGET
                assert progress.auxiliary_steps==(BUDGET if architecture=='resnet50' else 0)
                assert sum(progress.exposure.presentations['successful'].values())==64*BUDGET
                assert sum(map(len,progress.index_history))==64*BUDGET
                terminal=path/'checkpoints/ckpt_terminal.pth';payload,manifest=ck.read_checkpoint(terminal)
                assert payload['identity']['recovery']['capability']==recovery.CUDA_BOUNDARY_CAPABILITY
                assert len(recovery._read_index(path/'checkpoints'))==2
                if mode=='resume':assert len(restore_checks)==1
                outcome.update(status='PASS',successful_updates=BUDGET,auxiliary_updates=progress.auxiliary_steps,
                               index_history=list(progress.index_history),lr_trace=list(progress.lr_trace),
                               exposure=progress.exposure.summary(),mapping_digest=payload['identity']['map_digest'],
                               terminal_sha256=manifest['checkpoint_sha256'],terminal_bytes=terminal.stat().st_size,
                               checkpoint_integrity=True)
        outcome.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                       training_and_publication_seconds=time.perf_counter()-start,losses=losses,input_sha256=inputs,
                       step_seconds=steps,restore_checks=restore_checks,inspection_checks=inspection_checks,epoch_progress=counts,target_free_training=True)
        if mode!='interrupt':
            for name in ('model','criterion'):objects[name].cpu()
            for name in ('optimizer','auxiliary'):
                if objects[name]:objects[name].state.clear()
            objects.clear();gc.collect();torch.cuda.empty_cache()
            rebuilt=reconstruct_model(payload).to('cuda:0');assert rebuilt.checkpoint_metadata['num_classes']==2510
            outcome['reconstruction']=True;del rebuilt
        require_production_rejection(cfg)
    except BaseException as error:
        outcome.update(status='FAIL',error_type=type(error).__name__,error=str(error));raise
    finally:
        if path.is_dir():
            (path/f'{mode}-intent.json').write_bytes(ck.canonical_json(intent))
            (path/f'{mode}-outcome.json').write_bytes(ck.canonical_json(outcome))
        print(json.dumps({k:outcome.get(k) for k in ('architecture','mode','status','successful_updates','boundary_updates','peak_reserved_bytes','error')}),flush=True)
    return outcome


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--architecture',choices=['resnet50','pcb'],required=True)
    parser.add_argument('--campaign',required=True);parser.add_argument('--mode',choices=['continuous','interrupt','resume'],required=True)
    a=parser.parse_args();run(a.architecture,a.campaign,a.mode)
