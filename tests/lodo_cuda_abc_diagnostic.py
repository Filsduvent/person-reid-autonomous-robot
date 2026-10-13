"""NON-AUTHORITATIVE same-checkpoint A/B/C diagnostic using the original engine.

No production modifications, no image dumps, three updates after boundary two.
A/B retain bounded CPU state evidence; C compares both without retaining duplicates.
"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from lodo_cuda_admission_diagnostic import TITAN_UUID, logged_admission, write_once_json
from lodo_cuda_smoke import fixture_budget, require_production_rejection
from lodo_cuda_recovery_smoke import ShortTestEpoch
from lodo_cuda_numerical_diagnostic import nested_equal
from test_lodo_config import lodo_config
from reid.utils import lodo_checkpoint as ck, lodo_protocol as protocol, lodo_device as dev
from reid.utils import lodo_recovery as recovery
from reid.data import build as data_api
from reid.engine import evaluator
from scripts import train

ROOT=Path(__file__).resolve().parents[1]
UUID='GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5'


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def fingerprint(value):
    if torch.is_tensor(value) or isinstance(value,np.ndarray):
        a=value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.ascontiguousarray(value)
        return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):fingerprint(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [fingerprint(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def difference(actual,reference):
    """Original direction: assert_close(earlier path, later path). No new cutoffs."""
    assert actual.shape==reference.shape and actual.dtype==reference.dtype
    a=actual.detach().cpu();b=reference.detach().cpu()
    d=(a.double()-b.double()).abs();norm=torch.linalg.vector_norm(b.double())
    outside=~torch.isclose(a,b,rtol=.001,atol=.0001,equal_nan=False) if a.is_floating_point() else a!=b
    return dict(shape=list(a.shape),dtype=str(a.dtype),exact=torch.equal(a,b),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        max_abs=float(d.max()) if d.numel() else 0.,outside=int(outside.sum()),elements=a.numel(),
        relative_l2=float(torch.linalg.vector_norm(d)/norm) if norm else None,
        actual_abs_max=float(a.double().abs().max()) if a.numel() else 0.,
        reference_abs_max=float(b.double().abs().max()) if b.numel() else 0.)


def flatten(value,prefix=''):
    if torch.is_tensor(value):yield prefix,value
    elif isinstance(value,dict):
        for k,v in value.items():yield from flatten(v,prefix+'.'+str(k))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):yield from flatten(v,prefix+'.'+str(i))


def compare(earlier,later):
    left=dict(flatten(earlier));right=dict(flatten(later));assert left.keys()==right.keys()
    rows={k:difference(left[k],right[k]) for k in left}
    def non_tensors(v):
        if torch.is_tensor(v):return dict(tensor_shape=list(v.shape),dtype=str(v.dtype))
        if isinstance(v,dict):return {k:non_tensors(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [non_tensors(x) for x in v]
        return v
    return dict(non_tensor_equal=nested_equal(non_tensors(earlier),non_tensors(later)),
        nonexact_tensors=sum(not r['exact'] for r in rows.values()),outside=sum(r['outside'] for r in rows.values()),
        max_abs=max([r['max_abs'] for r in rows.values()] or [0.]),finite=all(r['finite'] for r in rows.values()),tensors=rows)


def output_path(cfg,*,project_root=None):
    path=Path(cfg['experiment']['output_dir'])
    if project_root is not None and Path(project_root).resolve()!=ROOT:raise ValueError('Root rebound')
    if path.resolve()!=path or '..' in path.parts:raise ValueError('Aliased output')
    parts=path.relative_to(ROOT/'exp/lodo_smoke').parts
    if len(parts)!=4 or not re.fullmatch(r'phase9r1-abc-[A-Za-z0-9._-]+',parts[0]) or parts[1:]!=('resnet50','LODO-1','producer'):
        raise ValueError('Outside diagnostic producer namespace')
    if path.exists():
        if not path.is_dir() or not cfg['train']['save'].get('resume'):
            raise ValueError('Fresh producer collision')
        # B/C reuse the immutable producer checkpoint, without creating outputs.
        selected=ck.validate_resume_path(cfg,path)
        policy='deterministic' if cfg['repro']['deterministic'] else 'original'
        _,checkpoint,_,_=read_common(path.parents[2],policy,os.environ.get('LODO_AUTHORIZED_GPU_UUID'))
        if selected!=checkpoint or Path(cfg['train']['save']['resume'])!=checkpoint:
            raise ValueError('Diagnostic resume must use the exact common update-2 checkpoint')
    return path


class Recorder:
    def __init__(self,base,path,*,reference_a=None):
        self.base=base;self.reference_a=reference_a;self.path=path;self.directory=base/path
        self.directory.mkdir(exist_ok=False)
        self.objects={};self.progress=None;self.current=0;self.events=[];self.comparisons=[]
        self.implementations={};self.handles=[];self.losses=[];self.started=time.perf_counter()

    def capture(self,stage,value,*,step=None):
        step=self.current if step is None else step
        if step<3 or step>5:return
        name=f'u{step:03d}-{stage}'
        if any(e['update']==step and e['stage']==stage for e in self.events):raise ValueError('Duplicate diagnostic event')
        data=cpu_tree(value)
        summary=dict(update=step,stage=stage,sha256=ck.digest(fingerprint(data)),
            devices={k:str(v.device) for k,v in flatten(value)},
            finite=all(bool(torch.isfinite(v).all()) for _,v in flatten(data)))
        assert summary['finite'],(step,stage,'nonfinite diagnostic state')
        if self.path in ('A','B'):torch.save(data,self.directory/(name+'.pth'))
        for prior in (() if self.path=='A' else ('A',) if self.path=='B' else ('A','B')):
            reference_directory=self.reference_a if prior=='A' and self.reference_a is not None else self.base/prior
            reference=torch.load(reference_directory/(name+'.pth'),map_location='cpu',weights_only=False)
            row=compare(reference,data) # preserve earlier path as actual, later as expected
            self.comparisons.append(dict(pair=prior+'/'+self.path,update=step,stage=stage,**row))
            del reference
        self.events.append(summary)
        if stage.startswith(('01_', '04_', '12_')):self.flush()

    def flush(self):
        value=dict(non_authoritative=True,path=self.path,events=self.events,
            comparisons=self.comparisons,adam_implementations=self.implementations,losses=self.losses)
        ck.atomic_write(self.directory/'trace.json',lambda stream:stream.write(json.dumps(value,indent=2).encode()))

    def named_state(self):
        o=self.objects;m=o['model'];opt=o['optimizer'];lookup={id(p):n for n,p in m.named_parameters()}
        owned={lookup[id(p)]:state for p,state in opt.state.items()}
        return dict(model=m.state_dict(),adam=owned)

    def ownership(self):
        o=self.objects;lookup=dict(o['model'].named_parameters());groups=o['optimizer'].param_groups
        assert all(len(g['params'])==1 and g['params'][0] is lookup[g['param_name']] for g in groups)
        module=ck.center_module(o['criterion'],o['auxiliary'])
        assert o['auxiliary'].param_groups[0]['params'][0] is module.centers
        return dict(main_parameter_association=True,center_parameter_association=True,
                    names=[g['param_name'] for g in groups])

    def install_model(self,model):
        self.objects['model']=model
        def before(module,args):
            self.current=self.progress.completed_primary_updates+1
            if self.current<3:return
            x=args[0];assert x.device==torch.device('cuda:0') and tuple(x.shape)==(64,3,256,128)
            indices=self.objects['loader'].latest_indices
            self.capture('05_batch',dict(input=fingerprint(x),indices=list(indices),
                expected_labels=[self.objects['loader'].dataset.labels[i] for i in indices],
                rng=fingerprint(ck.rng_state(self.objects['loader']))))
        self.handles.append(model.register_forward_pre_hook(before))
        for name,module in model.named_modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.BatchNorm2d,torch.nn.BatchNorm1d)):
                def layer(module,args,result,name=name):
                    if self.current<3:return
                    array=result.detach().cpu().contiguous()
                    # Full-output digest, bounded sample for numerical localization; no activation/image dump.
                    self.capture('06_layer_'+name,dict(output=fingerprint(array),sample=array.flatten()[:512].clone(),
                        abs_max=float(array.abs().max()),shape=list(array.shape)))
                self.handles.append(module.register_forward_hook(layer))
        def after(module,args,result):
            self.capture('06_forward',dict(outputs=result,buffers=dict(module.named_buffers())))
        self.handles.append(model.register_forward_hook(after))

    def install_criterion(self,criterion):
        self.objects['criterion']=criterion
        def after(module,args,result):
            if self.current<3:return
            outputs,labels=args;loss,logs=result
            expected=[self.objects['loader'].dataset.labels[i] for i in self.objects['loader'].latest_indices]
            assert labels.detach().cpu().tolist()==expected
            self.losses.append(dict(update=self.current,loss=float(loss.detach().cpu()),terms=logs))
            self.capture('06_losses',dict(loss=loss,terms={k:torch.tensor(v) for k,v in logs.items()},labels=labels))
        self.handles.append(criterion.register_forward_hook(after))

    def gradients(self):
        return dict(model={n:p.grad for n,p in self.objects['model'].named_parameters() if p.grad is not None},
                    center=self.objects['criterion'].center_loss.centers.grad)

    def attach_optimizer(self,opt):
        self.objects['optimizer']=opt
        def before(opt,args,kwargs):
            if self.current<3:return
            state=self.named_state();self.ownership()
            # Full-state SHA digests, plus complete first-convolution state before update.
            self.capture('09_before_main',dict(state_digest=fingerprint(state),
                weight=self.objects['model'].backbone[0].weight,
                first_adam=opt.state[self.objects['model'].backbone[0].weight],
                lr=[g['lr'] for g in opt.param_groups]))
        def after(opt,args,kwargs):self.capture('10_after_main',self.named_state())
        self.handles.extend([opt.register_step_pre_hook(before),opt.register_step_post_hook(after)])

    def attach_auxiliary(self,opt):
        self.objects['auxiliary']=opt
        def after(opt,args,kwargs):self.capture('11_after_center',dict(center=self.objects['criterion'].center_loss.state_dict(),optimizer=opt.state_dict()))
        self.handles.append(opt.register_step_post_hook(after))

    def accounting(self):
        p=self.progress;s=p.exposure.summary()
        self.capture('12_accounting',dict(completed=p.completed_primary_updates,main=p.main_steps_returned,
            auxiliary=p.auxiliary_steps,attempted=p.attempted_batches,forward=p.forward_batches,backward=p.backward_batches,
            consistent=p.consistent,lr_trace=list(p.lr_trace),indices=list(p.index_history),
            scheduler=self.objects['scheduler'].state_dict(),exposure={k:s[k] for k in p.exposure.phases},
            rng=fingerprint(ck.rng_state(self.objects['loader']))))


def install_engine_observation(stack,rec,engine):
    original_backward=torch.Tensor.backward
    def backward(value,*a,**k):
        result=original_backward(value,*a,**k)
        if rec.current>=3:rec.capture('07_backward',rec.gradients())
        return result
    stack.enter_context(patch.object(torch.Tensor,'backward',backward))
    criterion=rec.objects['criterion'];prepare=criterion.prepare_auxiliary_optimizer_step
    def scale():
        result=prepare();rec.capture('08_center_scaled',dict(center_gradient=criterion.center_loss.centers.grad));return result
    stack.enter_context(patch.object(criterion,'prepare_auxiliary_optimizer_step',scale))
    module=importlib.import_module('torch.optim.adam')
    for name in ('_multi_tensor_adam','_single_tensor_adam','_fused_adam'):
        fn=getattr(module,name)
        def observed(*a,_name=name,_fn=fn,**k):
            if rec.current>=3:rec.implementations[_name]=rec.implementations.get(_name,0)+1
            return _fn(*a,**k)
        stack.enter_context(patch.object(module,name,observed))
    source,start=inspect.getsourcelines(engine)
    target=start+next(i for i,line in enumerate(source) if 'for key, value in classification_accuracy(' in line)
    code=engine.__code__;seen=set();previous=sys.gettrace();assert previous is None
    def tracer(frame,event,arg):
        if frame.f_code is not code:return None
        if event=='line' and frame.f_lineno==target and rec.progress.completed_primary_updates>=3:
            step=rec.progress.completed_primary_updates
            if step not in seen:seen.add(step);rec.accounting()
        return tracer
    sys.settrace(tracer);stack.callback(sys.settrace,previous)


def guards(stack):
    def prohibited(*a,**k):raise AssertionError('Target/evaluation/legacy setup prohibited')
    for owner,name in [(train,'build_test_loader'),(train,'create_experiment_dirs'),(data_api,'build_test_loader'),(evaluator,'evaluate_reid')]:
        stack.enter_context(patch.object(owner,name,prohibited))
    for key in protocol.DATASET_ORDER:stack.enter_context(patch.object(data_api,f'_build_{key}_test_dataset',prohibited))


def boundary_capture(rec,checkpoint,arguments,*,capture=True):
    payload,manifest=ck.read_checkpoint(checkpoint);recovery.validate_boundary(payload)
    assert payload['progress']['completed_primary_updates']==2
    rec.ownership();o=rec.objects
    assert nested_equal(cpu_tree(o['model'].state_dict()),payload['model'])
    assert nested_equal(cpu_tree(o['criterion'].center_loss.state_dict()),payload['center'])
    assert nested_equal(cpu_tree(o['optimizer'].state_dict()),payload['optimizer'])
    assert nested_equal(cpu_tree(o['auxiliary'].state_dict()),payload['center_optimizer'])
    assert nested_equal(o['scheduler'].state_dict(),payload['scheduler'])
    assert nested_equal(ck.rng_state(o['loader']),payload['rng'])
    if capture:
        rec.capture('01_checkpoint',dict(checkpoint_sha256=manifest['checkpoint_sha256'],identity_digest=manifest['identity_digest'],boundary=payload['position']),step=3)
        rec.capture('02_restored_modules',dict(model=o['model'].state_dict(),center=o['criterion'].center_loss.state_dict()),step=3)
        rec.capture('03_restored_optimizers',dict(optimizer=o['optimizer'].state_dict(),center_optimizer=o['auxiliary'].state_dict(),ownership=rec.ownership()),step=3)
        rec.capture('04_restored_scheduler_rng',dict(scheduler=o['scheduler'].state_dict(),rng=fingerprint(ck.rng_state(o['loader']))),step=3)
    return payload,manifest


def observe_restore(stack,rec,checkpoint):
    """Observe each actual mutation boundary; never apply or substitute state."""
    payload,manifest=ck.read_checkpoint(checkpoint)
    recovery.validate_boundary(payload)
    rec.capture('01_checkpoint',dict(checkpoint_sha256=manifest['checkpoint_sha256'],
        identity_digest=manifest['identity_digest'],boundary=payload['position']),step=3)
    o=rec.objects
    original=o['criterion'].center_loss.load_state_dict
    def modules(*args,**kwargs):
        result=original(*args,**kwargs)
        rec.capture('02_restored_modules',dict(model=o['model'].state_dict(),
            center=o['criterion'].center_loss.state_dict()),step=3)
        return result
    stack.enter_context(patch.object(o['criterion'].center_loss,'load_state_dict',modules))
    original_opt=o['auxiliary'].load_state_dict
    def optimizers(*args,**kwargs):
        result=original_opt(*args,**kwargs)
        rec.capture('03_restored_optimizers',dict(optimizer=o['optimizer'].state_dict(),
            center_optimizer=o['auxiliary'].state_dict(),ownership=rec.ownership()),step=3)
        return result
    stack.enter_context(patch.object(o['auxiliary'],'load_state_dict',optimizers))
    original_rng=ck.restore_rng
    def rng(*args,**kwargs):
        result=original_rng(*args,**kwargs)
        rec.capture('04_restored_scheduler_rng',dict(scheduler=o['scheduler'].state_dict(),
            rng=fingerprint(ck.rng_state(o['loader']))),step=3)
        return result
    stack.enter_context(patch.object(ck,'restore_rng',rng))


def environment(gpu_uuid=UUID):
    return dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),uuid=gpu_uuid)


@contextmanager
def diagnostic_budget(policy):
    """Test-only deterministic variant; production validators remain unchanged."""
    with fixture_budget(5),ExitStack() as stack:
        if policy=='deterministic':
            original=protocol._expect
            def expect(cfg,path,value):
                if path in ('repro.deterministic','repro.benchmark'):
                    expected={'repro.deterministic':True,'repro.benchmark':False}[path]
                    assert cfg['repro'][path.split('.')[1]] is expected
                else:original(cfg,path,value)
            stack.enter_context(patch.object(protocol,'_expect',expect))
        yield


def continuation_artifacts(base,path,continuation_id):
    if continuation_id is None:return base
    if path not in ('B','C') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',continuation_id):
        raise ValueError('Only B/C may use a valid continuation ID')
    destination=base/'continuations'/continuation_id
    if destination.resolve()!=destination or not destination.is_dir():
        raise ValueError('Continuation directory must be existing and canonical')
    return destination


def _run(campaign,path,policy='original',gpu_uuid=UUID,continuation_id=None):
    assert re.fullmatch(r'phase9r1-abc-[A-Za-z0-9._-]+',campaign)
    assert gpu_uuid in (UUID,TITAN_UUID)
    if gpu_uuid==TITAN_UUID:assert campaign.startswith('phase9r1-abc-titan-')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ.get('LODO_AUTHORIZED_GPU_UUID')==gpu_uuid
    assert not torch.cuda.is_initialized()
    base=ROOT/'exp/lodo_smoke'/campaign
    cfg=lodo_config('resnet50','LODO-1');cfg['system'].update(device='cuda',gpu_id=0,amp=False,log_interval=1000)
    cfg['data'].update(root='/home/addirakoze/nobackup/Projects/Dataset',num_workers=0)
    cfg['train']['recovery']=dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2)
    cfg['experiment'].update(name='NON-AUTHORITATIVE-ABC',output_dir=str(base/'resnet50/LODO-1/producer'))
    protocol.scientific_configuration(cfg);cfg['train']['max_updates']=5;require_production_rejection(cfg)
    if policy=='deterministic':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        cfg['repro'].update(deterministic=True,benchmark=False)
        torch.use_deterministic_algorithms(True)
    else:assert policy=='original' and not torch.are_deterministic_algorithms_enabled()
    # No model downloads. Original frozen initialization stays cache-backed.
    weights=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    assert weights.is_file()
    artifact_base=continuation_artifacts(base,path,continuation_id)
    admission=dev.audit_gpu(gpu_uuid);rec=Recorder(artifact_base,path,reference_a=base/'A' if continuation_id is not None else None);started=time.perf_counter()
    outcome=dict(non_authoritative=True,path=path,policy=policy,admission=admission,pretrained_sha256=ck.file_digest(weights),status='RUNNING')
    objects=rec.objects
    loader_fn=train.build_train_loader;model_fn=train.build_model;criterion_fn=train.build_criterion
    opt_fn=train.build_optimizer;aux_fn=train.build_center_optimizer;sched_fn=train.build_scheduler;epoch_fn=train.train_one_epoch
    def loader(actual):
        obj,C=loader_fn(actual);assert C==2510 and obj.num_workers==0
        obj._observer.original=ShortTestEpoch(obj.batch_sampler);objects['loader']=obj;return obj,C
    def model(actual,num_classes,**kwargs):
        obj=model_fn(actual,num_classes,**kwargs);rec.install_model(obj);return obj
    def criterion(actual,num_classes,feat_dim):
        obj=criterion_fn(actual,num_classes,feat_dim);rec.install_criterion(obj);return obj
    def optimizer(actual,obj):
        opt=opt_fn(actual,obj);rec.attach_optimizer(opt);return opt
    def auxiliary(actual,obj):
        opt=aux_fn(actual,obj);rec.attach_auxiliary(opt);return opt
    def scheduler(actual,opt,**kwargs):
        obj=sched_fn(actual,opt,**kwargs);objects['scheduler']=obj;return obj
    engine_installed=False
    def epoch(*args,**kwargs):
        nonlocal engine_installed
        rec.progress=kwargs['update_progress']
        if not engine_installed:install_engine_observation(stack,rec,epoch_fn);engine_installed=True;torch.cuda.reset_peak_memory_stats(0)
        return epoch_fn(*args,**kwargs)
    original_save=ck.save_lodo_checkpoint
    def saved(destination,**kwargs):
        result=original_save(destination,**kwargs)
        if kwargs['kind']=='recovery' and kwargs['progress'].completed_primary_updates==2:
            request=copy.deepcopy(cfg);request['train']['save']['resume']=str(Path(cfg['experiment']['output_dir'])/'checkpoints/ckpt_last.pth')
            common,invalid=recovery.select_generation(request,Path(cfg['experiment']['output_dir']));assert not invalid
            objects['scheduler']=kwargs['scheduler'];rec.progress=kwargs['progress']
            boundary_capture(rec,common,kwargs)
            write_once_json(base/'common.json',dict(checkpoint=str(common),sha256=result['checkpoint_sha256'],identity_digest=result['identity_digest'],policy=policy,gpu_uuid=gpu_uuid))
        return result
    try:
        with diagnostic_budget(policy),ExitStack() as stack:
            stack.enter_context(patch.object(protocol,'validate_output_path',output_path));guards(stack)
            if path=='A':
                for name,fn in [('build_train_loader',loader),('build_model',model),('build_criterion',criterion),('build_optimizer',optimizer),('build_center_optimizer',auxiliary),('build_scheduler',scheduler),('train_one_epoch',epoch)]:stack.enter_context(patch.object(train,name,fn))
                stack.enter_context(patch.object(ck,'save_lodo_checkpoint',saved))
                progress=train.run_lodo_training(cfg)
                outcome['producer_never_restored']=True
            else:
                common,checkpoint,payload,proof=read_common(base,policy,gpu_uuid)
                cfg=copy.deepcopy(payload['cfg']);cfg['train']['save']['resume']=str(checkpoint)
                train.set_seed(cfg['repro']['seed'],cfg['repro']['deterministic'],cfg['repro']['benchmark'])
                value,C=loader(cfg);m=model(cfg,C,initialize_pretrained=False);c=criterion(cfg,C,m.feat_dim)
                opt=optimizer(cfg,m);aux=auxiliary(cfg,c);sched=scheduler(cfg,opt,steps_per_epoch=len(value))
                arguments=dict(cfg=cfg,loader=value,model=m,criterion=c,optimizer=opt,auxiliary=aux,scheduler=sched)
                p,position,kind=ck.restore_lodo_checkpoint(checkpoint,**arguments,cpu_inspection=True,validate_only=True)
                assert p.completed_primary_updates==2 and kind=='recovery' and not torch.cuda.is_initialized()
                outcome['cpu_validation_before_cuda']=True
                device,guard,evidence=dev.select_lodo_device(cfg);m.to(device);c.to(device)
                with ExitStack() as restoring:
                    observe_restore(restoring,rec,checkpoint)
                    progress,position,kind=ck.restore_lodo_checkpoint(checkpoint,**arguments)
                rec.progress=progress;boundary_capture(rec,checkpoint,arguments,capture=False)
                outcome.update(restoration_exact=True,device_admission=evidence)
                n=position['next_epoch']
                while not progress.budget_completed:
                    guard()
                    epoch(m,value,c,opt,aux,device,False,1000,scheduler=sched,epoch=n,update_progress=progress,before_update=guard);n+=1
                    guard()
                outcome['restore_source_sha256']=proof['checkpoint_sha256']
            assert progress.completed_primary_updates==5 and progress.auxiliary_steps==5 and progress.consistent
            assert progress.exposure.summary()['successful']['images']==320
            common=json.loads((base/'common.json').read_text());assert ck.file_digest(common['checkpoint'])==common['sha256']
            outcome.update(status='PASS',successful_updates=5,center_steps=5,accepted_images=320,
                common_sha256=common['sha256'],mapping_digest=objects['loader'].dataset.mapping_digest,
                environment=environment(gpu_uuid),peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                elapsed_seconds=time.perf_counter()-started,original_sampler=True,target_free=True)
        require_production_rejection(cfg)
    except BaseException as error:
        outcome.update(status='FAIL',error_type=type(error).__name__,error=str(error));raise
    finally:
        rec.flush();write_once_json(rec.directory/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    return outcome


def read_common(base,policy,gpu_uuid):
    common=json.loads((base/'common.json').read_text())
    assert common['policy']==policy and common['gpu_uuid']==gpu_uuid
    checkpoint=Path(common['checkpoint'])
    assert checkpoint.resolve()==checkpoint and checkpoint.parent==base/'resnet50/LODO-1/producer/checkpoints'
    payload,proof=ck.read_checkpoint(checkpoint)
    recovery.validate_boundary(payload)
    assert proof['checkpoint_sha256']==common['sha256']
    assert proof['identity_digest']==common['identity_digest']
    assert payload['progress']['completed_primary_updates']==2
    assert payload['identity']['recovery']['environment']['uuid']==gpu_uuid
    return common,checkpoint,payload,proof


def run(campaign,path,policy='original',gpu_uuid=UUID,continuation_id=None):
    assert re.fullmatch(r'phase9r1-abc-[A-Za-z0-9._-]+',campaign)
    assert path in ('A','B','C')
    if gpu_uuid==TITAN_UUID:
        assert campaign.startswith('phase9r1-abc-titan-')
        base=ROOT/'exp/lodo_smoke'/campaign
        artifact_base=continuation_artifacts(base,path,continuation_id)
        with logged_admission(artifact_base/('admission-'+path)):
            return _run(campaign,path,policy,gpu_uuid,continuation_id)
    assert gpu_uuid==UUID and continuation_id is None
    return _run(campaign,path,policy,gpu_uuid)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True)
    parser.add_argument('--path',choices=['A','B','C'],required=True);parser.add_argument('--policy',choices=['original','deterministic'],default='original')
    parser.add_argument('--gpu-uuid',choices=[UUID,TITAN_UUID],default=UUID)
    parser.add_argument('--continuation-id',default=None)
    args=parser.parse_args();run(args.campaign,args.path,args.policy,args.gpu_uuid,args.continuation_id)
