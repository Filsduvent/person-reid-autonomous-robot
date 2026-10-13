"""NON-AUTHORITATIVE Phase 9R.3: actual orchestration for every CUDA branch.

The controller archives each finished branch, then recreates the same canonical
run directory from an immutable update-2 snapshot. Checkpoint bytes and the
production same-run/config/runtime validators are never substituted. The only
fixtures are budget=5, a smoke output namespace and two original sampler batches
per test epoch. This is a controlled diagnostic, not a production fork feature.
"""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from unittest.mock import patch

# Launch-time workspace must be captured before importing Torch.
from reid.utils.lodo_cuda_policy import STRICT_POLICY, runtime_controls
import torch
from lodo_cuda_abc_diagnostic import cpu_tree, fingerprint, flatten, guards
from lodo_cuda_numerical_diagnostic import nested_equal
from lodo_cuda_recovery_smoke import ShortTestEpoch
from lodo_cuda_smoke import fixture_budget, require_production_rejection
from lodo_cuda_admission_diagnostic import TITAN_UUID, logged_admission, write_once_json
from test_lodo_config import lodo_config
from reid.utils import lodo_checkpoint as ck, lodo_protocol as protocol
from reid.utils import lodo_recovery as recovery, lodo_device as dev
from scripts import train

ROOT=Path(__file__).resolve().parents[1]
SMOKE=ROOT/'exp/lodo_smoke'
CASES=('A','B','C','interrupt','resume','terminal')


def campaign_path(name):
    if not re.fullmatch(r'phase9r3-[A-Za-z0-9._-]+',name):
        raise ValueError('Invalid Phase 9R.3 campaign')
    path=SMOKE/name
    if path.resolve()!=path:raise ValueError('Campaign alias')
    return path


def output_path(cfg,*,project_root=None):
    if project_root is not None and Path(project_root).resolve()!=ROOT:raise ValueError('Root rebound')
    path=Path(cfg['experiment']['output_dir'])
    try:
        parts=path.relative_to(SMOKE).parts
        assert len(parts)==4 and re.fullmatch(r'phase9r3-[A-Za-z0-9._-]+',parts[0])
        assert parts[1:]==(cfg['protocol']['architecture'],'LODO-1','active')
        assert path.resolve()==path and '..' not in path.parts
    except (ValueError,AssertionError) as error:raise ValueError('Unsafe diagnostic namespace') from error
    if cfg['train']['save'].get('resume'):ck.validate_resume_path(cfg,path)
    elif path.exists():raise ValueError('Fresh diagnostic collision')
    return path


def clone_boundary(source,destination):
    """Copy metadata; hard-link immutable checkpoint pairs without reserializing.

    Destination must be absent. No existing branch is repaired or overwritten.
    Writers use atomic replacement; source checkpoint bytes are read-only inputs.
    """
    source=Path(source);destination=Path(destination)
    if source.resolve()!=source or destination.resolve()!=destination or destination.exists():
        raise ValueError('Boundary clone collision/alias')
    if (source/'checkpoints/ckpt_terminal.pth').exists():raise ValueError('Boundary is already terminal')
    if any(p.is_symlink() for p in source.rglob('*')):raise ValueError('Aliased boundary input')
    def materialize(src,dst):
        if Path(src).is_symlink():raise ValueError('Aliased boundary input')
        if Path(src).parent.name=='checkpoints' and Path(src).suffix in ('.pth','.json') and Path(src).name!='recovery_index.json':
            os.link(src,dst)
        else:shutil.copy2(src,dst)
        return dst
    shutil.copytree(source,destination,copy_function=materialize,symlinks=False)


def compare_values(left,right,*,loss=False):
    """Earlier path is actual; later path reference. Original tolerances only."""
    a=dict(flatten(left));b=dict(flatten(right))
    if a.keys()!=b.keys():raise ValueError('Comparison tensor topology differs')
    rows={}
    for key,x in a.items():
        y=b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Comparison tensor metadata differs')
        finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        outside=~torch.isclose(x,y,rtol=.001,atol=.001 if loss else .0001,equal_nan=False) if x.is_floating_point() else x!=y
        rows[key]=dict(exact=torch.equal(x,y),finite=finite,outside=int(outside.sum()),max_abs=float((x.double()-y.double()).abs().max()) if x.numel() else 0.)
    def structure(value):
        if torch.is_tensor(value):return ('tensor',str(value.dtype),tuple(value.shape))
        if isinstance(value,dict):return {k:structure(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [structure(v) for v in value]
        return value
    return dict(non_tensor_equal=nested_equal(structure(left),structure(right)),
                nonexact_tensors=sum(not v['exact'] for v in rows.values()),
                outside=sum(v['outside'] for v in rows.values()),finite=all(v['finite'] for v in rows.values()),
                max_abs=max([v['max_abs'] for v in rows.values()] or [0.]),tensors=rows)


class Recorder:
    def __init__(self,base,case,architecture):
        self.base=base;self.case=case;self.architecture=architecture;self.directory=base/('trace-'+case)
        self.directory.mkdir(exist_ok=False);self.objects={};self.progress=None;self.current=0
        self.events=[];self.comparisons=[];self.losses=[];self.step_seconds=[];self.implementations={}
        self.started=None

    def capture(self,stage,value,step=None):
        step=self.current if step is None else step
        if not 3<=step<=5:return
        if any(e['update']==step and e['stage']==stage for e in self.events):raise ValueError('Repeated event')
        data=cpu_tree(value);name=f'u{step:03d}-{stage}.pth'
        assert all(bool(torch.isfinite(v).all()) for _,v in flatten(data))
        self.events.append(dict(update=step,stage=stage,digest=ck.digest(fingerprint(data))))
        # A/B are immutable references. Interrupted/resumed branches compare to A.
        if self.case in ('A','B'):torch.save(data,self.directory/name)
        prior=() if self.case=='A' else ('A','B') if self.case=='C' else ('A',)
        for case in prior:
            ref=torch.load(self.base/('trace-'+case)/name,map_location='cpu',weights_only=False)
            result=compare_values(ref,data,loss=stage=='06_losses')
            self.comparisons.append(dict(pair=case+'/'+self.case,update=step,stage=stage,**result))
        if stage in ('04_restored','12_accounting'):self.flush()

    def flush(self):
        ck.atomic_write(self.directory/'trace.json',lambda s:s.write(ck.canonical_json(dict(
            events=self.events,comparisons=self.comparisons,losses=self.losses,step_seconds=self.step_seconds,
            optimizer_implementations=self.implementations))))

    def ownership(self):
        o=self.objects;lookup=dict(o['model'].named_parameters());groups=o['optimizer'].param_groups
        assert all(len(g['params'])==1 and g['params'][0] is lookup[g['param_name']] for g in groups)
        module=ck.center_module(o['criterion'],o['auxiliary'])
        if self.architecture=='resnet50':assert o['auxiliary'].param_groups[0]['params'][0] is module.centers
        else:assert module is None and o['auxiliary'] is None
        return dict(names=[g['param_name'] for g in groups],center=module is not None)

    def state(self):
        o=self.objects;lookup={id(p):n for n,p in o['model'].named_parameters()}
        return dict(model=o['model'].state_dict(),optimizer={lookup[id(p)]:v for p,v in o['optimizer'].state.items()},
                    groups=[{k:v for k,v in g.items() if k!='params'} for g in o['optimizer'].param_groups])

    def restored(self,checkpoint):
        o=self.objects;payload,proof=ck.read_checkpoint(checkpoint)
        self.ownership()
        for value,saved in ((o['model'].state_dict(),payload['model']),(o['optimizer'].state_dict(),payload['optimizer']),
                            (o['scheduler'].state_dict(),payload['scheduler']),(ck.rng_state(o['loader']),payload['rng'])):
            assert nested_equal(cpu_tree(value),saved),'Exact restoration failed'
        if o['auxiliary'] is not None:
            assert nested_equal(cpu_tree(o['criterion'].center_loss.state_dict()),payload['center'])
            assert nested_equal(cpu_tree(o['auxiliary'].state_dict()),payload['center_optimizer'])
        self.capture('04_restored',dict(checkpoint_sha256=proof['checkpoint_sha256'],identity_digest=proof['identity_digest'],
            model=o['model'].state_dict(),optimizer=o['optimizer'].state_dict(),
            center=payload['center'],center_optimizer=payload['center_optimizer'],
            scheduler=o['scheduler'].state_dict(),rng=fingerprint(ck.rng_state(o['loader'])),ownership=self.ownership()),step=3)
        return proof

    def accounting(self):
        p=self.progress;o=self.objects
        torch.cuda.synchronize()
        self.step_seconds.append(dict(update=p.completed_primary_updates,seconds=time.perf_counter()-self.started,
                                      instrumented=p.completed_primary_updates>=3))
        self.capture('12_accounting',dict(completed=p.completed_primary_updates,auxiliary=p.auxiliary_steps,
            main=p.main_steps_returned,attempted=p.attempted_batches,forward=p.forward_batches,backward=p.backward_batches,
            consistent=p.consistent,lr_trace=list(p.lr_trace),indices=list(p.index_history),
            scheduler=o['scheduler'].state_dict(),exposure={k:p.exposure.summary()[k] for k in p.exposure.phases},
            rng=fingerprint(ck.rng_state(o['loader']))))


def run(campaign,architecture,case):
    base=campaign_path(campaign)/architecture/'LODO-1';active=base/'active'
    assert case in CASES and os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ.get('LODO_AUTHORIZED_GPU_UUID')==TITAN_UUID
    assert not torch.cuda.is_initialized()
    cfg=lodo_config(architecture,'LODO-1');cfg['system'].update(device='cuda',gpu_id=0,amp=False,log_interval=1000)
    cfg['data'].update(root='/home/addirakoze/nobackup/Projects/Dataset',num_workers=0)
    cfg['repro'].update(cuda_policy=STRICT_POLICY,deterministic=True,benchmark=False)
    cfg['train']['recovery']=dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2)
    cfg['experiment'].update(name='NON-AUTHORITATIVE-PHASE9R3',output_dir=str(active))
    protocol.scientific_configuration(cfg);cfg['train']['max_updates']=5;require_production_rejection(cfg)
    common=None
    if case!='A':
        common=json.loads((base/'common.json').read_text());checkpoint=active/'checkpoints'/common['name']
        assert ck.file_digest(checkpoint)==common['sha256']
        assert checkpoint.stat().st_ino==Path(common['immutable_checkpoint']).stat().st_ino
        cfg['train']['save']['resume']=str(active/'checkpoints/ckpt_last.pth' if case in ('resume','terminal') else checkpoint)
    weights=Path(torch.hub.get_dir())/'checkpoints'/('resnet50-11ad3fa6.pth' if architecture=='resnet50' else 'resnet50-19c8e357.pth')
    assert weights.is_file(),'Pretrained cache absent; downloads prohibited'
    rec=Recorder(base,case,architecture);o=rec.objects;cpu_checks=[];restore_checks=[];save_times=[];restored_times=[]
    result=dict(case=case,architecture=architecture,status='RUNNING',non_authoritative=True,production_orchestration=True,
                exact_same_checkpoint_bytes=case!='A',pretrained_sha256=ck.file_digest(weights),pid=os.getpid())
    started=time.perf_counter();engine=train.train_one_epoch
    with ExitStack() as stack:
        stack.enter_context(fixture_budget(5));stack.enter_context(patch.object(protocol,'validate_output_path',output_path));guards(stack)
        loader_fn=train.build_train_loader;model_fn=train.build_model;criterion_fn=train.build_criterion
        opt_fn=train.build_optimizer;aux_fn=train.build_center_optimizer;sched_fn=train.build_scheduler
        def loader(actual):
            value,C=loader_fn(actual);assert C==2510 and value.num_workers==0
            value._observer.original=ShortTestEpoch(value.batch_sampler);o['loader']=value;return value,C
        def model(actual,num_classes,**kwargs):
            value=model_fn(actual,num_classes,**kwargs);o['model']=value
            def before(module,args):
                rec.current=rec.progress.completed_primary_updates+1
                x=args[0];assert x.device==torch.device('cuda:0') and tuple(x.shape)==(64,3,*actual['data']['train']['images']['size'])
                torch.cuda.synchronize();rec.started=time.perf_counter()
                rec.capture('05_batch',dict(input=fingerprint(x),indices=list(o['loader'].latest_indices),
                    rng=fingerprint(ck.rng_state(o['loader']))))
            value.register_forward_pre_hook(before)
            value.register_forward_hook(lambda m,a,v:rec.capture('06_forward',dict(outputs=v,buffers=dict(m.named_buffers()))))
            return value
        def criterion(actual,num_classes,feat_dim):
            value=criterion_fn(actual,num_classes,feat_dim);o['criterion']=value
            def after(module,args,output):
                outputs,labels=args;loss,logs=output
                assert labels.cpu().tolist()==[o['loader'].dataset.labels[i] for i in o['loader'].latest_indices]
                if architecture=='pcb':assert len(outputs['logits'])==6 and all(tuple(v.shape)==(64,2510) for v in outputs['logits'])
                rec.losses.append(dict(update=rec.current,loss=float(loss.detach().cpu()),terms=logs))
                rec.capture('06_losses',dict(loss=loss,terms={k:torch.tensor(v) for k,v in logs.items()},labels=labels))
            value.register_forward_hook(after)
            if architecture=='resnet50':
                prepare=value.prepare_auxiliary_optimizer_step
                def scaled():
                    result=prepare();rec.capture('08_center_scaled',dict(gradient=value.center_loss.centers.grad));return result
                stack.enter_context(patch.object(value,'prepare_auxiliary_optimizer_step',scaled))
            return value
        def optimizer(actual,value):
            opt=opt_fn(actual,value);o['optimizer']=opt
            opt.register_step_pre_hook(lambda opt,a,k:rec.capture('09_before_main',rec.state()))
            opt.register_step_post_hook(lambda opt,a,k:rec.capture('10_after_main',rec.state()))
            return opt
        def auxiliary(actual,value):
            opt=aux_fn(actual,value);o['auxiliary']=opt
            if opt:opt.register_step_post_hook(lambda opt,a,k:rec.capture('11_after_center',dict(center=value.center_loss.state_dict(),optimizer=opt.state_dict())))
            return opt
        def scheduler(actual,opt,**kwargs):
            value=sched_fn(actual,opt,**kwargs);o['scheduler']=value;return value
        def epoch(*args,**kwargs):
            rec.progress=kwargs['update_progress'];return engine(*args,**kwargs)
        for name,fn in [('build_train_loader',loader),('build_model',model),('build_criterion',criterion),('build_optimizer',optimizer),
                        ('build_center_optimizer',auxiliary),('build_scheduler',scheduler),('train_one_epoch',epoch)]:
            stack.enter_context(patch.object(train,name,fn))
        original_backward=torch.Tensor.backward
        def backward(tensor,*args,**kwargs):
            value=original_backward(tensor,*args,**kwargs)
            rec.capture('07_backward',dict(model={n:p.grad for n,p in o['model'].named_parameters() if p.grad is not None},
                center=o['criterion'].center_loss.centers.grad if architecture=='resnet50' else None));return value
        stack.enter_context(patch.object(torch.Tensor,'backward',backward))
        import importlib
        module=importlib.import_module('torch.optim.adam' if architecture=='resnet50' else 'torch.optim.sgd')
        for name in ('_multi_tensor_adam','_single_tensor_adam','_fused_adam','_multi_tensor_sgd','_single_tensor_sgd','_fused_sgd'):
            if not hasattr(module,name):continue
            fn=getattr(module,name)
            def observed(*a,_fn=fn,_name=name,**k):
                rec.implementations[_name]=rec.implementations.get(_name,0)+1;return _fn(*a,**k)
            stack.enter_context(patch.object(module,name,observed))
        source,start=inspect.getsourcelines(engine)
        target=start+next(i for i,line in enumerate(source) if 'for key, value in classification_accuracy(' in line)
        seen=set();assert sys.gettrace() is None
        def trace(frame,event,arg):
            if frame.f_code is not engine.__code__:return None
            if event=='line' and frame.f_lineno==target:
                step=rec.progress.completed_primary_updates
                if step not in seen:
                    seen.add(step);rec.accounting()
                    if case=='interrupt' and step==3:
                        write_once_json(rec.directory/'interruption-marker.json',dict(completed=3,consistent=rec.progress.consistent,
                            boundary_updates=2,accepted_images=sum(rec.progress.exposure.presentations['successful'].values())))
                        os.kill(os.getpid(),signal.SIGTERM)  # only this diagnostic child; actual process termination
            return trace
        sys.settrace(trace);stack.callback(sys.settrace,None)
        original_restore=ck.restore_lodo_checkpoint
        def restored(path,**kwargs):
            before=time.perf_counter();value=original_restore(path,**kwargs)
            restored_times.append(dict(validate_only=bool(kwargs.get('validate_only')),seconds=time.perf_counter()-before))
            if kwargs.get('validate_only'):
                assert not torch.cuda.is_initialized();cpu_checks.append(dict(before_cuda=True,kind=value[2],updates=value[0].completed_primary_updates))
            elif value[2]!='terminal':
                assert value[0].completed_primary_updates==2;rec.progress=value[0];rec.restored(path)
                restore_checks.append(dict(exact=True,checkpoint_sha256=ck.file_digest(path)))
            return value
        stack.enter_context(patch.object(ck,'restore_lodo_checkpoint',restored))
        original_save=ck.save_lodo_checkpoint
        def saved(path,**kwargs):
            before=time.perf_counter();proof=original_save(path,**kwargs)
            save_times.append(dict(updates=kwargs['progress'].completed_primary_updates,kind=kwargs['kind'],seconds=time.perf_counter()-before))
            if case=='A' and kwargs['kind']=='recovery' and kwargs['progress'].completed_primary_updates==2:
                request=copy.deepcopy(cfg);request['train']['save']['resume']=str(active/'checkpoints/ckpt_last.pth')
                checkpoint,invalid=recovery.select_generation(request,active);assert not invalid
                rec.progress=kwargs['progress'];rec.restored(checkpoint)
                clone_boundary(active,base/'boundary')
                immutable=base/'boundary/checkpoints'/checkpoint.name
                assert immutable.stat().st_ino==checkpoint.stat().st_ino and ck.file_digest(immutable)==proof['checkpoint_sha256']
                write_once_json(base/'common.json',dict(name=checkpoint.name,sha256=proof['checkpoint_sha256'],
                    immutable_checkpoint=str(immutable),identity_digest=proof['identity_digest']))
            return proof
        stack.enter_context(patch.object(ck,'save_lodo_checkpoint',saved))
        if case=='terminal':
            def prohibited(*a,**k):raise AssertionError('Terminal no-op attempted GPU admission')
            stack.enter_context(patch.object(dev,'select_lodo_device',prohibited))
        else:
            original_select=dev.select_lodo_device
            def select(actual):
                assert not torch.cuda.is_initialized()
                if case!='A':assert cpu_checks
                value=original_select(actual);torch.cuda.reset_peak_memory_stats(0);return value
            stack.enter_context(patch.object(dev,'select_lodo_device',select))
        try:
            with logged_admission(base/('admission-'+case)):
                progress=train.run_lodo_training(cfg)
                assert progress.completed_primary_updates==5 and progress.auxiliary_steps==(5 if architecture=='resnet50' else 0)
                assert progress.consistent and sum(progress.exposure.presentations['successful'].values())==320
                assert len(progress.index_history)==5
                if case!='terminal':assert o['scheduler'].state_dict()['completed_updates']==5
                payload,proof=ck.read_checkpoint(active/'checkpoints/ckpt_terminal.pth')
                assert payload['lodo_schema_version']==2 and len(recovery._read_index(active/'checkpoints'))==2
                if case=='terminal':assert not torch.cuda.is_initialized() and not rec.events
                else:
                    torch.cuda.synchronize();result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(0),device_memory=dev._query('uuid,memory.total,memory.used,memory.free'))
                if case not in ('A','terminal'):assert len(restore_checks)==1 and cpu_checks
                require_original=copy.deepcopy(cfg);require_original['train']['save']['resume']=''
                result.update(status='PASS',successful_updates=5,auxiliary_updates=progress.auxiliary_steps,accepted_images=320,
                    restored=restore_checks,cpu_checks=cpu_checks,mapping_digest=payload['identity']['map_digest'],
                    environment=payload['identity']['recovery']['environment'],terminal_sha256=proof['checkpoint_sha256'],
                    terminal_bytes=(active/'checkpoints/ckpt_terminal.pth').stat().st_size,retained_generations=recovery._read_index(active/'checkpoints'),
                    save_times=save_times,restore_times=restored_times,elapsed_seconds=time.perf_counter()-started,
                    runtime_controls=runtime_controls(),target_free=True,producer_never_restored=case=='A' and not restored_times)
        except BaseException as error:
            result.update(status='FAIL',error_type=type(error).__name__,error=str(error));raise
        finally:
            rec.flush();write_once_json(rec.directory/'outcome.json',result)
    require_production_rejection(cfg)
    print(json.dumps({k:result.get(k) for k in ('case','architecture','status','peak_allocated_bytes','peak_reserved_bytes','elapsed_seconds')}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--architecture',required=True,choices=('resnet50','pcb'));p.add_argument('--case',required=True,choices=CASES)
    args=p.parse_args();run(args.campaign,args.architecture,args.case)
