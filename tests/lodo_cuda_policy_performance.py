"""Matched short policy timing through actual LODO orchestration; no qualification claim.

Five updates, batch64, frozen architecture-specific recipes. Steps1/2 warm up;
steps3/4/5 measured from model entry through completed-update accounting, excluding
image transfer, input hashing, device audit, checkpoint I/O and loader/preflight.
No parameter/gradient trace I/O. All measured inputs must match across policies.
"""
from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch
from reid.utils.lodo_cuda_policy import STRICT_POLICY,ORIGINAL_POLICY,runtime_controls
import torch
from lodo_cuda_policy_qualification import campaign_path,output_path
from lodo_cuda_recovery_smoke import ShortTestEpoch
from lodo_cuda_abc_diagnostic import fingerprint,guards
from lodo_cuda_smoke import fixture_budget,require_production_rejection
from lodo_cuda_admission_diagnostic import TITAN_UUID,logged_admission,write_once_json
from test_lodo_config import lodo_config
from reid.utils import lodo_protocol as protocol,lodo_device as dev,lodo_checkpoint as ck
from scripts import train


def measure(name,arch,policy):
    root=campaign_path(name);summary=json.loads((root/'campaign-summary.json').read_text())
    assert summary['status']=='COMPLETED' and summary['architectures'][arch]['status']=='PASS'
    assert policy in ('original','strict') and not torch.cuda.is_initialized()
    if policy=='original':assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    else:assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    weights=Path(torch.hub.get_dir())/'checkpoints'/('resnet50-11ad3fa6.pth' if arch=='resnet50' else 'resnet50-19c8e357.pth')
    if not weights.is_file():raise ValueError('Pretrained cache missing; downloads prohibited')
    base=root/'performance'/arch/policy;base.mkdir(parents=True,exist_ok=False)
    cfg=lodo_config(arch,'LODO-1');cfg['system'].update(device='cuda',gpu_id=0,amp=False,log_interval=1000)
    cfg['data'].update(root='/home/addirakoze/nobackup/Projects/Dataset',num_workers=0)
    cfg['repro'].update(cuda_policy=STRICT_POLICY if policy=='strict' else ORIGINAL_POLICY,
                        deterministic=policy=='strict',benchmark=policy=='original')
    cfg['train']['recovery']=dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2)
    # Dedicated fresh canonical active path; no restore and no reuse of recovery branches.
    active=root/'performance'/arch/policy/'active';cfg['experiment'].update(name='NON-AUTHORITATIVE-POLICY-TIMING',output_dir=str(active))
    protocol.scientific_configuration(cfg);cfg['train']['max_updates']=5;require_production_rejection(cfg)
    def timing_output(actual,**kwargs):
        if Path(actual['experiment']['output_dir'])!=active or active.resolve()!=active or active.exists() or actual['train']['save'].get('resume'):
            raise ValueError('Unsafe/nonfresh timing namespace')
        return active
    o={};steps=[];inputs=[];losses=[];pending={};engine=train.train_one_epoch;started=time.perf_counter()
    result=dict(status='RUNNING',architecture=arch,policy=policy,non_authoritative=True,pretrained_sha256=__import__('hashlib').sha256(weights.read_bytes()).hexdigest())
    with ExitStack() as stack,fixture_budget(5),logged_admission(base/'admission'):
        stack.enter_context(patch.object(protocol,'validate_output_path',timing_output));guards(stack)
        loader_fn=train.build_train_loader;model_fn=train.build_model;criterion_fn=train.build_criterion
        def loader(actual):
            value,C=loader_fn(actual);assert C==2510 and value.num_workers==0
            value._observer.original=ShortTestEpoch(value.batch_sampler);o['loader']=value;return value,C
        def model(actual,num_classes,**kwargs):
            value=model_fn(actual,num_classes,**kwargs);o['model']=value
            def before(module,args):
                x=args[0];assert tuple(x.shape)==(64,3,*actual['data']['train']['images']['size'])
                inputs.append(dict(indices=list(o['loader'].latest_indices),input=fingerprint(x)))
                torch.cuda.synchronize();pending['start']=time.perf_counter()
            value.register_forward_pre_hook(before);return value
        def criterion(actual,num_classes,feat_dim):
            value=criterion_fn(actual,num_classes,feat_dim)
            value.register_forward_hook(lambda module,args,result:losses.append(float(result[0].detach().cpu())))
            return value
        def epoch(*a,**k):o['progress']=k['update_progress'];return engine(*a,**k)
        for method,fn in [('build_train_loader',loader),('build_model',model),('build_criterion',criterion),('train_one_epoch',epoch)]:
            stack.enter_context(patch.object(train,method,fn))
        source,start=inspect.getsourcelines(engine);target=start+next(i for i,line in enumerate(source) if 'for key, value in classification_accuracy(' in line)
        seen=set();assert sys.gettrace() is None
        def tracer(frame,event,arg):
            if frame.f_code is not engine.__code__:return None
            if event=='line' and frame.f_lineno==target:
                step=o['progress'].completed_primary_updates
                if step not in seen:
                    seen.add(step);torch.cuda.synchronize();steps.append(dict(update=step,seconds=time.perf_counter()-pending['start']))
            return tracer
        sys.settrace(tracer);stack.callback(sys.settrace,None)
        try:
            progress=train.run_lodo_training(cfg)
            assert progress.completed_primary_updates==5 and len(steps)==5 and progress.auxiliary_steps==(5 if arch=='resnet50' else 0)
            terminal,_=ck.read_checkpoint(active/'checkpoints/ckpt_terminal.pth')
            result.update(mapping_digest=terminal['identity']['map_digest'],num_classes=terminal['identity']['num_classes'],scientific_configuration=terminal['identity']['scientific_config'])
            timed=[r['seconds'] for r in steps if r['update']>=3];median=statistics.median(timed)
            result.update(status='PASS',steps=steps,input_records=inputs,losses=losses,measured_updates=[3,4,5],
                median_step_seconds=median,median_images_per_second=64/median,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                device_memory=dev._query('uuid,memory.total,memory.used,memory.free'),
                current_allocated_bytes=torch.cuda.memory_allocated(0),current_reserved_bytes=torch.cuda.memory_reserved(0),
                cuda_memory_info=list(torch.cuda.mem_get_info(0)),runtime_controls=runtime_controls(),
                elapsed_seconds=time.perf_counter()-started,target_free=True)
        except BaseException as error:
            result.update(status='FAIL',error_type=type(error).__name__,error=str(error));raise
        finally:write_once_json(base/'result.json',result)
    require_production_rejection(cfg)
    print(json.dumps({k:result[k] for k in ('status','architecture','policy','median_step_seconds','median_images_per_second')}),flush=True)
    return result

if __name__=='__main__':measure(*sys.argv[1:])
