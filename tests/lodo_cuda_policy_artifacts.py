"""Read-only CPU inspection of actual Phase 9R.3 CUDA-produced artifacts.

CUDA visibility declarations are retained for exact compatibility validation,
but no CUDA discovery, context, image decoding or target loader is permitted.
All destructive negative fixtures are ordinary copies in temporary directories.
"""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch
from reid.utils.lodo_cuda_policy import activate_policy
import torch
from lodo_cuda_policy_qualification import campaign_path
from lodo_cuda_abc_diagnostic import guards,fingerprint
from lodo_cuda_smoke import fixture_budget
from lodo_cuda_admission_diagnostic import write_once_json
from reid.utils import lodo_checkpoint as ck,lodo_recovery as recovery,lodo_protocol as protocol
from scripts import train


def rehash(path,payload,proof):
    torch.save(payload,path);proof=copy.deepcopy(proof);identity=payload['identity']
    proof.update(checkpoint_sha256=ck.file_digest(path),identity_digest=ck.digest(identity),
        fold=identity['protocol']['fold'],map_digest=identity['map_digest'],config_digest=identity['config_digest'],data_digest=identity['data_digest'])
    path.with_suffix('.json').write_bytes(ck.canonical_json(proof));ck.read_checkpoint(path)


def inspect_arch(base):
    started=time.perf_counter();terminal=base/'branch-A/checkpoints/ckpt_terminal.pth'
    payload,proof=ck.read_checkpoint(terminal);cfg=copy.deepcopy(payload['cfg']);activate_policy(cfg)
    assert not torch.cuda.is_initialized();rows=[]
    with fixture_budget(5),ExitStack() as stack,tempfile.TemporaryDirectory(prefix='lodo9r3-artifact-') as temporary:
        guards(stack)
        # Archived-run inspection never reserves/resumes/writes the active run.
        def inspected_output(actual,*,project_root=None):
            path=Path(actual['experiment']['output_dir'])
            if path!=base/'active' or path.resolve()!=path or '..' in path.parts:
                raise ValueError('Read-only artifact namespace mismatch')
            if project_root is not None and Path(project_root).resolve()!=Path(__file__).resolve().parents[1]:
                raise ValueError('Artifact project root mismatch')
            return path
        stack.enter_context(patch.object(protocol,'validate_output_path',inspected_output))
        def prohibited(*a,**k):raise AssertionError('CPU artifact gate attempted CUDA or state mutation')
        for name in ('init','_lazy_init','is_available','device_count'):
            stack.enter_context(patch.object(torch.cuda,name,prohibited))
        loader,C=train.build_train_loader(cfg);model=train.build_model(cfg,C,initialize_pretrained=False)
        criterion=train.build_criterion(cfg,C,train.validate_model_loss_requirements(cfg,model))
        optimizer=train.build_optimizer(cfg,model);aux=train.build_center_optimizer(cfg,criterion)
        scheduler=train.build_scheduler(cfg,optimizer,steps_per_epoch=len(loader))
        args=dict(cfg=cfg,loader=loader,model=model,criterion=criterion,optimizer=optimizer,auxiliary=aux,scheduler=scheduler,
                  cpu_inspection=True,validate_only=True)
        for obj in (model,criterion,getattr(criterion,'center_loss',None),optimizer,scheduler,aux):
            if obj is not None:stack.enter_context(patch.object(obj,'load_state_dict',prohibited))
        p,_,kind=ck.restore_lodo_checkpoint(terminal,**args)
        assert kind=='terminal' and p.completed_primary_updates==5
        rows.append(dict(gate='valid actual v2 terminal, CPU-first/no-mutation',status='PASS'))
        before=ck.digest(fingerprint(dict(model=model.state_dict(),criterion=criterion.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())))
        scratch=Path(temporary)
        for mutation in ('same_C_wrong_map','wrong_fold','wrong_config','wrong_policy','wrong_runtime'):
            altered=copy.deepcopy(payload);identity=altered['identity']
            if mutation=='same_C_wrong_map':
                first,second=identity['mapping']['identities'][:2]
                first['global_label'],second['global_label']=second['global_label'],first['global_label']
            if mutation=='wrong_fold':identity['protocol']['fold']='LODO-2'
            if mutation=='wrong_config':altered['cfg']['repro']['seed']+=1
            if mutation=='wrong_policy':identity['recovery']['environment']['execution_policy']['version']=99
            if mutation=='wrong_runtime':identity['recovery']['environment']['driver']='incompatible-driver'
            if mutation=='same_C_wrong_map':identity['map_digest']=ck.digest(identity['mapping'])
            path=scratch/(mutation+'.pth');rehash(path,altered,proof)
            try:ck.restore_lodo_checkpoint(path,**args)
            except ValueError as error:rows.append(dict(gate=mutation,status='REJECTED_BEFORE_MUTATION',reason=str(error)))
            else:raise AssertionError('Incompatible real payload accepted: '+mutation)
        after=ck.digest(fingerprint(dict(model=model.state_dict(),criterion=criterion.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())))
        assert before==after
        directory=scratch/'retention';source=base/'branch-A/checkpoints';shutil.copytree(source,directory)
        run=directory.parent;pointer=str(directory/'ckpt_last.pth')
        # select_generation requires the literal run/checkpoints directory name.
        actual_run=scratch/'retained-run';actual_run.mkdir();directory.rename(actual_run/'checkpoints');directory=actual_run/'checkpoints'
        request=copy.deepcopy(cfg);request['train']['save']['resume']=str(directory/'ckpt_last.pth')
        indexed=recovery._read_index(directory);assert len(indexed)==2
        latest=directory/indexed[-1]['name'];previous=directory/indexed[0]['name']
        chosen,invalid=recovery.select_generation(request,actual_run);assert chosen==latest and not invalid
        rows.append(dict(gate='two actual immutable generations and SHA256 index',status='PASS'))
        saved=latest.read_bytes();latest.write_bytes(b'corrupt diagnostic copy')
        chosen,invalid=recovery.select_generation(request,actual_run);assert chosen==previous and len(invalid)==1
        rows.append(dict(gate='corrupt newest falls back to validated prior generation',status='PASS'))
        latest.write_bytes(saved);latest.unlink()
        chosen,invalid=recovery.select_generation(request,actual_run);assert chosen==previous and len(invalid)==1
        rows.append(dict(gate='missing newest falls back to validated prior generation',status='PASS'))
        latest.write_bytes(saved)
        explicit=copy.deepcopy(request);explicit['train']['save']['resume']=str(latest)
        latest.write_bytes(b'corrupt explicit copy')
        try:recovery.select_generation(explicit,actual_run)
        except ValueError:rows.append(dict(gate='explicit corrupt generation rejects without fallback',status='PASS'))
        else:raise AssertionError('Explicit corrupt accepted')
        latest.write_bytes(saved)
        original_index=(directory/'recovery_index.json').read_bytes()
        real_payload,real_proof=ck.read_checkpoint(latest)
        orphan=directory/recovery.generation_name(real_payload['position']['next_epoch'],real_payload['progress']['completed_primary_updates'])
        shutil.copy2(latest,orphan);shutil.copy2(latest.with_suffix('.json'),orphan.with_suffix('.json'))
        original_atomic=ck.atomic_write
        def interrupted(path,writer):
            if Path(path).name=='recovery_index.json':raise OSError('deliberate index-publication interruption')
            return original_atomic(path,writer)
        with patch.object(ck,'atomic_write',interrupted):
            try:recovery.publish_generation(orphan,real_proof)
            except OSError:pass
            else:raise AssertionError('Publication interruption not reached')
        assert (directory/'recovery_index.json').read_bytes()==original_index
        chosen,invalid=recovery.select_generation(request,actual_run);assert chosen==latest and not invalid
        explicit['train']['save']['resume']=str(orphan)
        try:recovery.select_generation(explicit,actual_run)
        except ValueError:rows.append(dict(gate='interrupted publication preserves index and rejects orphan promotion',status='PASS'))
        else:raise AssertionError('Uncommitted orphan promoted')
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',actual_v2_cuda_payload=True,cuda_initialized=False,gates=rows,seconds=time.perf_counter()-started)


def main(name):
    root=campaign_path(name);summary=json.loads((root/'campaign-summary.json').read_text())
    assert summary['status']=='COMPLETED'
    results={}
    for arch in ('resnet50','pcb'):
        results[arch]=inspect_arch(root/arch/'LODO-1');print(arch,results[arch]['status'],len(results[arch]['gates']),flush=True)
    write_once_json(root/'real-artifact-integrity.json',results)

if __name__=='__main__':main(sys.argv[1])
