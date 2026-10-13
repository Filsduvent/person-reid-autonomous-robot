"""Sequential Phase 9R.3 controller; never launches official training."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from lodo_cuda_policy_qualification import campaign_path,clone_boundary
from lodo_cuda_admission_diagnostic import TITAN_UUID,AdmissionJournal,write_once_json
from reid.utils import lodo_checkpoint as ck


def tree_hashes(root):
    return {str(p.relative_to(root)):ck.file_digest(p) for p in root.rglob('*') if p.is_file()}


def numerical_gate(base,cases=('B','C')):
    rows=[]
    for case in cases:
        value=json.loads((base/('trace-'+case)/'trace.json').read_text());rows.extend(value['comparisons'])
    summary=[]
    for pair in sorted(set(r['pair'] for r in rows)):
        for step in (3,4,5):
            events=[r for r in rows if r['pair']==pair and r['update']==step]
            if not events:continue
            failures=[r for r in events if r['outside'] or not r['non_tensor_equal'] or not r['finite']]
            summary.append(dict(pair=pair,update=step,events=len(events),exact=all(r['nonexact_tensors']==0 and r['non_tensor_equal'] for r in events),
                tolerance_pass=not failures,max_abs=max(r['max_abs'] for r in events),
                first_divergence=next((dict(stage=r['stage'],outside=r['outside']) for r in events if r['nonexact_tensors'] or not r['non_tensor_equal']),None)))
    required=9 if cases==('B','C') else 4 # interrupted update3 plus resumed3/4/5
    if len(summary)!=required or not all(r['tolerance_pass'] for r in summary):raise ValueError('Numerical qualification failed: '+str(summary))
    return summary


def main(name):
    root=campaign_path(name);root.mkdir(exist_ok=False)
    write_once_json(root/'authorization.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),phase='9R.3',
        gpu_uuid=TITAN_UUID,scope='short deterministic ResNet50 qualification; PCB only after ResNet50 PASS',
        user_confirmation='Explicit exact UUID allocation and authorization in current user message',
        official_training=False,phase10=False))
    old=Path(__file__).resolve().parents[1]/'docs/lodo_cuda_policy_integration.md'
    shutil.copy2(old,root/'prior-cpu-policy-report.md')
    shutil.copy2('/tmp/lodo9r3-gpu-before.json',root/'preservation-before.json')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=TITAN_UUID,LODO_AUTHORIZED_GPU_UUID=TITAN_UUID,
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tests')
    command=[sys.executable,'-B','tests/lodo_cuda_policy_qualification.py','--campaign',name]
    journal=AdmissionJournal(root/'controller-admission')
    campaign=dict(status='RUNNING',campaign=name,uuid=TITAN_UUID,commands=[],architectures={},started_utc=datetime.now(timezone.utc).isoformat())
    previous={}
    def execute(arch,case):
        with __import__('unittest.mock',fromlist=['patch']).patch.dict(os.environ,env):journal.check(TITAN_UUID)
        cmd=command+['--architecture',arch,'--case',case]
        print('START',arch,case,flush=True);start=time.perf_counter()
        logfile=root/f'{arch}-{case}.log'
        with logfile.open('xb') as out:
            process=subprocess.Popen(cmd,env=env,cwd=root.parents[2],stdout=out,stderr=subprocess.STDOUT)
            record=dict(command=cmd,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','LODO_AUTHORIZED_GPU_UUID','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH')},pid=process.pid)
            campaign['commands'].append(record)
            try:code=process.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=30);raise RuntimeError('Diagnostic timeout; child terminated, no automatic retry')
        record.update(returncode=code,seconds=time.perf_counter()-start,log=str(logfile))
        print('END',arch,case,code,round(record['seconds'],2),flush=True)
        if code!=(-15 if case=='interrupt' else 0):raise RuntimeError(f'{arch}/{case} failed; see {logfile}')
        time.sleep(3)
    try:
        for arch in ('resnet50','pcb'):
            base=root/arch/'LODO-1';base.mkdir(parents=True);active=base/'active'
            if arch=='pcb':assert campaign['architectures']['resnet50']['status']=='PASS'
            execute(arch,'A')
            (active).rename(base/'branch-A')
            immutable=tree_hashes(base/'boundary');previous[arch]=immutable
            archive_a=tree_hashes(base/'branch-A')
            for case in ('B','C'):
                clone_boundary(base/'boundary',active);execute(arch,case);active.rename(base/('branch-'+case))
                assert tree_hashes(base/'boundary')==immutable
                assert tree_hashes(base/'branch-A')==archive_a
            table=numerical_gate(base);write_once_json(base/'abc-comparison.json',table)
            # A separate actual process is terminated after successful step3.
            clone_boundary(base/'boundary',active);execute(arch,'interrupt')
            marker=json.loads((base/'trace-interrupt/interruption-marker.json').read_text())
            assert marker==dict(completed=3,consistent=True,boundary_updates=2,accepted_images=192)
            assert not (active/'checkpoints/ckpt_terminal.pth').exists()
            execute(arch,'resume')
            replay=numerical_gate(base,('interrupt','resume'))
            execute(arch,'terminal')
            active.rename(base/'branch-recovered')
            assert tree_hashes(base/'boundary')==immutable and tree_hashes(base/'branch-A')==archive_a
            campaign['architectures'][arch]=dict(status='PASS',abc=table,replay=replay,process_interruption=marker,
                rollback_discarded_updates=1,rollback_discarded_images=64,terminal_noop=True,immutable_boundary_sha256=immutable,
                outcomes={case:json.loads((base/('trace-'+case)/'outcome.json').read_text()) for case in ('A','B','C','resume','terminal')})
            write_once_json(base/'qualification.json',campaign['architectures'][arch])
        campaign['status']='COMPLETED'
    except BaseException as error:
        campaign.update(status='STOPPED',error_type=type(error).__name__,error=str(error));raise
    finally:
        campaign['ended_utc']=datetime.now(timezone.utc).isoformat()
        write_once_json(root/'campaign-summary.json',campaign)
        print(json.dumps(dict(status=campaign['status'],root=str(root))),flush=True)

if __name__=='__main__':main(sys.argv[1])
