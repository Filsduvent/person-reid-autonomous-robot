"""Sequential exact-UUID TITAN Xp diagnostics; no PCB or target evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/addirakoze/environments/Reid/bin/python'
UUID='GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c'
from lodo_cuda_admission_diagnostic import AdmissionJournal, publish_final_status


def require_allocation(receipt_path):
    receipt=json.loads(Path(receipt_path).read_text())
    if (receipt.get('uuid')!=UUID or receipt.get('allocated') is not True or
        receipt.get('authorized') is not True or receipt.get('phase')!='phase9r1_titan_abc' or
        receipt.get('source') not in ('user','admin') or not receipt.get('confirmation_text')):
        raise ValueError('Explicit exact-UUID TITAN allocation confirmation missing')
    return receipt


def run(receipt_path,policy='original'):
    require_allocation(receipt_path)  # Before directories, inventory or any CUDA discovery.
    assert policy in ('original','deterministic')
    from lodo_cuda_abc_diagnostic import ck,torch
    assert not torch.cuda.is_initialized()
    campaign='phase9r1-abc-titan-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+policy
    base=ROOT/'exp/lodo_smoke'/campaign
    base.mkdir(exist_ok=False)
    shutil.copy2(__file__,base/'driver.py')
    shutil.copy2(receipt_path,base/'allocation-confirmation.json')
    os.environ.update(CUDA_VISIBLE_DEVICES=UUID,LODO_AUTHORIZED_GPU_UUID=UUID)
    journal=AdmissionJournal(base/'admission-parent')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tests',CUDA_VISIBLE_DEVICES=UUID,LODO_AUTHORIZED_GPU_UUID=UUID)
    if policy=='deterministic':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    summary=dict(non_authoritative=True,policy=policy,campaign=campaign,status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),runs=[],qualification=None,pcb_executed=False)
    def flush(): ck.atomic_write(base/'campaign-summary.json',lambda stream:stream.write(json.dumps(summary,indent=2).encode()))
    (base/'source-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'tests/lodo_cuda_abc_diagnostic.py',ROOT/'tests/test_lodo_cuda_abc_diagnostic.py',ROOT/'tests/lodo_cuda_admission_diagnostic.py',ROOT/'tests/lodo_cuda_titan_abc_campaign.py',ROOT/'tests/test_lodo_cuda_admission_diagnostic.py',ROOT/'reid/engine/train_loop.py',ROOT/'reid/utils/lodo_checkpoint.py',ROOT/'scripts/train.py']},indent=2))
    flush();print('CAMPAIGN '+str(base),flush=True)
    try:
        for path in ('A','B','C'):
            if path!='A':time.sleep(3)  # Fixed settling interval BEFORE the first next admission; never retry.
            audit=journal.check(UUID)
            assert audit['free_mib']>=7500,'Insufficient fresh headroom'
            command=[PYTHON,'-B','tests/lodo_cuda_abc_diagnostic.py','--campaign',campaign,'--path',path,'--policy',policy,'--gpu-uuid',UUID]
            item=dict(path=path,command=command,environment={k:env.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH','CUDA_VISIBLE_DEVICES','LODO_AUTHORIZED_GPU_UUID','CUBLAS_WORKSPACE_CONFIG')},admission=audit,started_utc=datetime.now(timezone.utc).isoformat())
            summary['runs'].append(item);flush();print('START '+path,flush=True)
            started=time.perf_counter()
            with (base/(path+'-console.log')).open('w') as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            item.update(returncode=result.returncode,wall_seconds=time.perf_counter()-started)
            outcome=base/path/'outcome.json'
            if outcome.exists():item['outcome']=json.loads(outcome.read_text())
            flush();print('FINISH '+path+' '+str(item.get('outcome',{}).get('status'))+' '+str(item['wall_seconds']),flush=True)
            assert result.returncode==0 and item['outcome']['status']=='PASS','Child execution failed; stop'
        traces={p:json.loads((base/p/'trace.json').read_text()) for p in ('A','B','C')}
        rows=traces['B']['comparisons']+traces['C']['comparisons']
        selected=[];qualified=True
        for pair in ('A/B','A/C','B/C'):
            first,second=pair.split('/')
            for update in (3,4,5):
                state=next(r for r in rows if r['pair']==pair and r['update']==update and r['stage']=='10_after_main')
                center=next(r for r in rows if r['pair']==pair and r['update']==update and r['stage']=='11_after_center')
                groups={}
                for name,predicate in [('model',lambda k:k.startswith('.model.')),('adam_exp_avg',lambda k:k.startswith('.adam.') and k.endswith('.exp_avg')),('adam_exp_avg_sq',lambda k:k.startswith('.adam.') and k.endswith('.exp_avg_sq')),('adam_step',lambda k:k.startswith('.adam.') and k.endswith('.step'))]:
                    tensors={k:v for k,v in state['tensors'].items() if predicate(k)}
                    assert tensors
                    groups[name]=dict(outside=sum(v['outside'] for v in tensors.values()),nonexact=sum(not v['exact'] for v in tensors.values()),max_abs=max(v['max_abs'] for v in tensors.values()),tensors=len(tensors))
                structural=[]
                for stage in ('05_batch','12_accounting'):
                    row=next(r for r in rows if r['pair']==pair and r['update']==update and r['stage']==stage)
                    structural.append(row['non_tensor_equal'] and row['outside']==0 and row['nonexact_tensors']==0)
                if update==3:
                    for stage in ('01_checkpoint','02_restored_modules','03_restored_optimizers','04_restored_scheduler_rng'):
                        row=next(r for r in rows if r['pair']==pair and r['update']==3 and r['stage']==stage)
                        structural.append(row['non_tensor_equal'] and row['outside']==0 and row['nonexact_tensors']==0)
                la=next(r for r in traces[first]['losses'] if r['update']==update)
                lb=next(r for r in traces[second]['losses'] if r['update']==update)
                assert la['terms'].keys()==lb['terms'].keys()
                loss_pass=all(bool(torch.isclose(torch.tensor(a),torch.tensor(b),rtol=.001,atol=.001)) for a,b in [(la['loss'],lb['loss'])]+[(la['terms'][k],lb['terms'][k]) for k in la['terms']])
                passed=(state['outside']==0 and center['outside']==0 and state['finite'] and center['finite'] and state['non_tensor_equal'] and center['non_tensor_equal'] and groups['adam_step']['nonexact']==0 and all(structural) and loss_pass)
                qualified &= passed
                selected.append(dict(pair=pair,update=update,groups=groups,center=dict(outside=center['outside'],nonexact=center['nonexact_tensors'],max_abs=center['max_abs']),structural_exact=all(structural),loss_original_pass=loss_pass,loss_actual=la['loss'],loss_reference=lb['loss'],passed=passed))
        summary.update(status='COMPLETED',qualification='PASS' if qualified else 'FAIL',comparison_summary=selected,common=json.loads((base/'common.json').read_text()),finished_utc=datetime.now(timezone.utc).isoformat())
        assert not torch.cuda.is_initialized()
    except BaseException as e:
        summary.update(status='STOPPED',error_type=type(e).__name__,error=str(e),finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        summary['artifact_bytes']=sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
        publish_final_status(base/'final-summary.json',summary,journal.snapshot)
        flush();print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation-receipt',required=True)
    parser.add_argument('--policy',choices=['original','deterministic'],default='original')
    args=parser.parse_args();run(args.allocation_receipt,args.policy)
