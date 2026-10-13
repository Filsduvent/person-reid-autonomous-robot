"""Preserve completed deterministic A and finish B/C after a postflight-only error.

No production validator changes. Check both frozen-policy rejection and short
budget/namespace rejection independently, instead of expecting one error order.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch
import lodo_cuda_abc_diagnostic as abc
from lodo_cuda_admission_diagnostic import AdmissionJournal, write_once_json
from lodo_cuda_titan_abc_campaign import require_allocation

ROOT = abc.ROOT
ORIGIN = ROOT/'exp/lodo_smoke/phase9r1-abc-titan-20261009T234014-deterministic'
UUID = abc.TITAN_UUID
PYTHON = '/home/addirakoze/environments/Reid/bin/python'
ORIGINAL_REJECTION = abc.require_production_rejection


def checked_rejection(cfg):
    if cfg['repro']['deterministic'] is True and cfg['repro']['benchmark'] is False:
        try:
            abc.protocol.scientific_configuration(cfg)
        except ValueError as error:
            if 'repro.deterministic' not in str(error):
                raise AssertionError('Unexpected frozen-policy rejection') from error
        else:
            raise AssertionError('Production accepted changed deterministic policy')
        budget = copy.deepcopy(cfg)
        budget['repro'].update(deterministic=False, benchmark=True)
        ORIGINAL_REJECTION(budget)
    else:
        ORIGINAL_REJECTION(cfg)


def validate_completed_a(base):
    outcome = json.loads((base/'A/outcome.json').read_text())
    required = dict(status='FAIL', policy='deterministic', producer_never_restored=True,
                    successful_updates=5, center_steps=5, accepted_images=320,
                    error_type='AssertionError', error='')
    if any(outcome.get(k) != v for k,v in required.items()):
        raise ValueError('A failure is not the validated postflight-only case')
    log = (base/'A-console.log').read_text()
    if "assert 'max_updates' in str(error)" not in log or 'repro.deterministic must be False, got True' not in log:
        raise ValueError('A failure provenance differs')
    common,checkpoint,payload,proof = abc.read_common(base,'deterministic',UUID)
    terminal,terminal_proof = abc.ck.read_checkpoint(checkpoint.parent/'ckpt_terminal.pth')
    abc.recovery.validate_boundary(terminal)
    if terminal['kind'] != 'terminal' or terminal['progress']['completed_primary_updates'] != 5:
        raise ValueError('Completed A terminal missing')
    if terminal['identity'] != payload['identity']:
        raise ValueError('A common/terminal identity differs')
    trace = json.loads((base/'A/trace.json').read_text())
    if len(trace['events']) != 352 or len(trace['losses']) != 3 or trace['comparisons']:
        raise ValueError('A trace incomplete')
    for event in trace['events']:
        state = torch.load(base/'A'/f"u{event['update']:03d}-{event['stage']}.pth",map_location='cpu',weights_only=False)
        if abc.ck.digest(abc.fingerprint(state)) != event['sha256']:
            raise ValueError('A event integrity differs')
    last = torch.load(base/'A/u005-10_after_main.pth',map_location='cpu',weights_only=False)
    center = torch.load(base/'A/u005-11_after_center.pth',map_location='cpu',weights_only=False)
    if not abc.nested_equal(last['model'],terminal['model']) or not abc.nested_equal(center['center'],terminal['center']):
        raise ValueError('A terminal differs from traced state')
    named_adam = {g['param_name']: terminal['optimizer']['state'][g['params'][0]]
                  for g in terminal['optimizer']['param_groups']}
    accounting = torch.load(base/'A/u005-12_accounting.pth',map_location='cpu',weights_only=False)
    if (not abc.nested_equal(named_adam, last['adam']) or
        not abc.nested_equal(terminal['center_optimizer'], center['optimizer']) or
        not abc.nested_equal(terminal['scheduler'], accounting['scheduler']) or
        not abc.nested_equal(abc.fingerprint(terminal['rng']), accounting['rng'])):
        raise ValueError('A terminal optimizer/scheduler/RNG differs from trace')
    if common['sha256'] != outcome['common_sha256']:
        raise ValueError('A common checkpoint differs')
    checked_rejection(terminal['cfg'])
    if torch.cuda.is_initialized(): raise ValueError('Validation initialized CUDA')
    return dict(common=common,terminal_sha256=terminal_proof['checkpoint_sha256'],
                a_events_verified=352,a_runtime_completed=True,a_original_operational_status='FAIL',
                postflight_defect='Expected max_updates rejection before frozen deterministic-policy rejection')


def child(token,path):
    if path not in ('B','C'): raise ValueError('Only B/C continuation allowed')
    with patch.object(abc,'require_production_rejection',checked_rejection):
        return abc.run(ORIGIN.name,path,'deterministic',UUID,token)


def run():
    require_allocation(ORIGIN/'allocation-confirmation.json')
    proof = validate_completed_a(ORIGIN)
    history = {str(p.relative_to(ORIGIN)):abc.ck.file_digest(p) for p in ORIGIN.rglob('*') if p.is_file()}
    token = 'bc-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    parent=ORIGIN/'continuations'
    if parent.resolve()!=parent: raise ValueError('Aliased continuation namespace')
    parent.mkdir(exist_ok=True)
    base=ORIGIN/'continuations'/token; base.mkdir(exist_ok=False)
    write_once_json(base/'validated-A.json',proof)
    write_once_json(base/'preserved-A-campaign-sha256.json',history)
    os.environ.update(CUDA_VISIBLE_DEVICES=UUID,LODO_AUTHORIZED_GPU_UUID=UUID)
    journal=AdmissionJournal(base/'admission-parent')
    summary=dict(non_authoritative=True,status='RUNNING',policy='deterministic',runs=[],
                 common=proof['common'],a_validation=proof,qualification=None,pcb_executed=False)
    print('CAMPAIGN '+str(base),flush=True)
    try:
        for path in ('B','C'):
            if path=='C': time.sleep(3)
            audit=journal.check(UUID)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH='.:tests',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            command=[PYTHON,'-B',str(Path(__file__).resolve()),'--child',path,'--token',token]
            start=time.perf_counter()
            with (base/(path+'-console.log')).open('x') as log:
                process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            outcome=json.loads((base/path/'outcome.json').read_text())
            summary['runs'].append(dict(path=path,command=command,admission=audit,returncode=process.returncode,
                                      wall_seconds=time.perf_counter()-start,outcome=outcome))
            print('FINISH '+path+' '+outcome['status'],flush=True)
            if process.returncode or outcome['status']!='PASS': raise RuntimeError('Continuation failed; stop')
        traces={p:json.loads(((ORIGIN if p=='A' else base)/p/'trace.json').read_text()) for p in ('A','B','C')}
        rows=traces['B']['comparisons']+traces['C']['comparisons']
        selected=[]
        for pair in ('A/B','A/C','B/C'):
            for update in (3,4,5):
                compared=[r for r in rows if r['pair']==pair and r['update']==update]
                state=next(r for r in compared if r['stage']=='10_after_main')
                center=next(r for r in compared if r['stage']=='11_after_center')
                groups={}
                for name,predicate in [('model',lambda k:k.startswith('.model.')),
                    ('adam_exp_avg',lambda k:k.startswith('.adam.') and k.endswith('.exp_avg')),
                    ('adam_exp_avg_sq',lambda k:k.startswith('.adam.') and k.endswith('.exp_avg_sq')),
                    ('adam_step',lambda k:k.startswith('.adam.') and k.endswith('.step'))]:
                    tensors=[v for k,v in state['tensors'].items() if predicate(k)]
                    if not tensors: raise ValueError('Missing comparison group')
                    groups[name]=dict(outside=sum(v['outside'] for v in tensors),nonexact=sum(not v['exact'] for v in tensors),max_abs=max(v['max_abs'] for v in tensors))
                structural=[r for r in compared if r['stage'] in ('01_checkpoint','02_restored_modules','03_restored_optimizers','04_restored_scheduler_rng','05_batch','12_accounting')]
                structural_exact=all(r['non_tensor_equal'] and not r['outside'] and not r['nonexact_tensors'] for r in structural)
                earlier,later=pair.split('/')
                la=next(x for x in traces[earlier]['losses'] if x['update']==update)
                lb=next(x for x in traces[later]['losses'] if x['update']==update)
                loss_pass=all(abs(la['terms'][k]-lb['terms'][k])<=.001+.001*abs(lb['terms'][k]) for k in la['terms'])
                passed=(state['non_tensor_equal'] and center['non_tensor_equal'] and state['finite'] and center['finite'] and
                        not state['outside'] and not center['outside'] and groups['adam_step']['nonexact']==0 and structural_exact and loss_pass)
                selected.append(dict(pair=pair,update=update,groups=groups,center=dict(outside=center['outside'],nonexact=center['nonexact_tensors']),
                    structural_exact=structural_exact,loss_original_pass=loss_pass,passed=passed,
                    all_events_exact=all(r['non_tensor_equal'] and not r['nonexact_tensors'] for r in compared)))
        summary.update(status='COMPLETED',qualification='PASS' if all(r['passed'] for r in selected) else 'FAIL',comparison_summary=selected)
    except BaseException as error:
        summary.update(status='STOPPED',error_type=type(error).__name__,error=str(error));raise
    finally:
        changed=[p for p,h in history.items() if abc.ck.file_digest(ORIGIN/p)!=h]
        summary['preserved_campaign_changed']=changed
        if changed:summary['status']='STOPPED'
        try:summary['final_audit']=journal.snapshot()
        except BaseException as error:summary.update(status='STOPPED',final_audit_error=str(error))
        summary['parent_cuda_initialized']=torch.cuda.is_initialized()
        write_once_json(base/'final-summary.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='runs'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--child',choices=['B','C']);parser.add_argument('--token')
    args=parser.parse_args()
    if args.child:child(args.token,args.child)
    else:run()
