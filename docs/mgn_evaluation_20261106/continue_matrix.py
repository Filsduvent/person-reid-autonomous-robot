"""Continue only previously unevaluated authorized MGN cells; preserve every attempt."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_matrix as run
from run_matrix import (ROOT,SUITE,PYTHON,UUID,SOURCES,KEYS,TOLERANCE,write,identity,publish,
    file_hash,load_config,audit_gpu,verify_hashes,environment,build_cross_dataset_record,write_cross_dataset_record)
import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('diagonals','cross'),required=True)
    parser.add_argument('--accept-numerical-exceptions',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    registry=yaml.safe_load((SUITE/'registry.yaml').read_text())
    if registry['status'].startswith('COMPLETE'):raise RuntimeError('Already complete; no rerun')
    if args.scope=='cross' and not args.accept_numerical_exceptions and any(not c.get('within_tolerance',True) for c in registry['cells']):
        raise RuntimeError('Cross-domain progression awaits explicit disposition of numerical exceptions')
    manifest=json.loads((ROOT/'docs/mgn_experiment_manifest.json').read_text())
    if environment()!=manifest['environment']:raise RuntimeError('Environment drift')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or os.environ.get('MGN_AUTHORIZED_GPU_UUID')!=UUID:raise RuntimeError('UUID visibility/authorization missing')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=registry['release_commit']:raise RuntimeError('Release changed')
    verify_hashes(registry['frozen_runtime_sha256']);verify_hashes(manifest['qualification_receipt_sha256'])
    if file_hash(ROOT/'docs/mgn_experiment_manifest.json')!=registry['manifest']['sha256']:raise RuntimeError('Manifest changed')
    for cell in registry['cells']:
        for item in cell['artifacts']+cell['audit_artifacts']:
            if file_hash(ROOT/item['path'])!=item['sha256']:raise RuntimeError('Preserved cell artifact changed')
    audit=SUITE/'_audit';checkpoints=registry['source_checkpoints'];datasets=registry['dataset_provenance']
    original=yaml.safe_load((ROOT/'docs/mgn_training_results.yaml').read_text())
    registry.setdefault('continuation_records',[]).append(dict(scope=args.scope,
       utc=datetime.now(timezone.utc).isoformat(),script=identity(Path(__file__).resolve()),
       prior_status=registry['status'], prior_failure=registry.get('failure'),
       numerical_exception_disposition='Explicit user-approved flagged cross-host exceptions; tolerance unchanged' if args.accept_numerical_exceptions else 'Independent remaining diagonals only; no cross-domain progression implied'))
    if args.accept_numerical_exceptions:
        registry['numerical_exception_approval']='User approved continuing the full matrix with flagged cross-host numerical exceptions; strict1e-5 threshold remains unchanged.'
    registry['status']='RUNNING';registry.pop('failure',None);publish(registry)
    pairs=[(s,s) for s in SOURCES] if args.scope=='diagonals' else [(s,t) for s in SOURCES for t in SOURCES if s!=t]
    done={(c['source'],c['target']) for c in registry['cells']}
    try:
        for source,target in pairs:
            if (source,target) in done:continue
            name=f'{source}_to_{target}';output=SUITE/name;cell_audit=audit/name
            if output.exists() or output.is_symlink() or cell_audit.exists():raise RuntimeError('Existing incomplete cell preserved; no automatic rerun '+name)
            verify_hashes(registry['frozen_runtime_sha256'])
            checkpoint=checkpoints[source];path=ROOT/checkpoint['path']
            if file_hash(path)!=checkpoint['sha256']:raise RuntimeError('Checkpoint changed')
            verify_hashes(datasets[target]['evaluation']['metadata_sha256'])
            live=audit_gpu(UUID);cell_audit.mkdir();diagonal=source==target
            entry='scripts/evaluate.py' if diagonal else 'scripts/evaluate_cross_domain.py'
            arguments=(['--config',f'configs/mgn/{source}.yaml','--weight',str(path),'--opts',f'experiment.output_dir={output}'] if diagonal else
               ['--config',f'configs/mgn/{target}.yaml','--checkpoint',str(path),'--source-dataset',source,'--target-dataset',target,'--output-dir',str(output)])
            command=[PYTHON,'-u','-B',str(HERE/'gpu_entry.py'),'--audit',str(cell_audit),'--entry',entry,'--',*arguments]
            write(cell_audit/'admission.yaml',dict(source=source,target=target,checkpoint=checkpoint,command=command,
               common_entry_arguments=[PYTHON,'-u','-B',entry,*arguments],gpu=live,release=registry['release_commit'],output_fresh=True,
               created_utc=datetime.now(timezone.utc).isoformat()))
            registry['current_cell']=name;publish(registry);print('START '+name,flush=True)
            with (cell_audit/'terminal.log').open('x') as log:completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
            if completed.returncode:raise RuntimeError(f'{name} evaluator exit {completed.returncode}; preserved terminal.log')
            scores=json.loads((output/'metrics/eval_ckpt_best.json').read_text()) if diagonal else json.loads((output/'cross_dataset.json').read_text())
            cfg=load_config(str(output/'config.resolved.yaml'))
            if cfg['data']['train']['dataset']['name']!=source or cfg['data']['test']['dataset']['name']!=target:raise RuntimeError('Cell identity changed')
            if cfg['eval']['flip_aggregation']!='mean' or cfg['eval']['rerank']['enabled'] or cfg['system']['device']!='cuda':raise RuntimeError('Policy changed')
            values={k:float(scores[k]) for k in KEYS}
            if not all(math.isfinite(v) and 0<=v<=1 for v in values.values()):raise RuntimeError('Invalid metrics')
            if not values['Rank1']<=values['Rank5']<=values['Rank10']:raise RuntimeError('Invalid ranks')
            if diagonal:write_cross_dataset_record(output/'cross_dataset.json',build_cross_dataset_record(source_dataset=source,target_dataset=target,architecture='mgn',checkpoint_path=path,cfg=cfg,scores=values))
            if file_hash(path)!=checkpoint['sha256']:raise RuntimeError('Checkpoint changed during inference')
            gpu=json.loads((cell_audit/'gpu-completion.json').read_text())
            result=dict(source=source,target=target,setting='within-domain' if diagonal else 'cross-domain',epoch=checkpoint['epoch'],checkpoint=checkpoint,
                metrics=values,exit_code=completed.returncode,command=command,output=str(output.relative_to(ROOT)),gpu_execution=gpu,
                completed_utc=datetime.now(timezone.utc).isoformat(),artifacts=[identity(p) for p in sorted(output.rglob('*')) if p.is_file()],
                audit_artifacts=[identity(p) for p in sorted(cell_audit.rglob('*')) if p.is_file()])
            if diagonal:
                result['difference_from_training_fraction']={k:values[k]-original['sources'][source]['final_best'][k] for k in KEYS}
                result['within_tolerance']=all(abs(v)<=TOLERANCE for v in result['difference_from_training_fraction'].values())
                result['verification_status']='PASS' if result['within_tolerance'] else 'FLAGGED_CROSS_HOST_NUMERICAL_DIFFERENCE'
            write(cell_audit/'completion.yaml',result);registry['cells'].append(result)
            with (SUITE/'events.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            publish(registry);print('COMPLETE '+name+' '+json.dumps(values),flush=True);audit_gpu(UUID)
        registry.pop('current_cell',None)
        if args.scope=='diagonals':registry['status']='FOUR_DIAGONALS_RECORDED_REVIEW_PENDING'
        else:
            if len({(c['source'],c['target']) for c in registry['cells']})!=16:raise RuntimeError('Incomplete coverage')
            registry['status']='COMPLETE_16_CELLS_ARTIFACT_VERIFIED_REVIEW_PENDING'
            registry['gpu_after']=audit_gpu(UUID);registry['completed_utc']=datetime.now(timezone.utc).isoformat()
        publish(registry);print(registry['status'],flush=True)
    except BaseException as error:
        registry['status']='STOPPED_PRESERVED_FAILURE';registry['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat())
        publish(registry);raise


if __name__=='__main__':main()
