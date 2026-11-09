"""Execute the authorized original MGN 4x4 evaluation sequentially; never train."""
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.mgn_gpu_smoke import file_hash, source_provenance
from scripts.mgn_preflight import environment, evaluation_identity, verify_hashes
from reid.utils.config import load_config
from reid.utils.experiment_matrix import build_cross_dataset_record, write_cross_dataset_record
from reid.utils.lodo_device import audit_gpu

SOURCES = ('market1501','duke','cuhk03','msmt17')
KEYS = ('mAP','mINP','Rank1','Rank5','Rank10')
SUITE = ROOT/'results/mgn_selected_matrix/oneiros_20261106_attempt2'
PYTHON = '/home/addirakoze/environments/Reid/bin/python'
UUID = 'GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2'
TOLERANCE = 1e-5


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(yaml.safe_dump(json.loads(json.dumps(value)), sort_keys=False))
    temporary.replace(path)


def identity(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size, sha256=file_hash(path))


def publish(registry):
    write(SUITE/'registry.yaml', registry)
    write(ROOT/'docs/mgn_evaluation_results.yaml', registry)
    with (SUITE/'matrix.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=('source','target','epoch','checkpoint_sha256',*KEYS))
        writer.writeheader()
        for cell in registry['cells']:
            writer.writerow(dict(source=cell['source'], target=cell['target'], epoch=cell['epoch'],
                checkpoint_sha256=cell['checkpoint']['sha256'], **cell['metrics']))
    def table(metric):
        lookup={(c['source'],c['target']):c['metrics'][metric]*100 for c in registry['cells']}
        rows=['| Source → target | Market1501 | Duke | CUHK03 | MSMT17 |',
              '| --- | ---: | ---: | ---: | ---: |']
        for source in SOURCES:
            values=[f'{lookup[source,target]:.2f}' if (source,target) in lookup else 'pending' for target in SOURCES]
            rows.append('| '+source+' | '+' | '.join(values)+' |')
        return '\n'.join(rows)
    text='# MGN independent within-domain and cross-domain evaluation\n\n'
    text+=f"Status: **{registry['status']}**; {len(registry['cells'])}/16 cells recorded. Metrics below are percentages.\n\n"
    text+='Original four selected-best checkpoints only; one fixed source checkpoint across every target in its row. The separate40/60 schedule trial is excluded. GPU FP32 original/width-flip feature extraction, raw mean then global L2, source-owned RGB384x128 bicubic normalization, Euclidean distance, no reranking. Common CPU distance/ranking/metrics remain unchanged.\n\n'
    text+='## mAP\n\n'+table('mAP')+'\n\n## Rank-1\n\n'+table('Rank1')+'\n\n'
    text+='Full-precision Rank-1/5/10, mAP and mINP, checkpoint identities, commands, logs, GPU observations and source/target provenance are in [the registry](mgn_evaluation_results.yaml) and `results/mgn_selected_matrix/oneiros_20261106_attempt2/`. Diagonals use standalone `scripts/evaluate.py`; off-diagonals use `scripts/evaluate_cross_domain.py`.\n\n'
    text+='Periodic source-test mAP originally selected each checkpoint; standalone evaluation does not create an independent validation split. Single-seed results compare complete recipes. CUHK03 is detected/processed_partition; new/classic semantic correspondence remains unverified. Original Market1501 resumed from86 after partial87; no bitwise uninterrupted trajectory is claimed. No target-based checkpoint selection, tuning or training was performed. LODO remains a separate mandatory pending setting.\n'
    (ROOT/'docs/mgn_evaluation_matrix.md').write_text(text)


def main():
    os.chdir(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID or os.environ.get('MGN_AUTHORIZED_GPU_UUID') != UUID:
        raise RuntimeError('Explicit authorized Oneiros UUID required.')
    if SUITE.exists() or SUITE.is_symlink():
        raise RuntimeError('Suite output already exists; preserve it. No automatic resume/retry.')
    manifest=json.loads((ROOT/'docs/mgn_experiment_manifest.json').read_text())
    source_registry=yaml.safe_load((ROOT/'docs/mgn_training_results.yaml').read_text())
    if environment()!=manifest['environment']:
        raise RuntimeError('Qualified Python/package/runtime controls mismatch.')
    verify_hashes(manifest['runtime_source_sha256']); verify_hashes(manifest['qualification_receipt_sha256'])
    if file_hash('/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')!=manifest['weights_sha256']:
        raise RuntimeError('Pretrained artifact mismatch.')
    before_gpu=audit_gpu(UUID)
    checkpoints={};datasets={}
    from reid.data.build import build_train_loader, build_test_loader
    for source in SOURCES:
        item=source_registry['sources'][source]['checkpoints']['ckpt_best.pth']
        checkpoint=ROOT/item['path']
        if file_hash(checkpoint)!=item['sha256']:
            raise RuntimeError('Original checkpoint hash mismatch: '+source)
        checkpoints[source]=item
        entry=manifest['sources'][source]
        verify_hashes({entry['config_path']:entry['config_sha256']})
        for section in ('training','evaluation'):verify_hashes(entry[section]['metadata_sha256'])
        cfg=load_config(entry['config_path'])
        train,classes=build_train_loader(cfg);test=build_test_loader(cfg)
        tr=source_provenance(train.dataset,cfg);ev=evaluation_identity(test.dataset,cfg)
        if tr!=entry['training'] or ev!=entry['evaluation'] or classes!=item['reconstruction']['num_classes']:
            raise RuntimeError('Source/target data provenance mismatch: '+source)
        datasets[source]=dict(training=tr,evaluation=ev,config=identity(ROOT/entry['config_path']))
        del train,test
    SUITE.mkdir(parents=True)
    audit=SUITE/'_audit';audit.mkdir()
    git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    registry=dict(schema_version=1, status='RUNNING', authorized_user_instruction='ok you can do that and make sure you use GPU and go recording the results.',
       authorization_scope='Four original selected-best standalone diagonals and twelve off-diagonals; sequential GPU evaluation with incremental registration. No training/tuning/LODO/Part2/push/merge.',
       created_utc=datetime.now(timezone.utc).isoformat(), release_commit=git,
       source_checkpoint_selection='Original source-test periodic mAP best, fixed across targets; trial excluded',
       metric_units='fractions', diagonal_tolerance_fraction=TOLERANCE, environment=environment(), gpu_before=before_gpu,
       dataset_provenance=datasets, source_checkpoints=checkpoints,
       execution_sources={str(p.relative_to(ROOT)):file_hash(p) for p in (Path(__file__).resolve(),ROOT/'docs/mgn_evaluation_20261106/gpu_entry.py')},
       frozen_runtime_sha256=manifest['runtime_source_sha256'], manifest=identity(ROOT/'docs/mgn_experiment_manifest.json'), cells=[])
    publish(registry)
    pairs=[(s,s) for s in SOURCES]+[(s,t) for s in SOURCES for t in SOURCES if s!=t]
    try:
        for source,target in pairs:
            name=f'{source}_to_{target}';output=SUITE/name;cell_audit=audit/name
            if output.exists() or output.is_symlink() or cell_audit.exists():raise RuntimeError('Nonfresh cell '+name)
            if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=git:raise RuntimeError('Release changed during evaluation')
            verify_hashes(manifest['runtime_source_sha256'])
            checkpoint=checkpoints[source];path=ROOT/checkpoint['path']
            if file_hash(path)!=checkpoint['sha256']:raise RuntimeError('Checkpoint changed')
            verify_hashes(datasets[target]['evaluation']['metadata_sha256'])
            live=audit_gpu(UUID)
            cell_audit.mkdir()
            diagonal=source==target
            entry='scripts/evaluate.py' if diagonal else 'scripts/evaluate_cross_domain.py'
            if diagonal:
                arguments=['--config',f'configs/mgn/{source}.yaml','--weight',str(path),'--opts',f'experiment.output_dir={output}']
            else:
                arguments=['--config',f'configs/mgn/{target}.yaml','--checkpoint',str(path),'--source-dataset',source,'--target-dataset',target,'--output-dir',str(output)]
            command=[PYTHON,'-u','-B',str(ROOT/'docs/mgn_evaluation_20261106/gpu_entry.py'),'--audit',str(cell_audit),'--entry',entry,'--',*arguments]
            write(cell_audit/'admission.yaml',dict(source=source,target=target,checkpoint=checkpoint,command=command,
                  common_entry_arguments=[PYTHON,'-u','-B',entry,*arguments],gpu=live,release=git,output_fresh=True,
                  created_utc=datetime.now(timezone.utc).isoformat()))
            registry['current_cell']=name;publish(registry)
            print('START '+name,flush=True)
            with (cell_audit/'terminal.log').open('x') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
            if completed.returncode:raise RuntimeError(f'{name} evaluator exit {completed.returncode}; see preserved terminal.log')
            scores=json.loads((output/'metrics/eval_ckpt_best.json').read_text()) if diagonal else json.loads((output/'cross_dataset.json').read_text())
            cfg=load_config(str(output/'config.resolved.yaml'))
            if cfg['data']['train']['dataset']['name']!=source or cfg['data']['test']['dataset']['name']!=target:raise RuntimeError('Cell dataset identity changed')
            if cfg['eval']['flip_aggregation']!='mean' or cfg['eval']['rerank']['enabled'] or cfg['system']['device']!='cuda':raise RuntimeError('Retrieval/device policy changed')
            values={k:float(scores[k]) for k in KEYS}
            if not all(math.isfinite(v) and 0<=v<=1 for v in values.values()):raise RuntimeError('Nonfinite or out of range metrics')
            if not values['Rank1']<=values['Rank5']<=values['Rank10']:raise RuntimeError('Nonmonotonic ranks')
            if diagonal:
                row=build_cross_dataset_record(source_dataset=source,target_dataset=target,architecture='mgn',checkpoint_path=path,cfg=cfg,scores=values)
                write_cross_dataset_record(output/'cross_dataset.json',row)
            if file_hash(path)!=checkpoint['sha256']:raise RuntimeError('Checkpoint modified during evaluation')
            gpu_complete=json.loads((cell_audit/'gpu-completion.json').read_text())
            result=dict(source=source,target=target,setting='within-domain' if diagonal else 'cross-domain',
                epoch=checkpoint['epoch'],checkpoint=checkpoint,metrics=values,exit_code=completed.returncode,
                command=command,output=str(output.relative_to(ROOT)),gpu_execution=gpu_complete,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                artifacts=[identity(p) for p in sorted(output.rglob('*')) if p.is_file()],
                audit_artifacts=[identity(p) for p in sorted(cell_audit.rglob('*')) if p.is_file()])
            if diagonal:
                result['difference_from_training_fraction']={k:values[k]-source_registry['sources'][source]['final_best'][k] for k in KEYS}
                result['within_tolerance']=all(abs(v)<=TOLERANCE for v in result['difference_from_training_fraction'].values())
            write(cell_audit/'completion.yaml',result)
            registry['cells'].append(result)
            with (SUITE/'events.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            publish(registry)
            print('COMPLETE '+name+' '+json.dumps(values),flush=True)
            if diagonal and not result['within_tolerance']:raise RuntimeError('Diagonal exceeds preapproved tolerance; investigate without reselection or retrospective tolerance change')
            audit_gpu(UUID)
        if len({(c['source'],c['target']) for c in registry['cells']})!=16:raise RuntimeError('Missing or duplicate cells')
        registry['status']='COMPLETE_16_CELLS_ARTIFACT_VERIFIED_REVIEW_PENDING'
        registry['gpu_after']=audit_gpu(UUID);registry['completed_utc']=datetime.now(timezone.utc).isoformat();registry.pop('current_cell',None)
        publish(registry)
        print('COMPLETE ALL 16 CELLS',flush=True)
    except BaseException as error:
        registry['status']='STOPPED_PRESERVED_FAILURE'
        registry['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat())
        publish(registry)
        raise


if __name__=='__main__':
    main()
