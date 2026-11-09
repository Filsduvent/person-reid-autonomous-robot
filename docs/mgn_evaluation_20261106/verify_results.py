"""Read-only final verification of the authorized MGN matrix and provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.mgn_gpu_smoke import file_hash
from reid.utils.config import load_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--expected-cells',type=int,choices=(4,16),default=16);args=parser.parse_args()
    suite=ROOT/args.suite
    registry=yaml.safe_load((suite/'registry.yaml').read_text())
    assert registry['status']==('COMPLETE_16_CELLS_ARTIFACT_VERIFIED_REVIEW_PENDING' if args.expected_cells==16 else 'FOUR_DIAGONALS_RECORDED_REVIEW_PENDING')
    sources=('market1501','duke','cuhk03','msmt17');keys=('mAP','mINP','Rank1','Rank5','Rank10')
    expected={(s,t) for s in sources for t in sources} if args.expected_cells==16 else {(s,s) for s in sources}
    assert {(c['source'],c['target']) for c in registry['cells']}==expected
    assert len(registry['cells'])==args.expected_cells
    original=yaml.safe_load((ROOT/'docs/mgn_training_results.yaml').read_text())
    import torch
    saved={}
    for source in sources:
        ckpt=registry['source_checkpoints'][source]
        assert ckpt['sha256']==original['sources'][source]['checkpoints']['ckpt_best.pth']['sha256']
        assert file_hash(ROOT/ckpt['path'])==ckpt['sha256']
        c=torch.load(ROOT/ckpt['path'],map_location='cpu',weights_only=True,mmap=True)
        saved[source]=c['cfg'];del c
    for cell in registry['cells']:
        source,target=cell['source'],cell['target'];cfg=load_config(str(ROOT/cell['output']/'config.resolved.yaml'))
        assert cell['checkpoint']['sha256']==registry['source_checkpoints'][source]['sha256']
        assert cell['epoch']==original['sources'][source]['final_best']['epoch']
        assert cfg['model']==saved[source]['model']
        assert cfg['data']['train']==saved[source]['data']['train']
        for section in ('images','aug','loader'):
            assert cfg['data']['test'][section]==saved[source]['data']['test'][section]
        assert {k:v for k,v in cfg['eval'].items() if k!='weight'}=={k:v for k,v in saved[source]['eval'].items() if k!='weight'}
        assert cfg['data']['test']['dataset']==load_config(f'configs/mgn/{target}.yaml')['data']['test']['dataset']
        for artifact in cell['artifacts']+cell['audit_artifacts']:
            p=ROOT/artifact['path'];assert p.stat().st_size==artifact['bytes'];assert file_hash(p)==artifact['sha256'],p
        record=json.loads((ROOT/cell['output']/'cross_dataset.json').read_text())
        assert (record['source_dataset'],record['target_dataset'],record['architecture'])==(source,target,'mgn')
        assert Path(record['checkpoint_path']).resolve()==(ROOT/cell['checkpoint']['path']).resolve()
        assert all(record[k]==cell['metrics'][k] and 0<=record[k]<=1 for k in keys)
        assert record['Rank1']<=record['Rank5']<=record['Rank10']
        gpu=cell['gpu_execution'];assert gpu['device']=='cuda:0' and gpu['fp32'] and not gpu['amp']
        assert gpu['forwards']==2*gpu['batches']
        assert gpu['gpu']['uuid']==registry['gpu_before']['uuid']
        if source==target:
            differences={k:cell['metrics'][k]-original['sources'][source]['final_best'][k] for k in keys}
            assert differences==cell['difference_from_training_fraction']
            assert cell['within_tolerance']==all(abs(d)<=registry['diagonal_tolerance_fraction'] for d in differences.values())
    csv_rows=list(csv.DictReader((suite/'matrix.csv').open()));assert len(csv_rows)==args.expected_cells
    lookup={(c['source'],c['target']):c for c in registry['cells']}
    for row in csv_rows:
        cell=lookup[row['source'],row['target']]
        assert all(float(row[k])==cell['metrics'][k] for k in keys)
    max_difference=max(abs(d) for c in registry['cells'] if c['source']==c['target'] for d in c['difference_from_training_fraction'].values())
    result=dict(status='ARTIFACT_INTEGRITY_PASS',unique_cells=args.expected_cells,standalone_diagonals=4,off_diagonals=args.expected_cells-4,
        numerical_consistency_exceptions=[c['source'] for c in registry['cells'] if not c.get('within_tolerance',True)],
        numerical_exception_disposition=registry.get('numerical_exception_approval','REVIEW_PENDING; threshold unchanged'),
        fixed_original_checkpoint_per_source=True,source_owned_preprocessing_and_retrieval_preserved=True,
        target_owned_dataset_definitions_preserved=True,all_recorded_artifact_hashes_match=True,
        cuda_fp32_original_flip_forward_placement_verified=True,
        diagonal_maximum_difference_fraction=max_difference,csv_arithmetic_and_units_verified=True,
        registry_sha256=file_hash(suite/'registry.yaml'),verifier_sha256=file_hash(Path(__file__).resolve()))
    print(yaml.safe_dump(result,sort_keys=False))


if __name__=='__main__':main()
