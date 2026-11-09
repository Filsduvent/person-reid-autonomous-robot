"""Issue evidence-bound private launch files after observed bounded PASS only.

No trainer import, CUDA initialization, optimizer call or official launch.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from reid.utils import mgn_lodo_operational as operational, mgn_lodo_protocol as science
from reid.utils.config import load_config
from scripts.prepare_mgn_lodo_launch import prepare

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/mgn_lodo_phase21'
report=yaml.safe_load((DOC/'qualification_results.yaml').read_text())
assert report['status']=='PASS_BOUNDED_MGN_ONEIROS_QUALIFICATION_NOT_EXPERIMENTAL_VALIDATION'
assert report['matrix_cases_passed']==8 and report['corrected_attempt_optimizer_entries']==24
snapshot=operational.source_snapshot();operational.validate_committed_sources(ROOT,snapshot)
assert snapshot==report['source_sha256']
assert hashlib.sha256((ROOT/'tests/mgn_lodo_qualification.py').read_bytes()).hexdigest()==report['harness_sha256']
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff',report['executed_runtime_release'],head,'--','reid','scripts','configs/lodo/mgn_lodo_v2','tests/mgn_lodo_qualification.py'],cwd=ROOT)
assert not __import__('torch').cuda.is_initialized()
runtime=operational.environment()
proofs={str(DOC/name):hashlib.sha256((DOC/name).read_bytes()).hexdigest() for name in ['qualification_results.yaml','bounded_recovery_review.md','recovery_and_regression_cpu_tests.txt','real_index_rng.json','final_gpu_inventory.yaml','prepare_qualified_handoff.py']}
for row in report['evidence']:
    proof=ROOT/row['preserved_copy']
    assert hashlib.sha256(proof.read_bytes()).hexdigest()==row['sha256']
    proofs[str(proof)]=row['sha256']
folder=ROOT/'.local/mgn_lodo_launch/qualifications';folder.mkdir(parents=True,exist_ok=True,mode=0o700)
receipts=[]
for row in report['folds']:
    fold=int(row['fold'].split('-')[1]);cfg=load_config(str(ROOT/f'configs/lodo/mgn_lodo_v2/mgn_lodo-{fold}_seed42.yaml'))
    assert science.configuration_identity(cfg)==row['scientific_sha256']
    record=dict(schema_version=1,status='MGN_LODO_FOLD_QUALIFIED',fold=row['fold'],config_digest=row['scientific_sha256'],release_commit=head,source_sha256=snapshot,preflight_sha256=row['preflight_sha256'],environment=runtime,gpu_uuid=report['gpu_uuid'],capabilities=['fresh_fp32','completed_epoch_rollback','terminal_noop'],evidence_sha256=proofs,recovery_scope=dict(boundary_evidence='bounded_fixture',equivalence_review_reference=str(DOC/'bounded_recovery_review.md')))
    path=folder/f"{row['fold']}.json"
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(record,stream,indent=2)
    destination=prepare(fold,'official-20261109',path,'User 2026-11-09: official training launched by user only; prepared command invoked manually by user')
    receipts.append(dict(fold=row['fold'],qualification_path=str(path),qualification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config_path=str(destination/'config.yaml'),config_sha256=hashlib.sha256((destination/'config.yaml').read_bytes()).hexdigest(),execution_path=str(destination/'execution.json'),execution_sha256=hashlib.sha256((destination/'execution.json').read_bytes()).hexdigest(),training_launched=False))
assert not __import__('torch').cuda.is_initialized()
receipt=dict(status='FOUR_USER_LAUNCH_SCOPES_PREPARED_NOT_EXECUTED',reviewed_release=head,executed_runtime_release=report['executed_runtime_release'],runtime_bytes_identical=True,scope='Evidence review and handoff only; official training is launched by user.',folds=receipts,official_training_launched=False)
(DOC/'launch_preparation_receipt.yaml').write_text(yaml.safe_dump(receipt,sort_keys=False))
print('Prepared and admitted four user-operated launch scopes. Official launches:0.',flush=True)
