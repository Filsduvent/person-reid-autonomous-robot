"""Preserve observed bounded evidence; refuse incomplete/failed qualification."""
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/mgn_lodo_phase21'
CAMPAIGN=ROOT/'exp/mgn_lodo_qualification/mgn_lodo_v2/phase21-oneiros-20261109-attempt1'
ACTIVE=CAMPAIGN/'attempt-2'
manifest=json.loads((ACTIVE/'campaign.json').read_text())
assert manifest['status']=='PASS' and len(manifest['cases'])==12
assert all(row['exit_code']==0 and row['aggregate_optimizer_entries']<=3 for row in manifest['cases'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proofs=[]
for attempt,base in [(1,CAMPAIGN),(2,ACTIVE)]:
    for source in sorted(base.rglob('*')):
        if attempt==1 and source.is_relative_to(ACTIVE):continue
        if source.is_file() and source.suffix in ('.json','.txt'):
            target=DOC/'evidence'/f'attempt-{attempt}'/source.relative_to(base)
            if source.suffix=='.json':target=target.with_suffix('.json.txt')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            proofs.append(dict(source=str(source.relative_to(ROOT)),preserved_copy=str(target.relative_to(ROOT)),sha256=sha(target)))
initial_entries=sum(json.loads(p.read_text())['optimizer_entries'] for p in CAMPAIGN.glob('LODO-*/*/optimizer-ledger.json'))
assert initial_entries==0
rows=[]
for fold in range(1,5):
    forward=json.loads((ACTIVE/f'LODO-{fold}/forward/fresh-outcome.json').read_text())
    continuous=json.loads((ACTIVE/f'LODO-{fold}/rollback/continuous-outcome.json').read_text())
    replay=json.loads((ACTIVE/f'LODO-{fold}/rollback/replay-outcome.json').read_text())
    assert all(r['status']=='PASS' for r in (forward,continuous,replay))
    assert replay['restored_continuation_bitwise_equal'] and replay['retained_generation_count']==2
    assert len(forward['input_batches'])==3 and len(continuous['input_batches'])==2 and len(replay['input_batches'])==1
    assert forward['cuda_admission']['hardware_compute_capability']==[8,6]
    assert forward['numerical_policy']==dict(amp=False,convolution_tf32_permitted=True,full_fp32_arithmetic_guaranteed=False,matmul_tf32_permitted=False,tensor_dtype='float32')
    for kind in ('forward','rollback'):
        assert json.loads((ACTIVE/f'LODO-{fold}/{kind}/optimizer-ledger.json').read_text())['optimizer_entries']==3
    rows.append(dict(fold=f'LODO-{fold}',classes=forward['actual_classes'],status='PASS_BOUNDED_QUALIFICATION',forward_optimizer_entries=3,recovery_optimizer_entries_aggregate=3,restored_continuation_bitwise_equal=True,forward_peak_reserved_GiB=forward['peak_reserved_bytes']/2**30,recovery_peak_reserved_GiB=max(continuous['peak_reserved_bytes'],replay['peak_reserved_bytes'])/2**30,scientific_sha256=forward['scientific_sha256'],preflight_sha256=forward['preflight_sha256']))
index=json.loads((DOC/'real_index_rng.json').read_text())
assert len(index['folds'])==4 and all(r['next_full_metadata_epoch_exact'] for r in index['folds'])
report=dict(status='PASS_BOUNDED_MGN_ONEIROS_QUALIFICATION_NOT_EXPERIMENTAL_VALIDATION',executed_runtime_release=manifest['release_commit'],source_sha256=manifest['source_sha256'],harness_sha256=manifest['harness_sha256'],gpu_uuid=manifest['gpu_uuid'],gpu_name='NVIDIA RTX A4000',compute_capability=[8,6],approval_reference=manifest['approval_reference'],matrix_cases_passed=8,workers_passed=12,initial_attempt_optimizer_entries=initial_entries,corrected_attempt_optimizer_entries=24,max_optimizer_entries_per_case_attempt=3,max_worker_seconds=600,attempts_used=2,max_attempts=3,cpu_regressions='282 passed in24.22s',real_index_boundaries=index,review='docs/mgn_lodo_phase21/bounded_recovery_review.md',folds=rows,evidence=proofs,official_training_launched=False,benchmark_evaluations_launched=0,caveats=['One-batch fixture recovery is not a full actual-source CUDA epoch or long-run guarantee.','Float32 tensors and disabled AMP do not imply full-FP32 arithmetic.','Single seed and CUHK03 provenance limitations preserved.','Initial fixture-only zero-update failures retained; no capacity retry or recipe change.'])
(DOC/'qualification_results.yaml').write_text(yaml.safe_dump(report,sort_keys=False))
print('All8 cases PASS,24 optimizer entries; preserved',len(proofs),'evidence files.')
