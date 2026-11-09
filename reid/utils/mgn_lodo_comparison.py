"""Three fixed no-flip references and their mean; target-ranked bests are oracles."""
import json
import re
from pathlib import Path
from reid.utils.lodo_evaluation import metric_vector, METRICS, _require
from reid.utils.lodo_checkpoint import file_digest


def compatible(row,result):
    _require(row.get('record_type')=='mgn_single_source_reference' and type(row.get('reference_policy_version')) is int and row['reference_policy_version']==1,'Explicit no-flip supplementary identity required; original mean-flip matrix is not matched')
    source=row.get('source_dataset');cfg=row['resolved_config']
    _require(source in result['sources'] and row['target_dataset']==result['target'] and source!=result['target'],'Wrong source/target reference')
    _require(row['architecture']=='mgn' and cfg['model']['variant']==result['variant'] and cfg['model']['name']=='mgn','Wrong MGN variant')
    _require('protocol' not in cfg and cfg['data']['train']['dataset']['name']==source,'Source ownership mismatch')
    _require(cfg['data']['test']['images']==result['representation']['images'] and cfg['data']['test']['aug']==result['representation']['aug'],'Source pixel policy mismatch')
    _require(cfg['eval']['flip_aggregation']=='none' and cfg['eval']['normalize_feat'] and cfg['eval']['distance']=='euclidean' and not cfg['eval']['rerank']['enabled'],'Primary retrieval mismatch')
    provenance=row['evaluation_provenance'];target=result['evaluation']['configuration']['target']
    _require(provenance['target_metadata_hashes']==target['metadata_hashes'] and provenance['camera_filter']=='exclude_same_pid_same_camera' and provenance['target_bn_adaptation'] is False,'Target metadata/filter/BN evidence mismatch')
    _require(row['source_checkpoint_unchanged'] is True and re.fullmatch('[0-9a-f]{64}',row['checkpoint_sha256']),'Unpinned source checkpoint')
    _require(cfg['train'].get('epochs')==120 and cfg['train']['save'].get('metric')=='mAP' and cfg['sched']['milestones']==[60,90],'Source resolved budget/selection differs')
    _require(provenance.get('retrieval_code_sha256')==result['evaluation']['configuration'].get('retrieval_code_sha256') and bool(provenance.get('retrieval_code_sha256')),'Missing/different shared retrieval code identity')
    _require(row['training']['checkpoint_selection']=='periodic_source_test_mAP_best' and row['training']['budget']=={'unit':'completed_epochs','epochs':120},'Original source budget/selection provenance missing')
    return dict(source=source,target=result['target'],metrics=metric_vector(row),checkpoint_path=row['checkpoint_path'],
                checkpoint_sha256=row['checkpoint_sha256'],training=row['training'],evaluation_provenance=provenance)


def compare_stage_a(result,paths):
    _require(result.get('protocol')=='mgn_lodo_v2' and result.get('architecture')=='mgn','MGN v2 result required')
    cells={s:[] for s in result['sources']};issues=[];seen=set()
    for raw in paths:
        try:
            path=Path(raw).resolve(strict=True)
            _require(path not in seen,'Duplicate reference path');seen.add(path)
            row=json.loads(path.read_text());ref=compatible(row,result)
            ref['artifact']=dict(path=str(path),sha256=file_digest(path));cells[ref['source']].append(ref)
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
            issues.append(dict(path=str(raw),reason=str(error)))
    missing=[s for s,rows in cells.items() if not rows];ambiguous=[s for s,rows in cells.items() if len(rows)>1]
    complete=not missing and not ambiguous and not issues
    refs=[rows[0] for rows in cells.values() if len(rows)==1]
    report=dict(schema_version=1,record_type='mgn_lodo_v2_reference_comparison',status='complete' if complete else 'incomplete',
        required_sources=result['sources'],target=result['target'],references=refs,missing_sources=missing,ambiguous_sources=ambiguous,issues=issues,
        metric_units='fraction',difference_units='percentage_points',primary_reference_policy='all_three_and_arithmetic_mean',mean=None,
        retrospective_oracle_per_metric=None,retrospective_oracle_best_by_target_mAP=None,differences_pp=None,
        lodo_training=dict(budget={'unit':'successful_primary_updates','updates':85200},checkpoint_selection='predetermined_terminal'),
        oracle_interpretation='Retrospective target-ranked summaries, NOT target-independent source/checkpoint selection.',
        limitations=['Single seed42; whole training strategies/budgets/selection differ.','CUHK03 detected processed_partition; official-new correspondence unverified.'])
    if complete:
        mean={k:sum(r['metrics'][k] for r in refs)/3 for k in METRICS}
        envelope={k:max(r['metrics'][k] for r in refs) for k in METRICS}
        oracle=max(refs,key=lambda r:(r['metrics']['mAP'],r['metrics']['Rank1']))
        report.update(mean=mean,retrospective_oracle_per_metric=dict(metrics=envelope,is_single_run=False),
            retrospective_oracle_best_by_target_mAP=oracle,
            differences_pp={name:{k:100*(result['metrics'][k]-values[k]) for k in METRICS}
                            for name,values in [('mean',mean),('retrospective_oracle_envelope',envelope),('retrospective_oracle_run',oracle['metrics'])]})
    return report
