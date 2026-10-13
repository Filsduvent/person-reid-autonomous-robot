"""Read-only Stage A discovery and complete-strategy comparisons; no selection writes."""
import json
import hashlib
from pathlib import Path

from reid.utils.lodo_evaluation import METRICS, metric_vector, _require


def _compatible_reference(row,result):
    _require(row.get('schema_version')==1 and 'source_dataset' in row and 'protocol' not in row and
             row.get('record_type')!='lodo_held_out_result', 'Not a Stage A single-source record')
    cfg=row['resolved_config'];model=cfg['model'];test=cfg['data']['test']['dataset']
    architecture='pcb' if model['name']=='pcb' else 'resnet50' if model['name']=='reid_baseline' and model['backbone']['name']=='resnet50' else None
    _require(architecture==result['architecture'] and row['architecture']==model['name'], 'Architecture/config mismatch')
    if architecture=='resnet50':
        head=model.get('head',{})
        _require(head.get('embedding_dim')==2048 and head.get('bnneck') is True and head.get('eval_feat')=='bn', 'Incompatible ResNet embedding branch')
    if architecture=='pcb':
        _require(model.get('variant')==result['variant'], 'PCB variant mismatch')
    _require('protocol' not in cfg and cfg['data']['train']['dataset']['name']==row['source_dataset'], 'Source identity/config mismatch')
    _require(row['target_dataset']==result['target'] and test['name']==result['target'] and row['source_dataset'] in result['sources'], 'Wrong cross-domain cell')
    _require(cfg['data']['test']['images']==result['representation']['images'] and
             cfg['data']['test']['aug']==result['representation']['aug'], 'Incompatible architecture-specific preprocessing')
    target=result['evaluation']['configuration']['target']
    expected=target['dataset']
    for key,value in expected.items():
        if key in ('metadata_sha256','provenance'): continue
        if key in ('lists','image_subdir') and key not in test: continue  # Original raw loader has fixed authorized defaults.
        _require(test.get(key)==value, f'Target descriptor mismatch: {key}')
    if test['name']=='cuhk03':
        _require(test.get('protocol')=='processed_partition' and test.get('split_id') is None and
                 test.get('provenance',{}).get('verification','unverified')=='unverified', 'Unsupported CUHK03 official-new claim')
    evaluation=cfg['eval']
    _require(evaluation['normalize_feat'] is True and evaluation['distance']=='euclidean' and
             evaluation['topk']==[1,5,10] and not evaluation.get('rerank',{}).get('enabled',False) and
             not evaluation.get('test_time_flip',False), 'Incompatible retrieval protocol')
    provenance=row.get('evaluation_provenance',{})
    # Current metadata read today cannot prove a historical run used that split.
    _require(provenance.get('target_metadata_hashes')==target['metadata_hashes'], 'Missing/different historical target split hashes')
    _require(provenance.get('camera_filter')=='exclude_same_pid_same_camera', 'Unverified historical camera filtering')
    units=row.get('metric_units','fraction')  # existing schema-v1 writer emits fractions
    metrics=metric_vector(row,units)
    _require(isinstance(row.get('checkpoint_path'),str) and bool(row['checkpoint_path']) and bool(row.get('experiment_id')), 'Missing Stage A artifact/run reference')
    return dict(source=row['source_dataset'],target=row['target_dataset'],metrics=metrics,
                checkpoint_path=row['checkpoint_path'],experiment_id=row['experiment_id'],
                evidence=dict(target_metadata_hashes=provenance['target_metadata_hashes'],
                              camera_filter=provenance['camera_filter'],metric_units=units,
                              source_model=model,source_test_images=cfg['data']['test']['images'],
                              source_test_aug=cfg['data']['test']['aug']))


def compare_stage_a(result,paths):
    """Never pick a latest duplicate, average fewer than three, or fill missing cells."""
    _require(result.get('record_type')=='lodo_held_out_result' and result.get('metric_units')=='fraction', 'Expected isolated Stage B result')
    sources=result['sources'];target=result['target'];stage_b=metric_vector(result['metrics'])
    cells={source:[] for source in sources};issues=[];excluded=[];seen=set()
    for raw in paths:
        try:
            path=Path(raw).resolve(strict=True)
            _require(path not in seen, 'Duplicate reference artifact path')
            seen.add(path)
            raw_bytes=path.read_bytes()
            row=json.loads(raw_bytes)
            _require(isinstance(row,dict), 'Stage A record must be an object')
            if row.get('record_type')=='lodo_held_out_result' or 'protocol' in row:
                raise ValueError('Stage B records cannot be Stage A references')
            source=row.get('source_dataset')
            if source not in sources or row.get('target_dataset')!=target:
                excluded.append(dict(path=str(path),reason='Outside the three required cross-domain cells'))
                continue
            reference=_compatible_reference(row,result)
            reference['artifact']=dict(path=str(path),sha256=hashlib.sha256(raw_bytes).hexdigest())
            cells[source].append(reference)
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
            issues.append(dict(path=str(raw),reason=str(error)))
    missing=[source for source in sources if not cells[source]]
    ambiguous=[source for source in sources if len(cells[source])>1]
    complete=not missing and not ambiguous and not issues
    references=[cells[source][0] for source in sources if len(cells[source])==1]
    report=dict(schema_version=1,record_type='lodo_stage_a_comparison',architecture=result['architecture'],fold=result['fold'],
                target=target,required_sources=sources,synthetic=result['synthetic'],authoritative=result['authoritative'],
                stage_b_artifact=result['artifacts']['result'],status='complete' if complete else 'incomplete',
                references=references,missing_sources=missing,ambiguous_sources=ambiguous,issues=issues,excluded=excluded,
                stage_b=stage_b,metric_units='fraction',difference_units='percentage_points',
                mean=None,best_per_metric_envelope=None,best_by_mAP_run=None,differences_pp=None,
                interpretation='Comparison of complete training strategies; data, exposure, budgets, losses and selection differ. No diversity-only causal claim.')
    if complete:
        mean={key:sum(row['metrics'][key] for row in references)/3 for key in METRICS}
        envelope={key:max(row['metrics'][key] for row in references) for key in METRICS}
        best=max(references,key=lambda row:(row['metrics']['mAP'],row['metrics']['Rank1']))
        report.update(mean=mean,best_per_metric_envelope=dict(metrics=envelope,is_single_run=False,
                      contributing_sources={key:[row['source'] for row in references if row['metrics'][key]==envelope[key]] for key in METRICS}),
                      best_by_mAP_run=best,differences_pp={name:{key:100*(stage_b[key]-values[key]) for key in METRICS}
                      for name,values in [('mean',mean),('envelope',envelope),('best_by_mAP',best['metrics'])]})
    return report
