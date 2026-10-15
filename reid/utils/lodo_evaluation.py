"""Post-training lodo_v1 evaluation. No training/recovery or ranking implementation."""
import copy
import json
import platform
from pathlib import Path
import re

import torch

from reid.data.build import build_test_loader
from reid.data.multi_source import canonical_json
from reid.engine.evaluator import evaluate_reid
from reid.utils.checkpoint import reconstruct_model, make_reconstruction_metadata, normalized_model_state
from reid.utils.evaluation_config import build_evaluation_config
from reid.utils.lodo_checkpoint import read_checkpoint, digest, file_digest, atomic_write
from reid.utils.lodo_protocol import scientific_configuration, configuration_identity, validate_fold

RESULT_VERSION = 1
METRICS = ('mAP', 'mINP', 'Rank1', 'Rank5', 'Rank10')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_request(request):
    required = {'mode','version','architecture','fold','target','checkpoint','expected','output_dir','synthetic'}
    _require(isinstance(request,dict) and required <= request.keys(), 'Missing LODO evaluation request fields')
    _require(not request.keys()-required-{'data_root','batch_size','num_workers','stage_a_records','device'}, 'Unknown LODO evaluation fields')
    from reid.utils.lodo_evaluation_device import execution_spec
    execution_spec(request)
    _require(request['mode']=='lodo_held_out' and type(request['version']) is int and request['version']==1, 'Unsupported LODO evaluation mode/version')
    _require(type(request['synthetic']) is bool, 'synthetic must be explicit Boolean')
    _require(request['architecture'] in ('resnet50','pcb'), 'Unknown architecture')
    expected=request['expected']
    _require(isinstance(expected,dict) and expected.keys()=={'config_digest','map_digest','data_digest','num_classes'}, 'Expected frozen identities are required')
    for key in ('config_digest','map_digest','data_digest'):
        _require(isinstance(expected[key],str) and re.fullmatch('[0-9a-f]{64}',expected[key]), f'Invalid {key}')
    _require(type(expected['num_classes']) is int and expected['num_classes']>1, 'Invalid combined C')
    for key,default,minimum in (('batch_size',32,1),('num_workers',0,0)):
        value=request.get(key,default)
        _require(type(value) is int and value>=minimum, f'Invalid {key}')
    _require(isinstance(request.get('stage_a_records',[]),list), 'Stage A paths must be a list')


def validate_terminal(request):
    """No model construction/loading or target loader is permitted in this gate."""
    validate_request(request)
    path=Path(request['checkpoint']).expanduser().absolute()
    _require(path.name=='ckpt_terminal.pth' and path.resolve()==path, 'A canonical terminal checkpoint is required')
    _require(path.with_suffix('.json').resolve()==path.with_suffix('.json'), 'Terminal sidecar alias prohibited')
    payload,manifest=read_checkpoint(path)  # Phase 6 reader/schema, unchanged
    _require(payload['kind']=='terminal' and manifest['completion_reason']=='budget_completed', 'Not a completed terminal artifact')
    identity=payload['identity'];cfg=payload['cfg'];science=scientific_configuration(cfg)
    architecture=request['architecture'];budget=85200 if architecture=='resnet50' else 94200
    auxiliary=budget if architecture=='resnet50' else 0
    validate_fold(request['fold'],identity['sources'],request['target'])
    _require(identity['protocol']==cfg['protocol']==science['protocol'] and
             cfg['protocol']['fold']==request['fold'] and cfg['protocol']['architecture']==architecture and
             identity['target']==request['target'], 'Requested fold/target/architecture mismatch')
    _require(identity['scientific_config']==science and identity['config_digest']==configuration_identity(cfg), 'Frozen training config mismatch')
    for key,value in request['expected'].items():
        _require(identity[key]==value, f'Expected {key} mismatch')
    _require(identity['precision']=='fp32' and type(identity['schema_version']) is int and identity['schema_version']==payload['lodo_schema_version'] and identity['schema_version'] in (1,2), 'Unsupported precision/identity schema')
    _require(digest(identity['mapping'])==identity['map_digest'] and digest(identity['provenance'])==identity['data_digest'], 'Mapping/data provenance digest mismatch')
    mapping=identity['mapping'];rows=mapping['identities'];classes=identity['num_classes']
    _require(mapping['schema_version']==1 and mapping['num_classes']==classes and mapping['ordered_sources']==identity['sources'], 'Mapping schema/source/class mismatch')
    _require(len(rows)==classes and [r['global_label'] for r in rows]==list(range(classes)), 'Noncontiguous global labels')
    position=0
    for source in identity['sources']:
        selected=[r for r in rows if r['source_key']==source]
        pids=[r['source_exposed_pid'] for r in selected]
        _require(all(type(pid) is int and pid>=0 for pid in pids) and pids==sorted(set(pids)), 'Invalid PID membership/order')
        count=mapping['source_class_counts'][source]
        _require(count==len(selected) and selected==rows[position:position+count] and
                 sorted(r['local_label'] for r in selected)==list(range(count)), 'Mapping source/local labels mismatch')
        position+=count
    _require(position==classes, 'Mapping contains unknown sources')
    dim=2048 if architecture=='resnet50' else 1536
    declaration=make_reconstruction_metadata(cfg,classes,dim)
    _require(identity['feature_dim']==dim and identity['reconstruction']==payload['reconstruction']==declaration, 'Reconstruction C/D/variant mismatch')
    p=payload['progress']
    _require(p['consistent'] is True and p['failed'] is False and p['auxiliary']==(architecture=='resnet50'), 'Failed/inconsistent terminal state')
    for key in ('max_updates','completed_primary_updates','main_steps_returned','attempted_batches','forward_batches','backward_batches'):
        _require(type(p[key]) is int and p[key]==budget, f'Invalid terminal {key}')
    _require(type(p['auxiliary_steps']) is int and type(p['auxiliary']) is bool and p['auxiliary_steps']==auxiliary and manifest['actual_auxiliary_updates']==auxiliary and p['attempted_images']==64*budget, 'Terminal auxiliary/image count mismatch')
    _require(p['unattributed_attempted_images']==0 and p['unattributed_attempted_batches']==0, 'Unattributed failed attempts')
    _require(set(identity['provenance'])=={'market1501','duke','cuhk03','msmt17'}, 'Missing source/target provenance')
    _require(re.fullmatch('[0-9a-f]{64}',identity['sidecar_digest']) is not None, 'Invalid sidecar digest')
    identities={r['global_label']:r for r in rows}
    for phase in ('attempted','forward','backward','successful'):
        exposure=p['exposure'];source_counts=exposure['presentations'][phase];id_counts=exposure['identity_presentations'][phase]
        _require(exposure['batch_counts'][phase]==budget and sum(source_counts.values())==64*budget and sum(id_counts.values())==64*budget, 'Terminal exposure totals mismatch')
        _require(all(type(n) is int and n>=0 for n in list(source_counts.values())+list(id_counts.values())), 'Invalid exposure count')
        _require(exposure['distinct'][phase]=={label for label,n in id_counts.items() if n}, 'Distinct identity exposure mismatch')
        _require(not set(source_counts)-set(identity['sources']) and not set(id_counts)-set(identities), 'Unknown exposure provenance')
        for source in identity['sources']:
            _require(source_counts.get(source,0)==sum(n for label,n in id_counts.items() if identities[label]['source_key']==source), 'Source/identity exposure mismatch')
    state=normalized_model_state(payload)
    keys=['classifier.weight'] if architecture=='resnet50' else [f'classifiers.fc_list.{i}.weight' for i in range(6)]
    for key in keys:
        _require(key in state and tuple(state[key].shape)==(classes,2048 if architecture=='resnet50' else 256), 'Classifier must retain combined source C')
    _require(all(not t.is_floating_point() or t.dtype==torch.float32 for t in state.values()) and
             all(torch.isfinite(t).all() for t in state.values()), 'Invalid/nonfinite model tensors')
    if architecture=='resnet50':
        center=identity['center']
        _require(center is not None and center['num_classes']==classes and center['feature_dim']==2048 and
                 center['feature_branch']=='feat_raw' and center['weight']==.0005 and center['reduction']=='half_mean_squared_distance', 'Center convention mismatch')
        _require(payload['center'] is not None and tuple(payload['center']['centers'].shape)==(classes,2048) and
                 payload['center']['centers'].dtype==torch.float32 and torch.isfinite(payload['center']['centers']).all() and
                 payload['center_optimizer'] is not None and all(g['lr']==.5 for g in payload['center_optimizer']['param_groups']), 'Missing/incompatible Center state')
    else:
        _require(identity['center'] is None and payload['center'] is None and payload['center_optimizer'] is None, 'Unexpected PCB Center state')
    sched=payload['scheduler']
    _require(sched['completed_updates']==budget and {k:v for k,v in sched.items() if k!='completed_updates'}==identity['scheduler'], 'Terminal scheduler mismatch')
    _require(sched['milestones']==tuple(cfg['sched']['milestones']) and sched['warmup_iters']==cfg['sched']['warmup_iters'] and
             sched['warmup_factor']==cfg['sched']['warmup_factor'] and sched['gamma']==.1 and sched['unit']=='successful_primary_updates', 'Frozen schedule mismatch')
    return payload,manifest,path


def target_metadata(cfg,identity,root):
    """Only original target split metadata is read, never images/source loaders."""
    target=identity['target'];root=Path(root).expanduser().resolve(strict=True)
    base='cuhk03/detected' if target=='cuhk03' else target
    relative=[f'{base}/partitions.pkl'] if target!='msmt17' else [f'msmt17/MSMT17_V2/list_{role}.txt' for role in ('query','gallery')]
    hashes={}
    for name in relative:
        lexical=root/name;path=lexical.resolve(strict=True)
        _require(path==lexical and path.is_relative_to(root), 'Unauthorized target metadata alias/root')
        hashes[str(path)]=file_digest(path)
    frozen=identity['provenance'][target]
    _require(all(frozen['metadata_hashes'].get(name)==value for name,value in hashes.items()), 'Target partition/list hash differs from terminal provenance')
    if target=='msmt17':
        target_hash=digest(dict(schema_version=1,ordered_lists=[dict(name=Path(name).name,sha256=value) for name,value in hashes.items()]))
    else: target_hash=next(iter(hashes.values()))
    _require(frozen['descriptor_metadata_hashes']['target']==target_hash, 'Target descriptor hash mismatch')
    return dict(dataset=copy.deepcopy(scientific_configuration(cfg)['target']),metadata_hashes=hashes,
                descriptor_digest=target_hash,query='list_query.txt' if target=='msmt17' else 'test_marks == 0',
                gallery='list_gallery.txt' if target=='msmt17' else 'test_marks == 1')


def evaluation_output(request,project_root=None):
    project=Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    raw=Path(request['output_dir']).expanduser()
    _require('..' not in raw.parts, 'Output traversal prohibited')
    path=raw if raw.is_absolute() else project/raw
    prefix=project/'exp/lodo_eval/v1'/('synthetic' if request['synthetic'] else 'benchmark')/request['architecture']/request['fold']/'seed42'
    _require(path.parent==prefix and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',path.name) is not None, 'Invalid isolated evaluation namespace')
    _require(path.resolve()==path and not path.exists() and not path.is_symlink(), 'Evaluation output collision/alias')
    _require(all(not parent.exists() or parent.is_dir() for parent in path.parents), 'Evaluation output parent is not a directory')
    return path


def retrieval_contract(cfg):
    return dict(embedding='post_bn_l2_2048' if cfg['protocol']['architecture']=='resnet50' else 'concatenated_parts_1536',
                dimension=2048 if cfg['protocol']['architecture']=='resnet50' else 1536,
                normalization='l2',distance='euclidean',camera_filter='exclude_same_pid_same_camera',
                topk=[1,5,10],reranking=False,test_time_flip=False,
                images=copy.deepcopy(cfg['data']['test']['images']),aug=copy.deepcopy(cfg['data']['test']['aug']))


def metric_vector(scores,units='fraction'):
    _require(units in ('fraction','percentage'), 'Unsupported metric units')
    factor=1 if units=='fraction' else 100
    result={}
    for key in METRICS:
        value=scores.get(key)
        _require(type(value) in (int,float) and torch.isfinite(torch.tensor(value)) and 0<=value<=factor, f'Missing/invalid metric {key}')
        result[key]=float(value)/factor
    return result


def publish_result(path,record):
    path=Path(path)
    _require(path.name=='lodo_result.json' and not path.exists(), 'Stage B result collision/name mismatch')
    atomic_write(path,lambda stream:stream.write(canonical_json(record)))
    atomic_write(path.with_name('lodo_result.sha256.json'),lambda stream:stream.write(canonical_json(dict(schema_version=1,sha256=file_digest(path)))))


def read_result(path):
    path=Path(path)
    try:
        proof=json.loads(path.with_name('lodo_result.sha256.json').read_text())
        _require(proof==dict(schema_version=1,sha256=file_digest(path)), 'Stage B result integrity mismatch')
        record=json.loads(path.read_text())
        _require(record['schema_version']==1 and record['record_type']=='lodo_held_out_result', 'Unsupported Stage B result')
        metric_vector(record['metrics'],record['metric_units'])
        _require(file_digest(path.with_name('comparison.json'))==record['artifacts']['comparison_sha256'], 'Comparison integrity mismatch')
        return record
    except (OSError,KeyError,TypeError,json.JSONDecodeError) as error:
        raise ValueError(f'Invalid/incomplete Stage B result: {error}') from error


def evaluate_lodo(request,*,project_root=None):
    payload,manifest,checkpoint=validate_terminal(request)
    output=evaluation_output(request,project_root)
    cfg=payload['cfg'];identity=payload['identity'];root=request.get('data_root',cfg['data']['root'])
    metadata=target_metadata(cfg,identity,root)
    from reid.utils.lodo_evaluation_device import select_device, execution_spec, EvaluationProgress
    device, guard, device_evidence = select_device(request, cfg, output)
    progress = EvaluationProgress(device, guard)
    print(f'[Evaluation] device={device} precision=fp32 ranking=cpu checkpoint={checkpoint}', flush=True)
    runtime=dict(data=dict(root=str(Path(root).resolve()),test=dict(dataset=metadata['dataset'],batch=dict(size=request.get('batch_size',32))),
                           num_workers=request.get('num_workers',0),pin_memory=False),system=dict(device=device.type,gpu_id=0))
    evaluation=build_evaluation_config(cfg,runtime,output)
    contract=retrieval_contract(cfg)
    eval_identity=dict(schema_version=1,target=metadata,representation=contract,batch_size=request.get('batch_size',32),precision='fp32')
    if device.type == 'cuda':
        eval_identity['execution'] = execution_spec(request)
    # Reconstruction uses ONLY the embedded source cfg and source C.
    model=reconstruct_model(payload).to(device)
    loader=build_test_loader(evaluation)
    scores=evaluate_reid(evaluation,model,loader,device,progress=progress,before_batch=progress.before_batch)
    if guard is not None:
        guard()
    print('[Evaluation] metrics=' + json.dumps(metric_vector(scores)), flush=True)
    _require(not any(k.startswith('rerank') for k in scores), 'Unexpected reranking result')
    record=dict(schema_version=RESULT_VERSION,record_type='lodo_held_out_result',protocol='lodo_v1',protocol_version=1,
                architecture=request['architecture'],variant=cfg['model'].get('variant'),fold=request['fold'],
                sources=identity['sources'],target=request['target'],seed=42,synthetic=request['synthetic'],
                authoritative=not request['synthetic'],training=dict(config_digest=identity['config_digest'],map_digest=identity['map_digest'],
                data_digest=identity['data_digest'],sidecar_digest=identity['sidecar_digest'],source_provenance=identity['provenance'],num_classes=identity['num_classes']),
                terminal=dict(path=str(checkpoint),**manifest),evaluation=dict(config_digest=digest(eval_identity),configuration=eval_identity),
                representation=contract,metrics=metric_vector(scores),metric_units='fraction',
                environment=dict(python=platform.python_version(),torch=torch.__version__,device=device.type,num_workers=request.get('num_workers',0)),
                artifacts=dict(result=str(output/'lodo_result.json'),comparison=str(output/'comparison.json')),
                caveats=['Complete training strategies; differences do not isolate domain diversity.']+
                    (['CUHK03 detected processed_partition; official-new provenance remains unverified.'] if 'cuhk03' in identity['sources']+[identity['target']] else []))
    if device_evidence is not None:
        record['environment']['cuda_admission'] = device_evidence
    from reid.utils.lodo_comparison import compare_stage_a
    comparison=compare_stage_a(record,request.get('stage_a_records',[]))
    # Recheck immutable input integrity before publishing; no terminal writes.
    _require(file_digest(checkpoint)==manifest['checkpoint_sha256'], 'Terminal changed during evaluation')
    _require(target_metadata(cfg,identity,root)==metadata, 'Target metadata changed during evaluation')
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir(exist_ok=False)
    atomic_write(output/'comparison.json',lambda stream:stream.write(canonical_json(comparison)))
    record['artifacts']['comparison_sha256']=file_digest(output/'comparison.json')
    publish_result(output/'lodo_result.json',record)
    return record,comparison
