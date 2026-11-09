"""Explicit later no-flip reference requests; ordinary source guards remain intact.

This adapter never edits saved configuration/weights. Each request pins the
existing source registry/checkpoint and changes only an explicit evaluation
identity, target descriptor and approved execution/output fields.
"""
import copy
import json
from pathlib import Path
import re
import torch
import yaml
from reid.utils import lodo_evaluation as common
from reid.utils.mgn_lodo_evaluation import evaluate_frozen_bn, execution
from reid.utils.lodo_checkpoint import atomic_write, canonical_json, digest, file_digest
from reid.utils.checkpoint import reconstruct_model
from reid.utils.config import load_config
from reid.utils.evaluation_config import build_evaluation_config
from reid.data.build import build_test_loader
from reid.utils.lodo_protocol import DATASET_ORDER


def validate_request(request):
    keys={'mode','version','source','target','source_registry','source_registry_sha256','checkpoint',
          'checkpoint_sha256','target_config','target_config_sha256','effective_target_sha256','target_metadata_hashes','output_dir','device','flip_aggregation'}
    common._require(isinstance(request,dict) and request.keys()==keys,'Exact supplementary reference request required')
    common._require(request['mode']=='mgn_single_source_reference' and type(request['version']) is int and request['version']==1,'Wrong reference mode/version')
    common._require(request['source'] in DATASET_ORDER and request['target'] in DATASET_ORDER and request['source']!=request['target'],'Only12 off-diagonal reference cells are defined')
    common._require(request['device']=='cuda' and request['flip_aggregation']=='none','Explicit GPU/no-flip evaluation only')
    for key in ('checkpoint_sha256','source_registry_sha256','target_config_sha256','effective_target_sha256'):
        common._require(isinstance(request[key],str) and re.fullmatch('[0-9a-f]{64}',request[key]),'Pinned reference SHA required')


def source_checkpoint(request):
    validate_request(request)
    for path,key in ((request['source_registry'],'source_registry_sha256'),(request['checkpoint'],'checkpoint_sha256'),(request['target_config'],'target_config_sha256')):
        p=Path(path).absolute()
        common._require(p.resolve()==p and file_digest(p)==request[key],'Reference input hash/alias mismatch')
    registry=yaml.safe_load(Path(request['source_registry']).read_text())
    row=registry['sources'][request['source']];selected=row['checkpoints']['ckpt_best.pth']
    common._require(selected['sha256']==request['checkpoint_sha256'],'Only original registered best checkpoint is eligible')
    payload=torch.load(request['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    cfg=payload['cfg']
    common._require('protocol' not in cfg and cfg['model']['name']=='mgn' and cfg['data']['train']['dataset']['name']==request['source'],'Wrong source model/science')
    common._require(payload['epoch']==selected['epoch'] and payload['reconstruction']['num_classes']==row['source_identities'],'Original source C/epoch mismatch')
    common._require(cfg['train']['epochs']==120 and cfg['sched']['milestones']==[60,90] and cfg['train']['save']['metric']=='mAP','Original120-epoch recipe/selection required')
    common._require(digest(load_config(request['target_config']))==request['effective_target_sha256'],'Resolved target configuration changed')
    hashes=request['target_metadata_hashes']
    common._require(isinstance(hashes,dict) and bool(hashes),'Frozen target split hashes required')
    for path,expected in hashes.items():
        p=Path(path).absolute()
        common._require(p.resolve()==p and file_digest(p)==expected,'Target partition bytes changed')
    return payload


def evaluate_reference(request, *, request_path, project_root=None):
    validate_request(request)
    root=Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    from reid.utils.mgn_lodo_operational import validate_evaluation_admission
    admission=validate_evaluation_admission(request_path,request,root=root)
    payload=source_checkpoint(request);source=payload['cfg'];source_digest=digest(source)
    output=Path(request['output_dir']);output=output if output.is_absolute() else root/output
    prefix=root/'results/mgn_no_flip_references'/f"{request['source']}_to_{request['target']}"
    common._require(output.parent==prefix and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',output.name) and output.resolve()==output and not output.exists(),'Fresh reference namespace required')
    target=load_config(request['target_config'])
    common._require(target['data']['test']['dataset']['name']==request['target'],'Wrong target descriptor')
    evaluation=build_evaluation_config(source,target,output)
    evaluation['eval']['flip_aggregation']='none'
    # Explicit execution-only view for the reused strict CUDA device policy;
    # it is not a scientific training config and cannot pass the v2 validator.
    policy_view=dict(protocol=dict(name='multi_source_lodo',version=2,architecture='mgn'),
        system=dict(device='cuda',gpu_id=0,amp=False),data=dict(num_workers=0),
        repro=dict(seed=42,deterministic=True,benchmark=False,cuda_policy='strict_cuda_deterministic_v1'),
        train=dict(save=dict(resume='')))
    evaluation['system'].update(device='cuda',gpu_id=0,amp=False)
    evaluation['data'].update(num_workers=0,pin_memory=True)
    evaluation['data']['test']['batch']['size']=32
    evaluation['repro']={k:v for k,v in policy_view['repro'].items() if k!='cuda_policy'}
    loader=build_test_loader(evaluation)
    from scripts.mgn_preflight import evaluation_identity
    before=evaluation_identity(loader.dataset,evaluation)
    common._require(before['metadata_sha256']==request['target_metadata_hashes'],'Target definition differs from frozen reference request')
    from reid.utils.lodo_device import select_lodo_device
    device,guard,evidence=select_lodo_device(policy_view)
    common._require(device.type=='cuda','GPU reference extraction required')
    model=reconstruct_model(payload).to(device)
    scores=evaluate_frozen_bn(evaluation,model,loader,device,before_batch=guard)
    guard()
    common._require(evaluation_identity(loader.dataset,evaluation)==before,'Target inventory changed')
    common._require(file_digest(request['checkpoint'])==request['checkpoint_sha256'] and digest(payload['cfg'])==source_digest,'Source weights/config changed')
    from reid.utils.mgn_lodo_operational import retrieval_code_identity
    record=dict(schema_version=1,record_type='mgn_single_source_reference',reference_policy_version=1,
        architecture='mgn',source_dataset=request['source'],target_dataset=request['target'],
        checkpoint_path=str(Path(request['checkpoint']).absolute()),checkpoint_sha256=request['checkpoint_sha256'],
        source_checkpoint_unchanged=True,source_training_config_sha256=source_digest,
        resolved_config=evaluation,**common.metric_vector(scores),metric_units='fraction',
        training=dict(budget=dict(unit='completed_epochs',epochs=120),checkpoint_selection='periodic_source_test_mAP_best',
                      selected_epoch=payload['epoch'],successful_updates='not_authoritatively_recorded_in_ordinary_source_checkpoints'),
        evaluation_provenance=dict(target_metadata_hashes=before['metadata_sha256'],target_records_sha256=before['records_sha256'],
            camera_filter='exclude_same_pid_same_camera',target_bn_adaptation=False,
            execution=evidence,evaluation_release=admission['release_commit'],
            retrieval_code_sha256=retrieval_code_identity(admission['source_sha256']),
            execution_scope=admission,identity=digest(dict(request=request,source_config_sha256=source_digest,target=before,execution=execution(request)))))
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir(exist_ok=False)
    atomic_write(output/'cross_dataset.json',lambda f:f.write(canonical_json(record)))
    return record


def prepare_reference_plan(source_registry, *, project_root=None, run_id='prepared-no-flip'):
    """Read-only12-cell request preparation, not permission or evaluator execution."""
    root=Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    registry_path=Path(source_registry).resolve(strict=True);registry=yaml.safe_load(registry_path.read_text())
    common._require(set(registry['sources'])==set(DATASET_ORDER),'Original four-source registry required')
    common._require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',run_id),'Invalid reference run-id')
    target_data={}
    for target in DATASET_ORDER:
        path=root/f'configs/mgn/{target}.yaml';cfg=load_config(str(path))
        data_root=Path(cfg['data']['root']).resolve(strict=True)
        relative=([f'msmt17/MSMT17_V2/list_{r}.txt' for r in ('query','gallery')] if target=='msmt17'
                  else [('cuhk03/detected' if target=='cuhk03' else target)+'/partitions.pkl'])
        hashes={str(data_root/p):file_digest(data_root/p) for p in relative}
        target_data[target]=dict(target_config=str(path),target_config_sha256=file_digest(path),
                                 effective_target_sha256=digest(cfg),target_metadata_hashes=hashes)
    sources={};requests=[]
    for source in DATASET_ORDER:
        item=registry['sources'][source]['checkpoints']['ckpt_best.pth'];p=Path(item['path']);p=p if p.is_absolute() else root/p
        common._require(p.resolve()==p and file_digest(p)==item['sha256'],'Original source checkpoint missing/changed')
        sources[source]=dict(checkpoint=str(p),checkpoint_sha256=item['sha256'],epoch=item['epoch'])
        for target in DATASET_ORDER:
            if source==target:continue
            r=dict(mode='mgn_single_source_reference',version=1,source=source,target=target,
                source_registry=str(registry_path),source_registry_sha256=file_digest(registry_path),
                checkpoint=str(p),checkpoint_sha256=item['sha256'],**target_data[target],
                output_dir=str(root/'results/mgn_no_flip_references'/f'{source}_to_{target}'/run_id),
                device='cuda',flip_aggregation='none')
            validate_request(r);requests.append(r)
    return dict(status='PREPARED_NOT_EXECUTED_SEPARATE_APPROVAL_REQUIRED',cells=12,
                fixed_sources=sources,requests=requests,original_mean_flip_matrix_preserved=True)
