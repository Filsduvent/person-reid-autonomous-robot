"""Scoped MGN-v2 execution gate. Phase20 creates no approvals or qualifications.

Local UID-owned records attest operator intent, not an independent security
boundary. Exact code/config/data/runtime bindings prevent accidental scope drift.
"""
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess

from reid.utils.lodo_checkpoint import digest, file_digest
from reid.utils.lodo_operational import private_record
from reid.utils.mgn_lodo_protocol import configuration_identity, validate_lodo_config

ROOT = Path(__file__).resolve().parents[2]


def source_snapshot(root=ROOT):
    root=Path(root).resolve()
    return {str(p.relative_to(root)):file_digest(p) for folder in ('reid','scripts','configs/lodo/mgn_lodo_v2')
            for p in sorted((root/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.yml')}


def validate_committed_sources(root, snapshot):
    """Require actual committed runtime bytes/membership, even with unrelated dirty docs."""
    entries=subprocess.check_output(['git','-C',str(root),'ls-tree','-rz','HEAD','--',
                                      'reid','scripts','configs/lodo/mgn_lodo_v2']).split(b'\0')
    tracked=set()
    for entry in filter(None,entries):
        header,name=entry.split(b'\t',1);mode,kind,blob=header.decode().split();name=name.decode()
        if Path(name).suffix not in ('.py','.yaml','.yml'):continue
        if name not in snapshot:raise ValueError('Committed runtime/config file missing: '+name)
        path=Path(root)/name;raw=path.read_bytes()
        if (kind!='blob' or mode not in ('100644','100755') or path.resolve()!=path
                or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=blob
                or bool(path.stat().st_mode & 0o111)!=(mode=='100755')):
            raise ValueError('Runtime source differs from reviewed committed release: '+name)
        tracked.add(name)
    if tracked!=set(snapshot):
        raise ValueError('Uncommitted runtime/config membership cannot be qualified')


def environment():
    from scripts.mgn_preflight import environment as current
    value=current()
    import torch
    from reid.utils.lodo_device import _query
    uuid=os.environ.get('LODO_AUTHORIZED_GPU_UUID','')
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid):
        raise ValueError('Full GPU UUID required to bind qualification runtime')
    rows=[r for r in _query('uuid,driver_version,name,compute_cap') if len(r)==4 and r[0]==uuid]
    if len(rows)!=1:
        raise ValueError('Cannot bind actual GPU driver/name/compute capability without CUDA discovery')
    value.update(cuda_build=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        gpu_runtime=dict(uuid=uuid,driver=rows[0][1],name=rows[0][2],compute_capability=rows[0][3]))
    # Volatile occupancy/free-memory is checked by admission, not runtime identity.
    return json.loads(json.dumps(value))


def validate_training_admission(config, cfg, *, root=ROOT):
    validate_lodo_config(cfg,project_root=root)
    path=os.environ.get('MGN_LODO_ADMISSION_RECEIPT','')
    if not path:raise ValueError('MGN LODO requires a fold-qualified scoped execution receipt; Phase21 is not yet qualified')
    record,_=private_record(path)
    required={'schema_version','record_type','action','approval_reference','root','config_path','config_sha256',
              'effective_config_sha256','release_commit','qualification_path','qualification_sha256','gpu_uuid'}
    if record.keys()!=required or (type(record['schema_version']) is not int or record['schema_version']!=1) or record['record_type']!='mgn_lodo_v2_execution' or record['action']!='training':
        raise ValueError('Wrong MGN LODO execution record/action/schema')
    root=Path(root).resolve();config=Path(config).absolute()
    if config.resolve()!=config or not config.is_relative_to(root) or str(root)!=record['root'] or str(config)!=record['config_path']:
        raise ValueError('Wrong config/root or path alias')
    if not isinstance(record['approval_reference'],str) or not record['approval_reference'].strip():
        raise ValueError('Explicit operator approval reference required')
    if file_digest(config)!=record['config_sha256'] or digest(cfg)!=record['effective_config_sha256']:
        raise ValueError('Config/effective arguments differ from approved scope')
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if record['release_commit']!=head or not re.fullmatch('[0-9a-f]{40}',head):
        raise ValueError('Reviewed implementation release changed')
    snapshot=source_snapshot(root)
    validate_committed_sources(root,snapshot)
    qualification,sha=private_record(record['qualification_path'])
    if sha!=record['qualification_sha256'] or qualification.keys()!= {'schema_version','status','fold','config_digest','release_commit','source_sha256','preflight_sha256','environment','gpu_uuid','capabilities','evidence_sha256','recovery_scope'}:
        raise ValueError('Invalid qualification schema/integrity')
    if ((type(qualification['schema_version']) is not int or qualification['schema_version']!=1) or qualification['status']!='MGN_LODO_FOLD_QUALIFIED'
            or qualification['fold']!=cfg['protocol']['fold'] or qualification['config_digest']!=configuration_identity(cfg)
            or qualification['release_commit']!=head or qualification['source_sha256']!=snapshot
            or qualification['environment']!=environment()
            or qualification['capabilities']!=['fresh_fp32','completed_epoch_rollback','terminal_noop']):
        raise ValueError('Fold/code/runtime/recovery is not qualified for this request')
    scope=qualification['recovery_scope']
    if (not isinstance(scope,dict) or scope.keys()!= {'boundary_evidence','equivalence_review_reference'}
            or scope['boundary_evidence']!='bounded_fixture'
            or not isinstance(scope['equivalence_review_reference'],str) or not scope['equivalence_review_reference'].strip()):
        raise ValueError('Bounded recovery evidence requires explicit equivalence review; no universal recovery claim')
    evidence=qualification['evidence_sha256']
    if not isinstance(evidence,dict) or not evidence:
        raise ValueError('Preserved qualification evidence hashes required')
    for raw,expected_sha in evidence.items():
        proof=Path(raw).absolute()
        if proof.resolve()!=proof or not proof.is_relative_to(root) or file_digest(proof)!=expected_sha:
            raise ValueError('Qualification evidence changed/aliased/outside this release root')
    uuid=record['gpu_uuid']
    if (not isinstance(uuid,str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid)
            or qualification['gpu_uuid']!=uuid or os.environ.get('LODO_AUTHORIZED_GPU_UUID')!=uuid
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid):
        raise ValueError('Approved single GPU UUID scope differs')
    from reid.data.lodo_preflight import preflight_lodo
    if digest(preflight_lodo(cfg))!=qualification['preflight_sha256']:
        raise ValueError('Qualified source/target inventory differs')
    expected=cfg['model']['initialization']['sha256']
    if file_digest(cfg['model']['weights_path'])!=expected:
        raise ValueError('Pinned pretrained artifact differs')
    return record


def validate_training_session(cfg):
    path=os.environ.get('MGN_LODO_ADMISSION_RECEIPT','')
    if not path:raise ValueError('MGN LODO requires scoped operational admission before execution')
    record,_=private_record(path)
    return validate_training_admission(record.get('config_path',''),cfg)


def validate_evaluation_admission(request_path, request, *, root=ROOT):
    """Separate exact-request/terminal scope; training authority never admits evaluation."""
    path=os.environ.get('MGN_LODO_EVALUATION_RECEIPT','')
    if not path:raise ValueError('MGN evaluation requires separate scoped operator/resource approval')
    record,_=private_record(path)
    keys={'schema_version','record_type','action','approval_reference','root','request_path','request_sha256',
          'effective_request_sha256','release_commit','source_sha256','environment','gpu_uuid','checkpoint_sha256'}
    action='held_out_evaluation' if request.get('mode')=='lodo_held_out' else 'single_source_reference'
    if (record.keys()!=keys or (type(record['schema_version']) is not int or record['schema_version']!=1) or record['record_type']!='mgn_lodo_v2_execution'
            or record['action']!=action or not isinstance(record['approval_reference'],str) or not record['approval_reference'].strip()):
        raise ValueError('Wrong evaluation action/schema/approval reference')
    root=Path(root).resolve();p=Path(request_path).absolute()
    if (p.resolve()!=p or not p.is_relative_to(root) or record['root']!=str(root) or record['request_path']!=str(p)
            or file_digest(p)!=record['request_sha256'] or digest(request)!=record['effective_request_sha256']):
        raise ValueError('Evaluation request/root differs from approved scope')
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    snap=source_snapshot(root);validate_committed_sources(root,snap)
    if record['release_commit']!=head or record['source_sha256']!=snap or record['environment']!=environment():
        raise ValueError('Evaluation release/source/environment changed')
    checkpoint=Path(request['checkpoint']).absolute()
    if checkpoint.resolve()!=checkpoint or file_digest(checkpoint)!=record['checkpoint_sha256'] or request['checkpoint_sha256']!=record['checkpoint_sha256']:
        raise ValueError('Approved checkpoint SHA differs')
    uuid=record['gpu_uuid']
    if not isinstance(uuid,str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid):
        raise ValueError('Full evaluation GPU UUID required')
    if request.get('device')!='cuda' or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid or os.environ.get('LODO_AUTHORIZED_GPU_UUID')!=uuid:
        raise ValueError('Explicit approved GPU required; no CPU fallback')
    return record


def retrieval_code_identity(snapshot):
    """Exact shared pixel/parser/model/extraction/ranking bytes for matched references."""
    return digest({k:v for k,v in snapshot.items() if k.startswith(('reid/data/','reid/metrics/'))
        or k in ('reid/models/mgn.py','reid/models/build.py','reid/engine/evaluator.py','reid/utils/checkpoint.py')})
