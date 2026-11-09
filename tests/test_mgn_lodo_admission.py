"""Synthetic local receipt fixtures; never official permission or GPU execution."""
import copy
import json
from pathlib import Path
import subprocess
import pytest
import yaml
from reid.utils import mgn_lodo_operational as op
from reid.utils.lodo_checkpoint import digest,file_digest
from reid.utils.mgn_lodo_protocol import configuration_identity
from test_mgn_lodo_v2 import config

UUID='GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2'


def private(path,value):
    path.write_text(json.dumps(value));path.chmod(0o600)


@pytest.fixture
def receipt(tmp_path,monkeypatch):
    root=tmp_path;cfg=config(root=root);path=root/'configs/lodo/mgn_lodo_v2/fold.yaml';path.parent.mkdir(parents=True);path.write_text(yaml.safe_dump(cfg))
    for arguments in (['init','-q'],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','add','.'],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','synthetic fixture']):
        subprocess.run(['git','-C',str(root),*arguments],check=True,capture_output=True)
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    monkeypatch.setattr(op,'environment',lambda:{'fixture':'cpu-only'})
    import reid.data.lodo_preflight as pre
    monkeypatch.setattr(pre,'preflight_lodo',lambda c:{'frozen':'inventory fixture'})
    original=op.file_digest
    monkeypatch.setattr(op,'file_digest',lambda p:cfg['model']['initialization']['sha256'] if str(p)==cfg['model']['weights_path'] else original(p))
    evidence=root/'case-proof.yaml';evidence.write_text('synthetic unit fixture only')
    qualification=dict(schema_version=1,status='MGN_LODO_FOLD_QUALIFIED',fold=cfg['protocol']['fold'],config_digest=configuration_identity(cfg),
        release_commit=head,source_sha256=op.source_snapshot(root),preflight_sha256=digest({'frozen':'inventory fixture'}),
        environment=op.environment(),gpu_uuid=UUID,capabilities=['fresh_fp32','completed_epoch_rollback','terminal_noop'],
        evidence_sha256={str(evidence):file_digest(evidence)},
        recovery_scope=dict(boundary_evidence='bounded_fixture',equivalence_review_reference='Synthetic unit fixture review, not experimental validation'))
    qpath=root/'qualification.json';private(qpath,qualification)
    record=dict(schema_version=1,record_type='mgn_lodo_v2_execution',action='training',approval_reference='Synthetic unit fixture only',
        root=str(root),config_path=str(path),config_sha256=file_digest(path),effective_config_sha256=digest(cfg),release_commit=head,
        qualification_path=str(qpath),qualification_sha256=file_digest(qpath),gpu_uuid=UUID)
    admission=root/'admission.json';private(admission,record)
    monkeypatch.setenv('MGN_LODO_ADMISSION_RECEIPT',str(admission));monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID',UUID);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID)
    return root,cfg,path,qpath,qualification,admission,record


def test_exact_scope_passes_pure_binding_checks_without_gpu_or_images(receipt,monkeypatch):
    root,cfg,path,*_=receipt
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA discovered by metadata gate'))
    assert op.validate_training_admission(path,cfg,root=root)['action']=='training'
    assert not (root/'exp').exists()


@pytest.mark.parametrize('key,value',[('action','held_out_evaluation'),('approval_reference',''),('effective_config_sha256','0'*64),('release_commit','0'*40),('gpu_uuid','GPU-00000000-0000-0000-0000-000000000000')])
def test_operator_scope_drift_rejected(receipt,key,value):
    root,cfg,path,q,qualification,admission,record=receipt;record[key]=value;private(admission,record)
    with pytest.raises(ValueError):op.validate_training_admission(path,cfg,root=root)


@pytest.mark.parametrize('key,value',[('status','PREPARED'),('fold','LODO-2'),('config_digest','f'*64),('capabilities',['fresh_fp32']),('environment',{'wrong':'runtime'}),('preflight_sha256','a'*64)])
def test_unqualified_or_wrong_qualification_never_launches(receipt,key,value):
    root,cfg,path,q,qualification,admission,record=receipt;qualification[key]=value;private(q,qualification);record['qualification_sha256']=file_digest(q);private(admission,record)
    with pytest.raises(ValueError):op.validate_training_admission(path,cfg,root=root)


def test_source_membership_and_blob_changes_rejected(receipt):
    root,cfg,path,*_=receipt
    (root/'reid').mkdir();(root/'reid/new_unreviewed.py').write_text('x=1')
    with pytest.raises(ValueError,match='membership'):op.validate_training_admission(path,cfg,root=root)


def test_unsafe_receipt_permissions_rejected(receipt):
    root,cfg,path,q,qualification,admission,record=receipt;admission.chmod(0o644)
    with pytest.raises(ValueError,match='private'):op.validate_training_admission(path,cfg,root=root)


def test_committed_source_deletion_rejected(receipt):
    root,cfg,path,*_=receipt;path.unlink()
    with pytest.raises(ValueError,match='missing'):op.validate_committed_sources(root,op.source_snapshot(root))


def test_runtime_identity_binds_actual_driver_and_compute_capability_without_cuda(monkeypatch):
    import torch
    import reid.utils.lodo_device as device
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID',UUID)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA discovery in metadata runtime identity'))
    monkeypatch.setattr(device,'_query',lambda fields:[[UUID,'driver-A','GPU-fixture','8.6']])
    first=op.environment();assert first['gpu_runtime']['compute_capability']=='8.6'
    assert first['cuda_build']==torch.version.cuda and 'cudnn_version' in first
    monkeypatch.setattr(device,'_query',lambda fields:[[UUID,'driver-B','GPU-fixture','8.6']])
    assert op.environment()!=first
