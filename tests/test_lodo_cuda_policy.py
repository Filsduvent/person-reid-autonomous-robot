"""Phase 9R.3 CPU-only policy tests; fake CUDA payloads are explicitly metadata doubles."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from test_lodo_config import lodo_config
from test_lodo_checkpoint import synthetic, parts, epoch, assert_equal, FixtureModel, target_guards
from test_lodo_cuda_recovery import configured, write_boundary, POLICY, UUID
from test_lodo_device import inventory, cuda_double
from reid.engine.update_control import UpdateProgress
from reid.utils import lodo_cuda_policy as policy, lodo_checkpoint as ck, lodo_recovery as recovery, lodo_device as dev
from reid.utils import lodo_protocol as protocol
from reid.utils.config_schema import validate_config
from scripts import train


@pytest.fixture
def launch(monkeypatch):
    before=policy.runtime_controls()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',policy.WORKSPACE)
    # Explicit metadata double for an environment exported before Python launch.
    monkeypatch.setattr(policy,'WORKSPACE_AT_IMPORT',policy.WORKSPACE)
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID',UUID)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    yield
    torch.backends.cudnn.benchmark=before['cudnn_benchmark']
    torch.backends.cudnn.deterministic=before['cudnn_deterministic']
    torch.backends.cuda.matmul.allow_tf32=before['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=before['cudnn_allow_tf32']
    torch.use_deterministic_algorithms(before['deterministic_algorithms'],warn_only=before['deterministic_warn_only'])


def strict(cfg):
    cfg=copy.deepcopy(cfg)
    cfg['system'].update(device='cuda',gpu_id=0)
    cfg['data']['num_workers']=0
    cfg['repro'].update(cuda_policy=policy.STRICT_POLICY,deterministic=True,benchmark=False)
    return cfg


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('fold',list(protocol.FOLDS))
def test_strict_config_optin_preserves_frozen_science(architecture,fold,tmp_path):
    cfg=lodo_config(architecture,fold);old=protocol.scientific_configuration(cfg)
    selected=strict(cfg);validate_config(selected,project_root=tmp_path)
    new=protocol.scientific_configuration(selected)
    assert new['identity_schema_version']==2 and old['identity_schema_version']==1
    assert new['execution_policy']==policy.policy_contract(selected)
    assert protocol.configuration_identity(cfg)!=protocol.configuration_identity(selected)
    for key in old.keys()-{'identity_schema_version','benchmark','deterministic'}:assert old[key]==new[key]
    assert not tmp_path.joinpath('exp').exists() and not torch.cuda.is_initialized()


def test_default_original_identity_and_explicit_original_are_equal():
    cfg=lodo_config();before=protocol.serialize_scientific_configuration(cfg)
    cfg['repro']['cuda_policy']=policy.ORIGINAL_POLICY
    assert protocol.serialize_scientific_configuration(cfg)==before
    assert policy.checkpoint_schema(cfg)==1


@pytest.mark.parametrize('value',['unknown',None,True,{},'strict_cuda_deterministic_v2'])
def test_unknown_versions_fail_purely(value):
    cfg=lodo_config();cfg['repro']['cuda_policy']=value
    with pytest.raises(ValueError):protocol.scientific_configuration(cfg)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('change',[lambda c:c['system'].update(device='cpu'),lambda c:c['system'].update(device='auto'),lambda c:c['repro'].update(deterministic=False),lambda c:c['repro'].update(benchmark=True)])
def test_strict_requires_explicit_cuda_and_matching_declared_flags(change):
    cfg=strict(lodo_config());change(cfg)
    with pytest.raises(ValueError):protocol.scientific_configuration(cfg)


def test_stage_a_rejects_policy_without_optin():
    cfg=lodo_config();cfg.pop('protocol');cfg['repro']['cuda_policy']=policy.STRICT_POLICY
    with pytest.raises(ValueError,match='explicit protocol'):protocol.require_explicit_lodo(cfg)


@pytest.mark.parametrize('workspace',[None,'',':16:8',':4096:4'])
def test_missing_or_incompatible_workspace_fails_before_cuda_or_sources(launch,monkeypatch,workspace):
    cfg=strict(configured())
    if workspace is None:monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    else:monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',workspace)
    before=policy.runtime_controls()
    monkeypatch.setattr(dev,'audit_gpu',lambda *a,**k:pytest.fail('GPU discovery'))
    monkeypatch.setattr(train,'build_train_loader',lambda *a:pytest.fail('Sources accessed'))
    with pytest.raises(ValueError,match='CUBLAS_WORKSPACE'):train.run_lodo_training(cfg)
    assert policy.runtime_controls()==before and not torch.cuda.is_initialized()


def test_workspace_configured_after_module_import_is_rejected(launch,monkeypatch):
    monkeypatch.setattr(policy,'WORKSPACE_AT_IMPORT',None)
    with pytest.raises(ValueError,match='too late'):policy.activate_policy(strict(lodo_config()))


def test_initialized_context_rejected_without_backend_mutation(launch,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    before=policy.runtime_controls()
    with pytest.raises(ValueError,match='before CUDA initialization'):policy.activate_policy(strict(lodo_config()))
    assert before==policy.runtime_controls()


def test_activation_is_precuda_explicit_preserves_tf32_and_rechecks_drift(launch):
    cfg=strict(lodo_config());policy.activate_policy(cfg)
    assert policy.runtime_controls()=={k:v for k,v in policy.policy_contract(cfg).items() if k not in ('name','version')}
    assert not torch.cuda.is_initialized()
    torch.use_deterministic_algorithms(True,warn_only=True)
    with pytest.raises(ValueError,match='controls differ'):policy.validate_runtime(cfg)


@pytest.mark.parametrize('flag',['matmul','cudnn'])
def test_incompatible_tf32_is_rejected_without_silent_correction(launch,flag):
    if flag=='matmul':torch.backends.cuda.matmul.allow_tf32=True
    else:torch.backends.cudnn.allow_tf32=False
    before=policy.runtime_controls()
    with pytest.raises(ValueError,match='TF32'):policy.activate_policy(strict(lodo_config()))
    assert before==policy.runtime_controls()


def test_original_and_cpu_paths_do_not_change_global_flags(launch,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:pytest.fail('Original policy touched CUDA'))
    before=policy.runtime_controls();policy.activate_policy(lodo_config())
    assert before==policy.runtime_controls()


def test_device_admission_applies_policy_before_runtime_discovery(launch,inventory,cuda_double,monkeypatch):
    cfg=strict(configured())
    def available():
        policy.validate_runtime(cfg)
        return True
    monkeypatch.setattr(torch.cuda,'is_available',available)
    device,guard,evidence=dev.select_lodo_device(cfg)
    assert device==torch.device('cuda:0') and cuda_double==[0]
    assert evidence['execution_policy']==policy.policy_contract(cfg)
    guard();torch.backends.cudnn.benchmark=True
    with pytest.raises(ValueError,match='controls differ'):guard()


def strict_payload(synthetic,monkeypatch,*,architecture='resnet50',terminal=False):
    cfg=configured(architecture,budget=3 if terminal else 8);o=parts(cfg)
    p=UpdateProgress(cfg['train']['max_updates'],auxiliary=architecture=='resnet50');epoch(o,p)
    run=synthetic/cfg['experiment']['output_dir'];(run/'checkpoints').mkdir(parents=True)
    if terminal:
        path=run/'checkpoints/ckpt_terminal.pth'
        ck.save_lodo_checkpoint(path,progress=p,next_epoch=1,cursor=3,kind='terminal',**o)
    else:path=write_boundary(o,p,run,2)
    cfg=strict(cfg);policy.activate_policy(cfg);o['cfg']=cfg
    monkeypatch.setattr(dev,'_query',lambda fields,kind='gpu':[[UUID,'test-driver']])
    payload,proof=ck.read_checkpoint(path)
    payload.update(lodo_schema_version=2,cfg=copy.deepcopy(cfg),identity=copy.deepcopy(ck.make_identity(**o,cpu_inspection=True)))
    payload['rng']['cuda']=[torch.zeros(16,dtype=torch.uint8)]
    def publish():
        torch.save(payload,path)
        proof.update(schema_version=payload['lodo_schema_version'],checkpoint_sha256=ck.file_digest(path),identity_digest=ck.digest(payload['identity']),config_digest=payload['identity']['config_digest'],
                     fold=payload['identity']['protocol']['fold'],map_digest=payload['identity']['map_digest'],data_digest=payload['identity']['data_digest'])
        path.with_suffix('.json').write_bytes(ck.canonical_json(proof))
        if not terminal:
            (run/'checkpoints/recovery_index.json').write_bytes(ck.canonical_json(dict(version=1,generations=[dict(name=path.name,checkpoint_sha256=proof['checkpoint_sha256'],identity_digest=proof['identity_digest'])])))
    publish();return cfg,o,path,payload,publish


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_v2_checkpoint_cpu_validation_and_optimizer_binding(synthetic,launch,monkeypatch,architecture):
    cfg,o,path,payload,_=strict_payload(synthetic,monkeypatch,architecture=architecture)
    before={k:copy.deepcopy(o[k].state_dict()) for k in ('model','optimizer','scheduler')}
    p,position,kind=ck.restore_lodo_checkpoint(path,**o,cpu_inspection=True,validate_only=True)
    assert p.completed_primary_updates==3 and kind=='recovery'
    for k,state in before.items():assert_equal(state,o[k].state_dict())
    env=payload['identity']['recovery']['environment']
    assert env['execution_policy']['name']==policy.STRICT_POLICY and env['cudnn_version']==torch.backends.cudnn.version()
    assert env['runtime_controls']['deterministic_algorithms']
    assert bool(payload['center'] is not None)==(architecture=='resnet50')
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_wrong_policy_resume_rejected_before_tensor_mutation(synthetic,launch,monkeypatch,architecture):
    cfg,o,path,payload,_=strict_payload(synthetic,monkeypatch,architecture=architecture)
    o['cfg']['repro'].update(cuda_policy=policy.ORIGINAL_POLICY,deterministic=False,benchmark=True)
    monkeypatch.setattr(o['model'],'load_state_dict',lambda *a,**k:pytest.fail('State mutated'))
    before=copy.deepcopy(o['optimizer'].state_dict())
    with pytest.raises(ValueError,match='config|identity'):ck.restore_lodo_checkpoint(path,**o,cpu_inspection=True,validate_only=True)
    assert_equal(before,o['optimizer'].state_dict())


@pytest.mark.parametrize('field,value',[('uuid','GPU-other'),('driver','other'),('torch','other'),('cuda_build','other'),('cudnn_version',0),('execution_policy',{}),('runtime_controls',{})])
def test_strict_runtime_mismatch_rejected_before_cuda(synthetic,launch,monkeypatch,field,value):
    cfg,o,path,payload,publish=strict_payload(synthetic,monkeypatch)
    payload['identity']['recovery']['environment'][field]=value;publish()
    monkeypatch.setattr(dev,'select_lodo_device',lambda *a:pytest.fail('CUDA discovered'))
    with pytest.raises(ValueError,match='identity'):ck.restore_lodo_checkpoint(path,**o,cpu_inspection=True,validate_only=True)


@pytest.mark.parametrize('schema',[1,3,True])
def test_strict_payload_cannot_claim_old_or_unknown_schema(synthetic,launch,monkeypatch,schema):
    cfg,o,path,payload,publish=strict_payload(synthetic,monkeypatch)
    payload['lodo_schema_version']=schema;publish()
    with pytest.raises(ValueError):ck.read_checkpoint(path)


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_strict_terminal_noop_uses_production_orchestration_without_cuda(synthetic,launch,monkeypatch,architecture):
    cfg,o,path,payload,_=strict_payload(synthetic,monkeypatch,architecture=architecture,terminal=True)
    cfg['train']['save']['resume']=str(path)
    target_guards(monkeypatch)
    monkeypatch.setattr(train,'build_train_loader',lambda cfg:(o['loader'],o['loader'].dataset.num_classes))
    monkeypatch.setattr(train,'build_model',lambda cfg,num_classes,**k:FixtureModel(cfg,num_classes))
    monkeypatch.setattr(dev,'select_lodo_device',lambda *a:pytest.fail('Terminal accessed CUDA'))
    monkeypatch.setattr(train,'train_one_epoch',lambda *a,**k:pytest.fail('Terminal performed update'))
    hashes={p:ck.file_digest(p) for p in path.parent.iterdir() if p.is_file()}
    result=train.run_lodo_training(cfg)
    assert result.completed_primary_updates==3
    assert hashes=={p:ck.file_digest(p) for p in hashes}
    assert not torch.cuda.is_initialized()


def test_launch_environment_verified_in_fresh_cpu_only_subprocess(tmp_path):
    source="from reid.utils.lodo_cuda_policy import activate_policy; from test_lodo_config import lodo_config; import torch; c=lodo_config(); c['system']['device']='cuda'; c['repro'].update(cuda_policy='strict_cuda_deterministic_v1',deterministic=True,benchmark=False); activate_policy(c); assert not torch.cuda.is_initialized(); print('PRECUDA PASS')"
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tests')
    result=subprocess.run(['/home/addirakoze/environments/Reid/bin/python','-B','-c',source],env=env,cwd=train.REPO_ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'PRECUDA PASS' in result.stdout


# Schema-only synthetic metadata conversion; no CUDA or held-out loader is run.
from test_lodo_evaluation import setup_terminal, tree

@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_strict_terminal_schema_remains_cpu_inspectable(setup_terminal,architecture,monkeypatch):
    from reid.utils import lodo_evaluation as ev
    request,payload,proof,_=setup_terminal(architecture)
    cfg=strict(payload['cfg'])
    science=protocol.scientific_configuration(cfg)
    payload.update(lodo_schema_version=2,cfg=cfg)
    payload['identity'].update(schema_version=2,scientific_config=science,config_digest=protocol.configuration_identity(cfg))
    proof.update(schema_version=2,identity_digest=ck.digest(payload['identity']),config_digest=payload['identity']['config_digest'])
    path=Path(request['checkpoint']);torch.save(payload,path);proof['checkpoint_sha256']=ck.file_digest(path)
    path.with_suffix('.json').write_bytes(ck.canonical_json(proof))
    request['expected']['config_digest']=payload['identity']['config_digest']
    monkeypatch.setattr(ev,'build_test_loader',lambda *a,**k:pytest.fail('Target loader constructed'),raising=False)
    checked=ev.validate_terminal(request)
    assert checked[0]['lodo_schema_version']==2 and not torch.cuda.is_initialized()

@pytest.mark.parametrize('mutation',[
    lambda p:p['identity'].update(map_digest='0'*64),
    lambda p:p['identity']['protocol'].update(fold='LODO-2'),
    lambda p:p['identity']['scientific_config']['execution_policy'].update(deterministic_warn_only=True),
])
def test_strict_coherently_rehashed_wrong_map_fold_policy_rejected(synthetic,launch,monkeypatch,mutation):
    cfg,o,path,payload,publish=strict_payload(synthetic,monkeypatch)
    mutation(payload);publish()
    before=copy.deepcopy(o['model'].state_dict())
    with pytest.raises(ValueError):ck.restore_lodo_checkpoint(path,**o,cpu_inspection=True,validate_only=True)
    assert_equal(before,o['model'].state_dict())


def test_strict_generations_keep_atomic_index_and_reject_corrupt_or_missing(synthetic,launch,monkeypatch):
    cfg,o,path,payload,publish=strict_payload(synthetic,monkeypatch)
    run=path.parent.parent;request=copy.deepcopy(cfg);request['train']['save']['resume']=str(path.parent/'ckpt_last.pth')
    # Metadata-double generation of the SAME completed boundary, distinct immutable nonce.
    successor=path.with_name(recovery.generation_name(2,3))
    torch.save(payload,successor)
    proof=json.loads(path.with_suffix('.json').read_text());proof['checkpoint_sha256']=ck.file_digest(successor)
    successor.with_suffix('.json').write_bytes(ck.canonical_json(proof))
    replace=ck.os.replace
    def interrupted(src,dst):
        if str(dst).endswith('recovery_index.json'):raise OSError('publication interrupted')
        return replace(src,dst)
    with monkeypatch.context() as patch:
        patch.setattr(ck.os,'replace',interrupted)
        with pytest.raises(OSError,match='interrupted'):recovery.publish_generation(successor,proof)
    assert recovery.select_generation(request,run)[0]==path
    recovery.publish_generation(successor,proof)
    assert len(recovery._read_index(path.parent))==2
    successor.write_bytes(b'preserved corrupt diagnostic generation')
    selected,failures=recovery.select_generation(request,run)
    assert selected==path and len(failures)==1
    request['train']['save']['resume']=str(successor)
    with pytest.raises(ValueError):recovery.select_generation(request,run)
    assert successor.read_bytes()==b'preserved corrupt diagnostic generation'
    path.with_suffix('.json').unlink()
    request['train']['save']['resume']=str(path.parent/'ckpt_last.pth')
    with pytest.raises(ValueError,match='No retained'):recovery.select_generation(request,run)
    assert successor.exists() and not path.with_suffix('.json').exists()


def test_guard_rejects_switching_selected_policy_to_original(launch,inventory,cuda_double):
    cfg=strict(configured())
    _,guard,_=dev.select_lodo_device(cfg)
    cfg['repro'].update(cuda_policy=policy.ORIGINAL_POLICY,deterministic=False,benchmark=True)
    with pytest.raises(ValueError,match='policy changed'):guard()
