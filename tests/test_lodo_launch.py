"""External wrapper tests: tmp outputs, mocked tmux/exec/NVML; CPU only."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('lodo_launch',Path(__file__).resolve().parents[1]/'scripts/lodo_launch.py')
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)
FROZEN = launch.FROZEN_ROOT


def test_import_has_no_torch_or_training_dispatch():
    result = subprocess.run([str(launch.PYTHON),'-B','-c',
        'import runpy,sys;runpy.run_path('+repr(str(Path(launch.__file__)))+');assert "torch" not in sys.modules'],
        check=True,capture_output=True,text=True)
    assert result.returncode == 0


@pytest.mark.parametrize('row',json.loads((FROZEN/launch.INDEX).read_text())['runs'])
def test_all_eight_frozen_candidates_resolve(row):
    actual, manifest, path = launch.candidate(row['config'],FROZEN)
    assert actual == row
    assert manifest['immutable']['configuration']['path'] == row['config']
    assert path.is_file()


@pytest.mark.parametrize('value',['configs/stage_a.yaml','../escape.yaml'])
def test_unknown_and_traversing_config_refused(value):
    with pytest.raises(ValueError):
        launch.candidate(value,FROZEN)


def test_symlink_config_refused(tmp_path):
    (tmp_path/'alias').symlink_to(FROZEN/'configs')
    with pytest.raises(ValueError,match='alias'):
        launch.candidate(str(tmp_path/'alias/lodo/v1/resnet50_lodo-1_seed42.yaml'),FROZEN)


@pytest.fixture
def prepared(monkeypatch,tmp_path):
    root = tmp_path/'unified';root.mkdir();wrapper = root
    row = dict(config='configs/lodo/v1/resnet50_lodo-1_seed42.yaml',manifest='manifest.json')
    (root/'configs/lodo/v1').mkdir(parents=True)
    import shutil
    shutil.copyfile(FROZEN/row['config'], root/row['config'])
    (root/row['manifest']).write_text('manifest')
    policy = root/'policy.md'; policy.write_text('policy')
    run = root/'exp/lodo/v1/resnet50/LODO-1/seed42/unified-test'
    manifest = dict(immutable_sha256='immutable',operational=dict(proposed_output_dir=str(run)))
    from reid.utils.config import load_config
    from reid.utils.lodo_protocol import configuration_identity
    cfg = load_config(str(root/row['config']))
    binding = dict(path=str(root/'binding.json'), sha256='binding-sha', record=dict(
        root=str(root), commit='a'*40, config=row['config'], manifest=row['manifest'], run_dir=str(run),
        scientific_sha256=configuration_identity(cfg), dataset=dict(canonical_root=cfg['data']['root']),
        policy=dict(path=str(policy),sha256=launch.sha(policy)), interpreter=dict(path=str(launch.PYTHON),sha256='python-sha')))
    approval = dict(path='approval',sha256='approval-sha',record={'commit':'a'*40})
    monkeypatch.setattr(launch,'POLICY_SHA',launch.sha(policy))
    monkeypatch.setattr(launch,'load_binding',lambda *a:copy.deepcopy(binding))
    monkeypatch.setattr(launch,'candidate',lambda *a:(row,manifest,root/row['manifest']))
    monkeypatch.setattr(launch,'release_approval',lambda *a:copy.deepcopy(approval))
    from reid.utils import lodo_protocol as protocol
    original_namespace=protocol.validate_output_path
    def fixture_namespace(actual, project_root=None):
        if actual['train']['save'].get('resume'):
            assert (root/actual['experiment']['output_dir'])==run
            return run  # Full recovery compatibility is exercised by the worker tests.
        return original_namespace(actual, project_root=root)
    monkeypatch.setattr(protocol,'validate_output_path',fixture_namespace)
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    calls=[]
    def child(argv,**kwargs):
        calls.append((argv,kwargs))
        if '--_preflight' not in argv:
            return SimpleNamespace(stdout='',returncode=0)
        request=json.loads(argv[-1]); recovery = None
        if request['resume']:
            recovery=dict(path=str(run/'checkpoints/ckpt_epoch_000001_u000000003_g123456abcdef.pth'),
                sha256='checkpoint-sha',completed_primary_updates=3,center_updates=3,next_epoch=2,next_lr=[.000004782],fallback=[])
        return SimpleNamespace(stdout=json.dumps(dict(recovery=recovery,cuda_initialized=False)),returncode=0)
    monkeypatch.setattr(launch.subprocess,'run',child)
    return root,wrapper,row,run,calls


def test_successful_command_preparation_is_read_only(prepared):
    root,wrapper,row,run,calls=prepared
    plan=launch.prepare(row['config'],root=root,wrapper_root=wrapper)
    assert plan['command'][:5]==[str(launch.PYTHON),'-u',str(root/'scripts/train.py'),'--config',row['config']]
    assert plan['command'][5:]==['--opts', 'experiment.output_dir=exp/lodo/v1/resnet50/LODO-1/seed42/unified-test']
    assert not run.exists() and not (wrapper/'.local/lodo_launch').exists()
    assert len(calls)==1 and '--_preflight' in calls[0][0]
    assert plan['approval'] and plan['evidence']['cuda_initialized'] is False


def test_check_mode_does_not_consume_execution_approval(prepared,monkeypatch):
    root,wrapper,row,_,_=prepared
    monkeypatch.setattr(launch,'release_approval',lambda *a:pytest.fail('No approval in check-only mode'))
    assert launch.prepare(row['config'],root=root,wrapper_root=wrapper,require_release=False)['approval'] is None


@pytest.mark.parametrize('failure',['dirty','branch','policy','gpu','metadata'])
def test_preparation_failures_never_dispatch(prepared,monkeypatch,failure):
    root,wrapper,row,_,calls=prepared
    def fail(*a,**kw):raise ValueError(failure)
    if failure in ('dirty','branch'):monkeypatch.setattr(launch,'load_binding',fail)
    elif failure=='policy':monkeypatch.setattr(launch,'POLICY_SHA','bad')
    else:monkeypatch.setattr(launch.subprocess,'run',fail)
    with pytest.raises(ValueError):launch.prepare(row['config'],root=root,wrapper_root=wrapper)


def test_existing_run_rejected_before_worker(prepared):
    root,wrapper,row,run,calls=prepared
    run.mkdir(parents=True)
    with pytest.raises(ValueError,match='collision'):
        launch.prepare(row['config'],root=root,wrapper_root=wrapper)
    assert not calls


@pytest.mark.parametrize('state',['missing','empty','terminal','terminal_sidecar'])
def test_resume_never_silently_restarts(prepared,state):
    root,wrapper,row,run,calls=prepared
    if state!='missing':(run/'checkpoints').mkdir(parents=True)
    if state.startswith('terminal'):
        (run/'checkpoints'/('ckpt_terminal.json' if state=='terminal_sidecar' else 'ckpt_terminal.pth')).touch()
    with pytest.raises(ValueError):launch.prepare(row['config'],True,root=root,wrapper_root=wrapper)
    assert not calls


def test_resume_pins_validated_generation_without_recipe_override(prepared):
    root,wrapper,row,run,calls=prepared
    (run/'checkpoints').mkdir(parents=True)
    (run/'checkpoints/recovery_index.json').write_text('{}')
    plan=launch.prepare(row['config'],True,root=root,wrapper_root=wrapper)
    assert plan['command'][5]=='--opts'
    assert plan['command'][-1]=='train.save.resume='+plan['evidence']['recovery']['path']
    assert json.loads(calls[0][0][-1])['resume']==str(run/'checkpoints/ckpt_last.pth')
    assert plan['evidence']['recovery']['completed_primary_updates']==3


def test_insufficient_disk_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda p:SimpleNamespace(free=launch.MIN_DISK-1))
    with pytest.raises(ValueError,match='64 GiB'):launch.check_namespace(tmp_path/'new',False)


def test_environment_overrides_unsafe_inheritance(monkeypatch):
    for k in ['CUDA_VISIBLE_DEVICES','LODO_AUTHORIZED_GPU_UUID','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH','TORCH_HOME','PYTHONHOME']:
        monkeypatch.setenv(k,'wrong')
    env=launch.environment(FROZEN)
    assert env['CUDA_VISIBLE_DEVICES']==env['LODO_AUTHORIZED_GPU_UUID']==launch.GPU
    assert env['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert env['PYTHONPATH']==str(FROZEN)+os.pathsep+str(FROZEN/'scripts')
    assert env['PYTHONDONTWRITEBYTECODE']=='1'
    assert 'TORCH_HOME' not in env and 'PYTHONHOME' not in env


DECLARED_CONFIG = 'configs/lodo/v1/resnet50_lodo-1_seed42.yaml'


def declaration(root):
    return dict(schema_version=2, record_type='local_operator_execution_declaration',
        root=str(root), commit='a'*40, approved_by=launch.operator(),
        approval_reference='NON-AUTHORITATIVE synthetic local operator fixture',
        training={DECLARED_CONFIG:dict(binding_sha256='c'*64,allow_resume=True)}, evaluation={})


def approval_fixture(monkeypatch, root):
    binding = dict(sha256='c'*64,record=dict(root=str(root),commit='a'*40,config=DECLARED_CONFIG))
    monkeypatch.setattr(launch,'load_binding',lambda *a:copy.deepcopy(binding))


@pytest.mark.parametrize('change',['absent','base_commit','dirty','scope','resume','operator','reference'])
def test_release_gate_refuses_unreviewed_or_unapproved(tmp_path,monkeypatch,change):
    path=tmp_path/'.local/lodo_operational/v2/approval.json'
    path.parent.mkdir(parents=True)
    record=declaration(tmp_path);approval_fixture(monkeypatch,tmp_path)
    if change=='base_commit':record['commit']=launch.IMPLEMENTATION
    elif change=='scope':record['training']={}
    elif change=='resume':record['training'][DECLARED_CONFIG]['allow_resume']=False
    elif change=='operator':record['approved_by']='someone-else'
    elif change=='reference':record['approval_reference']=''
    if change!='absent':
        path.write_text(json.dumps(record));path.chmod(0o600)
    monkeypatch.setattr(launch,'verify_wrapper_source',lambda *a:None)
    def clean(*a):
        if change=='dirty':raise ValueError('dirty wrapper')
    if change=='dirty':monkeypatch.setattr(launch,'load_binding',clean)
    with pytest.raises(ValueError):launch.release_approval(tmp_path,DECLARED_CONFIG,True)


def test_valid_explicit_release_approval(tmp_path,monkeypatch):
    path=tmp_path/'.local/lodo_operational/v2/approval.json';path.parent.mkdir(parents=True)
    record=declaration(tmp_path);approval_fixture(monkeypatch,tmp_path)
    path.write_text(json.dumps(record));path.chmod(0o600)
    monkeypatch.setattr(launch,'verify_wrapper_source',lambda *a:None)
    monkeypatch.setattr(launch,'clean_head',lambda *a:None)
    assert launch.release_approval(tmp_path,DECLARED_CONFIG,True)['record']==record


def test_tmux_construction_is_quoted_and_forwards_environment(tmp_path):
    env=launch.environment(FROZEN)
    audit=tmp_path/'spaces and $shell; chars'
    command=launch.tmux_command('session',audit,env)
    assert command[:4]==['tmux','new-session','-d','-s']
    assert '--_execute' in command[-1]
    assert shlex.quote(str(audit/'launch-receipt.json')) in command[-1]
    assert shlex.quote(str(audit/'stdout.log')) in command[-1]
    for key in ('CUDA_VISIBLE_DEVICES','LODO_AUTHORIZED_GPU_UUID','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH'):
        assert key+'='+env[key] in command
    subprocess.run(['/bin/bash','-n','-c',command[-1]],check=True)


def test_dispatch_receipt_unique_logs_no_secrets_and_no_real_tmux(prepared,monkeypatch):
    root,wrapper,row,run,calls=prepared
    plan=launch.prepare(row['config'],root=root,wrapper_root=wrapper)
    monkeypatch.setattr(launch,'WRAPPER_ROOT',wrapper)
    monkeypatch.setattr(launch.shutil,'which',lambda p:'/usr/bin/tmux')
    monkeypatch.setenv('SECRET_TOKEN','must-not-appear')
    result=launch.dispatch(plan)
    receipt=Path(result['audit_dir'])/'launch-receipt.json'
    text=receipt.read_text()
    assert 'must-not-appear' not in text and 'SECRET_TOKEN' not in text
    assert not run.exists()
    assert calls[-1][0][0]=='tmux'
    assert str(receipt) in calls[-1][0][-1]
    assert json.loads(text)['plan']['approval']


def test_unapproved_plan_cannot_dispatch(tmp_path):
    with pytest.raises(ValueError,match='Check-only'):launch.dispatch({'approval':None})


def test_execute_rechecks_provenance_before_exec(tmp_path,monkeypatch):
    wrapper=tmp_path/'wrapper'
    audit=wrapper/'.local/lodo_launch/receipts/launch-test';audit.mkdir(parents=True)
    receipt=audit/'launch-receipt.json'
    old=dict(schema_version=2,root=str(wrapper),config='config',resume=False,command=['never-run'],config_sha256='original')
    receipt.write_text(json.dumps({'plan':old}));receipt.chmod(0o600)
    monkeypatch.setattr(launch,'WRAPPER_ROOT',wrapper)
    monkeypatch.setattr(launch,'prepare',lambda *a:dict(old,config_sha256='changed'))
    monkeypatch.setattr(launch.os,'execve',lambda *a:pytest.fail('No exec on changed provenance'))
    with pytest.raises(ValueError,match='Provenance changed'):launch.execute(receipt)
    assert (audit/'execution-failure.json').exists()


@pytest.mark.parametrize('revoked',[False,True])
def test_execute_keeps_lock_and_calls_only_frozen_trainer(tmp_path,monkeypatch,revoked):
    wrapper=tmp_path/'wrapper';audit=wrapper/'.local/lodo_launch/receipts/launch-test';audit.mkdir(parents=True)
    root=wrapper
    old=dict(schema_version=2,binding={'synthetic':True},config='config',resume=True,command=[str(launch.PYTHON),'-u','scripts/train.py','--config','config','--opts','train.save.resume=selected'],
        config_sha256='c',manifest_sha256='m',immutable_sha256='i',approval={'record':'test'},interpreter={'sha':'p'},policy={'sha':'s'},root=str(root),run_dir=str(root/'run'),
        evidence={'recovery':{'completed_primary_updates':7,'path':'selected','sha256':'checkpoint-sha'},'resolved_config':{'synthetic':True}},disk_free_bytes=launch.MIN_DISK,environment=launch.environment(root))
    receipt=audit/'launch-receipt.json';receipt.write_text(json.dumps({'plan':old}));receipt.chmod(0o600)
    monkeypatch.setattr(launch,'WRAPPER_ROOT',wrapper)
    monkeypatch.setattr(launch,'prepare',lambda *a:copy.deepcopy(old))
    monkeypatch.setattr(launch,'sha',lambda p:'checkpoint-sha')
    monkeypatch.setattr(launch.operational,'training_receipt',lambda *a,**k:{'synthetic':True})
    monkeypatch.setattr(launch,'release_approval',lambda *a:{'revoked':True} if revoked else copy.deepcopy(old['approval']))
    monkeypatch.chdir(tmp_path)
    class ExecObserved(BaseException):pass
    def execve(executable,argv,env):
        assert executable==str(launch.PYTHON) and argv==old['command']
        assert Path.cwd()==root and env['CUDA_VISIBLE_DEVICES']==launch.GPU
        assert (audit/'execution-admission.json').exists()
        raise ExecObserved()
    monkeypatch.setattr(launch.os,'execve',execve)
    if revoked:
        with pytest.raises(ValueError,match='changed before trainer exec'):launch.execute(receipt)
    else:
        with pytest.raises(ExecObserved):launch.execute(receipt)

# Reuse existing CPU-only checkpoint doubles; no real image or CUDA execution.
from test_lodo_checkpoint import synthetic, FixtureModel
from test_lodo_cuda_recovery import fake_cuda_generation


@pytest.mark.parametrize('mutation',[None,'center','rng','counter'])
def test_worker_reuses_full_cpu_recovery_validator(synthetic,monkeypatch,mutation):
    import torch
    import lodo_official_preparation as prep
    from reid.utils import config as config_module
    from reid.utils import lodo_device as device
    from reid.data import build as data_build
    from reid.models import build as model_build
    cfg,objects,path,payload,publish=fake_cuda_generation(synthetic,monkeypatch)
    monkeypatch.setattr(launch,'validate_preflight_request',lambda request:FROZEN)
    if mutation=='center':payload['center']=None
    elif mutation=='rng':payload['rng']['cuda']=[]
    elif mutation=='counter':payload['scheduler']['completed_updates']=0
    if mutation:publish()
    monkeypatch.setattr(config_module,'load_config',lambda *a:copy.deepcopy(cfg))
    monkeypatch.setattr(launch,'load_binding',lambda *a:{'synthetic':True})
    monkeypatch.setattr(launch,'configuration',lambda *a:copy.deepcopy(cfg))
    monkeypatch.setattr(launch,'candidate',lambda *a:(None,{},None))
    monkeypatch.setattr(launch,'verify_inputs',lambda *a,**k:dict(status='SYNTHETIC_CPU_ONLY'))
    monkeypatch.setattr(data_build,'build_train_loader',lambda c:(objects['loader'],objects['loader'].dataset.num_classes))
    builds=[]
    def build(c,num_classes,initialize_pretrained):
        assert initialize_pretrained is False
        builds.append('CPU reconstruction')
        return FixtureModel(c,num_classes)
    monkeypatch.setattr(model_build,'build_model',build)
    audit_calls=[]
    def audit(*a):
        audit_calls.append(a)
        return dict(name='NVIDIA TITAN Xp',free_mib=12000,processes=[])
    monkeypatch.setattr(device,'audit_gpu',audit)
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:'6.1')
    monkeypatch.chdir(synthetic)
    request=dict(root=str(FROZEN),config='unused',manifest='unused',run_dir=str(path.parent.parent),resume=cfg['train']['save']['resume'])
    if mutation:
        with pytest.raises(ValueError):launch.cpu_preflight(request)
        assert not audit_calls
    else:
        result=launch.cpu_preflight(request)
        assert result['recovery']['completed_primary_updates']==3
        assert result['recovery']['center_updates']==3
        assert result['recovery']['next_epoch']==2
        assert result['recovery']['path']==str(path)
        assert result['recovery']['next_lr']==list(objects['scheduler'].rates_after(3))
        assert audit_calls and result['cuda_initialized'] is False
    assert builds and not torch.cuda.is_initialized()


def test_preflight_failure_details_are_preserved(prepared,monkeypatch):
    root,wrapper,row,_,_=prepared
    def failed(*a,**k):raise subprocess.CalledProcessError(1,a[0],stderr='Metadata hash mismatch')
    monkeypatch.setattr(launch.subprocess,'run',failed)
    with pytest.raises(ValueError,match='Metadata hash mismatch'):
        launch.prepare(row['config'],root=root,wrapper_root=wrapper)


def test_worker_stdout_is_machine_readable_despite_model_logging(monkeypatch,capsys):
    def worker(request):
        print('[Model] ordinary frozen builder diagnostic')
        return dict(cuda_initialized=False)
    monkeypatch.setattr(launch,'cpu_preflight',worker)
    monkeypatch.setattr('sys.argv',['lodo_launch.py','--_preflight','{}'])
    launch.main()
    out=capsys.readouterr()
    assert json.loads(out.out)==dict(cuda_initialized=False)
    assert '[Model]' in out.err


def test_symlink_output_refused(tmp_path):
    target=tmp_path/'target';target.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(target)
    with pytest.raises(ValueError,match='alias'):launch.check_namespace(alias/'run',False)


def test_dirty_or_wrong_frozen_head_refused(monkeypatch):
    monkeypatch.setattr(launch,'git',lambda root,*args:'a'*40)
    with pytest.raises(ValueError,match='commit'):launch.clean_head(FROZEN,launch.IMPLEMENTATION)
    monkeypatch.setattr(launch,'git',lambda root,*args:launch.IMPLEMENTATION if 'rev-parse' in args else ' M changed')
    with pytest.raises(ValueError,match='Clean'):launch.clean_head(FROZEN,launch.IMPLEMENTATION)


def test_dependency_snapshot_failure_preserves_audit_without_tmux(prepared,monkeypatch):
    root,wrapper,row,_,_=prepared
    plan=launch.prepare(row['config'],root=root,wrapper_root=wrapper)
    monkeypatch.setattr(launch,'WRAPPER_ROOT',wrapper)
    monkeypatch.setattr(launch.shutil,'which',lambda p:'/usr/bin/tmux')
    def fail(argv,**kw):
        assert argv[0]!='tmux'
        raise subprocess.CalledProcessError(1,argv)
    monkeypatch.setattr(launch.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):launch.dispatch(plan)
    assert len(list((wrapper/'.local/lodo_launch/receipts').glob('*/dispatch-failure.json')))==1


@pytest.mark.parametrize('row',json.loads((FROZEN/launch.INDEX).read_text())['runs'])
def test_all_eight_build_original_trainer_commands_without_dispatch(monkeypatch,tmp_path,row):
    """Eight real candidate identities, synthetic namespaces, no official outputs."""
    import shutil
    root=tmp_path/'NON-AUTHORITATIVE-unified';root.mkdir()
    for relative in (row['config'],row['manifest'],launch.INDEX):
        dst=root/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(FROZEN/relative,dst)
    _,manifest,_=launch.candidate(row['config'],FROZEN)
    from reid.utils.config import load_config
    from reid.utils.lodo_protocol import configuration_identity
    cfg=load_config(str(root/row['config']))
    policy=root/'policy.md';policy.write_text('synthetic-policy')
    run=root/'exp/lodo/v1'/row['architecture']/row['fold']/'seed42/unified-command-fixture'
    binding=dict(path=str(root/'binding.json'),sha256='synthetic',record=dict(root=str(root),commit='a'*40,
        config=row['config'],manifest=row['manifest'],scientific_sha256=configuration_identity(cfg),
        dataset=dict(canonical_root=cfg['data']['root']),run_dir=str(run),policy=dict(path=str(policy),sha256=launch.sha(policy)),interpreter={}))
    monkeypatch.setattr(launch,'load_binding',lambda *a:copy.deepcopy(binding))
    monkeypatch.setattr(launch,'POLICY_SHA',launch.sha(policy))
    real_git=launch.operational.git
    monkeypatch.setattr(launch.operational,'git',lambda given,*a,**kw:real_git(FROZEN,*a,**kw))
    original_run=launch.subprocess.run
    def worker(argv,**kw):
        if '--_preflight' not in argv:
            assert argv[0]=='git'
            return original_run(argv,**kw)
        request=json.loads(argv[-1]);assert request['config']==row['config'] and request['manifest']==row['manifest']
        assert request['resume'] is None and kw['cwd']==root
        assert kw['env']['CUDA_VISIBLE_DEVICES']==launch.GPU
        return SimpleNamespace(stdout=json.dumps(dict(cuda_initialized=False,recovery=None)),returncode=0)
    monkeypatch.setattr(launch.subprocess,'run',worker)
    plan=launch.prepare(row['config'],require_release=False,root=root)
    assert plan['command'][:5]==[str(launch.PYTHON),'-u',str(root/'scripts/train.py'),'--config',row['config']]
    assert plan['immutable_sha256']==row['manifest_immutable_sha256'] and not run.exists()


@pytest.mark.parametrize('args',[
    ['--check','--_execute','receipt'],
    ['--resume','--_execute','receipt'],
    ['--config','config','--_execute','receipt'],
    ['--check','--_preflight','{}'],
    ['--_execute','receipt','--_preflight','{}'],
])
def test_internal_modes_cannot_override_read_only_or_public_flags(monkeypatch,args):
    monkeypatch.setattr('sys.argv',['lodo_launch.py',*args])
    for name in ('prepare','execute','cpu_preflight'):
        monkeypatch.setattr(launch,name,lambda *a,**kw:pytest.fail('No action before CLI rejection'))
    with pytest.raises(SystemExit) as error:launch.main()
    assert error.value.code==2


def test_private_record_rejects_shared_permissions_and_alias(tmp_path):
    path=tmp_path/'approval.json';path.write_text('{}');path.chmod(0o644)
    with pytest.raises(ValueError,match='private'):launch.private_record(path)
    path.chmod(0o600)
    assert launch.private_record(path)==b'{}'
    alias=tmp_path/'alias';alias.symlink_to(path)
    with pytest.raises(OSError):launch.private_record(alias)


def test_operator_identity_uses_uid_not_shell_claim(monkeypatch):
    expected=launch.pwd.getpwuid(os.getuid()).pw_name
    monkeypatch.setenv('USER','forged-user');monkeypatch.setenv('LOGNAME','forged-user')
    assert launch.operator()==expected


def test_executable_integrity_is_checked_even_when_git_claims_clean(tmp_path,monkeypatch):
    path=tmp_path/'scripts/lodo_launch.py';path.parent.mkdir();path.write_bytes(b'reviewed')
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**kw:b'reviewed')
    launch.verify_wrapper_source(tmp_path,'a'*40)
    path.write_bytes(b'changed despite skip-worktree')
    with pytest.raises(ValueError,match='differs'):launch.verify_wrapper_source(tmp_path,'a'*40)


def test_hidden_preflight_rejects_foreign_import_root(tmp_path):
    with pytest.raises(ValueError,match='canonical'):
        launch.validate_preflight_request(dict(root=str(tmp_path),config='config',manifest='manifest',run_dir='run',resume=None,binding_sha256='synthetic'))


@pytest.mark.parametrize('field,value',[
    ('manifest','other.json'),('run_dir','/tmp/other-run'),('resume','/tmp/other-checkpoint.pth')])
def test_hidden_preflight_rejects_tampered_candidate_namespace(monkeypatch,field,value):
    path='configs/lodo/v1/resnet50_lodo-1_seed42.yaml'
    row,manifest,_=launch.candidate(path,FROZEN)
    binding={'sha256':'synthetic','record':{'manifest':row['manifest'],'run_dir':str(FROZEN/'exp/lodo/v1/resnet50/LODO-1/seed42/unified-test')}}
    monkeypatch.setattr(launch,'load_binding',lambda *a:binding)
    request=dict(root=str(FROZEN),config=path,manifest=row['manifest'],binding_sha256='synthetic',
                 run_dir=binding['record']['run_dir'],resume=None)
    request[field]=value
    with pytest.raises(ValueError,match='mismatch'):launch.validate_preflight_request(request)


def test_revocation_during_preflight_prevents_dispatch(prepared,monkeypatch):
    root,wrapper,row,_,_=prepared
    values=iter([{'record':'initial'},{'record':'revoked'}])
    monkeypatch.setattr(launch,'release_approval',lambda *a:next(values))
    with pytest.raises(ValueError,match='changed during preflight'):
        launch.prepare(row['config'],root=root,wrapper_root=wrapper)


def test_existing_run_created_during_preflight_prevents_start(prepared,monkeypatch):
    root,wrapper,row,run,calls=prepared
    original=launch.subprocess.run
    def worker(*a,**kw):
        result=original(*a,**kw)
        run.mkdir(parents=True)
        return result
    monkeypatch.setattr(launch.subprocess,'run',worker)
    with pytest.raises(ValueError,match='collision'):
        launch.prepare(row['config'],root=root,wrapper_root=wrapper)


@pytest.mark.parametrize('mutation',[
    'scientific_commit','gpu','config_hash','manifest_path','manifest_hash','immutable',
    'architecture','fold','sources','target','run_dir','extra_binding','missing_binding',
    'duplicate_scope','record_type','old_schema'])
def test_operator_declaration_is_bound_to_exact_scientific_experiment(tmp_path,monkeypatch,mutation):
    record=declaration(tmp_path);approval_fixture(monkeypatch,tmp_path)
    binding=record['training'][DECLARED_CONFIG]
    if mutation=='scientific_commit':record['commit']='b'*40
    elif mutation=='duplicate_scope':record['training']={}
    elif mutation=='record_type':record['record_type']='cryptographically_authenticated'
    elif mutation=='old_schema':record['schema_version']=1
    elif mutation=='missing_binding':record['training']={}
    else:binding[mutation]='counterfeit'
    path=tmp_path/'.local/lodo_operational/v2/approval.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record));path.chmod(0o600)
    monkeypatch.setattr(launch,'clean_head',lambda *a:None)
    monkeypatch.setattr(launch,'verify_wrapper_source',lambda *a:None)
    with pytest.raises(ValueError):launch.release_approval(tmp_path,DECLARED_CONFIG,True)


@pytest.mark.parametrize('row',json.loads((FROZEN/launch.INDEX).read_text())['runs'])
def test_all_eight_operator_bindings_preserve_manifest_scope(row):
    actual,manifest,path=launch.candidate(row['config'],FROZEN)
    assert actual==row and manifest['immutable_sha256']==row['manifest_immutable_sha256']
    assert launch.sha(path)==launch.operational.sha(FROZEN/row['manifest'])


def test_operator_preparation_never_imports_torch(tmp_path):
    """Fresh parent prepares a mocked CPU-worker command using only stdlib."""
    source = '''
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('operator_fixture',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=Path(sys.argv[2]);r.mkdir()
config='configs/lodo/v1/resnet50_lodo-1_seed42.yaml'
p=r/config;p.parent.mkdir(parents=True);p.write_text('NON-AUTHORITATIVE config double')
manifest=r/'manifest.json';manifest.write_text('{}')
policy=r/'policy.md';policy.write_text('NON-AUTHORITATIVE policy double')
run=r/'exp/lodo/v1/resnet50/LODO-1/seed42/unified-parent-fixture'
binding={'sha256':'synthetic','record':{'commit':'a'*40,'run_dir':str(run),'policy':{'path':str(policy),'sha256':m.sha(policy)},'interpreter':{}}}
m.POLICY_SHA=m.sha(policy)
m.load_binding=lambda *a:binding
m.candidate=lambda *a:({'config':config,'manifest':'manifest.json'},{'immutable_sha256':'synthetic'},manifest)
m.subprocess.run=lambda *a,**k:SimpleNamespace(stdout=json.dumps({'cuda_initialized':False,'recovery':None}),returncode=0)
plan=m.prepare(config,root=r,require_release=False)
assert plan['command'][2]==str(r/'scripts/train.py')
assert 'torch' not in sys.modules and not run.exists()
print('STDLIB_PARENT_PASS')
'''
    result=subprocess.run([str(launch.PYTHON),'-B','-c',source,str(Path(launch.__file__)),str(tmp_path/'parent')],
                          check=True,capture_output=True,text=True)
    assert 'STDLIB_PARENT_PASS' in result.stdout
