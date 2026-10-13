"""CPU-only official-candidate integrity tests; no training/evaluation jobs."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official_preparation',ROOT/'scripts/lodo_official_preparation.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
EVIDENCE=ROOT/'docs/lodo_phase10_evidence'


@pytest.fixture(autouse=True)
def isolated_candidate_root(tmp_path, monkeypatch):
    """Synthetic operational copies; never depend on official output existence.

    Keep scientific identities/negative checks intact. Only local root, output
    paths and their file hashes differ from the historical fixture copies.
    """
    import shutil
    from reid.utils.config import load_config, save_yaml
    source = ROOT
    isolated = tmp_path/'NON-AUTHORITATIVE-candidates'
    shutil.copytree(EVIDENCE, isolated/'docs/lodo_phase10_evidence')
    (isolated/'scripts').mkdir()
    shutil.copyfile(source/'scripts/lodo_official_preparation.py', isolated/'scripts/lodo_official_preparation.py')
    for manifest_path in (isolated/'docs/lodo_phase10_evidence/manifests').glob('*.json'):
        manifest = json.loads(manifest_path.read_text()); imm = manifest['immutable']
        config_path = isolated/imm['configuration']['path']
        cfg = load_config(str(source/imm['configuration']['path']))
        cfg['experiment']['output_dir'] = str(isolated/cfg['experiment']['output_dir'])
        config_path.parent.mkdir(parents=True, exist_ok=True); save_yaml(cfg, str(config_path))
        imm['configuration']['file_sha256'] = prep.file_digest(config_path)
        request_path = isolated/imm['evaluation']['request']
        request = load_config(str(request_path))
        request['checkpoint'] = str(Path(cfg['experiment']['output_dir'])/'checkpoints/ckpt_terminal.pth')
        request['output_dir'] = str(isolated/request['output_dir']); save_yaml(request, str(request_path))
        imm['evaluation']['request_sha256'] = prep.file_digest(request_path)
        manifest['operational']['proposed_output_dir'] = cfg['experiment']['output_dir']
        manifest['immutable_sha256'] = prep.digest(imm)
        manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(prep, 'ROOT', isolated)
    monkeypatch.setattr(__import__('sys').modules[__name__], 'ROOT', isolated)
    monkeypatch.setattr(__import__('sys').modules[__name__], 'EVIDENCE', isolated/'docs/lodo_phase10_evidence')
    # Evaluation namespace validator remains real, now scoped to fixture root.
    from reid.utils import lodo_protocol as protocol
    original_namespace = protocol.validate_output_path
    monkeypatch.setattr(protocol, 'validate_output_path', lambda cfg, project_root=None: original_namespace(cfg, project_root=project_root or isolated))
    from reid.utils import lodo_evaluation as evaluation
    original = evaluation.evaluation_output
    monkeypatch.setattr(evaluation, 'evaluation_output', lambda request, project_root=None: original(request, project_root or isolated))


def proposal(arch='resnet50',fold='lodo-1'):
    return json.loads((EVIDENCE/'manifests'/f'{arch}_{fold}.json').read_text())


@pytest.mark.parametrize('arch',['resnet50','pcb'])
@pytest.mark.parametrize('fold',[f'lodo-{i}' for i in range(1,5)])
def test_eight_configs_use_actual_cli_without_dispatch(monkeypatch,arch,fold):
    from scripts import train
    from reid.utils.config import load_config
    from reid.utils.config_schema import validate_config
    from reid.utils.lodo_protocol import configuration_identity
    p=proposal(arch,fold);path=ROOT/p['immutable']['configuration']['path']
    monkeypatch.setattr('sys.argv',['train.py','--config',str(path)])
    def prohibited(*args,**kwargs):raise AssertionError('No dispatch/target/device/output allowed')
    for name in ('run_lodo_training','build_test_loader','build_train_loader','select_device','create_experiment_dirs'):
        monkeypatch.setattr(train,name,prohibited)
    args=train.parse_args();cfg=load_config(args.config,overrides=args.opts)
    validate_config(cfg);prep.validate_reid_config(cfg)
    assert configuration_identity(cfg)==p['immutable']['configuration']['scientific_sha256']
    assert cfg['train']['max_updates']==(85200 if arch=='resnet50' else 94200)
    assert cfg['data']['num_workers']==0
    assert not (ROOT/cfg['experiment']['output_dir']).exists()
    assert cfg['data']['test']['dataset']['name'] not in [s['name'] for s in cfg['data']['train']['dataset']['sources']]


def test_proposal_serialization_is_stable_and_operations_separate():
    p=proposal();prep.validate_envelope(p)
    assert prep.digest(json.loads(prep.canonical_json(p['immutable'])))==p['immutable_sha256']
    q=copy.deepcopy(p);q['operational']['timestamp_utc']='later';q['operational']['resume']='/operator/request'
    assert q['immutable_sha256']==p['immutable_sha256'];prep.validate_envelope(q)
    q['immutable']['dataset']['num_classes']+=1
    with pytest.raises(ValueError,match='digest'):prep.validate_envelope(q)


@pytest.mark.parametrize('mutation',['unknown','version','frozen_commit','approval','allocation','unknown_provenance'])
def test_false_freeze_unknown_provenance_and_approval_rejected(mutation):
    p=proposal()
    if mutation=='unknown':p['unexpected']=True
    elif mutation=='version':p['schema_version']=True
    elif mutation=='frozen_commit':p['immutable']['code']['frozen_commit']='0'*40
    elif mutation=='approval':p['operational']['approval_status']='APPROVED'
    elif mutation=='allocation':p['operational']['allocation_status']='ALLOCATED'
    else:p['immutable']['unknown_provenance']=True
    p['immutable_sha256']=prep.digest(p['immutable'])
    with pytest.raises(ValueError):prep.validate_envelope(p)


@pytest.mark.parametrize('field',['map_digest','sidecar_digest','data_digest','provenance'])
def test_same_class_counterfeit_provenance_rejected_before_execution(monkeypatch,tmp_path,field):
    monkeypatch.setattr(prep, 'verify_runtime', lambda execution: None)
    original_digest=prep.file_digest
    expected=proposal()['immutable']['pretrained']
    monkeypatch.setattr(prep, 'file_digest', lambda path: expected['sha256'] if str(path)==expected['path'] else original_digest(path))
    p=proposal();live=copy.deepcopy(p['immutable']['dataset'])
    p['immutable']['dataset'][field]={'unknown':'provenance'} if field=='provenance' else '0'*64
    assert p['immutable']['dataset']['num_classes']==live['num_classes']
    # Keep the synthetic request internally consistent with the counterfeit map/data,
    # so all four mutations reach the intended LIVE provenance comparison.
    if field in ('map_digest','data_digest'):
        from reid.utils.config import load_config, save_yaml
        request_path=ROOT/p['immutable']['evaluation']['request']
        request=load_config(str(request_path));request['expected'][field]=p['immutable']['dataset'][field]
        save_yaml(request,str(request_path))
        p['immutable']['evaluation']['request_sha256']=original_digest(request_path)
    p['immutable_sha256']=prep.digest(p['immutable'])
    path=tmp_path/'tampered.json';path.write_text(json.dumps(p));calls=[]
    monkeypatch.setattr(prep,'dataset_evidence',lambda cfg:(calls.append('live provenance') or live,{}))
    with pytest.raises(ValueError,match='Live dataset/map/sidecar/provenance'):
        prep.verify_manifest(path,check_code=False)
    assert calls==['live provenance']


def test_metadata_guard_blocks_images_target_and_cuda():
    import torch
    import reid.data.build as build
    from PIL import Image
    with prep.metadata_only():
        for action in (lambda:Image.open('/no/image'),lambda:torch.cuda.init(),
                       lambda:build.build_test_loader({}),lambda:build._build_market1501_test_dataset(None,None,None,None)):
            with pytest.raises(AssertionError,match='prohibited'):action()
    assert not torch.cuda.is_initialized()


def test_source_membership_and_hash_change_fail_closed(monkeypatch,tmp_path):
    root=tmp_path; (root/'reid').mkdir();(root/'reid/a.py').write_text('original')
    code=dict(base_head='1'*40,state='dirty_worktree_unfrozen',frozen_commit=None,
              source_sha256={'reid/a.py':prep.file_digest(root/'reid/a.py')})
    code['source_snapshot_sha256']=prep.digest(code['source_sha256'])
    monkeypatch.setattr(subprocess,'check_output',lambda *args,**kwargs:'1'*40+'\n')
    prep.verify_code(code,root=root)
    (root/'reid/b.py').write_text('unexpected')
    with pytest.raises(ValueError,match='membership'):prep.verify_code(code,root=root)
    (root/'reid/a.py').write_text('changed')
    with pytest.raises(ValueError,match='changed'):prep.verify_code(code,root=root)


def test_frozen_revision_requires_actual_full_head_and_clean_status(monkeypatch,tmp_path):
    code=dict(base_head='1'*40,state='dirty_worktree_unfrozen',frozen_commit=None,source_sha256={})
    code['source_snapshot_sha256']=prep.digest({})
    def query(command,**kwargs):return '2'*40+'\n' if 'rev-parse' in command else ' M dirty.py\n'
    monkeypatch.setattr(subprocess,'check_output',query)
    with pytest.raises(ValueError,match='actual full HEAD'):prep.verify_code(code,root=tmp_path,frozen_commit='3'*40)
    with pytest.raises(ValueError,match='clean'):prep.verify_code(code,root=tmp_path,frozen_commit='2'*40)


def test_launch_and_evaluation_are_not_verifier_cli_actions():
    import ast
    tree=ast.parse((ROOT/'scripts/lodo_official_preparation.py').read_text())
    calls={getattr(n.func,'id',getattr(n.func,'attr','')) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not calls.intersection({'run_lodo_training','train_one_epoch','evaluate_lodo','evaluate_reid','select_lodo_device'})


def test_rehashed_wrong_runtime_rejected_without_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, NVIDIA TITAN Xp, 570.211.01\n')
    p=proposal();p['immutable']['execution_candidate']['qualified_runtime']['torch']='wrong-runtime'
    p['immutable_sha256']=prep.digest(p['immutable']);path=tmp_path/'runtime.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Runtime differs'):
        prep.verify_manifest(path,check_code=False)


def test_rehashed_wrong_device_rejected_without_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c, NVIDIA TITAN Xp, 570.211.01\n')
    p=proposal();p['immutable']['execution_candidate']['qualified_runtime']['uuid']='GPU-wrong'
    p['immutable_sha256']=prep.digest(p['immutable']);path=tmp_path/'device.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Candidate device'):
        prep.verify_manifest(path,check_code=False)
