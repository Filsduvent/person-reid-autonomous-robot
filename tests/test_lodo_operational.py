"""Binding v2 fixtures are NON-AUTHORITATIVE, CPU-only, and never commit Git."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from reid.utils import lodo_operational as op
from reid.utils.config import load_config, save_yaml
from reid.utils.lodo_protocol import configuration_identity

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = [r['config'] for r in json.loads((REPOSITORY/op.INDEX).read_text())['runs']]


@pytest.fixture
def binding_fixture(tmp_path, monkeypatch):
    import shutil
    root=tmp_path/'NON-AUTHORITATIVE-unified';root.mkdir()
    index=json.loads((REPOSITORY/op.INDEX).read_text())
    for relative in [op.INDEX]+[r[k] for r in index['runs'] for k in ('config','manifest')]:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPOSITORY/relative,path)
    policy=root/'standing-policy.md';policy.write_text('NON-AUTHORITATIVE policy test double')
    monkeypatch.setattr(op,'POLICY_SHA',op.sha(policy))
    real_git=op.git
    monkeypatch.setattr(op,'git',lambda given,*a,**kw:real_git(REPOSITORY,*a,**kw))
    def synthetic_freeze(given, commit):
        if given != root or commit != 'a'*40:
            raise ValueError('Synthetic actual full commit mismatch')
        return 'b'*40
    monkeypatch.setattr(op,'clean_head',synthetic_freeze)
    created={}
    def create(config=CONFIGS[0]):
        binding=op.make_binding(config,root=root,commit='a'*40,run_id='unified-synthetic-test',policy_path=policy)
        path=op.binding_path(config,root);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():op.write_private(path,binding)
        value=op.load_binding(config,root);created[config]=value
        return value
    binding=create()
    def declare(*,training=True,resume=True,evaluation=None):
        record=dict(schema_version=2,record_type='local_operator_execution_declaration',root=str(root),commit='a'*40,
            approved_by=op.pwd.getpwuid(os.getuid()).pw_name,
            approval_reference='NON-AUTHORITATIVE CPU fixture, not user execution authority',
            training={binding['record']['config']:dict(binding_sha256=binding['sha256'],allow_resume=resume)} if training else {},evaluation={})
        if evaluation:
            request_path,terminal=evaluation
            record['evaluation'][str(request_path.relative_to(root))]=dict(request_sha256=op.sha(request_path),
                binding_sha256=binding['sha256'],terminal_sha256=op.sha(terminal))
        path=root/'.local/lodo_operational/v2/approval.json'
        if path.exists():path.unlink()  # Fixture-local replacement only.
        op.write_private(path,record)
        return record
    return root,binding,create,declare


@pytest.mark.parametrize('config',CONFIGS)
def test_all_eight_bind_actual_unified_root_and_historical_science(binding_fixture,config):
    root,_,create,_=binding_fixture;b=create(config);cfg=op.configuration(b)
    assert b['record']['root']==str(root) and b['record']['commit']=='a'*40
    assert b['record']['historical_scientific_commit']==op.HISTORICAL_SCIENCE
    assert cfg['experiment']['output_dir'].endswith('unified-synthetic-test')
    assert configuration_identity(cfg)==b['record']['scientific_sha256']
    assert cfg['data']['num_workers']==0 and cfg['train']['max_updates'] in (85200,94200)
    assert not (root/'exp').exists()


@pytest.mark.parametrize('field',['schema_version','record_type','root','commit','source_tree','config_sha256',
    'manifest','manifest_sha256','immutable_sha256','scientific_sha256','dataset','pretrained','execution',
    'recovery','gpu_uuid','num_workers','run_dir','policy','interpreter','historical_scientific_commit'])
def test_binding_mutation_refused(binding_fixture,field):
    root,b,_,_=binding_fixture;record=copy.deepcopy(b['record'])
    record[field]='counterfeit'
    Path(b['path']).write_text(json.dumps(record))
    with pytest.raises((ValueError,KeyError,TypeError,OSError)):
        op.load_binding(b['record']['config'],root)


def test_private_record_and_exclusive_publication(tmp_path):
    path=tmp_path/'record.json';op.write_private(path,{'scope':'synthetic'})
    assert path.stat().st_mode & 0o777==0o600
    with pytest.raises(FileExistsError):op.write_private(path,{})
    alias=tmp_path/'alias';alias.symlink_to(path)
    with pytest.raises(ValueError,match='alias'):op.private_record(alias)
    path.chmod(0o644)
    with pytest.raises(ValueError,match='private'):op.private_record(path)


def test_no_binding_cannot_prepare_or_admit(RECORD=None):
    from scripts import lodo_launch as launch
    with pytest.raises(ValueError,match='binding v2'):
        launch.prepare(CONFIGS[0],require_release=False)


def admit_fixture(binding_fixture,monkeypatch):
    root,b,_,declare=binding_fixture;declare()
    cfg=op.configuration(b);auth=op.approval(b,root=root)
    receipt=op.training_receipt(b,auth,cfg,root=root)
    path=root/'.local/lodo_launch/receipts/launch-synthetic/execution-admission.json'
    path.parent.mkdir(parents=True);op.write_private(path,receipt)
    for k,v in op.environment(root).items():
        if k in op.CONTROLLED:monkeypatch.setenv(k,v)
    monkeypatch.setenv('LODO_ADMISSION_RECEIPT',str(path));monkeypatch.setenv('LODO_ADMISSION_SHA256',op.sha(path))
    return root,b,cfg,path,receipt


def test_valid_scoped_training_receipt_reaches_shared_dispatch(binding_fixture,monkeypatch):
    from scripts import train
    root,b,cfg,path,_=admit_fixture(binding_fixture,monkeypatch)
    monkeypatch.setattr(train,'REPO_ROOT',root)
    monkeypatch.setattr(train,'parse_args',lambda:SimpleNamespace(config=str(root/b['record']['config']),opts=[
        'experiment.output_dir='+cfg['experiment']['output_dir']]))
    sentinel=object();calls=[]
    monkeypatch.setattr(train,'run_lodo_training',lambda actual:(calls.append(actual),sentinel)[1])
    assert train.main() is sentinel and calls==[cfg]
    assert not (root/'exp').exists()


@pytest.mark.parametrize('field',['status','root','commit','config','binding','approval','run_dir','resume',
                                 'recovery_sha256','resolved_config_sha256','environment','schema_version','process'])
def test_incompatible_receipt_refused_before_execution(binding_fixture,monkeypatch,field):
    root,b,cfg,path,receipt=admit_fixture(binding_fixture,monkeypatch)
    receipt[field]='counterfeit';path.write_text(json.dumps(receipt));monkeypatch.setenv('LODO_ADMISSION_SHA256',op.sha(path))
    with pytest.raises(ValueError,match='Scoped admission'):
        op.validate_training_admission(b['record']['config'],cfg,root=root)


@pytest.mark.parametrize('mutation',['config','budget','output','workers','resume','environment','hash','revocation','collision'])
def test_requested_run_mismatch_refused(binding_fixture,monkeypatch,mutation):
    root,b,cfg,path,_=admit_fixture(binding_fixture,monkeypatch);config=b['record']['config']
    if mutation=='config':config=CONFIGS[-1]
    elif mutation=='budget':cfg['train']['max_updates']-=1
    elif mutation=='output':cfg['experiment']['output_dir']+='-other'
    elif mutation=='workers':cfg['data']['num_workers']=2
    elif mutation=='resume':cfg['train']['save']['resume']='/tmp/foreign-checkpoint.pth'
    elif mutation=='environment':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-other')
    elif mutation=='hash':monkeypatch.setenv('LODO_ADMISSION_SHA256','0'*64)
    elif mutation=='revocation':Path(root/'.local/lodo_operational/v2/approval.json').write_text('{}')
    else:(root/cfg['experiment']['output_dir']).mkdir(parents=True)
    with pytest.raises((ValueError,OSError,KeyError)):
        op.validate_training_admission(config,cfg,root=root)


@pytest.mark.parametrize('config',CONFIGS)
def test_direct_stage_b_rejects_boolean_before_loaders_or_device(monkeypatch,config):
    from scripts import train
    monkeypatch.delenv('LODO_ADMISSION_RECEIPT',raising=False);monkeypatch.setenv('LODO_APPROVED','1')
    monkeypatch.setattr(train,'parse_args',lambda:SimpleNamespace(config=str(REPOSITORY/config),opts=[]))
    for name in ('run_lodo_training','build_train_loader','build_test_loader','select_device','create_experiment_dirs'):
        monkeypatch.setattr(train,name,lambda *a,**k:pytest.fail('No execution before receipt rejection'))
    with pytest.raises(ValueError,match='scoped operational admission receipt'):train.main()


def eval_fixture(binding_fixture):
    root,b,_,declare=binding_fixture;bound=b['record'];cfg=op.configuration(b)
    terminal=Path(bound['run_dir'])/'checkpoints/ckpt_terminal.pth';terminal.parent.mkdir(parents=True)
    terminal.write_bytes(b'NON-AUTHORITATIVE mock terminal; never deserialize')
    _,manifest,_=op.candidate(bound['config'],root)
    row=json.loads((root/op.INDEX).read_text())['runs'][0]
    request=dict(mode='lodo_held_out',version=1,architecture=row['architecture'],fold=row['fold'],target=row['target'],
        checkpoint=str(terminal),expected=dict(config_digest=bound['scientific_sha256'],map_digest=bound['dataset']['map_digest'],
        data_digest=bound['dataset']['data_digest'],num_classes=bound['dataset']['num_classes']),
        output_dir=f"exp/lodo_eval/v1/benchmark/{row['architecture']}/{row['fold']}/seed42/unified-synthetic-test",
        synthetic=False,num_workers=0)
    path=root/'requests/NON-AUTHORITATIVE-evaluation.yaml';path.parent.mkdir();save_yaml(request,str(path))
    return root,b,cfg,terminal,request,path,declare


def test_training_scope_never_authorizes_evaluation(binding_fixture,monkeypatch):
    from reid.utils import lodo_evaluation as ev
    root,b,cfg,terminal,request,path,declare=eval_fixture(binding_fixture);declare()
    monkeypatch.setattr(ev,'validate_terminal',lambda *a:pytest.fail('No terminal/target load without eval scope'))
    with pytest.raises(ValueError,match='separate execution approval scope'):
        op.evaluation_admission(path,request,root=root)


def test_separate_eval_scope_reuses_existing_cli_evaluator(binding_fixture,monkeypatch):
    from reid.utils import lodo_evaluation as ev
    from scripts import evaluate
    root,b,cfg,terminal,request,path,declare=eval_fixture(binding_fixture);declare(training=False,evaluation=(path,terminal))
    monkeypatch.setattr(ev,'validate_terminal',lambda *a:({'cfg':cfg},{'checkpoint_sha256':op.sha(terminal)},terminal))
    sentinel=({'artifacts':{'synthetic':True}},{});calls=[]
    monkeypatch.setattr(ev,'evaluate_lodo',lambda actual:(calls.append(actual),sentinel)[1])
    monkeypatch.setattr(evaluate,'REPO_ROOT',root)
    monkeypatch.setattr(evaluate,'parse_args',lambda:SimpleNamespace(config=str(path),opts=[],weight=''))
    assert evaluate.main() is sentinel and calls==[request]
    assert list((root/'.local/lodo_evaluation/v2').glob('*/execution-admission.json'))
    assert not (root/'exp/lodo_eval').exists()
    with pytest.raises(ValueError,match='Training/continuation'):
        op.approval(b,root=root)


@pytest.mark.parametrize('mutation',['checkpoint','expected','target','request_bytes','terminal_bytes','scope','existing_output','override'])
def test_evaluation_scope_identity_and_namespace_refused(binding_fixture,monkeypatch,mutation):
    from reid.utils import lodo_evaluation as ev
    from scripts import evaluate
    root,b,cfg,terminal,request,path,declare=eval_fixture(binding_fixture);declare(evaluation=(path,terminal))
    monkeypatch.setattr(ev,'validate_terminal',lambda *a:({'cfg':cfg},{'checkpoint_sha256':op.sha(terminal)},terminal))
    if mutation=='checkpoint':request['checkpoint']='/tmp/historical/ckpt_terminal.pth'
    elif mutation=='expected':request['expected']['map_digest']='0'*64
    elif mutation=='target':request['target']='msmt17'
    elif mutation=='request_bytes':path.write_text(path.read_text()+'\n# changed\n')
    elif mutation=='terminal_bytes':terminal.write_bytes(b'changed')
    elif mutation=='scope':declare()
    elif mutation=='existing_output':(root/request['output_dir']).mkdir(parents=True)
    else:
        monkeypatch.setattr(evaluate,'parse_args',lambda:SimpleNamespace(config=str(path),opts=['synthetic=true'],weight=''))
        with pytest.raises(ValueError,match='overrides'):evaluate.main()
        return
    with pytest.raises((ValueError,KeyError)):
        op.evaluation_admission(path,request,root=root)


def test_clean_head_checks_actual_blob_membership_and_modes(tmp_path,monkeypatch):
    path=tmp_path/'scripts/a.py';path.parent.mkdir();path.write_text('original')
    data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    def git(root,*args,binary=False):
        if args[0]=='status':return ''
        if args[0]=='ls-tree':return f'100644 blob {blob}\tscripts/a.py\0'.encode()
        return 'a'*40
    monkeypatch.setattr(op,'git',git)
    op.clean_head(tmp_path,'a'*40)
    path.write_text('changed despite hidden Git status')
    with pytest.raises(ValueError,match='blob changed'):op.clean_head(tmp_path,'a'*40)
    path.write_bytes(data);(path.parent/'ignored.py').write_text('injection')
    with pytest.raises(ValueError,match='membership'):op.clean_head(tmp_path,'a'*40)


@pytest.mark.parametrize('commit',[op.HISTORICAL_SCIENCE,op.HISTORICAL_LAUNCHER,'short','A'*40])
def test_historical_or_invented_freeze_refused(tmp_path,monkeypatch,commit):
    monkeypatch.setattr(op,'git',lambda *a,**k:commit)
    with pytest.raises(ValueError):op.clean_head(tmp_path,commit)


def test_stage_a_standalone_cli_evaluates_without_stage_b_authority(tmp_path,monkeypatch):
    """NON-AUTHORITATIVE data/compute doubles, real Stage A CLI/retrieval/writer."""
    import logging
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from reid.data.protocol import ReIDEvalSample
    from reid.data.collate import test_collate_fn
    from reid.utils.checkpoint import make_reconstruction_metadata
    from reid.data import build as data_build
    from reid.models import build as model_build
    from scripts import evaluate
    cfg=load_config(str(REPOSITORY/'configs/baseline_market1501_resnet50_triplet.yaml'))
    cfg['system'].update(device='cpu',amp=False)
    cfg['data'].update(num_workers=0,pin_memory=False)
    cfg['model']['backbone']['pretrained']=False;cfg['model']['head']['embedding_dim']=8
    cfg['experiment'].update(output_dir=str(tmp_path/'NON-AUTHORITATIVE-stage-a-evaluation'),
                             notes='Synthetic CLI exemption test; not a benchmark result')
    class TinyRepresentation(nn.Module):
        def __init__(self,actual,classes):
            super().__init__();self.projection=nn.Linear(3,8,bias=False);self.classifier=nn.Linear(8,classes,bias=False)
            self.checkpoint_metadata=make_reconstruction_metadata(actual,classes,8)
            with torch.no_grad():
                self.projection.weight.zero_();self.projection.weight[0,0]=1;self.projection.weight[1,1]=1
        def forward(self,images):
            feat=self.projection(images.mean((2,3)))
            return dict(emb=feat,feat_raw=feat,feat_bn=feat,logits=self.classifier(feat))
    model=TinyRepresentation(cfg,3);checkpoint=tmp_path/'NON-AUTHORITATIVE-stage-a.pth'
    torch.save(dict(model=model.state_dict(),cfg=copy.deepcopy(cfg),epoch=1,reconstruction=model.checkpoint_metadata),checkpoint)
    cfg['eval']['weight']=str(checkpoint)
    path=tmp_path/'NON-AUTHORITATIVE-stage-a-evaluation.yaml';save_yaml(cfg,str(path))
    red=torch.zeros(3,2,2);red[0]=1;green=torch.zeros(3,2,2);green[1]=1
    samples=[ReIDEvalSample(red,11,1,'query',0),ReIDEvalSample(red,11,2,'positive',1),ReIDEvalSample(green,22,2,'negative',1)]
    loader=DataLoader(samples,batch_size=2,collate_fn=test_collate_fn)
    def build(actual,num_classes,initialize_pretrained):
        assert num_classes==3 and initialize_pretrained is False
        return TinyRepresentation(actual,num_classes)
    def forbidden(*a,**k):pytest.fail('Stage A must not require Stage B authority')
    monkeypatch.setattr(op,'evaluation_admission',forbidden);monkeypatch.setattr(op,'validate_training_admission',forbidden)
    monkeypatch.setattr(model_build,'build_model',build)
    monkeypatch.setattr(data_build,'build_test_loader',lambda actual:loader)
    monkeypatch.setattr(evaluate,'parse_args',lambda:SimpleNamespace(config=str(path),opts=[],weight=''))
    monkeypatch.setattr(logging.getLogger('reid.evaluate'),'handlers',[])
    evaluate.main()
    records=[json.loads(p.read_text()) for p in (Path(cfg['experiment']['output_dir'])/'metrics').glob('*.json')]
    assert records and all(r['mAP']==1 and r['Rank1']==1 for r in records)
    assert not (tmp_path/'.local').exists() and not torch.cuda.is_initialized()
