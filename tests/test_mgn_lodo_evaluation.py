"""Synthetic metadata, request, comparison and common CLI tests; no GPU campaign."""
import copy
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from reid.utils import mgn_lodo_evaluation as ev
from reid.utils import lodo_evaluation as old
from reid.utils.mgn_lodo_comparison import compare_stage_a
from reid.utils.mgn_lodo_reference import validate_request as reference_request
from reid.utils.lodo_protocol import FOLDS
from reid.utils.lodo_checkpoint import digest
from test_mgn_lodo_v2 import config
from test_multi_source_dataset import tree


def request():
    return dict(mode='lodo_held_out',version=2,architecture='mgn',fold='LODO-1',target='market1501',
        checkpoint='/not-opened/ckpt_terminal.pth',checkpoint_sha256='d'*64,
        expected=dict(config_digest='a'*64,map_digest='b'*64,data_digest='c'*64,num_classes=2510),
        output_dir='exp/lodo_eval/mgn_lodo_v2/synthetic/mgn/LODO-1/seed42/fixture',synthetic=True,device='cpu')


def test_new_request_explicit_and_historical_reader_rejects():
    ev.validate_request(request())
    with pytest.raises(ValueError):old.validate_request(request())


@pytest.mark.parametrize('key,value',[('version',1),('version',True),('architecture','pcb'),('target','duke'),('batch_size',64),('num_workers',1),('checkpoint_sha256','bad'),('synthetic',False),('unknown',True)])
def test_bad_request_before_terminal_read(monkeypatch,key,value):
    r=request();r[key]=value
    monkeypatch.setattr(old,'read_checkpoint',lambda *a:pytest.fail('terminal read before request validation'))
    with pytest.raises(ValueError):ev.validate_terminal(r)


def test_missing_eval_authority_fails_before_terminal_data_cuda(monkeypatch):
    monkeypatch.delenv('MGN_LODO_EVALUATION_RECEIPT',raising=False)
    monkeypatch.setattr(old,'read_checkpoint',lambda *a:pytest.fail('unapproved terminal read'))
    with pytest.raises(ValueError,match='separate scoped'):ev.evaluate_mgn_lodo(request(),request_path='unused')


def result():
    cfg=config();target=dict(metadata_hashes={'partition':'a'*64})
    return dict(record_type='lodo_held_out_result',protocol='mgn_lodo_v2',architecture='mgn',variant=cfg['model']['variant'],sources=list(FOLDS['LODO-1'][0]),target='market1501',
        representation=dict(images=cfg['data']['test']['images'],aug=cfg['data']['test']['aug']),evaluation=dict(configuration=dict(target=target,retrieval_code_sha256='f'*64)),
        metrics=dict(mAP=.7,mINP=.5,Rank1=.8,Rank5=.9,Rank10=.95))


def reference(source,value=.3):
    cfg=config();cfg.pop('protocol');cfg['data']['train']['dataset']=dict(name=source,split='trainval',format='processed')
    cfg['data']['test']['dataset']=dict(name='market1501',split='test',format='processed')
    cfg['train']=dict(epochs=120,save=dict(metric='mAP'));cfg['sched']['milestones']=[60,90]
    return dict(record_type='mgn_single_source_reference',reference_policy_version=1,architecture='mgn',source_dataset=source,target_dataset='market1501',resolved_config=cfg,
        checkpoint_sha256='e'*64,checkpoint_path='/frozen/'+source+'/ckpt_best.pth',source_checkpoint_unchanged=True,
        evaluation_provenance=dict(target_metadata_hashes={'partition':'a'*64},camera_filter='exclude_same_pid_same_camera',target_bn_adaptation=False,retrieval_code_sha256='f'*64),
        training=dict(budget=dict(unit='completed_epochs',epochs=120),checkpoint_selection='periodic_source_test_mAP_best',selected_epoch=70),
        **{k:value for k in ('mAP','mINP','Rank1','Rank5','Rank10')})


def write_reference(tmp_path,source,**kwargs):
    row=reference(source,**kwargs);p=tmp_path/(source+'.json');p.write_text(json.dumps(row));return p


def test_three_reference_mean_primary_oracles_explicit(tmp_path):
    paths=[write_reference(tmp_path,s,value=v) for s,v in zip(FOLDS['LODO-1'][0],(.1,.3,.5))]
    report=compare_stage_a(result(),paths)
    assert report['status']=='complete' and report['mean']['mAP']==pytest.approx(.3)
    assert report['primary_reference_policy']=='all_three_and_arithmetic_mean'
    assert report['retrospective_oracle_best_by_target_mAP']['source']=='msmt17'
    assert 'NOT target-independent' in report['oracle_interpretation']
    assert report['differences_pp']['mean']['mAP']==pytest.approx(40)
    assert report['lodo_training']['budget']['updates']==85200
    assert all(r['training']['budget']['epochs']==120 for r in report['references'])


@pytest.mark.parametrize('damage',['missing','duplicate','mean_flip','bn_adapt','metadata','checkpoint_selection','original_matrix'])
def test_no_missing_or_unmatched_reference_average(tmp_path,damage):
    paths=[write_reference(tmp_path,s) for s in FOLDS['LODO-1'][0]]
    if damage=='missing':paths.pop()
    elif damage=='duplicate':paths.append(paths[0])
    else:
        p=paths[0];row=json.loads(p.read_text())
        if damage=='mean_flip':row['resolved_config']['eval']['flip_aggregation']='mean'
        elif damage=='bn_adapt':row['evaluation_provenance']['target_bn_adaptation']=True
        elif damage=='metadata':row['evaluation_provenance']['target_metadata_hashes']={}
        elif damage=='checkpoint_selection':row['training']['checkpoint_selection']='target_best'
        else:row.pop('record_type')
        p.write_text(json.dumps(row))
    report=compare_stage_a(result(),paths)
    assert report['status']=='incomplete' and report['mean'] is None
    assert report['retrospective_oracle_best_by_target_mAP'] is None


def test_reference_request_exact_twelve_cells_and_no_policy_override():
    r=dict(mode='mgn_single_source_reference',version=1,source='duke',target='market1501',source_registry='registry.yaml',source_registry_sha256='a'*64,
        checkpoint='ckpt_best.pth',checkpoint_sha256='b'*64,target_config='target.yaml',target_config_sha256='c'*64,effective_target_sha256='d'*64,
        target_metadata_hashes={'partition':'e'*64},output_dir='unused',device='cuda',flip_aggregation='none')
    reference_request(r)
    for key,value in [('source','market1501'),('flip_aggregation','mean'),('device','cpu'),('optimizer','override')]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):reference_request(bad)


def test_common_training_cli_refuses_unqualified_yaml_before_model(monkeypatch):
    from scripts import train
    monkeypatch.setattr(train,'parse_args',lambda:type('Args',(),{'config':str(Path(__file__).resolve().parents[1]/'configs/lodo/mgn_lodo_v2/mgn_lodo-1_seed42.yaml'),'opts':[]})())
    monkeypatch.delenv('MGN_LODO_ADMISSION_RECEIPT',raising=False)
    monkeypatch.setattr(train,'build_model',lambda *a,**k:pytest.fail('model initialized before admission'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('GPU touched before admission'))
    with pytest.raises(ValueError,match='Phase21'):train.main()


@pytest.mark.parametrize('mode',['lodo_held_out','mgn_single_source_reference'])
def test_both_new_evaluation_cli_modes_refuse_before_any_execution(monkeypatch,mode):
    from scripts import evaluate
    monkeypatch.delenv('MGN_LODO_EVALUATION_RECEIPT',raising=False)
    monkeypatch.setattr(evaluate,'parse_args',lambda:type('Args',(),{'config':'not-read','opts':[],'weight':''})())
    import reid.utils.config as config_module
    payload=request()
    if mode=='mgn_single_source_reference':
        payload=dict(mode=mode,version=1,source='duke',target='market1501',source_registry='registry.yaml',source_registry_sha256='a'*64,checkpoint='ckpt_best.pth',checkpoint_sha256='b'*64,target_config='target.yaml',target_config_sha256='c'*64,effective_target_sha256='d'*64,target_metadata_hashes={'partition':'e'*64},output_dir='unused',device='cuda',flip_aggregation='none')
    monkeypatch.setattr(config_module,'load_config',lambda *a,**k:payload)
    # evaluate.py imports load_config inside main.
    with pytest.raises(ValueError,match='separate scoped'):evaluate.main()


@pytest.fixture
def mgn_terminal(tree,tmp_path,monkeypatch):
    """Fabricated terminal counters at the frozen budget; ZERO optimizer calls.

    Source mapping/serialization use real shared code and synthetic image trees.
    Runtime/device functions are explicit CPU doubles, not CUDA qualification.
    """
    from collections import Counter
    from test_multi_source_dataset import partition,change_partition,image
    from test_mgn_lodo_v2 import TinyMGN
    from reid.utils import mgn_lodo_protocol as protocol
    from reid.utils import lodo_checkpoint as ck
    import reid.utils.lodo_device as dev
    import reid.utils.lodo_recovery as recovery
    from reid.data.multi_source import build_multi_source_dataset
    from reid.data.index_observation import ObservedTrainLoader
    from test_lodo_updates import loader_for
    from reid.engine.update_control import UpdateProgress
    from reid.losses.build import build_criterion
    from reid.optim.build import build_optimizer,build_scheduler
    from test_lodo_evaluation import rewrite
    names=['00000100_0001_00000000.jpg','00000100_0002_00000001.jpg']
    for name in names:image(partition(tree,'market1501').parent/'images'/name)
    change_partition(tree,'market1501',lambda p:p.update(test_im_names=names,test_marks=[0,1]))
    cfg=config(root=tmp_path);cfg['data']['root']=str(tree)
    for row in cfg['data']['train']['dataset']['sources']+[cfg['data']['test']['dataset']]:
        path=tree/'msmt17/MSMT17_V2/list_train.txt' if row['name']=='msmt17' else partition(tree,row['name'])
        row['metadata_sha256']=ck.file_digest(path)
    output=protocol.validate_output_path
    monkeypatch.setattr(protocol,'validate_output_path',lambda c,project_root=None:output(c,project_root=project_root or tmp_path))
    monkeypatch.setattr(ck,'validate_runtime',lambda c:None)
    monkeypatch.setattr(dev,'validate_placement',lambda *a:torch.device('cpu'))
    monkeypatch.setattr(dev,'authorization_uuid',lambda c:'GPU-00000000-0000-0000-0000-000000000000')
    monkeypatch.setattr(recovery,'execution_environment',lambda c:{'fixture':'cpu-not-qualified'})
    ds=build_multi_source_dataset(cfg,None);loader=ObservedTrainLoader(loader_for(ds,'pk',0),sampler_mode='pk')
    model=TinyMGN(cfg,ds.num_classes);criterion=build_criterion(cfg,ds.num_classes,None)
    opt=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,opt)
    p=UpdateProgress(85200);p.bind(loader);budget=85200
    for key in ('completed_primary_updates','main_steps_returned','attempted_batches','forward_batches','backward_batches'):setattr(p,key,budget)
    p.attempted_images=64*budget
    by_identity=Counter({label:64*budget//ds.num_classes for label in range(ds.num_classes)})
    by_source=Counter()
    for label,n in by_identity.items():by_source[p.exposure.identities[label]['source_key']]+=n
    for phase in p.exposure.phases:
        p.exposure.presentations[phase]=copy.deepcopy(by_source);p.exposure.identity_presentations[phase]=copy.deepcopy(by_identity)
        p.exposure.distinct[phase]=set(by_identity);p.exposure.batch_counts[phase]=budget
        p.exposure.identity_slots[phase]=Counter({s:n//4 for s,n in by_source.items()})
    scheduler.prepare(budget)
    path=Path(cfg['experiment']['output_dir'])/'checkpoints/ckpt_terminal.pth';path.parent.mkdir(parents=True)
    ck.save_lodo_checkpoint(path,cfg=cfg,loader=loader,model=model,criterion=criterion,optimizer=opt,auxiliary=None,
                           scheduler=scheduler,progress=p,next_epoch=1,cursor=1,kind='terminal')
    payload,proof=ck.read_checkpoint(path);r=request();r.update(checkpoint=str(path),checkpoint_sha256=ck.file_digest(path),
        expected={k:payload['identity'][k] for k in ('config_digest','map_digest','data_digest','num_classes')})
    calls=[]
    import reid.models.build as models
    def builder(c,num_classes,initialize_pretrained=True):
        assert initialize_pretrained is False;calls.append(num_classes);return TinyMGN(c,num_classes)
    monkeypatch.setattr(models,'build_model',builder)
    return r,payload,proof,calls


def test_real_shared_terminal_reader_and_synthetic_target_uses_source_c(mgn_terminal,tmp_path):
    r,p,proof,calls=mgn_terminal
    value=ev.validate_terminal(r)
    assert value[0]['identity']['num_classes']==24
    record,comparison=old._evaluate_lodo(r,project_root=tmp_path,terminal_validator=ev.validate_terminal,
        output_validator=ev.output_path,metadata_reader=ev.metadata,retrieval_reader=ev.representation,
        device_selector=ev.select_device,execution_reader=ev.execution,evaluation_function=ev.evaluate_frozen_bn,
        comparison_function=compare_stage_a,protocol='mgn_lodo_v2',protocol_version=2)
    assert calls==[24] and record['training']['num_classes']==24
    assert record['protocol']=='mgn_lodo_v2' and record['authoritative'] is False
    assert record['representation']['dimension']==2048 and record['representation']['target_bn_adaptation'] is False
    assert record['evaluation']['configuration']['terminal_sha256']==r['checkpoint_sha256']
    assert comparison['status']=='incomplete' and comparison['mean'] is None


@pytest.mark.parametrize('damage',['head','counter','scheduler','schema','numerical_policy','pk_slots','source_c'])
def test_bad_terminal_refused_before_target_loader(mgn_terminal,monkeypatch,damage):
    r,p,proof,calls=mgn_terminal
    from test_lodo_evaluation import rewrite
    from reid.utils.lodo_checkpoint import file_digest
    def mutate(payload):
        if damage=='head':payload['model']['classifiers.heads.0.weight']=torch.zeros(1,2048)
        elif damage=='counter':payload['progress']['main_steps_returned']-=1
        elif damage=='scheduler':payload['scheduler']['completed_updates']-=1
        elif damage=='schema':payload['identity']['schema_version']=1;payload['lodo_schema_version']=1
        elif damage=='numerical_policy':payload['identity']['numerical_policy']['convolution_tf32_permitted']=False
        elif damage=='pk_slots':payload['progress']['exposure']['identity_slots']['successful'].clear()
        else:payload['identity']['num_classes']=1
    rewrite(Path(r['checkpoint']),mutate);r['checkpoint_sha256']=file_digest(r['checkpoint'])
    monkeypatch.setattr(old,'build_test_loader',lambda *a:pytest.fail('target before terminal validation'))
    with pytest.raises(ValueError):ev.validate_terminal(r)
    assert calls==[]


def test_amp_context_and_batch_stat_only_bn_refused():
    from reid.utils.mgn_lodo_evaluation import evaluate_frozen_bn
    model=nn.Sequential(nn.BatchNorm1d(2),nn.Linear(2,2))
    with torch.autocast('cpu',enabled=True):
        with pytest.raises(ValueError,match='AMP'):evaluate_frozen_bn({},model,None,torch.device('cpu'))
    model=nn.BatchNorm1d(2,track_running_stats=False)
    with pytest.raises(ValueError,match='batch-statistics'):evaluate_frozen_bn({},model,None,torch.device('cpu'))
