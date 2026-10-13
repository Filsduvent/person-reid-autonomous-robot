"""Phase 7 synthetic CPU terminal/evaluation fixtures, never official runs.

Terminal counts below are fabricated metadata for schema tests at frozen budgets;
no 85,200/94,200-step training is executed and all outputs say synthetic.
"""
import copy
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from test_multi_source_dataset import tree, config, change_partition, partition, image
from reid.data.multi_source import build_multi_source_dataset, canonical_json
from reid.data.index_observation import ObservedTrainLoader
from reid.data.samplers import PKBatchSampler
from reid.data.collate import train_collate_fn
from reid.engine.update_control import UpdateProgress
from reid.losses.build import build_criterion
from reid.optim.build import build_optimizer, build_center_optimizer, build_scheduler
from reid.utils.checkpoint import make_reconstruction_metadata
from reid.utils import lodo_checkpoint as ck
from reid.utils import lodo_evaluation as ev
from reid.utils.lodo_comparison import compare_stage_a
from reid.utils.lodo_protocol import FOLDS
from reid.models.build import build_model as original_build_model


class EvalModel(nn.Module):
    def __init__(self,cfg,num_classes):
        super().__init__();self.architecture=cfg['protocol']['architecture']
        dim=2048 if self.architecture=='resnet50' else 1536
        self.feat_dim=dim
        self.backbone=nn.Linear(2,dim,bias=False)
        with torch.no_grad():
            self.backbone.weight.zero_();self.backbone.weight[0,0]=1;self.backbone.weight[1,1]=1
        if self.architecture=='resnet50': self.classifier=nn.Linear(dim,num_classes,bias=False)
        else:
            self.classifiers=nn.Module();self.classifiers.fc_list=nn.ModuleList(nn.Linear(256,num_classes) for _ in range(6))
        self.checkpoint_metadata=make_reconstruction_metadata(cfg,num_classes,dim)
    def forward(self,x):
        if x.ndim==4: x=x.mean((2,3))[:,:2]
        feat=self.backbone(x)
        return dict(emb=feat,feat_raw=feat*2,feat_bn=feat)


@pytest.fixture
def setup_terminal(tree,tmp_path,monkeypatch):
    # Fix only synthetic target fixtures to contain cross-camera positives.
    for name in ('market1501','duke','cuhk03'):
        base=partition(tree,name).parent
        names=['00000100_0001_00000000.jpg','00000100_0002_00000001.jpg']
        for ref in names: image(base/'images'/ref)
        change_partition(tree,name,lambda p,names=names:p.update(test_im_names=names,test_marks=[0,1]))
    base=tree/'msmt17/MSMT17_V2';ref='0000/0000_gallery_2_0000.jpg';image(base/'mask_test_v2'/ref)
    extra='0001/0001_gallery_1_0000.jpg';image(base/'mask_test_v2'/extra)
    Image.new('RGB',(4,8),color=(255,0,0)).save(base/'mask_test_v2'/extra)
    (base/'list_gallery.txt').write_text(f'{ref} 0\n{extra} 1\n')
    import reid.models.build as models
    calls=[]
    def builder(cfg,num_classes,initialize_pretrained=True):
        assert initialize_pretrained is False
        calls.append((cfg['protocol']['architecture'],num_classes))
        return EvalModel(cfg,num_classes)
    monkeypatch.setattr(models,'build_model',builder)
    def make(architecture='resnet50',fold='LODO-1'):
        cfg=config(tree,architecture,fold)
        ds=build_multi_source_dataset(cfg,None);classes=ds.num_classes
        if architecture=='resnet50': original=DataLoader(ds,batch_sampler=PKBatchSampler(ds.labels,16,4),collate_fn=train_collate_fn)
        else: original=DataLoader(ds,batch_size=64,shuffle=True,drop_last=True,collate_fn=train_collate_fn)
        original.effective_batch_size=64
        loader=ObservedTrainLoader(original,sampler_mode='pk' if architecture=='resnet50' else 'random')
        model=EvalModel(cfg,classes);criterion=build_criterion(cfg,classes,model.feat_dim)
        opt=build_optimizer(cfg,model);aux=build_center_optimizer(cfg,criterion);sched=build_scheduler(cfg,opt)
        budget=cfg['train']['max_updates'];p=UpdateProgress(budget,auxiliary=aux is not None);p.bind(loader)
        for key in ('completed_primary_updates','main_steps_returned','attempted_batches','forward_batches','backward_batches'): setattr(p,key,budget)
        p.auxiliary_steps=budget if aux else 0;p.attempted_images=64*budget
        by_identity=Counter({label:64*budget//classes for label in range(classes)})
        by_identity[0]+=64*budget-sum(by_identity.values())
        by_source=Counter()
        for label,n in by_identity.items(): by_source[p.exposure.identities[label]['source_key']]+=n
        for phase in p.exposure.phases:
            p.exposure.presentations[phase]=copy.deepcopy(by_source)
            p.exposure.identity_presentations[phase]=copy.deepcopy(by_identity)
            p.exposure.distinct[phase]=set(by_identity);p.exposure.batch_counts[phase]=budget
            if p.exposure.identity_slots is not None: p.exposure.identity_slots[phase]=Counter({s:n//4 for s,n in by_source.items()})
        sched.prepare(budget)
        path=tmp_path/'terminals'/architecture/fold/'checkpoints/ckpt_terminal.pth';path.parent.mkdir(parents=True)
        ck.save_lodo_checkpoint(path,cfg=cfg,loader=loader,model=model,criterion=criterion,optimizer=opt,auxiliary=aux,scheduler=sched,progress=p,next_epoch=1,cursor=1,kind='terminal')
        payload,manifest=ck.read_checkpoint(path)
        request=dict(mode='lodo_held_out',version=1,architecture=architecture,fold=fold,target=FOLDS[fold][1],checkpoint=str(path),
                     expected={k:payload['identity'][k] for k in ('config_digest','map_digest','data_digest','num_classes')},
                     output_dir=f'exp/lodo_eval/v1/synthetic/{architecture}/{fold}/seed42/test',synthetic=True,batch_size=8,num_workers=0)
        return request,payload,manifest,calls
    return make


def rewrite(path,mutate):
    payload,manifest=ck.read_checkpoint(path);mutate(payload);torch.save(payload,path)
    identity=payload['identity'];p=payload['progress']
    manifest.update(checkpoint_sha256=ck.file_digest(path),identity_digest=ck.digest(identity),kind=payload['kind'],
                    completion_reason='budget_completed' if payload['kind']=='terminal' else 'recovery_boundary',
                    planned_primary_updates=p['max_updates'],planned_auxiliary_updates=p['max_updates'] if p['auxiliary'] else 0,
                    actual_primary_updates=p['completed_primary_updates'],actual_auxiliary_updates=p['auxiliary_steps'],
                    fold=identity['protocol']['fold'],map_digest=identity['map_digest'],config_digest=identity['config_digest'],data_digest=identity['data_digest'])
    path.with_suffix('.json').write_bytes(canonical_json(manifest))


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('fold',list(FOLDS))
def test_all_folds_source_C_original_target_factory_and_result(setup_terminal,tmp_path,monkeypatch,architecture,fold):
    request,payload,manifest,calls=setup_terminal(architecture,fold)
    before=Path(request['checkpoint']).read_bytes();order=[]
    original_reconstruct=ev.reconstruct_model;original_loader=ev.build_test_loader
    def reconstruct(p): order.append('model');return original_reconstruct(p)
    def target(cfg):
        order.append('target')
        assert cfg['data']['test']['dataset']['name']==request['target']
        assert cfg['data']['test']['images']==payload['cfg']['data']['test']['images']
        loader=original_loader(cfg)
        assert len(loader.dataset.pids)==(3 if request['target']=='msmt17' else 2) and loader.dataset.pids[0]==loader.dataset.pids[1]
        assert loader.dataset.cams[0]!=loader.dataset.cams[1]
        return loader
    monkeypatch.setattr(ev,'reconstruct_model',reconstruct);monkeypatch.setattr(ev,'build_test_loader',target)
    result,comparison=ev.evaluate_lodo(request,project_root=tmp_path)
    assert order==['model','target'] and calls==[(architecture,24)]
    assert result['training']['num_classes']==24
    assert result['representation']['dimension']==(2048 if architecture=='resnet50' else 1536)
    assert result['synthetic'] and not result['authoritative']
    assert result['terminal']['checkpoint_sha256']==manifest['checkpoint_sha256'] and result['metrics']==pytest.approx({k:1 for k in ev.METRICS})
    assert comparison['status']=='incomplete' and comparison['missing_sources']==list(FOLDS[fold][0])
    path=tmp_path/request['output_dir']/'lodo_result.json';assert ev.read_result(path)==result
    assert Path(request['checkpoint']).read_bytes()==before
    assert 'source_dataset' not in result
    assert any('official-new' in caveat for caveat in result['caveats'])


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(kind='recovery'),lambda p:p['identity'].update(target='duke'),
    lambda p:p['identity']['protocol'].update(fold='LODO-2'),lambda p:p['identity'].update(map_digest='0'*64),
    lambda p:p['identity'].update(config_digest='0'*64),lambda p:p['identity'].update(data_digest='0'*64),
    lambda p:p['identity'].update(feature_dim=1536),lambda p:p['identity'].update(precision='fp16'),
    lambda p:p['progress'].update(completed_primary_updates=85199),lambda p:p['progress'].update(auxiliary_steps=0),
    lambda p:p['progress'].update(failed=True),lambda p:p['progress'].update(consistent=False),
    lambda p:p.update(center=None),lambda p:p['model'].update({'classifier.weight':torch.zeros(999,2048)}),
    lambda p:p['identity']['mapping']['identities'][0].update(source_exposed_pid=987),
    lambda p:p['identity']['center'].update(reduction='sum'),lambda p:p['scheduler'].update(completed_updates=0),
])
def test_invalid_terminal_rejected_before_reconstruction_or_target(setup_terminal,tmp_path,monkeypatch,mutation):
    request,_,_,_=setup_terminal();rewrite(Path(request['checkpoint']),mutation)
    monkeypatch.setattr(ev,'reconstruct_model',lambda *a:pytest.fail('Weights reconstructed before gate'))
    monkeypatch.setattr(ev,'build_test_loader',lambda *a:pytest.fail('Target constructed before gate'))
    with pytest.raises(ValueError): ev.evaluate_lodo(request,project_root=tmp_path)


@pytest.mark.parametrize('failure',['hash','missing_sidecar','legacy','schema','wrong_request_target','wrong_expected_map','unknown_option','wrong_version'])
def test_request_integrity_and_legacy_fail_closed(setup_terminal,tmp_path,monkeypatch,failure):
    request,_,_,_=setup_terminal();path=Path(request['checkpoint'])
    if failure=='hash': path.write_bytes(path.read_bytes()[:-25])
    elif failure=='missing_sidecar': path.with_suffix('.json').unlink()
    elif failure=='legacy': torch.save(dict(model={}),path)
    elif failure=='schema':
        manifest=json.loads(path.with_suffix('.json').read_text());manifest['schema_version']=2;path.with_suffix('.json').write_text(json.dumps(manifest))
    elif failure=='wrong_request_target': request['target']='duke'
    elif failure=='wrong_expected_map': request['expected']['map_digest']='1'*64
    elif failure=='unknown_option': request['reranking']=True
    else: request['version']=2
    monkeypatch.setattr(ev,'reconstruct_model',lambda *a:pytest.fail('Premature reconstruction'))
    monkeypatch.setattr(ev,'build_test_loader',lambda *a:pytest.fail('Premature target'))
    with pytest.raises(ValueError): ev.evaluate_lodo(request,project_root=tmp_path)


def test_known_existing_evaluator_metrics_and_same_camera_filter(setup_terminal,tmp_path,monkeypatch):
    request,_,_,_=setup_terminal()
    batch=(torch.tensor([[1.,0.],[1.,0.],[.99,.1],[0.,1.]]),torch.tensor([100,100,999,100]),
           torch.tensor([1,1,2,2]),['q','excluded','negative','positive'],torch.tensor([0,1,1,1]))
    before=copy.deepcopy(batch)
    class Loader(list): pass
    loader=Loader([batch]);loader.dataset=SimpleNamespace(num_classes=99999)
    monkeypatch.setattr(ev,'build_test_loader',lambda cfg:loader)
    result,_=ev.evaluate_lodo(request,project_root=tmp_path)
    assert result['metrics']==pytest.approx(dict(mAP=.5,mINP=.5,Rank1=0,Rank5=1,Rank10=1))
    assert torch.equal(batch[1],before[1]) and torch.equal(batch[2],before[2])
    assert result['training']['num_classes']==24


def stage_a_rows(result,tmp_path,values=None):
    values=values or [dict(mAP=.4,mINP=.3,Rank1=.9,Rank5=.8,Rank10=.9),
                      dict(mAP=.6,mINP=.4,Rank1=.5,Rank5=.9,Rank10=.95),
                      dict(mAP=.6,mINP=.5,Rank1=.7,Rank5=.85,Rank10=.92)]
    # Complete Stage A schema with explicit historical split/filter evidence.
    from test_lodo_config import lodo_config
    cfg=lodo_config(result['architecture']);cfg.pop('protocol');cfg['data']['test']['dataset']=result['evaluation']['configuration']['target']['dataset']
    paths=[]
    for source,metrics in zip(result['sources'],values):
        saved=copy.deepcopy(cfg);saved['data']['train']['dataset']=dict(name=source,format='processed',split='trainval')
        row=dict(schema_version=1,experiment_id=f'{source}-run',source_dataset=source,target_dataset=result['target'],
                 architecture=saved['model']['name'],checkpoint_path=f'/synthetic/stage-a/{source}/ckpt_best.pth',resolved_config=saved,
                 metric_units='fraction',evaluation_provenance=dict(target_metadata_hashes=result['evaluation']['configuration']['target']['metadata_hashes'],camera_filter='exclude_same_pid_same_camera'),**metrics)
        path=tmp_path/f'{source}-cross_dataset.json';path.write_text(json.dumps(row));paths.append(str(path))
    return paths


def test_three_reference_means_envelope_best_run_tie_break_and_pp(setup_terminal,tmp_path):
    request,_,_,_=setup_terminal();result,_=ev.evaluate_lodo(request,project_root=tmp_path)
    result['metrics']=dict(mAP=.7,mINP=.55,Rank1=.8,Rank5=.95,Rank10=.98)
    paths=stage_a_rows(result,tmp_path);report=compare_stage_a(result,paths)
    assert report['status']=='complete' and report['synthetic'] and not report['authoritative']
    (tmp_path/'complete-comparison-example.json').write_text(json.dumps(report,indent=2))
    assert report['mean']['mAP']==pytest.approx((.4+.6+.6)/3)
    assert report['best_per_metric_envelope']['metrics']['Rank1']==.9 and not report['best_per_metric_envelope']['is_single_run']
    assert report['best_by_mAP_run']['source']==result['sources'][2]
    assert report['best_by_mAP_run']['metrics']['Rank1']==.7
    assert report['differences_pp']['best_by_mAP']['mAP']==pytest.approx(10)
    assert report['differences_pp']['envelope']['Rank1']==pytest.approx(-10)
    assert report['differences_pp']['mean']['mAP']==pytest.approx(100*(.7-(.4+.6+.6)/3))
    # Percentage input normalizes to the same fractions, never relative change.
    for path in paths:
        row=json.loads(Path(path).read_text());row['metric_units']='percentage'
        for key in ev.METRICS: row[key]*=100
        Path(path).write_text(json.dumps(row))
    assert compare_stage_a(result,paths)['differences_pp']==report['differences_pp']


@pytest.mark.parametrize('problem',['missing','duplicate','wrong_architecture','wrong_target','missing_hashes','wrong_filter','rerank','within_domain','stage_b','unknown_units','missing_metric','cuhk_new'])
def test_missing_or_incomparable_references_never_partial_mean(setup_terminal,tmp_path,problem):
    request,_,_,_=setup_terminal(fold='LODO-3' if problem=='cuhk_new' else 'LODO-1');result,_=ev.evaluate_lodo(request,project_root=tmp_path)
    paths=stage_a_rows(result,tmp_path)
    if problem=='missing': paths=paths[:2]
    elif problem=='duplicate': paths.append(paths[0])
    else:
        row=json.loads(Path(paths[0]).read_text())
        if problem=='wrong_architecture': row['architecture']='pcb'
        elif problem=='wrong_target': row['target_dataset']='duke'
        elif problem=='missing_hashes': row.pop('evaluation_provenance')
        elif problem=='wrong_filter': row['evaluation_provenance']['camera_filter']='none'
        elif problem=='rerank': row['resolved_config']['eval']['rerank']['enabled']=True
        elif problem=='within_domain': row['source_dataset']=result['target']
        elif problem=='stage_b': row=result
        elif problem=='unknown_units': row['metric_units']='relative_percent'
        elif problem=='missing_metric': row['Rank10']=None
        else: row['resolved_config']['data']['test']['dataset']['protocol']='new'
        Path(paths[0]).write_text(json.dumps(row))
    report=compare_stage_a(result,paths)
    assert report['status']=='incomplete' and report['mean'] is None and report['differences_pp'] is None
    assert ev.read_result(tmp_path/request['output_dir']/'lodo_result.json')['synthetic']


def test_collision_symlink_corrupt_result_and_terminal_immutability(setup_terminal,tmp_path,monkeypatch):
    request,_,_,_=setup_terminal();result,_=ev.evaluate_lodo(request,project_root=tmp_path)
    monkeypatch.setattr(ev,'build_test_loader',lambda *a:pytest.fail('Target accessed on collision'))
    with pytest.raises(ValueError): ev.evaluate_lodo(request,project_root=tmp_path)
    path=tmp_path/request['output_dir']/'lodo_result.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError): ev.read_result(path)
    other=copy.deepcopy(request);other['output_dir']=other['output_dir'].replace('/test','/alias')
    (tmp_path/other['output_dir']).symlink_to(tmp_path/request['output_dir'],target_is_directory=True)
    with pytest.raises(ValueError): ev.evaluation_output(other,tmp_path)
    other['output_dir']='exp/pcb/market1501'
    with pytest.raises(ValueError): ev.evaluation_output(other,tmp_path)


def test_evaluate_cli_dispatch_preserves_legacy_defaults(setup_terminal,monkeypatch,tmp_path):
    from scripts import evaluate
    from reid.utils import config as cfgmodule
    request,_,_,_=setup_terminal()
    monkeypatch.setattr('reid.utils.lodo_operational.evaluation_admission', lambda *a, **k: {'synthetic': True})
    monkeypatch.setattr(evaluate, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(evaluate,'parse_args',lambda:SimpleNamespace(config='unused',opts=[],weight=''))
    monkeypatch.setattr(cfgmodule,'load_config',lambda *a,**k:request)
    sentinel=({'artifacts': {'synthetic': True}}, {});monkeypatch.setattr(ev,'evaluate_lodo',lambda cfg:sentinel)
    assert evaluate.main() is sentinel
    monkeypatch.setattr(evaluate,'parse_args',lambda:SimpleNamespace(config='unused',opts=[],weight='override.pth'))
    with pytest.raises(ValueError): evaluate.main()


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_actual_architecture_source_class_reconstruction(setup_terminal,monkeypatch,architecture):
    request,_,_,_=setup_terminal(architecture)
    payload,_,_=ev.validate_terminal(request)
    monkeypatch.setattr('reid.models.build.build_model',original_build_model)
    original=original_build_model(payload['cfg'],num_classes=24,initialize_pretrained=False).eval()
    payload['model']=original.state_dict()
    restored=ev.reconstruct_model(payload).eval()
    heads=[restored.classifier] if architecture=='resnet50' else list(restored.classifiers.fc_list)
    assert len(heads)==(1 if architecture=='resnet50' else 6)
    assert all(head.out_features==24 for head in heads)
    with torch.no_grad():
        batch=torch.randn(2,3,192 if architecture=='pcb' else 128,64)
        expected=original(batch)['emb'];actual=restored(batch)['emb']
    assert actual.shape==(2,2048 if architecture=='resnet50' else 1536)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('failure',['target_metadata','sidecar_alias','unattributed','center_nonfinite','duplicate_distinct_artifacts','malformed_reference','preprocessing','embedding'])
def test_additional_integrity_and_comparison_cases(setup_terminal,tmp_path,monkeypatch,failure):
    request,payload,_,_=setup_terminal()
    if failure in ('target_metadata','sidecar_alias','unattributed','center_nonfinite'):
        if failure=='target_metadata':
            path=Path(next(iter(payload['identity']['provenance'][request['target']]['metadata_hashes'])))
            path.write_bytes(path.read_bytes()+b' ')
        elif failure=='sidecar_alias':
            path=Path(request['checkpoint']).with_suffix('.json');replacement=path.with_name('alias.json')
            path.rename(replacement);path.symlink_to(replacement)
        elif failure=='unattributed': rewrite(Path(request['checkpoint']),lambda p:p['progress'].update(unattributed_attempted_images=1))
        else: rewrite(Path(request['checkpoint']),lambda p:p['center']['centers'].fill_(float('nan')))
        monkeypatch.setattr(ev,'reconstruct_model',lambda *a:pytest.fail('Premature reconstruction'))
        monkeypatch.setattr(ev,'build_test_loader',lambda *a:pytest.fail('Premature target'))
        with pytest.raises(ValueError): ev.evaluate_lodo(request,project_root=tmp_path)
    else:
        result,_=ev.evaluate_lodo(request,project_root=tmp_path);paths=stage_a_rows(result,tmp_path)
        if failure=='duplicate_distinct_artifacts':
            other=tmp_path/'other.json';other.write_bytes(Path(paths[0]).read_bytes());paths.append(str(other))
        elif failure=='malformed_reference': Path(paths[0]).write_text('[]')
        else:
            row=json.loads(Path(paths[0]).read_text())
            if failure=='preprocessing': row['resolved_config']['data']['test']['images']['height']=999
            else: row['resolved_config']['model']['head']['eval_feat']='raw'
            Path(paths[0]).write_text(json.dumps(row))
        report=compare_stage_a(result,paths)
        assert report['status']=='incomplete' and report['mean'] is None


def test_comparison_corruption_fails_result_integrity(setup_terminal,tmp_path):
    request,_,_,_=setup_terminal();ev.evaluate_lodo(request,project_root=tmp_path)
    output=tmp_path/request['output_dir'];(output/'comparison.json').write_text('{}')
    with pytest.raises(ValueError,match='Comparison integrity'): ev.read_result(output/'lodo_result.json')
