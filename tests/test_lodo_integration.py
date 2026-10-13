"""NON-AUTHORITATIVE Phase 8 CPU integration, original data/transform/engine path.

Only the expensive backbone/part computation is doubled. Short budgets are
fixture-local: Phase 6's existing _expect substitution and a checked AST clone
of the Phase 7 validator with ONLY its two official budget literals replaced.
Production validators must reject these short artifacts. No production smoke
mode/config or scientific setting is introduced. All YAML, images and outputs
are temporary; the real source loaders, transforms, samplers, losses, optimizer,
scheduler, checkpoint/recovery, evaluation and reporting code execute.
"""
import ast
import copy
from collections import Counter
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from test_multi_source_dataset import tree, config, partition, image, change_partition
from test_lodo_checkpoint import synthetic, FixtureModel, assert_equal, target_guards
from test_lodo_evaluation import stage_a_rows
from reid.data import build as data_build
from reid.data.protocol import ReIDTrainSample
from reid.engine.update_control import UpdateFailure
from reid.utils import lodo_checkpoint as checkpoint
from reid.utils import lodo_evaluation as evaluation
from reid.utils import lodo_protocol as protocol
from reid.utils.config import save_yaml
from scripts import train, evaluate


class IntegrationModel(FixtureModel):
    """Linear compute double retaining C, D, BN routing, dropout and six heads."""
    def __init__(self,cfg,classes):
        super().__init__(cfg,classes)
        if self.arch=='resnet50': self.bnneck=nn.BatchNorm1d(2048)
    def forward(self,x):
        if x.ndim==4: x=x.mean((2,3))[:,:2]
        self.inputs.append(x.detach().clone())
        if len(self.inputs)==self.fail_at: raise RuntimeError('synthetic interruption')
        raw=self.dropout(self.backbone(x/100))
        if self.arch=='resnet50':
            bn=self.bnneck(raw)
            return dict(emb=F.normalize(bn,dim=1),feat_raw=raw,feat_bn=bn,logits=self.classifier(bn))
        return dict(emb=raw,logits=tuple(head(part) for head,part in zip(self.classifiers.fc_list,raw.chunk(6,dim=1))))


def enlarge_synthetic_sources(root):
    """Three-plus batches per epoch permit persisted boundary + partial stop."""
    for source in protocol.DATASET_ORDER:
        if source=='msmt17':
            base=root/'msmt17/MSMT17_V2';rows=[]
            for pid in range(8):
                for slot in range(12):
                    ref=f'{pid:04d}/0000_train_1_{slot:04d}.jpg';image(base/'mask_train_v2'/ref)
                    rows.append(f'{ref} {pid}\n')
            (base/'list_train.txt').write_text(''.join(rows))
        else:
            base=partition(root,source).parent
            names=[f'{pid:08d}_0001_{slot:08d}.jpg' for pid in range(8) for slot in range(12)]
            for name in names: image(base/'images'/name)
            change_partition(root,source,lambda p,names=names:p.update(train_im_names=names[:48],val_im_names=names[48:],trainval_im_names=names,val_marks=[0]*48))
    # Original target loaders require a valid cross-camera positive.
    for source in ('market1501','duke','cuhk03'):
        base=partition(root,source).parent
        names=['00000100_0001_00000000.jpg','00000100_0002_00000001.jpg']
        for name in names: image(base/'images'/name)
        change_partition(root,source,lambda p,names=names:p.update(test_im_names=names,test_marks=[0,1]))
    base=root/'msmt17/MSMT17_V2';positive='0000/0000_gallery_2_0000.jpg';negative='0001/0001_gallery_1_0000.jpg'
    image(base/'mask_test_v2'/positive);image(base/'mask_test_v2'/negative)
    Image.new('RGB',(4,8),color=(255,0,0)).save(base/'mask_test_v2'/negative)
    (base/'list_gallery.txt').write_text(f'{positive} 0\n{negative} 1\n')


def smoke_terminal_gate(budget):
    """Test-only constant substitution, preserving EVERY other gate statement.

    Avoid adding a short-budget escape hatch to production evaluation. Assert
    the exact original literal set so implementation drift forces test review.
    """
    syntax=ast.parse(inspect.getsource(evaluation.validate_terminal));changed=[]
    class BudgetOnly(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in (85200,94200):
                changed.append(node.value)
                return ast.copy_location(ast.Constant(value=budget),node)
            return node
    syntax=BudgetOnly().visit(syntax);ast.fix_missing_locations(syntax)
    assert changed==[85200,94200]
    namespace=dict(evaluation.__dict__)
    exec(compile(syntax,'<NON-AUTHORITATIVE-smoke-budget-gate>','exec'),namespace)
    return namespace['validate_terminal']


def reconcile(progress,dataset):
    source_counts=Counter();identity_counts=Counter();seen=[]
    for indices in progress.index_history:
        assert len(indices)==64
        for idx in indices:
            row=dataset.resolve_index(idx);seen.append(idx)
            source_counts[row.source_key]+=1;identity_counts[row.global_label]+=1
    assert len(seen)==64*progress.completed_primary_updates
    for phase in progress.exposure.phases:
        assert progress.exposure.presentations[phase]==source_counts
        assert progress.exposure.identity_presentations[phase]==identity_counts
        assert progress.exposure.distinct[phase]==set(identity_counts)
    assert len(seen)>len(set(seen))  # repeats are presentations, never unique images
    progress.check_totals()
    return dict(per_source=dict(source_counts),per_identity=dict(identity_counts),unique_images=len(set(seen)),presentations=len(seen))


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('fold',list(protocol.FOLDS))
def test_eight_production_dispatch_workflows(tree,synthetic,tmp_path,monkeypatch,architecture,fold):
    # Test-only authority doubles for short CPU scientific fixtures; no production bypass.
    monkeypatch.setattr('reid.utils.lodo_operational.validate_training_admission', lambda *a, **k: None)
    monkeypatch.setattr('reid.utils.lodo_operational.evaluation_admission', lambda *a, **k: {'synthetic': True})
    monkeypatch.setattr(evaluate, 'REPO_ROOT', tmp_path)
    enlarge_synthetic_sources(tree)
    cfg=config(tree,architecture,fold);cfg['system']['log_interval']=1000
    # Validate the unmodified official recipe, including all input dimensions.
    official=protocol.scientific_configuration(cfg)
    loader,classes=data_build.build_train_loader(cfg)
    assert classes==24 and len(loader.dataset)==288
    assert tuple(loader.dataset.sources)==protocol.FOLDS[fold][0]
    assert loader.dataset.target not in loader.dataset.sources
    mapping=loader.dataset.mapping_digest
    duplicate=data_build.build_train_loader(cfg)[0]
    assert duplicate.dataset.mapping_digest==mapping and duplicate.dataset.serialize_sidecar()==loader.dataset.serialize_sidecar()
    sample=loader.dataset[0]
    assert type(sample) is ReIDTrainSample
    assert tuple(sample.image.shape[-2:])==((256,128) if architecture=='resnet50' else (384,128))
    first_epoch=len(list(copy.deepcopy(loader.batch_sampler))) if architecture=='resnet50' else len(loader)
    assert first_epoch>=3
    budget=first_epoch+2;cfg['train']['max_updates']=budget
    bad=copy.deepcopy(cfg);bad['data']['train']['dataset']['sources'][0]['name']=bad['data']['test']['dataset']['name']
    with pytest.raises(ValueError): data_build.build_train_loader(bad)
    # Keep original source factory/preprocessing/sampling; only observe construction.
    models=[];loaders=[];criteria=[];rates=[];auxiliary_rates=[];failure_at=[None]
    original_loader=train.build_train_loader;original_criterion=train.build_criterion
    original_optimizer=train.build_optimizer;original_auxiliary=train.build_center_optimizer
    def build_loader(actual):
        result=original_loader(actual);loaders.append(result[0]);return result
    def build_model(actual,num_classes,**kwargs):
        model=IntegrationModel(actual,num_classes);model.fail_at=failure_at[0];models.append(model);return model
    def build_criterion(actual,num_classes,feat_dim):
        criterion=original_criterion(actual,num_classes,feat_dim);criteria.append(criterion)
        criterion.observed_losses=[];criterion.raw_center_inputs=None
        def check_loss(module,args,result):
            outputs,labels=args;loss,terms=result
            assert tuple(outputs['emb'].shape)==(64,2048 if architecture=='resnet50' else 1536)
            if architecture=='resnet50':
                raw=outputs['feat_raw'];centers=module.center_loss.centers
                expected=.5*(raw-centers[labels]).square().sum(1).mean()
                assert terms['loss/center']==pytest.approx(float(expected.detach()),rel=1e-6)
                assert module.metric_feat_key=='feat_raw' and module.center is module.center_loss
                assert centers.shape==(24,2048) and module.w_center==.0005
                assert not torch.equal(raw,outputs['feat_bn'])
                module.raw_center_inputs=(raw.detach().clone(),labels.detach().clone(),centers.detach().clone())
                assert float(loss.detach())==pytest.approx(terms['loss/id']+terms['loss/triplet']+.0005*terms['loss/center'],rel=1e-6)
            else:
                assert len(outputs['logits'])==6 and all(head.shape==(64,24) for head in outputs['logits'])
                assert module.triplet is None and module.center_loss is None
                expected=sum(F.cross_entropy(head,labels) for head in outputs['logits'])
                torch.testing.assert_close(loss,expected,rtol=0,atol=0)
            module.observed_losses.append(float(loss.detach()))
        criterion.register_forward_hook(check_loss)
        return criterion
    def build_optimizer(actual,model):
        optimizer=original_optimizer(actual,model)
        optimizer.register_step_pre_hook(lambda opt,a,k:rates.append(tuple(g['lr'] for g in opt.param_groups)))
        return optimizer
    def build_auxiliary(actual,criterion):
        optimizer=original_auxiliary(actual,criterion)
        if optimizer:
            def before_step(opt,args,kwargs):
                auxiliary_rates.append(tuple(g['lr'] for g in opt.param_groups))
                raw,labels,centers=criterion.raw_center_inputs
                expected=torch.zeros_like(centers).index_add_(0,labels,(centers[labels]-raw)/64)
                torch.testing.assert_close(criterion.center_loss.centers.grad,expected,rtol=1e-5,atol=1e-6)
                assert [id(p) for g in opt.param_groups for p in g['params']]==[id(criterion.center_loss.centers)]
            optimizer.register_step_pre_hook(before_step)
        return optimizer
    monkeypatch.setattr(train,'build_train_loader',build_loader);monkeypatch.setattr(train,'build_model',build_model)
    monkeypatch.setattr(train,'build_criterion',build_criterion);monkeypatch.setattr(train,'build_optimizer',build_optimizer)
    monkeypatch.setattr(train,'build_center_optimizer',build_auxiliary)
    yaml_path=tmp_path/'NON-AUTHORITATIVE-training.yaml'
    def dispatch(actual):
        save_yaml(actual,str(yaml_path))
        monkeypatch.setattr(train,'parse_args',lambda:SimpleNamespace(config=str(yaml_path),opts=[]))
        return train.main()  # actual YAML loading/schema/semantic validation/dispatch
    with monkeypatch.context() as guards:
        target_guards(guards)
        guards.setattr(train,'create_experiment_dirs',lambda *a:pytest.fail('Stage A output setup'))
        reference_cfg=copy.deepcopy(cfg);reference_cfg['experiment']['output_dir']=reference_cfg['experiment']['output_dir'].replace('run-1','NON-AUTHORITATIVE-reference')
        reference=dispatch(reference_cfg);reference_model=models[-1];reference_criterion=criteria[-1]
        reference_rates=list(rates);rates.clear();auxiliary_rates.clear()
        assert len(reference_rates)==budget and len(reference_model.inputs)==budget
        assert reference.epoch_results[-1].partial_epoch and reference.epoch_results[-1].successful_updates==2
        assert reference.epoch_results[-1].loss_batches==2
        assert reference.epoch_results[-1].mean_loss==pytest.approx(sum(reference_criterion.observed_losses[-2:])/2)
        assert reference.epoch_results[-1].stop_reason=='budget_completed'
        exposure=reconcile(reference,loaders[-1].dataset)
        interrupted_cfg=copy.deepcopy(cfg);interrupted_cfg['experiment']['output_dir']=interrupted_cfg['experiment']['output_dir'].replace('run-1','NON-AUTHORITATIVE-recovered')
        failure_at[0]=first_epoch+2
        with pytest.raises(UpdateFailure) as error: dispatch(interrupted_cfg)
        assert error.value.progress.completed_primary_updates==first_epoch+1
        assert error.value.progress.attempted_batches==first_epoch+2
        run=tmp_path/interrupted_cfg['experiment']['output_dir'];last=run/'checkpoints/ckpt_last.pth';terminal=run/'checkpoints/ckpt_terminal.pth'
        boundary,_=checkpoint.read_checkpoint(last)
        assert boundary['progress']['completed_primary_updates']==first_epoch and not terminal.exists()
        assert boundary['position']['consumed_cursor']==0
        center_at_boundary=copy.deepcopy(boundary['center'])
        interrupted_cfg['train']['save']['resume']=str(last);failure_at[0]=None;rates.clear();auxiliary_rates.clear()
        recovered=dispatch(interrupted_cfg);recovered_model=models[-1];recovered_criterion=criteria[-1]
        assert_equal(reference_model.state_dict(),recovered_model.state_dict())
        assert_equal(reference_model.inputs[first_epoch:],recovered_model.inputs)
        assert_equal(reference_criterion.observed_losses[first_epoch:],recovered_criterion.observed_losses)
        assert_equal(list(reference.index_history),list(recovered.index_history))
        assert rates==reference_rates[first_epoch:] and len(rates)==2
        reconcile(recovered,loaders[-1].dataset)
        assert recovered.completed_primary_updates==budget and recovered.auxiliary_steps==(budget if architecture=='resnet50' else 0)
        if architecture=='resnet50':
            assert_equal(reference_criterion.center_loss.state_dict(),recovered_criterion.center_loss.state_dict())
            assert center_at_boundary is not None and auxiliary_rates==[(.5,),(.5,)]
        else: assert center_at_boundary is None and not auxiliary_rates
        payload,manifest=checkpoint.read_checkpoint(terminal)
        ref_terminal=tmp_path/reference_cfg['experiment']['output_dir']/'checkpoints/ckpt_terminal.pth'
        reference_payload,_=checkpoint.read_checkpoint(ref_terminal)
        for key in ('optimizer','scheduler','center','center_optimizer','rng'):
            assert_equal(payload[key],reference_payload[key])
        assert manifest['completion_reason']=='budget_completed' and manifest['checkpoint_sha256']==checkpoint.file_digest(terminal)
        assert payload['position']['consumed_cursor']==2 and not (run/'checkpoints/ckpt_best.pth').exists()
        events=json.loads((run/'recovery_events.json').read_text())
        assert events[0]['discarded_successful_updates']==1 and events[0]['discarded_attempted_batches']==2
        assert events[1]['event']=='rollback_resume'
        before=terminal.read_bytes();again=dispatch(interrupted_cfg)
        assert again.completed_primary_updates==budget and not models[-1].inputs and terminal.read_bytes()==before
    # Every training path above had throwing target guards. Evaluation now opens only target.
    request=dict(mode='lodo_held_out',version=1,architecture=architecture,fold=fold,target=protocol.FOLDS[fold][1],checkpoint=str(terminal),
                 expected={key:payload['identity'][key] for key in ('config_digest','map_digest','data_digest','num_classes')},
                 output_dir=f'exp/lodo_eval/v1/synthetic/{architecture}/{fold}/seed42/NON-AUTHORITATIVE-incomplete',synthetic=True,num_workers=0,batch_size=8)
    # Production gate still refuses a short synthetic budget. No forged full counts.
    with pytest.raises(ValueError,match='terminal'): evaluation.validate_terminal(request)
    monkeypatch.setattr(evaluation,'validate_terminal',smoke_terminal_gate(budget))
    monkeypatch.setattr('reid.models.build.build_model',lambda actual,num_classes,**kw:IntegrationModel(actual,num_classes))
    target_calls=[];original_target=evaluation.build_test_loader
    def target_only(actual):
        target_calls.append(actual['data']['test']['dataset']['name'])
        assert target_calls[-1]==request['target']
        result=original_target(actual)
        assert len(set(result.dataset.pids))!=24
        assert list(result.dataset.marks).count(0)==1 and list(result.dataset.marks).count(1)>=1
        return result
    monkeypatch.setattr(evaluation,'build_test_loader',target_only)
    original_evaluate=evaluation.evaluate_lodo
    monkeypatch.setattr(evaluation,'evaluate_lodo',lambda actual:original_evaluate(actual,project_root=tmp_path))
    eval_yaml=tmp_path/'NON-AUTHORITATIVE-evaluation.yaml'
    def eval_dispatch(actual):
        save_yaml(actual,str(eval_yaml));monkeypatch.setattr(evaluate,'parse_args',lambda:SimpleNamespace(config=str(eval_yaml),opts=[],weight=''))
        return evaluate.main()
    result,incomplete=eval_dispatch(request)
    assert incomplete['status']=='incomplete' and incomplete['mean'] is None
    request['stage_a_records']=stage_a_rows(result,tmp_path)
    request['output_dir']=request['output_dir'].replace('-incomplete','-complete')
    result,complete=eval_dispatch(request)
    assert complete['status']=='complete' and len(complete['references'])==3
    assert result['synthetic'] and not result['authoritative'] and complete['synthetic']
    assert target_calls==[request['target'],request['target']]
    assert result['training']['map_digest']==mapping and result['training']['num_classes']==24
    assert evaluation.read_result(tmp_path/request['output_dir']/'lodo_result.json')==result
    assert terminal.read_bytes()==before
    summary=dict(synthetic=True,authoritative=False,architecture=architecture,fold=fold,budget=budget,first_epoch_updates=first_epoch,
                 target=request['target'],classes=24,mapping_digest=mapping,terminal_hash=manifest['checkpoint_sha256'],
                 training='pass',terminal='pass',recovery='bitwise CPU multi-step equivalent',target_isolation='pass',evaluation='pass',
                 results='complete and incomplete comparison pass',exposure=exposure,lr_used=reference_rates,
                 official_config=official,compute_double='Linear backbone/part features; BN and dropout retained',
                 smoke_budget_double='fixture-only protocol expectation and checked validator AST literal substitution')
    (tmp_path/'NON-AUTHORITATIVE-workflow.json').write_text(json.dumps(summary,indent=2))


@pytest.mark.parametrize('architecture,update',[
    ('resnet50',u) for u in (1,500,501,28400,28401,49700,49701,85200)
]+[('pcb',u) for u in (1,31400,31401,94200)])
def test_official_boundary_lr_used_with_real_parameter_gradients(architecture,update):
    from test_lodo_config import lodo_config
    from test_lodo_updates import TinyModel
    from reid.optim.build import build_optimizer, build_scheduler
    cfg=lodo_config(architecture);model=TinyModel(24)
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer,steps_per_epoch=193)
    scheduler.prepare(update-1)
    used=[];optimizer.register_step_pre_hook(lambda opt,a,k:used.append(tuple(g['lr'] for g in opt.param_groups)))
    previous=[p.detach().clone() for p in model.parameters()]
    for parameter in model.parameters(): parameter.grad=torch.ones_like(parameter)
    optimizer.step()
    assert any(not torch.equal(old,p) for old,p in zip(previous,model.parameters()))
    if architecture=='resnet50':
        expected=.0003*(.01+.99*(update-1)/500) if update<=500 else .0003 if update<=28400 else .00003 if update<=49700 else .000003
        assert used[0]==pytest.approx([expected]*len(used[0]),rel=1e-14,abs=0)
    else:
        expected=[(.01 if update<=31400 else .001) if group['param_name'].startswith('backbone.') else (.1 if update<=31400 else .01) for group in optimizer.param_groups]
        assert used[0]==pytest.approx(expected,rel=1e-14,abs=0)
        assert max(used[0])/min(used[0])==pytest.approx(10)
    assert scheduler.completed_updates==update-1
    scheduler.step();assert scheduler.completed_updates==update


def test_production_protocol_still_rejects_smoke_budgets():
    from test_lodo_config import lodo_config
    for architecture in ('resnet50','pcb'):
        cfg=lodo_config(architecture);cfg['train']['max_updates']=5
        with pytest.raises(ValueError): protocol.scientific_configuration(cfg)


from test_lodo_evaluation import setup_terminal


def test_result_publication_interruption_has_no_valid_completion_proof(setup_terminal,tmp_path,monkeypatch):
    request,_,_,_=setup_terminal();terminal=Path(request['checkpoint']);before=terminal.read_bytes()
    atomic=evaluation.atomic_write
    def interrupted(path,writer):
        if Path(path).name=='lodo_result.sha256.json': raise OSError('synthetic proof publication interruption')
        return atomic(path,writer)
    monkeypatch.setattr(evaluation,'atomic_write',interrupted)
    with pytest.raises(OSError,match='synthetic proof'): evaluation.evaluate_lodo(request,project_root=tmp_path)
    result=tmp_path/request['output_dir']/'lodo_result.json'
    assert result.exists()
    with pytest.raises(ValueError): evaluation.read_result(result)
    with pytest.raises(ValueError,match='collision'): evaluation.evaluate_lodo(request,project_root=tmp_path)
    assert terminal.read_bytes()==before
