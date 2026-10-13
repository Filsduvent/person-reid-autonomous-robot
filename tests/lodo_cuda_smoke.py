"""Explicit NON-AUTHORITATIVE Phase 9 GPU harness; never an official launcher.

Run as a separate process with one allocated full UUID visible. Real source
loaders/architectures/weights/samplers/losses/shared orchestration execute. Only
budget and output validation are fixture-local substitutions; production must
reject the shortened cfg first. Post-terminal retrieval uses tiny synthetic
held-out images through the original target loader, never a real benchmark.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack, contextmanager
import copy
import gc
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import resource
import time
from unittest.mock import patch

from PIL import Image
import torch

from test_lodo_config import lodo_config
from reid.utils import lodo_protocol as protocol, lodo_checkpoint as ck
from reid.utils import lodo_evaluation as evaluation
from reid.utils.lodo_device import authorization_uuid, audit_gpu, validate_placement
from reid.utils.checkpoint import reconstruct_model
from reid.utils.evaluation_config import build_evaluation_config
from reid.data import build as data_build
from reid.engine import evaluator
from scripts import train

ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / 'exp/lodo_smoke'
CLASSES = dict(zip(protocol.FOLDS, (2510, 2559, 2494, 2220)))


def smoke_output(cfg, *, project_root=None):
    if project_root is not None and Path(project_root).resolve() != ROOT:
        raise ValueError("Smoke namespace cannot be rebound to another project")
    lexical = Path(cfg['experiment']['output_dir'])
    if not lexical.is_absolute(): lexical = ROOT / lexical
    try:
        parts = lexical.relative_to(SMOKE_ROOT).parts
        assert len(parts) == 4 and parts[1:3] == (cfg['protocol']['architecture'], cfg['protocol']['fold'])
        assert all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', p) for p in parts)
        assert lexical.resolve() == lexical and SMOKE_ROOT.resolve() == SMOKE_ROOT
        assert not lexical.exists() and not lexical.is_symlink()
        assert not cfg['train']['save'].get('resume')
    except (ValueError, AssertionError) as error:
        raise ValueError('Unsafe/nonfresh NON-AUTHORITATIVE smoke namespace') from error
    return lexical


@contextmanager
def fixture_budget(budget):
    if type(budget) is not int or not 2 <= budget <= 6:
        raise ValueError('Smoke-only budget must be 2..6')
    original = protocol._expect
    def expect(cfg, path, value):
        if path == 'train.max_updates':
            assert value in (85200, 94200) and cfg['train']['max_updates'] == budget
        else: original(cfg, path, value)
    with patch.object(protocol, '_expect', expect), patch.object(protocol, 'validate_output_path', smoke_output):
        yield


def terminal_gate(budget):
    syntax = ast.parse(inspect.getsource(evaluation.validate_terminal)); changed = []
    class BudgetOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in (85200, 94200):
                changed.append(node.value)
                return ast.copy_location(ast.Constant(value=budget), node)
            return node
    syntax = BudgetOnly().visit(syntax); ast.fix_missing_locations(syntax)
    assert changed == [85200, 94200]
    namespace = dict(evaluation.__dict__)
    exec(compile(syntax, '<NON-AUTHORITATIVE-budget-only-terminal>', 'exec'), namespace)
    return namespace['validate_terminal']


def synthetic_target(root, target):
    """One query, ten gallery records: original PID/camera/filtering contract."""
    def image(path, index):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (128, 384), (index*17 % 255, 40, 80)).save(path)
    if target != 'msmt17':
        base = root / ('cuhk03/detected' if target == 'cuhk03' else target)
        names = ['00000100_0001_00000000.jpg'] + [f'{100+i:08d}_0002_{i+1:08d}.jpg' for i in range(10)]
        for i, name in enumerate(names): image(base/'images'/name, i)
        (base/'partitions.pkl').write_bytes(pickle.dumps(dict(test_im_names=names, test_marks=[0]+[1]*10)))
    else:
        base = root/'msmt17/MSMT17_V2'
        for role in ('query', 'gallery'):
            refs = [f'{i:04d}/{i:04d}_{role}_{1 if role=="query" or i==1 else 2}_0000.jpg' for i in range(1 if role=='query' else 10)]
            for i, ref in enumerate(refs): image(base/'mask_test_v2'/ref, i)
            (base/f'list_{role}.txt').write_text(''.join(f'{ref} {i}\n' for i, ref in enumerate(refs)))


def require_production_rejection(cfg):
    try: protocol.scientific_configuration(cfg)
    except ValueError as error: assert 'max_updates' in str(error)
    else: raise AssertionError('Production accepted a short smoke budget')
    try: protocol.validate_output_path(cfg)
    except ValueError: pass
    else: raise AssertionError('Production accepted a smoke output namespace')


def run_case(architecture, fold, campaign, *, budget=3, interrupt=False):
    cfg = lodo_config(architecture, fold)
    cfg['system'].update(device='cuda', gpu_id=0, amp=False, log_interval=1000)
    # Explicit smoke execution setting: validated 0-worker path, not an official
    # worker policy change. Keep original pinning and all scientific fields.
    cfg['data'].update(root='/home/addirakoze/nobackup/Projects/Dataset', num_workers=0)
    cfg['experiment']['name'] = 'NON-AUTHORITATIVE-CUDA-SMOKE'
    kind = 'interrupted' if interrupt else 'fresh'
    cfg['experiment']['output_dir'] = str(SMOKE_ROOT/campaign/architecture/fold/kind)
    protocol.scientific_configuration(cfg)  # full recipe before test override
    cfg['train']['max_updates'] = budget
    require_production_rejection(cfg)
    uuid = authorization_uuid(cfg)
    admission = audit_gpu(uuid)  # mandatory before ANY CUDA setup/allocation
    run_dir = smoke_output(cfg)
    # Reserve only a unique, smoke-owned case container; never an official path.
    case_dir = run_dir.parent
    case_dir.mkdir(parents=True, exist_ok=False)
    outcome = dict(non_authoritative=True, architecture=architecture, fold=fold,
                   expected_classes=CLASSES[fold], authorized_uuid=uuid,
                   planned_smoke_updates=budget, admission=admission,
                   original_input=cfg['data']['train']['images']['size'], batch_size=64,
                   num_workers=0, precision='fp32', status='RUNNING')
    weights = Path(torch.hub.get_dir())/'checkpoints'/('resnet50-11ad3fa6.pth' if architecture=='resnet50' else 'resnet50-19c8e357.pth')
    if not weights.is_file():
        raise ValueError('Smoke requires existing pretrained cache; downloading is prohibited')
    outcome.update(pretrained_path=str(weights), pretrained_sha256=ck.file_digest(weights))
    (case_dir/'intent.json').write_bytes(ck.canonical_json(outcome))
    objects = {}; times=[]; losses=[]; rates=[]; aux_rates=[]; forward_times=[]
    original_loader=train.build_train_loader; original_model=train.build_model
    original_criterion=train.build_criterion; original_optimizer=train.build_optimizer
    original_aux=train.build_center_optimizer; original_epoch=train.train_one_epoch
    pending = {}; guard_calls = [0]
    def build_loader(actual):
        loader, classes = original_loader(actual)
        assert classes == CLASSES[fold]
        objects['loader'] = loader
        return loader, classes
    def build_model(actual, num_classes, **kwargs):
        model = original_model(actual, num_classes, **kwargs)
        objects['model'] = model
        def before(module, args):
            torch.cuda.synchronize(); pending['start'] = time.perf_counter()
            x=args[0]; assert x.device == torch.device('cuda:0') and x.dtype == torch.float32
            assert tuple(x.shape) == (64, 3, *cfg['data']['train']['images']['size'])
        model.register_forward_pre_hook(before)
        return model
    def build_criterion(actual, num_classes, feat_dim):
        criterion=original_criterion(actual,num_classes,feat_dim); objects['criterion']=criterion
        def observed(module, args, result):
            outputs, labels = args; loss, logs = result
            assert labels.device == torch.device('cuda:0') and torch.isfinite(loss).item()
            if architecture == 'resnet50':
                assert tuple(module.center_loss.centers.shape)==(CLASSES[fold],2048)
                assert module.center_loss.centers.device==labels.device
                expected=.5*(outputs['feat_raw']-module.center_loss.centers[labels]).square().sum(1).mean()
                torch.testing.assert_close(expected.detach(), torch.tensor(logs['loss/center'], device=labels.device), rtol=1e-5, atol=1e-4)
            else: assert len(outputs['logits'])==6 and all(x.shape==(64,CLASSES[fold]) for x in outputs['logits'])
            losses.append(float(loss.detach().cpu()))
        criterion.register_forward_hook(observed)
        return criterion
    def build_optimizer(actual, model):
        opt=original_optimizer(actual,model);objects['optimizer']=opt
        def before(opt,args,kwargs):
            rates.append(tuple(g['lr'] for g in opt.param_groups))
            assert all(torch.isfinite(p.grad).all().item() for g in opt.param_groups for p in g['params'] if p.grad is not None)
            torch.cuda.synchronize();pending['optimizer_start']=time.perf_counter()
        def after(opt,args,kwargs):
            torch.cuda.synchronize();now=time.perf_counter()
            forward_times.append(now-pending['optimizer_start'])
            if architecture=='pcb':times.append(now-pending['start'])
        opt.register_step_pre_hook(before);opt.register_step_post_hook(after)
        return opt
    def build_aux(actual, criterion):
        opt=original_aux(actual,criterion);objects['auxiliary']=opt
        if opt:
            opt.register_step_pre_hook(lambda opt,a,k:aux_rates.append(tuple(g['lr'] for g in opt.param_groups)))
            def after(opt,args,kwargs):
                torch.cuda.synchronize(); times.append(time.perf_counter()-pending['start'])
            opt.register_step_post_hook(after)
        return opt
    def epoch(*args,**kwargs):
        original_guard=kwargs['before_update']
        def guard():
            original_guard();guard_calls[0]+=1
            if interrupt and guard_calls[0]==2:raise RuntimeError('Intentional smoke-only interruption before update 2')
        kwargs['before_update']=guard
        torch.cuda.reset_peak_memory_stats(0)
        return original_epoch(*args,**kwargs)
    def prohibited(*a,**k):raise AssertionError('Target/legacy output access during smoke training')
    try:
        with fixture_budget(budget), ExitStack() as stack:
            for name, fn in [('build_train_loader',build_loader),('build_model',build_model),('build_criterion',build_criterion),
                             ('build_optimizer',build_optimizer),('build_center_optimizer',build_aux),('train_one_epoch',epoch),
                             ('build_test_loader',prohibited),('create_experiment_dirs',prohibited)]:
                stack.enter_context(patch.object(train,name,fn))
            stack.enter_context(patch.object(data_build,'build_test_loader',prohibited))
            stack.enter_context(patch.object(evaluator,'evaluate_reid',prohibited))
            for key in protocol.DATASET_ORDER:
                stack.enter_context(patch.object(data_build,f'_build_{key}_test_dataset',prohibited))
            started=time.perf_counter()
            try: progress=train.run_lodo_training(cfg)
            except Exception as error:
                if not interrupt or 'Intentional smoke-only interruption' not in str(error):raise
                events=json.loads((run_dir/'recovery_events.json').read_text())
                assert events[-1]['recover_from'] is None
                assert events[-1]['event']=='interruption_fresh_restart_required'
                assert events[-1]['discarded_successful_updates']==1
                assert not (run_dir/'checkpoints/ckpt_terminal.pth').exists()
                outcome.update(status='EXPECTED_INTERRUPTION', successful_updates=1, interruption=events[-1])
            else:
                assert not interrupt
                elapsed=time.perf_counter()-started
                validate_placement(cfg,objects['model'],objects['criterion'],objects['optimizer'],objects['auxiliary'])
                assert progress.completed_primary_updates==budget and progress.auxiliary_steps==(budget if architecture=='resnet50' else 0)
                source=Counter();ids=Counter()
                for indices in progress.index_history:
                    for i in indices:
                        row=objects['loader'].dataset.resolve_index(i);source[row.source_key]+=1;ids[row.global_label]+=1
                assert sum(source.values())==64*budget
                for phase in progress.exposure.phases:
                    assert progress.exposure.presentations[phase]==source and progress.exposure.identity_presentations[phase]==ids
                assert [tuple(row['lr_used']) for row in progress.lr_trace]==rates
                assert len(rates)==len(losses)==len(times)==budget
                assert aux_rates==[(.5,)]*budget if architecture=='resnet50' else not aux_rates
                outcome.update(status='TRAINING_PASSED', successful_updates=budget, auxiliary_updates=progress.auxiliary_steps,
                               wall_seconds=elapsed,losses=losses,lr_used=rates,step_seconds=times,optimizer_step_seconds=forward_times,
                               warm_step_seconds=times[0],steady_step_seconds=sum(times[1:])/len(times[1:]),
                               steady_images_per_second=64/(sum(times[1:])/len(times[1:])),
                               per_source_presentations=dict(source),mapping_digest=objects['loader'].dataset.mapping_digest)
            outcome.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                           peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            ckpt=run_dir/'checkpoints'/('ckpt_last.pth' if interrupt else 'ckpt_terminal.pth')
            payload,manifest=ck.read_checkpoint(ckpt)
            assert payload['identity']['recovery']['capability']=='fresh_cuda_no_resume_v1'
            outcome.update(checkpoint_sha256=manifest['checkpoint_sha256'],checkpoint_bytes=ckpt.stat().st_size)
            # Prove unsupported resume before discovery/model construction.
            bad=copy.deepcopy(cfg);bad['train']['save']['resume']=str(ckpt)
            try:train.run_lodo_training(bad)
            except ValueError as error:assert 'CUDA resume' in str(error)
            else:raise AssertionError('CUDA resume unexpectedly accepted')
            outcome['unsupported_resume_rejected']=True
            if not interrupt:
                request=dict(mode='lodo_held_out',version=1,architecture=architecture,fold=fold,target=protocol.FOLDS[fold][1],
                             checkpoint=str(ckpt),expected={k:payload['identity'][k] for k in ('config_digest','map_digest','data_digest','num_classes')},
                             output_dir=str(case_dir/'synthetic-evaluation'),synthetic=True)
                try:evaluation.validate_terminal(request)
                except ValueError:pass
                else:raise AssertionError('Production accepted smoke terminal')
                terminal_gate(budget)(request)
        # Leave ALL fixture substitutions before reconstruction/retrieval.
        if not interrupt:
            for module in (objects['model'],objects['criterion']):module.cpu()
            for name in ('optimizer','auxiliary'):
                if objects[name]:objects[name].state.clear()
            objects.clear();gc.collect();torch.cuda.empty_cache()
            reconstructed=reconstruct_model(payload).to('cuda:0')
            assert reconstructed.checkpoint_metadata['num_classes']==CLASSES[fold]
            root=case_dir/'synthetic-target';synthetic_target(root,protocol.FOLDS[fold][1])
            runtime=dict(data=dict(root=str(root),test=dict(dataset=copy.deepcopy(cfg['data']['test']['dataset']),batch=dict(size=11)),
                                   num_workers=0,pin_memory=False),system=dict(device='cuda',gpu_id=0))
            eval_cfg=build_evaluation_config(cfg,runtime,case_dir/'synthetic-evaluation')
            target_loader=data_build.build_test_loader(eval_cfg)
            assert len(target_loader.dataset)==11
            audit_gpu(uuid,allow_pid=os.getpid())
            metrics=evaluator.evaluate_reid(eval_cfg,reconstructed,target_loader,torch.device('cuda:0'))
            assert all(metrics[k] is not None and 0<=metrics[k]<=1 for k in ('mAP','mINP','Rank1','Rank5','Rank10'))
            assert ck.file_digest(ckpt)==manifest['checkpoint_sha256']
            outcome.update(status='PASS',checkpoint_integrity=True,reconstruction=True,
                           target_free_training=True,synthetic_target_plumbing=True,synthetic_metrics=metrics)
        require_production_rejection(cfg)  # substitutions must not escape
    except BaseException as error:
        outcome.update(status='FAIL',error_type=type(error).__name__,error=str(error))
        if torch.cuda.is_initialized():
            outcome.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        raise
    finally:
        (case_dir/f'{kind}-outcome.json').write_bytes(ck.canonical_json(outcome))
        print(json.dumps(outcome),flush=True)
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',choices=['resnet50','pcb'],required=True)
    parser.add_argument('--fold',choices=list(protocol.FOLDS),required=True)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--interrupt',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.campaign):parser.error('Invalid campaign identifier')
    run_case(args.architecture,args.fold,args.campaign,interrupt=args.interrupt)

if __name__=='__main__':main()
