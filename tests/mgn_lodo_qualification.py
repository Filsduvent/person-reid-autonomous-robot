"""Phase21 NON-AUTHORITATIVE bounded fixture harness, never an official trainer.

The parent supervises immutable cases. Fixture-only substitutions remain here;
production validation must first reject both the shortened budget and namespace.
"""
import argparse
from contextlib import ExitStack
import copy
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'exp/mgn_lodo_qualification/mgn_lodo_v2'
CLASSES = (2510, 2559, 2494, 2220)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2))
    temporary.replace(path)


def enter_step(path):
    """Count optimizer entries before calling SGD, aggregated across children."""
    value = json.loads(Path(path).read_text())
    if value['optimizer_entries'] >= 3:
        raise RuntimeError('Qualification aggregate optimizer-call limit reached')
    value['optimizer_entries'] += 1
    write(path, value)


def case_path(campaign, fold, kind):
    if not re.fullmatch(r'phase21-[A-Za-z0-9._-]+', campaign):
        raise ValueError('Invalid qualification campaign')
    if fold not in range(1, 5) or kind not in ('forward', 'rollback'):
        raise ValueError('Invalid fixed case')
    path = BASE / campaign / f'LODO-{fold}' / kind
    if path.resolve() != path:
        raise ValueError('Qualification path alias')
    return path


class OneBatchFixture:
    """One original PK batch; exhausted fixture epoch, not a benchmark epoch."""
    def __init__(self, original):
        self.original = original
    def __getattr__(self, name):
        return getattr(self.original, name)
    def __len__(self):
        return 1
    def __iter__(self):
        yield from islice(self.original, 1)


def equal(a, b):
    import numpy as np
    import torch
    if torch.is_tensor(b):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(b, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in b:
            equal(a[key], b[key])
    elif isinstance(b, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def worker(args):
    import torch
    import torch.nn.functional as F
    from reid.utils import mgn_lodo_protocol as science, mgn_lodo_operational as operational
    from reid.utils import lodo_checkpoint as ck, lodo_device as device
    from reid.utils import lodo_recovery as recovery
    from reid.utils.config import load_config
    from reid.data.lodo_preflight import preflight_lodo
    from reid.data import build as data_build
    from scripts import train

    path = case_path(args.campaign, args.fold, args.kind)
    cfg = load_config(str(ROOT / f'configs/lodo/mgn_lodo_v2/mgn_lodo-{args.fold}_seed42.yaml'))
    assert not torch.cuda.is_initialized()
    science.validate_lodo_config(cfg)
    operational.validate_committed_sources(ROOT, operational.source_snapshot())
    preflight = preflight_lodo(cfg)
    assert ck.file_digest(cfg['model']['weights_path']) == cfg['model']['initialization']['sha256']
    uuid = device.authorization_uuid(cfg)
    admission = device.audit_gpu(uuid)
    official_science = science.configuration_identity(cfg)
    cfg['train']['max_updates'] = 3
    try:
        science.contract(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('Production accepted fixture budget')
    cfg['experiment']['output_dir'] = str(path / 'run')
    try:
        science.validate_output_path(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('Production accepted fixture namespace')
    if args.stage == 'replay':
        cfg['train']['save']['resume'] = (path / 'boundary-one.txt').read_text()

    objects = {}; batches = []; losses = []; rates = []; memory = []; inspections = []
    outcome = dict(status='RUNNING', non_authoritative=True, fold=f'LODO-{args.fold}',
                   kind=args.kind, stage=args.stage, scientific_sha256=official_science,
                   actual_classes=CLASSES[args.fold-1], admission=admission,
                   preflight_sha256=ck.digest(preflight), fixture_epoch_batches=1,
                   max_optimizer_calls_aggregate_all_children=3, target_images_read=0)
    write(path / f'{args.stage}-outcome.json', outcome)
    original_loader = train.build_train_loader; original_model = train.build_model
    original_loss = train.build_criterion; original_opt = train.build_optimizer
    original_epoch = train.train_one_epoch; original_restore = ck.restore_lodo_checkpoint

    def loader(actual):
        value, classes = original_loader(actual)
        assert classes == CLASSES[args.fold-1]
        objects['loader'] = value
        return OneBatchFixture(value), classes

    def model(actual, num_classes, **kwargs):
        value = original_model(actual, num_classes, **kwargs)
        def before(module, inputs):
            x = inputs[0]
            assert x.dtype == torch.float32 and x.device == torch.device('cuda:0')
            assert tuple(x.shape) == (64, 3, 384, 128)
            assert not torch.is_autocast_enabled('cuda')
            indices = objects['loader'].latest_indices
            labels = [objects['loader'].dataset.labels[i] for i in indices]
            from collections import Counter
            assert sorted(Counter(labels).values()) == [4]*16
            records = [objects['loader'].dataset.resolve_index(i) for i in indices]
            batches.append(dict(indices=list(indices), input_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
                source_images=[dict(source=r.source_key, identity=r.global_label, path=r.canonical_path,
                    sha256=ck.file_digest(r.canonical_path)) for r in records]))
            memory.append(dict(point='before_forward', allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), free_total=list(torch.cuda.mem_get_info())))
        value.register_forward_pre_hook(before)
        return value

    def criterion(actual, num_classes, feat_dim):
        value = original_loss(actual, num_classes, feat_dim)
        def observed(module, inputs, result):
            outputs, labels = inputs
            assert len(outputs['logits']) == 8 and len(outputs['metric_features']) == 3
            assert outputs['emb'].shape == (64, 2048)
            assert all(t.shape == (64, num_classes) for t in outputs['logits'])
            with torch.no_grad():
                ce = torch.stack([F.cross_entropy(t, labels) for t in outputs['logits']]).mean()
                positive = labels[:, None].eq(labels[None, :])
                terms = []
                for feature in outputs['metric_features']:
                    distances = torch.cdist(feature.double(), feature.double())
                    ap = distances.masked_fill(~positive, -float('inf')).max(1).values
                    an = distances.masked_fill(positive, float('inf')).min(1).values
                    terms.append((ap-an+1.2).clamp_min(0).mean())
                triplet = torch.stack(terms).mean()
                torch.testing.assert_close(result[0].double(), ce.double()+triplet, rtol=1e-4, atol=1e-4)
                losses.append(dict(total=float(result[0].detach()), mean_ce=float(ce), mean_triplet=float(triplet)))
        value.register_forward_hook(observed)
        return value

    def optimizer(actual, value):
        opt = original_opt(actual, value)
        first = next(value.parameters()); previous = [None]
        def before(opt, inputs, kwargs):
            enter_step(path / 'optimizer-ledger.json')
            rates.append([g['lr'] for g in opt.param_groups])
            assert all(rate == .01 for rate in rates[-1])
            previous[0] = first.detach().clone()
        def after(opt, inputs, kwargs):
            assert not torch.equal(first, previous[0]); previous[0] = None
            assert all(torch.isfinite(state['momentum_buffer']).all().item() for state in opt.state.values())
            memory.append(dict(point='after_sgd', allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), free_total=list(torch.cuda.mem_get_info())))
        opt.register_step_pre_hook(before); opt.register_step_post_hook(after)
        return opt

    def epoch(*values, **kwargs):
        if args.kind == 'rollback' and kwargs['update_progress'].completed_primary_updates == 2:
            raise RuntimeError('PHASE21_PLANNED_STOP_BEFORE_UPDATE_THREE')
        return original_epoch(*values, **kwargs)

    def restored(checkpoint, **kwargs):
        if kwargs.get('validate_only'):
            assert not torch.cuda.is_initialized()
            inspections.append('cpu_integrity_before_cuda')
        return original_restore(checkpoint, **kwargs)

    def output(actual, **kwargs):
        assert actual['experiment']['output_dir'] == str(path / 'run')
        if args.stage != 'replay':
            assert not (path / 'run').exists()
        return path / 'run'

    expect = science._expect
    def fixture_expect(actual, key, value):
        if key == 'train.max_updates':
            assert actual['train']['max_updates'] == 3
        else:
            expect(actual, key, value)

    def prohibited(*values, **kwargs):
        raise AssertionError('Target/legacy loader forbidden during qualification training')

    started = time.monotonic()
    try:
        with ExitStack() as stack:
            for obj, name, fn in [(science, '_expect', fixture_expect), (science, 'validate_output_path', output),
                (operational, 'validate_training_session', lambda c: None), (ck, 'restore_lodo_checkpoint', restored),
                (train, 'build_train_loader', loader), (train, 'build_model', model),
                (train, 'build_criterion', criterion), (train, 'build_optimizer', optimizer),
                (train, 'train_one_epoch', epoch), (train, 'build_test_loader', prohibited),
                (data_build, 'build_test_loader', prohibited)]:
                stack.enter_context(patch.object(obj, name, fn))
            for name in ('market1501','duke','cuhk03','msmt17'):
                stack.enter_context(patch.object(data_build, f'_build_{name}_test_dataset', prohibited))
            try:
                progress = train.run_lodo_training(cfg)
            except RuntimeError as error:
                if args.kind != 'rollback' or str(error) != 'PHASE21_PLANNED_STOP_BEFORE_UPDATE_THREE':
                    raise
                directory = path / 'run/checkpoints'
                rows = recovery._read_index(directory)
                payloads = [(directory / row['name'], ck.read_checkpoint(directory / row['name'])[0]) for row in rows]
                latest_path, latest = next((p,v) for p,v in reversed(payloads) if v['progress']['completed_primary_updates'] == 2)
                if args.stage == 'continuous':
                    boundary = next(p for p,v in payloads if v['progress']['completed_primary_updates'] == 1)
                    (path / 'boundary-one.txt').write_text(str(boundary))
                    import shutil
                    shutil.copyfile(latest_path, path / 'continuous-two.pth')
                else:
                    expected = torch.load(path / 'continuous-two.pth', map_location='cpu', weights_only=False)
                    for key in ('model','optimizer','scheduler','rng','identity','position'):
                        equal(latest[key], expected[key])
                    for key in ('completed_primary_updates','main_steps_returned','attempted_batches','index_history','lr_trace'):
                        equal(latest['progress'][key], expected['progress'][key])
                    for key in ('presentations','identity_presentations','identity_slots','batch_counts','distinct'):
                        equal(latest['progress']['exposure'][key], expected['progress']['exposure'][key])
                    assert inspections == ['cpu_integrity_before_cuda']
                    outcome['restored_continuation_bitwise_equal'] = True
                outcome.update(retained_generation_count=len(rows), status='PASS')
            else:
                assert args.kind == 'forward' and progress.completed_primary_updates == 3 and progress.auxiliary_steps == 0
                assert sum(progress.exposure.presentations['successful'].values()) == 192
                outcome.update(status='PASS', successful_updates=3, exposure=progress.exposure.summary())
            if args.kind == 'forward':
                terminal, manifest = ck.read_checkpoint(path / 'run/checkpoints/ckpt_terminal.pth')
                outcome['terminal_sha256'] = manifest['checkpoint_sha256']
                outcome['next_lr'] = terminal['scheduler']
                outcome['numerical_policy'] = terminal['identity']['numerical_policy']
            outcome.update(cuda_admission=json.loads((path/'run/cuda_admission.json').read_text()),
                input_batches=batches, losses=losses, used_rates=rates, memory=memory,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        outcome.update(status='FAIL', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        write(path / f'{args.stage}-outcome.json', outcome)
        print(json.dumps({k:outcome.get(k) for k in ('status','fold','kind','stage','error')}), flush=True)


def parent(args):
    from reid.utils import mgn_lodo_operational as operational
    from reid.utils.lodo_device import audit_gpu
    operational.validate_committed_sources(ROOT, operational.source_snapshot())
    audit_gpu(args.gpu_uuid)
    destination = BASE / args.campaign
    if destination.exists():
        raise ValueError('Campaign exists; preserve attempts, no reset/retry')
    case_path(args.campaign, 1, 'forward')
    destination.mkdir(parents=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, LODO_AUTHORIZED_GPU_UUID=args.gpu_uuid,
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH='.:scripts:tests', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    manifest = dict(status='RUNNING', approval_reference=args.approval_reference,
        release_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=operational.source_snapshot(), harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu_uuid=args.gpu_uuid, cases=[], bounds=dict(cases=8, optimizer_calls_per_case=3, worker_seconds=600, attempts=1))
    write(destination/'campaign.json',manifest)
    for fold in range(1,5):
        for kind in ('forward','rollback'):
            path=case_path(args.campaign,fold,kind);path.mkdir(parents=True)
            write(path/'optimizer-ledger.json',dict(optimizer_entries=0))
            status='PASS'
            for stage in (['fresh'] if kind=='forward' else ['continuous','replay']):
                command=[sys.executable,str(Path(__file__).resolve()),'--campaign',args.campaign,'--fold',str(fold),'--kind',kind,'--stage',stage,'--worker']
                with (path/f'{stage}-stdout.txt').open('w') as log:
                    try:
                        result=subprocess.run(command,env=environment,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                        code=result.returncode
                    except subprocess.TimeoutExpired:
                        code=124
                count=json.loads((path/'optimizer-ledger.json').read_text())['optimizer_entries']
                manifest['cases'].append(dict(fold=fold,kind=kind,stage=stage,exit_code=code,aggregate_optimizer_entries=count,path=str(path)))
                write(destination/'campaign.json',manifest)
                print(f'LODO-{fold} {kind}/{stage}: exit={code} aggregate_calls={count}',flush=True)
                if code:
                    status='FAIL';break
            if status=='PASS':
                assert json.loads((path/'optimizer-ledger.json').read_text())['optimizer_entries']==3
    manifest['status']='PASS' if all(row['exit_code']==0 for row in manifest['cases']) and len(manifest['cases'])==12 else 'FAIL'
    write(destination/'campaign.json',manifest)
    return 0 if manifest['status']=='PASS' else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--approval-reference')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--fold',type=int)
    parser.add_argument('--kind',choices=['forward','rollback'])
    parser.add_argument('--stage',choices=['fresh','continuous','replay'])
    args=parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.gpu_uuid or not args.approval_reference:
            parser.error('Actual authorized GPU UUID and user approval reference required')
        sys.exit(parent(args))
