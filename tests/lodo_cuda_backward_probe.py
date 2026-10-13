"""Non-authoritative single-update backward probes; shared engine and restoration.

Restore the immutable original-policy TITAN update-2 checkpoint before applying
a labeled runtime-only policy. No checkpoint is produced or qualified here.
"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

import torch
import lodo_cuda_abc_diagnostic as abc
from lodo_cuda_admission_diagnostic import AdmissionJournal, logged_admission, write_once_json
from lodo_cuda_titan_bc_continuation import require_allocation

ROOT = abc.ROOT
ORIGIN = ROOT / 'exp/lodo_smoke/phase9r1-abc-titan-20261009T214240-original'
UUID = abc.TITAN_UUID
PYTHON = '/home/addirakoze/environments/Reid/bin/python'


def tensor_evidence(value):
    if torch.is_tensor(value):
        cpu = value.detach().cpu().contiguous()
        return dict(**abc.fingerprint(cpu), sample=cpu.flatten()[:128].tolist(),
                    finite=bool(torch.isfinite(cpu).all()),
                    abs_max=float((cpu if cpu.dtype == torch.bool else cpu.abs()).max()) if cpu.numel() else 0.)
    if isinstance(value, (tuple, list)):
        return [tensor_evidence(v) for v in value]
    if isinstance(value, dict):
        return {str(k): tensor_evidence(v) for k, v in value.items()}
    return value


class BackwardTrace:
    """Observe actual autograd node boundaries without replacing gradients.

    Full hashes cover every tensor. Samples are diagnostic only and never used
    to claim full-tensor tolerance counts. Node IDs follow next_functions order.
    """
    def __init__(self):
        self.nodes = []
        self.execution = []
        self.handles = []
        self.module_nodes = {}
        self.forward = {}
        self.batch = None
        self.loss = None

    def install_model(self, model, loader):
        def before(module, args):
            self.batch = dict(input=abc.fingerprint(args[0]), indices=list(loader.latest_indices),
                              rng=abc.fingerprint(abc.ck.rng_state(loader)))
        self.handles.append(model.register_forward_pre_hook(before))
        for name, module in model.named_modules():
            def after(module, args, outputs, name=name):
                self.forward[name] = abc.fingerprint(outputs)
                for _, value in abc.flatten(outputs):
                    if value.grad_fn is not None:
                        self.module_nodes.setdefault(value.grad_fn, []).append(name)
            self.handles.append(module.register_forward_hook(after))

    def install_graph(self, loss, parameters):
        if self.nodes:
            raise ValueError('Probe must execute exactly one backward graph')
        names = {id(p): name for name, p in parameters}
        graph = [loss.grad_fn]
        ids = {loss.grad_fn: 0}
        for node in graph:
            for child, _ in node.next_functions:
                if child is not None and child not in ids:
                    ids[child] = len(graph)
                    graph.append(child)
        for index, node in enumerate(graph):
            saved = {}
            for attr in dir(node):
                if attr.startswith('_saved_'):
                    saved[attr] = tensor_evidence(getattr(node, attr))
            if hasattr(node, 'saved_tensors'):
                saved['saved_tensors'] = tensor_evidence(node.saved_tensors)
            variable = getattr(node, 'variable', None)
            row = dict(id=index, operation=node.name(), modules=self.module_nodes.get(node, []),
                       parameter=names.get(id(variable)), saved=saved,
                       next=[[ids.get(child), slot] for child, slot in node.next_functions])
            self.nodes.append(row)
            def pre(grads, row=row):
                row['grad_outputs'] = tensor_evidence(grads)
                self.execution.append(row['id'])
            def post(grad_inputs, grad_outputs, row=row):
                row['grad_inputs'] = tensor_evidence(grad_inputs)
            self.handles.extend([node.register_prehook(pre), node.register_hook(post)])

    def close(self):
        for handle in self.handles:
            handle.remove()


def hashes(value):
    """Discard bounded samples/magnitudes while retaining structure/full hashes."""
    if isinstance(value, dict):
        return {k: hashes(v) for k, v in value.items() if k not in ('sample', 'abs_max')}
    if isinstance(value, list):
        return [hashes(v) for v in value]
    return value


def compare_traces(left, right):
    topology = lambda trace: [{k: row[k] for k in ('id', 'operation', 'modules', 'parameter', 'next')}
                              for row in trace['nodes']]
    if topology(left) != topology(right):
        raise ValueError('Backward graph topology differs; cannot align nodes')
    boundaries = []
    differing = []
    for index in left['execution']:
        a, b = left['nodes'][index], right['nodes'][index]
        if 'grad_inputs' not in a or 'grad_inputs' not in b:
            continue
        incoming_equal = hashes(a.get('grad_outputs')) == hashes(b.get('grad_outputs'))
        saved_equal = hashes(a['saved']) == hashes(b['saved'])
        outgoing_equal = hashes(a['grad_inputs']) == hashes(b['grad_inputs'])
        if not outgoing_equal:
            evidence = dict(id=index, operation=a['operation'], modules=a['modules'],
                            parameter=a['parameter'], incoming_exact=incoming_equal,
                            saved_operands_exact=saved_equal, outgoing_exact=False)
            differing.append(evidence)
            if incoming_equal and saved_equal:
                boundaries.append(evidence)
    return dict(batch_exact=left['batch'] == right['batch'],
                forward_exact=left['forward'] == right['forward'],
                execution_order_exact=left['execution'] == right['execution'],
                nodes=len(left['nodes']), differing_nodes=differing,
                equal_input_divergent_output_boundaries=boundaries,
                first_observed_differing_node=differing[0] if differing else None,
                limit='Node boundaries, not individual internal CUDA kernels; samples are not tolerance counts.')


def apply_policy(policy):
    if policy == 'deterministic':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('Strict probe requires workspace configured before CUDA initialization')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    elif policy != 'original':
        raise ValueError('Unknown diagnostic policy')


def probe_directory(base, name):
    if base.resolve() != base or base.parent != ROOT / 'exp/lodo_smoke':
        raise ValueError('Noncanonical/outside diagnostic namespace')
    if not re.fullmatch(r'phase9r1-backward-titan-[A-Za-z0-9_-]+', base.name):
        raise ValueError('Invalid diagnostic namespace')
    if name not in ('original-1', 'original-2', 'deterministic-1', 'deterministic-2'):
        raise ValueError('Invalid probe name')
    directory = base / name
    directory.mkdir(exist_ok=False)
    return directory


@contextmanager
def probe_contract():
    with abc.diagnostic_budget('original'), ExitStack() as stack:
        stack.enter_context(patch.object(abc.protocol, 'validate_output_path', abc.output_path))
        abc.guards(stack)
        yield stack


def execute(base, name):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID or os.environ.get('LODO_AUTHORIZED_GPU_UUID') != UUID:
        raise ValueError('Exact authorized TITAN UUID required')
    if torch.cuda.is_initialized():
        raise ValueError('Fresh process required')
    policy = name.rsplit('-', 1)[0]
    # Validate allocation and exact common origin before constructing sources.
    require_allocation(ORIGIN / 'allocation-confirmation.json')
    _, checkpoint, payload, proof = abc.read_common(ORIGIN, 'original', UUID)
    if proof['checkpoint_sha256'] != '562a0ed8fb673dbbfd3101fa4514fe41b2b80956a3ef664c1a4362a8ba6222ce':
        raise ValueError('Preserved common checkpoint changed')
    directory = probe_directory(base, name)
    trace = BackwardTrace()
    outcome = dict(non_authoritative=True, policy=policy, checkpoint=str(checkpoint),
                   checkpoint_sha256=proof['checkpoint_sha256'], status='RUNNING',
                   qualification=False, started_utc=datetime.now(timezone.utc).isoformat())
    started = time.perf_counter()
    objects = {}
    try:
        with logged_admission(directory / 'admission'), probe_contract() as stack:
            abc.dev.audit_gpu(UUID)
            cfg = copy.deepcopy(payload['cfg'])
            cfg['train']['save']['resume'] = str(checkpoint)
            abc.train.set_seed(cfg['repro']['seed'], cfg['repro']['deterministic'], cfg['repro']['benchmark'])
            loader, classes = abc.train.build_train_loader(cfg)
            if classes != 2510 or loader.num_workers != 0:
                raise ValueError('Unexpected source/class/worker contract')
            loader._observer.original = abc.ShortTestEpoch(loader.batch_sampler)
            model = abc.train.build_model(cfg, classes, initialize_pretrained=False)
            criterion = abc.train.build_criterion(cfg, classes, model.feat_dim)
            optimizer = abc.train.build_optimizer(cfg, model)
            auxiliary = abc.train.build_center_optimizer(cfg, criterion)
            scheduler = abc.train.build_scheduler(cfg, optimizer, steps_per_epoch=len(loader))
            objects = dict(cfg=cfg, loader=loader, model=model, criterion=criterion,
                           optimizer=optimizer, auxiliary=auxiliary, scheduler=scheduler)
            abc.ck.restore_lodo_checkpoint(checkpoint, **objects, cpu_inspection=True, validate_only=True)
            if torch.cuda.is_initialized():
                raise ValueError('CPU validation initialized CUDA')
            device, guard, admission = abc.dev.select_lodo_device(cfg)
            model.to(device); criterion.to(device)
            progress, position, _ = abc.ck.restore_lodo_checkpoint(checkpoint, **objects)
            rec = type('BoundaryView', (), {})()
            rec.objects = objects
            rec.ownership = lambda: ownership(model, optimizer, criterion, auxiliary)
            abc.boundary_capture(rec, checkpoint, objects, capture=False)
            outcome.update(restoration_exact=True, admission=admission,
                           restored_environment=abc.environment(UUID), mapping_digest=loader.dataset.mapping_digest)
            # Deliberate diagnostic override AFTER exact original-policy restoration.
            # Frozen cfg/identity/scheduler are never edited or checkpointed.
            apply_policy(policy)
            outcome['probe_environment'] = abc.environment(UUID)
            trace.install_model(model, loader)
            def loss_hook(module, args, result):
                loss, logs = result
                trace.loss = dict(total=float(loss.detach().cpu()), terms=logs,
                                  labels=abc.fingerprint(args[1]))
                trace.install_graph(loss, list(model.named_parameters()) +
                                    [('center', criterion.center_loss.centers)])
            trace.handles.append(criterion.register_forward_hook(loss_hook))
            original_backward = torch.Tensor.backward
            def backward(value, *args, **kwargs):
                result = original_backward(value, *args, **kwargs)
                torch.save(abc.cpu_tree(dict(model={n:p.grad for n,p in model.named_parameters()},
                                            center=criterion.center_loss.centers.grad)), directory / 'gradients.pth')
                return result
            stack.enter_context(patch.object(torch.Tensor, 'backward', backward))
            original_aux_step = auxiliary.step
            def step(*args, **kwargs):
                result = original_aux_step(*args, **kwargs)
                # Stop the shared engine naturally before it requests a second batch.
                # Read-only fixture adjustment only; no resumable snapshot is saved.
                progress.max_updates = 3
                return result
            stack.enter_context(patch.object(auxiliary, 'step', step))
            guard(); torch.cuda.reset_peak_memory_stats(0)
            result = abc.train.train_one_epoch(model, loader, criterion, optimizer, auxiliary,
                device, False, 1000, scheduler=scheduler, epoch=position['next_epoch'],
                update_progress=progress, before_update=guard)
            guard()
            if progress.completed_primary_updates != 3 or progress.auxiliary_steps != 3:
                raise ValueError('One-update probe exceeded its boundary')
            progress.check_totals()
            torch.save(abc.cpu_tree(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                center=criterion.center_loss.state_dict(), center_optimizer=auxiliary.state_dict())), directory / 'state.pth')
            outcome.update(status='PASS', completed_updates=3, center_steps=3,
                           lr_trace=list(progress.lr_trace), scheduler=scheduler.state_dict(),
                           source_exposure={k:progress.exposure.summary()[k] for k in progress.exposure.phases},
                           rng_after=abc.fingerprint(abc.ck.rng_state(loader)),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(0), stop_reason=result.stop_reason)
    except BaseException as error:
        outcome.update(status='FAIL', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        trace.close()
        outcome['elapsed_seconds'] = time.perf_counter() - started
        outcome['cuda_initialized'] = torch.cuda.is_initialized()
        write_once_json(directory / 'trace.json', dict(nodes=trace.nodes, execution=trace.execution,
            forward=trace.forward, batch=trace.batch, loss=trace.loss))
        write_once_json(directory / 'outcome.json', outcome)
        print(json.dumps(outcome), flush=True)


def ownership(model, optimizer, criterion, auxiliary):
    lookup = dict(model.named_parameters())
    if not all(len(g['params']) == 1 and g['params'][0] is lookup[g['param_name']] for g in optimizer.param_groups):
        raise ValueError('Main optimizer parameter association changed')
    if auxiliary.param_groups[0]['params'][0] is not criterion.center_loss.centers:
        raise ValueError('Center optimizer parameter association changed')


def campaign():
    require_allocation(ORIGIN / 'allocation-confirmation.json')
    if torch.cuda.is_initialized():
        raise ValueError('Parent must remain CPU-only')
    base = ROOT / 'exp/lodo_smoke' / ('phase9r1-backward-titan-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    base.mkdir(exist_ok=False)
    previous = json.loads((ORIGIN / 'continuations/bc-20261009T224416/cpu-closeout/lodo-titan-bc-origin-integrity.json').read_text())
    for relative, expected in previous['files_sha256'].items():
        if abc.ck.file_digest(ORIGIN / relative) != expected:
            raise ValueError('Historical A campaign changed: ' + relative)
    historical = {str(p.relative_to(ORIGIN)): abc.ck.file_digest(p) for p in ORIGIN.rglob('*') if p.is_file()}
    write_once_json(base / 'historical-sha256.json', historical)
    write_once_json(base / 'authorization.json', dict(uuid=UUID, allocated=True, source='user',
        confirmation='do it', scope='Four sequential one-update backward/deterministic probes; preserve prior results'))
    os.environ.update(CUDA_VISIBLE_DEVICES=UUID, LODO_AUTHORIZED_GPU_UUID=UUID)
    journal = AdmissionJournal(base / 'admission-parent')
    summary = dict(non_authoritative=True, status='RUNNING', runs=[], qualified=False)
    print('CAMPAIGN ' + str(base), flush=True)
    try:
        for name in ('original-1', 'original-2', 'deterministic-1', 'deterministic-2'):
            if summary['runs']:
                time.sleep(3)  # Before the next admission; never retry a rejection.
            audit = journal.check(UUID)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='.:tests')
            if name.startswith('deterministic'):
                env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            else:
                env.pop('CUBLAS_WORKSPACE_CONFIG', None)
            command = [PYTHON, '-B', str(Path(__file__).resolve()), '--base', str(base), '--probe', name]
            started = time.perf_counter()
            with (base / (name + '.log')).open('x') as log:
                child = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            item = dict(name=name, command=command, admission=audit, returncode=child.returncode,
                        wall_seconds=time.perf_counter()-started)
            outcome = base / name / 'outcome.json'
            if outcome.exists(): item['outcome'] = json.loads(outcome.read_text())
            summary['runs'].append(item)
            print('FINISH ' + name + ' ' + str(child.returncode), flush=True)
            if child.returncode:
                raise RuntimeError('Probe failed; preserve rejection/unsupported-operation evidence and stop')
        for policy in ('original', 'deterministic'):
            left = base / (policy + '-1'); right = base / (policy + '-2')
            pair = compare_traces(json.loads((left/'trace.json').read_text()), json.loads((right/'trace.json').read_text()))
            pair['gradients'] = abc.compare(torch.load(left/'gradients.pth', map_location='cpu', weights_only=False),
                                           torch.load(right/'gradients.pth', map_location='cpu', weights_only=False))
            pair['state'] = abc.compare(torch.load(left/'state.pth', map_location='cpu', weights_only=False),
                                       torch.load(right/'state.pth', map_location='cpu', weights_only=False))
            summary[policy] = pair
        summary['status'] = 'COMPLETED'
    except BaseException as error:
        summary.update(status='STOPPED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        changed = [p for p,h in historical.items() if not (ORIGIN/p).is_file() or abc.ck.file_digest(ORIGIN/p) != h]
        summary.update(historical_files=len(historical), historical_changed=changed,
                       parent_cuda_initialized=torch.cuda.is_initialized())
        if changed: summary['status'] = 'STOPPED'
        try: summary['final_audit'] = journal.snapshot()
        except BaseException as error:
            summary.update(status='STOPPED', final_audit_error=str(error))
        write_once_json(base/'summary.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('original','deterministic','runs')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--probe')
    args = parser.parse_args()
    if args.probe:
        execute(args.base, args.probe)
    else:
        campaign()
