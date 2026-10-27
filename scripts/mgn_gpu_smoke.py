"""Bounded, non-authoritative MGN Phase 12 FP32 admission.

This is never an official trainer. One fresh subprocess, three optimizer calls,
600 seconds maximum, immutable case identity and isolated per-attempt receipts.
Manual GPU authority is external: the operator must have current permission and
set MGN_AUTHORIZED_GPU_UUID. A free-memory snapshot never grants permission.
Only the LODO module's pure read-only inventory audit is reused; its admission,
protocol, whitelist and recovery routes are not invoked or modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / 'exp' / 'mgn_smoke'
CASE = 'phase12_synthetic_fp32'
STEPS = 3
MAX_SECONDS = 600
MAX_ATTEMPTS = 3
WEIGHTS = Path('/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')
SHA256 = '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097'
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def configuration():
    return dict(model=dict(name='mgn', variant='paper_v3_global2048_independent_v1', pretrained=True,
        weights_path=str(WEIGHTS), head=dict(classifier=True), initialization=dict(
            artifact_id='resnet50-19c8e357', sha256=SHA256, head_init_id='reference_a_384e820_bias_free_v1')),
        loss=dict(id=dict(enabled=True, weight=1., label_smoothing=0., head_aggregation='mean'),
            triplet=dict(enabled=True, weight=1., margin=1.2, mining='batch_hard', feature_source='model', feature_aggregation='mean'),
            center=dict(enabled=False)),
        optim=dict(name='sgd', lr=.01, momentum=.9, nesterov=False, weight_decay=.0005,
                   bias_lr_factor=1., weight_decay_bias=.0005),
        sched=dict(name='epoch_multistep', milestones=[60, 90], gamma=.1),
        train=dict(epochs=120, eval_interval=10, save=dict(metric='mAP', save_best=True, save_last=True,
                    resume='', require_selection_state_on_resume=True)),
        system=dict(device='cuda', gpu_id=0, amp=False),
        repro=dict(seed=42, deterministic=False, benchmark=True),
        synthetic=dict(shape=[64, 3, 384, 128], P=16, K=4, num_classes=1041,
                       generation='uniform RGB followed by fixed ImageNet normalization',
                       mean=[.485,.456,.406], std=[.229,.224,.225]))


def authorized_uuid(value, environ=None):
    environ = os.environ if environ is None else environ
    if not isinstance(value, str) or not UUID.fullmatch(value):
        raise ValueError('A full currently authorized GPU UUID is required.')
    if environ.get('MGN_AUTHORIZED_GPU_UUID') != value:
        raise ValueError('MGN_AUTHORIZED_GPU_UUID must match the external current manual authorization.')
    return value


def output_path(uuid, attempt):
    if not UUID.fullmatch(uuid) or type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
        raise ValueError('Invalid GPU/case attempt; at most three attempts are permitted.')
    path = SMOKE_ROOT / f'{CASE}_{uuid}_attempt{attempt}' / 'fp32'
    if path.resolve() != path or path.exists() or path.is_symlink():
        raise ValueError('Smoke output must be fresh, isolated and free of symlink redirection.')
    return path


def retry_guard(uuid, attempt, code_hash, correction_note):
    if attempt == 1:
        for number in range(1, MAX_ATTEMPTS+1):
            existing = SMOKE_ROOT / f'{CASE}_{uuid}_attempt{number}' / 'fp32'
            if existing.exists():
                raise ValueError('This case has an existing attempt; cannot reset its case identity.')
        return None
    prior = SMOKE_ROOT / f'{CASE}_{uuid}_attempt{attempt-1}' / 'fp32' / 'summary.json'
    if not prior.is_file():
        raise ValueError('Corrective retry requires the previous preserved outcome.')
    previous = json.loads(prior.read_text())
    if (not correction_note or previous.get('status') not in ('ERROR', 'OOM')
            or previous.get('code_hash') == code_hash or previous.get('case') != CASE
            or previous.get('gpu_uuid') != uuid
            or previous.get('configuration_sha256') != canonical_hash(configuration())):
        raise ValueError('Retry requires a documented implementation correction, changed code and identical scientific case.')
    # A true capacity OOM is never retried by the coordinator. An OOM retry is
    # only for an identified implementation defect, after passing CPU checks.
    return dict(path=str(prior), sha256=file_hash(prior), correction_note=correction_note)


def cpu_evidence():
    result = []
    for name in ('mgn_cpu_qualification.json', 'mgn_synthetic_qualification.json', 'mgn_gpu_smoke_contract.json'):
        path = ROOT/'docs'/name
        receipt = json.loads(path.read_text())
        expected_status = 'CPU_PASS' if name == 'mgn_gpu_smoke_contract.json' else 'PASS'
        if receipt.get('status') != expected_status:
            raise ValueError('Required CPU qualification has not passed.')
        if name == 'mgn_gpu_smoke_contract.json' and receipt.get('code_hash') != file_hash(__file__):
            raise ValueError('GPU harness changed after its CPU guard qualification.')
        if name == 'mgn_synthetic_qualification.json':
            for relative, digest in receipt['source_sha256'].items():
                if file_hash(ROOT/relative) != digest:
                    raise ValueError(f'CPU-qualified source changed: {relative}')
        result.append(dict(path=str(path), sha256=file_hash(path)))
    return result


def supervise(command, environment, output, *, popen=subprocess.Popen):
    """One child process group; timeout terminates only that owned group."""
    with (output/'stdout.log').open('x') as stdout, (output/'stderr.log').open('x') as stderr:
        child = popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            return child.wait(timeout=MAX_SECONDS), False
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            return child.returncode, True
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            raise


def summarize(manifest, worker, returncode, timed_out):
    status = 'TIMEOUT' if timed_out else worker.get('status', 'ERROR')
    if status not in ('PASS', 'OOM', 'ERROR', 'TIMEOUT'):
        status = 'ERROR'
    if (status == 'PASS' and (returncode != 0 or worker.get('completed_steps') != STEPS
            or worker.get('optimizer_step_calls') != STEPS or len(worker.get('steps', [])) != STEPS)):
        status = 'ERROR'
    return dict(**manifest, status=status, worker_returncode=returncode, timed_out=timed_out,
                worker=worker, finished_utc=datetime.now(timezone.utc).isoformat())


def failure_receipt(output, receipt, error, torch_backend=None):
    """Fault-injectable OOM/timeout path; never retry or alter the recipe."""
    receipt['status'] = ('OOM' if torch_backend is not None and isinstance(error, torch_backend.cuda.OutOfMemoryError)
                         else 'TIMEOUT' if isinstance(error, TimeoutError) else 'ERROR')
    receipt['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    if torch_backend is not None and torch_backend.cuda.is_initialized():
        try:
            cuda = torch_backend.cuda
            free, total = cuda.mem_get_info(0)
            receipt['failure_memory'] = dict(peak_allocated_bytes=cuda.max_memory_allocated(0),
                peak_reserved_bytes=cuda.max_memory_reserved(0), allocated_bytes=cuda.memory_allocated(0),
                reserved_bytes=cuda.memory_reserved(0), free_bytes=free, total_bytes=total)
        except Exception as memory_error:
            receipt['failure_memory_error'] = repr(memory_error)
    write_json(Path(output)/'worker_receipt.json', receipt)
    return receipt


def parent(args):
    uuid = authorized_uuid(args.gpu_uuid)
    output = output_path(uuid, args.attempt)
    code_hash = file_hash(__file__)
    correction = retry_guard(uuid, args.attempt, code_hash, args.correction_note)
    accepted_cpu = cpu_evidence()
    dirty_code = subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','reid','scripts'], cwd=ROOT, text=True)
    if dirty_code:
        raise ValueError('Commit the qualified implementation/harness before GPU admission; source worktree is dirty.')
    if not args.authorization_note.strip():
        raise ValueError('Record the source of current user/manual GPU authority.')
    if file_hash(WEIGHTS) != SHA256:
        raise ValueError('Required cached ImageNet artifact hash mismatch.')
    # This audit is read-only and provides no permission/LODO opt-in itself.
    from reid.utils.lodo_device import audit_gpu
    inventory = audit_gpu(uuid)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, case=CASE, gpu_uuid=uuid, attempt=args.attempt,
        precision='FP32', max_optimizer_calls=STEPS, worker_timeout_seconds=MAX_SECONDS,
        authorization_note=args.authorization_note, gpu_before=inventory,
        configuration=configuration(), configuration_sha256=canonical_hash(configuration()),
        weights_sha256=SHA256, weights_bytes=WEIGHTS.stat().st_size, code_hash=code_hash,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        cpu_evidence=accepted_cpu, correction=correction,
        started_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'manifest.json', manifest)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=uuid, MGN_AUTHORIZED_GPU_UUID=uuid,
                       PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--gpu-uuid', uuid,
               '--attempt', str(args.attempt), '--worker-output', str(output)]
    try:
        returncode, timed_out = supervise(command, environment, output)
        path = output/'worker_receipt.json'
        worker = json.loads(path.read_text()) if path.is_file() else {}
        result = summarize(manifest, worker, returncode, timed_out)
    except BaseException as error:
        result = summarize(manifest, dict(status='ERROR', error=repr(error)), -1, False)
        write_json(output/'summary.json', result)
        raise
    try:
        # Parent owns no CUDA context. Allow its PID only to avoid treating
        # NVML's trailing utilization sample as new work after child exit.
        result['gpu_after'] = audit_gpu(uuid, allow_pid=os.getpid())
    except ValueError as error:
        result['gpu_after_error'] = str(error)
        if result['status'] == 'PASS': result['status'] = 'ERROR'
    write_json(output/'summary.json', result)
    print(json.dumps(dict(status=result['status'], output=str(output), completed_steps=worker.get('completed_steps',0))))
    return 0 if result['status'] == 'PASS' else 2


def worker(args):
    uuid = authorized_uuid(args.gpu_uuid)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Worker visibility must contain only the authorized UUID before CUDA import/setup.')
    output = Path(args.worker_output)
    expected = SMOKE_ROOT/f'{CASE}_{uuid}_attempt{args.attempt}'/'fp32'
    if output != expected or output.resolve() != output:
        raise ValueError('Worker output does not match the immutable case identity.')
    manifest = json.loads((output/'manifest.json').read_text())
    if (manifest.get('case') != CASE or manifest.get('gpu_uuid') != uuid
            or manifest.get('code_hash') != file_hash(__file__)
            or manifest.get('configuration_sha256') != canonical_hash(configuration())
            or manifest.get('attempt') != args.attempt):
        raise ValueError('Worker manifest/code/scientific inputs changed.')
    signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError('MGN worker deadline')))
    signal.alarm(MAX_SECONDS)
    receipt = dict(status='RUNNING', stage='admission', attempted_batches=0,
                   optimizer_step_calls=0, completed_steps=0, skipped_steps=0, steps=[], events=[])
    started = time.monotonic()
    torch = None
    def record(stage):
        receipt['stage'] = stage
        receipt['events'].append(dict(stage=stage, elapsed_seconds=time.monotonic()-started))
        write_json(output/'worker_receipt.json', receipt)
    try:
        record('admission')
        from reid.utils.lodo_device import audit_gpu
        audit_gpu(uuid)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Authorized CUDA device unavailable; no fallback.')
        properties = torch.cuda.get_device_properties(0)
        observed = str(getattr(properties, 'uuid', ''))
        if not observed.startswith('GPU-'): observed = 'GPU-'+observed
        if observed != uuid:
            raise ValueError('CUDA logical/physical UUID mismatch.')
        supported = [int(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith('sm_') and a[3:].isdigit()]
        if not any(a//10 == properties.major and a%10 <= properties.minor for a in supported):
            raise ValueError('PyTorch build has no verified compatible device architecture.')
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        torch.manual_seed(42)
        device = torch.device('cuda:0')
        receipt['runtime'] = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu_name=properties.name, uuid=observed, visible=os.environ['CUDA_VISIBLE_DEVICES'], precision='FP32',
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            torch_architectures=torch.cuda.get_arch_list())
        audit_gpu(uuid, allow_pid=os.getpid())
        torch.cuda.reset_peak_memory_stats(0)
        record('model_initialization')
        from reid.models.build import build_model
        from reid.models.outputs import ensure_output_dict
        from reid.losses.build import build_criterion
        from reid.optim.build import build_optimizer
        from reid.utils.config import validate_reid_config
        cfg = configuration()
        validate_reid_config(cfg, num_classes=1041)
        model = build_model(cfg, 1041).to(device).train()
        criterion = build_criterion(cfg, 1041, model.feat_dim).to(device)
        optimizer = build_optimizer(cfg, model)
        assert all(p.device == device and p.dtype == torch.float32 for p in model.parameters())
        labels = torch.arange(16, device=device).repeat_interleave(4)
        mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
        std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
        def finite(tensors):
            checks = [torch.isfinite(t).all() for t in tensors if t.is_floating_point()]
            return bool(torch.stack(checks).all().item())
        for index in range(STEPS):
            audit_gpu(uuid, allow_pid=os.getpid())
            if index: torch.cuda.reset_peak_memory_stats(0)
            receipt['attempted_batches'] += 1
            record(f'step_{index+1}/input')
            images = (torch.rand(64,3,384,128,device=device)-mean)/std
            optimizer.zero_grad(set_to_none=True)
            before = model.backbone.shared.conv1.weight.detach().cpu().clone()
            record(f'step_{index+1}/forward')
            outputs = ensure_output_dict(model(images))
            if (outputs['emb'].shape != (64,2048) or len(outputs['logits']) != 8
                    or any(h.shape != (64,1041) for h in outputs['logits'])
                    or len(outputs['metric_features']) != 3
                    or any(f.shape != (64,256) for f in outputs['metric_features'])):
                raise ValueError('MGN GPU output shape contract failed.')
            record(f'step_{index+1}/loss')
            loss, logs = criterion(outputs, labels)
            receipt['latest_loss'] = {key: value if math.isfinite(value) else str(value) for key, value in logs.items()}
            if not all(math.isfinite(v) for v in logs.values()) or not torch.isfinite(loss).item():
                raise ValueError('Nonfinite GPU loss.')
            if not math.isclose(logs['loss/total'], logs['loss/id']+logs['loss/triplet'], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError('MGN family aggregation mismatch.')
            record(f'step_{index+1}/backward')
            loss.backward()
            parameters = list(model.parameters())
            if any(p.grad is None for p in parameters) or not finite([p.grad for p in parameters]):
                raise ValueError('Missing or nonfinite GPU gradients.')
            audit_gpu(uuid, allow_pid=os.getpid())
            record(f'step_{index+1}/optimizer')
            receipt['optimizer_step_calls'] += 1
            optimizer.step()
            torch.cuda.synchronize(0)
            momentum = [optimizer.state[p].get('momentum_buffer') for p in parameters]
            if (any(value is None for value in momentum) or not finite(momentum)
                    or not finite(list(model.parameters())+list(model.buffers()))
                    or torch.equal(before, model.backbone.shared.conv1.weight.detach().cpu())):
                raise ValueError('Nonfinite/unchanged GPU optimizer/model state.')
            free, total = torch.cuda.mem_get_info(0)
            receipt['completed_steps'] += 1
            receipt['steps'].append(dict(step=index+1, loss=logs,
                embedding_shape=[64,2048], logits_shape=[8,64,1041], metric_shape=[3,64,256],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                free_bytes=free,total_bytes=total,finite_gradients=True,finite_model=True,finite_momentum=True,parameter_changed=True,
                elapsed_seconds=time.monotonic()-started))
            del outputs, loss, images, parameters, momentum, before
            record(f'step_{index+1}/complete')
        receipt['status'] = 'PASS'
        record('complete')
        return 0
    except BaseException as error:
        receipt['elapsed_seconds'] = time.monotonic()-started
        failure_receipt(output, receipt, error, torch)
        traceback.print_exc()
        return 2
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--attempt', type=int, choices=range(1,MAX_ATTEMPTS+1), default=1)
    parser.add_argument('--authorization-note', default='')
    parser.add_argument('--correction-note', default='')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    def interrupted(*unused):
        raise InterruptedError('MGN supervisor interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    return parent(args)


if __name__ == '__main__':
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    raise SystemExit(main())
