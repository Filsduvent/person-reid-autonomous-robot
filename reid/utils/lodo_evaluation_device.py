"""Explicit LODO evaluation device admission; ranking remains on CPU.

CPU is the backward-compatible default. GPU feature extraction reuses the
training UUID/ownership/runtime guards, without modifying the source recipe.
"""
import copy
import shutil
from pathlib import Path

from reid.utils import lodo_device
from reid.utils.lodo_cuda_policy import STRICT_POLICY, policy_contract

MIN_FREE_MIB = 4096
MIN_DISK_BYTES = 2 * 1024**3


def execution_spec(request):
    device = request.get('device', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Evaluation device must be explicit cpu or cuda; no auto/fallback')
    if device == 'cpu':
        return dict(device='cpu', precision='fp32', ranking_device='cpu')
    from reid.utils.lodo_operational import GPU
    return dict(device='cuda', precision='fp32', ranking_device='cpu', gpu_uuid=GPU,
                cuda_policy=STRICT_POLICY)


def preflight(request, cfg, output):
    """NVML/disk only: permitted in a CUDA-hidden metadata-only --check child."""
    spec = execution_spec(request)
    if spec['device'] == 'cpu':
        return None
    source = copy.deepcopy(cfg)
    source['system']['device'] = 'cuda'
    contract = policy_contract(source)
    if contract is None or contract['name'] != STRICT_POLICY:
        raise ValueError('GPU evaluation requires the frozen strict deterministic CUDA policy')
    evidence = lodo_device.audit_gpu(spec['gpu_uuid'])
    if evidence['free_mib'] < MIN_FREE_MIB:
        raise ValueError('Insufficient evaluation GPU memory headroom (4096 MiB required)')
    parent = Path(output)
    while not parent.exists():
        parent = parent.parent
    disk = shutil.disk_usage(parent).free
    if disk < MIN_DISK_BYTES:
        raise ValueError('Insufficient evaluation disk headroom (2 GiB required)')
    return dict(gpu=evidence, disk_free_bytes=disk)


def select_device(request, cfg, output):
    import torch
    spec = execution_spec(request)
    if spec['device'] == 'cpu':
        return torch.device('cpu'), None, None
    resources = preflight(request, cfg, output)
    selection = copy.deepcopy(cfg)
    selection['system'].update(device='cuda', gpu_id=0, amp=False)
    # Evaluation restores no optimizer/training position, but uses the same
    # strict runtime and single-UUID placement checks as the trained model.
    selection['train']['save']['resume'] = ''
    device, guard, evidence = lodo_device.select_lodo_device(selection)
    evidence.update(evaluation_execution=spec, resources=resources)
    return device, guard, evidence


class EvaluationProgress:
    def __init__(self, device, guard=None, interval=10):
        self.device, self.guard, self.interval = str(device), guard, interval
        self.completed_batches = 0

    def before_batch(self):
        if self.guard is not None and self.completed_batches % self.interval == 0:
            self.guard()

    def __call__(self, event):
        if event['phase'] != 'features':
            print('[Evaluation] phase=' + event['phase'], flush=True)
            return
        self.completed_batches = event['batch']
        if event['batch'] not in (1, event['batches']) and event['batch'] % self.interval:
            return
        elapsed = event['elapsed_seconds']
        speed = event['images'] / elapsed if elapsed > 0 else 0
        remaining = (event['batches']-event['batch']) * elapsed / event['batch']
        print(f"[Evaluation] device={self.device} features {event['batch']}/{event['batches']} "
              f"({100*event['batch']/event['batches']:.1f}%) images={event['images']} "
              f"elapsed={elapsed:.1f}s speed={speed:.2f} images/s feature_eta={remaining:.1f}s",
              flush=True)
