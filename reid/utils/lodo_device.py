"""LODO-only explicit CUDA admission; no allocator or Stage A device fallback.

The operator must FIRST verify independent allocation and set the authorization
UUID. An idle NVML snapshot cannot grant allocation. Visibility is deliberately
restricted to that single full UUID before Python starts, preventing global RNG
helpers from initializing another user's GPU. CUDA continuation requires a separate opt-in boundary policy.
"""
import csv
from datetime import datetime, timezone
import io
import os
import re
import subprocess

import torch

CUDA_FRESH_CAPABILITY = 'fresh_cuda_no_resume_v1'
_UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def _query(fields, kind='gpu'):
    try:
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}',
                                 '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=10)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('Cannot verify GPU ownership through nvidia-smi') from error


def audit_gpu(uuid, *, allow_pid=None):
    """Read-only physical UUID inventory; reject contention or incomplete evidence."""
    rows = _query('index,uuid,name,driver_version,memory.total,memory.free,utilization.gpu,compute_mode,mig.mode.current')
    if any(len(row) != 9 for row in rows):
        raise ValueError('Incomplete GPU inventory')
    matches = [row for row in rows if row[1] == uuid]
    if len(matches) != 1:
        raise ValueError('Authorized GPU UUID unavailable or ambiguous')
    row = matches[0]
    if row[7] not in ('Default', 'Exclusive_Process') or row[8] not in ('Disabled', '[N/A]', 'N/A'):
        raise ValueError('Unsupported GPU compute/MIG allocation mode')
    processes = _query('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps')
    if any(len(p) != 4 for p in processes):
        raise ValueError('Incomplete GPU process inventory')
    selected = [p for p in processes if p[0] == uuid]
    if any(not p[1].isdigit() or int(p[1]) != allow_pid for p in selected):
        raise ValueError('Authorized GPU is occupied; no fallback permitted')
    try:
        total, free, utilization = map(int, (row[4], row[5], row[6]))
        physical_index = int(row[0])
    except ValueError as error:
        raise ValueError('Unverifiable GPU resource inventory') from error
    if total <= 0 or not 0 <= free <= total or not 0 <= utilization <= 100:
        raise ValueError('Invalid GPU resource inventory')
    if allow_pid is None and (free == 0 or utilization != 0):
        raise ValueError('GPU activity/headroom is not safe for a fresh launch')
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), uuid=uuid,
                physical_index=physical_index, name=row[2], driver=row[3],
                total_mib=total, free_mib=free, utilization_percent=utilization,
                compute_mode=row[7], mig_mode=row[8], processes=selected)


def authorization_uuid(cfg):
    """Pure checks before any CUDA query/setup, image access or output creation."""
    mode = cfg['system']['device']
    if mode == 'cpu':
        return None
    if mode != 'cuda':
        raise ValueError('CPU-validated recovery requires explicit cpu; auto is forbidden for LODO')
    from reid.utils.lodo_recovery import recovery_policy, cuda_boundary
    recovery_policy(cfg)
    if cfg['train']['save'].get('resume') and not cuda_boundary(cfg):
        raise ValueError('CUDA resume is unsupported; fresh restart requires a new namespace')
    uuid = os.environ.get('LODO_AUTHORIZED_GPU_UUID', '')
    if not _UUID.fullmatch(uuid):
        raise ValueError('CPU-validated recovery only without an explicitly authorized full GPU UUID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must contain only the authorized full GPU UUID')
    index = cfg['system'].get('gpu_id')
    if type(index) is not int or index != 0:
        raise ValueError('Single-UUID visibility requires logical system.gpu_id=0, not the physical index')
    if cfg['system'].get('amp') is not False:
        raise ValueError('LODO CUDA requires the frozen FP32 recipe')
    return uuid


def select_lodo_device(cfg):
    """Return device, recheck callback and operational evidence; never fallback.

    Real CUDA setup is permitted only AFTER admission. CPU tests mock the CUDA
    API; this function itself does not constitute resource authorization.
    """
    from reid.utils.lodo_recovery import cuda_boundary
    from reid.utils.lodo_cuda_policy import activate_policy, validate_runtime, policy_contract
    uuid = authorization_uuid(cfg)
    activate_policy(cfg)
    selected_execution_policy = policy_contract(cfg)
    if uuid is None:
        from reid.utils.device import select_device
        device, _ = select_device('cpu', cfg['system'].get('gpu_id', 0), cfg)
        return device, None, None
    evidence = audit_gpu(uuid)  # must precede even CUDA availability/setup
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Authorized CUDA device unavailable; no CPU/GPU fallback')
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(properties, 'uuid', ''))
    if not actual_uuid.startswith('GPU-'):
        actual_uuid = 'GPU-' + actual_uuid
    if actual_uuid != uuid:
        raise ValueError('CUDA logical device UUID differs from authorized physical GPU')
    # A compiled older minor in the same major supports Pascal 6.1 via sm_60.
    supported = [int(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith('sm_') and a[3:].isdigit()]
    if not any(a // 10 == properties.major and a % 10 <= properties.minor for a in supported):
        raise ValueError('PyTorch build has no verified compatible CUDA architecture')
    audit_gpu(uuid, allow_pid=os.getpid())  # recheck after CUDA runtime discovery
    torch.cuda.set_device(0)
    def recheck():
        if policy_contract(cfg) != selected_execution_policy:
            raise ValueError('CUDA execution policy changed during execution')
        validate_runtime(cfg)
        if authorization_uuid(cfg) != uuid:
            raise ValueError('GPU authorization changed during execution')
        audit_gpu(uuid, allow_pid=os.getpid())
    evidence.update(logical_index=0, visibility=uuid, torch_version=torch.__version__,
                    torch_cuda=torch.version.cuda, capability=('epoch_boundary_rollback_cuda_v1' if cuda_boundary(cfg) else CUDA_FRESH_CAPABILITY),
                    cudnn_benchmark=cfg['repro']['benchmark'],
                    cudnn_deterministic=cfg['repro']['deterministic'],
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if policy_contract(cfg) is not None:
        validate_runtime(cfg)
        evidence['execution_policy'] = policy_contract(cfg)
    from reid.utils.mgn_lodo_protocol import is_mgn_lodo
    if is_mgn_lodo(cfg):
        evidence.update(tensor_precision='float32', amp=False,
                        convolution_tf32_permitted=torch.backends.cudnn.allow_tf32,
                        hardware_compute_capability=[properties.major, properties.minor],
                        convolution_tf32_hardware_capable=properties.major >= 8,
                        actual_tf32_kernel_use_measured=False)
    return torch.device('cuda:0'), recheck, evidence


def validate_placement(cfg, model, criterion, optimizer, auxiliary):
    """One device for parameters/buffers and parameter-shaped optimizer state.

    Adam's non-capturable scalar step counter intentionally remains on CPU.
    Deserialization stays CPU-first for integrity and CPU evaluation, then
    load_state_dict copies into the existing device-owned parameters.
    """
    expected = torch.device('cuda:0' if cfg['system']['device'] == 'cuda' else 'cpu')
    for module in (model, criterion):
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.device != expected:
                raise ValueError('LODO model/Center/buffer device placement mismatch')
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError('LODO model/Center must remain FP32')
    for opt in (optimizer, auxiliary):
        if opt is None:
            continue
        for group in opt.param_groups:
            for parameter in group['params']:
                if parameter.device != expected:
                    raise ValueError('LODO optimizer parameter device mismatch')
        for parameter, values in opt.state.items():
            for name, value in values.items():
                if torch.is_tensor(value) and name != 'step' and value.device != parameter.device:
                    raise ValueError('LODO optimizer tensor device mismatch')
    return expected
