"""Explicit Stage B CUDA policy; pure selection, pre-context activation, no fallback.

Workspace must be supplied by the launching environment. Importing this module
is CUDA-free; scripts/train imports it before torch. Original/CPU paths are no-ops.
The strict bundle is immutable in code; configuration selects only its versioned ID.
"""
import os

ORIGINAL_POLICY = 'original_cuda_v1'
STRICT_POLICY = 'strict_cuda_deterministic_v1'
WORKSPACE = ':4096:8'
WORKSPACE_AT_IMPORT = os.environ.get('CUBLAS_WORKSPACE_CONFIG')


def policy_name(cfg):
    name = cfg.get('repro', {}).get('cuda_policy', ORIGINAL_POLICY)
    if name not in (ORIGINAL_POLICY, STRICT_POLICY):
        raise ValueError('Unsupported Stage B CUDA execution policy')
    if name == STRICT_POLICY:
        if cfg.get('protocol', {}).get('name') != 'multi_source_lodo' or cfg.get('system', {}).get('device') != 'cuda':
            raise ValueError('Strict CUDA policy requires explicit Stage B CUDA execution')
        if cfg['repro'].get('deterministic') is not True or cfg['repro'].get('benchmark') is not False:
            raise ValueError('Strict CUDA policy requires repro.deterministic=True and benchmark=False')
    return name


def policy_contract(cfg):
    if policy_name(cfg) != STRICT_POLICY:
        return None
    return dict(version=1, name=STRICT_POLICY, cudnn_benchmark=False,
                cudnn_deterministic=True, deterministic_algorithms=True,
                deterministic_warn_only=False, cublas_workspace_config=WORKSPACE,
                matmul_allow_tf32=False, cudnn_allow_tf32=True)


def validate_environment(cfg):
    """Pure checks; never set environment variables or touch CUDA."""
    if policy_name(cfg) == STRICT_POLICY:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != WORKSPACE:
            raise ValueError('Strict CUDA policy requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')
        if WORKSPACE_AT_IMPORT != WORKSPACE:
            raise ValueError('Strict CUDA workspace was configured too late; restart with it exported before imports')


def runtime_controls():
    import torch
    return dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def validate_runtime(cfg):
    contract = policy_contract(cfg)
    if contract is None:
        return
    validate_environment(cfg)
    expected = {k:v for k,v in contract.items() if k not in ('version', 'name')}
    if runtime_controls() != expected:
        raise ValueError('Strict CUDA execution controls differ from the selected policy')


def activate_policy(cfg):
    """Activate only while CUDA is uninitialized. Never retrofit a live context."""
    contract = policy_contract(cfg)
    if contract is None:
        return
    validate_environment(cfg)  # before even importing torch through this function
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('Strict CUDA policy must activate before CUDA initialization')
    if torch.backends.cuda.matmul.allow_tf32 is not False or torch.backends.cudnn.allow_tf32 is not True:
        raise ValueError('Strict CUDA policy requires the approved existing FP32/TF32 flags')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    validate_runtime(cfg)


def checkpoint_schema(cfg):
    return 2 if policy_name(cfg) == STRICT_POLICY else 1
