"""Operational GPU-only wrapper around unchanged common evaluation entry points."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def require_cuda_device(device):
    if device.type != 'cuda' or device.index not in (None, 0):
        raise RuntimeError('Evaluation requires the authorized logical cuda:0; CPU fallback refused.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True)
    parser.add_argument('--entry', choices=('scripts/evaluate.py', 'scripts/evaluate_cross_domain.py'), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch
    from scripts.mgn_gpu_smoke import file_hash
    from reid.utils.lodo_device import audit_gpu
    import reid.utils.device as device_module
    import reid.engine.evaluator as evaluator
    uuid = os.environ['MGN_AUTHORIZED_GPU_UUID']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('Single full-UUID visibility required.')
    audit_dir = Path(args.audit)
    audit_dir.mkdir(parents=True, exist_ok=True)
    before = audit_gpu(uuid)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Authorized CUDA device unavailable; no fallback.')
    prop = torch.cuda.get_device_properties(0)
    actual = str(prop.uuid)
    if not actual.startswith('GPU-'):
        actual = 'GPU-' + actual
    if actual != uuid:
        raise RuntimeError('CUDA logical0 UUID mismatch.')
    manifest = json.loads((ROOT/'docs/mgn_experiment_manifest.json').read_text())
    expected = manifest['sources']['market1501']['runtime']
    torch.backends.cuda.matmul.allow_tf32 = expected['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = expected['cudnn_allow_tf32']
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if any(runtime[k] != expected[k] for k in runtime) or before['driver'] != expected['driver']:
        raise RuntimeError('Framework/driver runtime differs from Oneiros qualification.')
    (audit_dir/'gpu-start.json').write_text(json.dumps(dict(pid=os.getpid(), gpu=before, runtime=runtime,
         visibility=uuid, device_name=prop.name, entry_sha256=file_hash(ROOT/args.entry)), indent=2)+'\n')
    original_select = device_module.select_device
    def select(device_str, gpu_id, cfg=None):
        if device_str != 'cuda' or gpu_id != 0:
            raise RuntimeError('Explicit cuda logical0 required.')
        device, config = original_select(device_str, gpu_id, cfg)
        require_cuda_device(device)
        return device, config
    device_module.select_device = select
    original_evaluate = evaluator.evaluate_reid
    def evaluate(cfg, model, loader, device, logger=None, **kwargs):
        require_cuda_device(device)
        if (cfg['model']['name'] != 'mgn' or cfg['system']['amp'] is not False
                or cfg['eval']['flip_aggregation'] != 'mean' or not cfg['eval']['normalize_feat']
                or cfg['eval']['distance'] != 'euclidean' or cfg['eval']['rerank']['enabled']):
            raise RuntimeError('Frozen MGN retrieval policy required.')
        torch.backends.cudnn.benchmark = cfg['repro']['benchmark']
        torch.backends.cudnn.deterministic = cfg['repro']['deterministic']
        for tensor in list(model.parameters()) + list(model.buffers()):
            require_cuda_device(tensor.device)
        forwards = [0]
        def placement(module, inputs):
            require_cuda_device(inputs[0].device)
            if inputs[0].dtype != torch.float32:
                raise RuntimeError('FP32 inputs required.')
            forwards[0] += 1
        hook = model.register_forward_pre_hook(placement)
        last_audit = [0.0]
        def before_batch():
            now = time.monotonic()
            if now-last_audit[0] >= 30:
                value = audit_gpu(uuid, allow_pid=os.getpid())
                with (audit_dir/'gpu-observed.jsonl').open('a') as stream:
                    stream.write(json.dumps(value)+'\n')
                last_audit[0] = now
        def progress(value):
            with (audit_dir/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(value)+'\n')
            if value.get('phase') != 'features' or value.get('batch', 0) % 100 == 0:
                print('[MGN progress] '+json.dumps(value), flush=True)
        try:
            scores = original_evaluate(cfg, model, loader, device, logger=logger,
                                       progress=progress, before_batch=before_batch, **kwargs)
        finally:
            hook.remove()
        if forwards[0] != 2*len(loader):
            raise RuntimeError('Original/flip forward count mismatch.')
        torch.cuda.synchronize()
        (audit_dir/'gpu-completion.json').write_text(json.dumps(dict(
            device=str(device), fp32=True, amp=False, forwards=forwards[0], batches=len(loader),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=audit_gpu(uuid, allow_pid=os.getpid())), indent=2)+'\n')
        return scores
    evaluator.evaluate_reid = evaluate
    sys.argv = [args.entry] + (args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments)
    runpy.run_path(str(ROOT/args.entry), run_name='__main__')


if __name__ == '__main__':
    main()
