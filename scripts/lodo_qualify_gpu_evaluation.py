"""Bounded CPU/CUDA evaluation diagnostic; never publishes benchmark results.

Run in a fresh interpreter with the approved UUID and strict workspace already
exported. Real target reads are limited to a fixed subset. No training occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def qualification(request, output):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from reid.utils.lodo_evaluation import validate_terminal, target_metadata, metric_vector
    from reid.utils.lodo_evaluation_device import select_device, preflight, EvaluationProgress
    from reid.utils.checkpoint import reconstruct_model
    from reid.utils.evaluation_config import build_evaluation_config
    from reid.data.build import build_test_loader
    from reid.engine.evaluator import extract_features, evaluate_reid
    from reid.metrics.distance import normalize, compute_dist
    from reid.utils.lodo_checkpoint import file_digest
    from reid.utils.lodo_cuda_policy import runtime_controls

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output or 'exp' in output.parts:
        raise ValueError('Use a new diagnostic path outside authoritative exp directories')
    payload, proof, checkpoint = validate_terminal(request)
    cfg, identity = payload['cfg'], payload['identity']
    gpu_request = dict(request, device='cuda')
    # Check physical resources before constructing/reading any target images.
    preflight(gpu_request, cfg, output)
    metadata = target_metadata(cfg, identity, request['data_root'])
    runtime = dict(data=dict(root=request['data_root'],test=dict(dataset=metadata['dataset'],
                        batch=dict(size=32)),num_workers=0,pin_memory=False),system=dict(device='cpu'))
    evaluation = build_evaluation_config(cfg, runtime, 'NOT_EXECUTED_DIAGNOSTIC')
    full = build_test_loader(evaluation)
    ds = full.dataset
    queries, seen = [], set()
    for i, (pid, mark) in enumerate(zip(ds.pids, ds.marks)):
        if int(mark) == 0 and int(pid) not in seen:
            seen.add(int(pid)); queries.append(i)
            if len(queries) == 4: break
    selected = list(queries)
    for q in queries:
        positives = [i for i, (pid, cam, mark) in enumerate(zip(ds.pids, ds.cams, ds.marks))
                     if int(mark) == 1 and int(pid) == int(ds.pids[q]) and int(cam) != int(ds.cams[q])]
        if not positives: raise ValueError('Diagnostic query has no cross-camera positive')
        selected.extend(positives[:3])
    selected.extend(i for i, (pid, mark) in enumerate(zip(ds.pids, ds.marks))
                    if int(mark) == 1 and int(pid) not in seen)
    selected = list(dict.fromkeys(selected))[:32]
    loader = DataLoader(Subset(ds, selected), batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=full.collate_fn)
    model = reconstruct_model(payload).to('cpu')
    manifest = [dict(index=int(i), name=str(ds.im_names[i]), pid=int(ds.pids[i]),
                     camera=int(ds.cams[i]), mark=int(ds.marks[i])) for i in selected]
    started = time.monotonic()
    cpu = extract_features(model, loader, torch.device('cpu'))
    cpu_seconds = time.monotonic()-started
    device, guard, evidence = select_device(gpu_request, cfg, output)
    model.to(device)
    progress = EvaluationProgress(device, guard)
    torch.cuda.synchronize()
    started = time.monotonic()
    gpu = extract_features(model, loader, device, progress=progress,before_batch=progress.before_batch)
    torch.cuda.synchronize()
    gpu_seconds = time.monotonic()-started
    repeated = extract_features(model, loader, device, before_batch=guard)
    guard()
    # Compare features after the same approved normalization used for ranking.
    cpu_norm, gpu_norm = normalize(cpu[0]), normalize(gpu[0])
    query, gallery = cpu[4] == 0, cpu[4] == 1
    cpu_dist = compute_dist(cpu_norm[query],cpu_norm[gallery])
    gpu_dist = compute_dist(gpu_norm[query],gpu_norm[gallery])
    features_pass = bool(np.allclose(cpu_norm, gpu_norm, atol=1e-5, rtol=1e-4))
    distances_pass = bool(np.allclose(cpu_dist, gpu_dist, atol=2e-5, rtol=1e-4))
    # Avoid a third forward pass: feed each captured embedding through the
    # original evaluator, retaining its exact PID/camera/ranking implementation.
    class EmbeddingModel(torch.nn.Module):
        def forward(self, x): return {'emb':x}
    def metrics(captured):
        batch=(torch.from_numpy(captured[0]),torch.from_numpy(captured[1]),
               torch.from_numpy(captured[2]),list(captured[3]),torch.from_numpy(captured[4]))
        return metric_vector(evaluate_reid(evaluation,EmbeddingModel(),[batch],torch.device('cpu')))
    cpu_metrics, gpu_metrics = metrics(cpu), metrics(gpu)
    metrics_pass = all(abs(cpu_metrics[k]-gpu_metrics[k])<=1e-6 for k in cpu_metrics)
    repeat_pass = bool(np.array_equal(gpu[0], repeated[0]))
    terminal_preserved = file_digest(checkpoint)==proof['checkpoint_sha256']
    report=dict(diagnostic_only=True,official_results_published=False,
                status='PASS' if all((features_pass,distances_pass,metrics_pass,repeat_pass,terminal_preserved)) else 'FAIL',
                architecture=request['architecture'],fold=request['fold'],target=request['target'],
                terminal_sha256=proof['checkpoint_sha256'],terminal_preserved=terminal_preserved,
                source_files={str(p.relative_to(ROOT)):file_digest(p) for p in
                    [ROOT/'reid/engine/evaluator.py',ROOT/'reid/utils/lodo_evaluation_device.py',Path(__file__).resolve()]},
                samples=manifest,sample_count=len(manifest),
                manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                cpu_seconds=cpu_seconds,gpu_seconds=gpu_seconds,
                bounded_feature_speedup=cpu_seconds/gpu_seconds,
                normalized_embedding_max_abs=float(np.max(np.abs(cpu_norm-gpu_norm))),
                distance_max_abs=float(np.max(np.abs(cpu_dist-gpu_dist))),
                tolerance=dict(normalized_embedding_atol=1e-5,normalized_embedding_rtol=1e-4,
                               distance_atol=2e-5,distance_rtol=1e-4,metric_atol=1e-6,metric_rtol=0),
                features_pass=features_pass,distances_pass=distances_pass,metrics_pass=metrics_pass,
                cuda_repeat_bitwise_equal=repeat_pass,cpu_metrics=cpu_metrics,gpu_metrics=gpu_metrics,
                gpu_admission=evidence,execution_controls=runtime_controls(),
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({k:report[k] for k in ('status','sample_count','cpu_seconds','gpu_seconds',
        'bounded_feature_speedup','normalized_embedding_max_abs','distance_max_abs','metrics_pass','cuda_repeat_bitwise_equal')}),flush=True)
    if report['status'] != 'PASS': raise ValueError('GPU evaluation qualification failed; preserve report')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from reid.utils.config import load_config
    qualification(load_config(args.request),args.output)
