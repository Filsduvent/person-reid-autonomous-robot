"""CPU-only tests for graph localization, policy isolation, and one-update scope."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import lodo_cuda_backward_probe as probe
import lodo_cuda_abc_diagnostic as abc
from test_lodo_checkpoint import synthetic, cfg_for, parts, epoch, seed
from reid.engine.update_control import UpdateProgress


def traced(model, x):
    trace = probe.BackwardTrace()
    root = model(x).sum()
    trace.install_graph(root, list(model.named_parameters()))
    root.backward()
    trace.close()
    return dict(nodes=trace.nodes, execution=trace.execution, batch=None, forward={})


def test_node_hooks_preserve_gradients_rng_and_complete_graph():
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(inplace=True), torch.nn.Linear(4, 1))
    control = copy.deepcopy(model)
    x = torch.randn(5, 3)
    rng = torch.get_rng_state().clone()
    left = traced(model, x)
    control(x).sum().backward()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a.grad, b.grad) for a,b in zip(model.parameters(), control.parameters()))
    assert all('grad_inputs' in row and 'grad_outputs' in row for row in left['nodes'])
    model.zero_grad(set_to_none=True)
    right = traced(model, x)
    result = probe.compare_traces(left, right)
    assert not result['differing_nodes']
    assert result['execution_order_exact']
    json.dumps(left)
    assert not torch.cuda.is_initialized()


class DifferentBackward(torch.autograd.Function):
    offset = 0.
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return grad + DifferentBackward.offset


def test_identifies_equal_operand_backward_divergence():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([1., 2.]))
        def forward(self, x):
            return DifferentBackward.apply(self.weight * x)
    model = Model(); x = torch.ones(2)
    DifferentBackward.offset = 0.
    left = traced(model, x)
    model.zero_grad(set_to_none=True)
    DifferentBackward.offset = .01
    right = traced(model, x)
    DifferentBackward.offset = 0.
    result = probe.compare_traces(left, right)
    boundaries = result['equal_input_divergent_output_boundaries']
    assert len(boundaries) == 1
    assert 'DifferentBackward' in boundaries[0]['operation']
    assert boundaries[0]['incoming_exact'] and boundaries[0]['saved_operands_exact']


def test_topology_mismatch_is_explicit():
    left = traced(torch.nn.Linear(2, 1), torch.ones(1, 2))
    right = copy.deepcopy(left); right['nodes'][0]['operation'] = 'other'
    with pytest.raises(ValueError, match='topology'):
        probe.compare_traces(left, right)


def test_full_hash_detects_changes_outside_bounded_sample():
    a = torch.zeros(1024); b = a.clone(); b[-1] = .001
    ea, eb = probe.tensor_evidence(a), probe.tensor_evidence(b)
    assert ea['sample'] == eb['sample'] and ea['sha256'] != eb['sha256']
    assert probe.hashes(ea) != probe.hashes(eb)


def test_nonfinite_evidence_is_explicit_and_empty_supported():
    assert not probe.tensor_evidence(torch.tensor([float('nan')]))['finite']
    assert probe.tensor_evidence(torch.empty(0))['abs_max'] == 0.
    assert probe.tensor_evidence((None, torch.tensor([1]))) [0] is None


def test_policy_requires_workspace_before_any_backend_mutation(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    before = abc.environment()
    with pytest.raises(ValueError, match='workspace'):
        probe.apply_policy('deterministic')
    assert abc.environment() == before and not torch.cuda.is_initialized()


def test_diagnostic_policy_is_explicit_preserves_tf32_and_restorable(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    before = abc.environment()
    try:
        probe.apply_policy('deterministic')
        after = abc.environment()
        assert not after['cudnn_benchmark'] and after['cudnn_deterministic'] and after['deterministic_algorithms']
        assert after['matmul_tf32'] == before['matmul_tf32']
        assert after['cudnn_tf32'] == before['cudnn_tf32']
    finally:
        torch.backends.cudnn.benchmark = before['cudnn_benchmark']
        torch.backends.cudnn.deterministic = before['cudnn_deterministic']
        torch.use_deterministic_algorithms(before['deterministic_algorithms'])
    with pytest.raises(ValueError): probe.apply_policy('unknown')
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('name', ['A', '../escape', 'deterministic-3', '/absolute'])
def test_invalid_probe_names_rejected_before_writes(tmp_path, monkeypatch, name):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    base = tmp_path/'exp/lodo_smoke/phase9r1-backward-titan-test'
    base.mkdir(parents=True)
    with pytest.raises(ValueError): probe.probe_directory(base, name)
    assert not list(base.iterdir())


def test_namespace_alias_and_collision_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    base = tmp_path/'exp/lodo_smoke/phase9r1-backward-titan-test'
    base.mkdir(parents=True)
    probe.probe_directory(base, 'original-1')
    with pytest.raises(FileExistsError): probe.probe_directory(base, 'original-1')
    alias = base.parent/'phase9r1-backward-titan-alias'
    alias.symlink_to(base, target_is_directory=True)
    with pytest.raises(ValueError): probe.probe_directory(alias, 'original-2')


def test_wrong_uuid_rejected_before_checkpoint_or_source_access(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', abc.UUID)
    monkeypatch.setenv('LODO_AUTHORIZED_GPU_UUID', abc.UUID)
    monkeypatch.setattr(abc, 'read_common', lambda *a: pytest.fail('Checkpoint accessed'))
    with pytest.raises(ValueError, match='TITAN'): probe.execute(tmp_path, 'original-1')
    assert not list(tmp_path.iterdir()) and not torch.cuda.is_initialized()


def test_ownership_fail_closed():
    model = torch.nn.Linear(2, 1)
    criterion = SimpleNamespace(center_loss=SimpleNamespace(centers=torch.nn.Parameter(torch.zeros(1))))
    auxiliary = torch.optim.SGD([criterion.center_loss.centers], lr=.5)
    opt = torch.optim.Adam([dict(params=[p], param_name=n) for n,p in model.named_parameters()])
    probe.ownership(model, opt, criterion, auxiliary)
    opt.param_groups[0]['params'] = [torch.nn.Parameter(torch.zeros_like(model.weight))]
    with pytest.raises(ValueError, match='association'): probe.ownership(model, opt, criterion, auxiliary)


def test_shared_engine_single_update_stop_retains_main_center_and_schedule(synthetic):
    cfg = cfg_for(); seed(); objects = parts(cfg)
    progress = UpdateProgress(5, auxiliary=True)
    objects['loader']._observer.original = abc.ShortTestEpoch(objects['loader'].batch_sampler)
    epoch(objects, progress)
    assert progress.completed_primary_updates == 2
    center_step = objects['auxiliary'].step
    def single(*args, **kwargs):
        result = center_step(*args, **kwargs)
        progress.max_updates = 3
        return result
    trace = probe.BackwardTrace()
    trace.install_model(objects['model'], objects['loader'])
    def loss_hook(module, args, result):
        trace.install_graph(result[0], list(objects['model'].named_parameters()) +
                            [('center', objects['criterion'].center_loss.centers)])
    trace.handles.append(objects['criterion'].register_forward_hook(loss_hook))
    try:
        with patch_step(objects['auxiliary'], single):
            epoch(objects, progress)
    finally:
        trace.close()
    assert trace.nodes and trace.batch and trace.forward
    assert all('grad_inputs' in row for row in trace.nodes)
    json.dumps(trace.nodes)
    assert progress.completed_primary_updates == progress.auxiliary_steps == 3
    assert objects['scheduler'].completed_updates == 3
    assert progress.forward_batches == progress.backward_batches == progress.attempted_batches == 3
    assert progress.consistent and not progress.failed
    progress.check_totals()
    assert cfg['train']['max_updates'] == 5


def patch_step(optimizer, fn):
    from unittest.mock import patch
    return patch.object(optimizer, 'step', fn)


def test_probe_contract_selects_existing_verified_resume_validator_and_restores():
    original = abc.protocol.validate_output_path
    with probe.probe_contract():
        assert abc.protocol.validate_output_path is abc.output_path
        with pytest.raises(AssertionError, match='prohibited'):
            abc.train.build_test_loader(None)
    assert abc.protocol.validate_output_path is original
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('dtype', [torch.bool, torch.int64, torch.float32])
def test_saved_mask_and_index_dtypes_supported(dtype):
    evidence = probe.tensor_evidence(torch.tensor([0, 1], dtype=dtype))
    assert evidence['finite'] and evidence['abs_max'] == 1.
    assert len(evidence['sha256']) == 64
