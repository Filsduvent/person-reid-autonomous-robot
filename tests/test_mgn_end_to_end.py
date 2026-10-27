"""Bounded CPU synthetic integration; no benchmark data or official training."""
import copy
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from reid.engine.train_loop import train_one_epoch
from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim.build import build_optimizer, build_scheduler
from reid.optim.lr_scheduler import step_completed_epoch
from reid.utils.checkpoint import save_checkpoint, reconstruct_model
from reid.utils.config import load_config, validate_reid_config
from scripts import train


def synthetic_config():
    return dict(model=dict(name='mgn', pretrained=False), loss=dict(
        id=dict(enabled=True, weight=1., label_smoothing=0., head_aggregation='mean'),
        triplet=dict(enabled=True, margin=1.2, weight=1., feature_source='model', feature_aggregation='mean'),
        center=dict(enabled=False)),
        optim=dict(name='sgd', lr=.01, momentum=.9, weight_decay=.0005,
                   weight_decay_bias=.0005, bias_lr_factor=1., nesterov=False),
        sched=dict(name='epoch_multistep', milestones=[60, 90], gamma=.1),
        train=dict(epochs=120, eval_interval=10, save=dict(metric='mAP', resume='', require_selection_state_on_resume=True)))


def digest(value):
    result = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                result.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item: visit(child)
        else:
            result.update(repr(item).encode())
    visit(value)
    return result.hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Recorder:
    def __init__(self): self.values = []
    def add_scalar(self, tag, value, global_step):
        assert math.isfinite(float(value))
        self.values.append((tag, float(value), global_step))


def test_real_mgn_synthetic_training_checkpoint_and_resume(tmp_path, monkeypatch):
    import reid.models.mgn as mgn
    torch.manual_seed(42)
    def forbidden(*a, **kw): pytest.fail('CPU synthetic test accessed pretrained artifacts')
    monkeypatch.setattr(mgn, '_read_historical_weights', forbidden)
    monkeypatch.setattr(torch.hub, 'download_url_to_file', forbidden)
    cfg = synthetic_config()
    validate_reid_config(cfg, num_classes=1041)
    model = build_model(cfg, 1041)
    criterion = build_criterion(cfg, 1041, model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=999)
    writer = Recorder()
    labels = torch.tensor([0, 0, 1, 1])
    generator = torch.Generator().manual_seed(80)
    batches = [torch.randn(4, 3, 384, 128, generator=generator) for _ in range(2)]
    completed = []
    losses = []
    shape_records = []

    def exercise_epoch(model, criterion, optimizer, scheduler, images, epoch):
        before = model.backbone.shared.conv1.weight.detach().clone()
        def check_step(opt, args, kwargs):
            for parameter in model.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for group in opt.param_groups:
                assert group['lr'] == .01 and group['weight_decay'] == .0005
                assert group['momentum'] == .9 and not group['nesterov']
        def check_loss(module, inputs, result):
            output, targets = inputs
            value, logs = result
            assert output['emb'].shape == (4, 2048)
            assert [tuple(head.shape) for head in output['logits']] == [(4, 1041)]*8
            assert [tuple(feature.shape) for feature in output['metric_features']] == [(4, 256)]*3
            assert torch.isfinite(value)
            assert logs['loss/total'] == pytest.approx(logs['loss/id']+logs['loss/triplet'], rel=1e-6)
            shape_records.append(dict(embedding=[4, 2048], logits=[8, 4, 1041], metric_features=[3, 4, 256]))
        handles = [optimizer.register_step_pre_hook(check_step), criterion.register_forward_hook(check_loss)]
        try:
            value = train_one_epoch(model, [(images, labels)], criterion, optimizer, None,
                                    torch.device('cpu'), False, 1, scheduler=scheduler, tb_writer=writer, epoch=epoch)
        finally:
            for handle in handles: handle.remove()
        assert scheduler.completed_epochs == epoch-1
        step_completed_epoch(scheduler, epoch)
        assert not torch.equal(before, model.backbone.shared.conv1.weight)
        for parameter in model.parameters():
            assert torch.isfinite(parameter).all()
            assert torch.isfinite(optimizer.state[parameter]['momentum_buffer']).all()
        completed.append(epoch)
        losses.append(value)
        return {name: digest(obj.state_dict()) for name, obj in [('model', model), ('optimizer', optimizer), ('scheduler', scheduler)]}

    first = exercise_epoch(model, criterion, optimizer, scheduler, batches[0], 1)
    selection = dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1)
    path = tmp_path/'synthetic_epoch1.pth'
    save_checkpoint(path, model, optimizer, scheduler, epoch=1, scores={'mAP': .8}, cfg=cfg, selection_state=selection)
    expected = exercise_epoch(model, criterion, optimizer, scheduler, batches[1], 2)
    model.eval()
    with torch.no_grad(): expected_output = model(batches[1][:1])
    del model, criterion, optimizer, scheduler
    gc.collect()

    model = build_model(cfg, 1041)
    criterion = build_criterion(cfg, 1041, None)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=1)
    cfg['train']['save']['resume'] = str(path)
    start, best, payload = train.maybe_resume_training(logging.getLogger('mgn-cpu'), cfg, model, optimizer, scheduler, None, 'cpu')
    assert start == 2 and best == .8 and scheduler.completed_epochs == 1
    actual_first = {name: digest(obj.state_dict()) for name, obj in [('model', model), ('optimizer', optimizer), ('scheduler', scheduler)]}
    assert actual_first == first
    del payload
    actual = exercise_epoch(model, criterion, optimizer, scheduler, batches[1], 2)
    assert actual == expected
    assert losses[1] == losses[2]
    last = tmp_path/'synthetic_epoch2.pth'
    save_checkpoint(last, model, optimizer, scheduler, epoch=2, scores={'mAP': .3}, cfg=cfg, selection_state=selection)
    tags = {tag for tag, _, _ in writer.values}
    assert {'loss/id', 'loss/triplet', 'loss/total', 'acc/id_mean_heads'} <= tags
    assert {f'loss/triplet/feature_{i}' for i in range(3)} <= tags
    assert completed == [1, 2, 2]  # Three actual updates, with the second replayed once.
    del model, criterion, optimizer, scheduler
    gc.collect()
    payload = torch.load(last, map_location='cpu', weights_only=True)
    restored = reconstruct_model(payload, cfg={'model': {'name': 'other', 'num_classes': 9999}}).eval()
    assert restored.num_classes == 1041 and restored.embedding_dim == 2048
    with torch.no_grad(): actual_output = restored(batches[1][:1])
    for field in ['emb', 'logits', 'metric_features']:
        left = actual_output[field] if isinstance(actual_output[field], tuple) else (actual_output[field],)
        right = expected_output[field] if isinstance(expected_output[field], tuple) else (expected_output[field],)
        for value, wanted in zip(left, right): torch.testing.assert_close(value, wanted, rtol=0, atol=0)
    evidence = dict(case='CPU synthetic MGN; not benchmark results', actual_optimizer_updates=3,
        input=[4, 3, 384, 128], synthetic_P=2, synthetic_K=2, source_classifier_count=1041,
        pretrained=False, torch_seed=42, data_generator_seed=80,
        primary_batch64_gpu_admission=False, losses=losses, shapes=shape_records,
        first_state=first, continuous_second_state=expected, resumed_second_state=actual,
        exact_controlled_state_match=True, scalar_tags=sorted(tags),
        checkpoint_sha256={path.name: file_digest(path), last.name: file_digest(last)},
        caveat='Fixed CPU inputs only; sampler/RNG/scaler stochastic replay is not qualified.')
    (tmp_path/'mgn_synthetic_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')


def test_common_entry_point_retains_historical_best_across_lower_last_and_resume(tmp_path, monkeypatch):
    import reid.engine.evaluator as evaluator
    cfg = load_config('configs/baseline_market1501_resnet50_triplet.yaml')
    cfg['system'].update(device='cpu', amp=False)
    cfg['repro'].update(deterministic=False, benchmark=False)
    cfg['logging']['tensorboard'] = False
    cfg['model']['backbone']['pretrained'] = False
    cfg['model']['head']['embedding_dim'] = 3
    cfg['loss']['triplet']['enabled'] = cfg['loss']['center']['enabled'] = False
    cfg['optim'].update(name='sgd', lr=.01, momentum=.9)
    cfg['sched'] = dict(name='epoch_multistep', milestones=[2], gamma=.1)
    cfg['experiment']['output_dir'] = str(tmp_path/'selection_only')
    cfg['train'].update(epochs=2, eval_interval=1)
    cfg['train']['save'].update(save_best=True, save_last=True, metric='mAP', resume='', require_selection_state_on_resume=True)
    class Loader(list): pass
    loader = Loader([None]); loader.dataset = SimpleNamespace(num_classes=3); loader.effective_batch_size=4
    def model(*args, **kwargs):
        result = nn.Linear(2, 3); result.feat_dim=3
        return result
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config='synthetic-only', opts=[]))
    monkeypatch.setattr(train, 'load_config', lambda *a, **k: copy.deepcopy(cfg))
    monkeypatch.setattr(train, 'build_train_loader', lambda c: (loader, 3))
    monkeypatch.setattr(train, 'build_test_loader', lambda c: object())
    monkeypatch.setattr(train, 'build_model', model)
    monkeypatch.setattr(train, 'save_run_artifacts', lambda *a, **k: dict(command='synthetic', environment='synthetic'))
    monkeypatch.setattr(train, 'save_training_plots', lambda **kwargs: None)
    used_rates, best_writes, saved_boundaries = [], [], []
    def epoch(**kwargs):
        used_rates.append(kwargs['optimizer'].param_groups[0]['lr'])
        return .25  # Selection/saving plumbing only; real MGN updates are tested above.
    monkeypatch.setattr(train, 'train_one_epoch', epoch)
    scores = iter([.8, .3, .8, .7, .8])  # First two epochs/final-best, resumed epoch/final-best.
    monkeypatch.setattr(evaluator, 'evaluate_reid', lambda *a, **k: dict(mAP=next(scores), mINP=.2, Rank1=.4, Rank5=.6, Rank10=.7, cmc=[.4,.6,.7]))
    real_save = train.save_checkpoint
    def save(**kwargs):
        assert kwargs['scheduler'].completed_epochs == kwargs['epoch']
        saved_boundaries.append(kwargs['epoch'])
        if Path(kwargs['path']).name == 'ckpt_best.pth': best_writes.append(kwargs['epoch'])
        return real_save(**kwargs)
    monkeypatch.setattr(train, 'save_checkpoint', save)
    previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        for final_epoch in [2, 3]:
            if final_epoch == 3:
                cfg['train']['epochs'] = 3
                cfg['train']['save']['resume'] = str(tmp_path/'selection_only/checkpoints/ckpt_last.pth')
            with monkeypatch.context() as local:
                local.setattr(logging.getLogger('reid.train'), 'handlers', [])
                train.main()
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous_cudnn
    root = tmp_path/'selection_only'
    best = torch.load(root/'checkpoints/ckpt_best.pth', map_location='cpu', weights_only=True)
    last = torch.load(root/'checkpoints/ckpt_last.pth', map_location='cpu', weights_only=True)
    assert best_writes == [1] and best['epoch'] == 1 and last['epoch'] == 3
    assert last['scores']['mAP'] == .7
    assert last['selection_state'] == dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1)
    assert used_rates == pytest.approx([.01, .01, .001])
    final = json.loads((root/'metrics/final_test.json').read_text())
    terminal = json.loads((root/'metrics/final_epoch_test.json').read_text())
    assert (final['epoch'], final['mAP'], final['checkpoint']) == (1, .8, 'ckpt_best.pth')
    assert (terminal['epoch'], terminal['mAP'], terminal['checkpoint']) == (3, .7, 'ckpt_last.pth')
    (tmp_path/'mgn_selection_evidence.json').write_text(json.dumps(dict(
        case='Generic orchestration with injected scores; not model performance',
        historical_best=.8, latest_score=.7, selected_epoch=1, last_epoch=3,
        best_writes=best_writes, optimizer_rates=used_rates,
        saved_epoch_boundaries=saved_boundaries, selected_and_final_state_separate=True), indent=2)+'\n')
