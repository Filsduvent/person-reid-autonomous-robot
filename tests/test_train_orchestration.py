import copy
import json
import logging
from pathlib import Path

import pytest
import torch

from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim import build_center_optimizer, build_optimizer, build_scheduler
from reid.utils.checkpoint import save_checkpoint
from reid.utils.logger import setup_logger
from scripts.train import (
    get_checkpoint_paths,
    is_better_score,
    maybe_resume_training,
    save_eval_metrics,
    save_resolved_config,
)
from reid.utils.metrics_artifacts import save_final_epoch_metrics, save_final_eval_metrics


BASE_CFG = {
    "model": {
        "name": "reid_baseline",
        "backbone": {
            "name": "resnet50",
            "pretrained": False,
            "last_conv_stride": 1,
        },
        "head": {
            "embedding_dim": 32,
            "pooling": "gap",
            "bnneck": True,
            "normalize": True,
            "metric_feat": "bn",
            "classifier": True,
        },
    },
    "loss": {
        "triplet": {
            "enabled": True,
            "margin": 0.3,
            "mining": "batch_hard",
            "weight": 1.0,
        },
        "id": {
            "enabled": True,
            "label_smoothing": 0.1,
            "weight": 1.0,
        },
        "center": {
            "enabled": True,
            "weight": 0.0005,
            "lr": 0.5,
        },
    },
    "optim": {
        "name": "adam",
        "lr": 3e-4,
        "weight_decay": 5e-4,
        "bias_lr_factor": 1.0,
        "weight_decay_bias": 5e-4,
        "momentum": 0.9,
    },
    "sched": {
        "name": "warmup_multistep",
        "milestones": [2, 3],
        "gamma": 0.1,
        "warmup_factor": 0.5,
        "warmup_iters": 2,
        "warmup_method": "linear",
    },
    "train": {
        "save": {
            "metric": "mAP",
            "resume": "",
        }
    },
}


def _make_cfg():
    return copy.deepcopy(BASE_CFG)


def test_save_resolved_config_writes_canonical_and_compatibility_files(tmp_path):
    cfg = _make_cfg()

    resolved_path = save_resolved_config(str(tmp_path), cfg)

    assert resolved_path.endswith("config.resolved.yaml")
    assert (tmp_path / "config.resolved.yaml").exists()
    assert (tmp_path / "config.yaml").exists()


def test_get_checkpoint_paths_uses_ckpt_names(tmp_path):
    paths = get_checkpoint_paths(str(tmp_path))

    assert paths["best"].endswith("ckpt_best.pth")
    assert paths["last"].endswith("ckpt_last.pth")


def test_train_logger_writes_train_log(tmp_path):
    logger = setup_logger("test.train.log.artifact", str(tmp_path), filename="train.log")

    logger.info("epoch=1 loss_total=1.0 lr=0.001 speed=10 imgs/s")

    for handler in logger.handlers:
        handler.flush()
    log_text = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert "epoch=1" in log_text
    assert "loss_total=1.0" in log_text
    assert "lr=0.001" in log_text
    assert "speed=10 imgs/s" in log_text


def test_save_eval_metrics_writes_latest_and_epoch_test_files(tmp_path):
    cfg = {
        "model": {
            "name": "reid_baseline",
            "backbone": {
                "name": "resnet50",
            },
        },
        "data": {
            "test": {
                "dataset": {
                    "name": "market1501",
                    "split": "test",
                }
            }
        }
    }
    scores = {
        "mAP": 0.5,
        "mINP": 0.4,
        "Rank1": 0.7,
        "Rank5": 0.8,
        "Rank10": 0.9,
    }

    paths = save_eval_metrics(str(tmp_path), cfg, epoch=3, scores=scores, checkpoint_name="ckpt_last.pth")

    assert paths["latest"].endswith("latest_val.json")
    assert paths["epoch"].endswith("val_epoch_003.json")
    latest = json.loads((tmp_path / "latest_val.json").read_text())
    epoch = json.loads((tmp_path / "val_epoch_003.json").read_text())
    assert latest == epoch
    assert latest["dataset"] == "market1501"
    assert latest["split"] == "test"
    assert latest["model"] == "resnet50_baseline"
    assert latest["epoch"] == 3
    assert latest["checkpoint"] == "ckpt_last.pth"
    assert latest["mAP"] == 0.5


def test_is_better_score_uses_configured_metric():
    assert is_better_score({"mAP": 0.6, "Rank1": 0.8}, "mAP", 0.5) is True
    assert is_better_score({"mAP": 0.4, "Rank1": 0.8}, "mAP", 0.5) is False

    with pytest.raises(KeyError, match="Configured checkpoint metric"):
        is_better_score({"Rank1": 0.8}, "mAP", 0.5)


def test_final_epoch_and_selected_checkpoint_metrics_are_separate(tmp_path):
    cfg = {
        "model": {"name": "reid_baseline", "backbone": {"name": "resnet50"}},
        "data": {"test": {"dataset": {"name": "market1501", "split": "test"}}},
    }
    scores = {"mAP": 0.5, "mINP": 0.4, "Rank1": 0.7, "Rank5": 0.8, "Rank10": 0.9}
    save_final_epoch_metrics(str(tmp_path), cfg, epoch=120, scores=scores)
    save_final_eval_metrics(str(tmp_path), cfg, epoch=110, scores=scores, checkpoint_name="ckpt_best.pth")

    final_epoch = json.loads((tmp_path / "final_epoch_test.json").read_text())
    selected = json.loads((tmp_path / "final_test.json").read_text())
    assert final_epoch["checkpoint"] == "ckpt_last.pth"
    assert final_epoch["epoch"] == 120
    assert selected["checkpoint"] == "ckpt_best.pth"
    assert selected["epoch"] == 110


def test_maybe_resume_training_restores_epoch_metric_and_full_state(tmp_path):
    cfg = _make_cfg()
    logger = logging.getLogger("test_train_resume")
    model = build_model(cfg, num_classes=3)
    criterion = build_criterion(cfg, num_classes=3, feat_dim=model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=2)
    center_optimizer = build_center_optimizer(cfg, criterion)
    ckpt_path = Path(tmp_path) / "checkpoints" / "ckpt_last.pth"

    optimizer.step()
    scheduler.step()
    if center_optimizer is not None:
        center_optimizer.step()

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        center_optimizer=center_optimizer,
        epoch=4,
        scores={"mAP": 0.88},
        cfg=cfg,
    )

    cfg["train"]["save"]["resume"] = str(ckpt_path)

    model2 = build_model(cfg, num_classes=3)
    criterion2 = build_criterion(cfg, num_classes=3, feat_dim=model2.feat_dim)
    optimizer2 = build_optimizer(cfg, model2)
    scheduler2 = build_scheduler(cfg, optimizer2, steps_per_epoch=2)
    center_optimizer2 = build_center_optimizer(cfg, criterion2)

    start_epoch, best_metric, checkpoint = maybe_resume_training(
        logger=logger,
        cfg=cfg,
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        center_optimizer=center_optimizer2,
        device="cpu",
    )

    assert start_epoch == 5
    assert best_metric == pytest.approx(0.88)
    assert checkpoint["epoch"] == 4
    assert optimizer2.state_dict() == optimizer.state_dict()
    assert scheduler2.state_dict() == scheduler.state_dict()
    assert center_optimizer2.state_dict() == center_optimizer.state_dict()


@pytest.mark.parametrize('latest', [None, {'mAP': .3}, {'mAP': .8}])
def test_resume_retains_historical_best_independently_of_latest_scores(tmp_path, latest):
    model = torch.nn.Linear(2, 2)
    path = tmp_path/'history.pth'
    state = dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1)
    save_checkpoint(path, model, epoch=3, scores=latest, selection_state=state)
    cfg = {'train': {'save': {'resume': str(path), 'metric': 'mAP', 'require_selection_state_on_resume': True}}}
    epoch, best, payload = maybe_resume_training(logging.getLogger('history'), cfg, model, None, None, None, 'cpu')
    assert epoch == 4 and best == .8
    assert payload['selection_state'] == state


def test_qualified_resume_requires_explicit_history_and_legacy_fallback_is_preserved(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = tmp_path/'legacy.pth'
    save_checkpoint(path, model, epoch=3, scores={'mAP': .3})
    cfg = {'train': {'save': {'resume': str(path), 'metric': 'mAP', 'require_selection_state_on_resume': True}}}
    with pytest.raises(ValueError, match='selection_state'):
        maybe_resume_training(logging.getLogger('history'), cfg, model, None, None, None, 'cpu')
    cfg['train']['save']['require_selection_state_on_resume'] = False
    assert maybe_resume_training(logging.getLogger('history'), cfg, model, None, None, None, 'cpu')[:2] == (4, .3)


@pytest.mark.parametrize('change', [{'schema_version': True}, {'schema_version': 2}, {'metric': ''},
    {'best_score': float('nan')}, {'best_score': True}, {'best_epoch': 0}, {'best_epoch': 4},
    {'best_epoch': True}, {'best_score': None}, {'unexpected': True}])
def test_selection_state_rejects_ambiguous_history(change):
    from reid.utils.checkpoint import validate_selection_state
    state = dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1)
    state.update(change)
    with pytest.raises(ValueError, match='selection_state'):
        validate_selection_state(state, epoch=3)


def test_selection_state_empty_history_and_metric_identity():
    from reid.utils.checkpoint import validate_selection_state
    state = dict(schema_version=1, metric='mAP', best_score=None, best_epoch=None)
    validate_selection_state(state, epoch=0, metric='mAP')
    with pytest.raises(ValueError, match='metric'):
        validate_selection_state(state, epoch=0, metric='Rank1')
    with pytest.raises(ValueError, match='epoch'):
        validate_selection_state(state, epoch=-1)


def test_corrupt_selection_state_is_rejected_before_model_mutation(tmp_path):
    from reid.utils.checkpoint import load_checkpoint
    source, target = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    payload = save_checkpoint(tmp_path/'valid.pth', source, epoch=3,
        selection_state=dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1))
    payload['model'] = {key: torch.full_like(value, 27) for key, value in payload['model'].items()}
    payload['selection_state']['best_epoch'] = 99
    path = tmp_path/'corrupt.pth'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='selection_state'):
        load_checkpoint(path, target)
    assert all(torch.equal(value, before[key]) for key, value in target.state_dict().items())


def test_resume_history_requirement_flag_is_boolean():
    cfg = {'train': {'save': {'resume': '', 'metric': 'mAP', 'require_selection_state_on_resume': 'false'}}}
    with pytest.raises(ValueError, match='boolean'):
        maybe_resume_training(logging.getLogger('history'), cfg, None, None, None, None, 'cpu')


def test_qualified_resume_requires_preserved_selected_weights(tmp_path):
    from scripts.train import validate_resume_best_checkpoint
    model = torch.nn.Linear(2, 2)
    state = dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1)
    cfg = {'model': {'name': 'synthetic_only'}}
    last = save_checkpoint(tmp_path/'last.pth', model, epoch=3, cfg=cfg,
                           scores={'mAP': .3}, selection_state=state)
    best_path = tmp_path/'best.pth'
    with pytest.raises(ValueError, match='preserved selected-best'):
        validate_resume_best_checkpoint(last, best_path, 'mAP')
    save_checkpoint(best_path, model, epoch=1, cfg=cfg, scores={'mAP': .8}, selection_state=state)
    validate_resume_best_checkpoint(last, best_path, 'mAP')
    empty = dict(last, selection_state=dict(schema_version=1, metric='mAP', best_score=None, best_epoch=None))
    validate_resume_best_checkpoint(empty, tmp_path/'never_selected.pth', 'mAP')


@pytest.mark.parametrize('damage', ['epoch', 'score', 'model', 'metadata', 'shape'])
def test_qualified_resume_rejects_conflicting_selected_checkpoint(tmp_path, damage):
    from scripts.train import validate_resume_best_checkpoint
    model = torch.nn.Linear(2, 2)
    state = dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1)
    cfg = {'model': {'name': 'synthetic_only'}}
    last = save_checkpoint(tmp_path/'last.pth', model, epoch=3, cfg=cfg,
                           scores={'mAP': .3}, selection_state=state)
    path = tmp_path/'best.pth'
    best = save_checkpoint(path, model, epoch=1, cfg=cfg, scores={'mAP': .8}, selection_state=state)
    if damage == 'epoch': best['epoch'] = 2
    elif damage == 'score': best['scores']['mAP'] = .7
    elif damage == 'model': best['cfg'] = {'model': {'name': 'other'}}
    elif damage == 'metadata': best['reconstruction'] = {'num_classes': 999}
    else: best['model']['weight'] = torch.zeros(1, 1)
    torch.save(best, path)
    with pytest.raises(ValueError, match='Selected-best'):
        validate_resume_best_checkpoint(last, path, 'mAP')


@pytest.mark.parametrize('content', [torch.zeros(1), {'scores': None}])
def test_qualified_resume_rejects_malformed_selected_payload(tmp_path, content):
    from scripts.train import validate_resume_best_checkpoint
    path = tmp_path/'bad-best.pth'
    torch.save(content, path)
    checkpoint = {'epoch': 3, 'selection_state': dict(schema_version=1, metric='mAP', best_score=.8, best_epoch=1)}
    with pytest.raises(ValueError, match='Selected-best'):
        validate_resume_best_checkpoint(checkpoint, path, 'mAP')
