"""CPU regression for diagnostic postflight error ordering; validators unchanged."""
import copy
import json
import pytest
import torch
import lodo_cuda_deterministic_continuation as continuation
from test_lodo_config import lodo_config


@pytest.mark.parametrize('policy',['original','deterministic'])
def test_rejection_checks_budget_namespace_and_frozen_policy_separately(policy):
    cfg=lodo_config();cfg['train']['max_updates']=5
    cfg['experiment']['output_dir']=str(continuation.ROOT/'exp/lodo_smoke/fixture-only/producer')
    if policy=='deterministic':cfg['repro'].update(deterministic=True,benchmark=False)
    before=copy.deepcopy(cfg)
    continuation.checked_rejection(cfg)
    assert cfg==before
    assert not torch.cuda.is_initialized()


def test_official_budget_not_accepted_as_short_fixture():
    cfg=lodo_config();cfg['repro'].update(deterministic=True,benchmark=False)
    with pytest.raises(AssertionError,match='short smoke budget'):
        continuation.checked_rejection(cfg)


def test_successful_policy_validation_cannot_be_silently_accepted(monkeypatch):
    cfg=lodo_config();cfg['train']['max_updates']=5;cfg['repro'].update(deterministic=True,benchmark=False)
    monkeypatch.setattr(continuation.abc.protocol,'scientific_configuration',lambda cfg:{})
    with pytest.raises(AssertionError,match='changed deterministic policy'):
        continuation.checked_rejection(cfg)


def test_unrelated_rejection_is_not_misclassified(monkeypatch):
    cfg=lodo_config();cfg['train']['max_updates']=5;cfg['repro'].update(deterministic=True,benchmark=False)
    def error(cfg):raise ValueError('wrong fold')
    monkeypatch.setattr(continuation.abc.protocol,'scientific_configuration',error)
    with pytest.raises(AssertionError,match='Unexpected'):
        continuation.checked_rejection(cfg)


def test_incomplete_A_rejected_before_checkpoint_access(tmp_path,monkeypatch):
    (tmp_path/'A').mkdir();(tmp_path/'A/outcome.json').write_text(json.dumps({'status':'PASS'}))
    monkeypatch.setattr(continuation.abc,'read_common',lambda *a:pytest.fail('Common accessed'))
    with pytest.raises(ValueError,match='postflight-only'):
        continuation.validate_completed_a(tmp_path)
    assert not torch.cuda.is_initialized()
