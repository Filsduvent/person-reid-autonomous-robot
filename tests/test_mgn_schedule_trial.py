"""Isolated Market1501 schedule trial; no real model or training execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from reid.utils.config import load_config, validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils.training_preflight import run_training_preflight
from reid.optim.build import build_optimizer, build_scheduler
from reid.optim.lr_scheduler import step_completed_epoch
from scripts import mgn_preflight as gate, train

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'configs/mgn_experiments/market1501_paper_schedule120.yaml'
PROFILE = 'market1501_paper_schedule120'
OUTPUT = 'exp/mgn/market1501_paper_schedule120'


def trial():
    return load_config(str(PATH))


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = set()
        for key in left.keys() | right.keys():
            path = prefix+'.'+key if prefix else key
            result |= differences(left.get(key), right.get(key), path)
        return result
    return {prefix} if left != right else set()


def test_only_schedule_and_operational_identity_change_from_original():
    cfg = trial(); baseline = load_config(str(ROOT/'configs/mgn/market1501.yaml'))
    assert differences(baseline, cfg) == {
        'sched.milestones', 'experiment.name', 'experiment.output_dir',
        'experiment.notes', 'train.preflight.name',
    }
    assert cfg['sched']['milestones'] == [40, 60]
    assert cfg['train']['epochs'] == 120
    assert cfg['train']['save']['resume'] == '' and cfg['model']['pretrained'] is True
    assert cfg['eval']['rerank']['enabled'] is False
    validate_config(cfg); validate_reid_config(cfg)


def test_manifest_preserves_original_four_contracts_and_admits_only_new_yaml():
    current = json.loads((ROOT/'docs/mgn_experiment_manifest.json').read_text())
    previous = json.loads((ROOT/'docs/mgn_freezes/20261105_before_schedule_trial/mgn_experiment_manifest.json').read_text())
    assert current['sources'] == previous['sources']
    assert set(current['experiments']) == {PROFILE}
    entry = current['experiments'][PROFILE]
    assert entry['resolved_configuration'] == trial()
    assert entry['config_path'] == str(PATH.relative_to(ROOT))
    assert gate.file_hash(PATH) == entry['config_sha256']
    for key in ('training', 'evaluation', 'runtime', 'smoke_image_sha256', 'maximum_reserved_bytes', 'planned_fresh_budget'):
        assert entry[key] == previous['sources']['market1501'][key]


def test_exact_120_epoch_learning_rate_trace_and_resume_state():
    cfg = trial(); opt = build_optimizer(cfg, nn.Linear(3, 2)); sched = build_scheduler(cfg, opt)
    for epoch in range(1, 121):
        expected = .01 if epoch <= 40 else .001 if epoch <= 60 else .0001
        assert all(group['lr'] == pytest.approx(expected) for group in opt.param_groups)
        step_completed_epoch(sched, epoch)
        if epoch in (39, 40, 59, 60, 87):
            restored_opt = build_optimizer(cfg, nn.Linear(3, 2))
            restored = build_scheduler(cfg, restored_opt)
            restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
            restored.load_state_dict(copy.deepcopy(sched.state_dict()))
            restored.validate_completed_epoch(epoch)
            assert restored.get_last_lr() == sched.get_last_lr()
    assert sched.completed_epochs == 120


@pytest.mark.parametrize('resume', [False, True])
def test_common_trainer_selects_trial_profile_and_stops_before_side_effects(monkeypatch, resume):
    options = [f'train.save.resume={OUTPUT}/checkpoints/ckpt_last.pth'] if resume else []
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config=str(PATH), opts=options))
    events = []
    def admit(source, **kwargs):
        assert source == 'market1501' and kwargs['profile'] == PROFILE
        assert kwargs['resume'] is resume
        assert kwargs['configuration'] == load_config(str(PATH), overrides=options)
        assert kwargs['config_path'] == PATH
        events.append('admission')
        return dict(source=source, status='READY_TO_RESUME' if resume else 'READY_FRESH',
                    experiment_profile=PROFILE, training_launched=False)
    monkeypatch.setattr(gate, 'preflight', admit)
    class Stop(Exception): pass
    def dispatch(path):
        assert events == ['admission'] and Path(path) == ROOT/OUTPUT
        raise Stop
    monkeypatch.setattr(train, 'create_experiment_dirs', dispatch)
    for name in ('build_model', 'select_device', 'set_seed', 'save_checkpoint', 'train_one_epoch'):
        monkeypatch.setattr(train, name, lambda *a, **k: pytest.fail('Premature side effect'))
    with pytest.raises(Stop): train.main()


def test_trial_does_not_accept_a_baseline_admission_receipt(monkeypatch):
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: dict(source='market1501', status='READY_FRESH', training_launched=False))
    with pytest.raises(ValueError, match='different experiment profile'):
        run_training_preflight(trial(), config_path=PATH)


def test_trial_cannot_overwrite_original_or_reuse_trial_output(tmp_path):
    original = tmp_path/'exp/mgn/market1501'; original.mkdir(parents=True)
    marker = original/'preserved'; marker.write_bytes(b'original results')
    cfg = trial()
    run = gate.output_guard('market1501', cfg, root=tmp_path, expected_output=OUTPUT)
    assert not run.exists() and marker.read_bytes() == b'original results'
    wrong = copy.deepcopy(cfg); wrong['experiment']['output_dir'] = 'exp/mgn/market1501'
    with pytest.raises(ValueError, match='isolated'):
        gate.output_guard('market1501', wrong, root=tmp_path, expected_output=OUTPUT)
    run.mkdir()
    with pytest.raises(ValueError, match='existing output'):
        gate.output_guard('market1501', cfg, root=tmp_path, expected_output=OUTPUT)
    run.rmdir(); run.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        gate.output_guard('market1501', cfg, root=tmp_path, expected_output=OUTPUT)
    assert marker.read_bytes() == b'original results'


def test_resume_is_bound_to_trial_last_checkpoint_not_original_weights():
    frozen = trial(); cfg = copy.deepcopy(frozen)
    cfg['train']['save']['resume'] = f'{OUTPUT}/checkpoints/ckpt_last.pth'
    assert gate.effective_configuration(frozen, cfg, PATH, PATH, resume=True) is cfg
    cfg['train']['save']['resume'] = 'exp/mgn/market1501/checkpoints/ckpt_last.pth'
    with pytest.raises(ValueError, match='original source ckpt_last'):
        gate.effective_configuration(frozen, cfg, PATH, PATH, resume=True)


@pytest.mark.parametrize('source', ['duke', 'cuhk03', 'msmt17'])
def test_trial_refuses_other_sources_before_manifest_or_gpu(source, monkeypatch):
    monkeypatch.setattr(gate, 'MANIFEST', SimpleNamespace(read_text=lambda: pytest.fail('Manifest reached')))
    with pytest.raises(ValueError, match='profile/source mismatch'):
        gate.preflight(source, profile=PROFILE)


@pytest.mark.parametrize('damage', ['lr', 'milestones', 'rerank', 'epochs', 'output'])
def test_unapproved_trial_changes_fail_before_output_and_device(damage):
    frozen = trial(); cfg = copy.deepcopy(frozen)
    if damage == 'lr': cfg['optim']['lr'] = .1
    elif damage == 'milestones': cfg['sched']['milestones'] = [30, 60]
    elif damage == 'rerank': cfg['eval']['rerank']['enabled'] = True
    elif damage == 'epochs': cfg['train']['epochs'] = 80
    else: cfg['experiment']['output_dir'] = 'exp/mgn/market1501'
    with pytest.raises(ValueError, match='differs from frozen'):
        gate.effective_configuration(frozen, cfg, PATH, PATH, resume=False)


def test_trial_profile_cannot_be_removed_by_cli_overrides():
    cfg = trial(); cfg['train'].pop('preflight')
    with pytest.raises(ValueError, match='must not change'):
        run_training_preflight(cfg, config_path=PATH, overrides=['train={}'])


def test_optional_standalone_trial_check_cannot_launch_training(monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, 'argv', ['mgn_preflight.py', '--source', 'market1501', '--profile', PROFILE])
    def admit(source, **kwargs):
        assert kwargs == dict(resume=False, profile=PROFILE)
        return dict(status='READY_FRESH', source=source, experiment_profile=PROFILE, training_launched=False)
    monkeypatch.setattr(gate, 'preflight', admit)
    assert gate.main() == 0
    assert json.loads(capsys.readouterr().out)['training_launched'] is False
