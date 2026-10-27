"""Launch admission ordering, effective-config binding and legacy dispatch."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reid.utils.config import load_config
from reid.utils.config_schema import validate_config
from reid.utils.training_preflight import run_training_preflight, validate_training_preflight
from scripts import mgn_preflight as gate
from scripts import train

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('market1501', 'duke', 'cuhk03', 'msmt17')


def preset(source='market1501'):
    return ROOT / f'configs/mgn/{source}.yaml'


class DispatchReached(Exception):
    pass


def guard_side_effects(monkeypatch):
    for name in ('build_model', 'select_device', 'set_seed', 'save_checkpoint',
                 'save_resolved_config', 'save_run_artifacts', 'setup_log_streams',
                 'build_train_loader', 'train_one_epoch', 'run_lodo_training'):
        monkeypatch.setattr(train, name, lambda *a, **k: pytest.fail('Premature training side effect'))


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('resume', [False, True])
def test_common_main_runs_preflight_before_training_dispatch(monkeypatch, capsys, source, resume):
    events = []
    options = ([f'train.save.resume=exp/mgn/{source}/checkpoints/ckpt_last.pth'] if resume else [])
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config=str(preset(source)), opts=options))
    guard_side_effects(monkeypatch)

    def admit(actual, **kwargs):
        assert actual == source and kwargs['resume'] is resume
        assert kwargs['config_path'] == preset(source)
        assert kwargs['configuration'] == load_config(str(preset(source)), overrides=options)
        events.append('admitted')
        return dict(status='READY_TO_RESUME' if resume else 'READY_FRESH', source=source, training_launched=False)

    def dispatch(path):
        assert events == ['admitted']
        assert Path(path) == ROOT / f'exp/mgn/{source}'
        events.append('dispatch')
        raise DispatchReached

    monkeypatch.setattr(gate, 'preflight', admit)
    monkeypatch.setattr(train, 'create_experiment_dirs', dispatch)
    with pytest.raises(DispatchReached):
        train.main()
    assert events == ['admitted', 'dispatch']
    assert json.loads(capsys.readouterr().out)['training_preflight']['source'] == source


def test_failed_preflight_stops_main_before_any_training_effect(monkeypatch):
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config=str(preset()), opts=[]))
    guard_side_effects(monkeypatch)
    monkeypatch.setattr(train, 'create_experiment_dirs', lambda *a: pytest.fail('Output created after refusal'))
    def refuse(*a, **k):
        raise ValueError('GPU/data/hash refusal')
    monkeypatch.setattr(gate, 'preflight', refuse)
    with pytest.raises(ValueError, match='GPU/data/hash refusal'):
        train.main()


@pytest.mark.parametrize('value', [None, False, '', 'mgn_offline_v1', {}, {'name': 'unknown'},
                                  {'name': 'mgn_offline_v1', 'source': 'duke'}])
def test_invalid_explicit_preflight_is_rejected_by_schema(value):
    cfg = load_config(str(preset()))
    cfg['train']['preflight'] = value
    with pytest.raises(ValueError, match='preflight'):
        validate_config(cfg)


def test_missing_config_fails_before_preflight_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config=str(tmp_path/'absent.yaml'), opts=[]))
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: pytest.fail('Missing YAML reached gate'))
    monkeypatch.setattr(train, 'create_experiment_dirs', lambda *a: pytest.fail('Missing YAML created output'))
    with pytest.raises(FileNotFoundError):
        train.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('name', ['configs/baseline_market1501_resnet50_triplet.yaml', 'configs/pcb/market1501.yaml'])
def test_baseline_pcb_no_opt_in_preserves_dispatch(monkeypatch, name):
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config=str(ROOT/name), opts=[]))
    guard_side_effects(monkeypatch)
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: pytest.fail('Legacy config ran MGN preflight'))
    def dispatch(*a):
        raise DispatchReached
    monkeypatch.setattr(train, 'create_experiment_dirs', dispatch)
    with pytest.raises(DispatchReached):
        train.main()


def test_override_cannot_remove_declared_admission(monkeypatch):
    cfg = load_config(str(preset()))
    cfg['train'].pop('preflight')
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: pytest.fail('Changed policy reached gate'))
    with pytest.raises(ValueError, match='must not change'):
        run_training_preflight(cfg, config_path=preset(), overrides=['train={}'])


@pytest.mark.parametrize('damage', ['recipe', 'source', 'config_path', 'resume_other', 'resume_symlink', 'resume_null'])
def test_effective_configuration_refuses_unsafe_launch_changes(tmp_path, damage):
    frozen = load_config(str(preset())); effective = copy.deepcopy(frozen)
    config_path = preset(); resume = False
    if damage == 'recipe': effective['optim']['lr'] = .1
    elif damage == 'source': effective['data']['train']['dataset']['name'] = 'duke'
    elif damage == 'config_path': config_path = tmp_path/'copy.yaml'
    elif damage == 'resume_null': effective['train']['save']['resume'] = None
    elif damage == 'resume_symlink':
        target = tmp_path/'checkpoint.pth'; target.touch()
        link = tmp_path/'ckpt_last.pth'; link.symlink_to(target)
        effective['train']['save']['resume'] = str(link); resume = True
    else:
        effective['train']['save']['resume'] = 'exp/mgn/duke/checkpoints/ckpt_last.pth'; resume = True
    with pytest.raises(ValueError):
        gate.effective_configuration(frozen, effective, config_path, preset(), resume=resume)


@pytest.mark.parametrize('absolute', [True, False])
def test_effective_resume_accepts_only_same_source_last_checkpoint(absolute):
    cfg = load_config(str(preset())); effective = copy.deepcopy(cfg)
    path = Path('exp/mgn/market1501/checkpoints/ckpt_last.pth')
    effective['train']['save']['resume'] = str(ROOT/path if absolute else path)
    before = copy.deepcopy(effective)
    assert gate.effective_configuration(cfg, effective, preset(), preset(), resume=True) is effective
    assert effective == before


@pytest.mark.parametrize('field,value', [('status', 'REFUSED'), ('source', 'duke'), ('training_launched', True)])
def test_non_admission_receipts_fail_closed(monkeypatch, field, value):
    receipt = dict(status='READY_FRESH', source='market1501', training_launched=False)
    receipt[field] = value
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: receipt)
    with pytest.raises(ValueError, match='admission receipt'):
        run_training_preflight(load_config(str(preset())), config_path=preset())


def test_wrong_model_or_missing_source_is_not_inferred_from_filename(monkeypatch):
    cfg = load_config(str(preset())); cfg['model']['name'] = 'pcb'
    with monkeypatch.context() as patch:
        patch.setattr(gate, 'preflight', lambda *a, **k: pytest.fail('Wrong model reached gate'))
        with pytest.raises(ValueError, match='requires model.name'):
            run_training_preflight(cfg, config_path=preset())
    with pytest.raises(ValueError, match='Unknown frozen MGN source'):
        gate.preflight(None)


def test_absence_is_noop_and_does_not_import_adapter_or_touch_environment(monkeypatch):
    import os
    before = dict(os.environ)
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: pytest.fail('Absent opt-in reached adapter'))
    assert validate_training_preflight({'train': {}}) is None
    assert run_training_preflight({'train': {}}, config_path='not-read.yaml') is None
    assert dict(os.environ) == before


def test_revised_presets_preserve_every_previous_scientific_setting():
    for source in SOURCES:
        current = load_config(str(preset(source)))
        previous = load_config(str(ROOT/'docs/mgn_freezes/20261027_phase15'/f'configs/mgn/{source}.yaml'))
        assert current['train'].pop('preflight') == {'name': 'mgn_offline_v1'}
        assert current == previous


def test_historical_lodo_uses_existing_admission_and_dispatch(monkeypatch):
    cfg = load_config(str(ROOT/'configs/lodo/v1/resnet50_lodo-2_seed42.yaml'))
    events = []
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config='historical.yaml', opts=[]))
    monkeypatch.setattr(train, 'load_config', lambda *a, **k: cfg)
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: pytest.fail('LODO ran MGN gate'))
    monkeypatch.setattr('reid.utils.lodo_operational.validate_training_admission', lambda *a, **k: events.append('historical_admission'))
    monkeypatch.setattr(train, 'validate_config', lambda *a: events.append('schema'))
    monkeypatch.setattr(train, 'validate_reid_config', lambda *a: events.append('model_contract'))
    monkeypatch.setattr(train, 'run_lodo_training', lambda *a: events.append('lodo_dispatch'))
    monkeypatch.setattr(train, 'create_experiment_dirs', lambda *a: pytest.fail('LODO used ordinary output path'))
    train.main()
    assert events == ['historical_admission', 'schema', 'model_contract', 'lodo_dispatch']


def test_success_receipt_is_retained_before_training_device_initialization(tmp_path, monkeypatch):
    cfg = load_config(str(preset())); cfg['experiment']['output_dir'] = str(tmp_path/'run')
    receipt = dict(status='READY_FRESH', source='market1501', training_launched=False, manifest_sha256='admitted-freeze')
    monkeypatch.setattr(train, 'parse_args', lambda: SimpleNamespace(config=str(preset()), opts=[]))
    monkeypatch.setattr(train, 'load_config', lambda *a, **k: cfg)
    monkeypatch.setattr(gate, 'preflight', lambda *a, **k: receipt)
    monkeypatch.setattr(train, 'setup_logger', lambda *a, **k: None)
    monkeypatch.setattr(train, 'save_run_artifacts', lambda *a, **k: {})
    def device_boundary(*a, **k):
        assert json.loads((tmp_path/'run/artifacts/training_preflight.json').read_text()) == receipt
        raise DispatchReached
    monkeypatch.setattr(train, 'select_device', device_boundary)
    monkeypatch.setattr(train, 'build_model', lambda *a, **k: pytest.fail('Receipt test initialized a model'))
    with pytest.raises(DispatchReached):
        train.main()
