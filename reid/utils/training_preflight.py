"""Explicit, fail-closed launch admission for the common trainer.

Absence of train.preflight preserves ordinary training. Profiles are named
adapters, not executable paths supplied by configuration.
"""
from pathlib import Path


def validate_training_preflight(cfg):
    if not isinstance(cfg, dict) or not isinstance(cfg.get('train', {}), dict):
        raise ValueError('Training configuration and train section must be mappings.')
    train = cfg.get('train', {})
    if 'preflight' not in train:
        return None
    spec = train['preflight']
    if not isinstance(spec, dict) or set(spec) != {'name'}:
        raise ValueError('train.preflight must be a mapping containing only name.')
    if spec['name'] != 'mgn_offline_v1':
        raise ValueError('Unknown train.preflight.name; supported: mgn_offline_v1.')
    return spec['name']


def run_training_preflight(cfg, *, config_path, overrides=()):
    """Return an admission receipt, or None for configs without an opt-in.

    Overrides may not remove/change the YAML's admission policy. The adapter
    validates the entire effective configuration, including controlled resume.
    No device is selected and no training output is created here.
    """
    name = validate_training_preflight(cfg)
    if overrides:
        from reid.utils.config import load_config
        declared = load_config(config_path)
        if validate_training_preflight(declared) != name:
            raise ValueError('Overrides must not change the declared training preflight.')
    if name is None:
        return None
    from scripts.mgn_preflight import preflight
    if cfg.get('model', {}).get('name') != 'mgn':
        raise ValueError('mgn_offline_v1 preflight requires model.name=mgn.')
    source = cfg.get('data', {}).get('train', {}).get('dataset', {}).get('name')
    resume_path = cfg.get('train', {}).get('save', {}).get('resume', '')
    if not isinstance(resume_path, str):
        raise ValueError('Training resume must be a checkpoint path string.')
    receipt = preflight(source, resume=bool(resume_path.strip()),
                        configuration=cfg, config_path=Path(config_path))
    expected = 'READY_TO_RESUME' if resume_path.strip() else 'READY_FRESH'
    if (receipt.get('status') != expected or receipt.get('source') != source
            or receipt.get('training_launched') is not False):
        raise ValueError('Training preflight did not return the required admission receipt.')
    return receipt
