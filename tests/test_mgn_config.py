"""MGN canonical configuration and common-builder contracts."""
import copy

import pytest
import torch

from reid.models.build import build_model
from reid.models.mgn import VARIANT, ARTIFACT_ID, IMAGENET_SHA256, HEAD_INIT_ID, validate_mgn_model_config
from reid.utils.config import validate_reid_config, model_requires_num_classes


def config():
    return {'model': {'name': 'mgn', 'pretrained': False}, 'loss': {
        'id': dict(enabled=True, weight=1., head_aggregation='mean'),
        'triplet': dict(enabled=True, margin=1.2, weight=1., feature_source='model', feature_aggregation='mean'),
        'center': dict(enabled=False)}}


def test_minimal_config_canonicalizes_variant_provenance_and_capabilities():
    cfg = config()
    validate_reid_config(cfg, num_classes=3)
    assert cfg['model']['variant'] == VARIANT
    assert cfg['model']['initialization'] == dict(artifact_id=ARTIFACT_ID, sha256=IMAGENET_SHA256, head_init_id=HEAD_INIT_ID)
    assert cfg['model']['head'] == {'classifier': True}
    assert model_requires_num_classes(cfg)
    before = copy.deepcopy(cfg)
    validate_reid_config(cfg, num_classes=3)
    assert cfg == before


@pytest.mark.parametrize('field,value', [('variant', 'other'), ('pretrained', 1),
    ('weights_path', ''), ('weights_path', '/unused'), ('pooling', 'avg'), ('branches', 2),
    ('embedding_dim', 256), ('head', {'classifier': False}), ('head', {'classifier': 1}),
    ('head', {'classifier': True, 'metric_feat': 'raw'}), ('head', None),
    ('initialization', {'artifact_id': 'V2'}), ('initialization', {'sha256': '0'*64}),
    ('initialization', {'sha256': 'bad'}), ('initialization', {'head_init_id': 'other'}),
    ('initialization', {'unknown': True}), ('initialization', None)])
def test_mgn_config_rejects_ambiguous_or_contradictory_settings(field, value):
    cfg = config()
    cfg['model'][field] = value
    with pytest.raises(ValueError, match='MGN'):
        validate_reid_config(cfg)


@pytest.mark.parametrize('loss,settings', [('center', {'enabled': True}),
    ('triplet', {'enabled': True, 'feature_source': 'legacy'})])
def test_mgn_rejects_unsupported_loss_capabilities(loss, settings):
    cfg = config()
    cfg['loss'][loss].update(settings)
    with pytest.raises(ValueError, match='MGN'):
        validate_reid_config(cfg)


def test_common_builder_offline_bypasses_all_artifact_io(monkeypatch):
    import reid.models.mgn as mgn
    def forbidden(*a, **kw): pytest.fail('offline builder accessed pretrained artifact')
    monkeypatch.setattr(mgn, '_read_historical_weights', forbidden)
    monkeypatch.setattr(mgn, 'Path', forbidden)
    monkeypatch.setattr(torch.hub, 'get_dir', forbidden)
    cfg = config()
    cfg['model'].update(pretrained=True, weights_path='/missing/original/artifact.pth')
    model = build_model(cfg, 3, initialize_pretrained=False)
    assert model.num_classes == 3 and model.embedding_dim == 2048 and model.feat_dim is None
    assert model.checkpoint_metadata == dict(schema_version=1, output_contract_version=1,
        model_name='mgn', variant=VARIANT, num_classes=3, embedding_dim=2048)
    assert cfg['model']['weights_path'] == '/missing/original/artifact.pth'


@pytest.mark.parametrize('classes', [None, 0, -1, True, 2.5])
def test_common_builder_requires_source_identity_count(classes):
    with pytest.raises(ValueError, match='num_classes'):
        build_model(config(), classes)
