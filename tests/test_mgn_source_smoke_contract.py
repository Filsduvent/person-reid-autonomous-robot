"""CPU real-source smoke contracts with synthetic files; no CUDA/benchmarks."""
import copy
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import mgn_gpu_smoke as smoke
from reid.utils.config import load_config,validate_reid_config
from reid.utils.config_schema import validate_config
from reid.data.build import build_train_loader
from reid.data.index_observation import ObservedTrainLoader


@pytest.mark.parametrize('source',smoke.SOURCES)
def test_real_source_contract_retains_parser_and_owns_complete_mgn_recipe(source):
    original=load_config(f'configs/baseline_{source}_resnet50_triplet.yaml')
    cfg=smoke.configuration(source)
    assert cfg['data']['train']['dataset']==original['data']['train']['dataset']
    assert cfg['data']['test']['dataset']==original['data']['test']['dataset']
    validate_config(cfg);validate_reid_config(cfg)
    assert cfg['data']['num_workers']==4 and cfg['data']['pin_memory'] is True
    assert cfg['data']['train']['batch']==dict(sampler='pk',P=16,K=4,batch_size=64)
    for role in ('train','test'):
        assert cfg['data'][role]['images']['interpolation']=='bicubic'
        assert cfg['data'][role]['images']['size']==[384,128]
        assert cfg['data'][role]['aug']['mean']==[.485,.456,.406]
        assert cfg['data'][role]['aug']['std']==[.229,.224,.225]
    assert cfg['data']['train']['aug']['mirror']=='random'
    assert all(cfg['data']['train']['aug'][name]['enabled'] is False for name in ['padding','random_crop','random_erasing'])
    assert cfg['eval']['flip_aggregation']=='mean' and cfg['eval']['normalize_feat'] is True
    assert cfg['eval']['rerank']['enabled'] is False
    assert cfg['optim']==smoke.configuration()['optim']
    assert cfg['sched']==smoke.configuration()['sched']
    assert cfg['model']==smoke.configuration()['model']
    assert cfg['loss']==smoke.configuration()['loss']
    assert cfg['train']['epochs']==120 and cfg['train']['eval_interval']==10
    assert cfg['train']['save']['require_selection_state_on_resume'] is True
    assert cfg['experiment']['output_dir'].startswith('exp/mgn_smoke/')
    assert smoke.case_id(source)==f'phase14_real_{source}_fp32'


def test_source_cases_are_separate_and_unknown_sources_cannot_reset_limits(tmp_path,monkeypatch):
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path)
    uuid='GPU-11111111-2222-3333-4444-555555555555'
    outputs=[smoke.output_path(uuid,1,source) for source in smoke.SOURCES]
    assert len(set(outputs))==4
    outputs[0].mkdir(parents=True)
    with pytest.raises(ValueError,match='reset'):smoke.retry_guard(uuid,1,'code','','market1501')
    assert smoke.retry_guard(uuid,1,'code','','duke') is None
    with pytest.raises(ValueError):smoke.configuration('new-source-to-reset-budget')


@pytest.fixture
def source_fixture(tmp_path):
    root=tmp_path/'data';directory=root/'market1501/images';directory.mkdir(parents=True)
    names=[];mapping={pid:pid-100 for pid in range(100,116)}
    rng=np.random.default_rng(20)
    for pid in mapping:
        for camera in range(1,5):
            name=f'{pid:08d}_{camera:04d}_00000000.jpg';names.append(name)
            Image.fromarray(rng.integers(0,256,(20,10,3),dtype=np.uint8)).save(directory/name)
    with (directory.parent/'partitions.pkl').open('wb') as stream:
        pickle.dump(dict(trainval_im_names=names,trainval_ids2labels=mapping),stream)
    cfg=smoke.configuration('market1501')
    cfg['data'].update(root=str(root),num_workers=0,pin_memory=False)
    loader,classes=build_train_loader(cfg)
    return cfg,loader,classes


def test_common_real_loader_observation_pk_rgb_pixels_and_map_identity(source_fixture):
    cfg,loader,classes=source_fixture
    assert classes==16
    provenance=smoke.source_provenance(loader.dataset,cfg)
    observed=ObservedTrainLoader(loader,sampler_mode='pk')
    iterator=iter(observed)
    images,labels=next(iterator)
    record=smoke.check_source_batch(images,labels,observed.latest_indices,observed.dataset)
    assert record['identities']==16 and record['images_per_identity']==4
    assert record['pixel_oracle'] in ['original','horizontal_flip']
    assert len(record['image_sha256'])==64
    iterator.close()
    assert observed.observation()['consumed_batches']==1
    assert provenance['num_classes']==16 and provenance['images']==64
    assert smoke.source_provenance(loader.dataset,cfg)==provenance
    image=Path(loader.dataset.im_dir)/loader.dataset.im_names[0]
    with image.open('ab') as stream:stream.write(b'changed')
    changed=smoke.source_provenance(loader.dataset,cfg)
    assert changed['file_inventory_sha256']!=provenance['file_inventory_sha256']
    assert changed['identity_map_sha256']==provenance['identity_map_sha256']


def test_bad_real_pk_or_pixels_fail_without_gpu_work(source_fixture):
    cfg,loader,classes=source_fixture
    observed=ObservedTrainLoader(loader,sampler_mode='pk');iterator=iter(observed)
    images,labels=next(iterator);indices=observed.latest_indices
    bad_labels=labels.clone();bad_labels[0]=labels[-1]
    with pytest.raises(ValueError,match='P16/K4'):smoke.check_source_batch(images,bad_labels,indices,observed.dataset)
    bad_images=images.clone();bad_images[0].zero_()
    with pytest.raises(ValueError,match='pixels'):smoke.check_source_batch(bad_images,labels,indices,observed.dataset)
    iterator.close()


@pytest.mark.parametrize('status,digest',[('ERROR','expected'),('PASS','wrong')])
def test_real_case_cannot_bypass_synthetic_admission(tmp_path,monkeypatch,status,digest):
    import json
    from types import SimpleNamespace
    uuid='GPU-11111111-2222-3333-4444-555555555555'
    monkeypatch.setenv('MGN_AUTHORIZED_GPU_UUID',uuid)
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path/'smokes')
    monkeypatch.setattr(smoke,'ROOT',tmp_path)
    monkeypatch.setattr(smoke,'cpu_evidence',lambda:[])
    original=smoke.configuration
    # Isolate the dependency gate; no source paths/cache/GPU inventory are read.
    monkeypatch.setattr(smoke,'configuration',lambda source=None:original() if source is None else {'source':source})
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/mgn_gpu_fp32_qualification.json').write_text(json.dumps(dict(status=status,configuration_sha256=digest)))
    args=SimpleNamespace(source='market1501',gpu_uuid=uuid,attempt=1,correction_note='',authorization_note='manual')
    with pytest.raises(ValueError,match='synthetic FP32'):
        smoke.parent(args)
    assert not (tmp_path/'smokes').exists()


def test_real_case_requires_synthetic_admission_on_same_physical_gpu(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    uuid='GPU-11111111-2222-3333-4444-555555555555'
    monkeypatch.setenv('MGN_AUTHORIZED_GPU_UUID',uuid)
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path/'smokes')
    monkeypatch.setattr(smoke,'ROOT',tmp_path)
    monkeypatch.setattr(smoke,'cpu_evidence',lambda:[])
    original=smoke.configuration
    frozen=original()
    monkeypatch.setattr(smoke,'configuration',lambda source=None:frozen if source is None else {'source':source})
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/mgn_gpu_fp32_qualification.json').write_text(json.dumps(dict(status='PASS',
        gpu_uuid='GPU-other',configuration_sha256=smoke.canonical_hash(frozen))))
    with pytest.raises(ValueError,match='synthetic FP32'):
        smoke.parent(SimpleNamespace(source='market1501',gpu_uuid=uuid,attempt=1,correction_note='',authorization_note='manual'))
