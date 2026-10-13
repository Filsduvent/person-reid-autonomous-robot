"""CPU fixtures use original loaders, real tiny image files and isolated trees."""
import copy
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler
from torchvision.transforms import ToTensor

from test_lodo_config import descriptor, lodo_config
from reid.data import build
from reid.data.lodo_preflight import preflight_lodo
from reid.data.multi_source import VirtualMultiSourceDataset, build_multi_source_dataset, canonical_json
from reid.data.protocol import ReIDTrainSample, validate_train_dataset
from reid.utils.lodo_protocol import DATASET_ORDER, FOLDS


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 8), color=(12, 24, 36)).save(path)


def partition(root, name):
    return root / ('cuhk03/detected' if name == 'cuhk03' else name) / 'partitions.pkl'


def change_partition(root, name, mutate):
    path = partition(root, name)
    value = pickle.loads(path.read_bytes()); mutate(value)
    path.write_bytes(pickle.dumps(value))


@pytest.fixture
def tree(tmp_path):
    for name in DATASET_ORDER:
        if name == 'msmt17':
            base = tmp_path / 'msmt17/MSMT17_V2'
            for role, sub in [('train', 'train'), ('val', 'train'), ('query', 'test'), ('gallery', 'test')]:
                rows = []
                pids = range(8) if role == 'train' else [0]  # shared PID is legitimate
                for pid in pids:
                    for slot in range(4 if role == 'train' else 1):
                        ref = f'{pid:04d}/0000_{role}_1_{slot:04d}.jpg'
                        image(base / f'mask_{sub}_v2' / ref)
                        rows.append(f'{ref} {pid}\n')
                (base / f'list_{role}.txt').write_text(''.join(rows))
        else:
            base = partition(tmp_path, name).parent
            training = [f'{pid:08d}_0001_{slot:08d}.jpg' for pid in range(8) for slot in range(4)]
            test = [f'{100:08d}_0001_{slot:08d}.jpg' for slot in range(2)]
            for ref in training + test:
                image(base / 'images' / ref)
            parts = dict(train_im_names=training[:16], val_im_names=training[16:],
                         trainval_im_names=training, test_im_names=test, test_marks=[0, 1], val_marks=[0] * 16,
                         trainval_ids2labels={p: 7-p for p in range(8)},
                         train_ids2labels={p: 3-p for p in range(4)})
            partition(tmp_path, name).write_bytes(pickle.dumps(parts))
    return tmp_path


def config(root, architecture='resnet50', fold='LODO-1'):
    cfg = lodo_config(architecture, fold)
    cfg['data'].update(root=str(root), num_workers=0, pin_memory=False)
    cfg['system']['device'] = 'cpu'
    return cfg


@pytest.mark.parametrize('fold', list(FOLDS))
def test_four_folds_delegate_and_align_every_index(tree, fold, monkeypatch):
    cfg = config(tree, fold=fold)
    monkeypatch.setattr(build, 'build_test_loader', lambda *a, **k: pytest.fail('target loader accessed'))
    calls = []
    def transform(img):
        calls.append(img.size)
        return ToTensor()(img) + 1
    ds = build_multi_source_dataset(cfg, transform)
    assert calls == []  # construction remains metadata-only
    assert len(ds) == 96 and ds.num_classes == 24
    validate_train_dataset(ds)
    expected = FOLDS[fold][0]
    assert tuple(ds.sources) == expected and ds.target not in ds.sources
    assert set(ds.labels) == set(range(24))
    for index in range(len(ds)):
        record = ds.resolve_index(index); sample = ds[index]
        assert type(sample) is ReIDTrainSample
        assert len(tuple(sample)) == 2
        assert record.global_index == index
        assert record.source_key == expected[index // 32]
        assert record.source_local_index == index % 32
        assert record.source_exposed_pid == (index % 32) // 4
        assert record.camera == (0 if record.source_key == 'msmt17' else 1)
        assert record.image_reference == ds.sources[record.source_key].im_names[index % 32]
        assert Path(record.canonical_path).is_file()
        assert sample.label == ds.labels[index] == record.global_label == (index // 32)*8 + record.source_exposed_pid
        assert sample.image.shape == (3, 8, 4) and sample.image.min() > 1
    assert len(calls) == len(ds)
    assert json.loads(ds.serialize_sidecar())['samples'][33]['global_index'] == 33
    for i in [-1, len(ds), 1000]:
        with pytest.raises(IndexError): ds.resolve_index(i)


def test_mapping_namespace_recovery_determinism_and_local_reordering(tree):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    repeat = build_multi_source_dataset(config(tree), ToTensor())
    assert ds.serialize_mapping() == repeat.serialize_mapping()
    assert ds.mapping_digest == hashlib.sha256(ds.serialize_mapping()).hexdigest() == repeat.mapping_digest
    metadata = ds.mapping_metadata()
    assert metadata['schema_version'] == 1 and metadata['num_classes'] == 24
    pid_zero = [r for r in metadata['identities'] if r['source_exposed_pid'] == 0]
    assert [r['global_label'] for r in pid_zero] == [0, 8, 16]  # no cross-source PID collision
    ds.validate_mapping(ds.serialize_mapping(), ds.mapping_digest)
    mutated = copy.deepcopy(metadata)
    mutated['identities'][0]['global_label'], mutated['identities'][1]['global_label'] = 1, 0
    raw = canonical_json(mutated)
    with pytest.raises(ValueError): ds.validate_mapping(raw, hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError): ds.validate_mapping(ds.serialize_mapping(), '0'*64)
    metadata['identities'].clear()
    assert len(ds.mapping_metadata()['identities']) == 24  # defensive recovery object
    change_partition(tree, 'duke', lambda p: p.update(trainval_ids2labels={p: p for p in range(8)}))
    reordered = build_multi_source_dataset(config(tree), ToTensor())
    assert reordered.labels == ds.labels
    assert [(r['source_key'], r['source_exposed_pid'], r['global_label']) for r in reordered.mapping_metadata()['identities']] == [(r['source_key'], r['source_exposed_pid'], r['global_label']) for r in ds.mapping_metadata()['identities']]
    # Full provenance includes local labels, so this changed local encoding is visible.
    assert reordered.mapping_digest != ds.mapping_digest
    other = build_multi_source_dataset(config(tree, fold='LODO-2'), ToTensor())
    assert other.mapping_digest != reordered.mapping_digest


def test_mapping_identity_excludes_host_paths(tree):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    evidence = copy.deepcopy(ds.preflight); evidence['configured_root'] = '/different/host'
    other = VirtualMultiSourceDataset(ds.sources, fold=ds.fold, target=ds.target, preflight=evidence)
    assert ds.mapping_digest == other.mapping_digest
    assert str(tree).encode() not in ds.serialize_mapping()


@pytest.mark.parametrize('architecture', ['resnet50', 'pcb'])
def test_existing_loader_collator_and_samplers(tree, architecture):
    loader, classes = build.build_train_loader(config(tree, architecture))
    assert classes == loader.dataset.num_classes == 24 and loader.effective_batch_size == 64
    images, labels = next(iter(loader))
    assert images.shape == (64, 3, 384 if architecture == 'pcb' else 256, 128)
    assert labels.dtype == torch.int64 and 0 <= labels.min() <= labels.max() < classes
    if architecture == 'resnet50':
        assert sorted(Counter(labels.tolist()).values()) == [4]*16
        assert loader.batch_sampler.__class__.__name__ == 'PKBatchSampler'
    else:
        assert isinstance(loader.sampler, RandomSampler) and loader.drop_last is True


@pytest.mark.parametrize('mutation', ['version', 'unknown', 'order', 'duplicate', 'missing', 'target', 'target_mismatch', 'train_split', 'msmt_val_list', 'cuhk_new', 'cuhk_labeled', 'cuhk_split'])
def test_invalid_descriptors_fail_before_image_or_loader_access(tree, mutation, monkeypatch):
    cfg = config(tree); sources = cfg['data']['train']['dataset']['sources']
    if mutation == 'version': cfg['protocol']['version'] = 2
    elif mutation == 'unknown': sources[0]['name'] = 'combined'
    elif mutation == 'order': sources.reverse()
    elif mutation == 'duplicate': sources[1] = sources[0]
    elif mutation == 'missing': sources.pop()
    elif mutation == 'target': sources[0] = descriptor('market1501')
    elif mutation == 'target_mismatch': cfg['data']['test']['dataset'] = descriptor('duke', True)
    elif mutation == 'train_split': sources[0]['split'] = 'train'
    elif mutation == 'msmt_val_list': sources[2]['lists'].append('list_val.txt')
    elif mutation == 'cuhk_new': sources[1]['protocol'] = 'new'
    elif mutation == 'cuhk_labeled': sources[1]['image_type'] = 'labeled'
    else: sources[1]['split_id'] = 0
    monkeypatch.setattr(Image, 'open', lambda *a, **k: pytest.fail('image decoded before failure'))
    monkeypatch.setattr(build, '_build_duke_train_dataset', lambda **k: pytest.fail('source loader built before rejection'))
    with pytest.raises(ValueError): build_multi_source_dataset(cfg, ToTensor())


@pytest.mark.parametrize('mutation', ['duplicate', 'constituent_overlap', 'wrong_union', 'source_test', 'traversal', 'absolute'])
def test_partition_integrity(tree, mutation, monkeypatch):
    def mutate(p):
        if mutation == 'duplicate': p['trainval_im_names'].append(p['trainval_im_names'][0])
        elif mutation == 'constituent_overlap': p['val_im_names'][0] = p['train_im_names'][0]
        elif mutation == 'wrong_union': p['trainval_im_names'].pop()
        else:
            name = (p['test_im_names'][0] if mutation == 'source_test' else
                    '../images/' + p['train_im_names'][0] if mutation == 'traversal' else
                    str(partition(tree, 'duke').parent / 'images' / p['train_im_names'][0]))
            p['train_im_names'][0] = p['trainval_im_names'][0] = name
    change_partition(tree, 'duke', mutate)
    monkeypatch.setattr(Image, 'open', lambda *a, **k: pytest.fail('prohibited image read'))
    with pytest.raises(ValueError): build_multi_source_dataset(config(tree), ToTensor())


@pytest.mark.parametrize('mutation', ['symlink_alias', 'escape_symlink', 'target_hardlink', 'cross_source_hardlink', 'missing', 'metadata_symlink', 'root_override', 'root_traversal', 'hash'])
def test_filesystem_and_provenance_fail_closed(tree, tmp_path, mutation):
    cfg = config(tree)
    train = tree / 'duke/images/00000000_0001_00000000.jpg'
    if mutation in ('symlink_alias', 'escape_symlink', 'target_hardlink', 'cross_source_hardlink'):
        destination = (tree / 'duke/images/00000000_0001_00000001.jpg' if mutation == 'symlink_alias' else
                       tree / 'market1501/images/00000100_0001_00000000.jpg' if mutation in ('escape_symlink', 'target_hardlink') else
                       tree / 'cuhk03/detected/images/00000000_0001_00000000.jpg')
        train.unlink()
        if mutation.endswith('hardlink'): os.link(destination, train)
        else: train.symlink_to(destination)
    elif mutation == 'missing': train.unlink()
    elif mutation == 'metadata_symlink':
        path = partition(tree, 'duke'); new = tree/'other.pkl'; path.rename(new); path.symlink_to(new)
    elif mutation == 'root_override':
        cfg['data']['train']['dataset']['sources'][0]['root'] = str(tree/'duke')
    elif mutation == 'root_traversal': cfg['data']['root'] = str(tree/'duke/..')
    else: cfg['data']['train']['dataset']['sources'][0]['metadata_sha256'] = '0'*64
    with pytest.raises((ValueError, FileNotFoundError)): build_multi_source_dataset(cfg, ToTensor())


@pytest.mark.parametrize('mutation', ['val', 'test', 'traversal', 'malformed'])
def test_msmt_train_only_and_raw_reference_checks(tree, mutation):
    base = tree/'msmt17/MSMT17_V2'; train = base/'list_train.txt'
    if mutation == 'val': train.write_text(train.read_text() + (base/'list_val.txt').read_text())
    elif mutation == 'test':
        test = base/'mask_test_v2/0000/0000_query_1_0000.jpg'
        destination = base/'mask_train_v2/0000/0000_train_1_0000.jpg'
        destination.unlink(); os.link(test, destination)
    elif mutation == 'traversal': train.write_text('../mask_train_v2/0000/0000_train_1_0000.jpg 0\n')
    else: train.write_text('malformed row extra-field\n')
    with pytest.raises(ValueError): build_multi_source_dataset(config(tree), ToTensor())


@pytest.mark.parametrize('mutation', ['pid_ambiguous', 'label_collision', 'noncontiguous', 'wrong_classes', 'sidecar_pid', 'sidecar_index', 'source_provenance', 'pid_type'])
def test_mapping_and_sidecar_corruption_rejected(tree, mutation):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    sources = ds.sources; source = sources['duke']; evidence = copy.deepcopy(ds.preflight)
    if mutation == 'pid_ambiguous': source.labels[0] = source.labels[4]
    elif mutation == 'label_collision': source.pids[0] = 7
    elif mutation == 'noncontiguous': source.labels[0] = 24
    elif mutation == 'wrong_classes': source.num_classes = 7
    elif mutation == 'sidecar_pid': evidence['inventories']['duke']['train'][0]['pid'] = 999
    elif mutation == 'sidecar_index': evidence['inventories']['duke']['train'].reverse()
    elif mutation == 'source_provenance': evidence['target'] = 'duke'
    else: source.pids[0] = '0'
    with pytest.raises(ValueError): VirtualMultiSourceDataset(sources, fold=ds.fold, target=ds.target, preflight=evidence)


def test_post_construction_source_changes_rejected(tree, monkeypatch):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    source = ds.sources['duke']; source.labels[0] = 0
    monkeypatch.setattr(Image, 'open', lambda *a, **k: pytest.fail('image read despite changed metadata'))
    with pytest.raises(ValueError): ds[0]


def test_returned_local_label_must_align(tree, monkeypatch):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    source = ds.sources['duke']
    monkeypatch.setattr(type(source), '__getitem__', lambda self, idx: ReIDTrainSample(torch.zeros(3, 8, 4), 999))
    with pytest.raises(ValueError): ds[0]


def test_valid_union_shared_pids_and_metadata_hashes(tree):
    cfg = config(tree)
    for source in cfg['data']['train']['dataset']['sources']:
        metadata = tree/'msmt17/MSMT17_V2/list_train.txt' if source['name'] == 'msmt17' else partition(tree, source['name'])
        source['metadata_sha256'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    evidence = preflight_lodo(cfg)
    assert len(evidence['inventories']['duke']['train']) == 32
    assert len(evidence['inventories']['duke']['val']) == 16
    assert evidence['inventories']['msmt17']['val'][0]['pid'] == evidence['inventories']['msmt17']['train'][0]['pid'] == 0
    assert evidence['source_descriptors'][1]['provenance']['verification'] == 'unverified'
    assert not (tree/'exp').exists()


def test_metadata_mutation_between_preflight_and_loader_rejected(tree, monkeypatch):
    original = build._build_duke_train_dataset
    def mutating_factory(**kwargs):
        ds = original(**kwargs)
        path = partition(tree, 'duke')
        path.write_bytes(path.read_bytes() + b'changed')
        return ds
    monkeypatch.setattr(build, '_build_duke_train_dataset', mutating_factory)
    with pytest.raises(ValueError, match='Metadata changed'):
        build_multi_source_dataset(config(tree), ToTensor())


def test_preflight_hash_corruption_rejected(tree):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    evidence = copy.deepcopy(ds.preflight)
    path = evidence['inventories']['duke']['metadata_path']
    evidence['inventories']['duke']['metadata_hashes'][path] = '0'*64
    with pytest.raises(ValueError, match='Metadata changed'):
        VirtualMultiSourceDataset(ds.sources, fold=ds.fold, target=ds.target, preflight=evidence)


@pytest.mark.parametrize('field,value', [('global_index', 99), ('source_local_index', 99),
                                        ('global_label', 99), ('camera', 99),
                                        ('source_key', 'market1501'), ('image_reference', 'different.jpg')])
def test_recovered_sidecar_must_match_all_sample_metadata(tree, field, value):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    raw = ds.serialize_sidecar()
    ds.validate_sidecar(raw, hashlib.sha256(raw).hexdigest())
    changed = json.loads(raw)
    changed['samples'][0][field] = value
    raw = canonical_json(changed)
    with pytest.raises(ValueError, match='Sidecar'):
        ds.validate_sidecar(raw, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('value', [True, 0.0, '0'])
def test_pid_types_are_not_silently_coerced(tree, value):
    ds = build_multi_source_dataset(config(tree), ToTensor())
    ds.sources['duke'].pids[0] = value
    with pytest.raises(ValueError, match='integer'):
        VirtualMultiSourceDataset(ds.sources, fold=ds.fold, target=ds.target, preflight=ds.preflight)



def test_original_msmt_in_tree_directory_aliases_are_validated(tree):
    base = tree/'msmt17/MSMT17_V2'
    for role, physical in [('train', 'train'), ('test', 'test')]:
        path = base/f'mask_{role}_v2'
        path.rename(base/physical)
        path.symlink_to(physical, target_is_directory=True)
    ds = build_multi_source_dataset(config(tree), ToTensor())
    evidence = ds.preflight['inventories']['msmt17']
    assert evidence['image_dir'] == str(base/'train')
    assert evidence['configured_image_dir'] == str(base/'mask_train_v2')
    assert ds[64].image.shape == (3, 8, 4)
    assert ds.sidecar[64].canonical_path.startswith(str(base/'train'))
    # A directory alias to test is never silently accepted as training.
    (base/'mask_train_v2').unlink()
    (base/'mask_train_v2').symlink_to('test', target_is_directory=True)
    with pytest.raises(ValueError, match='Unauthorized MSMT directory alias'):
        build_multi_source_dataset(config(tree), ToTensor())



def test_msmt_target_descriptor_hash_covers_query_and_gallery(tree):
    cfg = config(tree, fold='LODO-4')
    base = tree/'msmt17/MSMT17_V2'
    payload = dict(schema_version=1, ordered_lists=[
        dict(name=f'list_{role}.txt', sha256=hashlib.sha256((base/f'list_{role}.txt').read_bytes()).hexdigest())
        for role in ('query', 'gallery')])
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    cfg['data']['test']['dataset']['metadata_sha256'] = digest
    evidence = preflight_lodo(cfg)
    assert evidence['inventories']['msmt17']['descriptor_metadata_hashes']['target'] == digest
    cfg['data']['test']['dataset']['metadata_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Metadata SHA256 mismatch'):
        preflight_lodo(cfg)
