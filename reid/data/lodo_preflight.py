"""Read-only lodo_v1 inventory checks. No target dataset or image decoding.

The configured data.root authorizes one dataset tree. Descriptor overrides
must resolve to that tree. The existing MSMT mask_train_v2 -> train and
mask_test_v2 -> test directory aliases are resolved inside the V2 tree;
other nested symlinks and traversal fail closed. Pickles
are trusted project metadata, as in the original loaders. File identities use
canonical paths and (device, inode), not image-content hashes. Revalidate at
launch: this preflight is not a filesystem lock against concurrent mutation.
"""
from pathlib import Path
import hashlib
import json

from reid.data.cuhk03 import parse_processed_reid_name
from reid.data.market1501 import parse_processed_name
from reid.data.msmt17 import parse_msmt17_list
from reid.utils.io import load_pickle
from reid.utils.lodo_protocol import DATASET_ORDER
from reid.utils.lodo_dispatch import scientific_configuration, validate_lodo_config


def _root(value):
    path = Path(value).expanduser()
    if '..' in path.parts:
        raise ValueError('Dataset root traversal is prohibited')
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError('Dataset root must be a directory')
    return path


def _inside(root, relative, *, directory=False):
    reference = Path(relative)
    if reference.is_absolute() or '..' in reference.parts:
        raise ValueError(f'Path traversal/absolute reference: {relative}')
    lexical = root / reference
    canonical = lexical.resolve(strict=True)
    if not canonical.is_relative_to(root) or canonical != lexical:
        raise ValueError(f'Unauthorized path or symlink alias: {lexical}')
    if directory and not canonical.is_dir() or not directory and not canonical.is_file():
        raise ValueError(f'Wrong file type: {canonical}')
    return canonical


def _msmt_image_dir(root, base, subtree):
    lexical = root / base / subtree
    if lexical.is_symlink():
        # Original installed V2 layout: retain the authorized loader selector,
        # while allowing only its corresponding in-tree physical directory.
        physical = 'train' if subtree == 'mask_train_v2' else 'test'
        expected = _inside(root, f'{base}/{physical}', directory=True)
        if lexical.resolve(strict=True) != expected:
            raise ValueError(f'Unauthorized MSMT directory alias: {lexical}')
        return expected
    return _inside(root, f'{base}/{subtree}', directory=True)


def _names(values):
    return [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in values]


def _inventory(image_dir, names, pids, cams):
    rows = []
    for reference, pid, cam in zip(names, pids, cams):
        path = _inside(image_dir, reference)
        stat = path.stat()
        rows.append(dict(image_reference=reference, canonical_path=str(path),
                         file_identity=(stat.st_dev, stat.st_ino), pid=int(pid), camera=int(cam)))
    if len(rows) != len(names) or len(names) != len(pids) or len(names) != len(cams):
        raise ValueError('Inventory metadata lengths differ')
    if any(row['pid'] < 0 for row in rows):
        raise ValueError('Negative PID in authorized inventory')
    return rows


def _tokens(rows):
    return {row['canonical_path'] for row in rows}, {row['file_identity'] for row in rows}


def _disjoint(left, right, reason):
    lp, li = _tokens(left); rp, ri = _tokens(right)
    if lp & rp or li & ri:
        raise ValueError(f'Prohibited image overlap: {reason}')


def _unique(rows, reason):
    paths, inodes = _tokens(rows)
    if len(paths) != len(rows) or len(inodes) != len(rows):
        raise ValueError(f'Duplicate image path/file identity: {reason}')


def _read_inventory(root, name):
    hashes = {}
    if name == 'msmt17':
        base = 'msmt17/MSMT17_V2'
        lists = dict(train=('list_train.txt', 'mask_train_v2'),
                     val=('list_val.txt', 'mask_train_v2'),
                     query=('list_query.txt', 'mask_test_v2'),
                     gallery=('list_gallery.txt', 'mask_test_v2'))
        result = {}
        for role, (filename, subtree) in lists.items():
            metadata = _inside(root, f'{base}/{filename}')
            image_dir = _msmt_image_dir(root, base, subtree)
            raw = metadata.read_bytes(); hashes[str(metadata)] = hashlib.sha256(raw).hexdigest()
            names = [line.split()[0] for line in raw.decode('utf-8').splitlines() if line.strip()]
            # Check original references before the legacy parser joins them.
            for reference in names:
                ref = Path(reference)
                if ref.is_absolute() or '..' in ref.parts:
                    raise ValueError(f'Path traversal/absolute reference: {reference}')
            records = parse_msmt17_list(str(metadata), str(image_dir))
            result[role] = _inventory(image_dir, names, [r[1] for r in records], [r[2] for r in records])
        train, val = result['train'], result['val']
        _unique(train, f'{name} train'); _unique(val, f'{name} val')
        _disjoint(train, val, 'MSMT17 train/val')
        test = result['query'] + result['gallery']
        image_dir = _msmt_image_dir(root, base, 'mask_train_v2')
        metadata = root / base / 'list_train.txt'
    else:
        base = 'cuhk03/detected' if name == 'cuhk03' else name
        metadata = _inside(root, f'{base}/partitions.pkl')
        hashes[str(metadata)] = hashlib.sha256(metadata.read_bytes()).hexdigest()
        parts = load_pickle(str(metadata))
        image_dir = _inside(root, f'{base}/images', directory=True)
        parser = parse_processed_name if name == 'market1501' else parse_processed_reid_name
        result = {}
        for role in ('train', 'val', 'trainval', 'test'):
            names = _names(parts[f'{role}_im_names'])
            ids_cams = [parser(n) for n in names]
            result[role] = _inventory(image_dir, names, [r[0] for r in ids_cams], [r[1] for r in ids_cams])
            _unique(result[role], f'{name} {role}')
        _disjoint(result['train'], result['val'], f'{name} constituent train/val')
        if _tokens(result['trainval'])[0] != _tokens(result['train'] + result['val'])[0]:
            raise ValueError(f'Unauthorized trainval union: {name}')
        train, val, test = result['trainval'], result['val'], result['test']
    _disjoint(train, test, f'{name} source test/query/gallery')
    source_hash = hashes[str(metadata)]
    if name == 'msmt17':
        # A target descriptor covers BOTH authorized evaluation lists. Its
        # digest excludes paths and uses the same canonical UTF-8 JSON policy.
        payload = dict(schema_version=1, ordered_lists=[
            dict(name=f'list_{role}.txt', sha256=hashes[str(root / base / f'list_{role}.txt')])
            for role in ('query', 'gallery')])
        target_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                               ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    else:
        target_hash = source_hash  # partition file contains all original roles
    return dict(train=train, val=val, test=test, metadata_path=str(metadata),
                descriptor_metadata_hashes=dict(source=source_hash, target=target_hash),
                image_dir=str(image_dir), metadata_hashes=hashes,
                configured_image_dir=str(root / base / ('mask_train_v2' if name == 'msmt17' else 'images')))


def preflight_lodo(cfg):
    """Validate all inventories separately; return read-only evidence in memory.

    Metadata for the excluded target is inspected, never a target loader. Every
    referenced file must exist. Shared PIDs across partitions are not overlap.
    """
    validate_lodo_config(cfg)
    scientific = scientific_configuration(cfg)
    root = _root(cfg['data']['root'])
    originals = cfg['data']['train']['dataset']['sources'] + [cfg['data']['test']['dataset']]
    for descriptor in originals:
        if _root(descriptor.get('root', cfg['data']['root'])) != root:
            raise ValueError(f'Unauthorized dataset root override: {descriptor["name"]}')
    inventories = {name: _read_inventory(root, name) for name in DATASET_ORDER}
    for descriptor in originals:
        inventory = inventories[descriptor['name']]
        expected = descriptor.get('metadata_sha256')
        role = 'target' if descriptor['split'] == 'test' else 'source'
        actual = inventory['descriptor_metadata_hashes'][role]
        if expected is not None and expected != actual:
            raise ValueError(f'Metadata SHA256 mismatch: {descriptor["name"]}')
    names = tuple(s['name'] for s in scientific['sources'])
    target = scientific['target']['name']
    combined = [row for name in names for row in inventories[name]['train']]
    _unique(combined, 'combined source training')
    forbidden = inventories[target]['train'] + inventories[target]['val'] + inventories[target]['test']
    forbidden += [row for name in names for row in inventories[name]['test']]
    if 'msmt17' in names:
        forbidden += inventories['msmt17']['val']
    _disjoint(combined, forbidden, 'held-out target/source test/MSMT17 val')
    return dict(schema_version=1, fold=cfg['protocol']['fold'], sources=list(names), target=target,
                configured_root=cfg['data']['root'], canonical_root=str(root),
                source_descriptors=scientific['sources'], target_descriptor=scientific['target'],
                inventories=inventories, image_content_hash_policy='not performed')
