"""Virtual lodo_v1 training datasets with versioned identity and index metadata.

Source-exposed integer PIDs are preserved before local relabeling. Processed
PIDs need not be official raw PIDs. Local cameras retain loader semantics
(including existing MSMT camera-base normalization). No images are copied.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from operator import index as integer_index
from pathlib import Path

from torch.utils.data import Dataset
from reid.data.protocol import ReIDTrainSample, normalize_train_sample, validate_train_dataset
from reid.utils.lodo_protocol import validate_fold


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _integer(value):
    # Actual loaders expose Python/NumPy integers. Do not collapse string,
    # floating-point or bool identifiers into the same numeric namespace.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'PID/label must be an integer, got {value!r}')
    result = int(value)
    if result < 0:
        raise ValueError('PID/label must be nonnegative')
    return result


@dataclass(frozen=True)
class SourceIndex:
    global_index: int
    source_key: str
    source_local_index: int
    source_exposed_pid: int
    local_label: int
    global_label: int
    camera: int | None
    image_reference: str
    canonical_path: str


class VirtualMultiSourceDataset(Dataset):
    """Concatenate authorized source index spaces and replace only labels.

    Construct through build_multi_source_dataset for filesystem preflight.
    This lower-level adapter also validates every PID/local-label association.
    """
    def __init__(self, sources, *, fold, target, preflight):
        validate_fold(fold, tuple(sources), target)
        if (preflight['fold'] != fold or preflight['target'] != target
                or tuple(preflight['sources']) != tuple(sources)):
            raise ValueError('Inconsistent preflight provenance')
        for inventory in preflight['inventories'].values():
            for path, digest in inventory['metadata_hashes'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Metadata changed during construction: {path}')
        self.sources = dict(sources)
        self.fold, self.target = fold, target
        self.root = preflight['configured_root']; self.split = 'virtual_train'
        self.preflight = preflight
        rows, sidecar, statistics = [], [], {}
        self.local_to_global = {}
        for name, source in self.sources.items():
            validate_train_dataset(source)
            inventory = preflight['inventories'][name]['train']
            for attribute in ('pids', 'cams', 'im_names'):
                if len(getattr(source, attribute)) != len(source):
                    raise ValueError(f'Unaligned {name}.{attribute}')
            if len(inventory) != len(source):
                raise ValueError(f'Loader/inventory length differs: {name}')
            pid_to_local, local_to_pid = {}, {}
            for i, (pid, local) in enumerate(zip(source.pids, source.labels)):
                pid, local = _integer(pid), _integer(local)
                if (pid_to_local.get(pid, local) != local or local_to_pid.get(local, pid) != pid):
                    raise ValueError(f'Ambiguous PID/local label association: {name}')
                pid_to_local[pid] = local; local_to_pid[local] = pid
                record = inventory[i]
                if (pid != record['pid'] or int(source.cams[i]) != record['camera']
                        or str(source.im_names[i]) != record['image_reference']):
                    raise ValueError(f'Loader/inventory metadata differs: {name}:{i}')
                if Path(source.im_dir).resolve() != Path(preflight['inventories'][name]['image_dir']):
                    raise ValueError(f'Unauthorized loader image directory: {name}')
            if len(pid_to_local) != source.num_classes:
                raise ValueError(f'Incorrect source class count: {name}')
            offset = len(rows)
            mapping = {pid_to_local[pid]: offset + j for j, pid in enumerate(sorted(pid_to_local))}
            self.local_to_global[name] = mapping
            rows.extend(dict(source_key=name, source_exposed_pid=pid,
                             local_label=pid_to_local[pid], global_label=mapping[pid_to_local[pid]])
                        for pid in sorted(pid_to_local))
            for i, record in enumerate(inventory):
                local = int(source.labels[i])
                sidecar.append(SourceIndex(len(sidecar), name, i, int(source.pids[i]), local,
                                           mapping[local], record['camera'], record['image_reference'],
                                           record['canonical_path']))
            statistics[name] = dict(images=len(source), identities=source.num_classes)
        self.sidecar = tuple(sidecar)
        self.labels = tuple(row.global_label for row in sidecar)
        self.num_classes = len(rows)
        self.source_statistics = statistics
        self._mapping_bytes = canonical_json(dict(schema_version=1, pid_semantics='source_exposed_integer',
                                                  ordered_sources=list(sources), num_classes=self.num_classes,
                                                  source_class_counts={n: s['identities'] for n, s in statistics.items()},
                                                  identities=rows))
        self.mapping_digest = hashlib.sha256(self._mapping_bytes).hexdigest()
        validate_train_dataset(self)
        if self.num_classes != sum(source.num_classes for source in sources.values()):
            raise ValueError('Incorrect combined class count')

    def __len__(self):
        return len(self.sidecar)

    def resolve_index(self, idx):
        idx = integer_index(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.sidecar[idx]

    def __getitem__(self, idx):
        record = self.resolve_index(idx)
        source = self.sources[record.source_key]
        i = record.source_local_index
        if (int(source.labels[i]) != record.local_label or int(source.pids[i]) != record.source_exposed_pid
                or int(source.cams[i]) != record.camera or str(source.im_names[i]) != record.image_reference):
            raise ValueError('Source metadata changed after validation')
        sample = normalize_train_sample(source[i])
        if sample.label != record.local_label:
            raise ValueError('Returned source label does not match sidecar')
        return ReIDTrainSample(image=sample.image, label=record.global_label)

    def serialize_mapping(self):
        """Recoverable canonical JSON bytes, without host paths or timestamps.

        Global assignments survive local-label reorderings. A different local
        encoding changes this full-provenance digest because it is recorded.
        """
        return self._mapping_bytes

    def mapping_metadata(self):
        return json.loads(self._mapping_bytes)

    def serialize_sidecar(self):
        return canonical_json(dict(schema_version=1, mapping_digest=self.mapping_digest,
                                   samples=[asdict(row) for row in self.sidecar]))

    def validate_sidecar(self, payload, digest):
        """Validate recovered index metadata against the current adapter."""
        expected = self.serialize_sidecar()
        if canonical_json(json.loads(payload)) != expected or hashlib.sha256(expected).hexdigest() != digest:
            raise ValueError('Sidecar index/provenance SHA256 mismatch')

    def validate_mapping(self, payload, digest):
        """Reject restored/swapped maps, including equal-C different assignments."""
        if canonical_json(json.loads(payload)) != self._mapping_bytes or digest != self.mapping_digest:
            raise ValueError('Identity mapping/provenance SHA256 mismatch')


def build_multi_source_dataset(cfg, transform):
    from reid.data import build
    from reid.data.lodo_preflight import preflight_lodo
    evidence = preflight_lodo(cfg)  # no image reads, target dataset or output writes
    sources = {}
    for descriptor in cfg['data']['train']['dataset']['sources']:
        name = descriptor['name']; root = descriptor.get('root', cfg['data']['root'])
        kwargs = dict(root=root, split=descriptor['split'], dataset_format=descriptor['format'], transform=transform)
        factory = getattr(build, f'_build_{name}_train_dataset')
        if name == 'cuhk03':
            kwargs['dataset_cfg'] = descriptor
        sources[name] = factory(**kwargs)
    return VirtualMultiSourceDataset(sources, fold=evidence['fold'], target=evidence['target'], preflight=evidence)
