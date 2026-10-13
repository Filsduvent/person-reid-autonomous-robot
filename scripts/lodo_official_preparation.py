"""CPU/read-only verification of Phase 10 candidate manifests; never launches jobs.

Manifest v1 hashes canonical UTF-8 JSON (sorted keys, compact separators, no NaN).
The immutable proposal identifies science, inputs and an UNFROZEN source snapshot;
operational timestamps, allocation and resume paths are separate. A valid proposal
is neither a frozen Git commit nor authorization to execute training/evaluation.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid.data.multi_source import canonical_json
from reid.utils.config import load_config, validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils.lodo_protocol import configuration_identity, scientific_configuration
from reid.utils.lodo_checkpoint import file_digest, digest, loader_contract
from reid.utils.lodo_recovery import recovery_policy, CUDA_BOUNDARY_CAPABILITY


def seal(immutable, operational):
    return dict(schema_version=1, record_type='lodo_official_candidate',
                immutable=immutable, immutable_sha256=digest(immutable),
                operational=operational)


def validate_envelope(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
            'schema_version', 'record_type', 'immutable', 'immutable_sha256', 'operational'}:
        raise ValueError('Unknown/missing manifest fields')
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or manifest['record_type'] != 'lodo_official_candidate':
        raise ValueError('Unsupported manifest schema')
    if digest(manifest['immutable']) != manifest['immutable_sha256']:
        raise ValueError('Immutable candidate digest mismatch')
    if set(manifest['immutable']) != {'configuration','code','dataset','pretrained','recovery','mapping_file','execution_candidate','evaluation'}:
        raise ValueError('Unknown/missing immutable provenance fields')
    code = manifest['immutable']['code']
    if code['state'] != 'dirty_worktree_unfrozen' or code['frozen_commit'] is not None:
        raise ValueError('Phase 10 candidate must not claim an unapproved frozen commit')
    if manifest['operational']['approval_status'] != 'NOT_APPROVED' or manifest['operational']['allocation_status'] != 'NOT_ALLOCATED':
        raise ValueError('This candidate verifier cannot confer launch/allocation approval')


def verify_code(snapshot, root=ROOT, frozen_commit=None):
    if set(snapshot) != {'base_head', 'state', 'frozen_commit', 'source_sha256', 'source_snapshot_sha256'}:
        raise ValueError('Unknown code provenance fields')
    if digest(snapshot['source_sha256']) != snapshot['source_snapshot_sha256']:
        raise ValueError('Code snapshot digest mismatch')
    import subprocess
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if frozen_commit is not None:
        import re
        if not re.fullmatch(r'[0-9a-f]{40}', frozen_commit) or actual != frozen_commit:
            raise ValueError('Approved frozen commit must be the actual full HEAD')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True):
            raise ValueError('Frozen launch verification requires a clean checkout')
    if frozen_commit is None and actual != snapshot['base_head']:
        raise ValueError('Candidate base HEAD changed; re-review, do not silently refresh')
    for relative, expected in snapshot['source_sha256'].items():
        path = root / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or path.resolve() != path:
            raise ValueError('Code snapshot alias/traversal')
        if file_digest(path) != expected:
            raise ValueError(f'Code snapshot changed: {relative}')
    # Detect additions as well as changed/deleted files. Candidate YAMLs have
    # individual identities, and are excluded from this implementation snapshot.
    current = {str(p.relative_to(root)) for base in ('reid', 'scripts', 'tests', 'configs')
               for p in (root/base).rglob('*') if p.is_file() and not p.is_symlink()
               and p.suffix in ('.py', '.yaml', '.yml')
               and not p.is_relative_to(root/'configs/lodo')}
    if current != set(snapshot['source_sha256']):
        raise ValueError('Code snapshot membership changed')


@contextmanager
def metadata_only():
    """Throw before image decoding, CUDA initialization or target construction."""
    import torch
    import reid.data.build as build
    from PIL import Image
    if torch.cuda.is_initialized():
        raise ValueError('Preflight must start without CUDA context')
    def forbidden(*args, **kwargs):
        raise AssertionError('Image/CUDA/target execution is prohibited during preparation')
    with ExitStack() as stack:
        for obj, name in [(Image, 'open'), (torch.cuda, 'init'), (torch.cuda, '_lazy_init'),
                          (build, 'build_test_loader')]:
            stack.enter_context(patch.object(obj, name, forbidden))
        for name in ('market1501', 'duke', 'cuhk03', 'msmt17'):
            stack.enter_context(patch.object(build, f'_build_{name}_test_dataset', forbidden))
        yield
    if torch.cuda.is_initialized():
        raise AssertionError('Preparation initialized CUDA')


def dataset_evidence(cfg):
    from collections import Counter
    from reid.data.build import build_train_loader
    with metadata_only():
        validate_config(cfg)
        validate_reid_config(cfg)
        loader, classes = build_train_loader(cfg)  # construction only; never iterated
        contract = loader_contract(loader)
        ds = loader.dataset
        labels = Counter(ds.labels)
        if set(labels) != set(range(classes)):
            raise ValueError('Global class range differs from source labels')
        mapping = ds.mapping_metadata()
        ds.validate_mapping(ds.serialize_mapping(), ds.mapping_digest)
        sidecar = ds.serialize_sidecar()
        sidecar_hash = hashlib.sha256(sidecar).hexdigest()
        ds.validate_sidecar(sidecar, sidecar_hash)
        for i, record in enumerate(ds.sidecar):
            if (record.global_index != i or record.global_label != ds.labels[i] or
                    ds.sources[record.source_key].labels[record.source_local_index] != record.local_label):
                raise ValueError('Sidecar/source index misalignment')
        provenance = {key: dict(metadata_hashes=v['metadata_hashes'],
                               descriptor_metadata_hashes=v['descriptor_metadata_hashes'])
                      for key,v in ds.preflight['inventories'].items()}
        initial_batches = len(loader)
        # Each identity supplies max(1,floor(n/K)) chunks. Every PK batch uses P
        # chunks; residual identities can only REDUCE this conservative bound.
        maximum = sum(max(1, n//4) for n in labels.values())//16 if cfg['protocol']['architecture']=='resnet50' else len(ds)//64
        result = dict(images=len(ds), num_classes=classes, global_label_range=[0, classes-1],
                      source_statistics=ds.source_statistics, map_digest=ds.mapping_digest,
                      sidecar_digest=sidecar_hash, data_digest=digest(provenance), provenance=provenance,
                      canonical_root=ds.preflight['canonical_root'], initial_epoch_batches=initial_batches,
                      maximum_epoch_updates=maximum, public_sample='ReIDTrainSample(image,label)',
                      loader_contract=contract, target_exclusion='PASS', image_content_hash_policy='not performed')
        return result, mapping


def verify_manifest(path, *, check_code=True, frozen_commit=None, resume=None):
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    validate_envelope(manifest)
    imm = manifest['immutable']
    config_path = ROOT / imm['configuration']['path']
    if config_path.resolve() != config_path or not config_path.is_relative_to(ROOT/'configs/lodo/v1'):
        raise ValueError('Unauthorized candidate configuration path')
    if file_digest(config_path) != imm['configuration']['file_sha256']:
        raise ValueError('Configuration file SHA256 mismatch')
    cfg = load_config(str(config_path))
    if resume is not None:
        cfg['train']['save']['resume'] = str(resume)
    validate_config(cfg); validate_reid_config(cfg)
    if (configuration_identity(cfg) != imm['configuration']['scientific_sha256'] or
            scientific_configuration(cfg) != imm['configuration']['scientific']):
        raise ValueError('Scientific configuration differs from manifest')
    if (resume is None and cfg['train']['save']['resume']) or cfg['system']['device'] != 'cuda' or cfg['data']['num_workers'] != 0:
        raise ValueError('Official candidate requires explicit fresh/resume CUDA, zero workers')
    if cfg['repro'].get('cuda_policy') != 'strict_cuda_deterministic_v1':
        raise ValueError('Official candidate strict policy missing')
    if recovery_policy(cfg) != imm['recovery']['checkpoint_policy'] or imm['recovery']['capability'] != CUDA_BOUNDARY_CAPABILITY:
        raise ValueError('Recovery proposal mismatch')
    if imm['recovery']['checkpoint_policy'] != dict(version=1,policy='completed_epoch_boundary',every_epochs=1,keep_generations=2):
        raise ValueError('Unreviewed cadence/retention')
    if str((ROOT/cfg['experiment']['output_dir']).resolve()) != manifest['operational']['proposed_output_dir']:
        raise ValueError('Output directory mismatch')
    verify_runtime(imm['execution_candidate'])
    request_path=ROOT/imm['evaluation']['request']
    if request_path.resolve()!=request_path or not request_path.is_relative_to(ROOT/'docs/lodo_phase10_evidence/evaluation_requests'):
        raise ValueError('Unauthorized evaluation request path')
    if file_digest(request_path)!=imm['evaluation']['request_sha256']:
        raise ValueError('Evaluation request file SHA256 mismatch')
    from reid.utils.lodo_evaluation import validate_request, evaluation_output
    request=load_config(str(request_path));validate_request(request);evaluation_output(request)
    expected=dict(config_digest=configuration_identity(cfg),map_digest=imm['dataset']['map_digest'],data_digest=imm['dataset']['data_digest'],num_classes=imm['dataset']['num_classes'])
    if (request['expected']!=expected or request['architecture']!=cfg['protocol']['architecture'] or request['fold']!=cfg['protocol']['fold'] or request['target']!=cfg['data']['test']['dataset']['name'] or request['synthetic'] is not False or request['checkpoint']!=str(ROOT/cfg['experiment']['output_dir']/'checkpoints/ckpt_terminal.pth')):
        raise ValueError('Evaluation descriptor/expected identity mismatch')
    if check_code:
        verify_code(imm['code'], frozen_commit=frozen_commit)
    pretrained=imm['pretrained']
    import torch
    from torchvision.models import ResNet50_Weights
    from reid.models.pcb import HISTORICAL_IMAGENET_SHA256
    arch=cfg['protocol']['architecture']
    expected_name='resnet50-11ad3fa6.pth' if arch=='resnet50' else 'resnet50-19c8e357.pth'
    expected_path=Path(torch.hub.get_dir())/'checkpoints'/expected_name
    if pretrained['path'] != str(expected_path.resolve()) or (arch=='pcb' and pretrained['sha256'] != HISTORICAL_IMAGENET_SHA256):
        raise ValueError('Pretrained loader selection differs')
    if arch=='resnet50' and Path(ResNet50_Weights.IMAGENET1K_V2.url).name != expected_name:
        raise ValueError('Torchvision pretrained selection changed')
    expected_sha=('11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca' if arch=='resnet50' else HISTORICAL_IMAGENET_SHA256)
    if pretrained['sha256'] != expected_sha:
        raise ValueError('Pretrained provenance differs from qualified artifact')
    if file_digest(expected_path) != pretrained['sha256']:
        raise ValueError('Pretrained SHA256 mismatch; no download/fallback')
    observed, mapping=dataset_evidence(cfg)
    if observed != imm['dataset']:
        raise ValueError('Live dataset/map/sidecar/provenance differs from proposal')
    mapping_path=ROOT/imm['mapping_file']
    if mapping_path.resolve()!=mapping_path or not mapping_path.is_relative_to(ROOT/'docs/lodo_phase10_evidence/maps'):
        raise ValueError('Unauthorized mapping path')
    if mapping_path.read_bytes()!=canonical_json(mapping):
        raise ValueError('Recoverable mapping file differs')
    return dict(manifest=str(path.relative_to(ROOT)), status='PASS_CANDIDATE_ONLY',
                immutable_sha256=manifest['immutable_sha256'], scientific_sha256=configuration_identity(cfg),
                images=observed['images'],num_classes=observed['num_classes'],map_digest=observed['map_digest'],
                initial_epoch_batches=observed['initial_epoch_batches'],maximum_epoch_updates=observed['maximum_epoch_updates'],
                cuda_initialized=False,training_started=False,evaluation_started=False)


def verify_runtime(execution):
    import platform
    import subprocess
    import torch
    from importlib.metadata import version
    expected=execution['qualified_runtime']
    if execution['cuda_policy']!='strict_cuda_deterministic_v1' or execution['workspace']!=':4096:8':
        raise ValueError('Unsupported official policy proposal')
    current=dict(host=platform.node(),python=platform.python_version(),torch=str(torch.__version__),
                 torchvision=version('torchvision'),numpy=version('numpy'),cuda_build=torch.version.cuda,
                 cudnn_version=torch.backends.cudnn.version())
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[line.split(', ') for line in inventory.strip().splitlines()]
    selected=[row for row in rows if row[0]==expected['uuid']]
    if len(selected)!=1 or expected['uuid']!='GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c':
        raise ValueError('Candidate device differs from qualified TITAN UUID')
    current.update(uuid=selected[0][0],name=selected[0][1],driver=selected[0][2])
    if current!=expected:
        raise ValueError('Runtime differs from qualified candidate; separate review required')
    if torch.cuda.is_initialized():
        raise ValueError('Read-only runtime check initialized CUDA')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--frozen-commit', help='Read-only extra check: actual full HEAD and clean Git status; does not authorize launch')
    parser.add_argument('--resume', type=Path, help='Read-only same-run recovery validation; never restores tensors or launches training')
    args=parser.parse_args()
    # Every path is checked independently; no execution or mutation CLI exists.
    for path in args.manifest:
        print(json.dumps(verify_manifest(path, frozen_commit=args.frozen_commit, resume=args.resume),sort_keys=True))


if __name__=='__main__':
    main()
