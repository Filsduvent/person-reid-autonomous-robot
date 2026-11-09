#!/usr/bin/env python3
"""Verify migrated imports and dataset identity on CPU; never admit training."""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mapped(path, mapping):
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        if path == old or path.startswith(old + '/'):
            return Path(new + path[len(old):])
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output receipt already exists; preserve it and choose a fresh path')
    from scripts.mgn_gpu_smoke import file_hash, source_provenance
    from scripts.mgn_preflight import environment, evaluation_identity, verify_hashes
    from reid.utils.config import load_config
    from reid.data.build import build_train_loader, build_test_loader
    import torch
    manifest = json.loads((ROOT/'docs/mgn_experiment_manifest.json').read_text())
    mapping = json.loads((args.kit/'path-map.json').read_text())
    imports = {}
    for name in ('numpy','PIL','yaml','torch','torchvision','sklearn','matplotlib',
                 'tensorboard','cv2','onnx','onnxruntime'):
        module = importlib.import_module(name)
        imports[name] = getattr(module, '__version__', 'imported')
    verify_hashes(manifest['runtime_source_sha256'])
    verify_hashes(manifest['qualification_receipt_sha256'])
    checks = {}
    for source, entry in manifest['sources'].items():
        verify_hashes({entry['config_path']: entry['config_sha256']})
        cfg = load_config(str(ROOT/entry['config_path']))
        # Verify trusted partition bytes before the loader deserializes them.
        for section in ('training','evaluation'):
            for old, expected in entry[section]['metadata_sha256'].items():
                path = mapped(old, mapping)
                if file_hash(path) != expected:
                    raise ValueError(f'Migrated partition hash mismatch: {path}')
        if file_hash(Path(cfg['model']['weights_path'])) != manifest['weights_sha256']:
            raise ValueError('Pretrained initialization artifact differs from the frozen source')
        train, classes = build_train_loader(cfg)
        test = build_test_loader(cfg)
        actual_train = source_provenance(train.dataset, cfg)
        actual_eval = evaluation_identity(test.dataset, cfg)
        for key in ('images','num_classes','records_sha256','identity_map_sha256'):
            if actual_train[key] != entry['training'][key]:
                raise ValueError(f'{source}: migrated training {key} differs')
        for key in ('images','query','gallery','records_sha256'):
            if actual_eval[key] != entry['evaluation'][key]:
                raise ValueError(f'{source}: migrated evaluation {key} differs')
        checks[source] = dict(classes=classes, training=actual_train, evaluation=actual_eval,
                              portable_dataset_identity_matches=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU inspection unexpectedly initialized CUDA')
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free,driver_version','--format=csv'],
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  status='CPU_DATA_VERIFIED_GPU_QUALIFICATION_PENDING',
                  hostname=subprocess.check_output(['hostname'],text=True).strip(),
                  source_manifest_sha256=file_hash(ROOT/'docs/mgn_experiment_manifest.json'),
                  environment=environment(), imports=imports, datasets=checks,
                  gpu_inventory=dict(exit_code=gpu.returncode,output=gpu.stdout),
                  training_launched=False, evaluation_launched=False, cuda_initialized=False,
                  notice='Portable partition/record/map identity verified; host file identity and GPU admission require new qualification.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],receipt=str(args.output)),indent=2))


if __name__ == '__main__':
    main()
