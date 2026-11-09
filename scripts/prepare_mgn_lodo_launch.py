"""Prepare a private, pinned YAML/receipt pair for USER-operated official training.

This command never imports the trainer, initializes CUDA, or launches training.
Qualification must already exist and match the current committed release.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(fold, run_id, qualification_path, approval_reference):
    import yaml
    from reid.utils.config import load_config
    from reid.utils import mgn_lodo_operational as operational
    from reid.utils import mgn_lodo_protocol as science
    from reid.utils.lodo_checkpoint import digest, file_digest
    from reid.utils.lodo_operational import private_record
    import subprocess

    if fold not in range(1,5) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', run_id):
        raise ValueError('Valid fold1..4 and a single run ID required')
    if not approval_reference.strip():
        raise ValueError('User-operated launch declaration required')
    qualification_path = Path(qualification_path).absolute()
    qualification, qualification_sha = private_record(qualification_path)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    snapshot = operational.source_snapshot(); operational.validate_committed_sources(ROOT,snapshot)
    cfg = load_config(str(ROOT/f'configs/lodo/mgn_lodo_v2/mgn_lodo-{fold}_seed42.yaml'))
    if (qualification['status'] != 'MGN_LODO_FOLD_QUALIFIED' or qualification['fold'] != f'LODO-{fold}'
            or qualification['release_commit'] != head or qualification['source_sha256'] != snapshot
            or qualification['config_digest'] != science.configuration_identity(cfg)):
        raise ValueError('Fold has no qualification for the current code/science release')
    destination = ROOT/'.local/mgn_lodo_launch'/f'LODO-{fold}'/run_id
    if destination.resolve() != destination or destination.exists():
        raise ValueError('Use a new canonical launch directory')
    cfg['experiment']['output_dir'] = str(ROOT/'exp/lodo/mgn_lodo_v2/mgn'/f'LODO-{fold}'/'seed42'/run_id)
    science.validate_lodo_config(cfg)
    # Validate runtime/inventories/artifact through the actual public admission
    # before emitting a runnable command. Failure preserves files for inspection.
    destination.mkdir(parents=True, mode=0o700)
    config_path = destination/'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False)); config_path.chmod(0o600)
    receipt = dict(schema_version=1, record_type='mgn_lodo_v2_execution',action='training',
        approval_reference=approval_reference,root=str(ROOT),config_path=str(config_path),
        config_sha256=file_digest(config_path),effective_config_sha256=digest(cfg),release_commit=head,
        qualification_path=str(qualification_path),qualification_sha256=qualification_sha,
        gpu_uuid=qualification['gpu_uuid'])
    receipt_path=destination/'execution.json'
    descriptor=os.open(receipt_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as stream:json.dump(receipt,stream,indent=2)
    previous=os.environ.get('MGN_LODO_ADMISSION_RECEIPT')
    try:
        os.environ['MGN_LODO_ADMISSION_RECEIPT']=str(receipt_path)
        operational.validate_training_admission(config_path,cfg)
    finally:
        if previous is None:os.environ.pop('MGN_LODO_ADMISSION_RECEIPT',None)
        else:os.environ['MGN_LODO_ADMISSION_RECEIPT']=previous
    command='MGN_LODO_ADMISSION_RECEIPT='+shlex.quote(str(receipt_path))+' '+shlex.quote(sys.executable)+' scripts/train.py --config '+shlex.quote(str(config_path))
    print(command)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--qualification',required=True)
    parser.add_argument('--approval-reference',required=True)
    args=parser.parse_args()
    prepare(args.fold,args.run_id,args.qualification,args.approval_reference)
