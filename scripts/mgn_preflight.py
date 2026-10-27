"""Read-only MGN experiment admission. This program never launches training."""
import argparse
import contextlib
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.mgn_gpu_smoke import (SOURCES, authorized_uuid, file_hash, canonical_hash, source_provenance)
MANIFEST = ROOT/'docs/mgn_experiment_manifest.json'
PACKAGES = ('torch','torchvision','numpy','Pillow','PyYAML','tensorboard','matplotlib','scikit-learn')
CONTROLS = ('OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF',
            'CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','CUDA_LAUNCH_BLOCKING')


def environment():
    import torch,torchvision
    return dict(python=platform.python_version(),prefix=sys.prefix,torch_build=torch.__version__,torchvision_build=torchvision.__version__,
                packages={name:importlib.metadata.version(name) for name in PACKAGES},
                controls={name:os.environ.get(name) for name in CONTROLS})


def verify_hashes(hashes, *, root=ROOT):
    for relative,digest in hashes.items():
        path=Path(relative)
        if not path.is_absolute():path=root/path
        if file_hash(path)!=digest:raise ValueError(f'Frozen file hash mismatch: {relative}')


def evaluation_identity(dataset, cfg):
    """Definitions and host inventory only; no image decoding or ranking."""
    source=cfg['data']['test']['dataset']['name'];directory=Path(dataset.im_dir).resolve()
    metadata=([Path(dataset.dataset_dir)/name for name in ('list_query.txt','list_gallery.txt')]
              if source=='msmt17' else [directory.parent/'partitions.pkl'])
    rows=[];inventory=[]
    for name,pid,cam,mark in zip(dataset.im_names,dataset.pids,dataset.cams,dataset.marks):
        path=(directory/str(name)).resolve()
        if not path.is_relative_to(directory):raise ValueError('Evaluation image escapes its declared directory.')
        stat=path.stat();rows.append([str(name),int(pid),int(cam),int(mark)])
        inventory.append([str(name),str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns])
    if len(rows)!=len(dataset):raise ValueError('Evaluation metadata arrays are not aligned.')
    return dict(images=len(dataset),query=sum(row[3]==0 for row in rows),gallery=sum(row[3]==1 for row in rows),
        records_sha256=canonical_hash(rows),file_inventory_sha256=canonical_hash(inventory),
        metadata_sha256={str(p.resolve()):file_hash(p) for p in metadata},
        image_content_hash_policy='File inventory/metadata only; full evaluation image bytes are not hashed or decoded.')


def planned_pk_budget(labels):
    """Fresh-seed index-only trace; no loader iteration, images, model or updates."""
    from reid.data.samplers import PKBatchSampler
    import hashlib
    sampler=PKBatchSampler(labels,P=16,K=4,drop_last=True,seed=42)
    counts=[];stream=hashlib.sha256()
    for epoch in range(1,121):
        count=0
        for indices in sampler:
            stream.update(json.dumps([epoch,indices],separators=(',',':')).encode()+b'\n');count+=1
        counts.append(count)
    return dict(unit='optimization updates',epochs=120,batches_by_epoch=counts,total_updates=sum(counts),
        image_presentations=64*sum(counts),index_stream_sha256=stream.hexdigest(),
        caveat='Fresh uninterrupted sampler seed42 only; ordinary resume does not restore sampler/RNG trajectories.')


def output_guard(source,cfg,*,resume=False,root=ROOT):
    expected=root/'exp'/'mgn'/source
    supplied=Path(cfg['experiment']['output_dir'])
    if not supplied.is_absolute():supplied=root/supplied
    if supplied!=expected or supplied.resolve()!=supplied:
        raise ValueError('Output must be the isolated frozen source path without symlink redirection.')
    if resume:
        if not supplied.is_dir() or not (supplied/'checkpoints/ckpt_last.pth').is_file():
            raise ValueError('Resume requires the original run and last epoch-boundary checkpoint.')
    elif supplied.exists():raise ValueError('Fresh launch refuses an existing output path; preserve it, never overwrite or clean it.')
    return supplied


def normalized_training_cfg(cfg):
    result=copy.deepcopy(cfg)
    output=Path(result['experiment']['output_dir'])
    result['experiment']['output_dir']=str(output if output.is_absolute() else ROOT/output)
    result['train']['save']['resume']=''
    return result


def resume_guard(cfg,run,classes):
    """Validate continuation state only. No step, forward, files or CUDA model."""
    import torch
    from reid.utils.checkpoint import reconstruct_model,validate_selection_state
    from reid.optim.build import build_optimizer,build_scheduler
    from scripts.train import validate_resume_best_checkpoint
    path=run/'checkpoints/ckpt_last.pth'
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    epoch=checkpoint.get('epoch')
    if type(epoch) is not int or not 1<=epoch<120:raise ValueError('Resume must continue a completed epoch before epoch120.')
    if normalized_training_cfg(checkpoint.get('cfg',{}))!=normalized_training_cfg(cfg):
        raise ValueError('Resume configuration conflicts with the frozen scientific recipe.')
    metadata=checkpoint.get('reconstruction',{})
    if (metadata.get('model_name')!='mgn' or metadata.get('num_classes')!=classes
            or metadata.get('embedding_dim')!=2048):raise ValueError('Resume source-model identity/class count is incompatible.')
    validate_selection_state(checkpoint.get('selection_state'),epoch=epoch,metric='mAP')
    validate_resume_best_checkpoint(checkpoint,run/'checkpoints/ckpt_best.pth','mAP')
    model=reconstruct_model(checkpoint)
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer)
    if checkpoint.get('optimizer') is None or checkpoint.get('scheduler') is None:
        raise ValueError('Resume optimizer/scheduler state is missing.')
    optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
    scheduler.validate_completed_epoch(epoch)
    return dict(checkpoint=str(path),sha256=file_hash(path),next_epoch=epoch+1,
                best=checkpoint['selection_state'],next_epoch_rates=sorted(set(scheduler.get_last_lr())),
                caveat='Epoch-boundary continuation only, not stochastic/bitwise CUDA recovery. Preserve prior launch artifacts before restarting.')


def gpu_guard(uuid,expected,peak_reserved):
    """Current manual authority, no contention, one visible UUID; no model."""
    authorized_uuid(uuid)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:raise ValueError('Visibility must contain only the frozen authorized full UUID.')
    from reid.utils.lodo_device import audit_gpu
    before=audit_gpu(uuid)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Frozen CUDA device unavailable; do not launch or use CPU fallback.')
    properties=torch.cuda.get_device_properties(0)
    actual=str(getattr(properties,'uuid',''))
    if not actual.startswith('GPU-'):actual='GPU-'+actual
    if actual!=uuid or properties.name!=expected['gpu_name']:raise ValueError('Live GPU differs from qualification.')
    runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if any(runtime[key]!=expected[key] for key in runtime):raise ValueError('CUDA/framework runtime controls differ from qualification.')
    if before['driver']!=expected['driver']:raise ValueError('GPU driver differs from qualification.')
    current=audit_gpu(uuid,allow_pid=os.getpid())
    if current['free_mib']*1024**2<peak_reserved:raise ValueError('Available memory is below the observed source admission reservation peak.')
    return dict(before=before,after_runtime_query=current,runtime=runtime)


def effective_configuration(frozen, configuration, config_path, frozen_path, *, resume):
    """Bind common-trainer arguments to the admitted YAML; resume is operational."""
    if config_path is not None and Path(config_path).resolve()!=frozen_path.resolve():
        raise ValueError('Training config path must identify the frozen source preset.')
    if configuration is None:
        return frozen
    if normalized_training_cfg(configuration)!=normalized_training_cfg(frozen):
        raise ValueError('Effective training configuration differs from frozen admission.')
    supplied=configuration['train']['save'].get('resume','')
    if not isinstance(supplied,str):raise ValueError('Resume must be a checkpoint path string.')
    if resume:
        expected=ROOT/'exp/mgn'/frozen['data']['train']['dataset']['name']/'checkpoints/ckpt_last.pth'
        path=Path(supplied)
        if not path.is_absolute():path=ROOT/path
        if not supplied.strip() or path!=expected or path.resolve()!=expected:
            raise ValueError('Resume must use the original source ckpt_last.pth without redirection.')
    elif supplied:
        raise ValueError('Fresh admission cannot contain a resume checkpoint.')
    return configuration


def preflight(source,*,resume=False,configuration=None,config_path=None):
    from reid.utils.config import load_config,validate_reid_config
    from reid.utils.config_schema import validate_config
    if source not in SOURCES:raise ValueError('Unknown frozen MGN source.')
    manifest=json.loads(MANIFEST.read_text())
    if manifest.get('status')!='FROZEN':raise ValueError('MGN preparation freeze is not complete.')
    entry=manifest['sources'][source];uuid=manifest['gpu_uuid']
    authorized_uuid(uuid)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:raise ValueError('Set single-UUID CUDA visibility before preflight.')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','reid','scripts','configs/mgn','docs/mgn_experiment_manifest.json'],cwd=ROOT,text=True)
    if dirty:raise ValueError('Frozen runtime/config/manifest worktree is dirty; preserve and resolve it before launch.')
    subprocess.run(['git','merge-base','--is-ancestor',manifest['tested_implementation_commit'],'HEAD'],cwd=ROOT,check=True,capture_output=True)
    verify_hashes(manifest['runtime_source_sha256'])
    verify_hashes(manifest['qualification_receipt_sha256'])
    frozen_path=ROOT/entry['config_path']
    verify_hashes({str(frozen_path):entry['config_sha256']})
    cfg=load_config(str(frozen_path));validate_config(cfg);validate_reid_config(cfg)
    if cfg!=entry['resolved_configuration']:raise ValueError('Resolved source configuration differs from frozen admission.')
    cfg=effective_configuration(cfg,configuration,config_path,frozen_path,resume=resume)
    run=output_guard(source,cfg,resume=resume)
    if environment()!=manifest['environment']:raise ValueError('Python/packages/environment controls differ from the frozen admission environment.')
    verify_hashes({cfg['model']['weights_path']:manifest['weights_sha256']})
    verify_hashes(entry['training']['metadata_sha256']);verify_hashes(entry['evaluation']['metadata_sha256'])
    from reid.data.build import build_train_loader,build_test_loader
    with contextlib.redirect_stdout(sys.stderr):
        train_loader,classes=build_train_loader(cfg);test_loader=build_test_loader(cfg)
    actual=source_provenance(train_loader.dataset,cfg)
    if actual!=entry['training']:raise ValueError('Training data/map/file inventory differs from qualified source.')
    if evaluation_identity(test_loader.dataset,cfg)!=entry['evaluation']:
        raise ValueError('Evaluation partitions/records/file inventory differ from the freeze.')
    verify_hashes({str(Path(train_loader.dataset.im_dir)/name):digest for name,digest in entry['smoke_image_sha256'].items()})
    if shutil.disk_usage(ROOT/'exp').free<manifest['minimum_free_disk_bytes']:raise ValueError('Insufficient disk allowance for checkpoints/logs and retained provenance.')
    continuation=resume_guard(cfg,run,classes) if resume else None
    gpu=gpu_guard(uuid,entry['runtime'],entry['maximum_reserved_bytes'])
    return dict(status='READY_TO_RESUME' if resume else 'READY_FRESH',source=source,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        manifest_sha256=file_hash(MANIFEST),config=str(frozen_path),config_sha256=entry['config_sha256'],
        output=str(run),source_classes=classes,training_images=actual['images'],
        planned_fresh_updates=entry['planned_fresh_budget']['total_updates'],continuation=continuation,gpu=gpu,
        training_launched=False,notice='Read-only admission passed; the common trainer may now dispatch training.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=SOURCES,required=True)
    parser.add_argument('--resume',action='store_true',help='Validate an existing same-run epoch-boundary continuation; never launch it.')
    args=parser.parse_args()
    try:
        result=preflight(args.source,resume=args.resume)
    except Exception as error:
        print(json.dumps(dict(status='REFUSED',source=args.source,error=str(error),training_launched=False)))
        return 2
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
