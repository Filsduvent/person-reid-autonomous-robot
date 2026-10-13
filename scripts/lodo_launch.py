"""Operational launcher for future reviewed unified Stage B runs (binding v2).

Launch requires a clean wrapper release and an explicit local operator execution
declaration. The declaration is an audit guard, not cryptographic authorization. --check is read-only and never starts tmux.
Only the standard library is imported in the operator process.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import datetime
import fcntl
import hashlib
import json
import os
import pwd
import stat
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

IMPLEMENTATION = '4a03fb0d37cec2e41af92b8ff6e681c91ff5c387'
GPU = 'GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c'
POLICY_SHA = 'cf7b9e7bed01da13037e36e7ffd57f14a8645b6cca37a7febe29084d1d3b3895'
WRAPPER_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = WRAPPER_ROOT  # Compatibility name; execution is always this unified checkout.
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))
from reid.utils import lodo_operational as operational
load_binding = operational.load_binding
configuration = operational.configuration
INDEX = 'docs/lodo_phase10_evidence/candidate-index.json'
PYTHON = Path('/home/addirakoze/environments/Reid/bin/python')
MIN_DISK = 64 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def clean_head(root, expected):
    if git(root, 'rev-parse', 'HEAD') != expected:
        raise ValueError('Frozen commit mismatch')
    if git(root, 'status', '--porcelain=v1', '-uall'):
        raise ValueError('Clean worktree required')


def canonical(path):
    path = Path(path).absolute()
    if '..' in path.parts or path.resolve() != path:
        raise ValueError('Path alias or traversal prohibited')
    return path


def environment(root):
    return operational.environment(root)


def candidate(config, root):
    return operational.candidate(config, root)


def operator():
    return pwd.getpwuid(os.getuid()).pw_name


def private_record(path):
    """Read one owned, private regular file without a stat/read replacement gap."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Approval must be an operator-owned private regular file (0600)')
        return stream.read()


def verify_wrapper_source(root, commit):
    expected = subprocess.check_output(['git','-C',str(root),'show',commit+':scripts/lodo_launch.py'])
    if (root/'scripts/lodo_launch.py').read_bytes() != expected:
        raise ValueError('Wrapper executable differs from its approved release')


def experiment_binding(config, root=WRAPPER_ROOT):
    return load_binding(config, root)


def release_approval(wrapper_root, config, resume):
    """Scoped local audit declaration, not cryptographic authentication."""
    return operational.approval(load_binding(config, wrapper_root), resume=resume, root=wrapper_root)


def check_namespace(run, resume):
    run = canonical(run)
    if resume:
        if not run.is_dir():
            raise ValueError('Resume requires an existing same-run directory; never restart from zero')
        terminal = run/'checkpoints/ckpt_terminal.pth'
        if terminal.exists() or terminal.with_suffix('.json').exists():
            raise ValueError('Terminal checkpoint exists; inspect completion, do not resume')
        if not (run/'checkpoints/recovery_index.json').is_file():
            raise ValueError('No committed recovery index; manual investigation required')
    elif run.exists():
        raise ValueError('Existing-run collision; use --resume only with compatible recovery')
    parent = run
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < MIN_DISK:
        raise ValueError('At least 64 GiB disk headroom required')
    return run


def validate_preflight_request(request):
    if not isinstance(request, dict) or set(request) != {'root','config','manifest','run_dir','resume','binding_sha256'}:
        raise ValueError('Invalid internal preflight request')
    root = canonical(request['root'])
    if root != WRAPPER_ROOT:
        raise ValueError('Preflight accepts only the canonical unified implementation')
    binding = load_binding(request['config'], root)
    b = binding['record']
    if (request['manifest'] != b['manifest'] or request['run_dir'] != b['run_dir'] or
            request['binding_sha256'] != binding['sha256'] or
            request['resume'] not in (None, str(Path(b['run_dir'])/'checkpoints/ckpt_last.pth'))):
        raise ValueError('Internal preflight candidate/namespace mismatch')
    return root


def verify_inputs(cfg, manifest, root, *, check_runtime=True):
    """v2 inputs check; old v1 code/output fields remain historical references."""
    import lodo_official_preparation as prep
    from reid.utils.lodo_protocol import configuration_identity, scientific_configuration
    from reid.utils.lodo_checkpoint import file_digest
    from reid.utils.lodo_recovery import recovery_policy, CUDA_BOUNDARY_CAPABILITY
    import torch
    from torchvision.models import ResNet50_Weights
    from reid.models.pcb import HISTORICAL_IMAGENET_SHA256
    prep.validate_envelope(manifest)
    imm = manifest['immutable']
    if (configuration_identity(cfg) != imm['configuration']['scientific_sha256'] or
            scientific_configuration(cfg) != imm['configuration']['scientific']):
        raise ValueError('Scientific configuration differs from historical manifest')
    if (recovery_policy(cfg) != imm['recovery']['checkpoint_policy'] or
            imm['recovery']['capability'] != CUDA_BOUNDARY_CAPABILITY or
            imm['recovery']['checkpoint_schema'] != 2):
        raise ValueError('Recovery contract differs')
    if check_runtime:
        prep.verify_runtime(imm['execution_candidate'])
    arch = cfg['protocol']['architecture']
    name = 'resnet50-11ad3fa6.pth' if arch == 'resnet50' else 'resnet50-19c8e357.pth'
    weights = Path(torch.hub.get_dir())/'checkpoints'/name
    expected = ('11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
                if arch == 'resnet50' else HISTORICAL_IMAGENET_SHA256)
    if (imm['pretrained']['path'] != str(weights.resolve()) or imm['pretrained']['sha256'] != expected
            or file_digest(weights) != expected or
            (arch == 'resnet50' and Path(ResNet50_Weights.IMAGENET1K_V2.url).name != name)):
        raise ValueError('Pretrained loader/hash differs; no download/fallback')
    observed, mapping = prep.dataset_evidence(cfg)
    if observed != imm['dataset']:
        raise ValueError('Live dataset/map/sidecar/provenance differs from historical science')
    mapping_path = canonical(root/imm['mapping_file'])
    if not mapping_path.is_relative_to(root/'docs/lodo_phase10_evidence/maps') or mapping_path.read_bytes() != prep.canonical_json(mapping):
        raise ValueError('Recoverable mapping differs')
    return observed


def cpu_preflight(request):
    """Execute only in a fresh child; forbid CUDA/image/target execution."""
    root = validate_preflight_request(request)
    os.chdir(root)
    sys.path[:0] = [str(root), str(root/'scripts')]
    import lodo_official_preparation as prep
    from reid.utils.config import load_config, validate_reid_config
    from reid.utils.lodo_cuda_policy import activate_policy, validate_runtime, runtime_controls
    from reid.utils.lodo_device import audit_gpu
    import torch
    with prep.metadata_only():
        binding = load_binding(request['config'], root)
        cfg = configuration(binding, request['resume'])
        activate_policy(cfg)
        validate_runtime(cfg)
        resume_path = request['resume']
        if resume_path:
            cfg['train']['save']['resume'] = resume_path
        _, manifest, _ = candidate(request['config'], root)
        evidence = verify_inputs(cfg, manifest, root)
        recovery = None
        if resume_path:
            from reid.utils.lodo_recovery import select_generation
            from reid.utils.lodo_checkpoint import restore_lodo_checkpoint, read_checkpoint
            from reid.data.build import build_train_loader
            from reid.models.build import build_model
            from reid.losses.build import build_criterion
            from reid.optim.build import build_optimizer, build_center_optimizer, build_scheduler
            from reid.utils.config import validate_model_loss_requirements
            selected, fallback = select_generation(cfg, Path(request['run_dir']))
            loader, classes = build_train_loader(cfg)
            validate_reid_config(cfg, num_classes=classes)
            model = build_model(cfg, num_classes=classes, initialize_pretrained=False)
            dim = validate_model_loss_requirements(cfg, model)
            criterion = build_criterion(cfg, num_classes=classes, feat_dim=dim)
            optimizer = build_optimizer(cfg, model)
            auxiliary = build_center_optimizer(cfg, criterion)
            scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loader))
            progress, position, kind = restore_lodo_checkpoint(selected, cfg=cfg,loader=loader,
                model=model,criterion=criterion,optimizer=optimizer,auxiliary=auxiliary,
                scheduler=scheduler,cpu_inspection=True,validate_only=True)
            if kind != 'recovery' or progress.completed_primary_updates >= progress.max_updates:
                raise ValueError('Only incomplete validated recovery generations may continue')
            _, proof = read_checkpoint(selected)
            recovery = dict(path=str(selected),sha256=proof['checkpoint_sha256'],fallback=fallback,
                completed_primary_updates=progress.completed_primary_updates,
                center_updates=progress.auxiliary_steps,next_epoch=position['next_epoch'],
                next_lr=list(scheduler.rates_after(progress.completed_primary_updates)))
        audit = audit_gpu(GPU)  # Any selected-device compute PID, including own, is refused.
        if audit['name'] != 'NVIDIA TITAN Xp' or audit['free_mib'] < 10000:
            raise ValueError('TITAN Xp requires at least 10000 MiB free')
        cap = subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=compute_cap',
                                      '--format=csv,noheader,nounits'],text=True).strip()
        major, minor = map(int,cap.split('.'))
        supported = torch._C._cuda_getArchFlags().split()
        if not any(a.startswith('sm_') and a[3:].isdigit() and int(a[3:])//10 == major
                   and int(a[3:])%10 <= minor for a in supported):
            raise ValueError('No compatible compiled CUDA architecture')
        if torch.cuda.is_initialized():
            raise ValueError('Preflight initialized CUDA')
        if recovery is not None:
            cfg['train']['save']['resume'] = recovery['path']
        return dict(candidate=evidence,recovery=recovery,resolved_config=cfg,gpu=audit,
                    runtime_controls=runtime_controls(),cuda_initialized=False,
                    compute_capability=cap,compiled_architectures=supported)


def prepare(config, resume=False, *, require_release=True, root=None, wrapper_root=None):
    root = canonical(root or WRAPPER_ROOT)
    wrapper_root = canonical(wrapper_root or root)
    if wrapper_root != root:
        raise ValueError('Launcher and trainer must use the same unified checkout')
    binding = load_binding(config, root)
    b = binding['record']
    row, manifest, manifest_path = candidate(config, root)
    approval = release_approval(root, row['config'], resume) if require_release else None
    policy = canonical(b['policy']['path'])
    if sha(policy) != POLICY_SHA:
        raise ValueError('Standing allocation policy SHA256 mismatch')
    run = check_namespace(b['run_dir'], resume)
    if not run.is_relative_to(root/'exp/lodo/v1'):
        raise ValueError('Run must belong to this unified implementation')
    request = dict(root=str(root), config=row['config'], manifest=row['manifest'], run_dir=str(run),
                   resume=str(run/'checkpoints/ckpt_last.pth') if resume else None, binding_sha256=binding['sha256'])
    try:
        result = subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), '--_preflight', json.dumps(request)],
                                env=environment(root), cwd=root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise ValueError('CPU preflight refused: '+(error.stderr or str(error)).strip()) from error
    evidence = json.loads(result.stdout)
    if evidence.get('cuda_initialized') is not False:
        raise ValueError('CPU preflight did not prove CUDA remains uninitialized')
    if load_binding(config, root) != binding:
        raise ValueError('Operational binding changed during preflight')
    check_namespace(run, resume)
    if require_release and release_approval(root, row['config'], resume) != approval:
        raise ValueError('Execution approval changed during preflight')
    selected = evidence['recovery']['path'] if resume else None
    relative_run = str(run.relative_to(root))
    command = [str(PYTHON), '-u', str(root/'scripts/train.py'), '--config', row['config'],
               '--opts', 'experiment.output_dir='+relative_run]
    if resume:
        command.append('train.save.resume='+selected)
    return dict(schema_version=2, implementation_commit=b['commit'], binding=binding,
        config=row['config'], config_sha256=sha(root/row['config']), manifest=row['manifest'],
        manifest_sha256=sha(manifest_path), immutable_sha256=manifest['immutable_sha256'],
        root=str(root), run_dir=str(run), resume=resume, command=command, environment=environment(root),
        evidence=evidence, approval=approval, policy=b['policy'], interpreter=b['interpreter'],
        disk_free_bytes=shutil.disk_usage(run if run.exists() else root).free)


def write_json(path, value):
    operational.write_private(path, value)


def tmux_command(session, audit, env):
    # tmux parses a single shell-command string: quote every argument exactly once.
    shell = 'exec '+shlex.join(['/usr/bin/env','-u','PYTHONHOME','-u','PYTHONSTARTUP','-u','TORCH_HOME',str(PYTHON),'-B',str(Path(__file__).resolve()),'--_execute',str(audit/'launch-receipt.json')])
    shell += ' > '+shlex.quote(str(audit/'stdout.log'))+' 2>&1'
    command = ['tmux','new-session','-d','-s',session,'-c',str(WRAPPER_ROOT)]
    for key in ('CUDA_VISIBLE_DEVICES','LODO_AUTHORIZED_GPU_UUID','CUBLAS_WORKSPACE_CONFIG',
                'PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH'):
        command += ['-e',key+'='+env[key]]
    return command+[shell]


def dispatch(plan):
    if not plan['approval']:
        raise ValueError('Check-only plan cannot dispatch')
    if plan.get('schema_version') != 2 or Path(plan['root']) != WRAPPER_ROOT:
        raise ValueError('Only unified v2 plans can dispatch')
    if not shutil.which('tmux'):
        raise ValueError('tmux is unavailable')
    base = canonical(WRAPPER_ROOT/'.local/lodo_launch/receipts')
    base.mkdir(parents=True,exist_ok=True)
    audit = Path(tempfile.mkdtemp(prefix='launch-',dir=base))
    session = 'lodo-'+plan['config'].split('/')[-1].removesuffix('.yaml')+'-'+uuid.uuid4().hex[:12]
    record = dict(plan=plan,timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  operator=operator(),session=session,stdout=str(audit/'stdout.log'),
                  status='DISPATCH_REQUESTED_NOT_PROOF_OF_TRAINING')
    # Record only controlled environment variables; never persist secrets from the shell.
    record['plan'] = dict(plan,environment={k:plan['environment'][k] for k in (
        'CUDA_VISIBLE_DEVICES','LODO_AUTHORIZED_GPU_UUID','CUBLAS_WORKSPACE_CONFIG',
        'PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH')})
    try:
        with (audit/'pip-freeze.txt').open('x') as f:
            subprocess.run([str(PYTHON),'-m','pip','freeze','--all'],check=True,stdout=f,env=plan['environment'])
        record['dependency_freeze_sha256'] = sha(audit/'pip-freeze.txt')
        write_json(audit/'launch-receipt.json',record)
        subprocess.run(tmux_command(session,audit,plan['environment']),check=True,env=environment(Path(plan['root'])))
    except Exception as error:
        write_json(audit/'dispatch-failure.json',dict(error=str(error)))
        raise
    return dict(session=session,log=str(audit/'stdout.log'),run_dir=plan['run_dir'],audit_dir=str(audit))


def execute(receipt):
    """Detached admission guard; retain an advisory GPU lock across trainer exec."""
    receipt = canonical(receipt)
    if not receipt.is_relative_to(WRAPPER_ROOT/'.local/lodo_launch/receipts'):
        raise ValueError('Unrecognized operational receipt')
    record, _ = operational.private_record(receipt)
    old = record['plan']
    if old.get('schema_version') != 2 or Path(old['root']) != WRAPPER_ROOT:
        raise ValueError('Historical receipts cannot execute in the unified launcher')
    lock = WRAPPER_ROOT/'.local/lodo_launch'/('gpu-'+GPU+'.lock')
    fd = os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.set_inheritable(fd,True)
        current = prepare(old['config'],old['resume'])  # Full fresh admission inside tmux.
        for key in ('command','config_sha256','manifest_sha256','immutable_sha256','approval','interpreter','policy','root','run_dir','binding'):
            if current[key] != old[key]:
                raise ValueError('Provenance changed between dispatch and execution: '+key)
        if current['evidence']['recovery'] != old['evidence']['recovery']:
            raise ValueError('Recovery generation changed before execution')
        cfg = current['evidence']['resolved_config']
        if current['resume'] and sha(current['evidence']['recovery']['path']) != current['evidence']['recovery']['sha256']:
            raise ValueError('Recovery checkpoint changed after detached preflight')
        admission = receipt.parent/'execution-admission.json'
        write_json(admission, operational.training_receipt(current['binding'], current['approval'], cfg,
                                                         root=Path(current['root'])))
        if release_approval(WRAPPER_ROOT,current['config'],current['resume']) != current['approval']:
            raise ValueError('Execution approval changed before trainer exec')
        current['environment'].update(LODO_ADMISSION_RECEIPT=str(admission), LODO_ADMISSION_SHA256=sha(admission))
        os.chdir(current['root'])
        os.execve(current['command'][0],current['command'],current['environment'])
    except Exception as error:
        write_json(receipt.parent/'execution-failure.json',dict(error=str(error)))
        raise
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check',action='store_true',help='Read-only preparation; no release approval, receipts or tmux')
    parser.add_argument('--_preflight',help=argparse.SUPPRESS)
    parser.add_argument('--_execute',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._preflight and args._execute:
        parser.error('Internal modes are mutually exclusive')
    if (args._preflight or args._execute) and (args.config or args.resume or args.check):
        parser.error('Internal modes cannot be combined with public/check-only flags')
    try:
        if args._preflight:
            with redirect_stdout(sys.stderr):
                evidence = cpu_preflight(json.loads(args._preflight))
            print(json.dumps(evidence))
            return
        if args._execute:
            execute(args._execute)
            return
        if not args.config:
            parser.error('--config is required')
        plan = prepare(args.config,args.resume,require_release=not args.check)
        if args.check:
            # Avoid exposing unrelated inherited environment variables.
            plan['environment'] = {k:plan['environment'][k] for k in ('CUDA_VISIBLE_DEVICES','LODO_AUTHORIZED_GPU_UUID','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH')}
            print(json.dumps(dict(status='CHECK_ONLY_NOT_AUTHORIZED',plan=plan),indent=2))
        else:
            result = dispatch(plan)
            print('Detached launch dispatched; inspect admission/log to confirm trainer startup.')
            print(json.dumps(result,indent=2))
            print('Monitor: tail -f '+shlex.quote(result['log']))
            print('Attach: tmux attach-session -t '+shlex.quote(result['session']))
            print('GPU: nvidia-smi -i '+GPU)
    except (ValueError,OSError,subprocess.SubprocessError,KeyError) as error:
        print('REFUSED: '+str(error),file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
