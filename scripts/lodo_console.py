"""Main-checkout console for the frozen LODO runner; no torch import.

The six original user documents may stay dirty. Executable source must match a
reviewed console release; the maintained execution worktree must be fully clean.
Local release/approval records are audit guards, not authenticated signatures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.parent / 'person-reid-autonomous-robot-lodo-console-execution'
PYTHON = Path('/home/addirakoze/environments/Reid/bin/python')
USER_DOCUMENTS = {'plan.md', 'docs/preliminary_results_log.md',
                  *('docs/pcb_' + key + '_results.yaml' for key in ('market1501', 'duke', 'cuhk03', 'msmt17'))}
# Existing ignored local Stage A YAMLs are not executable console inputs. Only
# frozen candidate paths are ever accepted/forwarded by train_config().
LOCAL_STAGE_A_CONFIGS = {'configs/' + name + '.yaml' for name in (
    'benchmark', 'debug_train', 'run_video', 'runtime_qualification_common',
    'experiment_market1501_resnet50_center_rerank', 'experiment_market1501_resnet50_id_only',
    'experiment_market1501_resnet50_triplet_id', 'experiment_market1501_resnet50_triplet_id_center',
    'experiment_market1501_resnet50_triplet_only', 'experiment_msmt17_resnet50_triplet_id_center')}
CONSOLE_FILES = {'scripts/lodo_launch.py', 'scripts/lodo_console.py',
                 'scripts/lodo_evaluate.py', 'scripts/lodo_results.py'}
EVALUATION_REVISION_FILES = {
    'scripts/lodo_console.py', 'scripts/lodo_results.py',
    'scripts/lodo_gpu_evaluate_worker.py', 'scripts/lodo_qualify_gpu_evaluation.py',
    'reid/engine/evaluator.py', 'reid/utils/lodo_evaluation.py',
    'reid/utils/lodo_evaluation_device.py', 'reid/utils/lodo_operational.py',
    'docs/lodo_gpu_evaluation.md', 'tests/test_lodo_gpu_evaluation.py',
    'tests/test_lodo_console.py',
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reid.utils import lodo_operational as op
from lodo_results import read_release_record, execution_commit


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def verify_source(root, commit, *, documents=False):
    root = op.canonical(root)
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Full reviewed source commit required')
    if git(root, 'rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('Reviewed source HEAD mismatch')
    if git(root, 'diff', '--cached', '--name-only'):
        raise ValueError('Staged changes are not allowed at launch')
    allowed = USER_DOCUMENTS if documents else set()
    changes = git(root, 'status', '--porcelain=v1', '-z', '-uall').split(b'\0')
    for raw in filter(None, changes):
        name = raw[3:].decode()
        if raw[:2] not in (b' M', b'??') or name not in allowed:
            raise ValueError('Unreviewed source change: ' + name)
        path = op.canonical(root / name)
        if not path.is_file():
            raise ValueError('Preserved document is not a regular file')
    tracked = set()
    for raw in filter(None, git(root, 'ls-tree', '-rz', commit).split(b'\0')):
        header, name = raw.split(b'\t', 1)
        mode, kind, blob = header.decode().split(); name = name.decode()
        tracked.add(name); path = op.canonical(root / name)
        if kind != 'blob' or mode not in ('100644', '100755') or not path.is_file():
            raise ValueError('Unsupported source entry: ' + name)
        if name in allowed:
            continue
        data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != blob or bool(path.stat().st_mode & 0o111) != (mode == '100755'):
            raise ValueError('Reviewed source blob/mode changed: ' + name)
    current = {str(p.relative_to(root)) for base in ('reid', 'scripts', 'tests', 'configs')
               for p in (root / base).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')}
    permitted_local = LOCAL_STAGE_A_CONFIGS if documents else set()
    for name in current & permitted_local:
        op.canonical(root / name)  # No aliases even for excluded Stage A YAMLs.
    if not current <= tracked | permitted_local:
        raise ValueError('Untracked/ignored executable source is prohibited')
    return tracked


def release():
    path = ROOT / '.local/lodo_console/release.json'
    if not path.is_file():
        raise ValueError('Reviewed main-console release required; no experiment launched')
    record, digest = read_release_record()
    tracked = verify_source(ROOT, record['frontend_commit'], documents=True)
    if not CONSOLE_FILES <= tracked:
        raise ValueError('Console implementation is not included in the reviewed commit')
    op.clean_head(ENGINE, record['execution_commit'])
    if record['schema_version'] == 2:
        changes = set(git(ROOT, 'diff', '--name-only', record['execution_commit'],
                          record['frontend_commit']).decode().splitlines())
        if not changes <= EVALUATION_REVISION_FILES:
            raise ValueError('Evaluation frontend revision changes frozen training source')
    return dict(path=str(path), sha256=digest, record=record)


def train_config(config):
    raw = Path(config)
    path = op.canonical(raw if raw.is_absolute() else ROOT / raw)
    relative = str(path.relative_to(ROOT))
    op.candidate(relative, ENGINE)
    if path.read_bytes() != (ENGINE / relative).read_bytes():
        raise ValueError('Main configuration differs from frozen execution configuration')
    return relative


def evaluation_request(config):
    path = op.canonical(Path(config) if Path(config).is_absolute() else ENGINE / config)
    if not path.is_relative_to(ENGINE / '.local/lodo_operational/v2/evaluation_requests') or not path.is_file():
        raise ValueError('Use an approved maintained-root v2 evaluation request')
    return path


def environment(*, cpu=False):
    env = op.environment(ENGINE)
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def frozen_worker(code, values, *, cpu=False, source=None):
    source = source or ENGINE
    command = [str(PYTHON), '-B', '-c', code, json.dumps(values)]
    env = environment(cpu=cpu)
    env['PYTHONPATH'] = str(source) + os.pathsep + str(source / 'scripts')
    result = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Frozen runner refused: ' + (result.stderr or result.stdout).strip())
    return json.loads(result.stdout)


def evaluation_check(path):
    return frozen_worker('''import json,sys
from pathlib import Path
from reid.utils.config import load_config
from reid.utils.lodo_operational import evaluation_admission
from scripts.lodo_official_preparation import metadata_only
args=json.loads(sys.argv[1]);p=Path(args['request'])
with metadata_only(): proof=evaluation_admission(p,load_config(str(p)),root=Path(args['training_root']))
print(json.dumps(proof))
''', dict(request=str(path), training_root=str(ENGINE)), cpu=True, source=ROOT)


def atomic(path, value):
    path = op.canonical(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(op.encode(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def status(job_path, state, **fields):
    atomic(job_path.parent / 'status.json', dict(state=state, **fields))


def load_job(path):
    path = op.canonical(path)
    if not path.is_relative_to(ROOT / '.local/lodo_console/jobs') or path.name != 'job.json':
        raise ValueError('Unrecognized console job')
    job, _ = op.private_record(path)
    if job['execution_root'] != str(ENGINE) or job['execution_commit'] != execution_commit():
        raise ValueError('Foreign execution job')
    return job


def session_alive(session):
    r = subprocess.run(['tmux', 'has-session', '-t', '=' + session], capture_output=True, text=True)
    if r.returncode == 0:
        return True
    if r.returncode == 1 and any(s in r.stderr.lower() for s in ('can\'t find session', 'no server running', 'no sessions', 'no such file')):
        return False
    raise ValueError('Cannot establish tmux session state: ' + r.stderr.strip())


def start_supervisor(path, mode, session):
    env = environment(cpu=True)
    shell = 'exec ' + shlex.join(['/usr/bin/env', '-u', 'PYTHONHOME', '-u', 'PYTHONSTARTUP', '-u', 'TORCH_HOME',
                                '-u', 'LODO_ADMISSION_RECEIPT', '-u', 'LODO_ADMISSION_SHA256',
                                str(PYTHON), '-B', str(ROOT / 'scripts/lodo_console.py'), mode, str(path)])
    shell += ' > ' + shlex.quote(str(path.parent / 'supervisor.log')) + ' 2>&1'
    command = ['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT)]
    for key in op.CONTROLLED:
        command += ['-e', key + '=' + env[key]]
    subprocess.run(command + [shell], env=env, check=True)


def training_audit(job):
    path = op.canonical(job['launch_receipt'])
    base = ENGINE / '.local/lodo_launch/receipts'
    if not path.is_relative_to(base) or path.name != 'launch-receipt.json':
        raise ValueError('Foreign frozen launch receipt')
    receipt, digest = op.private_record(path)
    if digest != job['launch_receipt_sha256']:
        raise ValueError('Frozen launch receipt changed')
    plan = receipt['plan']
    if (plan['schema_version'] != 2 or plan['root'] != str(ENGINE) or plan['implementation_commit'] != execution_commit()
            or plan['config'] != job['config'] or plan['run_dir'] != job['run_dir'] or receipt['session'] != job['session']):
        raise ValueError('Frozen launch provenance mismatch')
    admission, _ = op.private_record(path.parent / 'execution-admission.json')
    if (admission['status'] != 'ADMITTED_TRAINING' or admission['root'] != str(ENGINE)
            or admission['commit'] != execution_commit() or admission['config'] != job['config']
            or admission['run_dir'] != job['run_dir'] or admission['binding'] != plan['binding']
            or admission['approval'] != plan['approval']):
        raise ValueError('Missing/incompatible actual training admission')
    return plan


def watch_training(path):
    job = load_job(path)
    try:
        if release() != job['release']:
            raise ValueError('Console release changed after dispatch')
        while session_alive(job['session']):
            time.sleep(1)
        terminal = Path(job['run_dir']) / 'checkpoints/ckpt_terminal.pth'
        if not terminal.exists():
            status(path, 'FAILED', message='No completed terminal. Inspect log; use same-run --resume only with valid recovery.', exit_code=1)
            return
        training_audit(job)
        from lodo_results import register_run
        records = register_run(Path(job['run_dir']))
        status(path, 'COMPLETE', records=records, exit_code=0)
    except Exception as error:
        status(path, 'REGISTRATION_FAILED', message=str(error), exit_code=1)


def execute_evaluation(path):
    job = load_job(path)
    try:
        if release() != job['release'] or op.sha(job['request']) != job['request_sha256']:
            raise ValueError('Evaluation request/release changed after dispatch')
        current = evaluation_check(Path(job['request']))
        if current != job['admission']:
            raise ValueError('Separate evaluation admission changed')
        command = [str(PYTHON), '-u', str(ENGINE / 'scripts/evaluate.py'), '--config', job['request']]
        with Path(job['log']).open('xb') as log:
            # GPU requests are separately hash-pinned by admission. Metadata
            # checks remain CUDA-hidden; only the admitted worker sees the GPU.
            gpu = current.get('status') == 'ADMITTED_CUDA_EVALUATION'
            env = environment(cpu=not gpu)
            if gpu:
                command = [str(PYTHON), '-u', '-B', str(ROOT / 'scripts/lodo_gpu_evaluate_worker.py'), '--job', str(path)]
                env['PYTHONPATH'] = str(ROOT) + os.pathsep + str(ROOT / 'scripts')
            result = subprocess.run(command, cwd=ROOT if gpu else ENGINE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(path, 'FAILED', message='Frozen evaluation failed; inspect log. No metrics registered.', exit_code=result.returncode)
            return
        from lodo_results import register_run
        records = register_run(Path(job['run_dir']), result=Path(current['output_dir']) / 'lodo_result.json')
        status(path, 'COMPLETE', records=records, exit_code=0)
    except Exception as error:
        status(path, 'REGISTRATION_FAILED', message=str(error), exit_code=1)


def follow(path):
    job = load_job(path); position = 0
    print('Live log:', job['log'], '\nCtrl+C stops viewing; the detached job continues.', flush=True)
    try:
        missing_supervisor = 0
        while True:
            log = Path(job['log'])
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(position); chunk = stream.read(); position = stream.tell()
                if chunk:
                    print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
            state_path = path.parent / 'status.json'
            state = op.private_record(state_path)[0] if state_path.exists() else {}
            if state.get('state') in ('COMPLETE', 'FAILED', 'REGISTRATION_FAILED'):
                # Drain output published just before the final status.
                if log.exists() and log.stat().st_size > position:
                    continue
                print('\nConsole status:', json.dumps(state), flush=True)
                return state['exit_code']
            missing_supervisor = 0 if session_alive(job['supervisor_session']) else missing_supervisor + 1
            if missing_supervisor >= 10:
                raise ValueError('Console supervisor is absent. Training may still be running; inspect ' + str(path.parent) + ' before any resume.')
            time.sleep(.5)
    except KeyboardInterrupt:
        print('\nStopped viewing. Job continues; use --monitor to return to this attempt.', flush=True)
        return 0


def latest_job(config, kind):
    choices = []
    for path in (ROOT / '.local/lodo_console/jobs').glob('*/job.json'):
        job = load_job(path)
        if job['config'] == config and job['kind'] == kind:
            choices.append((job['created_ns'], str(path), path))
    if not choices:
        raise ValueError('No matching console attempt to monitor')
    return max(choices)[2]


def launch(config, *, resume=False, check=False, no_follow=False, monitor=False, evaluate=False):
    kind = 'evaluation' if evaluate else 'training'
    if monitor:
        key = str(evaluation_request(config)) if evaluate else train_config(config)
        return follow(latest_job(key, kind))
    reviewed = release()
    if evaluate:
        request = evaluation_request(config); admission = evaluation_check(request)
        if check:
            print(json.dumps(dict(status='CHECK_ONLY_NOT_EXECUTED', admission=admission), indent=2)); return 0
        job = dict(kind=kind, config=str(request), request=str(request), request_sha256=op.sha(request),
                   admission=admission, run_dir=admission['binding']['record']['run_dir'])
    else:
        relative = train_config(config)
        print('Checking frozen source, dataset/map, recovery and GPU admission...', flush=True)
        if check:
            proof = frozen_worker('''import json,sys
from scripts import lodo_launch as runner
from reid.utils.lodo_operational import CONTROLLED
args=json.loads(sys.argv[1])
plan=runner.prepare(args['config'],args['resume'],require_release=False)
plan['environment']={key:plan['environment'][key] for key in CONTROLLED}
print(json.dumps(plan))
''', dict(config=relative, resume=resume))
            print(json.dumps(dict(status='CHECK_ONLY_NOT_EXECUTED', plan=proof), indent=2))
            return 0
        result = frozen_worker('''import json,sys
from scripts import lodo_launch as runner
args=json.loads(sys.argv[1])
plan=runner.prepare(args['config'],args['resume'])
print(json.dumps(runner.dispatch(plan)))
''', dict(config=relative, resume=resume))
        print('Frozen dispatch (inspect this receipt if console setup fails):', json.dumps(result), flush=True)
        receipt_path = op.canonical(Path(result['audit_dir']) / 'launch-receipt.json')
        if not receipt_path.is_relative_to(ENGINE / '.local/lodo_launch/receipts'):
            raise ValueError('Frozen dispatcher returned a foreign receipt')
        record, digest = op.private_record(receipt_path)
        plan = record['plan']
        if (plan['config'] != relative or plan['root'] != str(ENGINE)
                or plan['implementation_commit'] != execution_commit() or plan['schema_version'] != 2
                or record['session'] != result['session'] or record['stdout'] != result['log']
                or plan['run_dir'] != result['run_dir'] or result['log'] != str(receipt_path.parent / 'stdout.log')):
            raise ValueError('Frozen dispatcher returned an incompatible plan')
        job = dict(kind=kind, config=relative, session=result['session'], log=result['log'], run_dir=result['run_dir'],
                   launch_receipt=str(receipt_path), launch_receipt_sha256=digest)
    base = op.canonical(ROOT / '.local/lodo_console/jobs'); base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix='attempt-', dir=base)) / 'job.json'
    session = 'lodo-console-' + uuid.uuid4().hex[:12]
    job.update(execution_root=str(ENGINE), execution_commit=execution_commit(), release=reviewed,
               supervisor_session=session, created_ns=time.time_ns())
    if evaluate: job.update(session=session, log=str(path.parent / 'stdout.log'))
    op.write_private(path, job); status(path, 'DISPATCHED')
    print('Run:', job['run_dir'], '\nSession:', job['session'], '\nAudit:', path.parent, flush=True)
    try:
        start_supervisor(path, '--_evaluation' if evaluate else '--_watch', session)
    except Exception as error:
        status(path, 'REGISTRATION_FAILED', message='Supervisor could not start; training may already be running: ' + str(error), exit_code=1)
        raise ValueError('Supervisor failed; do not relaunch blindly. Inspect ' + str(path.parent)) from error
    return 0 if no_follow else follow(path)


def evaluation_main():
    parser = argparse.ArgumentParser(description='Launch separately approved held-out evaluation and register its verified metrics.')
    parser.add_argument('--config', required=True); parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-follow', action='store_true'); parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    if args.monitor and args.check: parser.error('--monitor and --check are mutually exclusive')
    try:
        return launch(args.config, check=args.check, no_follow=args.no_follow, monitor=args.monitor, evaluate=True)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        print('REFUSED:', error, file=sys.stderr); return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--_watch'); group.add_argument('--_evaluation'); args = parser.parse_args()
    path = op.canonical(args._watch or args._evaluation)
    (watch_training if args._watch else execute_evaluation)(path)
