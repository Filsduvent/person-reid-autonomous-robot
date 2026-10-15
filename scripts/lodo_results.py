"""Verified Stage B registry. Never trains, evaluates, or selects checkpoints.

Scientific validation runs in a fresh CUDA-hidden subprocess using the unchanged
source at the artifact's frozen checkout. Results stay outside tracked source.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.parent / 'person-reid-autonomous-robot-lodo-console-execution'
HISTORICAL = ROOT.parent / 'person-reid-autonomous-robot-lodo'
HISTORICAL_COMMIT = '4a03fb0d37cec2e41af92b8ff6e681c91ff5c387'
PYTHON = '/home/addirakoze/environments/Reid/bin/python'
REGISTRY = ROOT / '.local/lodo_results'


def canonical(path):
    path = Path(path).absolute()
    if '..' in path.parts or path.resolve() != path:
        raise ValueError('Registry artifact alias/traversal prohibited')
    return path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_release_record():
    """Read the private external release pin without importing scientific code."""
    path = canonical(ROOT / '.local/lodo_console/release.json')
    if not path.is_file():
        raise ValueError('Reviewed main-console release required; no experiment launched')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Release record must be owned, private and regular (0600)')
        raw = stream.read()
    record = json.loads(raw)
    keys = {'schema_version', 'record_type', 'frontend_root', 'frontend_commit', 'execution_root', 'execution_commit'}
    if (not isinstance(record, dict) or set(record) != keys or type(record['schema_version']) is not int
            or record['schema_version'] not in (1, 2) or record['record_type'] != 'lodo_main_console_release'
            or record['frontend_root'] != str(ROOT) or record['execution_root'] != str(ENGINE)
            or not isinstance(record['frontend_commit'], str)
            or re.fullmatch('[0-9a-f]{40}', record['frontend_commit']) is None
            or not isinstance(record['execution_commit'], str)
            or re.fullmatch('[0-9a-f]{40}', record['execution_commit']) is None
            or (record['schema_version'] == 1 and record['execution_commit'] != record['frontend_commit'])):
        raise ValueError('Invalid main-console release binding')
    return record, hashlib.sha256(raw).hexdigest()


def execution_commit():
    # Both checkouts use the actual release, stored externally to avoid a
    # self-referential SHA. No implicit fallback to the old integration commit.
    return read_release_record()[0]['execution_commit']


def check_frozen_source(root):
    root = canonical(root)
    expected = execution_commit() if root == ENGINE else HISTORICAL_COMMIT if root == HISTORICAL else None
    if expected is None:
        raise ValueError('Unsupported artifact source checkout')
    git = lambda *a: subprocess.check_output(['git', '-C', str(root), *a])
    if git('rev-parse', 'HEAD').decode().strip() != expected or git('status', '--porcelain=v1', '-uall'):
        raise ValueError('Artifact source checkout must retain its clean frozen HEAD')
    tracked = set()
    for entry in filter(None, git('ls-tree', '-rz', expected).split(b'\0')):
        header, raw = entry.split(b'\t', 1); mode, kind, blob = header.decode().split()
        name = raw.decode(); path = canonical(root / name); tracked.add(name)
        if kind != 'blob' or mode not in ('100644', '100755') or not path.is_file():
            raise ValueError('Unsupported artifact-source entry')
        data = path.read_bytes()
        if (hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != blob
                or bool(path.stat().st_mode & 0o111) != (mode == '100755')):
            raise ValueError('Artifact source changed: ' + name)
    current = {str(p.relative_to(root)) for base in ('reid', 'scripts', 'tests', 'configs')
               for p in (root / base).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')}
    if not current <= tracked:
        raise ValueError('Untracked executable artifact source')
    return expected


def inspect_artifacts(root, run, result=None):
    """CPU-only verification; this function must run in a fresh child."""
    root = canonical(root); run = canonical(run); commit = check_frozen_source(root)
    parts = run.relative_to(root / 'exp/lodo/v1').parts
    if (len(parts) != 4 or parts[0] not in ('resnet50', 'pcb') or parts[1] not in ('LODO-1', 'LODO-2', 'LODO-3', 'LODO-4')
            or parts[2] != 'seed42' or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', parts[3])):
        raise ValueError('Invalid official training namespace')
    if root == HISTORICAL and parts != ('resnet50', 'LODO-1', 'seed42', 'official-candidate-20261011'):
        raise ValueError('Historical import is scoped to the verified ResNet50 LODO-1 run')
    if root == ENGINE and not parts[3].startswith('unified-'):
        raise ValueError('Maintained-run namespace must begin unified-')
    # The reviewed frontend owns result validation (including GPU execution
    # metadata). The frozen training root still owns config/data/checkpoint
    # provenance and is independently verified above.
    sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
    import torch
    from reid.utils.config import load_config
    from reid.utils.lodo_protocol import configuration_identity
    from reid.utils.lodo_evaluation import validate_terminal, read_result, target_metadata, retrieval_contract
    from reid.utils.lodo_checkpoint import digest
    from scripts.lodo_official_preparation import metadata_only, validate_envelope
    if torch.cuda.is_initialized(): raise ValueError('Registry inspection must not initialize CUDA')
    rows = json.loads((root / 'docs/lodo_phase10_evidence/candidate-index.json').read_text())['runs']
    candidates = [r for r in rows if r['architecture'] == parts[0] and r['fold'] == parts[1]]
    if len(rows) != 8 or len(candidates) != 1: raise ValueError('Invalid frozen candidate membership')
    row = candidates[0]; manifest = json.loads((root / row['manifest']).read_text()); validate_envelope(manifest)
    if sha(root / row['config']) != row['config_file_sha256'] or manifest['immutable_sha256'] != row['manifest_immutable_sha256']:
        raise ValueError('Configuration/index/manifest mismatch')
    cfg = load_config(str(root / row['config'])); imm = manifest['immutable']
    if configuration_identity(cfg) != imm['configuration']['scientific_sha256']:
        raise ValueError('Scientific configuration mismatch')
    request = dict(mode='lodo_held_out', version=1, architecture=parts[0], fold=parts[1], target=row['target'],
                   checkpoint=str(run / 'checkpoints/ckpt_terminal.pth'), synthetic=False,
                   expected=dict(config_digest=imm['configuration']['scientific_sha256'], map_digest=imm['dataset']['map_digest'],
                                 data_digest=imm['dataset']['data_digest'], num_classes=imm['dataset']['num_classes']),
                   output_dir='NOT_EXECUTED', num_workers=0)
    with metadata_only():
        payload, proof, terminal = validate_terminal(request)
        actual_run = Path(payload['cfg']['experiment']['output_dir'])
        if canonical(actual_run if actual_run.is_absolute() else root / actual_run) != run:
            raise ValueError('Terminal belongs to a different run')
        identity = payload['identity']; progress = payload['progress']
        common = dict(schema_version=1, protocol='lodo_v1', architecture=parts[0], fold=parts[1], seed=42,
                      sources=identity['sources'], target=identity['target'], source_root=str(root), source_commit=commit,
                      config=row['config'], config_sha256=sha(root / row['config']), manifest_sha256=sha(root / row['manifest']),
                      run_dir=str(run), checkpoint=str(terminal), checkpoint_sha256=proof['checkpoint_sha256'],
                      checkpoint_sidecar_sha256=sha(terminal.with_suffix('.json')), selection='predetermined_exact_budget_terminal',
                      scientific_sha256=identity['config_digest'], map_sha256=identity['map_digest'],
                      sidecar_sha256=identity['sidecar_digest'], data_sha256=identity['data_digest'],
                      num_classes=identity['num_classes'], feature_dim=identity['feature_dim'],
                      historical_import=root == HISTORICAL)
        images = dict(progress['exposure']['presentations']['successful'])
        training = dict(common, record_type='lodo_training_completion', status='complete',
                        primary_updates=progress['completed_primary_updates'], center_updates=progress['auxiliary_steps'],
                        successful_images=sum(images.values()), per_source_successful_images=images,
                        metrics=None, metric_status='not_evaluated_in_this_training_record')
        records = [training]
        if result is not None:
            path = canonical(result)
            prefix = root / 'exp/lodo_eval/v1/benchmark' / parts[0] / parts[1] / 'seed42'
            if path.parent.parent != prefix or path.name != 'lodo_result.json':
                raise ValueError('Foreign held-out result namespace')
            measured = read_result(path)
            evaluated = measured['evaluation']['configuration']
            # The CPU reference has no execution field. GPU feature extraction
            # must identify the exact UUID/policy and keep ranking on CPU.
            from reid.utils.lodo_evaluation_device import execution_spec
            measured_device = measured.get('environment', {}).get('device', 'cpu')
            if measured_device not in ('cpu', 'cuda'):
                raise ValueError('Unsupported evaluation execution device')
            if measured_device == 'cuda':
                execution = execution_spec(dict(device='cuda'))
                evidence = measured['environment'].get('cuda_admission', {})
                if (evaluated.get('execution') != execution
                        or evidence.get('uuid') != execution['gpu_uuid']
                        or evidence.get('evaluation_execution') != execution):
                    raise ValueError('GPU evaluation execution provenance mismatch')
            elif 'execution' in evaluated:
                raise ValueError('CPU evaluation cannot claim GPU execution')
            if (measured['synthetic'] is not False or measured['authoritative'] is not True
                    or measured['metric_units'] != 'fraction' or measured['seed'] != 42
                    or measured['terminal'] != dict(path=str(terminal), **proof)
                    or measured['representation'] != retrieval_contract(payload['cfg'])
                    or measured['evaluation']['config_digest'] != digest(evaluated)
                    or evaluated['representation'] != measured['representation'] or evaluated['precision'] != 'fp32'
                    or measured['training']['source_provenance'] != identity['provenance']
                    or measured['architecture'] != parts[0] or measured['fold'] != parts[1]
                    or measured['sources'] != identity['sources'] or measured['target'] != identity['target']
                    or measured['terminal']['path'] != str(terminal) or measured['terminal']['checkpoint_sha256'] != proof['checkpoint_sha256']
                    or measured['training']['config_digest'] != identity['config_digest']
                    or measured['training']['map_digest'] != identity['map_digest']
                    or measured['training']['sidecar_digest'] != identity['sidecar_digest']
                    or measured['training']['data_digest'] != identity['data_digest']
                    or measured['training']['num_classes'] != identity['num_classes']
                    or measured['artifacts']['result'] != str(path)
                    or measured['evaluation']['configuration']['target'] != target_metadata(payload['cfg'], identity, payload['cfg']['data']['root'])):
                raise ValueError('Held-out metrics/terminal/provenance mismatch')
            comparison = json.loads(path.with_name('comparison.json').read_text())
            caveats = list(measured['caveats'])
            if root == HISTORICAL:
                caveats.append('Historical evaluation has no separate experiment-specific post-training declaration; its existing approval trail is preserved, not reconstructed.')
            records.append(dict(common, record_type='lodo_evaluation_registration', status='complete',
                                result=str(path), result_sha256=sha(path), result_proof_sha256=sha(path.with_name('lodo_result.sha256.json')),
                                comparison_sha256=sha(path.with_name('comparison.json')), comparison_status=comparison['status'],
                                metrics=measured['metrics'], metric_units=measured['metric_units'], caveats=caveats))
            if measured_device == 'cuda':
                records[-1]['evaluation_execution'] = evaluated['execution']
    if torch.cuda.is_initialized(): raise ValueError('Registry inspection initialized CUDA')
    return records


def registry_records():
    from lodo_console import op
    records = []
    for path in (REGISTRY / 'records').glob('*.json'):
        envelope, _ = op.private_record(path)
        if (set(envelope) != {'schema_version', 'sha256', 'record'} or envelope['schema_version'] != 1
                or envelope['sha256'] != op.digest(envelope['record']) or path.stem != envelope['sha256']):
            raise ValueError('Registry record integrity mismatch: ' + str(path))
        records.append(envelope['record'])
    return sorted(records, key=lambda r: (r['architecture'], r['fold'], r['run_dir'], r['record_type']))


def human_report(records):
    lines = ['# Stage B LODO results', '', 'Verified at registration; use `python scripts/lodo_results.py --verify` to revalidate artifacts.',
             'Training completion and separately authorized held-out metrics are distinct records. No target-based checkpoint selection.', '',
             '| Architecture | Fold | Run | Record | Updates | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        metrics = row.get('metrics') or {}
        score = lambda k: f"{100 * metrics[k]:.2f}%" if k in metrics else '—'
        lines.append(f"| {row['architecture']} | {row['fold']} | {Path(row['run_dir']).name} | {row['record_type']} | {row.get('primary_updates', '—')} | {score('mAP')} | {score('Rank1')} | {score('Rank5')} | {score('Rank10')} | {score('mINP')} |")
    for row in records:
        lines += ['', f"## {row['record_type']}: {row['architecture']} {row['fold']}", '',
                  f"Run: `{row['run_dir']}`", f"Terminal SHA256: `{row['checkpoint_sha256']}`"]
        if row.get('result'): lines += [f"Result: `{row['result']}`", f"Comparison: {row['comparison_status']}"]
        lines += [f'- {c}' for c in row.get('caveats', [])]
    return '\n'.join(lines) + '\n'


def publish(records):
    from lodo_console import op, atomic
    canonical(REGISTRY).mkdir(parents=True, exist_ok=True); canonical(REGISTRY / 'records').mkdir(exist_ok=True)
    fd = os.open(REGISTRY / 'registry.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    ids = []
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        registry_records()  # Refuse a corrupt existing registry before any new publication.
        for row in records:
            digest = op.digest(row); path = REGISTRY / 'records' / (digest + '.json')
            envelope = dict(schema_version=1, sha256=digest, record=row)
            if path.exists():
                if op.private_record(path)[0] != envelope: raise ValueError('Registry collision')
            else: op.write_private(path, envelope)
            ids.append(str(path))
        # Human view is derived exclusively from immutable records. Publication is atomic.
        view = human_report(registry_records()); path = canonical(REGISTRY / 'README.md')
        atomic(REGISTRY / 'index.json', dict(schema_version=1, records=registry_records()))
        import tempfile
        fd_view, temp = tempfile.mkstemp(prefix='.view-', dir=REGISTRY)
        try:
            with os.fdopen(fd_view, 'w') as stream: stream.write(view); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp): os.unlink(temp)
    finally:
        os.close(fd)
    return ids


def new_run_audit(run, result=None):
    from lodo_console import ROOT as front, load_job, training_audit, op
    training = []; evaluations = []
    for path in (front / '.local/lodo_console/jobs').glob('*/job.json'):
        job = load_job(path)
        if job['run_dir'] != str(run): continue
        if job['kind'] == 'training': training.append(job)
        elif result is not None and Path(job['admission']['output_dir']) / 'lodo_result.json' == result: evaluations.append(job)
    if not training: raise ValueError('No scoped actual training admission for this maintained run')
    failures = []
    for job in sorted(training, key=lambda j: j['created_ns'], reverse=True):
        try:
            plan = training_audit(job)
            launch = Path(job['launch_receipt'])
            audit = dict(training=dict(launch_receipt=str(launch), launch_receipt_sha256=job['launch_receipt_sha256'],
                         admission=str(launch.parent / 'execution-admission.json'),
                         admission_sha256=op.sha(launch.parent / 'execution-admission.json'),
                         binding_sha256=plan['binding']['sha256'], operator_declaration_sha256=plan['approval']['sha256']))
            break
        except (ValueError, OSError, KeyError) as error: failures.append(str(error))
    else: raise ValueError('No valid training audit: ' + '; '.join(failures))
    if result is not None:
        if not evaluations: raise ValueError('No separately scoped evaluation job')
        for path in (ENGINE / '.local/lodo_evaluation/v2').glob('*/completion.json'):
            completion, _ = op.private_record(path)
            admission, _ = op.private_record(path.parent / 'execution-admission.json')
            if (completion.get('status') == 'COMPLETED' and completion.get('result', {}).get('result') == str(result)
                    and any(admission == job['admission'] for job in evaluations)):
                if admission.get('status') == 'ADMITTED_CUDA_EVALUATION':
                    evaluator, evaluator_sha = op.private_record(path.parent / 'evaluator-release.json')
                    matches = [job for job in evaluations if admission == job['admission']]
                    if (not any(evaluator == job['release'] for job in matches)
                            or completion.get('evaluator_release_sha256') != evaluator['sha256']):
                        raise ValueError('GPU evaluator release/completion provenance mismatch')
                audit['evaluation'] = dict(completion=str(path), completion_sha256=op.sha(path),
                                           admission=str(path.parent / 'execution-admission.json'),
                                           admission_sha256=op.sha(path.parent / 'execution-admission.json'))
                if admission.get('status') == 'ADMITTED_CUDA_EVALUATION':
                    audit['evaluation'].update(evaluator_release=str(path.parent/'evaluator-release.json'),
                                               evaluator_release_sha256=evaluator_sha)
                return audit
        raise ValueError('No actual frozen evaluator completion receipt for these metrics')
    return audit


def register_run(run, result=None):
    run = canonical(run); result = canonical(result) if result else None
    source = next((p for p in (ENGINE, HISTORICAL) if run.is_relative_to(p / 'exp/lodo/v1')), None)
    if source is None: raise ValueError('Unknown experiment root')
    audit = new_run_audit(run, result) if source == ENGINE else None
    check_frozen_source(source)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(source) + ':' + str(source / 'scripts'))
    request = dict(root=str(source), run=str(run), result=str(result) if result else None)
    process = subprocess.run([PYTHON, '-B', str(Path(__file__).resolve()), '--_inspect', json.dumps(request)],
                             cwd=source, env=env, capture_output=True, text=True)
    if process.returncode: raise ValueError('Artifact verification failed: ' + (process.stderr or process.stdout).strip())
    records = json.loads(process.stdout)
    if len(records) != (2 if result else 1): raise ValueError('Incomplete verification worker result')
    check_frozen_source(source)
    if audit:
        for row in records:
            row['operational_audit'] = audit if row.get('result') else {'training': audit['training']}
    return publish(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--register-run'); group.add_argument('--import-historical', action='store_true')
    group.add_argument('--verify', action='store_true'); group.add_argument('--_inspect', help=argparse.SUPPRESS)
    parser.add_argument('--result'); args = parser.parse_args()
    if args.result and not args.register_run: parser.error('--result requires --register-run')
    if args._inspect:
        request = json.loads(args._inspect)
        print(json.dumps(inspect_artifacts(Path(request['root']), Path(request['run']), request['result'])))
        return 0
    try:
        if args.import_historical:
            suffix = Path('resnet50/LODO-1/seed42/official-candidate-20261011')
            ids = register_run(HISTORICAL / 'exp/lodo/v1' / suffix,
                               HISTORICAL / 'exp/lodo_eval/v1/benchmark' / suffix / 'lodo_result.json')
            print('Registered:', *ids, sep='\n')
        elif args.register_run:
            print('Registered:', *register_run(Path(args.register_run), args.result), sep='\n')
        elif args.verify:
            for row in registry_records():
                # Same strict verifier; idempotent publication never changes an existing record.
                register_run(Path(row['run_dir']), row.get('result'))
            print('All registry artifacts verified.')
        print(human_report(registry_records()), end='')
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        print('REFUSED:', error, file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
