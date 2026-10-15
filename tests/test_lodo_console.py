"""CPU-only main console tests. tmux, GPU admission and dispatch are doubled.

The unchanged runner's separate suite exercises actual safety/recovery gates.
No test grants live execution scope or creates an official output namespace.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import lodo_console as console
import lodo_results as results
from test_multi_source_dataset import tree
from test_lodo_evaluation import setup_terminal, rewrite

TEST_COMMIT = '17afc5eb6477cb6b93a0553408133d6dbc6185c6'


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    main = tmp_path / 'main'; engine = tmp_path / 'engine'
    main.mkdir(); engine.mkdir()
    monkeypatch.setattr(console, 'ROOT', main)
    monkeypatch.setattr(console, 'ENGINE', engine)
    monkeypatch.setattr(results, 'ROOT', main)
    monkeypatch.setattr(results, 'ENGINE', engine)
    monkeypatch.setattr(results, 'REGISTRY', main / '.local/lodo_results')
    monkeypatch.setattr(console, 'execution_commit', lambda: TEST_COMMIT)
    return main, engine


def private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    console.op.write_private(path, value)
    return path


def job(sandbox, **overrides):
    main, engine = sandbox
    value = dict(kind='training', config='candidate.yaml', execution_root=str(engine),
                 execution_commit=TEST_COMMIT, session='trainer', supervisor_session='watcher',
                 log=str(main / 'stdout.log'), run_dir=str(engine / 'run'), release={'sha256': 'reviewed'}, created_ns=1)
    value.update(overrides)
    return private(main / '.local/lodo_console/jobs/attempt-test/job.json', value), value


def git_repository(path):
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.name', 'CPU test'], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.email', 'cpu@example.invalid'], check=True)
    (path / 'scripts').mkdir(exist_ok=True); (path / 'scripts/entry.py').write_text('pass\n')
    (path / '.gitignore').write_text('.local/\nscripts/ignored.py\n')
    subprocess.run(['git', '-C', str(path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(path), 'commit', '-qm', 'fixture'], check=True)
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


@pytest.mark.parametrize('name', sorted(console.USER_DOCUMENTS))
def test_only_preserved_user_documents_may_be_dirty(sandbox, name):
    main, _ = sandbox; head = git_repository(main)
    path = main / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('user work')
    assert 'scripts/entry.py' in console.verify_source(main, head, documents=True)
    with pytest.raises(ValueError): console.verify_source(main, head)


@pytest.mark.parametrize('change', ['modified', 'untracked', 'ignored', 'mode', 'staged', 'hidden-modified', 'alias'])
def test_source_changes_fail_closed(sandbox, change):
    main, _ = sandbox; head = git_repository(main); entry = main / 'scripts/entry.py'
    if change == 'modified': entry.write_text('print(1)')
    elif change == 'untracked': (main / 'scripts/new.py').write_text('pass')
    elif change == 'ignored': (main / 'scripts/ignored.py').write_text('pass')
    elif change == 'mode': entry.chmod(0o755)
    elif change == 'staged':
        entry.write_text('new'); subprocess.run(['git', '-C', str(main), 'add', '.'], check=True)
    elif change == 'hidden-modified':
        subprocess.run(['git', '-C', str(main), 'update-index', '--assume-unchanged', 'scripts/entry.py'], check=True)
        entry.write_text('changed behind index')
    else:
        entry.unlink(); entry.symlink_to(main / '.gitignore')
    with pytest.raises(ValueError): console.verify_source(main, head, documents=True)


@pytest.mark.parametrize('head', ['a'*40, 'short', None, True])
def test_wrong_release_head_refused(sandbox, head):
    main, _ = sandbox; git_repository(main)
    with pytest.raises(ValueError): console.verify_source(main, head)


@pytest.mark.parametrize('name', sorted(console.LOCAL_STAGE_A_CONFIGS))
def test_original_ignored_stage_a_configs_preserved_without_entering_stage_b(sandbox, name):
    main, _ = sandbox; head = git_repository(main)
    (main / '.git/info/exclude').write_text('configs/*.yaml\n')
    file = main / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text('Stage A local config')
    console.verify_source(main, head, documents=True)
    with pytest.raises(ValueError): console.verify_source(main, head)
    extra = main / 'configs/unreviewed.yaml'; extra.write_text('unknown')
    with pytest.raises(ValueError): console.verify_source(main, head, documents=True)


def test_missing_release_refuses_before_worker_or_output(sandbox, monkeypatch):
    monkeypatch.setattr(console, 'frozen_worker', lambda *a, **k: pytest.fail('No worker before review'))
    with pytest.raises(ValueError, match='release required'): console.launch('candidate.yaml')
    assert not (sandbox[0] / '.local').exists()


@pytest.mark.parametrize('field,value', [('schema_version', True), ('execution_commit', 'b'*40),
                                      ('execution_root', '/foreign'), ('frontend_root', '/foreign'),
                                      ('record_type', 'approval')])
def test_release_binding_rejects_wrong_identity(sandbox, field, value):
    main, engine = sandbox
    record = dict(schema_version=1, record_type='lodo_main_console_release', frontend_root=str(main),
                  frontend_commit=TEST_COMMIT, execution_root=str(engine), execution_commit=TEST_COMMIT)
    record[field] = value
    private(main / '.local/lodo_console/release.json', record)
    with pytest.raises(ValueError, match='release binding'): console.release()


def test_release_requires_reviewed_frontend_and_clean_engine(sandbox, monkeypatch):
    main, engine = sandbox; calls = []
    record = dict(schema_version=1, record_type='lodo_main_console_release', frontend_root=str(main),
                  frontend_commit=TEST_COMMIT, execution_root=str(engine), execution_commit=TEST_COMMIT)
    private(main / '.local/lodo_console/release.json', record)
    monkeypatch.setattr(console, 'verify_source', lambda root, commit, **kw: calls.append((root, commit, kw)) or console.CONSOLE_FILES)
    monkeypatch.setattr(console.op, 'clean_head', lambda *a: calls.append(a))
    assert console.release()['record'] == record
    assert calls == [(main, TEST_COMMIT, {'documents': True}), (engine, TEST_COMMIT)]


def test_real_git_release_accepts_preserved_main_work_and_rejects_engine_change(sandbox, monkeypatch):
    main, engine = sandbox; git_repository(main); engine_head = git_repository(engine)
    for name in console.CONSOLE_FILES: (main / name).write_text('# temporary reviewed fixture\n')
    subprocess.run(['git', '-C', str(main), 'add', 'scripts'], check=True)
    subprocess.run(['git', '-C', str(main), 'commit', '-qm', 'fixture console release'], check=True)
    head = console.git(main, 'rev-parse', 'HEAD').decode().strip()
    (main / '.git/info/exclude').write_text('configs/*.yaml\n')
    for name in console.USER_DOCUMENTS | console.LOCAL_STAGE_A_CONFIGS:
        path = main / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('preserved local work\n')
    record = dict(schema_version=1, record_type='lodo_main_console_release', frontend_root=str(main),
                  frontend_commit=head, execution_root=str(engine), execution_commit=engine_head)
    private(main / '.local/lodo_console/release.json', record)
    subprocess.run(['git', '-C', str(engine), 'fetch', '-q', str(main), head], check=True)
    subprocess.run(['git', '-C', str(engine), 'checkout', '-q', '--detach', head], check=True)
    record['execution_commit'] = head
    console.atomic(main / '.local/lodo_console/release.json', record)
    monkeypatch.setattr(console, 'execution_commit', results.execution_commit)
    assert console.execution_commit() == head
    assert results.check_frozen_source(engine) == head
    assert console.release()['record'] == record
    (engine / 'scripts/entry.py').write_text('changed engine\n')
    with pytest.raises(ValueError, match='Clean unified worktree'): console.release()


@pytest.mark.parametrize('cpu', [False, True])
def test_environment_preserves_strict_policy_and_execution_root(sandbox, cpu):
    env = console.environment(cpu=cpu)
    assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert env['PYTHONPATH'].split(':') == [str(sandbox[1]), str(sandbox[1] / 'scripts')]
    assert env['CUDA_VISIBLE_DEVICES'] == ('' if cpu else console.op.GPU)
    assert env['OMP_NUM_THREADS'] == '1' and env['PYTHONDONTWRITEBYTECODE'] == '1'


def test_worker_uses_frozen_cwd_interpreter_and_propagates_refusal(sandbox, monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs)); return SimpleNamespace(returncode=1, stderr='collision', stdout='')
    monkeypatch.setattr(console.subprocess, 'run', run)
    with pytest.raises(ValueError, match='collision'): console.frozen_worker('code', {'resume': True})
    argv, kw = calls[0]
    assert argv[:3] == [str(console.PYTHON), '-B', '-c'] and kw['cwd'] == sandbox[1]
    assert json.loads(argv[-1]) == {'resume': True}


def dispatch_fixture(sandbox, monkeypatch):
    main, engine = sandbox; relative = 'candidate.yaml'; run = engine / 'exp/run'
    audit = engine / '.local/lodo_launch/receipts/launch-test'; audit.mkdir(parents=True)
    plan = dict(schema_version=2, root=str(engine), implementation_commit=TEST_COMMIT,
                config=relative, run_dir=str(run), binding={'sha256': 'binding'}, approval={'sha256': 'scope'})
    record = dict(plan=plan, session='trainer', stdout=str(audit / 'stdout.log'))
    private(audit / 'launch-receipt.json', record)
    output = dict(session='trainer', log=record['stdout'], run_dir=str(run), audit_dir=str(audit))
    calls = []
    monkeypatch.setattr(console, 'release', lambda: {'sha256': 'reviewed'})
    monkeypatch.setattr(console, 'train_config', lambda _: relative)
    monkeypatch.setattr(console, 'frozen_worker', lambda code, values: calls.append((code, values)) or output)
    monkeypatch.setattr(console, 'start_supervisor', lambda *a: calls.append(a))
    return output, record, calls


@pytest.mark.parametrize('resume', [False, True])
def test_start_and_resume_delegate_original_runner_and_follow(sandbox, monkeypatch, resume):
    output, _, calls = dispatch_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(console, 'follow', lambda path: calls.append(('follow', path)) or 0)
    assert console.launch('candidate.yaml', resume=resume) == 0
    assert "runner.prepare(args['config'],args['resume'])" in calls[0][0]
    assert 'runner.dispatch(plan)' in calls[0][0] and calls[0][1] == dict(config='candidate.yaml', resume=resume)
    path = next((sandbox[0] / '.local/lodo_console/jobs').glob('*/job.json'))
    saved = console.load_job(path)
    assert saved['run_dir'] == output['run_dir'] and saved['log'] == output['log']
    assert calls[1][1] == '--_watch' and calls[-1] == ('follow', path)


@pytest.mark.parametrize('field', ['session', 'log', 'run_dir', 'audit_dir'])
def test_foreign_dispatch_output_rejected(sandbox, monkeypatch, field):
    output, _, _ = dispatch_fixture(sandbox, monkeypatch); output[field] = '/foreign'
    with pytest.raises((ValueError, OSError)): console.launch('candidate.yaml', no_follow=True)
    assert not (sandbox[0] / '.local/lodo_console/jobs').exists()


def test_no_follow_keeps_detached_supervisor(sandbox, monkeypatch):
    _, _, calls = dispatch_fixture(sandbox, monkeypatch)
    monkeypatch.setattr(console, 'follow', lambda *a: pytest.fail('No viewer'))
    assert console.launch('candidate.yaml', no_follow=True) == 0
    assert calls[-1][1] == '--_watch'


def test_check_never_dispatches_or_writes(sandbox, monkeypatch):
    _, _, calls = dispatch_fixture(sandbox, monkeypatch)
    assert console.launch('candidate.yaml', check=True, resume=True) == 0
    code, values = calls[0]
    assert "runner.prepare(args['config'],args['resume'],require_release=False)" in code
    assert 'runner.dispatch' not in code and "plan['environment']={key:plan['environment'][key] for key in CONTROLLED}" in code
    assert values == dict(config='candidate.yaml', resume=True) and len(calls) == 1
    assert not (sandbox[0] / '.local/lodo_console/jobs').exists()


def test_monitor_chooses_latest_attempt_without_dispatch(sandbox, monkeypatch):
    first, value = job(sandbox)
    second = private(first.parent.parent / 'attempt-new/job.json', dict(value, created_ns=2))
    monkeypatch.setattr(console, 'train_config', lambda _: 'candidate.yaml')
    monkeypatch.setattr(console, 'release', lambda: pytest.fail('Monitor cannot launch'))
    monkeypatch.setattr(console, 'follow', lambda path: 0 if path == second else pytest.fail('Old log'))
    assert console.launch('candidate.yaml', monitor=True) == 0


def test_supervisor_command_quotes_paths_and_cpu_env(sandbox, monkeypatch):
    calls = []; path = sandbox[0] / "path with ' quote/job.json"
    monkeypatch.setattr(console.subprocess, 'run', lambda *a, **kw: calls.append((a, kw)))
    console.start_supervisor(path, '--_watch', 'unique-session')
    argv = calls[0][0][0]
    import shlex
    assert argv[:5] == ['tmux', 'new-session', '-d', '-s', 'unique-session']
    assert str(path) in shlex.split(argv[-1])
    assert 'CUDA_VISIBLE_DEVICES=' in argv and 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in argv
    assert 'PYTHONHOME' in shlex.split(argv[-1]) and 'LODO_ADMISSION_RECEIPT' in shlex.split(argv[-1])
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == ''


def test_follow_drains_final_log_and_returns_status(sandbox, capsys):
    path, value = job(sandbox); Path(value['log']).write_text('Epoch 1\nUpdate 85200 complete\n')
    console.status(path, 'COMPLETE', exit_code=0)
    assert console.follow(path) == 0
    assert 'Update 85200 complete' in capsys.readouterr().out


def test_ctrl_c_stops_only_viewer(sandbox, monkeypatch, capsys):
    path, _ = job(sandbox)
    monkeypatch.setattr(console, 'session_alive', lambda _: True)
    monkeypatch.setattr(console.time, 'sleep', lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert console.follow(path) == 0
    assert 'Job continues' in capsys.readouterr().out
    assert not (path.parent / 'status.json').exists()


def test_dead_supervisor_does_not_leave_viewer_hanging(sandbox, monkeypatch):
    path, _ = job(sandbox)
    monkeypatch.setattr(console, 'session_alive', lambda _: False)
    monkeypatch.setattr(console.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='may still be running'): console.follow(path)


@pytest.mark.parametrize('returncode,stderr,expected', [(0, '', True), (1, "can't find session: x", False),
                                                     (1, 'no server running on socket', False), (1, 'permission denied', None)])
def test_tmux_state_errors_are_not_false_completion(monkeypatch, returncode, stderr, expected):
    monkeypatch.setattr(console.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=returncode, stderr=stderr))
    if expected is None:
        with pytest.raises(ValueError): console.session_alive('session')
    else: assert console.session_alive('session') is expected


@pytest.mark.parametrize('terminal,admitted', [(False, False), (True, False), (True, True)])
def test_watcher_registers_only_completed_admitted_run(sandbox, monkeypatch, terminal, admitted):
    path, value = job(sandbox); calls = []
    if terminal:
        file = Path(value['run_dir']) / 'checkpoints/ckpt_terminal.pth'; file.parent.mkdir(parents=True); file.write_bytes(b'test')
    monkeypatch.setattr(console, 'release', lambda: value['release'])
    monkeypatch.setattr(console, 'session_alive', lambda _: False)
    def audit(_):
        if not admitted: raise ValueError('No admission')
    monkeypatch.setattr(console, 'training_audit', audit)
    monkeypatch.setattr(results, 'register_run', lambda run: calls.append(run) or ['verified'])
    console.watch_training(path)
    state = console.op.private_record(path.parent / 'status.json')[0]['state']
    assert state == ('FAILED' if not terminal else 'COMPLETE' if admitted else 'REGISTRATION_FAILED')
    assert bool(calls) == (terminal and admitted)


def sample_record(**overrides):
    row = dict(architecture='resnet50', fold='LODO-1', run_dir='/run/unified-test',
               record_type='lodo_training_completion', checkpoint_sha256='a'*64, primary_updates=85200,
               metrics=None, historical_import=False)
    row.update(overrides); return row


def test_registry_idempotent_and_training_has_no_target_metrics(sandbox):
    row = sample_record(); first = results.publish([row]); before = Path(first[0]).read_bytes()
    assert results.publish([row]) == first and Path(first[0]).read_bytes() == before
    assert len(results.registry_records()) == 1 and results.registry_records()[0]['metrics'] is None
    assert '85200' in (results.REGISTRY / 'README.md').read_text()
    assert Path(first[0]).stat().st_mode & 0o777 == 0o600


def test_evaluation_registers_separately_with_all_metrics_and_caveats(sandbox):
    train = sample_record()
    evaluation = sample_record(record_type='lodo_evaluation_registration', result='/eval/lodo_result.json',
        comparison_status='incomplete', metrics=dict(mAP=.4131, Rank1=.685, Rank5=.8126, Rank10=.8581, mINP=.0973),
        caveats=['Historical approval trail preserved.'])
    results.publish([train, evaluation]); text = (results.REGISTRY / 'README.md').read_text()
    assert len(results.registry_records()) == 2
    for value in ('41.31%', '68.50%', '81.26%', '85.81%', '9.73%', 'incomplete', 'Historical approval trail'): assert value in text


def test_corrupt_registry_fails_before_new_publication(sandbox):
    first = results.publish([sample_record()]); path = Path(first[0]); value = json.loads(path.read_text())
    value['record']['primary_updates'] = 1; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='integrity'): results.publish([sample_record(fold='LODO-2')])
    assert len(list(path.parent.glob('*.json'))) == 1


@pytest.mark.parametrize('alias', ['symlink', 'traversal'])
def test_registry_alias_paths_refused(sandbox, alias):
    main, _ = sandbox; real = main / 'real'; real.mkdir()
    if alias == 'symlink':
        path = main / 'alias'; path.symlink_to(real)
    else: path = real / '../escape'
    with pytest.raises(ValueError): results.canonical(path)


def test_unadmitted_maintained_run_cannot_register(sandbox):
    run = sandbox[1] / 'exp/lodo/v1/resnet50/LODO-1/seed42/unified-test'
    with pytest.raises(ValueError, match='No scoped actual'): results.register_run(run)
    assert not results.REGISTRY.exists()


def test_evaluation_check_is_metadata_only_in_cuda_hidden_child(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(console, 'frozen_worker', lambda *a, **k: calls.append((a, k)) or {'status': 'ADMITTED_CPU_EVALUATION'})
    assert console.evaluation_check(Path('/approved/request.yaml'))['status'] == 'ADMITTED_CPU_EVALUATION'
    code, values = calls[0][0]
    assert 'with metadata_only()' in code and 'evaluation_admission' in code
    assert values['request'] == '/approved/request.yaml' and values['training_root'] == str(console.ENGINE)
    assert calls[0][1] == {'cpu': True, 'source': console.ROOT}


def test_evaluation_needs_separate_admission_before_job_creation(sandbox, monkeypatch):
    monkeypatch.setattr(console, 'release', lambda: {'sha256': 'reviewed'})
    monkeypatch.setattr(console, 'evaluation_request', lambda _: Path('/request.yaml'))
    def refused(_): raise ValueError('No separate evaluation scope')
    monkeypatch.setattr(console, 'evaluation_check', refused)
    with pytest.raises(ValueError, match='separate evaluation'): console.launch('request.yaml', evaluate=True)
    assert not (sandbox[0] / '.local').exists()


def test_evaluation_failure_never_registers_metrics(sandbox, monkeypatch):
    request = sandbox[1] / 'request.yaml'; request.write_text('request')
    admission = {'output_dir': str(sandbox[1] / 'evaluation')}
    path, value = job(sandbox, kind='evaluation', request=str(request), request_sha256=console.op.sha(request), admission=admission)
    monkeypatch.setattr(console, 'release', lambda: value['release'])
    monkeypatch.setattr(console, 'evaluation_check', lambda _: admission)
    def run(argv, **kw):
        assert argv[2] == str(sandbox[1] / 'scripts/evaluate.py') and kw['env']['CUDA_VISIBLE_DEVICES'] == ''
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(console.subprocess, 'run', run)
    monkeypatch.setattr(results, 'register_run', lambda *a, **k: pytest.fail('Failed evaluation cannot register'))
    console.execute_evaluation(path)
    state = console.op.private_record(path.parent / 'status.json')[0]
    assert state['state'] == 'FAILED' and state['exit_code'] == 3


def test_evaluation_success_registers_frozen_output(sandbox, monkeypatch):
    request = sandbox[1] / 'request.yaml'; request.write_text('request')
    admission = {'output_dir': str(sandbox[1] / 'evaluation')}
    path, value = job(sandbox, kind='evaluation', request=str(request), request_sha256=console.op.sha(request), admission=admission)
    monkeypatch.setattr(console, 'release', lambda: value['release'])
    monkeypatch.setattr(console, 'evaluation_check', lambda _: admission)
    monkeypatch.setattr(console.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    calls = []
    monkeypatch.setattr(results, 'register_run', lambda run, **kw: calls.append((run, kw)) or ['verified'])
    console.execute_evaluation(path)
    assert calls == [(Path(value['run_dir']), {'result': Path(admission['output_dir']) / 'lodo_result.json'})]
    assert console.op.private_record(path.parent / 'status.json')[0]['state'] == 'COMPLETE'


def admitted_job(sandbox, monkeypatch):
    output, record, _ = dispatch_fixture(sandbox, monkeypatch)
    receipt = Path(output['audit_dir']) / 'launch-receipt.json'
    path, value = job(sandbox, run_dir=output['run_dir'], launch_receipt=str(receipt),
                      launch_receipt_sha256=console.op.sha(receipt))
    plan = record['plan']
    admission = dict(status='ADMITTED_TRAINING', root=plan['root'], commit=TEST_COMMIT,
                     config=plan['config'], run_dir=plan['run_dir'], binding=plan['binding'], approval=plan['approval'])
    private(receipt.parent / 'execution-admission.json', admission)
    return path, value, record, admission


@pytest.mark.parametrize('field', ['status', 'root', 'commit', 'config', 'run_dir', 'binding', 'approval'])
def test_mismatched_actual_training_admission_refuses_registration(sandbox, monkeypatch, field):
    _, value, _, admission = admitted_job(sandbox, monkeypatch)
    path = Path(value['launch_receipt']).parent / 'execution-admission.json'
    admission[field] = 'foreign'; console.atomic(path, admission)
    with pytest.raises(ValueError, match='admission'): console.training_audit(value)
    with pytest.raises(ValueError, match='No valid training audit'): results.new_run_audit(Path(value['run_dir']))


def test_registry_pins_actual_training_and_separate_evaluation_receipts(sandbox, monkeypatch):
    path, value, _, _ = admitted_job(sandbox, monkeypatch)
    audit = results.new_run_audit(Path(value['run_dir']))
    assert audit['training']['launch_receipt_sha256'] == value['launch_receipt_sha256']
    output = sandbox[1] / 'eval'; result = output / 'lodo_result.json'; admission = {'output_dir': str(output), 'approval': 'separate-scope'}
    private(path.parent.parent / 'attempt-eval/job.json', dict(value, kind='evaluation', admission=admission, created_ns=2))
    with pytest.raises(ValueError, match='No actual frozen evaluator completion'): results.new_run_audit(Path(value['run_dir']), result)
    base = sandbox[1] / '.local/lodo_evaluation/v2/evaluation-test'
    private(base / 'execution-admission.json', admission)
    private(base / 'completion.json', dict(status='COMPLETED', result={'result': str(result)}))
    verified = results.new_run_audit(Path(value['run_dir']), result)
    assert verified['evaluation']['completion_sha256'] == console.op.sha(base / 'completion.json')


def test_console_modules_import_without_torch():
    process = subprocess.run([str(console.PYTHON), '-B', '-c',
        'import lodo_console,lodo_results,sys; assert "torch" not in sys.modules'], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


@pytest.fixture
def registry_artifacts(sandbox, setup_terminal, monkeypatch):
    """Fabricated small CPU metadata, never a real official run or approval."""
    import shutil
    import yaml
    from reid.utils import lodo_evaluation as ev
    from reid.utils import lodo_checkpoint as ck
    _, engine = sandbox
    request, _, _, _ = setup_terminal()
    run = engine / 'exp/lodo/v1/resnet50/LODO-1/seed42/unified-fixture'
    run.mkdir(parents=True)
    shutil.copytree(Path(request['checkpoint']).parent, run / 'checkpoints')
    terminal = run / 'checkpoints/ckpt_terminal.pth'
    rewrite(terminal, lambda p: p['cfg']['experiment'].update(output_dir=str(run)))
    payload, proof = ck.read_checkpoint(terminal); identity = payload['identity']
    config = engine / 'config.yaml'; config.write_text(yaml.safe_dump(payload['cfg']))
    imm = dict(configuration={'scientific_sha256': identity['config_digest']},
               dataset={k: identity[k] for k in ('map_digest', 'data_digest', 'num_classes')})
    manifest = engine / 'manifest.json'
    manifest.write_text(json.dumps({'immutable': imm, 'immutable_sha256': ck.digest(imm)}))
    rows = [dict(architecture=arch, fold=f'LODO-{fold}', target='market1501', config='config.yaml', manifest='manifest.json',
                 config_file_sha256=results.sha(config), manifest_immutable_sha256=ck.digest(imm))
            for arch in ('resnet50', 'pcb') for fold in range(1, 5)]
    index = engine / 'docs/lodo_phase10_evidence/candidate-index.json'; index.parent.mkdir(parents=True)
    index.write_text(json.dumps({'runs': rows}))
    # Source/candidate freeze is separately tested against real Git fixtures and
    # real frozen sources; only these two fixture boundaries are substituted.
    monkeypatch.setattr(results, 'check_frozen_source', lambda _: TEST_COMMIT)
    import scripts.lodo_official_preparation as prep
    monkeypatch.setattr(prep, 'validate_envelope', lambda _: None)
    output = engine / 'exp/lodo_eval/v1/benchmark/resnet50/LODO-1/seed42/unified-fixture'
    output.mkdir(parents=True); comparison = output / 'comparison.json'; comparison.write_text('{"status":"incomplete"}')
    representation = ev.retrieval_contract(payload['cfg'])
    eval_identity = dict(schema_version=1, target=ev.target_metadata(payload['cfg'], identity, payload['cfg']['data']['root']),
                         representation=representation, batch_size=32, precision='fp32')
    result = dict(schema_version=1, record_type='lodo_held_out_result', architecture='resnet50', fold='LODO-1', seed=42,
                  synthetic=False, authoritative=True, sources=identity['sources'], target=identity['target'],
                  terminal=dict(path=str(terminal), **proof), representation=representation,
                  training={k: identity[k] for k in ('config_digest', 'map_digest', 'sidecar_digest', 'data_digest', 'num_classes')},
                  evaluation=dict(config_digest=ck.digest(eval_identity), configuration=eval_identity),
                  metrics={k: .5 for k in ev.METRICS}, metric_units='fraction',
                  artifacts=dict(result=str(output / 'lodo_result.json'), comparison_sha256=results.sha(comparison)), caveats=['Synthetic fixture only.'])
    result['training']['source_provenance'] = identity['provenance']
    ev.publish_result(output / 'lodo_result.json', result)
    return engine, run, output / 'lodo_result.json', result


def test_full_registry_inspection_checks_terminal_without_images_or_cuda(registry_artifacts):
    engine, run, path, _ = registry_artifacts
    import torch
    records = results.inspect_artifacts(engine, run, path)
    assert len(records) == 2 and records[0]['primary_updates'] == 85200
    assert records[0]['metrics'] is None and records[1]['metrics']['mAP'] == .5
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('failure', ['synthetic', 'terminal', 'map', 'sidecar', 'target', 'metrics', 'representation',
                                    'eval_digest', 'precision', 'source_provenance', 'result_path', 'result_hash'])
def test_registry_rejects_mismatched_results_even_with_resealed_json(registry_artifacts, failure):
    engine, run, path, record = registry_artifacts
    from reid.utils.lodo_checkpoint import digest
    if failure == 'synthetic': record['synthetic'] = True
    elif failure == 'terminal': record['terminal']['checkpoint_sha256'] = '0'*64
    elif failure == 'map': record['training']['map_digest'] = '0'*64
    elif failure == 'sidecar': record['training']['sidecar_digest'] = '0'*64
    elif failure == 'target': record['target'] = 'duke'
    elif failure == 'metrics': record['metrics']['mAP'] = 1.1
    elif failure == 'representation': record['representation']['distance'] = 'cosine'
    elif failure == 'eval_digest': record['evaluation']['config_digest'] = '0'*64
    elif failure == 'precision':
        record['evaluation']['configuration']['precision'] = 'fp16'
        record['evaluation']['config_digest'] = digest(record['evaluation']['configuration'])
    elif failure == 'source_provenance': record['training']['source_provenance'] = {}
    elif failure == 'result_path': record['artifacts']['result'] = '/foreign/result.json'
    path.write_text(json.dumps(record))
    path.with_name('lodo_result.sha256.json').write_text(json.dumps(dict(schema_version=1, sha256='0'*64 if failure == 'result_hash' else results.sha(path))))
    with pytest.raises(ValueError): results.inspect_artifacts(engine, run, path)
    assert not results.REGISTRY.exists()


def test_registry_rejects_incomplete_terminal(registry_artifacts):
    engine, run, _, _ = registry_artifacts
    rewrite(run / 'checkpoints/ckpt_terminal.pth', lambda p: p['progress'].update(completed_primary_updates=85199))
    with pytest.raises(ValueError): results.inspect_artifacts(engine, run)


@pytest.mark.parametrize('change', ['missing', 'different_commit', 'short_commit', 'public_permissions', 'unknown_key'])
def test_shared_release_pin_fails_closed(sandbox, change):
    main, engine = sandbox
    record = dict(schema_version=1, record_type='lodo_main_console_release', frontend_root=str(main),
                  frontend_commit=TEST_COMMIT, execution_root=str(engine), execution_commit=TEST_COMMIT)
    if change != 'missing':
        if change == 'different_commit': record['execution_commit'] = 'b'*40
        if change == 'short_commit': record['frontend_commit'] = record['execution_commit'] = 'short'
        if change == 'unknown_key': record['extra'] = 'unreviewed'
        path = private(main / '.local/lodo_console/release.json', record)
        if change == 'public_permissions': path.chmod(0o644)
    with pytest.raises(ValueError): results.execution_commit()


def test_historical_source_does_not_require_a_new_execution_release(sandbox, monkeypatch):
    historical = sandbox[0].parent / 'historical'; historical.mkdir()
    head = git_repository(historical)
    monkeypatch.setattr(results, 'HISTORICAL', historical)
    monkeypatch.setattr(results, 'HISTORICAL_COMMIT', head)
    assert results.check_frozen_source(historical) == head
    assert not (sandbox[0] / '.local/lodo_console').exists()
