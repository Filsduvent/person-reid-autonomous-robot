"""CPU tests for opt-in CUDA evaluation and backward-compatible retrieval."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reid.utils import lodo_evaluation as ev
from reid.utils import lodo_evaluation_device as devices
from reid.engine.evaluator import evaluate_reid
from test_multi_source_dataset import tree
from test_lodo_evaluation import setup_terminal, EvalModel
from test_lodo_console import sandbox, job, registry_artifacts
from test_lodo_operational import binding_fixture, eval_fixture
from reid.utils.config import save_yaml
from reid.utils import lodo_operational as op
import lodo_results as results
import lodo_console as console


def test_device_default_and_explicit_cpu():
    assert devices.execution_spec({}) == devices.execution_spec({'device': 'cpu'})
    assert devices.execution_spec({'device': 'cuda'})['ranking_device'] == 'cpu'


@pytest.mark.parametrize('value', ['auto', 'cuda:1', 'cuda:0', 'gpu', '', None, 0])
def test_no_implicit_device_fallback(value):
    with pytest.raises(ValueError):
        devices.execution_spec({'device': value})


def test_cpu_never_queries_or_initializes_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(devices.lodo_device, 'audit_gpu', lambda *a, **k: pytest.fail('CPU queried GPU'))
    monkeypatch.setattr(devices.lodo_device, 'select_lodo_device', lambda *a: pytest.fail('CPU selected GPU'))
    assert devices.preflight({}, {}, tmp_path) is None
    assert devices.select_device({}, {}, tmp_path) == (torch.device('cpu'), None, None)


@pytest.mark.parametrize('failure', ['occupied', 'memory', 'disk', 'policy'])
def test_preflight_fails_before_cuda_and_target_images(monkeypatch, tmp_path, failure):
    cfg = dict(system={'device': 'cuda'}, protocol={'name': 'multi_source_lodo'},
               repro=dict(cuda_policy='strict_cuda_deterministic_v1', deterministic=True, benchmark=False))
    if failure == 'policy': cfg['repro']['cuda_policy'] = 'original_cuda_v1'
    def audit(*a, **k):
        if failure == 'occupied': raise ValueError('occupied')
        return dict(free_mib=200 if failure == 'memory' else 12000)
    monkeypatch.setattr(devices.lodo_device, 'audit_gpu', audit)
    monkeypatch.setattr(devices.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0 if failure == 'disk' else 10**12))
    monkeypatch.setattr(devices.lodo_device, 'select_lodo_device', lambda *a: pytest.fail('CUDA setup before admission'))
    with pytest.raises(ValueError): devices.select_device({'device': 'cuda'}, cfg, tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_gpu_selection_delegates_existing_uuid_policy_and_preserves_cfg(monkeypatch, tmp_path):
    cfg = dict(system=dict(device='cuda', gpu_id=0, amp=False), train=dict(save=dict(resume='/old')))
    before = copy.deepcopy(cfg)
    monkeypatch.setattr(devices, 'preflight', lambda *a: {'snapshot': True})
    calls = []
    def select(value):
        calls.append(value)
        return torch.device('cuda:0'), lambda: None, dict(uuid='test')
    monkeypatch.setattr(devices.lodo_device, 'select_lodo_device', select)
    device, guard, proof = devices.select_device({'device': 'cuda'}, cfg, tmp_path)
    assert device == torch.device('cuda:0') and callable(guard)
    assert calls[0]['train']['save']['resume'] == '' and cfg == before
    assert proof['evaluation_execution'] == devices.execution_spec({'device': 'cuda'})


def test_progress_includes_counts_speed_eta_and_metric_stages(capsys):
    calls = []
    progress = devices.EvaluationProgress('cuda:0', lambda: calls.append(True))
    for i in range(1, 24):
        progress.before_batch()
        progress(dict(phase='features', batch=i, batches=23, images=i*32, elapsed_seconds=i))
    progress(dict(phase='mAP_cpu'))
    output = capsys.readouterr().out
    assert '1/23' in output and '10/23' in output and '20/23' in output and '23/23' in output
    assert 'feature_eta=' in output and 'images/s' in output and 'phase=mAP_cpu' in output
    assert len(calls) == 3


def test_same_evaluator_metrics_with_and_without_progress():
    class Model(torch.nn.Module):
        def forward(self, x): return {'emb': x}
    batch = (torch.tensor([[1.,0.], [1.,0.], [0.,1.]]), torch.tensor([1,1,2]),
             torch.tensor([1,2,2]), ['q','positive','negative'], torch.tensor([0,1,1]))
    cfg = dict(eval=dict(normalize_feat=True,distance='euclidean',topk=[1,5,10],rerank={'enabled':False}))
    events, guards = [], []
    reference = evaluate_reid(cfg, Model(), [batch], torch.device('cpu'))
    actual = evaluate_reid(cfg, Model(), [batch], torch.device('cpu'), progress=events.append,
                           before_batch=lambda: guards.append(True))
    assert actual == reference and actual['mAP'] == 1 and guards == [True]
    assert [e['phase'] for e in events] == ['features','distance_cpu','mAP_cpu','CMC_cpu','mINP_cpu']


@pytest.mark.parametrize('architecture', ['resnet50', 'pcb'])
def test_cpu_reference_unchanged_and_progress_visible(setup_terminal, tmp_path, architecture, capsys):
    request, _, _, _ = setup_terminal(architecture)
    result, _ = ev.evaluate_lodo(request, project_root=tmp_path)
    assert result['environment']['device'] == 'cpu'
    assert 'execution' not in result['evaluation']['configuration']
    assert all(result['metrics'][k] == 1 for k in ev.METRICS)
    output = capsys.readouterr().out
    assert '[Evaluation] device=cpu' in output and 'metrics=' in output


def test_invalid_gpu_admission_before_model_or_target(setup_terminal, tmp_path, monkeypatch):
    request, _, _, _ = setup_terminal()
    request['device'] = 'cuda'
    def refuse(*a): raise ValueError('occupied GPU')
    monkeypatch.setattr(devices, 'select_device', refuse)
    monkeypatch.setattr(ev, 'reconstruct_model', lambda *a: pytest.fail('Premature reconstruction'))
    monkeypatch.setattr(ev, 'build_test_loader', lambda *a: pytest.fail('Premature image loader'))
    with pytest.raises(ValueError, match='occupied'): ev.evaluate_lodo(request, project_root=tmp_path)


@pytest.mark.parametrize('status,visibility', [('ADMITTED_CPU_EVALUATION', ''),
                                           ('ADMITTED_CUDA_EVALUATION', console.op.GPU)])
def test_console_forwards_gpu_only_after_pinned_admission(sandbox, monkeypatch, status, visibility):
    request = sandbox[1]/'request.yaml'; request.write_text('request')
    admission = dict(status=status, output_dir=str(sandbox[1]/'evaluation'))
    path, value = job(sandbox, kind='evaluation', request=str(request),
                      request_sha256=console.op.sha(request), admission=admission)
    monkeypatch.setattr(console, 'release', lambda: value['release'])
    monkeypatch.setattr(console, 'evaluation_check', lambda _: admission)
    def run(argv, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == visibility
        assert kwargs['env']['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        if visibility:
            assert str(console.ROOT/'scripts/lodo_gpu_evaluate_worker.py') in argv
            assert kwargs['cwd'] == console.ROOT
            assert kwargs['env']['PYTHONPATH'].split(':')[0] == str(console.ROOT)
        else:
            assert str(console.ENGINE/'scripts/evaluate.py') in argv
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(console.subprocess, 'run', run)
    console.execute_evaluation(path)
    assert console.op.private_record(path.parent/'status.json')[0]['state'] == 'FAILED'


def test_cpu_approval_cannot_be_reused_for_gpu_request(binding_fixture, monkeypatch):
    root,b,cfg,terminal,request,path,declare = eval_fixture(binding_fixture)
    declare(evaluation=(path,terminal))
    request['device'] = 'cuda'
    save_yaml(request,str(path))
    monkeypatch.setattr(devices,'preflight',lambda *a:pytest.fail('Resources checked before scoped approval'))
    with pytest.raises(ValueError,match='separate execution approval scope'):
        op.evaluation_admission(path,request,root=root)


def test_gpu_admission_stays_cpu_first_and_checks_resources(binding_fixture, monkeypatch):
    root,b,cfg,terminal,request,path,declare = eval_fixture(binding_fixture)
    request['device'] = 'cuda'; save_yaml(request,str(path)); declare(evaluation=(path,terminal))
    calls = []
    monkeypatch.setattr(ev,'validate_terminal',lambda *a:(calls.append('terminal') or
        ({'cfg':cfg},{'checkpoint_sha256':op.sha(terminal)},terminal)))
    monkeypatch.setattr(devices,'preflight',lambda *a:calls.append('resources'))
    monkeypatch.setattr(devices.lodo_device,'select_lodo_device',lambda *a:pytest.fail('CUDA in metadata check'))
    admission = op.evaluation_admission(path,request,root=root)
    assert calls == ['terminal','resources'] and admission['status'] == 'ADMITTED_CUDA_EVALUATION'


@pytest.mark.parametrize('mutation',[None,'uuid','execution','device'])
def test_registry_gpu_provenance(registry_artifacts, mutation):
    engine,run,path,record = registry_artifacts
    spec = devices.execution_spec({'device':'cuda'})
    record['evaluation']['configuration']['execution'] = spec
    record['environment'] = dict(device='cuda',cuda_admission=dict(uuid=op.GPU,evaluation_execution=copy.deepcopy(spec)))
    if mutation == 'uuid': record['environment']['cuda_admission']['uuid'] = 'GPU-other'
    if mutation == 'execution': record['evaluation']['configuration']['execution'] = dict(device='auto')
    if mutation == 'device': record['environment']['device'] = 'cpu'
    record['evaluation']['config_digest'] = ev.digest(record['evaluation']['configuration'])
    path.write_text(json.dumps(record))
    path.with_name('lodo_result.sha256.json').write_text(json.dumps(dict(schema_version=1,sha256=op.sha(path))))
    if mutation is None:
        assert results.inspect_artifacts(engine,run,path)[1]['metrics'] == record['metrics']
    else:
        with pytest.raises(ValueError):results.inspect_artifacts(engine,run,path)


def test_v2_frontend_pin_keeps_original_training_commit(sandbox):
    main, engine = sandbox
    record = dict(schema_version=2,record_type='lodo_main_console_release',frontend_root=str(main),
                  frontend_commit='a'*40,execution_root=str(engine),execution_commit='b'*40)
    path=main/'.local/lodo_console/release.json';path.parent.mkdir(parents=True)
    op.write_private(path,record)
    assert results.read_release_record()[0] == record
    assert results.execution_commit() == 'b'*40


@pytest.mark.parametrize('changed,allowed', [('reid/utils/lodo_evaluation_device.py',True),
    ('reid/engine/train_loop.py',False),('configs/lodo/v1/resnet50_lodo-2_seed42.yaml',False),
    ('reid/models/resnet.py',False)])
def test_v2_revision_never_releases_training_or_config_changes(sandbox,monkeypatch,changed,allowed):
    main,engine=sandbox
    record=dict(schema_version=2,record_type='lodo_main_console_release',frontend_root=str(main),
                frontend_commit='a'*40,execution_root=str(engine),execution_commit='b'*40)
    path=main/'.local/lodo_console/release.json';path.parent.mkdir(parents=True);op.write_private(path,record)
    monkeypatch.setattr(console,'verify_source',lambda *a,**k:console.CONSOLE_FILES)
    monkeypatch.setattr(op,'clean_head',lambda *a:None)
    monkeypatch.setattr(console,'git',lambda *a:(changed+'\n').encode())
    if allowed: assert console.release()['record'] == record
    else:
        with pytest.raises(ValueError,match='frozen training source'):console.release()


def test_gpu_worker_refuses_changed_job_before_execution(sandbox,monkeypatch):
    import lodo_gpu_evaluate_worker as worker
    request=sandbox[1]/'request.yaml';request.write_text('pinned')
    path,value=job(sandbox,kind='evaluation',request=str(request),request_sha256=op.sha(request),admission={})
    monkeypatch.setattr(console,'release',lambda:value['release'])
    request.write_text('changed')
    monkeypatch.setattr(ev,'evaluate_lodo',lambda *a,**k:pytest.fail('Unpinned evaluation executed'))
    with pytest.raises(ValueError,match='request changed'):worker.execute(path)


def test_gpu_worker_writes_actual_evaluator_receipts(sandbox,monkeypatch):
    import lodo_gpu_evaluate_worker as worker
    from reid.utils import config as config_module
    request=sandbox[1]/'request.yaml';request.write_text('pinned')
    admission=dict(status='ADMITTED_CUDA_EVALUATION')
    path,value=job(sandbox,kind='evaluation',request=str(request),request_sha256=op.sha(request),admission=admission)
    monkeypatch.setattr(console,'release',lambda:value['release'])
    monkeypatch.setattr(config_module,'load_config',lambda *a:{'device':'cuda'})
    monkeypatch.setattr(op,'evaluation_admission',lambda *a,**k:admission)
    result=dict(artifacts={'result':'diagnostic-placeholder-not-a-real-benchmark'})
    calls=[]
    monkeypatch.setattr(ev,'evaluate_lodo',lambda *a,**k:(calls.append(k) or (result,{})))
    assert worker.execute(path) == (result,{})
    assert calls == [{'project_root':sandbox[1]}]
    base=sandbox[1]/'.local/lodo_evaluation/v2'
    assert len(list(base.glob('*/completion.json'))) == 1
    assert op.private_record(next(base.glob('*/evaluator-release.json')))[0] == value['release']
