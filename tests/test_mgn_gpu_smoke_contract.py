"""Pure CPU bounds/admission/receipt tests; never initialize CUDA."""
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from scripts import mgn_gpu_smoke as smoke

UUID = 'GPU-11111111-2222-3333-4444-555555555555'


def test_frozen_case_has_no_training_budget_or_science_overrides():
    cfg = smoke.configuration()
    assert smoke.CASE == 'phase12_synthetic_fp32'
    assert smoke.STEPS == 3 and smoke.MAX_SECONDS == 600 and smoke.MAX_ATTEMPTS == 3
    assert cfg['synthetic']['shape'] == [64,3,384,128]
    assert (cfg['synthetic']['P'],cfg['synthetic']['K'],cfg['synthetic']['num_classes']) == (16,4,1041)
    assert cfg['system']['amp'] is False and cfg['repro'] == dict(seed=42,deterministic=False,benchmark=True)
    assert cfg['optim'] == dict(name='sgd',lr=.01,momentum=.9,nesterov=False,weight_decay=.0005,bias_lr_factor=1.,weight_decay_bias=.0005)
    assert cfg['loss']['id']['head_aggregation'] == cfg['loss']['triplet']['feature_aggregation'] == 'mean'
    assert cfg['loss']['triplet']['feature_source'] == 'model' and cfg['loss']['triplet']['margin'] == 1.2
    assert cfg['model']['initialization']['sha256'] == smoke.SHA256
    assert cfg['sched'] == dict(name='epoch_multistep',milestones=[60,90],gamma=.1)


@pytest.mark.parametrize('uuid,environ', [('',{}),('GPU-0',{}),(UUID,{}),
    (UUID,{'MGN_AUTHORIZED_GPU_UUID':'GPU-other'})])
def test_idle_inventory_or_uuid_alone_cannot_authorize_execution(uuid,environ):
    with pytest.raises(ValueError): smoke.authorized_uuid(uuid,environ)


def test_external_manual_authority_requires_exact_uuid():
    assert smoke.authorized_uuid(UUID,{'MGN_AUTHORIZED_GPU_UUID':UUID}) == UUID


@pytest.mark.parametrize('attempt', [0,4,-1,True,1.0])
def test_attempt_bound_and_fresh_output_rejection(tmp_path,monkeypatch,attempt):
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path)
    with pytest.raises(ValueError): smoke.output_path(UUID,attempt)


def test_output_collision_symlink_and_case_reset_are_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path)
    path=smoke.output_path(UUID,1)
    path.mkdir(parents=True)
    with pytest.raises(ValueError,match='fresh'): smoke.output_path(UUID,1)
    with pytest.raises(ValueError,match='reset'): smoke.retry_guard(UUID,1,'code','')
    other=smoke.output_path(UUID,2)
    other.parent.mkdir(parents=True)
    other.symlink_to(path,target_is_directory=True)
    with pytest.raises(ValueError,match='fresh'): smoke.output_path(UUID,2)


def test_retries_require_preserved_error_correction_changed_code_and_same_case(tmp_path,monkeypatch):
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path)
    with pytest.raises(ValueError,match='previous'): smoke.retry_guard(UUID,2,'new','fixed defect')
    previous=tmp_path/f'{smoke.CASE}_{UUID}_attempt1'/'fp32'/'summary.json'
    previous.parent.mkdir(parents=True)
    state=dict(status='OOM',code_hash='old',case=smoke.CASE,gpu_uuid=UUID,
               configuration_sha256=smoke.canonical_hash(smoke.configuration()))
    previous.write_text(json.dumps(state))
    with pytest.raises(ValueError): smoke.retry_guard(UUID,2,'old','fixed defect')
    with pytest.raises(ValueError): smoke.retry_guard(UUID,2,'new','')
    assert smoke.retry_guard(UUID,2,'new','fixed retained-graph implementation defect')['sha256'] == smoke.file_hash(previous)
    for change in [dict(status='PASS'),dict(status='TIMEOUT'),dict(gpu_uuid='other'),dict(configuration_sha256='changed')]:
        previous.write_text(json.dumps(dict(state,**change)))
        with pytest.raises(ValueError): smoke.retry_guard(UUID,2,'new','fixed defect')


@pytest.mark.parametrize('completed,calls,returncode', [(0,0,0),(2,2,0),(3,4,0),(3,3,2)])
def test_pass_receipt_requires_three_complete_updates(completed,calls,returncode):
    result=smoke.summarize({},dict(status='PASS',completed_steps=completed,optimizer_step_calls=calls,steps=[{}]*completed),returncode,False)
    assert result['status']=='ERROR'


def test_oom_and_timeout_preserve_failure_stage_and_memory():
    worker=dict(status='OOM',stage='step_1/forward',completed_steps=0,optimizer_step_calls=0,
                failure_memory=dict(peak_allocated_bytes=123,peak_reserved_bytes=456))
    result=smoke.summarize({},worker,2,False)
    assert result['status']=='OOM' and result['worker']==worker
    assert smoke.summarize({},worker,-9,True)['status']=='TIMEOUT'
    passing=dict(status='PASS',completed_steps=3,optimizer_step_calls=3,steps=[{}]*3)
    assert smoke.summarize({},passing,0,False)['status']=='PASS'
    assert smoke.summarize({},dict(status='RUNNING'),-9,False)['status']=='ERROR'


def test_supervisor_kills_only_owned_group_at_hard_deadline(tmp_path,monkeypatch):
    signals=[]
    class Child:
        pid=12345;returncode=-9
        def __init__(self): self.calls=0
        def wait(self,timeout):
            self.calls+=1
            if self.calls==1:
                assert timeout==600
                raise subprocess.TimeoutExpired('worker',timeout)
            return self.returncode
    child=Child()
    def popen(command,**kwargs):
        assert kwargs['start_new_session'] is True and kwargs['env']['CUDA_VISIBLE_DEVICES']==UUID
        return child
    monkeypatch.setattr(smoke.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    assert smoke.supervise(['worker'],{'CUDA_VISIBLE_DEVICES':UUID},tmp_path,popen=popen)==(-9,True)
    assert signals==[(12345,signal.SIGKILL)]
    assert (tmp_path/'stdout.log').is_file() and (tmp_path/'stderr.log').is_file()


def test_worker_rejects_wrong_visibility_before_cuda_setup(monkeypatch):
    monkeypatch.setenv('MGN_AUTHORIZED_GPU_UUID',UUID)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(ValueError,match='visibility'):
        smoke.worker(SimpleNamespace(gpu_uuid=UUID))


def test_cpu_receipts_bind_source_and_harness_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(smoke,'ROOT',tmp_path)
    (tmp_path/'docs').mkdir()
    source=tmp_path/'source.py';source.write_text('qualified')
    (tmp_path/'docs/mgn_cpu_qualification.json').write_text(json.dumps(dict(status='PASS')))
    (tmp_path/'docs/mgn_synthetic_qualification.json').write_text(json.dumps(dict(status='PASS',source_sha256={'source.py':smoke.file_hash(source)})))
    guard=tmp_path/'docs/mgn_gpu_smoke_contract.json'
    guard.write_text(json.dumps(dict(status='CPU_PASS',code_hash=smoke.file_hash(smoke.__file__))))
    assert len(smoke.cpu_evidence())==3
    source.write_text('changed after qualification')
    with pytest.raises(ValueError,match='source changed'): smoke.cpu_evidence()
    source.write_text('qualified')
    guard.write_text(json.dumps(dict(status='CPU_PASS',code_hash='wrong')))
    with pytest.raises(ValueError,match='harness changed'): smoke.cpu_evidence()


def test_missing_manual_authority_fails_before_cache_receipts_or_inventory(tmp_path,monkeypatch):
    monkeypatch.delenv('MGN_AUTHORIZED_GPU_UUID',raising=False)
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path)
    monkeypatch.setattr(smoke,'cpu_evidence',lambda:pytest.fail('CPU/cache/inventory reached before permission guard'))
    with pytest.raises(ValueError,match='MGN_AUTHORIZED'):
        smoke.parent(SimpleNamespace(gpu_uuid=UUID,attempt=1,correction_note='',authorization_note=''))
    assert not list(tmp_path.iterdir())


def test_actual_oom_handler_preserves_counts_stage_memory_and_original_error(tmp_path):
    class FakeOOM(RuntimeError): pass
    cuda=SimpleNamespace(OutOfMemoryError=FakeOOM,is_initialized=lambda:True,
        mem_get_info=lambda device:(10,100),max_memory_allocated=lambda device:80,
        max_memory_reserved=lambda device:90,memory_allocated=lambda device:70,memory_reserved=lambda device:90)
    state=dict(stage='step_1/forward',attempted_batches=1,optimizer_step_calls=0,completed_steps=0,steps=[])
    try:
        raise FakeOOM('fixed batch64 allocation failed')
    except FakeOOM as error:
        smoke.failure_receipt(tmp_path,state,error,SimpleNamespace(cuda=cuda))
    saved=json.loads((tmp_path/'worker_receipt.json').read_text())
    assert saved['status']=='OOM' and saved['stage']=='step_1/forward'
    assert saved['optimizer_step_calls']==saved['completed_steps']==0
    assert saved['failure_memory']==dict(peak_allocated_bytes=80,peak_reserved_bytes=90,
        allocated_bytes=70,reserved_bytes=90,free_bytes=10,total_bytes=100)
    assert 'fixed batch64 allocation failed' in saved['error']['traceback']
    assert not (tmp_path/'summary.json').exists()  # Supervisor owns final publication.


def test_failure_reporting_survives_memory_query_failure(tmp_path):
    class FakeOOM(RuntimeError): pass
    def failed(device): raise RuntimeError('driver query failed')
    cuda=SimpleNamespace(OutOfMemoryError=FakeOOM,is_initialized=lambda:True,mem_get_info=failed)
    saved=smoke.failure_receipt(tmp_path,dict(stage='forward',completed_steps=0),FakeOOM('original OOM'),SimpleNamespace(cuda=cuda))
    assert saved['status']=='OOM' and saved['error']['message']=='original OOM'
    assert 'driver query failed' in saved['failure_memory_error']


def test_supervisor_interruption_cannot_leave_an_owned_worker_running(tmp_path,monkeypatch):
    killed=[]
    class Child:
        pid=12345
        def __init__(self): self.calls=0
        def wait(self,timeout):
            self.calls+=1
            if self.calls==1: raise KeyboardInterrupt('interrupted')
            return -9
    monkeypatch.setattr(smoke.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt):
        smoke.supervise(['worker'],{},tmp_path,popen=lambda *a,**k:Child())
    assert killed==[(12345,signal.SIGKILL)]
