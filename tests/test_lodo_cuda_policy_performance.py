"""CPU gates: timing must never bypass recovery or namespace qualification."""
import json
from pathlib import Path
import pytest
import torch
import lodo_cuda_policy_performance as perf

@pytest.mark.parametrize('status',['RUNNING','STOPPED','FAIL'])
def test_benchmark_rejects_incomplete_qualification_before_outputs(tmp_path,monkeypatch,status):
    (tmp_path/'campaign-summary.json').write_text(json.dumps(dict(status=status,architectures={})))
    monkeypatch.setattr(perf,'campaign_path',lambda _:tmp_path)
    with pytest.raises(AssertionError):perf.measure('ignored','resnet50','strict')
    assert not (tmp_path/'performance').exists() and not torch.cuda.is_initialized()

@pytest.mark.parametrize('arch',['resnet50','pcb'])
def test_benchmark_requires_architecture_recovery_pass(tmp_path,monkeypatch,arch):
    (tmp_path/'campaign-summary.json').write_text(json.dumps(dict(status='COMPLETED',architectures={arch:dict(status='FAIL')})))
    monkeypatch.setattr(perf,'campaign_path',lambda _:tmp_path)
    with pytest.raises(AssertionError):perf.measure('ignored',arch,'strict')
    assert not (tmp_path/'performance').exists()

@pytest.mark.parametrize('policy,workspace',[('unknown',':4096:8'),('strict',None),('original',':4096:8')])
def test_benchmark_policy_environment_fail_closed(tmp_path,monkeypatch,policy,workspace):
    (tmp_path/'campaign-summary.json').write_text(json.dumps(dict(status='COMPLETED',architectures={'resnet50':dict(status='PASS')})))
    monkeypatch.setattr(perf,'campaign_path',lambda _:tmp_path)
    if workspace is None:monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    else:monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',workspace)
    with pytest.raises(AssertionError):perf.measure('ignored','resnet50',policy)
    assert not (tmp_path/'performance').exists()
