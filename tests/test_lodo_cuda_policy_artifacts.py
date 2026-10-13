"""CPU-only schema-v2 metadata checks for the actual-artifact inspector."""
import copy
import pytest
from test_lodo_checkpoint import synthetic
from test_lodo_cuda_policy import launch,strict_payload
from lodo_cuda_policy_artifacts import rehash
from reid.utils import lodo_checkpoint as ck

@pytest.mark.parametrize('architecture',['resnet50','pcb'])
def test_rehash_preserves_reader_validation_for_coherent_wrong_map(synthetic,launch,monkeypatch,architecture):
    cfg,o,path,payload,_=strict_payload(synthetic,monkeypatch,architecture=architecture)
    _,proof=ck.read_checkpoint(path);changed=copy.deepcopy(payload)
    changed['identity']['map_digest']='f'*64
    rehash(path,changed,proof)
    with pytest.raises(ValueError,match='identity'):ck.restore_lodo_checkpoint(path,**o,cpu_inspection=True,validate_only=True)
    assert payload['identity']['map_digest']!='f'*64
