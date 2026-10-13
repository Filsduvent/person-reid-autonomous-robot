"""CPU-only guards for the non-authoritative CUDA recovery harness."""
from pathlib import Path
import pytest
from lodo_cuda_recovery_smoke import ShortTestEpoch,smoke_output,ROOT,SMOKE
from test_lodo_config import lodo_config


def test_short_epoch_never_changes_original_decisions_or_fetches_third():
    drawn=[]
    def original():
        for i in range(10):drawn.append(i);yield [i]*64
    wrapper=ShortTestEpoch(original())
    assert list(wrapper)==[[0]*64,[1]*64] and drawn==[0,1] and len(wrapper)==2


@pytest.mark.parametrize('suffix',['../escape','continuous/extra','other','continuous/../interrupted'])
def test_recovery_smoke_namespace_escape_rejected(suffix):
    cfg=lodo_config();cfg['experiment']['output_dir']=str(SMOKE/'phase9r-contract'/'resnet50/LODO-1'/suffix)
    with pytest.raises(ValueError):smoke_output(cfg)


def test_valid_fresh_namespace_and_root_rebinding_rejected():
    cfg=lodo_config();cfg['experiment']['output_dir']=str(SMOKE/'phase9r-contract-not-created'/'resnet50/LODO-1/continuous')
    assert smoke_output(cfg)==Path(cfg['experiment']['output_dir'])
    with pytest.raises(ValueError):smoke_output(cfg,project_root=Path('/tmp'))
