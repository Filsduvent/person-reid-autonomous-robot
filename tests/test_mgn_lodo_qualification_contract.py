"""CPU checks for fixture-only qualification bounds; no GPU allocation."""
import json
from pathlib import Path
import pytest
from mgn_lodo_qualification import enter_step, case_path, OneBatchFixture
from reid.utils.mgn_lodo_protocol import validate_lodo_config
from reid.utils.config import load_config


def test_ledger_aggregates_children_and_refuses_fourth_call(tmp_path):
    path=tmp_path/'ledger.json';path.write_text(json.dumps(dict(optimizer_entries=0)))
    for count in range(1,4):
        enter_step(path)
        assert json.loads(path.read_text())['optimizer_entries']==count
    with pytest.raises(RuntimeError,match='aggregate'):enter_step(path)
    assert json.loads(path.read_text())['optimizer_entries']==3


@pytest.mark.parametrize('campaign',['bad','phase21-../escape','phase21-bad space'])
def test_campaign_paths_reject_aliases(campaign):
    with pytest.raises(ValueError):case_path(campaign,1,'forward')


def test_fixture_epoch_reads_one_original_batch_only():
    class Original:
        def __init__(self):self.reads=[]
        def __iter__(self):
            for value in range(100):self.reads.append(value);yield value
    original=Original();fixture=OneBatchFixture(original)
    assert len(fixture)==1 and list(fixture)==[0] and original.reads==[0]


def test_production_refuses_qualification_budget_and_namespace():
    cfg=load_config('configs/lodo/mgn_lodo_v2/mgn_lodo-1_seed42.yaml')
    cfg['train']['max_updates']=3
    with pytest.raises(ValueError):validate_lodo_config(cfg)
    cfg['train']['max_updates']=85200
    cfg['experiment']['output_dir']=str(case_path('phase21-cpu-test',1,'forward')/'run')
    with pytest.raises(ValueError):validate_lodo_config(cfg)


def test_fixture_keeps_checkpoint_observed_loader_and_original_pk_sampler():
    from reid.data.index_observation import ObservedTrainLoader
    from test_lodo_updates import loader_for
    from test_lodo_checkpoint import StochasticDataset
    from reid.utils.lodo_checkpoint import loader_contract
    loader=ObservedTrainLoader(loader_for(StochasticDataset(194),'pk',0),sampler_mode='pk')
    original=loader.batch_sampler
    loader._observer.original=OneBatchFixture(original)
    assert loader_contract(loader)['P']==16
    assert len(list(loader))==1 and loader.batch_sampler is original and not loader._active


@pytest.mark.parametrize('attempt',[0,4,999])
def test_attempt_bound_cannot_be_reset(attempt):
    with pytest.raises(ValueError,match='correction'):case_path('phase21-test',1,'forward',attempt)
