"""CPU qualification of the isolated GPU smoke harness, without CUDA calls."""
import copy
from pathlib import Path

import pytest

import lodo_cuda_smoke as smoke
from reid.utils import lodo_protocol as protocol
from test_lodo_config import lodo_config


@pytest.mark.parametrize('architecture',['resnet50','pcb'])
@pytest.mark.parametrize('fold',list(protocol.FOLDS))
def test_testonly_budget_is_bounded_restored_and_recipe_preserved(tmp_path,monkeypatch,architecture,fold):
    monkeypatch.setattr(smoke,'ROOT',tmp_path)
    monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path/'exp/lodo_smoke')
    cfg=lodo_config(architecture,fold)
    original=protocol.scientific_configuration(cfg)
    cfg['experiment']['output_dir']=str(tmp_path/f'exp/lodo_smoke/campaign/{architecture}/{fold}/fresh')
    cfg['train']['max_updates']=3
    smoke.require_production_rejection(cfg)
    with smoke.fixture_budget(3):
        protocol.validate_lodo_config(cfg)
        science=protocol.scientific_configuration(cfg)
        assert protocol.validate_output_path(cfg)==Path(cfg['experiment']['output_dir'])
        original['update_budget']=3
        assert science==original
    smoke.require_production_rejection(cfg)
    assert not (tmp_path/'exp').exists()


@pytest.mark.parametrize('budget',[1,7,85200,True,3.0])
def test_invalid_smoke_budget_rejected(budget):
    with pytest.raises(ValueError):
        with smoke.fixture_budget(budget):pass


@pytest.mark.parametrize('path',[
    'exp/lodo/v1/resnet50/LODO-1/seed42/official',
    'exp/pcb/msmt17', 'exp/lodo_smoke/c/resnet50/LODO-2/fresh',
    'exp/lodo_smoke/c/resnet50/LODO-1/../escape',
])
def test_official_or_wrong_fold_output_prohibited(tmp_path,monkeypatch,path):
    monkeypatch.setattr(smoke,'ROOT',tmp_path);monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path/'exp/lodo_smoke')
    cfg=lodo_config();cfg['experiment']['output_dir']=str(tmp_path/path)
    with pytest.raises(ValueError):smoke.smoke_output(cfg)


def test_smoke_output_collision_alias_and_resume_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(smoke,'ROOT',tmp_path);monkeypatch.setattr(smoke,'SMOKE_ROOT',tmp_path/'exp/lodo_smoke')
    cfg=lodo_config();path=tmp_path/'exp/lodo_smoke/c/resnet50/LODO-1/fresh'
    cfg['experiment']['output_dir']=str(path);assert smoke.smoke_output(cfg)==path
    path.mkdir(parents=True)
    with pytest.raises(ValueError):smoke.smoke_output(cfg)
    alias=path.with_name('alias');alias.symlink_to(path,target_is_directory=True)
    cfg['experiment']['output_dir']=str(alias)
    with pytest.raises(ValueError):smoke.smoke_output(cfg)
    cfg['experiment']['output_dir']=str(path.with_name('new'));cfg['train']['save']['resume']='last'
    with pytest.raises(ValueError):smoke.smoke_output(cfg)


@pytest.mark.parametrize('target',list(protocol.DATASET_ORDER))
def test_synthetic_heldout_fixture_uses_original_parser_and_cross_camera_positive(tmp_path,target):
    from reid.data.build import build_test_loader
    smoke.synthetic_target(tmp_path,target)
    fold='LODO-'+str(list(protocol.DATASET_ORDER).index(target)+1)
    cfg=lodo_config(fold=fold)
    cfg['data'].update(root=str(tmp_path),num_workers=0,pin_memory=False)
    cfg['system']['device']='cpu'
    loader=build_test_loader(cfg);dataset=loader.dataset
    assert len(dataset)==11 and len(set(dataset.pids))==10
    query=[i for i,m in enumerate(dataset.marks) if m==0]
    gallery=[i for i,m in enumerate(dataset.marks) if m==1]
    assert len(query)==1 and len(gallery)==10
    q=query[0]
    assert any(dataset.pids[g]==dataset.pids[q] and dataset.cams[g]!=dataset.cams[q] for g in gallery)
    assert len(next(iter(loader))[0])==11
