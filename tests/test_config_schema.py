import copy

import pytest

from reid.utils.config import load_config
from reid.utils.config_schema import REQUIRED_TOP_LEVEL_SECTIONS, validate_config


VALID_CFG = {
    "experiment": {
        "name": "unit_test",
        "output_dir": "exp/unit_test",
    },
    "system": {
        "device": "auto",
    },
    "repro": {
        "seed": 42,
    },
    "logging": {
        "to_file": False,
    },
    "data": {
        "train": {
            "dataset": {
                "name": "market1501",
            }
        },
        "test": {
            "dataset": {
                "name": "market1501",
            }
        },
    },
    "model": {
        "name": "reid_baseline",
    },
    "loss": {
        "triplet": {
            "enabled": True,
        }
    },
    "optim": {
        "name": "adam",
    },
    "sched": {
        "name": "warmup_multistep",
    },
    "train": {
        "epochs": 1,
    },
    "eval": {
        "topk": [1],
    },
}


def _cfg():
    return copy.deepcopy(VALID_CFG)


def test_validate_config_accepts_required_schema():
    validate_config(_cfg())


def test_msmt17_config_disables_rerank_by_default():
    cfg = load_config("configs/baseline_msmt17_resnet50_triplet.yaml")

    assert cfg["data"]["train"]["dataset"]["name"] == "msmt17"
    assert cfg["data"]["train"]["dataset"]["format"] == "raw"
    assert cfg["eval"]["rerank"]["enabled"] is False


@pytest.mark.parametrize("section", REQUIRED_TOP_LEVEL_SECTIONS)
def test_validate_config_requires_top_level_sections(section):
    cfg = _cfg()
    del cfg[section]

    with pytest.raises(ValueError, match=f"Missing config section: {section}"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "key",
    [
        "experiment.name",
        "experiment.output_dir",
        "data.train.dataset.name",
        "data.test.dataset.name",
        "model.name",
        "optim.name",
        "train.epochs",
        "eval.topk",
    ],
)
def test_validate_config_requires_nested_keys(key):
    cfg = _cfg()
    parts = key.split(".")
    current = cfg
    for part in parts[:-1]:
        current = current[part]
    del current[parts[-1]]

    with pytest.raises(ValueError, match=f"Missing config key: {key}"):
        validate_config(cfg)


def test_validate_config_rejects_empty_required_values():
    cfg = _cfg()
    cfg["experiment"]["name"] = ""

    with pytest.raises(ValueError, match="experiment.name.*must not be empty"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_system_device():
    cfg = _cfg()
    cfg["system"]["device"] = "gpu"

    with pytest.raises(ValueError, match="Invalid config value for system.device"):
        validate_config(cfg)


def test_validate_config_rejects_non_mapping_sections():
    cfg = _cfg()
    cfg["data"] = []

    with pytest.raises(ValueError, match="Config section 'data' must be a mapping"):
        validate_config(cfg)


def test_validate_config_accepts_real_baseline_config():
    cfg = load_config("configs/baseline_market1501_resnet50_triplet.yaml")

    validate_config(cfg)


def test_config_overrides_are_validated_after_application(tmp_path):
    cfg_path = tmp_path / "bad_device.yaml"
    cfg_path.write_text(
        """
experiment:
  name: unit
  output_dir: exp/unit
system:
  device: gpu
repro: {}
logging: {}
data:
  train:
    dataset:
      name: market1501
  test:
    dataset:
      name: market1501
model:
  name: reid_baseline
loss: {}
optim:
  name: adam
sched: {}
train:
  epochs: 1
eval:
  topk: [1]
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path), overrides=["system.device=cpu"])

    validate_config(cfg)
