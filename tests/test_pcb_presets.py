"""Offline preset contracts; synthetic fixtures do not certify benchmark provenance."""
import copy
import pickle
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms as T

from reid.data.build import build_test_loader, build_train_loader
from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.models.pcb import PCB, HISTORICAL_IMAGENET_SHA256
from reid.optim.build import build_optimizer, build_scheduler
from reid.utils.config import load_config, validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils.evaluation_config import build_evaluation_config

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("market1501", "duke", "cuhk03", "msmt17")


def preset(name):
    return load_config(str(ROOT / "configs" / "pcb" / f"{name}.yaml"))


def test_cross_preset_consistency_and_dataset_ownership():
    recipes, outputs = [], []
    for name in NAMES:
        cfg = preset(name)
        baseline = load_config(str(ROOT / "configs" / f"baseline_{name}_resnet50_triplet.yaml"))
        outputs.append(cfg["experiment"]["output_dir"])
        assert cfg["experiment"]["output_dir"] == f"exp/pcb/{name}"
        assert cfg["experiment"]["output_dir"] != baseline["experiment"]["output_dir"]
        assert cfg["data"]["root"] == baseline["data"]["root"]
        for mode in ("train", "test"):
            assert cfg["data"][mode]["dataset"] == baseline["data"][mode]["dataset"]
            del cfg["data"][mode]["dataset"]
        del cfg["data"]["root"]
        del cfg["experiment"]["name"]
        del cfg["experiment"]["output_dir"]
        recipes.append(cfg)
    assert len(set(outputs)) == 4
    assert all(cfg == recipes[0] for cfg in recipes)


def synthetic_dataset(root, name):
    """65 train and 33 evaluation images exercise both final-batch policies."""
    def image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 8), (0, 0, 0)).save(path)

    if name == "msmt17":
        base = root / name / "MSMT17_V2"
        for filename, folder, count, offset in (
            ("list_train.txt", "mask_train_v2", 65, 0),
            ("list_query.txt", "mask_test_v2", 1, 100),
            ("list_gallery.txt", "mask_test_v2", 32, 101),
        ):
            rows = []
            for i in range(count):
                pid = i % 3 + (2 if folder == "mask_train_v2" else 20)
                rel = f"{pid:04d}/{pid:04d}_{i+offset:04d}_{i%2+1}_0001.jpg"
                image(base / folder / rel)
                rows.append(f"{rel} {pid}")
            (base / filename).write_text("\n".join(rows) + "\n")
    else:
        base = root / name
        if name == "cuhk03":
            base /= "detected"
        train = [f"{i%3+2:08d}_{i%2+1:04d}_{i:08d}.jpg" for i in range(65)]
        test = [f"{i%3+20:08d}_{i%2+1:04d}_{i+100:08d}.jpg" for i in range(33)]
        for filename in train + test:
            image(base / "images" / filename)
        with (base / "partitions.pkl").open("wb") as stream:
            pickle.dump(dict(trainval_im_names=train, trainval_ids2labels={2: 0, 3: 1, 4: 2},
                             test_im_names=test, test_marks=[0] + [1] * 32), stream)


@pytest.mark.parametrize("name", NAMES)
def test_preset_framework_and_synthetic_loader_preflight(name, tmp_path, monkeypatch):
    cfg = preset(name)
    validate_config(cfg)
    assert cfg["model"] == dict(name="pcb", variant="independent_part_reduction", pretrained=True)
    assert HISTORICAL_IMAGENET_SHA256 in cfg["experiment"]["notes"]
    assert "resnet50-19c8e357.pth" in cfg["experiment"]["notes"]
    assert cfg["loss"] == dict(
        id=dict(enabled=True, label_smoothing=0.0, weight=1.0, head_aggregation="sum"),
        triplet=dict(enabled=False), center=dict(enabled=False))
    assert cfg["optim"]["bias_lr_factor"] == 1.0
    assert cfg["optim"]["param_groups"] == [dict(prefix="backbone.", lr_mult=.1)]
    assert cfg["sched"]["milestones"] == [40] and cfg["sched"]["gamma"] == .1
    assert cfg["system"]["amp"] is False
    assert cfg["repro"] == dict(seed=42, deterministic=False, benchmark=True)
    assert cfg["train"] == dict(epochs=120, eval_interval=10,
                               save=dict(save_best=True, save_last=True, metric="mAP", resume=""))
    assert cfg["eval"]["normalize_feat"] is True
    assert cfg["eval"]["distance"] == "euclidean"
    assert cfg["eval"]["topk"] == [1, 5, 10]
    assert cfg["eval"]["rerank"]["enabled"] is False
    synthetic_dataset(tmp_path, name)
    cfg["data"].update(root=str(tmp_path), num_workers=0, pin_memory=False)
    train, classes = build_train_loader(cfg)
    test = build_test_loader(cfg)
    assert classes == 3 and set(train.dataset.labels) == {0, 1, 2}
    assert type(train.sampler) is RandomSampler
    assert type(test.sampler) is SequentialSampler
    assert train.drop_last and not test.drop_last
    assert len(train.dataset) == 65 and len(train) == 1
    assert len(test.dataset) == 33 and len(test) == 2
    images, labels = next(iter(train))
    assert images.shape == (64, 3, 384, 128) and images.dtype == torch.float32
    assert torch.isfinite(images).all()
    assert labels.shape == (64,) and labels.dtype == torch.long
    assert set(labels.tolist()) <= {0, 1, 2}
    batches = list(test)
    assert [len(batch[0]) for batch in batches] == [32, 1]
    assert torch.cat([batch[4] for batch in batches]).tolist() == [0] + [1] * 32
    expected_cams = [i % 2 + 1 for i in range(33)]
    if name == "msmt17":
        expected_cams = [0] + [i % 2 for i in range(32)]
    assert torch.cat([batch[2] for batch in batches]).tolist() == expected_cams
    expected_pids = [i % 3 + 20 for i in range(33)]
    if name == "msmt17":
        expected_pids = [20] + [i % 3 + 20 for i in range(32)]
    assert torch.cat([batch[1] for batch in batches]).tolist() == expected_pids
    for mode, loader, types in (
        ("train", train, [T.Resize, T.RandomHorizontalFlip, T.ToTensor, T.Normalize]),
        ("test", test, [T.Resize, T.ToTensor, T.Normalize]),
    ):
        transforms = loader.dataset.transform.transforms
        assert [type(t) for t in transforms] == types
        assert tuple(transforms[0].size) == (384, 128)
        assert list(transforms[-1].mean) == [.486, .459, .408]
        assert list(transforms[-1].std) == [.229, .224, .225]
        assert cfg["data"][mode]["aug"]["mirror"] == ("random" if mode == "train" else "none")
    assert train.dataset.transform.transforms[1].p == .5
    expected_pixel = -torch.tensor([.486, .459, .408]) / torch.tensor([.229, .224, .225])
    for batch in batches:
        assert batch[0].shape[1:] == (3, 384, 128)
        assert batch[0].dtype == torch.float32 and torch.isfinite(batch[0]).all()
        torch.testing.assert_close(batch[0][0, :, 0, 0], expected_pixel)

    def forbidden(*args, **kwargs):
        pytest.fail("Offline preflight must not load pretrained weights or step an optimizer")
    monkeypatch.setattr("reid.models.pcb._read_historical_weights", forbidden)
    validate_reid_config(cfg, num_classes=classes)
    model = build_model(cfg, num_classes=classes, initialize_pretrained=False)
    assert isinstance(model, PCB) and model.embedding_dim == 1536 and model.feat_dim is None
    assert len(model.classifiers.fc_list) == len(model.reductions.local_conv_list) == 6
    assert model.backbone.layer4[0].conv2.stride == (1, 1)
    assert all(module.dilation == (1, 1) for module in model.backbone.modules()
               if isinstance(module, torch.nn.Conv2d))
    assert all(part[0].in_channels == 2048 and part[0].out_channels == 256
               for part in model.reductions.local_conv_list)
    assert all(head.in_features == 256 and head.out_features == classes
               for head in model.classifiers.fc_list)
    criterion = build_criterion(cfg, num_classes=classes, feat_dim=model.feat_dim)
    assert criterion.triplet is None and criterion.center_loss is None
    logits = [torch.randn(4, classes) for _ in range(6)]
    targets = torch.tensor([0, 1, 2, 0])
    total, _ = criterion({"logits": logits}, targets)
    torch.testing.assert_close(total, sum(torch.nn.functional.cross_entropy(x, targets) for x in logits))
    optimizer = build_optimizer(cfg, model)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == .9 and not optimizer.defaults["nesterov"]
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(.01 if group["param_name"].startswith("backbone.") else .1)
        assert group["weight_decay"] == .0005
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train))
    monkeypatch.setattr(optimizer, "step", forbidden)
    assert scheduler.milestones == [40] and scheduler.warmup_iters == 0
    # Inspect the formula without training or stepping either optimizer/scheduler.
    for iteration, factor in ((0, 1), (39, 1), (40, .1), (119, .1)):
        scheduler.last_epoch = iteration
        assert scheduler.get_lr() == pytest.approx([lr * factor for lr in scheduler.base_lrs])


def test_real_preset_source_preprocessing_ownership(tmp_path):
    source = preset("market1501")
    target = load_config(str(ROOT / "configs/baseline_duke_resnet50_triplet.yaml"))
    original = copy.deepcopy(source)
    assert target["data"]["test"]["images"]["size"] == [256, 128]
    merged = build_evaluation_config(source, target, tmp_path)
    assert merged["data"]["test"]["dataset"] == target["data"]["test"]["dataset"]
    assert merged["data"]["test"]["images"]["size"] == [384, 128]
    for key in ("images", "aug", "loader"):
        assert merged["data"]["test"][key] == source["data"]["test"][key]
    assert merged["model"] == source["model"] and merged["eval"] == source["eval"]
    assert source == original
