# ReID Baseline Progress

## Current State

This file is the handoff note for future Codex sessions.

## Completed

### Phase 3.2: Optimizer + Warmup Scheduler

Implemented:
- `reid/optim/` package
- `WarmupMultiStepLR`
- `build_optimizer(cfg, model)`
- `build_center_optimizer(cfg, center_criterion)`
- `build_scheduler(cfg, optimizer)`
- training integration in `scripts/train.py`
- per-iteration scheduler stepping in `reid/engine/train_loop.py`
- LR logging in console and TensorBoard
- center-loss optimizer gradient rescaling and separate optimizer step

Validation:
- optimizer/scheduler/train-loop tests passed on CPU
- loss-mode coverage passed for:
  - triplet only
  - triplet + ID
  - triplet + ID + center

Notes:
- scheduler is currently stepped per iteration
- current milestone semantics therefore behave per iteration unless later converted from epochs

Relevant commit:
- `17d7485` `Add ReID optimizer and warmup scheduler support`

### Random Erasing Augmentation

Implemented:
- custom `RandomErasing` in `reid/data/transforms.py`
- train transform refactored to accept `aug_cfg`
- train transform order now matches:
  - `Resize`
  - `RandomHorizontalFlip`
  - `Pad`
  - `RandomCrop`
  - `ToTensor`
  - `Normalize`
  - `RandomErasing`
- test transform remains deterministic
- training data builder updated to use `build_train_tf(image_size, aug_cfg)`
- optional visual inspection script:
  - `scripts/debug_transforms.py`
- baseline config updated with:
  - `padding`
  - `random_crop`
  - `random_erasing`

Validation:
- transform composition tests passed
- `RandomErasing` appears only when `random_erasing.enabled: true`
- disabling it in config removes it without code changes
- test transform does not include `RandomErasing`
- no sampler files changed

Notes:
- dataset-backed Market1501 smoke run is currently blocked in this environment by pickle / NumPy compatibility for `partitions.pkl`
- CUDA was not verified in the current environment because `torch.cuda.is_available()` was `False`

## Current Uncommitted Work

At the time this note was written, Random Erasing changes are present in the working tree and not yet committed.

Touched files:
- `configs/baseline_market1501_resnet50_triplet.yaml`
- `reid/data/build.py`
- `reid/data/transforms.py`
- `scripts/debug_transforms.py`
- `tests/test_data_transforms.py`

## Suggested Next Step

Before starting the next trick:
1. review the Random Erasing diff
2. run the transform/debug checks you want
3. commit the Random Erasing task as its own commit

## Useful Commands

Train debug run:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/debug_train.yaml
```

Transform debug images:

```bash
PYTHONPATH=. python3 scripts/debug_transforms.py --config configs/baseline_market1501_resnet50_triplet.yaml
```

Run current test coverage:

```bash
PYTHONPATH=. pytest -q tests/test_data_transforms.py tests/test_train_loop_optim.py tests/test_reid_loss_modes.py tests/test_optim_build.py
```
