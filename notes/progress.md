# ReID Baseline Progress

## Current State

This file is the handoff note for future Codex sessions.

## Completed

### Label Smoothing Verification

Verified:
- `loss.id.label_smoothing` is config-driven in YAML
- `label_smoothing: 0.0` selects standard CE behavior
- `label_smoothing: 0.1` selects `CrossEntropyLabelSmooth`
- ID loss consumes classifier `logits` and dataset `labels`
- NaN guard added for ID loss in `reid/losses/build.py`
- CPU training-path checks cover:
  - triplet only
  - ID only
  - triplet + ID
  - triplet + ID + center

Validation:
- `PYTHONPATH=. pytest -q tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `11 passed, 3 skipped`
- label smoothing tests verify:
  - `label_smoothing: 0.0` matches `F.cross_entropy(...)`
  - `label_smoothing: 0.1` changes the scalar ID-loss value on fixed logits

Notes:
- CUDA runtime verification could not be executed in the current environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- CUDA-path tests were added and will run automatically on a GPU-visible machine

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

No task-specific uncommitted work was intended at the time this note was last updated.

## Suggested Next Step

Implement the Bag-of-Tricks last stride trick in the current ReID baseline.

Expected focus:
1. verify how `last_conv_stride` is currently wired from YAML into the backbone
2. confirm the default baseline setting matches the intended trick
3. validate training/eval still run correctly when comparing stride variants

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
