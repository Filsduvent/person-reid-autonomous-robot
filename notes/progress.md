# ReID Baseline Progress

## Current State

This file is the handoff note for future Codex sessions.

## Completed

### Center Loss Integration

Implemented:
- `loss.center` remains YAML-controlled with:
  - `enabled`
  - `weight`
  - `lr`
- Center Loss stays disabled by default in the baseline config but can be enabled without code changes
- `build_criterion(cfg, num_classes, feat_dim)` now:
  - creates `CenterLoss` only when enabled
  - validates `num_classes` and `feat_dim`
  - stores the module as `criterion.center_loss`
  - keeps `criterion.center` as a compatibility alias
- triplet loss and center loss now share the same `model.head.metric_feat` routing
- ID loss continues using `outputs["logits"]` only
- `build_center_optimizer(cfg, criterion)` now:
  - returns `None` when center loss is disabled
  - returns `None` when the criterion has no center loss module
  - otherwise returns SGD over the center loss parameters with `loss.center.lr`
- train loop now:
  - zeros the center optimizer when present
  - rescales center gradients by `1 / center_weight`
  - steps the center optimizer safely alongside the main optimizer
- Center Loss remains device-agnostic and moves with the criterion module

Validation:
- `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `22 passed, 3 skipped`
- verified modes:
  - triplet only
  - ID only
  - triplet + ID
  - triplet + ID + center
- verified behavior:
  - center optimizer is created only when needed
  - `loss/center` is logged through the criterion and train loop
  - center gradients exist and are stepped when enabled

How To Test:
- focused Center Loss checks:
  - `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- debug training path:
  - `PYTHONPATH=. python3 scripts/train.py --config configs/debug_train.yaml`

Notes:
- CUDA execution remains skip-gated in tests and was not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- no `.cuda()` calls were introduced into `CenterLoss`

### BNNeck Trick

Implemented:
- `model.head.bnneck`, `metric_feat`, and `eval_feat` are now explicit YAML controls
- strong-baseline defaults are now:
  - `bnneck: true`
  - `metric_feat: raw`
  - `eval_feat: bn`
- `ReidBaseline` now exposes the BNNeck contract as:
  - `feat_raw`
  - `feat_bn`
  - `emb`
  - `logits`
- `emb` is now selected from `eval_feat` and then normalized if enabled
- BNNeck is now implemented as a dedicated bottleneck layer with:
  - frozen BN bias
  - BN weight/bias initialization
- classifier remains bias-free and is explicitly initialized with small normal weights
- loss routing now uses `model.head.metric_feat` for:
  - triplet loss
  - center loss
- ID loss still uses `logits` only
- evaluator now reads `outputs["emb"]` from dict outputs and does not use `logits`
- builder validation now raises explicit `ValueError`s for invalid:
  - `metric_feat`
  - `eval_feat`
- active baseline/debug/ablation configs were aligned to the BNNeck contract
- stale `loss.center.feat` routing was removed from the active YAML configs

Validation:
- `PYTHONPATH=. pytest -q tests/test_model_forward.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `22 passed, 6 skipped`
- forward tests verify:
  - BNNeck enabled works
  - BNNeck disabled works
  - `eval_feat: raw` works
  - `eval_feat: bn` works
  - output dict contains `feat_raw`, `feat_bn`, `emb`, `logits`
- loss-routing tests verify:
  - triplet uses configured metric feature
  - center uses configured metric feature
  - ID loss uses `logits` only
- existing train-loop tests verify training still runs on CPU with the dict-output contract

How To Test:
- focused BNNeck checks:
  - `PYTHONPATH=. pytest -q tests/test_model_forward.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- quick syntax checks:
  - `python3 -m py_compile reid/models/baseline.py reid/models/build.py reid/losses/build.py reid/engine/evaluator.py tests/test_model_forward.py tests/test_reid_loss_modes.py`
- debug train path:
  - `PYTHONPATH=. python3 scripts/train.py --config configs/debug_train.yaml`

Notes:
- CUDA checks remain skip-gated in tests and were not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- evaluator smoke tests are present but skip when `sklearn` is not installed in the environment

### Last Stride Trick

Implemented:
- `last_conv_stride` remains YAML-controlled through `model.backbone.last_conv_stride`
- `reid/models/build.py` now reads the value into a local variable and prints:
  - `[Model] backbone=resnet50 last_conv_stride=...`
- `ReidBaseline` now explicitly supports:
  - `last_conv_stride: 1`
  - `last_conv_stride: 2`
- `last_conv_stride: 1` modifies only `layer4[0].conv2` and `layer4[0].downsample[0]`
- invalid stride values now raise:
  - `ValueError("last_conv_stride must be 1 or 2")`
- optional ablation config added:
  - `configs/ablation_stride2.yaml`

Validation:
- `PYTHONPATH=. pytest -q tests/test_model_forward.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `13 passed, 5 skipped`
- forward tests verify:
  - `last_conv_stride=1` runs on CPU
  - `last_conv_stride=2` runs on CPU
  - pooled embedding shape stays unchanged
  - final backbone feature map is spatially larger for stride `1` than stride `2`
- config/build tests verify:
  - YAML alone controls the stride setting
  - builder reflects stride `1` vs `2` without hardcoding

Notes:
- CUDA forward verification is covered by a skip-gated test and was not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- evaluation smoke coverage is present but was skipped in this environment because `sklearn` is not installed

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

Center Loss integration hardening is present in the working tree and not yet committed.

## Suggested Next Step

Pick the next Bag-of-Tricks item to implement and validate in the current ReID baseline.

Expected focus:
1. confirm the target trick against the current repo state
2. keep it YAML-controlled when applicable
3. add focused validation before committing the step

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
