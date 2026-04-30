# ReID Baseline Progress

## Current State

This file is the handoff note for future Codex sessions.

## Completed

### Train And Evaluation Orchestration

Implemented:
- experiment logger utility added in `reid/utils/logger.py`
- checkpoint utility added in `reid/utils/checkpoint.py`
- checkpoint loading now supports:
  - full training checkpoints with `ckpt["model"]`
  - raw model-only state dicts
  - `module.` prefix stripping for non-DataParallel loads
- `scripts/train.py` was refactored into an orchestration-style entrypoint that now:
  - loads config plus `--opts` overrides
  - creates experiment artifact directories
  - sets up logger, device, seed, TensorBoard, loaders, model, criterion, optimizer, center optimizer, and scheduler
  - resumes from `train.save.resume` when configured
  - saves canonical checkpoints and eval artifacts
- `scripts/evaluate.py` added as a standalone evaluation entrypoint that:
  - loads config plus optional `--weight`
  - builds only the test-side path
  - loads a checkpoint
  - runs evaluation without training
  - writes metrics JSON under the experiment directory
- shared `build_test_loader(cfg)` added to `reid/data/build.py` so train/evaluate reuse the same test loader logic
- evaluation JSON export schema is now consistent across training and standalone evaluation:
  - `dataset`
  - `split`
  - `epoch`
  - `checkpoint`
  - `mAP`
  - `mINP`
  - `Rank1`
  - `Rank5`
  - `Rank10`
  - rerank fields only when reranking is enabled
- resolved config artifacts are now saved as:
  - `config.resolved.yaml`
  - `config.yaml` compatibility alias
- canonical checkpoints are now saved as:
  - `checkpoints/ckpt_best.pth`
  - `checkpoints/ckpt_last.pth`
- train artifact layer now also writes:
  - `logs/stdout.txt`
  - `logs/stderr.txt`
  - `plots/loss_curve.png`
  - `plots/rank1_curve.png`
  - `plots/map_bar.png`
  - `plots/minp_bar.png`
  - `plots/cmc_curve.png`
- standalone evaluation now also tee-captures raw output to:
  - `logs/eval_stdout.txt`
  - `logs/eval_stderr.txt`

Validation:
- `python3 -m py_compile reid/utils/logger.py reid/utils/checkpoint.py scripts/train.py scripts/evaluate.py reid/data/build.py`
- result in this environment: passed
- `PYTHONPATH=. pytest -q tests/test_checkpoint.py tests/test_train_orchestration.py tests/test_train_loop_optim.py tests/test_model_forward.py tests/test_rerank.py`
- result in this environment: `32 passed, 11 skipped`
- orchestration tests now verify:
  - canonical resolved-config artifact naming
  - canonical checkpoint naming
  - resume restores epoch, best metric, optimizer, scheduler, and center optimizer state
  - model-only checkpoint load works
  - `module.` prefix stripping works

How To Test:
- focused orchestration checks:
  - `PYTHONPATH=. pytest -q tests/test_checkpoint.py tests/test_train_orchestration.py tests/test_train_loop_optim.py`
- direct standalone evaluator CLI check:
  - `python3 scripts/evaluate.py --help`
- example standalone evaluation run:
  - `python3 scripts/evaluate.py --config configs/experiment_market1501_resnet50_center_rerank.yaml --weight exp/market1501_r50_triplet_id_center_rerank/checkpoints/ckpt_best.pth`

### Optional Re-Ranking Evaluation

Implemented:
- optional k-reciprocal re-ranking utility added in `reid/metrics/rerank.py`
- evaluator now always reports original metrics:
  - `mAP`
  - `mINP`
  - `Rank1`
  - `Rank5`
  - `Rank10`
- evaluator now adds reranked metrics only when `eval.rerank.enabled: true`:
  - `rerank_mAP`
  - `rerank_mINP`
  - `rerank_Rank1`
  - `rerank_Rank5`
  - `rerank_Rank10`
- current evaluator flow remains:
  - features -> distance matrices -> optional reranking -> metrics
- metric JSON export now naturally includes rerank keys only when enabled because eval scores are serialized directly
- TensorBoard eval logging now:
  - always logs original eval metrics
  - logs rerank eval metrics only when present
- evaluator now prints a memory warning when reranking is enabled because large galleries can be expensive on CPU memory
- active Market1501 baseline config now exposes:
  - `eval.rerank.enabled`
  - `eval.rerank.k1`
  - `eval.rerank.k2`
  - `eval.rerank.lambda_value`

Validation:
- `python3 -m py_compile reid/metrics/rerank.py reid/engine/evaluator.py tests/test_rerank.py scripts/train.py`
- result in this environment: passed
- `PYTHONPATH=. pytest -q tests/test_rerank.py tests/test_model_forward.py`
- result in this environment: `11 passed, 5 skipped`
- rerank tests verify:
  - rerank utility returns query-gallery distance matrices with finite values
  - evaluator works with rerank disabled
  - evaluator works with rerank enabled
  - original metrics are preserved
  - rerank metrics are reported separately

How To Test:
- focused rerank and evaluator checks:
  - `PYTHONPATH=. pytest -q tests/test_rerank.py tests/test_model_forward.py`
- rerank experiment toggle:
  - `PYTHONPATH=. python3 scripts/train.py --config configs/experiment_market1501_resnet50_center.yaml`
  - duplicate the config or temporarily set `eval.rerank.enabled: true` in the experiment YAML for the rerank comparison run

### Warmup Scheduler Milestone Semantics

Implemented:
- `warmup_multistep` milestones are now interpreted as epochs, matching the strong-baseline recipe
- scheduler stepping remains per iteration in the train loop
- `build_scheduler(cfg, optimizer, steps_per_epoch=...)` now converts YAML epoch milestones into iteration milestones internally
- warmup remains controlled by `warmup_iters` and therefore still operates per iteration
- `scripts/train.py` now passes `len(train_loader)` into the scheduler builder so active training uses the corrected semantics
- scheduler builder now raises a clear error if `warmup_multistep` is requested without a positive `steps_per_epoch`

Validation:
- `python3 -m py_compile reid/optim/build.py scripts/train.py tests/test_optim_build.py tests/test_train_loop_optim.py`
- result in this environment: passed
- `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_train_loop_optim.py`
- result in this environment: `15 passed, 3 skipped`
- scheduler tests now verify:
  - epoch milestones are converted to iteration milestones
  - warmup still ramps per iteration
  - LR decay occurs at the expected epoch boundary after conversion
  - missing `steps_per_epoch` fails early for `warmup_multistep`

How To Test:
- focused scheduler and train-loop checks:
  - `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_train_loop_optim.py`
- quick syntax checks:
  - `python3 -m py_compile reid/optim/build.py scripts/train.py tests/test_optim_build.py tests/test_train_loop_optim.py`

### Optimizer Refinements

Implemented:
- SGD optimizer builder now supports YAML-controlled `nesterov`
- optimizer param groups now keep `param_name` for debugging and verification
- bias-group overrides remain YAML-controlled through:
  - `bias_lr_factor`
  - `weight_decay_bias`
- optimizer tests now verify:
  - bias params use `base_lr * bias_lr_factor`
  - bias params use `weight_decay_bias`
  - non-bias params use base values
  - SGD honors `nesterov: true`

Validation:
- `PYTHONPATH=. pytest -q tests/test_optim_build.py`
- result in this environment: `10 passed`

How To Test:
- focused optimizer checks:
  - `PYTHONPATH=. pytest -q tests/test_optim_build.py`
- quick syntax checks:
  - `python3 -m py_compile reid/optim/build.py tests/test_optim_build.py`

### PK Sampler And Training Modes

Implemented:
- `PKBatchSampler` is now finite and no longer uses infinite iteration
- PK sampler now:
  - builds per-identity index pools
  - pads identities with replacement up to `K` when needed
  - shuffles locally with seeded RNGs only
  - yields finite `List[int]` PK batches
- sampler reproducibility is local to the sampler through:
  - `random.Random(seed)`
  - `np.random.default_rng(seed)`
- sampler length is now derived from the actual finite batch plan, so `len(train_loader)` matches the yielded epoch batches
- `build_train_loader(cfg)` now supports:
  - `sampler: pk` for triplet / triplet+ID style training
  - `sampler: random` for ID-only / softmax-only style training
- train loop no longer uses fake `steps_per_epoch` limits
- one epoch is now defined by the dataloader / sampler length
- PK loader safety checks now fail early for:
  - triplet loss with `sampler: random`
  - `P <= 1`
  - `K <= 1`
  - `batch_size != P * K` in PK mode
- strong-baseline Market1501 config now explicitly includes:
  - `sampler: pk`
  - `P: 16`
  - `K: 4`
  - `batch_size: 64`

Validation:
- `PYTHONPATH=. pytest -q tests/test_sampler.py`
- result in this environment: `9 passed`
- `PYTHONPATH=. pytest -q tests/test_sampler.py tests/test_train_loop_optim.py`
- result in this environment: `10 passed, 3 skipped`
- sampler tests verify:
  - PK batches have size `P*K`
  - each batch contains exactly `P` identities
  - each selected identity appears exactly `K` times
  - sampler iteration is finite
  - `len(sampler)` is finite and positive
  - PK loader length is finite
  - random mode works for ID-only
  - triplet + random raises a clear error
  - PK mode rejects invalid `P`, `K`, or mismatched `batch_size`

How To Test:
- focused sampler checks:
  - `PYTHONPATH=. pytest -q tests/test_sampler.py`
- sampler plus train-loop regression:
  - `PYTHONPATH=. pytest -q tests/test_sampler.py tests/test_train_loop_optim.py`
- quick syntax checks:
  - `python3 -m py_compile reid/data/samplers.py reid/data/build.py reid/engine/train_loop.py scripts/train.py tests/test_sampler.py`

Notes:
- CUDA train-loop checks remain skip-gated and were not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026

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
- scheduler still steps per iteration
- milestone semantics are now corrected by converting YAML epoch milestones to iteration milestones during scheduler construction

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

Orchestration / artifact-layer changes are present and should be committed as a separate step after review.

## Suggested Next Step

Run at least one real end-to-end experiment with the refactored orchestration and inspect the new artifacts.

Expected focus:
1. verify `config.resolved.yaml`, `logs/`, `checkpoints/`, `metrics/`, and `plots/` are all created as expected
2. verify `train.save.resume` through a real interrupted-and-resumed run
3. verify `scripts/evaluate.py` on a real checkpoint in both rerank-off and rerank-on settings

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

Run next ResNet50 center-loss experiment:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/experiment_market1501_resnet50_center.yaml
```

Run standalone evaluation:

```bash
python3 scripts/evaluate.py --config configs/experiment_market1501_resnet50_center_rerank.yaml --weight exp/market1501_r50_triplet_id_center_rerank/checkpoints/ckpt_best.pth
```
