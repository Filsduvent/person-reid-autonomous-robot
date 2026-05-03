# Lab Validation Plan

Use this document before launching full training on the university machine.
The order goes from small, fast checks to full one-epoch dataset/model smoke
runs. Stop at the first failure, fix it, then continue.

Assumed command prefix:

```bash
cd ~/UFPR/2026/person_reid_auto_robot_offline
export PYTHONPATH=.
PYTHON=/home/filsduvent/environments/windflow_env/bin/python
```

If the lab environment uses a different Python path, replace `$PYTHON` with the
active environment Python.

## 1. Verify Test Runner

Command:

```bash
$PYTHON -m pytest --version
```

What it checks:

- Confirms `pytest` is installed in the active environment.

Expected result:

- Prints a pytest version, for example `pytest 9.0.3`.

## 2. Documentation Protocol Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_baseline_protocol_doc.py tests/test_model_plugin_protocol_doc.py
```

What it checks:

- `docs/baseline_protocol_v1.md` exists and contains the frozen baseline protocol sections.
- `docs/model_plugin_protocol.md` exists and contains the allowed/forbidden future model integration boundaries.

Expected result:

- All tests pass.

## 3. Config Schema Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_config_schema.py
```

What it checks:

- Required YAML sections exist.
- Required config keys exist.
- Invalid `system.device` fails early.
- MSMT17 config keeps reranking disabled by default.

Expected result:

- All tests pass.

## 4. Collate And Dataset Protocol Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_collate.py tests/test_dataset_protocol.py
```

What it checks:

- Train batches collate as `(imgs, labels)`.
- Test batches collate as `(imgs, pids, camids, names, marks)`.
- Dataset builders return the locked loader contracts.
- Train loader returns `(loader, num_classes)`.
- Test loader works without evaluator changes.

Expected result:

- All tests pass.

## 5. Dataset Parser And Loader Unit Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_market1501_dataset.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_msmt17_dataset.py
```

What it checks:

- Market1501 processed/raw parsers.
- Duke processed/raw parsers.
- CUHK03 processed detected/labeled parser and partitions contract.
- MSMT17 raw list parser and camera normalization.
- Query-before-gallery ordering.
- Train returns `(image, label)`.
- Test returns `(image, pid, camid, image_name, mark)`.

Expected result:

- All tests pass.
- These tests use synthetic temporary data where possible, so they do not require your full datasets.

## 6. Transform And Sampler Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_data_transforms.py tests/test_sampler.py
```

What it checks:

- Train transform order:
  - resize
  - horizontal flip
  - padding
  - random crop
  - tensor
  - normalize
  - random erasing
- Test transform order:
  - resize
  - tensor
  - normalize
- PK sampler produces finite batches.
- `len(train_loader)` is valid.
- Triplet loss requires `sampler: pk`.
- Random sampler works for ID-only mode.

Expected result:

- All tests pass.

## 7. Ranking, Distance, And Evaluation Harness Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_rerank.py tests/test_evaluation_harness.py
```

What it checks:

- Re-ranking returns finite query-gallery matrices.
- Re-ranking metrics are reported separately.
- MSMT17 rerank warning is dataset-specific.
- Evaluator depends only on `outputs["emb"]`.
- Core metrics always include:
  - `mAP`
  - `mINP`
  - `Rank1`
  - `Rank5`
  - `Rank10`
- Distance modes work:
  - `euclidean`
  - `cosine`

Expected result:

- All tests pass.

## 8. Model Interface Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_model_interface.py tests/test_model_forward.py
```

What it checks:

- ResNet50 baseline forwards on CPU.
- `last_conv_stride: 1|2` works.
- Model output dict contains:
  - `feat_raw`
  - `feat_bn`
  - `emb`
  - `logits`
- `eval_feat: raw|bn` controls `emb`.
- Evaluator works with both eval feature settings.

Expected result:

- CPU tests pass.
- CUDA-specific tests may be skipped if CUDA is not visible.

## 9. Model Plug-In Contract Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_model_plugin_contract.py
```

What it checks:

- A dummy model that does not go through `build_model` works with:
  - criterion
  - optimizer
  - scheduler
  - train loop
  - evaluator
- This proves the framework is model-agnostic.

Expected result:

- All tests pass.

## 10. Loss Interface Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_loss_interface.py tests/test_reid_loss_modes.py
```

What it checks:

- `criterion(outputs, labels)` returns `(loss, logs)`.
- Logs contain standard keys.
- Triplet and Center route through `feat_raw` or `feat_bn`.
- ID loss routes through `logits`.
- Loss builder does not depend on model class names.
- These modes run:
  - Triplet only
  - ID only
  - Triplet + ID
  - Triplet + ID + Center

Expected result:

- All tests pass.

## 11. Optimizer And Scheduler Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_optim_build.py tests/test_train_loop_optim.py
```

What it checks:

- Optimizers:
  - SGD
  - Adam
  - AdamW
- `bias_lr_factor`
- `weight_decay_bias`
- frozen parameters are skipped
- WarmupMultiStepLR
- MultiStepLR
- optional center optimizer
- training loop logs loss, LR, speed, and ID accuracy when available

Expected result:

- CPU tests pass.
- CUDA tests may be skipped if CUDA is not visible.

## 12. Checkpoint And Artifact Format Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_checkpoint.py tests/test_artifact_format.py tests/test_reproducibility_artifacts.py tests/test_train_orchestration.py
```

What it checks:

- Checkpoint payload includes:
  - epoch
  - model
  - optimizer
  - scheduler
  - center optimizer
  - scores
  - cfg
- Resume restores optimizer/scheduler/center optimizer.
- Metrics write the locked schema.
- `latest_test.json`, `test_epoch_XXX.json`, and `final_test.json` share schema.
- Command and environment artifacts are written.
- Resolved config is saved.
- Train log exists and contains useful fields.

Expected result:

- All tests pass.

## 13. ResNet50 Strong Baseline Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_resnet50_strong_baseline.py
```

What it checks:

- Baseline config includes:
  - ResNet50 backbone
  - BNNeck
  - Triplet loss
  - ID loss
  - label smoothing
  - optional center loss
  - random erasing
  - WarmupMultiStepLR
  - bias LR factor
  - PK sampler
  - reranking config
- YAML-only smoke for:
  - Triplet only
  - ID only
  - Triplet + ID
  - Triplet + ID + Center

Expected result:

- All tests pass.

## 14. Existing Pipeline Smoke Unit Checks

Command:

```bash
$PYTHON -m pytest -q tests/test_smoke_reid_pipeline.py tests/test_smoke_test_matrix.py
```

What it checks:

- The old smoke helper validates loader, batch, model forward, and loss checks.
- The new smoke matrix covers all four datasets.
- Smoke matrix commands include one-epoch overrides.

Expected result:

- All tests pass.

## 15. Focused Framework Regression

Command:

```bash
$PYTHON -m pytest -q \
  tests/test_config_schema.py \
  tests/test_dataset_protocol.py \
  tests/test_data_transforms.py \
  tests/test_sampler.py \
  tests/test_evaluation_harness.py \
  tests/test_model_interface.py \
  tests/test_model_plugin_contract.py \
  tests/test_loss_interface.py \
  tests/test_optim_build.py \
  tests/test_artifact_format.py \
  tests/test_smoke_test_matrix.py
```

What it checks:

- The full locked framework interfaces without running the slower ResNet-heavy tests.

Expected result:

- All tests pass.

## 16. Full Test Suite

Command:

```bash
$PYTHON -m pytest -q tests
```

What it checks:

- Everything in the repository test suite.

Expected result:

- All tests pass.
- Some CUDA tests may be skipped if CUDA is not visible.
- On the CPU environment used during development, the latest broad run was:
  - `235 passed, 4 skipped`

## 17. Real Dataset Loader Smoke On Lab Machine

Set the dataset root first:

```bash
DATASET_ROOT=/path/to/Dataset
```

Dry-run the commands:

```bash
$PYTHON scripts/smoke_test.py --dry-run --root "$DATASET_ROOT"
```

What it checks:

- Prints the exact one-epoch training commands for:
  - Market1501
  - Duke
  - CUHK03
  - MSMT17

Expected result:

- Four commands are printed.
- No training is executed.

## 18. Real Dataset Full Smoke Matrix

Command:

```bash
$PYTHON scripts/smoke_test.py --root "$DATASET_ROOT"
```

What it checks for each dataset:

- builds train loader
- builds test loader
- runs one train epoch
- runs evaluation
- saves `ckpt_last.pth`
- saves `ckpt_best.pth` if the configured best metric improves
- saves metric JSON artifacts

Expected result:

- Four smoke runs complete.
- Check these folders:
  - `exp/smoke_market1501_resnet50/`
  - `exp/smoke_duke_resnet50/`
  - `exp/smoke_cuhk03_resnet50/`
  - `exp/smoke_msmt17_resnet50/`
- Each should contain:
  - `config.resolved.yaml`
  - `train.log`
  - `checkpoints/ckpt_last.pth`
  - `metrics/latest_test.json`
  - `metrics/test_epoch_001.json`
  - `metrics/final_test.json`
  - `artifacts/command.txt`
  - `artifacts/environment.txt`

If one dataset is not available yet, run a single dataset:

```bash
$PYTHON scripts/smoke_test.py --datasets market1501 --root "$DATASET_ROOT"
```

## 19. First Real One-Epoch Training Test

Start with Market1501 because it is the baseline reference.

Command:

```bash
$PYTHON scripts/train.py \
  --config configs/baseline_market1501_resnet50_triplet.yaml \
  -o \
  data.root="$DATASET_ROOT" \
  model.backbone.pretrained=false \
  train.epochs=1 \
  train.eval_interval=1 \
  data.train.batch.P=4 \
  data.train.batch.K=2 \
  data.train.batch.batch_size=8 \
  data.test.batch.size=8 \
  data.num_workers=0 \
  logging.tensorboard=false \
  experiment.name=manual_market1501_one_epoch \
  experiment.output_dir=exp/manual_market1501_one_epoch
```

What it checks:

- Real dataset loading
- Real training loop
- Real evaluation
- Real checkpoints
- Real metric files
- Real artifact files

Expected result:

- Training completes one epoch.
- Inspect:
  - `exp/manual_market1501_one_epoch/train.log`
  - `exp/manual_market1501_one_epoch/checkpoints/ckpt_last.pth`
  - `exp/manual_market1501_one_epoch/metrics/latest_test.json`
  - `exp/manual_market1501_one_epoch/metrics/final_test.json`

## 20. First Full Baseline Training

After the one-epoch test passes, launch the full Market1501 baseline.

Command:

```bash
$PYTHON scripts/train.py \
  --config configs/baseline_market1501_resnet50_triplet.yaml \
  -o data.root="$DATASET_ROOT"
```

What it checks:

- Full baseline protocol under real training duration.
- This is the point where model performance starts to matter.

Expected result:

- Long-running training job.
- Check `train.log`, TensorBoard, checkpoints, metrics, and plots during training.

## Recommended Lab Order

Run in this order tomorrow:

1. Steps 1-4
2. Step 5
3. Steps 6-14
4. Step 16 full suite
5. Step 17 dry-run smoke matrix
6. Step 18 real dataset smoke matrix
7. Step 19 one-epoch Market1501 training
8. Step 20 full baseline training

Only move to full training after the smoke matrix and one-epoch Market1501 run
complete successfully.
