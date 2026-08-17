# Cross-Dataset Baseline Experiment Series

This document defines the professional training and evaluation campaign for the
current `ResNet50 + Triplet + ID + BNNeck` baseline across all baseline
datasets already present in this repo:

- `Market1501`
- `DukeMTMC-ReID`
- `CUHK03`
- `MSMT17`

The purpose is to finish the baseline family properly before moving to the next
model such as `PCB`, `MGN`, or `TransReID`.

This file complements:

- `docs/baseline_protocol_v1.md`
- `docs/model_plugin_protocol.md`
- `docs/msmt17_baseline_experiment_series.md`

## Why This Scope

The model is not validated just because one dataset trained successfully.

For a ReID research workflow, the baseline should be closed on every supported
benchmark that you intend to use for future model comparison. In this repo,
that means:

- `Market1501` as the classic baseline benchmark
- `DukeMTMC-ReID` as a second large benchmark with different camera/domain bias
- `CUHK03` as the harder smaller benchmark with protocol sensitivity
- `MSMT17` as the large-scale difficult benchmark

If the next model is evaluated on all four, then the baseline must also be
evaluated on all four.

## Canonical Configs

Use these repo configs as the canonical starting points:

- `configs/baseline_market1501_resnet50_triplet.yaml`
- `configs/baseline_duke_resnet50_triplet.yaml`
- `configs/baseline_cuhk03_resnet50_triplet.yaml`
- `configs/baseline_msmt17_resnet50_triplet.yaml`

These already encode the current strong-baseline protocol:

- image size `256x128`
- `resnet50`
- last stride `1`
- `Adam`
- warmup multi-step scheduler
- Triplet + ID loss
- label smoothing
- BNNeck enabled
- metric feature `raw`
- eval feature `bn`

## Experiment Philosophy

Use the same experimental ladder on each dataset:

1. Smoke run
2. Canonical full baseline run
3. Standalone `ckpt_best` evaluation
4. Standalone `ckpt_last` evaluation
5. Optional post-hoc rerank evaluation
6. Reproducibility runs over multiple seeds
7. Two controlled ablations:
   - Center loss
   - last stride `2`

Only after that should the next architecture be introduced.

## Shared Command Prefix

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
export PYTHONPATH=.
PYTHON=python3
```

Replace `$PYTHON` if you use a different environment.

## Dataset Run Matrix

### 1. Market1501

Canonical config:

- `configs/baseline_market1501_resnet50_triplet.yaml`

Smoke:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=smoke_market1501_r50_triplet \
  experiment.output_dir=exp/smoke_market1501_r50_triplet \
  train.epochs=1 \
  train.eval_interval=1 \
  data.num_workers=0 \
  data.train.batch.P=4 \
  data.train.batch.K=2 \
  data.train.batch.batch_size=8 \
  data.test.batch.size=8 \
  logging.tensorboard=false
```

Full baseline:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml
```

Best checkpoint evaluation:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_market1501_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_market1501/checkpoints/ckpt_best.pth
```

Last checkpoint evaluation:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_market1501_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_market1501/checkpoints/ckpt_last.pth
```

Reproducibility seeds:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_market1501_seed42 \
  experiment.output_dir=exp/baseline_r50_triplet_market1501_seed42 \
  repro.seed=42

$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_market1501_seed43 \
  experiment.output_dir=exp/baseline_r50_triplet_market1501_seed43 \
  repro.seed=43

$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_market1501_seed44 \
  experiment.output_dir=exp/baseline_r50_triplet_market1501_seed44 \
  repro.seed=44
```

Center-loss ablation:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=ablation_market1501_r50_triplet_id_center \
  experiment.output_dir=exp/ablation_market1501_r50_triplet_id_center \
  loss.center.enabled=true
```

Stride-2 ablation:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=ablation_market1501_r50_stride2 \
  experiment.output_dir=exp/ablation_market1501_r50_stride2 \
  model.backbone.last_conv_stride=2
```

### 2. DukeMTMC-ReID

Canonical config:

- `configs/baseline_duke_resnet50_triplet.yaml`

Smoke:

```bash
$PYTHON scripts/train.py --config configs/baseline_duke_resnet50_triplet.yaml -o \
  experiment.name=smoke_duke_r50_triplet \
  experiment.output_dir=exp/smoke_duke_r50_triplet \
  train.epochs=1 \
  train.eval_interval=1 \
  data.num_workers=0 \
  data.train.batch.P=4 \
  data.train.batch.K=2 \
  data.train.batch.batch_size=8 \
  data.test.batch.size=8 \
  logging.tensorboard=false
```

Full baseline:

```bash
$PYTHON scripts/train.py --config configs/baseline_duke_resnet50_triplet.yaml
```

Best checkpoint evaluation:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_duke_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_duke/checkpoints/ckpt_best.pth
```

Last checkpoint evaluation:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_duke_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_duke/checkpoints/ckpt_last.pth
```

Reproducibility seeds:

```bash
$PYTHON scripts/train.py --config configs/baseline_duke_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_duke_seed42 \
  experiment.output_dir=exp/baseline_r50_triplet_duke_seed42 \
  repro.seed=42

$PYTHON scripts/train.py --config configs/baseline_duke_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_duke_seed43 \
  experiment.output_dir=exp/baseline_r50_triplet_duke_seed43 \
  repro.seed=43

$PYTHON scripts/train.py --config configs/baseline_duke_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_duke_seed44 \
  experiment.output_dir=exp/baseline_r50_triplet_duke_seed44 \
  repro.seed=44
```

Center-loss ablation:

```bash
$PYTHON scripts/train.py --config configs/baseline_duke_resnet50_triplet.yaml -o \
  experiment.name=ablation_duke_r50_triplet_id_center \
  experiment.output_dir=exp/ablation_duke_r50_triplet_id_center \
  loss.center.enabled=true
```

Stride-2 ablation:

```bash
$PYTHON scripts/train.py --config configs/baseline_duke_resnet50_triplet.yaml -o \
  experiment.name=ablation_duke_r50_stride2 \
  experiment.output_dir=exp/ablation_duke_r50_stride2 \
  model.backbone.last_conv_stride=2
```

### 3. CUHK03

Canonical config:

- `configs/baseline_cuhk03_resnet50_triplet.yaml`

The repo default is:

- `format: processed`
- `image_type: detected`
- `protocol: new`
- `split_id: 0`

Do not mix CUHK03 protocols inside one comparison table.

Smoke:

```bash
$PYTHON scripts/train.py --config configs/baseline_cuhk03_resnet50_triplet.yaml -o \
  experiment.name=smoke_cuhk03_r50_triplet \
  experiment.output_dir=exp/smoke_cuhk03_r50_triplet \
  train.epochs=1 \
  train.eval_interval=1 \
  data.num_workers=0 \
  data.train.batch.P=4 \
  data.train.batch.K=2 \
  data.train.batch.batch_size=8 \
  data.test.batch.size=8 \
  logging.tensorboard=false
```

Full baseline:

```bash
$PYTHON scripts/train.py --config configs/baseline_cuhk03_resnet50_triplet.yaml
```

Best checkpoint evaluation:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_cuhk03_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_cuhk03_detected/checkpoints/ckpt_best.pth
```

Last checkpoint evaluation:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_cuhk03_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_cuhk03_detected/checkpoints/ckpt_last.pth
```

Reproducibility seeds:

```bash
$PYTHON scripts/train.py --config configs/baseline_cuhk03_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_cuhk03_detected_seed42 \
  experiment.output_dir=exp/baseline_r50_triplet_cuhk03_detected_seed42 \
  repro.seed=42

$PYTHON scripts/train.py --config configs/baseline_cuhk03_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_cuhk03_detected_seed43 \
  experiment.output_dir=exp/baseline_r50_triplet_cuhk03_detected_seed43 \
  repro.seed=43

$PYTHON scripts/train.py --config configs/baseline_cuhk03_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_cuhk03_detected_seed44 \
  experiment.output_dir=exp/baseline_r50_triplet_cuhk03_detected_seed44 \
  repro.seed=44
```

Center-loss ablation:

```bash
$PYTHON scripts/train.py --config configs/baseline_cuhk03_resnet50_triplet.yaml -o \
  experiment.name=ablation_cuhk03_detected_r50_triplet_id_center \
  experiment.output_dir=exp/ablation_cuhk03_detected_r50_triplet_id_center \
  loss.center.enabled=true
```

Stride-2 ablation:

```bash
$PYTHON scripts/train.py --config configs/baseline_cuhk03_resnet50_triplet.yaml -o \
  experiment.name=ablation_cuhk03_detected_r50_stride2 \
  experiment.output_dir=exp/ablation_cuhk03_detected_r50_stride2 \
  model.backbone.last_conv_stride=2
```

### 4. MSMT17

Canonical config:

- `configs/baseline_msmt17_resnet50_triplet.yaml`

Use the detailed dataset-specific notes in:

- `docs/msmt17_baseline_experiment_series.md`

Core full baseline command:

```bash
$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml
```

Core best-checkpoint evaluation:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_msmt17_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_msmt17/checkpoints/ckpt_best.pth
```

## Result Table To Maintain

For each dataset, maintain one results table with:

- `dataset`
- `experiment`
- `seed`
- `checkpoint`
- `best_epoch`
- `mAP`
- `mINP`
- `Rank1`
- `Rank5`
- `Rank10`
- `rerank_mAP` if applicable
- `wall_clock_hours`
- `notes`

## Decision Rules

### Reproducibility

Before moving on, each dataset baseline should have at least three full-seed
runs.

Preferred stability threshold:

- `std(mAP) <= 0.01`

Still acceptable:

- `std(mAP) <= 0.015`

If a dataset baseline is more unstable than that, do not over-interpret small
model differences on that dataset.

### Ablations

Use only targeted ablations that answer a concrete question:

- does `Center loss` help this dataset in this framework?
- does `last_conv_stride=2` hurt or help enough to matter?

Keep the ablation only if it improves the seed-mean baseline by a meaningful
margin. A good working minimum is:

- `>= 0.5` absolute mAP points

### Reranking

Reranking is analysis, not the core identity of the model.

- report base metrics first
- report rerank metrics separately
- compare rerank only against rerank

## When You Can Move To The Next Model

You can move to the next architecture only when:

1. all intended baseline datasets have passed smoke
2. all intended baseline datasets have one canonical full run
3. each canonical run has standalone `ckpt_best` evaluation
4. each dataset has a small reproducibility block
5. the two controlled ablations have been assessed
6. you have frozen one canonical baseline per dataset

Then the next model should be judged against:

- per-dataset baseline mean over seeds
- per-dataset best single run
- training cost
- inference cost if runtime deployment matters

## Practical Recommendation

If you want the most pragmatic order, do this:

1. finish `Market1501`
2. finish `DukeMTMC-ReID`
3. finish `CUHK03`
4. finish `MSMT17`
5. freeze the four-dataset baseline table
6. only then start the next architecture

That gives you one clean, defensible baseline family for the whole repo.
