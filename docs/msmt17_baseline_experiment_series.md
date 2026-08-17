# MSMT17 Baseline Experiment Series

This document defines the exact training and evaluation campaign to finish the
current ResNet50 baseline on MSMT17 before moving to the next model.

It is not a brainstorming list. It is a gated run order with:

- fixed baseline
- copy-pasteable commands
- targeted ablations only
- decision criteria for when the baseline is complete enough

The baseline protocol remains the frozen framework in
`docs/baseline_protocol_v1.md`. This file only defines the run series.

## Current Anchor

Current completed run:

- experiment: `baseline_r50_triplet_msmt17`
- config: `configs/baseline_msmt17_resnet50_triplet.yaml`
- checkpoint: `exp/baseline_r50_triplet_msmt17/checkpoints/ckpt_best.pth`
- final reported test metrics in `exp/baseline_r50_triplet_msmt17/metrics/final_test.json`:
  - `mAP = 0.5238`
  - `mINP = 0.8044`
  - `Rank-1 = 0.7560`
  - `Rank-5 = 0.8620`
  - `Rank-10 = 0.8948`

Use this run as the reference point for every new result.

## Baseline Under Study

Locked baseline recipe:

- dataset: `msmt17`
- backbone: `resnet50`
- last stride: `1`
- image size: `256x128`
- train sampler: `pk`
- batch layout: `P=16`, `K=4`, `batch_size=64`
- losses: Triplet + ID
- label smoothing: `0.1`
- BNNeck: enabled
- metric feature: `raw`
- eval feature: `bn`
- optimizer: `Adam`
- learning rate: `3e-4`
- scheduler: warmup multi-step
- milestones: `40`, `70`
- epochs: `120`
- rerank: disabled during normal training-time evaluation

## Environment

Assumed command prefix:

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
export PYTHONPATH=.
PYTHON=python3
```

If you use a different environment Python, replace `$PYTHON`.

## Experiment Order

Run the experiments in this exact order. Do not start ablations before the
reproducibility block is complete.

### 1. One-Epoch Smoke On MSMT17

Purpose:

- confirms the machine, dataset path, loader, train loop, evaluation, and
  checkpoint writing are healthy before another long run

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml -o \
  experiment.name=smoke_msmt17_r50_triplet \
  experiment.output_dir=exp/smoke_msmt17_r50_triplet \
  train.epochs=1 \
  train.eval_interval=1 \
  data.num_workers=0 \
  data.train.batch.P=4 \
  data.train.batch.K=2 \
  data.train.batch.batch_size=8 \
  data.test.batch.size=8 \
  logging.tensorboard=false
```

Accept only if:

- `exp/smoke_msmt17_r50_triplet/checkpoints/ckpt_last.pth` exists
- `exp/smoke_msmt17_r50_triplet/metrics/final_test.json` exists
- no NaN/Inf appears in the log

### 2. Canonical Full Baseline Re-Run

Purpose:

- produces the official baseline run that all later models must beat
- verifies the current anchor is reproducible on the target machine

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml
```

Required artifacts:

- `exp/baseline_r50_triplet_msmt17/checkpoints/ckpt_best.pth`
- `exp/baseline_r50_triplet_msmt17/checkpoints/ckpt_last.pth`
- `exp/baseline_r50_triplet_msmt17/metrics/final_test.json`
- `exp/baseline_r50_triplet_msmt17/plots/`
- `exp/baseline_r50_triplet_msmt17/config.resolved.yaml`

Record:

- best epoch by `mAP`
- final `mAP`, `mINP`, `Rank-1`, `Rank-5`, `Rank-10`
- total wall-clock time
- average images/sec from the training log

### 3. Standalone Best-Checkpoint Evaluation

Purpose:

- verifies that post-training evaluation of `ckpt_best.pth` matches the
  training-produced artifact path
- gives you a clean evaluation command to reuse later for future models

Command:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_msmt17_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_msmt17/checkpoints/ckpt_best.pth
```

Accept only if:

- `metrics/eval_ckpt_best.pth.json` appears under the same experiment
- metrics are numerically consistent with the training-selected best checkpoint

### 4. Standalone Last-Checkpoint Evaluation

Purpose:

- measures whether the training tail is still improving or has already peaked

Command:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_msmt17_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_msmt17/checkpoints/ckpt_last.pth
```

Interpretation:

- if `ckpt_last` is clearly worse than `ckpt_best`, training is peaking before
  the end and the best-checkpoint protocol is justified
- if `ckpt_last` is equal or better, the schedule is still reasonable and the
  run is not degrading late

### 5. Post-Hoc Rerank Evaluation On Best Checkpoint

Purpose:

- measures the retrieval ceiling of the current baseline without changing the
  training recipe
- keeps reranking additive, never a replacement for base metrics

Command:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_msmt17_resnet50_triplet.yaml \
  --weight exp/baseline_r50_triplet_msmt17/checkpoints/ckpt_best.pth \
  --opts \
  experiment.name=baseline_r50_triplet_msmt17_rerank_eval \
  experiment.output_dir=exp/baseline_r50_triplet_msmt17_rerank_eval \
  eval.rerank.enabled=true
```

Record separately:

- base metrics
- rerank metrics

Decision rule:

- use rerank only as an analysis number
- do not use rerank to declare the model itself better than another model unless
  the comparison also uses rerank on both sides

### 6. Reproducibility Block: Three Full-Seed Runs

Purpose:

- establishes whether the baseline is stable enough that model-to-model
  comparisons will be trustworthy

Run these three experiments:

```bash
$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_msmt17_seed42 \
  experiment.output_dir=exp/baseline_r50_triplet_msmt17_seed42 \
  repro.seed=42

$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_msmt17_seed43 \
  experiment.output_dir=exp/baseline_r50_triplet_msmt17_seed43 \
  repro.seed=43

$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml -o \
  experiment.name=baseline_r50_triplet_msmt17_seed44 \
  experiment.output_dir=exp/baseline_r50_triplet_msmt17_seed44 \
  repro.seed=44
```

Collect from each run:

- best-checkpoint `mAP`
- best-checkpoint `Rank-1`
- best epoch

Then compute:

- mean `mAP`
- std `mAP`
- mean `Rank-1`
- std `Rank-1`

Completion criterion:

- preferred: `std(mAP) <= 0.01`
- acceptable: `std(mAP) <= 0.015`

If variance is larger than that, do not move to the next model yet. First make
sure the comparison protocol is stable enough.

### 7. Targeted Ablation A: Center Loss

Purpose:

- tests the most meaningful additive baseline extension already supported by the
  framework

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml -o \
  experiment.name=ablation_msmt17_r50_triplet_id_center \
  experiment.output_dir=exp/ablation_msmt17_r50_triplet_id_center \
  loss.center.enabled=true
```

Compare against the seed-mean baseline, not against a single lucky run.

Decision rule:

- keep Center loss only if it improves mean `mAP` or mean `Rank-1` by a
  meaningful margin
- for this repo, use `>= 0.5` absolute mAP points as the minimum meaningful
  gain

### 8. Targeted Ablation B: Last Stride 2

Purpose:

- tests a lower-resolution backbone tail that may reduce detail but is useful as
  a controlled baseline sensitivity check

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_msmt17_resnet50_triplet.yaml -o \
  experiment.name=ablation_msmt17_r50_stride2 \
  experiment.output_dir=exp/ablation_msmt17_r50_stride2 \
  model.backbone.last_conv_stride=2
```

Decision rule:

- if stride-2 loses clearly to stride-1, keep stride-1 as the canonical
  baseline and stop exploring this branch
- if stride-2 is competitive while being lighter or easier to train on your
  hardware, record that as an engineering tradeoff, not necessarily as the main
  scientific baseline

## Minimal Result Table To Maintain

Keep one table for all runs with these columns:

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

This table is the evidence you need before claiming the baseline is complete.

## When You Are Allowed To Move To The Next Model

You can move to the next architecture only when all of these are true:

1. The one-epoch smoke run passed.
2. The canonical baseline full run completed cleanly.
3. Standalone evaluation of `ckpt_best.pth` completed cleanly.
4. You have at least three full-seed baseline runs.
5. Baseline variance is acceptable.
6. You tested the two targeted ablations above.
7. You selected one canonical MSMT17 baseline to beat:
   - usually `ResNet50 + Triplet + ID + BNNeck`, stride `1`
   - optionally `+ Center loss` if it gives a real average gain

At that point, the next model should be judged against:

- baseline mean over seeds
- best single run
- training cost
- inference cost if deployment matters

## What Not To Do

- do not compare a new model against only one weak baseline run
- do not use rerank as the only comparison metric
- do not change multiple core variables at once
- do not declare victory from a single lucky seed
- do not move to the next model before the reproducibility block is done

## Recommended Final Outcome For This Stage

The expected final output of this stage is one short conclusion such as:

`The canonical MSMT17 baseline is ResNet50 + Triplet + ID + BNNeck + stride1,
trained for 120 epochs with PK 16x4. Over 3 seeds it achieved mean mAP X and
mean Rank-1 Y. Center loss did or did not help. Stride-2 did or did not help.
This is the baseline that the next model must beat.`
