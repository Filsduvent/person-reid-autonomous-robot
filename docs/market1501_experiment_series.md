# Market1501 Experiment Series

This document defines the first full training and evaluation campaign for the
current baseline on `Market1501`.

The goal is:

1. train and test `Market1501` first
2. sweep the meaningful configuration families already supported by this repo
3. select one winning `Market1501` recipe
4. only then reproduce that same recipe family on the next dataset

This is the correct first step before any broader cross-dataset comparison.

## Scope

This campaign covers the configuration families you explicitly want to test on
`Market1501`:

- `triplet only`
- `id only`
- `triplet + id`
- `triplet + id + center`
- `BNNeck` ablation
- `last_conv_stride = 1` versus `2`
- all implemented Bag-of-Tricks together
- rerank analysis

This is not an exhaustive combinatorial explosion. It is a controlled sweep
that answers the main research questions with a manageable number of runs.

## Canonical Dataset And Base Config

Dataset:

- `Market1501`

Base config:

- `configs/baseline_market1501_resnet50_triplet.yaml`

This base config already includes the current strong baseline defaults:

- `resnet50`
- `last_conv_stride=1`
- `Triplet + ID`
- `label_smoothing=0.1`
- `random erasing`
- `BNNeck=true`
- `metric_feat=raw`
- `eval_feat=bn`
- `PK sampler`

In other words, the current base config is already the main
`Bag-of-Tricks-style strong baseline` in this repo.

## Existing Anchors In This Repo

Current baseline anchor:

- `exp/baseline_r50_triplet_market1501/metrics/final_test.json`
  - `mAP = 0.8578`
  - `mINP = 0.9601`
  - `Rank-1 = 0.9439`

Current center-loss anchor:

- `exp/market1501_r50_triplet_id_center/metrics/latest_val.json`
  - `mAP = 0.8609`
  - `mINP = 0.9618`
  - `Rank-1 = 0.9463`

Current center-loss + rerank anchor:

- `exp/market1501_r50_triplet_id_center_rerank/metrics/latest_val.json`
  - base `mAP = 0.8626`
  - rerank `mAP = 0.9426`
  - rerank `Rank-1 = 0.9561`

Use these only as anchors. The point of this document is to rerun the full
series cleanly and compare all variants under one structured protocol.

## Shared Command Prefix

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
export PYTHONPATH=.
PYTHON=python3
```

Replace `$PYTHON` if you use a different environment.

## Current Execution State

Currently running:

- `market1501_r50_triplet_only`
- config: `configs/experiment_market1501_resnet50_triplet_only.yaml`
- launch command:

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
python3 scripts/train.py --config configs/experiment_market1501_resnet50_triplet_only.yaml
```

Queued next in Phase 1:

1. `market1501_r50_id_only`
2. `market1501_r50_triplet_id`
3. `market1501_r50_triplet_id_center`

If you come back later, finish those three before moving to Phase 2.

## Phase 0: Mandatory Smoke

Run this before every long campaign restart.

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

Accept only if:

- `ckpt_last.pth` exists
- `metrics/final_test.json` exists
- no NaN/Inf appears in the log

## Phase 1: Loss-Mode Sweep

This phase answers: which loss family is fundamentally strongest on
`Market1501` inside this framework?

### 1. Triplet Only

Purpose:

- tests pure metric learning without ID classification

Command:

```bash
$PYTHON scripts/train.py --config configs/experiment_market1501_resnet50_triplet_only.yaml
```

### 2. ID Only

Purpose:

- tests pure classification without triplet
- should use `random` sampler, not `pk`

Command:

```bash
$PYTHON scripts/train.py --config configs/experiment_market1501_resnet50_id_only.yaml
```

### 3. Triplet + ID

Purpose:

- this is the current canonical strong baseline

Command:

```bash
$PYTHON scripts/train.py --config configs/experiment_market1501_resnet50_triplet_id.yaml
```

### 4. Triplet + ID + Center

Purpose:

- tests whether center loss adds value on top of the strong baseline

Command:

```bash
$PYTHON scripts/train.py --config configs/experiment_market1501_resnet50_triplet_id_center.yaml
```

### Phase 1 Resume Queue

After the current `triplet_only` run ends, run these commands in order:

1. ID only

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
python3 scripts/train.py --config configs/experiment_market1501_resnet50_id_only.yaml
```

2. Triplet + ID

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
python3 scripts/train.py --config configs/experiment_market1501_resnet50_triplet_id.yaml
```

3. Triplet + ID + Center

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
python3 scripts/train.py --config configs/experiment_market1501_resnet50_triplet_id_center.yaml
```

### Phase 1 Decision Rule

At the end of Phase 1, compare:

- `triplet_only`
- `id_only`
- `triplet_id`
- `triplet_id_center`

Pick the strongest family by:

1. `mAP`
2. `Rank-1`
3. training stability

The expected likely winner is `triplet_id` or `triplet_id_center`, but do not
assume it; verify it.

## Phase 2: Bag-of-Tricks Structure Sweep

This phase answers: among the training tricks already implemented, which ones
matter most on `Market1501`?

### 5. Full Bag-of-Tricks Strong Baseline

Purpose:

- this is the reference “all implemented tricks together” run

Included tricks:

- `triplet + id`
- `label smoothing`
- `random erasing`
- `BNNeck`
- `stride 1`
- `eval_feat=bn`

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_full_bag_of_tricks \
  experiment.output_dir=exp/market1501_full_bag_of_tricks
```

### 6. BNNeck Off Ablation

Purpose:

- isolates the value of BNNeck

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_bnneck_off \
  experiment.output_dir=exp/market1501_bnneck_off \
  model.head.bnneck=false \
  model.head.metric_feat=raw \
  model.head.eval_feat=raw
```

### 7. Stride-2 Ablation

Purpose:

- compares standard ImageNet stride against ReID-favored stride 1

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_stride2 \
  experiment.output_dir=exp/market1501_stride2 \
  model.backbone.last_conv_stride=2
```

### 8. No Random Erasing Ablation

Purpose:

- isolates the value of random erasing inside the current strong baseline

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_no_random_erasing \
  experiment.output_dir=exp/market1501_no_random_erasing \
  data.train.aug.random_erasing.enabled=false
```

### 9. No Label Smoothing Ablation

Purpose:

- isolates the value of label smoothing

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_no_label_smoothing \
  experiment.output_dir=exp/market1501_no_label_smoothing \
  loss.id.label_smoothing=0.0
```

### 10. Full Bag-of-Tricks + Center

Purpose:

- tests the strongest “everything together” version currently supported

Command:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_full_bag_of_tricks_center \
  experiment.output_dir=exp/market1501_full_bag_of_tricks_center \
  loss.center.enabled=true
```

### Phase 2 Decision Rule

Measure each ablation relative to `market1501_full_bag_of_tricks`.

Keep a trick in the final recipe only if it gives a meaningful gain or a clear
stability improvement. A good minimum threshold is:

- `>= 0.3` absolute mAP points on `Market1501`

## Phase 3: Post-Hoc Evaluation Sweep

This phase answers: how much comes from the learned model versus the evaluation
strategy?

Important:

- rerank is an evaluation trick
- it does not change training
- evaluate the same checkpoint twice with rerank off/on

### 11. Best Checkpoint Standalone Eval

Run this for every candidate you want to compare seriously.

Example for the full strong baseline:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_market1501_resnet50_triplet.yaml \
  --weight exp/market1501_full_bag_of_tricks/checkpoints/ckpt_best.pth \
  --opts \
  experiment.name=market1501_full_bag_of_tricks_eval \
  experiment.output_dir=exp/market1501_full_bag_of_tricks_eval
```

### 12. Same Checkpoint With Rerank Enabled

Example for the same checkpoint:

```bash
$PYTHON scripts/evaluate.py \
  --config configs/baseline_market1501_resnet50_triplet.yaml \
  --weight exp/market1501_full_bag_of_tricks/checkpoints/ckpt_best.pth \
  --opts \
  experiment.name=market1501_full_bag_of_tricks_rerank_eval \
  experiment.output_dir=exp/market1501_full_bag_of_tricks_rerank_eval \
  eval.rerank.enabled=true
```

Apply the same pair of commands to:

- `market1501_triplet_id_center`
- `market1501_full_bag_of_tricks_center`
- any other top candidate

## Phase 4: Reproducibility For The Winning Recipes

Do not run seeds for every weak configuration. That wastes time.

After Phases 1 to 3, pick the top `2` Market1501 recipes and run three seeds
for each.

Example for the winning recipe:

```bash
$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_winner_seed42 \
  experiment.output_dir=exp/market1501_winner_seed42 \
  repro.seed=42

$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_winner_seed43 \
  experiment.output_dir=exp/market1501_winner_seed43 \
  repro.seed=43

$PYTHON scripts/train.py --config configs/baseline_market1501_resnet50_triplet.yaml -o \
  experiment.name=market1501_winner_seed44 \
  experiment.output_dir=exp/market1501_winner_seed44 \
  repro.seed=44
```

If the winner uses extra overrides such as `loss.center.enabled=true`, keep the
same overrides on all three seed runs.

Target stability:

- preferred: `std(mAP) <= 0.01`
- acceptable: `std(mAP) <= 0.015`

## Minimal Result Table

Keep one table with:

- `experiment`
- `family`
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

## Final Decision For Market1501

At the end of this campaign, you must select exactly one canonical
`Market1501` recipe to carry to the next dataset.

The chosen recipe should be the winner after considering:

1. base `mAP`
2. base `Rank-1`
3. rerank gain as analysis only
4. reproducibility across seeds
5. training cost

## When To Leave Market1501

You are ready to leave `Market1501` only when:

1. the smoke run passed
2. the loss-mode sweep is complete
3. the Bag-of-Tricks sweep is complete
4. rerank A/B was done on the top candidates
5. the winning recipe was repeated across seeds
6. one canonical `Market1501` recipe was frozen

Only then should you reproduce that same winning recipe on the next dataset.
