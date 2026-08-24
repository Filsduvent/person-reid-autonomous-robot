# Baseline Protocol v1

This document freezes the ReID baseline protocol used by the offline framework.
Future model configs should inherit this protocol and change only the model
entry, model-specific settings, and optional model-specific losses.

## Scope

Supported datasets:

- Market1501: processed and raw
- DukeMTMC-ReID: processed and raw
- CUHK03: processed detected/labeled
- MSMT17: raw

Locked dataset contract:

- Train item: `(image, label)`
- Train metadata: `dataset.labels`, `dataset.num_classes`
- Test item: `(image, pid, camid, image_name, mark)`
- Test marks: `0=query`, `1=gallery`, `2=multi-query optional`
- Test ordering: query samples first, gallery samples second

All dataset-specific parsing and path rules stay inside `reid/data/*` modules.
Training, evaluation, metrics, checkpointing, and artifact code must not add
dataset-specific branches.

## Config Schema

Every experiment config must contain:

- `experiment`
- `system`
- `repro`
- `logging`
- `data`
- `model`
- `loss`
- `optim`
- `sched`
- `train`
- `eval`

Required keys:

- `experiment.name`
- `experiment.output_dir`
- `system.device`: `auto`, `cpu`, or `cuda`
- `data.train.dataset.name`
- `data.test.dataset.name`
- `model.name`
- `optim.name`
- `train.epochs`
- `eval.topk`

## Dataset Preparation

Expected dataset roots are controlled by YAML through `data.root`.

Market1501:

- Processed: `root/market1501/images`, `root/market1501/partitions.pkl`
- Raw: `root/market1501/Market-1501-v15.09.15/{bounding_box_train,query,bounding_box_test}`
- Raw junk pid `-1` is ignored
- Raw camera IDs are normalized to zero-based

DukeMTMC-ReID:

- Processed: `root/duke/images`, `root/duke/partitions.pkl`
- Raw: `root/duke/DukeMTMC-reID/{bounding_box_train,query,bounding_box_test}`
- Raw camera IDs are normalized to zero-based

CUHK03:

- Processed detected: `root/cuhk03/detected/images`, `root/cuhk03/detected/partitions.pkl`
- Processed labeled: `root/cuhk03/labeled/images`, `root/cuhk03/labeled/partitions.pkl`
- Supports `image_type: detected|labeled`
- Canonical current setting: `protocol: processed_partition`, `split_id: null`
- The supplied flat partition files do not encode CUHK03 `new|classic` metadata
  or split IDs. Those settings are rejected rather than silently claimed.

MSMT17:

- Raw: `root/msmt17/MSMT17_V2`
- Uses official list files: `list_train.txt`, `list_val.txt`, `list_query.txt`, `list_gallery.txt`
- Image dirs: `mask_train_v2`, `mask_test_v2`
- Camera IDs are normalized consistently from list-file names

## Transforms

Train transform order:

1. Resize
2. Random horizontal flip when `data.train.aug.mirror: random`
3. Padding when enabled
4. Random crop when enabled
5. To tensor
6. Normalize
7. Random erasing when enabled

Test transform order:

1. Resize
2. To tensor
3. Normalize

Test transforms must not include train-only augmentation.

## Sampler

Supported train sampler modes:

- `sampler: pk` for Triplet and Triplet+ID modes
- `sampler: random` for ID-only mode

PK sampler requirements:

- `P > 1`
- `K > 1`
- `batch_size == P * K`
- finite `len(train_loader)`

If Triplet loss is enabled, `sampler` must be `pk`.

## Model Settings

Frozen ResNet50 strong baseline:

- `model.name: reid_baseline`
- `model.backbone.name: resnet50`
- `model.backbone.last_conv_stride: 1|2`
- `model.head.pooling: gap`
- `model.head.bnneck: true`
- `model.head.embedding_dim: 2048`
- `model.head.normalize: true`
- `model.head.metric_feat: raw|bn`
- `model.head.eval_feat: raw|bn`
- `model.head.classifier: true|false`

Locked model output contract:

```python
{
    "feat_raw": Tensor,
    "feat_bn": Tensor,
    "emb": Tensor,
    "logits": Tensor | None,
}
```

Usage:

- `emb` is used by evaluation
- `logits` is used by ID loss
- `feat_raw` or `feat_bn` is used by metric losses through `model.head.metric_feat`

Future models must satisfy this output contract before entering training.

## Losses

Supported baseline loss components:

- Batch-hard Triplet loss
- ID cross entropy
- ID cross entropy with label smoothing
- Optional Center loss

Loss interface:

```python
loss, logs = criterion(outputs, labels)
```

Required/standard log keys:

- `loss/total`
- `loss/triplet`
- `loss/id`
- `loss/center`

Feature routing:

- Triplet uses `feat_raw` or `feat_bn`
- Center uses the same metric feature as Triplet
- ID uses `logits`

Loss code must not depend on model class names.

## Optimizer

Supported optimizers:

- SGD
- Adam
- AdamW

Parameter group behavior:

- Uses `model.named_parameters()`
- Skips frozen parameters
- Applies `bias_lr_factor` to parameter names containing `bias`
- Applies `weight_decay_bias` to parameter names containing `bias`

Optional Center loss optimizer:

- Created only when Center loss is enabled and exposes trainable center parameters
- Uses `loss.center.lr`

## Scheduler

Supported scheduler modes:

- `warmup_multistep`
- `step` / PyTorch `MultiStepLR`
- disabled with `none`, `disabled`, or empty name

Warmup multi-step behavior:

- YAML milestones are epoch milestones
- Builder converts them to iteration milestones with `steps_per_epoch=len(train_loader)`
- Training loop steps the scheduler once per iteration

No hardcoded `steps_per_epoch` is allowed.

## Evaluation

Evaluator input:

- The evaluator only depends on `outputs["emb"]`
- No model internals are accessed

Distance options:

- `eval.distance: euclidean`
- `eval.distance: cosine`

Always reported metrics:

- `mAP`
- `mINP`
- `Rank1`
- `Rank5`
- `Rank10`

`mINP` uses the last valid positive match after same-identity/same-camera
filtering: `number_of_valid_positives / rank_of_last_valid_positive`.

Optional reranking metrics:

- `rerank_mAP`
- `rerank_mINP`
- `rerank_Rank1`
- `rerank_Rank5`
- `rerank_Rank10`

Reranking metrics are reported separately and never replace original metrics.
MSMT17 reranking is disabled by default because its gallery can be large.

## Checkpoint Format

Checkpoints are written under `checkpoints/`:

- `ckpt_last.pth`
- `ckpt_best.pth`

Checkpoint payload:

```python
{
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "center_optimizer": center_optimizer.state_dict() or None,
    "scores": scores,
    "cfg": cfg,
}
```

Best checkpoint selection is controlled by:

```yaml
train:
  save:
    metric: mAP
```

`ckpt_last.pth` is saved after every epoch. `ckpt_best.pth` is replaced only
when the configured metric strictly improves at an evaluation epoch.

## Artifact Schema

Each run writes:

```text
exp/<experiment.name>/
  config.resolved.yaml
  config.yaml
  train.log
  logs/
    stdout.txt
    stderr.txt
  checkpoints/
    ckpt_last.pth
    ckpt_best.pth
  metrics/
    latest_val.json
    val_epoch_XXX.json
    final_epoch_test.json  # final training checkpoint metrics, ckpt_last.pth
    final_test.json        # selected checkpoint metrics, normally ckpt_best.pth
  tensorboard/
  plots/
  artifacts/
    command.txt
    environment.txt
    git_commit.txt  # when available
```

Metric JSON schema:

```json
{
  "dataset": "market1501",
  "split": "test",
  "model": "resnet50_baseline",
  "epoch": 120,
  "checkpoint": "ckpt_best.pth",
  "mAP": 0.0,
  "mINP": 0.0,
  "Rank1": 0.0,
  "Rank5": 0.0,
  "Rank10": 0.0
}
```

## Cross-Domain Evaluation

Within-domain evaluation uses the same source and target dataset. Cross-domain
evaluation loads a source checkpoint directly against a target dataset's normal
query/gallery protocol. The target labels are used only for retrieval metrics;
the source classifier is not replaced and is not used for matching.

```bash
python scripts/evaluate_cross_domain.py \
  --config configs/baseline_duke_resnet50_triplet.yaml \
  --checkpoint exp/baseline_r50_triplet_market1501/checkpoints/ckpt_best.pth \
  --source-dataset market1501 \
  --target-dataset duke \
  --output-dir exp/cross_market_to_duke
```

Each evaluation writes `cross_dataset.json` with source/target datasets,
architecture, checkpoint path, metrics, timestamp, experiment ID, and resolved
configuration. `scripts/report_model_selection.py` ranks complete 4x4 result
matrices by mAP, uses Rank-1 as the tie-breaker, and leaves incomplete
architectures unranked.

## Common And Architecture-Specific Settings

The common protocol owns dataset splits, transforms, PK sampling, evaluation
metrics, query/gallery filtering, seed controls, checkpoint artifacts, and
re-ranking policy. Architectures must expose the locked output contract above.
Backbone construction, input-tokenization requirements, embedding head details,
and architecture-specific optimizer or schedule changes remain explicit model
configuration, rather than being silently inherited by PCB, MGN, or TransReID.

When reranking is enabled, rerank fields are appended to the same payload.

## Smoke Matrix

Before full training, run the one-epoch smoke matrix:

```bash
PYTHONPATH=. python scripts/smoke_test.py --root /path/to/Dataset
```

Dry-run command preview:

```bash
PYTHONPATH=. python scripts/smoke_test.py --dry-run
```

Default smoke overrides:

- `train.epochs=1`
- `train.eval_interval=1`
- `data.train.batch.P=4`
- `data.train.batch.K=2`
- `data.train.batch.batch_size=8`
- `data.test.batch.size=8`
- `data.num_workers=0`
- `model.backbone.pretrained=false`
- `logging.tensorboard=false`

The smoke matrix must build train/test loaders, run one train epoch, run
evaluation, save checkpoints, and save metrics for every supported dataset.

## Future Model Rule

A future model should require only:

1. A new model implementation file
2. A model builder entry
3. A model-specific YAML config inheriting this protocol
4. An optional model-specific loss only if the locked losses are insufficient

Future models must not require changes to:

- dataset loaders
- samplers
- transforms
- generic training loop
- evaluator
- ranking metrics
- checkpointing
- logging
- artifact export
