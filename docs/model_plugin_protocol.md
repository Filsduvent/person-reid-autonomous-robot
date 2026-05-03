# Model Plug-In Protocol

This document defines how future ReID models plug into the frozen offline
framework. It extends `docs/baseline_protocol_v1.md`.

Target future models:

- PCB
- MGN
- TransReID
- other custom ReID models that satisfy the locked output contract

## Allowed Files

Adding a future model may modify only:

- `reid/models/<model_name>.py`
- `reid/models/build.py`
- `configs/<model>_<dataset>.yaml`
- an optional loss file under `reid/losses/` if the model requires a loss that is not covered by the baseline loss bundle

Examples:

- `reid/models/pcb.py`
- `reid/models/mgn.py`
- `reid/models/transreid.py`
- `configs/pcb_market1501.yaml`
- `configs/mgn_duke.yaml`
- `configs/transreid_msmt17.yaml`

## Forbidden Files

Future model integration must not modify:

- `reid/data/`
- `reid/engine/evaluator.py`
- `reid/engine/train_loop.py`
- `reid/metrics/`
- `scripts/train.py`
- `scripts/evaluate.py`

These modules are frozen framework infrastructure. If a model appears to need
changes in these files, the model implementation or config is violating the
plug-in contract and should be adapted instead.

## Required Model Output

Every model must return:

```python
{
    "feat_raw": Tensor,
    "feat_bn": Tensor,
    "emb": Tensor,
    "logits": Tensor | None,
}
```

Field usage:

- `emb`: used by the evaluator and retrieval metrics
- `logits`: used by ID loss
- `feat_raw`: available for Triplet and Center losses
- `feat_bn`: available for Triplet and Center losses

The evaluator depends only on `outputs["emb"]`. Losses depend only on the
locked output keys and `model.head.metric_feat`.

## Required Model Attributes

Every trainable model must expose:

- `feat_dim`: positive integer feature dimension for metric and center losses
- standard `named_parameters()` behavior inherited from `nn.Module`
- standard `state_dict()` and `load_state_dict()` behavior inherited from `nn.Module`

The optimizer builder uses `named_parameters()` and does not need model-specific
branches.

## Builder Integration

`reid/models/build.py` is the only core framework file that should be updated
for a new model. The builder entry should:

1. Read only `cfg["model"]`
2. Validate model-specific config values early
3. Instantiate the model
4. Pass `num_classes` only when the model needs a classifier head
5. Return an `nn.Module` satisfying the locked output contract

The builder must not add dataset-specific, evaluator-specific, checkpoint-
specific, or training-loop-specific behavior.

## Config Requirements

Each model config must inherit the baseline protocol sections:

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

Model configs may change:

- `model.name`
- model-specific fields under `model`
- loss enables/weights if the model requires them
- optimizer/scheduler values when justified
- dataset name/split/format for the target experiment

Model configs must not require changes to dataset loaders, samplers, transforms,
training, evaluation, metrics, checkpoint, or artifact code.

## Loss Integration

Default losses are model-agnostic:

- Triplet uses `feat_raw` or `feat_bn`
- Center uses the same metric feature as Triplet
- ID uses `logits`

If a future model needs an additional loss, add it as an optional loss module and
wire it through the loss builder without teaching the training loop about that
specific model.

The training loop must continue to call:

```python
loss, logs = criterion(outputs, labels)
```

## Evaluation Integration

No evaluation integration should be needed beyond returning `emb`.

Evaluation must continue to report:

- `mAP`
- `mINP`
- `Rank1`
- `Rank5`
- `Rank10`

Optional reranking metrics remain separate:

- `rerank_mAP`
- `rerank_mINP`
- `rerank_Rank1`
- `rerank_Rank5`
- `rerank_Rank10`

## Artifact Compatibility

New models must write the same artifacts:

- `config.resolved.yaml`
- `train.log`
- `checkpoints/ckpt_last.pth`
- `checkpoints/ckpt_best.pth`
- `metrics/latest_test.json`
- `metrics/test_epoch_XXX.json`
- `metrics/final_test.json`
- `artifacts/command.txt`
- `artifacts/environment.txt`

Metric JSON must keep the baseline schema and set `model` to a stable model
label derived from config.

## Verification Checklist

Before a new model is considered integrated:

- Model-specific unit test verifies output keys and tensor shapes
- `tests/test_model_plugin_contract.py` still passes
- Loss interface tests still pass
- Evaluation harness tests still pass
- Smoke matrix or one-dataset smoke run succeeds with the new config
- No forbidden files were modified

## PCB Notes

PCB may produce part features internally, but its public output must still expose
the locked keys. It may concatenate or pool part descriptors into `emb`.

## MGN Notes

MGN may have multiple branches internally, but the public output must still expose
the locked keys. Branch-specific losses should be optional loss modules, not
training-loop changes.

## TransReID Notes

TransReID may use transformer-specific inputs and heads internally, but the
dataset, transform, evaluator, metric, checkpoint, and artifact protocols remain
unchanged. Its public output must still expose `emb` for evaluation and `logits`
for ID loss when classifier training is enabled.
