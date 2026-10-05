# Model Plug-In Protocol

This document defines the shared contract for ReID models such as PCB, MGN,
TransReID and custom architectures. Preserve the established baseline protocol
in `docs/baseline_protocol_v1.md`; architecture-specific recipes may differ.

## Integration boundaries

Implement architecture behavior in `reid/models/<model_name>.py` and dispatch
in `reid/models/build.py`. Experiment configurations may use paths such as
`configs/<model>_<dataset>.yaml` or model subdirectories; paths are not a contract.

Necessary shared capability extensions belong in their generic owners, with
explicit scope and regression tests: model outputs, configuration validation,
checkpoint reconstruction, `scripts/train.py`, and smoke orchestration. Future
loss extensions belong in `reid/losses/`; diagnostic extensions belong in
`reid/engine/train_loop.py`. Do not add architecture-name branches to consumers
or create architecture-specific training/evaluation entry points.

Preserve `reid/data/`, `reid/engine/evaluator.py`, `reid/metrics/`, dataset
membership, ranking policy and artifact formats. Model integration does not
itself authorize changes to these protocols. `scripts/evaluate.py` and common
cross-domain evaluation use source-model reconstruction; target identities must
not determine source classifier dimensions. Any separately authorized generic
extension must be documented and validated; models must not invent fake features
to satisfy an outdated restriction.

## Required model output

Models expose the same dictionary structure in train and eval mode:

```python
{
    "emb": Tensor[B, D],
    "feat_raw": Tensor[B, F] | None,
    "feat_bn": Tensor[B, F] | None,
    "logits": Tensor[B, C] | tuple[Tensor[B, C], ...] | list[Tensor[B, C]] | None,
}
```

- `emb` is the required retrieval tensor. It stays connected to autograd during
  training. Evaluation performs no-grad collection externally.
- `feat_raw` and `feat_bn` are optional model-provided metric features. Preserve
  the baseline's pre/post-BNNeck meanings; do not alias `emb` into these fields
  solely to satisfy a consumer. Unsupported metric losses must fail clearly.
- `logits` can be one tensor, a nonempty ordered flat tensor sequence, or None.
  All heads must be floating 2-D tensors with positive, equal batch and class
  dimensions and the same dtype/device. Head count is generic, not fixed at six.
  Empty/nested/mixed sequences, mismatched dimensions, integer tensors and
  incompatible dtype/device are rejected by `validate_logits` in
  `reid/models/outputs.py`. Callers can provide the expected label batch size.

`ensure_output_dict` preserves dictionary identity and tensor/sequence order,
validates logits against the embedding batch when available, and retains legacy
single-tensor and `(embedding, logits)` transport. It never averages logits or
normalizes embeddings. The evaluator consumes only `outputs["emb"]`.

## Dimension and loss capabilities

- `embedding_dim`: positive retrieval width D.
- `feat_dim`: optional metric-feature width F, distinct from retrieval width.
  It may be None for an ID-only model without metric features.
- Standard `named_parameters()`, `state_dict()` and strict `load_state_dict()`
  behavior inherited from `nn.Module` remain mandatory.

`validate_model_loss_requirements` resolves optional metric width in train/smoke
orchestration. Center loss requires a positive integer `feat_dim` at construction.
Triplet consumes a runtime metric tensor rather than a declared width. Enabled
Triplet/Center consumers reject missing selected `feat_raw`/`feat_bn`; they never
silently disable themselves or fall back to `emb`. Configuration validation
rejects known architecture/loss capability conflicts before execution.

ID classification does not require metric features or `feat_dim`. A config can
omit a baseline-specific `model.head` when its architecture does not use one.
The baseline keeps its single tensor logits, metric dimensions, feature choices,
normalization and numerical behavior.

**Current implementation boundary:** multi-head logits can be represented and
validated, but `LossBundle` still computes ID loss for a single tensor. Multi-head
CE aggregation and multi-head training statistics require subsequent, separately
validated extensions. A full multi-head training/smoke step is not yet supported.
Do not confuse successful construction/evaluation/reconstruction with training
integration. The generic training call remains `loss, logs = criterion(outputs, labels)`.

## Builder and configuration

The common `build_model` dispatches on `cfg["model"]["name"]` before reading any
architecture-specific fields. Each branch validates its own config, receives
source `num_classes` explicitly where needed, and returns the public model.
Fixed variants must reject unsupported architectural overrides, rather than
silently creating hybrids. If an omitted variant has one supported default,
validation/builder canonicalize it into `cfg.model.variant` before saving cfg and
metadata. The resolved configuration remains the single source of provenance.

The common experiment sections remain `experiment`, `system`, `repro`, `logging`,
`data`, `model`, `loss`, `optim`, `sched`, `train`, and `eval`. Model-specific fields
need not resemble baseline heads. Architectural recipes must not change dataset
membership, evaluation metrics or checkpoint-selection policy without explicit
project authorization.

New training may opt into the architecture's required pretrained initialization.
`build_model(..., initialize_pretrained=False)` must suppress all such reads and
downloads, including configured local weight paths, for trained-state loading.

## Checkpoint reconstruction

Builders declare the existing versioned `checkpoint_metadata`: `schema_version`,
`output_contract_version`, `model_name`, `variant`, source `num_classes` and
`embedding_dim`. `save_checkpoint(..., cfg=cfg)` persists this as `reconstruction`
with the canonical cfg; no architecture-specific duplicate format.

`reconstruct_model` validates metadata/config consistency, dispatches through the
common builder with initialization disabled, validates the constructed declaration
and strictly loads state. Source class count comes from checkpoint metadata, never
from a target evaluation dataset. Historical baseline checkpoints retain the
bounded `classifier.weight` fallback. New multi-head models must use metadata.

## Evaluation and artifacts

Feature extraction collects `emb`. The common evaluator applies configured global
normalization using `norm + 1e-12`, then the configured distance. Model-side
normalization policy must be documented per architecture; baseline behavior is
preserved. Neither output transport nor checkpoint reconstruction changes it.

Core metrics remain `mAP`, `mINP`, `Rank1`, `Rank5`, `Rank10`. Optional
`rerank_mAP`, `rerank_mINP`, `rerank_Rank1`, `rerank_Rank5`, `rerank_Rank10` remain
separate. Preserve common artifacts:

- `config.resolved.yaml`, `train.log`
- `checkpoints/ckpt_last.pth`, `checkpoints/ckpt_best.pth`
- `metrics/latest_test.json`, `metrics/test_epoch_XXX.json`, `metrics/final_test.json`
- `artifacts/command.txt`, `artifacts/environment.txt`

## Verification checklist

- Builder/model tests prove dimensions, output structure, initialization and
  architecture-specific behavior; malformed generic outputs fail clearly.
- `tests/test_model_plugin_contract.py`, loss interface tests, configuration,
  evaluator and baseline regressions pass.
- Real-model checkpoint save/reconstruction passes strictly, without pretrained
  initialization, preserving source heads and evaluation outputs.
- Metric-capability tests cover ID-only optional features and rejection of
  unsupported enabled metric losses.
- A full training smoke is a later gate once the required objective, diagnostics
  and optimizer capabilities are implemented; component checks do not replace it.
