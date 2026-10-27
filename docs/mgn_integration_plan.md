# MGN implementation contract and integration roadmap

Created: 2026-10-27 (America/Sao_Paulo). Documentation freeze: Phase 0D.

## Authorization, status, and persistent workflow

The user reviewed and accepted the Phase 0A, 0B, and 0C investigations and authorized this document in Phase 0D. The scientific choices and reusable contracts below are frozen project directions. This document is not evidence that the proposed implementation exists or works.

**Current authorization: revise and validate this document only under the user's final Phase 0D scope revision. The accepted technical design and framework contracts are preserved. The active roadmap covers Part 1 offline experiments exclusively and ends with Phase 23 LODO results registration. All implementation/execution Phases 1-23 remain NOT STARTED and NOT AUTHORIZED; Phase 1 requires a subsequent explicit instruction. No code/configuration/test/other-documentation/checkpoint/artifact changes, training, GPU experiments, ONNX export, runtime tests, commit or push are authorized by this revision.**

At every implementation session, Codex must:

1. Read this entire plan, its append-only records, applicable repository guidance, and current Git state.
2. Identify the exact currently authorized phase or named subunit. Prerequisite completion and a listed next step do not grant authorization.
3. Verify preceding evidence was reviewed by the user and research advisor and that the user committed and pushed accepted changes. Record that boundary; do not commit or push on the user's behalf.
4. Implement only the authorized unit. A scope conflict or required scientific deviation must be recorded and resolved before dependent work, rather than silently broadening scope.
5. Run focused tests and relevant regressions, provide the change summary and validation report, append the phase evidence, and stop for review.
6. After user/advisor acceptance, the user commits and pushes the accepted changes. Only then may the next phase be separately authorized.

Mark implementation, validation, review, and commit/push separately. Passing tests is not human acceptance. Prepared commands are not observed launches. Failed, skipped, and unexecuted required checks cannot be marked passed. Optional work can be explicitly waived with a reason; it is never silently treated as qualified.

Completed phase records must never be overwritten. Append corrections with their date, reason, prior record reference, and approving user decision. A revised scientific contract needs an explicit revision record; it must not retrospectively change an experiment's provenance.

**PCB implementation history remains untouched.** Do not rewrite `plan.md`, PCB protocols, result registries, checkpoints, outputs, or historical LODO artifacts as part of MGN work. Shared code changes require regressions that preserve their behavior. Preserve local artifacts; do not delete them to obtain a clean checkout.

## Current development scope and future model selection

The project has two parts: Part 1 is offline ReID model training and evaluation; Part 2 is computational evaluation of selected models. We are currently working exclusively on Part 1. This active MGN roadmap includes within-domain, cross-domain and mandatory LODO experiments, ending with offline LODO evaluation and registration in Phase 23. GPU admission, memory checks and bounded training smokes in Part 1 establish training feasibility; they do not authorize computational benchmarking or robotic-pipeline evaluation.

After Part 1 results are complete and reviewed, the user and research advisor will select the exact architectures, scientifically justified configurations and immutable checkpoints for Part 2. That later stage may compare PyTorch and ONNX Runtime inference, latency, FPS, memory and complete robotic-pipeline behavior on selected hardware. No assumption is made that every trained architecture will be exported or evaluated in the robotic pipeline. Preserve generic model interfaces and strict checkpoint reconstruction needed for offline evaluation and future selection, but do not implement runtime adapters, export mechanisms, ONNX configuration or computational benchmarks during Part 1. This is a future-scope statement only, not a Part 2 implementation roadmap; Part 2 needs a later independently reviewed scope and authorization after model/checkpoint selection.

## Evidence basis and interpretation

- **PAPER:** Wang et al., [arXiv:1804.01438v3](https://arxiv.org/html/1804.01438v3), especially Figure 3, Table 1, and Sections 3.1, 3.2, and 4.1. [v1](https://arxiv.org/html/1804.01438v1) and [v2](https://arxiv.org/html/1804.01438v2) retain classifier bias in Equation 1; v3 explicitly removes it.
- **REFERENCE CODE:** [seathiefwang/MGN-pytorch](https://github.com/seathiefwang/MGN-pytorch/tree/384e82002e40b71157506e23d71e6709fa830757), pinned commit `384e82002e40b71157506e23d71e6709fa830757`. It is a reproduction; official authorship was not established. Relevant files are `model/mgn.py`, `loss/__init__.py`, `loss/triplet.py`, `option.py`, `data/__init__.py`, `data/sampler.py`, `trainer.py`, `utils/utility.py`, and `demo.sh`.
- **CURRENT FRAMEWORK:** Phase 0D read-only inspection at `be628bbb0bcb98ba218a7d383bb637a28167393e`. Existing dirty documentation/results were present before this file. That SHA is an inspection anchor, not the future MGN production freeze.
- **PROJECT ADAPTATION:** Explicit accepted choices that complete unspecified details or adapt the experiment. They are not recovered authors' hyperparameters.
- **UNVERIFIED:** MGN numerical correctness, memory feasibility, performance, and training results until their named gates pass.

This plan governs the specifically authorized generic extensions. `docs/model_plugin_protocol.md` contains a stale statement that multi-head CE is not implemented; current `LossBundle` supports it. Reconcile that generic documentation in its scoped phase, without revising PCB history. Existing baseline/PCB tests and source code, rather than that stale statement, define preserved behavior.

## Frozen scientific and architecture contract

The model is a paper-aligned v3 architecture with project-adapted training. It is not an exact reproduction of either the paper's full experimental procedure or Reference A's default/final experiment.

| Property | Frozen selection | Evidence/qualification |
| --- | --- | --- |
| Architecture variant | `paper_v3_global2048_independent_v1` | Project identifier binding all fixed topology and supervision decisions below; incompatible changes require a new identifier. |
| Shared trunk | ResNet50 stem, `layer1`, `layer2`, `layer3[0]` | Paper split after `res_conv4_1`; reference mapping. |
| Independent branches | Each has `layer3[1:]` and a final three-bottleneck residual stage | Pretrained values initially equal; parameter/storage independence mandatory. |
| Final-stage stride | Global branch retains stride 2; Part-2 and Part-3 use stride 1 in the strided convolution and projection; dilation remains 1 | Paper-aligned; historical torchvision topology. |
| Input/map geometry | `[B,3,384,128]`; shared output `[B,1024,24,8]`; global final map `[B,2048,12,4]`; local final maps `[B,2048,24,8]` | Geometry of selected topology. |
| Pooling | Whole-map max pooling in all three branches; equal full-width horizontal max-pooled stripes in local branches | Paper choice; Reference A's average-pooling default is deliberately not selected. |
| Local partitions | Part-2 top/bottom; Part-3 top/middle/bottom | No adaptive stripe counts or alternative partitions. |
| Reductions | Eight independent bias-free Conv2d(2048,256,1), BatchNorm2d(256), ReLU modules | Paper independence; reference implementation details. BN epsilon `1e-5`, momentum `0.1`, affine parameters and running statistics. |
| Global classifiers | Three independent bias-free Linear(2048,C), on unreduced whole-map pooled globals | v3 supervision location. Reference A's 256-D global CE inputs are not selected. |
| Local classifiers | Five independent bias-free Linear(256,C), on reduced locals | v3 local supervision and bias policy. |
| Metric features | Three reduced global vectors, each `[B,256]`, post-reduction BN/ReLU and without L2 normalization | Paper feature selection; reference normalization behavior. No local or concatenated-descriptor Triplet. |
| Retrieval | Eight reduced vectors, concatenated to `[B,2048]`; no extra BNNeck, dropout, per-part normalization, or classifier-dependent embedding | Paper composition; exact ordering below is reference-derived. |

Retrieval order is global G, global P2, global P3, P2 top, P2 bottom, P3 top, P3 middle, P3 bottom. Metric order is G, P2 global, P3 global. Classifier order is the corresponding three unreduced globals followed by the five reduced locals. These orders are part of the immutable variant.

Reference A details: `model/mgn.py:17-41` split/copies; `43-66` pooling/reductions; `68-83` heads; `89-103` initialization; `105-152` forward/order. These are reference inspection locations, not implemented project MGN symbols.

### Initialization and artifact provenance

Use the official historical artifact, not trained ReID weights or modern V2 initialization:

| Field | Frozen value |
| --- | --- |
| Artifact identifier | `resnet50-19c8e357` |
| Source URL | `https://download.pytorch.org/models/resnet50-19c8e357.pth` |
| Full SHA-256 | `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097` |
| Known size | 102502400 bytes |
| Existing Constantine path | `/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth` |
| Head initialization identifier | `reference_a_384e820_bias_free_v1` |

Phase 0B hashed the existing local file without downloading, deserializing, or initializing a model. Phase 0D does not repeat that qualification. Historical torchvision `v0.2.1/torchvision/models/resnet.py` maps pretrained ResNet50 to this artifact; Reference A leaves torchvision unpinned, so its reported-run bytes remain unverified. Modern `pretrained=True` is not a sufficient provenance specification.

Load and validate the pretrained ResNet50 state on CPU before dividing it. Shared layers and each independent branch receive corresponding pretrained tensors and BN buffers. Local final-stage tensors are copied from pretrained `layer4` despite the changed stride; none of that residual stage is left randomly initialized. Discard only the ImageNet classifier. Legacy missing BN `num_batches_tracked` buffers may be initialized to zero explicitly; other missing, unexpected, wrong-shape, or wrong-dtype tensors fail.

Reductions and ReID classifiers are new. Follow Reference A's initializer rules: Kaiming normal fan-in reduction convolution; reduction BN weight Normal(1,0.02), bias zero, running mean zero/variance one; Kaiming normal fan-out classifier weight, without classifier bias. Initialize one reduction template and copy it independently eight times, as the reference does. This records initializer rules, not bitwise equality with Reference A after changing classifier dimensions/bias.

Before any legacy weight deserialization, verify the full approved hash against the same open file stream. Wrong or missing explicitly selected weights fail; do not substitute another artifact or random weights. Smoke and production admission require an existing verified artifact and prohibit downloads. With pretrained initialization disabled for tests/reconstruction, do not access that artifact at all.

### Loss, optimization, and data recipe

| Setting | Frozen value | Provenance |
| --- | --- | --- |
| CE | Ordinary batch-mean CE per head; arithmetic mean over eight heads; label smoothing 0 | Project aggregation using Reference A default; paper specifies supervision locations, not this complete scalar reduction. |
| Triplet | Positive batch-hard Euclidean hinge, margin 1.2, batch mean per feature; arithmetic mean over three globals | Paper mining/margin/selection, reference executable formulation/reduction. |
| Total | Mean CE + mean Triplet, family coefficients 1 and 1 | Project choice; exact authors' aggregate coefficients were not established. |
| Other losses | Center disabled | No invented single raw/BN metric feature. |
| Optimizer | SGD LR 0.01, momentum 0.9, weight decay 0.0005 | Paper-derived principal settings. |
| Parameter rules | Uniform LR; bias LR factor 1; same weight decay for all trainable parameters; no prefix multipliers; Nesterov false | Explicit project completion of recipe; do not inherit PCB's differential LR. |
| Duration | Exactly 120 completed epochs | Project policy; paper uses 80. Equal epochs are not equal updates or compute. |
| Scheduler | `epoch_multistep`, gamma 0.1, changes after completed epochs 60 and 90; no warmup | Project adaptation preserving the paper's schedule proportions. |
| Batch | Framework PK sampler, P=16, K=4, batch 64 | Paper batch composition; framework epoch exposure differs from Reference A. |
| Train preprocessing | RGB, 384x128 bicubic resize, flip probability 0.5, ToTensor, reference normalization | Resolution/flip paper-derived; interpolation/constants reference-derived. |
| Normalization | Mean `[0.485,0.456,0.406]`; std `[0.229,0.224,0.225]` | Reference A; do not inherit PCB/baseline's different mean. |
| Other augmentation | No padding, random crop, or random erasing | Selected paper-aligned initial recipe; A's longer erasing experiment is not selected. |
| Test preprocessing | Same RGB resize/normalization; no random transform | Project/reference contract. |
| Offline extraction | Mean raw original/width-flipped descriptors, then one global L2 normalization | Paper flip aggregation plus reference normalization order. |
| Retrieval | Euclidean; primary reranking disabled | Common framework/reference retrieval; reranked metrics are a separate optional future result. |

Paper Equation 2 prints a leading negative hinge-sum sign. Use the conventional nonnegative hinge implemented by Reference A; do not reverse the ranking objective. The apparent sign error and unspecified aggregate coefficients must remain disclosed.

Reference A samples only K images per identity per epoch and retains a short final batch. Framework PK sampling consumes multiple chunks and may leave different remaining chunks when identities become exhausted. Keep the established framework sampler unchanged. Record actual batches, presentations, and successful updates; never infer them from 120 epochs or PCB random-loader counts. With AMP, successful updates can be fewer than attempted batches.

### Checkpoint selection and scientific limits

Keep common `scripts/train.py`, `scripts/evaluate.py`, and `scripts/evaluate_cross_domain.py`. Evaluate every 10 epochs; save `ckpt_best.pth` on strictly improving periodic mAP; save `ckpt_last.pth` as the latest state through epoch 120. Preserve selected-best epoch/hash/scores separately from final-epoch state. Final selected-best evaluation uses the same offline flip/normalization policy as checkpoint selection. Do not select source checkpoints using cross-domain target scores.

The configured test query/gallery split supplies periodic selection scores, not an independent validation split. Preserve and disclose this common methodological limitation. Results compare complete training strategies, not a matched architecture-only ablation. Future computation measurements must identify whether flip is enabled and pair accuracy with that same extraction policy.

## Frozen reusable framework contracts

### Output transport and feature-source schema

Existing dictionary identity, tensor identity, gradients, ordered logits, legacy tensor output, and legacy `(embedding, logits)` interpretation remain unchanged. Do not reinterpret extra tuple positions or alias `emb` into fake raw/BN fields. A new optional field extends the dictionary contract:

```text
emb: floating Tensor[B,D]                         # retrieval
logits: None | Tensor[B,C] | nonempty flat sequence[Tensor[B,C]]
feat_raw: None | Tensor[B,F]                     # legacy semantics
feat_bn: None | Tensor[B,F]                      # legacy semantics
metric_features: absent | None | Tensor[B,F_i] | flat sequence[Tensor[B,F_i]]
```

Absent, None, or an empty metric sequence means zero metric features. Every provided feature is a floating rank-2 tensor with B>0 and F_i>0; B must match labels/embedding where available. Require matching batch, dtype, and device across metric features and matching label device at loss consumption. Feature widths may differ. Reject nested/mixed sequences, integer tensors, invalid ranks, zero dimensions, and mismatched batch/dtype/device. Preserve the input dictionary/container and tensors; a helper may expose an ordered tuple view without cloning, detaching, stacking, or concatenating tensors.

| Configuration key | Allowed/default | Frozen meaning |
| --- | --- | --- |
| `loss.triplet.feature_source` | `legacy` (default), `model` | Legacy selects existing raw/BN tensor; model exclusively consumes `metric_features`. No fallback between modes. |
| `loss.triplet.feature_aggregation` | `sum` (default), `mean` | Independent batch-mean Triplet per feature, then selected aggregation. MGN explicitly chooses mean. |
| `model.head.metric_feat` | Existing `raw`/`bn` behavior | Remains the legacy selector; do not add architecture-specific values. |
| `loss.id.head_aggregation` | Existing sum default, optional mean | MGN mean; PCB sum unchanged; no averaging logits. |

Enabled model-source Triplet with zero features fails clearly; disabled Triplet permits zero features. Legacy single-tensor arithmetic and error behavior remain on their original path. Center continues consuming the legacy selected tensor and positive `feat_dim`; it never consumes the new sequence implicitly. MGN declares `feat_dim=None`, so enabled Center is rejected. Its `embedding_dim=2048` must never be substituted for a Center feature width.

For MGN in both train/eval modes: `emb` is raw `[B,2048]`, `logits` is eight `[B,C]` tensors, `metric_features` is three `[B,256]` tensors, and raw/BN fields are None. Embedding and metric features remain attached during training; no-grad belongs to evaluation. Classifiers may still be evaluated in eval mode; no public shape/ordering change is implied by mode.

Keep `loss/id`, `loss/triplet`, `loss/center`, `loss/total` and their meanings. Family diagnostics are unweighted aggregates; total applies each family coefficient once. Legacy mode preserves its existing key set and exact single-tensor arithmetic. Model-source mode additionally emits unweighted `loss/triplet/feature_0`, etc., in stable order. Per-feature diagnostics are not additional total-loss terms. Classification accuracy remains existing single-head accuracy or mean independent-head accuracy, never an ensemble prediction. Avoid changing TensorBoard step numbering or historical logging arithmetic in this work.

### Scheduler cadence and resume schema

`sched.name: epoch_multistep`, `milestones: [60,90]`, `gamma: 0.1` is a distinct opt-in scheduler. No warmup fields apply. Declare `step_interval='epoch'` and state unit `completed_epochs`; cadence is derived from scheduler type, not an independent contradictory user override. Legacy schedulers without a cadence declaration retain batch stepping. Do not change existing `step` raw-iteration semantics or `warmup_multistep` conversion/trace.

Epoch 1 starts at 0.01; all batches of epoch 60 use 0.01; all batches of epoch 61 use 0.001; all batches of epoch 90 use 0.001; epoch 91 begins at 0.0001. Step an epoch scheduler once after successful completion of the entire epoch and before the first last/best checkpoint publication. Failed/incomplete epochs do not advance it. Do not step this scheduler in `train_one_epoch`.

State records a version, cadence/unit, completed epochs, milestones, gamma, and base group rates. Validate state against the requested new scheduler, parameter-group layout/rates, and saved checkpoint epoch. Loading optimizer before scheduler must preserve the next epoch's LR. Resume at saved epoch+1 without another scheduler step. Reject wrong cadence/unit/configuration and counter/LR disagreement before resumed training. Older scheduler states remain supported on their existing paths; do not require new fields in all checkpoints.

Standard checkpoints are epoch-boundary continuation, not bitwise replay: sampler/RNG/AMP scaler state is not fully preserved by current ordinary saving. Do not import LODO recovery guarantees. Historical best-score recovery on last-checkpoint resume is an open engineering issue below; no production resume is qualified until resolved and tested.

### Minimum MGN configuration and strict reconstruction

The following is a documentation schema, not a created preset:

```yaml
model:
  name: mgn
  variant: paper_v3_global2048_independent_v1
  pretrained: true
  weights_path: /home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
  initialization:
    artifact_id: resnet50-19c8e357
    sha256: 19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097
    head_init_id: reference_a_384e820_bias_free_v1
  head:
    classifier: true
loss:
  id: {enabled: true, label_smoothing: 0.0, weight: 1.0, head_aggregation: mean}
  triplet: {enabled: true, margin: 1.2, mining: batch_hard, weight: 1.0, feature_source: model, feature_aggregation: mean}
  center: {enabled: false}
sched:
  name: epoch_multistep
  milestones: [60, 90]
  gamma: 0.1
eval:
  flip_aggregation: mean
  normalize_feat: true
  distance: euclidean
  rerank: {enabled: false}
```

Full presets still require existing experiment/system/repro/logging/data/optim/train sections. The fixed variant owns branching, pooling, dimensions, orders, and bias policy; reject contradictory free overrides. `head.classifier` is a true capability declaration, not an alternative topology switch. Source C comes from the train loader, not a hard-coded YAML class count. The builder declares C and 2048-D embedding in existing reconstruction metadata. Canonicalize the single supported variant and initialization identifiers before saving new configurations; reject unknown identifiers, malformed/mismatched hashes, invalid pretrained/path combinations, and nonpositive source C.

Both `data.train.images.interpolation` and `data.test.images.interpolation` explicitly select `bicubic` in MGN presets. An omitted interpolation field preserves the current transform default for existing configurations. Pass this generic image setting through the common loader/transform builders without changing dataset membership. Unsupported interpolation identifiers fail validation. Test image settings, including interpolation, remain source-owned in cross-domain evaluation.

Keep metadata/configuration as the reconstruction authority and `strict=True` tensor loading. The generic checkpoint format and dictionary output-contract version 1 remain compatible; this is an additive optional output field, not a global version invalidation. MGN requires its self-describing metadata; do not broaden the historical baseline-only raw-state fallback. Uniform DataParallel prefix handling remains unchanged.

`initialize_pretrained=False` must prevent all cache reads, hash reads, path-existence requirements, downloads, and pretrained deserialization. Validate declared identifiers syntactically/semantically but reconstruct exclusively from trained state, even if the original artifact path no longer exists. Do not overwrite source topology, C, preprocessing, or evaluation with a target preset. New initialization provenance is retained in saved cfg; no historical checkpoint or manifest is rewritten.

### Evaluation policy and LODO preservation

`eval.flip_aggregation` accepts `none` (default) or `mean`. Missing and explicit none are semantically equivalent. Existing single-forward calls and default metrics remain unchanged. For mean, require NCHW inputs, flip width, run original and flipped views sequentially, validate matching floating `[B,D]` embeddings, average raw model-provided descriptors, and only then perform the existing evaluator's global L2 normalization. Do not normalize views/parts first or double query/gallery metadata. MGN returns raw embeddings; the generic extractor does not undo normalization performed internally by other models.

`extract_features` receives an optional policy keyword whose omitted value remains none. `evaluate_reid` resolves policy and passes it explicitly; direct legacy extractor callers keep working, including feature-only test fixtures. Source cfg owns policy in cross-domain evaluation; target execution/data overrides cannot replace it. Standalone comparison normalizes only the new default equivalence without mutating saved configurations. Invalid values and actual policy conflicts fail before output creation/model/data execution where the entry point permits.

Keep ranking, same-camera filtering, distances, reranking mathematics/parameters, and core metric keys unchanged. Saved resolved configuration records the effective policy; cross-dataset records already include it. Do not mix policies in one output namespace or relabel a single-forward timing as flip-enabled offline computation.

**LODO v1 remains frozen:** whitelist ResNet50/PCB, existing budgets/successful-update scheduler, terminal selection, no target-based selection, no test-time flip, and existing scientific identities. The new field must be rejected when it requests mean for LODO v1. Its scientific hash currently selects only known retrieval fields (`reid/utils/lodo_protocol.py:357`), so honoring mean without this guard would silently change execution while omitting it from the hash. Add only enforcement of the existing no-flip invariant, not a new retrieval recipe or default field in old scientific identities. Preserve no-flip retrieval records and test old digests. Existing strict loss/model key checks must continue rejecting unsupported MGN fields. No MGN whitelist entry in v1.

## Mandatory offline settings and LODO comparability

The dissertation requires MGN in three separately reported experimental settings: within-domain evaluation (Phases 16-17), cross-domain evaluation (Phase 18), and LODO evaluation (Phases 19-21 and 22-23). MGN's offline experimental study is not complete until all required source runs, within/cross-domain records, four official LODO folds, and four independently evaluated held-out results are accepted and registered. Incomplete or unavailable evidence remains explicitly incomplete; it is not a waiver or a fabricated result.

Architectures may use different scientifically justified optimization recipes. LODO comparisons describe architectures together with their training configurations, not a matched optimization or architecture-only ablation unless independently established. Do not silently force MGN to inherit ResNet50/PCB learning rates, epochs/update budgets, scheduler units, loss coefficients, sampling exposure, precision, or recovery behavior. The within-domain 120-epoch recipe does not silently define the new LODO budget either.

Phase 19 must agree on common held-out dataset definitions, identical intended dataset partitions, Rank-1/5/10, mAP and mINP definitions/units, and an explicit primary LODO retrieval protocol. Resolve training exposure and optimization budgets; scheduler units and actual update counts; ordered combined-source identity mapping; checkpoint selection; flip aggregation and normalization; precision and recovery; and scientific identity/versioning. Record justified differences and their consequences alongside the common comparison fields. Preserve unresolved dataset-provenance limitations instead of claiming unverified partition equivalence.

Primary retrieval agreement includes input/preprocessing ownership, descriptor extraction/flip aggregation, normalization, distance, reranking status, and camera filtering. If the agreed primary policy differs from a historical ResNet50/PCB result, do not relabel that result as matching. Keep it as historical evidence with its original identity and policy. Any needed supplementary reevaluation must be separately authorized, use the frozen checkpoint in a fresh result namespace, and preserve the original record. Report unmatched or unavailable primary-protocol evidence explicitly; do not manufacture comparability by editing v1 identities, checkpoints, metrics, or results.

Phase 23 defines a comparison structure that accommodates ResNet50, PCB, MGN, and future TransReID records while retaining each architecture's protocol/recipe identity. Within-domain, cross-domain, and LODO tables remain separate. TransReID must also undergo all three settings in its future integration roadmap. Generic MGN extensions should remain reusable, but no TransReID design implementation, configuration, qualification, or experiment is authorized here.

## Phase ordering and authorization gates

Each phase below carries the nine required fields. All status entries are NOT STARTED / NOT AUTHORIZED. A phase's evidence includes the common evidence record plus its specifically named receipts. Commit boundaries always mean user/advisor review followed by a user-owned commit and push; Codex stops before that boundary. Optional/future branches do not authorize their execution.

| Phase | Deliverable | Required accepted predecessors |
| --- | --- | --- |
| 1 | Optional metric output validation | Phase 0D roadmap accepted, committed/pushed |
| 2 | Generic multi-feature Triplet and diagnostics | 1 |
| 3 | Exact epoch scheduler and resume state | 2 |
| 4 | MGN backbone and verified initialization | 3 |
| 5 | Max pooling and independent reductions | 4 |
| 6 | Bias-free identity classifiers | 5 |
| 7 | Public MGN forward and descriptor assembly | 6 |
| 8 | Builder/configuration/reconstruction integration | 7 |
| 9 | Preprocessing, flip evaluation, source ownership, LODO guard | 8 |
| 10 | CPU model/loss qualification gate | 1-9 |
| 11 | Synthetic training/checkpoint qualification | 10 |
| 12 | GPU FP32 batch-64 admission | 11 plus explicit GPU allocation |
| 13 | Optional AMP feasibility | 12 attempt reviewed; separately authorized |
| 14 | Real-data FP32 GPU smoke | 12 PASS accepted; 13 not required for FP32 |
| 15 | Production preset/provenance freeze | 14 PASS accepted |
| 16 | Within-domain source training, separate A-D units | 15 accepted; explicit per-source authorization |
| 17 | Independent within-domain evaluation/registration | Corresponding 16 unit accepted |
| 18 | Cross-domain evaluation/registration | Corresponding 16/17 source units accepted |
| 19 | Required versioned MGN LODO scientific decision | 16-18 evidence reviewed; separate authorization |
| 20 | Required MGN LODO protocol extension implementation | 19 accepted, explicitly versioned |
| 21 | Required MGN LODO bounded qualification | 20 accepted; fold-specific qualification required before 22 |
| 22 | Official MGN LODO training, separate A-D folds | 19-21 accepted for the corresponding fold; frozen launch reviewed; separate allocation/authorization |
| 23 | Independent held-out LODO evaluation and registration | Corresponding 22 fold accepted; Phase 19 primary retrieval/comparison contract accepted |

Only Phase 13 is optional AMP feasibility. Phases 19-23 are mandatory for completion of Part 1 offline dissertation experiments. Each official LODO fold requires its corresponding accepted qualification, frozen scientific configuration and explicit execution authorization; each held-out evaluation requires acceptance of the corresponding training fold. Mandatory LODO status never grants execution permission. The active roadmap ends after Phase 23 and contains no ONNX/export/runtime/computational implementation phase. Part 2 is deferred until offline results and exact model/checkpoint selection have been reviewed; it is not a parallel active workstream. If optional AMP fails, FP32 status is unchanged; an AMP-only success cannot authorize the frozen FP32 training recipe or an unqualified LODO precision policy.

### Phase 1 — Optional metric output validation

- **Objective:** Transport zero/one/multiple metric features without altering legacy outputs.
- **Files and modules in scope:** `reid/models/outputs.py`; `tests/test_model_interface.py`; output section of `docs/model_plugin_protocol.md`; this plan's evidence record.
- **Exact implementation contract:** Optional metric field, validation, identity/order/autograd preservation, heterogenous feature widths, and unchanged legacy adapters as frozen above; no feature selection or loss execution here.
- **Required focused tests:** Missing/None/empty/tensor/one/many features; varying widths; identity and gradient connectivity; rejection of malformed rank/type/batch/dtype/device/zero dimensions.
- **Required regression tests:** Existing model-interface transport, baseline forward/interface, PCB model outputs, and plugin contract tests.
- **Acceptance criteria:** New field validates correctly; existing dictionary/container/tensor identities, baseline outputs, and tuple interpretations remain unchanged; all required tests pass.
- **Evidence Codex must provide:** Diff, exact commands/interpreter/counts, validation matrix, preserved behavior examples, and common evidence record; no model/GPU qualification claim.
- **Explicit out-of-scope work:** Losses, MGN implementation, scheduler, presets, checkpoint changes, evaluation, and GPU work.
- **Commit boundary:** One reviewed output-contract commit owned and pushed by the user, then stop. Phase 2 requires separate authorization.

### Phase 2 — Generic multi-feature Triplet

- **Objective:** Support independent metric losses with explicit selection and stable diagnostics.
- **Files and modules in scope:** `reid/losses/build.py`, generic loss checks in `reid/utils/config.py`; `tests/test_loss_interface.py`, `tests/test_multi_head_id_loss.py`, `tests/test_training_diagnostics.py`, `tests/test_model_plugin_contract.py`; generic protocol documentation.
- **Exact implementation contract:** Legacy/model selection, sum/mean Triplet, separate Center selection, unchanged single-tensor arithmetic, preserved CE defaults, and opt-in per-feature diagnostics. No architecture-name conditions in the criterion.
- **Required focused tests:** Independent manual scalar/gradient oracle for one/three/different-width features and both reductions; zero-feature enabled error; combined eight-head CE/three-feature Triplet with nonunit family weights proving one-time weighting; Center isolation.
- **Required regression tests:** Baseline raw/BN loss modes, single-head exact values/gradients, six-head PCB sum/backward, diagnostics, and common train-loop/plugin tests.
- **Acceptance criteria:** Every intended feature receives the independent gradient; no concatenated mining; legacy log keys/defaults preserved; generic documentation reconciles its stale CE statement.
- **Evidence Codex must provide:** Arithmetic and gradient comparisons, logged meanings, exact focused/regression results, change summary, and common record.
- **Explicit out-of-scope work:** New model, epoch scheduler, optimizer recipe changes, GPU execution, and Center multi-feature support.
- **Commit boundary:** One reviewed generic-loss commit, user commit/push, then stop.

### Phase 3 — Exact epoch scheduling

- **Objective:** Decay after actual completed epochs without changing existing scheduler semantics.
- **Files and modules in scope:** `reid/optim/lr_scheduler.py`, `reid/optim/build.py`, `reid/engine/train_loop.py`, `scripts/train.py`, generic scheduler validation; `tests/test_scheduler_trace.py`, `tests/test_train_loop_optim.py`, `tests/test_train_orchestration.py`, `tests/test_optim_build.py`.
- **Exact implementation contract:** New epoch scheduler/cadence/state; step after full epoch before any save; optimizer-before-scheduler restore; validate counter/configuration/group-rate invariants; no sampler edits or global legacy migration.
- **Required focused tests:** Unequal synthetic epoch lengths; every LR around 60/61 and 90/91; resume at 59/60/61 and 89/90/91; no double step; failures/incomplete epochs do not advance; wrong cadence/unit/config/state rejected.
- **Required regression tests:** Full existing PCB LR trace and prefix-rate ratios, legacy raw-iteration `step`, baseline warmup, checkpoint/orchestration state restore, LODO successful-update tests.
- **Acceptance criteria:** Correct rates for all 120 epoch boundaries independent of batches; preserved legacy traces; saved state represents next epoch's rate and restores without extra advancement.
- **Evidence Codex must provide:** Independent expected/observed LR tables, continuous/resumed traces, serialized-state assertions, regression results, and common record.
- **Explicit out-of-scope work:** Altering PK membership/exposure, LODO scheduling, full stochastic replay, MGN architecture, and training.
- **Commit boundary:** One reviewed scheduler/orchestration commit, user commit/push, then stop.

### Phase 4 — Backbone and pretrained initialization

- **Objective:** Build the exact split/branch backbone and checksum-verified initialization.
- **Files and modules in scope:** New architecture-local `reid/models/mgn.py`; proposed `tests/test_mgn_model.py`; existing historical artifact read-only for explicit loader tests. No PCB loader rewrite is implied.
- **Exact implementation contract:** Shared/branch topology, strides/dilation, copied pretrained parameters/BN buffers, legacy counter handling, discarded ImageNet fc only, independent branch storage, full-hash-before-load and offline initialization bypass.
- **Required focused tests:** CPU map geometry; key/shape/dtype mapping; wrong hash rejected before deserialization; missing artifact rejection; no reader/download call when disabled; equal initial values but distinct branch storage; zero legacy BN counters only where absent.
- **Required regression tests:** Baseline and PCB forward/initialization tests, model interface, and existing checkpoint reconstruction tests.
- **Acceptance criteria:** Correct maps at 384x128, no random residual layers after pretrained load, no accidental shared branch parameters, and no constructor network access in offline mode.
- **Evidence Codex must provide:** Layer/weight mapping, artifact hash/size when inspected, parameter/storage checks, CPU test results, and common record.
- **Explicit out-of-scope work:** Pool/head/public model integration, source training, downloading weights, GPU tests, and modifying PCB history.
- **Commit boundary:** One reviewed backbone/initialization commit, user commit/push, then stop.

### Phase 5 — Pooling and reductions

- **Objective:** Implement ordered global/local max features and eight independent reductions.
- **Files and modules in scope:** Architecture-local components in `reid/models/mgn.py`; `tests/test_mgn_model.py`.
- **Exact implementation contract:** Whole-map/stripe max pooling on frozen geometry, partitions 2/3, eight Conv/BN/ReLU reductions with exact order and initializer; no adaptive alternate recipe.
- **Required focused tests:** Controlled spatial maps proving maxima and stripe boundaries; raw/reduced shapes; ordering; independent parameter/BN-buffer storage; template initialization semantics; invalid geometry fails before partial BN mutation.
- **Required regression tests:** PCB stripe/reduction tests, baseline forward/interface, and Phase 4 backbone tests.
- **Acceptance criteria:** Max rather than average results, no cross-stripe mixing, exactly eight independent 256-D reductions, and attached gradients.
- **Evidence Codex must provide:** Controlled expected/output tensors, shape/storage/initialization checks, tests, and common record.
- **Explicit out-of-scope work:** Classifiers, public dispatch, losses, presets, and GPU work.
- **Commit boundary:** One reviewed pooling/reduction commit, user commit/push, then stop.

### Phase 6 — Identity classifiers

- **Objective:** Supervise unreduced globals and reduced locals with correct source dimensions.
- **Files and modules in scope:** Classifier component in `reid/models/mgn.py`; `tests/test_mgn_model.py`.
- **Exact implementation contract:** Three bias-free 2048-to-C heads and five bias-free 256-to-C heads, fixed order, source C supplied explicitly, selected initializer, no dataset lookup.
- **Required focused tests:** C values representative of all intended sources, head inputs/weights/bias absence, controlled routing, independent storage and gradients, invalid C/feature shapes.
- **Required regression tests:** Generic logits/CE tests and PCB classifier/model tests.
- **Acceptance criteria:** Eight `[B,C]` outputs with global CE connected before reduction; no target-derived or hard-coded class count.
- **Evidence Codex must provide:** Head/input table, gradient routing assertions, test results, and common record.
- **Explicit out-of-scope work:** Public MGN assembly, common builder registration, dataset/preset changes, and execution qualification.
- **Commit boundary:** One reviewed classifiers commit, user commit/push, then stop.

### Phase 7 — Public MGN forward

- **Objective:** Assemble the frozen architecture into the generic output interface.
- **Files and modules in scope:** Public model in `reid/models/mgn.py`; `tests/test_mgn_model.py`, generic model/plugin interface tests where needed.
- **Exact implementation contract:** Raw 2048-D emb, three reduced metric features, eight logits, None raw/BN fields, `feat_dim=None`, exact train/eval order, and no added normalization/BNNeck/dropout.
- **Required focused tests:** CPU train/eval shapes, concatenation oracle from captured components, metric/CE routing, descriptor classifier independence, all relevant gradient paths, eval batch one, and invalid input geometry.
- **Required regression tests:** Output validation, generic eight-head CE/multi-feature Triplet, baseline and PCB model/loss tests.
- **Acceptance criteria:** One CPU forward/backward through real components agrees with independent loss arithmetic; retrieval order and width are fixed; no fake legacy metric features.
- **Evidence Codex must provide:** Public output/shape table, component-to-loss/descriptor assertions, numerical results, and common record.
- **Explicit out-of-scope work:** Builder/checkpoint/presets integration, train orchestration, GPU and real data.
- **Commit boundary:** One reviewed model-assembly commit, user commit/push, then stop.

### Phase 8 — Configuration, builder, and reconstruction

- **Objective:** Make MGN constructible and self-describing through existing common entry points.
- **Files and modules in scope:** `reid/models/build.py`, architecture-local validation in `reid/models/mgn.py`, generic checks in `reid/utils/config.py`/`config_schema.py` as needed, `reid/utils/checkpoint.py` only if a demonstrated generic gap requires it; config/reconstruction/plugin tests.
- **Exact implementation contract:** Frozen configuration schema, canonical identifiers, source C, explicit dimensions, existing metadata versions, initialize-pretrained bypass, strict state loading, unchanged baseline fallback and source authority.
- **Required focused tests:** Valid/minimal/contradictory configs; unknown IDs/hash/path combinations; real MGN checkpoint round trip and output equality; absent artifact path and prohibited downloads; metadata/state/global-head shape/bias corruption; source C survives conflicting target cfg.
- **Required regression tests:** Existing baseline legacy and PCB reconstruction, DataParallel prefix handling, configuration schema, common train orchestration and cross-domain entry-point tests.
- **Acceptance criteria:** Standard builder and reconstruction support MGN without network access on trained-state loading; no format migration or classifier-name inference for MGN.
- **Evidence Codex must provide:** Resolved schema/metadata examples, exact tensor/output round trips, negative-case matrix, tests, and common record.
- **Explicit out-of-scope work:** Production dataset presets, independent trainers/evaluators, actual training, GPU, and LODO admission.
- **Commit boundary:** One reviewed integration commit, user commit/push, then stop.

### Phase 9 — Preprocessing, flip evaluation, and source ownership

- **Objective:** Add optional offline aggregation while preserving default and frozen LODO behavior.
- **Files and modules in scope:** `reid/engine/evaluator.py`, `reid/utils/evaluation_config.py`, generic interpolation plumbing in `reid/data/build.py` and `reid/data/transforms.py`, generic policy validation, existing-rule guard in `reid/utils/lodo_protocol.py`; transform, evaluation, cross-domain, config, rerank and LODO tests. Update generic protocol documentation.
- **Exact implementation contract:** Explicit bicubic image interpolation with omitted-field default preserved; none/mean flip policy, sequential width flips, mean before final L2, one metadata row per image, source-owned settings, historical default equivalence, unchanged ranking/reranking, and v1 rejection of mean without identity changes. Preserve frozen LODO image/pixel behavior and reject an interpolation override that would change it rather than omit the change from scientific identity.
- **Required focused tests:** Bicubic pixel oracle and old-default equivalence; one/two forward counts, controlled unequal-norm descriptor oracle, flipping/order, shape mismatch, invalid policy, source/target conflicts, missing-vs-none equivalence, LODO execution-changing override rejection before execution and unchanged historical hashes.
- **Required regression tests:** Existing transforms and source loader tests, evaluation harness/rerank/metrics tests, cross-domain and standalone entry points, LODO CPU/GPU-evaluation contract tests that require no GPU, baseline/PCB default descriptors.
- **Acceptance criteria:** Default outputs/metrics remain unchanged; MGN policy is explicit and recorded in cfg; no policy can change LODO execution under an unchanged identity.
- **Evidence Codex must provide:** Aggregation oracle, default metrics comparison, source ownership and identity checks, tests, and common record.
- **Explicit out-of-scope work:** MGN LODO support, altered metric mathematics, benchmark evaluation, export, and runtime timing.
- **Commit boundary:** One reviewed generic-evaluation commit, user commit/push, then stop.

### Phase 10 — CPU model and loss qualification gate

- **Objective:** Review the integrated CPU contract before training or device qualification.
- **Files and modules in scope:** CPU model/interface/loss/reconstruction/regression tests, proposed MGN CPU qualification tests, this plan and a separate MGN evidence location when authorized.
- **Exact implementation contract:** Exercise real MGN components and independent mathematical oracles with pretrained disabled except explicit artifact-reader checks; no GPU/data-loader benchmark or network access.
- **Required focused tests:** Full output/CE/Triplet shapes, routing/order, parameter independence, finite backward, eight/three means and weights, model mode behavior, offline reconstruction.
- **Required regression tests:** Baseline/PCB model/loss/init/presets, outputs, optimizer groups, schedulers, config, checkpoint, evaluation and LODO CPU contracts relevant to shared changes.
- **Acceptance criteria:** Required CPU suites pass; all skips/gaps explained; no claim of GPU feasibility or reference score reproduction.
- **Evidence Codex must provide:** Consolidated CPU acceptance matrix, commands/environment/counts/tolerances, uncovered limitations, and common record for user/advisor review.
- **Explicit out-of-scope work:** New scientific choices, training loops, GPU/AMP, real-data execution, and official presets.
- **Commit boundary:** User-reviewed CPU qualification receipt/documentation commit; code fixes, if necessary, remain in this gate only when explicitly scoped and retested. User commit/push, then stop.

### Phase 11 — Synthetic training and checkpoint qualification

- **Objective:** Prove common training, scheduler, save/load and selection plumbing together.
- **Files and modules in scope:** Proposed `tests/test_mgn_end_to_end.py`, common orchestration/checkpoint tests and MGN evidence; `scripts/train.py`/checkpoint code only for separately identified generic integration corrections within this authorization.
- **Exact implementation contract:** Small CPU synthetic PK batches through common builders/criterion/train loop; controlled evaluation scores through common selection; strict checkpoint round trips; epoch LR continuation without stochastic-replay claims.
- **Required focused tests:** Finite updates/momentum, complete output/loss diagnostics, varied-length epoch boundaries, continuous/resumed optimizer-LR state, selected-best versus last-state separation, source-class reconstruction, resume selection-threshold retention.
- **Required regression tests:** Existing orchestration, diagnostics, checkpoint/reconstruction, optimizer groups, baseline/PCB end-to-end and LODO update/recovery CPU contracts.
- **Acceptance criteria:** Synthetic training and reconstruction pass; resumed historical-best handling is resolved or production resume explicitly remains disabled pending a scoped decision; no incorrect overwrite of a better selected checkpoint.
- **Evidence Codex must provide:** Update/LR traces, independent expected scores and selected epoch, checkpoint hashes/metadata, resume limitations, exact tests, and common record.
- **Explicit out-of-scope work:** Real source images, GPU, full epochs over benchmarks, qualification of bitwise recovery, and official training launch.
- **Commit boundary:** One reviewed synthetic integration/receipt commit, user commit/push, then stop.

### Phase 12 — GPU FP32 batch-64 admission

- **Objective:** Establish bounded single-GPU feasibility for the unchanged primary recipe.
- **Files and modules in scope:** New isolated MGN GPU smoke harness and CPU harness-contract tests, common builders/criterion/optimizer, proposed `exp/mgn_smoke/<unique-run>/fp32/` receipts. Existing LODO smoke scripts are inspection references, not mutable admission routes.
- **Exact implementation contract:** Fresh allocated-UUID subprocess, cached hash-verified weights, FP32, synthetic `[64,3,384,128]`, P16/K4, source C=1041, three complete updates, fixed ten-minute worker limit, cold and subsequent-step memory measurements including SGD momentum.
- **Required focused tests:** CPU guard/timeout/OOM/receipt tests before launch; GPU eight `[64,1041]` logits, three `[64,256]` metric features, `[64,2048]` emb, finite means/gradients/state, actual optimizer steps and parameter changes, no graph retention.
- **Required regression tests:** Prior accepted CPU model/loss/orchestration contracts and smoke guard tests; no unrelated GPU campaigns.
- **Acceptance criteria:** Three FP32 steps complete with peak allocated/reserved/free/total bytes and runtime controls recorded. OOM/timeout/failure is preserved as failed admission, not repaired by changing batch, resolution, precision, or device.
- **Evidence Codex must provide:** Allocation/UUID and environment audit, commit/config/weight hashes, per-stage status/step counts, first/steady memory peaks, stderr/log and machine-readable outcome, common record.
- **Explicit out-of-scope work:** Real data, automatic AMP retry, smaller batches, accumulated microbatches, official outputs/checkpoints, training, and changing LODO guards.
- **Commit boundary:** User/advisor reviews the actual FP32 receipt; user commits/pushes harness and accepted evidence metadata, not large temporary tensors. Stop regardless of pass/failure.

### Phase 13 — Optional AMP feasibility

- **Objective:** Measure a separate optional precision capability without substituting it for FP32 admission.
- **Files and modules in scope:** MGN smoke harness/contract tests and distinct `amp` receipt namespace; common AMP path.
- **Exact implementation contract:** Separately authorized fresh process, same batch/geometry/class count/initialization and bounded three-update budget; FP16 autocast with FP32 parameters and GradScaler; report actual successful/skipped steps and precision/runtime settings.
- **Required focused tests:** AMP control/receipt tests; GPU shapes, finite unscaled gradients, optimizer/momentum state, scale behavior and peak memory; diagnose distance/mining precision issues without silently changing frozen loss arithmetic.
- **Required regression tests:** FP32 smoke contract and CPU loss/orchestration tests affected by any correction.
- **Acceptance criteria:** Separate PASS/FAIL/UNSUPPORTED result; optional waiver allowed. No speed or memory benefit assumed. AMP success never overwrites FP32 failure or changes primary training precision.
- **Evidence Codex must provide:** AMP-specific controls, successful/skipped counts, finite checks, allocated/reserved peaks, limitations and common record.
- **Explicit out-of-scope work:** BF16 assumptions, production AMP authorization, batch reduction, scientific recipe substitution, and full training.
- **Commit boundary:** Separate reviewed optional-precision evidence commit or recorded waiver; user commit/push, then stop.

### Phase 14 — Real-data GPU smoke

- **Objective:** Qualify source loading/preprocessing and real PK batches after accepted FP32 admission.
- **Files and modules in scope:** MGN smoke harness, previously accepted source loaders/transforms read-only in behavior, proposed source smoke configs/evidence in isolated smoke namespace. Preprocessing implementation must already be accepted under Phase 9; this execution gate does not bundle a transform correction with a GPU run.
- **Exact implementation contract:** Bounded three FP32 updates per separately authorized source; P16/K4, batch 64, 384x128, actual source C, frozen normalization/flip; no target evaluation/selection. Keep dataset membership and partitions unchanged.
- **Required focused tests:** Previously accepted interpolation option/default tests; source labels/PK structure, RGB/pixel-transform checks, finite real-data forward/backward/step and per-source peak memory. CPU tests establish source/target preprocessing ownership.
- **Required regression tests:** Existing data transforms, sampler, four dataset protocols, baseline/PCB preset behavior and accepted MGN smoke contracts.
- **Acceptance criteria:** Every intended training source has accepted real-data evidence, exact metadata/provenance, fixed batch and no silent parser/protocol correction. If any source fails, that source is not admitted.
- **Evidence Codex must provide:** Source metadata hashes, parser/protocol identity including CUHK03 provenance limits, actual class/step/shape logs, transformed input checks, memory peaks, common record.
- **Explicit out-of-scope work:** Benchmark galleries/ranking, full epochs, official output directories, data rewriting, LODO, and deployment.
- **Commit boundary:** One source smoke unit at a time, reviewed evidence and user commit/push; no automatic next source or training.

### Phase 15 — Production configuration and provenance freeze

- **Objective:** Make one concrete reproducible training recipe reviewable before any official run.
- **Files and modules in scope:** Proposed `configs/mgn/{market1501,duke,cuhk03,msmt17}.yaml`, MGN preset tests, new MGN experiment protocol/manifest/environment records and this plan. Never rewrite PCB freeze files.
- **Exact implementation contract:** Full frozen schema/recipe, 120 epochs, exact epoch cadence 60/90, periodic evaluation10, selected-best/last outputs, source datasets/provenance, independent output roots `exp/mgn/<source>`, FP32, initialization and all execution controls resolved. Repro seed42, deterministic=false, benchmark=true follow established experiment conventions unless explicitly revised before admission testing; no later silent change.
- **Required focused tests:** All preset validation, optimizer groups/rates, scheduler complete traces, enabled/disabled features, source class metadata and policy ownership; read-only preflight commands must refuse wrong code/config/weights/device or reused outputs.
- **Required regression tests:** All accepted shared-component suites and baseline/PCB presets; matching real-data smoke flags must be confirmed.
- **Acceptance criteria:** User/advisor accepts exact code SHA, clean-state policy, environment, GPU allocation/UUID, data/preset/weight hashes, output isolation, disk allowance, exact command and resume restrictions. All mandatory CPU/synthetic/FP32/real-data gates are accepted. No launch occurs in this phase.
- **Evidence Codex must provide:** Manifest, exact terminal commands and failure guards, launch-environment exports, checkpoint/result policy, matching admission receipts, common record and explicit review status.
- **Explicit out-of-scope work:** Training, code changes beyond separately authorized preset integration fixes, batch/precision adaptation, and changing historical provenance.
- **Commit boundary:** User commits/pushes accepted freeze. Distinguish implementation SHA from documentation/publication SHA; if code-equivalent revisions differ, document and review the difference before launch. Stop.

### Phase 16 — Within-domain source training

- **Objective:** Train the four selected source models under the accepted freeze.
- **Files and modules in scope:** Common `scripts/train.py`, frozen source preset, isolated `exp/mgn/<source>`, new MGN result registry/receipts. No source-code or configuration edits during a run.
- **Exact implementation contract:** Independent units 16A Market1501, 16B Duke, 16C CUHK03, 16D MSMT17; each needs its own explicit launch authorization and review. Run 120 epochs, periodic source-test mAP selection, fixed offline aggregation, best/last separation; resume only under an accepted compatible policy.
- **Required focused tests:** Source-specific launch guards; monitor actual process/log progression, losses, LR at60/90, finite saved state, periodic scores, selected epoch/hash, final epoch120/hash and selected-best final evaluation.
- **Required regression tests:** Frozen commit's accepted preset/model/loss/scheduler/checkpoint suites; re-run only when justified by drift/corrections, not by altering the run environment.
- **Acceptance criteria:** Authorized source completed with 120 epochs and 12 periodic evaluations, all provenance/metrics consistent, last and selected-best registered independently, failures/deviations explicit. No completion claim from a prepared command alone.
- **Evidence Codex must provide:** Actual launch/terminal environment, GPU and process evidence, final log, resolved cfg, exact update count, checkpoint hashes, final/selected metrics, registry and machine-readable verification receipt.
- **Explicit out-of-scope work:** Starting another source automatically, target-based selection, live code edits, unapproved restart/resume, cross-domain evaluation, and export.
- **Commit boundary:** After each A-D unit the user/advisor reviews the source evidence; user commits/pushes accepted registry/documentation before the next authorized unit. Keep checkpoint binaries/local artifacts preserved outside unnecessary Git additions.

### Phase 17 — Independent within-domain evaluation

- **Objective:** Verify selected source checkpoints through the standalone evaluator.
- **Files and modules in scope:** Common `scripts/evaluate.py`, saved source cfg/checkpoint, fresh MGN diagonal output/registry locations, evidence documentation.
- **Exact implementation contract:** One separately authorized source at a time; exact registered best hash/source C/preprocessing/flip policy, no retraining or reselection, source query/gallery protocol unchanged.
- **Required focused tests:** Output freshness, reconstruction/no-download checks, source policy identity, finite metrics, comparison with training selected-best evaluation under a reviewed tolerance justified before execution.
- **Required regression tests:** Existing standalone reconstruction/evaluation/source-config tests on the frozen evaluation release; no repeated full gallery runs solely as a test substitute.
- **Acceptance criteria:** All four diagonal records separately verified or explicitly incomplete; checkpoint/epoch/hash/config agree; metric tolerances and environment differences disclosed.
- **Evidence Codex must provide:** Actual commands, release/checkpoint/config hashes, resolved evaluation config, scores/logs, diagonal receipt and common record.
- **Explicit out-of-scope work:** New checkpoint selection, target overrides of source semantics, training, reranking the primary results, or overwriting previous cells.
- **Commit boundary:** Each source diagonal review and user-owned registration commit/push is separate; next evaluation requires authorization.

### Phase 18 — Cross-domain evaluation

- **Objective:** Register the 12 off-diagonal source-to-target cells without changing source models.
- **Files and modules in scope:** Common `scripts/evaluate_cross_domain.py`, four registered source best checkpoints/configurations, fresh MGN matrix cell paths and result documentation.
- **Exact implementation contract:** Execute separately authorized source rows; fixed source checkpoint per row, source-owned preprocessing/evaluation/flip policy, target-owned dataset/parser/protocol and permitted runtime overrides only; no target-C reconstruction.
- **Required focused tests:** Source/target identity and output guards, same checkpoint hash across a row, retained C, correct target query/gallery semantics, finite metrics and complete unique pair coverage.
- **Required regression tests:** Accepted source-ownership, reconstruction, ranking, experiment-matrix and default evaluation tests; no scientific protocol changes.
- **Acceptance criteria:** Four diagonals plus12 off-diagonals form 16 unique registered cells; no target used to choose a checkpoint; arithmetic recomputed and recipe/test-split limitations preserved.
- **Evidence Codex must provide:** Commands, release/hash/config provenance, per-row verification receipts, matrix and bounded comparisons with verified baseline/PCB evidence, common record.
- **Explicit out-of-scope work:** Retraining, selecting a different checkpoint per target, fabricated missing comparisons, LODO or runtime deployment.
- **Commit boundary:** Review and user commit/push each accepted source row; no automatic next row or later phase.

### Phase 19 — Versioned MGN LODO decision

- **Objective:** Resolve the required MGN LODO scientific protocol and cross-architecture comparison contract without expanding historical v1 implicitly.
- **Files and modules in scope:** New MGN LODO decision/contract documentation, this roadmap, read-only inspection of existing LODO contracts/evidence. No LODO code/configuration edits yet.
- **Exact implementation contract:** Choose an explicit extension identifier/version and scope; resolve training exposure/optimization budgets, epoch versus successful-update scheduler units and update accounting, ordered combined-source identity mapping, terminal versus source-only checkpoint selection, primary retrieval including flip/normalization, precision/recovery, and training/evaluation scientific identities. Freeze common held-out datasets, partitions and metric definitions across architectures; justify different optimization recipes rather than copying v1 settings. Held-out target images/labels/metrics must not influence training, checkpoint selection, or recipe tuning.
- **Required focused tests:** Documentation validation and independently checked budget/identity examples; enumerate negative cases the extension must reject.
- **Required regression tests:** Read-only baseline comparison of frozen v1 schemas/identities and rules; do not run official LODO jobs as validation.
- **Acceptance criteria:** User/advisor accepts every scientific choice, primary retrieval/comparability rules, four source/target folds and new identity boundary, with concrete scope for Phase 20 and later Phases 22-23. Missing historical comparability evidence is identified with any separately authorized reevaluation needed; no v1 result is rewritten. If the extension is too broad for one unit, append smaller reviewed subunits before implementation.
- **Evidence Codex must provide:** Decision table covering all comparison dimensions, evidence/unknowns, budgets/exposure and scheduler examples, source-map/fold definitions, exact training/evaluation identity contents, primary-policy compatibility matrix against historical records, untouched-v1 comparison and common record.
- **Explicit out-of-scope work:** Whitelist changes, historical hash migration, prototype training, and implicit target selection.
- **Commit boundary:** Separate user-owned scientific-decision commit/push; no implementation authorized by the decision's existence alone.

### Phase 20 — Versioned MGN LODO implementation

- **Objective:** Implement the mandatory, accepted versioned extension's offline contracts and admission rules.
- **Files and modules in scope:** New versioned LODO schema/admission/checkpoint/evaluation integration modules and tests identified by Phase 19; existing generic components only where explicitly required. Historical v1 acceptance paths remain unchanged.
- **Exact implementation contract:** New version dispatch and scientific identities, source C/mapping for each of the four folds, chosen schedule/exposure budget, checkpoint/retrieval policy and Phase 23 comparison record support, no target-based selection; compatibility and recovery claims limited to tests actually performed. Keep the existing v1 whitelist, configuration identities, checkpoints, artifacts and results unchanged.
- **Required focused tests:** New protocol positive/negative cases, identity determinism/sensitivity, source/target leakage rejection, checkpoint/reconstruction and request validation on CPU, explicit MGN rejection under v1.
- **Required regression tests:** Frozen v1 config/scientific hashes, budgets, sampler/recovery/checkpoint/evaluation/device contract suites; baseline/PCB paths unchanged.
- **Acceptance criteria:** Offline extension tests pass and all previously accepted v1 identities remain identical; GPU/official execution still disabled pending its own gate.
- **Evidence Codex must provide:** Versioned schema and dispatch diff, identity vectors, rejection matrix, focused/regression commands/results and common record.
- **Explicit out-of-scope work:** Official MGN LODO training/evaluation, changing v1 rules, or claiming generic device recovery from CPU tests.
- **Commit boundary:** User reviews and commits/pushes the scoped extension only. Any larger operational/recovery work requires appended explicit subunits and authorization.

### Phase 21 — MGN LODO bounded qualification

- **Objective:** Qualify the mandatory versioned extension's accepted data/device/checkpoint boundaries before official fold training.
- **Files and modules in scope:** New isolated versioned qualification harness/receipts and frozen extension modules; allocated GPU only when separately authorized.
- **Exact implementation contract:** Explicit bounded budgets, synthetic targets/no target selection, existing verified weights, new source class dimensions and mapping, chosen scheduler/retrieval policy; recovery qualification only if expressly included in Phase 19's contract.
- **Required focused tests:** For each intended fold, verify source mapping/identity/reconstruction and held-out exclusion; perform bounded real combined-source device admission and optimizer checks at actual combined C and the approved precision/batch, plus synthetic target evaluation plumbing under the primary retrieval policy. Required recovery interruptions/continuation checks apply only under their accepted scope. Do not generalize one fold's admission to untested folds or device/worker/precision settings.
- **Required regression tests:** New protocol CPU suite and frozen v1 tests/identity vectors; protected-artifact checks before/after execution.
- **Acceptance criteria:** Evidence proves precisely the corresponding fold/device/batch/precision/worker/scheduler/checkpoint/retrieval scope and leaves historical v1 artifacts untouched. Each Phase 22 fold requires its own accepted qualification or explicitly reviewed equivalence evidence covering that fold; failed bounds remain failed. No official run follows automatically, and full-study completion requires coverage of all four folds.
- **Evidence Codex must provide:** All data/code/config/runtime hashes, actual bounded steps, selection-leakage and checkpoint evidence, qualification limits, common record.
- **Explicit out-of-scope work:** Official MGN LODO experiments, unqualified folds/devices, new scientific choices, and v1 artifacts.
- **Commit boundary:** User/advisor accepts the qualification evidence and the user commits/pushes the receipts. Stop before official training; each later Phase 22 fold requires its own reviewed frozen launch scope and explicit execution authorization.

### Phase 22 — Official MGN LODO training

The following are four independent execution/review units. A-D label these units, not historical v1 fold numbers; Phase 19 assigns the new protocol's immutable fold identifiers and source order. All nine phase fields below apply separately to every unit, with its own qualification, tests, launch authorization, provenance receipt, acceptance and user-owned commit boundary.

| Unit | Ordered training sources | Held-out target | Required accepted qualification |
| --- | --- | --- | --- |
| 22A | Market1501 + Duke + CUHK03 | MSMT17 | Phase 21 evidence covering this fold and its approved execution settings |
| 22B | Market1501 + Duke + MSMT17 | CUHK03 | Phase 21 evidence covering this fold and its approved execution settings |
| 22C | Market1501 + CUHK03 + MSMT17 | Duke | Phase 21 evidence covering this fold and its approved execution settings |
| 22D | Duke + CUHK03 + MSMT17 | Market1501 | Phase 21 evidence covering this fold and its approved execution settings |

- **Objective:** Train the four required combined-source MGN LODO models under the scientifically approved versioned protocol, one independently authorized fold at a time.
- **Files and modules in scope:** Phase 20's versioned common launch/training machinery, reviewed fold-specific frozen configurations/manifests, approved source datasets and mapping sidecars, new isolated MGN LODO checkpoint/log/receipt locations. Existing v1 code behavior, namespaces and historical records remain untouched; no live code/configuration changes during a run.
- **Exact implementation contract:** Before each launch, the user/advisor accepts the fold's protocol/version, implementation/release SHA, source order/partitions/content hashes, contiguous combined-source identity map and hash, source C, exposure/budget and scheduler units, actual batch/precision/recovery policy, target-free checkpoint rules, environment/device allocation, exact command and fresh output paths. Use a new versioned namespace such as `exp/lodo/<approved-protocol-id>/mgn/<fold>/<run-id>` fixed by Phase 19, never an existing v1 path. Full training starts only after corresponding Phase 21 qualification is accepted and the user explicitly authorizes that named fold. A frozen configuration is required for every fold; a failed or interrupted run is not a completed fold. Resume requires compatible checkpoints and previously qualified recovery semantics, with no silent fresh restart or budget change. Held-out data never enter training, optimization, checkpoint selection, or recipe tuning.
- **Required focused tests:** Apply separately to 22A-D: read-only launch/admission guards for exact code/config/weight/map/data/device identities and fresh outputs; source/target disjointness and global-label validity; approved schedule/budget and checkpoint-policy contract tests; observed progress, finite losses/state and actual epoch/batch/successful-update/exposure accounting; terminal/completion verification against the accepted budget and selection rule. Preserve each fold's actual launch/failure/recovery evidence.
- **Required regression tests:** The frozen extension's mapping, scientific identity, scheduler, checkpoint/recovery, target-leakage and admission suites plus relevant generic MGN and historical v1 regression receipts. Confirm release/configuration equivalence to accepted qualification for that fold; any material drift requires a reviewed correction and requalification before launch, not an in-run change.
- **Acceptance criteria:** Each authorized fold independently meets its approved complete training budget, retained combined-source C/map, frozen target-free checkpoint policy and exact provenance; final log/completion state, selected/terminal and latest checkpoints are verified with hashes and states distinguished. User/advisor reviews the fold and any recovery/deviations before its completion is registered. Overall Phase 22 remains incomplete until all four fold units are accepted; none is auto-launched after another completes.
- **Evidence Codex must provide:** Per-fold authorization and accepted qualification references, implementation/config/environment/GPU/data/weight/map hashes, exact command and observed process/log evidence, actual exposures and successful updates, LR trace/boundaries, checkpoint names/hashes/counters/selection provenance, final completion/failure/recovery logs, machine-readable fold receipt and common evidence record. Do not infer a launch or completion from a prepared command.
- **Explicit out-of-scope work:** Starting another fold automatically, using held-out performance for selection or tuning, changing a frozen recipe/precision/batch/budget, reusing v1 output paths, deleting failed artifacts, independent held-out benchmark evaluation before Phase 23 authorization, modifying historical ResNet50/PCB identities or results, and Part 2 export/runtime/computational implementation or execution.
- **Commit boundary:** Separate 22A, 22B, 22C and 22D boundaries: after each authorized fold, stop for user/research-advisor evidence review; the user commits/pushes accepted manifests, completion registration and evidence metadata while preserving local checkpoint artifacts. Only a subsequent explicit authorization can launch another fold or its Phase 23 evaluation. No full LODO run is authorized by this roadmap revision.

### Phase 23 — LODO evaluation and results registration

Units 23A-D correspond one-to-one to 22A-D and their held-out targets. They are independently authorized after acceptance of the corresponding training fold; completing all Phase 22 folds is not necessary to evaluate one accepted fold, but all four accepted held-out results are required for the completed offline study.

- **Objective:** Independently evaluate and register all four required held-out MGN results, with a comparison structure compatible with existing ResNet50/PCB and future TransReID LODO evidence.
- **Files and modules in scope:** Phase 20's approved versioned evaluator/registry/comparison machinery, frozen source checkpoint/configuration and identity-map manifests, held-out dataset query/gallery definitions, fresh per-fold result directories and new MGN comparison/verification receipts. Historical result files remain read-only.
- **Exact implementation contract:** Each 23A-D authorization names the accepted Phase 22 fold, immutable checkpoint path/hash and target-free selection evidence, model/variant/C, protocol/training/evaluation identity, target/partition/parser/content hashes and Phase 19's primary retrieval policy. Reconstruct strictly without pretrained downloads or target-derived classifier changes. Evaluate only the held-out target under that frozen policy, without selecting a different checkpoint from its scores. Preserve source-owned preprocessing and record original/flip aggregation, normalization, distance, reranking, camera filtering, precision and execution details explicitly. Publish Rank-1, Rank-5, Rank-10, mAP and mINP with declared units and finite/range validation; keep the LODO setting separate from within/cross-domain reports.
- **Required focused tests:** Apply separately to 23A-D: accepted fold/target/partition/checkpoint/map/config identity checks before evaluation, target exclusion from training/selection evidence, strict source-C reconstruction, retrieval-policy equivalence and metric oracles, fresh-output/collision guards, exact observed metrics and provenance-to-receipt agreement. Then verify four unique held-out targets, all four source triples, complete metric coverage, independent summary arithmetic and explicit missing/unverified evidence markers.
- **Required regression tests:** Approved generic retrieval/ranking/reranking, versioned LODO evaluation/registration/comparison and leakage suites; historical v1 scientific identities/result parsing and integrity assertions. Ensure read-only ResNet50/PCB import preserves their original units, metrics, checkpoint hashes, protocol identities and retrieval labels. Future TransReID support is schema compatibility only, not model implementation.
- **Acceptance criteria:** Each fold has an independently observed, verified held-out evaluation and user/advisor-accepted registration tied to its exact frozen checkpoint and primary policy. Overall Phase 23 requires all four accepted MGN results and a comparison table with common held-out partitions/metrics and explicit recipe/budget/exposure/precision/recovery differences. Unavailable historical or future results remain missing, never zero-filled or invented. Primary-protocol mismatches remain labeled non-comparable until separately authorized supplementary evidence exists; never overwrite or relabel historical records to satisfy comparison.
- **Evidence Codex must provide:** Per-fold authorization, accepted training receipt, evaluation release/environment/device and exact command, model/checkpoint/configuration/map/data hashes, parser/partition and selection provenance, full retrieval/metric policy, observed Rank-1/5/10/mAP/mINP, logs, machine-readable result and registration verification receipt. The comparison structure must retain architecture/variant; ordered sources and target; dataset partition/content identities; training recipe/config hash and protocol/version; schedule/budget units and actual exposures/updates; source C/map hash; checkpoint/selection provenance; precision/recovery; primary versus secondary retrieval-policy identity; metric units and availability/review status. These fields accommodate ResNet50, PCB, MGN and future TransReID without fabricating rows or merging experimental settings.
- **Explicit out-of-scope work:** Target-based checkpoint selection or hyperparameter tuning, retraining, changing the agreed primary protocol after seeing scores, overwriting v1 checkpoints/artifacts/results, fabricating unavailable comparisons, executing TransReID or supplementary historical reevaluations without separate authorization and fresh namespaces, and Part 2 adapters/export/ONNX configuration/runtime tests/computational benchmarks.
- **Commit boundary:** Separate 23A-D review/registration boundaries: stop after each authorized held-out result, obtain user/research-advisor acceptance, and let the user commit/push the accepted result metadata/receipt. A final separately reviewed four-fold comparison/coverage receipt closes Phase 23 and the mandatory LODO deliverable. The active Part 1 roadmap ends with this accepted offline LODO results registration. No next evaluation, experiment, export, runtime work or deployment follows automatically; subsequent model/checkpoint selection and Part 2 require their later reviewed scope.

## Qualification execution conventions

These are future requirements, not commands executed in Phase 0D. The known Constantine convention uses `/home/addirakoze/environments/Reid/bin/python`, a persistent tmux session, and single full-UUID visibility before Python starts. Historical TITAN Xp UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` is an identity reference, not a current allocation or availability claim. Reverify host, allocation, processes, device capacity/runtime support and fresh output paths before authorized execution. Do not kill another workload, bypass failed admission, switch GPUs, or fall back to CPU silently.

Reuse the ownership/resource checks demonstrated by `reid/utils/lodo_device.py:audit_gpu` where suitable, not LODO-only authorization/configuration or fixture-budget patches to admit MGN. GPU parent supervision must bound the worker and preserve failure evidence. Catch CUDA OOM distinctly; record failed stage/completed updates and peak allocated/reserved/current/free memory when available; release process-owned tensors and exit the worker. Parent reports timeout/crash if cleanup cannot finish. Never auto-retry with changed scientific inputs.

For FP32 memory, record context baseline, reset peak stats before CUDA model/input placement, and retain lifecycle peaks across forward/backward and first momentum allocation. Synchronize measurement boundaries; do not retain graphs, clone full GPU models for checking, or erase cold peaks before reporting. Instrumentation overhead is part of the recorded smoke. Optional AMP has a separate fresh process and receipt. Full training is blocked until accepted CPU, synthetic, FP32 batch 64, real-data source admission and production freeze gates; optional AMP cannot waive those gates.

Pretrained-free CPU shape/loss tests do not validate pretrained artifact mapping. Synthetic device tests do not validate real source preprocessing. Three successful steps do not establish full-run memory stability, benchmark accuracy, bitwise determinism, or recovery qualification. Preserve these limits in every receipt.

## Open engineering decisions and stop conditions

The scientific choices above are frozen; these remaining items do not authorize silent recipe changes:

1. **Historical best on resume:** current `scripts/train.py:185-205` derives `best_metric` from the loaded checkpoint's scores, which need not be the historical maximum when resuming last. Phase 11 must define/test an approved way to retain the selection threshold, or keep production resume unqualified. Do not falsely certify full best-selection continuity from LR restore alone.
2. **Exact epoch scheduler implementation:** internal class/helper details may be chosen in Phase 3, but cadence/unit/state and boundary acceptance oracles here cannot change. PyTorch constructor/resume off-by-one behavior must be observed in tests, not assumed.
3. **Bicubic pixel qualification:** Phase 9 implements the frozen generic interpolation field before GPU gates. Numerical pixel agreement with the selected Pillow/torchvision versions remains to be observed; no MGN-name branch, baseline/PCB pixel change, or execution-changing LODO override is permitted.
4. **Execution environment and device feasibility:** historical environment/UUID data are not live verification. Primary FP32 batch 64 remains fixed; failure stops admission and requires a separately reviewed decision. No estimate permits reducing P/K or using accumulation.
5. **Future Part 2 selection:** exact models/configurations/checkpoints, hardware and any inference/parity/computational protocol remain deferred until Part 1 offline evidence is complete and reviewed. They introduce no export/runtime implementation or qualification gate into this active roadmap.
6. **Mandatory MGN LODO:** training exposure/budgets, scheduler units/update counts, combined-source identity mapping, checkpoint selection, agreed primary retrieval/flip/normalization, precision/recovery and scientific identity/versioning must be resolved in Phase 19 before Phases 20-21 and 22-23. Recipe differences are allowed with justification and explicit comparison metadata. Do not answer these questions by inheriting all v1 optimization settings or modifying historical v1 behavior/results. These decisions are mandatory Part 1 prerequisites; Part 2 remains deferred until offline results and model/checkpoint selection are reviewed.

## Common evidence record and append-only phase history

Every phase record must include: authorization text/date and scoped unit; prerequisites and user/advisor review evidence; user commit/push boundary; starting branch/HEAD/status; exact changed files and rationale; interpreter and actual commands; focused/regression pass/fail/skip counts and relevant tolerances; artifacts/receipts and hashes; failures/resolution; deviations and approving decision; preserved baseline/PCB/LODO assertions; completion/review state; next proposed phase explicitly marked unauthorized. Never include memory-system citations in repository/PR messages.

Use a new dated subsection for each attempt, correction or review. Updating a status summary must not remove earlier evidence. For gates with no implementation change, the commit boundary is the reviewed receipt/plan update. Large local checkpoints and temporary smoke tensors need not be committed, but their paths/hashes and preservation policy must be recorded. User-owned commit/push evidence is never fabricated.

### Phase 0D — Creation record, 2026-10-27

- Authorization: user's Phase 0D attachment, documentation only; prior Phase 0A/B/C findings explicitly reviewed and accepted.
- Starting HEAD: `be628bbb0bcb98ba218a7d383bb637a28167393e`; existing modified `plan.md`, `docs/preliminary_results_log.md`, untracked PCB registration files and selected matrix were present before this task.
- Authorized deliverable: this new file only. No existing documentation or PCB history edited; no source/configuration/test/experiment-script modification, training/GPU execution, commit or push.
- Validation: documentation structure, schemas, phase dependencies/required fields, Markdown links/fences, whitespace and authorized change scope are checked during creation. Final observed results are reported to the user; no software test pass count is implied.
- Status: documentation created; roadmap awaits user/research-advisor review and user commit/push. Phases1-23 remain NOT STARTED / NOT AUTHORIZED.
- Next proposed unit: Phase 1 optional metric output validation only, after this roadmap's review/commit/push and explicit authorization.

### Phase 0D — Documentation validation record, 2026-10-27

- Structural checks observed all 23 ordered phases and all 9 required fields per phase, balanced Markdown fences, valid parsed YAML example with expected key types, and no trailing whitespace.
- Before/after integrity checks observed identical hashes for 48 existing PCB/history/result files, identical tracked-diff digest, and unchanged HEAD. The only added Git status entry was this roadmap. Existing dirty files were preserved; none were staged, committed, or pushed.
- `git diff --no-index --check /dev/null docs/mgn_integration_plan.md` is the whitespace check; the corresponding `--stat` reports this untracked new document without staging it. Final structural/scope checks are repeated after the documentation validation record is appended.
- No source tests or experiments executed; documentation validation is not software qualification. User/advisor roadmap review and user commit/push remain pending; Phase 1 is not authorized.

### Phase 0D — Final roadmap revision record, 2026-10-27

- Authorization: user's final Phase 0D revision request permits changes only to this file. The user reviewed and accepted the technical architecture, scientific configuration, generic contracts and existing implementation boundaries; Phase 1 remains unauthorized. Earlier creation/validation records above are preserved verbatim as historical records, including their original review/status observations.
- Scientific correction: MGN must contribute within-domain, cross-domain and separately reported LODO experiments to the dissertation. Phases 19-21 are mandatory preparation/qualification, and new Phases 24-25 require four independently authorized official folds and four independently authorized held-out evaluations/registrations. Mandatory deliverables do not constitute execution permission.
- Exact sections changed: `Authorization, status, and persistent workflow` updates current documentation-only authorization; new `Mandatory offline settings and LODO comparability` defines completeness, justified recipe differences, common partitions/metrics/primary retrieval and future TransReID scope; `Phase ordering and authorization gates` makes 19-21 mandatory, adds 24-25 and preserves independent 22/23 prerequisites; Phases 19-21 retain their purposes with comparison/per-fold qualification requirements; Phase 23's commit-boundary wording only clarifies the independent runtime workstream; new Phases 24/25 define fold execution/evaluation contracts and commit boundaries; open engineering decision 6 reflects required LODO. Phase 22's body and Phase 23's scope/prerequisites otherwise remain unchanged. No accepted architecture/loss/within-domain recipe or Phase 1-18 implementation boundary is changed.
- Comparability requirements: Phase 19 must resolve exposure/budgets, scheduler/update accounting, combined-source mapping, checkpoint selection, flip/normalization, precision/recovery and versioned identity. Architecture-specific justified recipes remain permitted. Historical v1 whitelist, identities, checkpoints, artifacts and results are preserved; mismatched historical retrieval policies require explicit labels and separately authorized supplementary evidence, never retrospective rewriting.
- Validation requirements for this revision: all 25 numbered phases must each contain the nine contract fields; dependency checks must cover fold-specific qualification before Phase 24, accepted corresponding training before Phase 25, and no new LODO prerequisite for Phases 22/23. Compare the entire original Phase 0D history, frozen technical sections, Phase 1-18 bodies, Phase 22 body and Phase 23 scope against their pre-revision text. Verify all other existing dirty/PCB/history/result files and tracked changes remain unchanged; only this authorized document may differ. Final observed validation and revision diff/stat are returned in the session report; no software test result is implied.
- Observed documentation validation: 25 ordered phases each contain all nine fields; the dependency graph is acyclic, per-fold qualification/training gates are present, and neither independent workstream gains a dependency on the other. The frozen technical section, Phases 1-18, Phase 22 body, Phase 23 scope/prerequisites and original Phase 0D records compare unchanged as specified above. YAML/fence/whitespace checks pass; hashes of all 48 protected PCB/history/result files, the pre-existing tracked diff, Git status and HEAD remain unchanged. The only content change is this roadmap revision; these checks are repeated after appending this observed record.
- Status: documentation revised for final review; all implementation/execution Phases 1-25 are NOT STARTED / NOT AUTHORIZED. No source/configuration/test/other-documentation/artifact edits, training, GPU experiments, commit or push occurred. Optional AMP remains optional. TransReID's future roadmap must cover all three settings; no TransReID implementation is authorized now.

### Phase 0D — Final offline-scope revision record, 2026-10-27

- Authorization: the user's final Phase 0D scope revision permits modification of this roadmap only. Current development is exclusively Part 1 offline ReID training/evaluation. No implementation, training, GPU experiment, export, runtime test, commit or push is authorized; Phase 1 remains unstarted.
- Scope correction: remove the former active Phase 22 ONNX export/parity and Phase 23 runtime computational evaluation bodies and table rows. Retain only a short future Part 2/model-selection scope statement. Runtime adapters, export mechanisms, ONNX configuration and computational benchmarking are outside Part 1; not every trained architecture is automatically selected for those later activities.
- Numbering correction: the previously numbered official LODO training Phase 24 and held-out registration Phase 25 are now active Phases 22 and 23, including corresponding A-D units and all current dependency/cross-references. Earlier Phase 0D records remain verbatim with their then-current numbering/scope; those historical references do not reauthorize removed computational phases or redefine current LODO phase identities.
- Preserved contracts: Phases 1-18 are unchanged; Phases 19-21 retain their accepted technical/scientific/qualification contracts, with only downstream references renumbered. MGN LODO remains mandatory, alongside separately reported within-domain and cross-domain results. Historical LODO v1 whitelist, identities, partitions, checkpoints, artifacts and ResNet50/PCB results remain unchanged. Architecture-specific justified recipes and generic offline interfaces/reconstruction remain intact; Phase 19 resolves exposure/update budgets, mapping, checkpoint selection, evaluation aggregation/normalization, precision/recovery and scientific identity rather than inheriting another model's recipe.
- Exact sections revised: current authorization; new `Current development scope and future model selection`; active LODO comparability references; phase ordering/dependencies and optionality paragraph; downstream references in Phases 19-21; removal of old computational Phase 22/23 bodies; renumbered official LODO Phases 22/23 and their exclusions/completion boundary; open engineering decisions 5-6. The original creation, validation and prior final-revision records below their historical headings are not edited or erased.
- Validation requirements: active numbering/table must be exactly 1-23, each phase must retain all nine fields, dependencies must be acyclic and end with qualified fold training then corresponding held-out registration, and no computational phase may remain active. Compare frozen technical contracts, Phase 1-18 bodies and reference-renumbered Phase 19-21 bodies; preserve the complete pre-existing Phase 0D history verbatim. Check YAML/fences/whitespace and verify all other pre-existing tracked/dirty/PCB/history/result files, Git state and HEAD are unchanged. Final diff/stat and observed validation are reported in the session; no source-test result is implied.
- Observed documentation checks pass: all 23 active phases have the nine fields, numbering/dependencies and four-fold gates are consistent, and no computational phase remains active. The frozen contracts and Phases 1-18 are unchanged; Phases 19-21 differ only in downstream reference numbering. Original Phase 0D records remain verbatim; all 48 protected files and the pre-existing tracked diff/Git status/HEAD remain unchanged. YAML/fences/whitespace checks pass. These checks are repeated after this record is appended; no source test, model initialization or experiment is executed.
- Status: revised active Part 1 roadmap ends with Phase 23 offline LODO evaluation/results registration. All active implementation/execution Phases 1-23 are NOT STARTED / NOT AUTHORIZED. User/advisor review of this scope revision and user-owned commit/push remain pending. The next proposed implementation unit is Phase 1 optional metric output validation only, after those workflow prerequisites and separate explicit authorization.

## Proposed autonomous execution amendment — 2026-10-27

**Amendment status: PROPOSED / AWAITING EXPLICIT USER APPROVAL.** The user's attached instruction, `MGN — Autonomous Implementation and Bounded Testing`, states: "First reconcile it with the existing roadmap's phase-by-phase authorization rules through an append-only amendment" and "After that amendment is approved" Codex may execute the specified phases autonomously. This amendment is the concrete proposal for that approval. Its preparation authorizes documentation changes only; it does not itself activate implementation, Git commits, GPU testing or official experiments. Silence or elapsed time is not approval. Append the actual approval message/date and execution scope before activation; do not fabricate an approval record.

### Precedence and preserved technical scope

Upon explicit user approval, this amendment supersedes only the conflicting execution-policy statements in `Authorization, status, and persistent workflow`, the phase-ordering authorization paragraph, phase prerequisites requiring a user-owned commit/push between eligible implementation phases, and the corresponding per-phase review/commit boundaries. It also supersedes historical "all phases unauthorized" status summaries only for the ranges actually activated below. It does not rewrite those statements or any completed history. Read this amendment together with the original phase contracts; the latest explicitly approved execution policy governs eligible work.

The accepted architecture, losses, configuration, batch/resolution/precision, dataset partitions, target-exclusion rules, scheduler boundaries, checkpoint semantics, acceptance criteria, and Part 1 scope remain unchanged. Existing source/PCB/LODO artifacts and results remain protected. All nine fields in every phase still apply, except that eligible technical review/commit/progression gates follow this approved policy. A passing technical gate is not a claim of user/advisor scientific acceptance or official training completion.

### Activation and authority matrix

| Phases | Authority after amendment approval | Required boundary |
| --- | --- | --- |
| 1-11 | Autonomous implementation, focused CPU tests, synthetic integration/checkpoint tests and relevant regressions | Continue when mandatory technical acceptance passes, evidence is appended and a local phase commit exists; no new prompt or user push required between these phases. |
| 12-14 | Autonomous bounded GPU/synthetic/real-data smoke testing within the limits below, only on a verifiably allocated approved GPU | Resource availability and matching qualification prerequisites remain mandatory; Phase 13 is optional and must be explicitly recorded as executed or not exercised. No per-source prompt is required for the Phase 14 cases within this approved scope. |
| 15 | Autonomous frozen source configuration, provenance/preflight and experiment-handoff preparation | Finish the concrete handoff, mark human review pending, and stop before any official launch. Required admission gates cannot be skipped. |
| 16 | Official within-domain full training is user-operated | Codex prepares commands and later verifies supplied evidence; it never launches the official training or counts the handoff as a completed run. |
| 17-18 | Conditional evaluation authority only after corresponding user-generated checkpoints/evidence exist and the evaluation policy, source/target cells, resources and execution scope are approved | Verify checkpoint identity and target-free source selection before executing the approved evaluation units. Missing checkpoints or unapproved scope is a stop boundary. |
| 19 | Human scientific decision/approval gate | Codex may prepare the required proposal when eligible, but cannot invent or approve the versioned LODO scientific protocol, budget or primary retrieval policy. |
| 20-21 | Autonomous implementation and bounded qualification after explicit human approval of the Phase 19 protocol and its covered execution scope | Frozen protocol, per-fold qualification and bounded-test limits remain mandatory. Prepare the four-fold official training handoff and stop before Phase 22. |
| 22 | Official full LODO fold training is user-operated | No complete fold launch by Codex. User training evidence must be observed and verified before later evaluation. |
| 23 | Conditional held-out evaluation/registration after corresponding user-produced checkpoints exist and evaluation policy/execution scope are approved | Target data cannot select checkpoints or tune recipes. Register only observed results; missing evidence remains incomplete. |

This amendment does not grant authority for Part 2 adapters, ONNX export/configuration/parity work, runtime tests, computational benchmarking or robotic-pipeline implementation. It does not authorize a push, merge, destructive cleanup, historical-result modification, or another architecture's implementation.

### Autonomous phase loop and local commits

Before activation, only this amendment is prepared; no branch, worktree or commit is created. After approval, use a dedicated implementation branch or isolated worktree. Inspect and record HEAD, branch, tracked/untracked changes, applicable repository instructions and protected output paths. Do not reset, stash, clean, overwrite, move or commit unrelated user changes to obtain isolation. If a safe branch cannot retain those changes, use an authorized isolated worktree; stop if safe isolation cannot be established. The existing dirty PCB documentation/results must remain untouched.

The approved roadmap/amendment may be committed alone as a local policy baseline on the dedicated branch before Phase 1. This replaces the old prerequisite that the user first commit and push the roadmap. Stage only explicitly reviewed in-scope paths; never use broad staging that captures unrelated work. Do not automatically push, merge, amend or rewrite existing commits.

For every eligible phase, Codex must read the full roadmap/amendment and phase contract, verify prerequisites/Git state, implement the scoped changes, write and run meaningful focused tests, run applicable regressions, diagnose/correct ordinary failures within scope, and verify every mandatory acceptance criterion. Append observed evidence and limitations, create a dedicated local commit for that phase, then automatically continue to the next eligible preauthorized phase. A successful phase does not require another user prompt, advisor review or user commit/push merely to continue within the activated range. Send progress updates and keep the technical results available for later user/advisor review.

Do not mark a mandatory gate passed when its checks fail, are skipped without an applicable approved waiver, or were not executed. Preserve distinctions between implementation done, tests passed, technical phase completion, human review pending, official training not run, and unavailable evaluation evidence. A failed prerequisite prevents dependent phase completion; optional Phase 13 may remain unexercised with its status/reason recorded and does not block the frozen FP32 path.

Each phase record must include phase status, authorization/approval reference, starting Git state, changed files, exact commands/interpreter/environment, focused and regression pass/fail/skip results, relevant numerical checks, GPU receipts where applicable, failures/corrections, remaining limits, and its actual local commit. Preserve all original records. To avoid a self-referential commit hash, include the pre-commit evidence in the phase commit, then append that observed commit SHA in a new receipt entry; carry the receipt into the next scoped commit or a final documentation-only receipt commit. Do not claim an uncreated hash or rewrite history to insert one. No human scientific approval is inferred from a local commit.

### Bounded testing and resource policy

CPU unit/regression/synthetic/checkpoint tests may execute autonomously after activation, using the minimal operations needed for their contracts. They must not execute complete benchmark-training epochs or become a prolonged training campaign. GPU tests are qualification smokes, not official runs.

- Every smoke attempt, including its child workers collectively, has a maximum of **three optimizer updates**; each worker has a maximum **ten-minute** wall-clock lifetime enforced by supervision. Record attempted/completed/skipped steps and failure stage. Do not split a test into extra workers to evade the update bound.
- Use only a verifiably allocated, explicitly approved GPU identity. Check current processes, ownership/allocation and available memory before CUDA execution and recheck during the bounded case. An idle snapshot or the historical TITAN UUID is not a new allocation. Do not choose a different GPU, use an occupied/unavailable device, kill another workload, or silently fall back to CPU.
- Use a fresh isolated subprocess and output directory for each attempt, with immutable case identity: phase, source/fold, GPU, precision and frozen scientific inputs. Run device cases sequentially. Keep FP32 and optional AMP cases and receipts separate; AMP success never substitutes for FP32 admission.
- Record actual GPU identity/allocation evidence, model/config/weight/data/map identities where applicable, runtime controls, peak allocated/reserved memory, loss values, gradient/finite-state checks, actual optimizer steps, timing/timeout and final status. Require existing verified pretrained artifacts; no silent dependency or weight substitutions.
- Preserve original failure, OOM, timeout, interruption and recovery evidence. Do not change frozen batch, P/K, image size, precision, optimizer, loss, schedule or other scientific recipe to obtain a pass. No microbatch/accumulation workaround is authorized.
- Corrective retries are finite: at most one initial attempt plus two corrected attempts per declared case under this policy. Each retry must identify the implementation defect and correction, pass the relevant CPU/contract checks first, reverify resource safety, and link to preserved earlier evidence. Do not mint new case identifiers to reset this limit or rerun a true resource-capacity failure without a material within-contract correction. Stop sooner if a listed blocker applies; additional attempts require an explicit policy extension.
- The initial covered case matrix is Phase 12 synthetic FP32, optional Phase 13 synthetic AMP, and Phase 14 one real-data FP32 case per intended source. Phase 21's concrete fold/qualification case matrix is bounded and approved with the scientific protocol before execution; it cannot silently expand these update/time limits or authorize official training.

If resource constraints, missing dependencies or unavailable source data prevent a required smoke, report the block and preserve the passing work already completed. Do not mark Phase 12/14/15 complete, waive required admission, or continue into official training. Existing allocation authorization can be reused while still valid; it does not require repeated permission requests merely because the next bounded case begins.

### User-operated training handoffs

At Phase 15 provide the frozen source configurations, reproducibility/environment manifest, code/configuration/data/weight hashes, exact preflight and launch commands, GPU allocation requirements, isolated output paths, monitoring/failure-detection instructions, checkpoint selection/evaluation policy, and allowed resume conditions. Distinguish the tested implementation commit from the preparation/manifest commit. Do not execute those official commands. Stop at the handoff for the user to launch and supervise training.

After approved Phase 21 qualification, prepare the equivalent four-fold LODO handoff under the approved versioned protocol, including combined-source identity maps and hashes, source/target exclusion evidence, per-fold budget/scheduler/checkpoint/recovery/retrieval rules and isolated paths. Stop before complete-fold execution. Full training remains user-operated regardless of an earlier smoke PASS, idle GPU, available launch script or prepared manifest.

After the user supplies completed checkpoints and evidence, verify actual completion/provenance and proceed only with the corresponding approved evaluation scope in Phases 17-18 or 23. Held-out targets cannot influence LODO training, checkpoint choice or recipe tuning. Do not fabricate missing results, import another architecture's optimization settings, overwrite historical results or count official training as complete without evidence.

### Stop conditions, completion and amendment record

Resolve ordinary implementation/test failures automatically within the accepted scope. Do not stop merely because the next eligible phase lacks a new prompt or a user push. Stop for a scientific decision, a required specification change, unresolved mandatory validation failure, unsafe/unavailable resources, missing required dependencies/data/checkpoints, the official full-training boundary, unapproved evaluation scope, or a prohibited/destructive next action. Do not use a full training launch to investigate a failing smoke. Report the concrete blocker, completed evidence/commits and prepared handoff; no unlimited retry loop is permitted.

The authorized implementation objective is complete only when the activated implementation/integration/validation/preparation phases have actually passed. A prepared Phase 15 handoff is not completion of Phases 16-18; a Phase 21 handoff is not completion of Phases 22-23; the mandatory offline study remains incomplete until its user-operated training and approved evaluations are verified. Human scientific approvals, missing checkpoints and future Part 2 work cannot be silently counted as complete.

- Amendment authorization: user attachment `MGN — Autonomous Implementation and Bounded Testing`, received 2026-10-27; preparation is documentation only and execution is conditional on subsequent explicit approval.
- Reconciliation: once approved, local per-phase commits and automatic progression replace the former per-phase user/advisor review/user commit/push gate only for the activated eligible ranges. Scientific decisions, user-operated official training, resource safety and later evaluation approval boundaries remain explicit. All accepted technical phase contracts and original history above are preserved.
- Approval record: **PENDING — no execution activation recorded.** Append the user's actual approval and scope in a new dated record before implementation. This amendment may be reviewed as a concrete policy without any code/test/configuration/experiment changes already having occurred.
- Validation requirement: verify the entire pre-amendment document is an unchanged prefix, all 23 phase contracts retain the nine fields, the authority matrix preserves scientific/training/evaluation gates and the finite smoke bounds, and only this roadmap changes. Report observed documentation checks separately from software tests.
- Observed documentation validation: the complete pre-amendment document matches its original byte length/SHA-256 as an unchanged prefix; all 23 phases retain the nine fields; the pending-approval authority matrix and finite update/time/retry limits are present; YAML/fence/whitespace checks pass. All 48 protected PCB/history/result files, prior tracked diff, HEAD, branch and Git status remain unchanged. These checks are repeated after this receipt is added; no software tests or experiments have run.
- Current status: amendment prepared for review; no implementation branch/worktree, local commit, source/config/test modification, model initialization, training, GPU smoke, export or runtime execution has been started by this amendment session.

## Autonomous execution activation — 2026-10-27

The user explicitly approved the proposed autonomous execution amendment dated 2026-10-27 and authorized activation. Phases 1–11 may proceed consecutively, followed by eligible bounded qualification and preparation in Phases 12–15; Phase 13 remains optional. Successfully validated phases require no further approval. Separate local commits are authorized. Existing work and protected artifacts must be preserved.

GPU execution requires verified access to an approved allocated device; historical identity or an idle GPU is insufficient. Without allocation, stop at the GPU boundary. Full within-domain/LODO training, ONNX/runtime experiments, push and merge remain prohibited. Official training is user-operated. The original pending-approval record above is preserved as history and superseded by this activation.

Activation baseline: main at be628bbb0bcb98ba218a7d383bb637a28167393e. Pre-existing PCB/history/results and unrelated working changes are excluded from implementation commits. No GPU allocation has been established.

### Phase 1 implementation receipt — 2026-10-27

Implemented optional `metric_features` validation and tuple-view helper, preserving dictionary/container/tensor identity, order, heterogeneous widths and autograd. Legacy transport and outputs remain unchanged. Added malformed-input and gradient tests and documented the optional field.

Validation: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider -q tests/test_model_interface.py tests/test_model_forward.py tests/test_pcb_model.py tests/test_model_plugin_contract.py`: **143 passed, 1 skipped**, 100.90 seconds. Skip: existing CUDA-only baseline test, CUDA unavailable; no GPU execution. Focused cases cover missing/None/empty/tensor/list/tuple, mixed widths and gradients, malformed rank/type/batch/dtype/device/zero dimensions. Scoped `git diff --check` passed. CPU output-contract acceptance passed; no model-training or GPU-feasibility claim. Baseline activation commit: `6fa61fc`.

### Phase 2 implementation receipt — 2026-10-27

Phase 1 commit: `5c80938`. Generic Triplet now selects legacy raw/BN (unchanged default arithmetic) or explicit model features, independently evaluates features, and aggregates sum/default or mean. No fallback/concatenation; Center remains isolated on its legacy selection. CE behavior is preserved; model-source feature diagnostics are unweighted and family weights apply once. Generic configuration checks and stale protocol documentation are reconciled.

Validation uses the Phase 1 interpreter/environment/pytest flags. `tests/test_loss_interface.py tests/test_reid_loss_modes.py tests/test_multi_head_id_loss.py tests/test_training_diagnostics.py tests/test_model_plugin_contract.py tests/test_train_loop_optim.py`: **137 passed, 3 skipped**, 50.05 seconds (existing unavailable-CUDA cases). Additional newly added device/config checks: `tests/test_loss_interface.py -k 'labels_device or config_rejects_invalid_triplet'`: **3 passed, 19 deselected**, 2.77 seconds. Manual per-anchor Euclidean mining/hinge and independent CE/autograd oracle validates one/three heterogeneous features, both aggregations, eight heads and nonunit weights at 2e-6 tolerances. Zero-feature errors, no fallback, Center isolation, device/batch mismatch, defaults and legacy baseline/PCB gradients pass. Scoped whitespace check passes. CPU loss acceptance passed; no GPU execution.

### Phase 3 implementation receipt — 2026-10-27

Phase 2 commit: `e5305ec`. Added opt-in `epoch_multistep`, completed-epoch unit/cadence, strict version/configuration/group-layout/base-rate/counter/optimizer-LR restore checks. Standard batch stepping skips epoch schedulers; standard orchestration advances once after successful epoch completion before first checkpoint save and checks saved epoch on resume. Legacy batch and LODO successful-update paths are preserved; sampler unchanged. All 120 boundaries are checked using 300 synthetic optimizer calls (no benchmark training); boundaries 59/60/61 and 89/90/91 resume without double steps. Failed synthetic epochs do not advance.

Initial sandbox regression was interrupted at an existing two-worker DataLoader test after stalled workers; original log `/tmp/mgn-phase3-tests.log` retained, not accepted as a pass. Only the verified test process and its two workers were stopped. A bounded host CPU retry with CUDA hidden passed: `timeout 300s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider -q tests/test_scheduler_trace.py tests/test_train_loop_optim.py tests/test_train_orchestration.py tests/test_optim_build.py tests/test_optimizer_param_groups.py tests/test_lodo_updates.py tests/test_lodo_checkpoint.py`: **239 passed, 3 skipped**, 100.29 seconds. Skips are existing CUDA cases. Final stricter schema-version rejection and epoch-focused suite: same CPU flags, `timeout 60s`, `tests/test_scheduler_trace.py -k epoch`: **33 passed, 19 deselected**, 5.59 seconds. Scoped whitespace check and all 48 protected-file hashes pass. CPU scheduler acceptance passed; no GPU execution or stochastic-replay qualification.

### Phase 4 implementation receipt — 2026-10-27

Phase 3 commit: `2edbe88`. Architecture-local backbone now shares stem/layer1/layer2/layer3[0], then independently copies five layer3 and three layer4 bottlenecks into each branch. Only local layer4 strides change; dilation stays one. Tests verify shared `[1,1024,24,8]`, global `[1,2048,12,4]`, local `[1,2048,24,8]` maps, equal initial tensors with independent storage, and exhaustive pretrained key/buffer mapping. ImageNet fc is validated then discarded; only absent legacy counters are zero-filled. Missing/wrong artifacts and malformed states fail without substitution or download; disabled initialization never reads artifacts.

Validation: `timeout 180s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider -q tests/test_mgn_model.py tests/test_model_forward.py tests/test_pcb_model.py tests/test_checkpoint_reconstruction.py`: **141 passed, 1 skipped**, 128.95 seconds; skip is unavailable-CUDA baseline test. Existing local historical artifact observed at 102502400 bytes, SHA-256 `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`; approved legacy reader succeeds only after verifying that hash. CPU backbone acceptance passed. No public model/head integration or GPU execution yet.

### Phase 5 implementation receipt — 2026-10-27

Phase 4 commit: `d3a08f7`. Added fixed-geometry whole-map/stripe max pooling in descriptor order and eight independent Conv/BN/ReLU reductions copied from one initialized template. Complete shape/batch/dtype/device validation precedes BN updates; single-item training is rejected before buffer mutation. Controlled maps verify global/stripe maxima and boundaries; manual matrix/BN/ReLU oracle verifies reduction ordering, values and attached gradients; parameter and BN-buffer storage is independent with equal initial template values.

Validation uses Phase 4 CPU flags: `timeout 150s ... pytest -p no:cacheprovider -q tests/test_mgn_model.py tests/test_pcb_model.py tests/test_model_interface.py -k 'mgn or stripe or reduction or resnet50_baseline'`: **61 passed, 89 deselected**, 45.33 seconds. Pooling was then refined to standard `max_pool2d` tie-gradient behavior, with an explicit first-maximum gradient oracle; focused final `timeout 60s ... tests/test_mgn_model.py -k 'pool or reduction'`: **11 passed, 10 deselected**, 5.40 seconds. Scoped whitespace checks passed. CPU pooling/reduction acceptance passed; no classifier/public-builder/GPU work in this phase.

### Phase 6 implementation receipt — 2026-10-27

Phase 5 commit: `7dbc5a2`. Added eight independent bias-free classifiers, ordered three 2048-to-C globals followed by five 256-to-C locals, explicitly Kaiming fan-out initialized. C is supplied by the caller and validated as a positive integer; no dataset lookup. Tests cover C=751/702/767/1041, dimensions/storage, controlled scalar routing, every head/input gradient, initializer calls and malformed counts/features.

Validation uses Phase 4 CPU flags: `timeout 120s ... pytest -p no:cacheprovider -q tests/test_mgn_model.py tests/test_model_interface.py tests/test_multi_head_id_loss.py tests/test_pcb_model.py -k 'mgn or logits or classifier or multi_head or single_tensor or six_ce or aggregation'`: **144 passed, 85 deselected**, 37.33 seconds. Generic logits/CE, single-tensor arithmetic, six-head aggregation and PCB classifier regressions pass; scoped whitespace checks pass. CPU classifier acceptance passed. Public model, builder and GPU qualification remain later gates.

### Phase 7 implementation receipt — 2026-10-27

Phase 6 commit: `a3d63d4`. Public `MGN` assembles backbone/pools/reductions/classifiers with the same train/eval dictionary: raw 2048-D concatenation, three reduced global metric features, eight logits, None raw/BN legacy fields, `embedding_dim=2048`, `feat_dim=None`. Source C remains explicit. Shape/batch-one training guards run before BN mutation. There is no added dropout or descriptor normalization.

Validation uses Phase 4 CPU flags: `timeout 180s ... pytest -p no:cacheprovider -q tests/test_mgn_model.py tests/test_model_interface.py tests/test_loss_interface.py tests/test_reid_loss_modes.py tests/test_multi_head_id_loss.py`: **180 passed**, 86.03 seconds. Real train/eval/eval-B1 forwards verify descriptor concatenation and feature identity against component hooks. A real B4 CPU backward verifies finite gradients for all parameters, mean-eight CE plus mean-three Triplet arithmetic, global CE bypassing reductions, and Triplet reaching global reductions exclusively. Classifier mutation leaves retrieval unchanged. Generic output/loss and baseline/PCB regression cases pass. CPU assembly acceptance passed; builder and GPU qualification remain unexecuted.

### Phase 8 implementation receipt — 2026-10-27

Phase 7 commit: `c8c8ee1`. Common builder dispatch and architecture-local canonical configuration now bind the single variant, artifact ID/full SHA/head initializer, classifier capability and source C. Contradictory topology/head/provenance/path declarations fail. Known unsupported Center/legacy Triplet configurations fail early. Existing reconstruction/output schema versions remain 1; existing metadata declares model=mgn, frozen variant, source C and embedding_dim=2048. No generic checkpoint-format or legacy-fallback migration was needed.

Validation uses Phase 4 CPU flags. An initial invocation named nonexistent `tests/test_cross_domain_evaluation.py`; collection failed with no tests run (`/tmp/mgn-phase8-tests.log`). Corrected `timeout 180s ... pytest -p no:cacheprovider -q tests/test_mgn_config.py tests/test_checkpoint_reconstruction.py tests/test_model_plugin_contract.py tests/test_config_schema.py tests/test_train_orchestration.py tests/test_cross_domain_config.py`: **142 passed**, 100.95 seconds. Added MGN uniform DataParallel-prefix check: `timeout 60s ... tests/test_checkpoint_reconstruction.py::test_real_mgn_uniform_dataparallel_prefix_roundtrip`: **1 passed**, 12.54 seconds. Real MGN save/load produces exact state/descriptor/logit/metric outputs; conflicting target C cannot overwrite source C. Artifact Path/cache/reader hooks are prohibited during offline reconstruction, even when saved pretrained=true and artifact path is absent. Missing/bias/wrong-head shapes, missing metadata and variant/dimension corruption fail strictly. Baseline/PCB reconstruction, configuration and cross-domain regressions pass. Scoped whitespace checks pass. CPU configuration/reconstruction acceptance passed; no GPU execution or production preset freeze.

### Phase 9 implementation receipt — 2026-10-27

Phase 8 commit: `428237f`. Added generic bilinear/bicubic interpolation selection through both loaders, omitted-field pixel behavior unchanged. Added default-none/opt-in-mean flip extraction with sequential width flips, validated floating matching descriptors, raw averaging before existing global L2, and one metadata row per image. Cross-domain source ownership is preserved; standalone missing/none comparison equivalence is local and nonmutating. Ranking/reranking mathematics and metrics are unchanged. LODO v1 explicitly rejects mean and execution-changing interpolation; whitelist/identity serialization remain unchanged.

Validation: initial collection exposed the existing `lodo_console` script import requirement (no tests executed, `/tmp/mgn-phase9-tests.log`). Corrected bounded host CPU invocation: `timeout 240s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider -q tests/test_data_transforms.py tests/test_dataset_protocol.py tests/test_evaluation_harness.py tests/test_cross_domain_config.py tests/test_rerank.py tests/test_lodo_config.py tests/test_lodo_evaluation.py tests/test_lodo_gpu_evaluation.py`: **395 passed**, 70.48 seconds. Despite the latter filename, all executed cases are CPU/mocked admission/evaluation contracts; CUDA was hidden and no GPU benchmark ran. PIL bicubic and old-default pixel oracles, raw `[3,0]`/`[0,1]` mean-before-L2 oracle, metadata counts, malformed policy/views, source ownership/default equivalence, existing rerank/metric regressions pass. Pure identity functions loaded read-only from `be628bbb` reproduce all eight ResNet50/PCB fold digests exactly; those frozen fixture digests and equivalent explicit defaults now have regression assertions. Generic protocol documentation reconciled. Scoped whitespace checks pass. CPU preprocessing/evaluation acceptance passed; MGN is not admitted to historical LODO v1.

### Phase 10 CPU qualification receipt — 2026-10-27

Phase 9 commit: `518c5e1`. Consolidated CPU acceptance covers real MGN components/public output, independent loss/gradient oracles, strict initialization/reconstruction, shared framework integration and baseline/PCB/LODO/data/evaluation regressions. The stale generic documentation test asserting unsupported multi-head training was reconciled with the accepted implemented contract. Review found a scheduler restore gap: live/restored optimizer group metadata must match the declared layout too. Added that check before restore/advance and a fail-before-counter-change regression; legacy scheduler classes are unchanged.

Exact 41-file command, environment, source/test/log hashes, protected-file fingerprint and eight historical identity fixtures are recorded in `docs/mgn_cpu_qualification.json`. Bounded host CPU suite with CUDA hidden and `PYTHONPATH=.:scripts:tests`: **1231 passed, 4 skipped**, 526.67 seconds. All skips are existing CUDA-only baseline/loop tests. After the targeted group-layout correction, complete scheduler suite: **53 passed**, 20.88 seconds. The consolidated suite tested base `518c5e1`; the final scheduler suite covers the subsequent new-scheduler guard and preserves the full legacy traces. All 48 protected pre-existing files match their initial hashes. No frozen configuration/history/result changes. Scoped whitespace checks pass.

Technical CPU gate accepted under the activated amendment. This is not advisor scientific review, GPU batch-64 admission, benchmark reproduction, full stochastic recovery, or production preset qualification. The unresolved historical-best continuation issue remains explicitly assigned to Phase 11. No official training, GPU experiment or Part 2 work has executed.

### Current manual GPU authorization — 2026-10-27

While CPU qualification continued, Codex requested a current allocation identity because the Constantine launch environment contained no SLURM/PBS/LSF job or GPU-visibility allocation fields. The user explicitly replied: “no allocation is needed just use, the GPU is for me so Use it”. This is fresh user manual authorization for their GPU for the already authorized bounded MGN work; it supersedes the earlier requirement to obtain a scheduler/job or reservation record. No scheduler allocation is claimed or fabricated. All finite update/time/retry limits, live resource checks, phase dependencies, protected-artifact rules and the prohibition on official training/Part 2 remain in force.

Read-only live inventory identifies physical GPU 0 as NVIDIA TITAN Xp, UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, 12288 MiB total and 12179 MiB free, with no listed compute processes at inspection. This current inventory, rather than a historical identity or idle state alone, binds the user's manual authorization to the selected default GPU 0. GPU execution still requires accepted CPU/harness gates and a fresh uncontended-device audit immediately before and during each bounded case. No CUDA model execution has occurred at this record.

### Phase 11 synthetic integration receipt — 2026-10-27

Phase 10 code/test commit: `4660921`; its JSON receipt was initially omitted by the blanket JSON ignore rule and then explicitly added alone in documentation commit `197105c`. No unrelated work was staged. The dedicated MGN synthetic lifecycle now exercises common builders/loss/optimizer/train loop/checkpoints, C=1041, real 384x128 CPU inputs, and exactly three SGD updates using a small P2/K2 batch solely for CPU integration (the required production/GPU P16/K4 batch64 is unchanged). Fixed seeds are model42/data80. Losses were 17.3281364441, 8.9501132965 and 8.9501132965; complete model/optimizer/scheduler hashes match exactly for controlled continuous/resumed inputs. This is not benchmark performance or stochastic replay.

Resolved historical-best retention generically with optional validated `selection_state` (metric, best score/epoch and version), saved on every new standard selected/last publication. Latest score or absent latest evaluation cannot reset the historical threshold. New qualified MGN presets must set `train.save.require_selection_state_on_resume: true`; qualified continuation requires the preserved selected-best file, matching epoch/score/model/metadata/tensor layout. Legacy checkpoints retain their existing fallback without newly certifying an unknown historical maximum. Loaded payload tensors are released before the standard training loop. Model reconstruction schema/output version 1 and LODO recovery paths remain unchanged.

Validation uses Phase 10 host CPU environment with CUDA hidden and `PYTHONPATH=.:scripts:tests`. Initial focused run: **2 passed**, 86.79 seconds; the fixture was then given explicit initial seed42 and retested in the final regression. Fourteen-file checkpoint/orchestration/diagnostic/baseline/PCB/LODO update and CPU recovery regression: **381 passed, 3 skipped**, 366.40 seconds. Skips are existing CUDA loop cases; CUDA recovery metadata cases are CPU doubles. Additional preserved-best guard/orchestration suite: **31 passed**, 12.55 seconds. Final malformed-selected-payload checks plus targeted history/main suite: **26 passed, 7 deselected**, 5.96 seconds. Injected selection scores demonstrate best epoch1/.8 surviving last epoch2/.3 and resumed epoch3/.7, with correct saved completed-epoch rates [.01,.01,.001]; final selected and last-state metric artifacts stay separate. Exact commands, checkpoint/state/source/test/log hashes and caveats are in `docs/mgn_synthetic_qualification.json`.

All 48 protected files remain unchanged; scoped whitespace checks pass. Technical Phase 11 acceptance passed under the activated amendment. Ordinary epoch-boundary continuation is qualified only for this explicit state/selection contract, with frozen matching inputs and retained best artifact; bitwise stochastic/CUDA recovery is not claimed. No official training or benchmark evaluation has run. Phase 12 now requires its CPU harness guards and fresh audit of the manually authorized live UUID before CUDA execution.

### Phase 12 harness preparation receipt — 2026-10-27

Phase 11 commit: `acc74bf`. Added isolated `scripts/mgn_gpu_smoke.py` and CPU contract tests. The harness binds fresh manual GPU authority to one full visible UUID, frozen FP32 B64/P16/K4/384x128/C1041 inputs, verified cached weights, qualified source hashes and a committed source tree. One fresh worker process group has a hard 600-second supervisor limit and at most three optimizer calls. Immutable attempt namespaces and preserved error/correction/code identities enforce at most three attempts per scientific case; no automatic retry, batch/precision/device fallback or official training route exists. Only the existing pure read-only GPU inventory audit is reused; LODO admission/protocol/recovery code is untouched.

CPU guard/config/loss validation: initial **69 passed**, 6.44 seconds; after direct OOM/memory-query/interruption fault-injection coverage, final **73 passed**, 6.22 seconds. Command/source/test/log hashes are in `docs/mgn_gpu_smoke_contract.json` (status CPU_PASS, explicitly no GPU result). Guards cover authority before resources, UUID visibility, source/receipt binding, fresh/symlink/collision paths, case retry limits/corrections, hard timeout/process-group isolation, OOM stage/count/memory preservation and three-completed-step PASS requirements. The Phase 11 raw-evidence path is also corrected from the moving pytest-current alias to its physical run directory; values/hashes/acceptance are unchanged.

Phase 12 is PREPARED, not admitted. Next action under the user's current manual authorization is one initial FP32 attempt on live UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, after another fresh audit. No CUDA model execution has yet occurred. A true OOM/resource failure stops downstream Phases 14–15 without scientific recipe changes.

### Phase 12 observed FP32 admission — 2026-10-27

Tested harness/implementation commit: `6874aae`; one initial attempt, no retries. Under the user's recorded current manual authorization, a fresh single-UUID worker completed exactly three FP32 B64/P16/K4/384x128/C1041 updates on TITAN Xp `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`. Cached ImageNet bytes and qualified source/harness receipts matched before launch. Fresh pre/during/post NVML audits passed; the GPU was released with no compute processes afterward. No other GPU was initialized or used.

Result PASS: eight `[64,1041]` logits, three `[64,256]` metric tensors and raw `[64,2048]` descriptors; finite losses/gradients/parameters/momentum and observed parameter changes on all three optimizer calls; no skipped steps. First/cold peak allocated **11901345792 bytes**, reserved **12155092992 bytes** (includes model/first-step/autotuning); steady steps peak allocated **10143774720 bytes**, reserved **11106516992 bytes**. CUDA-reported total 12774801408 bytes. Completion elapsed 30.286 seconds, below the 600-second hard bound. Runtime Torch2.7.1+cu126, CUDA12.6, cuDNN90501, driver570.211.01; FP32, cuDNN benchmark=true/deterministic=false, matmul TF32=false, cuDNN TF32=true (Pascal hardware). This is bounded feasibility, not long-run stability, accuracy reproduction or computational benchmarking.

Full configuration, manual authority note, cold/steady memory, per-feature/family losses, step counts, runtime controls, source/weight/CPU-receipt identities and raw-file hashes are recorded in `docs/mgn_gpu_fp32_qualification.json`. Raw evidence remains isolated under `exp/mgn_smoke/phase12_synthetic_fp32_GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c_attempt1/fp32/`. Technical FP32 admission accepted under the activated amendment; Phase 14 source qualification may proceed after this local evidence commit. No official training, target retrieval or Part 2 work ran.

### Phase 13 optional precision disposition — 2026-10-27

AMP is explicitly waived for this execution: the primary frozen recipe is FP32 and has passed its required synthetic gate. No AMP capability, speed/memory advantage or production AMP qualification is claimed. Optional Phase 13 is not counted as executed; primary FP32/source requirements remain unchanged.

### Phase 14 source-harness preparation — 2026-10-27

Phase 12 observed admission commit: `2c502d2`. Extended only the isolated smoke harness for the four approved source cases. Source dataset/parser/split definitions are copied read-only from their established baseline definitions; MGN model/loss/optimizer/scheduler/120-epoch/retrieval settings replace the complete architecture-specific recipe. Execution controls are explicitly fixed before admission: four loader workers, pinned memory, test batch32, seed42, benchmark=true/deterministic=false; train P16/K4/B64 and bicubic384x128/normalization/flip/no-erasing remain frozen. No production preset or official output is created yet.

The common source loader and its original PK sampler are observed through the existing transparent index observer without dataset/collator changes or extra RNG draws. Receipts will distinguish consumed batches from prefetched/unconsumed work, bind actual C, labels/maps/ordered records/metadata and file inventory, and hash consumed image bytes. Full corpus image-byte hashing is explicitly not claimed. CUHK03 retains detected/processed_partition with semantic new/classic verification unverified. Deterministic per-batch pixel oracles permit only the agreed bicubic/normalization/original-or-horizontal-flip behavior. Every real case requires the unchanged accepted synthetic FP32 recipe on the same GPU UUID and its own fresh three-update/600-second attempt namespace.

CPU validation: source/data/sampler/four parser/PCB preset suites **151 passed**, 12.02 seconds; final source-configuration and same-device dependency guard suite **36 passed**. Earlier intermediate focused guard run **35 passed**, 6.23 seconds, is retained in `/tmp/mgn-phase14-final-guard-tests.log`. The updated CPU harness receipt appends the original Phase 12 receipt in its history and binds the current harness and all four source configuration hashes. No common model/loader/transform/loss/optimizer/scheduler code changed after prior acceptance. Next actions are four separate, sequential, current-user-authorized real-data FP32 cases, with fresh resource audits and no target evaluation or official training.

### Phase 14A Market1501 observed qualification — 2026-10-27

Tested source harness commit `b719551`, initial attempt only: PASS, three complete FP32 updates through the real common Market1501 processed/trainval loader, **751 identities / 12936 images**, fixed P16/K4/B64/384x128. Actual eight-head/three-feature shapes, finite loss/gradient/model/momentum, parameter changes, PK labels, exact RGB/bicubic/normalize/optional-flip pixels and consumed-index/map/image provenance passed. Four workers/pinned memory and prefetched-versus-consumed counts are recorded; only 192 images reached the model. The complete frozen case/provenance/memory/runtime summary is the Market1501 entry in `docs/mgn_real_source_qualification.json`; raw logs/receipts remain in its isolated namespace. Post-worker audit released the GPU. No target evaluation, official training or historical artifact modification. Source technical acceptance passed; the other three sources remain pending.

### Phase 14B Duke observed qualification — 2026-10-27

Execution commit `27e41f6` is code-equivalent to tested source harness `b719551` (intervening change is the Market1501 documentation receipt only). Initial Duke attempt PASS: three complete FP32 updates through the real processed/trainval source loader, **702 identities / 16522 images**, unchanged P16/K4/B64/384x128 and all frozen recipe/worker controls. Finite losses/gradients/model/momentum, parameter changes, correct output shapes, actual PK labels and RGB/bicubic/normalize/flip pixel oracle passed. Source map/metadata/inventory/consumed-image identities and cold/steady memory are registered independently in the Duke entry of `docs/mgn_real_source_qualification.json`. Exactly 192 images reached the model; prefetched unused work is separately recorded. The post-case device audit is clear. No target evaluation/official training; other source records and historical results remain unchanged. Duke technical acceptance passed.

### Phase 14C CUHK03 observed qualification — 2026-10-27

Execution commit `35dcbd1` differs from source harness `b719551` only by prior source receipts. Initial CUHK03 attempt PASS: three complete FP32 updates, **767 identities / 7365 images**, detected/processed_partition/trainval, split_id=null. Semantic new/classic provenance remains explicitly unverified; no partition, parser or identity-map correction was made. Frozen batch/geometry/loss/SGD/worker controls, output shapes, finite gradient/model/momentum and parameter updates, real PK labels and bicubic/RGB/normalize/flip pixel oracle passed. Exactly 192 source images reached the model, with unused prefetch separately counted. CUHK03's independent metadata/map/inventory/image hashes, runtime and memory proof are added to `docs/mgn_real_source_qualification.json`; all earlier source records are retained. Post-case GPU audit is clear. No target evaluation/official training. CUHK03 technical acceptance passed; MSMT17 remains pending.
