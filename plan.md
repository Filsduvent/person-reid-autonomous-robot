# PCB implementation and validation roadmap

Created: 2026-10-02. Operational source of truth for PCB integration, validation, training, within-domain evaluation, and cross-domain evaluation.

## Current authorization and session protocol

**Only documentation creation/correction and validation of this document are authorized in the documentation sessions. Phase 4 and all later work are NOT STARTED and NOT AUTHORIZED. No PCB code, configuration presets, tests, or training runs have been created by this task.**

At the beginning of every future PCB session:

1. Read this entire plan and inspect current Git state, applicable repository guidance, and implementation records.
2. Identify the first incomplete step that the user has explicitly authorized. A listed next step is not authorization.
3. Verify prerequisites, including review of preceding steps. If authorization or a scientific decision is missing, report it; do not implement dependent work.
4. Implement ONLY that step, within its listed scope. Do not bundle later phases for convenience.
5. Execute that step's meaningful validation and record commands, environment, results, failures, artifacts, and limitations.
6. Update this plan, report the result for user review, and STOP. Never automatically continue to the next step.

The user is learning and reviewing the model and framework incrementally. This user-requested review cadence overrides ordinary autonomous continuation. No commit, push, or deployment is implied.

Status convention:

| Marker | Meaning |
| --- | --- |
| `[ ]` | NOT STARTED |
| `[~]` | IN PROGRESS |
| `[x]` | IMPLEMENTED + VALIDATED; for Phases 0–3, the completed deliverable is investigation/documentation, not implementation |
| `[!]` | BLOCKED; record concrete blocker and required resolution |
| `[?]` | NEEDS HUMAN REVIEW |

Track completion and review separately: after validation passes, use completion `[x]` and review `[?]`, then STOP. Failed or unexecuted required validation cannot be marked complete. Review acceptance does not authorize the next phase unless the user says to proceed. Every future record must identify the authorization message/scope.

Both supplied roadmap attachments end during Phase 7's test list, at “no parameter sharing”. Phases 0–7 below preserve the supplied sequence. Completion of Phase 7's validation and Phases 8–25 are a documented continuation derived from the frozen contract and requested end-to-end scope, not recovered missing attachment text. Their ordering is a planning proposal pending review, not execution authorization.

## Scientific scope and precedence

Method: established architecture → defensible strong/reference implementation → faithful adaptation to one modular framework → behavior validation → train ourselves → within-domain evaluation → cross-domain evaluation → architecture comparison → model selection → later robotic/edge analysis.

Architecture sequence: ResNet50 + Bag of Tricks qualification completed; PCB current; MGN and TransReID future. Completion of ResNet50 qualification does not mean every historical artifact is available locally or every reported result has been independently reverified.

Use ImageNet backbone initialization as part of the selected recipe. Do not substitute downloaded final PCB ReID weights for our trained experimental models. One selected PCB configuration; no planned Triplet, Center, erasing, PK, shared-reduction, stripe-count, feature-width, or alternative-loss ablation campaign. Synthetic checks, overfit checks, and diagnostic smokes verify implementation and are not scientific ablations.

Keep one configuration system and common `scripts/train.py`, `scripts/evaluate.py`, and `scripts/evaluate_cross_domain.py`. No architecture-specific training/evaluation entry points. Keep common dataset interfaces/partitions, CMC, Rank-1/5/10, mAP, mINP, checkpoint persistence, periodic checkpoint evaluation, best-checkpoint selection, artifacts, and reproducibility recording; architecture recipes may differ.

Proposed preset paths: `configs/pcb/market1501.yaml`, `duke.yaml`, `cuhk03.yaml`, `msmt17.yaml`. `load_config()` in `reid/utils/config.py:112` accepts arbitrary paths; no config inheritance mechanism is implied. Do not move existing ResNet50 presets for symmetry.

The user's **120-epoch project decision** supersedes the preceding Phase 3 report's proposed 60-epoch project run; reference duration remains 60. The subsequent checkpoint-policy correction supersedes the earlier final-epoch-only selection proposal: project training lasts 120 epochs, evaluation occurs every 10 epochs, and the selected model is `ckpt_best.pth` by strictly improving periodic mAP. `ckpt_last.pth` records the latest training state and reaches epoch 120 after successful completion. The epoch-41 LR decay is unchanged. Equal epochs do not imply equal compute, batches, or optimization updates across architectures/datasets.

**Common framework behavior:** datasets and dataset protocol; evaluation and metrics; checkpoint persistence; periodic checkpoint evaluation; best-checkpoint selection; artifact infrastructure; within-domain and cross-domain evaluation.

**Architecture-specific behavior:** architecture, input resolution, heads, embedding construction, loss formulation, optimizer, learning rates, scheduler, sampling, augmentation, and architecture-specific hyperparameters. This distinction also governs future MGN and TransReID integration.

**Known methodological limitation:** the established ResNet50 protocol evaluates the configured test query/gallery split periodically rather than an independent validation split. ResNet50, PCB, and future integrated architectures deliberately use this common checkpoint-selection protocol for experimental consistency unless a later project-wide methodological revision is explicitly approved. Preserve and disclose the test-split selection limitation in dissertation methodology; do not introduce PCB-only validation selection or refactor dataset splitting here.

`docs/model_plugin_protocol.md` currently forbids several generic extensions and mandates positive `feat_dim` and tensor logits. Those older restrictions conflict with the user's explicit verified PCB contract. This plan governs PCB scope; the old document must be reconciled in Phase 10. Do not invent a new approval gate from the stale document, change it during this document-only task, or remove legitimate regression tests just to conceal incompatibility.

## Baseline evidence and limits

Creation-session read-only checks (2026-10-02):

- Repository: `/home/filsduvent/UFPR/person-reid-autonomous-robot`; branch `main`, tracking `origin/main`; HEAD `37ed8a50dedc5aba0bade8765f2190824e217482`; clean before creating this plan.
- Reference: `/home/filsduvent/UFPR/beyond-part-models`; clean `master`, tracking `origin/master`; HEAD `1686e889eb01c28a54b633051418012e15d9c9f3`.
- Interpreter: `/home/filsduvent/environments/Reid/bin/python`, Python 3.12.3, torch 2.7.1+cpu, torchvision 0.22.1+cpu; CUDA unavailable. This machine cannot validate GPU feasibility or authoritative GPU training.
- No applicable `AGENTS.md` found in the repository listing or checked ancestors `/`, `/home`, `/home/filsduvent`, `/home/filsduvent/UFPR`.
- Shell sandbox currently fails with `mountinfo path is not absolute`; approved escalation was needed for read-only shell checks. This is environment state, not a request to bypass future permissions.

Prior-session audit evidence, not rerun during plan creation:

- Focused baseline suite: 51 passed, 1 deselected, 7.49 s; sampler, evaluation harness, experiment matrix, model interface, and config schema coverage. Invocation used the Reid interpreter, `-B`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, pytest `-p no:cacheprovider --assert=plain -s`; the file-writing override test was excluded. The exact deselection command is not preserved here; do not present this as a fresh, reproducible full-suite result.
- Selected local ResNet50 checkpoint strictly loaded and a two-image synthetic forward produced 2048-D descriptors and 1041-way logits. Checkpoint: `exp/msmt17_no_label_smoothing/checkpoints/ckpt_best.pth`, SHA256 `5c751a6a2f3d2b19456684d66c72b3f86ad8d8a7cf7ab061aacd2071720bfc3f`.
- Baseline outputs: `feat_raw` before BNNeck, `feat_bn` after BNNeck, configured retrieval `emb`, single tensor/optional `logits`; preserve numerical behavior. `feat_dim` currently feeds Center-loss width. Evaluator consumes only `emb`.
- Only the selected MSMT17 offline checkpoint was located. Complete other-source checkpoints and the raw 4×4 offline matrix were not available in the audit. Historical mINP in the selected checkpoint is stale; do not mix it with corrected metrics. Dataset roots in historical configs were absent; available replacement roots were not proven partition-equivalent.
- Full tests and GPU training were not performed. Exact resume equivalence is not established: existing checkpointing lacks RNG/sampler/scaler state, and the unrelated Center-loss parameter persistence issue remains. PCB disables Center and AMP; do not claim this fixes general resume reproducibility.

## Frozen reference and implementation contract

Primary reference: [huanghoujing/beyond-part-models](https://github.com/huanghoujing/beyond-part-models), commit `1686e889eb01c28a54b633051418012e15d9c9f3`, independent 1×1 reduction variant. Original paper and [authors' implementation](https://github.com/syfafterzy/PCB_RPP_for_reID) provide provenance only. Previously inspected authors' commit: `e29cf54486427d1423277d4c793e39ac0eeff87c`. Do not import its shared reduction, dropout, pre-reduction 12288-D descriptor, per-part normalization, or optimizer choices into an unnamed hybrid.

External non-reranked README results are Market Rank-1/mAP 92.87/78.54%, Duke 84.47/69.94%, CUHK03 59.14/53.93%. These are not our results or acceptance thresholds. At equal precision, 1536-D descriptors use one-eighth the descriptor storage of 12288-D descriptors; no proportional network-speed claim follows.

### Architecture and initialization

- RGB input `[B,3,384,128]`; ResNet50 bottleneck stages `[3,4,6,3]`; stride on bottleneck 3×3 convolution; last-stage first bottleneck/projection stride 1, dilation 1; feature map `[B,2048,24,8]`. No ImageNet FC/global pooling in the feature backbone.
- Six top-to-bottom, non-overlapping height slices. Require positive spatial extent and feature-map height divisible by six; reject invalid heights explicitly. For standard input, each stripe is `[B,2048,4,8]`; average its whole height and width to `[B,2048,1,1]`. Do not use adaptive pooling to silently accept incompatible heights.
- Six independent `Conv2d(2048,256,1,bias=True)` → `BatchNorm2d(256)` → in-place ReLU modules. Keep the four-dimensional pooled tensor through Conv/BN, then flatten to `[B,256]`.
- Six independent `Linear(256,source_num_classes,bias=True)` classifiers. Weights `Normal(mean=0,std=0.001)`; biases zero. No global branch, shared reduction, dropout, or extra BNNeck.
- Backbone initialization: historical ImageNet `resnet50-19c8e357.pth`, with only ImageNet `fc.*` excluded and any justified modern BN buffer compatibility explicitly validated. No silent ImageNet V2 substitution. Record fetched weight provenance/hash during implementation; do not invent a full hash now. Reconstructing trained checkpoints must disable initialization downloads and strictly load the trained state.
- Reduction convolution weight and bias: historical uniform distribution `[-1/sqrt(2048), +1/sqrt(2048)]`.
- Reduction BN: affine/trainable scale and bias; epsilon `1e-5`, momentum `0.1`, running mean zero/variance one; **scale initialized Uniform[0,1], bias zero**. This is PyTorch 0.3 constructor behavior inherited by Huang's heads, not modern unit-scale BN initialization. Backbone BN is independently initialized/loaded. No blanket initializer may overwrite pretrained weights or these head choices.
- Train/eval use the same output structure; BN changes statistics behavior normally. Single-item training batches are incompatible with pooled BN; configured random loader drops the incomplete final batch. Evaluation with one image must work.

Reference anchors (paths relative to the reference checkout): `bpm/model/PCBModel.py:9–70`; `bpm/model/resnet.py:56–147,182–190`; `script/experiment/train_pcb.py:39–128,196–205,260–277,340–352,437–495`. Historical initializer evidence: [PyTorch 0.3 BatchNorm](https://raw.githubusercontent.com/pytorch/pytorch/v0.3.0/torch/nn/modules/batchnorm.py), `_BatchNorm.__init__/reset_parameters` lines 11–37, and [convolution](https://raw.githubusercontent.com/pytorch/pytorch/v0.3.0/torch/nn/modules/conv.py), `_ConvNd.reset_parameters` lines 37–44. Source lines are audit anchors; recheck by function after changes.

### Outputs, loss, dimensions, and logging

```text
PCB.embedding_dim = 1536          # retrieval width
PCB.feat_dim = None               # no exposed legacy metric-loss feature
{
  "emb": Tensor[B,1536],          # concatenate six post-BN/ReLU parts, top to bottom
  "feat_raw": None,
  "feat_bn": None,
  "logits": (Tensor[B,C], ... six heads ...)
}
```

No per-part or global model-side L2 normalization. `extract_features()` collects descriptors. The evaluator globally normalizes once, then uses Euclidean distance. Keep `eval.normalize_feat=true`, `topk=[1,5,10]`, reranking false. Reference normalization divides by `norm + float32 epsilon`; framework uses `norm + 1e-12`. Preserve the common numerical policy and document it; both leave an all-zero descriptor zero, but small-norm results are not exactly equivalent.

Generic logits: one `[B,C]` floating tensor, nonempty flat ordered tuple/list of such tensors, or `None`. Validate rank, equal batch/class dimensions, label batch size, compatible device/dtype; reject malformed/nested/mixed sequences clearly. Generic consumers do not hardcode six. Keep tensor-only ResNet50 arithmetic unchanged.

Ordinary batch-mean CE per head, then sum six scalars. `loss.id.head_aggregation` is a planned generic `sum|mean` field, default `sum`; PCB uses only sum, weight 1, smoothing 0. Triplet/Center explicitly disabled. Do not average logits. Missing metric features fail only when enabled metric losses require them; positive width remains mandatory for Center construction. Remove unconditional positive `feat_dim` in orchestration. `embedding_dim` is retrieval metadata, not a redefinition of ResNet50's metric width.

Single-head `acc/id` remains unchanged. Multi-head `acc/id_mean_heads` is the arithmetic mean of independently computed head accuracies, detached/no-grad and diagnostic only. No per-head logs or designated-head option initially. Keep aggregate loss logs. Do not label head LR as `lr/bias`; use generic group labels when prefix rules are active, preserving existing labels without rules.

### Project recipe and explicit adaptations

| Setting | Project PCB |
| --- | --- |
| Duration | **120 epochs; reference is 60** |
| Training batch | 64; random shuffled images, drop incomplete batch |
| Evaluation batch | 32 initially, no shuffle, keep final batch |
| Optimizer | SGD, momentum 0.9, Nesterov false |
| Weight decay | 0.0005 for all trainable parameters, including bias and BN |
| LR | backbone 0.01; all new layers 0.1 |
| Schedule | decay ×0.1 beginning epoch 41; no later milestones; no warmup |
| Augmentation | resize 384×128; horizontal flip p=0.5; no padding/crop/erasing |
| Pixels | RGB /255; mean [0.486,0.459,0.408]; std [0.229,0.224,0.225] |
| Precision | FP32, AMP disabled |
| Reproducibility | seed 42, deterministic false, cuDNN benchmark true |
| Retrieval | 1536-D concat; evaluator global normalization; Euclidean |
| Test-time flip / reranking | disabled |
| Periodic evaluation / selection | Every 10 epochs; strict improvement in mAP updates `checkpoints/ckpt_best.pth` |
| Authoritative selected checkpoint | `checkpoints/ckpt_best.pth`; best epoch may be 10,20,…,120 |
| Latest/final training state | `checkpoints/ckpt_last.pth`; epoch 120 after successful completion |

Existing common torchvision preprocessing is retained rather than Huang's OpenCV `INTER_LINEAR`; pixel-identical reproduction is not claimed. Reference defaults to unseeded training, with an optional seed-1 path disabling cuDNN; project seeding is a recorded adaptation. Keep existing benchmark dataset membership authoritative; a `trainval` name alone does not prove equivalence. CUHK03 partition/image-type provenance requires evidence. MSMT17 is a new application of the chosen recipe, not reproduction of a Huang MSMT17 result.

Training batch 64 is the target. If hardware makes it infeasible, record evidence and request a methodological decision before an authoritative run; do not silently reduce it, rescale LR, enable AMP, or equate gradient accumulation with BN batch 64. Smaller bounded diagnostic batches are explicitly non-authoritative. Evaluation batch may be reduced for memory after checking equivalent outputs.

Planned generic optimizer schema (not implemented yet):

```yaml
optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  nesterov: false
  weight_decay: 0.0005
  bias_lr_factor: 1.0
  weight_decay_bias: 0.0005
  param_groups:
    - prefix: "backbone."
      lr_mult: 0.1
sched:
  name: warmup_multistep
  milestones: [40]
  gamma: 0.1
  warmup_iters: 0
  warmup_factor: 1.0
  warmup_method: linear
train:
  epochs: 120
  eval_interval: 10
  save:
    save_best: true
    save_last: true
    metric: mAP
    resume: ""
```

Each trainable parameter occurs exactly once. Unmatched parameters use multiplier 1. Reject ambiguous overlapping rules and unmatched rules. Effective LR = base LR × prefix multiplier × applicable existing bias multiplier. Existing bias decay policy remains; no rules means unchanged historical behavior. Integrated PCB backbone module is named `backbone`.

Scheduler advances after each optimizer update. For fixed `S=len(train_loader)`, updates 1 through `40S` use 0.01/0.1; update `40S+1` through `120S` use 0.001/0.01. A log immediately after update `40S` can show the next update's LR. Verify constructor behavior, boundary, terminal value, and restored optimizer/scheduler state. `milestones:[41]` is wrong for this builder. Do not use the legacy iteration-stepped `step` branch with epoch units. If the intended trace cannot be achieved, block/report; do not shift milestones silently.

### Common checkpoint protocol and selection provenance

Evaluate the current model through the common evaluator at epochs 10,20,30,…,120. On each strict mAP improvement over the previously recorded best, save/update `ckpt_best.pth`; equal mAP does not replace the earlier best. Continue persisting `ckpt_last.pth` as latest training state, reaching epoch 120. The authoritative selected PCB model for within-domain and cross-domain evaluation is `ckpt_best.pth`, not automatically the last checkpoint. Keep final-epoch metrics distinct from selected-best metrics and label both with their actual checkpoint and epoch.

For every source model record: dataset; training epochs = 120; evaluation interval = 10; selection metric = mAP; best epoch; best mAP at selection; `ckpt_best` SHA256; `ckpt_last` epoch = 120; `ckpt_last` SHA256. Persist this evidence in the eventual experiment manifest/artifacts and dissertation evidence. The selected best epoch must belong to {10,20,…,120} and agree with the full periodic evaluation history, using the first occurrence of the maximum when mAP ties.

Run directories must be fresh or explicitly validated for resume; stale checkpoints from unrelated runs must never be selected. The existing resume helper initializes best mAP from resumed checkpoint scores, which may not preserve the historical maximum. Before any resumed authoritative run, verify preservation of the historical best checkpoint and full selection history; if not established, block that run and report a separately scoped generic correction rather than silently selecting a worse checkpoint. No resume code is changed by this plan update.

Source checkpoint selection uses only that source run's configured within-domain query/gallery evaluation. Freeze the resulting `ckpt_best.pth` hash across the entire matrix row, including its diagonal; never choose source epochs based on cross-domain target scores. Disclose the common test-split selection limitation for all architectures. This checkpoint protocol does not change the frozen PCB optimization recipe.

### Reconstruction and cross-domain ownership

Add versioned reconstruction metadata containing model identity, variant, source `num_classes`, `embedding_dim`, contract version, and resolved configuration (the existing top-level `cfg` may remain the single canonical configuration). Validate consistency among metadata, config, and tensors; reject unsupported versions and mismatches. Preserve legacy ResNet50 `classifier.weight` inference. Strip a uniform `module.` prefix before inference/load; mixed-prefix state must not be silently rewritten. Rebuild without pretrained downloads; strict state loading stays mandatory. Never use target class count to rebuild source heads.

For cross-domain evaluation, preserve the trained source architecture, resolution, pixel normalization, descriptor, and evaluation policy. Substitute target root/dataset/split/format/protocol; target batch/workers/device are execution settings and must be applied intentionally. Current whole-`data.test` replacement must be corrected. A target 256×128 baseline preset must not override PCB's 384×128 preprocessing. No adaptation, fine-tuning, target-label training, or target-driven model selection.

## Compatibility map and source/test index

| Requirement | Classification | Owner / evidence at audited HEAD |
| --- | --- | --- |
| YAML loader and common entry points | REUSE AS-IS / CONFIGURE | `reid/utils/config.py:112`, `scripts/train.py` |
| PCB backbone, stripes, heads, descriptor | MODEL-SPECIFIC IMPLEMENTATION | proposed `reid/models/pcb.py`; reference `PCBModel.py` |
| Model dispatch before baseline parsing | GENERIC EXTENSION | `reid/models/build.py:4–35` |
| Dictionary transport | REUSE AS-IS | `reid/models/outputs.py:4–31`; add shared logits validation here if appropriate |
| Multi-head CE / disabled metric semantics | GENERIC EXTENSION | `reid/losses/build.py:44–63,89–117`; `reid/utils/config.py:130` |
| Optional metric width | GENERIC EXTENSION | `scripts/train.py:371`; `scripts/smoke_reid_pipeline.py:146` |
| Accuracy / LR labels | GENERIC EXTENSION | `reid/engine/train_loop.py:63–96,115–122` |
| Prefix LR groups | GENERIC EXTENSION | `reid/optim/build.py:9–42` |
| Exact epoch-41 schedule | CONFIGURE | `reid/optim/build.py:88–103`; `reid/optim/lr_scheduler.py:34–45` |
| Random loader / transformations | CONFIGURE | `reid/data/build.py:144–225`; `reid/data/transforms.py:51–95` |
| Four dataset parsers / partitions | REUSE AS-IS | `reid/data/build.py` and imported dataset implementations |
| Embedding evaluator / ranking | REUSE AS-IS | `reid/engine/evaluator.py:17–36,60–98`; `reid/metrics/` |
| Reconstruction metadata and strict helper | GENERIC EXTENSION | `reid/utils/checkpoint.py`; `scripts/evaluate.py:80`; `scripts/evaluate_cross_domain.py:38` |
| Source preprocessing preservation | GENERIC EXTENSION | `scripts/evaluate_cross_domain.py:43–54` |
| Common metrics/artifacts / periodic mAP best-selection policy | REUSE AS-IS / CONFIGURE | `reid/utils/metrics_artifacts.py`; `scripts/train.py:404–528` |
| Historical protocol documentation | GENERIC EXTENSION | `docs/model_plugin_protocol.md` and its documentation tests |
| Matrix aggregation | REUSE AS-IS, verify later | `reid/utils/experiment_matrix.py`; `scripts/aggregate_results.py`; `scripts/report_model_selection.py` |
| New dataset loaders, sampler, evaluator, ranking | NOT REQUIRED | no changes planned |
| Runtime/export/deployment, MGN, TransReID, ablations | NOT REQUIRED | outside this PCB execution scope |

Existing tests to retain/extend by owner: `tests/test_checkpoint.py`, `test_evaluation_harness.py`, `test_model_interface.py`, `test_model_plugin_contract.py`, `test_model_plugin_protocol_doc.py`, `test_model_forward.py`, `test_loss_interface.py`, `test_reid_loss_modes.py`, `test_train_loop_optim.py`, `test_optim_build.py`, `test_train_orchestration.py`, `test_config_schema.py`, `test_sampler.py`, `test_data_transforms.py`, `test_dataset_protocol.py`, `test_market1501_dataset.py`, `test_duke_dataset.py`, `test_cuhk03_dataset.py`, `test_msmt17_dataset.py`, `test_smoke_reid_pipeline.py`, `test_artifact_format.py`, `test_reproducibility_artifacts.py`, `test_experiment_matrix.py`, `test_resnet50_strong_baseline.py`. These filenames exist; new test filenames below are explicitly proposed, not claimed to exist.

## Execution phases

All phases must update their implementation record, decisions/deviations, review notes, and this plan. Listed expected changes are future bounds, not authorization. Every phase ends in STOP for review.

### Phase 0 — Freeze pre-PCB baseline

- **Objective:** Preserve the starting repository/environment/ResNet50 contract.
- **Why this step exists:** Prevent accidental attribution of old behavior or missing evidence to PCB.
- **Prerequisites:** Repository audit and read-only current-state checks.
- **Files to inspect:** `reid/models/baseline.py`, `docs/baseline_protocol_v1.md`, baseline configurations, baseline evidence above.
- **Files expected to change:** `plan.md` only for this documentation deliverable.
- **Implementation tasks:** Record branch, HEAD, status, environment, prior test evidence, behavior, and limitations.
- **Validation/tests:** Current Git/environment inspection; distinguish historical tests from newly run tests.
- **Exit criteria:** Pre-PCB state recorded without asserting a new full-suite pass.
- **Status:** Completion `[x]` audit/documentation; review `[?]` consolidated record.
- **Implementation record:** Prior audit completed; branch/HEAD/clean state and CPU environment rechecked 2026-10-02. No baseline tests rerun in roadmap creation.
- **Decisions/deviations:** Revalidate affected baseline tests during implementation; full coverage remains a later gate.
- **Review notes:** Historical command/artifact limits are explicit above.
- **Next step:** Phase 1 documentation; no implementation authorized.

### Phase 1 — Freeze reference specification

- **Objective:** Identify one unambiguous executable PCB variant.
- **Why this step exists:** Prevent mixing Huang and original-author behaviors.
- **Prerequisites:** Reference investigation and selection completed.
- **Files to inspect:** Reference `README.md`, `bpm/model/PCBModel.py`, `resnet.py`, `script/experiment/train_pcb.py`, `bpm/dataset/TestSet.py`, `PreProcessImage.py`, `bpm/utils/distance.py`.
- **Files expected to change:** `plan.md` only.
- **Implementation tasks:** Record pinned commit, architecture, objective, descriptor, optimizer, preprocessing, initialization, and provenance differences.
- **Validation/tests:** Source-based Phase 3 findings; local reference HEAD/status rechecked during creation.
- **Exit criteria:** Selected variant and historical-default dependencies are explicit.
- **Status:** Completion `[x]` investigation/documentation; review `[?]` consolidated record.
- **Implementation record:** Commit verified; frozen contract and source anchors preserved above. No model executed in this phase's documentation task.
- **Decisions/deviations:** Historical BN initialization is explicit; modern defaults must not silently replace it.
- **Review notes:** README results remain external claims, not reproduced measurements.
- **Next step:** Phase 2 documentation.

### Phase 2 — Freeze compatibility map

- **Objective:** Assign every integration requirement to a framework owner.
- **Why this step exists:** Keep architecture implementation separate from necessary generic extensions.
- **Prerequisites:** Framework audit and Phase 1.
- **Files to inspect:** Source/test index and compatibility table above.
- **Files expected to change:** `plan.md` only.
- **Implementation tasks:** Classify reuse, configuration, generic extension, model implementation, and excluded work.
- **Validation/tests:** Cross-check current filenames, entry-point dependencies, and actual test inventory.
- **Exit criteria:** Every frozen requirement has an owner; no duplicate training pipeline planned.
- **Status:** Completion `[x]` analysis/documentation; review `[?]` consolidated record.
- **Implementation record:** Prior compatibility investigation consolidated; source/test inventory checked 2026-10-02.
- **Decisions/deviations:** Stale plug-in protocol restrictions explicitly superseded by user contract; reconcile later.
- **Review notes:** Test updates must strengthen intended contracts, not erase baseline behavior checks.
- **Next step:** Phase 3 documentation.

### Phase 3 — Freeze integration contract

- **Objective:** Record decisions sufficient to start bounded implementation.
- **Why this step exists:** Prevent silent methodological changes between sessions.
- **Prerequisites:** Phases 0–2; user's current roadmap instructions.
- **Files to inspect:** Frozen contract above and prior Phase 3 report.
- **Files expected to change:** `plan.md` only.
- **Implementation tasks:** Consolidate outputs, dimensions, loss, statistics, groups, normalization, metadata, preprocessing ownership, 120-epoch decision, and common periodic mAP checkpoint selection.
- **Validation/tests:** Document consistency checks; verify 120 training epochs, 10-epoch evaluation, mAP-selected best, and separate latest/final state throughout the plan.
- **Exit criteria:** No unresolved architectural contract blocks Phase 4; operational training prerequisites remain explicit later gates.
- **Status:** Completion `[x]` contract documentation; review `[?]` roadmap and derived continuation.
- **Implementation record:** Original verification completed read-only; project duration is 120 epochs; the subsequent user correction restores common periodic mAP best-checkpoint selection. This plan persists both decisions.
- **Decisions/deviations:** 120 epochs, decay still epoch 41; evaluate every 10 epochs, select strict-best mAP, retain epoch-120 last state. Common test-split selection limitation remains explicit.
- **Review notes:** No future phase authorized by creation of this document.
- **Next step:** Phase 4, only after explicit authorization.

### Phase 4 — Generic checkpoint reconstruction

- **Objective:** Remove reconstruction's exclusive dependency on `classifier.weight`.
- **Why this step exists:** Multi-head checkpoints must rebuild source classifiers without downloads or target-class leakage.
- **Prerequisites:** Reviewed contract and explicit Phase 4 authorization; inspect current checkpoint API/tests.
- **Files to inspect:** `reid/utils/checkpoint.py`, `reid/models/build.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/evaluate_cross_domain.py`, `tests/test_checkpoint.py`.
- **Files expected to change:** Those checkpoint/build/entry-point files only as needed for generic reconstruction, `tests/test_checkpoint.py`; proposed `tests/test_checkpoint_reconstruction.py`; `plan.md`.
- **Implementation tasks:** Add versioned metadata and a common reconstruction path; validate metadata/config/tensors; preserve legacy fallback; normalize uniform prefixes; disable initialization downloads; retain strict loading and source class count. Do not implement PCB here.
- **Validation/tests:** Historical ResNet50 fallback; synthetic metadata single/multi-head round trips; prefix handling; unknown version/mismatch rejection; target identity count independence; missing/unexpected/shape-mismatched tensors; download function patched to fail if called. Existing checkpoint test asserts exact payload keys and needs intentional additive-schema coverage.
- **Exit criteria:** Generic reconstruction verified with test-only multi-head surrogate; no PCB production code required. Actual PCB round trip is a mandatory Phase 10/18 gate, not falsely marked passed here.
- **Status:** Completion `[x] IMPLEMENTED + VALIDATED` on 2026-10-05; review `[?] NEEDS HUMAN REVIEW`. Phase 4 was explicitly authorized by the user attachment `153f6c9b-9749-40f1-ada8-5786a11e6838/Pasted text.txt`. Earlier documentation-session statements elsewhere in this plan are historical; this Phase 4 authorization/record supersedes them for Phase 4 only.
- **Implementation record:** Phase 4 only completed on 2026-10-05. Starting branch `main` tracking `origin/main`, HEAD `37ed8a50dedc5aba0bade8765f2190824e217482`; starting Git status `?? plan.md`, all tracked files clean. Read the complete roadmap, confirmed Phases 0–3 recorded complete and the latest 120-epoch/every-10-epoch strict-best-mAP policy, and inspected checkpoint callers/tests before editing. No applicable AGENTS.md was found in the repository or checked ancestors.

  Exact changed files:
  - `reid/utils/checkpoint.py`: metadata construction/validation, shared state normalization, metadata-first class resolution, strict source-model reconstruction; additive metadata saving and preserved optimizer/scheduler loading.
  - `reid/models/build.py`: explicit metadata declaration and keyword-only `initialize_pretrained` override, default true; model computations/state tensors unchanged.
  - `scripts/evaluate.py`: shared class helper compatibility wrapper and shared reconstruction call.
  - `scripts/evaluate_cross_domain.py`: shared class helper compatibility wrapper and shared reconstruction call; existing data-config merge intentionally unchanged.
  - `tests/test_checkpoint.py`: existing exact-payload assertion updated for additive `reconstruction` field.
  - `tests/test_checkpoint_reconstruction.py`: new tests using a test-only single/multi-head surrogate, actual baseline models, and a mocked cross-domain entry-point test.
  - `plan.md`: only this Phase 4 section updated.

  Design and persisted schema:
  ```yaml
  reconstruction:
    schema_version: 1
    output_contract_version: 1
    model_name: reid_baseline  # selected builder identity
    variant: null             # cfg.model.variant when relevant
    num_classes: 1041         # example source classifier count; null for classifier-free model
    embedding_dim: 2048       # example retrieval width
  cfg: ...                    # existing canonical resolved config; not duplicated in metadata
  model: ...                  # unchanged state_dict representation
  ```
  Builders declare `model.checkpoint_metadata` via `make_reconstruction_metadata(cfg, num_classes, embedding_dim)`. Saving a declared model with cfg persists a copied reconstruction block; DataParallel/DDP declarations are read from the wrapped module. Unknown/undeclared models retain the existing save API but cannot use metadata-based reconstruction until their builder declares the contract. `save_checkpoint(..., cfg=None)` remains supported and omits self-describing metadata; reconstruction then requires a caller-supplied legacy baseline source configuration.

  `reconstruct_model(checkpoint, cfg=None)` normalizes wrapped/raw state, strips a uniform `module.` prefix (including PyTorch state-version metadata), rejects mixed prefixes, validates versions/identity/variant/dimensions against canonical source config, resolves source classes, invokes the common builder with `initialize_pretrained=False`, checks the constructed declaration, then strictly loads tensors. Unsupported/null/incomplete explicit metadata never silently falls back. Metadata-free reconstruction is bounded to historical `reid_baseline`; only its `classifier.weight` is inferred (or None for classifier-free state). Embedded source cfg takes precedence; caller cfg is a fallback only for old/raw checkpoints without cfg. Dataset/target class counts are never used. Future architectures must register their normal builder and declare metadata; no production surrogate or PCB registry entry was added.

  Validation environment: `/home/filsduvent/environments/Reid/bin/python`, Python 3.12.3, torch 2.7.1+cpu, torchvision 0.22.1+cpu. Exact focused commands, from repository root:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider tests/test_checkpoint_reconstruction.py tests/test_checkpoint.py > /tmp/phase4-focused.log 2>&1
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider tests/test_checkpoint_reconstruction.py tests/test_checkpoint.py > /tmp/phase4-focused-final.log 2>&1
  ```
  Initial captured focused result: **33 passed in 17.58 s**. After preserving optional-cfg saves and adding configuration-save rejection/cross-domain entry-point coverage, final focused result: **36 passed in 18.30 s**. The initial streaming test invocation's session handle was not retained, so it was allowed to finish and rerun with captured output; its uncollected result is not counted. No test failures were observed in the captured runs.

  Exact affected regression command:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider tests/test_train_orchestration.py tests/test_model_interface.py tests/test_model_forward.py tests/test_model_plugin_contract.py tests/test_loss_interface.py tests/test_reid_loss_modes.py tests/test_optim_build.py tests/test_train_loop_optim.py tests/test_evaluation_harness.py tests/test_config_schema.py tests/test_experiment_matrix.py tests/test_artifact_format.py tests/test_reproducibility_artifacts.py tests/test_resnet50_strong_baseline.py > /tmp/phase4-regression.log 2>&1
  ```
  Result: **112 passed, 4 skipped in 36.76 s**. Skips are CUDA-only: one model-forward case and three parameterized training-loop cases; CUDA is unavailable. Final focused plus affected regressions total **148 passed, 4 skipped**; repeated earlier focused runs are not added. Not a claim of a full repository/GPU suite.

  Covered failures: unsupported schema/output versions; malformed/missing/null metadata; identity/variant/config dimension conflicts; source class-count mismatches; mixed prefixes; missing/unexpected/shape-incompatible state tensors; constructed declaration mismatch. Positive coverage: legacy raw/wrapped baseline and uniform prefixes, exact baseline forward equality, new metadata single/multi-head round trips, independent source classes despite target count 999, DataParallel state-version metadata, classifier-free baseline, optimizer/scheduler restoration, no pretrained calls, and preserved optional-cfg API. Network-dependent initialization is forbidden by assertions/mocks in reconstruction tests.

  Additional read-only historical checkpoint validation used `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -` with this inline script:
  ```python
  from pathlib import Path
  import hashlib
  import torch
  from reid.utils.checkpoint import reconstruct_model
  p=Path('exp/msmt17_no_label_smoothing/checkpoints/ckpt_best.pth')
  if not p.is_file():
      raise SystemExit('Historical checkpoint missing; read-only verification unavailable.')
  def digest():
      with p.open('rb') as f:
          return hashlib.file_digest(f,'sha256').hexdigest()
  before=digest()
  checkpoint=torch.load(p,map_location='cpu',weights_only=True)
  assert 'reconstruction' not in checkpoint
  from torchvision.models import ResNet50_Weights
  original=type(ResNet50_Weights.IMAGENET1K_V2).get_state_dict
  def no_download(*a,**k):
      raise AssertionError('Pretrained initialization requested')
  type(ResNet50_Weights.IMAGENET1K_V2).get_state_dict=no_download
  try:
      model=reconstruct_model(checkpoint).eval()
      with torch.no_grad():
          outputs=model(torch.zeros(2,3,256,128))
      assert outputs['emb'].shape==(2,2048)
      assert outputs['logits'].shape==(2,1041)
      assert torch.isfinite(outputs['emb']).all()
  finally:
      type(ResNet50_Weights.IMAGENET1K_V2).get_state_dict=original
  assert digest()==before
  print('PASS historical checkpoint: strict legacy reconstruction; no pretraining; emb(2,2048), logits(2,1041); epoch',checkpoint['epoch'],'unchanged SHA256',before)
  ```
  Result: PASS; checkpoint epoch 120; expected shapes; file unchanged at SHA256 `5c751a6a2f3d2b19456684d66c72b3f86ad8d8a7cf7ab061aacd2071720bfc3f`. No historical checkpoint rewritten, no dataset experiment/training started.

  Final checks: `git diff --check`; inspect `git diff --stat` and `git status --short`; verify plan text outside Phase 4 is byte-for-byte unchanged. Expected resulting status: modified `reid/models/build.py`, `reid/utils/checkpoint.py`, `scripts/evaluate.py`, `scripts/evaluate_cross_domain.py`, `tests/test_checkpoint.py`; untracked existing `plan.md` and new `tests/test_checkpoint_reconstruction.py`. No commit or push.
- **Decisions/deviations:** No scope deviation. Test-only surrogate avoids the dependency on Phases 5–9; real PCB round trips remain deferred. The pre-existing broken filesystem sandbox required approved escalated execution. No training recipe, checkpoint-selection policy, losses, optimizer behavior, datasets, evaluator metrics, or runtime code changed. No `scripts/train.py` change was needed because its existing common save calls automatically persist builder-declared metadata.
- **Review notes:** Awaiting human review; STOP. Limitations: actual PCB integration/round trip is not validated; CUDA coverage unavailable; source preprocessing/config-artifact reconciliation remains Phase 15, whose merge behavior was not changed here; existing historical-best resume and RNG/scaler persistence limitations remain. Global plan header/older records were left unchanged because this authorization requires updating Phase 4 only; this dated section is the current Phase 4 state.
- **Next step:** Phase 5, separately authorized.

### Phase 5 — PCB backbone

- **Objective:** Add the reference-compatible ResNet50 feature backbone in a separate PCB module.
- **Why this step exists:** Preserve baseline internals and historical weight provenance.
- **Prerequisites:** Satisfied. The user explicitly reviewed and accepted Phase 4 and authorized ONLY Phase 5 in attachment `d306c23f-bb70-43a1-8cd5-3e5ef7c10908/Pasted text.txt` on 2026-10-05. Phase 4 is committed/pushed as `26508529016387db699182acee7b467288581820`. Its earlier awaiting-review wording elsewhere is historical; acceptance is recorded here because this task permits updating Phase 5 only. The frozen initialization contract remains in force.
- **Files to inspect:** Reference `bpm/model/resnet.py`; `reid/models/baseline.py`; `tests/test_model_forward.py`.
- **Files expected to change:** Proposed `reid/models/pcb.py`, proposed `tests/test_pcb_model.py`, `plan.md`.
- **Implementation tasks:** Implement backbone construction, stride/dilation and controlled historical initialization, with an explicit no-pretraining mode. Do not register an incomplete end-to-end model in the public builder.
- **Validation/tests:** Stage topology/stride assertions; `[B,3,384,128]` → `[B,2048,24,8]`; gradient flow; mock weight mapping and separately controlled real historical-weight load with provenance recorded; no final ReID weights. No network dependency in ordinary unit tests.
- **Exit criteria:** Backbone structure, shape, gradient path, and actual initialization compatibility verified; unavailable required weights must be reported, not replaced.
- **Status:** Completion `[x] IMPLEMENTED + VALIDATED` on 2026-10-05; review `[x] REVIEWED AND ACCEPTED` by the user on 2026-10-05. Only Phase 5 authorized; Phase 6 remains unstarted and unauthorized.
- **Implementation record:** Starting branch `main`, tracking `origin/main`; HEAD `26508529016387db699182acee7b467288581820`; working tree clean. Read the complete current plan, confirmed Phase 4 completion/user acceptance and the unchanged common checkpoint-selection policy, inspected the clean pinned reference checkout at `1686e889eb01c28a54b633051418012e15d9c9f3`, checked ancestor/repository guidance (no applicable AGENTS.md found), and inspected the installed torchvision bottleneck/initialization code before implementation.

  Exact files changed:
  - New `reid/models/pcb.py`: internal `PCBBackbone` feature component and historical ImageNet loading; no public builder registration.
  - New `tests/test_pcb_model.py`: 20 network-independent Phase 5 topology/forward/backward/loading tests.
  - `plan.md`: this Phase 5 section only. All text outside it is preserved byte-for-byte.

  Architecture: reuse torchvision's uninitialized ResNet50 components, retaining named `conv1`, `bn1`, `relu`, `maxpool`, `layer1`–`layer4` only. Stage lengths `[3,4,6,3]`, bottleneck expansion 4, stride located on `conv2` (3×3). Set `layer4[0].conv2.stride=(1,1)`, `layer4[0].downsample[0].stride=(1,1)`, and the block's descriptive `stride=1`. Dilation remains `(1,1)`; no stride-to-dilation substitution. No global pooling/FC modules are registered or executed. Input `[1,3,384,128]` produces `[1,2048,24,8]` in train and eval modes. A scalar output backward check reaches input/stem/all stages/projection and yields finite gradients; no optimizer or training run was used for this check.

  Reference trace: `bpm/model/resnet.py:56–92` defines bottleneck/main/projection behavior; lines 95–147 define stem/stages, normal fan-out convolution initialization, unit/zero backbone BN, and feature-map forward; lines 182–190 specify `[3,4,6,3]` and ImageNet loading. `bpm/model/PCBModel.py:10–23` selects last stride 1/dilation 1. The installed torchvision implementation matches these selected backbone operations and initialization distributions; controlled loaded-weight parity below establishes actual output agreement. Same-seed historical random-initialization draw order is not claimed.

  Construction and initialization:
  ```python
  from reid.models.pcb import PCBBackbone
  backbone = PCBBackbone(pretrained=False)  # no weight read/download; future checkpoint construction
  # Explicit initialization from the pinned file:
  backbone = PCBBackbone(pretrained=True, weights_path="/tmp/resnet50-19c8e357.pth")
  # Alternatively pretrained=True alone uses the exact URL and verified torch-hub cache.
  ```
  The default is explicitly non-pretrained; eventual new PCB training must opt into historical initialization in its later builder. Passing weights_path while pretrained=False fails rather than silently loading. `ReidBaseline`, its V2 initialization, the common model builder, and Phase 4 helpers are unchanged. This component returns a feature tensor, not the full ReID output dictionary, and does not claim complete PCB metadata/reconstruction support yet.

  Historical weight provenance:
  - Exact URL read from reference `bpm/model/resnet.py:11`: `https://download.pytorch.org/models/resnet50-19c8e357.pth`.
  - Controlled artifact location: `/tmp/resnet50-19c8e357.pth` (temporary, outside repository); 102502400 bytes.
  - Verified SHA256: `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`, matching the historical filename prefix. Full hash is pinned in the module and recorded here. No ImageNet V2 or final ReID weights substituted.
  - Raw file contains 267 float32 tensors: 265 backbone entries plus `fc.weight[1000,2048]` and `fc.bias[1000]`. It has no BN batch counters.
  - Loading validates keys, shapes, and dtypes before copying; excludes precisely the two validated ImageNet FC entries; explicitly supplies 53 zero-valued modern BN `num_batches_tracked` buffers, then uses strict loading. Missing/unexpected/shape/dtype mismatches beyond this documented legacy compatibility fail clearly; source mappings are not mutated.
  - Every cached/local file is checked against the full pinned SHA256 before deserialization. Downloads use the same full hash, exact URL, and no alternate-source fallback.

  Encountered failure and resolution: initial inspection using `torch.load(..., weights_only=True)` failed because this official file uses the legacy tar serialization format unsupported by that mode in torch 2.7.1. After full checksum verification, the controlled load succeeded with `weights_only=False`. Production code permits that legacy mode only after checking the exact pinned full checksum, hashes and deserializes the same open file, and rejects a wrong cached/local file before reaching torch.load. Ordinary tests mock this boundary and never deserialize an arbitrary legacy artifact. This is serialization compatibility, not a different weight initialization. No unit-test failures occurred.

  Exact commands from repository root:
  ```bash
  curl --fail --location --retry 2 --connect-timeout 15 --output /tmp/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
  sha256sum /tmp/resnet50-19c8e357.pth
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -p no:cacheprovider tests/test_pcb_model.py > /tmp/phase5-focused.log 2>&1
  PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B /tmp/phase5_reference_check.py > /tmp/phase5-reference.log 2>&1
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -rs -p no:cacheprovider tests/test_model_forward.py tests/test_model_interface.py tests/test_resnet50_strong_baseline.py tests/test_checkpoint_reconstruction.py > /tmp/phase5-regression.log 2>&1
  git diff --check
  git status --short
  ```
  Environment: existing Reid Python 3.12.3 / torch 2.7.1+cpu / torchvision 0.22.1+cpu; no dependency changes. Focused result: **20 passed in 11.43 s**. Affected baseline/reconstruction result: **51 passed, 1 skipped in 26.85 s**; skipped `tests/test_model_forward.py:192` is CUDA-only, CUDA unavailable. Combined **71 passed, 1 skipped**. No ordinary test requires network, a cached historical file, or the external reference checkout. Tests verify topology, both stride paths, unchanged dilation, train/eval shape, all-parameter finite gradients, absence of pooling/classifier heads, pretrained-disabled behavior, mocked historical mapping, explicit BN-counter reset, malformed state rejection before mutation, exact URL/hash constants, cache/download checksum enforcement, and refusal of missing/wrong files without substitution.

  Separate controlled reference parity script `/tmp/phase5_reference_check.py` (temporary; reproduce from the exact source below):
  ```python
  from pathlib import Path
  import hashlib
  import importlib.util
  import subprocess
  import torch
  from reid.models.pcb import PCBBackbone, HISTORICAL_IMAGENET_SHA256

  root=Path('/home/filsduvent/UFPR/beyond-part-models')
  assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=='1686e889eb01c28a54b633051418012e15d9c9f3'
  path=Path('/tmp/resnet50-19c8e357.pth')
  def digest():
      with path.open('rb') as stream:
          return hashlib.file_digest(stream,'sha256').hexdigest()
  assert digest()==HISTORICAL_IMAGENET_SHA256
  model=PCBBackbone(pretrained=True,weights_path=path).eval()
  spec=importlib.util.spec_from_file_location('pinned_reference_resnet',root/'bpm/model/resnet.py')
  reference=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(reference)
  ref=reference.resnet50(pretrained=False,last_conv_stride=1,last_conv_dilation=1).eval()
  # This exact file was verified above; historical tar requires legacy loading.
  state=torch.load(path,map_location='cpu',weights_only=False)
  feature_state={k:v for k,v in state.items() if not k.startswith('fc.')}
  ref.load_state_dict(feature_state,strict=True)
  assert list(model.state_dict())==list(ref.state_dict())
  for key,value in model.state_dict().items():
      assert torch.equal(value,ref.state_dict()[key]),key
  for name,module in ref.named_modules():
      if isinstance(module,torch.nn.Conv2d):
          actual=model.get_submodule(name)
          assert (actual.stride,actual.dilation,actual.padding,actual.kernel_size)==(module.stride,module.dilation,module.padding,module.kernel_size),name
  torch.manual_seed(42)
  x=torch.randn(1,3,384,128)
  with torch.no_grad():
      actual=model(x)
      expected=ref(x)
  assert actual.shape==(1,2048,24,8)
  torch.testing.assert_close(actual,expected,rtol=0,atol=0)
  assert digest()==HISTORICAL_IMAGENET_SHA256
  print('PASS: exact historical weights; all backbone tensors and convolution topology match pinned reference; output [1,2048,24,8] matches at rtol=0, atol=0.')
  print('Weight bytes:',path.stat().st_size,'SHA256:',digest())
  print('Source keys:',len(state),'loaded backbone keys:',len(feature_state),'explicit new BN counters:',sum(k.endswith('num_batches_tracked') for k in model.state_dict()))
  ```
  Result: PASS. All backbone state tensors and convolution strides/dilations/padding/kernel sizes matched the pinned reference. Loaded-weight forward output matched exactly at `rtol=0, atol=0` for the recorded CPU input. The weight-file checksum remained unchanged. This is a backbone compatibility check, not a dataset experiment or a claim of historical training reproducibility.

  Pre-commit validation Git status: modified `plan.md`; untracked `reid/models/pcb.py` and `tests/test_pcb_model.py`; all other tracked files unchanged. Reference checkout remains clean. No commit/push had been performed at implementation handoff. The user subsequently accepted Phase 5 and authorized its commit/push on 2026-10-05. Files in /tmp are validation artifacts only; no weights added to the repository.
- **Decisions/deviations:** No architectural/recipe deviation. Reuse matching torchvision backbone components rather than copying the legacy repository. Explicit zero BN counters and checksum-gated legacy deserialization are the required modern-runtime compatibility measures. No changes to `ReidBaseline`, public builder, losses, optimizer, datasets, metrics, checkpoint-selection policy, or runtime. No stripes, stripe pooling, part reductions, PCB heads/classifiers, retrieval concatenation, configuration presets, or training implemented.
- **Review notes:** Phase 5 reviewed and accepted by the user on 2026-10-05; commit/push authorized. STOP after publishing this step; Phase 6 requires separate authorization. No unresolved historical-weight blocker. Limitations: CPU validation only; temporary downloaded file may need re-fetching; same-seed historical random draw order is not reproduced; this feature component is not yet a complete/registered PCB ReID model, and full PCB checkpoint reconstruction remains for later integration. Phase 6 is not started.
- **Next step:** Phase 6, separately authorized.

### Phase 6 — Stripe partition and pooling

- **Objective:** Implement exact six-stripe partition and full-stripe average pooling.
- **Why this step exists:** Pooling order and boundaries define the architecture.
- **Prerequisites:** Satisfied. The user reviewed and accepted Phase 5 and explicitly authorized ONLY Phase 6 in attachment `87361208-6c83-4b3c-a594-83ab2aba150a/Pasted text.txt` on 2026-10-05. Phase 5 is recorded as implemented/validated and reviewed/accepted and is committed/pushed as `837112115fba818bb72fe3bddc2cd65ab7f8bd75`. Earlier authorization statements are historical; this record supersedes them for Phase 6 only.
- **Files to inspect:** Pinned reference `bpm/model/PCBModel.py:42–58`; current PCB backbone and its tests; complete roadmap and relevant baseline/checkpoint tests.
- **Files expected to change:** `reid/models/pcb.py`, `tests/test_pcb_model.py`, `plan.md` only.
- **Implementation tasks:** Slice equal-height stripes top-to-bottom; validate divisibility/positive sizes; preserve `[B,2048,1,1]` pooled tensors for later reductions.
- **Validation/tests:** Row-coded deterministic maps prove exact boundaries, ordering, complete coverage, no overlap, full-width arithmetic means; invalid heights fail before pooling; gradients match average-pooling semantics; actual backbone integration and baseline/checkpoint regressions.
- **Exit criteria:** Satisfied: six correctly ordered pooled stripes with no adaptive fallback; backbone contract preserved.
- **Status:** Completion `[x] IMPLEMENTED + VALIDATED` on 2026-10-05; review `[x] REVIEWED AND ACCEPTED` by the user on 2026-10-05. Phase 7 remains unstarted and unauthorized.
- **Implementation record:** Starting branch `main`, tracking `origin/main`; HEAD `837112115fba818bb72fe3bddc2cd65ab7f8bd75`; clean working tree. Read the complete current roadmap and authorization, confirmed the Phase 5 review prerequisite, inspected the existing backbone/tests and the clean Huang reference checkout at `1686e889eb01c28a54b633051418012e15d9c9f3`. No applicable AGENTS.md was found in the repository or checked ancestors.

  Exact files changed:
  - `reid/models/pcb.py`: added parameter-free `PCBStripePool`, with a `partition` helper returning six full-width tensor views and `forward` returning six pooled tensors; updated module description/import. `PCBBackbone` implementation and initialization are unchanged.
  - `tests/test_pcb_model.py`: retained all 20 Phase 5 cases and added 16 Phase 6 cases, independent of network, cached weights, or reference-checkout availability.
  - `plan.md`: this Phase 6 section only; all text outside it remains byte-for-byte unchanged.

  Reference and implementation: pinned `PCBModel.py:50–58` checks height divisibility, computes stripe height, slices top-to-bottom, and calls `F.avg_pool2d` with `(stripe_height, full_width)`. `PCBStripePool` follows these exact spatial operations. Six is architecture-owned, with no configurable stripe-count variant. Input must be a four-dimensional tensor, with positive height divisible by six and positive width; invalid input raises a descriptive `ValueError` before slicing/pooling. No adaptive or uneven partition fallback. Channels and batch items remain separate; output stays four-dimensional, with no normalization, channel reduction, or flattening.

  Standard flow:
  ```text
  [B,3,384,128] -> PCBBackbone -> [B,2048,24,8]
  -> six [B,2048,4,8] views, rows [0:4], [4:8], [8:12], [12:16], [16:20], [20:24]
  -> full 4x8 average per batch/channel -> tuple of six [B,2048,1,1] tensors
  ```
  Usage is `pooled = PCBStripePool()(backbone(images))`; the backbone still independently returns its feature map. The helper also accepts smaller channel counts for exact isolated tests; this does not create another PCB variant.

  Validation evidence:
  - Rows coded 0 through 23 establish exact ordered boundaries and means `1.5, 5.5, 9.5, 13.5, 17.5, 21.5`. Per-row coverage counts equal one and concatenated slices recover the entire map, proving complete coverage without overlap or reordering.
  - Distinct batch/channel/row/column values verify full-width averaging without mixing channels or examples. Valid `(H,W)` cases `(6,1)`, `(12,3)`, `(24,8)`, `(30,5)` exercise equal integer partitioning. Train/eval results agree, inputs remain unchanged, and the pooling module has no parameters or buffers.
  - Heights `0,1,5,7,16,25`, zero width, incorrect rank, and non-tensor input fail clearly. Invalid-height tests forbid any pooling call; actual-backbone integration forbids adaptive pooling.
  - For each individual pooled stripe, autograd gives exactly `1/32` at its source positions and zero everywhere else. A combined scalar with distinct stripe/channel weights calls backward and verifies every source gradient equals its coefficient divided by 32. No optimizer step or training performed.
  - Actual `PCBBackbone(pretrained=False)` on `[1,3,384,128]` verifies feature, unpooled, and pooled shapes, finite values, and agreement with a separate spatial mean. All prior Phase 5 topology, forward, gradient, and historical-loading tests remain passing.

  Exact command from repository root:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -rs -p no:cacheprovider tests/test_pcb_model.py tests/test_model_forward.py tests/test_model_interface.py tests/test_resnet50_strong_baseline.py tests/test_checkpoint_reconstruction.py tests/test_checkpoint.py > /tmp/phase6-tests.log 2>&1
  git diff --check
  git diff --stat
  git status --short --branch
  ```
  Result: **93 passed, 1 skipped in 41.06 s; zero failures**. This comprises 36 PCB tests (20 retained backbone + 16 new stripe cases) and 57 passing baseline/interface/checkpoint regressions. The single skip is `tests/test_model_forward.py:192`, CUDA unavailable. Regression coverage includes ResNet50/BoT forward behavior, model interface, strong-baseline behavior, generic reconstruction, and checkpoint save/load. Environment remains Reid Python 3.12.3 / torch 2.7.1+cpu / torchvision 0.22.1+cpu; no dependency changes. This is a targeted CPU suite, not a full repository or GPU validation claim.

  Final scope checks: `git diff --check` passed; text outside Phase 6 matched HEAD exactly; the complete `PCBBackbone` class matched HEAD exactly. Pre-commit validation Git status: modified `plan.md`, `reid/models/pcb.py`, `tests/test_pcb_model.py` only, on `main`; no untracked files. No commit or push had been performed at implementation handoff. The user subsequently reviewed and accepted Phase 6 and authorized its commit/push on 2026-10-05. The reference checkout remains clean. Test log is temporary in `/tmp`, not a repository artifact.
- **Decisions/deviations:** No architectural or scope deviations. A separate parameter-free consumer preserves the validated backbone API. Explicit `ValueError` checks strengthen the reference assertion with positive-size/rank validation while retaining exact valid-input behavior. No test failures or unresolved implementation blockers. The environment's broken filesystem sandbox required approved escalated command execution; no repository workaround was introduced.
- **Review notes:** Phase 6 reviewed and accepted by the user on 2026-10-05; commit/push authorized. STOP after publishing this step; Phase 7 requires separate authorization. CPU-only validation; full PCB is still an incomplete, unregistered component. No reductions, reduction BN/ReLU, classifiers, retrieval descriptor, loss/optimizer/configuration integration, training, or Phase 7 work implemented. No baseline, checkpoint-selection, dataset, evaluator, or generic infrastructure changes.
- **Next step:** Phase 7 — independent reduction modules, only after Phase 6 review and separate explicit authorization.

### Phase 7 — Independent reduction modules

- **Objective:** Produce six independent post-Conv/BN/ReLU 256-D local features.
- **Why this step exists:** Independent reduction is the selected Huang variant's defining choice.
- **Prerequisites:** Satisfied. The user reviewed and accepted Phase 6 and explicitly authorized ONLY Phase 7 in attachment `a238967c-953f-47f6-a7b4-ca0358496648/Pasted text.txt` on 2026-10-05. Phase 6 is recorded as implemented/validated and reviewed/accepted and is committed/pushed as `4f1bbe9ebcce5f072b44dfd61dba2df1963feb7a`. Earlier authorization wording is historical; this dated record supersedes it for Phase 7 only.
- **Files to inspect:** Complete current roadmap; `reid/models/pcb.py` and PCB tests; pinned `bpm/model/PCBModel.py`; PyTorch v0.3.0 Conv and BatchNorm initialization sources; relevant baseline/interface/checkpoint tests.
- **Files expected to change:** `reid/models/pcb.py`, `tests/test_pcb_model.py`, `plan.md` only.
- **Implementation tasks:** Six independent Conv/BN/ReLU modules, explicit historical initialization, ordered local outputs flattened only after reduction; preserve backbone and stripe components.
- **Validation/tests:** Exact structure; distinct parameter and buffer identities/storage; mutation isolation; deterministic initializer invocation/bounds and values; loaded-backbone protection; ordered outputs; train/eval BN arithmetic and running statistics; single-image evaluation; all-reduction and integrated-backbone gradients; prior PCB and baseline/checkpoint regressions.
- **Exit criteria:** Satisfied: six independent reductions and historical initialization verified, producing six ordered `[B,256]` features without classifiers or descriptor concatenation.
- **Status:** Completion `[x] IMPLEMENTED + VALIDATED` on 2026-10-05; review `[x] REVIEWED AND ACCEPTED` by the user on 2026-10-05. Phase 8 remains unstarted and unauthorized.
- **Implementation record:** Starting branch `main`, tracking `origin/main`, HEAD `4f1bbe9ebcce5f072b44dfd61dba2df1963feb7a`; working tree clean. Read the complete roadmap and authorization, verified Phase 6 acceptance, inspected the current PCB components/tests and the clean pinned Huang checkout at `1686e889eb01c28a54b633051418012e15d9c9f3`. No applicable AGENTS.md found in the repository or checked ancestors.

  Reference sources rechecked:
  - Huang `bpm/model/PCBModel.py:26–32` constructs a fresh Conv2d/BatchNorm2d/in-place ReLU sequence for each part; lines 60–63 apply the corresponding reduction before flattening.
  - [PyTorch v0.3.0 convolution source](https://raw.githubusercontent.com/pytorch/pytorch/v0.3.0/torch/nn/modules/conv.py), `_ConvNd.reset_parameters`: fan-in is input channels times kernel area; weight and bias are uniform within `±1/sqrt(fan_in)`. For this 1x1 reduction, fan-in is 2048.
  - [PyTorch v0.3.0 BatchNorm source](https://raw.githubusercontent.com/pytorch/pytorch/v0.3.0/torch/nn/modules/batchnorm.py), `_BatchNorm.__init__/reset_parameters`: eps 1e-5, momentum 0.1, affine enabled, running mean zero/variance one, scale Uniform[0,1], bias zero. No contradiction with the frozen contract.

  Exact files changed:
  - `reid/models/pcb.py`: added internal `PCBPartReductions` and math import; updated module description. Historical loading, `PCBBackbone`, and `PCBStripePool` implementations are byte-for-byte unchanged.
  - `tests/test_pcb_model.py`: retained all 36 prior PCB cases and added 16 Phase 7 cases; all ordinary tests remain independent of network/cache/reference checkout.
  - `plan.md`: this Phase 7 section only; text before Phase 7 and from Phase 8 onward remains byte-for-byte unchanged.

  Design and shape contract: `PCBPartReductions.local_conv_list` owns six separately constructed `nn.Sequential` modules. Each is `Conv2d(2048,256,1,bias=True)` -> `BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)` -> `ReLU(inplace=True)`. Each module receives its corresponding `[B,2048,1,1]` pooled stripe, retains four dimensions through the reduction, then flattens to `[B,256]`. Forward returns a tuple of six tensors in input/top-to-bottom order. The narrow interface accepts a tuple/list of exactly six positive-batch `[B,2048,1,1]` tensors with equal batch sizes; structural input validation precedes all reductions to avoid partial BN updates for malformed shapes/counts. Standard PyTorch device/dtype requirements and BN errors remain in force.

  Initialization is applied directly to each newly created Conv and BN, with no model-wide initializer and no backbone reference. Conv weights and biases explicitly use `nn.init.uniform_` with `±1/math.sqrt(2048)`; BN scales explicitly use Uniform[0,1], biases/running means zero, running variances one, and modern `num_batches_tracked` buffers zero. Modern constructor defaults are overwritten for these tensors only. This preserves historical distributions; exact whole-model random-number draw order relative to PyTorch 0.3 is not claimed.

  Standard internal usage:
  ```python
  backbone = PCBBackbone(pretrained=False)
  pool = PCBStripePool()
  reductions = PCBPartReductions()
  # In evaluation, call eval() on backbone and reductions, including for B=1.
  local_features = reductions(pool(backbone(images)))
  # [B,3,384,128] -> [B,2048,24,8] -> six [B,2048,1,1] -> six [B,256]
  ```
  This remains separate internal components, with no complete PCB public output or builder registration.

  Validation evidence:
  - Structure checks establish six Conv/BN/ReLU sequences with the exact dimensions/settings and no Linear/dropout/pooling modules. All 42 parameter/buffer tensors (seven per part) have distinct object and storage identities; all module objects are distinct. A controlled mutation of all parameters/buffers of part 3 changes its output while the other five parts' complete state and outputs remain exactly unchanged.
  - Deterministic initialization test poisons modern constructor defaults with 17, records every explicit uniform call and its exact bounds/target, and compares all resulting uniform tensors to independent generator draws with seed 42 at rtol=0/atol=0. BN bias/mean/variance/counter values are checked exactly. This tests explicit overrides rather than approximate sample statistics. A mocked historical backbone load followed by reduction construction preserves every backbone tensor exactly and shares no storage with reductions; actual historical-file compatibility remains covered by Phase 5's recorded controlled check.
  - Controlled part/channel routing verifies ordered outputs and ReLU before flattening, including negative input values. Train-mode BN results match manually computed batch means and biased variances; running means/variances match momentum-0.1 updates using unbiased variance. Distinct input distributions establish independent statistics for all six parts. Eval outputs match stored-statistic arithmetic, leave state unchanged, and single-item outputs agree with the corresponding row of larger-batch evaluation. Training B=1 retains PyTorch's expected pooled-BN error; no special fallback was added. The planned authoritative batch64/drop_last recipe remains unchanged.
  - Seed-7 train/eval synthetic backward tests verify non-None, finite gradients for all 24 reduction parameters, including every Conv bias and BN scale/bias, and finite nonzero gradients at every pooled input. Conv-bias gradients need not be nonzero in training because BN removes a channelwise shift. Single-image eval through actual uninitialized backbone -> pool -> reductions verifies six `[1,256]` outputs, all-parameter finite gradients in both components, and nonzero input/stem/layer4 gradients. No optimizer step, training run, or evaluation experiment occurred.

  Exact test commands from repository root:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -rs -p no:cacheprovider tests/test_pcb_model.py > /tmp/phase7-focused.log 2>&1
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -rs -p no:cacheprovider tests/test_pcb_model.py tests/test_model_forward.py tests/test_model_interface.py tests/test_resnet50_strong_baseline.py tests/test_checkpoint_reconstruction.py tests/test_checkpoint.py > /tmp/phase7-validation.log 2>&1
  git diff --check
  git diff --stat
  git status --short --branch
  ```
  Initial focused result: **51 passed, 1 failed in 15.32 s**. The initialization test used `2048 ** -0.5` for its expected bounds, whose last float64 digit differs from the historical `1/math.sqrt(2048)` expression; exact initializer-call comparison caught that difference. Corrected the test to use the recorded historical formula; production initialization was already correct. Final combined result: **109 passed, 1 skipped in 43.77 s; zero failures**: 52 PCB cases (36 retained + 16 new) and 57 passing baseline/interface/checkpoint regressions. One skip at `tests/test_model_forward.py:192` because CUDA is unavailable. Regression files cover ResNet50/BoT forward behavior, existing interface, strong baseline, generic reconstruction and checkpoint persistence. Environment rechecked: Python 3.12.3, torch 2.7.1+cpu, torchvision 0.22.1+cpu, CUDA false; no dependency changes. No full-repository/GPU-suite claim.

  Source retrieval issue: browser fetch of historical BatchNorm returned a cache miss; the same pinned raw URL was successfully retrieved/read with `curl --fail --location --connect-timeout 15 --max-time 45 https://raw.githubusercontent.com/pytorch/pytorch/v0.3.0/torch/nn/modules/batchnorm.py -o /tmp/phase7-historical-batchnorm.py`. Convolution source was successfully read through the browser. Temporary source/log files stay in `/tmp`, outside the repository. The existing broken filesystem sandbox required approved escalated commands; no code workaround introduced.

  Final scope checks: `git diff --check` passed; prior PCB implementations and roadmap text outside Phase 7 matched HEAD exactly. Pre-commit validation Git status on `main`: modified `plan.md`, `reid/models/pcb.py`, `tests/test_pcb_model.py` only; no untracked files. No commit or push had been performed at implementation handoff. The user subsequently reviewed and accepted Phase 7 and authorized its commit/push on 2026-10-05. Reference checkout remains clean.
- **Decisions/deviations:** No architecture/recipe deviation and no changes to generic infrastructure. Separate reduction ownership protects the accepted backbone and stripe interfaces. Explicit modern BN counters are compatibility state, not another normalization policy. Earlier plan records remain historical because only Phase 7 is updated.
- **Review notes:** Phase 7 reviewed and accepted by the user on 2026-10-05; commit/push authorized. STOP after publishing this step; Phase 8 requires separate authorization. No unresolved implementation blocker. Limitations: CPU-only validation, no whole-model historical RNG parity claim, and incomplete/unregistered PCB component. Classifiers, final descriptor/public output, loss integration, statistics, optimizer/configuration changes, training/evaluation experiments and Phase 8 are not implemented.
- **Next step:** Phase 8 — identity classifiers, only after Phase 7 review and separate explicit authorization.

### Phase 8 — Identity classifiers

- **Objective:** Add six independent source-identity classification heads.
- **Why this step exists:** PCB supervises each ordered local representation independently in a common source label space.
- **Prerequisites:** Satisfied. The user reviewed and accepted Phase 7 and explicitly authorized ONLY Phase 8 in attachment `7e74c2aa-9600-4769-9808-034d8394a38e/Pasted text.txt` on 2026-10-05. Phase 7 is recorded as implemented/validated and reviewed/accepted and is committed/pushed as `1f812654ff420601d923fa04e3ec2b67d0724e10`. Earlier authorization wording is historical; this dated record supersedes it for Phase 8 only.
- **Files to inspect:** Complete roadmap; current `reid/models/pcb.py` and component tests; pinned Huang `bpm/model/PCBModel.py`; source-class validation in common model/checkpoint construction; relevant regression tests.
- **Files expected to change:** `reid/models/pcb.py`, `tests/test_pcb_model.py`, `plan.md` only.
- **Implementation tasks:** Six biased `Linear(256,C)` heads, explicit Normal(0,0.001)/zero initialization, deterministic part association, positive source-class validation; preserve all earlier component behavior.
- **Validation/tests:** Multiple class counts; distinct modules/parameter identities/storage; mutation isolation; deterministic initializer checks; backbone/reduction state preservation; ordered association; manual six-CE backward through every head/reduction/backbone; valid train mode and single-image evaluation; prior PCB and baseline/checkpoint regressions.
- **Exit criteria:** Satisfied: six correctly initialized independent heads produce ordered `[B,C]` logits with verified gradient connectivity, while six `[B,256]` local features remain available separately.
- **Status:** Completion `[x] IMPLEMENTED + VALIDATED` on 2026-10-05; review `[x] REVIEWED AND ACCEPTED` by the user on 2026-10-05. Phase 9 remains unstarted and unauthorized.
- **Implementation record:** Starting branch `main`, tracking `origin/main`; HEAD `1f812654ff420601d923fa04e3ec2b67d0724e10`; working tree clean. Read the complete current roadmap and authorization, confirmed Phase 7 review acceptance, inspected current PCB components/tests and the clean Huang reference checkout at `1686e889eb01c28a54b633051418012e15d9c9f3`. No applicable AGENTS.md found in the repository or checked ancestors.

  Reference trace: pinned `bpm/model/PCBModel.py:34–40` creates one biased Linear per stripe, explicitly initializes its weight with Normal(std=0.001, default mean=0) and bias with zero, and stores the heads in `fc_list`. Lines 64–65 apply classifier i directly to local feature i. No shared/averaged/global classifier is used.

  Exact files changed:
  - `reid/models/pcb.py`: added internal `PCBIdentityClassifiers(num_classes)` and updated module description. Historical loading, backbone, stripes/pooling and reductions remain byte-for-byte unchanged.
  - `tests/test_pcb_model.py`: retained all 52 previous cases and added 23 Phase 8 cases; updated test-module description. Ordinary tests have no network/cache/reference-checkout dependency.
  - `plan.md`: this Phase 8 section only; text outside it remains byte-for-byte unchanged.

  Design: six separately constructed `nn.Linear(256,num_classes,bias=True)` modules in `fc_list`. A required positive Python integer supplies C once, so all six heads have identical source label dimensions but independent weights/biases. Invalid values `0,-1,None,True,False,3.0,2.5,"3"` fail with descriptive ValueError; this follows the strict positive-integer convention in Phase 4's metadata validation without importing or changing checkpoint code. No dataset count or target dataset is looked up or hardcoded. When builder integration is authorized in Phase 10, new training must supply source training identities through the common constructor flow, and reconstruction must use the persisted source C established by Phase 4. That wiring and an actual full PCB checkpoint round trip are not implemented/claimed here.

  Initialization is narrowly scoped to each new classifier: `nn.init.normal_(weight,mean=0,std=0.001)` and `nn.init.zeros_(bias)`. No initializer traverses or owns backbone/reduction modules. Forward accepts a tuple/list of exactly six `[B,256]` local tensors with positive equal batch sizes and returns a tuple of six `[B,C]` logits in the same order. Invalid count/rank/width/batch/empty/non-tensor inputs fail clearly. Standard PyTorch device/dtype requirements remain. No feature or logit averaging, normalization, concatenation, or public output dictionary is introduced.

  The existing reduction interface provides local features and the new consumer provides logits; both ordered collections remain available without a premature full-model wrapper:
  ```python
  # Given backbone, pool and reductions from Phases 5–7, in the desired mode:
  classifiers = PCBIdentityClassifiers(num_classes=source_num_classes)
  local_features = reductions(pool(backbone(images)))  # tuple: six [B,256]
  logits = classifiers(local_features)                # tuple: six [B,source_num_classes]
  ```
  Use train/eval on each component (or their containing module). The tests compose them in a test-only ModuleDict and exercise train with B=2 and eval with B=1. This is not builder registration or production training integration.

  Validation evidence:
  - Synthetic source counts C=1,3,17 verify all six exact weight/bias dimensions and logits shapes. Six module identities and all 12 trainable parameter identities/storage pointers are distinct; classifiers have no buffers. Classifier-only train/eval results match exactly for identical inputs, and input local features remain unchanged.
  - A controlled test assigns distinct per-head weights/biases and per-part features, then checks hand-computed logits exactly. Changing both weight and bias of head 3 changes only its output/state; the other five heads' state and logits remain bitwise equal. This proves ordered feature_i -> classifier_i association and behavioral isolation.
  - Deterministic initializer test poisons Linear constructor defaults with 17, spies on each explicit normal initializer's parameter/mean/std, and compares resulting weights against independent generator draws with seed 42 at rtol=0/atol=0; every bias is exactly zero. This does not rely on noisy empirical mean/std thresholds. Mocked historical backbone loading followed by reduction construction and then classifier construction preserves every earlier parameter/buffer exactly and establishes no shared storage with classifiers. No new historical-weight download was needed.
  - Seed-8 integrated tests use actual `PCBBackbone(pretrained=False)` -> stripe pool -> reductions -> classifiers on `[B,3,384,128]`; verify `[B,2048,24,8]`, six pooled `[B,2048,1,1]`, six local `[B,256]` and six logit `[B,5]` tensors. A test-only sum of six ordinary cross-entropies, with the same labels for all heads, produces non-None finite gradients for every classifier, reduction and backbone parameter. Every classifier weight/bias, reduction Conv weight and local feature has nonzero gradients; input/stem/layer4 gradients are finite and nonzero. Train-mode reduction BN counters advance once; eval counters remain unchanged, including valid single-image evaluation. Conv-bias gradients in train-mode reductions need not be nonzero because BN removes channelwise shifts. No optimizer steps or real training/evaluation experiments.

  Exact command from repository root:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -rs -p no:cacheprovider tests/test_pcb_model.py tests/test_model_forward.py tests/test_model_interface.py tests/test_resnet50_strong_baseline.py tests/test_checkpoint_reconstruction.py tests/test_checkpoint.py > /tmp/phase8-validation.log 2>&1
  git diff --check
  git diff --stat
  git status --short --branch
  ```
  Result: **132 passed, 1 skipped in 47.08 s; zero failures**. This comprises 75 PCB cases (52 retained + 23 new) and 57 passing baseline/interface/checkpoint regressions. One skip at `tests/test_model_forward.py:192`: CUDA unavailable. Coverage includes ResNet50/BoT forward behavior, existing interface, strong baseline, generic checkpoint reconstruction and checkpoint persistence. Environment rechecked: Python 3.12.3, torch 2.7.1+cpu, torchvision 0.22.1+cpu; CUDA false. No dependency changes; targeted CPU validation, not a full-repository/GPU-suite claim.

  Final checks: `git diff --check` passed; all prior PCB component implementations and roadmap text outside Phase 8 matched HEAD exactly. Pre-commit validation Git status on `main`: modified `plan.md`, `reid/models/pcb.py`, `tests/test_pcb_model.py` only; no untracked files. No commit or push had been performed at implementation handoff. The user subsequently reviewed and accepted Phase 8 and authorized its commit/push on 2026-10-05. Reference checkout remains clean. Temporary validation log stays at `/tmp/phase8-validation.log`.
- **Decisions/deviations:** No architecture/recipe or scope deviation. A narrow classifier consumer preserves existing local-feature ownership and avoids introducing a final model/output API early. No test failures or unresolved implementation blocker. Approved escalated execution was used for the existing filesystem sandbox issue; no repository workaround introduced.
- **Review notes:** Phase 8 reviewed and accepted by the user on 2026-10-05; commit/push authorized. STOP after publishing this step; Phase 9 requires separate authorization. CPU-only validation; full PCB remains unregistered, source-class builder/metadata wiring and full PCB reconstruction remain later gates. No 1536-D descriptor, public output dictionary, production multi-head CE/LossBundle changes, training-loop statistics, optimizer groups, configs, training/evaluation experiments or Phase 9 work implemented. ResNet50 and generic infrastructure are unchanged.
- **Next step:** Phase 9 — retrieval and public PCB outputs, only after Phase 8 review and separate explicit authorization.

### Phase 9 — Retrieval and public PCB outputs

- **Objective:** Complete the raw PCB retrieval descriptor, public model outputs and dimension metadata.
- **Why this step exists:** Evaluation must receive the selected local-feature descriptor without fabricated BNNeck features.
- **Prerequisites:** Satisfied. The user reviewed and accepted Phase 8 and explicitly authorized ONLY Phase 9 in attachment `7b0ec34a-fb48-44ce-93c5-49663453bd21/Pasted text.txt` on 2026-10-05. Phase 8 is recorded as implemented/validated and reviewed/accepted and is committed/pushed as `0fdb6cae80c14a9f0d9e9619766087c8910c155a`. Earlier authorization wording is historical; this dated record supersedes it for Phase 9 only.
- **Files to inspect:** Complete roadmap; current PCB components/tests; pinned Huang `PCBModel.py`, `train_pcb.py:ExtractFeature`, `TestSet.py`, `distance.py`; framework `reid/models/outputs.py`, `reid/engine/evaluator.py`, `reid/metrics/distance.py` and evaluator tests.
- **Files expected to change:** `reid/models/pcb.py`, `tests/test_pcb_model.py`, `plan.md` only.
- **Implementation tasks:** Compose existing components; concatenate six post-ReLU local features in spatial order; expose raw `emb`, six-logit tuple, `feat_raw=None`, `feat_bn=None`, `embedding_dim=1536`, `feat_dim=None` consistently in train/eval. Keep normalization external.
- **Validation/tests:** Exact sentinel descriptor blocks/order and lack of normalization; classifier-independent embeddings; actual train/single-image eval outputs; descriptor gradients to backbone/reductions; unchanged extraction/evaluator compatibility and one external global normalization; earlier PCB and relevant baseline/checkpoint/evaluator regressions.
- **Exit criteria:** Satisfied: public output contract and retrieval path verified independently of builder registration.
- **Status:** Completion `[x] IMPLEMENTED + VALIDATED` on 2026-10-05; review `[x] REVIEWED AND ACCEPTED` by the user on 2026-10-05. Phase 10 remains unstarted and unauthorized.
- **Implementation record:** Starting branch `main`, tracking `origin/main`; HEAD `0fdb6cae80c14a9f0d9e9619766087c8910c155a`; working tree clean. Read the complete current roadmap and authorization; confirmed Phase 8 acceptance; inspected the existing PCB components, framework output/evaluation interfaces, and clean Huang reference checkout at `1686e889eb01c28a54b633051418012e15d9c9f3`. No applicable AGENTS.md found in the repository or checked ancestors.

  Reference behavior rechecked: `bpm/model/PCBModel.py:60–65` retains the post-Conv/BN/ReLU local features separately from classifier logits. `script/experiment/train_pcb.py:250–277` (`ExtractFeature`) collects those local tensors and concatenates them along feature axis 1. `bpm/dataset/TestSet.py:122–129` stacks collected descriptors and optionally normalizes each complete row; line 188 uses Euclidean query/gallery distance. `bpm/utils/distance.py:7–10` uses norm plus float32 epsilon. Our existing evaluator uses global normalization with `1e-12` and Euclidean distance; the previously frozen numerical adaptation is preserved.

  Exact files changed:
  - `reid/models/pcb.py`: added `PCB(nn.Module)` composing `backbone`, `pool`, `reductions`, and `classifiers`; raw concatenation, output dictionary and dimension metadata; updated module description. All Phase 5–8 component implementations and historical weight loading remain byte-for-byte unchanged.
  - `tests/test_pcb_model.py`: retained all 75 prior cases and added eight Phase 9 cases, including narrow evaluator compatibility tests; updated module description. No production evaluator or output-helper edits.
  - `plan.md`: this Phase 9 section only; text outside it remains byte-for-byte unchanged.

  Direct construction and output:
  ```python
  model = PCB(num_classes=source_num_classes, pretrained=False)
  # pretrained=True and optional weights_path delegate to the verified backbone loader.
  outputs = model(images)
  # {
  #   "emb": Tensor[B,1536],
  #   "feat_raw": None,
  #   "feat_bn": None,
  #   "logits": (Tensor[B,C], ... six tensors ...),
  # }
  ```
  Positive source C is validated before constructing/loading the backbone. The existing six heads receive the same C; no target-dataset inference. Construction initializes only the existing components through their validated constructors, with no blanket initializer. A mocked historical-load composition test verifies every backbone tensor survives subsequent construction and source C reaches all six classifiers. The default non-pretrained path remains offline; future training builder wiring must opt into the frozen historical initialization.

  Forward computes `local_features = reductions(pool(backbone(images)))`, then `torch.cat(local_features, dim=1)`. Blocks `[0:256]`, `[256:512]`, `[512:768]`, `[768:1024]`, `[1024:1280]`, `[1280:1536]` correspond to parts 1–6, top-to-bottom. These are post-BN/ReLU features. The classifier branch consumes the same local features separately; embeddings neither depend on logits nor detach from autograd. No model-side per-part/global normalization, additional BNNeck, dropout, global branch, normalization/clamping fallback, or extra output field. `embedding_dim=1536` is retrieval width; `feat_dim=None` honestly declares no exposed legacy metric-loss feature. Both modes return exactly the same four keys with all six logits; only standard BN behavior changes. Single-image evaluation works; single-image pooled-BN training remains unsupported as established earlier.

  Validation evidence:
  - A test-only sentinel producer feeds the real public forward and real classifiers with parts filled by 1–6 and a second batch row 7–12. Exact comparison with repeated blocks proves order, 256-value block widths, correct concatenation axis and no per-part/global normalization. Norms exceed one by construction. Controlled zero local features produce a finite zero descriptor. Existing `ensure_output_dict` and `get_embedding` accept the dictionary directly.
  - For both modes, expected logits match each head applied to its own descriptor block. Mutating every classifier's weights/biases changes logits while the descriptor remains exactly equal, establishing the branch before classification.
  - Seed-9 actual-model tests on `[2,3,384,128]` in train mode and `[1,3,384,128]` in eval mode verify metadata, exact keys, None metric fields, six finite logits, finite `[B,1536]` embeddings, and exact equality to captured post-ReLU local concatenation. Child-module structure is exactly the four accepted components; no dropout/adaptive pooling added.
  - A descriptor-only squared-mean scalar backward in train mode yields non-None finite gradients for all backbone/reduction parameters, and finite nonzero gradients at input, backbone map, every pooled stripe/local feature, every reduction Conv weight and stem weight. Classifier gradients remain None for this descriptor-only objective, proving graph separation. Existing Phase 8 manual six-head CE tests still verify the supervision branch. No optimizer step or training experiment.
  - Seed-10 actual PCB single-image feature extraction through unchanged `extract_features` returns an array exactly equal to direct raw model `emb`, preserving ID/camera/name/mark metadata. The extractor switches to eval and applies no-grad externally; it does not normalize.
  - A controlled four-sample query/gallery fixture uses the real public PCB forward with sentinel local features and real classifiers. Unmodified `evaluate_reid` completes with finite core metrics for normalization both enabled and disabled. Delegating spies verify the collected raw `[4,1536]` matrix, exactly one axis-1 normalization when enabled (none when disabled), and Euclidean distance inputs exactly equal to `raw / (norm + 1e-12)` or raw respectively. Normalized ordinary nonzero rows have unit norm within rtol=1e-6. Actual ranking/distance functions run unchanged; this is synthetic compatibility validation, not a dataset result.

  Exact validation command from repository root:
  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -rs -p no:cacheprovider tests/test_pcb_model.py tests/test_model_forward.py tests/test_model_interface.py tests/test_resnet50_strong_baseline.py tests/test_checkpoint_reconstruction.py tests/test_checkpoint.py tests/test_evaluation_harness.py > /tmp/phase9-validation.log 2>&1
  git diff --check
  git diff --stat
  git status --short --branch
  ```
  Result: **146 passed, 1 skipped in 53.57 s; zero failures**. This comprises 83 PCB cases (75 retained + eight new), 57 passing baseline/interface/checkpoint cases, and six existing evaluation-harness cases. One skip at `tests/test_model_forward.py:192`: CUDA unavailable. Regression coverage preserves ResNet50/BoT forward behavior, model interfaces, strong baseline, generic reconstruction/checkpoint persistence and evaluator behavior. Environment rechecked: Python 3.12.3, torch 2.7.1+cpu, torchvision 0.22.1+cpu; CUDA false. No dependency changes. Targeted CPU validation only, not a full repository/GPU suite.

  Final checks: `git diff --check` passed; existing PCB component code and roadmap text outside Phase 9 matched HEAD exactly. Pre-commit validation Git status on `main`: modified `plan.md`, `reid/models/pcb.py`, `tests/test_pcb_model.py` only; no untracked files. No commit or push had been performed at implementation handoff. The user subsequently reviewed and accepted Phase 9 and authorized its commit/push on 2026-10-05. Reference checkout remains clean. Temporary log: `/tmp/phase9-validation.log`.
- **Decisions/deviations:** No architecture/recipe or scope deviation. Descriptor concatenation occurs in torch inside the model to preserve the training graph, while normalization stays in the common evaluator. No test failures or unresolved implementation blocker. Approved escalated execution was used for the existing filesystem sandbox issue; no repository workaround introduced.
- **Review notes:** Phase 9 reviewed and accepted by the user on 2026-10-05; commit/push authorized. STOP after publishing this step; Phase 10 requires separate authorization. CPU-only validation. Direct PCB construction/output is complete, but public builder/config-schema integration, generic optional metric-feature/multi-head handling, reconstruction metadata and full PCB checkpoint round trip remain later gates. No loss/training-loop/optimizer changes, presets, training/dataset experiments, cross-domain changes or Phase 10 work implemented. Baseline and evaluation production code remain unchanged.
- **Next step:** Phase 10 — generic model/output/dimension integration, only after Phase 9 review and separate explicit authorization.

### Phase 10 — Generic model/output/dimension integration

- **Objective:** Register PCB and reconcile generic contracts without changing baseline computations.
- **Why this step exists:** Builder and orchestration currently assume baseline head fields and positive metric width.
- **Prerequisites:** Phases 4–9 reviewed and explicit authorization.
- **Files to inspect:** `reid/models/build.py`, `outputs.py`, `reid/utils/config.py`, `config_schema.py`, `scripts/train.py`, `scripts/smoke_reid_pipeline.py`, plug-in protocol/tests.
- **Files expected to change:** Those generic contract files as needed; optional metadata-only addition to `reid/models/baseline.py`; `docs/model_plugin_protocol.md`; `tests/test_model_interface.py`, `test_model_plugin_contract.py`, `test_model_plugin_protocol_doc.py`, `test_config_schema.py`, `test_checkpoint.py`; `plan.md`.
- **Implementation tasks:** Dispatch before baseline-specific parsing; validate fixed PCB variant; add reusable logits validation; make metric width optional except where required; expose retrieval metadata without baseline numerical change; reconcile old documentation and tests; connect real PCB metadata reconstruction.
- **Validation/tests:** Build both models; malformed heads rejected generically; CE-only config admits None metric features/dimension; metric-enabled incompatible outputs fail clearly; actual PCB checkpoint round trip/no-download test from Phase 4; ResNet50 output regression. Full multi-head training awaits Phase 11.
- **Exit criteria:** Real PCB builds/reconstructs and generic contract validation is coherent; no fake metric features.
- **Status:** Completion `[x] IMPLEMENTED + VALIDATED` on 2026-10-05; review `[x] REVIEWED AND ACCEPTED` by the user on 2026-10-05. Phase 11 remains unstarted and requires separate authorization.
- **Implementation record:**

  Authorization: user attachment `/home/filsduvent/.codex/attachments/221903fe-50f1-4882-96fe-cb51bbdcb94a/Pasted text.txt`, explicitly authorizing Phase 10 only and accepting Phase 9. Starting branch `main`, HEAD `661cdbe04849790e46142d86cf6fc87db3cffff9` (`Add PCB retrieval descriptor and public model outputs`), clean working tree. Phase 9 implementation/validation and user review prerequisite confirmed. Implementation-session commit/push was deferred; the user subsequently reviewed and accepted Phase 10 and explicitly authorized committing and pushing on 2026-10-05.

  Inspected: root `plan.md`; `reid/models/pcb.py`, `baseline.py`, `build.py`, `outputs.py`; `reid/utils/config.py`, `config_schema.py`, `checkpoint.py`; `reid/losses/build.py`; `reid/engine/evaluator.py`; `scripts/train.py`, `smoke_reid_pipeline.py`, `evaluate.py`, `evaluate_cross_domain.py`; `docs/model_plugin_protocol.md`; and the model/config/plugin/checkpoint/evaluation/loss/orchestration/smoke regression files listed in the commands below. Reinspected the Phase 4 reconstruction metadata/helper and its surrogate tests before adding real PCB coverage.

  Changed files: `reid/models/build.py`, `baseline.py`, `outputs.py`; `reid/utils/config.py`; `reid/losses/build.py`; `scripts/train.py`, `scripts/smoke_reid_pipeline.py`; `docs/model_plugin_protocol.md`; `tests/test_model_interface.py`, `test_model_plugin_contract.py`, `test_model_plugin_protocol_doc.py`, `test_config_schema.py`, `test_checkpoint_reconstruction.py`, `test_pcb_model.py`, `test_smoke_reid_pipeline.py`; and only this Phase 10 section of `plan.md`.

  Builder dispatch now precedes baseline-only parsing. PCB accepts `name`, optional fixed `variant=independent_part_reduction`, boolean `pretrained` (default true), and optional historical `weights_path`; unknown model fields and alternative variants are rejected. Architecture internals remain fixed in the existing PCB implementation. The validator materializes the default variant in canonical `cfg.model`, keeping the existing Phase 4 metadata and saved config consistent. Dynamic source C constructs six independent classifiers; retrieval width remains 1536 and metric width remains None. `initialize_pretrained=False` disables historical initialization even when the saved config requests it, including a weights path. No second checkpoint schema was introduced.

  Generic `validate_logits` accepts None, one floating [B,C] tensor, or any nonempty flat tuple/list of matching floating [B,C] tensors. It validates rank, positive dimensions, matching batch/class shape, dtype/device, and an optional expected batch size; `ensure_output_dict` supplies embedding batch size. Heads/container identity and ordering are preserved, with no aggregation, copies or detach. Tests include 1/2/3/6 heads and malformed empty/nested/mixed/rank/shape/dtype/device cases. Label-batch validation can use the same optional argument when loss integration is authorized; CE arithmetic is unchanged here.

  Generic training/smoke orchestration permits absent/None metric width for ID-only models. Center still requires a positive integer metric width, Triplet still rejects absent requested runtime metric features, and PCB configuration rejects enabled Triplet or Center with an actionable error. No embedding-to-metric substitution or silent loss disabling occurs. The sole criterion-construction change makes the unused baseline head lookup optional; the entire LossBundle implementation is unchanged. Baseline gained only `embedding_dim = feat_dim` metadata; its construction/state/forward computations are preserved.

  Smoke construction now uses the common builder's `initialize_pretrained` switch instead of injecting `model.backbone.pretrained=false` into every architecture's config. Defaults still avoid initialization/downloads; `--use-config-pretrained` honors the selected model's configuration. To request pretraining through smoke `--opts`, that explicit flag is now also needed. Four orchestration cases verify both flag values for both architectures without entering the deferred multi-head loss step.

  Real PCB tests exercise public builder identity, six dynamic-C heads, dictionary output transport and common feature extraction of the 1536-D embedding. A synthetic/untrained real PCB checkpoint round trip uses existing save/reconstruct helpers and schema: exact state equality and exact deterministic CPU evaluation embedding/logit equality; six source-C=3 heads preserved against target C=999; no top-level classifier-name guessing; historical initialization and download hooks forbidden. Missing/unexpected/wrong-shape state, wrong embedding metadata and unsupported variant are rejected. Historical baseline fallback and prior checkpoint tests pass. This is not validation of an authoritative trained PCB checkpoint.

  Plugin documentation now describes optional model-provided metric features, separate retrieval/metric dimensions, ordered multi-head logits, generic validation ownership and existing reconstruction schema. It explicitly records that current CE still consumes a single tensor and that full multi-head training awaits Phase 11. No PCB-specific evaluator, ranking change or parallel pipeline was introduced.

  Environment: `/home/filsduvent/environments/Reid/bin/python`, Python 3.12.3, torch 2.7.1+cpu, torchvision 0.22.1+cpu; CUDA unavailable. Every pytest command below used the exact prefix `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/filsduvent/environments/Reid/bin/python -B -m pytest -q -rs -p no:cacheprovider`:

  1. `tests/test_model_interface.py tests/test_config_schema.py tests/test_model_plugin_contract.py tests/test_pcb_model.py -k 'generic or malformed or dispatch or exact_state or pcb_config or unsupported_architecture or metric_consumers or id_only_plugin or output_transport' > /tmp/phase10-focused.log 2>&1` — **50 passed, 0 failed, 0 skipped, 103 deselected**, 9.98 s.
  2. `tests/test_checkpoint_reconstruction.py -k real_pcb tests/test_model_plugin_protocol_doc.py > /tmp/phase10-reconstruction-focused.log 2>&1` — **6 passed, 0 failed, 0 skipped, 31 deselected**, 11.41 s. The selector excludes the documentation test; it is covered in the next command.
  3. `tests/test_pcb_model.py tests/test_model_interface.py tests/test_model_forward.py tests/test_resnet50_strong_baseline.py tests/test_model_plugin_contract.py tests/test_model_plugin_protocol_doc.py tests/test_config_schema.py tests/test_checkpoint.py tests/test_checkpoint_reconstruction.py tests/test_evaluation_harness.py tests/test_loss_interface.py tests/test_reid_loss_modes.py tests/test_train_orchestration.py tests/test_smoke_reid_pipeline.py tests/test_train_loop_optim.py > /tmp/phase10-regression.log 2>&1` — **252 passed, 0 failed, 4 skipped**, 74.98 s. Skips: one CUDA model-forward test and three CUDA train-loop tests because CUDA is unavailable. Covers baseline models/outputs/losses, historical checkpoint fallback, evaluator and common orchestration. This is the affected Phase 10 regression suite, not the Phase 17 full offline gate.
  4. After the final smoke override correction: `tests/test_smoke_reid_pipeline.py` — **10 passed, 0 failed, 0 skipped**, 4.10 s, including the four new initialization-control cases. Other tested production files were unchanged after command 3.

  Failures/resolutions: no test failures. Final inspection found the smoke default injecting a baseline-only field, resolved with the existing generic builder initialization switch and focused regression above. Shell execution required approved escalation because the current sandbox fails with `mountinfo path is not absolute`; this is an environment issue. Final `git diff --check` and byte-for-byte comparison of all plan content outside Phase 10 pass. Implementation-session Git state: 16 modified tracked files listed above, no untracked files; changes were uncommitted and unpushed pending review.

- **Decisions/deviations:** No scientific contract deviation or new ablation surface. Narrow additional files: `reid/losses/build.py` only for optional head configuration at construction (no loss arithmetic); `tests/test_smoke_reid_pipeline.py` for generic initialization control. Existing structural schema and checkpoint implementation required no changes. No future MGN feature abstraction beyond the approved generic output contract.
- **Review notes:** Phase 10 reviewed and accepted by the user on 2026-10-05; commit/push authorized. STOP after publishing this step. CPU synthetic validation only; no authoritative PCB training/checkpoint, GPU verification, multi-head CE/statistics, differential optimizer groups, scheduler changes, presets, dataset work, cross-domain preprocessing or full pipeline training. Full PCB training/smoke loss execution remains unsupported until Phase 11; this is the intended phase boundary. Phase 11 requires separate explicit authorization; no next-phase work is authorized.
- **Next step:** Phase 11 — Generic multi-head identity loss, only after Phase 10 review and separate explicit authorization.

### Phase 11 — Generic multi-head identity loss

- **Objective:** Support independent CE aggregation while retaining single-head behavior.
- **Why this step exists:** Existing ID loss receives one tensor.
- **Prerequisites:** Phase 10 reviewed and explicit authorization.
- **Files to inspect:** `reid/losses/build.py`, `id.py`, `tests/test_loss_interface.py`, `test_reid_loss_modes.py`.
- **Files expected to change:** Loss/config validation as needed; those loss tests; `plan.md`.
- **Implementation tasks:** Tensor direct path unchanged; sequence per-head CE then configured sum/mean; default sum; PCB smoothing zero/weight one; no logits averaging or metric feature requirement for CE-only.
- **Validation/tests:** Value AND gradient equality to explicit six-CE sum; generic two/three-head sum/mean; unchanged baseline smoothing/weights; malformed sequences and missing logits fail; disabled metric features accepted; enabled metric errors preserved.
- **Exit criteria:** Objective and every head's gradients match the reference mathematical loss.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Generic mean support is infrastructure coverage, not a PCB experiment.
- **Review notes:** STOP.
- **Next step:** Phase 12, separately authorized.

### Phase 12 — Multi-head training diagnostics

- **Objective:** Support useful diagnostics without affecting optimization.
- **Why this step exists:** Tensor-only `argmax` fails on PCB outputs.
- **Prerequisites:** Phase 11 reviewed and explicit authorization.
- **Files to inspect:** `reid/engine/train_loop.py`, `tests/test_train_loop_optim.py`, `test_model_plugin_contract.py`.
- **Files expected to change:** Training-loop diagnostic handling and those tests; `plan.md`.
- **Implementation tasks:** Preserve single-head `acc/id`; multi-head `acc/id_mean_heads`; retain loss aggregates; no architecture-name checks/per-head logs.
- **Validation/tests:** Hand-computed head accuracies including disagreeing heads; unchanged single-head values/tags; identical gradients/updates with diagnostics; synthetic generic multi-head loop.
- **Exit criteria:** Logging is correct and optimization invariant.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** LR label change belongs to Phase 13 with prefix-group metadata.
- **Review notes:** STOP.
- **Next step:** Phase 13, separately authorized.

### Phase 13 — Differential learning rates and group logging

- **Objective:** Add reusable prefix multipliers and truthful group LR diagnostics.
- **Why this step exists:** Reference backbone/new-layer rates differ by tenfold.
- **Prerequisites:** Phase 12 reviewed and explicit authorization.
- **Files to inspect:** `reid/optim/build.py`, `reid/engine/train_loop.py`, `tests/test_optim_build.py`, `test_train_loop_optim.py`.
- **Files expected to change:** Those files, config validation if necessary, `plan.md`.
- **Implementation tasks:** Implement frozen prefix schema, reject overlaps/unmatched rules, compose bias multipliers, preserve decay and no-rule behavior; add accurate group labels without falsely naming head LR as bias LR.
- **Validation/tests:** Every trainable parameter covered once by identity; frozen parameters excluded; exact LR/decay for weights/biases/BN; ambiguous/unmatched rules fail; baseline no-rule optimizer state/group order unchanged; no PCB architecture check.
- **Exit criteria:** PCB rates 0.01/0.1 with correct all-parameter weight decay; diagnostics truthful.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Do not automatically exempt biases or BN from decay.
- **Review notes:** STOP.
- **Next step:** Phase 14, separately authorized.

### Phase 14 — Verify exact 120-epoch LR trace

- **Objective:** Prove the epoch-41 boundary under iteration stepping.
- **Why this step exists:** Schedule names alone do not establish actual optimizer rates.
- **Prerequisites:** Phase 13 reviewed and explicit authorization.
- **Files to inspect:** `reid/optim/build.py`, `lr_scheduler.py`, train-loop step order, `tests/test_optim_build.py`.
- **Files expected to change:** Scheduler tests and `plan.md`; scheduler production changes are not expected.
- **Implementation tasks:** Test frozen no-warmup `[40]` recipe on fixed synthetic loader lengths, including resumed optimizer/scheduler state.
- **Validation/tests:** Rates actually used at updates 1, `40S`, `40S+1`, `120S`; all groups scale once and retain ratio; no extra decay; constructor and restored-state boundary checks.
- **Exit criteria:** Required trace established with exact commands/results; no full training needed.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** If behavior cannot express the frozen trace, mark BLOCKED and report; do not move/add milestones.
- **Review notes:** STOP.
- **Next step:** Phase 15, separately authorized.

### Phase 15 — Source preprocessing in common evaluation

- **Objective:** Make within/cross-domain reconstruction and preprocessing faithful to the source model.
- **Why this step exists:** Current cross-domain merge can replace PCB resolution with target baseline resolution.
- **Prerequisites:** Phases 10–14 reviewed and explicit authorization.
- **Files to inspect:** `scripts/evaluate.py`, `scripts/evaluate_cross_domain.py`, `reid/utils/checkpoint.py`, `tests/test_evaluation_harness.py`, `test_experiment_matrix.py`.
- **Files expected to change:** Common evaluation preparation/merge helpers and tests; proposed `tests/test_cross_domain_config.py`; `plan.md`. No evaluator or ranking implementation changes expected.
- **Implementation tasks:** Preserve source model/descriptor/preprocessing; substitute target dataset protocol/root; handle execution overrides intentionally; share reconstruction; fail incompatible standalone config clearly rather than silently evaluating different settings.
- **Validation/tests:** Source PCB plus target 256×128 baseline config still yields 384×128/source mean/std/source C; target parser/root/split retained; config copies unmutated; source/target labels cannot trigger adaptation; legacy baseline paths preserved.
- **Exit criteria:** Source preprocessing ownership demonstrated through common entry points.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Target runtime device/batch/workers may differ without changing learned-model preprocessing.
- **Review notes:** STOP.
- **Next step:** Phase 16, separately authorized.

### Phase 16 — PCB presets and dataset preflight

- **Objective:** Express the single project recipe for four authoritative benchmark loaders.
- **Why this step exists:** Configuration must encode all deviations and use real, verified dataset membership.
- **Prerequisites:** Phase 15 reviewed and explicit authorization; available dataset roots/partition evidence.
- **Files to inspect:** Existing baseline YAMLs, `reid/data/build.py`, transforms, dataset protocol/tests, common config validation.
- **Files expected to change:** Proposed four `configs/pcb/*.yaml`, preset/config tests, `plan.md`; no parser/sampler changes expected.
- **Implementation tasks:** Full existing schema plus model-specific settings; random64, CE-only, explicit disabled Triplet section, reference mean/std, initialization identity, 120 epochs, eval_interval 10, save_best true/save_last true, selection metric mAP; unique output dirs. Verify actual roots/partitions/camera/label conventions and dataset counts; document CUHK03 and MSMT17 provenance.
- **Validation/tests:** All four YAMLs load/validate; builder/loss/optimizer agree; read-only loader batches show shapes/dtypes/labels; random drop-last and fixed length; train/eval transforms correct; partition membership checks, not directory-name guesses; no full training.
- **Exit criteria:** Four usable presets and dataset evidence. Missing data/partition provenance blocks the affected preflight and authoritative run; placeholders are not validated configurations.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Same selected recipe on MSMT17 is a project extension. Do not reorganize baseline YAMLs.
- **Review notes:** STOP; no new ablations or silent recipe changes.
- **Next step:** Phase 17, separately authorized.

### Phase 17 — ResNet50 and shared-framework regression gate

- **Objective:** Establish that required generic extensions preserve baseline behavior.
- **Why this step exists:** Shared infrastructure changes must not invalidate qualification comparisons.
- **Prerequisites:** Phases 4–16 reviewed and explicit authorization.
- **Files to inspect:** Full source/test index, actual diff against pre-PCB HEAD, legacy checkpoint and presets.
- **Files expected to change:** `plan.md` and validation artifacts in approved output location; fixes require a clearly bounded recorded follow-up, not unrelated refactoring.
- **Implementation tasks:** Run focused changed-component tests and then relevant full offline suite; inspect production diff for forbidden scope creep; reproduce historical checkpoint load/forward where available.
- **Validation/tests:** Model interfaces, losses, optimizer/bias/schedule, train orchestration, checkpoint fallback, config, dataset/sampler/transforms, evaluator/ranking, artifact schema, plugin/documentation tests; fixed-state baseline output/loss/gradient comparisons. Record skips and reasons; no unplanned external downloads.
- **Exit criteria:** Required offline tests pass, or concrete blockers recorded; no new baseline metric claims from synthetic checks.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Documentation assertions may be updated for approved contract evolution; numerical regressions may not be dismissed as documentation changes.
- **Review notes:** STOP.
- **Next step:** Phase 18, separately authorized.

### Phase 18 — Synthetic end-to-end integration validation

- **Objective:** Verify the complete PCB path before consuming real training resources.
- **Why this step exists:** Component tests do not prove builder→loss→optimizer→checkpoint→evaluation integration.
- **Prerequisites:** Phase 17 passed/reviewed and explicit authorization.
- **Files to inspect:** Common train/eval/smoke entry points, PCB presets/tests, checkpoint and artifact writers.
- **Files expected to change:** Integration tests if needed; `plan.md`; isolated temporary validation artifacts, not authoritative experiment dirs.
- **Implementation tasks:** Exercise actual PCB forward/backward/optimizer steps on synthetic inputs, save/reconstruct strictly, run evaluator on a valid synthetic query/gallery fixture, verify periodic strict-best mAP selection and separate latest/final checkpoint behavior through common orchestration.
- **Validation/tests:** Finite loss/gradients across backbone and all six heads; correct shapes/dimensions; restored eval embeddings agree within stated tolerance; no pretrained download on reload; all five core metrics and expected artifact identity; source/target class isolation; normalization once; a controlled periodic score sequence with an early maximum, equal-score tie and later lower score proves best retention while last advances, and a later strict improvement proves replacement. Verify selected-best versus final-epoch artifact identity separately.
- **Exit criteria:** Complete synthetic pipeline passes with transparent scope and tolerances.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Synthetic optimizer steps are implementation diagnostics, not experiments or benchmark results.
- **Review notes:** STOP.
- **Next step:** Phase 19, separately authorized.

### Phase 19 — Bounded real-data GPU smoke and feasibility

- **Objective:** Verify real preprocessing, finite training, checkpoint reload, and batch-64 hardware feasibility.
- **Why this step exists:** Current local CPU environment cannot establish GPU memory/performance or data-path viability.
- **Prerequisites:** Phase 18 reviewed; explicit bounded-run authorization; identified GPU host/environment, verified data, fixed smoke batch/step limits and separate output dir.
- **Files to inspect:** Presets, real dataset summaries, GPU/environment details, common training/smoke commands.
- **Files expected to change:** `plan.md`; isolated diagnostic logs/checkpoints/config snapshots. Production source changes are not assumed.
- **Implementation tasks:** Run a short predeclared real-data check; optional tiny-subset overfit diagnostic only if needed/authorized; collect memory, batch/step time, finite losses, branch gradients and reload evidence.
- **Validation/tests:** Batch64 FP32 feasibility, all-head gradients, no NaNs, source labels contiguous, complete runtime/preprocessing provenance, checkpoint evaluation works. Extrapolate resource needs with uncertainty; do not demand reference accuracy from a smoke.
- **Exit criteria:** Feasibility documented or blocked with concrete hardware/data issue; no automatic full run.
- **Status:** `[ ] NOT STARTED`; authorization absent; GPU resource not yet established.
- **Implementation record:** None.
- **Decisions/deviations:** Training-batch reduction, AMP, gradient accumulation, LR changes require explicit project decision; stop on OOM rather than silently changing recipe.
- **Review notes:** STOP.
- **Next step:** Phase 20, separately authorized.

### Phase 20 — Authoritative-run readiness and experiment freeze

- **Objective:** Prepare a concrete reviewable four-source run manifest before long training.
- **Why this step exists:** Dataset, config, hardware, checkpoint policy, and artifact identity must be fixed before results exist.
- **Prerequisites:** Phase 19 passed/reviewed; all preceding gates complete; explicit authorization to prepare manifest only.
- **Files to inspect:** Four presets, dataset evidence, repository diff/status, environment, artifact helpers, `scripts/train.py` final/resume paths.
- **Files expected to change:** `plan.md`; proposed `docs/pcb_experiment_protocol.md` and manifest/artifacts in a user-reviewed location.
- **Implementation tasks:** Freeze per-source resolved config/hash, code commit plus dirty-state evidence, reference weight identity/hash, partition identities/counts/hashes where available, hardware/software versions, seed, fresh output dir, command, 120-epoch schedule, common 10-epoch mAP selection policy, selected-best/latest-state provenance fields and resources. Present exact run command and destination for each source. Resolve missing paths before claiming readiness.
- **Validation/tests:** No stale outputs; source preprocessing and loader length verified; LR trace for actual S; periodic strict-improvement selection, tie retention, best/last epoch and hash recording, and distinct selected-best/final-state artifact naming verified; training root/test root correct; source/target identities never mixed. Review resume limitations and verify historical best-score/checkpoint preservation before relying on resumed authoritative training.
- **Exit criteria:** Concrete manifest ready for approval; no unresolved methodological/hardware/data issue. Preparing a manifest does not authorize any long run.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Record reference60/project120 and all adaptations. If interrupted later, preserve evidence and obtain a bounded resume decision; do not claim bitwise continuous equivalence.
- **Review notes:** STOP; request source-specific run authorization only after manifest exists.
- **Next step:** Phase 21, first explicitly authorized source run.

### Phase 21 — Train four source models, one reviewed run at a time

- **Objective:** Train our selected PCB model on Market1501, Duke, CUHK03, and MSMT17.
- **Why this step exists:** Architecture comparison requires our own source-trained models under the frozen project recipe.
- **Prerequisites:** Phase 20 reviewed; explicit authorization identifying one source, exact command/host/output/resources.
- **Files to inspect:** That source's frozen manifest/preset and latest Git/environment/data state.
- **Files expected to change:** That source's approved experiment artifacts and `plan.md`; no source-code/config edits during a run.
- **Implementation tasks:** Treat 21A Market1501, 21B Duke, 21C CUHK03, 21D MSMT17 as separate authorization/review units. Execute only the authorized unit for 120 epochs through common train.py; monitor failures; preserve logs/checkpoints; record epoch-120 ckpt_last hash and selected ckpt_best epoch/hash/mAP, together with all periodic evaluation records. Do not automatically start the next source.
- **Validation/tests:** Correct source C, batch64/FP32, exact LR boundary, 120 completed epochs, no nonfinite failures; ckpt_last.pth exists at epoch 120 with final metadata/config/optimizer/scheduler and recorded SHA256; all 12 periodic evaluations occurred at epochs 10,20,…,120; ckpt_best.pth exists and matches the highest recorded mAP under strict-improvement/tie-retention semantics; record best epoch, best selection mAP and best SHA256. Do not assume best epoch is 120. Verify separately labeled final-epoch and selected-best metrics. The training entry point's automatic selected-best evaluation is later independently checked in Phase 22.
- **Exit criteria:** Each source unit has validated epoch-120 last state, a periodic-mAP-selected best checkpoint, complete selection provenance and a reviewed record; parent phase completes only when all four units pass.
- **Status:** `[ ] NOT STARTED`; 21A `[ ]`, 21B `[ ]`, 21C `[ ]`, 21D `[ ]`; no run authorized.
- **Implementation record:** None. Maintain separate command/date/host/duration/artifacts/hash/result/error records per source unit.
- **Decisions/deviations:** No automatic retries that change recipe; interruptions/OOM/nonfinite loss trigger explicit recorded diagnosis. Never substitute smoke or external pretrained checkpoints.
- **Review notes:** STOP after each source unit; all four are not bundled by this roadmap.
- **Next step:** Next specifically authorized source unit, or Phase 22 after all four are reviewed.

### Phase 22 — Within-domain selected-best checkpoint evaluation

- **Objective:** Independently verify four authoritative diagonal results from each source's selected ckpt_best.pth.
- **Why this step exists:** The reported metrics must correspond exactly to the authoritative weights and common protocol.
- **Prerequisites:** Four Phase 21 source units validated/reviewed; explicit evaluation authorization.
- **Files to inspect:** Selected-best checkpoint hashes/epochs/configs, last-state provenance and common `scripts/evaluate.py`/artifact behavior.
- **Files expected to change:** Separate evaluation result directories and `plan.md`; do not overwrite training provenance/config snapshots.
- **Implementation tasks:** Evaluate each source ckpt_best.pth on its own frozen query/gallery partition; no rerank/flip; record source dataset, best epoch, best checkpoint SHA256, mAP, mINP, Rank-1, Rank-5 and Rank-10; compare with the training entry point's selected-best evaluation using justified tolerances. Retain epoch-120 ckpt_last.pth for training-state/provenance analysis; it is not automatically the selected model.
- **Validation/tests:** Exactly four diagonal records; Rank1/5/10, mAP, mINP finite and valid; source normalization/resolution; same-camera/junk policy common; no stale historical mINP or checkpoint from another run; selected best hash/epoch match Phase 21 provenance and highest periodic mAP. Preserve disagreements for diagnosis.
- **Exit criteria:** Four traceable diagonal results agree with the intended selected-best source models/protocol.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None; record one row/path/hash per source.
- **Decisions/deviations:** Reference README scores are context, not required pass thresholds.
- **Review notes:** STOP.
- **Next step:** Phase 23, separately authorized.

### Phase 23 — Cross-domain evaluation matrix

- **Objective:** Evaluate all 12 ordered off-diagonal source→target pairs without adaptation.
- **Why this step exists:** Cross-domain generalization is central to the project objective.
- **Prerequisites:** Phase 22 reviewed; four fixed source ckpt_best.pth hashes matching Phase 22 diagonals, all target partitions validated; explicit evaluation scope authorization.
- **Files to inspect:** `scripts/evaluate_cross_domain.py`, frozen source manifests, target dataset presets, existing matrix record builder.
- **Files expected to change:** Unique source→target result dirs/JSON and `plan.md`; no model training or checkpoint modification.
- **Implementation tasks:** Market→Duke/CUHK03/MSMT17; Duke→Market/CUHK03/MSMT17; CUHK03→Market/Duke/MSMT17; MSMT17→Market/Duke/CUHK03. Use the SAME source ckpt_best.pth selected by that source's within-domain periodic mAP procedure for every target, including the Phase 22 diagonal. Preserve each source's architecture/preprocessing/C and identical selected-best SHA256 across the entire row; use target query/gallery protocol only. For example, the Market selected-best hash is identical in Market→Market, Market→Duke, Market→CUHK03 and Market→MSMT17.
- **Validation/tests:** Exactly 12 distinct pairs; no missing/duplicate cell; correct source/target names and selected-best hashes/epochs; row hashes equal the Phase 22 diagonal hash and Phase 21 selected-best provenance; no target-label adaptation; 384×128 maintained; all core metrics; matrix records retain descriptor/variant/config provenance; reranking disabled.
- **Exit criteria:** Twelve traceable off-diagonal results; combine with four diagonal results to obtain all 16 cells.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None; maintain per-pair status/command/output/checkpoint hash and errors.
- **Decisions/deviations:** Do not select a different source epoch for a favorable target result.
- **Review notes:** STOP after the explicitly authorized evaluation scope; do not proceed to selection automatically.
- **Next step:** Phase 24, separately authorized.

### Phase 24 — Matrix audit and architecture comparison

- **Objective:** Produce an evidence-backed PCB matrix and a valid comparison with available baseline results.
- **Why this step exists:** Completed files alone do not establish comparable experimental provenance.
- **Prerequisites:** Phase 23 reviewed and all 16 PCB cells available; explicit analysis authorization.
- **Files to inspect:** `reid/utils/experiment_matrix.py`, `scripts/aggregate_results.py`, `scripts/report_model_selection.py`, `tests/test_experiment_matrix.py`, baseline registry/artifacts.
- **Files expected to change:** Proposed PCB results documentation/machine-readable summaries, `plan.md`; generic aggregation changes only if a verified incompatibility is separately scoped.
- **Implementation tasks:** Audit cell uniqueness, selected-best checkpoint consistency across every complete row, Phase 21/22 best epoch/mAP/hash provenance, separately retained epoch-120 last-state hashes, metric units/protocol, row mapping, missing evidence and summary arithmetic; report within-domain and off-diagonal summaries separately; disclose project/reference differences and common test-split checkpoint-selection limitation affecting ResNet50, PCB and future architectures.
- **Validation/tests:** All16 coverage and 12 cross-domain count; independent recomputation of summary arithmetic; no duplicate overwrite ambiguity; no fabricated missing ResNet50 cells. Existing matrix ranking compares architectures and must not be mistaken for ranking source-trained models within PCB.
- **Exit criteria:** Reproducible PCB matrix/report, with baseline comparison limited to verified comparable evidence; remaining evidence gaps explicit.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** Missing baseline artifacts block claims requiring them, not preservation of complete PCB results. No automatic MGN/TransReID integration.
- **Review notes:** STOP.
- **Next step:** Phase 25, separately authorized.

### Phase 25 — Model-selection review and offline handoff

- **Objective:** Deliver validated PCB evidence for later model-selection/deployment decisions.
- **Why this step exists:** Periodic checkpoint selection within each source run, later comparison/selection among source-trained models, and runtime qualification are distinct decisions.
- **Prerequisites:** Phase 24 reviewed; explicit analysis/handoff authorization; selection rule approved before applying it.
- **Files to inspect:** Completed matrix/report, source checkpoint manifest, existing model-selection methodology and artifact gaps.
- **Files expected to change:** Handoff/selection documentation and `plan.md`; no runtime adapters, export code, or deployment configuration.
- **Implementation tasks:** Present four selected ckpt_best.pth identities and their within/cross-domain evidence, with dataset, training epochs=120, evaluation interval=10, metric=mAP, best epoch, best selection mAP, best SHA256, last epoch=120 and last SHA256; apply only the reviewed selection unit/rule; record result/ties/limitations. If current tooling ranks the wrong unit, document and obtain a bounded generic reporting task rather than silently changing methodology.
- **Validation/tests:** Every claimed score links to a recorded matrix cell; selected-best weights hash/epoch verified against each source row; epoch-120 last state is retained separately; no descriptor-size-to-speed extrapolation; methodological changes documented; unresolved evidence gaps carried forward.
- **Exit criteria:** Offline PCB implementation/validation/training/evaluation deliverables reviewed, selection recorded or explicitly deferred with reason; later runtime work remains separately authorized.
- **Status:** `[ ] NOT STARTED`; authorization absent.
- **Implementation record:** None.
- **Decisions/deviations:** No deployment, ONNX export, robotic runtime integration, MGN, or TransReID work under this roadmap's execution authorization.
- **Review notes:** STOP and hand off to the user.
- **Next step:** None automatically; future work requires a new explicit scope.

## Record template and document validation

For each future implemented unit, replace “None” with: authorization and date; starting branch/HEAD/status; changed files; implementation rationale; exact test commands/interpreter/environment; pass/fail/skip counts; relevant assertions/numerical tolerances; artifact paths and hashes; failures and resolution; deviations from contract and approving decision; remaining limits; completion/review status; proposed next step (not authorized by implication).

Document-creation validation scope: only `plan.md` added; verify required phase fields, phases 0–25, protected 120-epoch/epoch-41 policies and common 10-epoch mAP checkpoint selection, existing-path references (distinguish proposed files), status/authorization consistency, and `git diff --check`/whitespace. No implementation tests or phases 4+ are executed merely to validate this Markdown roadmap.

Creation record: 2026-10-02 — read both supplied attachments, inspected Git/environment/reference state and relevant source/test inventory, consolidated Phase 3 with the superseding project-duration decision, and created only this root roadmap. Completion of document validation is reported in the creation-session response; future sessions must inspect repository state rather than assume it remains unchanged.

Validation record (2026-10-02): The in-memory structural check passed for phases 0–25, all 13 required fields per phase, unstarted/unauthorized phases 4+, balanced fences, policy anchors, whitespace, and 35 source/test/document references (explicitly proposed files excluded from existence checks). `git diff --check` passed. Git status and untracked-file inventory showed only `plan.md`; its untracked content was separately checked for whitespace. No implementation tests or training ran. Roadmap completion: [x]; human review: [?].

Checkpoint-policy correction record (2026-10-02): Documentation only. The user superseded final-epoch-only selection with the established common-framework periodic mAP protocol. Updated session authorization wording, methodology/classification/limitation, recipe/configuration, common checkpoint protocol and provenance, compatibility map, Phase 3, Phases 16, 18, 20–25, and document-validation requirements. PCB optimization and LR policy remain unchanged; Phase 4 and all later phases remain unstarted and unauthorized. Validation: obsolete-policy scan, phase-field/status consistency, protected optimization/initialization contract comparison, whitespace checks and Git change-scope checks; no implementation tests or training.
