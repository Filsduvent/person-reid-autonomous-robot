# Stage B LODO implementation plan

Planning version 1; protocol `lodo_v1`; prepared on 2026-10-08. Companion documents: [scientific protocol](lodo_experiment_protocol.md) and [decision register](lodo_decision_register.md).

**Scope of this phase:** planning documentation only. No adapter, schema, sampler, scheduler, checkpoint, evaluator, test, executable LODO config, branch or worktree is created here. No training/evaluation/GPU job, commit or push is authorized. Stop after document validation and await explicit approval for Phase 3. Official training needs a separate explicit approval in Phase 10.

## Repository baseline and isolation

Observed branch: `main`. HEAD: `3b015c964b99bfe8ad4b7842bd174d3bc58af607`. The configured checkout `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot` and `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot` refer to the same directory.

Pre-existing work to preserve:

- Modified `docs/preliminary_results_log.md` and `plan.md`.
- Untracked `docs/pcb_duke_results.yaml` and `docs/pcb_market1501_results.yaml`.
- All Stage A/PCB configurations, result directories and active experiments.

Do not stash, reset, clean, move or absorb this work. After Phase 3 approval, create a dedicated branch and sibling worktree from an explicitly recorded base commit, for example `feature/lodo-v1` and `../person-reid-autonomous-robot-lodo`. Inspect availability before choosing either name. Transfer only reviewed LODO planning material into that worktree. Do not change the active checkout's branch or share output directories. No branch/worktree creation has occurred in this phase.

## Architecture and code anchors

Use one shared implementation across all folds and preserve the existing training engine. New behavior is explicit opt-in; absent Stage B configuration, Stage A paths retain existing behavior.

| Concern | Current file/symbol | Planned extension / verified limitation |
|---|---|---|
| Structural schema | `reid/utils/config_schema.py:validate_config` | Conditional requirements for successful-update mode; Stage A required keys remain intact |
| Semantic validation | `reid/utils/config.py:validate_reid_config` | Fold, source/target, recipe, schedule and policy validation |
| Dataset dispatch | `reid/data/build.py:build_train_loader` | Build virtual sources through existing factories; reuse sampling/collation |
| Source descriptors | `market1501.py`, `duke.py`, `cuhk03.py`, `msmt17.py` | Inspect source attributes without changing original loaders' default semantics |
| Public sample API | `reid/data/protocol.py:ReIDTrainSample`, `collate.py:train_collate_fn` | Preserve image/label contract; metadata stays out of samples |
| Sampling | `reid/data/samplers.py:PKBatchSampler` | Reuse selection algorithm; opt-in order/state recovery and exposure instrumentation |
| Model dimensions | `reid/models/build.py:build_model`, `baseline.py:ReidBaseline`, `pcb.py:PCB` | Dynamic combined source class count; same architecture definitions |
| Loss routing | `reid/losses/build.py:build_criterion`, `center.py:CenterLoss` | Preserve local half-mean and raw features; validate effective feature routing |
| Optimizer/schedule | `reid/optim/build.py`, `lr_scheduler.py:WarmupMultiStepLR` | Explicit successful-update units; no epoch-length milestone multiplication in Stage B |
| Training | `reid/engine/train_loop.py:train_one_epoch` | Opt-in exact stopping, result accounting and counters; retain legacy return/call behavior by default |
| Orchestration | `scripts/train.py:main`, `maybe_resume_training` | Target-free mode; existing unconditional test-loader construction is a blocker |
| Checkpoints | `reid/utils/checkpoint.py:save_checkpoint/load_checkpoint/reconstruct_model` | Center state, versioned protocol/recovery metadata; legacy model evaluation preserved |
| Evaluation | `scripts/evaluate.py`, `evaluate_cross_domain.py`, `reid/utils/evaluation_config.py` | Separate fold-aware evaluation; source-owned representation and target-owned parsing |
| Metrics/results | `reid/engine/evaluator.py`, `metrics/ranking.py`, `utils/experiment_matrix.py`, `metrics_artifacts.py` | Reuse retrieval metrics; separate Stage B records from Stage A matrix/selection |
| Provenance | `reid/utils/artifacts.py`, `utils/seed.py` | Versioned manifests, hashes, RNG/recovery and environment evidence |

Candidate new modules, only after approval: `reid/data/multi_source.py`, `reid/utils/lodo_protocol.py`, `reid/utils/lodo_artifacts.py`. Prefer small focused additions over duplicated training/evaluation engines. Exact names remain conditional on implementation review.

## Proposed minimum opt-in YAML contract

The following is **illustrative YAML inside documentation only**. It is not supported by the current schema, contains placeholders, and must not be launched. No runnable config is created here. Keep existing `model`, `loss`, `optim`, `repro`, `data` and `eval` ownership; add a protocol/version/fold block, virtual source descriptors, explicit update units and training policy. `data.test` holds inert target metadata during training, not a loader to construct.

```yaml
protocol:
  name: multi_source_lodo
  version: 1
  architecture: resnet50  # must agree with model.name capability
  fold: LODO-1
experiment:
  name: lodo_v1_resnet50_lodo1_seed42
  output_dir: exp/lodo/v1/resnet50/LODO-1/seed42/<unique-run-id>
system:
  device: cuda
  gpu_id: <validated-dedicated-gpu>
  amp: false
repro:
  seed: 42
  deterministic: false
  benchmark: true
logging:
  to_file: true
  tensorboard: true
data:
  root: <validated-dataset-root>
  train:
    dataset:
      name: multi_source
      sources:
        - {name: duke, format: processed, split: trainval}
        - name: cuhk03
          format: processed
          split: trainval
          image_type: detected
          protocol: processed_partition
          split_id: null
          provenance:
            intended_protocol: detected_new
            verification: unverified
        - name: msmt17
          format: raw
          split: train
          lists: [list_train.txt]
          image_subdir: msmt17/MSMT17_V2/mask_train_v2
    images: {size: [256, 128]}
    batch: {sampler: pk, P: 16, K: 4, batch_size: 64}
    aug: <resolved-existing-resnet50-training-transform>
  test:
    dataset: {name: market1501, format: processed, split: test}
    images: {size: [256, 128]}
    aug: <resolved-existing-resnet50-evaluation-transform>
    loader: {shuffle: false}
    batch: {size: 32}
model:
  name: reid_baseline
  backbone: {name: resnet50, pretrained: true, last_conv_stride: 1}
  head:
    embedding_dim: 2048
    pooling: gap
    bnneck: true
    normalize: true
    metric_feat: raw
    eval_feat: bn
    classifier: true
loss:
  id: {enabled: true, weight: 1.0, label_smoothing: 0.1}
  triplet: {enabled: true, weight: 1.0, margin: 0.3, mining: batch_hard}
  center:
    enabled: true
    weight: 0.0005
    lr: 0.5
    feat: raw
    reduction: half_mean_squared_distance
optim:
  name: adam
  lr: 0.0003
  weight_decay: 0.0005
  bias_lr_factor: 1.0
  weight_decay_bias: 0.0005
sched:
  name: warmup_multistep
  unit: successful_primary_updates
  milestones: [28400, 49700]
  gamma: 0.1
  warmup_iters: 500
  warmup_factor: 0.01
  warmup_method: linear
train:
  mode: successful_updates
  max_updates: 85200
  evaluate_during_training: false
  checkpoint_selection: terminal
  save: {save_best: false, save_last: true, save_terminal: true, resume: ''}
eval:
  normalize_feat: true
  distance: euclidean
  topk: [1, 5, 10]
  rerank: {enabled: false}
```

`lists`, `image_subdir` and CUHK03 `provenance` are proposed validation/provenance fields, not existing loader overrides. Verify they match the original loader's actual selection. Support a per-source root override only if needed; inherited roots remain explicit in the resolved manifest. Unknown fields and unsupported versions fail validation. Canonical order applies to all folds; no duplicated fold-specific adapter code.

In successful-update mode the schema no longer requires `train.epochs`; reject ambiguous simultaneous epoch and update stopping criteria. In default Stage A mode retain epoch requirements and legacy scheduling units. Do not fabricate an epoch cap to express a fixed update budget.

Illustrative PCB replacement fields below describe the same opt-in policy. The complete future resolved config must retain the remaining protocol/source/target/runtime sections and PCB preprocessing; these are not a standalone runnable config.

```yaml
protocol: {name: multi_source_lodo, version: 1, architecture: pcb, fold: LODO-1}
model: {name: pcb, variant: independent_part_reduction, pretrained: true}
loss:
  id: {enabled: true, weight: 1.0, label_smoothing: 0.0, head_aggregation: sum}
  triplet: {enabled: false}
  center: {enabled: false}
data:
  train:
    images: {size: [384, 128]}
    batch: {sampler: random, batch_size: 64, drop_last: true}
    aug: <resolved-existing-pcb-training-transform>
  test:
    images: {size: [384, 128]}
    aug: <resolved-existing-pcb-evaluation-transform>
optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  nesterov: false
  weight_decay: 0.0005
  weight_decay_bias: 0.0005
  bias_lr_factor: 1.0
  param_groups: [{prefix: backbone., lr_mult: 0.1}]
sched:
  name: warmup_multistep
  unit: successful_primary_updates
  milestones: [31400]
  gamma: 0.1
  warmup_iters: 0
  warmup_factor: 1.0
  warmup_method: linear
train:
  mode: successful_updates
  max_updates: 94200
  evaluate_during_training: false
  checkpoint_selection: terminal
  save: {save_best: false, save_last: true, save_terminal: true, resume: ''}
```

## Blocking preflight specification

All official preflight failures abort before optimizer updates and before writing to an existing experiment directory. Do not silently coerce a mismatch to the approved value. Write diagnostics only to a new preflight namespace. No success status can substitute for missing evidence.

| Check | Fail-closed condition | Future synthetic acceptance test |
|---|---|---|
| Fold membership | Not three distinct sources and exactly one target; wrong complement/order; unknown key/version | Every correct fold accepted; duplicates, target-as-source and swapped target rejected |
| Authorized splits | Unexpected processed split, CUHK03 image type/protocol, MSMT17 trainval/val list or manual combined dataset | Descriptor/list mutations fail before any image access |
| CUHK03 provenance | `new`/split-ID claim unsupported by flat file; hash or reviewed provenance differs | Unsupported selector rejected; explicit processed-partition limitation retained |
| Canonical paths | Missing paths, path escape, symlink alias or shared file across prohibited roles | Symlink/hardlink/relative alias fixtures expose overlap |
| Source membership | Source training includes target files or source test-query/gallery members | Injected forbidden records rejected |
| Train/validation partitions | Constituent `train`/`val` image overlap or unexpected trainval membership; separate validation drawn from trainval | Disjoint constituents accepted; authorized trainval union accepted; reuse as independent validation rejected |
| MSMT17 lists | Wrong image subtree/list hashes; train/val image overlap; malformed rows/cameras | Train-only selection accepted; val inclusion and malformed fixtures rejected |
| Mapping | Noncontiguous labels, missing PIDs, inconsistent local/global map, nondeterministic digest | Same metadata produces identical map despite local-label order; equal-C swapped map rejected |
| Class dimensions | C differs from validated sum; any ResNet/PCB classifier width differs; Center not C×2048 | Validate all six PCB heads and ResNet/Center shapes |
| Recipe | Incorrect loss/feature/reduction, sampler/batch/input, optimizer groups, precision or budget | Mutate each fixed architecture setting and require rejection |
| Schedule | Epoch units in Stage B; incorrect warmup/decay boundary; center LR scheduled with main LR | Boundary and restored-trace fixtures fail on one-update shifts |
| Target isolation | Training constructs a target loader or invokes evaluator/metric selection | Monkeypatch target builder/evaluator to raise; Stage B training succeeds without calling them |
| Resume | Wrong fold, mapping, recipe/config identity, counters or recovery capability; missing centers | Same-C wrong-map, legacy-center exact-resume and incomplete-step fixtures rejected |
| Output isolation | Existing unrelated output, canonical collision with Stage A/other run, unsupported overwrite | Cross-worktree/symlink collision rejected; explicit compatible resume separately accepted |
| Terminal status | Too few/many updates, auxiliary mismatch, wrong completion reason or unverified checkpoint hash | Boundary fixtures distinguish interruption from terminal completion |
| Target retrieval | Wrong excluded target/query/gallery, changed target PIDs/cameras/marks or preprocessing | Original marks/filtering retained; target-derived classifier count rejected |

Metadata-only leakage checks may read original partition/list inventories in a separate preflight stage. They must not instantiate a target dataset/loader during training. Check files using canonical paths and, where available, inode identities and frozen hashes; lists with different names can point at the same file. Parent-directory equality alone is not leakage because processed train/test share an image directory.

The inspected processed metadata has zero train/val constituent filename overlap and zero trainval/test filename overlap. This is limited evidence: canonical/inode/content validation and cross-dataset duplicate-person detection were not performed. Shared query/gallery PIDs are expected. Source namespacing prevents label collisions but does not prove different real people. Report duplicate-content findings or uncertainty; never claim path checks establish semantic identity disjointness.

## Exact-update and accounting contract

Count one primary update only after a successful parameter update. In the initial FP32 campaign, nonfinite loss/gradients fail before either optimizer step; no silent skipped-update recovery. ResNet center rescaling and its auxiliary step must complete before a recovery checkpoint is considered consistent. If an optimizer/auxiliary step fails partway, abort and restore the prior consistent checkpoint; do not save the partially updated state as resumable or terminal.

Before the next batch, stop if `primary_updates == budget`; never execute update `budget+1`. Prefetch may fetch unused images, but unused prefetched records are not counted as model-processed exposure. Define attempted batches, model-forward/backward image draws, successful-update images, and unconsumed prefetched records separately. Only successful primary updates advance the scheduler. Initial Center auxiliary count must equal primary count at each consistent checkpoint; PCB auxiliary count is zero.

Use explicit successful-update schedule units and LR-used logging. For ResNet updates 1/500/501 the factors are .01/.99802/1; decays affect update 28,401 and 49,701. For PCB the single decay affects 31,401. Record each group's LR used and LR next, and test continuation immediately before/after every boundary. Do not multiply Stage B milestones by loader length.

Partial-epoch averages use actual processed batches or sample-weighted totals, with an explicit denominator and batch/sample count. Do not divide by the full loader length after early termination. Report raw and weighted loss terms distinctly. Preserve Stage A's existing return interface and logs by default, introducing structured counters/accounting only in the opt-in path.

Exposure counters use consumed batch indices plus adapter sidecar metadata. For ResNet, labels may identify source identity slots, but a resumable index trace is needed for exact image exposure. For PCB, labels alone cannot recover image draws; capture batch indices without changing `ReIDTrainSample`. Bookkeeping must not consume sampler RNG, shuffle again, or alter which records are trained.

## Versioned checkpoint, recovery and artifact contract

Extend existing reconstruction metadata rather than breaking its bounded legacy fallback. Store a new recovery/protocol schema containing model, main optimizer, Center module and optimizer when enabled, scheduler, attempted/primary/auxiliary counts, exposure, epoch/cursor, RNG states, sampler/order state, fold, mapping digest, data manifests, resolved-config identity and precision/scaler state when applicable. Save the canonical Center module state, not only SGD state; avoid duplicate aliases in a whole-bundle state dictionary.

Validate all scientific metadata before resuming: C, D, actual feature branch, local reduction, coefficient, architecture/fold, map digest, optimizer/schedule and frozen config identity. Restore centers to the freshly constructed module associated with the center optimizer; prove the restored next update matches continuous training. Do not replace parameter objects after optimizer construction without rebuilding their association.

The frozen resolved training config is immutable and hashed. Store permitted operational overrides and a compatibility digest separately, so a new resume path does not spuriously redefine the scientific recipe. Define canonical serialization before relying on hashes. Evaluation config has a separate identity and cannot alter the frozen training config.

Exact recovery must account for sampler-owned Python/NumPy RNG, epoch batch order and consumed cursor, global Python/NumPy/Torch CPU/CUDA RNG, worker transform RNG, prefetch and any PCB stochastic layers. Merely saving global RNG and advancing a cursor is insufficient. Test checkpointing after full main+aux steps, interrupted partial epochs and schedule boundaries. A checkpoint records its validated recovery capability.

Use deterministic synthetic fixtures for numerical resumed-next-update equivalence. Separately validate actual GPU batch order, RNG/augmentation continuation and numerical behavior. The inherited nondeterministic GPU settings do not promise bitwise trajectories; report this distinction and block any stronger claim without evidence. A change to deterministic execution requires documented review, not an implicit alteration of the approved campaign.

If exact mid-epoch recovery cannot initially be demonstrated, fail an exact request at unsupported positions. A tested restricted policy may roll back to a compatible persisted boundary and discard/re-record unfinished attempts; it must not silently continue under a new stochastic trajectory. Worker-count restrictions or other execution changes require documented review. Official launch remains gated on accepting and testing the recovery contract.

Terminal artifact specification: a distinct `checkpoints/ckpt_terminal.pth` (proposed name), `completion_reason: budget_completed`, configured and actual counters, architecture/fold/C/D, config/map/data digests and checkpoint SHA256 recorded in its sidecar manifest. Recovery `ckpt_last.pth` is not automatically terminal. Use atomic write/replace within the run directory and write completion metadata only after checkpoint integrity is verified. A resumed already-terminal run must not step again.

Keep legacy Stage A model-only evaluation and reconstruction intact. No missing legacy Center state can be reconstructed from momentum-free SGD metadata. Reject legacy Center exact-resume in Stage B; retain the historical Stage A path with its existing limitations rather than rewriting artifacts. Future AMP requires synchronized skip/counter handling and scaler persistence; it is outside the finalized FP32 campaign.

## Dependency-aware roadmap and stop gates

All tests below are **future acceptance requirements**, not tests run during this planning phase. Each phase must leave default Stage A behavior intact. Dependency order: 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10. Center schema design can start after Phase 3, but integration acceptance in Phase 6 depends on Phase 5 counters and recovery positions.

### Phase 3 — Isolated development and protocol/schema contracts

**Objective:** establish opt-in boundaries and fail-closed protocol contracts without touching the active Stage A checkout.

**Likely files:** the three LODO documents; `reid/utils/config_schema.py`, `config.py`; proposed `lodo_protocol.py`; future `tests/test_lodo_config.py` and extensions to `test_config_schema.py`.

**Tasks:** after approval, create the separate branch/worktree; record base commit and dirty-state exclusions; define all four folds/canonical order; introduce versioned schema, conditional epoch/update requirements, immutable config identity, output namespace and preflight/manifest contracts. Resolve the unsupported CUHK03 `new` claim as explicit provenance, not a loader workaround.

**Tests:** legacy schema acceptance; all four folds; ambiguous units/modes; forbidden target-source membership; unsupported versions; wrong recipe; output-path collision and unsupported CUHK03 fields.

**Acceptance/stop gate:** reviewed schema and manifests; no LODO fallback from malformed input; Stage A configs accepted unchanged; isolated paths confirmed. Stop if provenance requirements demand unavailable official-new evidence or if isolation cannot preserve existing work.

**Risks:** overbroad validation changes and config ambiguity. **Rollback:** revert only Phase 3 commits in the isolated worktree; preserve the original checkout, datasets and artifacts. No official runs.

### Phase 4 — Virtual sources and deterministic identity mapping

**Objective:** build a virtual adapter through existing source loaders with recoverable labels and metadata.

**Likely files:** proposed `reid/data/multi_source.py`; `data/build.py`; protocol/preflight helpers; future `tests/test_multi_source_dataset.py`, `test_lodo_preflight.py`; existing dataset/collator tests.

**Tasks:** factory extraction/reuse without default behavior changes; concatenate source index spaces; deterministic PID map/digest; original/local/global PID provenance; source/camera/index sidecars; validated combined C; source/target membership and descriptor/path gates. Delegate transforms/image reads to source loaders; do not copy images or change the sample API.

**Tests:** toy equal-PID datasets remain distinct; stable maps across local-label ordering; contiguous labels; class sums; sidecar alignment; unmodified collation; delegated transforms; overlapping paths/symlinks; authorized trainval union; MSMT train-only; no target loader.

**Acceptance/stop gate:** adapter satisfies `ReIDTrainDataset`, mapping is reconstructable, every sample aligns with labels/source metadata, all prohibited overlap fixtures fail. Metadata counts agree with the frozen inventories or discrepancies are explicitly reviewed. Stop on map ambiguity, leakage or unverified descriptor changes.

**Risks:** confusing original PIDs with local labels; duplicated transforms. **Rollback:** revert Phase 4 adapter/dispatch changes in the isolated worktree; retain legacy factories and all data files.

### Phase 5 — Sampling reuse, exact updates, LR and exposure

**Objective:** preserve sampling algorithms while controlling successful updates and accounting accurately.

**Likely files:** `data/build.py`, `data/samplers.py`, `engine/train_loop.py`, `optim/build.py`, `optim/lr_scheduler.py`; future `tests/test_lodo_updates.py`, `test_lodo_exposure.py`; existing sampler/loop/scheduler tests.

**Tasks:** opt-in update counter and mid-epoch stop; structured partial-epoch accounting; index-sidecar exposure; sampler state/order contract; update-unit schedule; LR-used/next trace; main-versus-aux counters; keep Stage A epoch units and API behavior as defaults.

**Tests:** tiny budgets shorter than an epoch; variable PK epoch lengths; no update beyond budget; same sampled order without bookkeeping; exposure sum/per-source/identity slots; partial denominators; nonfinite failure; ResNet 1/500/501, 28,400/28,401, 49,700/49,701 and final boundary; PCB 31,400/31,401 and final boundary. Use cheap synthetic optimizers/gradients for full scheduler traces.

**Acceptance/stop gate:** exactly 85,200/94,200 planned successful-update traces; correct groups/boundaries; no dependence on fold loader length; no counting center steps as primary. Stop for shifted milestones, altered sampling or unverifiable exposure.

**Risks:** off-by-one scheduling, sampler RNG perturbation and incomplete-update accounting. **Rollback:** revert only opt-in control/instrumentation commits; no Stage A behavior/config changes.

### Phase 6 — Center persistence, recovery and target-free terminal policy

**Objective:** make consistent checkpoints resumable and training independent of held-out evaluation.

**Likely files:** `utils/checkpoint.py`, `losses/build.py`, `scripts/train.py`, provenance helpers; future `tests/test_lodo_checkpoint.py`, `test_lodo_resume.py`; existing checkpoint/reconstruction/orchestration tests.

**Tasks:** versioned Center/module/optimizer state and compatibility checks; RNG/sampler/cursor/exposure recovery; target-free orchestration; distinct terminal checkpoint and completion reason; terminal count verification; atomic artifact completion. Restore the correct next LR and center tensor without changing the local formula.

**Tests:** nonzero Center round-trip; restored optimizer and gradients; resumed-next-update and multi-step equivalence; same-C wrong-map/config/fold rejection; legacy model-only evaluation; unsupported legacy Center exact-resume; no target builder/evaluator calls; no extra step from terminal resume; interrupted main/aux step not marked complete; prefetched-worker recovery capability tests.

**Acceptance/stop gate:** learned centers persist; main/aux counts remain synchronized; exactly terminal counts; demonstrated recovery capability recorded; default Stage A orchestration/reconstruction unchanged. Stop if exact continuation is claimed without evidence; obtain review of any restricted recovery policy before official training.

**Risks:** missing center tensors, stale optimizer parameter associations, worker RNG/prefetch and false terminal completion. **Rollback:** revert isolated opt-in/checkpoint additions without deleting old artifacts; versioning must retain legacy readers. Never convert a partially updated checkpoint to terminal.

### Phase 7 — Held-out evaluation and strategy comparison

**Objective:** evaluate one excluded target per fold after terminal completion and report separately from Stage A.

**Likely files:** `scripts/evaluate.py`/`evaluate_cross_domain.py`, `utils/evaluation_config.py`, proposed LODO artifact/report helpers; future `tests/test_lodo_evaluation.py`, `test_lodo_comparison.py`; existing evaluator/cross-domain tests.

**Tasks:** validate terminal completion/hash; reconstruct with combined source C; use original target query/gallery, source-owned transforms and unchanged metrics; emit fold-aware manifests/results; reference three Stage A source→target cells, arithmetic means and best values; percentage-point differences; explicit missing/confounded references.

**Tests:** combined C retained for ResNet and all six PCB classifiers; raw target labels/cameras preserved; same-camera filtering unchanged; target options and source representation ownership; three-cell means/deltas/best-envelope versus best-by-mAP; missing-cell handling; Stage B records cannot contaminate Stage A matrix/model selection.

**Acceptance/stop gate:** four valid fold records per architecture are representable; no target-derived reconstruction or training selection; missing references are not fabricated. Stop if target protocol differs or results would overwrite Stage A.

**Risks:** mixing training/eval config ownership and causal overinterpretation. **Rollback:** remove/revert isolated new reporting integration; retain existing evaluator/metrics and Stage A outputs.

### Phase 8 — CPU regression and synthetic integration gate

**Objective:** establish Stage A compatibility and end-to-end Stage B invariants before GPU work.

**Likely files:** existing dataset, sampler, config, model, loss, optimizer, scheduler, training, checkpoint, evaluator and PCB tests; new opt-in LODO tests from Phases 3–7.

**Tasks:** run appropriate Stage A regression plus synthetic LODO training/recovery/evaluation for every fold and architecture; verify public contracts, full boundary traces, leakage rejection, legacy checkpoint reconstruction, exact stop and resume; record commands/environment/pass/fail/skip evidence.

**Tests:** fresh and resumed tiny-budget integration, variable PK order, PCB six-head loss, Center formula/gradient weighting, every preflight negative fixture, default Stage A epoch-selected checkpoints and existing frozen PCB group behavior.

**Acceptance/stop gate:** relevant regressions and synthetic integrations pass; skipped CUDA tests are explicit and cannot count as GPU validation; implementation and scientific config remain consistent. Stop on any regression or unresolved integrity/recovery failure.

**Risks:** tests mirror implementation rather than expected behavior or hide skips. **Rollback:** repair/revert isolated changes causing regressions; keep Stage A checkout untouched. No GPU or official training before this gate.

### Phase 9 — Non-authoritative GPU smoke validation

**Objective:** check actual device behavior, larger heads/centers and recovery without claiming official results.

**Likely files/artifacts:** reviewed smoke configuration material and future launch helpers; only an isolated `exp/lodo_smoke/...` namespace, separate from official/Stage A directories.

**Tasks:** after the CPU gate and resource review, run explicitly short GPU smoke jobs; construct all fold class counts and both architectures; check memory/throughput, LR trace, center/main counters, saved-state recovery and separate target evaluation. Mark any target smoke metrics non-authoritative and prohibit their use to change the approved recipe or select checkpoints. Prefer synthetic targets for plumbing checks; original target evaluation, if needed, occurs only after smoke training ends.

**Tests:** actual-device finite gradients, checkpoint reconstruction and recovery, terminal-at-smoke-budget counts, loader isolation, dimensions and measured peak VRAM. Record the shortened budget as smoke-only; retain approved recipes, and test official milestones via the CPU traces rather than using smoke results to retune them.

**Acceptance/stop gate:** observed GPU feasibility, class counts, memory margin and validated recovery capability; no overlap/interference with ongoing work; no target-based tuning. Stop on OOM, state divergence, target leakage or absent resource evidence.

**Risks:** smoke artifacts mistaken for final evidence, shared-GPU interference. **Rollback:** stop only the newly launched smoke job if needed; quarantine its directory and repair isolated implementation. Do not kill or modify Stage A jobs. Recipe/batch/budget changes require explicit protocol review.

### Phase 10 — Freeze and guarded official launch

**Objective:** freeze a reproducible campaign and present reviewable launch material before authorization.

**Likely files/artifacts:** future finalized LODO config files, immutable protocol/config/data/map manifests, code commit identity, guarded launch and monitoring commands, separate official output directories. None is created in this planning phase.

**Tasks:** verify all prior gates; freeze implementation commit and clean/recorded state, eight resolved configs, descriptor/inventory hashes, pretrained artifacts, maps and environment; review CUHK03 provenance and recovery limitations; estimate resources from measured smoke throughput; preflight each fold, verify dedicated GPU/output paths and prepare guarded commands with exact seed/budget/schedule.

**Tests:** launch dry/preflight checks; immutable hashes; correct eight-run identities and terminal policies; output collisions; no target loader; consistent class/map/config metadata; rerun only checks justified by final changes.

**Acceptance/stop gate:** implementation is ready for official training only when all evidence is observed and reviewed. **Request explicit user approval before executing any official training command.** Approval of science or earlier implementation phases is not official-launch approval. Do not launch when any required gate is incomplete.

**Risks:** stale manifest/config or implicit authorization. **Rollback:** withdraw unexecuted commands; re-freeze only after review. Never overwrite existing official/Stage A artifacts; failed launched runs retain provenance and require compatible recovery or separately identified restart.

## Scientific limitations versus engineering blockers

Scientific limitations: one seed, unequal source contributions, fixed budgets not demonstrated convergence optima, strategy-level Stage A comparison confounds, mixed Center/smoothing findings and CUHK03 official-new provenance uncertainty. These must be reported even after successful implementation.

Engineering blockers before official training: missing Center module checkpoint state, no exact-update stop/counters, unconditional target-loader construction, mapping/digest and fold-aware compatibility not implemented, stochastic recovery not demonstrated, update-unit schedule/schema absent, exposure/provenance/result manifests absent, and GPU VRAM/throughput unmeasured for LODO.

Open operational decisions: tested worker/recovery configuration, recovery checkpoint cadence/storage retention, dedicated GPU/launch ordering, final artifact names and resource-dependent extra seeds. None permits changing the finalized folds, recipe, budget or target-isolation rule silently.

## Current-phase validation and stop condition

Review the three planning documents for all four folds, authorized splits, deterministic map/public contract, exact architecture recipes and LR-used boundaries, Center persistence, target isolation, terminal/recovery metadata, comparison caveats, complete Phase 3–10 gates and truthful decision statuses. Parse illustrative YAML only as documentation; do not validate it as a runnable config or execute it.

Check Git status and hashes of pre-existing modified/untracked Stage A documents before/after. The only task-created files may be the three designated LODO documents. Documentation consistency checks are not implementation tests or GPU evidence.

Current readiness: **READY FOR IMPLEMENTATION**, subject to explicit approval to begin Phase 3. **Not ready for official training.** No phase 3–10 task or test has been completed by preparing this plan. Stop here.
