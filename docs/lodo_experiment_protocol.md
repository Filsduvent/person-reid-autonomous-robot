# Stage B LODO scientific protocol

## Phase 10F current user sign-off — 2026-10-11

The [final sign-off](lodo_phase10_final_signoff.md) records the explicit later user instruction. It accepts the exact hashed CUHK03 detected `processed_partition`/null split with official-new correspondence **UNVERIFIED**, approves Stage B-only strict deterministic CUDA/FP32/zero workers and completed-epoch/every-epoch/two-validated-generation recovery plus distinct terminal, and accepts incomplete historical Stage A comparison eligibility. Scientific recipes, splits and budgets are unchanged. Earlier dated implementation/approval statements below remain historical. Advisor/institutional approval is not supplied or inferred. One local feature-branch freeze is conditionally authorized after all gates; official training, resume, target evaluation, GPU work, allocation, merge and push remain unauthorized. Actual commit identity is recorded in a separate ignored operational receipt, never self-referentially in these manifests.


Protocol identifier: `lodo_v1`. Planning version: 1. Prepared on 2026-10-08.

Scientific settings are **FINALIZED** by the user's Phase 2 instructions. Engineering requirements remain unimplemented and require validation. This document authorizes neither implementation nor official training. See the [implementation plan](lodo_implementation_plan.md) and [decision register](lodo_decision_register.md).

## Scientific question and scope

For *Long-Term Person Re-Identification in Autonomous Robotic Systems: From Benchmark Evaluation to Real-Time Edge Deployment*, Stage B asks whether a predefined multi-source training strategy improves recognition on an excluded dataset relative to the existing single-source cross-domain strategies.

Stage A remains frozen: independent training on four datasets and the existing 4×4 within-domain/cross-domain evaluation, with original query/gallery and CMC Rank-1/5/10, mAP and mINP. Preserve all existing configurations, model behavior, checkpoints, outputs, evaluation matrices, provenance and ongoing PCB work. Stage B does not replace or retroactively standardize Stage A's heterogeneous ResNet50 losses.

Stage B has four training runs and four separate held-out evaluations **per architecture**: four ResNet50 and four PCB runs in the initial seed-42 campaign. There are no Stage B results yet. Development, recovery and smoke runs must use separate namespaces and must not be counted as official experiments.

## Four folds

Canonical dataset order is `market1501`, `duke`, `cuhk03`, `msmt17`; each fold uses the ordered complement of its target.

| Fold | Ordered training sources | Held-out target |
|---|---|---|
| LODO-1 | Duke + CUHK03 + MSMT17 | Market-1501 |
| LODO-2 | Market-1501 + CUHK03 + MSMT17 | DukeMTMC-ReID |
| LODO-3 | Market-1501 + Duke + MSMT17 | CUHK03 |
| LODO-4 | Market-1501 + Duke + CUHK03 | MSMT17 |

The target is excluded from training, checkpoint selection, early stopping and hyperparameter tuning. No previously manually combined dataset is authorized. Source datasets are combined virtually through their existing loaders.

## Authorized dataset descriptors and provenance

All paths below are relative to the resolved dataset root. The observed root is `/home/addirakoze/nobackup/Projects/Dataset`, resolving to `/mnt/hd-data/addirakoze/Projects/Dataset`. Record both configured and canonical paths; revalidate them on the training host.

| Dataset key | Authorized source format/split | Image location | Split metadata |
|---|---|---|---|
| `market1501` | Processed / `trainval` | `market1501/images/` | `market1501/partitions.pkl`, `trainval_im_names` |
| `duke` | Processed / `trainval` | `duke/images/` | `duke/partitions.pkl`, `trainval_im_names` |
| `cuhk03` | Processed / `trainval`, detected images | `cuhk03/detected/images/` | `cuhk03/detected/partitions.pkl`, `trainval_im_names` |
| `msmt17` | Raw / `train` | `msmt17/MSMT17_V2/mask_train_v2/` | `msmt17/MSMT17_V2/list_train.txt` only |

Existing dispatch and descriptors: [data/build.py](../reid/data/build.py), [Market loader](../reid/data/market1501.py), [Duke loader](../reid/data/duke.py), [CUHK03 loader](../reid/data/cuhk03.py), [MSMT17 loader](../reid/data/msmt17.py).

**CUHK03 provenance blocker:** the approved scientific description refers to detected/new data, but `cuhk03._validate_protocol()` accepts only `protocol: processed_partition`, `split_id: null`. The flat file has no official-new/classic or split-ID metadata. Preserve the detected partition; record `intended_protocol: detected_new` as a provenance assertion with `verification: unverified`, not as a supported loader selector or established fact. Do not pass `protocol: new`, invent a split ID, rewrite Stage A metadata, or silently convert data. Official Stage B launch requires review of the exact partition hash and this limitation. A strict official-new claim requires independent conversion/split evidence; absent that evidence, the report must say detected processed-partition. Any requirement to change data needs explicit protocol review.

MSMT17 source training excludes `list_val.txt`; it does not silently use `trainval`. Its target evaluation uses `mask_test_v2/`, `list_query.txt` and `list_gallery.txt` through the original test loader. Processed targets use their original `test_im_names` and `test_marks`. Never remap target identities or cameras into source training labels.

Metadata-only inspection on 2026-10-08 recovered these counts. They are a planning baseline, not a substitute for launch-time descriptor, file and hash validation.

| Source | Training images | Training identities | SHA256 of partition/list file |
|---|---:|---:|---|
| Market | 12,936 | 751 | `3853cfdc9be6814d0193b9aad8025b61133d7f7b9ef8449b5d0f2964811a53ab` |
| Duke | 16,522 | 702 | `25929417e0a8780548924d4f58f4f644ee18098e957df41a571650eb05e6b638` |
| CUHK03 detected | 7,365 | 767 | `a1fd9a06748dd217325820b2ed5c5814de364a6b5e91cb7a93e1924915034f97` |
| MSMT17 train | 30,248 | 1,041 | `00930d5911e66ec6ac4d9b177e61b49433658758cc091231bf7cfcb83c70d0bf` |

| Fold | Calculated source images | Calculated combined identities |
|---|---:|---:|
| LODO-1 | 54,135 | 2,510 |
| LODO-2 | 50,549 | 2,559 |
| LODO-3 | 59,706 | 2,494 |
| LODO-4 | 36,823 | 2,220 |

These identity counts assume dataset-namespaced identities. Live class counts must come from validated source metadata, never target classes or hard-coded table values.

## Virtual dataset and identity contract

The future adapter delegates `__getitem__` to the original source loader, retaining its image loading and architecture-owned transforms. It returns the existing `ReIDTrainSample(image, label)` with only the label replaced. Preserve [data/protocol.py](../reid/data/protocol.py), the collator and the engine's `(images, labels)` contract. Source/domain metadata stays in adapter attributes and a sidecar index.

Identity keys are `(dataset_key, original_pid)`. Here `original_pid` means the PID exposed by the authorized loader before local training relabeling; for processed datasets it may be an encoded processed PID, not a recoverable official raw PID. Record this provenance explicitly.

Build the map by canonical source order and numerically sorted source PIDs. Assign contiguous labels `0..C-1`. Persist recoverable rows containing dataset key, original PID, local label and global label; also persist source-local index-to-global index associations and camera metadata. Hash versioned canonical UTF-8 JSON with sorted object keys, fixed separators and ordered rows using SHA256. The implementation must specify the canonical serialization and test it.

All classifier heads and the Center Loss tensor share this mapping. Equal class counts do not imply equal mappings. Source-local cameras remain unchanged. Camera namespacing is unnecessary for independent target retrieval and is reserved for a future pooled-source retrieval protocol.

## Sampling and exposure

| Architecture | Primary sampling | Batch |
|---|---|---:|
| ResNet50 | Existing PK, P=16, K=4 | 64 |
| PCB | Existing random shuffled image sampling, `drop_last=True` | 64 |

Do not add domain-balanced sampling. [PKBatchSampler](../reid/data/samplers.py) has variable epoch length, per-identity remainder dropping, and replacement for identities with fewer than K images. Its length is not a guaranteed fixed number of updates. PCB image sampling gives larger image sources more opportunities. Neither policy equalizes source contributions.

Record realized processed images, successful-update images, per-source samples, per-source identity selections, and distinct identities encountered. For PK, report identity-slot selections as well as image draws; distinguish repeats from unique exposure. Obtain source membership from sidecar indices or instrumented batch indices without adding fields to the public sample contract. Do not infer all exposure from dataset-size proportions.

## Final ResNet50 Candidate C recipe

One common recipe applies to all four folds. It is a **reference-oriented project implementation**, not an exact upstream numerical reproduction and not a configuration shown to be empirically superior in Stage A.

| Component | Final setting |
|---|---|
| Model | ResNet50, final stride 1, global average pooling |
| Neck / feature width | BNNeck; 2048 dimensions |
| Input | 256×128 |
| Identity | Cross-entropy weight 1, smoothing ε=0.1, post-BN bias-free classifier |
| Triplet | Batch-hard, weight 1, margin 0.3, pre-BN features |
| Center | Enabled, weight 0.0005, pre-BN 2048-D features |
| Center convention | Local `0.5 × mean ‖feature-center‖²` |
| Center optimizer | SGD LR 0.5, no momentum, no weight decay |
| Center gradient | Inverse-weight rescaling of center-parameter gradients only |
| Retrieval | Post-BN L2-normalized embedding |
| Main optimizer | Adam LR 0.0003, weight decay 0.0005; retain existing Adam defaults |
| Bias policy | LR factor 1, weight decay 0.0005 |
| Sampling | PK P16×K4, batch 64 |
| Precision / initial seed | FP32, AMP disabled; seed 42 |
| Primary budget | 85,200 successful optimizer updates |

Preserve existing architecture-specific preprocessing: RGB mean `[0.486, 0.459, 0.408]`, std `[0.229, 0.224, 0.225]`, resize, random flip, padding 10, crop and random erasing probability 0.5 with existing parameters. Evaluation uses existing deterministic preprocessing, no reranking or test-time flip in the primary protocol. Preserve the current pretrained-weight policy and record the actual weight identifier/hash before launch.

Code anchors: [baseline model](../reid/models/baseline.py), [loss construction](../reid/losses/build.py), [Center Loss](../reid/losses/center.py), [optimizer construction](../reid/optim/build.py). The current loss builder uses `model.head.metric_feat` for both metric losses; validate raw/pre-BN routing rather than trusting an independently declared center feature field.

The upstream methodological reference is [Luo et al.](https://arxiv.org/abs/1903.07071) and its [author implementation](https://github.com/michuanhaohao/reid-strong-baseline). Prior audits found mixed Center effects and improved mAP from removing smoothing in Center-disabled comparisons. The Center-enabled unsmoothed combination was untested. These findings remain limitations; they do not override the user's finalized Candidate C decision.

### ResNet50 LR in successful-update units

Let `u` be the one-based primary update being executed and `n=u-1` the completed primary updates before it. The LR in this table is the LR **used by that optimizer step**, not the LR stored after advancing the scheduler.

| Update used | Main LR |
|---|---|
| 1–500 | `0.0003 × [0.01 + 0.99 × (u-1)/500]` |
| 501–28,400 | 0.0003 |
| 28,401–49,700 | 0.00003 |
| 49,701–85,200 | 0.000003 |

Gamma is 0.1; decays occur after completed updates 28,400 and 49,700. Center LR stays 0.5 throughout. Preserve [WarmupMultiStepLR](../reid/optim/lr_scheduler.py) boundary semantics: update 1 uses factor 0.01, update 500 uses factor 0.99802 (LR 0.000299406), and update 501 first uses factor 1. This is the precise meaning of the 500-update warmup; do not silently change the denominator to 499. Constructor and used-versus-next-LR traces require future tests.

One Center step accompanies each successful FP32 primary update. Center steps are auxiliary: 85,200 primary and 85,200 Center updates, not 170,400 primary updates.

## Final PCB recipe

Preserve the existing independent-part PCB architecture and its distinct optimization recipe.

| Component | Final setting |
|---|---|
| Model | ResNet50-based backbone, six horizontal parts, six classifiers |
| Retrieval embedding | Six 256-D part features concatenated to 1536 dimensions; existing evaluator normalization |
| Input | 384×128 |
| Loss | Six-head summed Identity CE, weight 1; no smoothing, Triplet or Center |
| Sampling | Random shuffled images, batch 64, `drop_last=True` |
| Optimizer | SGD, momentum 0.9, weight decay 0.0005, Nesterov false |
| Initial groups | Backbone LR 0.01; reductions/classifiers LR 0.1 |
| Bias policy | LR factor 1; weight decay 0.0005 |
| Warmup | None |
| Precision / initial seed | FP32, AMP disabled; seed 42 |
| Primary budget | 94,200 successful optimizer updates; zero auxiliary updates |

| Update used | Backbone LR | Reduction/classifier LR |
|---|---:|---:|
| 1–31,400 | 0.01 | 0.1 |
| 31,401–94,200 | 0.001 | 0.01 |

Gamma is 0.1. Preserve the 10:1 new-layer/backbone ratio. Preprocessing remains 384×128 resize, random flip, the existing RGB normalization, and no padding, random crop or random erasing. Primary evaluation retains existing normalization, Euclidean distance, no reranking and no test-time flip. Preserve checksum-pinned PCB ImageNet initialization.

Code/config anchors: [PCB model](../reid/models/pcb.py), [multi-head loss](../reid/losses/build.py), [optimizer groups](../reid/optim/build.py), and the frozen [PCB Market preset](../configs/pcb/market1501.yaml). These Stage A presets must not be edited for LODO.

## Budget interpretation and terminal selection

The budgets are rounded mean 120-epoch LODO workload reference budgets from the prior investigation. They are not measured convergence optima. They equalize primary updates within an architecture, not training time, unique images, effective passes or compute across architectures.

At batch 64, successful-update exposure is 5,452,800 image draws per ResNet50 fold and 6,028,800 per PCB fold. Across four folds these are 340,800 and 376,800 primary updates respectively. Repeated draws do not represent that many distinct images. Realized source exposure must still be measured.

Training stops after exactly the configured successful primary count, including inside an epoch. A predetermined terminal checkpoint is selected solely because the budget completed. An interruption/recovery checkpoint is not a terminal checkpoint. Partial epochs and attempted/failed batches must be accounted for separately.

The opt-in Stage B mode must never construct the held-out target evaluation loader during training. Disabling `save_best` is insufficient: [scripts/train.py](../scripts/train.py) currently constructs a test loader unconditionally. No target evaluation, early stopping or metric-selected checkpoints are permitted. Stage A remains epoch-controlled with its existing metric-selected policy by default.

## Recovery and checkpoint contract

Before official ResNet50 training, fix the verified defect in [checkpoint.py](../reid/utils/checkpoint.py): center optimizer state is saved but learned centers are not. New checkpoints must contain Center module state plus optimizer state and validate class count, D=2048, pre-BN branch, local half-mean reduction, coefficient, fold, mapping digest and frozen resolved-config identity.

Recovery also requires model, main optimizer, scheduler, successful primary/auxiliary counters, attempted batches, epoch and within-epoch batch position, Python/NumPy/Torch CPU/CUDA RNG states, sampler state or deterministic order reconstruction, consumed batch cursor, exposure counters, and AMP scaler if AMP is later approved. Store recovery capability explicitly; exact stochastic continuation must be demonstrated, including prefetched-worker transforms, rather than assumed from restoring global RNG alone.

Distinguish stochastic-state/order continuation from bitwise numerical reproducibility. The inherited `deterministic=false`/benchmark settings do not guarantee bitwise-identical GPU trajectories. Use deterministic fixtures to demonstrate next-update equivalence, and report the actual GPU guarantee separately. If bitwise GPU continuation is required, this is an explicit validation blocker until compatible deterministic execution is reviewed; do not silently change execution settings or advertise an unsupported guarantee.

Preserve legacy Stage A model-only evaluation/reconstruction. A legacy Center-enabled checkpoint cannot support exact continuation because its centers are absent; reject an exact-resume request rather than silently reinitialize. Keep legacy behavior available for existing Stage A defaults while making its limitations explicit. Do not relabel legacy artifacts as exact Stage B checkpoints.

If exact recovery is initially infeasible, specify and validate a restricted capability, such as exact recovery only from completed epoch boundaries under a tested loader configuration. A failure outside supported boundaries must stop or roll back to a compatible persisted boundary with discarded attempts recorded. No silent trajectory changes or altered budget. Official launch requires acceptance of the documented recovery capability.

## Integrity and target-isolation gates

Preflight must fail closed for invalid folds, descriptors, identity maps, model dimensions, sampling/loss/schedule/budget combinations, checkpoint compatibility and output collisions. Canonicalize paths and symlinks, verify metadata and image inventories, and check authorized source membership against excluded target and source query/gallery inventories. Metadata-only leakage preflight may inspect target partition/list paths in a separate stage; it must not construct a target dataset/loader, transform target images, extract features or evaluate a model during training.

For processed sources, `train` and `val` constituent image lists must be disjoint. The authorized `trainval` union includes those constituent images by design; it must not also be treated as a disjoint validation set. For MSMT17, `train` and `val` image paths must be disjoint; shared training/validation PIDs are not automatically an error. No source validation/model-selection loader is used in the primary terminal policy. Query/gallery share identities by design; preserve their established retrieval semantics.

Path checks cannot prove absence of cross-dataset duplicate content or the same real person under unrelated PIDs. Record the limits of path, inode and content-hash checks. Do not claim cross-dataset identity disjointness beyond the namespace protocol without independent evidence. Detailed checks and synthetic acceptance tests are specified in the implementation plan.

## Held-out evaluation and Stage A comparison

After terminal completion, reconstruct the model with its **combined-source** class count, model variant and embedding dimension. Do not rebuild classifier dimensions from the target. Use [checkpoint reconstruction](../reid/utils/checkpoint.py), existing [feature extraction/evaluator](../reid/engine/evaluator.py), [distance](../reid/metrics/distance.py) and [ranking/filtering](../reid/metrics/ranking.py) unchanged.

The separate evaluation stage opens only the excluded target's original query/gallery. Preserve raw target PID/camera semantics, same-person/same-camera filtering, source-owned image preprocessing and normalization, Euclidean distance, Rank-1/5/10, mAP and mINP. CUHK03 uses the available detected processed-partition semantics with its provenance limitation; do not invent official-new validation.

Store four fold-aware results per architecture separately from Stage A's 4×4 matrix, under an isolated versioned Stage B namespace. Do not coerce three sources into a single-source matrix row or pass Stage B records to unchanged Stage A model-selection logic.

For each target and architecture, report Stage B against each of its three Stage A single-source→target results, their arithmetic mean and their best result. For every metric stored as a fraction, `difference_pp = 100 × (StageB - reference)`. Report a per-metric best-of-three envelope and also identify the single Stage A run best by mAP (Rank-1 tie-break); do not imply that an envelope of different runs is one model. Missing/non-comparable reference cells remain explicit; do not silently average fewer than three or replace them with within-domain results.

This compares **complete training strategies**, not the isolated causal effect of domain diversity. Data size, identity count, primary updates, source exposure, checkpoint selection and historical ResNet50 loss settings differ. Never attribute an improvement to diversity alone. Evaluation outputs must not feed back into this campaign's recipe, budgets, fold-specific selection or retraining decisions.

## Versioned manifests and provenance

Future machine-readable manifests, separate from this planning phase, must record:

| Group | Required information |
|---|---|
| Identity | Schema/protocol version, architecture, fold, ordered sources, excluded target, seed, run ID |
| Data | Configured/canonical roots, format, split/options, partition/list hashes, image-inventory/content-hash policy, image/identity/camera counts, CUHK03 provenance |
| Mapping | Recoverable PID map, canonical serialization version, digest, sample-index/source sidecar |
| Recipe | Resolved training config and SHA256, pretrained artifact/hash, sampler, losses, Center convention, optimizer, precision, schedule |
| Execution | Planned/actual primary and auxiliary updates, attempts, partial epoch/cursor, successful/attempted images, realized per-source sample and identity exposure |
| LR | LR used versus LR prepared for the next update; group names, boundary events and recoverable trace |
| Environment | Code commit and dirty diff identity, dependency environment, host/GPU, timing and interruption/recovery history |
| Completion | Completion reason `budget_completed`, terminal count/checkpoint/hash, checkpoint schema and recovery capability |
| Evaluation | Separate resolved evaluation config/hash, target split/query/gallery hashes, normalization/distance/filter settings, metric values and artifact paths |

The immutable frozen training-config hash identifies the recipe. Resume paths and allowed execution overrides are separately recorded and validated; they must not silently redefine the frozen scientific config. Official Phase 10 runs use a frozen implementation commit and reviewed config/data manifests. Volatile timing/path fields are not part of identity-map canonicalization.

Use an isolated namespace such as `exp/lodo/v1/<architecture>/<fold>/seed42/<run_id>/`, with separate training, recovery, terminal and evaluation artifacts; exact names are an engineering decision. Canonical collision checks apply across worktrees and symlinked paths. Do not overwrite Stage A directories, reuse an unverified interrupted directory, or modify its results registry.

## Limitations and readiness

One seed does not establish variability or statistical significance. Extra seeds are a resource-dependent follow-up, not completed work. Unequal source contributions, different effective passes, single-source versus multi-source data size, Stage A selection/loss differences and uncertain CUHK03 official-new provenance constrain interpretation. Fixed budgets do not prove convergence. Larger classifiers/centers may affect runtime and VRAM; no LODO GPU measurement exists.

Planning is **READY FOR IMPLEMENTATION** with explicit gates in the companion plan. It is not ready for official training. Center persistence, exact stopping/recovery, mapping/preflight, target-free orchestration, Stage A regression and GPU smoke validation remain required. A strict official-new CUHK03 claim remains blocked without evidence. Stop after the three planning documents; await explicit approval before Phase 3 and explicit approval before official Phase 10 training.
