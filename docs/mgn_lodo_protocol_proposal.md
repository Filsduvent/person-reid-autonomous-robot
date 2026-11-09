# MGN LODO protocol proposal — Phase19

Date:2026-11-09 (America/Sao_Paulo). **PROPOSED; scientific approval pending.** This is a review document, not a runnable configuration, frozen launch manifest or training authorization. The user's direction was: “let's continue i think better to use what we already did for the other already implemented”. It authorizes preparing this reuse-based proposal; it does not supply approval of numerical choices that had not yet been presented.

The proposal adopts the established ResNet50/PCB LODO experimental structure, integrity rules and operational design, while preserving MGN's own architecture/optimization recipe. Existing `lodo_v1` identities, whitelist, configurations, results and all completed MGN source experiments remain unchanged. Within-domain/cross-domain MGN results remain separate from LODO results.

## Proposed decisions

| Dimension | Proposed MGN choice | Reuse or deliberate adaptation |
| --- | --- | --- |
| Scientific protocol | `mgn_lodo_v2`; `protocol.name=multi_source_lodo`, `protocol.version=2`, architecture `mgn` | New explicit scientific branch; never add MGN to historical v1 |
| Folds/source order/partitions | Existing LODO-1–4, canonical Market/Duke/CUHK03/MSMT17 order | Reuse existing definitions and metadata, not copied/combined image folders |
| Labels | `(source_key, source_exposed_pid)` sorted canonically into contiguous global labels | Reuse mapping schema1 and exact existing map bytes; separate new training identity |
| Model | `paper_v3_global2048_independent_v1`, eight classifier heads,2048-D retrieval | Retain original qualified MGN topology; each head has actual combined-source C |
| Initialization | Fresh checksum-pinned historical ResNet50 ImageNet artifact, existing MGN initializer | No source-trained checkpoint, trial checkpoint or old LODO weights as initialization |
| Loss | Mean8-head CE + mean3-global batch-hard Triplet, weights1/1, margin1.2, no smoothing/Center | Preserve MGN recipe; do not copy PCB sum-only CE or ResNet50 Center recipe |
| Main optimizer | SGD LR0.01 uniformly, momentum0.9, decay0.0005, Nesterov=false, bias factor1/same decay | Preserve MGN; no PCB10:1 groups, Adam or warmup |
| Sampler/batch | Existing global PK P16/K4/B64, seed42; no domain balancing | Same PK policy as ResNet50; differs from PCB shuffled images |
| Stopping budget | **85,200 successful primary updates per fold**, zero auxiliary updates | Reuse ResNet50's rounded mean120-epoch PK workload reference; fixed across folds, not120 epochs per fold |
| Scheduler | Successful-update units; gamma0.1 after42,600 and63,900 completed updates; no warmup | MGN's original50%/75% decay proportions mapped onto the fixed budget |
| Checkpoint selection | Predetermined exact-budget terminal; no source/target metric selection or early stopping | Reuse existing target-free LODO policy; distinct recovery/latest and terminal artifacts |
| Train preprocessing | RGB384×128 bicubic; mean[.485,.456,.406], std[.229,.224,.225]; random horizontal flip; no crop/padding/erasing | Preserve MGN; do not impose ResNet50 augmentation or PCB normalization |
| Primary retrieval | **No test-time flip**, one global L2, Euclidean, no reranking, unchanged camera filtering and metrics | Align extraction/metric procedure with historical LODO; architecture-specific input/normalization remain disclosed |
| Precision/execution | FP32 tensors, AMP=false, seed42, strict deterministic CUDA policy, zero ordered/nonpersistent workers | Reuse current LODO operating design; different from MGN StageA benchmark=true/four workers |
| Strict controls | benchmark=false, cuDNN deterministic=true, deterministic algorithms=true/warn_only=false, workspace`:4096:8` before imports; matmul TF32=false/cuDNN TF32=true | Reuse `strict_cuda_deterministic_v1`; TF32 convolution capability differs by hardware and is disclosed |
| Recovery design | Completed-epoch rollback; initial/every-epoch persisted boundary; latest2 validated immutable generations plus distinct terminal | Reuse design; MGN/Oneiros numerical and filesystem implementation requires its own qualification |
| Host | Oneiros, dataset root `/home/addirakoze/nobackup/Projects/Dataset`; approved GPU identity/runtime established again before execution | No historical TITAN UUID, `/mnt/hd-data` binding or previous allocation is reused |

The85,200-update proposal is a workload/comparison policy, not a convergence optimum, matched compute budget, or an automatic consequence of completed StageA training. It matches ResNet50 primary updates and image draws; PCB retains94,200 updates and its own sampler/recipe. No architecture-only or domain-diversity-only causal claim follows.

## Four folds and roadmap numbering

| Existing fold | Ordered sources | Excluded target | Source images | Combined C | Roadmap official unit |
| --- | --- | --- | ---: | ---: | --- |
| LODO-1 | Duke + CUHK03 + MSMT17 | Market1501 | 54,135 | 2,510 | 22D /23D |
| LODO-2 | Market1501 + CUHK03 + MSMT17 | Duke | 50,549 | 2,559 | 22C /23C |
| LODO-3 | Market1501 + Duke + MSMT17 | CUHK03 | 59,706 | 2,494 | 22B /23B |
| LODO-4 | Market1501 + Duke + CUHK03 | MSMT17 | 36,823 | 2,220 | 22A /23A |

Roadmap A–D are execution units, not replacements for historical fold numbers. Sources remain Market/Duke processed trainval; CUHK03 detected/processed_partition/trainval/split_id=null; MSMT17 V2 raw `list_train.txt` only, excluding validation. Targets retain their original query/gallery definitions. CUHK03 correspondence to official-new/classic is **unverified**, including when historical prose called it new. This proposal preserves the exact partition limitation rather than correcting data or claims retrospectively.

Live source metadata counts and maps were rechecked without decoding images. Each newly reconstructed map equals the historical canonical JSON bytes and SHA-256. Reusing map bytes does not reuse a training identity, operational path binding or checkpoint. PID keys are loader-exposed processed identifiers, not claimed raw official identities. Source cameras stay unchanged. Source indices are concatenated in canonical order; sidecars retain source-local index, PID/local/global label, camera and image reference. Host-bound sidecars/inventories must be regenerated and bound on Oneiros; old path-bound sidecar digests are not silently portable.

## Independent budget and identity examples

An index-only fresh-seed42 trace of the unchanged PK sampler gives:

| Fold | Updates across120 reference epochs | Epoch containing update85,200 | Batch within that epoch |
| --- | ---: | ---: | ---: |
| LODO-1 | 91,670 |112 |411 |
| LODO-2 | 85,892 |120 |24 |
| LODO-3 |102,740 |100 |427 |
| LODO-4 |60,687 |169 |249 |

Mean reference workload is85,247.25; rounding to the nearest100 gives85,200, the existing ResNet50 budget. Fixed successful updates intentionally imply different effective epoch counts. The trace is a scientific workload example, not observed training or a guarantee that an unqualified recovery reproduces it. Training stops at the budget even inside an epoch; a partial-epoch terminal is complete-only, not a recovery boundary.

Per fold:85,200×64=5,452,800 accepted image draws and85,200×16=1,363,200 identity slots. Across four folds:340,800 primary updates and21,811,200 image draws; zero auxiliary optimizer updates. Record attempts, accepted steps, skipped/failed/discarded work, processed versus accepted source images/identity slots and unique identities separately. Do not infer source exposure from dataset-size ratios or count replayed discarded work twice. Per-source index-only exposure forecasts are preserved in [the evidence](mgn_lodo_phase19/metadata-budget-evidence.yaml).

| Optimizer update being executed | Completed updates before it | LR used |
| ---: | ---: | ---: |
|1 |0 |0.01 |
|42,600 |42,599 |0.01 |
|42,601 |42,600 |0.001 |
|63,900 |63,899 |0.001 |
|63,901 |63,900 |0.0001 |
|85,200 |85,199 |0.0001 |

Advance the successful-update scheduler once after each completed primary optimizer step. Failures do not advance accepted counters or scheduler. Serialized LR represents the next update; optimizer state restores before scheduler validation. This is an explicit StageB scheduler adaptation, not a rewrite of StageA's completed-epoch60/90 milestones.

Map example: LODO-4 labels Market0–750, Duke751–1452, CUHK031453–2219. Equal raw PID values in two datasets remain distinct identities; equal C with a different mapping is incompatible.

## Target isolation and checkpoint rules

Training constructs only the three authorized source loaders and their virtual union. Metadata-only leakage preflight may inspect target partition/list inventories in a separate stage; it cannot construct a target loader, transform target images, extract target features, compute target metrics or select a model. No periodic real-source validation is needed under the terminal policy. Never treat source trainval constituents as disjoint selection data.

All eight classifiers use combined-source C. Held-out evaluation reconstructs that same trained topology/C from metadata; it never rebuilds classifiers with target C. No target-derived updates, checkpoints, early stopping, tuning or post-hoc budget changes. Fresh ImageNet initialization is mandatory; none of the original four single-source runs is silently continued as LODO.

An initial recovery boundary and each exhausted epoch are published with model/buffers, optimizer/scheduler, successful and attempted counters, RNG/sampler ownership, exposure and immutable protocol/map/data identities. Preserve two independently validated generations, committed integrity metadata and original failure evidence; integrity fallback follows the reviewed existing policy. Interrupted accepted state rolls back to the last supported persisted boundary; distinguish discarded physical work from the retained trajectory. Terminal completion is separate and reopening it performs no optimizer step. No exact arbitrary mid-epoch or universal bitwise recovery claim is made.

## Proposed scientific and operational identity boundary

`mgn_lodo_v2` is a **scientific protocol** identifier. It is not the already existing operational binding schema2 or checkpoint schema2. The historical `lodo_v1` whitelist remains exactly ResNet50/PCB, its serialized scientific payloads and eight regression digests unchanged. Existing map schema1 and generic strict reconstruction/output contracts can be reused without upgrading old artifacts.

The new training scientific payload must contain exactly the following semantic groups under an explicit versioned schema; concrete implementation is Phase20 and must reject unknown/contradictory fields:

1. Identity schema version, scientific protocol ID, protocol name/version, architecture, fold, canonical ordered sources and excluded target, seed and logical new output namespace.
2. Each canonical source/target descriptor and portable partition/list/ordered-record identities; source C, map schema/digest and source-index ordering identity. Target descriptors are metadata-only during training.
3. Complete MGN variant/topology/output contract, embedding dimension, eight classifier dimensions/bias policy, initialization artifact ID/full SHA/head initializer; no absolute pretrained file path in semantic identity.
4. Complete loss and optimizer declarations, PK/sampler policy, batch64/P16/K4, successful primary/auxiliary budget, complete scheduler units/boundaries and no-warmup policy.
5. Train/test geometry, interpolation, normalization and augmentations; explicit retrieval `flip_aggregation=none`, global normalization, distance, reranking, filtering and Rank1/5/10/mAP/mINP policy.
6. Terminal checkpoint selection/no evaluation during training, precision/AMP, strict deterministic execution-policy controls and zero-worker restriction, recovery capability/cadence/retention and accepted accounting semantics.

Canonicalization reuses the established UTF-8/sorted-key/compact/noNaN representation and integer-valued-number normalization. New scientific payloads are domain-separated by `mgn_lodo_v2`; none is passed through the v1 whitelist or serialized as v1. The raw file hash additionally binds exact configuration bytes. Timing, approval/allocation assertions and run-leaf identifiers are operational, not fabricated scientific evidence.

An operational manifest separately pins the actual reviewed implementation commit/source membership, config file hash, interpreter/packages/driver/GPU UUID, current canonical dataset root, full host inventories and sidecar/map hashes, pretrained file bytes, private allocation/binding declarations, fresh run path and approved recovery scope. Checkpoints validate both scientific and host-bound data/runtime compatibility. Relocation, old UUIDs, equal C or matching portable split hashes alone do not authorize continuation. Do not copy old v1 operational approvals, manifests or checkpoint generations into the new namespace.

The held-out evaluation identity separately binds training scientific ID, exact terminal checkpoint SHA, combined-source map/C/model metadata, target parser/source-code release, partition/list/query/gallery ordered-record identities, source-owned pixels/extraction/normalization/distance/filter policy, metric implementation/units and precision policy. Runtime/hardware and exact output/command are preserved in its operational receipt. Any optional mean-flip result would require a different evaluation identity and result namespace, not replacement of the primary result.

Proposed isolated logical namespace: `exp/lodo/mgn_lodo_v2/mgn/LODO-<n>/seed42/<run-id>/`; evaluation records under a separate matching versioned result root. This is a path specification only; no production configuration, run directory, checkpoint or binding is created in Phase19.

## Retrieval and historical comparison compatibility

| Evidence family | Primary extraction | Compatibility / action |
| --- | --- | --- |
| Historical ResNet50 LODO v1 | No flip, global L2, Euclidean, no rerank, existing filters | Same proposed retrieval procedure;256×128/architecture preprocessing and Adam/Center recipe remain different; actual historical per-fold records/partition hashes still need import verification |
| Historical PCB LODO v1 | No flip, global L2, Euclidean, no rerank, existing filters | Same proposed retrieval procedure;PCB normalization/1536-D descriptor/CE/random sampling/94,200 updates remain different; no relabeling |
| Proposed MGN LODO v2 | No flip, global L2, Euclidean, no rerank, existing filters | Common primary procedure with MGN384×128 bicubic/ImageNet pixels/2048-D descriptor/PK/SGD/loss recipe disclosed |
| Completed MGN StageA matrix | Mean original/width-flip descriptors before global L2 | **Not primary-policy-matched** to proposed no-flip LODO; preserve all16 cells and numerical flags; no direct matched-policy improvement claim |

For MGN's primary LODO-versus-single-source comparison, separately authorize supplemental **no-flip** evaluation of the12 fixed original off-diagonal checkpoints/cells in a fresh namespace, with identical source-owned pixels and target partitions. No training, checkpoint reselection or overwrite is needed. This supplemental campaign is **not executed or implicitly authorized** by this proposal. Until those references exist and pass exact partition/filter/policy checks, comparison eligibility is incomplete; never average fewer than three compatible references, use a diagonal as a missing reference, or relabel the mean-flip matrix.

The existing standalone evaluator deliberately rejects source retrieval-policy conflicts, and the cross-domain evaluator preserves the checkpoint's mean-flip setting rather than importing a target override. A CLI `eval.flip_aggregation=none` is therefore not a sufficient supplementary workflow. A separately reviewed explicit evaluation-request adapter must pin the original checkpoint/pixels, declare the approved no-flip evaluation identity and preserve ordinary source-ownership behavior. This is future scoped engineering/evaluation work, not a bypass of current guards or permission to edit saved checkpoint configuration.

For each held-out target, retain the three eligible single-source references, their arithmetic mean and per-metric best-of-three envelope; additionally identify the single reference run best by mAP (Rank1 tie-break). Compute differences in percentage points as100×(LODO−reference). These are descriptive whole-strategy comparisons, not model selection using held-out scores. Preserve unmatched historical labels and missing hash/filter evidence rather than repairing old provenance.

## Phase20 implementation and Phase21 qualification scope to review

After explicit scientific approval and its review/commit boundary, implement a separate versioned protocol/dispatch path and MGN configuration validation; reuse pure mapping/data/sampler/accounting/checkpoint/retrieval utilities where their contracts fit. No MGN branch in the v1 whitelist, altered v1 hashes/budgets/configs or source-training presets. New strict loss/model/schedule/recovery validation must bind all8 heads/3 metric features, source C and map. Generic framework changes, if necessary, need focused MGN/baseline/PCB and frozen-v1 regressions and a new qualified release; old receipts remain historical.

Before official folds, propose the bounded Phase21 matrix: one real-combined-source FP32 qualification case for each of the4 actual class counts, plus one separately bounded rollback/replay case per fold and synthetic held-out plumbing with no real target training/selection. Each case/attempt has at most3 aggregate optimizer calls across its children and at most600 seconds per supervised worker; initial plus at most2 documented implementation-correction attempts, no capacity workaround. One recovery example uses1 initial boundary step plus1 uninterrupted continuation and1 restored replay step (3 aggregate), with a fixture epoch and separate non-authoritative identity. It qualifies only the tested boundary/scope, not full real epochs or long-run bitwise equivalence. Source/index/RNG accounting gets independent CPU oracles. No full epoch or official85,200-update run is a test.

Use current approved Oneiros identity/runtime, real source C, B64/P16/K4, strict FP32 controls and zero workers. Existing within-domain benchmark=true/four-worker feasibility or old ResNet50/PCB TITAN recovery does not qualify MGN strict behavior on this host. CUDA deterministic-operation failure, OOM, wrong identity or recovery discrepancy stops the affected gate; no silent batch/precision/worker/policy change or unqualified resume. Actual allocations and future per-fold execution scope must still be verified. Qualification-case execution requires its approved scope, not this proposed table alone.

Official Phase22 folds remain user-operated after corresponding accepted qualification and frozen handoff. Phase23 held-out evaluation is separately approved after terminal evidence; it must not feed back into training or selection. No LODO training/evaluation, GPU smoke, ONNX/runtime work, commit/push/merge or Phase20 implementation occurs in Phase19.

The extension's negative-case contract must reject: MGN presented as v1; reordered/extra/missing sources or included target; source/target path aliases and prohibited query/gallery overlap; duplicate or inconsistent PID/local/global labels; equal-C/different-map checkpoint substitution; any head/variant/class-count/embedding mismatch; altered losses, optimizer, batch, schedule, counters or terminal rule; off-by-one LR advancement and extra optimizer calls after the budget; target loader/metric use during training or selection; unbound flip/interpolation/normalization changes; mismatched runtime/worker/precision/GPU or CPU fallback; legacy/moved/corrupt recovery generations or mid-epoch/terminal resume; existing/symlink output collisions and unapproved scientific/operational bindings. Historical v1 identities and rejection tests must still pass exactly.

## Evidence, unresolved review and authorization

Read-only CPU evidence: source metadata and exact historical map equality for all4 folds; index-only reference/budget/exposure/LR examples; **4 passed,209 deselected** focused historical scientific-identity and flip/interpolation preservation cases. Full commands/logs and workload traces are in `docs/mgn_lodo_phase19/`. Deselection is intentional narrow preservation validation, not a full regression qualification. No image batches, CUDA contexts, model optimization or production configs were created.

The current proposal resolves candidate choices rather than assuming that “use the others” approves unseen MGN scientific settings. Required review is concrete: new version;85,200-update budget/42,600–63,900 schedule; preserved MGN training recipe; primary no-flip retrieval and separately pending no-flip StageA references; strict zero-worker/recovery design with bounded qualification limits; exact processed CUHK03 provenance limitation; target-free terminal policy and host-specific admission. User/advisor acceptance remains separate from CPU checks. Explicit approval is required before Phase20; no advisor acceptance, publication or standing GPU authorization is invented.

## Scientific approval and authoritative clarifications — 2026-11-09

The user explicitly approved the core `mgn_lodo_v2` protocol and authorized Phase20 implementation, tests, fixes, documentation and scoped local commits under the autonomous policy. This supersedes the historical pending-approval text above; it does not turn proposal examples into runtime bindings or authorize GPU qualification, official training, benchmark campaigns, ONNX/runtime work or eventual retrieval-result acceptance.

Authoritative implementation requirements:

1.85,200 successful updates is a reference workload, not matched compute or demonstrated convergence. There is no120-epoch cap. Retained per-source exposure and attempted/discarded/replayed work are separately logged.
2.One consumed batch produces one primary optimizer update. Scheduler, counters and terminal completion advance consistently; no additional optimizer call after completion. Boundary-used LR values are0.01 through42,600,0.001 through63,900 and0.0001 through85,200.
3.Only authorized source training partitions train the model. The target cannot choose checkpoints/recipes or adapt BatchNorm statistics. Target inference is eval/no-grad; test-time adaptation is prohibited.
4.The later no-flip campaign is12 cells using the four unchanged source checkpoints. Preserve original mean-flip records and source-ownership guards. Primary comparison uses all3 compatible references and their mean. Target-ranked best-of-three/envelopes are retrospective oracle summaries, never target-independent source selection. Preserve training-budget and metric-selected-versus-terminal differences.
5.Preserve numerical controls; record float32 parameter/input tensors and AMP=false separately from full-FP32 arithmetic. cuDNN convolution TF32 permission and actual hardware capability must be explicit; no silent precision-policy changes.
6.Single-seed and CUHK03 semantic-protocol limitations remain. The original focused preservation command ran4 tests (209 deselected); the separate MGN-under-v1 rejection command ran2 tests (211 deselected). Thus6 passing focused cases across2 actual commands;8 legacy digest vectors and8 official config digests are assertions, not additional test cases.

The approved new protocol uses the established common framework and single-command YAML interface; no separate MGN trainer or weaker v1 guards. Phase21 is prepared only, with resource approval and the existing short smoke bounds still required. Full Phase22 training remains user-operated. Phase20 completion means implementation readiness, not experimental validation, successful GPU recovery or approval of eventual scores. Separate user/advisor review status is not fabricated.
