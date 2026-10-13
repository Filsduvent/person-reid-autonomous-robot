# LODO decision register

Version 1, `lodo_v1`, 2026-10-08. Read with the [scientific protocol](lodo_experiment_protocol.md) and [implementation plan](lodo_implementation_plan.md).

## Status and evidence definitions

- **FINALIZED**: the user has fixed the scientific choice or required behavior. It does not mean the code exists or validation passed.
- **CONDITIONAL / REQUIRES VALIDATION**: implementation or evidence must satisfy a specified gate before official training.
- **OPEN**: an operational choice or additional evidence remains unresolved. It cannot silently change finalized science.

Evidence labels: **USER-FINALIZED**, **VERIFIED FROM CODE**, **VERIFIED FROM CONFIGURATION**, **VERIFIED FROM EXPERIMENT ARTIFACT** (prior audits), **CALCULATED FROM VERIFIED METADATA**, and **NOT VERIFIED**. No LODO implementation, training, evaluation or GPU validation is complete. No formal approval beyond the user's established scientific choices is inferred.

## Finalized scientific decisions

| ID | Decision / status | Evidence and rationale | Implementation implication | Approval requirement |
|---|---|---|---|---|
| D01 | Stage A preservation — **FINALIZED** | USER-FINALIZED; existing selections, 4×4 matrices and PCB work remain historical evidence | Opt-in defaults; separate branch/worktree and outputs; never rewrite existing configs/artifacts | No changes authorized; any exception requires explicit scope review |
| D02 | Four complement folds — **FINALIZED** | USER-FINALIZED: LODO-1 Duke/CUHK/MSMT→Market; 2 Market/CUHK/MSMT→Duke; 3 Market/Duke/MSMT→CUHK; 4 Market/Duke/CUHK→MSMT | One implementation, canonical `market1501,duke,cuhk03,msmt17` order, three distinct sources | Changing membership requires protocol approval |
| D03 | Virtual combination — **FINALIZED** | USER-FINALIZED; original loader capabilities verified in `reid/data/build.py` | Delegate image loading/transforms; no copied images or manually combined dataset | Implementation starts only after Phase 3 approval |
| D04 | Source splits — **FINALIZED** | USER-FINALIZED; processed trainval Market/Duke/CUHK detected; raw MSMT train | Freeze exact descriptors and hashes; MSMT `list_train.txt` only; see D20 for CUHK provenance | Split/data changes require review |
| D05 | Global identity namespace — **FINALIZED** | USER-FINALIZED; loader exposes source PIDs/local labels | Key `(dataset_key, original_pid)`; ordered contiguous labels, recoverable map and SHA256; classifier/Center same map | Map integrity is a mandatory launch gate |
| D06 | Public sample/camera contract — **FINALIZED** | CODE: `ReIDTrainSample(image,label)`, original collator; target PID/camera/marks | Domain/index metadata in attributes/sidecar; source-local cameras preserved; original target semantics | Public API or retrieval changes need review |
| D07 | Architecture-specific sampling — **FINALIZED** | USER-FINALIZED; CODE: PK sampler and random loader | ResNet PK16×4/batch64; PCB random shuffled batch64/drop_last; no primary domain balancing | Sampling changes require scientific review |
| D08 | Fixed successful-primary budgets — **FINALIZED** | USER-FINALIZED from workload investigation; not convergence optima | ResNet 85,200; PCB 94,200 in every fold; main/aux counted separately | Budget changes require explicit approval |
| D09 | ResNet Candidate C — **FINALIZED** | USER-FINALIZED; prior loss audit: reference-oriented choice, mixed Center effects, no unsmoothed-Center result | Stride1/GAP/BNNeck/2048-D; input256×128; ID1/ε.1 post-BN; Triplet1/margin.3 pre-BN; normalized post-BN evaluation | No further candidate selection needed; changes need protocol review |
| D10 | Local Center convention — **FINALIZED** | CODE: `reid/losses/center.py` uses `0.5×mean` squared distance; differs from author code | Center enabled, C×2048, raw/pre-BN, weight .0005; preserve local reduction, document adaptation | No automatic upstream normalization correction |
| D11 | Center optimizer — **FINALIZED** | CODE: separate SGD and inverse-weight gradient rescaling | LR .5, momentum0/decay0; one auxiliary step per successful FP32 primary update | Synchronization/persistence tests required; no disabling Center for convenience |
| D12 | ResNet Adam / LR — **FINALIZED** | USER-FINALIZED; CODE/configuration: Adam .0003, decay .0005, existing warmup semantics | Updates1–500 warmup; 501–28,400 .0003; 28,401–49,700 .00003; 49,701–85,200 .000003; Center LR fixed | Boundary validation mandatory; values not open |
| D13 | Warmup boundary convention — **FINALIZED** | CODE: `WarmupMultiStepLR.get_lr()` uses completed updates/500 | Update1 factor.01; update500 .99802; update501 1; preserve denominator500 | Tests must verify LR used; do not replace denominator with499 |
| D14 | PCB architecture/loss — **FINALIZED** | USER-FINALIZED; CODE/config: independent parts, six heads, six summed CE losses | 1536-D retrieval; input384×128; ID1 only; no smoothing/Triplet/Center | Preserve Stage A PCB presets and representation |
| D15 | PCB SGD / LR — **FINALIZED** | USER-FINALIZED; CODE/config: prefix group policy | Momentum.9/decay.0005/Nesterovfalse; backbone/new .01/.1 through31,400, .001/.01 from31,401 through94,200; no warmup | Group and boundary gate required |
| D16 | Terminal target-free policy — **FINALIZED** | USER-FINALIZED; prior audits identify unconditional test loader as blocker | No target loader during training; predetermined terminal after exact budget; separate original-query/gallery evaluation | Official launch prohibited until isolation/terminal gates pass |
| D17 | Seed/precision initial campaign — **FINALIZED** | USER-FINALIZED seed42; FP32 primary/Center synchronization contract | Eight official training runs planned; AMP disabled; preserve existing nondeterministic/benchmark settings and record them | Additional seeds/AMP require separate resource/validation review |
| D18 | Evaluation/comparison semantics — **FINALIZED** | USER-FINALIZED; existing evaluator/filter/metrics code | Four separate results per architecture; compare each target to three Stage A source→target cells, mean and best; differences in pp | No target-driven tuning; no claims of diversity-only causation |
| D19 | Documentation-only stop — **FINALIZED** | USER-FINALIZED strict constraints | Only these three planning files; no implementation/tests/configs/branch/worktree/jobs/commits/pushes now | Explicit approval to begin Phase 3; separate explicit official-training approval in Phase 10 |

## Required validation and evidence gates

| ID | Decision / status | Evidence and rationale | Implementation implication | Approval requirement |
|---|---|---|---|---|
| D20 | CUHK03 official-new provenance — **CONDITIONAL / REQUIRES VALIDATION** | CODE: `_validate_protocol()` rejects `new` and non-null split ID; flat partition lacks provenance; detected-new intent is unverified | Use truthful processed_partition/null descriptor and detected images; freeze hash; obtain independent evidence for official-new claim or retain explicit limitation | Official launch review must acknowledge limitation; strict-new requirement blocks launch without evidence; changing data needs approval |
| D21 | Dataset inventories and leakage checks — **CONDITIONAL / REQUIRES VALIDATION** | METADATA: counts/hashes in protocol; zero constituent train/val and trainval/test filename overlap; canonical/content/semantic checks incomplete | Launch-time descriptor/path/symlink/inode/hash checks; authorized trainval union is not a disjoint validation set; no source query/gallery or target data | Fail closed; discrepancies block freeze/launch |
| D22 | Map/class dimension enforcement — **CONDITIONAL / REQUIRES VALIDATION** | CODE accepts dynamic C; CALCULATED counts 2510/2559/2494/2220; adapter absent | Stable original/local/global map, digest and sidecar; all six PCB classifiers and ResNet/Center dimensions verified | Same-C mismatched maps must reject resume |
| D23 | Exact-update stopping / scheduling — **CONDITIONAL / REQUIRES VALIDATION** | CODE current loop is epoch-complete and scheduler builder converts epoch milestones using loader length | Opt-in successful-update counter, stop before extra step, partial averages, used/next LR and fixed update units | Synthetic boundary and Stage A regression gates before GPU |
| D24 | Center checkpoint persistence — **CONDITIONAL / REQUIRES VALIDATION** | CODE and prior artifact audit: optimizer saved, learned centers omitted | Versioned Center module plus optimizer state; C/D/branch/reduction/weight/fold/map/config checks; legacy model-only evaluation preserved | Mandatory before official ResNet training; no false legacy exact continuation |
| D25 | Exact stochastic recovery — **CONDITIONAL / REQUIRES VALIDATION** | CODE lacks RNG/sampler/cursor state; multiworker prefetch complicates continuation | Model/optimizers/scheduler/RNG/order/cursor/exposure counters; complete main+aux boundary; demonstrate next-update and multi-step equivalence | Review validated capability; reject unsupported exact requests and silent trajectory changes |
| D26 | Target-free orchestration / terminal status — **CONDITIONAL / REQUIRES VALIDATION** | CODE `scripts/train.py` constructs test loader even when save_best false | Guard construction/evaluation; terminal completion reason/count/hash; no target-derived reconstruction C | Tests must prohibit target loader/evaluator calls; false completion blocks launch |
| D27 | Exposure / manifest / comparison implementation — **CONDITIONAL / REQUIRES VALIDATION** | CODE has per-batch logs but no complete fold-aware successful-update/source provenance contract | Versioned recoverable map, source indices, counts, LR trace, frozen config/data/code/environment/checkpoint/eval identities; separate Stage B records | Synthetic accounting/report tests; no Stage A record overwrites |
| D28 | Stage A compatibility — **CONDITIONAL / REQUIRES VALIDATION** | Existing configs/code/tests are evidence anchors; no implementation regression run here | Default epoch mode, loss formula, PCB groups, public data API, legacy checkpoints/evaluator unchanged | Phase 8 regression gate mandatory |
| D29 | GPU feasibility — **CONDITIONAL / REQUIRES VALIDATION** | No multi-source LODO GPU measurements; Center tensor estimate 17.34–19.99 MiB plus gradients; heads/activations also consume VRAM | Phase 9 isolated short smoke runs for both architectures/all C values, observed memory/throughput/recovery; no retuning from target metrics | CPU gates and resource review before smoke; official-launch approval remains separate |
| D30 | Campaign freeze — **CONDITIONAL / REQUIRES VALIDATION** | Science final; implementation/configs/manifests absent | Freeze implementation commit, eight configs, data/map/pretrained hashes, environment and guarded commands after Phases3–9 | Explicit approval immediately before official training |

## Open operational decisions

| ID | Decision / status | Evidence and rationale | Implementation implication | Approval requirement |
|---|---|---|---|---|
| D31 | Recovery capability and worker configuration — **OPEN** | Exact mid-epoch stochastic continuation not demonstrated | Prefer exact replay; otherwise tested restricted boundaries/rollback with explicit limitations; choose worker/prefetch strategy from evidence | Accept capability before official launch; no silent changes to finalized sampling/recipe |
| D32 | Recovery cadence and retention — **OPEN** | Larger checkpoints and interruption risk; no LODO storage/runtime measurement | Choose update cadence and retention/storage after smoke evidence; preserve terminal artifacts and interruption history | Review resource plan before freeze |
| D33 | Dedicated GPUs and launch order — **OPEN** | No LODO resource reservation or measured fold throughput | Explicit system.gpu_id and unique canonical outputs; protect ongoing experiments | Resource review, then explicit official-launch approval |
| D34 | Extra seeds / later ablations — **OPEN** | One seed limits inference; Candidate B untested; no follow-up resource commitment | Additional campaigns only if approved/resources permit; predefined budgets and no target-tuned recipe changes | Separate scientific/resource approval; never report them as completed |
| D35 | Artifact/schema naming details — **OPEN** | Protocol fields specified, implementation serialization not finalized | Proposed ckpt_terminal.pth and isolated versioned namespaces; choose stable names/canonical hash encoding in Phase3/6 | Implementation review; must preserve integrity contract |

## Contradictions and limits that must remain visible

1. **CUHK03 detected/new versus available loader:** the descriptor conflict is explicit in D20. No change of approved data or invented official-new claim is authorized.
2. **Warmup wording versus endpoint:** “500 warmup updates .01→1” means current boundary semantics: .99802 at update500 and 1 at501. This clarifies the approved request; it does not change the schedule.
3. **Trainval versus no train/validation overlap:** disjoint constituent train/val lists are required; their authorized trainval union intentionally includes both. It cannot also serve as an independent selection set.
4. **Reference-oriented Candidate C versus empirical evidence:** the choice is finalized, but mixed Center effects and Center-disabled smoothing ablations do not establish superiority. The local half-mean differs from upstream and stays explicit.
5. **Fair workload versus causal attribution:** fixed updates do not equalize images/identities, effective passes or compute across architectures; Stage A comparisons concern complete strategies.
6. **Exact recovery versus GPU numerics:** stochastic state/order recovery and deterministic-fixture next-update equivalence must be tested. Inherited nondeterministic GPU execution does not establish bitwise numerical continuation; record the demonstrated capability and review any stronger requirement or execution change.

Readiness classification: **READY FOR IMPLEMENTATION** because the scientific protocol and required engineering/validation gates are specified. **Not ready for official training.** All conditional gates remain pending; no test, smoke run, implementation phase, executable LODO configuration or launch is reported complete. Stop after validating these three documents and await Phase 3 approval.


## Phase 9R.3 implementation addendum

This addendum records the later user-authorized implementation phase. The dated
Phase 2 planning tables above are retained as historical decisions and pending-gate
wording; they do not describe current implementation completeness. Full evidence:
[lodo_cuda_policy_integration.md](lodo_cuda_policy_integration.md).

| ID | Current decision/status | Verified evidence and limit |
|---|---|---|
| D36 | Explicit Stage B strict CUDA policy — **IMPLEMENTED, CPU VERIFIED** | User Phase 9R.3 authorizes `strict_cuda_deterministic_v1` as opt-in. Actual LODO training activates it before CUDA discovery, requires prelaunch `:4096:8` workspace, checks unchanged TF32 values and rejects late initialization/control drift. Original defaults remain unchanged. This amends D17 only for explicit strict selection; seed42, FP32 and all scientific recipes/budgets remain frozen. |
| D37 | Strict policy/checkpoint compatibility — **IMPLEMENTED, CPU VERIFIED** | Strict identity/payload/manifest schema v2; original v1 preserved. Wrong policy, map/fold/config, runtime/driver/GPU/cuDNN and tensor contracts reject before restoration mutation. No historical checkpoints are rewritten or upgraded. Legacy Stage A readers remain intact. |
| D38 | Production-path CUDA recovery qualification — **CONDITIONAL, GPU DEFERRED** | Prior short diagnostic ResNet50 PASS remains historical candidate evidence. New production-path ResNet50 A/B/C and independent PCB A/B/C, actual process interruption/rollback, retained GPU generations and resource/overhead measurements are not executed. Fresh Phase 9R.3 exact-UUID authorization is pending; CPU doubles/skips are not GPU evidence. Original-policy numerical failures remain failures. |

D20 CUHK03 official-new provenance remains unverified; detected processed_partition
with null split ID remains the truthful descriptor. D31/D32 worker capability,
cadence/retention and D33 dedicated official allocation require review before D30
freeze/launch. CUDA recovery is restricted to completed persisted epoch boundaries,
zero workers and the bounded recorded environment. No arbitrary mid-epoch, all-worker,
all-hardware/version or long-run bitwise guarantee follows from five-update diagnostics.

Current Phase 9R.3 classification: **PARTIAL — GPU AUTHORIZATION REQUIRED**.
No official LODO training, Phase 10, benchmark evaluation, commit or push occurred.


## Phase 9R.3 authorized CUDA continuation — verified closeout

The preceding D36–D38 CPU-only closeout remains historical at its original time.
The later explicit exact-TITAN allocation authorized this continuation; current
evidence and capability limits are in [lodo_cuda_policy_integration.md](lodo_cuda_policy_integration.md).

| ID | Current decision/status | Verified evidence and limit |
|---|---|---|
| D39 | Phase 9R.3 authorized CUDA recovery — **VERIFIED, BOUNDED PASS** | Actual LODO orchestration for ResNet50 and independently gated PCB; same exact update2 checkpoint per architecture, A never restored, B/C independent; all nine pair/update comparisons3–5 bitwise exact. Original tolerances unchanged. TITAN exact UUID/runtime, FP32, zero workers, LODO-1, two-batch fixture epochs and five-update diagnostic scope only. |
| D40 | Process rollback and retained v2 artifacts — **VERIFIED** | Separate child SIGTERM after completed3; last persisted boundary2, replay to5, 320 accepted images and no double-counting. Separate terminal no-op, two immutable generations, actual v2 artifact wrong-map/fold/config/policy/runtime rejection, corruption/missing fallback and index-publication fault checks passed on CPU copies. No arbitrary cursor/full-epoch/all-worker guarantee. |
| D41 | Resource and policy timing — **MEASURED, LIMITED** | Frozen batch/resolution/recipes retained; allocator/device samples, checkpoint save/restore costs and matched original/strict three-observation timing recorded. Sequential short measurement has order/thermal/cache and scope limits; original policy timing does not qualify its recovery. |

Current classification: **COMPLETE — READY FOR PHASE 10 REVIEW**. This supersedes
the initial GPU-authorization-pending classification only; historical failed
original-policy runs and the prior CPU-only evidence remain unchanged. Original
Stage A and completed PCB/LODO artifacts are preserved by source/status and bounded
full SHA256/stat checks. Official launch, provenance (including CUHK03 official-new
claim), cadence/retention and dedicated official allocation still need review.
Stop here: no official training, held-out evaluation, Phase 10, commit or push.


## Phase 10 preparation — verified candidates, approval pending

D39–41 and all original-policy failure records remain historical and unchanged.
Preparation evidence and explicit limits are in [lodo_phase10_readiness.md](lodo_phase10_readiness.md).

| ID | Decision/status | Evidence and remaining gate |
|---|---|---|
| D42 | Eight official candidates — **CPU VERIFIED, NOT FROZEN** | Actual source inventories match all four folds; original schema/CLI, maps/sidecars/pretrained/source hashes and eight manifests pass. Dirty base3b015c9 is not an implementation commit. |
| D43 | Official Git freeze — **EXPLICIT APPROVAL REQUIRED** | Complete pre-existing/untracked implementation review and proposed file list prepared. No staging/commit/merge/push; actual clean full SHA and separate operational receipt remain mandatory. |
| D44 | Strict policy and zero-worker CUDA boundary recovery — **PROPOSED OFFICIAL ADOPTION** | Reverified18 artifact/16 executed source hashes and four actual v2 checkpoints; bounded LODO-1/five-update/TITAN qualification only. Stage A defaults and numerical tolerances unchanged. |
| D45 | Every-epoch/two-generation retention plus terminal — **IMPLEMENTATION VERIFIED, OFFICIAL APPROVAL OPEN** | Existing integrity/index/failover/no-op behavior covered by CPU and actual historical artifacts. Formal PK/fixed PCB replay bounds, trained-size storage and64GiB headroom proposal documented. No arbitrary mid-epoch extension. |
| D46 | CUHK03 detected processed partition — **OFFICIAL-NEW STILL UNVERIFIED** | Local reference counts/signatures agree but no verified raw→processed sample/PID correspondence. Keep processed_partition/null; advisor acceptance or separate provenance review needed. |
| D47 | Historical Stage A comparison — **INCOMPLETE** | ResNet12 cross-domain records found/0eligible under actual Phase7 provenance validator; PCB0found. Never enrich originals with current hashes or average incomplete cells. Standalone Stage B evaluation remains separate. |
| D48 | Official resources/launch — **NOT AUTHORIZED** | Fresh read-only NVML finds idle devices, not allocation. Strict all-fold memory and sustained/end-to-end time unverified; conservative windows/headroom require explicit review and qualified full-UUID allocation. No GPU work/official run/evaluation/Phase11. |

Current Phase10 preparation classification: **READY FOR LAUNCH APPROVAL REVIEW**.
Full CPU1,382passed/4CUDA-onlyskipped; focused25passed. Candidates and documented
limits are reviewable, not launch permission. See the preservation closeout for
unchanged original Stage A/PCB and historical LODO artifacts. Stop pending explicit
Git freeze, provenance/resource decisions and official-launch approval.


## Phase 10F final user sign-off — 2026-10-11

Decision authority: explicit user Phase10F instruction; exact source SHA and scope in
[lodo_phase10f_evidence/authority.json](lodo_phase10f_evidence/authority.json).
Prior tables remain dated history. No factual claim is upgraded by approval.

| ID | Current decision | Limit / separate gate |
|---|---|---|
| D49 | Stage B `strict_cuda_deterministic_v1`, FP32, zero workers — **APPROVED** | Workspace before imports/context; benchmark false, deterministic true, warn-only false, recorded TF32/schema2. Stage A/schema1 unchanged; bounded five-update LODO-1/TITAN evidence only. |
| D50 | Completed-epoch rollback, every-epoch cadence, latest two independently validated immutable generations + distinct terminal — **APPROVED** | Existing production YAML/schema verified; no mid-epoch continuation. Integrity fallback only under existing policy; exact-budget partial-epoch terminal is complete-only. |
| D51 | CUHK03 detected processed_partition/null — **ACCEPTED WITH LIMITATION** | Exact hashes/counts; official detected/new correspondence still UNVERIFIED. User acceptance is not advisor/institutional sign-off; none supplied. |
| D52 | Incomplete Stage A references — **ACCEPTED, NOT A TRAINING BLOCKER** | ResNet 12 found/0 eligible; PCB 0 found in inspected records. No fabricated/backfilled history; separate future qualification/evaluation task; whole-strategy comparison. |
| D53 | Sequential TITAN plan/headroom/monitoring — **PROVISIONAL** | Exact preferred UUID, fresh at least 10,000 MiB free/64 GiB disk checks ; 48 h planning window not maximum/reservation. Strict all-fold sustained memory/time unverified; actual lab allocation pending. |
| D54 | One local `feature/lodo-v1` freeze commit — **AUTHORIZED AFTER GATES** | Complete reviewed explicit paths, full CPU/preflight/preservation gates; actual full SHA/cleanliness/frozen checks in separate operational receipt. No official train/resume/eval/GPU work, merge, push, tag or Phase 11 approval. |

See [lodo_phase10_final_signoff.md](lodo_phase10_final_signoff.md) for exact candidate identities,
current CPU evidence and first-run approval checklist. Phase 10 candidate immutable JSONs
retain truthful original prefreeze snapshot fields; NOT_APPROVED/NOT_ALLOCATED continues
to mean official launch/allocation status. No hash refresh or historical failure rewrite.
