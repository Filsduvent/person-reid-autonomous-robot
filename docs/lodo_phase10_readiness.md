# Phase 10 preparation and guarded-launch readiness

Prepared 2026-10-11 (client date); evidence timestamps are UTC. **READY FOR LAUNCH APPROVAL REVIEW** — preparation only. No frozen implementation commit, resource allocation or permission to train is claimed. Full evidence is under `lodo_phase10_evidence/`; canonical science remains `lodo_experiment_protocol.md`.

## A. Repository state and preservation

VERIFIED FROM CODE / EXPERIMENT ARTIFACT: isolated checkout `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo`, branch `feature/lodo-v1`; original `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot`, branch `main`. Both observed HEADs are `3b015c964b99bfe8ad4b7842bd174d3bc58af607`. This is the historical base, **not** a commit containing the uncommitted LODO implementation. Nothing was staged, committed, pushed, switched, stashed or reset.

Both initial complete statuses and all pre-existing source hashes are in [pre-edit-state.json](lodo_phase10_evidence/pre-edit-state.json). Main already had modified `docs/preliminary_results_log.md`, `plan.md` and seven untracked protocol/PCB registry documents. Those changes belong to the user and remain untouched. The feature tree already had nine tracked modified files (460 insertions/80 deletions), plus the earlier Phase 3–9R.3 untracked implementation/tests/docs. Phase 10 preserves that work; existing production implementation bytes are unchanged.

The only pre-existing feature file edited in this phase is `docs/lodo_decision_register.md`, by an evidence-supported append. New deliverables are eight YAMLs under `configs/lodo/v1/`, the CPU-only `scripts/lodo_official_preparation.py`, 25 tests in `tests/test_lodo_official_preparation.py`, three Phase 10 documents and preparation evidence. Exact new/modified membership, sizes, hashes, both final statuses and per-phase classification are in [repository-delta.json](lodo_phase10_evidence/repository-delta.json). [implementation-review.patch](lodo_phase10_evidence/implementation-review.patch) includes prior opt-in production changes and new Python modules/tests, including untracked implementation. [proposed-freeze-files.txt](lodo_phase10_evidence/proposed-freeze-files.txt) is a review list, not staging authorization. The complete delta includes ignored local JSON/log evidence. Existing `*.json` ignore rules remain unchanged; manifests/maps and selected preparation receipts require individually reviewed exact-path force-add after Git approval, while local logs stay preserved and ignored.

The final [preservation-closeout.json](lodo_phase10_evidence/preservation-closeout.json) links the artifact audit: all 4,629 recorded experiment file stat/membership records and 4,511 selected full SHA256s match the pre-edit baseline across both trees, including all feature `.pth` tensors/checkpoints and original PCB MSMT17 best/last. All 175 original source files and 33 original `results/` files are unchanged. All 233 pre-existing feature files are unchanged except the intentional decision-register append. No official run directory or historical experiment file was created/changed. Dataset metadata are revalidated through all eight original-loader preflights; images were neither copied nor decoded. This is bounded source/artifact/metadata evidence, not a full image-content audit.

Completed original PCB MSMT17 best epoch50 and last epoch120 remain protected: best SHA `5657281a3750b63f1335d63e1bdde387a82c6d8bc5708c60d9ef20a81efc6914`; last SHA `d9ce09debf72e99072d7e9a15d1f68e70c13ac39ba0bdaa8f1f6232ab27cda91`. Selected-best mAP 0.4339965879917145 and final-epoch mAP 0.413950651884079 remain distinct. The historical recorded run commit3b015c9 versus Phase20 freeze66ea88d provenance exception is retained for review, not silently reconciled.

## B. Eight executable candidates and provenance

VERIFIED FROM CONFIGURATION / LIVE METADATA: all eight actual production-schema and CLI parses, source-only loader constructions, global contiguous labels, mapping/sidecar alignment and exclusion checks passed. Counts match every planning expectation. No DataLoader was iterated during real metadata preflight; no target loader/image transformation was constructed.

| Fold | Ordered sources | Excluded target | Images | Identities / label range | Map SHA256 |
|---|---|---|---:|---|---|

| LODO-1 | duke, cuhk03, msmt17 | market1501 | 54,135 | 2,510 / 0–2509 | `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b` |

| LODO-2 | market1501, cuhk03, msmt17 | duke | 50,549 | 2,559 / 0–2558 | `13f82a58afe58a9df499126a53f6aa0ae48633c8bd036992f66fdc7c41ed2b15` |

| LODO-3 | market1501, duke, msmt17 | cuhk03 | 59,706 | 2,494 / 0–2493 | `5caf9706605d6626d557637713e794cf959f54cd2c8d9a6eebeb3c95a5b44970` |

| LODO-4 | market1501, duke, cuhk03 | msmt17 | 36,823 | 2,220 / 0–2219 | `332f999102894db105360a2765483b23a4653c52fec41443a7f5ff5a7b3618ff` |

| Architecture / fold | Primary / auxiliary updates | Scientific configuration SHA256 | Immutable candidate manifest SHA256 | Gate |
|---|---:|---|---|---|

| resnet50 / LODO-1 | 85,200 / 85,200 | `3e00961d17d4158163cb311ddcf821834fd12a8fd47cd6b9a48f574bc5c7d25c` | `81d1ae641d088e6cb1f25814216782ba51d5b936a05fb63be9b2da45b39d42cf` | PASS candidate only |

| resnet50 / LODO-2 | 85,200 / 85,200 | `e85e95040d8469615d6d62915ee3bedc24dc1c73135b1420de3d59e01009ade2` | `694cacb5d038344ab7fc5316fcf15b1348b236fab44f09135266086bea6c0384` | PASS candidate only |

| resnet50 / LODO-3 | 85,200 / 85,200 | `e94328887c1abdef675afc974d68ee81a941d8af637e21ef3265ca912f908d1a` | `8ba2d151e7619d16345b4e46cbb7baa0bfd0e25097598cee75675c251c57b0f2` | PASS candidate only |

| resnet50 / LODO-4 | 85,200 / 85,200 | `47986b17b19373d09678718217da61a8b319a671450bdbb9d3e5f684ed5bad77` | `3f2eb1c5497421bad78c5470e0feafab37061decf19cd531bba912197550e7c7` | PASS candidate only |

| pcb / LODO-1 | 94,200 / 0 | `04251b99dca9a00cf13bc2e88e8a2fe47b752c07993c4773b7e6c0a987855d1e` | `4e87c209af6abdfc44ecf40a95b3df4de7608997f6fe875293621a510f56f19b` | PASS candidate only |

| pcb / LODO-2 | 94,200 / 0 | `0993e9454b345c6d204b89e8ed180e9b6a63e3fdef607966a3837e73094351ec` | `32e951f16add5faa3e4fe6d86e5f41e5d06831e4ea6997f2f4ab425b2d2676c0` | PASS candidate only |

| pcb / LODO-3 | 94,200 / 0 | `e0f59a7689f1a5bb856f3e08ad4b591089f1f016557ad065db6335041229b839` | `1b656e8db9cc3027b9f6e63e80ddddbbde89ef226edb63fbe3b0cd2c267d5370` | PASS candidate only |

| pcb / LODO-4 | 94,200 / 0 | `d53b5b2c7345e43d05e3ea3d2809efe73be66f92973c70f775496b04332d87c1` | `01c474466262d7604119e0843f7b27160c358c4077b7ba39f3fb7c7e4a6a1bc4` | PASS candidate only |

Exact YAML paths and full YAML-file SHA256s are in [candidate-index.json](lodo_phase10_evidence/candidate-index.json); full per-run immutable provenance is in `lodo_phase10_evidence/manifests/`. Eight separate future evaluation requests and four recoverable canonical maps are adjacent. Proposed run namespace: `exp/lodo/v1/<architecture>/<LODO-n>/seed42/official-candidate-20261011`; no leaf is reserved/created now.

Live sources: Market12,936/751; Duke16,522/702; CUHK03 7,365/767; MSMT17 30,248/1,041 images/IDs. Market/Duke processed trainval, CUHK03 detected processed trainval/processed_partition/null, MSMT17 V2 raw train/list_train only. Processed trainval is its authorized train+val union; MSMT list_val/query/gallery and excluded-target images are prohibited. PIDs are source-exposed identifiers, namespaced by dataset; an unavailable official raw PID is not claimed recovered. Stable sample sidecars preserve original source-local cameras and paths. Source shares are inventory shares, not presumed exposure shares.

Partition/list SHA256s: Market `3853cfdc9be6814d0193b9aad8025b61133d7f7b9ef8449b5d0f2964811a53ab`; Duke `25929417e0a8780548924d4f58f4f644ee18098e957df41a571650eb05e6b638`; CUHK03 detected `a1fd9a06748dd217325820b2ed5c5814de364a6b5e91cb7a93e1924915034f97`; MSMT train `00930d5911e66ec6ac4d9b177e61b49433658758cc091231bf7cfcb83c70d0bf`. Manifests additionally pin all source/target/prohibited inventories and sidecar/data hashes. Canonical data root: `/mnt/hd-data/addirakoze/Projects/Dataset`.

Cached pretrained full hashes were read, with no downloads: ResNet V2 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`; historical PCB ImageNet `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`. Eight CPU model/criterion/optimizer/scheduler constructions verified all classifier shapes and ResNet C×2048 Center dimensions without pretrained initialization, forward/backward or updates. Exact LR-used points include ResNet500=.000299406,501=.0003,28401=.00003,49701=.000003; PCB31401 preserves backbone/new ratio10:1. Science, original transforms/samplers, FP32, losses and successful-update budgets are unchanged.

Manifest schema1 uses canonical UTF-8 sorted compact JSON/noNaN, with a SHA256 over `immutable`; timestamps, proposed output, allocation/approval/resume declarations are operational. Production scientific identity and canonical map schema remain their existing definitions. Code records `dirty_worktree_unfrozen`, base HEAD above, `frozen_commit:null`, and a custom implementation source membership/hash snapshot. That snapshot is not a Git tree or commit. Source refinement is explicitly recorded in `candidate-source-refinement.json`. No manifest asserts approval or allocation. The verifier rejects config/runtime/weight/code drift, unknown provenance, wrong same-C mapping, stale sidecar, invalid output and target/fold mismatches.

## C. Deterministic CUDA and recovery compatibility

VERIFIED FROM CODE / HISTORICAL ARTIFACT: propose `strict_cuda_deterministic_v1`, workspace `:4096:8` before imports/context, benchmarkfalse, cuDNN deterministictrue, deterministic algorithmstrue/warnonlyfalse, recorded TF32 controls, FP32 and zero workers. Stage A defaults/schema1 remain intact; strict checkpoint schema2 refuses incompatible runtime/policy resumes. Actual CUDA capability is `epoch_boundary_rollback_cuda_v1` (the raw generic loader contract retains its CPU base capability; the CUDA checkpoint identity overrides it explicitly).

Reverified 18 final Phase9R.3 evidence hashes and 16 executed implementation/harness source hashes. Four actual schema2 boundary/terminal artifacts were read on CPU. A/B/C events (ResNet84, PCB66) and rollback replay compare exactly; terminal no-op is true. Historical failures under the original policy remain failures. [qualification-reverified-final.json](lodo_phase10_evidence/qualification-reverified-final.json) provides current verification.

The result covers LODO-1, five updates, two-batch fixture epochs, zero workers, the exact TITAN/runtime. It does not qualify full real epochs, other folds, long runs, other workers/devices or arbitrary cursors. Exact saved-state restoration and bounded bitwise continuation are separate claims. Full model/buffers, both optimizers/Center where applicable, scheduler, RNG/sampler ownership, counters/exposure and position are checked CPU-first before mutation/CUDA; RNG restores last. Accepted counters/exposure roll back with state, avoiding double-counting.

All nine requested documents were read fully. Earlier plan/readiness statements that a feature was not yet implemented, CPU-only/fresh-no-resume restrictions and initial numerical failures are dated historical states explicitly superseded by later addenda/D36–41, not instructions to alter current science. D17 original CUDA defaults are expressly amended by optional D36 strict policy. No active recipe contradiction was found. CUHK03 provenance, official freeze, cadence adoption, worker limits and allocation remain open decisions; no claim is silently upgraded.

## D. Cadence, retention, replay and storage

PROPOSAL, implementation already verified: checkpoint initial state and every exhausted complete epoch; retain latest two validated immutable generations plus a distinct terminal. `train.recovery={version:1,policy:completed_epoch_boundary,every_epochs:1,keep_generations:2}`. Official adoption requires approval. Final budget may end in a partial epoch; only its complete terminal is accepted. There is no resumable mid-epoch cursor or update-boundary checkpoint extension.

| Fold | PK initial; observed20 epochs | PK conservative maximum | PCB fixed batches | Normal replay maximum ResNet / PCB | Latest-corrupt fallback maximum ResNet / PCB |
|---|---|---:|---:|---|---|

| LODO-1 | 764; 762–766 | 782 | 845 | 782 / 845 updates | 1564 / 1690 updates |

| LODO-2 | 716; 714–717 | 727 | 789 | 727 / 789 updates | 1454 / 1578 updates |

| LODO-3 | 858; 854–859 | 874 | 932 | 874 / 932 updates | 1748 / 1864 updates |

| LODO-4 | 505; 504–508 | 519 | 575 | 519 / 575 updates | 1038 / 1150 updates |

PK epoch length can vary: each PID supplies max(1,floor(n/K)) chunks; each batch consumes P chunks, so floor(sum(chunks)/P) bounds its length. Residual-ID exhaustion only reduces it. Twenty metadata-only epochs use private instances of the unchanged sampler and do not establish the bound themselves. PCB floor(N/64) is fixed with original drop_last. A latest-valid boundary loses at most one bounded epoch; rollback past a bad latest pair loses at most two. More than one unusable retained generation has no claimed recovery guarantee.

| Architecture / fold | Measured trained checkpoint MiB (historical original policy) | Projected terminal+2 generations MiB | Compute-only normal / fallback replay hours |
|---|---:|---:|---:|

| resnet50 / LODO-1 | 348.14 | 1044.42 | 0.134 / 0.269 |

| resnet50 / LODO-2 | 349.68 | 1049.04 | 0.125 / 0.250 |

| resnet50 / LODO-3 | 347.64 | 1042.93 | 0.150 / 0.301 |

| resnet50 / LODO-4 | 339.07 | 1017.21 | 0.089 / 0.179 |

| pcb / LODO-1 | 233.52 | 700.57 | 0.185 / 0.369 |

| pcb / LODO-2 | 234.10 | 702.31 | 0.172 / 0.345 |

| pcb / LODO-3 | 233.34 | 700.01 | 0.204 / 0.407 |

| pcb / LODO-4 | 230.10 | 690.31 | 0.126 / 0.251 |

Eight-run retained checkpoint projection: **6.784 GiB**; not measured strict full-duration storage. Trained optimizer-state sizes, not initial empty optimizers, underpin it. Strict LODO-1 trained terminals are about348.16/233.54MiB for ResNet/PCB. Proposed campaign headroom **64GiB**, e.g.10GiB checkpoints/publication scratch,8GiB stdout,2GiB manifests/maps,44GiB safety; no bytes are reserved. Observed free space 3,361,166,323,712 bytes (~3.06TiB). Preserve corrupt/orphan history; accumulation is a stop/review condition, not permission to delete. Exposure/epoch history and logs grow; LR/index traces are bounded to32. Transient publication requires an additional payload/pair before pruning. No TensorBoard retention assumed.

Short strict save observations: ResNet8.249/14.117/17.394/10.688s; PCB8.296/11.931/14.133/9.412s. BranchB CPU compatibility/actual restore about8.025/8.224s and7.072/7.458s. These include the recorded storage/runtime scope, not sustained I/O guarantees. At the short sampled step time, typical every-epoch writes suggest roughly1.1–5.3% compute+save overhead depending on architecture/fold; shorter PK epochs, duplicate validation, startup, contention and filesystem stalls can cost more. Lost-work hours in the table exclude those costs.

Pair and generation-index publications are atomic individually, not a single transaction. Latest missing/corrupt indexed integrity pair can fall back to the prior validated pair. An explicit corrupt path, semantic identity mismatch, missing/bad index or all invalid generations fails closed. Orphans are not promoted. `ckpt_last.pth` is a logical index pointer and need not physically exist. Same-run canonical namespace is enforced. A valid terminal is a CPU-first no-op; incomplete terminal is refused. These behaviors are covered by existing CPU tests and actual v2-artifact checks; no official checkpoint exists now.

## E. Read-only resources and feasibility

VERIFIED FROM READ-ONLY NVML at 2026-10-12T00:58:41.613957+00:00: TITAN Xp UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`,12,288MiB total/5used/12,179free; GTX1080 UUID `GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5`,8,192total/5used/8,103free. Both utilization0%, no compute PIDs/owners listed. No active training/evaluation job was found in the initial own-process audit. Idle devices are not allocated. Driver570.211.01; Constantine/Python3.11.2/Torch2.7.1+cu126/torchvision0.22.1/NumPy2.1.3/CUDA12.6/cuDNN90501 match recorded core qualification versions. Complete dependency snapshot is `pip-freeze.txt`.

| Architecture / fold | Strict reserved GPU GiB evidence | Strict short img/s basis | Compute-only projection hours | Sustained throughput / end-to-end duration | Candidate allocation window |
|---|---|---:|---:|---|---|

| resnet50 / LODO-1 | 4.988, short LODO-1 measured | 103.36 | 14.65 | UNVERIFIED | 48h proposal, unallocated |

| resnet50 / LODO-2 | UNVERIFIED; LODO-1 reference 4.988 | 103.36 | 14.65 | UNVERIFIED | 48h proposal, unallocated |

| resnet50 / LODO-3 | UNVERIFIED; LODO-1 reference 4.988 | 103.36 | 14.65 | UNVERIFIED | 48h proposal, unallocated |

| resnet50 / LODO-4 | UNVERIFIED; LODO-1 reference 4.988 | 103.36 | 14.65 | UNVERIFIED | 48h proposal, unallocated |

| pcb / LODO-1 | 7.166, short LODO-1 measured | 81.40 | 20.57 | UNVERIFIED | 48h proposal, unallocated |

| pcb / LODO-2 | UNVERIFIED; LODO-1 reference 7.166 | 81.40 | 20.57 | UNVERIFIED | 48h proposal, unallocated |

| pcb / LODO-3 | UNVERIFIED; LODO-1 reference 7.166 | 81.40 | 20.57 | UNVERIFIED | 48h proposal, unallocated |

| pcb / LODO-4 | UNVERIFIED; LODO-1 reference 7.166 | 81.40 | 20.57 | UNVERIFIED | 48h proposal, unallocated |

Strict LODO-1 qualification maxima reserved ResNet4.988GiB/PCB7.166GiB, allocated approximately4.362/6.358GiB, leave roughly7.01/4.83GiB versus nominal12GiB, before other processes/context variability. Matched timing is only three measured updates: .6191774323/.7862352026s, +38.95%/+47.48% versus paired original. Full-duration allocator drift, real I/O, thermal state and other-fold strict peaks are UNVERIFIED. Historical original-policy all-fold ~6.70/7.52GiB allocator observations and GTX PCB~7.69GiB do not qualify strict recovery or substitute for all-fold strict measurements.

The 14.65/20.57h projections transfer LODO-1 strict step samples across folds for planning only, not measured lower bounds or confidence intervals. A proposed48h sequential window per run (384h across8 windows) is an explicit review candidate, not a reliable maximum or reservation. Before launch, approve sustained-throughput/strict-memory uncertainty with a conservative monitored window or separately authorize qualification; insufficient evidence stays an open admission gate. Requested fresh TITAN free>=10,000MiB and disk>=64GiB are extra operator headroom proposals. Fresh full-UUID ownership/process checks and actual lab allocation are mandatory. GTX is not an automatically qualified alternative. No GPU workload/allocation was performed in Phase10; recipes/tolerances are unchanged.

## F. CUHK03 and Stage A reference eligibility

VERIFIED FROM METADATA: local `splits_new_detected.json` SHA `d25235134237c6b84f5200098a53d264335c1ef530901067d486505f51b1bb70` and `cuhk03_new_protocol_config_detected.mat` SHA `ef438e7775fce6d08ef6dff84d2235bd4bed63850dbbf7025f3251e45c7fa721` exist. Processed train7,365/767, query1,400/700, gallery5,332/700 image/ID counts and sorted per-PID image counts match references. PID sets differ; no verified raw PNG/MAT→processed JPG/PID mapping/generator was found. Matching counts/signatures do not prove corresponding sample membership. **Official detected/new remains UNVERIFIED**. Preserve processed_partition/null, and seek advisor acceptance of this exact hashed-partition limitation or separately reviewed provenance work. No dataset metadata was repaired.

Actual Phase7 `_compatible_reference` rejects every required historical ResNet cross-domain cell:12 available across4 targets, zero eligible, first reason missing/different historical target split hashes; historical camera-filter evidence also absent. Four within-domain records are not substitutes. No PCB cross-domain record was found in inspected original result/experiment trees. Each fold therefore has ResNet3found/0eligible and PCB0found/0eligible of3required. No historical file is rewritten using current hashes. [stage-a-reference-eligibility.json](lodo_phase10_evidence/stage-a-reference-eligibility.json) records full paths/digests/reasons. All8 future requests truthfully use empty Stage A lists: standalone Stage B evaluation can be valid with incomplete comparison; a complete report needs exactly three qualified records per fold/architecture. Comparisons concern whole strategies, not diversity-only causal effects.

## G. Guarded commands and dry-run evidence

[lodo_official_launch_runbook.md](lodo_official_launch_runbook.md) contains concrete actual CLI commands for full source/config/map/weights/runtime checks; approved clean full commit and allocation assertions; exact TITAN UUID with logicalcuda:0; tmux sequential launch; unbuffered external logs; NVML/disk/durable counters; safe own-process interruption; logical-index same-run resume; incompatible/corrupt state handling; exact terminal and separately approved target evaluation. Installed tmux3.3a supports explicit environment forwarding, avoiding stale CUDA settings in an existing server. Launch/resume/evaluation blocks carry DO NOT EXECUTE BEFORE APPROVAL labels.

All6 bash blocks pass `bash -n`; all4 embedded Python blocks parse, without executing official commands (`runbook-syntax.json`). Actual train/evaluate `--help` and all8 training CLI parses ran CPU-only; schema, hashed evaluation request descriptors and actual code signatures were checked. Eight real metadata/manifest verifier calls passed with CUDA/image/target guards; `eight-preflights.log` records them. The verifier has no launch API. Its optional resume flag performs integrity/config/same-run namespace preflight through the production path; full tensor/optimizer/RNG compatibility is additionally enforced by production CPU-first validation before CUDA, not inferred from the candidate-only status.

Monitoring limitation is explicit: LODO does not invoke Stage A file/TensorBoard initialization; candidates disable those flags and the runbook captures stdout externally. Exact auxiliary/source counters are durable-checkpoint fields, not emitted on every console iteration. Wrong target access, nonfinite/partial step, hash/runtime mismatch, wrong/competing PID, OOM/kernel fault, disk/VRAM headroom violation and orphan/corrupt growth require stop/review. No killing other users' jobs, automatic recipe changes or historical artifact deletion.

## H. CPU tests and regression evidence

VERIFIED FROM EXECUTION: full suite **1,382 passed,0failed,4skipped in511.10s** (JUnit511.057s); focused Phase10 **25passed,0failed,0skipped in6.97s**. Exact receipts/JUnit/logs are `cpu-tests.json`, `cpu-full.xml/.log`, `cpu-focused.xml/.log`. Four explicitly CUDA-only model/train-loop cases skip with CUDA hidden; nothing else is excluded. The prior1,357 CPU passes plus25 new tests all pass; no baseline pytest failures or regressions observed.

Executed from isolated canonical feature checkout with interpreter `/home/addirakoze/environments/Reid/bin/python`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:tests CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8`:

```bash
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_official_preparation.py -q --junitxml=/tmp/lodo-phase10-focused.xml
timeout 1800 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests -q --junitxml=/tmp/lodo-phase10-full.xml
```

Tests cover8 actual configurations/CLI without dispatch, canonical immutable vs operational serialization, false freeze/allocation/schema/provenance, same-C alteredmap/sidecar/data, code membership/clean freeze, wrong runtime/UUID and prohibited images/target/CUDA. Full suite includes original single-source configuration/loaders/samplers/collator, model/loss/optimizer/scheduler/evaluation interface and all prior LODO data/update/exposure/recovery/terminal/comparison contracts. No sampler/model/training engine was edited in this phase. `git diff --check` passes.

One independent audit attempt failed on a Python literal equality expectation (.01 versus frozen `.1*.1` representation), not a training result or tolerance test. Original attempted source/log and partial receipt are preserved. Only the audit expectation was corrected to exact existing config arithmetic; production, recipes and numerical CUDA tolerances were unchanged. Corrected audit passed (`independent-audit-final.log`). No automatic GPU retry occurred.

## I. Open decisions and explicit approvals

1. **Git freeze approval blocker:** Phase10§2 forbids automatic commit. Review complete earlier implementation and Phase10 additions using exact file list/diff, then explicitly authorize staging/one feature-branch commit. Require clean actual full new SHA, verified snapshot and separate operational launch receipt. Dirty base HEAD cannot satisfy immutable official provenance.
2. Accept or resolve the precise CUHK03 processed detected-partition/official-new limitation for dissertation claims and launch. No invented split ID or conversion is proposed.
3. Accept Stage B-only strict CUDA/FP32/zero-workers and bounded completed-epoch rollback/replay, every-epoch cadence/two validated generations+terminal, normal/fallback lost-work/storage/logging plan. If epochs are unacceptable, separately review a safe checkpoint extension; none is implemented here.
4. Explicitly review all-fold strict-memory and sustained/end-to-end uncertainties and the conservative candidate window; authorize separate measurements if required. Allocate the exact qualified TITAN UUID independently, then repeat ownership/free-memory/process/disk/runtime admission. Previous smoke authorization is insufficient. No reservation exists.
5. Explicitly approve each official launch after freeze/provenance/resource gates. Separately approve any resume and post-terminal real target evaluation. No Phase11 is implied.
6. Accept incomplete Stage A comparison status or separately qualify missing historical cells. Standalone Stage B and complete comparison eligibility remain distinct; original records must stay untouched.

## J. Classification and mandatory stop

**READY FOR LAUNCH APPROVAL REVIEW.** Eight internally consistent, hashed, CPU-validated candidates, demonstrated existing restricted recovery/storage proposal, full passing CPU regressions and preservation evidence support preparation completion. CUHK03 provenance is truthfully unverified, Stage A comparisons incomplete, all-fold strict memory/sustained duration unverified and all admission/approval conditions above remain open. This classification grants no permission and is not a claim of a frozen/ready-to-run official campaign.

No official training/resume, real held-out evaluation, GPU work, commit/push/merge, destructive operation, allocation or Phase11 occurred. Stop after this report. Git freeze, resource reservation, official launch and subsequent phase require explicit review/authorization.
