# Official Stage B LODO campaign — Phase 10 freeze candidate

## Phase 10F current user sign-off — 2026-10-11

The [final sign-off](lodo_phase10_final_signoff.md) records the explicit later user instruction. It accepts the exact hashed CUHK03 detected `processed_partition`/null split with official-new correspondence **UNVERIFIED**, approves Stage B-only strict deterministic CUDA/FP32/zero workers and completed-epoch/every-epoch/two-validated-generation recovery plus distinct terminal, and accepts incomplete historical Stage A comparison eligibility. Scientific recipes, splits and budgets are unchanged. Earlier dated implementation/approval statements below remain historical. Advisor/institutional approval is not supplied or inferred. One local feature-branch freeze is conditionally authorized after all gates; official training, resume, target evaluation, GPU work, allocation, merge and push remain unauthorized. Actual commit identity is recorded in a separate ignored operational receipt, never self-referentially in these manifests.


Prepared 2026-10-11 (client date). **CANDIDATE, NOT FROZEN OR APPROVED TO LAUNCH.** Canonical science remains `lodo_experiment_protocol.md`; current recovery is `lodo_recovery_contract.md` and strict-policy evidence is `lodo_cuda_policy_integration.md`. This document binds proposed executable artifacts to that science; it does not replace Stage A or authorize another phase.

## Candidate campaign and identities

Eight configurations under `configs/lodo/v1/` cover ResNet50 Candidate C and PCB × LODO-1–4, seed42. Exact paths, full config/scientific/map/manifest SHA256s, actual source images/classes, budgets and epoch lengths are in `lodo_phase10_evidence/candidate-index.json` and the A–J readiness report. All future run leaves use `exp/lodo/v1/<architecture>/<fold>/seed42/official-candidate-20261011`; none is created during preparation. A name containing official is a proposed namespace, not evidence of an official experiment.

Every source is built virtually by its unchanged original loader. Authorized data are Market/Duke processed trainval; CUHK03 detected processed trainval, processed_partition/null split ID; MSMT17 V2 raw train/list_train only. The excluded dataset is metadata-only during preflight and is never a training/evaluation loader during training. Source-local PIDs are namespaced and numerically ordered into contiguous labels. Processed PIDs are source-exposed encoded identifiers, not claimed recoverable official raw IDs. Sidecar digests bind exact source/global indices, labels, canonical image references and unchanged source-local cameras. No copied/combined images.

Each source descriptor carries its production-validated metadata SHA256; MSMT target metadata uses the production canonical ordered query/gallery list composite digest. Partition/list files, recoverable map JSONs, sidecar/data digests and pretrained artifact hashes are pinned in per-run manifests. Path/inode checks reject prohibited overlap/aliases and allow the original in-tree MSMT V2 train/test directory aliases; they do not prove content uniqueness or distinct real people across datasets. No every-image content hash or semantic identity audit is claimed.

## Scientific recipe remains fixed

| Component | ResNet50 Candidate C | PCB |
|---|---|---|
| Model / representation | stride1/GAP/BNNeck, 2048-D post-BN L2 | independent_part_reduction, six heads, 1536-D concatenated parts |
| Input / batch / sampler | 256×128, 64, original PK P16K4 | 384×128,64, original shuffled random/drop_last |
| Loss | ID1/epsilon.1 +Triplet1/margin.3 +Center.0005 | sum of six ID CE heads; no smoothing/Triplet/Center |
| Center | C×2048 pre-BN, local half-mean squared distance; separate SGD.5, inverse-weight center-gradient rescaling | absent |
| Main optimizer | existing Adam defaults, LR.0003, decay.0005 | SGD momentum.9/Nesterovfalse/decay.0005; backbone/new.01/.1 |
| Primary / auxiliary budget | 85,200 /85,200 | 94,200 /0 |
| Precision / seed | FP32/AMPfalse,42 | FP32/AMPfalse,42 |
| Schedule | denominator500 warmup; update500 factor.99802,501 full; decays used at28,401/49,701 | no warmup; decay used at31,401; 10:1 ratio preserved |
| Terminal selection | exact-budget predetermined terminal, no target metrics | same |

All architecture-owned augmentation/normalization/retrieval settings inherit the existing finalized presets and are included in scientific identity. Center/Triplet use raw features while ID uses post-BN classification. Stage A recipes, loaders, samplers, losses, models, evaluators, result matrices and selection remain unchanged. These whole-strategy comparisons are not an isolated causal ablation of dataset diversity; one seed/fixed workloads do not establish convergence or significance.

## Proposed execution/recovery freeze

Propose Stage B-only `strict_cuda_deterministic_v1`, workspace `:4096:8` before imports, cuDNN benchmarkfalse/deterministictrue, deterministic algorithmstrue/warnonlyfalse, unchanged approved TF32 controls, FP32 and zero nonpersistent ordered workers. Stage A defaults and historical schema1 semantics remain intact; strict scientific identity/checkpoints use schema2.

Actual production capability is `epoch_boundary_rollback_cuda_v1`, selected by `train.recovery` version1/policy completed_epoch_boundary/every_epochs1/keep_generations2. Cadence/retention are already implemented and tested in prior phases; **using them officially is proposed, not approved by this document**. Validate and commit an initial boundary, every exhausted epoch, retain latest two validated immutable generations, and publish a separate terminal. Final exact budget may stop inside an epoch; that terminal is complete-only and cannot continue.

Saved state includes model/buffers, canonical Center/module optimizer, main optimizer, update scheduler/next LR, RNG/sampler ownership, epoch position, primary/auxiliary/attempted counters and source exposure. CPU-first compatibility precedes CUDA discovery/tensor mutation; RNG is restored last. Accepted counters/exposure roll back with state. Corrupt/missing latest integrity pair can fall back via the committed index; explicit bad generations or incompatible identities fail closed. No orphan promotion, repair or legacy migration. Two-generation retention never prunes an unverified/unrelated artifact or the sole valid generation; forensic corrupt bytes can accumulate and require a stop/review.

Phase9R.3's actual same-checkpoint A/B/C is bitwise exact at updates3–5 for both models; actual SIGTERM rollback2→5, terminal no-op and actual-artifact integrity checks passed. This is **LODO-1, five updates, two-batch fixture epochs, zero workers, one TITAN UUID/runtime**. Exact saved-state restoration differs from a universal bitwise-continuation guarantee. Full real epochs, other folds, long durations, different workers/hardware/libraries and arbitrary mid-epoch cursors are not qualified by that result.

Proposed device is the recorded TITAN Xp full UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, logical cuda:0 with single-device visibility. Historical diagnostic allocation is not official allocation. GTX1080 is not a qualified substitute. Actual allocation-window/e2e performance and strict all-fold memory remain explicit launch-review gates. Do not alter the recipe to fit resources.

## Manifest and freeze semantics

Manifest v1 is canonical UTF-8 JSON, sorted keys/compact separators/noNaN. `immutable_sha256` hashes only `immutable`; timestamps/resume/allocation/approval metadata remain `operational`. Production `configuration_identity` retains its existing versioned scientific JSON (integer-valued float normalization). Map serialization remains its existing schema1 and excludes paths/timestamps. Data/sidecar identities retain production canonical-root binding; relocation is not silently compatible. Full configuration file hash additionally pins proposed operational declarations.

Immutable proposal fields bind scientific/config-file identity, current implementation source bytes/membership, pretrained artifact, actual dataset/map/sidecar/provenance, recovery, qualified runtime scope and separate evaluation descriptor. `code.base_head` is the observed historical base; `code.state=dirty_worktree_unfrozen` and `code.frozen_commit=null`. A custom source snapshot SHA is **not a Git commit/tree SHA**. No manifest claims an uncreated commit. The guard checks source membership as well as hashes, full recorded runtime/driver/UUID, cached weights, evaluation request identity, schema and fresh output collisions without CUDA/images/jobs.

After review/explicit Git approval, a clean implementation commit plus separate signed-off launch receipt must bind that real SHA to the already verified proposal. Additional `--frozen-commit` checks full current HEAD/clean status/exact implementation snapshot; it never grants approval. No launch manifest may be called frozen while these gates remain pending.

## CUHK03 and Stage A comparison eligibility

Local new-detected JSON/MAT references exist. Processed train/query/gallery sizes, identity counts and sorted per-identity image counts agree with those references; PID sets differ and no verified raw-PNG/MAT→processed-JPG/PID conversion manifest/generator was found. This is corroborating metadata evidence, not proof of sample correspondence. Preserve processed_partition/null and the explicit **unverified official detected/new** claim. Advisor review must accept this exact hashed partition limitation or commission separate provenance/conversion work; no data/split modification is authorized here.

Stage A reference eligibility is determined by the actual Phase7 validator, not filenames or metric values. Historical ResNet50 matrix records lack required historical target split hashes/filter evidence; PCB cross-domain records are absent in the inspected original result/experiment trees. Exact issues and missing cells are in `stage-a-reference-eligibility.json`. Do not rewrite/enrich originals, fabricate cells or average fewer than three. Standalone post-training Stage B evaluation may be valid with comparison status incomplete; complete comparison needs exactly three compatible source→target references per architecture/fold.

## Review and mandatory stop

See `lodo_phase10_readiness.md` for measured/projected memory, checkpoint cost/storage, normal/failover replay bounds, guarded commands, tests and preservation evidence. Approval decisions remain open: exact staged freeze contents/commit, processed CUHK03 limitation, restricted recovery/worker/policy/cadence, storage/logging conditions, dedicated GPU/runtime/allocation window and each launch. Real target evaluation remains separately authorized only after terminal completion. No official training/evaluation, GPU workload, commit/push/merge or Phase11 occurs in Phase10 preparation.
