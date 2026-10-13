# Phase 10F final scientific/operational sign-off

Decision source: the explicit user Phase 10F instruction, client date 2026-10-11; exact attachment SHA and approved scope in [authority.json](lodo_phase10f_evidence/authority.json). This document approves the existing candidate science and a **single conditional local freeze commit**, not official execution. No advisor/institutional approval, GPU reservation, launch ID or benchmark result is invented. Historical Phase 2/9/10 states and failures remain dated evidence; current decisions supersede only their pending-approval wording.

## Approved scope and remaining limits

The four canonical complement folds, original-loader virtual sources, source-owned splits and deterministic identity maps are approved unchanged. ResNet50 retains 85,200 successful primary and Center updates. PCB retains 94,200 primary updates and no Center steps. Frozen losses, transforms, dimensions, sampling, schedules and seed 42 remain unchanged. Training uses no target loader or target-selected checkpoint; completion uses the exact-budget terminal.

Stage B-only `strict_cuda_deterministic_v1`, FP32 and zero workers are approved. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization; cuDNN benchmark is false, deterministic algorithms are true with warn-only disabled, matmul TF32 is false and cuDNN TF32 is true. Schema-v2 runtime identity fails closed on incompatible resume. Stage A defaults and schema-v1 behavior remain unchanged. CUDA qualification covers LODO-1, five updates, zero workers and the recorded TITAN runtime; full-duration, all-fold and all-device equivalence remains unverified.

Recovery uses `completed_epoch_boundary`, every completed epoch, the latest two independently validated immutable generations and a distinct terminal. Restore and replay unfinished epochs without double-counting durable updates or exposure. Existing integrity fallback and incompatibility rejection remain unchanged. Pair/index publications are individually atomic, not one transaction; orphan generations are not promoted. There is no arbitrary mid-epoch continuation. An exact-budget terminal inside an epoch is complete-only.

CUHK03 detected `processed_partition`, `split_id: null` is accepted with limitation. Its partition SHA256 is `a1fd9a06748dd217325820b2ed5c5814de364a6b5e91cb7a93e1924915034f97`; train has 7,365 images/767 identities, query 1,400/700 and gallery 5,332/700. Official detected/new correspondence remains **UNVERIFIED**. Reference counts do not establish raw-to-processed sample correspondence. User acceptance is not advisor or institutional sign-off; none was supplied. Any separately required institutional approval remains pending.

Incomplete Stage A comparisons are accepted and do not block training preparation: 12 ResNet cross-domain records were found, zero qualify under current provenance, and no PCB cross-domain records were found in inspected artifacts. Do not backfill historical hashes, substitute within-domain results or average fewer than three sources. Future reference qualification and real held-out evaluation require separate approval. Comparison concerns complete strategies, not an isolated causal effect of domain diversity.

The sequential TITAN plan is provisional: preferred UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, one run at a time, no fallback, fresh allocation/ownership/process checks, at least 10,000 MiB free VRAM and 64 GiB free disk. Numeric headroom thresholds are operator/runbook gates, not production resource reservations. The proposed 48-hour window is planning only. Full-fold sustained memory and end-to-end duration remain unverified. The retained-checkpoint estimate of 6.784 GiB excludes publication scratch, growth and forensic artifacts. Actual allocation and official launch/resume/evaluation remain separate gates.

## Exact candidates — unchanged identities

| Architecture / fold | Images / IDs | Primary / auxiliary | YAML SHA256 | Immutable manifest SHA256 |
|---|---:|---:|---|---|
| resnet50 / LODO-1 | 54,135 / 2,510 | 85,200 / 85,200 | `c3af6e294ba941f08a122996309e7d3dda302b55fd7dd4585885aeb656fb30e5` | `81d1ae641d088e6cb1f25814216782ba51d5b936a05fb63be9b2da45b39d42cf` |
| resnet50 / LODO-2 | 50,549 / 2,559 | 85,200 / 85,200 | `2c0d82cedd21fb2d0272af6d6052b223e936ccf7e1a2174b0b549969e193fafd` | `694cacb5d038344ab7fc5316fcf15b1348b236fab44f09135266086bea6c0384` |
| resnet50 / LODO-3 | 59,706 / 2,494 | 85,200 / 85,200 | `fc02eed6e1d0a0d85bd06d481386f409afb11af8cc0db0a0c4e5b3a2735ebd5b` | `8ba2d151e7619d16345b4e46cbb7baa0bfd0e25097598cee75675c251c57b0f2` |
| resnet50 / LODO-4 | 36,823 / 2,220 | 85,200 / 85,200 | `0d68d05d6404c0403a3532a1ae57769a2f9a5e7e94db47414802a76b9d891175` | `3f2eb1c5497421bad78c5470e0feafab37061decf19cd531bba912197550e7c7` |
| pcb / LODO-1 | 54,135 / 2,510 | 94,200 / 0 | `2fd3246c530c8feff9c5566a5b3a2643a359fdca39713ce04ebcf996c6468466` | `4e87c209af6abdfc44ecf40a95b3df4de7608997f6fe875293621a510f56f19b` |
| pcb / LODO-2 | 50,549 / 2,559 | 94,200 / 0 | `c7b309fd12bfa0b8196eb2b999540d4a80b0d5b77006ee6f21b4688211622700` | `32e951f16add5faa3e4fe6d86e5f41e5d06831e4ea6997f2f4ab425b2d2676c0` |
| pcb / LODO-3 | 59,706 / 2,494 | 94,200 / 0 | `eeab4cae608834a6cc61058e940e86cc30d40dd1f4bb8c8ab45f04d06bba832b` | `1b656e8db9cc3027b9f6e63e80ddddbbde89ef226edb63fbe3b0cd2c267d5370` |
| pcb / LODO-4 | 36,823 / 2,220 | 94,200 / 0 | `374f97dbe23f2246a0ba8aa89f1367e729c4a6b63131ddbb850832480deab190` | `01c474466262d7604119e0843f7b27160c358c4077b7ba39f3fb7c7e4a6a1bc4` |

Full scientific/map/data/sidecar/pretrained identities and evaluation descriptors remain in [candidate-index.json](lodo_phase10_evidence/candidate-index.json) and its eight manifests/four maps. Source counts (images/identities): Market 12,936/751; Duke 16,522/702; CUHK03 7,365/767; MSMT17 30,248/1,041. Fold totals: 54,135/2,510; 50,549/2,559; 59,706/2,494; 36,823/2,220. Canonical root `/mnt/hd-data/addirakoze/Projects/Dataset`. All original metadata/hash/path/inode/exclusion gates reran CPU-only, no images/target loaders/CUDA. Image-content/semantic-person uniqueness is not established.

ResNet pretrained SHA256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`; PCB SHA256: `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`. No downloads. Immutable source snapshot SHA256: `70f69a7b66f5996e0fdaf02dcbf8bd5699b84aa856c23196abc45db5deb82a4f`. No production/config/test changes since hashing; narrow documentation updates are explicitly recorded in docs-signoff-delta.json.

## Freeze validation and provenance semantics

Before staging, both full porcelain-uall statuses/inventories/HEADs/worktrees were captured; original main remains at `3b015c964b99bfe8ad4b7842bd174d3bc58af607` with existing uncommitted work preserved. The entire 10,200-line/590,622-byte patch was read/reconstructed/AST-reviewed, all 45 full new Python files match disk and all nine tracked changes match the patch. Critical data/policy/shared-engine/checkpoint/orchestration paths were inspected. All 143 original proposal paths were reviewed, including ignored JSONs; bounded sensitive/prohibited-file screening passed. New Phase 10F records are documentation/provenance only. Full review scope/method and exact membership are recorded in source-review.json/freeze-membership.json; no claim of universal secret detection is made.

Fresh full CPU suite: 1,382 passed, zero failed and four CUDA-only skips (531.89 s; JUnit 531.855 s). Focused suite: 25 passed, zero failed/skipped (8.53 s; JUnit 8.534 s). Exact commands/environment/skip names in cpu-validation.json and JUnit files. All eight manifest preflights pass; diff/syntax checks pass. Additional static-audit path allowlist fixture was corrected only in /tmp; production/scientific identities and numerical tolerances unchanged. Original attempt source is retained locally. No pytest baseline failures/regressions observed.

Protected artifacts rehash: 4,511 selected full SHA256s plus 4,629 stat/membership records and 33 original result files; original 175 source files unchanged. The fresh original status/source and post-stage preservation checks are required again before the single commit. Dataset metadata/weights validated; no all-image-content audit is claimed. No historical diagnostics or completed PCB artifact is changed.

Candidate JSONs intentionally keep their original prefreeze base/dirty-snapshot/frozen_commit:null and NOT_APPROVED/NOT_ALLOCATED launch/allocation declarations. These fields are historical candidate provenance; this user sign-off is recorded separately. They are not silently relabeled or rehashed. The actual **full new Git SHA** is written after the one commit into `.local/lodo_phase10f/operational-freeze-receipt.json`, outside the commit it describes. Its companion final-report.md records actual outcome, cleanliness and eight --frozen-commit results. No self-referential commit hash is inserted. A failed post-commit gate means FREEZE INCOMPLETE; no amend/reset/additional commit is authorized.

## First run — read-only launch approval checklist, NOT EXECUTED

1. Read actual full frozen commit from the separate operational receipt; require feature/lodo-v1/clean intended source state and all eight frozen-commit checks. Bind ResNet50 LODO-1 YAML/immutable/science/map/pretrained full hashes above and in its unchanged manifest.
2. **Current lab allocation ID: PENDING. Official launch approval ID: PENDING.** No placeholder is a real authorization. Confirm exact preferred TITAN UUID independently and a conservative monitored window; short-diagnostic permission does not carry over.
3. Repeat read-only NVML/process owner/UUID/runtime/VRAM >=10,000 MiB / disk >=64 GiB checks; expose only the authorized full UUID/logicalcuda:0; export workspace before Python. Stop on competitors, mismatch, nonfinite/OOM/kernel/storage fault. Never modify others' jobs.
4. Use the [guarded runbook](lodo_official_launch_runbook.md) only after separate approval; tmux explicit environment/unbuffered own stdout, fresh operational receipt/hash/dependency freeze. No official leaf is reserved now. Monitor losses/LR plus durable primary/Center/source-exposure counters, generation integrity and disk.
5. ResNet50 LODO-1 first; subsequent ResNet folds then PCB sequentially only with separate approvals. Interruption/resume uses same-run validated logical index and completed-epoch rollback; no mid-epoch continuation.
6. Exact terminal: 85,200 primary / 85,200 Center steps with 5,452,800 presentations; verify all hashes/provenance/completion first. Real target-only evaluation requires separate approval after completion; Stage A comparisons remain explicitly incomplete.

**NOT APPROVED:** official training/resume, real target evaluation, GPU workloads/allocation, merge/push/tag or Phase 11. Stop after one conditional local freeze and verification/reporting.
