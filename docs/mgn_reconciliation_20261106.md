# MGN reconciliation on Oneiros — 2026-11-06

The user's continuation request authorizes read-only reconciliation, separate registration of the completed schedule trial, and preparation/CPU validation within existing authorization. It explicitly retains the four original source runs as the selected study configuration and prohibits further tuning. No training or benchmark evaluation was launched during this reconciliation.

## Repository and migration

Starting branch `feature/mgn-offline`, HEAD `7185a7667b2d21a5ad805538ace64326b2bfa28e`. No applicable AGENTS.md was found in the checkout or its ancestor directories. Local commits `b72c4a3`, `81d356b`, and `f4aea84` all exist and are ancestors of HEAD. Later commits `2a62658` and `7185a76` preserve the Oneiros host qualification and successful admission evidence. Local existence does not establish remote publication; no fetch, commit, push or merge was performed.

Pre-existing changes are preserved: modified `plan.md` and `docs/preliminary_results_log.md`; untracked migration helpers/docs, original MGN result registries, PCB results/comparison documents and `results/pcb_selected_matrix/`. In particular, the original MGN result Markdown/YAML are present but untracked. Human/advisor acceptance and their publication are not inferred.

The migration receipt at `/home/addirakoze/.local/share/reid-migration/execution-receipt/migration_receipt.json` reports seven transferred worktrees, 91.89 GiB logical payload, matching dataset transport hash and checksum verification with no content differences. All seven target repositories and their transferred commits are present. This session independently rechecked original MGN artifact hashes, rather than treating that receipt as sufficient completion evidence.

Current CPU inspection matches the active Oneiros manifest: Python 3.11.2, Torch 2.7.1+cu126, torchvision 0.22.1+cu126 and its package/control fields. All 57 runtime hashes and twelve qualification-receipt hashes match. The historical pretrained artifact matches SHA-256 `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`. All four datasets match the active host manifest, including partition bytes, ordered records/maps and host file inventories; portable identities also match the original result registry. No full corpus byte rehash or exhaustive image decoding is claimed.

The old registry's manifest hash `b289ea8a...` resolves to `docs/mgn_freezes/20261105_before_schedule_trial/mgn_experiment_manifest.json`; the current manifest is a later host revision. Old documents referring to an active manifest with that old hash are historical snapshots. The Phase15 archive, schedule-trial archive and Constantine host-admission archive remain unchanged. Original saved output paths under `/mnt/hd-data/...` were mapped to this checkout only for comparison of `experiment.output_dir`; no stored provenance or scientific setting was rewritten.

A read-only host query found NVIDIA RTX A4000 `GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2`, driver 590.44.01, 15967 MiB free and no compute processes. No common training/standalone evaluation Python process was listed. This is a time-specific snapshot, not an allocation or evaluation authorization. Sandbox-only NVML failed; the host query succeeded. Prior Oneiros synthetic/four-source three-update training qualification and READY_FRESH receipts remain historical evidence of their named scope, not an independent retrieval evaluation qualification.

## Reconciled phase state

| Phase | Observed state | Evidence / remaining gate |
| --- | --- | --- |
| 1–12, 14–15 | Mandatory technical work completed historically | Appended roadmap records, qualification/freeze receipts and current frozen byte checks |
| 13 | Explicitly waived | FP32 primary policy; no AMP qualification claimed |
| 16A–D | Four trainings and trainer-produced evaluations complete, reverified | Eight matching checkpoint hashes, 120 epoch markers per run, 12 periodic evaluations per run, best/last separation; human review/publication tracked separately |
| 17A–D | Pending; CPU preparation completed | No independent MGN standalone artifacts found in this checkout's results/exp scan; trainer final evaluation is insufficient |
| 18 | Pending | Twelve off-diagonal cells need corresponding accepted 16/17 evidence and approved row scope |
| 19 | Pending mandatory scientific decision | Requires reviewed 16–18 evidence and explicit approval of versioned MGN LODO protocol |
| 20–21 | Pending | Approved Phase19 protocol before implementation and fold-specific bounded qualification |
| 22–23 | Pending | Four user-operated fold trainings, then approved independent held-out evaluation/registration |

The roadmap's original “NOT STARTED” statements and old preparation-only protocol/trial statements are retained as history. Its activated amendment and later evidence govern completed technical work. This record does not retroactively invent launch approval messages, shell exit receipts or advisor acceptance.

## Original selected results

Percentages below are trainer-produced source-test results; the machine receipt retains full precision and checkpoint hashes.

| Source | Selected epoch | Selected mAP | Selected Rank-1 | Epoch120 mAP | Epoch120 Rank-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Market1501 | 70 | 71.00 | 88.18 | 70.74 | 88.03 |
| Duke | 40 | 66.87 | 81.64 | 66.03 | 81.37 |
| CUHK03 detected / processed_partition | 120 | 45.86 | 46.07 | 45.86 | 46.07 |
| MSMT17 | 70 | 19.67 | 43.13 | 19.21 | 42.49 |

Reverified all eight registry checkpoint hashes, finite model tensors, source C/reconstruction metadata, optimizer-state presence, checkpoint/scheduler epochs, original 60/90 milestones, selection history, resolved/saved configuration, 97 registry-listed supporting artifacts and the 23-file Market1501 resume archive. Checkpoint/final metric agreement uses the existing registration tolerance of 1e-5 in fractional units.

Market1501's original partial epoch87 and continuation from epoch86 remain disclosed. Ordinary checkpoints do not establish authoritative successful-update counts or bitwise stochastic recovery. Formal shell exit-code receipts were not saved. CUHK03 semantic new/classic correspondence remains unverified. Source-test selection is not independent validation. These limitations are preserved rather than “repaired” by relabeling results.

## Separate schedule trial

The actual saved configuration differs scientifically only in milestones 40/60 instead of 60/90; operational changes are experiment name/notes/output and named preflight profile. Logs verify all 120 epochs, all 12 evaluations and LR0.01/0.001/0.0001 in epochs1–40/41–60/61–120. Last checkpoint/scheduler both record120; best checkpoint and selection history record90. Both checkpoint hashes, all final/periodic metrics, resolved configuration and launch artifacts are recorded separately in [the trial receipt](mgn_market1501_paper_schedule120_results.yaml).

| Role | Original Market1501 mAP | Trial mAP | Delta, percentage points | Original Rank-1 | Trial Rank-1 | Delta, percentage points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected-best | 71.0009 | 71.2293 | +0.2284 | 88.1829 | 87.9157 | −0.2672 |
| Epoch120 | 70.7414 | 71.0561 | +0.3147 | 88.0344 | 88.1829 | +0.1485 |

The small mixed changes do not establish a meaningful improvement or statistical significance. Different hardware and the original run's continuation prevent a bitwise paired comparison. Per the user's explicit decision, this trial remains unselected supplementary evidence; original four-run results stay selected. No additional tuning is proposed.

## Next eligible unit and concrete handoff

The next roadmap work is **Phase17A: standalone Market1501 evaluation of the original epoch70 checkpoint**, SHA-256 `5067cb70fb2e3fe7609ca7338487c44f62699596088852abe060c9672ebe8024`. Preparation for all four diagonals is in [phase17-preparation.yaml](mgn_reconciliation_20261106/phase17-preparation.yaml), including exact checkpoint/config identities, proposed commands and fresh output paths. All four effective configurations were validated on CPU through the existing standalone source-policy preparation; no gallery was evaluated and no result directory was created.

Before execution:

1. Review the original completion evidence and resolve the result-registry acceptance/publication boundary. No new training is required.
2. Explicitly approve the Phase17A cell, original fixed checkpoint, source-owned RGB384x128 bicubic preprocessing, raw original/flip mean then global L2, Euclidean distance, no reranking, FP32 and named Oneiros resource scope. The amendment explicitly conditions Phases17–18 on approved evaluation policy, cells, resources and execution scope; the present request preserves those separate gates.
3. Review the proposed absolute comparison tolerance **1e-5 in fractional units per metric** against trainer selected-best results before execution. This follows the existing artifact-registration tolerance, but cross-host inference equivalence remains unobserved. A larger discrepancy is preserved/investigated, not used to select another checkpoint or widen tolerance retrospectively.
4. Immediately before launch, recheck live GPU UUID/runtime/occupancy, release/source/config/checkpoint/data identities and fresh output path. Standalone `evaluate.py` creates directories with `ensure_dir` and does not enforce fresh output itself, so the approved orchestration must refuse any existing/symlink destination. Do not reuse training output directories or invoke training preflight's fresh-run checks on completed runs.
5. Capture command, environment/device/process and output receipts, evaluate once through common `scripts/evaluate.py`, compare/register the observed diagonal, and stop at its review boundary. Later diagonals and cross-domain rows are not auto-authorized.

The first proposed command, **prepared only; not executed**, is:

```bash
/home/addirakoze/environments/Reid/bin/python -u -B scripts/evaluate.py \
  --config configs/mgn/market1501.yaml \
  --weight exp/mgn/market1501/checkpoints/ckpt_best.pth \
  --opts experiment.output_dir=results/mgn_independent_within_domain/oneiros_20261106/market1501
```

Phase18 must fix the original source checkpoint across all targets in each row and preserve source preprocessing/retrieval settings. Phase19 must separately approve LODO version, budgets/exposure, scheduler units, ordered combined-source maps, target-free selection, primary retrieval/comparability, precision and recovery before any Phase20 implementation. Historical ResNet50/PCB LODO v1 stays frozen.

## Validation and preserved work

Read-only artifact/data audit passed without CUDA initialization or model inference. Focused CPU suites `test_cross_domain_config.py`, `test_evaluation_harness.py`, and `test_checkpoint_reconstruction.py`: **93 passed, 0 failed, 0 skipped in 23.71 seconds**. These include synthetic evaluation fixtures and strict reconstruction/no-download checks; they are not benchmark runs. Full command/log/source hashes and evidence are in [artifact-verification.yaml](mgn_reconciliation_20261106/artifact-verification.yaml). The audit script and logs are retained beside it.

Only new reconciliation/trial evidence and this roadmap's append-only record are changed. Existing training outputs, original result registries, protected PCB/LODO material, runtime/configuration/preflight source, historical freezes and unrelated dirty work remain untouched. No local commit, push or merge was performed; review remains pending.
