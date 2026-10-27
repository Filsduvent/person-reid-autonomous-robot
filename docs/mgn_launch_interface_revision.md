# MGN automatic training admission — freeze revision

Date: 2026-10-27. Revision: `20261027_automatic_preflight_v1`.

The user explicitly authorized this launch-interface change after Phase15 qualification. No official training or full benchmark evaluation is authorized by this revision. Starting point: branch `feature/mgn-offline`, commit `b568c6b6ea35014bc5c1a61a26889eb11b9770ad`, with existing unrelated PCB documentation/results left untouched.

## Investigation and implementation

The common trainer previously created output directories before selecting the device and constructing the model. The existing MGN preflight was already a read-only callable, but reloaded the canonical YAML and did not bind CLI overrides to its checks. The smallest reusable extension is an optional named `train.preflight` mapping, dispatched by `reid/utils/training_preflight.py` at the start of common `scripts/train.py`, immediately after configuration loading and before either ordinary or historical LODO dispatch.

The four MGN presets declare `train.preflight: {name: mgn_offline_v1}`. Source identity comes from the effective training dataset name; the adapter checks the canonical source YAML path and exact effective frozen configuration. Only the operational output-path normalization and controlled same-source last-checkpoint resume are allowed. Overrides cannot remove/change the declared admission profile. Unknown profiles and malformed explicit configurations fail closed. No executable command/import path is accepted from YAML, and no device is chosen or changed.

The existing MGN gate continues to check committed/clean runtime, ancestry and all source/config/receipt/weight hashes, environment, datasets/maps/inventories/smoke-image bytes, output isolation, disk space, current GPU authority/visibility/identity/contention/runtime/memory, and existing checkpoint/selection/optimizer/scheduler continuation state. A refusal propagates before trainer side effects. On success the trainer prints the receipt, then saves `artifacts/training_preflight.json` after creating the admitted run output. Standalone preflight remains an optional diagnostic. Baseline/PCB and historical LODO configs have no opt-in and retain their existing dispatch and admission rules.

## Freeze preservation and scientific equivalence

The complete previous manifest, protocol, four presets and Phase15 receipts are copied byte-for-byte under `docs/mgn_freezes/20261027_phase15`, with original commit and per-file SHA-256 in `snapshot.json`. Earlier qualification receipts and original roadmap records remain unchanged. The previous manifest SHA-256 is `5d0173f2448726520f8a3a3381d88c9683f8a15c74fe81ec9f7131ee7159fdd2`.

Every new preset equals its archived predecessor after removing only `train.preflight`. Architecture, weights, losses, optimizer, epoch schedule, batch, transforms, evaluation policy, dataset partitions, output namespaces and resume scientific policy remain unchanged. The new manifest binds the changed common trainer, schema, adapter and new generic hook, and revised raw/resolved configuration hashes. The old hash-qualified implementation is not claimed identical. Previous bounded GPU evidence is retained only for the unchanged model/scientific recipe; no new GPU optimizer workload is needed or claimed for a launch-order change.

The manifest stays `PREPARING` while qualification is pending and will be returned to `FROZEN` only after CPU checks, hash verification and a separate local implementation commit. The revised code commit and exact test evidence will be recorded below. Live admission checks will exercise common `train.main()` with an explicit sentinel stopping immediately before output creation; they cannot launch training.

## Validation history

Initial focused suite: 89 passed, one test-fixture failure in 7.49 seconds. The test's adapter mock remained active during a subsequent real-guard assertion; narrowed its scope. Original log: `/tmp/mgn-launch-focused.log`.

The first regression command referenced nonexistent `tests/test_lodo_recovery.py`; collection stopped with zero tests, no workload executed. Corrected to existing CPU-only `tests/test_lodo_cuda_recovery.py`. Original collection-error log: `/tmp/mgn-launch-regression-collection-error.log`. No scientific correction or GPU retry occurred.

Qualification is pending; no official output exists. This record will be extended with observed results and commit identities without rewriting the original Phase0D or Phase15 history.

## Observed CPU acceptance

The corrected broad CPU regression command passed **591 tests**, zero failures/skips, in328.41 seconds. It covers the common trainer, MGN synthetic continuation, checkpoint reconstruction/selection, exact scheduler traces, baseline/PCB presets and end-to-end behavior, historical LODO configuration/admission and CPU recovery. Final focused suite passed **92 tests**, in7.24 seconds, including the additional admission-receipt persistence check. Exact commands, file lists, source/test/log hashes and preserved-freeze assertions are in `docs/mgn_launch_interface_qualification.json`. CUDA was hidden in both suites; test workloads are isolated synthetic fixtures, not official dataset training or benchmark evaluation.

All57 runtime-file hashes have been recomputed. Only four runtime paths differ from the previous freeze: `scripts/train.py`, `scripts/mgn_preflight.py`, `reid/utils/config_schema.py` and new `reid/utils/training_preflight.py`. All four active preset hashes/resolved configs were revised solely for the launch opt-in. Prior GPU/weight/environment/selection/retrieval/resume policies and every dataset/provenance field are unchanged. All48 protected original files remain byte-identical; no baseline/PCB/LODO source/config or original roadmap history changes. No official output root exists. CPU technical acceptance passes; the next local commit records this implementation with manifest still PREPARING, then a separate freeze commit will bind that observed code SHA before live read-only admission checks.

## Observed implementation commit and revised freeze

Implementation commit: `569e8d533c1a3b863c2681ad5898160c546364bc`. CPU qualification passed before this commit; its runtime bytes exactly match the revised57-file manifest. The active manifest is now FROZEN and anchors this observed commit, with the new CPU receipt hash and explicit previous-freeze identity. The original `b719551` GPU-tested implementation remains separately identified in the preserved receipts. This does not imply the changed common trainer was GPU-training-qualified. Live checks follow only after the freeze metadata is committed; each is stopped before output creation. No official training has run.

## Observed common-trainer live qualification and closure

Freeze commit: `031bb3237499c7df2fd2216c0e93fcaccb5d6be0`. Revised manifest SHA-256: `b289ea8a1ee1ee03d2a2b41267e0286168241dcda3b07817d864df0f1d8f286e`. All four actual source presets returned READY_FRESH through common `train.main()`, then hit the unconditional stop before output creation (exit0). Only `create_experiment_dirs` was replaced by the stop sentinel; real admission/config/data/environment/GPU guards executed. No training model, optimizer step, training output or ranking evaluation ran in these live checks. The exact driver, response/log hashes, live device audits and current code/config/manifest identities are preserved in `docs/mgn_launch_interface_live_qualification.json`.

Two additional real CLI negative checks refused a changed LR and a resume request without the original run/checkpoint (both exit1, before output/device training). Existing controlled valid continuation is covered by the CPU/synthetic regressions; no production MGN checkpoint exists to resume. All57 runtime hashes, six qualification-receipt hashes, eight prior-freeze file copies and48 protected original files validate. Original roadmap history and baseline/PCB/LODO source/config are unchanged; all four official output roots remain absent. No push/merge, official training, benchmark evaluation or new GPU training was performed. Implementation and freeze technical qualification pass. The unchanged scientific caveats and human review status remain explicit in the protocol/manifest.

The single-command interface is now ready under the existing qualified environment/GPU prerequisites: `python scripts/train.py --config configs/mgn/<source>.yaml`. A separate manual preflight invocation is unnecessary. No automatic GPU selection or fallback policy was added. Work stops after this documentation/qualification commit.
