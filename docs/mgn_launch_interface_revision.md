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
