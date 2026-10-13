# Phase 7: post-training LODO evaluation and comparison contract

This is an implementation contract, not authorization to run official evaluation.
Phase 7 validation uses synthetic CPU inputs and temporary outputs only. Scientific
protocol, source loaders, architectures, training and Phase 6 recovery are unchanged.

## Explicit entry point

`scripts/evaluate.py --config <reviewed-request.yaml>` dispatches only when `mode`
is `lodo_held_out`. Existing Stage A requests retain their existing execution path.
The LODO request is a separate YAML mapping, not a training configuration. No
runnable official request is shipped. `--weight` cannot override its checkpoint.

Required fields:

| Field | Contract |
|---|---|
| mode / version | `lodo_held_out` / integer `1` |
| architecture | `resnet50` or `pcb` |
| fold / target | A frozen Phase 3 fold and its excluded target |
| checkpoint | Canonical `ckpt_terminal.pth`, with Phase 6 `.json` sidecar |
| expected | Pinned `config_digest`, `map_digest`, `data_digest`, `num_classes` |
| synthetic | Explicit Boolean; synthetic results are non-authoritative |
| output_dir | Fresh isolated path described below |

Optional fields: `data_root`, positive integer `batch_size` (default 32),
nonnegative integer `num_workers` (default 0), and `stage_a_records` (explicit JSON
artifact paths, default empty). Unknown keys are rejected. Execution is CPU fp32;
GPU evaluation is deferred. A relocated data root must still match the canonical
metadata paths and hashes in terminal provenance; there is no silent rebinding.

## Read-only terminal gate and evaluation

`reid.utils.lodo_evaluation.validate_terminal` uses the unchanged Phase 6 reader
and its exact schema/sidecar hash. It requires terminal completion at 85,200
primary/auxiliary ResNet updates or 94,200 primary and zero auxiliary PCB updates.
It validates the frozen configuration, ordered fold, expected identities, mapping
membership/order/contiguity, class dimensions, precision, source/target provenance,
exposure totals, model tensor finiteness, Center convention/presence and scheduler
counter/contract. Recovery and legacy Stage A artifacts are rejected. SHA256
provides integrity evidence, not cryptographic authentication; torch checkpoints
must remain trusted local artifacts.

The original target partition/list hashes must match the terminal provenance
before reconstructing weights or constructing a target loader. Reconstruction
uses the existing checkpoint utility and combined-source C. Strict state loading
preserves one ResNet classifier or all six PCB heads. Target class count never
sizes classifiers. Only the excluded target's original evaluation loader is built.

Existing source-owned test preprocessing and the existing evaluator perform
feature extraction, L2 normalization, Euclidean distance, same-PID/same-camera
exclusion, CMC Rank-1/5/10, mAP and mINP. ResNet uses its 2048-D post-BN branch;
PCB uses the 1536-D concatenated part representation. No fine-tuning, adaptation,
reranking, flip, checkpoint selection or training feedback is introduced.

## Isolated publication and result schema

Outputs must be new leaves under the canonical worktree path:

`exp/lodo_eval/v1/{synthetic|benchmark}/{architecture}/{fold}/seed42/{identifier}`

Aliases, traversal, existing paths and other experiment namespaces are rejected.
Phase 7 tests override the project root to a temporary directory. No worktree
experiment output is created by the tests. Existing terminal bytes and target
metadata hashes are rechecked before publication. A newly reserved directory
receives atomic files: comparison first, then result, then result hash proof.
An interrupted publication without the final proof fails `read_result`; there
is no automatic repair, overwrite or directory deletion.

`lodo_result.json` schema 1 uses `record_type: lodo_held_out_result`. It records
architecture/variant, fold/sources/target/seed, synthetic/authoritative flags,
training config/map/data/sidecar digests and provenance, verified terminal manifest
and checkpoint path, a separate evaluation config digest/target descriptors and
hashes, representation/retrieval policy, all five metrics in explicit fractions,
CPU environment, artifact paths and comparison SHA256, and scientific caveats.
`lodo_result.sha256.json` is the final integrity proof. `read_result` verifies it
and the sibling comparison hash. Stage A matrix writers and selection functions
are never called; Stage B records do not use the Stage A source_dataset schema.

## Three-reference comparison

`reid.utils.lodo_comparison.compare_stage_a` reads only the explicitly supplied
Stage A cross-domain JSON artifacts. It requires exactly one compatible run for
each of the fold's three source-to-target cells. There is no latest-run selection
or implicit discovery across unrelated output trees. It verifies schema/source/
target/config, architecture/PCB variant, representation, preprocessing, target
split descriptors, normalization/distance/top-k, no reranking/flip, run/checkpoint
references, explicit historical target metadata hashes and camera-filter evidence.
MSMT17 original raw loader defaults are permitted when optional list/subtree
selectors are absent; supplied selectors must match the authorized descriptor.
CUHK03 remains detected `processed_partition`, null split ID, unverified official-new
provenance. A claimed official-new split is rejected.

Existing Stage A schema-1 records emit fractions. An explicit `metric_units:
percentage` is normalized by dividing by 100. Unknown units, nonfinite values and
out-of-range metrics are rejected. Historical records lacking target split hashes
or filtering evidence are incomparable; today's metadata hashes cannot establish
what a historical run used. Originals are never enriched or rewritten silently.

`comparison.json` schema 1 contains each compatible vector and source artifact
hash/evidence; complete reports contain the arithmetic mean of exactly three,
the per-metric envelope with contributing sources and `is_single_run: false`,
one actual run selected by highest mAP then Rank-1 (canonical source order breaks
any remaining tie), and all Stage B differences using
`difference_pp = 100 * (stage_b_fraction - reference_fraction)`.

Missing, invalid or ambiguous required references produce status `incomplete`,
explicit missing/ambiguous/issues/excluded entries and null aggregates/deltas.
The standalone Stage B result is still published. Stage B input cannot serve as
a Stage A reference. Comparisons concern complete strategies: data quantities,
identity counts, budgets, exposure, losses and historical checkpoint selection
are confounders. No causal domain-diversity claim or training feedback is valid.

## Limits and next approval boundary

CUHK03 official-new provenance remains unverified. Path/metadata hashes do not
prove image-content uniqueness or distinct real people across datasets. No
full benchmark/GPU result or GPU recovery guarantee follows from CPU tests.
Phase 6 retains its tested CPU epoch-boundary rollback capability only.
Phases 8–10, GPU qualification, official training/evaluation and deployment remain
subject to their separate explicit approval. Phase 7 does not begin them.
