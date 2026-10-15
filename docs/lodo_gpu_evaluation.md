# GPU feature extraction for held-out LODO evaluation

This is a proposed release revision. The running CPU evaluation, prior release
and scientific training configurations must stay unchanged during review.

An evaluation request may add `device: cuda`. Omission, or `device: cpu`, keeps
the historical CPU FP32 path. `auto`, other devices and fallback are rejected.
The existing `scripts/lodo_evaluate.py --config <approved-request.yaml>` command
remains the interface, with detached execution, live logs and result registration.

The exact request bytes require their own scoped operator declaration. CPU
approval never approves a modified GPU request. Use a distinct result leaf such
as `unified-official-20261013-gpu`; never overwrite the CPU reference. Execution
must not begin until the reviewed source revision and external release/binding
records agree. Do not regenerate a training binding for an already trained run
as a way to disguise its original source provenance.

The external console release may use schema version 2: `frontend_commit` pins
the new reviewed console/evaluator, while `execution_commit` continues to pin
the original clean training checkout. Version 1 still requires equal commits.
Version 2 permits only the explicitly listed evaluation, console, test and
documentation files to differ between commits; training engines, models,
scientific configurations and historical manifests cannot change. Training and
CPU execution still delegate to the original checkout. CUDA admission runs in
a CUDA-hidden frontend child against the original training binding; only the
reviewed GPU worker executes the new extractor. Its receipt separately records
the evaluator release. Result registration preserves the original training
commit, terminal and launch receipts, and verifies the new evaluator receipt.

Release steps after review: commit only this evaluation revision, update the
frontend release pin using schema 2 while retaining the old execution pin,
publish a separately scoped `device: cuda` request with a distinct result leaf,
then run its `--check` before execution. Preserve the old release receipt and
approval bytes. Do not change release files while an old supervisor still needs
them without first verifying compatibility or arranging an explicit handover.
Implementation/qualification does not itself approve committing or publishing
this release, and does not stop or replace the existing CPU benchmark attempt.

Only feature extraction moves to CUDA. The source model, checkpoint, dimensions,
preprocessing, image order, batch size, target PID/camera filtering, FP32 precision,
normalization, NumPy distance, CMC, mAP, mINP and no-reranking policy are unchanged.
The shared extractor and metric implementation are reused. Stage A's default
evaluation API and quiet behavior remain unchanged.

GPU execution is restricted to the standing TITAN Xp UUID
`GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`. Before CUDA initialization the launcher
checks the request, terminal, source, data, same-account operator declaration,
GPU ownership/contention, at least 4096 MiB free GPU memory and 2 GiB disk space.
The frozen strict CUDA policy, single-UUID visibility, architecture compatibility
and runtime controls are reused. Checks repeat before inference and every ten
batches, with a final check before publication. Failures abort; no CPU fallback,
other GPU selection, AMP or silent reduction in batch size is allowed.

Logs show the first batch, every ten batches, the last batch, image count,
throughput, elapsed time and feature-extraction ETA. The ETA excludes later CPU
ranking. Distance, mAP, CMC and mINP stages and the final metrics are printed.
GPU results include the execution device, exact UUID, CUDA admission evidence
and deterministic policy in their provenance. Historical CPU identities remain
unchanged. Result SHA proofs and immutable training checkpoints are preserved.

CPU/GPU qualification is distinct from same-device training recovery. The
bounded diagnostic uses a fixed manifest of real Duke query/gallery samples
and the trained terminal. It compares normalized embeddings (atol=1e-5,
rtol=1e-4), CPU-computed distances (atol=2e-5, rtol=1e-4), and retrieval metrics
(atol=1e-6, rtol=0), and repeats CUDA extraction for exact equality. These are
new evaluation acceptance bounds; no existing training/recovery tolerance is
changed. Record maximum errors and the actual sample size, runtime and timings.
This does not establish full-Duke or every-architecture equivalence. Full CPU
and GPU benchmark comparison remains a separate post-evaluation gate.

No active evaluation should be terminated automatically. Keep its logs and
artifacts. Any later retirement of the CPU attempt requires the user's decision.
