# Oneiros GPU qualification — 2026-11-06

Status: **MGN_FP32_GPU_QUALIFICATION_PASS**. This is bounded MGN feasibility
qualification, not training admission, a benchmark result, or LODO qualification.

The user authorized Oneiros GPU qualification in the migration conversation.
The device was idle before and after the sequential checks:

- NVIDIA RTX A4000, 16,376 MiB VRAM, driver 590.44.01.
- UUID: `GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2`.
- Python environment: `/home/addirakoze/environments/Reid`.
- Isolated checkout on Oneiros:
  `/home/addirakoze/nobackup/Projects/oneiros-gpu-qualification-20261106`.

The existing `scripts/mgn_gpu_smoke.py` harness ran unchanged. Each case used a
fresh subprocess, a fresh output directory, at most three optimizer updates,
and the harness's 600-second deadline. The recipe remained FP32, batch 64,
P16/K4, and 384×128 images, with the original model, losses, optimizer and
verified cached pretrained weights. No official trainer was launched.

| Case | Completed updates | Status | Peak reserved memory (GiB) |
| --- | ---: | --- | ---: |
| Synthetic | 3 | PASS | 10.7852 |
| Market1501 | 3 | PASS | 10.7871 |
| Duke | 3 | PASS | 10.7891 |
| CUHK03 | 3 | PASS | 10.7773 |
| MSMT17 | 3 | PASS | 10.7324 |

All cases passed finite loss, gradient, momentum and model-state checks, output
shape checks, and actual parameter-update checks. Real-data cases also verified
the consumed PK batches, pixels and image hashes. Their portable training record
and identity-map hashes match the original frozen source definitions.

CPU admission guards: **36 passed**. Current-code CPU synthetic/checkpoint/resume
and regression suite: **389 passed, 3 skipped**. The initial prelaunch refusal
was a stale historical CPU synthetic receipt's `scripts/train.py` hash. It
occurred before a GPU attempt; its log is preserved. The regression suite was
rerun before issuing replacement CPU evidence inside the isolated checkout.
Historical receipts and the original project's frozen code, configurations,
active manifest and experiment outputs were preserved.

## Evidence

On Oneiros, `qualification-evidence/qualification_receipt.json` contains the
consolidated report. Each raw worker manifest, receipt, stdout/stderr and summary
is retained under `exp/mgn_smoke/` in the isolated checkout. CPU logs and the
original receipts used during preparation remain in `qualification-evidence/`.

A local copy is retained under
`.local/server_migration/oneiros-gpu-qualification-20261106/`, with the consolidated
report in `evidence/qualification_receipt.json` and raw GPU receipts in
`mgn_smoke/`.

## Active Oneiros preflight integration

The user authorized installing the Oneiros evidence into this checkout only.
The active `docs/mgn_experiment_manifest.json` now identifies the RTX A4000 UUID,
current packages/runtime and Oneiros dataset inventories. The original manifest
is preserved in Git history and
`.local/host_admission/oneiros-20261106/constantine-active-manifest.json`.
The Constantine repository was not modified.

The active manifest references versioned evidence under
`docs/host_qualification/oneiros_20261106/`. Runtime code and scientific YAMLs
remain unchanged. The historical schedule-trial test now checks preserved
scientific contracts and portable data identity alongside the new qualification
evidence, instead of requiring Constantine's hardware/filesystem inventory.

Validation: **109 passed** in the focused preset, schedule-trial, training
preflight and smoke-guard suites. Both the standalone preflight and common
trainer's `run_training_preflight` adapter returned **READY_FRESH** for
`market1501_paper_schedule120`. Their receipts are saved beside the qualification
evidence. No training output directory was created and no training was launched.

Run this command from this checkout on Oneiros only when ready to start the
fresh trial; it is provided here without executing it:

```bash
MGN_AUTHORIZED_GPU_UUID=GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2 CUDA_VISIBLE_DEVICES=GPU-202f74b9-f9fe-8896-5cd7-78332b0e49e2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/addirakoze/environments/Reid/bin/python scripts/train.py --config configs/mgn_experiments/market1501_paper_schedule120.yaml
```

Preflight rechecks authorization, occupancy, memory, code/configuration/evidence
hashes, dataset inventories, environment and output freshness on every launch.
The preserved completed four-source outputs are not eligible for a fresh launch
at their existing paths. This integration does not authorize overwriting or
resuming those completed runs.

These checks do not qualify full-run stability, retrieval evaluation, AMP, LODO
execution or cross-host CUDA recovery. The three-update checks do not exercise
the later LR milestones in the isolated Market1501 paper-schedule trial.
