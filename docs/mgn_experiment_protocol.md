# MGN offline experiment freeze

Date: 2026-10-27. Protocol: `mgn_offline_v1_fp32_sgd120`.

The admitted implementation is on `feature/mgn-offline`. Runtime/model source was qualified at `b719551` and its source-equivalent receipt commits. Launch-interface freeze revision `20261027_automatic_preflight_v1` adds automatic admission in the common trainer. Its implementation and qualification are recorded separately in [mgn_launch_interface_revision.md](mgn_launch_interface_revision.md). Original GPU evidence remains bounded to its recorded implementation; the revised trainer is not claimed byte-identical. Exact byte identities are in [mgn_experiment_manifest.json](mgn_experiment_manifest.json). The concrete handoff remains available for user/advisor review. **Official training has not been launched.**

This is Part 1 only. Within-domain, cross-domain and separately reported MGN LODO results are mandatory for the completed offline study. The four runs below are within-domain source training. LODO still requires the roadmap's versioned scientific/protocol/qualification gates; historical v1 ResNet50/PCB identities and results remain unchanged. Computational model selection, ONNX and robotic/runtime comparisons belong to later Part 2. TransReID has no implementation in this change.

## Frozen scientific recipe

Paper-v3-aligned `paper_v3_global2048_independent_v1`: shared ResNet50 through layer3[0], three independent residual branches, max pooling, eight independent 256-D reductions, three unreduced 2048-D global CE heads and five reduced local CE heads, all bias-free. Triplet uses the three reduced global features. Retrieval concatenates eight reduced features into a raw 2048-D descriptor. The roadmap contains the exact topology/order/initializer contract and evidence distinctions.

The training recipe is a project adaptation, not an exact reproduction of the authors' complete experiments. Total loss is mean eight-head ordinary batch-mean CE plus mean three-feature batch-hard Euclidean Triplet, family coefficients 1/1, margin1.2, smoothing0, Center disabled. SGD uses LR0.01, momentum0.9, weight decay0.0005 on all parameters, bias factor1, no prefix multipliers, Nesterov=false. Exactly120 completed epochs; no warmup; LR0.01 throughout epochs1–60, 0.001 during61–90 and 0.0001 during91–120. The scheduler advances after each completed epoch, before saving its next-epoch LR. Epoch numbers and optimizer-update counts are distinct.

Training uses P16/K4/B64, RGB384×128, explicit bicubic resize, mean[.485,.456,.406], std[.229,.224,.225], horizontal flip0.5; no padding/crop/erasing. Four workers, pinned memory, seed42, deterministic=false, cuDNN benchmark=true, FP32. Test batch32; raw original/width-flip mean before global L2; Euclidean distance; primary reranking disabled. Evaluation policy is source-owned during cross-domain evaluation. Periodic source test query/gallery mAP every10 epochs selects strictly improved `ckpt_best.pth`. This is a disclosed test-based selection policy with no independent validation split. Best-selected and final-epoch120 results must stay separate.

| Source | Training partition | Identities | Images | Planned fresh updates | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Market1501 | processed trainval | 751 | 12936 | 21999 | `exp/mgn/market1501` |
| Duke | processed trainval | 702 | 16522 | 27130 | `exp/mgn/duke` |
| CUHK03 | detected processed_partition trainval | 767 | 7365 | 11156 | `exp/mgn/cuhk03` |
| MSMT17 | raw list_train.txt only | 1041 | 30248 | 52426 | `exp/mgn/msmt17` |

Update budgets above are **index-only simulations**, without images, forward/backward or optimizer steps, of the unchanged PK sampler for120 fresh uninterrupted epochs. Per-epoch counts and index-stream hashes are recorded in the manifest. PK epoch lengths vary; do not multiply epoch1 length by120 or convert these epoch milestones into batch milestones. Actual interrupted/resumed exposure can differ because ordinary checkpoints do not restore sampler/RNG trajectories.

CUHK03's actual processed partition is preserved. Its semantic new/classic provenance is unverified; do not relabel it or claim verified paper split reproduction. MSMT17's separate validation list is not silently included. Dataset identities bind metadata/list hashes, ordered records, label maps, canonical file inventories (size/mtime/file identity), and all consumed smoke-image bytes. Full corpora are not byte-hashed or exhaustively decoded; this limitation remains explicit.

## Initialization and qualification

Fresh initialization requires the existing official historical `resnet50-19c8e357.pth` at `/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth`, 102502400 bytes, SHA-256 `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`. The loader verifies the same stream before legacy deserialization and never downloads or substitutes weights. Head initializer ID: `reference_a_384e820_bias_free_v1`. Trained-state reconstruction separately bypasses pretrained reads/downloads.

CPU gates include1231 passing consolidated cases (four CUDA-only skips), followed by the scoped continuation/source/preset regressions. Synthetic FP32 and each of the four real-source cases completed exactly three optimizer updates on TITAN Xp UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c`, each in a fresh process under600 seconds. CPU/source/GPU receipts are linked by the manifest. Cold autotuning peaked above steady training; consult the measured per-source bytes rather than assuming long-run headroom. These smokes do not establish long-run stability or benchmark accuracy. Optional AMP was not run and is not part of this primary recipe.

The user explicitly supplied current manual owner authorization for this GPU; no scheduler allocation is claimed. The read-only preflight requires the same full UUID, correct visibility, current uncontended access, matching admitted runtime/environment, source/config/code/weight/data identities, measured memory headroom, and at least10GiB free disk allowance. Reused outputs are refused. Existing unrelated dirty documentation/results may remain; frozen runtime/config/manifest changes are refused. No automatic push or merge occurred.

## Common-trainer launch commands

Use the Reid environment on Constantine and run **one source at a time**. Inspect completion before proceeding to another source. Each preset explicitly opts into `train.preflight: {name: mgn_offline_v1}`. The common trainer automatically runs the existing read-only preflight against the effective configuration before training dispatch, output creation, training model initialization, checkpoint writes or GPU training. Failure stops the command. The preflight does not select or change GPUs; the existing device/environment prerequisites below still apply. There is no separate mandatory preflight command. Do not bypass a refused guard or reuse another architecture's paths.

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
export CUDA_VISIBLE_DEVICES=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
export MGN_AUTHORIZED_GPU_UUID="$CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF CUBLAS_WORKSPACE_CONFIG NVIDIA_TF32_OVERRIDE CUDA_LAUNCH_BLOCKING
```

Market1501:

```bash
/home/addirakoze/environments/Reid/bin/python -u -B scripts/train.py --config configs/mgn/market1501.yaml
```

Duke:

```bash
/home/addirakoze/environments/Reid/bin/python -u -B scripts/train.py --config configs/mgn/duke.yaml
```

CUHK03:

```bash
/home/addirakoze/environments/Reid/bin/python -u -B scripts/train.py --config configs/mgn/cuhk03.yaml
```

MSMT17:

```bash
/home/addirakoze/environments/Reid/bin/python -u -B scripts/train.py --config configs/mgn/msmt17.yaml
```

These commands are prepared for the user; Codex has not executed them. The trainer prints the successful admission JSON before dispatch and saves it as `artifacts/training_preflight.json` only after admission succeeds and the run output is created. Keep that JSON, frozen config and manifest with the run provenance. Standalone `scripts/mgn_preflight.py --source <source>` remains available as an optional read-only diagnostic. The common trainer records its actual Git revision, resolved config, environment and command. Preflight accepts later documentation/preparation commits only when the qualified runtime bytes match and the tested implementation remains an ancestor; it never treats a different runtime recipe as equivalent merely because a branch name matches.

## Monitoring, completion and continuation

Monitor `exp/mgn/<source>/train.log` and `logs/{stdout,stderr}.txt`, plus current GPU/disk usage. Confirm CUDA logical0, FP32, B64/P16/K4, expected source C, finite losses, correct epoch LR windows and ongoing checkpoint writes. Stop and preserve evidence on OOM/nonfinite state, parser/data failures, wrong device/recipe or incomplete epoch; do not change batch, AMP, optimizer or schedule to obtain a pass. The read-only guard cannot replace monitoring throughout a long run.

Completion requires process success, epoch120 in `checkpoints/ckpt_last.pth`, scheduler `completed_epochs=120`, configuration/reconstruction/source provenance, and separate selected-best versus final-state artifacts. `metrics/final_test.json` identifies the re-evaluated selected checkpoint; `metrics/final_epoch_test.json` identifies the last evaluated final state. Their epochs/scores may differ. Never call a run complete from an intermediate best score or an existing output directory alone. Independent benchmark evaluation/registration remains a later authorized phase.

No automatic resume is performed. For a user-operated continuation, retain the original output and last completed-epoch checkpoint (epoch1–119), its optimizer/scheduler state, identical frozen code/config/data/maps/environment, explicit `selection_state` and the preserved matching selected-best artifact when one exists. The common trainer derives resume admission from the effective `train.save.resume` path and validates these before dispatch. It accepts only the same source's original `exp/mgn/<source>/checkpoints/ckpt_last.pth`, without symlink redirection. The standalone preflight's `--resume` mode remains an optional read-only diagnostic. Before restarting, preserve immutable copies/hashes of prior command/environment/resolved-config receipts and checkpoints; common training will publish new launch/config receipts. Reuse the frozen YAML and supply only the operational `train.save.resume` override to the common trainer. Never use the legacy missing-history fallback for this MGN study. Example (Market1501 only): `python scripts/train.py --config configs/mgn/market1501.yaml --opts train.save.resume=exp/mgn/market1501/checkpoints/ckpt_last.pth`. The automatic gate rejects scientific overrides, wrong config/source paths and attempts to remove/change the declared preflight via overrides.

Ordinary continuation preserves model/optimizer/epoch LR and historical selection, but not sampler/RNG/scaler trajectories or bitwise CUDA recovery. Record the continuation boundary and actual exposure separately; the manifest's fresh index trace no longer certifies its total updates. Do not describe a resumed trajectory as the uninterrupted fresh run. If the last checkpoint/selected weights or any identity is missing/inconsistent, preserve the failed attempt and resolve its provenance before continuing. No ad-hoc deletion, path reuse, target-based tuning or recipe correction is authorized by this handoff.

## Launch-interface freeze revision — 2026-10-27

The user authorized automatic launch admission without changing the scientific recipe. Revision `20261027_automatic_preflight_v1` changes the common trainer, admission adapter and opt-in schema, plus only the `train.preflight` operational field in each preset. The complete previous manifest, protocol, four YAMLs and Phase15 receipts are preserved byte-for-byte in [mgn_freezes/20261027_phase15](mgn_freezes/20261027_phase15), with their original commit and checksums. Earlier qualification receipts and roadmap history are not overwritten. The new manifest binds revised runtime/config hashes and separate launch-interface qualification; old GPU qualification is retained as prior architecture/recipe feasibility evidence, not a test of the changed trainer. No scientific setting, dataset definition, GPU choice, historical result or LODO v1 identity changes.
