# Market1501 MGN paper-schedule trial

Date: 2026-11-05. Status: preparation; training has not been launched.

The user authorized one isolated Market1501 experiment to test LR reductions after epochs40 and60, while retaining120 epochs and no reranking. The original paper (Wang et al., arXiv:1804.01438v3, Section4.1, https://arxiv.org/pdf/1804.01438v3) used these milestones with an80-epoch horizon. This is an extended120-epoch schedule trial, not an exact paper reproduction or a change to the completed baseline study.

## Experiment contract

- Config: `configs/mgn_experiments/market1501_paper_schedule120.yaml`.
- Experiment: `mgn_market1501_paper_schedule120`.
- Separate output: `exp/mgn/market1501_paper_schedule120`.
- The only scientific delta from `configs/mgn/market1501.yaml` is `sched.milestones: [40,60]`, replacing `[60,90]`.
- LR0.01 for epochs1–40,0.001 for41–60,0.0001 for61–120; scheduler steps after each completed epoch.
- Architecture, historical pretrained artifact/hash, initialization, loss heads/weights/margin/aggregation, SGD/momentum/weight decay, seed42, FP32, P16/K4/B64, data definitions, preprocessing/augmentation, workers and evaluation/checkpoint-selection policy are unchanged. Reranking stays disabled.
- Start fresh from ImageNet initialization; never resume/warm-start from the completed original run. No completed checkpoint is changed or reused for initialization.
- Operational deltas are only experiment name, output directory, explanatory notes and the dedicated admission profile.

The control run is `exp/mgn/market1501`: selected epoch70, Rank-1 88.18%, mAP71.00%; both selected and final-epoch metrics/hashes are documented in `docs/mgn_training_results.md` and its full-precision receipt. The control resumed from epoch86 after a partial epoch87; the fresh trial cannot be claimed a bitwise paired/uninterrupted sampling trajectory. Report this exposure difference when interpreting any change. Do not infer exact update counts from120 epochs or reuse index-only forecasts as measured counters.

Compare selected-best and final-epoch120 metrics separately, using Rank-1/5/10,mAP,mINP and the same retrieval protocol. Preserve the trial even if it does not improve; do not overwrite the baseline or silently replace dissertation results. No performance outcome is fabricated by this preparation.

## Admission and freeze lineage

The existing common trainer is used. Its generic named-preflight registry accepts the dedicated `mgn_market1501_paper_schedule120` profile; the MGN adapter selects one immutable Market1501 experiment entry. Unknown profiles/sources, wrong source/config/output, existing or redirected outputs, altered effective scientific configuration, weights/code/environment/data mismatches, insufficient disk, unavailable/contended or different GPU, and invalid resume state fail before training side effects. Profile changes via CLI overrides are refused. There is no automatic GPU selection, CPU fallback admission or new trainer.

The original four YAML presets and their complete manifest source entries remain identical. The prior active manifest was copied byte-for-byte to `docs/mgn_freezes/20261105_before_schedule_trial/mgn_experiment_manifest.json`, SHA-256 `b289ea8a1ee1ee03d2a2b41267e0286168241dcda3b07817d864df0f1d8f286e`. Historical run/registration records carrying that hash resolve to this immutable snapshot. No historical run receipt or completed result is rewritten. The active manifest receives a documented runtime/admission revision and the additional experiment entry. Earlier lineage and bounded GPU evidence remain separately identified; new runtime bytes are not claimed identical to the previous freeze.

Only `reid/utils/training_preflight.py` and `scripts/mgn_preflight.py` change in the runtime. Model, losses, optimizer, scheduler and common trainer code remain unchanged. Existing Market1501 FP32 batch/memory evidence is inherited for the identical model/batch/data geometry, not claimed as a new training-convergence test. No new GPU forward/backward smoke or full experiment is executed here.

Manifest status remains PREPARING until CPU qualification and an observed local implementation commit; a separate freeze commit will bind the qualified bytes before a read-only live admission test. The live test uses common `train.main()` and an unconditional stop before output creation. Qualification records and observed commit identities will be appended below.

## User-operated launch

In the existing qualified Reid environment with the previously agreed GPU/environment exports:

```bash
python scripts/train.py --config configs/mgn_experiments/market1501_paper_schedule120.yaml
```

Optional terminal-independent equivalent (run only one launch command):

```bash
nohup python -u scripts/train.py --config configs/mgn_experiments/market1501_paper_schedule120.yaml > /tmp/mgn-market1501-paper-schedule120.log 2>&1 < /dev/null &
```

Watch background startup/progress with `tail -n 50 -F /tmp/mgn-market1501-paper-schedule120.log`; persistent logs/checkpoints are created under the separate trial output only after automatic admission passes. Do not create the trial output directory manually before first launch.

Completion requires epoch120 in last checkpoint and scheduler, final selected/final-epoch metric artifacts, final-evaluation log and consistent provenance. Preserve original launch/config/checkpoint receipts before any continuation. A future user-operated resume must use the trial's own `checkpoints/ckpt_last.pth` and identical frozen configuration; the original completed Market1501 checkpoint is refused. Ordinary continuation preserves model/optimizer/epoch LR/selection, not bitwise stochastic replay.

Official training remains user-operated; Codex stops after preparation, tests, freeze verification and command handoff.

## Observed qualification

CPU qualification: 408 tests passed in 90.81 seconds. The exact test command, stdout, hashes and limitations are recorded in `docs/mgn_market1501_paper_schedule120_qualification.yaml`. Original four source contracts remain identical; 57 runtime hashes and seven qualification receipt hashes were verified. History additions are append-only and existing result/artifact documents remain unchanged.

Implementation commit: `b72c4a3979ff0d20e3036908bbccb7fce7eb681f`. The separate freeze revision binds these tested runtime bytes and this trial YAML. Training has not been launched; live admission evidence follows separately.

Read-only live admission at freeze commit `81d356be37ed0e1c8551bcb020aae7e4633b33a9`: file/hash/ancestry/configuration/environment/weights/data/disk checks passed, then GPU admission refused the occupied authorized device (`REFUSED_GPU_OCCUPIED`). Another user's KITTI/DPVO processes were observed; none was changed. Full evidence and the stop-before-output driver are in `docs/mgn_market1501_paper_schedule120_live_admission.yaml`. This is a resource refusal, not a successful READY_FRESH receipt. Wait until the authorized GPU is free before running the command. No trial output, model initialization, GPU forward/backward, optimizer step or training occurred.
