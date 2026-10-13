# Phase 9R.1 — Backward localization and deterministic recovery candidate

**NON-AUTHORITATIVE. Original-policy numerical FAIL remains unchanged.**

Original backward nondeterminism is now localized to actual autograd operations. Diagnostic strict execution produced 0 nonexact gradient tensors and 0 nonexact post-update state tensors across the two independent update-3 probes. The fresh same-policy A/B/C campaign then passed all nine comparisons at updates 3–5; all 1,056 recorded event comparisons were bitwise equal. Production execution policy is unchanged.

## A. Repository and preservation

Implementation worktree: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`, branch `feature/lodo-v1`. Original remains `main`. Both HEADs remain `3b015c964b99bfe8ad4b7842bd174d3bc58af607`. Canonical roots start `/mnt/hd-data/addirakoze/Projects/`.

Added only `tests/lodo_cuda_backward_probe.py`, `tests/test_lodo_cuda_backward_probe.py`, `tests/lodo_cuda_deterministic_continuation.py`, `tests/test_lodo_cuda_deterministic_continuation.py`, and this report. Existing A/B/C harness and every production module/configuration are unchanged. All 174 original source hashes, all 212 preexisting LODO source hashes, both HEADs, original Git status and the preexisting artifact size/mtime/inode inventories (1,777 main; 1,513 LODO entries) match. Full statuses and exact new-source list are retained in `cpu-evidence/lodo-backward-preservation-final.json`.

The original TITAN campaign's **810 files**, including historical A/B/C artifacts, were checked by full SHA256 before/after probes and again after the deterministic qualification; changed files: []. Both failed new harness attempts were separately frozen and preserved. Dataset images and historical/PCB checkpoints/logs were never written. Artifact-stat checks outside the TITAN campaign are not full-content hashes. No reset/stash/clean, commit/push or official output creation occurred.

## B. Authorization and ownership

The user's exact TITAN allocation confirmation plus their **“do it”** approval authorized this diagnostic scope. Target: `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` only. GTX `GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5` was excluded.

Fresh host audit showed TITAN idle, 12,179 MiB free, utilization zero, driver 570.211.01, no compute process; historical PCB PID 2370601 was absent. Prior PCB completion evidence was preserved; no PCB training/evaluation/registration was performed. Parent/child journals retain timestamped numeric metrics, ownership and exact UUID -> logical `cuda:0` checks. Fresh headroom remains 7,500 MiB. No admission rejection was retried or threshold changed. Monitoring found only this campaign's own GPU PID. Parent CUDA initialized: False.

## C. Probe and public execution contracts

Probe directory: `exp/lodo_smoke/phase9r1-backward-titan-20261009T231550`. Every child uses original source loaders/transforms, PK16×4 batch64, 256×128 FP32, actual ResNet50/BNNeck/2048-D, ID smoothing .1 + Triplet .3 + Center .0005, Adam and Center SGD .5, and the original update scheduler. LODO-1 has live validated C=2510 and map SHA256 `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`.

Duke/CUHK detected processed trainval and MSMT raw train are unchanged. Market target access remains metadata-only, with throwing guards against target loaders/evaluation/legacy output setup. CUHK official-new provenance remains unverified.

The same original common update-2 SHA256 is used by every probe: `562a0ed8fb673dbbfd3101fa4514fe41b2b80956a3ef664c1a4362a8ba6222ce`. Real restoration validates on CPU before CUDA selection, then verifies live modules, learned centers, optimizer Parameter associations, both optimizers, scheduler and RNG exactly. Workspace is preconfigured for strict children before CUDA initialization; other strict runtime overrides are applied after original-identity restoration. These cross-policy one-update probes are diagnostics, not compatible production resumptions.

The shared engine executes only update 3. A test-only in-memory progress cap becomes 3 after the Center step, stopping before another batch; official config, validators and budgets remain unchanged. No checkpoint/terminal is published by these probes. Raw `state.pth` is diagnostic evidence, not a resumable checkpoint.

| Probe | Execution | Main count | Center count | Harness seconds | Peak allocated / reserved bytes |
|---|---|---:|---:|---:|---|
| original-1 | PASS | 3 | 3 | 315.95 | 9308494848 / 9915334656 |
| original-2 | PASS | 3 | 3 | 316.72 | 9308494848 / 9915334656 |
| deterministic-1 | PASS | 3 | 3 | 326.03 | 4683734016 / 5356126208 |
| deterministic-2 | PASS | 3 | 3 | 326.80 | 4683734016 / 5356126208 |

## D. Operation-level cause

Both original-policy probes have exact batch/input hashes, labels, RNG, forward hashes, losses, LR and exposure; all 374 autograd nodes execute in matching order. Yet **81 gradient tensors differ**, max abs 1.87754631042e-06. Update-3 gradient/state outside counts remain zero under the original criteria. The previously preserved update-4/5 original-policy failures are not replaced by this one-update PASS.

At the following nodes, full hashes prove identical recorded saved operands and incoming gradients, with different outgoing gradients. Convolution slot 1 is the filter/weight gradient; slot 0 is the input gradient. The Center branch and backbone branches show independent numerical variability.

| Node | Operation | Source module | Differing outputs |
|---:|---|---|---|
| 21 | IndexSelectBackward0 | Center indexed selection | 0 -> center |
| 39 | ConvolutionBackward0 | backbone.7.2.conv3 | 1 -> backbone.7.2.conv3.weight |
| 84 | ConvolutionBackward0 | backbone.7.2.conv1 | 1 -> backbone.7.2.conv1.weight |
| 98 | ConvolutionBackward0 | backbone.7.1.conv1 | 1 -> backbone.7.1.conv1.weight |
| 191 | ConvolutionBackward0 | backbone.6.0.downsample.0 | 1 -> backbone.6.0.downsample.0.weight |
| 212 | ConvolutionBackward0 | backbone.6.0.conv2 | 0 -> input gradient, 1 -> backbone.6.0.conv2.weight |

The first observed differing operation is Center `IndexSelectBackward0`; the first backbone boundary is `backbone.7.2.conv3` convolution weight-gradient computation. Center gradient max abs difference is `1.4551915228366852e-11`. `backbone.6.0.conv2` additionally produces a different input gradient, which propagates further backward. This establishes variability inside backward before Center inverse-weight scaling and Adam; checkpoint corruption/optimizer restoration is not the demonstrated source.

Internal CUDA/cuDNN algorithm IDs and individual kernel names were not captured. The result identifies autograd operation boundaries, not a specific internal reduction implementation. Full hashes detect changes outside the 128-element bounded samples; sampled differences are never substituted for full-tensor tolerance counts. Complete parameter gradients and post-update state are compared separately. Instrumentation synchronizes/transfers to CPU and can affect timing/autotuning; original-policy affected tensors differ somewhat from historical traces. No full images were saved or copied.

## E. Strict candidate and repeatability

| Runtime setting | Original | Strict diagnostic |
|---|---|---|
| cuDNN benchmark | true | false |
| cuDNN deterministic | false | true |
| deterministic algorithms | false | true |
| CUBLAS_WORKSPACE_CONFIG | unset | `:4096:8` before process/CUDA setup |
| matmul TF32 | false | false |
| cuDNN TF32 | true | true |

Torch 2.7.1+cu126, CUDA build 12.6, cuDNN 90501 and the exact TITAN UUID are held fixed. The complete setting bundle was tested; necessity of each individual setting was not isolated. No loss/model/sampler/optimizer formula or tolerance was changed. Strict mode was not bypassed or downgraded. The actual global-pooling graph uses `MeanBackward1`; no pooling replacement was required.

Strict pair: batch/forward exact True/True; loss/labels/LR/exposure/RNG exact True/True/True/True/True. Nonexact gradients/state: 0/0; outside counts 0/0. Differing node count: 0.

## F. Fresh A/B/C qualification

Fresh deterministic campaign: `exp/lodo_smoke/phase9r1-abc-titan-20261009T234014-deterministic/continuations/bc-20261009T234829`. Continuation controller execution **COMPLETED**; original numerical criteria **PASS**. A completed all five updates, but its preserved operational outcome remains **FAIL** due solely to the diagnostic postflight assertion described in section G. Its complete runtime and terminal artifacts were independently validated before B/C; A was not rerun. New common checkpoint SHA256 `474b6c595dea0fd5d4df7af81f3e7f4c24da3c34543a9148d75c2a9ad5caf056`; identity `d8cf592fcc0054de5495b2cb32e2f2e70a855aecd7cb7379cf25aa0ddd770a1d`. A produced this boundary and continued without loading it; B/C independently restored that same new artifact. This campaign does not replace preserved original-policy A.

| Pair | Update | Gate | Adam first-moment outside | Adam first-moment nonexact tensors | Model outside | Center outside | Structural exact |
|---|---:|---|---:|---:|---:|---:|---|
| A/B | 3 | PASS | 0 | 0 | 0 | 0 | True |
| A/B | 4 | PASS | 0 | 0 | 0 | 0 | True |
| A/B | 5 | PASS | 0 | 0 | 0 | 0 | True |
| A/C | 3 | PASS | 0 | 0 | 0 | 0 | True |
| A/C | 4 | PASS | 0 | 0 | 0 | 0 | True |
| A/C | 5 | PASS | 0 | 0 | 0 | 0 | True |
| B/C | 3 | PASS | 0 | 0 | 0 | 0 | True |
| B/C | 4 | PASS | 0 | 0 | 0 | 0 | True |
| B/C | 5 | PASS | 0 | 0 | 0 | 0 | True |

Independent CPU closeout checked all 352 event hashes per run and all **1,056 pairwise event comparisons**: full hashes, finite flags, tensor equality and non-tensor structure are exact. Adam used `_multi_tensor_adam` in all runs (483 instrumented calls per path across updates 3–5). Both resumes restore on CPU before CUDA admission; A never restored. All paths reach five main/Center updates and 320 cumulative accepted images.

| Update | Identical total loss in A/B/C |
|---:|---:|
| 3 | 10.19058609008789 |
| 4 | 10.04645633697510 |
| 5 | 10.11817359924316 |

| Path | Harness seconds | Peak allocated bytes | Peak reserved bytes |
|---|---:|---:|---:|
| A | 230.51 | 4681079808 | 5154799616 |
| B | 254.06 | 4683734016 | 5356126208 |
| C | 291.91 | 4683734016 | 5356126208 |

These instrumented timings include intensive evidence capture and CPU transfers; they are not throughput comparisons. Final campaign audit at `2026-10-09T23:58:09.451718Z` found TITAN at 0% utilization with 12,179 MiB free and no compute processes. Parent CUDA stayed uninitialized. Evidence: continuation `cpu-evidence/lodo-deterministic-final-audit.json`.

State/gradient criterion remains `rtol=0.001, atol=0.0001`; loss `rtol=0.001, atol=0.001`; integer/structural invariants exact. Success under strict settings cannot qualify the original nondeterministic policy. Observed bitwise equality is limited to these runs/hardware/software and does not establish a universal or long-run guarantee.

## G. CPU tests and harness failures

**215 unique passed, 0 failed, 3 skipped** after deduplicating overlapping cases: 19 new backward cases, 6 postflight-wrapper cases, 31 existing A/B/C cases, 22 admission cases, 91 device/recovery/forensic cases, and 46 Stage A cases. The three existing CUDA-dependent Stage A skips are not GPU evidence.

All commands use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests`, interpreter `/home/addirakoze/environments/Reid/bin/python -B`, pytest `-p no:cacheprovider`, from the isolated worktree.

```bash
/home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_backward_probe.py tests/test_lodo_cuda_abc_diagnostic.py tests/test_lodo_cuda_admission_diagnostic.py -q --junitxml=/tmp/lodo-backward-first.xml
timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_device.py tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_recovery_smoke_contract.py tests/test_lodo_cuda_numerical_diagnostic.py tests/test_checkpoint.py tests/test_checkpoint_reconstruction.py tests/test_train_loop_optim.py -q --junitxml=/tmp/lodo-backward-regressions.xml
/home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_backward_probe.py tests/test_lodo_cuda_abc_diagnostic.py -q --junitxml=/tmp/lodo-backward-graph-complete.xml
/home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_deterministic_continuation.py tests/test_lodo_cuda_backward_probe.py tests/test_lodo_cuda_abc_diagnostic.py -q --junitxml=/tmp/lodo-deterministic-postflight-final.xml
```

Initial group 68 passed/7.72 s; recovery/Stage A group 137 passed/3 skipped/81.29 s; corrected final group 50 passed/7.89 s. Intermediate corrected validator run 47 passed/7.65 s overlaps these and is not added. Production bytes were unchanged throughout. Postflight wrapper/focused run: 56 passed/7.95 s. Two new fixture cases initially failed because they used a valid production output namespace; the fixture was corrected to a smoke namespace and the guards were preserved. These were new-test fixture failures, not baseline production regressions. Full suite was not rerun. A separate CPU Conv/BN/in-place-ReLU/MaxPool/AdaptiveAvgPool graph check passed with 16 complete nodes; it is not added to pytest counts.

Two probe collector errors were preserved: `231010` inherited a fresh-output validator and stopped before CUDA; `231154` restored exactly and performed forward, but Boolean `abs_cpu` summary failed before backward or optimizer steps. Exact executed sources, logs, outcomes and STOPPED summaries remain in those namespaces. The first was corrected by reusing the existing verified resume validator; the second by supporting Boolean evidence and adding full loss-graph integration. There was no resource rejection, scientific numerical failure, silent repair, or production workaround in these attempts.

A separate deterministic A/B/C attempt completed A through update 5, published the valid terminal SHA256 `fbc838e603a6ad322838173a63c27f6277d42ca9633439be7eeaac5223355c3b`, then stopped on an old diagnostic postflight assertion. The real production validator correctly rejected `repro.deterministic=True`; the old smoke assertion incorrectly expected the first error to mention `max_updates`. A's original FAIL outcome and original STOPPED driver summary are unchanged. The new wrapper independently proves frozen-policy rejection, then short-budget/output rejection on a copy with original flags; no existing validator/harness file is edited. CPU validation checked all 352 A event hashes and exact terminal model/centers/Adam/Center-optimizer/scheduler/RNG consistency before B/C. A was neither rerun nor replaced by a restored reference.

## H. Commands, evidence and resource limits

Executed probe parent:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests timeout 1800 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_backward_probe.py
```

Children set both `CUDA_VISIBLE_DEVICES` and `LODO_AUTHORIZED_GPU_UUID` to the exact TITAN UUID, and invoke the same script with `--base <new canonical campaign> --probe original-1` (then original-2, deterministic-1, deterministic-2). Strict children additionally set workspace `:4096:8`. Each child timeout is 1200 s; all are sequential. Exact arrays, admissions, timings and outcomes are retained in `summary.json`. Slow hash/CPU synchronization time is not training throughput. No batch/resolution change was used.

Executed new deterministic A campaign (complete runtime, then preserved postflight failure):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests timeout 1800 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_titan_abc_campaign.py --allocation-receipt exp/lodo_smoke/phase9r1-abc-titan-20261009T214240-original/allocation-confirmation.json --policy deterministic
```

B/C after the postflight correction were executed with the same CPU-hidden parent environment:

```bash
timeout 1500 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_deterministic_continuation.py
```

The new child wrapper calls the existing `abc.run` with `policy=deterministic` and a fresh B/C continuation ID, patching only diagnostic postflight checking. Exact commands and full unchanged-A campaign byte-hash verification are retained in that continuation's `final-summary.json` and `preserved-A-campaign-sha256.json`. Parent validates A before admission. B/C read the same common SHA and retain the same original numerical criteria.

Full node evidence, whole named gradients/state, numeric analyses, source hashes, JUnit XML, initial/final preservation snapshots and failed-attempt hashes are archived in this non-authoritative campaign. SHA256 is integrity evidence for trusted local files, not authentication or proof of absence of duplicate dataset content/people.

## I. Readiness and protected work

**STRICT CANDIDATE — SHORT TITAN RESNET50 RECOVERY QUALIFICATION PASS.** The new deterministic candidate passed the numerical gate for these tested conditions; production adoption remains gated. Original-policy classification remains **INCOMPLETE — NUMERICAL INVESTIGATION REQUIRED**. Its historical FAIL is unchanged. A successful strict-policy qualification is a working diagnostic execution-policy candidate only. It does not approve changing D17's frozen production settings, imply official readiness, qualify PCB/GTX/all folds/long-run recovery, or establish arbitrary mid-epoch continuation.

Stage A/source loaders/configs/models/losses/samplers/optimizer/scheduler/recovery/orchestration/evaluation APIs remain byte-unchanged. Original checkout and completed PCB artifacts remain protected. No official LODO training, real target evaluation, PCB recovery qualification, Phase 10, commits or pushes occurred.

## J. Next review

Review adoption of the strict execution-policy bundle and its performance/resource implications before changing production protocol/configuration. Production checkpoint compatibility would also need to enforce all approved runtime controls, including global deterministic algorithms and workspace (not currently part of the bounded saved environment contract). Preserve original failures and keep policies distinct. PCB recovery remains a separately approved gate. Stop after this report; no official launch or subsequent GPU scope is implied.
