# Phase 9R.1 A/B/C continuation — resource-gated closeout

**GPU DIAGNOSTICS DEFERRED — RESOURCE/OWNERSHIP GATE**

A completed five successful original-policy updates. The fresh admission check immediately before B rejected GPU activity/headroom; B and C never launched. This is an incomplete numerical investigation, not a recovery PASS or a demonstrated numerical recovery FAIL. No deterministic GPU campaign, PCB recovery, official training, evaluation or Phase 10 occurred.

Evidence directory: `exp/lodo_smoke/phase9r1-abc-20261009T210631-original/` in the isolated LODO worktree. Times below are UTC (local São Paulo time is UTC−03:00).

## A. Repository preservation and changed files

Worktree: `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot-lodo`, canonical `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot-lodo`, branch `feature/lodo-v1`. Original checkout remains on `main`. Both HEADs remain `3b015c964b99bfe8ad4b7842bd174d3bc58af607`.

New files in this continuation only:

- `tests/lodo_cuda_abc_diagnostic.py`: diagnostic hooks around the original engine and checkpoint loader; original-policy and separately scoped deterministic fixture.
- `tests/test_lodo_cuda_abc_diagnostic.py`: 14 CPU contracts.
- `docs/lodo_cuda_abc_qualification.md`: this report.

The baseline was recorded before edits in `lodo-phase9r1-abc-before.json`, retained inside the evidence directory. SHA256 checks preserved all 174 original-checkout source files and 202 preexisting LODO source files. Existing Git changes were retained; the original checkout's exact Git status is unchanged. Existing artifact inventories (1,810 main and 325 LODO files) retain size, nanosecond mtime and inode. This artifact check is not a full content rehash of every artifact or image. `preservation-final.json` records both statuses and checks. Neither Stage A nor any production module/configuration was edited in this continuation. No stash/reset/clean, commits or pushes occurred.

The seven authoritative documents and relevant implementation/tests were read earlier in this continuing investigation; preservation hashes were rechecked, and the numerical investigation document reread. Earlier failed Phase 9R and CPU Phase 9R.1 evidence remains unchanged.

## B. GPU admission, ownership and exclusion

User authorization names only GTX 1080 `GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5` on Constantine. Initial audit: `2026-10-09T20:55:14.412236Z`. Admission before A: `21:06:31.575210Z`; child recheck: `21:06:37.987707Z`. GTX was physical index 1, logical `cuda:0`, 8,103 MiB free, 0% utilization, no compute process, driver `570.211.01`. Both visibility/authorization environment variables contained that exact UUID. Production guards also rechecked before updates.

After A exited successfully (249.345 seconds after its launch at `21:06:31.575266Z`), admission before B raised:

```
ValueError: GPU activity/headroom is not safe for a fresh launch
```

That predicate rejects zero free memory or nonzero reported utilization; its failing numeric sample was not retained. The error does not establish a competing process or a recovery defect. Transient utilization accounting after A is plausible but unproven. The final read-only audit at `21:11:54.812509Z` showed GTX idle, 8,103 MiB free and no compute processes. No retry or B/C launch followed the rejection, in accordance with the requested stop rule.

TITAN Xp `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` was excluded throughout. Initial audit observed the preexisting PCB PID 2370601 on TITAN. Later monitoring found that PID absent and TITAN idle (5 MiB used). No signal or training/configuration action was taken against PCB; its completion/result quality was not evaluated. TITAN availability did not authorize using it.

## C. Single common checkpoint and provenance

A's durable completed-update-2 generation:

```
resnet50/LODO-1/producer/checkpoints/ckpt_epoch_000001_u000000002_g74b4767782cc.pth
SHA256: 4daa3bda52eb138a2ec878bc84473a7e7e39df4b47a0bbdaee89fe98c804b178
identity_digest: 4742949a38fe428d32e49db240ff6e65ae7c1749846e5adb8b22881cc640acf1
```

The real checkpoint reader verified schema, manifest, scientific identity and completed-epoch boundary. Producer live model, Center, both optimizers, scheduler and RNG were checked exactly against serialized state. A continued in the producing process without invoking restoration, reaching update 5. The common generation remains retained alongside update 4; its SHA256 was rechecked after A and at closeout.

LODO-1 source inventory: Duke, detected CUHK03 processed trainval, MSMT17 raw train only; 54,135 images, 2,510 classifier/Center classes. Mapping SHA256: `e090e49fde838001a11fd7b41b8b23238bf38df34a36feeb601881082fcd312b`. Target Market-1501 was limited to authorized metadata inventory checks; no target loader or evaluation ran. Pretrained cached weight SHA256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`; no download occurred.

B and C have not restored this checkpoint. A/B/C common origin is therefore established for A and implemented for B/C, but not GPU-observed for B/C. Do not present the intended design as completed evidence.

## D. Per-update comparisons

| Update | A loss | A/B | A/C | B/C |
|---|---:|---|---|---|
| 3 | 10.1905860901 | NOT RUN | NOT RUN | NOT RUN |
| 4 | 10.0464496613 | NOT RUN | NOT RUN | NOT RUN |
| 5 | 10.1181545258 | NOT RUN | NOT RUN | NOT RUN |

A recorded 352 finite diagnostic events. For updates 3–5 these include independently observed indices/labels/full transformed-input hashes; forward outputs and loss terms; every convolution/BatchNorm output digest and bounded 512-element sample; all named model and Center gradients before Center scaling; Center gradients afterward; pre-Adam state fingerprints and complete first-convolution state; complete post-Adam model/moments/steps; post-Center state; scheduler/counters/LR/source exposure/RNG.

The comparison implementation computes exactness, max absolute difference, unchanged-tolerance failure counts, relative L2 and magnitudes, retaining earlier path as actual and later path as reference. Integer state requires equality; qualification additionally requires exact Adam step counters. **There are no observed A/B, A/C or B/C error counts, Adam first/second-moment comparisons, Center comparisons or bitwise results from this campaign.** A's trace cannot supply missing pairwise results.

## E. Earliest divergence and localization

Not determined: no restored GPU path ran. No numerical root cause or specific CUDA kernel can be attributed. The old independent-execution comparison already differed before restoration (15 Adam first-moment failures at update 2); its later 521 failures remain confounded and are not substituted for this missing controlled comparison.

Hooks instrument individual convolution and BatchNorm modules, including the first convolution and named parameter/moment/gradient state. Full activation hashes locate exact differences; bounded activation samples do not supply full-activation tolerance counts. Parameter/state comparisons use full tensors. Instrumentation synchronizes and hashes on CPU and can affect timing/autotuning; this remains a limitation to assess in any completed comparison.

## F. Numerical policy and results

Original state criterion remains `rtol=0.001, atol=0.0001`; loss criterion `rtol=0.001, atol=0.001`; integer/topology/provenance checks exact. No criterion, comparison direction, production flag, seed, loss, optimizer, sampler, transform, architecture, batch size or resolution was changed.

A runtime: PyTorch `2.7.1+cu126`, CUDA build `12.6`, cuDNN `90501`; benchmark `true`, cuDNN deterministic `false`, deterministic algorithms `false`, matmul TF32 `false`, cuDNN TF32 `true`, `CUBLAS_WORKSPACE_CONFIG` unset. Actual Adam implementation: `_multi_tensor_adam`, 161 calls per observed update, 483 over updates 3–5; optimizer group settings are retained in stage-03 evidence. No fused Adam path was observed.

Original-policy numerical qualification: **INCOMPLETE**. Deterministic GPU diagnostic: **NOT RUN**. A separate deterministic fixture is CPU-tested and requires `:4096:8` workspace before GPU initialization; it is not an executable scientific approval or evidence of deterministic CUDA success. Production validators still reject modified deterministic flags and short budgets outside diagnostic contexts.

## G. Defects, corrections and tests

No production recovery defect was demonstrated; no production correction or tolerance change was made. During pre-GPU harness development, CPU tests caught/prevented tracing a wrapper instead of the original engine and verified actual module/optimizer/RNG restore ordering and parameter ownership. A post-GPU harness-only correction scopes deterministic fixture flag exceptions locally; original-policy behavior is unchanged. Its 14 CPU tests and final 64-test recovery suite pass; deterministic GPU behavior remains untested.

A diagnostic-driver logging defect occurred: the final unprotected admission audit raised the same error and prevented the already-selected STOPPED status from being flushed. CPU-only closeout preserved the prior summary as `campaign-summary-before-stop-repair.json`, wrote explicit `failure-history.json`, and finalized `campaign-summary.json` as STOPPED/INCOMPLETE. It neither altered the checkpoint nor resumed GPU work. The historical executed driver remains intact. Before another campaign, make final error publication unconditional and retain the full rejected admission sample; review a fixed settling interval before fresh admission after a child exits, without bypassing/retrying a rejected gate.

## H. Readiness and PCB gate

A's execution/invariants PASS does not qualify ResNet50 numerical continuation. ResNet50 remains unqualified pending B/C and all pairwise checks. PCB CUDA recovery remains gated and was not started. TITAN use, official LODO runs and Phase 10 require separate explicit authorization/review.

## I. Commands, tests, duration and memory

All Python commands used `/home/addirakoze/environments/Reid/bin/python`, `-B`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. CPU tests set `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=.:tests` and pytest `-p no:cacheprovider`.

Final focused command (outside sandbox for two-worker tests):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests timeout 300 /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_lodo_cuda_abc_diagnostic.py tests/test_lodo_cuda_recovery.py tests/test_lodo_cuda_recovery_smoke_contract.py tests/test_lodo_cuda_numerical_diagnostic.py -q --junitxml=/tmp/lodo-abc-postchange-cpu.xml
```

Stage A command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests /home/addirakoze/environments/Reid/bin/python -B -m pytest -p no:cacheprovider tests/test_checkpoint.py tests/test_checkpoint_reconstruction.py tests/test_train_loop_optim.py -q --junitxml=/tmp/lodo-abc-stagea-cpu.xml
```

| Test run | Passed | Failed | Skipped | Pytest seconds |
|---|---:|---:|---:|---:|
| Initial staged harness | 13 | 0 | 0 | 16.75 |
| Pre-GPU focused recovery/harness/forensics | 63 | 0 | 0 | 28.56 |
| Final staged harness including deterministic-fixture isolation | 14 | 0 | 0 | 7.11 |
| Final installed focused suite | 64 | 0 | 0 | 28.46 |
| Stage A checkpoint/reconstruction/train-loop | 46 | 0 | 3 | 61.12 |

These runs overlap; the final focused plus Stage A evidence is **110 passed, 3 skipped**, not their cumulative sum. CUDA-dependent Stage A tests were skipped intentionally. An initial sandboxed broad run stalled after 15 tests in worker handling and was terminated (only identified pytest PIDs); its outside-sandbox rerun passed. It is not reported as a completed failing suite or a production regression. Earlier full-suite counts are historical, not rerun claims for this continuation.

Exact executed GPU child command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5 LODO_AUTHORIZED_GPU_UUID=GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5 /home/addirakoze/environments/Reid/bin/python -B tests/lodo_cuda_abc_diagnostic.py --campaign phase9r1-abc-20261009T210631-original --path A --policy original
```

The parent command was `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/addirakoze/environments/Reid/bin/python -B /tmp/lodo_abc_driver.py original`; its exact source is retained as `driver.py`. GPU execution: A PASS (5 primary, 5 Center steps, 320 accepted images), B/C NOT RUN. A measured 241.558 seconds inside the harness and 249.345 seconds child wall time, including metadata setup, checkpoint I/O and instrumentation—not a training-throughput benchmark. Peak allocated 7,004,904,448 bytes (~6,680 MiB), reserved 7,193,231,360 bytes (6,860 MiB). No OOM, nonfinite-state or execution failure occurred in A.

A retains 352 bounded CPU snapshots totaling 1,936,399,044 bytes (~1.80 GiB); whole initial campaign evidence ~3.03 GB including three checkpoint generations/terminal evidence. Full numerical state is retained for later comparisons, activation copies are bounded, and no training images were dumped or copied. The exact executed harness/test source is retained and matches pre-GPU `source-hashes.json`; the final harness differs only by the explicitly documented diagnostic fixture correction.

## J. Recovery capability, limitations and approval

The existing capability remains `epoch_boundary_rollback_cuda_v1`: completed-epoch recovery, zero workers, frozen FP32 recipe, same scientific/source-map/environment contracts. No mid-epoch cursor/prefetch recovery or bitwise GPU guarantee is added. This campaign observed a valid update-2 save and uninterrupted continuation, not CUDA restore/continuation.

No alternative numerical criterion is proposed or adopted from this incomplete evidence; CUDA variability itself has not been established by the controlled comparison. Further GPU execution requires explicit continuation approval after this safety stop and renewed safe admission. A later idle audit does not retroactively validate the rejected launch. Dataset path checks cannot establish absence of duplicate image content or shared real-world identities across datasets; CUHK03 official-new provenance remains unverified.

## K. Exact remaining work

1. Review this resource stop, repair driver failure-publication/admission-sample recording, and decide whether to authorize B/C against the preserved common checkpoint/A trace or a fresh complete A/B/C campaign. No retry is running.
2. Observe B and C CPU-first validation and exact live CUDA restoration, module/Center optimizer ownership and scheduler/RNG restoration from the same checkpoint.
3. Complete all three pairwise per-update numerical/structural/loss comparisons, including first/second moments, integer steps, Center and first-divergent named parameter/operation. Assess instrumentation effects if necessary.
4. Only if the original-policy comparison requires it and safe execution is explicitly reauthorized, run a separately labeled deterministic diagnostic; report unsupported operations without substitution. No production-policy change is authorized by that result.
5. Address a production defect only if demonstrated, with regression tests and controlled reruns. PCB recovery, wider Phase 9R completion, later GPUs and official launches remain separate review gates.

Stop status: **GPU DIAGNOSTICS DEFERRED — RESOURCE/OWNERSHIP GATE**.
