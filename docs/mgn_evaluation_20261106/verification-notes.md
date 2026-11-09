# Operational evidence and numerical verification notes

Scope: user-approved four original checkpoint diagonals and twelve off-diagonal GPU evaluations, retaining source settings/checkpoints and recording results incrementally. No training or tuning.

Precision terminology: parameters and inputs are FP32 and AMP is disabled. The qualified Oneiros controls set matmul TF32=false and cuDNN TF32=true. Ampere hardware can therefore use TF32 convolution arithmetic under the recorded cuDNN policy; the previous Pascal GPU did not have TF32 hardware. “FP32” here identifies tensors/AMP policy, not a guarantee of full-mantissa internal convolution arithmetic or bitwise cross-host equivalence. Both control values are retained in every cell's GPU-start receipt.

## Preserved preparation error

The first recorder attempt passed environment/checkpoint/data/GPU inventory checks but failed before launching a child evaluator because PyYAML cannot represent PyTorch's `TorchVersion` string subclass. The failed namespace `results/mgn_selected_matrix/oneiros_20261106/` retains its code, traceback and preparation-failure receipt. Serialization was corrected by conversion through JSON primitives and tested against the actual `torch.__version__` object. No scientific/runtime evaluator code changed. The fresh execution namespace is `results/mgn_selected_matrix/oneiros_20261106_attempt2/`.

## Market1501 standalone verification exception

The common standalone evaluator completed on CUDA logical0 / Oneiros RTX A4000, using original selected epoch70 and the exact registered checkpoint SHA. Recorded GPU model/input placement is FP32,2000 original/flip forwards over1000 batches. Source model, complete test data/preprocessing and evaluation settings match the saved checkpoint exactly; dataset inventories/partition bytes and checkpoint hashes pass.

| Metric | Original trainer | Oneiros standalone | Delta in fractions |
| --- | ---: | ---: | ---: |
| mAP | 0.7100086212158203 | 0.70999675989151 | −0.000011861324310302734 |
| mINP | 0.34995803236961365 | 0.34993869066238403 | −0.000019341707229614258 |
| Rank-1 | 0.8818289637565613 | 0.8818289637565613 | 0 |
| Rank-5 | 0.9483373165130615 | 0.9483373165130615 | 0 |
| Rank-10 | 0.9664489030838013 | 0.9664489030838013 | 0 |

The strict pre-execution absolute threshold1e-5 per fractional metric is exceeded for mAP and mINP. The first execution stopped and preserved its registry/failure before cross-domain evaluation. The difference is0.001186 and0.001934 percentage points, respectively. Different Pascal/Ampere hardware, cuDNN algorithm selection with benchmark=true/deterministic=false, and hardware-dependent TF32 support are possible numerical causes, not a proven diagnosis. The declared controls match the qualified host runtime; no recipe/data/checkpoint mismatch has been found. No tolerance was widened, checkpoint reselected, ranking policy changed or benchmark rerun to obtain a pass.

A user clarification was requested for recording cross-host numerical exceptions while continuing the full matrix. Remaining independent diagonal evaluations may proceed under the existing four-cell authorization without claiming Market1501 passed its strict tolerance. Final review status and exception disposition must remain explicit; cross-domain continuation must respect the user's response.

## Four-diagonal completion and remaining scope

All four independently evaluated original checkpoints completed successfully on GPU. Every evaluator exited0. The final artifact verifier confirms4 unique diagonal cells, source-owned preprocessing/retrieval, target-owned dataset definitions, fixed original checkpoint identities, original/flip CUDA input/model placement, all recorded artifact hashes and CSV full-precision values. See `diagonal-integrity-check.yaml` and `diagonal-discrepancies.yaml`.

| Source | Epoch | mAP (%) | Rank-1 (%) | Largest absolute difference from original trainer, percentage points |
| --- | ---: | ---: | ---: | ---: |
| market1501 | 70 | 70.999676 | 88.182896 | 0.001934 |
| duke | 40 | 66.873205 | 81.687611 | 0.044882 |
| cuhk03 | 120 | 45.847631 | 46.000001 | 0.071430 |
| msmt17 | 70 | 19.670908 | 43.159792 | 0.025731 |

All four have at least one metric outside the strict1e-5 fractional threshold. Artifact integrity PASS does not certify numerical consistency or human scientific acceptance. The maximum difference is0.0714302 percentage points (one CUHK03 query at Rank-1). The source/target/configuration/checkpoint checks found no semantic drift; a hardware-dependent numerical explanation remains plausible but unproven. The original results remain intact, these are separately recorded Oneiros standalone observations, and the proposed tolerance is unchanged.

The asynchronous user question asks whether to continue the full matrix with explicitly flagged cross-host numerical exceptions. No answer has been received in this execution segment. Consequently, none of the12 cross-domain cells was launched. The continuation script supports new cells only, checks preserved completed-cell hashes, and refuses existing incomplete outputs; it does not automatically retry or overwrite evaluations. Cross-domain invocation with a numerical-exception override requires the user's explicit response; the prepared flag is not evidence of approval.
