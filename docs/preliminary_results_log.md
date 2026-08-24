# Preliminary Results Log

This file is the working registry for the preliminary results chapter.

It has two goals:

1. keep one clean record of experiments already completed
2. give us one place to append the next results as new training runs finish

Date of consolidation: 2026-08-18

## Current Experimental Stage

The project is currently entering the `MSMT17` recipe-selection phase.

The intended order remains:

1. establish the MSMT17 strong-baseline reference
2. run the selected controlled MSMT17 ablations
3. consolidate the cross-dataset recipe conclusions

## Cross-Dataset Baseline Anchors

These are the strongest completed baseline anchors already present in `exp/`.
All mINP values in this registry use the corrected last-valid-positive-rank
calculation and were verified against their listed metric artifact.

| Dataset | Experiment | Artifact | mAP | mINP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Market1501 | `baseline_r50_triplet_market1501` | `exp/baseline_r50_triplet_market1501/metrics/final_test.json` | 0.8578 | 0.5906 | 0.9439 | 0.9813 | 0.9884 |
| DukeMTMC-ReID | `baseline_r50_triplet_duke` | `exp/baseline_r50_triplet_duke/metrics/final_test.json` | 0.7697 | 0.4104 | 0.8698 | 0.9452 | 0.9659 |
| CUHK03 detected | `baseline_r50_triplet_cuhk03_detected` | `exp/baseline_r50_triplet_cuhk03_detected/metrics/final_test.json` | 0.5610 | 0.4379 | 0.5757 | 0.7693 | 0.8471 |
| MSMT17 | `baseline_r50_triplet_msmt17` | `exp/baseline_r50_triplet_msmt17/metrics/final_test.json` | 0.5238 | 0.1170 | 0.7560 | 0.8620 | 0.8948 |

## Market1501 Loss-Mode Sweep Results

These are the main completed `Market1501` comparison runs already available.

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Smoke | `smoke_market1501_resnet50` | `exp/smoke_market1501_resnet50/metrics/final_test.json` | 1 | 0.0347 | 0.0053 | 0.0766 | 0.1876 | 0.2604 | one-epoch smoke only |
| One-epoch manual | `manual_market1501_one_epoch` | `exp/manual_market1501_one_epoch/metrics/final_test.json` | 1 | 0.0349 | 0.0055 | 0.0811 | 0.1915 | 0.2613 | one-epoch manual sanity run |
| Triplet only | `market1501_r50_triplet_only_official_experimental_settings` | `exp/market1501_r50_triplet_only_official_experimental_settings/metrics/final_test.json` | 30 | 0.0012 | 0.0010 | 0.0006 | 0.0048 | 0.0086 | clearly non-viable in current setup |
| ID only | `market1501_r50_id_only_official_experimental_settings` | `exp/market1501_r50_id_only_official_experimental_settings/metrics/final_test.json` | 120 | 0.8420 | 0.5601 | 0.9385 | 0.9783 | 0.9852 | strong classification-only baseline |
| Triplet + ID | `market1501_r50_triplet_id_official_experimental_settings` | `exp/market1501_r50_triplet_id_official_experimental_settings/metrics/final_test.json` | 120 | 0.8590 | 0.5916 | 0.9448 | 0.9837 | 0.9896 | main strong-baseline family |
| Triplet + ID + Center | `market1501_r50_triplet_id_center_official_experimental_settings` | `exp/market1501_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json` | 120 | 0.8604 | 0.5954 | 0.9436 | 0.9843 | 0.9902 | best completed base result in final-test artifacts |
| Triplet + ID + Center + rerank eval | `market1501_r50_triplet_id_center_rerank` | `exp/market1501_r50_triplet_id_center_rerank/metrics/latest_val.json` | 120 | 0.8626 | 0.9625 | 0.9463 | 0.9843 | 0.9887 | rerank mAP 0.9426, rerank Rank-1 0.9561 |

## Market1501 Phase 2 Results

These runs start the structure/ablation comparison around the strong baseline.

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `market1501_full_bag_of_tricks` | `exp/market1501_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.8583 | 0.5895 | 0.9427 | 0.9828 | 0.9893 | first Phase 2 reference run; slightly below the best center-loss variants |
| BNNeck off ablation | `market1501_bnneck_off` | `exp/market1501_bnneck_off/metrics/final_test.json` | 120 | 0.0009 | 0.0011 | 0.0003 | 0.0024 | 0.0033 | catastrophic collapse; BNNeck is essential in this setup |
| Stride-2 ablation | `market1501_stride2` | `exp/market1501_stride2/metrics/final_test.json` | 120 | 0.8457 | 0.5636 | 0.9391 | 0.9828 | 0.9896 | clearly worse than stride-1 reference; keep stride 1 in the canonical recipe |
| No random erasing ablation | `market1501_no_random_erasing` | `exp/market1501_no_random_erasing/metrics/final_test.json` | 120 | 0.8372 | 0.5579 | 0.9474 | 0.9798 | 0.9881 | lower mAP than reference; random erasing still helps retrieval quality overall |
| No label smoothing ablation | `market1501_no_label_smoothing` | `exp/market1501_no_label_smoothing/metrics/final_test.json` | 120 | 0.8600 | 0.5948 | 0.9397 | 0.9819 | 0.9881 | slightly higher mAP than reference, but lower Rank-1 and lower than center-loss winner on mAP |
| Full Bag-of-Tricks + Center | `market1501_r50_triplet_id_center_official_experimental_settings` | `exp/market1501_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json` | 120 | 0.8604 | 0.5954 | 0.9436 | 0.9843 | 0.9902 | equivalent coverage of the Phase 2 center-enabled variant already exists in repo |

## Additional Results Worth Keeping Separate

These are useful anchors but should not be mixed directly with the main final-test
table without noting their status.

| Experiment | Artifact | Status | Key Result |
| --- | --- | --- | --- |
| `market1501_r50_triplet_id` | `exp/market1501_r50_triplet_id/metrics/latest_val.json` | partial/intermediate | epoch 60, mAP 0.8360, Rank-1 0.9376 |
| `market1501_r50_triplet_id_center` | `exp/market1501_r50_triplet_id_center/metrics/latest_val.json` | latest-val anchor | epoch 120, mAP 0.8609, Rank-1 0.9463 |
| `msmt17_r50_triplet_id_center_official_experimental_settings` | `exp/msmt17_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json` | additive ablation | mAP 0.5300, Rank-1 0.7575 |

## Current Interpretation

- `Market1501` Phase 1 is effectively already populated with usable evidence.
- `triplet_only` is not competitive in this framework.
- `id_only` is strong, but both `triplet + id` and `triplet + id + center` are better.
- the strongest completed `Market1501` base run so far is `triplet + id + center`.
- the new `market1501_full_bag_of_tricks` reference is essentially aligned with the plain `triplet + id` family, but does not beat the center-loss variants.
- removing `BNNeck` is decisively invalid for this baseline; it should stay in the canonical recipe.
- switching to `last_conv_stride=2` degrades `mAP` and `Rank-1` versus the stride-1 reference, so stride 1 remains the correct baseline choice.
- removing random erasing improves `Rank-1` slightly but hurts `mAP` materially, so random erasing should stay in the canonical recipe unless final priorities shift toward top-1 only.
- removing label smoothing gives a tiny `mAP` increase over the plain reference but loses `Rank-1` and still does not beat the center-loss variant overall.
- the strongest completed `Market1501` recipe in the current record is still the center-enabled strong baseline.
- reranking provides a very large retrieval boost, but it should remain analysis-only.

## Market1501 Freeze Decision

`Market1501` is sufficiently covered to move on.

Current canonical winner to carry forward:

- base training recipe: `Triplet + ID + BNNeck + stride 1 + random erasing`
- strongest completed additive variant: `center.enabled=true`
- winning artifact anchor:
  `exp/market1501_r50_triplet_id_center_official_experimental_settings/metrics/final_test.json`

## Next Planned Dataset

The next dataset should be `DukeMTMC-ReID`, reproducing the same controlled
recipe family used on `Market1501`.

## DukeMTMC-ReID Phase 2 Results

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `duke_full_bag_of_tricks` | `exp/duke_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.7707 | 0.4130 | 0.8667 | 0.9443 | 0.9641 | Duke reference run; essentially matches the older Duke baseline anchor |
| BNNeck off ablation | `duke_bnneck_off` | `exp/duke_bnneck_off/metrics/final_test.json` | 120 | 0.0009 | 0.0009 | 0.0000 | 0.0036 | 0.0036 | catastrophic collapse; BNNeck is essential on Duke too |
| Stride-2 ablation | `duke_stride2` | `exp/duke_stride2/metrics/final_test.json` | 120 | 0.7574 | 0.3939 | 0.8546 | 0.9327 | 0.9556 | clearly worse than the stride-1 Duke reference; keep stride 1 |
| No random erasing ablation | `duke_no_random_erasing` | `exp/duke_no_random_erasing/metrics/final_test.json` | 120 | 0.7206 | 0.3634 | 0.8469 | 0.9237 | 0.9479 | materially worse than reference across all reported retrieval metrics; retain random erasing |
| No label smoothing ablation | `duke_no_label_smoothing` | `exp/duke_no_label_smoothing/metrics/final_test.json` | 120 | 0.7741 | 0.4045 | 0.8797 | 0.9448 | 0.9605 | improves mAP, mINP, and Rank-1 over the reference; label smoothing is not beneficial in this Duke run |
| Full Bag-of-Tricks + Center | `duke_full_bag_of_tricks_center` | `exp/duke_full_bag_of_tricks_center/metrics/final_test.json` | 120 | 0.7688 | 0.4154 | 0.8654 | 0.9443 | 0.9618 | slightly below the no-center Duke reference; center loss is not retained for the Duke recipe |

## DukeMTMC-ReID Current Interpretation

- the Duke full strong-baseline reference is now registered and stable.
- its result is effectively aligned with the earlier `baseline_r50_triplet_duke` anchor.
- removing `BNNeck` is decisively invalid on Duke as well.
- switching to `last_conv_stride=2` degrades Duke on both `mAP` and `Rank-1`, so stride 1 stays locked here as well.
- removing random erasing substantially degrades every reported Duke metric, so random erasing stays in the Duke recipe.
- removing label smoothing improves `mAP` by 0.0034 and `Rank-1` by 0.0130 over the plain Duke reference; this is the current strongest completed Duke variant.
- adding center loss produces mAP 0.7688 and Rank-1 0.8654, slightly below the no-center reference; it is not beneficial on Duke in this run.
- the planned Duke sweep is complete. The current Duke winner is `duke_no_label_smoothing`.

## CUHK03 Detected Phase 2 Results

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `cuhk03_full_bag_of_tricks` | `exp/cuhk03_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.5602 | 0.4409 | 0.5657 | 0.7714 | 0.8500 | matched no-center reference; broadly aligned with the prior CUHK03 detected baseline anchor |
| Full Bag-of-Tricks + Center | `cuhk03_full_bag_of_tricks_center` | `exp/cuhk03_full_bag_of_tricks_center/metrics/final_test.json` | 120 | 0.5698 | 0.4499 | 0.5814 | 0.7657 | 0.8457 | first completed CUHK03 Phase 2 run; exceeds the prior CUHK03 detected baseline anchor on mAP and Rank-1 |
| No label smoothing ablation | `cuhk03_no_label_smoothing` | `exp/cuhk03_no_label_smoothing/metrics/final_test.json` | 120 | 0.6352 | 0.5119 | 0.6571 | 0.8271 | 0.8914 | strongest completed CUHK03 run; substantially improves every reported metric over the matched reference |
| No label smoothing replication (seed 1337) | `cuhk03_no_label_smoothing_seed1337` | `exp/cuhk03_no_label_smoothing_seed1337/metrics/final_test.json` | 120 | 0.6439 | 0.5136 | 0.6757 | 0.8386 | 0.9029 | independently seeded replication; confirms the high no-label-smoothing result and is the strongest CUHK03 run so far |
| No random erasing ablation | `cuhk03_no_random_erasing` | `exp/cuhk03_no_random_erasing/metrics/final_test.json` | 120 | 0.5141 | 0.4016 | 0.5193 | 0.7221 | 0.8029 | worse than the reference across every reported metric; retain random erasing |
| Stride-2 ablation | `cuhk03_stride2` | `exp/cuhk03_stride2/metrics/final_test.json` | 120 | 0.5194 | 0.4003 | 0.5243 | 0.7336 | 0.8236 | worse than the stride-1 reference; keep stride 1 |

## CUHK03 Detected Current Interpretation

- `cuhk03_full_bag_of_tricks_center` completed successfully using `ckpt_best.pth` at epoch 120.
- `cuhk03_full_bag_of_tricks` completed successfully using `ckpt_best.pth` at epoch 120.
- in the matched comparison, center loss improves mAP from 0.5602 to 0.5698 and Rank-1 from 0.5657 to 0.5814.
- removing label smoothing produces a strong seed-42 result: mAP improves by 0.0750 and Rank-1 by 0.0914 versus the matched reference.
- the seed-1337 no-label-smoothing replication is even stronger (mAP 0.6439, Rank-1 0.6757), so the high result is reproducible across two no-label-smoothing seeds. To attribute the seed-1337 gain specifically to removing label smoothing, a seed-1337 label-smoothed reference is still needed as the matched control.
- removing random erasing reduces mAP by 0.0461 and Rank-1 by 0.0464 versus the matched reference; random erasing stays in the CUHK03 recipe.
- switching to stride 2 reduces mAP by 0.0409 and Rank-1 by 0.0414 versus the matched reference; stride 1 stays in the CUHK03 recipe.
- the CUHK03 recipe sweep is practically complete. A seed-1337 label-smoothed control remains an optional verification run, not a blocker for opening MSMT17.

## MSMT17 Phase 2 Results

| Family | Experiment | Artifact | Epoch | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Bag-of-Tricks reference | `msmt17_full_bag_of_tricks` | `exp/msmt17_full_bag_of_tricks/metrics/final_test.json` | 120 | 0.5214 | 0.1177 | 0.7531 | 0.8607 | 0.8931 | fresh matched MSMT17 reference; aligned with the earlier MSMT17 baseline anchor |
| No label smoothing ablation | `msmt17_no_label_smoothing` | `exp/msmt17_no_label_smoothing/metrics/final_test.json` | 120 | 0.5432 | 0.1260 | 0.7669 | 0.8741 | 0.9049 | improves every reported metric over the matched reference; current strongest completed MSMT17 variant |

## MSMT17 Current Interpretation

- both initial MSMT17 Phase 2 runs completed successfully using `ckpt_best.pth` at epoch 120.
- removing label smoothing improves mAP by 0.0218 and Rank-1 by 0.0138 versus the matched reference, matching the positive direction observed on Duke and CUHK03.
- the remaining planned MSMT17 comparisons are random erasing, stride, and the matched center-loss variant.

## Update Rule For New Runs

After each new experiment finishes, append:

- experiment name
- exact config or override command
- dataset
- checkpoint used for reporting
- epoch
- mAP
- mINP
- Rank-1
- Rank-5
- Rank-10
- rerank metrics if applicable
- one short interpretation note
