# MGN versus historical PCB: descriptive recipe comparison

Historical PCB:16 saved result hashes and4 source checkpoint hashes verified. MGN observations are separate GPU evaluations on Oneiros. These are complete-recipe observations with different training and extraction policies; they do not isolate architecture effects. MGN uses original/flip mean; historical PCB uses no flip. Both use Euclidean distance, global normalization and no reranking. Declared target selectors match. CUHK03 new/classic semantic correspondence remains unverified. No architecture winner or significance claim is made.

| Source → target | MGN mAP (%) | PCB mAP (%) | Difference (points) | MGN Rank-1 (%) | PCB Rank-1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| market1501 → market1501 | 71.00 | 79.54 | -8.54 | 88.18 | 91.92 |
| duke → duke | 66.87 | 70.12 | -3.25 | 81.69 | 84.74 |
| cuhk03 → cuhk03 | 45.85 | 59.66 | -13.81 | 46.00 | 63.36 |
| msmt17 → msmt17 | 19.67 | 43.40 | -23.73 | 43.16 | 70.56 |
| market1501 → duke | 21.35 | 22.63 | -1.28 | 37.84 | 39.54 |
| market1501 → cuhk03 | 7.47 | 7.12 | +0.35 | 6.79 | 6.36 |
| market1501 → msmt17 | 2.81 | 3.32 | -0.51 | 8.87 | 10.70 |
| duke → market1501 | 26.24 | 27.54 | -1.30 | 53.41 | 55.94 |
| duke → cuhk03 | 6.12 | 6.46 | -0.34 | 5.93 | 6.36 |
| duke → msmt17 | 4.58 | 5.28 | -0.69 | 15.10 | 16.85 |
| cuhk03 → market1501 | 18.92 | 25.46 | -6.54 | 42.13 | 49.79 |
| cuhk03 → duke | 11.55 | 15.81 | -4.26 | 21.36 | 29.08 |
| cuhk03 → msmt17 | 1.71 | 2.37 | -0.66 | 5.84 | 8.29 |
| msmt17 → market1501 | 19.01 | 26.93 | -7.93 | 42.22 | 52.67 |
| msmt17 → duke | 22.18 | 34.46 | -12.28 | 40.31 | 54.08 |
| msmt17 → cuhk03 | 9.10 | 11.66 | -2.56 | 9.07 | 11.64 |

Full precision, five metrics and hash checks are in [historical-pcb-comparison.yaml](mgn_evaluation_20261106/historical-pcb-comparison.yaml). Historical records remain unchanged.
