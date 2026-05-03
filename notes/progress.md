# ReID Baseline Progress

## Current State

This file is the handoff note for future Codex sessions.

## Completed

### Reproducibility Controls Lock

Implemented:
- verified and hardened `reid/utils/seed.py`
  - `random.seed(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `torch.backends.cudnn.deterministic = deterministic`
  - `torch.backends.cudnn.benchmark = benchmark`
- run artifact utility added:
  - `reid/utils/artifacts.py`
- artifact utility writes:
  - `command.txt`
  - `environment.txt`
  - `git_commit.txt` when a git commit is available
- `environment.txt` records:
  - Python version
  - executable path
  - platform
  - current working directory
  - PyTorch version
  - CUDA availability/version
  - cuDNN version
  - CUDA device names when available
- run artifact writing is wired into:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/smoke_reid_pipeline.py`
- existing resolved config artifact remains:
  - `config.resolved.yaml`
- tests added:
  - `tests/test_reproducibility_artifacts.py`

What It Locks:
- the `repro:` YAML section controls the major deterministic/runtime seed knobs used by training
- train/evaluate/smoke runs leave enough metadata to reconstruct how the run was launched
- each real experiment run saves:
  - `exp/<experiment>/config.resolved.yaml`
  - `exp/<experiment>/command.txt`
  - `exp/<experiment>/environment.txt`
  - `exp/<experiment>/git_commit.txt` when git is available
- missing git metadata does not fail a run; `git_commit.txt` is optional by design

Validation:
- broad framework regression:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_reproducibility_artifacts.py tests/test_config_schema.py tests/test_smoke_reid_pipeline.py tests/test_dataset_protocol.py tests/test_msmt17_dataset.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py tests/test_rerank.py tests/test_model_forward.py tests/test_train_loop_optim.py tests/test_reid_loss_modes.py tests/test_train_orchestration.py tests/test_checkpoint.py`
- result in this environment: `172 passed, 4 skipped in 51.15s`

How To Test:
- focused reproducibility checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_reproducibility_artifacts.py`
- run a loader-only smoke and inspect artifacts:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python scripts/smoke_reid_pipeline.py --config configs/baseline_msmt17_resnet50_triplet.yaml --root /path/to/Dataset --skip-batch`
  - check `exp/baseline_r50_triplet_msmt17/command.txt`
  - check `exp/baseline_r50_triplet_msmt17/environment.txt`
  - check `exp/baseline_r50_triplet_msmt17/git_commit.txt` if git is available

### YAML Configuration Schema Lock

Implemented:
- experiment-level YAML schema validator added:
  - `reid/utils/config_schema.py`
  - public API: `validate_config(cfg: dict) -> None`
- required top-level sections are now checked:
  - `experiment`
  - `system`
  - `repro`
  - `logging`
  - `data`
  - `model`
  - `loss`
  - `optim`
  - `sched`
  - `train`
  - `eval`
- required keys are now checked:
  - `experiment.name`
  - `experiment.output_dir`
  - `system.device`
  - `data.train.dataset.name`
  - `data.test.dataset.name`
  - `model.name`
  - `optim.name`
  - `train.epochs`
  - `eval.topk`
- `system.device` is constrained to:
  - `auto`
  - `cpu`
  - `cuda`
- validation errors are explicit:
  - missing section: `Missing config section: data`
  - missing key: `Missing config key: experiment.name`
  - bad device: `Invalid config value for system.device: 'gpu'. Expected one of: auto, cpu, cuda.`
- schema validation is wired into full-config entrypoints:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/smoke_reid_pipeline.py`
  - `scripts/debug_transforms.py`
  - `scripts/smoke_cuhk03_dataset.py`
- validation runs after `load_config(..., overrides=...)`, so command-line overrides can still repair or change config values before schema checks
- schema tests added:
  - `tests/test_config_schema.py`

What It Locks:
- one YAML controls the whole experiment before dataset/model/loss builders run
- invalid experiment YAML fails early with a clear error instead of surfacing later as an unrelated builder or key error
- model/loss semantic validation remains separate in `validate_reid_config`
- partial unit-test configs can still call lower-level builders directly without pretending to be full experiment configs

Validation:
- full framework regression:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_config_schema.py tests/test_smoke_reid_pipeline.py tests/test_dataset_protocol.py tests/test_msmt17_dataset.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py tests/test_rerank.py tests/test_model_forward.py tests/test_train_loop_optim.py tests/test_reid_loss_modes.py tests/test_train_orchestration.py tests/test_checkpoint.py`
- result in this environment: `168 passed, 4 skipped in 51.13s`
- locked dataset baseline configs validated:
  - `configs/baseline_market1501_resnet50_triplet.yaml`
  - `configs/baseline_cuhk03_resnet50_triplet.yaml`
  - `configs/baseline_duke_resnet50_triplet.yaml`
  - `configs/baseline_msmt17_resnet50_triplet.yaml`

How To Test:
- focused schema checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_config_schema.py`
- validate locked configs manually:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python - <<'PY'`
  - `from reid.utils.config import load_config`
  - `from reid.utils.config_schema import validate_config`
  - `for p in ['configs/baseline_market1501_resnet50_triplet.yaml', 'configs/baseline_cuhk03_resnet50_triplet.yaml', 'configs/baseline_duke_resnet50_triplet.yaml', 'configs/baseline_msmt17_resnet50_triplet.yaml']: validate_config(load_config(p)); print('ok', p)`
  - `PY`

### Offline Framework Foundation Check

Implemented:
- verified the expected project foundation exists:
  - `configs/`
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `reid/data/`
  - `reid/models/`
  - `reid/losses/`
  - `reid/optim/`
  - `reid/engine/`
  - `reid/metrics/`
  - `reid/utils/`
  - `tests/`
  - `exp/`
- scanned orchestration and framework boundaries:
  - no dataset-specific logic in `scripts/train.py`
  - no model-specific logic in `reid/engine/evaluator.py`
  - no raw dataset paths outside YAML/dataset modules
- hardened the training loop to be more future-model/future-loss friendly:
  - `train_one_epoch(...)` now accepts a generic `aux_optimizer` instead of a center-specific optimizer argument
  - loss-specific center-gradient scaling moved from `reid/engine/train_loop.py` into `LossBundle.prepare_auxiliary_optimizer_step()`
  - training-loop logging now consumes generic `logs` keys returned by the criterion instead of hardcoding `triplet`, `id`, and `center`
  - TensorBoard logging now writes any criterion-provided loss keys dynamically
- updated call sites and tests:
  - `scripts/train.py`
  - `tests/test_train_loop_optim.py`

Why It Matters:
- future model additions should not require changes to dataset loaders, samplers, evaluator, checkpointing, logging, or artifact export
- future loss additions can provide their own log keys and optional auxiliary-optimizer preparation without editing `train_one_epoch`
- existing center-loss behavior is preserved, but the center-specific rule now lives with the loss bundle instead of the engine loop
- the training loop is now closer to a pure engine: move tensors, run model, call criterion, step optimizers, log generic metrics

Validation:
- full focused framework regression:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_smoke_reid_pipeline.py tests/test_dataset_protocol.py tests/test_msmt17_dataset.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py tests/test_rerank.py tests/test_model_forward.py tests/test_train_loop_optim.py tests/test_reid_loss_modes.py tests/test_train_orchestration.py tests/test_checkpoint.py`
- result in this environment: `143 passed, 4 skipped in 77.46s`
- static checks:
  - required foundation paths exist
  - `reid/engine/train_loop.py` no longer contains hardcoded `triplet`, `center_loss`, `w_center`, `loss/id`, `loss/triplet`, or `loss/center` handling

How To Test:
- focused train-loop/loss hardening checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_train_loop_optim.py tests/test_reid_loss_modes.py tests/test_train_orchestration.py tests/test_checkpoint.py`
- broader framework regression:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_smoke_reid_pipeline.py tests/test_dataset_protocol.py tests/test_msmt17_dataset.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py tests/test_rerank.py tests/test_model_forward.py tests/test_train_loop_optim.py tests/test_reid_loss_modes.py tests/test_train_orchestration.py tests/test_checkpoint.py`

### Shared Pipeline Smoke Test

Implemented:
- shared smoke-test entrypoint added:
  - `scripts/smoke_reid_pipeline.py`
- the script loads a normal training config and applies safe smoke defaults:
  - `data.num_workers=0`
  - `system.device=cpu` unless overridden
  - `model.backbone.pretrained=false` unless `--use-config-pretrained` is passed
- supported flags:
  - `--config`
  - `--root`
  - `--device`
  - `--skip-batch`
  - `--skip-model`
  - `--use-config-pretrained`
  - `--opts key=value ...`
- smoke checks include:
  - config load and validation
  - train loader construction
  - test loader construction
  - train dataset `num_classes`
  - one train batch contract: `(imgs, labels)`
  - one eval batch contract: `(imgs, pids, camids, names, marks)`
  - model construction with `num_classes`
  - one train forward pass
  - one loss computation
  - one eval forward pass
- script tests added in `tests/test_smoke_reid_pipeline.py`
  - override behavior
  - train batch contract validation
  - eval batch contract validation

How To Test:
- local unit checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_smoke_reid_pipeline.py`
- Constantine loader-only smoke:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python scripts/smoke_reid_pipeline.py --config configs/baseline_msmt17_resnet50_triplet.yaml --root /path/to/Dataset --skip-batch`
- Constantine full smoke:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python scripts/smoke_reid_pipeline.py --config configs/baseline_msmt17_resnet50_triplet.yaml --root /path/to/Dataset`
- for limited GPU memory, lower eval batch size:
  - `--opts data.test.batch.size=32`

### MSMT17 Raw Dataset Lock

Implemented:
- raw MSMT17 dataset support added in `reid/data/msmt17.py`
  - `MSMT17RawTrain`
  - `MSMT17RawTest`
  - `parse_msmt17_list`
- MSMT17 uses the same global ReID dataset protocol as Market1501, CUHK03, and Duke
  - train samples return `(image, label)`
  - eval samples return `(image, pid, camid, image_name, mark)`
  - `mark=0` means query and `mark=1` means gallery
- supported official raw layout:
  - `root/msmt17/MSMT17_V2/mask_train_v2`
  - `root/msmt17/MSMT17_V2/mask_test_v2`
  - `root/msmt17/MSMT17_V2/list_train.txt`
  - `root/msmt17/MSMT17_V2/list_val.txt`
  - `root/msmt17/MSMT17_V2/list_query.txt`
  - `root/msmt17/MSMT17_V2/list_gallery.txt`
- supported train splits:
  - `train`
  - `val`
  - `trainval`
- supported eval split:
  - `test`
- MSMT17 list parsing reads official rows:
  - `relative/image/path.jpg pid`
  - extracts camera id with the original convention: `rel_path.split("_")[2]`
  - converts pids to `int`
  - normalizes camera ids to zero-based when the list is one-based
  - keeps already zero-based camera ids unchanged
  - raises clear errors for malformed rows, invalid pids, or ambiguous camera-id base
- train datasets expose:
  - `samples`
  - `labels`
  - `num_classes`
  - `pids`
  - `cams`
  - `im_names`
  - `num_cameras`
- test datasets expose:
  - `samples`
  - `pids`
  - `cams`
  - `marks`
  - `im_names`
  - `num_query`
  - `num_gallery`
  - `num_cameras`
- raw test ordering is locked:
  - query samples first with `mark=0`
  - gallery samples second with `mark=1`
- MSMT17 constructors fail early with `FileNotFoundError(path)` for missing:
  - `MSMT17_V2`
  - `mask_train_v2`
  - `mask_test_v2`
  - `list_train.txt`
  - `list_val.txt` when `split=val` or `split=trainval`
  - `list_query.txt`
  - `list_gallery.txt`
- `reid/data/build.py` now dispatches Market1501, CUHK03, Duke, and MSMT17
  - MSMT17 currently supports `format: raw`
  - unsupported MSMT17 formats raise a clear `ValueError`
- MSMT17 dataset statistics are printed by the builders:
  - train: `[MSMT17] format=raw split=train`
  - train stats: `num images`, `num identities`, `num cameras`
  - test: `[MSMT17] format=raw split=test`
  - test stats: `query images`, `gallery images`, `query identities`, `gallery identities`, `cameras`
- MSMT17 baseline config added:
  - `configs/baseline_msmt17_resnet50_triplet.yaml`
  - default train dataset: `name=msmt17`, `split=train`, `format=raw`
  - default test dataset: `name=msmt17`, `split=test`, `format=raw`
  - default `data.num_workers=8`
  - default `data.test.batch.size=64`
  - default `eval.rerank.enabled=false`
- MSMT17 re-ranking warning added in `reid/engine/evaluator.py`
  - message: `[Warning] MSMT17 reranking may require large memory due to gallery size.`
  - warning only appears when MSMT17 evaluation has reranking explicitly enabled
- processed MSMT17 support is intentionally deferred
  - do not add `MSMT17ProcessedTrain` or `MSMT17ProcessedTest` until a standard `images/partitions.pkl` contract exists or the alternate `data/data.pkl` format is inspected

What It Locks:
- MSMT17 raw train/eval can plug into the same train loop, sampler, loss, model, and evaluator as Market1501, CUHK03, and Duke
- train labels are contiguous labels for the training split
- `num_classes` is exposed by train datasets
- `num_query` and `num_gallery` are exposed by the test dataset
- query/gallery ordering is deterministic and tested
- camera ids are normalized consistently to the pipeline's zero-based convention
- evaluator ranking logic remains dataset-agnostic; only a memory warning was added for MSMT17 reranking

Validation:
- `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_msmt17_dataset.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py tests/test_rerank.py`
- result in this environment: `105 passed in 6.50s`

How To Test:
- focused MSMT17/unit checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_msmt17_dataset.py tests/test_dataset_protocol.py`
- focused dataset regression checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_msmt17_dataset.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py`
- reranking warning checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_rerank.py`
- real-data raw smoke check on Constantine:
  - set `data.root` to the parent directory containing `msmt17/MSMT17_V2`
  - run the normal train/eval loader path with `configs/baseline_msmt17_resnet50_triplet.yaml`

### DukeMTMC-ReID Dataset Lock

Implemented:
- DukeMTMC-ReID dataset support added in `reid/data/duke.py`
  - `DukeProcessedTrain`
  - `DukeProcessedTest`
  - `DukeRawTrain`
  - `DukeRawTest`
  - `parse_duke_raw_name`
  - shared processed parser import: `parse_processed_reid_name`
- Duke uses the same global ReID dataset protocol as Market1501 and CUHK03
  - train samples return `(image, label)`
  - eval samples return `(image, pid, camid, image_name, mark)`
  - `mark=0` means query and `mark=1` means gallery
- supported processed layout:
  - `root/duke/images`
  - `root/duke/partitions.pkl`
- supported raw layout:
  - `root/duke/DukeMTMC-reID/bounding_box_train`
  - `root/duke/DukeMTMC-reID/query`
  - `root/duke/DukeMTMC-reID/bounding_box_test`
- supported train splits:
  - `train`
  - `trainval`
- supported eval splits:
  - `val`
  - `test`
- raw Duke filename parsing is strict enough for the official naming convention:
  - uses regex `([-\d]+)_c(\d)`
  - skips junk pid `-1`
  - validates camera range `[1, 8]`
  - converts raw camera ids to zero-based ids
- processed Duke uses the shared transformed ReID parser:
  - `parse_processed_reid_name("00000002_0001_00000000.jpg") -> (2, 1)`
- train datasets expose:
  - `labels`
  - `num_classes`
  - `pids`
  - `cams`
  - `im_names`
- test datasets expose:
  - `pids`
  - `cams`
  - `marks`
  - `im_names`
  - `num_query`
  - `num_gallery`
- raw test ordering is locked:
  - query samples first with `mark=0`
  - gallery samples second with `mark=1`
- Duke constructors fail early with `FileNotFoundError(path)` for missing:
  - processed `images`
  - processed `partitions.pkl`
  - raw `bounding_box_train`
  - raw `query`
  - raw `bounding_box_test`
- `reid/data/build.py` now dispatches Market1501, CUHK03, and Duke
  - Duke `format: processed` uses `images/partitions.pkl`
  - Duke `format: raw` uses the official folders
  - missing `format` defaults to `processed`
- Duke dataset statistics are printed by the builders:
  - `[DukeMTMC-ReID] format=processed split=trainval`
  - `num images`
  - `num identities`
  - `num query images`
  - `num gallery images`
  - `num cameras`
- Duke baseline config added:
  - `configs/baseline_duke_resnet50_triplet.yaml`
  - default train dataset: `name=duke`, `split=trainval`, `format=processed`
  - default test dataset: `name=duke`, `split=test`, `format=processed`

What It Locks:
- Duke processed train/eval can plug into the same train loop, sampler, loss, model, and evaluator as Market1501 and CUHK03
- Duke raw train/eval can be selected through config when the official folders exist
- no evaluator logic changes are required
- `num_classes` is exposed by train datasets
- query/gallery ordering is deterministic and tested

Validation:
- `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py`
- result in this environment: `83 passed in 3.69s`
- evaluator files were not modified

How To Test:
- focused Duke/unit checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_duke_dataset.py tests/test_dataset_protocol.py`
- focused dataset regression checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_duke_dataset.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py`
- real-data processed smoke check on Constantine:
  - set `data.root` to the parent directory containing `duke/images` and `duke/partitions.pkl`
  - run the normal train/eval loader path with `configs/baseline_duke_resnet50_triplet.yaml`
- real-data raw smoke check on Constantine:
  - override Duke train/test dataset `format: raw`
  - ensure `root/duke/DukeMTMC-reID/{bounding_box_train,query,bounding_box_test}` exists

### CUHK03 Processed Dataset Lock

Implemented:
- processed CUHK03 dataset support added in `reid/data/cuhk03.py`
  - `CUHK03ProcessedTrain`
  - `CUHK03ProcessedTest`
  - `parse_processed_reid_name`
- CUHK03 uses the same global ReID dataset protocol as Market1501
  - train samples return `(image, label)`
  - eval samples return `(image, pid, camid, image_name, mark)`
  - `mark=0` means query and `mark=1` means gallery
- supported processed layouts:
  - `root/cuhk03/detected/images`
  - `root/cuhk03/detected/partitions.pkl`
  - `root/cuhk03/labeled/images`
  - `root/cuhk03/labeled/partitions.pkl`
- supported train splits:
  - `train`
  - `trainval`
- supported eval splits:
  - `val`
  - `test`
- supported CUHK03 config fields:
  - `format: processed`
  - `image_type: detected|labeled`
  - `protocol: new|classic`
  - `split_id: 0`
- processed filename parsing is strict and reusable:
  - accepts `8digits_4digits_8digits.jpg`
  - accepts `8digits_4digits_8digits.png`
  - keeps `camid = int(name[9:13])` for compatibility with the transformed tri-loss naming convention
- train partitions require:
  - `{split}_im_names`
  - `{split}_ids2labels`
- eval partitions require:
  - `{split}_im_names`
  - `{split}_marks`
- CUHK03 constructors fail early with `FileNotFoundError(path)` for missing:
  - `images`
  - `partitions.pkl`
- `reid/data/build.py` now dispatches both Market1501 and CUHK03
  - Market1501 processed/raw behavior is unchanged
  - CUHK03 currently supports processed format only
- CUHK03 dataset statistics are printed by the builders:
  - `[CUHK03] format=processed image_type=detected split=trainval`
  - `num images`
  - `num identities`
  - `num query images`
  - `num gallery images`
- CUHK03 baseline config added:
  - `configs/baseline_cuhk03_resnet50_triplet.yaml`
- real-data smoke script added for the Constantine environment:
  - `scripts/smoke_cuhk03_dataset.py`
- raw CUHK03 `.mat` preprocessing is intentionally deferred
  - `# TODO: Add raw CUHK03 .mat preprocessing support later if needed.`

What It Locks:
- CUHK03 processed train/eval can plug into the same train loop, sampler, loss, model, and evaluator as Market1501
- no evaluator logic changes are required
- `image_type=detected` and `image_type=labeled` are both represented in tests using synthetic fixtures
- `num_classes` is exposed by the train dataset as `len(ids2labels)`

Validation:
- `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py`
- result in this environment: `61 passed in 3.80s`
- `git diff -- reid/engine/evaluator.py`
- result in this environment: no diff

How To Test:
- focused CUHK03/unit checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_cuhk03_dataset.py tests/test_dataset_protocol.py`
- focused dataset regression checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_cuhk03_dataset.py tests/test_market1501_dataset.py tests/test_sampler.py`
- Constantine real-data smoke check:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python scripts/smoke_cuhk03_dataset.py --config configs/baseline_cuhk03_resnet50_triplet.yaml --root /path/to/Dataset`
- if the real dataset is large and only construction should be checked first:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python scripts/smoke_cuhk03_dataset.py --config configs/baseline_cuhk03_resnet50_triplet.yaml --root /path/to/Dataset --skip-batch`

### Dataset Protocol Lock

Implemented:
- explicit dataset protocol module added in `reid/data/protocol.py`
  - `ReIDTrainSample(image, label)`
  - `ReIDEvalSample(image, pid, camid, image_name, mark)`
  - train dataset validation for `labels` and `num_classes`
  - eval dataset validation for `pids`, `cams`, `marks`, and `im_names`
- train/eval mark constants are now centralized:
  - `MARK_QUERY = 0`
  - `MARK_GALLERY = 1`
  - `MARK_MULTI_QUERY = 2`
- Market1501 train/test datasets now return named protocol samples instead of raw anonymous tuples
- Market1501 raw-directory parser added for standard `bounding_box_train`, `query`, and `bounding_box_test` folders
  - uses regex `([-\d]+)_c(\d)`
  - skips junk pid `-1`
  - validates pid range `[0, 1501]` and camera range `[1, 6]`
  - converts raw camera ids to zero-based ids
  - supports train relabeling through `relabel=True`
- explicit single-name parsers are now available:
  - `parse_market1501_name("0002_c1s1_000451_01.jpg") -> (2, 0)`
  - `parse_processed_name("00000002_0001_00000000.jpg") -> (2, 1)`
- `Market1501RawTrain` added for the official raw train folder
  - resolves `root/market1501/Market-1501-v15.09.15/bounding_box_train`
  - exposes `samples = [(img_path, pid, camid, label), ...]`
  - exposes `labels`, `pids`, `cams`, `im_names`, and `num_classes`
  - returns train protocol samples equivalent to `(image, label)`
- `Market1501RawTest` added for the official raw query/gallery folders
  - resolves `query` and `bounding_box_test`
  - builds a single sample list with all query samples first, then all gallery samples
  - assigns `mark=0` for query and `mark=1` for gallery
  - returns eval protocol samples equivalent to `(image, pid, camid, image_name, mark)`
- processed Market1501 classes are preserved and aliased as:
  - `Market1501ProcessedTrain = Market1501FromPartitions`
  - `Market1501ProcessedTest = Market1501TestFromPartitions`
- `reid/data/build.py` now chooses Market1501 datasets through `dataset.format`
  - `processed` uses the existing `images/partitions.pkl` classes
  - `raw` uses the official-folder raw classes
  - missing `format` defaults to `processed`
- train/test loader builders now print Market1501 dataset statistics:
  - selected format
  - train identities/images
  - query/gallery images
  - camera count
- Market1501 constructors now fail early with `FileNotFoundError` for missing required paths
  - raw: `bounding_box_train`, `query`, `bounding_box_test`
  - processed: `images`, `partitions.pkl`
- collate functions still accept the old tuple shapes for compatibility, then normalize into protocol samples internally
- shared dataloader builders now validate datasets immediately after construction
- tests added in `tests/test_dataset_protocol.py` for:
  - train sample collation
  - eval sample collation
  - required train metadata
  - required eval query/gallery marks
  - unsupported eval marks
- tests added in `tests/test_market1501_dataset.py` for:
  - raw filename parsing and junk pid handling
  - processed filename parsing
  - processed train/eval dataset item contracts
  - raw train/eval dataset item contracts
  - raw query-before-gallery ordering
  - required processed/raw path safety checks
- evaluator logic was not changed; the existing evaluator still consumes `(imgs, pids, cams, names, marks)`

What It Locks:
- every train dataset must expose:
  - `labels`: one class label per sample
  - `num_classes`: positive class count
  - contiguous non-negative labels covering `[0, num_classes)`
- every eval dataset must expose:
  - `pids`
  - `cams`
  - `marks`
  - `im_names`
  - at least one query sample and one gallery sample
- loader outputs remain unchanged:
  - train batches: `(imgs, labels)`
  - eval batches: `(imgs, pids, camids, names, marks)`

Validation:
- `python3 -m py_compile reid/data/protocol.py reid/data/collate.py reid/data/market1501.py reid/data/market1501_test.py reid/data/build.py`
- result in this environment: passed
- `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_market1501_dataset.py tests/test_sampler.py`
- result in this environment: `41 passed in 29.42s`
- `git diff -- reid/engine/evaluator.py`
- result in this environment: no diff

How To Test:
- focused dataset protocol checks:
  - `PYTHONPATH=. /home/filsduvent/environments/windflow_env/bin/python -m pytest -q tests/test_dataset_protocol.py tests/test_market1501_dataset.py tests/test_sampler.py`
- syntax checks:
  - `python3 -m py_compile reid/data/protocol.py reid/data/collate.py reid/data/market1501.py reid/data/market1501_test.py reid/data/build.py`

### Collate Functions And Training Observability

Implemented:
- explicit train/test collate functions added in `reid/data/collate.py`
  - `train_collate_fn(batch)` for `(image, label)` batches
  - `test_collate_fn(batch)` for `(image, pid, camid, name, mark)` batches
- both shared dataloader builders now use explicit `collate_fn` wiring in `reid/data/build.py`
  - train loader now uses `train_collate_fn`
  - test loader now uses `test_collate_fn`
- train/test batch contracts are now explicit instead of relying on PyTorch default collation for metadata
- `scripts/train.py` and `scripts/evaluate.py` continue to use the shared builders, so there is still no duplicated test-loader construction
- `reid/engine/train_loop.py` now computes `acc_id` when classifier logits are present
  - this is conditional and only runs when `outputs["logits"]` exists
  - triplet-only mode still works without accuracy logging
- train loop now logs time/speed observability data at each log interval:
  - `time/batch`
  - `speed=... imgs/s`
  - `lr/base`
- TensorBoard now receives:
  - `acc/id` when logits are present
  - `time/batch`
  - `speed/img_per_sec`
  - `lr/base`

What It Adds:
- clearer and more durable dataset/dataloader contracts
- safer metadata handling for evaluation batches
- better training sanity checks when ID logits are enabled
- easier throughput debugging and machine-to-machine comparison
- no change to model architecture, loss behavior, or metric definitions

How It Works:
- train collation stacks images and converts labels to `torch.long`
- test collation stacks images, converts numeric metadata to `torch.long`, and preserves filenames as a Python `list`
- dataloader builders pass those collate functions directly into `DataLoader(...)`
- train-loop accuracy is computed from `logits.argmax(dim=1)` versus labels only when logits exist
- speed is computed from logged samples over interval wall-clock time, while `time/batch` is derived from the same interval

Validation:
- `python3 -m py_compile reid/data/collate.py reid/data/build.py reid/engine/train_loop.py tests/test_collate.py tests/test_train_loop_optim.py`
- result in this environment: passed
- `PYTHONPATH=. pytest -q tests/test_collate.py tests/test_train_loop_optim.py tests/test_model_forward.py`
- result in this environment: `14 passed, 6 skipped`
- collate and observability tests verify:
  - train collate returns `NCHW` images and `torch.long` labels
  - test collate returns images, pids, camids, names, and marks with the expected types
  - train loop logs LR, time per batch, and speed
  - `acc_id` appears only when classifier logits are available
  - triplet-only mode still runs

How To Test:
- focused collate and train-loop checks:
  - `PYTHONPATH=. pytest -q tests/test_collate.py tests/test_train_loop_optim.py tests/test_model_forward.py`
- quick syntax checks:
  - `python3 -m py_compile reid/data/collate.py reid/data/build.py reid/engine/train_loop.py`

### Train And Evaluation Orchestration

Implemented:
- experiment logger utility added in `reid/utils/logger.py`
- checkpoint utility added in `reid/utils/checkpoint.py`
- checkpoint loading now supports:
  - full training checkpoints with `ckpt["model"]`
  - raw model-only state dicts
  - `module.` prefix stripping for non-DataParallel loads
- `scripts/train.py` was refactored into an orchestration-style entrypoint that now:
  - loads config plus `--opts` overrides
  - creates experiment artifact directories
  - sets up logger, device, seed, TensorBoard, loaders, model, criterion, optimizer, center optimizer, and scheduler
  - resumes from `train.save.resume` when configured
  - saves canonical checkpoints and eval artifacts
- `scripts/evaluate.py` added as a standalone evaluation entrypoint that:
  - loads config plus optional `--weight`
  - builds only the test-side path
  - loads a checkpoint
  - runs evaluation without training
  - writes metrics JSON under the experiment directory
- shared `build_test_loader(cfg)` added to `reid/data/build.py` so train/evaluate reuse the same test loader logic
- evaluation JSON export schema is now consistent across training and standalone evaluation:
  - `dataset`
  - `split`
  - `epoch`
  - `checkpoint`
  - `mAP`
  - `mINP`
  - `Rank1`
  - `Rank5`
  - `Rank10`
  - rerank fields only when reranking is enabled
- resolved config artifacts are now saved as:
  - `config.resolved.yaml`
  - `config.yaml` compatibility alias
- canonical checkpoints are now saved as:
  - `checkpoints/ckpt_best.pth`
  - `checkpoints/ckpt_last.pth`
- train artifact layer now also writes:
  - `logs/stdout.txt`
  - `logs/stderr.txt`
  - `plots/loss_curve.png`
  - `plots/rank1_curve.png`
  - `plots/map_bar.png`
  - `plots/minp_bar.png`
  - `plots/cmc_curve.png`
- standalone evaluation now also tee-captures raw output to:
  - `logs/eval_stdout.txt`
  - `logs/eval_stderr.txt`

Validation:
- `python3 -m py_compile reid/utils/logger.py reid/utils/checkpoint.py scripts/train.py scripts/evaluate.py reid/data/build.py`
- result in this environment: passed
- `PYTHONPATH=. pytest -q tests/test_checkpoint.py tests/test_train_orchestration.py tests/test_train_loop_optim.py tests/test_model_forward.py tests/test_rerank.py`
- result in this environment: `32 passed, 11 skipped`
- orchestration tests now verify:
  - canonical resolved-config artifact naming
  - canonical checkpoint naming
  - resume restores epoch, best metric, optimizer, scheduler, and center optimizer state
  - model-only checkpoint load works
  - `module.` prefix stripping works

How To Test:
- focused orchestration checks:
  - `PYTHONPATH=. pytest -q tests/test_checkpoint.py tests/test_train_orchestration.py tests/test_train_loop_optim.py`
- direct standalone evaluator CLI check:
  - `python3 scripts/evaluate.py --help`
- example standalone evaluation run:
  - `python3 scripts/evaluate.py --config configs/experiment_market1501_resnet50_center_rerank.yaml --weight exp/market1501_r50_triplet_id_center_rerank/checkpoints/ckpt_best.pth`

### Optional Re-Ranking Evaluation

Implemented:
- optional k-reciprocal re-ranking utility added in `reid/metrics/rerank.py`
- evaluator now always reports original metrics:
  - `mAP`
  - `mINP`
  - `Rank1`
  - `Rank5`
  - `Rank10`
- evaluator now adds reranked metrics only when `eval.rerank.enabled: true`:
  - `rerank_mAP`
  - `rerank_mINP`
  - `rerank_Rank1`
  - `rerank_Rank5`
  - `rerank_Rank10`
- current evaluator flow remains:
  - features -> distance matrices -> optional reranking -> metrics
- metric JSON export now naturally includes rerank keys only when enabled because eval scores are serialized directly
- TensorBoard eval logging now:
  - always logs original eval metrics
  - logs rerank eval metrics only when present
- evaluator now prints a memory warning when reranking is enabled because large galleries can be expensive on CPU memory
- active Market1501 baseline config now exposes:
  - `eval.rerank.enabled`
  - `eval.rerank.k1`
  - `eval.rerank.k2`
  - `eval.rerank.lambda_value`

Validation:
- `python3 -m py_compile reid/metrics/rerank.py reid/engine/evaluator.py tests/test_rerank.py scripts/train.py`
- result in this environment: passed
- `PYTHONPATH=. pytest -q tests/test_rerank.py tests/test_model_forward.py`
- result in this environment: `11 passed, 5 skipped`
- rerank tests verify:
  - rerank utility returns query-gallery distance matrices with finite values
  - evaluator works with rerank disabled
  - evaluator works with rerank enabled
  - original metrics are preserved
  - rerank metrics are reported separately

How To Test:
- focused rerank and evaluator checks:
  - `PYTHONPATH=. pytest -q tests/test_rerank.py tests/test_model_forward.py`
- rerank experiment toggle:
  - `PYTHONPATH=. python3 scripts/train.py --config configs/experiment_market1501_resnet50_center.yaml`
  - duplicate the config or temporarily set `eval.rerank.enabled: true` in the experiment YAML for the rerank comparison run

### Warmup Scheduler Milestone Semantics

Implemented:
- `warmup_multistep` milestones are now interpreted as epochs, matching the strong-baseline recipe
- scheduler stepping remains per iteration in the train loop
- `build_scheduler(cfg, optimizer, steps_per_epoch=...)` now converts YAML epoch milestones into iteration milestones internally
- warmup remains controlled by `warmup_iters` and therefore still operates per iteration
- `scripts/train.py` now passes `len(train_loader)` into the scheduler builder so active training uses the corrected semantics
- scheduler builder now raises a clear error if `warmup_multistep` is requested without a positive `steps_per_epoch`

Validation:
- `python3 -m py_compile reid/optim/build.py scripts/train.py tests/test_optim_build.py tests/test_train_loop_optim.py`
- result in this environment: passed
- `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_train_loop_optim.py`
- result in this environment: `15 passed, 3 skipped`
- scheduler tests now verify:
  - epoch milestones are converted to iteration milestones
  - warmup still ramps per iteration
  - LR decay occurs at the expected epoch boundary after conversion
  - missing `steps_per_epoch` fails early for `warmup_multistep`

How To Test:
- focused scheduler and train-loop checks:
  - `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_train_loop_optim.py`
- quick syntax checks:
  - `python3 -m py_compile reid/optim/build.py scripts/train.py tests/test_optim_build.py tests/test_train_loop_optim.py`

### Optimizer Refinements

Implemented:
- SGD optimizer builder now supports YAML-controlled `nesterov`
- optimizer param groups now keep `param_name` for debugging and verification
- bias-group overrides remain YAML-controlled through:
  - `bias_lr_factor`
  - `weight_decay_bias`
- optimizer tests now verify:
  - bias params use `base_lr * bias_lr_factor`
  - bias params use `weight_decay_bias`
  - non-bias params use base values
  - SGD honors `nesterov: true`

Validation:
- `PYTHONPATH=. pytest -q tests/test_optim_build.py`
- result in this environment: `10 passed`

How To Test:
- focused optimizer checks:
  - `PYTHONPATH=. pytest -q tests/test_optim_build.py`
- quick syntax checks:
  - `python3 -m py_compile reid/optim/build.py tests/test_optim_build.py`

### PK Sampler And Training Modes

Implemented:
- `PKBatchSampler` is now finite and no longer uses infinite iteration
- PK sampler now:
  - builds per-identity index pools
  - pads identities with replacement up to `K` when needed
  - shuffles locally with seeded RNGs only
  - yields finite `List[int]` PK batches
- sampler reproducibility is local to the sampler through:
  - `random.Random(seed)`
  - `np.random.default_rng(seed)`
- sampler length is now derived from the actual finite batch plan, so `len(train_loader)` matches the yielded epoch batches
- `build_train_loader(cfg)` now supports:
  - `sampler: pk` for triplet / triplet+ID style training
  - `sampler: random` for ID-only / softmax-only style training
- train loop no longer uses fake `steps_per_epoch` limits
- one epoch is now defined by the dataloader / sampler length
- PK loader safety checks now fail early for:
  - triplet loss with `sampler: random`
  - `P <= 1`
  - `K <= 1`
  - `batch_size != P * K` in PK mode
- strong-baseline Market1501 config now explicitly includes:
  - `sampler: pk`
  - `P: 16`
  - `K: 4`
  - `batch_size: 64`

Validation:
- `PYTHONPATH=. pytest -q tests/test_sampler.py`
- result in this environment: `9 passed`
- `PYTHONPATH=. pytest -q tests/test_sampler.py tests/test_train_loop_optim.py`
- result in this environment: `10 passed, 3 skipped`
- sampler tests verify:
  - PK batches have size `P*K`
  - each batch contains exactly `P` identities
  - each selected identity appears exactly `K` times
  - sampler iteration is finite
  - `len(sampler)` is finite and positive
  - PK loader length is finite
  - random mode works for ID-only
  - triplet + random raises a clear error
  - PK mode rejects invalid `P`, `K`, or mismatched `batch_size`

How To Test:
- focused sampler checks:
  - `PYTHONPATH=. pytest -q tests/test_sampler.py`
- sampler plus train-loop regression:
  - `PYTHONPATH=. pytest -q tests/test_sampler.py tests/test_train_loop_optim.py`
- quick syntax checks:
  - `python3 -m py_compile reid/data/samplers.py reid/data/build.py reid/engine/train_loop.py scripts/train.py tests/test_sampler.py`

Notes:
- CUDA train-loop checks remain skip-gated and were not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026

### Center Loss Integration

Implemented:
- `loss.center` remains YAML-controlled with:
  - `enabled`
  - `weight`
  - `lr`
- Center Loss stays disabled by default in the baseline config but can be enabled without code changes
- `build_criterion(cfg, num_classes, feat_dim)` now:
  - creates `CenterLoss` only when enabled
  - validates `num_classes` and `feat_dim`
  - stores the module as `criterion.center_loss`
  - keeps `criterion.center` as a compatibility alias
- triplet loss and center loss now share the same `model.head.metric_feat` routing
- ID loss continues using `outputs["logits"]` only
- `build_center_optimizer(cfg, criterion)` now:
  - returns `None` when center loss is disabled
  - returns `None` when the criterion has no center loss module
  - otherwise returns SGD over the center loss parameters with `loss.center.lr`
- train loop now:
  - zeros the center optimizer when present
  - rescales center gradients by `1 / center_weight`
  - steps the center optimizer safely alongside the main optimizer
- Center Loss remains device-agnostic and moves with the criterion module

Validation:
- `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `22 passed, 3 skipped`
- verified modes:
  - triplet only
  - ID only
  - triplet + ID
  - triplet + ID + center
- verified behavior:
  - center optimizer is created only when needed
  - `loss/center` is logged through the criterion and train loop
  - center gradients exist and are stepped when enabled

How To Test:
- focused Center Loss checks:
  - `PYTHONPATH=. pytest -q tests/test_optim_build.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- debug training path:
  - `PYTHONPATH=. python3 scripts/train.py --config configs/debug_train.yaml`

Notes:
- CUDA execution remains skip-gated in tests and was not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- no `.cuda()` calls were introduced into `CenterLoss`

### BNNeck Trick

Implemented:
- `model.head.bnneck`, `metric_feat`, and `eval_feat` are now explicit YAML controls
- strong-baseline defaults are now:
  - `bnneck: true`
  - `metric_feat: raw`
  - `eval_feat: bn`
- `ReidBaseline` now exposes the BNNeck contract as:
  - `feat_raw`
  - `feat_bn`
  - `emb`
  - `logits`
- `emb` is now selected from `eval_feat` and then normalized if enabled
- BNNeck is now implemented as a dedicated bottleneck layer with:
  - frozen BN bias
  - BN weight/bias initialization
- classifier remains bias-free and is explicitly initialized with small normal weights
- loss routing now uses `model.head.metric_feat` for:
  - triplet loss
  - center loss
- ID loss still uses `logits` only
- evaluator now reads `outputs["emb"]` from dict outputs and does not use `logits`
- builder validation now raises explicit `ValueError`s for invalid:
  - `metric_feat`
  - `eval_feat`
- active baseline/debug/ablation configs were aligned to the BNNeck contract
- stale `loss.center.feat` routing was removed from the active YAML configs

Validation:
- `PYTHONPATH=. pytest -q tests/test_model_forward.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `22 passed, 6 skipped`
- forward tests verify:
  - BNNeck enabled works
  - BNNeck disabled works
  - `eval_feat: raw` works
  - `eval_feat: bn` works
  - output dict contains `feat_raw`, `feat_bn`, `emb`, `logits`
- loss-routing tests verify:
  - triplet uses configured metric feature
  - center uses configured metric feature
  - ID loss uses `logits` only
- existing train-loop tests verify training still runs on CPU with the dict-output contract

How To Test:
- focused BNNeck checks:
  - `PYTHONPATH=. pytest -q tests/test_model_forward.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- quick syntax checks:
  - `python3 -m py_compile reid/models/baseline.py reid/models/build.py reid/losses/build.py reid/engine/evaluator.py tests/test_model_forward.py tests/test_reid_loss_modes.py`
- debug train path:
  - `PYTHONPATH=. python3 scripts/train.py --config configs/debug_train.yaml`

Notes:
- CUDA checks remain skip-gated in tests and were not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- evaluator smoke tests are present but skip when `sklearn` is not installed in the environment

### Last Stride Trick

Implemented:
- `last_conv_stride` remains YAML-controlled through `model.backbone.last_conv_stride`
- `reid/models/build.py` now reads the value into a local variable and prints:
  - `[Model] backbone=resnet50 last_conv_stride=...`
- `ReidBaseline` now explicitly supports:
  - `last_conv_stride: 1`
  - `last_conv_stride: 2`
- `last_conv_stride: 1` modifies only `layer4[0].conv2` and `layer4[0].downsample[0]`
- invalid stride values now raise:
  - `ValueError("last_conv_stride must be 1 or 2")`
- optional ablation config added:
  - `configs/ablation_stride2.yaml`

Validation:
- `PYTHONPATH=. pytest -q tests/test_model_forward.py tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `13 passed, 5 skipped`
- forward tests verify:
  - `last_conv_stride=1` runs on CPU
  - `last_conv_stride=2` runs on CPU
  - pooled embedding shape stays unchanged
  - final backbone feature map is spatially larger for stride `1` than stride `2`
- config/build tests verify:
  - YAML alone controls the stride setting
  - builder reflects stride `1` vs `2` without hardcoding

Notes:
- CUDA forward verification is covered by a skip-gated test and was not executed in this environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- evaluation smoke coverage is present but was skipped in this environment because `sklearn` is not installed

### Label Smoothing Verification

Verified:
- `loss.id.label_smoothing` is config-driven in YAML
- `label_smoothing: 0.0` selects standard CE behavior
- `label_smoothing: 0.1` selects `CrossEntropyLabelSmooth`
- ID loss consumes classifier `logits` and dataset `labels`
- NaN guard added for ID loss in `reid/losses/build.py`
- CPU training-path checks cover:
  - triplet only
  - ID only
  - triplet + ID
  - triplet + ID + center

Validation:
- `PYTHONPATH=. pytest -q tests/test_reid_loss_modes.py tests/test_train_loop_optim.py`
- result in this environment: `11 passed, 3 skipped`
- label smoothing tests verify:
  - `label_smoothing: 0.0` matches `F.cross_entropy(...)`
  - `label_smoothing: 0.1` changes the scalar ID-loss value on fixed logits

Notes:
- CUDA runtime verification could not be executed in the current environment because `torch.cuda.is_available()` was `False` on April 28, 2026
- CUDA-path tests were added and will run automatically on a GPU-visible machine

### Phase 3.2: Optimizer + Warmup Scheduler

Implemented:
- `reid/optim/` package
- `WarmupMultiStepLR`
- `build_optimizer(cfg, model)`
- `build_center_optimizer(cfg, center_criterion)`
- `build_scheduler(cfg, optimizer)`
- training integration in `scripts/train.py`
- per-iteration scheduler stepping in `reid/engine/train_loop.py`
- LR logging in console and TensorBoard
- center-loss optimizer gradient rescaling and separate optimizer step

Validation:
- optimizer/scheduler/train-loop tests passed on CPU
- loss-mode coverage passed for:
  - triplet only
  - triplet + ID
  - triplet + ID + center

Notes:
- scheduler still steps per iteration
- milestone semantics are now corrected by converting YAML epoch milestones to iteration milestones during scheduler construction

Relevant commit:
- `17d7485` `Add ReID optimizer and warmup scheduler support`

### Random Erasing Augmentation

Implemented:
- custom `RandomErasing` in `reid/data/transforms.py`
- train transform refactored to accept `aug_cfg`
- train transform order now matches:
  - `Resize`
  - `RandomHorizontalFlip`
  - `Pad`
  - `RandomCrop`
  - `ToTensor`
  - `Normalize`
  - `RandomErasing`
- test transform remains deterministic
- training data builder updated to use `build_train_tf(image_size, aug_cfg)`
- optional visual inspection script:
  - `scripts/debug_transforms.py`
- baseline config updated with:
  - `padding`
  - `random_crop`
  - `random_erasing`

Validation:
- transform composition tests passed
- `RandomErasing` appears only when `random_erasing.enabled: true`
- disabling it in config removes it without code changes
- test transform does not include `RandomErasing`
- no sampler files changed

Notes:
- dataset-backed Market1501 smoke run is currently blocked in this environment by pickle / NumPy compatibility for `partitions.pkl`
- CUDA was not verified in the current environment because `torch.cuda.is_available()` was `False`

## Current Uncommitted Work

Orchestration / artifact-layer changes are present and should be committed as a separate step after review.

## Suggested Next Step

Run at least one real end-to-end experiment with the refactored orchestration and inspect the new artifacts.

Expected focus:
1. verify `config.resolved.yaml`, `logs/`, `checkpoints/`, `metrics/`, and `plots/` are all created as expected
2. verify `train.save.resume` through a real interrupted-and-resumed run
3. verify `scripts/evaluate.py` on a real checkpoint in both rerank-off and rerank-on settings

## Useful Commands

Train debug run:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/debug_train.yaml
```

Transform debug images:

```bash
PYTHONPATH=. python3 scripts/debug_transforms.py --config configs/baseline_market1501_resnet50_triplet.yaml
```

Run current test coverage:

```bash
PYTHONPATH=. pytest -q tests/test_data_transforms.py tests/test_train_loop_optim.py tests/test_reid_loss_modes.py tests/test_optim_build.py
```

Run next ResNet50 center-loss experiment:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/experiment_market1501_resnet50_center.yaml
```

Run standalone evaluation:

```bash
python3 scripts/evaluate.py --config configs/experiment_market1501_resnet50_center_rerank.yaml --weight exp/market1501_r50_triplet_id_center_rerank/checkpoints/ckpt_best.pth
```
