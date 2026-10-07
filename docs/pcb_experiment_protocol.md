# Phase 20 — Authoritative PCB experiment freeze

Prepared on Constantine, 2026-10-06. **Reviewed and accepted by the user.** No Phase 21 run is authorized or executed by this document. Execute only the separately authorized source, then review it before another run.

## Code and environment identity

Repository: `person-reid-autonomous-robot`; origin: `git@github-personal:Filsduvent/person-reid-autonomous-robot.git`. Audit started on clean `main`; HEAD, origin/main and the live remote main agreed on **`66ea88d06ba0f79c298b19abe1652ae892d69fdd`**, the published accepted Phase 19 commit. This is the training commit for all four sources. The later commit containing this documentation is a publication record, not a replacement training commit. A production or preset correction requires explicitly revising the freeze.

After this documentation is accepted and committed, launch instructions below select the frozen commit with a clean, detached checkout. They do not force checkout or discard changes. Copy the chosen instructions before switching: these new documents do not exist at the older training commit. They remain accessible using `git show main:docs/pcb_experiment_protocol.md`. Do not switch, pull, edit code, install packages or launch a second run in this checkout during training. Return with `git switch main` only when no run is active. Ignored experiment artifacts persist across checkout changes.

| Item | Frozen value |
| --- | --- |
| Host/user | constantine / addirakoze |
| Repository physical directory | `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot` |
| Activation | `source /home/addirakoze/environments/Reid/bin/activate` |
| Python | `/home/addirakoze/environments/Reid/bin/python`, 3.11.2 |
| PyTorch / torchvision | 2.7.1+cu126 / 0.22.1+cu126 |
| CUDA runtime / cuDNN | 12.6 / 90501 |
| Driver | 570.211.01; nvidia-smi advertises CUDA compatibility 12.8 |
| GPU | NVIDIA TITAN Xp, physical index 0, UUID `GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c` |
| CUDA-visible VRAM | 12,774,801,408 bytes = 11.8974609375 GiB |
| OS platform | Linux-6.1.0-50-amd64-x86_64-with-glibc2.36 |
| Session manager | `/usr/bin/tmux`, 3.3a |

The package inventory is `pcb_environment_freeze.txt`; it records installed distribution names/versions, not an installer lockfile. No dependencies were changed. At 21:17 on October 6, GPU0 had no compute processes, 0% utilization, 31°C, 5 MiB used and 12,179 MiB free according to nvidia-smi. Availability is a snapshot, not a reservation: inspect again before launching. The GTX1080 is not the approved execution device. UUID visibility pins the tested physical TITAN Xp; the unchanged preset's gpu_id=0 selects logical device 0 within that visibility. A preflight prevents `device:auto` silently falling back to CPU.

## Architecture and initialization

Huang Houjing `beyond-part-models`, reference commit `1686e889eb01c28a54b633051418012e15d9c9f3`, independent-part reduction variant, adapted to the common framework. ResNet50, last stride 1, dilation 1; six fixed horizontal stripes; six independent 2048→256 Conv2d 1×1 / BatchNorm / ReLU reductions and six independent 256→source-C identity classifiers. Ordered concatenation produces a 1536-dimensional post-ReLU retrieval embedding. No alternative variant, shared reduction, dropout, additional BNNeck or per-part normalization is selected.

Historical ImageNet backbone artifact:

- Path: `/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth`.
- Size: 102,502,400 bytes.
- SHA256: `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`.
- Original official source: `https://download.pytorch.org/models/resnet50-19c8e357.pth`.
- Matches the pinned artifact from Phases 5/19; Phase 20 rehashed the local file. Phase 19 already verified the 265 backbone tensors. No substitution with V2 weights, random initialization or trained PCB weights.

The model loader checks the complete hash before deserialization. With this cache present, the training entry point needs no external network resource. It has no Ultralytics import dependency; `YOLO_OFFLINE=true` is included consistently as a harmless guard for optional imports, not as a network firewall. W&B is disabled. Do not delete/move the historical cache; otherwise the model loader would attempt a download. No PYTHONPATH or thread-count override is needed by the training CLI.

## Data and protocol

Logical root `/home/addirakoze/nobackup/Projects/Dataset` resolves to `/mnt/hd-data/addirakoze/Projects/Dataset`. Twelve relevant Phase 16 partition/list/provenance hashes were refreshed successfully; all are recorded in the YAML manifest. Phase 20 reloads metadata and verifies counts/loader lengths; it does not repeat the full image-level Phase 16 audit or claim a content hash of every image.

| Source | Physical dataset path below `/mnt/hd-data/addirakoze/Projects/Dataset` | Train IDs | Train images | Query | Gallery | Train steps/epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Market-1501 | `market1501/images` | 751 | 12936 | 3368 | 15913 | 202 |
| DukeMTMC-ReID | `duke/images` | 702 | 16522 | 2228 | 17661 | 258 |
| CUHK03 detected/new | `cuhk03/detected/images` | 767 | 7365 | 1400 | 5332 | 115 |
| MSMT17_V2 | `msmt17/MSMT17_V2` | 1041 | 30248 | 11659 | 82161 | 472 |

Market and Duke use processed partitions, trainval/test. Market excludes 3819 raw PID−1 junk gallery images and retains 2798 PID0 background gallery images. Its evaluation loader also extracts 12688 mark-2 multi-query images; only marks 0/1 enter the standard query/gallery metrics. There is no multi-query aggregation. Dataset length is 31969, evaluation batches 1000. Duke evaluation length 19889, batches 622; gallery includes distractors. Standard same-person/same-camera exclusion remains in the common ranking evaluator.

CUHK03 is **detected images, new protocol**, verified in Phase 16 against MAT indices, re-ranking split pickle, JSON and detected source data. The parser selects `processed_partition`, split_id=null; this API spelling does not mean unknown scientific protocol. The preset's old provenance comment is superseded by the accepted Phase 16 evidence; its bytes are deliberately unchanged. Test length 6732, batches 211; camera labels represent local pair sides 0/1.

MSMT17 uses raw official list parsing, `MSMT17_V2/list_train.txt` only. `list_val.txt` contains 2373 additional images of the same 1041 training IDs and is excluded. Existing `mask_train_v2`→`train` and `mask_test_v2`→`test` links remain unchanged. Cameras 1–15 map to 0–14. Test length 93820, batches 2932. **MSMT17 is a project extension of the selected Huang configuration, not reproduction of a Huang-reported MSMT17 experiment.**

## Common recipe

All four YAMLs were loaded by `load_config`, structural and semantic validators. After excluding only experiment name/output and train/test dataset selectors, their remaining fields are identical. Full resolved configurations (including absolute output paths), hashes and observed loader details are in `pcb_experiment_manifest.yaml`.

- Input 384×128. Train: Resize → RandomHorizontalFlip(p=0.5) → ToTensor → Normalize(mean=[0.486,0.459,0.408], std=[0.229,0.224,0.225]). Evaluation: Resize → ToTensor → same Normalize. No padding, crop or random erasing.
- Ordinary random shuffled images, batch64, drop_last=true; no PK sampling. Test batch32, sequential, drop_last=false. Four workers, pinned memory. Per-epoch discarded tails: Market8, Duke10, CUHK5, MSMT40 images; shuffling changes which images form the tail.
- Six cross-entropies summed, ID weight1, label smoothing0; triplet and center disabled.
- SGD baseLR0.1, momentum0.9, Nesterov=false; decay0.0005 for all trainable weights, biases and BN parameters; biasLRfactor1.0. `backbone.` multiplier0.1: backboneLR0.01, new layersLR0.1.
- 120 epochs, warmup_multistep, milestone[40], gamma0.1, warmup_iters0. Epochs1–40: backbone0.01/new0.1; epochs41–120: backbone0.001/new0.01. No second decay. Actual converted milestones: Market8080, Duke10320, CUHK4600, MSMT18880 completed updates; first reduced-LR updates are one later. Total updates: 24240/30960/13800/56640. Scheduler boundary inspection used the actual builders without optimizer updates.
- FP32, AMP=false; seed42 for every source; deterministic=false, cuDNN benchmark=true. Full RNG/sampler-state persistence is absent: no claim of bitwise deterministic full-training resume.
- Reference duration60 versus common project duration120 is an explicit adaptation. The common 120-epoch/10-epoch evaluation protocol is preserved across architectures.

## Selection, evaluation and future result provenance

Evaluate every10 epochs (10,20,…,120). Save latest state as `checkpoints/ckpt_last.pth`; select `checkpoints/ckpt_best.pth` by highest periodically observed mAP, strictly `>` so ties retain the earlier best. Last reaches epoch120 on successful completion. Best is not automatically epoch120.

**Methodological limitation:** periodic selection evaluates the configured test query/gallery protocol, not an independent validation split. This common protocol is deliberately preserved for consistency with completed ResNet50 experiments and must be disclosed in the dissertation.

Evaluation uses one global L2 feature normalization, Euclidean distance, no re-ranking, no test-time flip, topk=[1,5,10]. Report Rank-1/5/10, mAP and mINP. Phase22 independently checks each selected source best on its own dataset. Phase23 uses exactly one fixed source `ckpt_best.pth` for all four targets in each of four source rows (16 cells including the four diagonals); no target-dependent checkpoint choice or adaptation. Phase15 source preprocessing ownership applies; only target dataset protocol changes.

After training, record best epoch, periodic selection mAP, selected-best SHA256 and within-domain metrics; separately record epoch120 last-state SHA256. Manifest result fields are null because no such results exist in this phase. The framework does not automatically create checkpoint checksum files.

Resume limitation: `maybe_resume_training` initializes its best score from the resumed checkpoint's scores, which can be absent on a non-evaluation epoch or lower than an earlier historical best. Automatic/blind resume could overwrite historical best. After interruption, preserve best, last, metric history and logs, then obtain a bounded recovery decision that explicitly protects historical best. Do not delete the output directory, restart into it, or change recipe to work around failure. This freeze is ready for fresh runs, not unconditional resumed runs.

## Actual artifact contract and storage

For each output root, the current framework creates:

```text
config.resolved.yaml
config.yaml                         # compatibility alias/copy
train.log
logs/stdout.txt
logs/stderr.txt
artifacts/command.txt
artifacts/environment.txt
artifacts/git_commit.txt
command.txt                         # root compatibility copies
environment.txt
git_commit.txt
tensorboard/events.out.tfevents.*
tb                                  # compatibility symlink when supported
checkpoints/ckpt_last.pth
checkpoints/ckpt_best.pth
metrics/val_epoch_010.json           # then 020,...,120
metrics/latest_val.json
metrics/final_epoch_test.json       # final training state
metrics/final_test.json             # selected-best state
plots/loss_curve.png                # if plotting succeeds
plots/rank1_curve.png
plots/map_bar.png
plots/minp_bar.png
plots/cmc_curve.png
```

Checkpoint payload includes epoch, model, optimizer, scheduler, center_optimizer (disabled), scores, cfg and reconstruction metadata. There is no archive of 120 epoch checkpoints. Latest is overwritten each epoch and again with scores after evaluation; best is overwritten only on strict improvement. Writes are not atomic: checkpoint write failure can leave a truncated file. `command.txt` records script argv, not the activating shell or environment exports; this freeze records those missing details. Git provenance records the actual training commit, but not dirty status/preset hashes/weight hashes; the launch guards and this manifest complement it.

Storage estimates below count actual model state tensors plus one FP32 SGD momentum tensor per trainable parameter, not a measured completed-run file. Serialization/config/metadata overhead and logs add space. Two persistent checkpoints per source are expected; keep **at least10 GiB free for the four runs** as a conservative operational allowance, plus separate space for any manually preserved recovery copies. No automatic cleanup is authorized. Current `/dev/sdb1` has 3,405,158,223,872 available bytes (about3.10 TiB); existing `exp` uses24 GiB. Storage is sufficient at audit time.

Phase19 observed cold Market batch64 allocated/reserved peaks9.660/10.021 GiB, leaving1.876 GiB (15.77%) against CUDA-visible capacity; three finite updates and separate batch32 evaluation succeeded. This is measured Market feasibility, not a new measurement of all four full runs. Full MSMT ranking also uses host RAM (its query×gallery float32 distance array alone is about3.57 GiB, plus features/copies/sorting); available host RAM was about26 GiB. Keep other large workloads off this host/GPU during the run.

| Source | Parameters | Estimated bytes/checkpoint | MiB/checkpoint |
| --- | ---: | ---: | ---: |
| market1501 | 27816410 | 222756520 | 212.44 |
| duke | 27740852 | 222152056 | 211.86 |
| cuhk03 | 27841082 | 222953896 | 212.63 |
| msmt17 | 28263590 | 226333960 | 215.85 |

## Preset and output identities

All four outputs were absent (including `exp/pcb` itself); resolved paths are distinct, do not overlap baseline outputs, and contain no smoke evidence. Phase18 evidence remains under `/tmp/pytest-of-addirakoze`; Phase19 remains under `/tmp/pcb-phase19.D2rVUXeO`. Nothing was deleted.

- `configs/pcb/market1501.yaml` — SHA256 `a0d689383e3e054a406d14172129072f1660da3852c8cab4177cbefccb746f77`; output `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot/exp/pcb/market1501`.
- `configs/pcb/duke.yaml` — SHA256 `a30afbcec6b55f874923bab36dc6fba99114dd24947cabd3b8fb1170c6448f85`; output `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot/exp/pcb/duke`.
- `configs/pcb/cuhk03.yaml` — SHA256 `0b13581f5de394a2b3f2587ee6ec61b4a9ba05c219a887ae3bd9b7e20d10c1c8`; output `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot/exp/pcb/cuhk03`.
- `configs/pcb/msmt17.yaml` — SHA256 `3d618f16698e13192b485a6a1b9424db67e25e7e13e99b1f5d49fafc5598a8f8`; output `/mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot/exp/pcb/msmt17`.

## Manual launch workflow — after acceptance and source-specific authorization

Log in and create one session for the chosen source (only one GPU0 training at a time):

```bash
ssh addirakoze@constantine
tmux new-session -s pcb-market1501
```

For the other sources use session names `pcb-duke`, `pcb-cuhk03`, or `pcb-msmt17`. Before pasting a launch block, inspect current processes/free memory in a second terminal:

```bash
nvidia-smi -i GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
pgrep -af 'scripts/train.py'
df -h /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot/exp
```

Require the TITAN Xp to be idle, free memory comparable to the audited idle state, no other training using this checkout, sufficient disk space, and the frozen environment/data still available. A listed unrelated process on GPU0 means wait and coordinate; do not kill it. These commands do not reserve the GPU against another user's later launch.

Each block below is complete **inside the tmux shell**: it activates the environment, selects the exact published commit, checks clean state/config/cache/GPU/environment, refuses an existing output (including broken symlinks), and starts only the chosen run. Subshell `set -e` stops on a failed guard while preserving the tmux shell. If a guard fails, stop and report; do not bypass it or delete output. No YAML override is applied. Full package and data hashes remain in the manifest for investigating changes; the command checks its source's relevant metadata again. Untracked files also stop the checkout guard; commit accepted Phase20 documents first.

### market1501

```bash
(
set -e
source /home/addirakoze/environments/Reid/bin/activate
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot
test "$(hostname -s)" = constantine
test -z "$(git status --porcelain)"
git switch --detach 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test "$(git rev-parse HEAD)" = 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test ! -e exp/pcb/market1501
test ! -L exp/pcb/market1501
export CUDA_VISIBLE_DEVICES=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
export YOLO_OFFLINE=true
/home/addirakoze/environments/Reid/bin/python - <<'PY_PREFLIGHT'
import sys, hashlib
from pathlib import Path
import torch, torchvision
assert sys.version_info[:3] == (3,11,2)
assert torch.__version__ == "2.7.1+cu126" and torchvision.__version__ == "0.22.1+cu126"
assert torch.version.cuda == "12.6" and torch.backends.cudnn.version() == 90501
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
assert p.name == "NVIDIA TITAN Xp" and p.total_memory == 12774801408
assert str(p.uuid).removeprefix("GPU-") == "bb0efb3f-f2ea-d40d-8769-e4e989bab10c"
def check(path, expected):
    with open(path, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, str(path)
check("configs/pcb/market1501.yaml", "a0d689383e3e054a406d14172129072f1660da3852c8cab4177cbefccb746f77")
w = Path(torch.hub.get_dir()) / "checkpoints/resnet50-19c8e357.pth"
assert str(w) == "/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"
check(w, "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097")
base = Path("/home/addirakoze/nobackup/Projects/Dataset")
assert str(base.resolve()) == "/mnt/hd-data/addirakoze/Projects/Dataset"
for relative, digest in {'market1501/partitions.pkl': '3853cfdc9be6814d0193b9aad8025b61133d7f7b9ef8449b5d0f2964811a53ab', 'market1501/train_test_split.pkl': '0501b8b8c11ccf5a5b8aea19f2cce509ae3f2c2602063c92afaf4446c5529949'}.items():
    check(base / relative, digest)
print("Frozen source preflight passed; starting fresh market1501 PCB training.")
PY_PREFLIGHT
/home/addirakoze/environments/Reid/bin/python -u scripts/train.py --config configs/pcb/market1501.yaml
)
```

### duke

```bash
(
set -e
source /home/addirakoze/environments/Reid/bin/activate
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot
test "$(hostname -s)" = constantine
test -z "$(git status --porcelain)"
git switch --detach 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test "$(git rev-parse HEAD)" = 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test ! -e exp/pcb/duke
test ! -L exp/pcb/duke
export CUDA_VISIBLE_DEVICES=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
export YOLO_OFFLINE=true
/home/addirakoze/environments/Reid/bin/python - <<'PY_PREFLIGHT'
import sys, hashlib
from pathlib import Path
import torch, torchvision
assert sys.version_info[:3] == (3,11,2)
assert torch.__version__ == "2.7.1+cu126" and torchvision.__version__ == "0.22.1+cu126"
assert torch.version.cuda == "12.6" and torch.backends.cudnn.version() == 90501
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
assert p.name == "NVIDIA TITAN Xp" and p.total_memory == 12774801408
assert str(p.uuid).removeprefix("GPU-") == "bb0efb3f-f2ea-d40d-8769-e4e989bab10c"
def check(path, expected):
    with open(path, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, str(path)
check("configs/pcb/duke.yaml", "a30afbcec6b55f874923bab36dc6fba99114dd24947cabd3b8fb1170c6448f85")
w = Path(torch.hub.get_dir()) / "checkpoints/resnet50-19c8e357.pth"
assert str(w) == "/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"
check(w, "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097")
base = Path("/home/addirakoze/nobackup/Projects/Dataset")
assert str(base.resolve()) == "/mnt/hd-data/addirakoze/Projects/Dataset"
for relative, digest in {'duke/partitions.pkl': '25929417e0a8780548924d4f58f4f644ee18098e957df41a571650eb05e6b638', 'duke/train_test_split.pkl': 'beb21338194506fb9b04ce0f400ac38d920c861cc5c75c581a8ace0e4429845a'}.items():
    check(base / relative, digest)
print("Frozen source preflight passed; starting fresh duke PCB training.")
PY_PREFLIGHT
/home/addirakoze/environments/Reid/bin/python -u scripts/train.py --config configs/pcb/duke.yaml
)
```

### cuhk03 — detected/new protocol

```bash
(
set -e
source /home/addirakoze/environments/Reid/bin/activate
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot
test "$(hostname -s)" = constantine
test -z "$(git status --porcelain)"
git switch --detach 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test "$(git rev-parse HEAD)" = 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test ! -e exp/pcb/cuhk03
test ! -L exp/pcb/cuhk03
export CUDA_VISIBLE_DEVICES=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
export YOLO_OFFLINE=true
/home/addirakoze/environments/Reid/bin/python - <<'PY_PREFLIGHT'
import sys, hashlib
from pathlib import Path
import torch, torchvision
assert sys.version_info[:3] == (3,11,2)
assert torch.__version__ == "2.7.1+cu126" and torchvision.__version__ == "0.22.1+cu126"
assert torch.version.cuda == "12.6" and torch.backends.cudnn.version() == 90501
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
assert p.name == "NVIDIA TITAN Xp" and p.total_memory == 12774801408
assert str(p.uuid).removeprefix("GPU-") == "bb0efb3f-f2ea-d40d-8769-e4e989bab10c"
def check(path, expected):
    with open(path, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, str(path)
check("configs/pcb/cuhk03.yaml", "0b13581f5de394a2b3f2587ee6ec61b4a9ba05c219a887ae3bd9b7e20d10c1c8")
w = Path(torch.hub.get_dir()) / "checkpoints/resnet50-19c8e357.pth"
assert str(w) == "/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"
check(w, "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097")
base = Path("/home/addirakoze/nobackup/Projects/Dataset")
assert str(base.resolve()) == "/mnt/hd-data/addirakoze/Projects/Dataset"
for relative, digest in {'cuhk03/detected/partitions.pkl': 'a1fd9a06748dd217325820b2ed5c5814de364a6b5e91cb7a93e1924915034f97', 'cuhk03/re_ranking_train_test_split.pkl': '22558735b7a423dacd9158352b6bcc76944c048eaf1df9ada253dbe58a175c3c', 'cuhk03/cuhk03_new_protocol_config_detected.mat': 'ef438e7775fce6d08ef6dff84d2235bd4bed63850dbbf7025f3251e45c7fa721', 'cuhk03/splits_new_detected.json': 'd25235134237c6b84f5200098a53d264335c1ef530901067d486505f51b1bb70'}.items():
    check(base / relative, digest)
print("Frozen source preflight passed; starting fresh cuhk03 PCB training.")
PY_PREFLIGHT
/home/addirakoze/environments/Reid/bin/python -u scripts/train.py --config configs/pcb/cuhk03.yaml
)
```

### msmt17

```bash
(
set -e
source /home/addirakoze/environments/Reid/bin/activate
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot
test "$(hostname -s)" = constantine
test -z "$(git status --porcelain)"
git switch --detach 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test "$(git rev-parse HEAD)" = 66ea88d06ba0f79c298b19abe1652ae892d69fdd
test ! -e exp/pcb/msmt17
test ! -L exp/pcb/msmt17
export CUDA_VISIBLE_DEVICES=GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
export YOLO_OFFLINE=true
/home/addirakoze/environments/Reid/bin/python - <<'PY_PREFLIGHT'
import sys, hashlib
from pathlib import Path
import torch, torchvision
assert sys.version_info[:3] == (3,11,2)
assert torch.__version__ == "2.7.1+cu126" and torchvision.__version__ == "0.22.1+cu126"
assert torch.version.cuda == "12.6" and torch.backends.cudnn.version() == 90501
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
assert p.name == "NVIDIA TITAN Xp" and p.total_memory == 12774801408
assert str(p.uuid).removeprefix("GPU-") == "bb0efb3f-f2ea-d40d-8769-e4e989bab10c"
def check(path, expected):
    with open(path, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, str(path)
check("configs/pcb/msmt17.yaml", "3d618f16698e13192b485a6a1b9424db67e25e7e13e99b1f5d49fafc5598a8f8")
w = Path(torch.hub.get_dir()) / "checkpoints/resnet50-19c8e357.pth"
assert str(w) == "/home/addirakoze/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"
check(w, "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097")
base = Path("/home/addirakoze/nobackup/Projects/Dataset")
assert str(base.resolve()) == "/mnt/hd-data/addirakoze/Projects/Dataset"
for relative, digest in {'msmt17/MSMT17_V2/list_train.txt': '00930d5911e66ec6ac4d9b177e61b49433658758cc091231bf7cfcb83c70d0bf', 'msmt17/MSMT17_V2/list_val.txt': '1b42a24d568a37648405e209f3c50d0d82e908e7009d22326e3089a48f9a7f8c', 'msmt17/MSMT17_V2/list_query.txt': '46b568a56e6e7567175c3350f74362f522d3b898242e0c6d4f87e27ee71cab03', 'msmt17/MSMT17_V2/list_gallery.txt': 'fccd30161c9185bc05f4ee5fa86af5768aab256799b526f87dc5f3a1900288fd'}.items():
    check(base / relative, digest)
print("Frozen source preflight passed; starting fresh msmt17 PCB training.")
PY_PREFLIGHT
/home/addirakoze/environments/Reid/bin/python -u scripts/train.py --config configs/pcb/msmt17.yaml
)
```


Detach without stopping training: press **Ctrl-b**, release, then **d**. After SSH reconnect:

```bash
tmux list-sessions
tmux attach-session -t pcb-market1501
```

Use the corresponding session name for another source. Run remains foreground inside tmux; there is no bare `&` and no automatic restart/next-source launch. A training failure returns to its tmux shell so the error remains visible.

## Monitor the first Market run

From a separate login, use the physical checkout path. These are read-only commands, independently runnable:

```bash
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot
pgrep -af 'scripts/train.py'
ps -u addirakoze -o pid,etime,pcpu,pmem,rss,args
watch -n 5 nvidia-smi -i GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c
tail -F exp/pcb/market1501/train.log
```

Use another terminal or Ctrl-C to leave `watch`/`tail`; that does not stop the separate tmux training process.

```bash
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot
tail -n 50 exp/pcb/market1501/logs/stderr.txt
ls -lh --time-style=long-iso exp/pcb/market1501/checkpoints
ls -lh --time-style=long-iso exp/pcb/market1501/metrics
cat exp/pcb/market1501/metrics/latest_val.json
df -h exp
du -sh exp/pcb/market1501
cat exp/pcb/market1501/artifacts/git_commit.txt
```

Early missing files are expected before their creation. Logs report finite loss/progress every20 training iterations and epoch summaries. `ckpt_last` updates each epoch; `ckpt_best` first appears at the epoch10 evaluation and changes only on improvement. Expect12 periodic JSON records by completion. Evaluation has no per-batch heartbeat and can be quiet, especially MSMT ranking; inspect process CPU/RSS and GPU behavior before calling it stalled. Low GPU utilization during CPU ranking is normal. Do not require a best-checkpoint timestamp change at every evaluation.

Stop the affected training process (Ctrl-C inside its tmux session), retain evidence and report before continuing for NaN/Inf loss, CUDA OOM/error, full disk, dataset I/O failures, failed checkpoint writes, config/commit mismatch, process crash, or thermal instability (persistent throttling/driver thermal warnings, abnormal sustained temperature or shutdown). Do not impose a fabricated universal temperature threshold. Normal loss fluctuations and low early accuracy are not failures. Do not silently reduce batch size, enable AMP, change GPU or resume/restart.

After completion, independently verify epoch120, all12 periodic evaluations, selected-best maximum/tie policy, final-state versus selected-best metrics, and hashes:

```bash
cd /mnt/hd-data/addirakoze/Projects/person-reid-autonomous-robot
sha256sum exp/pcb/market1501/checkpoints/ckpt_best.pth exp/pcb/market1501/checkpoints/ckpt_last.pth
cat exp/pcb/market1501/metrics/final_epoch_test.json
cat exp/pcb/market1501/metrics/final_test.json
```

Do not register completion from checkpoint existence alone. Preserve all evidence for the source-specific review.

## Phase 20 verification and scope

Read-only audit evidence is retained in `/tmp/pcb-phase20.2YDE1Deu/` (`audit.py`, `audit.log`, `audit.json`, package inventory and generated shell blocks). The real CLI `--help` exited0 and confirmed `--config` plus optional `-o/--opts`; launch blocks use only `--config`. Four config validators/loaders, actual source counts, random/sequential samplers, drop-last policies, CPU model structures, parameter/optimizer grouping and actual-step scheduler boundaries passed. CPU audit models used `initialize_pretrained=False` only to inspect structure/count bytes without downloading; the authoritative YAMLs retain `pretrained:true` and the launch commands load the verified historical cache.

No loader was iterated and no model forward, backward, optimizer update, training or evaluation metric was executed. Network audit recorded zero connection/DNS attempts. Importing TensorBoard emitted installed TensorFlow duplicate CUDA-factory warnings; matplotlib used a temporary cache because the sandbox home cache was unwritable. These did not fail the read-only audit and required no dependency or environment changes. Full regression and GPU smoke tests were not repeated for this documentation-only phase.

Validation additionally parses this manifest, compares resolved configurations and hashes with the live loader, syntax-checks each launch block with `bash -n` (without execution), and confirms every line of `plan.md` outside Phase20 is unchanged. All four extracted read-only launch preflights also passed against the actual UUID-selected TITAN Xp, environment, cache and source data hashes; no training entry point was executed. Production code, presets, datasets, weights and prior results are unchanged. Phase21 remains unstarted. Review/acceptance and a separate source-specific authorization are required before any launch.
