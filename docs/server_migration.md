# Constantine to Oneiros

Destination: `addirakoze@oneiros.inf.ufpr.br`. The approved layout uses Oneiros's
large `/home` disk. Your normal terminal paths remain the same:

| Resource | Path on Oneiros |
| --- | --- |
| Main project | `/home/addirakoze/nobackup/Projects/person-reid-autonomous-robot` |
| Other worktrees | `/home/addirakoze/nobackup/Projects/<worktree-name>` |
| Dataset | `/home/addirakoze/nobackup/Projects/Dataset` |
| Python environment | `/home/addirakoze/environments/Reid` |
| Torch pretrained cache | `/home/addirakoze/.cache/torch/hub/checkpoints` |
| Migration kit | `/home/addirakoze/nobackup/Projects/migration-kit` |
| Isolated base Python | `/home/addirakoze/.local/share/reid-migration/python3112` |

Oneiros does not provide `/mnt/hd-data`, and the account cannot create it. The
user approved using the existing `/home` paths. Separate configuration candidates
relocate the old `/mnt/hd-data` paths into `/home`; historical receipts and source
configs remain unchanged. The temporary evaluation-review worktree is also
preserved under the project parent instead of `/tmp`.

Observed host inventory: RTX A4000, 16,376 MiB VRAM, driver 590.44.01, and roughly
2.2 TiB available on `/home`. No compute processes were reported at inspection.
This is a time-specific snapshot, not a GPU reservation or training admission.

## SSH access

Run on Constantine, in your normal terminal. Do not overwrite an existing key
when `ssh-keygen` prompts; use the existing key or a different filename.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_oneiros -C "constantine-to-oneiros"
ssh-copy-id -i ~/.ssh/id_ed25519_oneiros.pub addirakoze@oneiros.inf.ufpr.br
ssh -i ~/.ssh/id_ed25519_oneiros -o IdentitiesOnly=yes addirakoze@oneiros.inf.ufpr.br hostname
```

Enter the Oneiros account password only in the terminal. If you set a key
passphrase, unlock it using `ssh-add ~/.ssh/id_ed25519_oneiros` in a terminal with
a running SSH agent. Automated access also requires that agent to be available
in the execution session; otherwise run the transfer interactively.

## Prepare the kit

The preparer writes only a fresh local kit; it never connects or runs training:

```bash
/home/addirakoze/environments/Reid/bin/python deploy/server_migration.py --output .local/server_migration/oneiros-home-2026-11-06 --target-parent /home/addirakoze/nobackup/Projects
```

Use a fresh output name for subsequent exports. The kit includes live pip pins,
Python/CUDA build metadata, Git history, worktree changes, per-file inventories,
dataset/cache inventories, config candidates, transfer scripts, and checksums.
All seven registered worktrees are included, including the temporary evaluation
review tree. Worktrees become independent repositories at the same commits and
branches, avoiding source-machine `.git` pointers. All source branches/tags
are retained in the bundle; source branches are fetched under `refs/remotes/source`.

Original config files, result receipts, checkpoints, maps, and frozen code are
copied unchanged. Config candidates are separate, retain preflight policies,
and require review in `config-index.json`; they are never installed automatically.
Historical absolute paths inside receipts remain historical provenance.

## Inspect and transfer

First inspect Oneiros using SSH: `df -h`, `python3.11 --version`, `command -v rsync`,
and `nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv`.
The initial inventory is approximately **92 GiB of file payload** across all
worktrees, datasets, and cache. Allow additional space for the environment, new
checkpoints, and logs. Inventory sizes are logical sizes, not a disk reservation.
Use a destination without existing project files; scripts never delete, but an
rsync copy can update matching destination filenames.

From Constantine, copy only the setup script to start:

```bash
scp -i ~/.ssh/id_ed25519_oneiros .local/server_migration/oneiros-home-2026-11-06/target-setup.sh addirakoze@oneiros.inf.ufpr.br:/tmp/reid-target-setup.sh
ssh -i ~/.ssh/id_ed25519_oneiros addirakoze@oneiros.inf.ufpr.br 'bash /tmp/reid-target-setup.sh directories'
export RSYNC_RSH='ssh -i /home/addirakoze/.ssh/id_ed25519_oneiros -o IdentitiesOnly=yes'
MODE=preview bash .local/server_migration/oneiros-home-2026-11-06/transfer.sh
MODE=copy bash .local/server_migration/oneiros-home-2026-11-06/transfer.sh
MODE=verify bash .local/server_migration/oneiros-home-2026-11-06/transfer.sh
```

`verify` checks every transferred file's content using rsync checksums without
writing. Expect no differences; inspect any output before proceeding. Transfers
are resumable, retain symlinks, include ignored artifacts and local configs, and
exclude virtual environments, `.git`, credential locations, and Python caches.
Only the project data and Torch checkpoint cache are copied from outside the
worktrees; home-directory credentials and unrelated projects are not transferred.
Review symlinks in `manifest.json` for external dependencies before transferring.

Quiesce your own jobs before the final copy and verification. Do not stop other
users' processes. A live rsync of changing checkpoints is not an atomic snapshot.

On Oneiros, verify the kit, reconstruct Git metadata without checking files out
over the copied working trees, and recreate the environment:

```bash
cd /home/addirakoze/nobackup/Projects/migration-kit
sha256sum -c SHA256SUMS
bash target-setup.sh git
PATH=/home/addirakoze/.local/share/reid-migration/python3112/bin:$PATH bash target-environment.sh
```

Oneiros originally provided Python 3.10.12. `deploy/oneiros_prepare_python.sh`
creates an isolated Python 3.11.2 using [micromamba's documented installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
and conda-forge, without changing system Python. Its archive checksum and explicit
package list are retained under `.local/share/reid-migration/bootstrap`.

The environment script refuses an existing environment. It requires the observed
source Python **3.11.2** and installs the observed PyTorch **2.7.1+cu126** /
torchvision **0.22.1+cu126** builds, followed by the live environment pins. It does
not install drivers or system Python. If Oneiros cannot support these builds,
record and qualify a new environment instead of silently changing dependencies.
The old repository `requirements.txt` alone is insufficient (for example, the
live environment also contains pytest, ONNX, and ONNX Runtime).

## Working on Oneiros

Connect and activate the same environment path you used on Constantine:

```bash
ssh addirakoze@oneiros.inf.ufpr.br
source /home/addirakoze/environments/Reid/bin/activate
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
```

The read-only migration inspection checks imports, frozen code/config hashes,
pretrained weights, and all four sources' portable partition/record/map identity.
It creates no model, does not iterate loaders, and does not initialize CUDA:

```bash
python -B deploy/check_migrated_environment.py --kit /home/addirakoze/nobackup/Projects/migration-kit --output /home/addirakoze/.local/share/reid-migration/cpu-data-check.json
```

Choose a fresh receipt path when repeating the inspection. A successful result
is `CPU_DATA_VERIFIED_GPU_QUALIFICATION_PENDING`, which explicitly leaves GPU
qualification outstanding. The full byte verification is the separate source-side
`MODE=verify` rsync operation. Target Git status may show the same preexisting
local changes/untracked results as Constantine; preserve them.

## Qualification before continuing experiments

Copying the project does not qualify Oneiros for training. In particular, MGN
preflight binds the UUID, GPU name/driver, Python prefix and package versions,
dataset absolute paths, inode/device/mtime inventory, and qualification receipts.
Even identical paths and byte-identical datasets will have different filesystem
identity on the new server. Setting `MGN_AUTHORIZED_GPU_UUID` alone will not pass.

Preserve the old manifest/receipts and perform a separate Oneiros qualification:

1. Record the available, authorized GPU UUID and its runtime and occupancy.
2. Validate dataset partitions, label maps, image counts, and transferred content.
3. Qualify the unchanged batch-64 FP32 MGN recipe on the new hardware. The old
   Market1501 receipt recorded roughly 11.31 GiB peak reservation; the new peak
   must be measured rather than inferred from a free-memory snapshot.
4. Establish a versioned host-specific admission manifest and bind it to the
   unchanged scientific configuration, preserving Constantine's evidence. The
   current preflight uses a fixed manifest path: a reviewed portability change
   is needed before installing new admission evidence. Do not remove preflight.
5. Qualify LODO operational requests separately: prior UUIDs, launch declarations,
   and path-bound runtime receipts do not grant authority on Oneiros.
6. Inspect any intended resume checkpoint, best checkpoint, saved configuration,
   optimizer/scheduler state, and terminal/registry evidence before resuming.
   Preserve selected-best separately from final training state. Cross-host resume
   is not a promise of bitwise/stochastic continuation.

The isolated `market1501_paper_schedule120` trial remains a fresh run, with 120
epochs, milestones 40/60, and no reranking. Do not resume the baseline into it.
No migration script launches training, evaluation, or GPU qualification.
