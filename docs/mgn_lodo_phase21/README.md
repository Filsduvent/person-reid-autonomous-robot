# Phase21 qualification and user-operated training

Observed qualification status, per-fold class counts, exact executed release, numerical evidence, memory, failure history and preserved evidence hashes are in `qualification_results.yaml`. The eight-case bounds and the bounded-recovery limitations remain in `execution_scope.md` and `bounded_recovery_review.md`. No official training or benchmark evaluation is performed by the qualification harness.

After corresponding successful qualification and private launch preparation, run one fold at a time on the single authorized RTX A4000:

```bash
cd /home/addirakoze/nobackup/Projects/person-reid-autonomous-robot
bash docs/mgn_lodo_phase21/official_training_commands.sh 1
bash docs/mgn_lodo_phase21/official_training_commands.sh 2
bash docs/mgn_lodo_phase21/official_training_commands.sh 3
bash docs/mgn_lodo_phase21/official_training_commands.sh 4
```

Each wrapper invokes the existing `scripts/train.py --config <pinned YAML>` with its separate private execution receipt, a single visible full GPU UUID and the strict workspace setting before Python imports. It does not choose a source recipe or change the successful-update budget. Each fold initializes freshly from the pinned ImageNet artifact and trains to85,200 successful primary updates. The folds hold out Market1501/Duke/CUHK03/MSMT17 respectively.

Console logs append to `.local/mgn_lodo_launch/LODO-<n>/official-20261109/train.log`. For example, in another terminal:

```bash
tail -f .local/mgn_lodo_launch/LODO-1/official-20261109/train.log
```

Training artifacts use `exp/lodo/mgn_lodo_v2/mgn/LODO-<n>/seed42/official-20261109/`. The selected result is only `checkpoints/ckpt_terminal.pth` after the exact successful-update budget. Source exposure, discarded/unknown work, recovery events, checkpoints and actual runtime evidence are recorded by the common framework. No target evaluation or target BN adaptation occurs during training.

The prepared launch scope pins the exact current commit, runtime/config/data and qualification evidence. Do not edit runtime/config/receipt bytes to bypass a refusal. A changed release, dataset, GPU/runtime or existing output requires corresponding validation and an explicit new or resume scope. Repeating the fresh command after an interrupted run does not silently resume or overwrite it.

The private qualification and execution files under `.local/mgn_lodo_launch/` are local0600 operational records; they are not committed, institutional authentication, or evidence of retrieval quality. `launch_preparation_receipt.yaml` records only their hashes and paths after actual successful admission, with zero official launches. Documentation-only promotion from the executed runtime commit must preserve every runtime/config byte and the tested harness identity; any runtime edit requires requalification rather than a copied PASS.
