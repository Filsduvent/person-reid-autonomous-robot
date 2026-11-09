# Publication and two-host training handoff

The user explicitly requested committing/pushing the accumulated work so it can be pulled on Constantine, with official training launched manually by the user. Publish branch `feature/mgn-lodo-v2`; no merge to another branch, force push, full training or benchmark evaluation is part of this publication.

On Constantine, fetch/switch/update without discarding local work:

```bash
cd ~/nobackup/Projects/person-reid-autonomous-robot
git fetch origin
git switch feature/mgn-lodo-v2
git pull --ff-only origin feature/mgn-lodo-v2
```

The publication includes framework code/configurations/tests, MGN Phase19-21 evidence, completed source/evaluation registries, related PCB comparison evidence and migration utilities. Datasets, pretrained/model weights, full experiment directories, Python environments and private `.local/` launch/qualification records remain external dependencies. Git pull does not transfer these large/local resources.

Oneiros has observed bounded qualification for all four folds on its RTX A4000. The publication changes only documents/results and handoff preparation; runtime/configuration bytes and the qualified harness must still match the executed qualification before any release-specific launch records are refreshed. The previous private records and original handoff receipt are retained as historical evidence. Fresh Oneiros records use a release-specific qualification directory and a new `official-20261109-published` run ID. No official output directory is created by preparation.

Constantine does not inherit Oneiros qualification, hardware/runtime identity, absolute data/cache paths or execution receipts. Before a Constantine official command is ready, verify its actual allocated full GPU UUID, Python/CUDA/driver/capability, dataset/cache paths and pinned artifact/source inventories, then run the same approved bounded qualification for the intended folds on that host. Preserve the strict numerical policy, P16/K4/B64 and smoke limits; no automatic fallback or capacity tuning. The existing Oneiros wrapper names Oneiros's UUID and must not be used as a Constantine launcher.

The four folds are independent fresh ImageNet initializations and can be split across the two qualified hosts, with each fold launched once. Record the actual host/runtime per result; deterministic controls do not establish cross-hardware bitwise equivalence. A practical host assignment can be chosen when the user requests the final four commands. Do not run two folds concurrently on the same single GPU. The user starts official training; no training is started by this push/handoff task.
