# Unified Stage A/B integration — Phase 1 review

The review worktree `person-reid-autonomous-robot-unified-review`, branch
`review/unified-stage-ab-v1`, starts at main
`3b015c964b99bfe8ad4b7842bd174d3bc58af607`. The linear 158-path scientific and
launcher delta was applied exactly once as unstaged changes. There is no merge,
commit or official execution in this phase.

The frozen Stage B experiment protocol, implementation plan and decision register
are authoritative here. Divergent original-main planning documents and all nine
original-main edits remain byte-identical in the original checkout and external
archival evidence. Exact differences/hashes are in the Phase 1 evidence bundle.
No duplicate scientific protocol is introduced.

The shared scientific implementation is unchanged from the validated Stage B
freeze. Stage A keeps its existing loaders, preprocessing, models, losses,
optimizer/scheduler defaults, epoch stopping, selected-best checkpoints, metrics
and output interfaces. The new operational layer is described in
[lodo_operational_binding_v2.md](lodo_operational_binding_v2.md).

| Stage/action | Interface | Phase 1 execution status |
| --- | --- | --- |
| A train | `python scripts/train.py --config <stage-a.yaml>` | Original CPU regressions and synthetic fixtures; no official job |
| A evaluate | `python scripts/evaluate.py --config <stage-a-eval.yaml>` | Existing eval.weight / --weight behavior retained |
| A cross-domain | `python scripts/evaluate_cross_domain.py --config <target.yaml> --checkpoint <source.pth> --source-dataset <source> --target-dataset <target> --output-dir <isolated-cell>` | CLI and implementation unchanged |
| B train | `python scripts/lodo_launch.py --config configs/lodo/v1/resnet50_lodo-1_seed42.yaml` | CPU preparation fixtures; live execution closed pending reviewed unified freeze/binding/approval |
| B resume | Same command plus `--resume` | CPU same-run restoration fixtures; never restart from zero |
| B evaluate | `python scripts/evaluate.py --config <future-stage-b-eval-request.yaml>` | Separate CPU scope and shared evaluator fixtures; no official evaluation |

## Corrected historical state assumptions

All seven frozen failures were in `test_lodo_official_preparation.py`: one valid
LODO-1 CLI case, four counterfeit provenance cases, and wrong runtime/device cases.
They now use temporary candidate roots, YAML/request copies and corresponding file
hashes. Production namespace validators remain active against those temporary
roots. All four counterfeit map/sidecar/data/provenance tests additionally assert
that the live dataset comparison was reached. Negative runtime/GPU cases retain
their specific exception assertions; malformed scope/science cannot pass early.

The launcher row-0 failure in `test_all_eight_build_original_trainer_commands_without_dispatch`
now shares the all-eight fixture using real historical candidate bytes and fresh
temporary unified output leaves. Git reads remain real; only worker execution and
operational freeze declarations are CPU doubles. Independent collision, terminal,
missing-index, fallback/state compatibility, revocation and changed-output cases
remain covered. Official LODO-1 directories are never hidden, moved or deleted.

Existing short scientific end-to-end fixtures explicitly double only operational
authority, just as they already double model computation and official budgets.
Dedicated v2 tests exercise the actual receipt/approval validators independently;
there is no synthetic override flag in production. Test records cannot authorize
real execution. CPU synthetic equivalence is not new GPU qualification.

## Preservation and review gates

Protected original main, scientific and released-launcher worktrees retain their
branches, HEADs, dirty states and artifacts. Their shared Git store gains only the
explicitly authorized review branch/worktree registration. Source bytes, existing
refs, staged states and historical experiment files remain unchanged.

The archived `docs/lodo_phase10_evidence/implementation-review.patch` retains the
previous narrowly approved 39 context-space whitespace lines and exact SHA256
`d40833321643c4c559ccde08cf7443c16fff80405831bb6d9acef6ae44da7d98`.
All other proposed files use strict whitespace checks. The exception does not
apply to new code, tests or documentation.

Full external evidence contains preservation inventories/rehashes, all nine
archival copies, reconciliation diffs, exact commands/environments/JUnit, all eight
metadata preflights, complete integration patch and proposed staging membership.
No files are staged. See the final external A–J report for observed counts and
readiness; this document does not substitute planned gates for executed evidence.

## Future merge procedure — requires explicit approval

Review the complete patch and exact staging membership first. After separate
approval, commit only the reviewed integration paths on the review branch and
record the actual SHA externally. Revalidate source identity and run namespaces
from the intended future canonical checkout. Bind that actual root/commit using
v2; never copy the original LODO-1 execution declaration as future approval.

Before merging main, recheck its HEAD/status and preserve its nine user documents.
The three untracked main LODO paths collide with incoming tracked documentation;
a future approved reconciliation must explicitly preserve/archive those original
paths before any merge that would replace them. Do not reset/stash/clean or absorb
the preliminary-results/plan edits and PCB registries. Stop if main has changed or
new conflicts arise. This phase does not authorize moving/deleting main files.

A future merge/freeze must rerun appropriate regression and provenance gates on
the actual final tree. Obtain separately scoped execution approval before GPU
qualification or any official training/evaluation. The standing GPU allocation
remains valid subject to sharing rules and fresh technical checks. There is no
new scientific campaign, reservation or launch authority in this review.
