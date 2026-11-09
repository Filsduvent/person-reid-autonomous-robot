"""Register verified row coverage and all metric tables after the 16-cell run."""
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.mgn_gpu_smoke import file_hash


def main():
    suite=ROOT/'results/mgn_selected_matrix/oneiros_20261106_attempt2'
    command=[sys.executable,'-B',str(Path(__file__).with_name('verify_results.py')),'--suite',str(suite.relative_to(ROOT))]
    completed=subprocess.run(command,check=True,capture_output=True,text=True)
    verification=yaml.safe_load(completed.stdout)
    (ROOT/'docs/mgn_evaluation_20261106/matrix-integrity-check.yaml').write_text(completed.stdout)
    r=yaml.safe_load((suite/'registry.yaml').read_text())
    sources=('market1501','duke','cuhk03','msmt17');keys=('mAP','mINP','Rank1','Rank5','Rank10')
    for source in sources:
        path=suite/f'{source}_row_verification.yaml'
        if path.exists():raise RuntimeError('Preserve existing row receipt: '+str(path))
        cells=[c for c in r['cells'] if c['source']==source]
        assert len(cells)==4 and {c['target'] for c in cells}==set(sources)
        row=dict(status='FOUR_CELLS_ARTIFACT_VERIFIED_REVIEW_PENDING',source=source,release_commit=r['release_commit'],
            checkpoint=r['source_checkpoints'][source],fixed_original_checkpoint_across_all_targets=True,
            no_target_based_selection=True,gpu_uuid=r['gpu_before']['uuid'],
            cells={c['target']:dict(setting=c['setting'],metrics=c['metrics'],output=c['output'],
                 metric_record_sha256=file_hash(ROOT/c['output']/'cross_dataset.json'),
                 completion_receipt_sha256=file_hash(suite/'_audit'/f"{source}_to_{c['target']}"/'completion.yaml')) for c in cells},
            diagonal_tolerance_exception=next(c for c in cells if c['target']==source)['difference_from_training_fraction'],
            numerical_exception_disposition=r.get('numerical_exception_approval','REVIEW_PENDING'),
            protected_original_results=True,training_performed=False)
        path.write_text(yaml.safe_dump(row,sort_keys=False))
    summaries={}
    for setting in ('within-domain','cross-domain'):
        cells=[c for c in r['cells'] if c['setting']==setting]
        summaries[setting]=dict(cells=len(cells),unweighted_mean_metrics={k:sum(c['metrics'][k] for c in cells)/len(cells) for k in keys})
    (ROOT/'docs/mgn_evaluation_20261106/descriptive-summary.yaml').write_text(yaml.safe_dump(summaries,sort_keys=False))
    path=ROOT/'docs/mgn_evaluation_matrix.md';text=path.read_text()
    text+='\n## Additional metrics (%)\n'
    for key in ('Rank5','Rank10','mINP'):
        lookup={(c['source'],c['target']):c['metrics'][key]*100 for c in r['cells']}
        text+=f'\n### {key}\n\n| Source → target | Market1501 | Duke | CUHK03 | MSMT17 |\n| --- | ---: | ---: | ---: | ---: |\n'
        for source in sources:text+='| '+source+' | '+' | '.join(f'{lookup[source,target]:.2f}' for target in sources)+' |\n'
    text+='\n## Verification and interpretation\n\n'
    text+='All16 unique cells and4 fixed source rows passed artifact/configuration/GPU-placement verification; all evaluator processes exited0. Row receipts are stored beside the matrix CSV. Four standalone diagonals retain numerical differences beyond the strict1e-5 threshold; the user explicitly authorized Phase18 continuation while preserving those flags. No threshold was widened and no hardware cause was proven. FP32 parameters/inputs and AMP=false use qualified matmul TF32=false/cuDNN TF32=true controls; effective convolution arithmetic differs in capability from the previous Pascal host.\n\n'
    text+='Full verification: [matrix-integrity-check.yaml](mgn_evaluation_20261106/matrix-integrity-check.yaml). Descriptive averages: [descriptive-summary.yaml](mgn_evaluation_20261106/descriptive-summary.yaml). Bounded historical recipe comparison: [MGN versus PCB](mgn_historical_pcb_comparison.md). Human/advisor review remains separate from observed execution and artifact integrity. The next roadmap phase is the versioned MGN LODO scientific decision; no LODO implementation or training is started here.\n'
    path.write_text(text)
    print(completed.stdout)
    print('Four immutable row verification receipts and five metric tables registered.')


if __name__=='__main__':main()
