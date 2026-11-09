"""Bounded read-only comparison with independently registered historical PCB cells."""
import json
import math
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.mgn_gpu_smoke import file_hash


def main():
    current=yaml.safe_load((ROOT/'docs/mgn_evaluation_results.yaml').read_text())
    sources=('market1501','duke','cuhk03','msmt17');keys=('mAP','mINP','Rank1','Rank5','Rank10')
    historical={};checks=[]
    for source in sources:
        path=ROOT/f'results/pcb_selected_matrix/{source}_row_verification.yaml'
        receipt=yaml.safe_load(path.read_text())
        checkpoint=ROOT/f'exp/pcb/{source}/checkpoints/ckpt_best.pth'
        assert file_hash(checkpoint)==receipt['source_checkpoint_sha256']
        checks.append(dict(source=source,receipt=str(path.relative_to(ROOT)),receipt_sha256=file_hash(path),
                           checkpoint_sha256=receipt['source_checkpoint_sha256'],checkpoint_current_bytes_match=True))
        for target in sources:
            item=receipt['cells'][target];artifact=ROOT/item['artifact']
            assert file_hash(artifact)==item['artifact_sha256']
            row=json.loads(artifact.read_text());assert (row['source_dataset'],row['target_dataset'],row['architecture'])==(source,target,'pcb')
            assert Path(row['checkpoint_path']).name=='ckpt_best.pth'
            assert all(math.isfinite(row[k]) and 0<=row[k]<=1 and row[k]==item[k] for k in keys)
            historical[source,target]=row
    comparisons=[]
    for cell in current['cells']:
        pair=cell['source'],cell['target'];row=historical[pair]
        config=yaml.safe_load((ROOT/cell['output']/'config.resolved.yaml').read_text())
        assert config['data']['test']['dataset']==row['resolved_config']['data']['test']['dataset']
        comparisons.append(dict(source=pair[0],target=pair[1],
            setting=cell['setting'],mgn=cell['metrics'],pcb={k:row[k] for k in keys},
            difference_mgn_minus_pcb_percentage_points={k:100*(cell['metrics'][k]-row[k]) for k in keys},
            declared_target_selector_matches=True,pcb_artifact=f'results/pcb_selected_matrix/{pair[0]}_to_{pair[1]}/cross_dataset.json'))
    output=dict(status='DESCRIPTIVE_RECIPE_COMPARISON_ONLY',mgn_cells_available=len(current['cells']),
        pcb_cells_hash_verified=16,pcb_checkpoints_hash_verified=4,historical_verification=checks,
        caveats=['MGN flip=mean versus historical PCB flip=none; different normalization/training recipes and hardware. Not a matched architecture-only ablation or protocol-matched accuracy ranking.',
                'Declared target selectors match; CUHK03 detected/processed_partition semantic new/classic correspondence remains unverified despite historical documentation labels.',
                'Historical PCB bytes/results preserved. No supplementary evaluation or baseline/PCB rerun.',
                'ResNet50 comparison not recomputed here; existing historical comparison remains at docs/resnet50_pcb_comparison_matrix.md.',
                'MGN diagonal cross-host numerical differences remain flagged; user authorized Phase18 continuation without changing thresholds or source weights.'],
        cells=comparisons)
    (ROOT/'docs/mgn_evaluation_20261106/historical-pcb-comparison.yaml').write_text(yaml.safe_dump(output,sort_keys=False))
    text='# MGN versus historical PCB: descriptive recipe comparison\n\n'
    text+='Historical PCB:16 saved result hashes and4 source checkpoint hashes verified. MGN observations are separate GPU evaluations on Oneiros. These are complete-recipe observations with different training and extraction policies; they do not isolate architecture effects. MGN uses original/flip mean; historical PCB uses no flip. Both use Euclidean distance, global normalization and no reranking. Declared target selectors match. CUHK03 new/classic semantic correspondence remains unverified. No architecture winner or significance claim is made.\n\n'
    text+='| Source → target | MGN mAP (%) | PCB mAP (%) | Difference (points) | MGN Rank-1 (%) | PCB Rank-1 (%) |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for c in comparisons:
        text+=f"| {c['source']} → {c['target']} | {100*c['mgn']['mAP']:.2f} | {100*c['pcb']['mAP']:.2f} | {c['difference_mgn_minus_pcb_percentage_points']['mAP']:+.2f} | {100*c['mgn']['Rank1']:.2f} | {100*c['pcb']['Rank1']:.2f} |\n"
    text+='\nFull precision, five metrics and hash checks are in [historical-pcb-comparison.yaml](mgn_evaluation_20261106/historical-pcb-comparison.yaml). Historical records remain unchanged.\n'
    (ROOT/'docs/mgn_historical_pcb_comparison.md').write_text(text)
    print('Verified16 PCB result hashes and4 original PCB checkpoint hashes; descriptive comparison saved.')


if __name__=='__main__':main()
