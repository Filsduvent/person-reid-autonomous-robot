"""CPU/index-only full real source epoch boundaries; no pixels/models/CUDA."""
import copy
import json
from pathlib import Path
import random
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from reid.utils.config import load_config
from reid.data.build import build_train_loader
from reid.utils.lodo_checkpoint import rng_state, restore_rng, digest

ROOT=Path(__file__).resolve().parents[2]
report=dict(status='PASS_INDEX_ONLY_NOT_FULL_EPOCH_CUDA_RECOVERY',image_batches=0,models=0,cuda_initialized=False,folds=[])
def prohibited(*a,**k):raise AssertionError('Pixel/CUDA access in index-only boundary evidence')
with patch.object(Image,'open',prohibited),patch.object(torch.cuda,'is_available',prohibited):
    for fold in range(1,5):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        cfg=load_config(str(ROOT/f'configs/lodo/mgn_lodo_v2/mgn_lodo-{fold}_seed42.yaml'))
        loader,classes=build_train_loader(cfg)
        epoch1=[tuple(batch) for batch in loader.batch_sampler]
        state=copy.deepcopy(rng_state(loader))
        epoch2=[tuple(batch) for batch in loader.batch_sampler]
        restore_rng(state,loader)
        replay=[tuple(batch) for batch in loader.batch_sampler]
        assert epoch2==replay and len(epoch1)>1 and len(epoch2)>1
        assert all(len(b)==64 for b in epoch1+epoch2)
        assert not torch.cuda.is_initialized()
        report['folds'].append(dict(fold=f'LODO-{fold}',classes=classes,first_epoch_batches=len(epoch1),second_epoch_batches=len(epoch2),first_epoch_index_sha256=digest(epoch1),next_epoch_replay_sha256=digest(epoch2),next_full_metadata_epoch_exact=True,map_sha256=loader.dataset.mapping_digest))
        print(f'LODO-{fold}: full index epochs {len(epoch1)}/{len(epoch2)}; RNG restored, next index epoch identical',flush=True)
(ROOT/'docs/mgn_lodo_phase21/real_index_rng.json').write_text(json.dumps(report,indent=2))
