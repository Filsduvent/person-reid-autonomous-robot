import os.path as osp
import json
import numpy as np
import torch

from reid.metrics.distance import compute_dist, normalize
from reid.metrics.ranking import cmc, mean_ap, mean_inp
from reid.utils.io import ensure_dir

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, cams, names, marks = [], [], [], [], []
    for imgs, pid, cam, name, mark in loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs).detach().cpu().numpy()
        feats.append(emb)
        pids.append(pid.numpy())
        cams.append(cam.numpy())
        names.append(np.array(name))
        marks.append(mark.numpy())
    return (np.vstack(feats),
            np.hstack(pids),
            np.hstack(cams),
            np.hstack(names),
            np.hstack(marks))

def evaluate_reid(cfg, model, test_loader, device):
    feat, ids, cams, im_names, marks = extract_features(model, test_loader, device)

    if cfg["eval"]["normalize_feat"]:
        feat = normalize(feat, axis=1)

    q = (marks == 0)
    g = (marks == 1)

    dist = compute_dist(feat[q], feat[g], metric=cfg["eval"]["distance"])

    mAP = mean_ap(dist, ids[q], ids[g], cams[q], cams[g], average=True)
    cmc_scores = cmc(
        dist, ids[q], ids[g], cams[q], cams[g],
        topk=max(cfg["eval"]["topk"]),
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True,
        average=True
    )
    mINP = mean_inp(dist, ids[q], ids[g], cams[q], cams[g], average=True)

    topk = cfg["eval"]["topk"]
    out = {
        "mAP": float(mAP),
        "mINP": float(mINP),
        "Rank1": float(cmc_scores[0]) if len(cmc_scores) > 0 else None,
        "Rank5": float(cmc_scores[4]) if len(cmc_scores) > 4 else None,
        "Rank10": float(cmc_scores[9]) if len(cmc_scores) > 9 else None,
        "cmc": [float(cmc_scores[k-1]) for k in topk],
    }
    return out
