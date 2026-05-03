import os.path as osp
import json
import numpy as np
import torch

from reid.metrics.distance import compute_dist, normalize
from reid.metrics.ranking import cmc, mean_ap, mean_inp
from reid.metrics.rerank import re_ranking
from reid.models.outputs import ensure_output_dict
from reid.utils.io import ensure_dir


MSMT17_RERANK_WARNING = "[Warning] MSMT17 reranking may require large memory due to gallery size."


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, cams, names, marks = [], [], [], [], []
    for imgs, pid, cam, name, mark in loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = ensure_output_dict(model(imgs))
        emb = outputs.get("emb")
        if emb is None:
            raise ValueError("Model output dict is missing 'emb'.")
        emb = emb.detach().cpu().numpy()
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


def _test_dataset_name(cfg):
    return str(
        cfg.get("data", {})
        .get("test", {})
        .get("dataset", {})
        .get("name", "")
    ).lower()


def _warn_rerank_memory(cfg, logger=None):
    if _test_dataset_name(cfg) == "msmt17":
        message = MSMT17_RERANK_WARNING
    else:
        message = "Re-ranking enabled. This may use significant CPU memory for large galleries."

    if logger is not None:
        logger.warning(message)
    else:
        print(message)


def evaluate_reid(cfg, model, test_loader, device, logger=None):
    feat, ids, cams, im_names, marks = extract_features(model, test_loader, device)

    if cfg["eval"]["normalize_feat"]:
        feat = normalize(feat, axis=1)

    q = (marks == 0)
    g = (marks == 1)

    q_feat = feat[q]
    g_feat = feat[g]
    q_ids = ids[q]
    g_ids = ids[g]
    q_cams = cams[q]
    g_cams = cams[g]

    dist = compute_dist(q_feat, g_feat, metric=cfg["eval"]["distance"])

    mAP = mean_ap(dist, q_ids, g_ids, q_cams, g_cams, average=True)
    cmc_scores = cmc(
        dist, q_ids, g_ids, q_cams, g_cams,
        topk=max(cfg["eval"]["topk"]),
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=True,
        average=True
    )
    mINP = mean_inp(dist, q_ids, g_ids, q_cams, g_cams, average=True)

    topk = cfg["eval"]["topk"]
    out = {
        "mAP": float(mAP),
        "mINP": float(mINP),
        "Rank1": float(cmc_scores[0]) if len(cmc_scores) > 0 else None,
        "Rank5": float(cmc_scores[4]) if len(cmc_scores) > 4 else None,
        "Rank10": float(cmc_scores[9]) if len(cmc_scores) > 9 else None,
        "cmc": [float(cmc_scores[k-1]) for k in topk],
    }

    rerank_cfg = cfg["eval"].get("rerank", {})
    if rerank_cfg.get("enabled", False):
        _warn_rerank_memory(cfg, logger=logger)
        q_q_dist = compute_dist(q_feat, q_feat, metric=cfg["eval"]["distance"])
        g_g_dist = compute_dist(g_feat, g_feat, metric=cfg["eval"]["distance"])
        rerank_dist = re_ranking(
            dist,
            q_q_dist,
            g_g_dist,
            k1=int(rerank_cfg.get("k1", 20)),
            k2=int(rerank_cfg.get("k2", 6)),
            lambda_value=float(rerank_cfg.get("lambda_value", 0.3)),
        )

        rerank_map = mean_ap(rerank_dist, q_ids, g_ids, q_cams, g_cams, average=True)
        rerank_cmc = cmc(
            rerank_dist, q_ids, g_ids, q_cams, g_cams,
            topk=max(topk),
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True,
            average=True
        )
        rerank_minp = mean_inp(rerank_dist, q_ids, g_ids, q_cams, g_cams, average=True)
        out.update({
            "rerank_mAP": float(rerank_map),
            "rerank_mINP": float(rerank_minp),
            "rerank_Rank1": float(rerank_cmc[0]) if len(rerank_cmc) > 0 else None,
            "rerank_Rank5": float(rerank_cmc[4]) if len(rerank_cmc) > 4 else None,
            "rerank_Rank10": float(rerank_cmc[9]) if len(rerank_cmc) > 9 else None,
            "rerank_cmc": [float(rerank_cmc[k-1]) for k in topk],
        })
    return out
