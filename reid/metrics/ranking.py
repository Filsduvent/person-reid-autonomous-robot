from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool_)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

def cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams,
        topk=100, separate_camera_set=False, single_gallery_shot=False,
        first_match_break=False, average=True):
    assert isinstance(distmat, np.ndarray)
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    ret = np.zeros([m, topk], dtype=np.float32)
    is_valid_query = np.zeros(m, dtype=np.float32)
    num_valid = 0

    for i in range(m):
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue

        is_valid_query[i] = 1
        if single_gallery_shot:
            repeat = 100
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1

        for _ in range(repeat):
            if single_gallery_shot:
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]

            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[i, k - j] += 1
                    break
                ret[i, k - j] += delta

        num_valid += 1

    if num_valid == 0:
        raise RuntimeError("No valid query")
    ret = ret.cumsum(axis=1)
    if average:
        return np.sum(ret, axis=0) / num_valid
    return ret, is_valid_query

def mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, average=True):
    assert isinstance(distmat, np.ndarray)
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    aps = np.zeros(m, dtype=np.float32)
    is_valid = np.zeros(m, dtype=np.float32)

    for i in range(m):
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        is_valid[i] = 1
        aps[i] = average_precision_score(y_true, y_score)

    if average:
        return float(np.sum(aps) / (np.sum(is_valid) + 1e-12))
    return aps, is_valid

def mean_inp(distmat, query_ids, gallery_ids, query_cams, gallery_cams, average=True):
    assert isinstance(distmat, np.ndarray)
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

    inps = np.zeros(m, dtype=np.float32)
    is_valid = np.zeros(m, dtype=np.float32)

    for i in range(m):
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if not np.any(matches[i, valid]):
            continue
        is_valid[i] = 1
        match_idx = np.where(matches[i, valid])[0]
        first_rank = match_idx[0] + 1
        inps[i] = 1.0 / first_rank

    if average:
        return float(np.sum(inps) / (np.sum(is_valid) + 1e-12))
    return inps, is_valid
