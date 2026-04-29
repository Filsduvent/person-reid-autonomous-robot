import numpy as np


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    assert q_g_dist.ndim == 2
    assert q_q_dist.ndim == 2
    assert g_g_dist.ndim == 2
    assert q_q_dist.shape[0] == q_q_dist.shape[1]
    assert g_g_dist.shape[0] == g_g_dist.shape[1]
    assert q_g_dist.shape[0] == q_q_dist.shape[0]
    assert q_g_dist.shape[1] == g_g_dist.shape[0]

    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0,
    )

    original_dist = np.power(original_dist, 2).astype(np.float32)
    max_per_col = np.maximum(np.max(original_dist, axis=0), 1e-12)
    original_dist = np.transpose(original_dist / max_per_col)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        forward_k = initial_rank[i, : k1 + 1]
        backward_k = initial_rank[forward_k, : k1 + 1]
        fi = np.where(backward_k == i)[0]
        k_reciprocal = forward_k[fi]

        expansion = k_reciprocal
        for candidate in k_reciprocal:
            candidate_forward = initial_rank[candidate, : int(np.around(k1 / 2)) + 1]
            candidate_backward = initial_rank[candidate_forward, : int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward == candidate)[0]
            candidate_k = candidate_forward[fi_candidate]

            if len(np.intersect1d(candidate_k, k_reciprocal)) > 2.0 / 3 * len(candidate_k):
                expansion = np.append(expansion, candidate_k)

        expansion = np.unique(expansion)
        weight = np.exp(-original_dist[i, expansion])
        V[i, expansion] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, :]

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    del initial_rank

    inv_index = []
    for i in range(gallery_num):
        inv_index.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros((1, gallery_num), dtype=np.float32)
        ind_non_zero = np.where(V[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_non_zero]

        for j, ind in enumerate(ind_non_zero):
            temp_min[0, ind_images[j]] += np.minimum(
                V[i, ind],
                V[ind_images[j], ind],
            )

        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    final_dist = final_dist[:query_num, query_num:]

    return final_dist
