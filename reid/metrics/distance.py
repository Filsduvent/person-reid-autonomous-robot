import numpy as np

def normalize(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def compute_dist(a, b, metric="euclidean"):
    assert metric in ["euclidean", "cosine"]
    if metric == "cosine":
        a = normalize(a, axis=1)
        b = normalize(b, axis=1)
        # cosine similarity; to make it a distance, use 1 - sim if you want
        sim = a @ b.T
        dist = 1.0 - sim
        return dist.astype(np.float32)
    # euclidean
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    d2 = aa + bb - 2.0 * (a @ b.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).astype(np.float32)
