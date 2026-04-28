from collections import defaultdict
from typing import Dict, Iterator, List
import random
import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[List[int]]):
    """
    Yields batches with P identities and K instances each => batch size = P*K.
    Assumes dataset provides `labels` attribute aligned with indices.
    """
    def __init__(self, labels: List[int], P: int, K: int, drop_last: bool = True, seed: int = 42):
        self.labels = labels
        self.P = int(P)
        self.K = int(K)
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.label_to_indices[int(lab)].append(idx)
        self.unique_labels = list(self.label_to_indices.keys())

    def _build_batches(self) -> List[List[int]]:
        """Construct one finite epoch of PK batches."""
        batch_indices_per_label: Dict[int, List[List[int]]] = {}
        available_labels: List[int] = []

        for lab, indices in self.label_to_indices.items():
            idxs = indices[:]
            if len(idxs) < self.K:
                idxs = self.np_rng.choice(idxs, size=self.K, replace=True).tolist()

            self.rng.shuffle(idxs)
            chunks = [idxs[i:i + self.K] for i in range(0, len(idxs), self.K)]
            chunks = [chunk for chunk in chunks if len(chunk) == self.K]
            if chunks:
                batch_indices_per_label[lab] = chunks
                available_labels.append(lab)

        batches: List[List[int]] = []
        while len(available_labels) >= self.P:
            chosen_labels = self.rng.sample(available_labels, self.P)
            batch: List[int] = []
            for lab in chosen_labels:
                batch.extend(batch_indices_per_label[lab].pop(0))
                if not batch_indices_per_label[lab]:
                    available_labels.remove(lab)
            batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())
