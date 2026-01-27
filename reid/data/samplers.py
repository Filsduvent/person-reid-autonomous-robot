from collections import defaultdict
from typing import Dict, Iterator, List
import random
import torch
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

        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.label_to_indices[int(lab)].append(idx)
        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        labels = self.unique_labels[:]
        self.rng.shuffle(labels)

        # make per-label pools we can sample from repeatedly
        pools = {lab: inds[:] for lab, inds in self.label_to_indices.items()}
        for lab in pools:
            self.rng.shuffle(pools[lab])

        batch: List[int] = []
        while True:
            # sample P labels (with replacement if not enough)
            if len(labels) >= self.P:
                chosen = labels[:self.P]
                labels = labels[self.P:]
            else:
                chosen = labels[:]
                # refill
                labels = self.unique_labels[:]
                self.rng.shuffle(labels)
                while len(chosen) < self.P:
                    chosen.append(labels.pop())

            for lab in chosen:
                inds = pools[lab]
                if len(inds) >= self.K:
                    picked = inds[:self.K]
                    pools[lab] = inds[self.K:]
                else:
                    # not enough left -> sample with replacement from full list
                    full = self.label_to_indices[lab]
                    picked = [self.rng.choice(full) for _ in range(self.K)]
                    pools[lab] = inds  # keep whatever remained
                batch.extend(picked)

            yield batch
            batch = []

    def __len__(self) -> int:
        # undefined meaningful epoch length; let DataLoader run until you stop.
        # We’ll control “steps per epoch” in the train loop for now.
        return 10**9
