import torch

from reid.data.protocol import normalize_eval_sample, normalize_train_sample


def train_collate_fn(batch):
    samples = [normalize_train_sample(sample) for sample in batch]
    imgs = [sample.image for sample in samples]
    labels = [sample.label for sample in samples]
    labels = torch.tensor(labels, dtype=torch.long)
    return torch.stack(imgs, dim=0), labels


def test_collate_fn(batch):
    samples = [normalize_eval_sample(sample) for sample in batch]
    imgs = [sample.image for sample in samples]
    pids = [sample.pid for sample in samples]
    camids = [sample.camid for sample in samples]
    names = [sample.image_name for sample in samples]
    marks = [sample.mark for sample in samples]
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.long),
        torch.tensor(camids, dtype=torch.long),
        names,
        torch.tensor(marks, dtype=torch.long),
    )


train_collate_fn.__test__ = False
test_collate_fn.__test__ = False
