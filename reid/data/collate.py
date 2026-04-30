import torch


def train_collate_fn(batch):
    imgs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return torch.stack(imgs, dim=0), labels


def test_collate_fn(batch):
    imgs, pids, camids, names, marks = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.long),
        torch.tensor(camids, dtype=torch.long),
        list(names),
        torch.tensor(marks, dtype=torch.long),
    )


train_collate_fn.__test__ = False
test_collate_fn.__test__ = False
