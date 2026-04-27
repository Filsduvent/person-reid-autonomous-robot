"""Learning-rate scheduler implementations for ReID training."""

from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLR(_LRScheduler):
    """Multi-step decay with Bag of Tricks-style warmup."""

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        milestones = list(milestones)
        if milestones != sorted(milestones):
            raise ValueError("milestones must be a sorted list of increasing integers")
        if warmup_method not in {"constant", "linear"}:
            raise ValueError("warmup_method must be 'constant' or 'linear'")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        num_decays = bisect_right(self.milestones, self.last_epoch)
        decay = self.gamma ** num_decays
        return [base_lr * warmup_factor * decay for base_lr in self.base_lrs]
