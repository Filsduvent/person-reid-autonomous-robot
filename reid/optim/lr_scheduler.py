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


class SuccessfulUpdateLR:
    """Opt-in rates for the NEXT step, indexed by completed primary updates.

    No epoch-length conversion or constructor auto-step.
    prepare(n) sets the next rate; validated state loading restores the counter.
    Neither API restores
    optimizer tensors, samplers, stochastic state or a training trajectory.
    """
    unit = 'successful_primary_updates'

    def __init__(self, optimizer, milestones, *, gamma, warmup_iters, warmup_factor):
        self.optimizer = optimizer
        self.milestones = tuple(milestones)
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.base_lrs = tuple(float(group['lr']) for group in optimizer.param_groups)
        self.completed_updates = 0
        self.prepare(0)

    def rates_after(self, completed):
        if type(completed) is not int or completed < 0:
            raise ValueError('Completed updates must be a nonnegative integer')
        factor = 1.0
        if completed < self.warmup_iters:
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * completed / self.warmup_iters
        decay = self.gamma ** bisect_right(self.milestones, completed)
        return tuple(base * factor * decay for base in self.base_lrs)

    def prepare(self, completed):
        rates = self.rates_after(completed)
        for group, rate in zip(self.optimizer.param_groups, rates):
            group['lr'] = rate
        self.completed_updates = completed

    def step(self):
        self.prepare(self.completed_updates + 1)

    def get_last_lr(self):
        return list(self.rates_after(self.completed_updates))

    def state_dict(self):
        return dict(schema_version=1, unit=self.unit, milestones=self.milestones,
                    gamma=self.gamma, warmup_iters=self.warmup_iters,
                    warmup_factor=self.warmup_factor, base_lrs=self.base_lrs,
                    completed_updates=self.completed_updates)

    def validate_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError('Unsupported successful-update scheduler state')
        for key in expected.keys() - {'completed_updates'}:
            if state[key] != expected[key]:
                raise ValueError(f'Incompatible scheduler {key}')
        rates = self.rates_after(state['completed_updates'])
        if tuple(group['lr'] for group in self.optimizer.param_groups) != rates:
            raise ValueError('Restored optimizer LR does not match scheduler count')

    def load_state_dict(self, state):
        # Construct scheduler on the fresh optimizer, load optimizer state FIRST,
        # then validate/load this state. Never step on resume.
        self.validate_state_dict(state)
        self.prepare(state['completed_updates'])
