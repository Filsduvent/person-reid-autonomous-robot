"""In-memory FP32 update control and consumed-index exposure; no recovery state.

One consumed minibatch corresponds to one main optimizer step. Accumulation
and AMP are unsupported in this opt-in path. Failed attempts remain visible,
while a partially executed main/auxiliary pair makes the state inconsistent.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path

import torch


class UpdateFailure(RuntimeError):
    """Abort with inspectable in-memory progress; never a resumable checkpoint."""
    def __init__(self, message, progress):
        super().__init__(message)
        self.progress = progress


def nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


class ExposureAccounting:
    phases = ('attempted', 'forward', 'backward', 'successful')

    def __init__(self, dataset, *, sampler_mode, P=None, K=None):
        self.dataset = dataset
        self.mapping_digest = dataset.mapping_digest
        self.identities = {r['global_label']: r for r in dataset.mapping_metadata()['identities']}
        self.source_offsets = {}
        self.image_dirs = {}
        offset = 0
        for name, source in dataset.sources.items():
            self.source_offsets[name] = offset
            self.image_dirs[name] = Path(source.im_dir).resolve()
            offset += len(source)
        if offset != len(dataset):
            raise ValueError('Source index spaces do not cover the dataset')
        self.sampler_mode, self.P, self.K = sampler_mode, P, K
        if sampler_mode == 'pk' and (P is None or K is None or P <= 1 or K <= 1):
            raise ValueError('PK accounting requires the original P/K')
        self.presentations = {phase: Counter() for phase in self.phases}
        self.identity_presentations = {phase: Counter() for phase in self.phases}
        self.identity_slots = {phase: Counter() for phase in self.phases} if sampler_mode == 'pk' else None
        self.batch_counts = Counter()
        self.distinct = {phase: set() for phase in self.phases}
        self.lookup_seconds = 0.0

    def resolve(self, indices, labels):
        if indices is None or len(indices) != len(labels):
            raise ValueError('Missing/unaligned consumed indices')
        if self.dataset.mapping_digest != self.mapping_digest:
            raise ValueError('Dataset identity changed during accounting')
        records = []
        for idx, label in zip(indices, labels.tolist()):
            idx = nonnegative_integer(idx, 'global index')
            record = self.dataset.resolve_index(idx)
            identity = self.identities.get(record.global_label)
            if (record.global_index != idx or record.global_label != int(label)
                    or record.global_label != int(self.dataset.labels[idx]) or identity is None
                    or identity['source_key'] != record.source_key
                    or identity['source_exposed_pid'] != record.source_exposed_pid
                    or identity['local_label'] != record.local_label):
                raise ValueError('Sidecar/global identity mismatch')
            source = self.dataset.sources[record.source_key]
            i = nonnegative_integer(record.source_local_index, 'source-local index')
            if (i + self.source_offsets[record.source_key] != idx
                    or record.canonical_path != str(self.image_dirs[record.source_key] / record.image_reference)
                    or i >= len(source) or int(source.pids[i]) != record.source_exposed_pid
                    or int(source.labels[i]) != record.local_label or int(source.cams[i]) != record.camera
                    or str(source.im_names[i]) != record.image_reference):
                raise ValueError('Sidecar/source-local metadata mismatch')
            records.append(record)
        if self.sampler_mode == 'pk':
            counts = Counter(r.global_label for r in records)
            if len(counts) != self.P or set(counts.values()) != {self.K}:
                raise ValueError('Consumed PK batch does not match original P/K')
        return tuple(records)

    def record(self, phase, records):
        self.batch_counts[phase] += 1
        self.presentations[phase].update(r.source_key for r in records)
        self.identity_presentations[phase].update(r.global_label for r in records)
        ids = {r.global_label: r.source_key for r in records}
        self.distinct[phase].update(ids)
        if self.identity_slots is not None:
            self.identity_slots[phase].update(ids.values())  # one slot per chosen identity

    def summary(self):
        result = {}
        for phase in self.phases:
            counts = self.presentations[phase]; total = sum(counts.values())
            result[phase] = dict(batches=self.batch_counts[phase], images=total,
                                per_source_images=dict(counts),
                                per_source_percent={s: 100 * counts[s] / total if total else 0.0
                                                    for s in self.dataset.sources},
                                per_identity_presentations=dict(self.identity_presentations[phase]),
                                distinct_global_identities=sorted(self.distinct[phase]),
                                per_source_distinct_identities={s: sum(self.identities[g]['source_key'] == s
                                                                      for g in self.distinct[phase])
                                                                for s in self.dataset.sources},
                                per_source_identity_slots=dict(self.identity_slots[phase])
                                if self.identity_slots is not None else None)
        result['lookup_seconds'] = self.lookup_seconds
        return result


@dataclass
class UpdateProgress:
    max_updates: int
    auxiliary: bool = False
    history_limit: int = 32
    completed_primary_updates: int = 0
    auxiliary_steps: int = 0
    main_steps_returned: int = 0
    attempted_images: int = 0
    unattributed_attempted_images: int = 0
    unattributed_attempted_batches: int = 0
    attempted_batches: int = 0
    forward_batches: int = 0
    backward_batches: int = 0
    consistent: bool = True
    failed: bool = False
    failure_reason: str | None = None
    exposure: ExposureAccounting | None = None
    epoch_results: list = field(default_factory=list)

    def __post_init__(self):
        self.max_updates = nonnegative_integer(self.max_updates, 'max_updates')
        if not self.max_updates:
            raise ValueError('max_updates must be positive')
        self.history_limit = nonnegative_integer(self.history_limit, 'history_limit')
        if type(self.auxiliary) is not bool:
            raise ValueError('auxiliary must be Boolean')
        if self.completed_primary_updates or self.auxiliary_steps or self.main_steps_returned:
            raise ValueError('Phase 5 progress starts fresh; recovery is deferred')
        self.index_history = deque(maxlen=self.history_limit)
        self.lr_trace = deque(maxlen=self.history_limit)

    @property
    def budget_completed(self):
        return (not self.failed and self.consistent and
                self.completed_primary_updates == self.max_updates)

    def bind(self, loader):
        if self.exposure is None:
            batch_sampler = loader.batch_sampler
            self.exposure = ExposureAccounting(loader.dataset, sampler_mode=loader.sampler_mode,
                                                P=getattr(batch_sampler, 'P', None), K=getattr(batch_sampler, 'K', None))
        elif self.exposure.dataset is not loader.dataset or self.exposure.sampler_mode != loader.sampler_mode:
            raise ValueError('Progress cannot be reused with a different dataset/sampler')

    def fail(self, error, *, inconsistent=False):
        self.failed = True
        self.consistent = self.consistent and not inconsistent
        self.failure_reason = str(error)

    def check_totals(self):
        if self.exposure is None:
            return
        e = self.exposure
        for phase, count in [('attempted', self.attempted_batches), ('forward', self.forward_batches),
                             ('backward', self.backward_batches), ('successful', self.completed_primary_updates)]:
            unattributed = self.unattributed_attempted_batches if phase == 'attempted' else 0
            if e.batch_counts[phase] + unattributed != count:
                raise ValueError(f'Exposure/batch count mismatch: {phase}')
            if sum(e.presentations[phase].values()) != sum(e.identity_presentations[phase].values()):
                raise ValueError(f'Exposure image/identity totals mismatch: {phase}')
        if sum(e.presentations['attempted'].values()) + self.unattributed_attempted_images != self.attempted_images:
            raise ValueError('Attempted image totals mismatch')
        if not self.failed:
            if self.main_steps_returned != self.completed_primary_updates or self.auxiliary_steps != (
                    self.completed_primary_updates if self.auxiliary else 0):
                raise ValueError('Primary/auxiliary consistency boundary mismatch')


@dataclass(frozen=True)
class UpdateEpochResult:
    epoch: int
    stop_reason: str  # budget_completed, epoch_exhausted, or failed
    partial_epoch: bool
    attempted_batches: int
    forward_batches: int
    backward_batches: int
    successful_updates: int
    auxiliary_steps: int
    forward_images: int
    successful_images: int
    loss_batches: int
    mean_loss: float | None
    mean_loss_terms: dict
    mean_weighted_loss_terms: dict
    lr_used: tuple | None
    lr_next: tuple | None
    observation: dict


def validate_update_mode(progress, loader, model, criterion, optimizer, aux_optimizer, scheduler, amp):
    from reid.data.index_observation import ObservedTrainLoader
    from reid.optim.lr_scheduler import SuccessfulUpdateLR
    if not isinstance(progress, UpdateProgress) or not isinstance(loader, ObservedTrainLoader):
        raise ValueError('Update mode requires UpdateProgress and an observed loader')
    if amp:
        raise ValueError('Successful-update mode supports FP32 only; AMP is deferred')
    if progress.failed or not progress.consistent:
        raise ValueError('Failed/inconsistent progress cannot continue; recovery is deferred')
    if progress.completed_primary_updates > progress.max_updates:
        raise ValueError('Progress exceeds the update budget')
    if aux_optimizer is not None and any(group['lr'] != 0.5 for group in aux_optimizer.param_groups):
        raise ValueError('Center auxiliary LR must remain 0.5')
    if progress.auxiliary != (aux_optimizer is not None):
        raise ValueError('Expected auxiliary optimizer does not match progress')
    if not isinstance(scheduler, SuccessfulUpdateLR) or scheduler.optimizer is not optimizer:
        raise ValueError('Update mode requires the matching successful-update scheduler')
    if scheduler.completed_updates != progress.completed_primary_updates:
        raise ValueError('Scheduler and successful-update counters disagree')
    for module in (model, criterion):
        if hasattr(module, 'parameters') and any(p.is_floating_point() and p.dtype != torch.float32 for p in module.parameters()):
            raise ValueError('Successful-update mode requires FP32 parameters')
    progress.bind(loader)
    progress.check_totals()


def finite_gradients(optimizer, *, require_any=True):
    gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if require_any and not gradients:
        raise ValueError('Optimizer has no gradients for this update')
    if any(not torch.isfinite(g).all().item() for g in gradients):
        raise ValueError('Nonfinite optimizer gradient')


def finite_parameters(optimizer):
    if any(not torch.isfinite(p).all().item() for group in optimizer.param_groups for p in group['params']):
        raise ValueError('Nonfinite parameter after optimizer step')


def epoch_batches(loader, progress):
    """Budget guard runs BEFORE next(iterator), including before iterator creation."""
    if progress is not None and progress.budget_completed:
        return
    iterator = iter(loader)
    try:
        while progress is None or progress.completed_primary_updates < progress.max_updates:
            try:
                batch = next(iterator)
            except StopIteration:
                return
            yield batch
    finally:
        if progress is not None:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
