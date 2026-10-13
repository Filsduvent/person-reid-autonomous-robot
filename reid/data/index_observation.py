"""Opt-in observation of the ORIGINAL DataLoader batch sampler.

No dataset/sample/collator changes and no RNG draws. The parent process records
batches dispatched to the loader, then pairs each in-order returned batch with
its indices. Dispatched/prefetched batches are NOT model-processed exposure.
This is in-memory observation, not sampler replay or worker RNG recovery.
"""
from collections import deque
from torch.utils.data import DataLoader


class _ObservedBatchSampler:
    def __init__(self, original):
        self.original = original
        self.pending = deque()
        self.scheduled_batches = self.scheduled_images = 0

    def __iter__(self):
        for indices in self.original:
            recorded = tuple(indices)
            self.pending.append(recorded)
            self.scheduled_batches += 1
            self.scheduled_images += len(recorded)
            yield indices  # same indices/order; no extra iteration of original

    def __len__(self):
        return len(self.original)


class ObservedTrainLoader:
    """Expose latest consumed indices beside an unchanged (images, labels) batch.

    The uniterated loader's original sampler, generator and worker initialization
    are reused. Constructor does not draw RNG; only one resulting loader iterates.
    Concurrent iterators, persistent workers and out-of-order loading are rejected
    here; recovery and persistent-worker lifecycle are Phase 6 concerns.
    """
    def __init__(self, loader, *, sampler_mode):
        if sampler_mode not in ('pk', 'random'):
            raise ValueError('Unsupported observed sampler mode')
        if loader.persistent_workers or not getattr(loader, 'in_order', True):
            raise ValueError('Observation requires nonpersistent, in-order workers')
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler  # preserve public original sampler
        self.sampler = loader.sampler
        self.batch_size, self.drop_last = loader.batch_size, loader.drop_last
        self.sampler_mode = sampler_mode
        self.effective_batch_size = getattr(loader, 'effective_batch_size', loader.batch_size)
        self._observer = _ObservedBatchSampler(loader.batch_sampler)
        kwargs = dict(batch_sampler=self._observer, num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory, collate_fn=loader.collate_fn,
                      worker_init_fn=loader.worker_init_fn, generator=loader.generator,
                      timeout=loader.timeout)
        if loader.num_workers:
            kwargs.update(prefetch_factor=loader.prefetch_factor,
                          multiprocessing_context=loader.multiprocessing_context)
        if loader.pin_memory_device:
            kwargs['pin_memory_device'] = loader.pin_memory_device
        self._loader = DataLoader(self.dataset, **kwargs)
        self.latest_indices = None
        self._active = False
        self.consumed_batches = self.consumed_images = 0
        self.unconsumed_indices = ()

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        if self._active:
            raise ValueError('Concurrent observed loader iterators are prohibited')
        self._active = True
        observer = self._observer
        observer.pending.clear()
        observer.scheduled_batches = observer.scheduled_images = 0
        self.consumed_batches = self.consumed_images = 0
        self.unconsumed_indices = ()
        iterator = None
        try:
            iterator = iter(self._loader)
            for batch in iterator:
                if not observer.pending:
                    raise ValueError('Returned batch has no observed indices')
                self.latest_indices = observer.pending.popleft()
                self.consumed_batches += 1
                self.consumed_images += len(self.latest_indices)
                yield batch
        finally:
            self.unconsumed_indices = tuple(observer.pending)
            observer.pending.clear()
            self._active = False
            # Dropping the nonpersistent iterator releases its worker resources.
            # No private PyTorch shutdown API and no reconstruction of order.
            del iterator

    def observation(self):
        pending = tuple(self._observer.pending) if self._active else self.unconsumed_indices
        return dict(scheduled_batches=self._observer.scheduled_batches,
                    scheduled_images=self._observer.scheduled_images,
                    consumed_batches=self.consumed_batches, consumed_images=self.consumed_images,
                    unconsumed_scheduled_batches=len(pending),
                    unconsumed_scheduled_images=sum(len(b) for b in pending))
