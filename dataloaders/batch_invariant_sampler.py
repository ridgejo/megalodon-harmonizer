import random


class BatchInvariantSampler:
    """
    Provides batches randomly selected from multiple dataloaders. Ensures that nothing within a batch from
    one dataloader is ever mixed with data from another dataloader.
    """

    def __init__(self, dataloaders, shuffle=True):
        self.dataloaders = dataloaders
        self.data_sizes = [len(dataloader) for dataloader in dataloaders]
        self.data_len = sum(self.data_sizes)
        self.batch_order = []
        self.shuffle = shuffle

        # If not randomly shuffling on each loop, define a random order on instantiation
        if not self.shuffle:
            self._reset()
            self.fixed_batch_order = (
                self.batch_order.copy()
            )  # Store a copy of this defined order

    def __iter__(self):
        if self.shuffle:
            self._reset()  # Generate a new random batch sampling order
        else:
            self.batch_order = self.fixed_batch_order.copy()  # Use the defined order
        return self

    def __next__(self):
        # Return next sample from randomly selected iterable
        if not self.batch_order:
            raise StopIteration
        dl_idx = self.batch_order.pop(0)
        return next(self.dataloader_iters[dl_idx])

    def __len__(self):
        return self.data_len

    def _generate_batch_order(self):
        batch_order = []
        for i, data_size in enumerate(self.data_sizes):
            batch_order.extend([i for _ in range(data_size)])
        random.shuffle(batch_order)
        return batch_order

    def _reset(self):
        self.batch_order = self._generate_batch_order()
        self.dataloader_iters = [iter(dataloader) for dataloader in self.dataloaders]
