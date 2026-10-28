"""Optional PyTorch adapters. Indexing never decodes audio."""
try:
    from torch.utils.data import Dataset, IterableDataset, get_worker_info
except ImportError as exc:
    raise ImportError('Install parlo-loader[torch] to use parlo.torch') from exc
from .corpus import iter_rows


class ParloDataset(Dataset):
    def __init__(self, query):
        self.frame = query.collect()

    def __len__(self):
        return self.frame.height

    def __getitem__(self, index):
        return self.frame.row(index, named=True)


class ParloIterableDataset(IterableDataset):
    def __init__(self, query, *, shuffle=None, seed=42, buffer_size=1024, batch_size=256):
        self.query = query
        self.options = dict(shuffle=shuffle, seed=seed, buffer_size=buffer_size, batch_size=batch_size)

    def __iter__(self):
        worker = get_worker_info()
        for i, row in enumerate(iter_rows(self.query, **self.options)):
            if worker is None or i % worker.num_workers == worker.id:
                yield row
