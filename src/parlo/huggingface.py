"""Optional Hugging Face adapters with relative paths and no Audio decoder."""
from .corpus import iter_rows


def to_huggingface(query, *, iterable=False, shuffle=None, seed=42, buffer_size=1024, batch_size=256):
    try:
        from datasets import Dataset, IterableDataset
    except ImportError as exc:
        raise ImportError('Install parlo-loader[huggingface] to use this adapter') from exc
    if iterable:
        def generate():
            yield from iter_rows(query, shuffle=shuffle, seed=seed, buffer_size=buffer_size, batch_size=batch_size)
        return IterableDataset.from_generator(generate)
    if shuffle is None:
        return Dataset(query.collect().to_arrow())
    return Dataset.from_list(list(iter_rows(query, shuffle=shuffle, seed=seed, buffer_size=buffer_size, batch_size=batch_size)))
