import soundfile as sf
import pytest
from parlo import Corpus


def test_torch_adapters_no_decode(corpus_root,monkeypatch):
    pytest.importorskip('torch')
    from torch.utils.data import DataLoader
    from parlo.torch import ParloDataset, ParloIterableDataset
    monkeypatch.setattr(sf.SoundFile,'read',lambda *a,**k: pytest.fail('Unexpected decoding'))
    c=Corpus(corpus_root)
    query=c.split().apply(c.asr(),'train')
    dataset=ParloDataset(query)
    assert dataset[0]['audio_path'].endswith('.WAV')
    rows=list(DataLoader(dataset,batch_size=None,num_workers=0))
    streamed=list(ParloIterableDataset(query,shuffle='buffer',buffer_size=3))
    assert {r['utterance_id'] for r in streamed} == {r['utterance_id'] for r in rows}


def test_torch_worker_sharding(corpus_root,monkeypatch):
    pytest.importorskip('torch')
    from types import SimpleNamespace
    import parlo.torch as adapter
    query=Corpus(corpus_root).recordings
    chunks=[]
    for worker in range(3):
        monkeypatch.setattr(adapter,'get_worker_info',lambda worker=worker: SimpleNamespace(id=worker,num_workers=3))
        chunks.append({r['recording_id'] for r in adapter.ParloIterableDataset(query,shuffle='buffer')})
    assert len(set.union(*chunks))==33
    assert sum(map(len,chunks))==33


def test_huggingface_adapters_no_decode(corpus_root,monkeypatch):
    pytest.importorskip('datasets')
    from parlo.huggingface import to_huggingface
    monkeypatch.setattr(sf.SoundFile,'read',lambda *a,**k: pytest.fail('Unexpected decoding'))
    query=Corpus(corpus_root).asr()
    ds=to_huggingface(query)
    assert len(ds)==33 and isinstance(ds[0]['audio_path'],str)
    stream=to_huggingface(query,iterable=True,shuffle='buffer',buffer_size=3)
    assert len(list(stream))==33
    assert 'reference_verbatim' in next(iter(stream))


def test_torch_spawn_workers(corpus_root):
    pytest.importorskip('torch')
    from torch.utils.data import DataLoader
    from parlo.torch import ParloIterableDataset
    query=Corpus(corpus_root).asr().sort('utterance_id')
    dataset=ParloIterableDataset(query,shuffle='buffer',buffer_size=3)
    loader=DataLoader(dataset,batch_size=None,num_workers=2,multiprocessing_context='spawn')
    ids=[row['utterance_id'] for row in loader]
    assert len(ids)==len(set(ids))==33
