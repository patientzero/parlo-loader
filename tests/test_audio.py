import numpy as np
import pytest
from parlo import Corpus, load_audio


def test_native_slice_mono_and_resampling(corpus_root):
    c=Corpus(corpus_root)
    row=c.recordings.collect().row(0,named=True)
    full=c.load_audio(row)
    assert full.sample_rate == 8000 and full.samples.shape == (8000,2)
    excerpt=c.load_audio(row,start_ms=100,end_ms=900)
    np.testing.assert_array_equal(excerpt.samples,full.samples[800:7200])
    mono=c.load_audio(row,mono=True)
    np.testing.assert_allclose(mono.samples,full.samples.mean(axis=1))
    resampled=c.load_audio(row,mono=True,sample_rate=16000)
    assert resampled.sample_rate == 16000 and resampled.samples.shape == (16000,)
    turn=c.asr().collect().row(0,named=True)
    assert c.load_audio(turn).samples.shape == (6400,2)
    assert c.load_audio(row['recording_id']).sample_rate == 8000


@pytest.mark.parametrize('start,end', [(-1,100),(10,10),(900,800),(0,1001),(1.5,200),(False,100)])
def test_reject_invalid_boundaries(corpus_root,start,end):
    with pytest.raises(ValueError):
        load_audio(next(corpus_root.rglob('*.WAV')),start_ms=start,end_ms=end)
