import json
import os
from pathlib import Path
import polars as pl
import pytest
import soundfile as sf
from parlo import Corpus, iter_rows


def test_lazy_filtering_metadata_and_audio_never_decoded(corpus_root, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Audio decoded during metadata work')
    monkeypatch.setattr(sf.SoundFile, 'read', forbidden)
    c = Corpus(corpus_root)
    assert c.cache_status()['status'] == 'missing'
    assert isinstance(c.recordings, pl.LazyFrame)
    assert c.recordings.collect().height == 33
    assert c.participants.collect().height == 18
    assert c.filter(task=['reading', 'recall'], diagnosis='hc').recordings.collect().height == 11
    assert c.participant('HC1').recordings.collect().height == 2
    assert c.complete_cases(['reading', 'recall']).participants.collect().height == 15
    assert c.complete_cases(['reading', 'unknown']).participants.collect().height == 0
    assert c.participants.select('custom').collect().item(0,0) == '001'
    assert 'Synthetic' in c.metadata_description
    assert c.utterances.filter(~pl.col('is_participant')).collect().height == 33
    assert c.words.collect().height > 33
    for mode in [None, 'index', 'buffer']:
        first = list(c.iter_rows(shuffle=mode, seed=7, buffer_size=3))
        assert first == list(c.iter_rows(shuffle=mode, seed=7, buffer_size=3))
        assert len({r['recording_id'] for r in first}) == 33
    for frame in c.iter_batches(batch_size=4):
        assert frame.height <= 4


def test_cache_reuse_invalidation_and_hashes(corpus_root, monkeypatch):
    c = Corpus(corpus_root)
    c.refresh(content_hash=True)
    generation = c.cache_status()['generation']
    assert c.cache_status(verify_hashes=True)['hashes_verified']
    other = Corpus(corpus_root)
    monkeypatch.setattr(other, 'refresh', lambda **kwargs: pytest.fail('Cache rebuilt'))
    assert other.recordings.collect().height == 33
    assert other.cache_status()['generation'] == generation
    path = next(corpus_root.rglob('*.CHA'))
    before = path.stat()
    path.write_text(path.read_text().replace('komme', 'gehee'))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert c.cache_status()['status'] == 'valid'
    assert c.cache_status(verify_hashes=True)['status'] == 'stale'
    c.refresh()
    assert c.cache_status(verify_hashes=True)['status'] == 'unverified'
    path.write_text(path.read_text() + '\n')
    assert c.cache_status()['status'] == 'stale'
    assert Corpus(corpus_root).recordings.collect().height == 33


def test_atomic_refresh_failure_keeps_committed_manifest(corpus_root, monkeypatch):
    c=Corpus(corpus_root)
    c.refresh()
    old=(c.cache_dir/'manifest.json').read_bytes()
    def fail(*args, **kwargs):
        raise OSError('simulated write failure')
    monkeypatch.setattr(pl.DataFrame, 'write_parquet', fail)
    with pytest.raises(OSError):
        c.refresh()
    assert (c.cache_dir/'manifest.json').read_bytes() == old
    assert c.recordings.collect().height == 33


def test_pair_and_metadata_validation(corpus_root):
    missing = next(corpus_root.rglob('*.WAV'))
    missing.unlink()
    (corpus_root/'hc'/'orphan.wav').write_bytes(b'bad')
    with pytest.warns(UserWarning, match='validation'):
        c = Corpus(corpus_root)
        c.recordings.collect()
    assert c.validation_report.filter(pl.col('code') == 'incomplete_pair').collect().height == 2
    csv = corpus_root/'0metadata.csv'
    csv.write_text(csv.read_text().replace('HC1,control', 'HC1,patient_ad'))
    with pytest.warns(UserWarning):
        c.refresh()
    assert c.validation_report.filter(pl.col('code') == 'diagnosis_conflict').collect().height == 2
    assert c.participant('HC1').recordings.collect().height == 0


def test_placeholders_and_asr(corpus_root):
    path=next(corpus_root.rglob('*.CHA'))
    path.write_text('@Begin\n@Media:\tx, audio, notrans\n*CHI:\txxx .\n@End\n')
    c=Corpus(corpus_root)
    assert c.recordings.filter(pl.col('transcript_status')=='placeholder').collect().height == 1
    asr=c.asr().collect()
    assert asr.height == 32
    assert set(asr['reference']) == {'ähm ich ich komme'}
    assert set(asr['reference_verbatim']) == {'Ähm ich ich komme .'}
    assert c.asr('recording').collect().height == 32
    assert c.asr('recording').collect()['reference'][0].startswith('bitte erzählen')
    assert 'xxx' not in ' '.join(asr['reference'])


def test_cache_override_and_relative_paths(corpus_root, tmp_path):
    c=Corpus(corpus_root, cache_dir=tmp_path/'custom-cache')
    assert all(not Path(p).is_absolute() for p in c.recordings.collect()['audio_path'])
    assert (tmp_path/'custom-cache'/'manifest.json').exists()
    assert not (corpus_root/'.parlo-cache').exists()


def test_invalid_timing_stays_visible(corpus_root):
    path=next(corpus_root.rglob('*.CHA'))
    path.write_text(path.read_text().replace('100_900', '100_1900'))
    c=Corpus(corpus_root)
    with pytest.warns(UserWarning):
        turns=c.utterances.filter(pl.col('timing_status')=='invalid').collect()
    assert turns.height == 1 and turns['end_ms'][0] == 1900
    with pytest.raises(ValueError, match='invalid timing'):
        c.load_audio(turns.row(0,named=True))
    assert c.asr().collect().height == 32


def test_shuffle_validation():
    q=pl.LazyFrame({'a':list(range(30))})
    a=list(iter_rows(q,shuffle='index',seed=1))
    assert a != list(iter_rows(q,shuffle='index',seed=2))
    assert list(iter_rows(q,shuffle='buffer',buffer_size=1)) == list(iter_rows(q))
    with pytest.raises(ValueError):
        list(iter_rows(q,shuffle='invalid'))


def test_duplicate_metadata_and_unreadable_audio(corpus_root):
    path=corpus_root/'0metadata.csv'
    lines=path.read_text().splitlines()
    path.write_text('\n'.join(lines+[lines[1]])+'\n')
    next(corpus_root.rglob('HC1_reading.WAV')).write_bytes(b'not a WAV')
    c=Corpus(corpus_root)
    with pytest.warns(UserWarning):
        records=c.recordings.collect()
    assert records.height==31
    codes=c.validation_report.collect()['code'].to_list()
    assert 'duplicate_metadata' in codes and 'unreadable_audio' in codes


def test_missing_metadata_and_unreadable_transcript(corpus_root):
    path=corpus_root/'0metadata.csv'
    lines=path.read_text().splitlines()
    path.write_text('\n'.join([lines[0]]+lines[2:])+'\n')
    next(corpus_root.rglob('HC1_reading.CHA')).write_bytes(b'\xff')
    c=Corpus(corpus_root)
    with pytest.warns(UserWarning):
        assert c.recordings.collect().height==33
    assert c.participant('HC0').participants.collect()['diagnosis'].to_list()==['HC']
    codes=c.validation_report.collect()['code'].to_list()
    assert 'missing_metadata' in codes and 'unreadable_transcript' in codes
    assert c.asr('recording').collect().height==32


def test_missing_tables_and_versions_invalidate_cache(corpus_root):
    c=Corpus(corpus_root)
    c.refresh()
    manifest_path=c.cache_dir/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['parser_version']='obsolete'
    manifest_path.write_text(json.dumps(manifest))
    assert c.cache_status()['status']=='stale'
    c.refresh()
    (c.cache_dir/c._manifest['tables']['words']).unlink()
    assert c.cache_status()['status']=='stale'
    assert Corpus(corpus_root).words.collect().height>0
