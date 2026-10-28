"""Lazy corpus views and transactional, source-aware Parquet caches."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import uuid
import warnings

import polars as pl
import soundfile as sf

from .audio import load_audio
from .chat import parse_chat, normalize_text

PARSER_VERSION = '3'
SCHEMA_VERSION = '2'
DIAGNOSES = {'control': 'HC', 'patient_mci': 'MCI', 'patient_ad': 'AD', 'hc': 'HC', 'mci': 'MCI', 'ad': 'AD'}
COMMON = {'recording_id': pl.String, 'participant_id': pl.String, 'task': pl.String, 'diagnosis': pl.String}
UTTERANCE = {**COMMON, 'utterance_id': pl.String, 'utterance_index': pl.Int64, 'speaker': pl.String,
             'is_participant': pl.Boolean, 'raw_text': pl.String, 'text': pl.String,
             'start_ms': pl.Int64, 'end_ms': pl.Int64, 'timing_status': pl.String,
             'is_placeholder': pl.Boolean, 'valid_reference': pl.Boolean, 'dependent_tiers': pl.String, 'line': pl.Int64}
WORD = {**COMMON, 'utterance_id': pl.String, 'utterance_index': pl.Int64, 'word_index': pl.Int64,
        'speaker': pl.String, 'is_participant': pl.Boolean, 'raw_text': pl.String, 'text': pl.String,
        'start_ms': pl.Int64, 'end_ms': pl.Int64, 'timing_status': pl.String}
RECORDING = {**COMMON, 'audio_path': pl.String, 'transcript_path': pl.String, 'duration_ms': pl.Float64,
             'sample_rate': pl.Int64, 'channels': pl.Int64, 'frames': pl.Int64, 'headers': pl.String,
             'raw_transcript': pl.String, 'transcript_status': pl.String, 'valid_reference': pl.Boolean,
             'text': pl.String, 'all_speakers_text': pl.String}
ISSUE = {'severity': pl.String, 'code': pl.String, 'path': pl.String, 'detail': pl.String}


def _values(value):
    return [value] if isinstance(value, (str, int)) else list(value)


def iter_batches(query, batch_size=256):
    """Stream Polars batches without decoding any audio."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    batches = query.collect_batches(chunk_size=batch_size, engine='streaming', lazy=True)
    try:
        yield from batches
    finally:
        stop = getattr(batches, 'stop', None)
        if stop is not None:
            stop()


def iter_rows(query, *, batch_size=256, shuffle=None, seed=42, buffer_size=1024):
    """shuffle=None, 'index' (materialize rows), or 'buffer' (bounded stream)."""
    if shuffle not in (None, 'index', 'buffer'):
        raise ValueError("shuffle must be None, 'index', or 'buffer'")
    rng = random.Random(seed)
    if shuffle == 'index':
        frame = query.collect()
        indices = list(range(frame.height))
        rng.shuffle(indices)
        for i in indices:
            yield frame.row(i, named=True)
        return
    rows = (row for batch in iter_batches(query, batch_size) for row in batch.iter_rows(named=True))
    if shuffle is None:
        yield from rows
        return
    if buffer_size < 1:
        raise ValueError('buffer_size must be positive')
    buffer = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
        else:
            i = rng.randrange(len(buffer))
            yield buffer[i]
            buffer[i] = row
    rng.shuffle(buffer)
    yield from buffer


class CorpusView:
    """Filters propagate by recording ID to all levels; tasks stay available by default."""
    def __init__(self, corpus, recordings):
        self.corpus = corpus
        self.recordings = recordings

    @property
    def participants(self):
        return self.corpus.participants.join(self.recordings.select('participant_id').unique(), on='participant_id', how='semi', maintain_order='left')

    @property
    def utterances(self):
        return self.corpus.utterances.join(self.recordings.select('recording_id'), on='recording_id', how='semi', maintain_order='left')

    @property
    def words(self):
        return self.corpus.words.join(self.recordings.select('recording_id'), on='recording_id', how='semi', maintain_order='left')

    def filter(self, *, task=None, diagnosis=None, participant=None):
        query = self.recordings
        for column, value in [('task', task), ('diagnosis', diagnosis), ('participant_id', participant)]:
            if value is not None:
                values = _values(value)
                values = [str(v).upper() if column == 'diagnosis' else str(v) for v in values]
                query = query.filter(pl.col(column).is_in(values))
        return CorpusView(self.corpus, query)

    def participant(self, participant_id):
        return self.filter(participant=str(participant_id))

    def complete_cases(self, tasks):
        tasks = sorted(set(_values(tasks)))
        if not tasks:
            raise ValueError('Specify at least one required task')
        ids = (self.recordings.filter(pl.col('task').is_in(tasks)).group_by('participant_id')
               .agg(pl.col('task').n_unique().alias('n')).filter(pl.col('n') == len(tasks)).select('participant_id'))
        return CorpusView(self.corpus, self.recordings.join(ids, on='participant_id', how='semi', maintain_order='left'))

    def asr(self, level='utterance', *, normalize=True, lowercase=True, punctuation=True):
        if level not in ('recording', 'utterance'):
            raise ValueError("level must be 'recording' or 'utterance'")
        if level == 'utterance':
            query = self.utterances.filter(pl.col('valid_reference') & (pl.col('timing_status') == 'valid'))
            query = query.join(self.recordings.select('recording_id', 'audio_path'), on='recording_id', maintain_order='left')
        else:
            query = self.recordings.filter(pl.col('valid_reference')).with_columns(
                pl.col('all_speakers_text').alias('text'),
                pl.lit(None, dtype=pl.Int64).alias('start_ms'), pl.lit(None, dtype=pl.Int64).alias('end_ms'))
        query = query.with_columns(pl.col('text').alias('reference_verbatim'))
        ref = pl.col('text')
        if normalize:
            ref = ref.map_elements(lambda t: normalize_text(t, lowercase=lowercase, punctuation=punctuation), return_dtype=pl.String)
        return query.with_columns(ref.alias('reference'))

    def targets(self, target='diagnosis', *, labels=None):
        from .evaluation import target_view
        return target_view(self, target, labels=labels)

    def split(self, *, proportions=(.7, .15, .15), seed=42):
        from .evaluation import stratified_split
        return stratified_split(self.participants, proportions=proportions, seed=seed)

    def kfold(self, *, folds=5, seed=42):
        from .evaluation import stratified_kfold
        return stratified_kfold(self.participants, folds=folds, seed=seed)

    def leave_one_out(self):
        from .evaluation import leave_one_out
        return leave_one_out(self.participants)

    def iter_batches(self, table='recordings', **kwargs):
        return iter_batches(getattr(self, table), **kwargs)

    def iter_rows(self, table='recordings', **kwargs):
        return iter_rows(getattr(self, table), **kwargs)

    def load_audio(self, example, **kwargs):
        return self.corpus.load_audio(example, **kwargs)


class Corpus(CorpusView):
    def __init__(self, root, cache_dir=None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f'Corpus directory does not exist: {self.root}')
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else self.root / '.parlo-cache'
        self.corpus = self
        self._manifest = None

    def _sources(self, hashes=False):
        entries = []
        directories = sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.upper() in {'AD', 'MCI', 'HC'})
        names = [p.name.upper() for p in directories]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate diagnosis directories differing only in case')
        if set(names) != {'AD', 'MCI', 'HC'}:
            raise ValueError('Corpus root must contain AD, MCI, HC directories (case insensitive)')
        files = [self.root / '0metadata.csv', self.root / '0metadata.cdc']
        files += [p for d in directories for p in d.rglob('*') if p.is_file() and p.suffix.lower() in {'.cha', '.wav'}]
        for path in sorted(files):
            stat = path.stat()
            entry = {'path': path.relative_to(self.root).as_posix(), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if hashes:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                entry['sha256'] = digest.hexdigest()
            entries.append(entry)
        return entries

    def cache_status(self, *, verify_hashes=False):
        path = self.cache_dir / 'manifest.json'
        if not path.exists():
            return {'status': 'missing', 'reason': 'No manifest'}
        try:
            manifest = json.loads(path.read_text())
            if manifest['parser_version'] != PARSER_VERSION or manifest['schema_version'] != SCHEMA_VERSION:
                return {'status': 'stale', 'reason': 'Parser/schema version changed'}
            if set(manifest['tables']) != {'participants', 'recordings', 'utterances', 'words', 'validation'}:
                return {'status': 'stale', 'reason': 'Incomplete table manifest'}
            if any(not (self.cache_dir / p).is_file() for p in manifest['tables'].values()):
                return {'status': 'stale', 'reason': 'Missing Parquet table'}
            sources = self._sources(hashes=verify_hashes)
            old = manifest['sources']
            keys = ['path', 'size', 'mtime_ns']
            if [{k: r[k] for k in keys} for r in sources] != [{k: r[k] for k in keys} for r in old]:
                return {'status': 'stale', 'reason': 'Source paths, sizes, or modification times changed'}
            if verify_hashes:
                if not all('sha256' in r for r in old):
                    return {'status': 'unverified', 'reason': 'Refresh with content_hash=True to establish hashes'}
                if sources != old:
                    return {'status': 'stale', 'reason': 'Source content hashes changed'}
            return {'status': 'valid', 'generation': manifest['generation'], 'hashes_verified': verify_hashes}
        except (ValueError, KeyError, OSError) as exc:
            return {'status': 'stale', 'reason': str(exc)}

    def _ensure(self):
        # A Corpus instance is a snapshot; call refresh() to observe later changes.
        if self._manifest is None:
            if self.cache_status()['status'] != 'valid':
                self.refresh()
            else:
                self._manifest = json.loads((self.cache_dir / 'manifest.json').read_text())
                self._warn()

    def _warn(self):
        n = self._manifest.get('issue_count', 0)
        if n:
            warnings.warn(f'PARLO validation reported {n} issues; inspect corpus.validation_report', UserWarning, stacklevel=3)

    def _table(self, name):
        self._ensure()
        return pl.scan_parquet(self.cache_dir / self._manifest['tables'][name])

    @property
    def participants(self):
        return self._table('participants')

    @property
    def recordings(self):
        return self._table('recordings')

    @property
    def utterances(self):
        return self._table('utterances')

    @property
    def words(self):
        return self._table('words')

    @property
    def validation_report(self):
        return self._table('validation')

    @property
    def metadata_description(self):
        self._ensure()
        return self._manifest['metadata_description']

    def refresh(self, *, content_hash=False):
        sources = self._sources(hashes=content_hash)
        issues, recordings, utterances, words = [], [], [], []
        def report(code, path, detail, severity='warning'):
            issues.append(dict(severity=severity, code=code, path=str(path), detail=detail))
        with (self.root / '0metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if 'patient_id' not in fields:
                raise ValueError('0metadata.csv must have a patient_id column')
            metadata, ambiguous = {}, set()
            for row in reader:
                pid = (row.get('patient_id') or '').strip()
                if not pid or None in row:
                    report('malformed_metadata', '0metadata.csv', 'Missing ID or extra fields')
                    continue
                if pid in metadata:
                    ambiguous.add(pid)
                    report('duplicate_metadata', '0metadata.csv', f'Duplicate participant {pid}; excluding associated recordings')
                metadata[pid] = row
        participants = {}
        for pid, row in metadata.items():
            diagnosis = DIAGNOSES.get((row.get('type') or '').lower())
            participants[pid] = {**{k: v for k, v in row.items() if k is not None}, 'participant_id': pid,
                                 'diagnosis': diagnosis, 'metadata_json': json.dumps(row, ensure_ascii=False),
                                 'has_recordings': False}
        pairs = {}
        for item in sources:
            path = Path(item['path'])
            if path.suffix.lower() not in {'.wav', '.cha'}:
                continue
            key = path.with_suffix('').as_posix().lower()
            pair = pairs.setdefault(key, {})
            suffix = path.suffix.lower()
            if suffix in pair:
                report('duplicate_source', path, 'Ambiguous case-insensitive source pair; excluded')
                pair['ambiguous'] = True
            pair[suffix] = path
        for key, pair in sorted(pairs.items()):
            if pair.get('ambiguous'):
                continue
            if '.wav' not in pair or '.cha' not in pair:
                report('incomplete_pair', key, 'Missing WAV or CHAT counterpart; excluded')
                continue
            wav, cha = pair['.wav'], pair['.cha']
            pid, sep, task = wav.stem.partition('_')
            if not sep or not task:
                report('invalid_filename', wav, 'Expected participant_task.wav; excluded')
                continue
            diagnosis = wav.parts[0].upper()
            if pid in ambiguous:
                continue
            if pid not in participants:
                report('missing_metadata', wav, f'No metadata for participant {pid}')
                participants[pid] = {**{f: None for f in fields}, 'participant_id': pid, 'diagnosis': diagnosis,
                                     'metadata_json': '{}', 'has_recordings': False}
            p = participants[pid]
            if p['diagnosis'] and p['diagnosis'] != diagnosis:
                report('diagnosis_conflict', wav, f'Metadata/group conflict for {pid}; excluded')
                continue
            if not p['diagnosis']:
                report('unknown_metadata_diagnosis', wav, f'Using directory diagnosis for {pid}')
                p['diagnosis'] = diagnosis
            try:
                info = sf.info(str(self.root / wav))  # header only; no sample decoding
                if info.frames <= 0:
                    raise ValueError('Empty audio')
            except (RuntimeError, OSError, ValueError) as exc:
                report('unreadable_audio', wav, str(exc), 'error')
                continue
            try:
                with (self.root / cha).open(encoding='utf-8-sig', newline='') as stream:
                    raw = stream.read()
                headers, turns, tokens, errors = parse_chat(raw, info.duration * 1000)
            except (UnicodeError, OSError) as exc:
                report('unreadable_transcript', cha, str(exc), 'error')
                raw, headers, turns, tokens, errors = '', [], [], [], []
            for code, detail in errors:
                report(code, cha, detail)
            for h in headers:
                if h['label'] == '@ID':
                    parts = h['text'].split('|')
                    if len(parts) > 5 and parts[5] and DIAGNOSES.get(parts[5].lower(), diagnosis) != diagnosis:
                        report('chat_diagnosis_conflict', cha, f'CHAT @ID diagnosis {parts[5]} differs from {diagnosis}')
            shared = dict(recording_id=key, participant_id=pid, task=task.lower(), diagnosis=diagnosis)
            part_turns = [u for u in turns if u['is_participant']]
            valid = (bool(part_turns) and all(u['valid_reference'] for u in part_turns)
                     and all(not u['is_placeholder'] and any(c.isalpha() for c in u['text']) for u in turns)
                     and not errors)
            status = 'transcribed' if valid else ('placeholder' if any(u['is_placeholder'] for u in turns) else 'unusable')
            recordings.append({**shared, 'audio_path': wav.as_posix(), 'transcript_path': cha.as_posix(),
                               'duration_ms': info.duration * 1000, 'sample_rate': info.samplerate,
                               'channels': info.channels, 'frames': info.frames, 'headers': json.dumps(headers, ensure_ascii=False),
                               'raw_transcript': raw, 'transcript_status': status, 'valid_reference': valid,
                               'text': ' '.join(u['text'] for u in part_turns),
                               'all_speakers_text': ' '.join(u['text'] for u in turns)})
            if status == 'unusable':
                report('unusable_transcript', cha, 'Audio retained; no usable recording-level ASR reference')
            for u in turns:
                u['dependent_tiers'] = json.dumps(u['dependent_tiers'], ensure_ascii=False)
                # Structural corruption makes reference eligibility uncertain.
                if any(code in {'malformed_line', 'missing_end', 'orphan_tier'} for code, _ in errors):
                    u['valid_reference'] = False
                utterances.append({**shared, 'utterance_id': f'{key}:{u["utterance_index"]}', **u})
            words.extend({**shared, 'utterance_id': f'{key}:{w["utterance_index"]}', **w} for w in tokens)
            p['has_recordings'] = True
        for pid, p in participants.items():
            if not p['has_recordings']:
                report('metadata_without_recordings', '0metadata.csv', f'Participant {pid} has no usable paired recordings')
        participant_schema = {**{f: pl.String for f in fields}, 'participant_id': pl.String, 'diagnosis': pl.String,
                              'metadata_json': pl.String, 'has_recordings': pl.Boolean}
        tables = {'participants': pl.DataFrame(list(participants.values()), schema=participant_schema),
                  'recordings': pl.DataFrame(recordings, schema=RECORDING), 'utterances': pl.DataFrame(utterances, schema=UTTERANCE),
                  'words': pl.DataFrame(words, schema=WORD), 'validation': pl.DataFrame(issues, schema=ISSUE)}
        # Never commit a snapshot assembled while sources are changing.
        if self._sources(hashes=content_hash) != sources:
            raise RuntimeError('Corpus changed during refresh; retry after source writes finish')
        generation = uuid.uuid4().hex
        folder = self.cache_dir / f'v{SCHEMA_VERSION}-{generation}'
        folder.mkdir(parents=True)
        paths = {}
        for name, frame in tables.items():
            target = folder / f'{name}.parquet'
            frame.write_parquet(target)
            paths[name] = target.relative_to(self.cache_dir).as_posix()
        manifest = dict(parser_version=PARSER_VERSION, schema_version=SCHEMA_VERSION, generation=generation,
                        sources=sources, tables=paths, issue_count=len(issues),
                        metadata_description=(self.root / '0metadata.cdc').read_text(encoding='utf-8-sig'))
        pending = self.cache_dir / f'.manifest-{generation}.tmp'
        with pending.open('w') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, self.cache_dir / 'manifest.json')
        self._manifest = manifest
        self._warn()
        return self.cache_status()

    def load_audio(self, example, **kwargs):
        if isinstance(example, str):
            rows = self.recordings.filter(pl.col('recording_id') == example).collect()
            if rows.height != 1:
                raise KeyError(f'Unknown recording ID: {example}')
            example = rows.row(0, named=True)
        if 'audio_path' not in example:
            rec = self.recordings.filter(pl.col('recording_id') == example['recording_id']).collect()
            if rec.height != 1:
                raise KeyError(example['recording_id'])
            example = {**rec.row(0, named=True), **example}
        if 'utterance_id' in example:
            if example.get('timing_status', 'valid') != 'valid' or example.get('start_ms') is None or example.get('end_ms') is None:
                raise ValueError('Cannot slice an utterance/word with missing or invalid timing')
        path = (self.root / example['audio_path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Audio path escapes corpus root')
        for boundary in ('start_ms', 'end_ms'):
            if boundary in example:
                kwargs.setdefault(boundary, example[boundary])
        return load_audio(path, **kwargs)
