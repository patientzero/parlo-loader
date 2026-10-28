"""Participant assignments are created before expansion to task recordings."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import warnings
import polars as pl


@dataclass
class SplitManifest:
    assignments: dict[str, list[str]]
    seed: int | None = 42
    strategy: str = 'stratified'

    def __post_init__(self):
        if not isinstance(self.assignments, dict) or not self.assignments:
            raise ValueError('Split manifest needs named partitions')
        seen = set()
        for name, ids in self.assignments.items():
            if not isinstance(name, str) or not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise ValueError('Partitions must map names to lists of string participant IDs')
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError('Participant leakage: duplicate IDs within or across partitions')
            seen.update(ids)

    def save(self, path):
        self.__post_init__()
        Path(path).write_text(json.dumps({'version': 1, 'strategy': self.strategy, 'seed': self.seed,
                                        'assignments': self.assignments}, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('version') != 1:
            raise ValueError('Unsupported split manifest version')
        return cls(data['assignments'], data.get('seed'), data['strategy'])

    def apply(self, query, split, *, strict=True):
        """Filter any participant-keyed LazyFrame; absent task participants are fine.

        strict=True rejects query IDs missing from the entire manifest so a changed
        cohort is never silently assigned or dropped. Manifest IDs absent from a
        task-specific query are expected and allowed.
        """
        self.__post_init__()
        ids = self.assignments[split]
        if strict:
            known = [i for values in self.assignments.values() for i in values]
            unknown = query.select('participant_id').unique().filter(~pl.col('participant_id').is_in(known)).collect()
            if unknown.height:
                raise ValueError(f'{unknown.height} query participants are absent from the saved manifest; regenerate it or use strict=False')
        return query.filter(pl.col('participant_id').is_in(ids))


def _groups(participants):
    if 'has_recordings' in participants.collect_schema():
        participants = participants.filter(pl.col('has_recordings'))
    frame = participants.select('participant_id', 'diagnosis').collect().sort('participant_id')
    if frame.is_empty():
        raise ValueError('No eligible participants; broaden the selected tasks/groups')
    if frame['participant_id'].n_unique() != frame.height:
        raise ValueError('Expected one diagnosis per unique participant')
    groups = {}
    for pid, label in frame.iter_rows():
        if label not in {'HC', 'MCI', 'AD'}:
            raise ValueError(f'Participant {pid} has unknown diagnosis {label!r}')
        groups.setdefault(label, []).append(pid)
    return groups


def stratified_split(participants, *, proportions=(.7, .15, .15), seed=42):
    if len(proportions) != 3 or any(not math.isfinite(p) or p < 0 for p in proportions) or not math.isclose(sum(proportions), 1):
        raise ValueError('Provide three nonnegative proportions summing to one (train, validation, test)')
    active = [i for i, p in enumerate(proportions) if p > 0]
    assignments = {k: [] for k in ('train', 'validation', 'test')}
    rng = random.Random(seed)
    for diagnosis, original in sorted(_groups(participants).items()):
        ids = original.copy()
        n = len(ids)
        if n < len(active):
            raise ValueError(f'{diagnosis} has {n} participants; need at least {len(active)} for nonempty stratified partitions. Broaden the cohort or use fewer nonzero partitions.')
        ideal = [n * p for p in proportions]
        counts = [max(1, math.floor(ideal[i])) if i in active else 0 for i in range(3)]
        while sum(counts) < n:
            i = max(active, key=lambda i: (ideal[i] - counts[i], -i))
            counts[i] += 1
        while sum(counts) > n:
            i = max((i for i in active if counts[i] > 1), key=lambda i: (counts[i] - ideal[i], -i))
            counts[i] -= 1
        rng.shuffle(ids)
        offset = 0
        for name, count in zip(assignments, counts):
            assignments[name].extend(ids[offset:offset + count])
            offset += count
    return SplitManifest({k: sorted(v) for k, v in assignments.items()}, seed)


def stratified_kfold(participants, *, folds=5, seed=42):
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError('folds must be an integer >= 2')
    groups = _groups(participants)
    for diagnosis, ids in groups.items():
        if len(ids) < folds:
            raise ValueError(f'{diagnosis} has {len(ids)} participants; {folds} stratified folds require at least {folds}. Reduce folds or broaden the cohort.')
    rng = random.Random(seed)
    test_ids = [[] for _ in range(folds)]
    all_ids = set()
    for _, original in sorted(groups.items()):
        ids = original.copy()
        rng.shuffle(ids)
        all_ids.update(ids)
        for i, pid in enumerate(ids):
            test_ids[i % folds].append(pid)
    return [SplitManifest({'train': sorted(all_ids - set(test)), 'test': sorted(test)}, seed, f'kfold:{i}/{folds}')
            for i, test in enumerate(test_ids)]


def leave_one_out(participants):
    ids = sorted(i for values in _groups(participants).values() for i in values)
    for pid in ids:
        yield SplitManifest({'train': [i for i in ids if i != pid], 'test': [pid]}, None, 'leave-one-participant-out')


@dataclass
class TargetView:
    participants: pl.LazyFrame
    mapping: dict
    exclusions: pl.LazyFrame
    source: object
    target: str

    def expand(self, table='recordings'):
        """Join participant targets to recordings, utterances, or words lazily."""
        if table not in ('recordings', 'utterances', 'words'):
            raise ValueError('table must be recordings, utterances, or words')
        return getattr(self.source, table).join(self.participants.select('participant_id', 'target'), on='participant_id', how='inner', maintain_order='left')

    def split(self, **kwargs):
        return stratified_split(self.participants, **kwargs)


def target_view(view, target='diagnosis', *, labels=None):
    participants = view.participants
    if 'has_recordings' in participants.collect_schema():
        participants = participants.filter(pl.col('has_recordings'))
    empty = participants.limit(0).select('participant_id').with_columns(pl.lit('', dtype=pl.String).alias('reason'))
    if target in {'mmse', 'cdr'}:
        if labels is not None:
            raise ValueError('labels is only supported for diagnosis classification')
        column = f'{target}_value'
        if column not in participants.collect_schema():
            value = pl.lit(None, dtype=pl.Float64)
        else:
            value = pl.col(column).str.replace(',', '.').cast(pl.Float64, strict=False)
        query = participants.with_columns(value.alias('target'))
        valid = pl.col('target').is_not_null() & pl.col('target').is_finite()
        excluded = query.filter(~valid.fill_null(False)).select('participant_id').with_columns(pl.lit(f'missing_or_invalid_{target}').alias('reason'))
        count = excluded.select(pl.len()).collect().item()
        if count:
            warnings.warn(f'Excluded {count} participants with missing/invalid {target.upper()} from this target view only', UserWarning, stacklevel=2)
        return TargetView(query.filter(valid), {}, excluded, view, target)
    if target == 'diagnosis':
        labels = ['HC', 'MCI', 'AD'] if labels is None else [label.upper() for label in labels]
        if len(labels) < 2 or len(set(labels)) != len(labels) or set(labels) - {'HC', 'MCI', 'AD'}:
            raise ValueError('labels must contain two or three distinct labels from HC, MCI, AD')
        mapping = {label: i for i, label in enumerate(labels)}
    elif target == 'hc_vs_impaired':
        if labels is not None:
            raise ValueError('labels is only supported for diagnosis classification')
        mapping = {'HC': 0, 'MCI': 1, 'AD': 1}
    else:
        raise ValueError("target must be diagnosis, hc_vs_impaired, mmse, or cdr")
    excluded = participants.filter(~pl.col('diagnosis').is_in(list(mapping))).select('participant_id').with_columns(pl.lit('outside_requested_labels').alias('reason'))
    query = participants.filter(pl.col('diagnosis').is_in(list(mapping))).with_columns(
        pl.col('diagnosis').replace_strict(mapping, return_dtype=pl.Int64).alias('target'))
    return TargetView(query, mapping, excluded if labels else empty, view, target)
