"""Print an aggregate audit without decoding audio or exporting corpus text."""
import argparse
import json
import polars as pl
from parlo import Corpus


def audit(root):
    corpus = Corpus(root)
    tasks = corpus.recordings.select('task').unique().collect().to_series().to_list()
    result = {
        'participants': corpus.participants.group_by('diagnosis').len(name='participants').sort('diagnosis').collect().to_dicts(),
        'recordings': corpus.recordings.group_by('task').agg(
            pl.len().alias('recordings'),
            (pl.col('transcript_status') == 'placeholder').sum().alias('placeholders'),
            pl.col('valid_reference').sum().alias('asr_recordings')).sort('task').collect().to_dicts(),
        'complete_case_participants': corpus.complete_cases(tasks).participants.select(pl.len()).collect().item(),
        'asr_utterances': corpus.asr().select(pl.len()).collect().item(),
        'utterances': corpus.utterances.select(pl.len()).collect().item(),
        'words': corpus.words.select(pl.len()).collect().item(),
        'duration_hours': corpus.recordings.select(pl.col('duration_ms').sum() / 3600000).collect().item(),
        'issues': corpus.validation_report.group_by('code').len(name='issues').sort('code').collect().to_dicts(),
        'missing_targets': {name: corpus.targets(name).exclusions.select(pl.len()).collect().item() for name in ['mmse', 'cdr']},
        'split_participants': {k: len(v) for k, v in corpus.split().assignments.items()},
        'cache_status': corpus.cache_status(),
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus_root')
    print(json.dumps(audit(parser.parse_args().corpus_root), indent=2))
