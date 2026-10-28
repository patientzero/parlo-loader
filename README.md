# parlo-loader

A local Python loader for the PARLO Dementia Corpus. Import it as `parlo`.
Explore metadata and transcripts with Polars, decode audio explicitly, and split
participants before expanding to recordings or utterances. No model training,
ASR inference, or metric scoring is included.

## Installation

Python 3.10 or newer:

```bash
uv venv
uv pip install .
# Or add this package to another uv project:
uv add /absolute/path/to/parlo-loader
# Optional integrations:
uv pip install '.[notebook,torch,huggingface]'
```

The core requires Polars, NumPy, SoundFile, and SciPy. The optional extras are
independent; importing `parlo` does not import PyTorch or Hugging Face. SoundFile
wheels include libsndfile on common platforms; other platforms may need a system
libsndfile installation. The corpus is supplied separately under its own access
terms. This package neither downloads nor distributes it.

## Corpus layout and tasks

Pass the directory containing both metadata files and the diagnosis directories:

```text
corpus-root/
  0metadata.csv
  0metadata.cdc
  AD/901_reading.wav
  AD/901_reading.cha
  HC/...
  MCI/...
```

Diagnosis directory names and WAV/CHAT extensions are case insensitive. Discovery
is recursive. Matching files have the same relative stem, with a
`participant_task` filename. An incomplete or ambiguous pair is excluded with a
validation issue. The loader supports the supplied PARLO metadata schema
(`patient_id`, `type`, `mmse_value`, `cdr_value`, etc.).

The task descriptions below follow section 3.2 of Braun et al. (2026),
*The PARLO Dementia Corpus: A German Multi-Center Resource for Alzheimer's
Disease*, LREC, pp. 9581–9591. The [official corpus page](https://talkbank.org/dementia/access/German/PARLO.html)
provides the citation and paper; corpus DOI: [10.21415/29hp-z825](https://doi.org/10.21415/29hp-z825).
The paper uses **DEM** for the group stored in the download's **AD** directory.

| Task key | Elicitation |
| --- | --- |
| `reading` | Read the “Johanna Subway” story aloud. |
| `boston` | Name 15 pictured objects, a Boston Naming Test variant. |
| `animal` | Name as many animals as possible in one minute. |
| `pictdes` | Describe a mountain scene. |
| `pataka` | Repeat “pataka” for ten seconds. |
| `sischafu` | Repeat “sischafu” for ten seconds. |
| `recall` | Retell the story after intervening tasks. |
| `pictrecall` | Describe the picture from memory after intervening tasks. |

### Downloaded files are not a curated benchmark

The supplied download contains **208 participants** (83 HC, 59 MCI, 66 AD) and
**1,633 paired recordings**. Table 2 of the paper describes curated subsets; its
eight per-task counts sum to 1,591. It reports 171 participants in its all-task
subset, while 201 participants in this download have all eight file pairs.
File availability is not a reproduction of the paper's quality selection.
The exact reasons for each difference cannot be recovered from the download.

| Task | Downloaded recordings | Paper Table 2 | Placeholder CHAT files | Loader recording ASR examples |
| --- | ---: | ---: | ---: | ---: |
| `animal` | 205 | 205 | 0 | 205 |
| `boston` | 205 | 204 | 1 | 204 |
| `pataka` | 202 | 201 | 202 | 0 |
| `pictdes` | 208 | 195 | 14 | 192 |
| `pictrecall` | 204 | 200 | 2 | 202 |
| `reading` | 205 | 199 | 5 | 200 |
| `recall` | 202 | 185 | 11 | 191 |
| `sischafu` | 202 | 202 | 202 | 0 |

These are local audit results, not guarantees about other releases. The 437
placeholder CHAT files are explicitly marked; two further recordings are excluded
from recording ASR eligibility by the loader's conservative checks. An eligible
reference has passed structural checks, not a new human transcription audit.

## Lazy exploration

```python
from parlo import Corpus
import polars as pl

corpus = Corpus('/path/to/corpus-root')
corpus.participants.group_by('diagnosis').len().collect()  # participant counts
corpus.recordings.group_by('task', 'diagnosis').len().collect()  # recording counts

view = corpus.filter(task=['reading', 'recall'], diagnosis=['HC', 'MCI'])
view.participants.collect()
view.utterances.filter(pl.col('is_participant')).select('text').collect()
view.utterances.filter(~pl.col('is_participant')).collect()  # interviewer/other turns
corpus.participant('901').recordings.collect()  # all available tasks for this ID

# Explicitly require both tasks; retain all tasks already selected in the view.
complete = view.complete_cases(['reading', 'recall'])
```

Construction only validates the root directory. The first table access builds or
reuses a cache; subsequent table properties return `polars.LazyFrame` scans.
Initial indexing reads CSV/CHAT text and WAV headers into memory, but never audio
samples. Participants include metadata-only rows with `has_recordings=False`;
filtered views and evaluation helpers select participants with usable audio pairs.
Available tasks are kept by default. Complete cases mean **file availability**,
not complete valid transcripts or clinical targets.

All original CSV columns are preserved as strings, including empty values and
leading zeroes; `metadata_json` preserves the original row independently of
canonical fields. Cast demographic values explicitly, e.g.
`pl.col('age').cast(pl.Float64, strict=False)`. Canonical fields include
`participant_id`, `diagnosis`, and `has_recordings`. Original `type` labels remain.
Duplicate metadata IDs and conflicting directory/metadata diagnoses exclude the
associated recordings; missing metadata is reported and directory labels are used.
`corpus.metadata_description` preserves the complete CDC description.

Recording rows include relative paths, native sample rate/channels/frame count,
duration, full `raw_transcript`, JSON `headers`, cleaned participant `text`, and
`all_speakers_text`. Utterances retain `speaker`, `raw_text`, JSON `dependent_tiers`,
line number, participant-role flag, placeholder flag, and reference eligibility.
Repeated headers and dependent tiers remain ordered. The entire source survives
in `raw_transcript`, including annotations the parser does not interpret.

`clean_text` removes CHAT structural codes and timing markers while keeping
spoken fillers, retracings, and repetitions. It is a conservative reader for this
release, not a complete CLAN implementation. It does not reconstruct words from
replacement tiers, infer missing speech, or recover the paper's original
transcription-service documents. Review raw text for unfamiliar CHAT extensions.

## Timing and audio

```python
row = corpus.recordings.collect().row(0, named=True)
audio = corpus.load_audio(row)  # or pass row['recording_id']
print(audio.sample_rate, audio.samples.shape)
excerpt = corpus.load_audio(row, start_ms=100, end_ms=500, mono=True)
resampled = corpus.load_audio(row, mono=True, sample_rate=16_000)
```

Default samples are float32 with shape `(frames, channels)` at the native sample
rate. Explicit `mono=True` averages channels to `(frames,)`; explicit
`sample_rate=` uses polyphase resampling. No implicit resampling or decoding occurs.

Word and utterance boundaries are nullable **integer milliseconds**, relative to
the beginning of the recording. `timing_status` is `valid`, `missing`, or `invalid`.
A `%wor` bullet is associated with the immediately preceding token; tokens lacking
bullets keep null times. Without `%wor`, lexical tokens are retained with null
times; timings are never inferred from utterances. Word boundaries outside a
valid parent utterance are reported as invalid. Overlaps are preserved because
speech may overlap; no monotonicity repair is performed.

Audio slicing uses half-open sample intervals
`[floor(start_ms * sr / 1000), floor(end_ms * sr / 1000))`. Negative, reversed,
zero-length, out-of-range, and noninteger boundaries raise `ValueError`.
`None` means the recording edge for an explicit recording excerpt. An utterance
or word row with missing/invalid timing cannot be sliced; its raw values remain
available for inspection. Passing an ASR utterance row to `load_audio` uses its
validated boundaries automatically.

## Cache and validation

```python
corpus.cache_status()
corpus.validation_report.group_by('code').len().collect()
corpus.refresh()  # explicit rebuild
corpus.refresh(content_hash=True)  # opt-in: reads every source byte
corpus.cache_status(verify_hashes=True)
other = Corpus('/path/to/corpus-root', cache_dir='/path/to/local-cache')
```

The default is `<root>/.parlo-cache`. Each generation contains versioned Parquet
tables and a manifest with relative source paths, sizes, nanosecond modification
times, parser/schema versions, and optional SHA-256 digests. A new instance checks
source metadata; stale caches rebuild automatically on first table access.
`verify_hashes=True` detects same-size, same-mtime edits when a hash baseline was
established; otherwise it returns `unverified`. It reports status and does not
rebuild. `refresh(content_hash=True)` establishes or replaces the baseline.

Each live Corpus/view is a snapshot. After source edits use a new `Corpus`, or
call `refresh()` and create new views. Refresh writes a unique generation and
atomically replaces the manifest only after all tables are complete. Interrupted
writes cannot replace the previous snapshot. Old generations remain to keep
existing lazy scans usable; remove the cache directory only when no readers use
it. Sources changing during indexing cause refresh to fail with a retry message.
Caches contain corpus text/metadata and should remain local like the corpus.

Validation issues have `severity`, `code`, `path`, and `detail`. A visible warning
summarizes issue counts on build/reuse. Broken audio and incomplete pairs are
excluded. Audio with unusable text remains queryable with `transcript_status`;
it does not become an ASR reference. Placeholders are expected data, exposed as
flags and status rather than thousands of repeated warnings.

## Batches and shuffling

```python
for batch in view.iter_batches('utterances', batch_size=256):
    pass
for row in view.iter_rows(shuffle='buffer', buffer_size=1024, seed=42):
    pass
rows = list(view.iter_rows(shuffle='index', seed=42))
```

`shuffle=None` preserves query order. `index` collects the selected metadata rows
and shuffles an O(N) index list, giving a uniform permutation. It uses O(N) row
storage plus indices. `buffer` streams batches and replaces random entries in a
bounded buffer; its row storage is O(batch_size + buffer_size), apart from Polars
query execution state. It is not a uniform global permutation unless the whole
query fits in the buffer. A buffer of one preserves order. Both use local RNGs,
default seed 42, and never decode audio. Reproducibility requires the same query
and input order; explicitly sort custom grouped/joined queries when needed.
Changing batch/buffer size or worker configuration can change stream order.
Polars can need additional memory for sorts, joins, and aggregations.
The batching interface uses [Polars collect_batches](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_batches.html).

## Participant-safe evaluation

```python
from parlo import SplitManifest

split = corpus.split()  # seed=42, proportions=(.70, .15, .15)
split.save('participant-split.json')
split = SplitManifest.load('participant-split.json')
train_recordings = split.apply(view.recordings, 'train')
test_segments = split.apply(corpus.asr(), 'test')
folds = corpus.kfold()  # five diagnosis-stratified folds, seed=42
for split in corpus.leave_one_out():
    pass
```

Each participant is assigned once before expansion to tasks/segments. Reuse the
same saved ID manifest for all task views and adapters. IDs absent from a selected
task are allowed; query IDs absent from the entire manifest raise by default
(`strict=False` explicitly permits dropping them). Manifests reject duplicate IDs
within or across partitions.

Train/validation/test allocation is per diagnosis, using nearest proportional
integer counts while guaranteeing at least one member per nonzero partition.
Consequently small cohorts can differ substantially from requested proportions.
A diagnosis with fewer participants than nonzero partitions is infeasible. K-fold
requires at least `folds` participants in every selected diagnosis. Errors suggest
reducing folds/partitions or broadening the cohort. Each k-fold manifest has
`train` and `test`; leave-one-out has one held-out participant. Splitting a filtered
view deliberately changes the eligible cohort; create the master manifest first
when comparing tasks.

```python
three_way = corpus.targets()  # HC=0, MCI=1, AD=2
pair = corpus.targets(labels=['HC', 'AD'])  # requested order defines 0/1
binary = corpus.targets('hc_vs_impaired')  # HC=0, MCI=AD=1
mmse = corpus.targets('mmse')
cdr = corpus.targets('cdr')
print(pair.mapping)
print(mmse.exclusions.collect())
regression_recordings = mmse.expand('recordings')
regression_split = mmse.split()  # stratified by original diagnosis
```

Target views keep original diagnosis and metadata and add `target`. MMSE/CDR
conversion accepts decimal commas and excludes missing, nonnumeric, or nonfinite
values only from that target view, with a warning and an exclusion table.
Regression mappings are empty because the numerical values are not relabeled.
Pairwise views report participants outside the requested labels in `exclusions`.

## ASR examples

```python
utterances = corpus.asr('utterance')
recordings = corpus.asr('recording')
verbatim = corpus.asr(normalize=False)
case_sensitive = corpus.asr(lowercase=False, punctuation=True)
example = utterances.collect().row(0, named=True)
audio = corpus.load_audio(example)
```

Utterance examples contain **participant speech only**, a valid reference, and
valid utterance timing. Recording examples use the **full audio and all speakers'
text in transcript order**, including interviewer speech, so the reference
matches the audio scope. The recording table's original `text` column remains
participant-only. Raw utterance/word rows expose other speaker turns separately.
Recording-level eligibility conservatively rejects any reported parsing/timing
issue and requires participant speech; usable participant utterances from such a
recording can still be selected independently.

Examples carry both cleaned `reference_verbatim` and normalized `reference`.
Default normalization applies Unicode NFC, lowercase, punctuation removal, and
whitespace collapse. Fillers and repetitions remain. This does **not** duplicate
the paper's filler/numeric normalization. Customize the resulting lazy query for
another protocol. `xxx`, `yyy`, `www`, `notrans`, absent text, and missing or invalid
utterance timings cannot supply timed ASR references. Placeholder tasks remain
available for acoustic work. Structural eligibility does not establish that a
reference covers every audible sound; inspect the data before benchmarking.

## Optional integrations

```python
from parlo.torch import ParloDataset, ParloIterableDataset
from parlo.huggingface import to_huggingface

query = corpus.split().apply(corpus.asr(), 'train')
map_dataset = ParloDataset(query)  # materializes metadata, not audio
stream_dataset = ParloIterableDataset(query, shuffle='buffer', seed=42)
hf_dataset = to_huggingface(query)
hf_stream = to_huggingface(query, iterable=True, shuffle='buffer', seed=42)
audio = corpus.load_audio(map_dataset[0])  # explicit decoding
```

All adapters return dictionaries with relative `audio_path` strings and optional
millisecond bounds. They do not register an automatic audio decoder. Map adapters
materialize metadata; iterable adapters evaluate the query on iteration. Hugging
Face uses its [generator dataset interface](https://huggingface.co/docs/datasets/create_dataset)
for streaming. Polars Python normalization expressions need `cloudpickle`, included
in both extras.

PyTorch iterable workers shard by row position after the same deterministic
shuffle, avoiding duplicate examples; each worker scans metadata. Use a stable
query order and `spawn` multiprocessing when using Polars with workers. The Hugging
Face generator represents one shard. For basic exploration use `num_workers=0`.
Distributed rank sharding is left to the caller. Nullable/raw fields may need a
custom collate function: use `DataLoader(dataset, batch_size=None)` for rows, or
select only the columns your collator supports. No padding or training loop is
provided.

## Notebook and verification

```bash
uv pip install '.[notebook]'
.venv/bin/jupyter lab examples/explore_parlo.ipynb
```

Edit only `CORPUS_ROOT` in the first code cell. The notebook contains task, group,
and participant selectors; participant demographics; recording duration and
coverage plots; transcript/timing inspection; explicit audio playback; incomplete
case comparisons; leakage checks; target views; and optional adapter examples.
It ships with outputs cleared. Notebook-generated split manifests are written
inside the local cache, not packaged with the project.

```bash
uv pip install '.[test,notebook,torch,huggingface]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/audit_corpus.py /path/to/corpus-root
.venv/bin/python examples/verify_notebook.py /path/to/corpus-root
uv build
```

`verify_notebook.py` executes a temporary copy and never saves corpus outputs into
the shipped notebook. See [VALIDATION.md](VALIDATION.md) for the local audit and
verification results. Builds include only package source, tests, documentation,
and examples: no corpus, paper PDF, caches, or audio.
