# Validation report

Verified locally on 28 October 2026 with Python 3.12.14, Polars 1.44.2,
SoundFile 0.14.0, PyTorch 2.14.1, and datasets 5.1.0 on macOS arm64.

## Automated checks

- **32 tests passed**, including actual PyTorch `spawn` workers and Hugging Face
  map/iterable adapter smoke tests.
- Synthetic fixtures cover multiline CHAT tiers, participant/interviewer roles,
  raw annotations, placeholders, nullable/invalid timings, missing pairs, broken
  audio/text, duplicate/missing/conflicting metadata, filtering, complete cases,
  cache reuse/invalidation, optional SHA-256 verification, and interrupted writes.
- Audio tests compare exact native-rate slices, channel averaging, resampling,
  and rejected out-of-range/reversed/noninteger boundaries.
- Split checks verify seeded reproducibility, participant isolation at recording
  and segment levels, saved manifest reuse, five-fold coverage, leave-one-out,
  target mappings, missing-target exclusions, and infeasible cohort errors.
- Metadata queries, shuffling, and adapter access are tested with waveform reads
  patched to fail, establishing that these operations do not decode audio.
- The exploration notebook executed **all 13 code cells** against the supplied
  corpus, including explicit excerpt playback and both optional integrations.
  A temporary additional cell exercised task/group/participant selectors and
  disabled playback for an untranscribed task. The shipped notebook retains
  empty outputs and null execution counts.
- A wheel installed and imported in a separate core-only uv environment; corpus
  queries and default splits worked with no PyTorch/datasets installed or loaded.
  `uv add /absolute/package/path` also succeeded in a disposable project.
- Wheel and source archives were inspected for corpus files, metadata files,
  caches, Parquet tables, audio, and the supplied paper; none are included.

## Supplied corpus audit

| Measure | Result |
| --- | ---: |
| Participants | 208: 83 HC, 59 MCI, 66 AD |
| WAV/CHAT pairs | 1,633 |
| Audio duration | 19.931 hours |
| Participant and interviewer utterances, including placeholders | 2,575 |
| Word-table rows, including untimed tokens | 75,273 |
| Placeholder recordings | 437 |
| Other recordings excluded from recording-level ASR | 2 |
| Eligible full-recording ASR examples | 1,194 |
| Eligible timed participant-utterance ASR examples | 1,572 |
| Participants with all eight file pairs | 201 |
| Participants missing/invalid MMSE | 2 |
| Participants missing/invalid CDR | 32 |
| Default participant train/validation/test sizes | 145 / 32 / 31 |

There were no incomplete pairs or directory/metadata diagnosis conflicts in this
snapshot. Four validation rows concern two recordings: one has only interviewer
speech; another has an interviewer utterance and word boundary beyond the audio
end. Those original annotations remain visible. The latter recording's valid
participant utterances remain eligible independently. Neither recording supplies
a full-recording ASR reference under the conservative policy.

The paper's curated all-task subset has 171 participants, not the download's 201
participants with eight file pairs. The paper's per-task counts sum to 1,591,
not 1,633. See README for the per-task comparison. File presence and structural
reference validity do not recreate the paper's curation or establish transcript
accuracy. The audit did not manually listen to every file or hash the full audio
corpus; content-hash verification was exercised on synthetic fixtures.

## Reproduction and limits

```bash
uv pip install '.[test,notebook,torch,huggingface]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/audit_corpus.py /path/to/corpus-root
.venv/bin/python examples/verify_notebook.py /path/to/corpus-root
uv build
```

Hugging Face may write its standard local cache; notebook execution starts a
local Jupyter kernel. The host sandbox required authorization for those accesses.
No corpus source files were edited; only the local cache was generated.
No package was published. Compatibility was exercised on the versions above,
not every supported Python/platform combination.

The enclosing repository already ignores this `parlo_dataset` working directory.
Package deliverables exist on disk, but that pre-existing parent ignore rule was
left unchanged; tracking them in the parent repository requires an explicit
exception or selective force-add. Corpus files must remain excluded.
