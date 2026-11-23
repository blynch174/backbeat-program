# Backbeat — Automatic Drum Transcription

CSE 5717, Big Data Analytics — Fall 2026
Brandon Lynch and Aman Patel

Backbeat listens to a drum recording and writes down what was played:
when each hit occurred and which drum produced it. Sheet music is easy
to find for guitar, bass, keys and vocals, and hard to find for drums —
what exists is often inaccurate or paid. This project builds the
machine learning stage that a drum-charting tool would need.

Scope is the three drums that carry almost every groove: kick, snare
and hi-hat.

## How it works

Transcription is split into two stages that fail for different reasons
and are therefore measured separately.

```
audio file
   │
   ├─ onset detection ──────── when was a drum struck?
   │
   ├─ 100 ms window cut at each onset
   │
   ├─ feature extraction ───── A. hand-designed descriptors
   │                           B. mel spectrogram + PCA
   │
   ├─ classification ───────── which drum was it?
   │
   └─ transcription ───────── printed table, or a MIDI drum chart
```

## Repository layout

```
src/
  loading.py      audio and annotation readers, filename conventions
  onsets.py       onset detection and detector diagnostics
  features.py     windowing, both feature representations, table building
  splits.py       kit- and source-aware data splitting
  models.py       classifiers, ensembles, Keras network
  evaluation.py   classification and event-level scoring

scripts/
  explore_data.py   dataset inventory and annotation statistics
  train.py          model comparison, evaluation, saves the chosen model
  transcribe.py     transcribe a recording; optional MIDI output
  check_loading.py  sanity-check the loaders against one file
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset into `data/` — see `data/README.md` — then:

```bash
python scripts/explore_data.py        # what is in the data

# best configuration
python scripts/train.py --held-out-kits --representation spectrogram \
                        --multilabel --synthesize 1.0

python scripts/transcribe.py some.wav --midi out.mid
```

## Dataset

[IDMT-SMT-Drums](https://zenodo.org/records/7544164), from Fraunhofer
IDMT. Three kinds of file, distinguished by a tag in the filename:

| Pattern | Contents | Role |
|---|---|---|
| `*#MIX.wav` | polyphonic drum loop, XML-annotated | evaluation |
| `*#KD#train.wav` | isolated single-instrument exemplar | training |
| `*#KD.wav` | perfectly separated track | source separation work |

Training uses the isolated recordings, where each window holds exactly
one drum and the label comes from the filename. Evaluation uses the
annotated loops, where drums overlap. The gap between the two is the
difference between recognising a drum and transcribing drumming.

Recordings come from three sound sources — `RealDrum` (acoustic kits),
`WaveDrum` (sample libraries) and `TechnoDrum` (synthesizers) — which
supports a harder question: does a model trained on some kinds of drum
recognise kinds it has never heard?

Audio is not committed to this repository. See `data/README.md`.

## Method

**Splitting is by drum kit.** `WaveDrum02_51#KD#train.wav` and
`WaveDrum02_51#MIX.wav` are the same drums recorded two ways, so
splitting file by file would let a model be tested on a kit it had
already learned. Every split groups on the kit identifier instead.

**Preprocessing lives inside the pipeline.** Standardisation and PCA
are fitted within each cross-validation fold, never once on the full
table, so no test data influences the transform.

**Macro F1 is the headline metric.** The classes are unbalanced — over
half of all annotated hits are hi-hats — so plain accuracy rewards a
model that neglects the rarer drums.

**Event-level scores are reported at three tolerances**: 50 ms, which
is the convention in onset-detection research and keeps results
comparable with published work, plus 25 ms and 10 ms to show how
precisely hits are actually placed.

## Results

Best configuration: **0.907 macro F1** on held-out drum kits — soft
voting over random forest, MLP and SVC, on PCA-reduced mel
spectrograms, with multi-label targets and synthesized mixtures in
training. Both drums recovered in 97% of simultaneous windows.

| configuration | hand-designed | spectrogram + PCA |
|---|---|---|
| single-label | 0.674 | 0.641 |
| multi-label, isolated training only | 0.707 | — |
| multi-label + synthesized mixtures | 0.866 | **0.907** |

`docs/EXPERIMENTS.md` records what was tried, what worked, and what did
not.

## Models

Following the ensemble methods in Géron, *Hands-On Machine Learning*
(ch. 7) and the Keras workflow in Chollet, *Deep Learning with Python*
(ch. 2):

1. Random forest, MLP and support vector classifier, individually
2. Hard voting — majority of the three
3. Soft voting — mean of the three probability estimates
4. Stacking — a random forest blends the three
5. A Keras dense network on the same PCA-reduced inputs

Each runs over both feature representations, and under both
single-label and multi-label targets.

## Simultaneous hits

Drummers routinely strike two drums at once; in this dataset more than
half of all annotated hits share their moment with another. That breaks
two assumptions:

- A 100 ms window cut at a kick also contains the hi-hat struck with it.
- Onset detection resolves the pair as one event, because the analysis
  frame is wider than the gap between them.

Two responses are implemented and compared. The **single-label** model
predicts one drum per onset and simply cannot represent a pair. The
**multi-label** model predicts each drum independently, so a kick and
hi-hat together is `[1, 0, 1]`, and `evaluation.cooccurrence_breakdown`
reports how often both are recovered.

## Limitations and future work

- **Three instruments.** Toms, cymbals and ride are not modelled. Real
  charts need them.
- **One dataset.** [MDB Drums](https://github.com/CarlSouthall/MDBDrums)
  — 23 real musical performances with 7,994 annotated onsets — is the
  natural next test set, and would show whether this transfers from
  isolated drum loops to drums inside full mixes. It is scoped out of
  the semester, not out of the project.
- **Drums only.** Transcribing a full song needs source separation
  first; [Demucs](https://github.com/facebookresearch/demucs) is the
  standard route.
- **MIDI, not notation.** The output opens in MuseScore as a drum
  track. Turning that into a readable chart needs tempo estimation and
  quantisation to a beat grid.
- **No playing technique.** Ghost notes, flams and open versus closed
  hi-hat are all collapsed into their parent instrument.

## References

- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*.
- Chollet, F. *Deep Learning with Python*.
- Dittmar, C. and Gärtner, D. "Real-time transcription and separation of
  drum recordings based on NMF decomposition." DAFx, 2014. (IDMT-SMT-Drums)
- Southall, C., Wu, C.-W., Lerch, A. and Hockman, J. "MDB Drums: An
  annotated subset of MedleyDB for automatic drum transcription." ISMIR
  late-breaking demo, 2017.
- McFee, B. et al. "librosa: Audio and music signal analysis in Python."
  SciPy, 2015.
