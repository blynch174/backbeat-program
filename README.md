# Automatic Drum Transcription

CSE 5717 — Big Data Analytics, Fall 2026

## Problem

Jam clubs and cover bands hand out sheet music for vocals, guitar, bass, and
keys, but almost never for the drums. Drum charts are scarce, often inaccurate,
and usually behind a paywall. This project trains a model that listens to a
recording and identifies every drum hit: when it happened and which drum it
was.

For this imeplementation, we will focus on three core drums: kick, snare, and hi-hat.

## Problem framing

Two connected tasks:

1. **Onset detection** — find the timestamps where a drum was struck.
2. **Hit classification** — label each onset as kick, snare, or hi-hat.

A predicted hit counts as correct if it lands within 50 ms maximum of a true hit and
carries the right label.

## Metrics

- Per-class precision, recall, and F1
- Macro-averaged F1 as the headline number
- Confusion matrix for error analysis

## Data

| Dataset | What it is | Use |
|---|---|---|
| [IDMT-SMT-Drums](https://www.idmt.fraunhofer.de/en/publications/datasets/drums.html) | Isolated drum recordings, kick/snare/hi-hat onsets annotated | Primary training set |
| [MDB Drums](https://github.com/CarlSouthall/MDBDrums) | Drum annotations on full MedleyDB music tracks | Real-world test set |
| [E-GMD](https://magenta.withgoogle.com/datasets/e-gmd) | Expanded Groove MIDI — hundreds of hours of human drumming, audio aligned to MIDI | Scale-up if needed |
| [ENST-Drums](https://perso.telecom-paristech.fr/grichard/ENST-drums/) | Real drummers, multi-mic, annotated | Optional harder set |

Datasets are **not** committed to this repo. See `data/README.md`.

## Pipeline

```
audio file
  -> (optional) Demucs source separation to isolate drums
  -> onset detection
  -> 100 ms window around each onset
  -> mel spectrogram
  -> classifier -> kick / snare / hi-hat
  -> MIDI export
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Repo layout

```
src/          importable modules
notebooks/    exploration and experiments
scripts/      one-off runnable scripts
data/         datasets (gitignored)
docs/         proposal, progress report, final report
```

## Team

| Name | Role |
|---|---|
| Brandon Lynch | TBD |
| Aman | TBD |

## Deliverables

- [ ] Proposal — Week 4
- [ ] Progress report — Week 8?
- [ ] Final report + code — Week 14
- [ ] Presentation — Week 14
