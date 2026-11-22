"""Reading audio files and drum-hit annotations from IDMT-SMT-Drums.

The dataset stores three kinds of audio file, distinguished by a tag in
the filename:

    WaveDrum02_01#MIX.wav        a polyphonic drum loop
    WaveDrum02_01#KD#train.wav   an isolated kick-drum exemplar
    WaveDrum02_01#KD.wav         a perfectly isolated kick-drum track

Only the #MIX loops carry XML annotations. For the isolated files the
instrument is given by the filename itself, which is why they serve as
clean, uncontaminated training material.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import librosa

# Every file is resampled to this rate so that a fixed-length window is
# always the same number of samples.
SAMPLE_RATE = 44100

# The three instruments this project transcribes, in a fixed order used
# by every array in the pipeline.
LABELS = ["kick", "snare", "hihat"]

# IDMT uses two-letter instrument codes. MDB Drums uses the same codes
# for these three, plus TT/CY/OT for toms, cymbals and other percussion,
# which map to None and are discarded.
LABEL_MAP = {
    "KD": "kick",
    "SD": "snare",
    "HH": "hihat",
    "TT": None,
    "CY": None,
    "OT": None,
}


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def load_audio(path, sample_rate=SAMPLE_RATE):
    """Load an audio file as a mono waveform.

    Returns (samples, sample_rate) where samples is a 1-D float array.
    """
    samples, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return samples, sr


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------


def load_annotations(path):
    """Read ground-truth hits from an IDMT annotation file.

    Each hit is stored as an <event> carrying an <onsetSec> in seconds
    and an <instrument> code. Returns a time-sorted list of
    (time_seconds, label) tuples; instruments outside the three target
    classes are dropped.
    """
    root = ET.parse(str(path)).getroot()
    hits = []

    for event in root.iter("event"):
        onset = event.findtext("onsetSec")
        code = event.findtext("instrument")
        if onset is None or code is None:
            continue

        label = LABEL_MAP.get(code.strip().upper())
        if label is not None:
            hits.append((float(onset), label))

    return sorted(hits)


# --------------------------------------------------------------------------
# Filename conventions
# --------------------------------------------------------------------------


def file_kind(path):
    """Classify a WAV file by its filename tag.

    Returns "mix" for a polyphonic loop, "train" for an isolated
    exemplar, "isolated" for a perfectly separated single-instrument
    track, or None when the name matches no known pattern.
    """
    stem = Path(path).stem
    parts = stem.split("#")

    if len(parts) == 2 and parts[1].upper() == "MIX":
        return "mix"
    if len(parts) == 3 and parts[2].lower() == "train":
        return "train"
    if len(parts) == 2 and parts[1].upper() in LABEL_MAP:
        return "isolated"
    return None


def file_label(path):
    """Instrument label carried by an isolated file's name.

    WaveDrum02_01#KD#train.wav -> "kick". Returns None for mix loops and
    for instrument codes outside the three target classes.
    """
    parts = Path(path).stem.split("#")
    if len(parts) < 2:
        return None
    return LABEL_MAP.get(parts[1].strip().upper())


def annotation_for(audio_path, annotation_dir=None):
    """Locate the XML annotation belonging to a mix loop.

    Audio and annotation share a basename and differ only in directory
    and extension. Returns a Path, or None when no annotation exists.
    """
    audio_path = Path(audio_path)
    if annotation_dir is None:
        annotation_dir = audio_path.parent.parent / "annotation_xml"

    candidate = Path(annotation_dir) / f"{audio_path.stem}.xml"
    return candidate if candidate.exists() else None


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------


def find_files(root, kind=None):
    """List the dataset's WAV files, optionally filtered by kind.

    root is the dataset directory containing audio/ and
    annotation_xml/. Returns a sorted list of Paths.
    """
    audio_dir = Path(root) / "audio"
    files = sorted(audio_dir.glob("*.wav"))

    if kind is None:
        return files
    return [f for f in files if file_kind(f) == kind]
