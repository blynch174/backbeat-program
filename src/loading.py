"""Reading audio files and their hit annotations.

Owner: Brandon
"""
import librosa
import xml.etree.ElementTree as ET

SAMPLE_RATE = 44100
LABEL_MAP = {"KD": "kick", "SD": "snare", "HH": "hihat"}

def load_audio(path, sample_rate=44100):
    """Load an audio file as a mono waveform.

    Returns (samples, sample_rate).
    """
    samples, sr = librosa.load(path, sr=sample_rate, mono=True)
    return samples, sr


def load_annotations(path):
    """Read a dataset's ground-truth hits.

    Returns a list of (time_seconds, label) where label is one of
    "kick", "snare", "hihat".
    """
    root = ET.parse(path).getroot()
    hits = []

    for event in root.iter("event"):
        onset = float(event.findtext("onsetSec"))
        raw = event.findtext("instrument")

        label = LABEL_MAP.get(raw.strip().upper())
        if label is not None:
            hits.append((onset, label))

    return sorted(hits)
