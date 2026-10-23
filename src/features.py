"""Turning each hit into something a model can learn from.

Owner: TBD
"""

WINDOW_MS = 100


def extract_window(samples, sample_rate, onset_time, window_ms=WINDOW_MS):
    """Slice the audio starting at an onset."""
    raise NotImplementedError


def mel_spectrogram(window, sample_rate):
    """Convert one hit's waveform into a 2D mel spectrogram."""
    raise NotImplementedError


def summary_features(window, sample_rate):
    """Hand-picked scalar features for the baseline model.

    Candidates: spectral centroid, zero-crossing rate, RMS energy,
    spectral rolloff, a handful of MFCCs.
    """
    raise NotImplementedError
