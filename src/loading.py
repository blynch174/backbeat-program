"""Reading audio files and their hit annotations.

Owner: TBD
"""


def load_audio(path, sample_rate=44100):
    """Load an audio file as a mono waveform.

    Returns (samples, sample_rate).
    """
    raise NotImplementedError


def load_annotations(path):
    """Read a dataset's ground-truth hits.

    Returns a list of (time_seconds, label) where label is one of
    "kick", "snare", "hihat".
    """
    raise NotImplementedError
