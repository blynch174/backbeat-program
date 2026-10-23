"""Finding the moments where a drum was struck.

Owner: TBD
"""


def detect_onsets(samples, sample_rate):
    """Return onset times in seconds.

    Starting point: librosa.onset.onset_detect. Expect to tune the
    threshold — this is the fiddliest part of the pipeline.
    """
    raise NotImplementedError
