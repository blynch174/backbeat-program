"""Scoring predictions against ground truth.

Owner: TBD
"""

TOLERANCE_MS = 50


def match_hits(predicted, actual, tolerance_ms=TOLERANCE_MS):
    """Pair up predicted and true hits.

    A pair matches when the times are within tolerance AND the labels
    agree. Returns (matches, false_positives, false_negatives).
    """
    raise NotImplementedError


def score(predicted, actual):
    """Per-class precision/recall/F1 plus macro F1."""
    raise NotImplementedError


def confusion(predicted, actual):
    """Confusion matrix over LABELS, for the error analysis section."""
    raise NotImplementedError
