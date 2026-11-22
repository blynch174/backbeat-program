"""Measuring how well the system transcribes.

Two kinds of question are asked separately, because they fail for
different reasons.

    Classification   given that a hit occurred here, was the right drum
                     identified? Onsets come from the annotations, so
                     detection errors are excluded.

    Event level      running on its own, did the system find the hits at
                     all, and label them correctly? A prediction counts
                     only if it lands close enough in time and names the
                     right drum.

Separating them keeps a weak detector from being mistaken for a weak
classifier, and the other way round.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
)

from .loading import LABELS

# Timing tolerances, in milliseconds, at which event-level results are
# reported. 50 ms is the long-standing convention in onset detection
# research, which makes the headline figure comparable with published
# work. The tighter values show how precise the timing really is: a
# score that holds up at 10 ms is placing hits almost exactly.
TOLERANCES_MS = [50, 25, 10]


# --------------------------------------------------------------------------
# Classification scoring
# --------------------------------------------------------------------------


def score_classification(y_true, y_pred, multilabel=False):
    """Per-drum precision, recall and F1, plus the macro average.

    Returns a dict keyed by drum name with an added "macro_f1" entry.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS if multilabel else None,
        labels=None if multilabel else LABELS,
        output_dict=True,
        zero_division=0,
    )

    scores = {}
    for label in LABELS:
        if label in report:
            scores[label] = {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }

    scores["macro_f1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    return scores


def confusion(y_true, y_pred, multilabel=False):
    """Which drums get mistaken for which.

    Single-label returns one matrix whose rows are true drums and
    columns predicted ones. Multi-label returns one two-by-two matrix
    per drum, since each is decided independently.
    """
    if multilabel:
        return multilabel_confusion_matrix(y_true, y_pred)
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def format_confusion(matrix):
    """Render a single-label confusion matrix as aligned text."""
    width = max(len(name) for name in LABELS) + 2
    lines = [" " * width + "".join(f"{name:>10}" for name in LABELS)]

    for name, row in zip(LABELS, matrix):
        lines.append(f"{name:<{width}}" + "".join(f"{value:>10}" for value in row))

    return "\n".join(lines)


def cooccurrence_breakdown(y_true, y_pred):
    """How simultaneous hits are handled, under multi-label targets.

    Restricted to windows whose true target names more than one drum,
    and reports how often both, one, or neither were recovered. This is
    the question the multi-label formulation exists to answer.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    multiple = y_true.sum(axis=1) > 1
    if not multiple.any():
        return {"windows": 0}

    true_rows = y_true[multiple]
    pred_rows = y_pred[multiple]

    recovered = (true_rows & pred_rows).sum(axis=1)
    expected = true_rows.sum(axis=1)

    return {
        "windows": int(multiple.sum()),
        "all_recovered": int((recovered == expected).sum()),
        "partly_recovered": int(((recovered > 0) & (recovered < expected)).sum()),
        "none_recovered": int((recovered == 0).sum()),
    }


# --------------------------------------------------------------------------
# Event-level scoring
# --------------------------------------------------------------------------


def match_events(predicted, actual, tolerance_ms):
    """Pair predicted events with annotated ones by time.

    Both arguments are lists of (time_seconds, label). Pairing is
    nearest-first and strictly one to one, so a single prediction can
    never satisfy two annotations.

    Returns (pairs, unmatched_predicted, unmatched_actual).
    """
    tolerance = tolerance_ms / 1000

    candidates = []
    for i, (pred_time, _) in enumerate(predicted):
        for j, (true_time, _) in enumerate(actual):
            distance = abs(pred_time - true_time)
            if distance <= tolerance:
                candidates.append((distance, i, j))

    candidates.sort()

    used_pred, used_true, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in used_pred or j in used_true:
            continue
        used_pred.add(i)
        used_true.add(j)
        pairs.append((predicted[i], actual[j]))

    unmatched_pred = [p for i, p in enumerate(predicted) if i not in used_pred]
    unmatched_true = [a for j, a in enumerate(actual) if j not in used_true]

    return pairs, unmatched_pred, unmatched_true


def score_events(predicted, actual, tolerance_ms):
    """Per-drum precision, recall and F1 at one timing tolerance.

    Credit is counted per instrument rather than per event, so a matched
    pair naming the right time but the wrong drum produces one false
    positive and one false negative instead of a single opaque error.
    """
    pairs, extra, missing = match_events(predicted, actual, tolerance_ms)

    tp = {label: 0 for label in LABELS}
    fp = {label: 0 for label in LABELS}
    fn = {label: 0 for label in LABELS}

    for (_, pred_label), (_, true_label) in pairs:
        for label in _as_set(pred_label) & _as_set(true_label):
            tp[label] += 1
        for label in _as_set(pred_label) - _as_set(true_label):
            fp[label] += 1
        for label in _as_set(true_label) - _as_set(pred_label):
            fn[label] += 1

    for _, label in extra:
        for name in _as_set(label):
            fp[name] += 1
    for _, label in missing:
        for name in _as_set(label):
            fn[name] += 1

    scores, f1s = {}, []
    for label in LABELS:
        precision = _ratio(tp[label], tp[label] + fp[label])
        recall = _ratio(tp[label], tp[label] + fn[label])
        f1 = _ratio(2 * precision * recall, precision + recall)
        scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp[label] + fn[label],
        }
        f1s.append(f1)

    scores["macro_f1"] = float(np.mean(f1s))
    return scores


def score_events_all_tolerances(predicted, actual):
    """Event-level scores at every tolerance in TOLERANCES_MS.

    A score that falls away sharply as the tolerance tightens means the
    right drums are being named at slightly wrong moments, which points
    at onset detection rather than at classification.
    """
    return {ms: score_events(predicted, actual, ms) for ms in TOLERANCES_MS}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_scores(scores, title=""):
    """Render a score dict as an aligned table."""
    lines = []
    if title:
        lines.append(title)

    lines.append(f"{'':<8}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in LABELS:
        if label not in scores:
            continue
        row = scores[label]
        lines.append(
            f"{label:<8}{row['precision']:>12.3f}{row['recall']:>10.3f}"
            f"{row['f1']:>10.3f}{int(row['support']):>10}"
        )

    lines.append(f"{'macro F1':<8}{scores['macro_f1']:>12.3f}")
    return "\n".join(lines)


def _as_set(label):
    """Normalise a single label or an indicator vector to a set of names."""
    if isinstance(label, str):
        return {label}
    if isinstance(label, (set, frozenset)):
        return set(label)
    return {LABELS[i] for i, present in enumerate(label) if present}


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0
