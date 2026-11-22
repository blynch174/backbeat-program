"""Locating the moments at which a drum is struck.

Onset detection is the first of the pipeline's two stages. It answers
"when did something happen", leaving "what was it" to the classifier.

Drum hits are percussive: the signal jumps sharply at the strike and
decays afterwards. Detection therefore looks for rapid increases in
spectral energy, which librosa computes as an onset strength envelope.

Simultaneous hits are a known limitation. A kick and a hi-hat played
together are annotated a few milliseconds apart, but the analysis frame
is roughly 11 ms wide at 44.1 kHz, so the pair is reported as a single
onset. Detected counts are therefore expected to fall slightly below
annotated counts even when detection is working correctly.
"""

from __future__ import annotations

import librosa
import numpy as np

# Analysis frame advance, in samples. 512 at 44.1 kHz is about 11.6 ms.
HOP_LENGTH = 512

# Sensitivity of the peak picker. Higher values demand a stronger peak
# and therefore detect fewer onsets. Recall is largely insensitive to
# this setting on IDMT-SMT-Drums, because the ceiling is set by
# simultaneous strokes being merged rather than by threshold choice.
DELTA = 0.03

# Minimum separation between detected onsets, in seconds. Drum strokes
# closer together than this are treated as one event.
MIN_SEPARATION_S = 0.02


def detect_onsets(
    samples,
    sample_rate,
    hop_length=HOP_LENGTH,
    delta=DELTA,
    min_separation_s=MIN_SEPARATION_S,
):
    """Return the times, in seconds, at which drum hits begin.

    Peaks are picked from a percussive onset strength envelope and then
    backtracked to the start of each transient, so the returned time
    marks the attack rather than the energy peak.
    """
    envelope = librosa.onset.onset_strength(
        y=samples,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )

    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=True,
        delta=delta,
        units="frames",
    )

    times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
    return _enforce_separation(times, min_separation_s)


def onset_strength(samples, sample_rate, hop_length=HOP_LENGTH):
    """Return the onset strength envelope and its frame times.

    Useful for plotting detection behaviour against the waveform when
    diagnosing missed or spurious hits.
    """
    envelope = librosa.onset.onset_strength(
        y=samples, sr=sample_rate, hop_length=hop_length, aggregate=np.median
    )
    times = librosa.frames_to_time(
        np.arange(len(envelope)), sr=sample_rate, hop_length=hop_length
    )
    return envelope, times


def detection_summary(detected, annotated, tolerance_s=0.05):
    """Compare detected onsets against annotated ones, ignoring labels.

    Reports how many annotated hits were found, how many detections had
    no annotation nearby, and the mean absolute timing error of the
    matched pairs. This measures the detector alone, with classifier
    performance excluded.
    """
    detected = np.asarray(sorted(detected), dtype=float)
    annotated_times = np.asarray(
        sorted(t for t, _ in annotated) if annotated and isinstance(annotated[0], tuple)
        else sorted(annotated),
        dtype=float,
    )

    matched_errors = []
    unmatched_detections = 0
    available = list(annotated_times)

    for t in detected:
        if not available:
            unmatched_detections += 1
            continue
        distances = [abs(t - a) for a in available]
        best = int(np.argmin(distances))
        if distances[best] <= tolerance_s:
            matched_errors.append(distances[best])
            available.pop(best)
        else:
            unmatched_detections += 1

    found = len(matched_errors)
    total = len(annotated_times)

    return {
        "annotated": total,
        "detected": len(detected),
        "matched": found,
        "missed": total - found,
        "spurious": unmatched_detections,
        "recall": found / total if total else 0.0,
        "mean_error_ms": float(np.mean(matched_errors) * 1000) if matched_errors else None,
    }


def _enforce_separation(times, min_separation_s):
    """Collapse detections that fall closer together than the minimum."""
    if len(times) == 0:
        return np.asarray([], dtype=float)

    kept = [float(times[0])]
    for t in times[1:]:
        if float(t) - kept[-1] >= min_separation_s:
            kept.append(float(t))
    return np.asarray(kept, dtype=float)
