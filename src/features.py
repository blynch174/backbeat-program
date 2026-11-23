"""Converting drum hits into feature vectors.

Each detected or annotated onset becomes one training row. A short
window is cut from the waveform at the onset and described in one of
two ways:

    Representation A  a small set of hand-designed scalar descriptors
                      (spectral shape, energy, MFCCs)

    Representation B  a mel spectrogram flattened into a long vector,
                      reduced by PCA inside the model pipeline

A is interpretable and compact; B is high-dimensional and lets the model
find its own structure. The two are compared directly in the results.

A window is 100 ms, long enough to capture a drum's attack and early
decay without running into the next stroke at ordinary tempos.
"""

from __future__ import annotations

import librosa
import numpy as np

from . import loading
from .loading import LABELS

# Length of the audio window cut at each onset.
WINDOW_MS = 100

# Two hits closer together than this are treated as one simultaneous
# event rather than two separate strokes. The MDB Drums annotators used
# 50 ms for the same purpose; 20 ms is deliberately stricter so that
# fast but genuinely separate strokes are not merged.
SIMULTANEITY_MS = 20

# Mel bands used by the spectrogram representation.
N_MELS = 64

# Number of MFCCs retained by the hand-designed representation.
N_MFCC = 13


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def extract_window(samples, sample_rate, onset_time, window_ms=WINDOW_MS):
    """Cut a fixed-length window beginning at an onset.

    Windows near the end of a file are zero-padded so that every row of
    the feature table has identical length.
    """
    start = int(round(onset_time * sample_rate))
    length = int(round(window_ms / 1000 * sample_rate))

    start = max(0, start)
    window = samples[start : start + length]

    if len(window) < length:
        window = np.pad(window, (0, length - len(window)))

    return window


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def window_label(onset_time, hits):
    """Single-label target: the instrument nearest this onset.

    Lossy where two drums are struck together, since only one survives.
    """
    if not hits:
        return None
    return min(hits, key=lambda h: abs(h[0] - onset_time))[1]


def window_multilabel(onset_time, hits, tolerance_ms=SIMULTANEITY_MS):
    """Multi-label target: every instrument struck at this onset.

    Returns a three-element indicator vector ordered as LABELS, so a
    kick and hi-hat played together give [1, 0, 1].
    """
    tolerance = tolerance_ms / 1000
    vector = np.zeros(len(LABELS), dtype=int)

    for time, label in hits:
        if abs(time - onset_time) <= tolerance:
            vector[LABELS.index(label)] = 1

    if not vector.any():
        nearest = window_label(onset_time, hits)
        if nearest is not None:
            vector[LABELS.index(nearest)] = 1

    return vector


def isolated_hits(hits, tolerance_ms=SIMULTANEITY_MS):
    """Keep only hits with no neighbour inside the simultaneity window.

    Used by the single-label path, where a window containing two drums
    cannot be represented honestly.
    """
    tolerance = tolerance_ms / 1000
    times = [t for t, _ in hits]
    kept = []

    for i, (time, label) in enumerate(hits):
        neighbours = [
            abs(other - time) <= tolerance for j, other in enumerate(times) if j != i
        ]
        if not any(neighbours):
            kept.append((time, label))

    return kept


def simultaneity_rate(hits, tolerance_ms=SIMULTANEITY_MS):
    """Fraction of hits sharing their moment with another drum."""
    if not hits:
        return 0.0
    return 1.0 - len(isolated_hits(hits, tolerance_ms)) / len(hits)


def group_events(hits, tolerance_ms=SIMULTANEITY_MS):
    """Collapse hits struck together into one musical event.

    The annotations list a simultaneous kick and hi-hat as two entries a
    few milliseconds apart. Taking one window per entry would describe
    the same moment twice, with near-identical audio and identical
    targets, which inflates every count that follows.

    Returns a list of (time_seconds, labels) where time is the earliest
    attack in the group and labels is the set of drums it contains.
    """
    if not hits:
        return []

    tolerance = tolerance_ms / 1000
    events = []
    start, labels = hits[0][0], {hits[0][1]}

    for time, label in hits[1:]:
        if time - start <= tolerance:
            labels.add(label)
        else:
            events.append((start, labels))
            start, labels = time, {label}

    events.append((start, labels))
    return events


# --------------------------------------------------------------------------
# Representation A: hand-designed descriptors
# --------------------------------------------------------------------------

FEATURE_NAMES = (
    ["spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
     "zero_crossing_rate", "rms_energy"]
    + [f"mfcc_{i + 1}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i + 1}_std" for i in range(N_MFCC)]
)


def summary_features(window, sample_rate, n_mfcc=N_MFCC):
    """Describe one hit with a short vector of acoustic statistics.

    Spectral centroid separates low drums from high ones: a kick sits
    far lower than a hi-hat. Rolloff and bandwidth describe how energy
    is spread, zero-crossing rate responds to noisiness, and RMS gives
    loudness. The MFCCs summarise overall timbre.

    Column order matches FEATURE_NAMES, so a fitted model's feature
    importances can be read back by name.
    """
    spec = np.abs(librosa.stft(window, n_fft=512, hop_length=128))

    centroid = librosa.feature.spectral_centroid(S=spec, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(S=spec, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(S=spec, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(window, frame_length=512, hop_length=128)
    rms = librosa.feature.rms(S=spec, frame_length=512)

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(
            librosa.feature.melspectrogram(S=spec**2, sr=sample_rate, n_mels=N_MELS)
        ),
        n_mfcc=n_mfcc,
    )

    return np.concatenate(
        [
            [float(np.mean(centroid))],
            [float(np.mean(rolloff))],
            [float(np.mean(bandwidth))],
            [float(np.mean(zcr))],
            [float(np.mean(rms))],
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
        ]
    )


# --------------------------------------------------------------------------
# Representation B: mel spectrogram
# --------------------------------------------------------------------------


def mel_spectrogram(window, sample_rate, n_mels=N_MELS):
    """Render one hit as a time-frequency image in decibels.

    Rows are mel-spaced frequency bands, columns are time frames. Drum
    types produce visibly different patterns: a kick concentrates energy
    in the lowest bands, a hi-hat spreads it across the highest.
    """
    mel = librosa.feature.melspectrogram(
        y=window, sr=sample_rate, n_fft=512, hop_length=128, n_mels=n_mels
    )
    return librosa.power_to_db(mel, ref=np.max)


def spectrogram_vector(window, sample_rate, n_mels=N_MELS):
    """Flatten a mel spectrogram into a single feature vector.

    Deliberately high-dimensional. PCA inside the model pipeline reduces
    it, fitted separately on each cross-validation fold so that no test
    data influences the transform.
    """
    return mel_spectrogram(window, sample_rate, n_mels).flatten()


# --------------------------------------------------------------------------
# Building the feature table
# --------------------------------------------------------------------------


def featurize(window, sample_rate, representation="summary"):
    """Apply the chosen representation to a single window."""
    if representation == "summary":
        return summary_features(window, sample_rate)
    if representation == "spectrogram":
        return spectrogram_vector(window, sample_rate)
    raise ValueError(f"unknown representation: {representation}")


def collect_windows(files, max_hits_per_file=None):
    """Cut raw audio windows from isolated single-instrument recordings.

    Onsets are detected within each file and the instrument comes from
    the filename, so every window is a clean example of exactly one
    drum. Raw audio is returned rather than features so that windows can
    be summed into artificial mixtures before being described.

    Returns (windows, labels, groups, sample_rate).
    """
    from .onsets import detect_onsets
    from .splits import kit_id

    windows, labels, groups = [], [], []
    sample_rate = None

    for path in files:
        label = loading.file_label(path)
        if label is None:
            continue

        samples, sample_rate = loading.load_audio(path)
        times = detect_onsets(samples, sample_rate)
        if max_hits_per_file is not None:
            times = times[:max_hits_per_file]

        for t in times:
            windows.append(extract_window(samples, sample_rate, t))
            labels.append(label)
            groups.append(kit_id(path))

    return windows, labels, groups, sample_rate


def synthesize_mixtures(
    windows, labels, groups, sample_rate, ratio=1.0, max_overlap=2, seed=0
):
    """Create artificial simultaneous hits by summing isolated windows.

    A detector trained only on isolated drums has never encountered a
    kick with a hi-hat layered over it, and in practice stays silent
    when one arrives. IDMT ships perfectly separated tracks precisely so
    that mixtures can be rebuilt from them, which supplies multi-label
    training data with exact ground truth.

    Components are drawn from the same drum kit so the result is
    acoustically coherent and the row keeps a valid group. Each is
    scaled by a random gain, since drummers do not strike every drum
    equally hard.

    Returns (mixed_windows, multilabel_targets, groups).
    """
    rng = np.random.default_rng(seed)

    by_kit = {}
    for window, label, group in zip(windows, labels, groups):
        by_kit.setdefault(group, {}).setdefault(label, []).append(window)

    usable = [kit for kit, bank in by_kit.items() if len(bank) >= 2]
    if not usable:
        return [], [], []

    n_wanted = int(len(windows) * ratio)
    mixed, targets, mixed_groups = [], [], []

    for _ in range(n_wanted):
        kit = usable[rng.integers(len(usable))]
        bank = by_kit[kit]

        n_components = min(max_overlap, len(bank))
        chosen = rng.choice(
            list(bank.keys()), size=n_components, replace=False
        )

        total = None
        target = np.zeros(len(LABELS), dtype=int)

        for label in chosen:
            pool = bank[label]
            component = pool[rng.integers(len(pool))]
            gain = rng.uniform(0.6, 1.0)
            total = component * gain if total is None else total + component * gain
            target[LABELS.index(label)] = 1

        # A real recording cannot exceed its own headroom, so a summed
        # pair is limited rather than left louder than anything in the
        # genuine material. Without this the energy features alone would
        # betray which rows are synthetic.
        peak = float(np.max(np.abs(total)))
        if peak > 1.0:
            total = total / peak * 0.99

        mixed.append(total)
        targets.append(target)
        mixed_groups.append(kit)

    return mixed, targets, mixed_groups


def build_training_set(
    files,
    representation="summary",
    max_hits_per_file=None,
    multilabel=False,
    synthesize=0.0,
):
    """Build a feature table from isolated single-instrument recordings.

    With synthesize above zero, that many artificial mixtures are added
    relative to the number of real windows, so the training distribution
    contains overlapping hits rather than clean ones only. Synthesis
    requires multi-label targets, since a mixture has no single answer.

    Returns (X, y, groups).
    """
    windows, labels, groups, sample_rate = collect_windows(files, max_hits_per_file)

    if multilabel:
        targets = [
            window_multilabel(0.0, [(0.0, label)]) for label in labels
        ]
    else:
        targets = list(labels)

    if synthesize > 0:
        if not multilabel:
            raise ValueError("synthesis requires multilabel targets")
        extra_windows, extra_targets, extra_groups = synthesize_mixtures(
            windows, labels, groups, sample_rate, ratio=synthesize
        )
        windows = windows + extra_windows
        targets = targets + extra_targets
        groups = groups + extra_groups

    rows = [featurize(window, sample_rate, representation) for window in windows]

    return np.asarray(rows), np.asarray(targets), np.asarray(groups)


def build_evaluation_set(
    files,
    representation="summary",
    multilabel=False,
    exclude_simultaneous=False,
    annotation_dir=None,
):
    """Build a feature table from annotated polyphonic drum loops.

    Onsets come from the XML annotations rather than from detection, so
    classifier performance can be measured independently of detection
    error.

    With multilabel set, each row's target is a three-element indicator
    vector; otherwise it is a single label. exclude_simultaneous drops
    windows containing more than one drum, which the single-label
    formulation cannot represent.

    Returns (X, y, groups).
    """
    from .splits import kit_id

    rows, targets, groups = [], [], []

    for path in files:
        annotation = loading.annotation_for(path, annotation_dir)
        if annotation is None:
            continue

        all_hits = loading.load_annotations(annotation)
        if not all_hits:
            continue

        samples, sr = loading.load_audio(path)

        if multilabel:
            # One window per musical event, so a simultaneous pair is
            # described once rather than twice.
            for time, labels in group_events(all_hits):
                if exclude_simultaneous and len(labels) > 1:
                    continue
                target = np.zeros(len(LABELS), dtype=int)
                for label in labels:
                    target[LABELS.index(label)] = 1
                rows.append(featurize(extract_window(samples, sr, time), sr,
                                      representation))
                targets.append(target)
                groups.append(kit_id(path))
        else:
            hits = isolated_hits(all_hits) if exclude_simultaneous else all_hits
            for time, label in hits:
                rows.append(featurize(extract_window(samples, sr, time), sr,
                                      representation))
                targets.append(label)
                groups.append(kit_id(path))

    return np.asarray(rows), np.asarray(targets), np.asarray(groups)
