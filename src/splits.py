"""Dividing the data so that results mean what they appear to mean.

Every file in IDMT-SMT-Drums belongs to a drum kit, identified by the
part of its filename before the first '#':

    WaveDrum02_51#KD#train.wav  ->  WaveDrum02_51
    WaveDrum02_51#MIX.wav       ->  WaveDrum02_51

Those two files are the same drums recorded two ways. Training on one
and testing on the other would let the model recognise a kit it had
already heard, so splits are made at kit level rather than file level.

The dataset also spans three sound sources, which support a second and
harder question: does a model trained on some kinds of drum recognise
kinds it has never encountered?

    RealDrum     acoustic kits recorded in a room
    WaveDrum     commercial drum sample libraries
    TechnoDrum   drum synthesizers
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

SOURCES = ["RealDrum", "WaveDrum", "TechnoDrum"]


def kit_id(path):
    """Return the drum kit a file belongs to.

    This is the grouping key for every split in this module.
    """
    stem = Path(path).stem
    if "#" not in stem:
        raise ValueError(
            f"{stem!r} has no '#' separator; kit_id expects IDMT-SMT-Drums "
            "filenames such as WaveDrum02_51#MIX.wav"
        )
    return stem.split("#")[0]


def source_id(path):
    """Return which of the three sound sources a file came from."""
    kit = kit_id(path)
    match = re.match(r"([A-Za-z]+)", kit)
    name = match.group(1) if match else ""
    return name if name in SOURCES else None


def train_test_split_by_kit(X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Hold out whole drum kits for testing.

    Returns (X_train, X_test, y_train, y_test, groups_train).
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        groups[train_idx],
    )


def cv_splitter(n_folds=N_FOLDS):
    """Cross-validation that keeps each kit inside a single fold.

    Pass the result to cross_val_score together with groups=... so that
    fold scores reflect performance on unseen kits.
    """
    return GroupKFold(n_splits=n_folds)


def safe_n_folds(groups, requested=N_FOLDS):
    """Clamp the fold count to the number of distinct kits available."""
    return max(2, min(requested, len(np.unique(groups))))


def split_by_source(X, y, sources, held_out):
    """Train on two sound sources and test on the third.

    held_out is one of SOURCES. Returns
    (X_train, X_test, y_train, y_test, groups_train_sources).
    """
    if held_out not in SOURCES:
        raise ValueError(f"held_out must be one of {SOURCES}, got {held_out!r}")

    sources = np.asarray(sources)
    test_mask = sources == held_out
    train_mask = ~test_mask

    return (
        X[train_mask],
        X[test_mask],
        y[train_mask],
        y[test_mask],
        sources[train_mask],
    )
