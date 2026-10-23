"""Classifiers that label a hit as kick, snare, or hi-hat.

Owner: TBD
"""

LABELS = ["kick", "snare", "hihat"]


def train_baseline(X, y):
    """Simple model on summary features. Our floor to beat.

    Suggested: logistic regression or a small random forest.
    """
    raise NotImplementedError


def train_cnn(X, y):
    """Small convolutional net over mel spectrograms."""
    raise NotImplementedError


def predict(model, X):
    """Return predicted labels."""
    raise NotImplementedError
