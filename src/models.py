"""Classifiers that identify which drum produced a hit.

The progression follows the ensemble methods in Geron, Hands-On Machine
Learning, chapter 7, and the Keras workflow in Chollet, Deep Learning
with Python, chapter 2:

    1. three independent classifiers   random forest, MLP, support vector
    2. hard voting                     majority of the three predictions
    3. soft voting                     mean of the three probabilities
    4. stacking                        a random forest blends the three
    5. Keras network                   a small dense network on the same
                                       PCA-reduced inputs

Every estimator is wrapped in a Pipeline so that standardisation and
PCA are fitted inside each cross-validation fold. Fitting them once on
the full table would let test data influence the transform and inflate
every score that follows.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .loading import LABELS

RANDOM_STATE = 42

# Proportion of variance PCA retains, following the dimensionality
# reduction step used on Fashion-MNIST in the course assignments.
PCA_VARIANCE = 0.95


# --------------------------------------------------------------------------
# Base estimators
# --------------------------------------------------------------------------


def base_estimators(multilabel=False):
    """The three classifiers compared individually and then combined.

    Support vector machines produce no probabilities unless asked, and
    soft voting needs them, so probability estimation is enabled here.
    """
    return [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=600,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "svc",
            SVC(probability=True, random_state=RANDOM_STATE),
        ),
    ]


def build_pipeline(classifier, use_pca=False, multilabel=False):
    """Attach preprocessing to an estimator.

    Standardisation comes first because the descriptors live on very
    different scales: spectral centroid is measured in thousands of
    hertz while zero-crossing rate lies between zero and one.

    Multi-label targets are handled by MultiOutputClassifier, which
    fits one copy of the estimator per instrument. It wraps the whole
    classifier, ensembles included, because voting and stacking accept
    only single-column targets.
    """
    steps = [("scaler", StandardScaler())]

    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)))

    if multilabel:
        classifier = MultiOutputClassifier(classifier)

    steps.append(("clf", classifier))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Individual models
# --------------------------------------------------------------------------


def random_forest(use_pca=False, multilabel=False):
    """An ensemble of decision trees, used here as the reference model."""
    return build_pipeline(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        use_pca,
        multilabel,
    )


def mlp(use_pca=False, multilabel=False):
    """A fully connected neural network trained by scikit-learn."""
    return build_pipeline(
        MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=600, random_state=RANDOM_STATE
        ),
        use_pca,
        multilabel,
    )


def svc(use_pca=False, multilabel=False):
    """A support vector machine with a radial basis function kernel."""
    return build_pipeline(
        SVC(probability=True, random_state=RANDOM_STATE), use_pca, multilabel
    )


# --------------------------------------------------------------------------
# Ensembles
# --------------------------------------------------------------------------


def hard_voting(use_pca=False, multilabel=False):
    """Predict the class that most of the three classifiers choose."""
    return build_pipeline(
        VotingClassifier(estimators=base_estimators(), voting="hard"),
        use_pca,
        multilabel,
    )


def soft_voting(use_pca=False, multilabel=False):
    """Average the three classifiers' probabilities and take the largest.

    Usually beats hard voting because a confident classifier outweighs
    two uncertain ones instead of being outvoted by them.
    """
    return build_pipeline(
        VotingClassifier(estimators=base_estimators(), voting="soft"),
        use_pca,
        multilabel,
    )


def stacking(use_pca=False, multilabel=False, cv=3):
    """Train a second-stage model on the first stage's predictions.

    The blender learns which classifier to trust in which situation,
    rather than weighting them equally as voting does.
    """
    return build_pipeline(
        StackingClassifier(
            estimators=base_estimators(),
            final_estimator=RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            ),
            cv=cv,
        ),
        use_pca,
        multilabel,
    )


def manual_stacking_features(X, y, cv=3):
    """Produce blender training data by hand, as done in the coursework.

    Each base classifier predicts out-of-fold probabilities for every
    row; those probabilities are concatenated into the feature matrix a
    blender is then trained on. StackingClassifier does this internally,
    and this function exposes the same construction explicitly.
    """
    columns = []

    for _, estimator in base_estimators():
        pipeline = build_pipeline(estimator)
        probabilities = cross_val_predict(
            pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1
        )
        columns.append(probabilities)

    return np.hstack(columns)


# --------------------------------------------------------------------------
# Keras network
# --------------------------------------------------------------------------


def keras_network(input_dim, n_classes=len(LABELS), multilabel=False):
    """A small dense network built with Keras.

    Two hidden layers with rectified linear activations feed an output
    layer whose form depends on the task. Single-label classification
    uses softmax with categorical cross-entropy, since the classes are
    mutually exclusive. Multi-label uses sigmoid with binary
    cross-entropy, so each instrument is decided independently.
    """
    from tensorflow import keras

    keras.utils.set_random_seed(RANDOM_STATE)

    output_activation = "sigmoid" if multilabel else "softmax"
    loss = "binary_crossentropy" if multilabel else "categorical_crossentropy"

    network = keras.models.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(n_classes, activation=output_activation),
        ]
    )
    network.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return network


def fit_keras(X_train, y_train, X_valid=None, y_valid=None, epochs=30, multilabel=False):
    """Standardise, reduce with PCA, and train the Keras network.

    Returns (network, scaler, pca) so the same transforms can be applied
    to held-out data later.
    """
    from tensorflow import keras

    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE).fit(
        scaler.transform(X_train)
    )

    def prepare(X):
        return pca.transform(scaler.transform(X))

    X_train_prepared = prepare(X_train)

    if multilabel:
        y_train_prepared = np.asarray(y_train)
    else:
        y_train_prepared = keras.utils.to_categorical(
            [LABELS.index(label) for label in y_train], num_classes=len(LABELS)
        )

    validation = None
    if X_valid is not None and y_valid is not None:
        if multilabel:
            y_valid_prepared = np.asarray(y_valid)
        else:
            y_valid_prepared = keras.utils.to_categorical(
                [LABELS.index(label) for label in y_valid], num_classes=len(LABELS)
            )
        validation = (prepare(X_valid), y_valid_prepared)

    network = keras_network(
        X_train_prepared.shape[1], multilabel=multilabel
    )
    network.fit(
        X_train_prepared,
        y_train_prepared,
        validation_data=validation,
        epochs=epochs,
        batch_size=128,
        verbose=0,
    )

    return network, scaler, pca


def predict_keras(network, scaler, pca, X, multilabel=False, threshold=0.5):
    """Apply a trained Keras network to new rows."""
    probabilities = network.predict(pca.transform(scaler.transform(X)), verbose=0)

    if multilabel:
        return (probabilities >= threshold).astype(int)
    return np.asarray([LABELS[i] for i in probabilities.argmax(axis=1)])


# --------------------------------------------------------------------------
# Comparing models
# --------------------------------------------------------------------------


def all_models(use_pca=False, multilabel=False):
    """The scikit-learn models entered into the comparison, by name."""
    return {
        "random forest": random_forest(use_pca, multilabel),
        "mlp": mlp(use_pca, multilabel),
        "svc": svc(use_pca, multilabel),
        "hard voting": hard_voting(use_pca, multilabel),
        "soft voting": soft_voting(use_pca, multilabel),
        "stacking": stacking(use_pca, multilabel),
    }


def compare_models(X, y, groups, use_pca=False, multilabel=False, n_folds=5):
    """Cross-validate every model and collect macro F1 scores.

    Macro F1 weights each drum equally, so a model that ignores the
    rarest instrument cannot hide behind the others. Plain accuracy is
    avoided: under multi-label targets scikit-learn scores it as exact
    set agreement, which penalises two correct instruments out of three
    as heavily as none.

    Returns a dict of model name to (mean, standard deviation).
    """
    from .splits import cv_splitter, safe_n_folds

    folds = safe_n_folds(groups, n_folds)
    splitter = cv_splitter(folds)
    scoring = "f1_macro" if not multilabel else "f1_samples"

    results = {}
    for name, model in all_models(use_pca, multilabel).items():
        if multilabel and name == "hard voting":
            continue  # hard voting cannot average indicator columns
        scores = cross_val_score(
            model, X, y, groups=groups, cv=splitter, scoring=scoring, n_jobs=-1
        )
        results[name] = (float(scores.mean()), float(scores.std()))

    return results
