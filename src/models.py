"""Classifiers that label a hit as kick, snare, or hi-hat.

Owner: TBD
"""

LABELS = ["kick", "snare", "hihat"]


LABELS = ["kick", "snare", "hihat"]
 
RANDOM_STATE = 42
PCA_VARIANCE = 0.95
 
 
# Pipelines
 
 
def build_pipeline(classifier, use_pca=False):
    """Wrap a classifier with its preprocessing.
 
    Steps: StandardScaler, then PCA(n_components=PCA_VARIANCE) when
    use_pca is set (for the spectrogram representation), then the
    classifier.
 
    Returns an unfitted sklearn Pipeline.
    """
    raise NotImplementedError
 
 
# Individual models
 
 
def random_forest(use_pca=False):
    """RandomForestClassifier in a pipeline."""
    raise NotImplementedError
 
 
def mlp(use_pca=False):
    """MLPClassifier in a pipeline.
 
    Raise max_iter above the default — 200 hits the iteration cap and
    throws a ConvergenceWarning.
    """
    raise NotImplementedError
 
 
def svc(use_pca=False):
    """SVC in a pipeline. probability=True is required for soft voting."""
    raise NotImplementedError
 
 
# Ensembles
 
 
def soft_voting(use_pca=False):
    """VotingClassifier(voting="soft") over the three models above."""
    raise NotImplementedError
 
 
def stacking(use_pca=False):
    """StackingClassifier over the three models, random forest blender.
 
    Pass the group-aware splitter from src/splits.py as cv= so the
    blender's training data doesn't leak across recordings either.
    """
    raise NotImplementedError
 
 
# Stretch goal
 
 
def train_cnn(X, y):
    """Small convolutional net over 2D mel spectrograms.
 
    Only if the ensembles are done and reported first.
    """
    raise NotImplementedError
 
 
# Running the comparison
 
 
def compare_models(X, y, groups, use_pca=False):
    """Cross-validate every model and collect the scores.
 
    Use cross_val_score with the group-aware splitter and macro F1 as
    the scoring metric — not accuracy, the classes are imbalanced.
 
    Returns a table of model name -> mean and std of the fold scores.
    """
    raise NotImplementedError
 
 
def predict(model, X):
    """Return predicted labels."""
    raise NotImplementedError