"""Train the drum classifier and compare the candidate models.

Training rows come from the isolated single-instrument recordings,
where every window contains exactly one drum and the label is given by
the filename. Evaluation rows come from the annotated polyphonic loops,
which are harder because drums overlap there.

    python scripts/train.py
    python scripts/train.py --representation spectrogram
    python scripts/train.py --multilabel
    python scripts/train.py --quick            (small subset, for a fast check)

The chosen model is written to models/backbeat.joblib for use by
scripts/transcribe.py.
"""

from __future__ import annotations

import argparse
import sys
import warnings
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import joblib
import numpy as np

from src import evaluation, features, loading, models, splits


def main(args):
    data_root = Path(args.data)
    started = time.time()

    train_files = loading.find_files(data_root, "train")
    mix_files = loading.find_files(data_root, "mix")

    if args.quick:
        train_files = train_files[: args.quick]
        mix_files = mix_files[: max(4, args.quick // 8)]

    print(f"training files {len(train_files)}   evaluation loops {len(mix_files)}")
    print(f"representation {args.representation}   "
          f"targets {'multi-label' if args.multilabel else 'single-label'}")
    print()

    print("building training table from isolated recordings ...")
    X_train, y_train, groups_train = features.build_training_set(
        train_files, representation=args.representation
    )
    print(f"  {X_train.shape[0]} rows x {X_train.shape[1]} features")
    print(f"  {len(np.unique(groups_train))} distinct kits")
    counts = {label: int((y_train == label).sum()) for label in loading.LABELS}
    print(f"  class counts {counts}")
    print()

    print("building evaluation table from annotated loops ...")
    X_test, y_test, groups_test = features.build_evaluation_set(
        mix_files,
        representation=args.representation,
        multilabel=args.multilabel,
        exclude_simultaneous=args.exclude_simultaneous,
    )
    print(f"  {X_test.shape[0]} rows")
    print()

    use_pca = args.representation == "spectrogram"

    print("=" * 62)
    print("CROSS-VALIDATED MODEL COMPARISON (on training material)")
    print("=" * 62)
    print(f"{'model':<16}{'macro F1':>12}{'std':>10}")

    if args.multilabel:
        y_train_cv = np.array(
            [features.window_multilabel(0, [(0, label)]) for label in y_train]
        )
    else:
        y_train_cv = y_train

    results = models.compare_models(
        X_train, y_train_cv, groups_train,
        use_pca=use_pca, multilabel=args.multilabel, n_folds=args.folds,
    )
    for name, (mean, std) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:<16}{mean:>12.3f}{std:>10.3f}")

    best_name = max(results, key=lambda k: results[k][0])
    print(f"\nbest by cross-validation: {best_name}")
    print()

    print("=" * 62)
    print("HELD-OUT EVALUATION (annotated polyphonic loops)")
    print("=" * 62)

    best = models.all_models(use_pca, args.multilabel)[best_name]
    best.fit(X_train, y_train_cv)
    predictions = best.predict(X_test)

    scores = evaluation.score_classification(
        y_test, predictions, multilabel=args.multilabel
    )
    print(evaluation.format_scores(scores))
    print()

    if args.multilabel:
        breakdown = evaluation.cooccurrence_breakdown(y_test, predictions)
        print("simultaneous-hit windows")
        for key, value in breakdown.items():
            print(f"  {key:<20}{value:>6}")
    else:
        print("confusion matrix (rows true, columns predicted)")
        print(evaluation.format_confusion(
            evaluation.confusion(y_test, predictions)
        ))

    if args.keras:
        print()
        print("=" * 62)
        print("KERAS NETWORK")
        print("=" * 62)
        network, scaler, pca = models.fit_keras(
            X_train, y_train_cv, multilabel=args.multilabel, epochs=args.epochs
        )
        keras_pred = models.predict_keras(
            network, scaler, pca, X_test, multilabel=args.multilabel
        )
        keras_scores = evaluation.score_classification(
            y_test, keras_pred, multilabel=args.multilabel
        )
        print(evaluation.format_scores(keras_scores))

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best,
            "representation": args.representation,
            "multilabel": args.multilabel,
            "labels": loading.LABELS,
            "name": best_name,
        },
        args.out,
    )

    print()
    print(f"saved {best_name} to {args.out}")
    print(f"elapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/idmt_smt_drums")
    parser.add_argument("--representation", choices=["summary", "spectrogram"],
                        default="summary")
    parser.add_argument("--multilabel", action="store_true")
    parser.add_argument("--exclude-simultaneous", action="store_true",
                        help="drop overlapping hits from the evaluation set")
    parser.add_argument("--keras", action="store_true",
                        help="also train the Keras network")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--quick", type=int, default=0,
                        help="use only the first N training files")
    parser.add_argument("--out", default="models/backbeat.joblib")
    sys.exit(main(parser.parse_args()))
