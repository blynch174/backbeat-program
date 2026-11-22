"""Survey the dataset and report the figures used in the write-up.

Counts the files, measures how often drums are struck simultaneously,
and checks how much of the annotated material onset detection recovers.

    python scripts/explore_data.py [--data data/idmt_smt_drums]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np

from src import features, loading, onsets, splits


def main(data_root, limit):
    data_root = Path(data_root)
    if not (data_root / "audio").is_dir():
        print(f"No audio/ directory under {data_root}")
        return 1

    print("=" * 62)
    print("FILE INVENTORY")
    print("=" * 62)

    kinds = Counter()
    sources = Counter()
    for path in loading.find_files(data_root):
        kind = loading.file_kind(path)
        kinds[kind] += 1
        if kind == "mix":
            sources[splits.source_id(path)] += 1

    print(f"  polyphonic loops (#MIX)        {kinds['mix']:>5}")
    print(f"  isolated exemplars (#train)    {kinds['train']:>5}")
    print(f"  separated tracks               {kinds['isolated']:>5}")
    print(f"  total                          {sum(kinds.values()):>5}")
    print()
    print("  loops by sound source")
    for source, count in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"    {source:<14}{count:>5}")

    mix_files = loading.find_files(data_root, "mix")[:limit]

    print()
    print("=" * 62)
    print("ANNOTATIONS")
    print("=" * 62)

    label_counts = Counter()
    simultaneity = []
    durations = []

    for path in mix_files:
        annotation = loading.annotation_for(path)
        if annotation is None:
            continue
        hits = loading.load_annotations(annotation)
        label_counts.update(label for _, label in hits)
        simultaneity.append(features.simultaneity_rate(hits))
        durations.append(hits[-1][0] if hits else 0.0)

    total_hits = sum(label_counts.values())
    print(f"  loops examined                 {len(mix_files):>5}")
    print(f"  annotated hits                 {total_hits:>5}")
    print()
    print("  class balance")
    for label in loading.LABELS:
        count = label_counts[label]
        share = 100 * count / total_hits if total_hits else 0
        print(f"    {label:<14}{count:>5}  ({share:4.1f}%)")

    print()
    print(f"  hits sharing their moment with another drum, within "
          f"{features.SIMULTANEITY_MS} ms:")
    print(f"    mean across loops            {100 * np.mean(simultaneity):>5.1f}%")
    print(f"    range                        "
          f"{100 * np.min(simultaneity):.1f}% - {100 * np.max(simultaneity):.1f}%")

    print()
    print("=" * 62)
    print("ONSET DETECTION")
    print("=" * 62)

    annotated = matched = spurious = 0
    errors = []

    for path in mix_files:
        annotation = loading.annotation_for(path)
        if annotation is None:
            continue
        hits = loading.load_annotations(annotation)
        samples, sr = loading.load_audio(path)
        detected = onsets.detect_onsets(samples, sr)
        summary = onsets.detection_summary(detected, hits)

        annotated += summary["annotated"]
        matched += summary["matched"]
        spurious += summary["spurious"]
        if summary["mean_error_ms"] is not None:
            errors.append(summary["mean_error_ms"])

    print(f"  annotated hits                 {annotated:>5}")
    print(f"  recovered by detection         {matched:>5}  "
          f"({100 * matched / annotated:.1f}%)")
    print(f"  detections with no annotation  {spurious:>5}")
    print(f"  mean timing error              {np.mean(errors):>5.1f} ms")
    print()
    print("  Recall is bounded above by simultaneity: two drums struck")
    print("  together fall inside one analysis frame and are reported as")
    print("  a single onset, so they can never both be recovered.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/idmt_smt_drums")
    parser.add_argument("--limit", type=int, default=None,
                        help="examine only the first N loops")
    args = parser.parse_args()
    sys.exit(main(args.data, args.limit))
