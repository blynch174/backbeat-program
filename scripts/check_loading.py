"""Verify the loaders against a single file before trusting them.

Catches the two failures that are otherwise silent: annotation times
read in the wrong units, and a parser that matches no tags and returns
an empty list.

    python scripts/check_loading.py audio.wav annotation.xml
    python scripts/check_loading.py audio.wav          (finds the annotation)
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src import features, loading, splits


def main(audio_path, annotation_path=None):
    audio_path = Path(audio_path)

    if annotation_path is None:
        annotation_path = loading.annotation_for(audio_path)
        if annotation_path is None:
            print(f"No annotation found for {audio_path.name}")
            print("Mix loops are the annotated files; isolated recordings are not.")
            return 1

    samples, sr = loading.load_audio(audio_path)
    hits = loading.load_annotations(annotation_path)
    duration = len(samples) / sr

    print(f"audio        {audio_path.name}")
    print(f"kind         {loading.file_kind(audio_path)}")
    print(f"kit          {splits.kit_id(audio_path)}")
    print(f"source       {splits.source_id(audio_path)}")
    print(f"sample rate  {sr} Hz")
    print(f"duration     {duration:.3f} s  ({len(samples)} samples)")
    print()

    if not hits:
        print("FAIL  no hits parsed")
        print("      check the tag names inside load_annotations()")
        return 1

    print(f"hits         {len(hits)}")
    print(f"labels       {dict(Counter(label for _, label in hits))}")
    print()
    print("first 5:")
    for onset, label in hits[:5]:
        print(f"  {onset:9.5f}  {label}")
    print()

    last = hits[-1][0]
    print(f"last hit at  {last:.3f} s   (audio ends at {duration:.3f} s)")
    if last > duration:
        print("FAIL  last hit is past the end of the audio")
        print("      onset times are being read as samples or milliseconds")
        return 1
    print("OK    every hit falls inside the audio")
    print()

    rate = features.simultaneity_rate(hits)
    simultaneous = round(rate * len(hits))
    print(f"hits sharing their moment, within {features.SIMULTANEITY_MS} ms:")
    print(f"  {simultaneous}/{len(hits)}  ({100 * rate:.1f}%)")

    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
