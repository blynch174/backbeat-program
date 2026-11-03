"""Sanity-check src/loading.py against one IDMT-SMT-Drums file.

Usage, from the repo root:

    python scripts/check_loading.py <audio.wav> <annotation.xml>
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loading import load_audio, load_annotations

SIMULTANEOUS_MS = 20


def main(audio_path, annotation_path):
    samples, sr = load_audio(audio_path)
    hits = load_annotations(annotation_path)

    duration = len(samples) / sr

    print(f"audio        {Path(audio_path).name}")
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

    # The check that catches the units bug.
    last = hits[-1][0]
    print(f"last hit at  {last:.3f} s   (audio ends at {duration:.3f} s)")
    if last > duration:
        print("FAIL  last hit is past the end of the audio")
        print("      onsetSec is being read as samples or milliseconds")
        return 1
    print("OK    every hit falls inside the audio")
    print()

    # How often hits arrive together — contaminates windows, and the
    # onset detector will merge them.
    window = SIMULTANEOUS_MS / 1000
    times = [onset for onset, _ in hits]
    clustered = sum(
        any(abs(other - t) <= window for j, other in enumerate(times) if j != i)
        for i, t in enumerate(times)
    )

    pct = 100 * clustered / len(hits)
    print(f"hits with a neighbour within {SIMULTANEOUS_MS} ms:")
    print(f"  {clustered}/{len(hits)}  ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
