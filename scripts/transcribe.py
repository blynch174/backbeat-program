"""Transcribe a drum recording with a trained model.

Reads an audio file, finds the hits, names the drum behind each one,
and prints the result as a time-ordered transcription. With --midi it
also writes a MIDI file that notation software such as MuseScore can
open as a drum chart.

    python scripts/transcribe.py recording.wav
    python scripts/transcribe.py recording.wav --midi out.mid
    python scripts/transcribe.py recording.wav --compare annotation.xml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import joblib
import numpy as np

from src import evaluation, features, loading, onsets

# General MIDI percussion note numbers, so the output opens correctly
# in notation software.
MIDI_NOTES = {"kick": 36, "snare": 38, "hihat": 42}

# Single-character marks used by the printed grid.
MARKS = {"kick": "O", "snare": "X", "hihat": "x"}


def transcribe(audio_path, bundle):
    """Return a list of (time_seconds, label) for one recording."""
    samples, sr = loading.load_audio(audio_path)
    times = onsets.detect_onsets(samples, sr)

    if len(times) == 0:
        return []

    rows = np.asarray(
        [
            features.featurize(
                features.extract_window(samples, sr, t), sr, bundle["representation"]
            )
            for t in times
        ]
    )

    predictions = bundle["model"].predict(rows)

    events = []
    for time, prediction in zip(times, predictions):
        if bundle["multilabel"]:
            labels = [
                bundle["labels"][i]
                for i, present in enumerate(prediction)
                if present
            ]
            if labels:
                events.append((float(time), set(labels)))
        else:
            events.append((float(time), str(prediction)))

    return events


def print_transcription(events):
    """Print each hit with its timestamp."""
    print(f"{'time':>9}  drum")
    print("-" * 28)
    for time, label in events:
        names = label if isinstance(label, str) else "+".join(sorted(label))
        print(f"{time:>9.3f}  {names}")


def print_grid(events, resolution=0.125):
    """Print a coarse grid showing where each drum falls over time.

    Rows are instruments, columns are time steps. Intended as a quick
    visual check rather than as notation.
    """
    if not events:
        return

    end = max(t for t, _ in events) + resolution
    n_steps = int(end / resolution) + 1
    lanes = {label: [" "] * n_steps for label in loading.LABELS}

    for time, label in events:
        names = {label} if isinstance(label, str) else label
        step = int(round(time / resolution))
        for name in names:
            if name in lanes and step < n_steps:
                lanes[name][step] = MARKS[name]

    print()
    print(f"grid, one column per {int(resolution * 1000)} ms")
    for label in loading.LABELS:
        print(f"  {label:<6}|{''.join(lanes[label])}|")


def write_midi(events, path, tempo_bpm=120):
    """Write the transcription to a MIDI drum track."""
    import mido

    ticks_per_beat = 480
    seconds_per_beat = 60 / tempo_bpm

    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)))

    messages = []
    for time, label in events:
        names = {label} if isinstance(label, str) else label
        tick = int(round(time / seconds_per_beat * ticks_per_beat))
        for name in names:
            note = MIDI_NOTES[name]
            messages.append((tick, "note_on", note))
            messages.append((tick + 30, "note_off", note))

    messages.sort()

    previous = 0
    for tick, kind, note in messages:
        track.append(
            mido.Message(kind, note=note, velocity=90, channel=9, time=tick - previous)
        )
        previous = tick

    midi.save(str(path))


def main(args):
    bundle = joblib.load(args.model)
    print(f"model: {bundle['name']} ({bundle['representation']} features)")
    print(f"audio: {Path(args.audio).name}")
    print()

    events = transcribe(args.audio, bundle)
    if not events:
        print("No drum hits detected.")
        return 1

    print_transcription(events)
    if args.grid:
        print_grid(events)

    counts = {label: 0 for label in loading.LABELS}
    for _, label in events:
        for name in ({label} if isinstance(label, str) else label):
            counts[name] += 1

    print()
    print(f"{len(events)} hits  " + "  ".join(f"{k} {v}" for k, v in counts.items()))

    if args.compare:
        actual = loading.load_annotations(args.compare)
        print()
        print("=" * 48)
        print("COMPARED WITH ANNOTATION")
        print("=" * 48)
        for ms, scores in evaluation.score_events_all_tolerances(events, actual).items():
            print(evaluation.format_scores(scores, title=f"\ntolerance {ms} ms"))

    if args.midi:
        write_midi(events, args.midi)
        print(f"\nwrote {args.midi}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="path to a WAV file")
    parser.add_argument("--model", default="models/backbeat.joblib")
    parser.add_argument("--midi", help="also write a MIDI file here")
    parser.add_argument("--compare", help="score against an XML annotation")
    parser.add_argument("--grid", action="store_true",
                        help="print a coarse time grid")
    sys.exit(main(parser.parse_args()))
