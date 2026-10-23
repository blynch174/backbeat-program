# Data

Audio datasets are large, so they are gitignored. They live in **this folder**
on each of our machines, but never get committed. Each of us downloads them
separately.

Expected layout once downloaded:

```
data/
  idmt_smt_drums/
    audio/
    annotation_xml/
    annotation_svl/
  mdb_drums/
    audio/
    annotations/
```

## IDMT-SMT-Drums (primary training set)

Direct download from Zenodo, no request form:
https://zenodo.org/records/7544164

`IDMT-SMT-DRUMS-V2.zip` — 287 MB, 608 WAV files, about 2h10m of audio.
License: CC BY-NC-ND 4.0 (non-commercial, no derivatives — fine for coursework,
cite it).

```bash
cd data
curl -L -o idmt.zip https://zenodo.org/records/7544164/files/IDMT-SMT-DRUMS-V2.zip
unzip idmt.zip -d idmt_smt_drums
rm idmt.zip
```

## MDB Drums (real-world test set)

Annotations *and* audio are both in the GitHub repo — drum-only stems, full
mixes, and original multitracks for 23 MedleyDB tracks, 7,994 annotated onsets.
License: CC BY-NC-SA 4.0.

```bash
cd data
git clone https://github.com/CarlSouthall/MDBDrums.git mdb_drums
```

Note: this clone is large because of the multitrack WAVs. If it's too big, grab
just the annotations and the drum-only audio through the GitHub web UI.

Two annotation levels: `class` (6 drum instruments) and `subclass` (21
instrument + technique combinations). We want **class**, then collapse it down
to kick / snare / hi-hat.

## E-GMD (only if we scale up)

https://magenta.withgoogle.com/datasets/e-gmd — hundreds of hours of human
drumming with audio aligned to MIDI. Large. Don't download this until we've
proven the pipeline works on IDMT.


### Format notes

(fill in — annotation column layout, sample rates, any files that fail to load)