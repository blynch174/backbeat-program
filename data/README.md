# Data

Audio is large and licensed, so it is gitignored. It lives in this
folder on each machine and is downloaded separately.

Expected layout:

```
data/idmt_smt_drums/
  audio/
  annotation_xml/
  annotation_svl/
```

## IDMT-SMT-Drums

https://zenodo.org/records/7544164 — `IDMT-SMT-DRUMS-V2.zip`, 287 MB,
608 WAV files, about 2 hours 10 minutes. CC BY-NC-ND 4.0.

```bash
cd data
curl -L -o idmt.zip https://zenodo.org/records/7544164/files/IDMT-SMT-DRUMS-V2.zip
md5 idmt.zip        # expect d2664b4c2aaa34b90ba2f57b389c5663
unzip -q idmt.zip -d idmt_smt_drums
rm idmt.zip
```

Verify the extraction — macOS Archive Utility silently truncates large
archives, so extract from the terminal and check the counts:

```bash
find idmt_smt_drums -name '*.wav'     | wc -l   # expect 608
find idmt_smt_drums -name '*MIX.wav'  | wc -l   # expect 104
```

Then confirm the loaders agree:

```bash
python scripts/explore_data.py
```

## Future work

[MDB Drums](https://github.com/CarlSouthall/MDBDrums) — 23 real musical
performances, 7,994 onsets across 6 classes, with the MIREX 2017
train/test splits included. Its KD/SD/HH codes match the ones used
here; TT, CY and OT map to None in `src.loading.LABEL_MAP` and are
discarded.
