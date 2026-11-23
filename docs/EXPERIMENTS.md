# Experiment log

What was tried and why

All figures use macro F1 unless stated. Evaluation is on the annotated
polyphonic loops, which are harder than the isolated recordings the
classifier trains on.

Two evaluation settings appear throughout:

- **all kits** — trained on every isolated recording, tested on every
  mix loop. Since a kit's isolated files and its mix loop are the same
  drums, the model has heard every kit it is tested on. This matches
  the dataset's intended workflow and measures kit-adapted
  transcription.
- **held-out kits** — 24 of the 95 kits reserved entirely for testing.
  Nothing tested was seen in training. This is the honest
  generalisation figure.

---

## 1. Dataset survey

**What:** counted the files, measured how often drums are struck
together, and checked what onset detection recovers.

**Result:**

| | |
|---|---|
| annotated loops | 95 |
| isolated exemplars | 285 |
| annotated hits | 7,927 |
| class balance | hi-hat 53.6%, kick 27.5%, snare 18.9% |
| hits sharing their moment (within 20 ms) | **53.2%** |

**Why it mattered:** the published description claims 104 loops and 608
files; this release contains 95 and 560. So, reported figures come from the
actual data and not the webpage.

The simultaneity figure set the direction for everything that followed.
Over half of all hits overlap with another drum, and per-loop rates
range from 0% to 100%, so difficulty varies enormously across the test
set.

---

## 2. Onset detection

**What:** librosa's percussive onset detector, with a threshold sweep.

**Result:** 66.7% of annotated hits recovered, 49 spurious detections
across all 95 loops, mean timing error **4.5 ms**.

**Why it worked, and why it stopped there:** precision is near perfect —
about one false alarm every two files. Timing is accurate enough that
event-level scores are identical at 50 ms, 25 ms and 10 ms tolerance,
which most published work does not report (most only go to 50 ms).

Recall did not improve with threshold tuning. Two drums struck together
fall inside one analysis frame (~11.6 ms at 44.1 kHz) and are reported
as a single onset. Given 53.2% simultaneity, roughly 5,800 of the 7,927
hits are reachable at all, and detection found 5,286 — about 91% of the
ceiling. **Detection was not the bottleneck, and was left alone.**

---

## 3. Single-label classification, hand-designed features

**What:** 31 scalar descriptors per hit — spectral centroid, rolloff,
bandwidth, zero-crossing rate, RMS, and 13 MFCCs with their standard
deviations. One label per window, three mutually exclusive classes.

**Result:**

| | macro F1 |
|---|---|
| cross-validation, isolated hits | 0.983 |
| held-out, all kits | 0.668 |
| held-out kits | 0.674 |

**Why the gap:** recognising a drum alone is nearly solved; transcribing
drumming is not. Two distribution shifts occur at once — windows go
from containing one drum to containing several, and the class mix moves
from roughly balanced to hi-hat heavy.

The confusion matrix located the failure precisely:

```
             kick     snare     hihat
kick         1367       194       618
snare           8       800       691
hihat         437       383      3429
```

- 618 kicks and 691 snares were labelled hi-hat — **1,309 hits lost in
  one direction**.
- 46% of all snares became hi-hat.
- Only **8** snares were called kick. Kick and snare essentially never
  confuse with each other.

That last point was the useful one. The features separate drums that do
not overlap perfectly well; the errors are specifically contamination
from a hi-hat sharing the window. The problem was not because of the 
features. It was because of the output formulation. 

---

## 4. Ensemble methods

**What:** hard voting, soft voting, and stacking over random forest,
MLP and SVC, following Géron ch. 7.

**Result, cross-validated on training material:**

| model | macro F1 | std |
|---|---|---|
| SVC | 0.983 | 0.012 |
| hard voting | 0.980 | 0.012 |
| random forest | 0.973 | 0.018 |
| soft voting | 0.966 | 0.033 |
| MLP | 0.964 | 0.036 |
| stacking | 0.956 | 0.041 |

**Why it failed:** no ensemble beat the best single model. Voting and
stacking cancel *uncorrelated* errors; here all three classifiers sit
near ceiling on clean isolated hits and fail on the same small set of
ambiguous windows, so there is nothing to cancel and averaging lets the
weaker members drag the strongest down. Stacking fared worst and was
also the least stable (std 0.041) — a blender trained on near-identical
inputs has little to learn and overfits the fold.

Ensembles became useful later, once the training data contained genuine
variety (see §7).

---

## 5. Keras network

**What:** a dense network on PCA-reduced inputs, following Chollet
ch. 2 — two hidden ReLU layers, softmax output, categorical
cross-entropy.

**Result:** 0.597, against 0.668 for scikit-learn's SVC on identical
features.

**Why it underperformed:** 2,322 training rows across 31 features is
small and tabular, which is where kernel methods and tree ensembles are
strongest. A network has no structural advantage until there is either
much more data or genuine spatial structure to exploit, and a dense
network on flattened features discards the time-frequency layout that
would give it one.

---

## 6. Multi-label classification

**What:** replaced the three-way exclusive choice with three
independent yes/no detectors, so a simultaneous kick and hi-hat becomes
`[1, 0, 1]`. Same features, same models, `MultiOutputClassifier`
wrapping each.

**Result:**

| | macro F1 | both drums recovered |
|---|---|---|
| all kits | 0.563 | **45 / 3,929 (1.1%)** |
| held-out kits | 0.707 | 12 / 516 (2.3%) |

**Why it failed:** the formulation could now represent simultaneous
hits and almost never did. Precision was extremely high and recall very
low — the kick detector was right 98% of the time it fired, and fired
on under a third of the kicks present.

The cause was the training distribution. Each detector learned from
isolated recordings only, so it had never encountered a kick with a
hi-hat layered over it and stayed silent when one arrived. Making the
output capable of expressing overlap achieves nothing if the model has
never been shown overlap.

**Fixed along the way:** building one window per annotated *hit* meant a
simultaneous pair produced two windows 3–6 ms apart, overlapping by 94%,
carrying identical targets — the same musical moment counted twice.
Annotations are now grouped into events first, which reduced the
evaluation set from 7,927 rows to 5,952.

---

## 7. Synthesized mixtures

**What:** built artificial simultaneous hits by summing isolated
windows from the same drum kit, with random per-component gain and
peak limiting so the synthetic rows are not simply louder than real
ones. Added one synthetic mixture per real window, giving a training
set that is 50% overlapping hits. IDMT ships perfectly separated tracks
specifically so mixtures can be rebuilt this way.

**Result:**

| | macro F1 | both drums recovered |
|---|---|---|
| all kits | 0.860 | 1,842 / 1,952 (94.4%) |
| held-out kits | **0.866** | **485 / 516 (94.0%)** |

Per-drum, held-out kits:

| drum | precision | recall | F1 |
|---|---|---|---|
| kick | 0.738 | 0.956 | 0.833 |
| snare | 0.732 | 0.978 | 0.838 |
| hi-hat | 0.877 | 0.986 | 0.928 |

**Why it worked:** recall rose from 0.65 to 0.96 for kick and from 0.43
to 0.98 for snare, at a modest cost in precision. Having seen
contaminated examples during training, the detectors now fire when a
drum is present underneath another. Windows where neither drum was
recovered fell to **zero**.

Soft voting won this configuration, having lost every earlier one.
Once the training data contained real variation, the base classifiers
began making different mistakes, which is the condition ensembles need.

**The caveat:** synthetic mixtures are sums of separately recorded hits.
Real simultaneous strokes involve one drum's resonance exciting
another, microphone bleed, and a single room response. The gain between
0.866 and 0.674 is real but is measured against mixtures partly built
the same way the training data was.

---

## 8. Feature representation

**What:** compared the 31 hand-designed descriptors against a flattened
mel spectrogram (64 bands, 2,240 values per window) reduced by PCA at
95% explained variance inside the pipeline, following the
dimensionality reduction step in the course assignments.

**Result, held-out kits:**

| representation | single-label | multi-label + synthesis |
|---|---|---|
| hand-designed, 31 features | **0.674** | 0.866 |
| mel spectrogram + PCA | 0.641 | **0.907** |

**Why the ranking flips:** under single-label targets the compact
hand-designed features win. The spectrogram carries far more
information, but the single-label formulation throws most of it away —
it must still answer with one drum, so the extra detail describing the
second drum in the window is wasted and simply adds noise. Snare
precision fell to 0.389, the worst figure recorded in any run.

Once the targets can express overlap and the training data contains
overlap, the extra detail becomes usable and the spectrogram pulls
ahead. Snare precision rose to 0.885 against 0.732 for the
hand-designed features — the largest single improvement on the hardest
drum in the project.

**The lesson:** a richer representation only pays off when the output
formulation and the training distribution can make use of it. Tested in
isolation under the original setup, the spectrogram would have been
dismissed as worse.

---

## Summary

Held-out kits, macro F1:

| configuration | hand-designed | spectrogram + PCA |
|---|---|---|
| single-label | 0.674 | 0.641 |
| multi-label, isolated training only | 0.707 | — |
| **multi-label + synthesized mixtures** | 0.866 | **0.907** |
| Keras network, single-label | 0.597 | — |

Best result: **0.907 macro F1**, soft voting over random forest, MLP and
SVC, on PCA-reduced mel spectrograms, with multi-label targets and
synthesized mixtures in training. Both drums recovered in 500 of 516
simultaneous windows (96.9%); neither drum recovered in zero.

The arc in four steps:

1. Single-label cannot represent a kick and hi-hat struck together, and
   loses 1,309 hits to hi-hat because of it.
2. Multi-label can represent them, but a model trained only on isolated
   drums does not recognise overlapping ones — both recovered 1.1% of
   the time.
3. Training on synthesized mixtures supplies the missing examples, and
   both drums are recovered 94% of the time.
4. With overlap representable and present in training, the richer
   spectrogram representation finally beats the hand-designed one.

Held-out-kit figures track the all-kit figures closely, so the result
does not depend on having heard the test kits.

Three findings contradicted the expected result and are worth stating
plainly: ensembles lost to a single SVC until the training data gained
variety, the Keras network lost to classical models on small tabular
data, and the spectrogram representation lost until the formulation
could exploit it.

---

## Not yet run

- **Cross-source generalisation** — train on two of RealDrum,
  WaveDrum and TechnoDrum, test on the third. `splits.split_by_source`
  is implemented but no script calls it. Note the imbalance: 70, 14 and
  11 loops respectively.
- **Manual stacking** — `models.manual_stacking_features` reproduces the
  by-hand blender construction from the coursework; unused so far.
- **MDB Drums** — drums inside real musical performances rather than
  isolated loops. Scoped to future work.
