# Dataset decision

Internal record. This is deliberately not in the manuscript: the supervisor's
instruction for the integrated paper is that the superseded dataset is not to be
discussed there. The reasoning is kept here so the decision stays auditable.

## What changed

The original written specification fixed an image dataset of micro-Doppler
spectrograms. The project instead uses the Karlsson et al. 77 GHz FMCW raw I/Q
corpus, Zenodo 10.5281/zenodo.5845259, and derives its own representations from
the complex range-compressed data.

Approved by Dr Salman Liaquat.

## Why

The images in the specified set are overlapping frames of continuous
recordings. Measured frame-to-frame cosine similarity is 0.94 to 0.99 for
consecutive frames, against 0.68 to 0.85 for frames drawn from distant parts of
the same recording. A random split over such frames therefore places near
duplicates of the same physical moment on both sides of the split, and any
classifier can reach a high score by recognising the recording rather than the
target. There is no per-recording identifier distributed with the set, so the
leak cannot be closed by regrouping.

The FMCW corpus carries 130 identifiable parent measurements. Grouping by
parent is possible, which makes a measurement-disjoint protocol possible, and
that protocol is the premise of the whole study.

## Consequences that follow from the change

- Six original labels were mapped onto four operational classes, because
  several labels are carried by fewer than five parent measurements and cannot
  populate five grouped folds. The mapping is frozen in `class_mapping.json`.
- Image preprocessing was replaced by I/Q signal processing. Each branch of the
  wider study derives its own representation from the same complex data, which
  is what makes the four representations comparable in input even where they
  are not comparable in scoring unit.
- The dataset authors' own train/test indicator is not disjoint by measurement.
  Their published accuracy is therefore not comparable with the numbers
  reported here, and it is not cited as a baseline.

## Where this is reflected

The limitations section of the manuscript states that statistics rest on 130
measurements rather than 75,801 segments, and that the authors' own split
indicator is not measurement-disjoint. Neither statement names the superseded
dataset.
