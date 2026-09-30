# SE-801 Term Project, Physics-Guided Models

Physics-guided micro-Doppler descriptors with classical learners, a small MLP,
and machine-checked reporting, for four-class small aerial target
classification.

Branch: `student/s1-physics`. Author: Ghulam Mustafa.

The integrated paper refers to this work as the **physics-guided feature
models** and does not use track labels. `s1` survives here in the branch name,
the directory name and the script prefixes because those are internal
coordination names and renaming them would break every stored path and every
Atlas job script. Nothing in the paper carries them.

## What this work does

Ten handcrafted descriptors are computed per 15 ms radar segment from raw I/Q,
then compared across an RBF-SVM, a Random Forest, XGBoost and a small MLP under
one frozen cross-validation protocol, with calibration, confidence, ablation,
importance, data-efficiency and robustness analyses. A separate sub-study asks
whether a language model can summarise the resulting numbers without inventing
or misreporting them, checked by a deterministic verifier.

The question the descriptors answer is how much of the attainable
discrimination survives reducing 1280 complex values to ten named physical
quantities, which quantities carry it, and what that reduction buys in
interpretability and inference cost.

## Dataset

Karlsson et al. (KTH), Zenodo **10.5281/zenodo.5845259**, CC-BY-4.0. SAAB SIRS
1600, 77 GHz FMCW, 160 MHz bandwidth, PRF 17 kHz. 130 parent measurements,
75,868 segments of 5 range cells by 256 slow-time samples; 75,801 after
excluding the 67 segments the authors flag as field-of-view edge.

Four coarse classes, mapping frozen in `class_mapping.json`: drone (44
measurements), bird (56), human (11), corner reflector (19).

The raw data is **not** in this repository. Download the master file from
Zenodo, then:

```bash
python convert_dataset.py --source path/to/data_SAAB_SIRS_77GHz_FMCW.npy
```

The project uses this corpus rather than the image set named in the original
written specification. The change was approved by Dr Salman Liaquat and the
reasoning is recorded in `docs/DATASET_DECISION.md`. The integrated paper does
not discuss the superseded dataset, on the supervisor's instruction, so that
record lives here and not in the manuscript.

## Frozen protocol

- Nested 5x5 cross-validation, **grouped by measurement**, seed 2026, outer
  fold seeds 2026 to 2030.
- `fold_manifest.csv` is **write-once**. Its sha256 must read
  `fbece6fb30a8d9e8468cac2ba89748d298d0b8debe54402ffc278302ad8fb7a6`
  and all four tracks share it. Never run `generate_manifest.py` again.
- Primary metric macro-F1; ECE with 15 bins, Brier score, batch-one latency
  (100 warm-up, 1,000 timed, repeated five times).
- Uncentred linear power, `|FFT|^2` divided by its own mean, summed over the
  five range cells. No log, no min-max, no bulk-Doppler centring.
- Zero-Doppler half-width 0.075 on the normalised axis (19 of 256 bins,
  +/-1.24 m/s), chosen by sweep: drone-versus-human rank AUC 0.814 at 0.075
  against 0.412 at 0.150.

## Reproducing, in order

```bash
python check_files.py                      # run this first, see below

python convert_dataset.py --source <zenodo master .npy>
python src/common/generate_manifest.py     # ONCE. Already frozen; do not re-run.
python src/common/preprocess_iq.py         # writes the power memmap
python src/s1_physics/feature_extractor.py # writes s1_features.csv
python src/s1_physics/validate_features.py # numerical gate, must pass
python src/s1_physics/zero_doppler_sweep.py

python src/s1_physics/train_classical.py --models svm rf xgb --latency-repeats 5
python src/s1_physics/train_mlp.py --demo overfit
python src/s1_physics/train_mlp.py --demo repro
python src/s1_physics/train_mlp.py
python src/s1_physics/measure_mlp_latency.py --also-cpu

python src/s1_physics/run_ablations.py --models svm xgb rf
python src/s1_physics/run_robustness.py --models svm xgb rf
python src/s1_physics/run_robustness.py --models mlp
python src/s1_physics/mlp_data_efficiency.py
python src/s1_physics/make_reliability.py

python src/s1_physics/analyze_results.py --all
python src/s1_physics/compare_models.py    # exports the paper tables

python src/s1_physics/llm_experiment.py --provider ollama --model llama3.2:3b
python -m pytest src/s1_physics/test_llm_verify.py -q   # 27 tests, 26 run, 1 skipped
```

There is no top-level `tests/` directory. The regression suite is the single
file `src/s1_physics/test_llm_verify.py`, so `pytest tests` will find nothing;
name the file.

### Checking the reported numbers without re-running anything

Every value the paper attributes to this work can be checked against the stored
files in a few seconds, with no training and no raw data:

```powershell
$env:S1_ARTIFACT_DIR = "C:\Projects\ANN_Project\results\atlas"
python src/s1_physics/verify_paper_numbers.py
```

The script holds each paper value next to the file and key path it must come
from, follows the paths, and prints one line per claim. The last full run
reported **121 verified, 0 wrong, 0 not found**. `--section` limits it to one
group, `--json` is machine-readable, and the exit code is non-zero on any
failure. `docs/RESULT_TRACEABILITY.md` is the same mapping in prose.

Claims derived from the out-of-fold probability files are skipped rather than
failed when those files are absent, since they are too large to commit.

`python check_files.py` verifies that all 21 tracked scripts are the current
revision before any run whose numbers reach the paper. It tests several marker
strings per file, including the command-line flags the Atlas job scripts pass,
because a single marker once let a stale `train_classical.py` pass as current
and the batch job died 19 seconds in.

### Paths and environments

`src/common/config.py` resolves every path. Local runs need nothing set. On
Atlas set `PROJECT_ENV=atlas`, and paths then come from `ATLAS_DATA_DIR`,
`ATLAS_WORK_DIR` and `ATLAS_ARTIFACT_DIR`; the job scripts in `atlas/` do this
for you.

`S1_DATA_DIR`, `S1_WORK_DIR` and `S1_ARTIFACT_DIR` override any of the three in
**either** environment. They exist so that a local machine can re-analyse the
Atlas results without pretending to be Atlas, which is how the reliability
diagrams, the MLP robustness run and the MLP data-efficiency curve were
produced after the cluster jobs finished:

```powershell
$env:S1_ARTIFACT_DIR = "C:\Projects\ANN_Project\results\atlas"
$env:S1_DATA_DIR     = "C:\Projects\ANN_Project\artifacts\work\zenodo"
```

The banner every script prints names which variable is in force, so a run
cannot quietly read the wrong directory.

## Layout

```
src/common/        config, manifest generation, preprocessing, corruption, metrics
src/s1_physics/    features, training, analysis, ablations, robustness, LLM study
atlas/             the SLURM job scripts that produced the reported numbers
results/atlas/     RESULTS OF RECORD: jobs 2305, 2308, 2309 on argon01
results/local_dev/ the local development run, kept for the reproduction claim
results/notes/     small evidence files
data/manifests/    manifests.zip (both manifests) plus the sha256 files
logs/              Atlas job logs, including the failed job 2238
tools/             one-off inspection helper
docs/              traceability record and the dataset decision
paper/             the report source, its figures and its generated tables
```

Results of record are `results/atlas/`. The local run is kept because the two
together are the evidence for the reproducibility finding below.

`paper/tables/*.tex` are written by `compare_models.py` from the stored result
files and are included verbatim, never retyped. `paper/figures/reliability.pdf`
is written by `make_reliability.py`. `paper/s1_draft_report.tex` is the
standalone track report; the integrated four-approach manuscript is maintained
separately in Overleaf.

Not in git, and regenerable: the raw measurements, the power memmap,
`s1_features.csv`, the out-of-fold probability files, and the Random Forest and
XGBoost model files (a 500-tree uncapped forest serialises to about 280 MB per
fold). The five MLP checkpoints are kept, since they are 20 KB each and make
the latency measurement repeatable. `reliability_bins.csv` is kept for the same
reason: the out-of-fold probabilities it was computed from are too large to
commit, so the binned numbers behind the figure stay in the repository even
though their input does not.

## Headline results, from `results/atlas/`

Segment-level macro-F1, mean and standard deviation across the five grouped
outer folds, on 75,801 segments.

| Model | Macro-F1 | Acc. | ECE | Brier | Latency (batch 1) | Train |
|---|---|---|---|---|---|---|
| Majority baseline | 0.2183 +/- 0.0007 | 0.775 | | | | |
| RBF-SVM | 0.8771 +/- 0.0248 | 0.9379 | 0.0281 | 0.1108 | 0.388 ms | 9m 09s |
| XGBoost | 0.8625 +/- 0.0369 | 0.9300 | 0.0258 | 0.1091 | 0.145 ms | 1m 30s |
| Random Forest | 0.8353 +/- 0.0396 | 0.9181 | 0.0213 | 0.1249 | 10.235 ms | 21m 06s |
| MLP (3,108 params) | 0.8294 +/- 0.0319 | 0.9106 | 0.0260 | 0.1332 | 0.023 ms CPU, 0.056 ms GPU | 49m 47s |

A paired sign-flip test over the five folds gives p = 0.25 for the SVM against
XGBoost and p = 0.0625 against the Random Forest and the MLP. With five folds
0.0625 is the smallest attainable value, so none of these differences is
separable at this sample size and the ordering is descriptive only.

Six findings worth reading the report for:

1. **Discrimination does not separate these four learners.** Their macro-F1
   values span 0.048, which is about twice the fold-to-fold spread of the best
   of them, and no pairwise difference reaches significance. Inference cost
   spans more than two orders of magnitude over the same set.
2. **The most accurate model is not the most trustworthy.** The Random Forest
   has the lowest ECE, XGBoost the best Brier score and by far the largest
   high-confidence operating point, 65.31 % of segments at p >= 0.99 with
   98.87 % accuracy, against 3.47 % for the SVM.
3. **A calibration number has to say how folds were combined.** Averaging ECE
   over folds gives 0.0281, 0.0258, 0.0213 and 0.0260 for SVM, XGBoost, Random
   Forest and MLP, while pooling all out-of-fold predictions into one set of
   bins gives 0.0241, 0.0212, 0.0157 and 0.0118. Both agree that the SVM is
   worst, but the MLP moves from third to first, with no prediction changed.
4. **Intermediate noise is worse than total noise.** At -10 dB the tree models
   sit at the majority baseline; at 0 dB they fall to 0.09, well below it,
   because shifted-but-informative features produce confident errors. The SVM
   and the MLP never cross the baseline.
5. **Two of the ten descriptors earn almost nothing here.** Dropping the whole
   temporal family costs 0.0094 macro-F1, because a 15 ms window is shorter
   than a gait cycle or a wingbeat. This is a property of the segmentation, not
   of the descriptors.
6. **Reproducibility is a property of a machine, not of a seed.** The three
   classical models reproduce to four decimal places across two machines and
   two scikit-learn versions; the MLP does not.

And one about the reporting itself: seven defects were found in the LLM claim
checker, six of which each changed a reported rate, three of those being false
negatives. Numerical fidelity at or above 0.976 was observed alongside
incorrect comparative claims, which is why `verify_paper_numbers.py` exists.

## Limitations

- Statistics rest on 130 measurements, not 75,801 segments. Only 11 human and
  19 corner reflector recordings exist, so two to three human parents fall in
  each outer test fold and the fold variance for that class is large.
- Every model selected a hyper-parameter at the edge of its mandated grid, so
  all four may be under-tuned.
- The ten descriptors were fixed before the folds were frozen and were not
  searched over. They are a defensible choice, not an optimised one.
- The zero-Doppler half-width was selected on development folds by a single
  criterion, the drone-against-human rank separation. A different criterion
  could have selected a different value.
- Perturbations are synthetic and applied after preprocessing. They are not a
  substitute for recordings made under degraded conditions, and the
  signal-to-noise ratios are signal-domain, not calibrated receiver SNR.
- Under Doppler-axis blur the two temporal descriptors are untouched by
  construction, since blur is applied to the power spectrum while they are
  computed from the I/Q.
- The dataset authors' own train/test indicator is not disjoint by measurement,
  so their published accuracy is not comparable with these numbers.
- Local environment scikit-learn 1.9.0 with torch 2.13.0; Atlas 1.8.0 with
  2.5.1. The MLP's cross-machine difference therefore has two candidate causes,
  device and library version, which one run each cannot separate. Measured
  per-fold differences run from -0.0014 to +0.0110, local minus cluster.
- The MLP robustness and data-efficiency curves were produced on the
  development machine, not on Atlas, because each point needs its own training
  run. The 100 % point agrees with the full nested run to 0.008.

## Further work

Not attempted, and no result from any of these appears in the paper:

- **A measured grouping comparison.** The argument that segment-wise
  partitioning rewards measurement memorisation is asserted rather than shown,
  because every reported number comes from the grouped side of it.
  `src/s1_physics/s1_grouping_ablation.py` scores one model on the frozen
  grouped folds against random stratified segment folds with hyper-parameters
  held fixed. It has not been run.
- **Parent-level rescoring of this branch.** The four approaches are compared
  across different scoring units. `src/s1_physics/s1_parent_level.py` averages
  the stored out-of-fold probabilities within each measurement and scores over
  the 130 parents, which would put this branch in the same unit as the CNN
  branch without retraining. It has not been run.
- **A compact five-descriptor subset chosen without touching the outer folds.**
  The two compact subsets in `results_ablations.json` were selected using
  information from the full fits, so they are diagnostics rather than a
  nested-selection result.

Both scripts are committed, documented and import cleanly. Neither writes a
file that anything else reads, so nothing in the reported results depends on
them.

## Supervisors

Dr Raja Farrukh Ali (ML), Dr Salman Liaquat (radar signal processing),
Shaiq-e-Mustafa (dataset and class mapping).
