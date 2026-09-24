# SE-801 Term Project, Track S1

Physics-guided micro-Doppler descriptors with classical learners, an MLP, and
verified LLM reporting, for four-class small aerial target classification.

Branch: `student/s1-physics`. Student: Ghulam Mustafa (S1).

## What this track does

Ten handcrafted descriptors are computed per 15 ms radar segment from raw I/Q,
then compared across an RBF-SVM, Random Forest, XGBoost and a small MLP under
one frozen cross-validation protocol, with calibration, ablation, importance,
data-efficiency and robustness analyses. A separate sub-study asks whether an
LLM can summarise the resulting numbers without inventing or misreporting
them, checked by a deterministic verifier.

## Dataset

Karlsson et al. (KTH), Zenodo **10.5281/zenodo.5845259**, CC-BY-4.0. SAAB SIRS
1600, 77 GHz FMCW, 160 MHz bandwidth, PRF 17 kHz. 130 measurements, 75,868
segments of 5 range cells by 256 slow-time samples; 75,801 after excluding the
67 segments the authors flag as field-of-view edge.

Four coarse classes, mapping frozen in `class_mapping.json`: drone (44
measurements), human (11), bird (56), corner reflector (19).

The raw data is **not** in this repository. Download the master file from
Zenodo, then:

```bash
python convert_dataset.py --source path/to/data_SAAB_SIRS_77GHz_FMCW.npy
```

This project departs from the written specification, which fixes DIAT-uSAT:
that dataset's images were found to be overlapping frames of continuous
recordings (consecutive-frame similarity 0.94 to 0.99 against 0.68 to 0.85 for
distant frames), so any random split leaks. Dr Salman Liaquat approved the move
to raw radar data. The deviations are listed in the report.

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
python convert_dataset.py --source <zenodo master .npy>
python src/common/generate_manifest.py     # ONCE. Already frozen; do not re-run.
python src/common/preprocess_iq.py         # writes the power memmap
python src/s1_physics/feature_extractor.py # writes s1_features.csv
python src/s1_physics/validate_features.py # numerical gate, must pass
python src/s1_physics/train_classical.py --models svm rf xgb --latency-repeats 5
python src/s1_physics/train_mlp.py --demo overfit
python src/s1_physics/train_mlp.py --demo repro
python src/s1_physics/train_mlp.py
python src/s1_physics/measure_mlp_latency.py --also-cpu
python src/s1_physics/run_ablations.py --models svm xgb rf
python src/s1_physics/run_robustness.py --models svm xgb rf
python src/s1_physics/analyze_results.py
python src/s1_physics/compare_models.py    # exports the paper tables
python src/s1_physics/llm_experiment.py --provider ollama --model llama3.2:3b
pytest src/s1_physics/test_llm_verify.py -q
```

`python check_files.py` verifies that all 18 scripts are the current revision
before any run whose numbers reach the paper. Run it first. It tests several
marker strings per file, including the command-line flags the Atlas job scripts
pass, because a single marker once let a stale `train_classical.py` pass as
current and the batch job died 19 seconds in.

On Atlas set `PROJECT_ENV=atlas`; paths then come from `ATLAS_DATA_DIR`,
`ATLAS_WORK_DIR` and `ATLAS_ARTIFACT_DIR` (see `src/common/config.py`). The job
scripts in `atlas/` do this for you.

## Layout

```
src/common/        config, manifest generation, preprocessing, corruption, metrics
src/s1_physics/    features, training, analysis, ablations, robustness, LLM study
atlas/             the five SLURM job scripts that produced the reported numbers
results/atlas/     RESULTS OF RECORD: jobs 2305, 2308, 2309 on argon01
results/local_dev/ the local development run, kept for the reproduction claim
results/notes/     small evidence files
data/manifests/    manifests.zip (both manifests) plus the sha256 files
logs/              Atlas job logs, including the failed job 2238
tools/             one-off inspection helper
paper/             the report source
```

Results of record are `results/atlas/`. The local run is kept because the two
together are the evidence for the reproducibility finding below.

Not in git, and regenerable: the raw measurements, the power memmap,
`s1_features.csv`, the out-of-fold probability files, and the Random Forest and
XGBoost model files (a 500-tree uncapped forest serialises to about 280 MB per
fold). The five MLP checkpoints are kept, since they are 20 KB each and make
the latency measurement repeatable.

## Headline results, from `results/atlas/`

| Model | Macro-F1 | Acc. | ECE | Brier | Latency (batch 1) | Train |
|---|---|---|---|---|---|---|
| Majority baseline | 0.2183 +/- 0.0007 | 0.775 | | | | |
| RBF-SVM | **0.8771 +/- 0.0248** | **0.9379** | 0.0281 | 0.1108 | 0.388 ms | 9m 09s |
| XGBoost | 0.8625 +/- 0.0369 | 0.9300 | 0.0258 | **0.1091** | 0.145 ms | **1m 30s** |
| Random Forest | 0.8353 +/- 0.0396 | 0.9181 | **0.0213** | 0.1249 | 10.235 ms | 21m 06s |
| MLP (3,108 params) | 0.8294 +/- 0.0319 | 0.9106 | 0.0260 | 0.1332 | **0.023 ms** CPU, 0.056 ms GPU | 49m 48s |

Five findings worth reading the report for:

1. **The MLP does not beat the classical learners.** It is last on macro-F1 and
   on Brier score, at the highest training cost, and it is the cheapest model
   to serve.
2. **The most accurate model is not the most trustworthy.** Random Forest has
   the lowest ECE, XGBoost the best Brier score and by far the best
   high-confidence operating point (65% of segments at p >= 0.99 with 98.87%
   accuracy, against 3.5% for the SVM).
3. **Intermediate noise is worse than total noise.** At -10 dB the tree models
   sit at the majority baseline; at 0 dB they fall to 0.09, well below it,
   because shifted-but-informative features produce confident errors.
4. **Reproducibility is a property of a machine, not of a seed.** The three
   classical models reproduce to four decimal places across two machines and
   two scikit-learn versions; the MLP does not.
5. **The verifier needed its own validation.** Six bugs in the LLM claim
   checker each changed a reported rate, three of them false negatives.

## Deviations, limitations, known gaps

Documented in the report, and summarised here so nobody has to guess:

- Dataset changed from DIAT-uSAT, six classes to four, image preprocessing
  replaced by I/Q signal processing.
- The dataset authors' own train/test indicator is not disjoint by
  measurement, so their published accuracy is not comparable with ours.
- Every model selected a hyperparameter at the edge of its mandated grid, so
  all four may be under-tuned.
- Statistics rest on 130 measurements, not 75,801 segments. Only 11 human and
  19 reflector recordings exist.
- Under Doppler-axis blur the two temporal descriptors are untouched by
  construction, since blur is applied to the power spectrum while they are
  computed from the I/Q.
- Local environment scikit-learn 1.9.0 and torch 2.13.0; Atlas 1.8.0 and
  2.5.1.
- Not run: MLP robustness, and the MLP data-efficiency curve.

## Supervisors

Dr Raja Farrukh Ali (ML), Dr Salman Liaquat (radar signal processing),
Shaiq-e-Mustafa (dataset and class mapping).
