# Result traceability, physics-guided track

Every number this track contributes to the integrated paper, and the stored file
it comes from. No value in the paper was typed from memory or from a previous
draft.

This document is checkable rather than declarative. Running

```powershell
$env:S1_ARTIFACT_DIR = "C:\Projects\ANN_Project\results\atlas"
python src/s1_physics/verify_paper_numbers.py
```

loads each file below, follows the key path, and compares it with the value as
the paper prints it. The last full run reported **121 verified, 0 wrong, 0 not
found**. `--json` gives machine-readable output and the exit code is non-zero if
anything fails, so it can gate a commit.

The results of record are `results/atlas/`, produced by SLURM jobs 2305, 2308
and 2309 on node `argon01`. The language-model study was run locally and lives
in `results/local_dev/`. Where a file exists in both places, the Atlas copy is
the one the paper cites.

## Files

| File | Written by | Holds |
|---|---|---|
| `results/atlas/manifest_meta.json` | `src/common/generate_manifest.py` | Segment and parent counts, per-class composition, seed, fold-manifest digest |
| `results/atlas/results_classical_full.json` | `src/s1_physics/train_classical.py` | SVM, Random Forest and XGBoost: macro-F1, accuracy, ECE, Brier, per-fold scores, selected hyper-parameters, latency, wall time, peak memory; majority baseline |
| `results/atlas/results_mlp.json` | `src/s1_physics/train_mlp.py` | The same fields for the MLP, plus architecture and parameter count |
| `results/atlas/results_mlp_latency.json` | `src/s1_physics/measure_mlp_latency.py` | Batch-one MLP latency on CPU and CUDA under the 100 warm-up, 1,000 timed, 5 repeat protocol |
| `results/atlas/results_ablations.json` | `src/s1_physics/run_ablations.py` | Family and single-descriptor ablations, classical data-efficiency curves, permutation importance |
| `results/atlas/results_robustness_classical.json` | `src/s1_physics/run_robustness.py` | Classical models under nine corruption conditions, feature stability, clean against corrupted importance, pooled per-class F1 |
| `results/atlas/results_robustness.json` | `src/s1_physics/run_robustness.py --models mlp` | The same conditions for the MLP |
| `results/atlas/results_mlp_data_efficiency.json` | `src/s1_physics/mlp_data_efficiency.py` | MLP data-efficiency curve at five training fractions |
| `results/atlas/oof_predictions_{svm,xgb,rf,mlp}.csv` | `train_classical.py`, `train_mlp.py` | Per-segment held-out class probabilities, with `measurement_id` and `outer_fold` |
| `results/atlas/s1_features.csv` | `src/s1_physics/feature_extractor.py` | The ten descriptors per segment |
| `results/atlas/reliability_bins.csv` | `src/s1_physics/make_reliability.py` | The 15 reliability bins behind the calibration figure |
| `results/atlas/run_ledger.csv` | `train_classical.py` | One row per model with job id, commit, seed, configuration and headline metrics |
| `results/local_dev/results_llm_ollama_llama3.1-8b.json` | `src/s1_physics/llm_experiment.py` | Llama-3.1-8B across three prompting arms, with the seeded-error injection test |
| `results/local_dev/results_llm_ollama_llama3.2-3b.json` | same | Llama-3.2-3B across three prompting arms |
| `results/local_dev/results_llm_gemini_gemini-3.5-flash.json` | same | Gemini-3.5-Flash across two prompting arms |
| `fold_manifest.csv` | `src/common/generate_manifest.py`, run once | The frozen outer and inner fold assignment shared by all four tracks |

## Dataset and protocol

| Paper value | File | Key |
|---|---|---|
| 130 parent measurements | `manifest_meta.json` | `n_measurements` |
| 75,868 segments in total | `manifest_meta.json` | `n_segments` |
| 67 field-of-view-edge segments | `manifest_meta.json` | `n_edge_flagged` |
| 75,801 segments after exclusion | `results_classical_full.json` | `n_segments` |
| Segments per class, 58,768 / 7,792 / 6,028 / 3,280 | `manifest_meta.json` | `segments_per_class.{drone,bird,human,reflector}` |
| Parents per class, 44 / 56 / 11 / 19 | `manifest_meta.json` | `measurements_per_class.{drone,bird,human,reflector}` |
| Seed 2026 | `manifest_meta.json` | `seed` |
| Fold manifest digest `fbece6fb…8db4e402` | `manifest_meta.json` | `fold_manifest_sha256` |

## Headline scores, segment level

Mean and standard deviation across the five grouped outer folds.

| Paper value | File | Key |
|---|---|---|
| SVM 0.8771 ± 0.0248, accuracy 0.9379 | `results_classical_full.json` | `models.svm.{mean_macro_f1,std_macro_f1,accuracy}` |
| XGBoost 0.8625 ± 0.0369, accuracy 0.9300 | `results_classical_full.json` | `models.xgb.…` |
| Random Forest 0.8353 ± 0.0396, accuracy 0.9181 | `results_classical_full.json` | `models.rf.…` |
| MLP 0.8294 ± 0.0319, accuracy 0.9106 | `results_mlp.json` | `model.…` |
| MLP 3,108 parameters, 10-64-32-4 | `results_mlp.json` | `model.{parameter_count,architecture}` |
| Majority baseline 0.2183 ± 0.0007 | `results_classical_full.json` | `majority_baseline.{mean_macro_f1,std_macro_f1}` |
| SVM per-fold 0.8306, 0.8822, 0.9036, 0.8904, 0.8786 | `results_classical_full.json` | `models.svm.outer_fold_scores` |

The paired sign-flip p-values, 0.25 for the SVM against XGBoost and 0.0625
against the Random Forest and the MLP, are computed from the per-fold arrays
above by `src/s1_physics/compare_models.py`, which writes `table_folds.csv`.

## Per-class F1, pooled over the grouped folds

| Paper value | File | Key |
|---|---|---|
| SVM 0.9687 / 0.7923 / 0.8327 / 0.9200 | `results_robustness_classical.json` | `scores.clean.svm.per_class_f1.*` |
| MLP 0.9512 / 0.7097 / 0.8015 / 0.8487 | `results_robustness.json` | `scores.clean.mlp.per_class_f1.*` |

## Calibration and confidence

| Paper value | File | Key |
|---|---|---|
| Fold-averaged ECE 0.0281 / 0.0258 / 0.0213 | `results_classical_full.json` | `models.{svm,xgb,rf}.ece` |
| Fold-averaged ECE 0.0260 for the MLP | `results_mlp.json` | `model.ece` |
| Brier 0.1108 / 0.1091 / 0.1249 | `results_classical_full.json` | `models.{svm,xgb,rf}.brier_score` |
| Brier 0.1332 for the MLP | `results_mlp.json` | `model.brier_score` |
| Pooled ECE 0.0241 / 0.0212 / 0.0157 / 0.0118 | `oof_predictions_*.csv` | 15 equal-width bins over the maximum class probability; also in `reliability_bins.csv` |
| Top-2 accuracy 0.9891 / 0.9853 / 0.9824 / 0.9860 | `oof_predictions_*.csv` | Two highest probabilities against the true label |
| Coverage and accuracy at p ≥ 0.99: 3.47 % at 99.66 %, 65.31 % at 98.87 %, 42.15 % at 99.33 %, 53.24 % at 99.46 % | `oof_predictions_*.csv` | Threshold on the maximum probability |
| Confident errors at p ≥ 0.9: 2.77 %, 2.35 %, 1.06 %, 1.40 % | `oof_predictions_*.csv` | Incorrect rows above the threshold |
| 1,517 of the SVM's 2,100 confident errors assign a non-Drone segment to Drone | `oof_predictions_svm.csv` | Incorrect rows above p ≥ 0.9 with `pred` equal to Drone |

The pooled and fold-averaged ECE columns are both reported because they
disagree. Nothing but the aggregation rule differs between them.

## Feature analysis

| Paper value | File | Key |
|---|---|---|
| Ablation reference 0.8780 | `results_ablations.json` | `ablations.svm.baseline.mean` |
| Spectral shape only 0.8016 | `results_ablations.json` | `ablations.svm.only_spectral_shape.mean` |
| Energy distribution only 0.7290 | `results_ablations.json` | `ablations.svm.only_energy_distribution.mean` |
| Temporal only 0.2896 | `results_ablations.json` | `ablations.svm.only_temporal.mean` |
| Dropping spectral shape costs 0.1015 | `results_ablations.json` | `baseline.mean` minus `drop_spectral_shape.mean` |
| Dropping energy costs 0.0545 | `results_ablations.json` | `baseline.mean` minus `drop_energy_distribution.mean` |
| Dropping temporal costs 0.0094 | `results_ablations.json` | `baseline.mean` minus `drop_temporal.mean` |
| No single descriptor worth more than 0.0260 | `results_ablations.json` | Maximum over the fourteen `drop_<feature>` entries; the largest is `drop_spectral_entropy` at 0.0260 |
| Permutation importance 0.4636 / 0.4547 / 0.4327 / 0.3570 / 0.2802 | `results_ablations.json` | `permutation.svm[].mean`, matched on `feature` |
| Kurtosis: forest 0.0341, ensemble 0.0740 | `results_ablations.json` | `permutation.{rf,xgb}[].mean` where `feature` is `kurtosis` |
| Drone recall by zero-Doppler quartile 0.923 / 0.980 / 0.997 / 0.986 | `oof_predictions_svm.csv` joined to `s1_features.csv` | Quartiles of `zero_doppler_ratio` within the Drone class |

The ablation arms are stored in the order SVM, XGBoost, Random Forest under
`ablations`, and the paper quotes the SVM arm. This ordering was confirmed
directly in the file.

## Data efficiency

| Paper value | File | Key |
|---|---|---|
| SVM 0.7820 / 0.8197 / 0.8487 / 0.8572 / 0.8780 | `results_ablations.json` | `data_efficiency.svm.{0.2,0.4,0.6,0.8,1.0}.mean` |
| SVM fold spread 0.0704 at 20 %, 0.0233 at 100 % | `results_ablations.json` | `data_efficiency.svm.{0.2,1.0}.std` |
| MLP 0.7109 / 0.7988 / 0.7948 / 0.8131 / 0.8372 | `results_mlp_data_efficiency.json` | `data_efficiency.{…}.mean` |

Subsampling is at measurement level and applies to the training partition only;
each outer test fold is untouched. Hyper-parameters are reused from the nested
selection rather than re-tuned, which is recorded in the file's `note` field.

## Robustness

Noise is added to the complex signal before the Doppler transform, so the
signal-to-noise ratios are signal-domain and not image-domain. The file records
this in its `note` field.

| Paper value | File | Key |
|---|---|---|
| SVM 0.4610 at +10 dB, 0.2269 at −10 dB | `results_robustness_classical.json` | `scores.snr+10dB.svm.mean_macro_f1`, `scores.snr-10dB.svm.…` |
| MLP 0.4244 at +10 dB, 0.2110 at −10 dB | `results_robustness.json` | `scores.snr±10dB.mlp.mean_macro_f1` |
| SVM blur 0.5272 at one bin, 0.3941 at four | `results_robustness_classical.json` | `scores.blur{1,4}px.svm.mean_macro_f1` |
| MLP blur 0.5954 at one bin, 0.3093 at four | `results_robustness.json` | `scores.blur{1,4}px.mlp.mean_macro_f1` |
| Combined 0 dB and 2-bin blur: 0.1631, 0.1770, 0.0716, 0.0735 | both robustness files | `scores.combined.{svm,mlp,xgb,rf}.mean_macro_f1` |

## Cost

| Paper value | File | Key |
|---|---|---|
| Batch-one latency 0.388 / 0.145 / 10.235 ms | `results_classical_full.json` | `models.{svm,xgb,rf}.mean_latency_ms` |
| MLP 0.023 ms CPU, 0.056 ms CUDA | `results_mlp_latency.json` | `devices.{cpu,cuda}.mean_latency_ms` |
| Training 9 min 09 s, 1 min 30 s, 21 min 06 s | `results_classical_full.json` | `models.{svm,xgb,rf}.wall_seconds` = 549.7, 90.7, 1266.3 |
| MLP training 49 min 47 s | `results_mlp.json` | `model.wall_seconds` = 2987.0 |
| Peak memory 2.31 to 2.40 GB per model | `results_classical_full.json` | `models.*.peak_memory_mb` = 2313.9, 2377.0, 2398.2 |
| Total classical wall time 1907.3 s | `results_classical_full.json` | `total_wall_seconds` |

## Language-model reporting study

| Paper value | File | Key |
|---|---|---|
| Llama-3.1-8B passes 6, 18, 19 of 20 | `results_llm_ollama_llama3.1-8b.json` | `arms.{direct,constrained,generate_verify}.summary.n_pass` |
| Llama-3.2-3B passes 15, 18, 18 of 20 | `results_llm_ollama_llama3.2-3b.json` | same keys |
| Gemini-3.5-Flash passes 3 of 5 direct and 5 of 5 constrained | `results_llm_gemini_gemini-3.5-flash.json` | same keys; **only two arms exist for this model** |
| Numerical fidelity at or above 0.976 in every arm | all three files | `arms.*.summary.mean_factual_accuracy`; the minimum is 0.9762, Llama-3.1-8B direct |
| Record coverage 43 to 47 % | `results_llm_ollama_llama3.2-3b.json` | `arms.{constrained,generate_verify}.summary.mean_record_coverage` = 0.4296, 0.4704. In the direct arm coverage is 0.1439, so the claim must be scoped to the two structured arms |
| Verifier detects 100 % of seeded errors | all three files | `injection_test.by_type.*.detected`, true for all five error types |
| Seven verifier defects, six of which changed a reported rate | `src/s1_physics/test_llm_verify.py` | The seven regression tests marked as real-output regressions |
| 27 regression tests | `src/s1_physics/test_llm_verify.py` | 27 test functions; 26 run and one skips without a network model |

Gemini-3.5-Flash was run on the direct and constrained arms only. There is no
generate-and-verify arm for that model, and any sentence implying a result for
it is unsupported.

## Cross-machine reproducibility

| Paper value | Source |
|---|---|
| The three classical models agree to four decimal places on every outer fold | `results/atlas/results_classical_full.json` against `results/local_dev/results_classical_full.json`, comparing `models.*.outer_fold_scores` |
| MLP 0.8294 on the cluster against 0.8338 ± 0.0305 locally | `results/atlas/results_mlp.json` and `results/local_dev/results_mlp.json`, `model.mean_macro_f1` |
| Per-fold differences between −0.0014 and +0.0110 | The same two files, element-wise on `model.outer_fold_scores`; local minus cluster |

Cluster environment scikit-learn 1.8.0 with PyTorch 2.5.1 on an RTX 4090.
Development machine scikit-learn 1.9.0 with PyTorch 2.13.0 on CPU. The MLP's
difference therefore has two candidate causes, device and library version, and
one run on each cannot separate them.

## What is deliberately not in the paper

`src/s1_physics/s1_grouping_ablation.py` and `src/s1_physics/s1_parent_level.py`
are present in this repository and have not been run. They are listed under
further work in the README. No number attributed to either appears in the paper,
and neither writes a file that anything else reads.
