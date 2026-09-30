"""
src/s1_physics/verify_paper_numbers.py

Checks every physics-guided number in the paper against the stored result files.

Dr Farrukh's instruction for the integrated paper is that every reported value
must come from a result file and that nothing may be entered from memory. This
script makes that checkable rather than promised. Each entry in CLAIMS below
records one number as the paper prints it, together with the file and the exact
key path it must come from. The script loads the files, follows the paths and
reports any value that does not agree to the precision the paper quotes.

A claim that cannot be located is reported as MISSING, which is treated as a
failure, because a number with no stored source is the thing this check exists
to catch.

    python src/s1_physics/verify_paper_numbers.py
    python src/s1_physics/verify_paper_numbers.py --section results
    python src/s1_physics/verify_paper_numbers.py --json

Needs S1_ARTIFACT_DIR pointing at the results of record, or a local run whose
artifacts sit where config.py expects them:

    $env:S1_ARTIFACT_DIR = "C:\\Projects\\ANN_Project\\results\\atlas"

The out-of-fold probability files are not in git, so the claims derived from
them (top-2 accuracy, confidence coverage, confident-error direction, pooled
ECE, the Drone zero-Doppler quartiles) are skipped with a SKIP line when the
CSVs are absent rather than counted as failures. Regenerate them with
train_classical.py and train_mlp.py to check those too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))

from config import ARTIFACT_DIR, banner                                # noqa: E402

CLASSICAL = "results_classical_full.json"
MLP = "results_mlp.json"
ABL = "results_ablations.json"
ROB_C = "results_robustness_classical.json"
ROB_M = "results_robustness.json"
LAT = "results_mlp_latency.json"
MLP_DE = "results_mlp_data_efficiency.json"
META = "manifest_meta.json"
LLM = {"llama3.1:8b": "results_llm_ollama_llama3.1-8b.json",
       "llama3.2:3b": "results_llm_ollama_llama3.2-3b.json",
       "gemini-3.5-flash": "results_llm_gemini_gemini-3.5-flash.json"}

PROB_COLS = ["p_drone", "p_bird", "p_human", "p_reflector"]
CLASS_ID = {"drone": 0, "bird": 1, "human": 2, "reflector": 3}

# (section, what the paper says, printed value, file, key path, tolerance)
# Key path segments are dict keys, or integers for list positions.
CLAIMS: list[tuple] = [

    # ---- dataset and protocol ----
    ("dataset", "130 parent measurements", 130, META, ["n_measurements"], 0),
    ("dataset", "75,868 segments in total", 75868, META, ["n_segments"], 0),
    ("dataset", "67 field-of-view-edge segments", 67, META, ["n_edge_flagged"], 0),
    ("dataset", "Drone 58,768 segments", 58768, META, ["segments_per_class", "drone"], 0),
    ("dataset", "Bird 7,792 segments", 7792, META, ["segments_per_class", "bird"], 0),
    ("dataset", "Human 6,028 segments", 6028, META, ["segments_per_class", "human"], 0),
    ("dataset", "Corner Reflector 3,280 segments", 3280, META, ["segments_per_class", "reflector"], 0),
    ("dataset", "Drone 44 parents", 44, META, ["measurements_per_class", "drone"], 0),
    ("dataset", "Bird 56 parents", 56, META, ["measurements_per_class", "bird"], 0),
    ("dataset", "Human 11 parents", 11, META, ["measurements_per_class", "human"], 0),
    ("dataset", "Corner Reflector 19 parents", 19, META, ["measurements_per_class", "reflector"], 0),
    ("dataset", "seed 2026", 2026, META, ["seed"], 0),
    ("dataset", "75,801 segments after edge exclusion", 75801, CLASSICAL, ["n_segments"], 0),

    # ---- headline scores, segment level ----
    ("results", "SVM macro-F1 0.8771", 0.8771, CLASSICAL, ["models", "svm", "mean_macro_f1"], 5e-5),
    ("results", "SVM sd 0.0248", 0.0248, CLASSICAL, ["models", "svm", "std_macro_f1"], 5e-5),
    ("results", "SVM accuracy 0.9379", 0.9379, CLASSICAL, ["models", "svm", "accuracy"], 5e-5),
    ("results", "XGBoost macro-F1 0.8625", 0.8625, CLASSICAL, ["models", "xgb", "mean_macro_f1"], 5e-5),
    ("results", "XGBoost sd 0.0369", 0.0369, CLASSICAL, ["models", "xgb", "std_macro_f1"], 5e-5),
    ("results", "XGBoost accuracy 0.9300", 0.9300, CLASSICAL, ["models", "xgb", "accuracy"], 5e-5),
    ("results", "Random Forest macro-F1 0.8353", 0.8353, CLASSICAL, ["models", "rf", "mean_macro_f1"], 5e-5),
    ("results", "Random Forest sd 0.0396", 0.0396, CLASSICAL, ["models", "rf", "std_macro_f1"], 5e-5),
    ("results", "Random Forest accuracy 0.9181", 0.9181, CLASSICAL, ["models", "rf", "accuracy"], 5e-5),
    ("results", "MLP macro-F1 0.8294", 0.8294, MLP, ["model", "mean_macro_f1"], 5e-5),
    ("results", "MLP sd 0.0319", 0.0319, MLP, ["model", "std_macro_f1"], 5e-5),
    ("results", "MLP accuracy 0.9106", 0.9106, MLP, ["model", "accuracy"], 5e-5),
    ("results", "MLP parameter count 3,108", 3108, MLP, ["model", "parameter_count"], 0),
    ("results", "majority baseline 0.2183", 0.2183, CLASSICAL, ["majority_baseline", "mean_macro_f1"], 5e-5),
    ("results", "majority baseline sd 0.0007", 0.0007, CLASSICAL, ["majority_baseline", "std_macro_f1"], 5e-5),

    # ---- SVM per-fold ----
    ("results", "SVM fold 0 = 0.8306", 0.8306, CLASSICAL, ["models", "svm", "outer_fold_scores", 0], 5e-5),
    ("results", "SVM fold 1 = 0.8822", 0.8822, CLASSICAL, ["models", "svm", "outer_fold_scores", 1], 5e-5),
    ("results", "SVM fold 2 = 0.9036", 0.9036, CLASSICAL, ["models", "svm", "outer_fold_scores", 2], 5e-5),
    ("results", "SVM fold 3 = 0.8904", 0.8904, CLASSICAL, ["models", "svm", "outer_fold_scores", 3], 5e-5),
    ("results", "SVM fold 4 = 0.8786", 0.8786, CLASSICAL, ["models", "svm", "outer_fold_scores", 4], 5e-5),

    # ---- cost ----
    ("cost", "SVM latency 0.388 ms", 0.388, CLASSICAL, ["models", "svm", "mean_latency_ms"], 5e-4),
    ("cost", "XGBoost latency 0.145 ms", 0.145, CLASSICAL, ["models", "xgb", "mean_latency_ms"], 5e-4),
    ("cost", "Random Forest latency 10.235 ms", 10.235, CLASSICAL, ["models", "rf", "mean_latency_ms"], 5e-4),
    ("cost", "MLP latency 0.023 ms CPU", 0.023, LAT, ["devices", "cpu", "mean_latency_ms"], 5e-4),
    ("cost", "MLP latency 0.056 ms GPU", 0.056, LAT, ["devices", "cuda", "mean_latency_ms"], 5e-4),
    ("cost", "SVM training 549.7 s", 549.7, CLASSICAL, ["models", "svm", "wall_seconds"], 0.05),
    ("cost", "XGBoost training 90.7 s", 90.7, CLASSICAL, ["models", "xgb", "wall_seconds"], 0.05),
    ("cost", "Random Forest training 1266.3 s", 1266.3, CLASSICAL, ["models", "rf", "wall_seconds"], 0.05),
    ("cost", "MLP training 2987.0 s", 2987.0, MLP, ["model", "wall_seconds"], 0.05),
    ("cost", "total classical wall 1907.3 s", 1907.3, CLASSICAL, ["total_wall_seconds"], 0.05),

    # ---- calibration, fold averaged ----
    ("calibration", "SVM ECE 0.0281", 0.0281, CLASSICAL, ["models", "svm", "ece"], 5e-5),
    ("calibration", "XGBoost ECE 0.0258", 0.0258, CLASSICAL, ["models", "xgb", "ece"], 5e-5),
    ("calibration", "Random Forest ECE 0.0213", 0.0213, CLASSICAL, ["models", "rf", "ece"], 5e-5),
    ("calibration", "MLP ECE 0.0260", 0.0260, MLP, ["model", "ece"], 5e-5),
    ("calibration", "SVM Brier 0.1108", 0.1108, CLASSICAL, ["models", "svm", "brier_score"], 5e-5),
    ("calibration", "XGBoost Brier 0.1091", 0.1091, CLASSICAL, ["models", "xgb", "brier_score"], 5e-5),
    ("calibration", "Random Forest Brier 0.1249", 0.1249, CLASSICAL, ["models", "rf", "brier_score"], 5e-5),
    ("calibration", "MLP Brier 0.1332", 0.1332, MLP, ["model", "brier_score"], 5e-5),

    # ---- ablation, SVM arm ----
    ("ablation", "ablation reference 0.8780", 0.8780, ABL, ["ablations", "svm", "baseline", "mean"], 5e-5),
    ("ablation", "spectral shape only 0.8016", 0.8016, ABL, ["ablations", "svm", "only_spectral_shape", "mean"], 5e-5),
    ("ablation", "energy only 0.7290", 0.7290, ABL, ["ablations", "svm", "only_energy_distribution", "mean"], 5e-5),
    ("ablation", "temporal only 0.2896", 0.2896, ABL, ["ablations", "svm", "only_temporal", "mean"], 5e-5),

    # ---- permutation importance, SVM ----
    ("ablation", "spectral entropy importance 0.4636", 0.4636, ABL, ["permutation", "svm", "@spectral_entropy"], 5e-5),
    ("ablation", "Doppler spread importance 0.4547", 0.4547, ABL, ["permutation", "svm", "@doppler_spread"], 5e-5),
    ("ablation", "side-lobe energy ratio importance 0.4327", 0.4327, ABL, ["permutation", "svm", "@side_lobe_energy_ratio"], 5e-5),
    ("ablation", "kurtosis importance 0.3570", 0.3570, ABL, ["permutation", "svm", "@kurtosis"], 5e-5),
    ("ablation", "temporal entropy importance 0.2802", 0.2802, ABL, ["permutation", "svm", "@temporal_entropy"], 5e-5),
    ("ablation", "kurtosis, forest 0.034", 0.034, ABL, ["permutation", "rf", "@kurtosis"], 5e-4),
    ("ablation", "kurtosis, ensemble 0.074", 0.074, ABL, ["permutation", "xgb", "@kurtosis"], 5e-4),

    # ---- data efficiency, SVM ----
    ("data_efficiency", "SVM 20 % = 0.7820", 0.7820, ABL, ["data_efficiency", "svm", "0.2", "mean"], 5e-5),
    ("data_efficiency", "SVM 40 % = 0.8197", 0.8197, ABL, ["data_efficiency", "svm", "0.4", "mean"], 5e-5),
    ("data_efficiency", "SVM 60 % = 0.8487", 0.8487, ABL, ["data_efficiency", "svm", "0.6", "mean"], 5e-5),
    ("data_efficiency", "SVM 80 % = 0.8572", 0.8572, ABL, ["data_efficiency", "svm", "0.8", "mean"], 5e-5),
    ("data_efficiency", "SVM 100 % = 0.8780", 0.8780, ABL, ["data_efficiency", "svm", "1.0", "mean"], 5e-5),
    ("data_efficiency", "SVM fold sd at 20 % = 0.0704", 0.0704, ABL, ["data_efficiency", "svm", "0.2", "std"], 5e-5),
    ("data_efficiency", "SVM fold sd at 100 % = 0.0233", 0.0233, ABL, ["data_efficiency", "svm", "1.0", "std"], 5e-5),
    ("data_efficiency", "MLP 20 % = 0.7109", 0.7109, MLP_DE, ["data_efficiency", "0.2", "mean"], 5e-5),
    ("data_efficiency", "MLP 100 % = 0.8372", 0.8372, MLP_DE, ["data_efficiency", "1.0", "mean"], 5e-5),

    # ---- per-class F1, pooled over the grouped folds ----
    ("per_class", "SVM Drone F1 0.9687", 0.9687, ROB_C, ["scores", "clean", "svm", "per_class_f1", "drone"], 5e-5),
    ("per_class", "SVM Bird F1 0.7923", 0.7923, ROB_C, ["scores", "clean", "svm", "per_class_f1", "bird"], 5e-5),
    ("per_class", "SVM Human F1 0.8327", 0.8327, ROB_C, ["scores", "clean", "svm", "per_class_f1", "human"], 5e-5),
    ("per_class", "SVM Reflector F1 0.9200", 0.9200, ROB_C, ["scores", "clean", "svm", "per_class_f1", "reflector"], 5e-5),
    ("per_class", "MLP Drone F1 0.9512", 0.9512, ROB_M, ["scores", "clean", "mlp", "per_class_f1", "drone"], 5e-5),
    ("per_class", "MLP Bird F1 0.7097", 0.7097, ROB_M, ["scores", "clean", "mlp", "per_class_f1", "bird"], 5e-5),
    ("per_class", "MLP Human F1 0.8015", 0.8015, ROB_M, ["scores", "clean", "mlp", "per_class_f1", "human"], 5e-5),
    ("per_class", "MLP Reflector F1 0.8487", 0.8487, ROB_M, ["scores", "clean", "mlp", "per_class_f1", "reflector"], 5e-5),

    # ---- robustness ----
    ("robustness", "SVM at +10 dB = 0.4610", 0.4610, ROB_C, ["scores", "snr+10dB", "svm", "mean_macro_f1"], 5e-5),
    ("robustness", "SVM at -10 dB = 0.2269", 0.2269, ROB_C, ["scores", "snr-10dB", "svm", "mean_macro_f1"], 5e-5),
    ("robustness", "MLP at +10 dB = 0.4244", 0.4244, ROB_M, ["scores", "snr+10dB", "mlp", "mean_macro_f1"], 5e-5),
    ("robustness", "MLP at -10 dB = 0.2110", 0.2110, ROB_M, ["scores", "snr-10dB", "mlp", "mean_macro_f1"], 5e-5),
    ("robustness", "SVM 1-bin blur 0.5272", 0.5272, ROB_C, ["scores", "blur1px", "svm", "mean_macro_f1"], 5e-5),
    ("robustness", "SVM 4-bin blur 0.3941", 0.3941, ROB_C, ["scores", "blur4px", "svm", "mean_macro_f1"], 5e-5),
    ("robustness", "MLP 1-bin blur 0.5954", 0.5954, ROB_M, ["scores", "blur1px", "mlp", "mean_macro_f1"], 5e-5),
    ("robustness", "MLP 4-bin blur 0.3093", 0.3093, ROB_M, ["scores", "blur4px", "mlp", "mean_macro_f1"], 5e-5),
    ("robustness", "combined, SVM 0.1631", 0.1631, ROB_C, ["scores", "combined", "svm", "mean_macro_f1"], 5e-5),
    ("robustness", "combined, XGBoost 0.0716", 0.0716, ROB_C, ["scores", "combined", "xgb", "mean_macro_f1"], 5e-5),
    ("robustness", "combined, Random Forest 0.0735", 0.0735, ROB_C, ["scores", "combined", "rf", "mean_macro_f1"], 5e-5),
    ("robustness", "combined, MLP 0.1770", 0.1770, ROB_M, ["scores", "combined", "mlp", "mean_macro_f1"], 5e-5),

    # ---- LLM reporting study ----
    ("llm", "Llama-3.1-8B direct 6 of 20", 6, LLM["llama3.1:8b"], ["arms", "direct", "summary", "n_pass"], 0),
    ("llm", "Llama-3.1-8B constrained 18 of 20", 18, LLM["llama3.1:8b"], ["arms", "constrained", "summary", "n_pass"], 0),
    ("llm", "Llama-3.1-8B generate-verify 19 of 20", 19, LLM["llama3.1:8b"], ["arms", "generate_verify", "summary", "n_pass"], 0),
    ("llm", "Llama-3.2-3B direct 15 of 20", 15, LLM["llama3.2:3b"], ["arms", "direct", "summary", "n_pass"], 0),
    ("llm", "Llama-3.2-3B constrained 18 of 20", 18, LLM["llama3.2:3b"], ["arms", "constrained", "summary", "n_pass"], 0),
    ("llm", "Llama-3.2-3B generate-verify 18 of 20", 18, LLM["llama3.2:3b"], ["arms", "generate_verify", "summary", "n_pass"], 0),
    ("llm", "Gemini direct 3 of 5", 3, LLM["gemini-3.5-flash"], ["arms", "direct", "summary", "n_pass"], 0),
    ("llm", "Gemini constrained 5 of 5", 5, LLM["gemini-3.5-flash"], ["arms", "constrained", "summary", "n_pass"], 0),
]

# Claims computed from the out-of-fold probability files. Skipped when absent.
# model -> (top-2, coverage at p>=0.99, accuracy in that band, confident-error
#           rate at p>=0.9, pooled 15-bin ECE)
DERIVED = {
    "svm": (0.9891, 3.47, 99.66, 2.77, 0.0241),
    "xgb": (0.9853, 65.31, 98.87, 2.35, 0.0212),
    "rf": (0.9824, 42.15, 99.33, 1.06, 0.0157),
    "mlp": (0.9860, 53.24, 99.46, 1.40, 0.0118),
}
DRONE_QUARTILE_RECALL = (0.923, 0.980, 0.997, 0.986)
SVM_DRONE_DIRECTED_ERRORS = (1517, 2100)


def dig(blob, path: list):
    """Follow a key path. '@name' selects the dict in a list whose feature is name."""
    cur = blob
    for step in path:
        if isinstance(step, str) and step.startswith("@"):
            want = step[1:]
            match = [r for r in cur if r.get("feature") == want]
            if not match:
                raise KeyError(f"no entry with feature={want}")
            cur = match[0]["mean"]
        else:
            cur = cur[step]
    return cur


def pooled_ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    total, edges = 0.0, np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def check_derived(art: Path, rows: list) -> None:
    for key, (p_top2, p_cov, p_acc, p_cerr, p_ece) in DERIVED.items():
        path = art / f"oof_predictions_{key}.csv"
        if not path.exists():
            rows.append(("derived", f"{key.upper()} out-of-fold claims", "SKIP",
                         f"{path.name} not in the repository; regenerate to check"))
            continue
        df = pd.read_csv(path)
        probs = df[PROB_COLS].to_numpy(float)
        conf = probs.max(axis=1)
        correct = (df["pred"] == df["true"]).to_numpy()
        order = np.argsort(-probs, axis=1)
        top2 = float(np.mean([t in row[:2] for t, row in zip(df["true"], order)]))
        band = conf >= 0.99
        for label, got, want, tol in (
                (f"{key.upper()} top-2 accuracy", top2, p_top2, 5e-5),
                (f"{key.upper()} coverage at p>=0.99 (%)", band.mean() * 100, p_cov, 5e-3),
                (f"{key.upper()} accuracy in that band (%)", correct[band].mean() * 100, p_acc, 5e-3),
                (f"{key.upper()} confident errors at p>=0.9 (%)",
                 ((~correct) & (conf >= 0.9)).mean() * 100, p_cerr, 5e-3),
                (f"{key.upper()} pooled 15-bin ECE", pooled_ece(conf, correct.astype(float)), p_ece, 1e-4)):
            rows.append(("derived", label, "PASS" if abs(got - want) <= tol else "FAIL",
                         f"stored {got:.4f} vs paper {want}"))

        if key == "svm":
            wrong = df[(~correct) & (conf >= 0.9)]
            pred = wrong["pred"]
            to_drone = int((pred == CLASS_ID["drone"]).sum() if pred.dtype != object
                           else (pred == "drone").sum())
            n_want, d_want = SVM_DRONE_DIRECTED_ERRORS
            ok = (to_drone == n_want) and (len(wrong) == d_want)
            rows.append(("derived", "SVM confident errors directed at Drone",
                         "PASS" if ok else "FAIL",
                         f"stored {to_drone} of {len(wrong)} vs paper {n_want} of {d_want}"))

    feat = art / "s1_features.csv"
    oof = art / "oof_predictions_svm.csv"
    if feat.exists() and oof.exists():
        f = pd.read_csv(feat, usecols=["segment_uid", "zero_doppler_ratio"])
        o = pd.read_csv(oof)
        m = o.merge(f, on="segment_uid", how="left")
        d = m[m["true"] == CLASS_ID["drone"]].copy()
        d["q"] = pd.qcut(d["zero_doppler_ratio"], 4, labels=[1, 2, 3, 4])
        got = d.groupby("q", observed=True).apply(
            lambda g: float((g["pred"] == CLASS_ID["drone"]).mean())).tolist()
        ok = all(abs(g - w) <= 5e-4 for g, w in zip(got, DRONE_QUARTILE_RECALL))
        rows.append(("derived", "Drone recall by zero-Doppler quartile",
                     "PASS" if ok else "FAIL",
                     f"stored {[round(g, 4) for g in got]} vs paper {list(DRONE_QUARTILE_RECALL)}"))
    else:
        rows.append(("derived", "Drone zero-Doppler quartile recall", "SKIP",
                     "needs s1_features.csv and oof_predictions_svm.csv"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+", default=None,
                    help="limit to these sections, e.g. results calibration")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--artifact-dir", default=None,
                    help="override S1_ARTIFACT_DIR for this run")
    args = ap.parse_args()

    art = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_DIR
    if not args.json:
        print(banner(), "\n")
        print(f"checking against {art}\n")

    cache: dict[str, dict] = {}

    def load(name: str):
        if name not in cache:
            for base in (art, art.parent / "local_dev", art.parent / "atlas"):
                p = base / name
                if p.exists():
                    cache[name] = json.loads(p.read_text())
                    break
            else:
                cache[name] = None
        return cache[name]

    rows: list[tuple] = []
    for section, label, paper, fname, path, tol in CLAIMS:
        if args.section and section not in args.section:
            continue
        blob = load(fname)
        if blob is None:
            rows.append((section, label, "MISSING", f"{fname} not found"))
            continue
        try:
            got = dig(blob, path)
        except (KeyError, IndexError, TypeError) as exc:
            rows.append((section, label, "MISSING",
                         f"{fname}:{'/'.join(map(str, path))} -> {exc}"))
            continue
        ok = abs(float(got) - float(paper)) <= tol
        rows.append((section, label, "PASS" if ok else "FAIL",
                     f"stored {got} vs paper {paper}  [{fname}]"))

    if not args.section or "derived" in args.section:
        check_derived(art, rows)

    counts = {k: sum(1 for r in rows if r[2] == k)
              for k in ("PASS", "FAIL", "MISSING", "SKIP")}

    if args.json:
        print(json.dumps({"artifact_dir": str(art), "counts": counts,
                          "rows": [{"section": s, "claim": c, "status": st,
                                    "detail": d} for s, c, st, d in rows]}, indent=2))
    else:
        current = None
        for section, label, status, detail in rows:
            if section != current:
                print(f"\n-- {section} --")
                current = section
            mark = {"PASS": "  ok  ", "FAIL": " FAIL ",
                    "MISSING": "MISSING", "SKIP": " skip "}[status]
            line = f"  {mark}  {label}"
            print(line if status == "PASS" else f"{line}\n              {detail}")
        print(f"\n{counts['PASS']} verified, {counts['FAIL']} wrong, "
              f"{counts['MISSING']} not found, {counts['SKIP']} skipped")
        if counts["FAIL"] or counts["MISSING"]:
            print("\nA wrong or unfindable number must be corrected in the paper "
                  "or traced to the file it really came from before submission.")
        else:
            print("\nEvery checked number in the paper matches a stored result file.")

    return 1 if (counts["FAIL"] or counts["MISSING"]) else 0


if __name__ == "__main__":
    sys.exit(main())
