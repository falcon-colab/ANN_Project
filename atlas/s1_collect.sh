#!/bin/bash
# s1_collect.sh -- run on raven01, from anywhere:  bash ~/ANN_Project/s1_collect.sh
#
# Checks that every result the paper needs exists and is non-trivial, prints a
# summary you can paste into the chat, verifies the fold-manifest hash, and
# packages everything into three tarballs in your home directory ready to scp.
#
# Safe to run more than once. It reads and packages; it never deletes or
# regenerates anything.

set -uo pipefail

PROJECT=/home/gmustafa/ANN_Project
ART=/data/gmustafa/se801/artifacts
LOGS=$PROJECT/logs
EXPECTED_SHA=fbece6fb30a8d9e8468cac2ba89748d298d0b8debe54402ffc278302ad8fb7a6

echo "=============================================================="
echo " S1 result collection, $(date)"
echo "=============================================================="

# --- 1. jobs ---------------------------------------------------------------
echo
echo "--- job history ---"
sacct -u "$USER" -S 2026-09-17 \
      --format=JobID%12,JobName%16,State%12,Elapsed,MaxRSS,ExitCode 2>/dev/null \
  | grep -v '\.extern' || echo "sacct unavailable"

# --- 2. the fold manifest must not have changed ----------------------------
echo
echo "--- fold manifest hash ---"
if [ -f "$ART/fold_manifest.csv" ]; then
    GOT=$(sha256sum "$ART/fold_manifest.csv" | cut -d' ' -f1)
    echo "$GOT"
    if [ "$GOT" = "$EXPECTED_SHA" ]; then
        echo "OK: matches the frozen manifest all four tracks share"
    else
        echo "STOP: hash does NOT match the frozen manifest."
        echo "      Expected $EXPECTED_SHA"
        echo "      Do not report results from this run until that is explained."
    fi
else
    echo "MISSING $ART/fold_manifest.csv"
fi

# --- 3. every artefact the paper needs -------------------------------------
echo
echo "--- required artefacts ---"
REQUIRED=(
  results_classical_full.json
  results_mlp.json
  results_mlp_latency.json
  results_ablations.json
  results_robustness.json
  run_ledger.csv
  manifest_meta.json
  mlp_training_curves.csv
  s1_features.csv
  feature_correlation.png
  feature_distributions.png
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ -s "$ART/$f" ]; then
        printf "  present  %-28s %s\n" "$f" "$(du -h "$ART/$f" | cut -f1)"
    else
        printf "  MISSING  %s\n" "$f"
        MISSING=$((MISSING + 1))
    fi
done
echo "  models: $(ls -1 "$ART"/models/*.joblib 2>/dev/null | wc -l) joblib, "\
"$(ls -1 "$ART"/models/mlp_fold*.pt 2>/dev/null | wc -l) checkpoints"
echo "  oof:    $(ls -1 "$ART"/oof_predictions_*.csv 2>/dev/null | wc -l) files"

# --- 4. headline numbers, straight out of the JSON -------------------------
echo
echo "--- headline numbers (paste these into the chat) ---"
python3 - <<'PY'
import json, pathlib
art = pathlib.Path("/data/gmustafa/se801/artifacts")

def load(name):
    p = art / name
    if not p.exists() or p.stat().st_size == 0:
        print(f"  {name}: absent")
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"  {name}: unreadable ({e})")
        return None

d = load("results_classical_full.json")
if d:
    for k, m in d["models"].items():
        print(f"  {m['model_name']:14s} macro-F1 {m['mean_macro_f1']:.4f} "
              f"+/- {m['std_macro_f1']:.4f} | acc {m['accuracy']:.4f} | "
              f"ece {m['ece']:.4f} | brier {m['brier_score']:.4f} | "
              f"lat {m['mean_latency_ms']:.3f} ms")

m = load("results_mlp.json")
if m:
    mm = m["model"]
    print(f"  {'MLP':14s} macro-F1 {mm['mean_macro_f1']:.4f} "
          f"+/- {mm['std_macro_f1']:.4f} | acc {mm['accuracy']:.4f} | "
          f"ece {mm['ece']:.4f} | brier {mm['brier_score']:.4f}")

lat = load("results_mlp_latency.json")
if lat:
    for dev, r in lat["devices"].items():
        print(f"  MLP latency {dev:5s} {r['mean_latency_ms']:.4f} ms "
              f"(p95 {r['p95_latency_ms']:.4f})")

a = load("results_ablations.json")
if a:
    print(f"  ablations: {len(a.get('ablations', {}))} models, "
          f"data efficiency {len(a.get('data_efficiency', {}))}, "
          f"permutation {len(a.get('permutation', {}))}"
          + (f" | PARTIAL RUN: {a['partial_run']}" if a.get("partial_run") else ""))
    for key, conds in (a.get("ablations") or {}).items():
        for name in ("baseline", "compact5_correlation", "compact5_importance"):
            if name in conds:
                print(f"    {key:4s} {name:22s} {conds[name]['mean']:.4f}")

r = load("results_robustness.json")
if r:
    sc = r.get("scores", r)
    conds = list(sc.keys()) if isinstance(sc, dict) else []
    print(f"  robustness conditions: {len(conds)} -> {', '.join(conds[:12])}")
    for cond in conds:
        row = sc[cond]
        if isinstance(row, dict):
            vals = ", ".join(f"{k} {v:.4f}" if isinstance(v, (int, float))
                             else f"{k} {v}" for k, v in row.items())
            print(f"    {cond:10s} {vals}")
PY

# --- 5. package ------------------------------------------------------------
echo
echo "--- packaging ---"
cd "$ART" || exit 1
sha256sum fold_manifest.csv > "$HOME/fold_manifest.sha256" 2>/dev/null

tar czf "$HOME/s1_atlas_results.tgz" \
    results_*.json run_ledger.csv manifest_meta.json mlp_training_curves.csv \
    feature_correlation.png feature_distributions.png models/mlp_fold*.pt \
    2>/dev/null
echo "  $HOME/s1_atlas_results.tgz  $(du -h "$HOME/s1_atlas_results.tgz" | cut -f1)"

tar czf "$HOME/s1_atlas_bulk.tgz" \
    oof_predictions_*.csv s1_features.csv fold_manifest.csv dataset_manifest.csv \
    2>/dev/null
echo "  $HOME/s1_atlas_bulk.tgz     $(du -h "$HOME/s1_atlas_bulk.tgz" | cut -f1)"

cd "$LOGS" || exit 1
tar czf "$HOME/s1_atlas_logs.tgz" ./*.log ./*.err done_*.txt 2>/dev/null
echo "  $HOME/s1_atlas_logs.tgz     $(du -h "$HOME/s1_atlas_logs.tgz" | cut -f1)"

echo
if [ "$MISSING" -gt 0 ]; then
    echo "WARNING: $MISSING required artefact(s) missing. Check the job logs"
    echo "         before treating this as the final result set."
else
    echo "All required artefacts present."
fi
echo
echo "Now, from PowerShell on the workstation:"
echo "  scp gmustafa@192.168.24.100:~/s1_atlas_*.tgz \"\$env:USERPROFILE\\Downloads\\\""
echo "  scp gmustafa@192.168.24.100:~/fold_manifest.sha256 \"\$env:USERPROFILE\\Downloads\\\""
echo "=============================================================="
