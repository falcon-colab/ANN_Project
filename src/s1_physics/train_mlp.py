"""
src/s1_physics/train_mlp.py

The mandated neural model for S1, plus the section 14 training demonstration.

Protocol section 10 fixes the architecture as 10 -> 64 -> 32 -> output with
ReLU, batch normalisation and dropout. The output layer is 4 rather than the
document's 6, following the coarse class mapping agreed on 30 August.

Section 14 requires a clearly written training and validation loop showing all
ten of: mini-batch loading, gradient reset, forward pass, loss computation,
backpropagation, optimizer update, validation, early stopping, checkpoint
saving and checkpoint restoration. Each is marked with a numbered comment in
`train_one` so it can be pointed at during the live demonstration. It also
requires training loss, validation loss, training macro-F1, validation
macro-F1, gradient norms and epoch time to be recorded; all six go into the
per-run history.

Two further section 14 obligations are implemented as subcommands:

    --demo overfit     the small-batch overfitting test. If a network cannot
                       drive the loss to near zero on a handful of samples,
                       something is broken: labels, gradient flow, learning
                       rate or preprocessing. Passing it proves the machinery
                       works before any conclusion is drawn from it.
    --demo repro       the fixed-seed reproducibility test. Two runs of the
                       same configuration and seed must produce identical
                       numbers.

On model.train() versus model.eval(), which section 14 asks you to explain:
dropout randomly zeroes activations during training and is disabled at
evaluation; batch normalisation uses the current mini-batch statistics during
training while accumulating running estimates, and uses those running
estimates at evaluation. Forgetting eval() therefore both injects noise and
lets the batch composition of the test set influence its own predictions.

The whole feature matrix is moved to the device once and mini-batches are cut
from a shuffled permutation. With ten features a DataLoader would spend more
time on worker overhead than on arithmetic, and on Windows num_workers>0
deadlocks. The batching is still explicit, which is what section 14 asks for.

Run from the project root:
    python src/s1_physics/train_mlp.py --demo overfit
    python src/s1_physics/train_mlp.py --demo repro
    python src/s1_physics/train_mlp.py
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))
from config import ARTIFACT_DIR, N_INNER, N_OUTER, OUTER_FOLD_SEEDS, SEED, banner
from feature_transforms import (COMPACT_SUBSET, FEATURE_NAMES,
                                FeaturePreprocessor)
from metrics import evaluate_classification

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}
N_CLASSES = 4

# Protocol section 15
GRID = {"lr": [1e-4, 3e-4, 1e-3], "weight_decay": [1e-5, 1e-4]}
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8
GRAD_CLIP = 1.0
HIDDEN = (64, 32)
DROPOUT = 0.30


class MLP(nn.Module):
    """10 -> 64 -> 32 -> 4, with ReLU, batch norm and dropout."""

    def __init__(self, n_in: int, hidden=HIDDEN, n_out=N_CLASSES,
                 dropout=DROPOUT):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, X, y) -> tuple[float, float, np.ndarray]:
    model.eval()                       # dropout off, batch-norm running stats
    logits = model(X)
    loss = nn.functional.cross_entropy(logits, y).item()
    proba = torch.softmax(logits, dim=1).cpu().numpy()
    f1 = f1_score(y.cpu().numpy(), proba.argmax(1), average="macro",
                  labels=list(range(N_CLASSES)), zero_division=0)
    return loss, float(f1), proba


def train_one(Xtr, ytr, Xva, yva, params, seed, device, max_epochs=MAX_EPOCHS,
              patience=PATIENCE, verbose=False):
    """
    One training run. The ten numbered steps are the section 14 checklist.
    Returns (best_val_f1, history, best_state).
    """
    set_seed(seed)
    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])
    criterion = nn.CrossEntropyLoss()          # unweighted, per section 15

    best_f1, best_state, best_epoch, since = -1.0, None, -1, 0
    history = []
    n = Xtr.shape[0]

    for epoch in range(max_epochs):
        t0 = time.perf_counter()
        model.train()                          # dropout on, batch-norm batch stats
        perm = torch.randperm(n, device=device)
        run_loss, n_batches, grad_norms = 0.0, 0, []

        for start in range(0, n, BATCH_SIZE):
            # (1) mini-batch loading
            idx = perm[start:start + BATCH_SIZE]
            if idx.numel() < 2:                # batch norm needs at least 2
                continue
            xb, yb = Xtr[idx], ytr[idx]

            opt.zero_grad(set_to_none=True)    # (2) gradient reset
            logits = model(xb)                 # (3) forward pass
            loss = criterion(logits, yb)       # (4) loss computation
            loss.backward()                    # (5) backpropagation
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()                         # (6) optimizer update

            run_loss += loss.item()
            n_batches += 1
            grad_norms.append(float(gn))

        train_loss = run_loss / max(n_batches, 1)
        _, train_f1, _ = evaluate(model, Xtr, ytr)
        val_loss, val_f1, _ = evaluate(model, Xva, yva)   # (7) validation

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_macro_f1": train_f1, "val_macro_f1": val_f1,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
            "epoch_seconds": round(time.perf_counter() - t0, 3),
        })
        if verbose:
            h = history[-1]
            print(f"    epoch {epoch:3d} train {train_loss:.4f} "
                  f"val {val_loss:.4f} val-F1 {val_f1:.4f} "
                  f"|g| {h['grad_norm_mean']:.3f} {h['epoch_seconds']:.2f}s")

        if val_f1 > best_f1:                   # (9) checkpoint saving
            best_f1, best_epoch, since = val_f1, epoch, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:              # (8) early stopping
                break

    if best_state is not None:
        model.load_state_dict(best_state)      # (10) checkpoint restoration
    return best_f1, history, model, best_epoch


# --------------------------------------------------------------------------
# Section 14 demonstrations
# --------------------------------------------------------------------------

def demo_overfit(df, cols, device) -> int:
    """
    Small-batch overfitting test.

    A working network must be able to memorise a handful of examples. Failure
    points at implementation errors, wrong labels, broken gradient flow, an
    unsuitable learning rate or a preprocessing defect, and it must be ruled
    out before any result is believed.
    """
    print("\n=== Section 14: small-batch overfitting test ===")
    sub = df.groupby("class_id", group_keys=False).head(8)   # 32 samples
    pre = FeaturePreprocessor(cols)
    X = torch.tensor(pre.fit_transform(sub), dtype=torch.float32, device=device)
    y = torch.tensor(sub["class_id"].to_numpy(), dtype=torch.long, device=device)
    print(f"  {len(sub)} samples, {len(cols)} features, "
          f"class counts {sub['class_id'].value_counts().sort_index().tolist()}")

    f1, hist, model, _ = train_one(X, y, X, y, {"lr": 1e-3, "weight_decay": 0.0},
                                   SEED, device, max_epochs=400, patience=400)
    _, final_f1, proba = evaluate(model, X, y)
    acc = float((proba.argmax(1) == y.cpu().numpy()).mean())
    print(f"  final training loss {hist[-1]['train_loss']:.6f} | "
          f"accuracy {acc:.4f} | macro-F1 {final_f1:.4f} | "
          f"{len(hist)} epochs | {model.n_params()} parameters")
    ok = acc >= 0.95
    print(f"  {'PASS' if ok else 'FAIL'}: the network "
          f"{'can' if ok else 'CANNOT'} overfit a small subset")
    return 0 if ok else 1


def _stratified_sample(d: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Take roughly n rows with every class represented.

    Plain .head(n) is wrong here: the manifest is ordered by measurement and
    the measurements are grouped by class, so the first rows are all drones.
    That produced a single-class subset, a degenerate model and a meaningless
    macro-F1 of exactly 0.25.
    """
    per = max(2, n // d["class_id"].nunique())
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in d.groupby("class_id", sort=True):
        take = min(per, len(g))
        picks.append(g.iloc[rng.choice(len(g), size=take, replace=False)])
    return pd.concat(picks).reset_index(drop=True)


def demo_repro(df, cols, device) -> int:
    """Two identical runs at the same seed must give identical numbers."""
    print("\n=== Section 14: fixed-seed reproducibility test ===")
    tr = _stratified_sample(df[df["outer_fold"] != 0], 8000, SEED)
    va = _stratified_sample(df[df["outer_fold"] == 0], 2000, SEED)
    print(f"  train {len(tr)} rows {tr['class_id'].value_counts().sort_index().tolist()}"
          f" | val {len(va)} rows {va['class_id'].value_counts().sort_index().tolist()}")
    if tr["class_id"].nunique() < N_CLASSES or va["class_id"].nunique() < N_CLASSES:
        print("  FAIL: the subset does not contain all four classes")
        return 1
    pre = FeaturePreprocessor(cols)
    Xtr = torch.tensor(pre.fit_transform(tr), dtype=torch.float32, device=device)
    ytr = torch.tensor(tr["class_id"].to_numpy(), dtype=torch.long, device=device)
    Xva = torch.tensor(pre.transform(va), dtype=torch.float32, device=device)
    yva = torch.tensor(va["class_id"].to_numpy(), dtype=torch.long, device=device)

    out, preds = [], []
    for run in (1, 2):
        f1, hist, model, best_ep = train_one(Xtr, ytr, Xva, yva,
                                             {"lr": 1e-3, "weight_decay": 1e-4},
                                             SEED, device, max_epochs=25,
                                             patience=25)
        _, _, proba = evaluate(model, Xva, yva)
        out.append((f1, [h["train_loss"] for h in hist], best_ep))
        preds.append(proba)
        print(f"  run {run}: best val macro-F1 {f1:.10f}, "
              f"best epoch {best_ep}, {len(hist)} epochs, "
              f"predicts {len(np.unique(proba.argmax(1)))} distinct classes")

    # A reproducible degenerate model is still degenerate. The test only means
    # something if the run it reproduces is a real one.
    n_pred = len(np.unique(preds[0].argmax(1)))
    if n_pred < 2 or out[0][0] <= 1.0 / N_CLASSES + 1e-9:
        print(f"  FAIL: the run is degenerate (macro-F1 {out[0][0]:.4f}, "
              f"{n_pred} class(es) predicted). Reproducibility is not "
              f"demonstrated by repeating a collapsed model.")
        return 1

    same_f1 = out[0][0] == out[1][0]
    same_curve = np.array_equal(out[0][1], out[1][1])
    same_proba = np.array_equal(preds[0], preds[1])
    print(f"  identical best val macro-F1:            {same_f1}")
    print(f"  identical per-epoch training loss curve: {same_curve}")
    print(f"  identical predicted probabilities:       {same_proba}")
    ok = same_f1 and same_curve and same_proba
    print(f"  {'PASS' if ok else 'FAIL'}: seed {SEED} "
          f"{'reproduces exactly on a non-degenerate run' if ok else 'DOES NOT reproduce'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def load(exclude_edge=True, feature_set="full"):
    feats = pd.read_csv(ARTIFACT_DIR / "s1_features.csv")
    folds = pd.read_csv(ARTIFACT_DIR / "fold_manifest.csv")
    # s1_features.csv already carries class_id and measurement_id, so only
    # the columns unique to the fold manifest are pulled across. Merging both
    # copies would silently produce class_id_x / class_id_y.
    keep = ["segment_uid", "class_name", "edge_flag", "outer_fold"] + [
        f"inner_fold_for_outer_{k}" for k in range(N_OUTER)]
    df = feats.merge(folds[keep], on="segment_uid", validate="one_to_one")
    if exclude_edge:
        n0 = len(df)
        df = df[df["edge_flag"] == 0].reset_index(drop=True)
        print(f"Excluded {n0 - len(df)} edge segments ({n0} -> {len(df)})")
    cols = list(FEATURE_NAMES if feature_set == "full" else COMPACT_SUBSET)
    return df, cols


def to_tensors(pre, fit_df, use_df, device, fit=False):
    X = pre.fit_transform(fit_df) if fit else pre.transform(use_df)
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(use_df["class_id"].to_numpy() if not fit
                         else fit_df["class_id"].to_numpy(),
                         dtype=torch.long, device=device))


def run_nested(df, cols, device) -> dict:
    grid = list(ParameterGrid(GRID))
    print(f"\n{'='*66}\nMLP: {len(grid)} configs, "
          f"{len(grid)*N_INNER*N_OUTER} inner runs + {N_OUTER} final\n{'='*66}")
    fold_results, selected, oof, histories = [], [], [], {}
    t_model = time.perf_counter()

    for k in range(N_OUTER):
        seed = OUTER_FOLD_SEEDS[k]
        outer_tr = df[df["outer_fold"] != k]
        outer_te = df[df["outer_fold"] == k]
        icol = f"inner_fold_for_outer_{k}"
        t0 = time.perf_counter()

        scores = np.zeros((len(grid), N_INNER))
        for ci, params in enumerate(grid):
            for j in range(N_INNER):
                tr = outer_tr[outer_tr[icol] != j]
                va = outer_tr[outer_tr[icol] == j]
                pre = FeaturePreprocessor(cols)
                Xtr, ytr = to_tensors(pre, tr, tr, device, fit=True)
                Xva, yva = to_tensors(pre, tr, va, device)
                f1, _, _, _ = train_one(Xtr, ytr, Xva, yva, params, seed, device)
                scores[ci, j] = f1

        mean_inner = scores.mean(axis=1)
        best_i = int(np.argmax(mean_inner))
        best = grid[best_i]
        selected.append(best)
        inner_s = time.perf_counter() - t0

        # Final model on the full outer-training portion. The inner fold 0 of
        # this outer fold serves as the early-stopping signal, so no outer-test
        # data is ever used to decide when to stop.
        fit_tr = outer_tr[outer_tr[icol] != 0]
        stop_va = outer_tr[outer_tr[icol] == 0]
        pre = FeaturePreprocessor(cols)
        Xtr, ytr = to_tensors(pre, fit_tr, fit_tr, device, fit=True)
        Xva, yva = to_tensors(pre, fit_tr, stop_va, device)
        Xte, yte = to_tensors(pre, fit_tr, outer_te, device)

        t1 = time.perf_counter()
        _, hist, model, best_ep = train_one(Xtr, ytr, Xva, yva, best, seed,
                                            device)
        train_s = time.perf_counter() - t1
        histories[k] = hist

        _, _, proba = evaluate(model, Xte, yte)
        oof.append(pd.DataFrame({
            "segment_uid": outer_te["segment_uid"].to_numpy(),
            "measurement_id": outer_te["measurement_id"].to_numpy(),
            "outer_fold": k, "true": outer_te["class_id"].to_numpy(),
            "pred": proba.argmax(1),
            **{f"p_{CLASS_NAMES[c]}": proba[:, c] for c in range(N_CLASSES)},
        }))
        m = evaluate_classification(proba, outer_te["class_id"].to_numpy())
        m.update(outer_fold=k, seed=seed, params=best,
                 mean_inner_macro_f1=float(mean_inner[best_i]),
                 best_epoch=best_ep, n_epochs=len(hist),
                 parameter_count=model.n_params(),
                 inner_seconds=round(inner_s, 1),
                 final_fit_seconds=round(train_s, 1))
        fold_results.append(m)

        model_dir = ARTIFACT_DIR / "models"
        model_dir.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "features": cols,
                    "params": best, "seed": seed, "outer_fold": k,
                    "scaler_mean": pre.scaler.mean_,
                    "scaler_scale": pre.scaler.scale_,
                    "class_names": CLASS_NAMES},
                   model_dir / f"mlp_fold{k}.pt")

        elapsed = time.perf_counter() - t_model
        print(f"  fold {k}: inner {inner_s/60:.1f} min | final {train_s:.1f}s "
              f"| best {best} | stopped epoch {best_ep}")
        print(f"          inner F1 {mean_inner[best_i]:.4f} | "
              f"TEST F1 {m['macro_f1']:.4f} | elapsed {elapsed/60:.1f} min")

    pd.concat(oof, ignore_index=True).to_csv(
        ARTIFACT_DIR / "oof_predictions_mlp.csv", index=False)
    pd.concat([pd.DataFrame(h).assign(outer_fold=k)
               for k, h in histories.items()], ignore_index=True).to_csv(
        ARTIFACT_DIR / "mlp_training_curves.csv", index=False)

    f1s = [r["macro_f1"] for r in fold_results]
    summary = {
        "model_name": "MLP", "architecture": f"{len(cols)}-64-32-{N_CLASSES}",
        "parameter_count": fold_results[0]["parameter_count"],
        "outer_fold_scores": [round(v, 4) for v in f1s],
        "mean_macro_f1": float(np.mean(f1s)), "std_macro_f1": float(np.std(f1s)),
        "accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
        "ece": float(np.mean([r["ece"] for r in fold_results])),
        "brier_score": float(np.mean([r["brier_score"] for r in fold_results])),
        "selected_params_per_fold": selected,
        "wall_seconds": round(time.perf_counter() - t_model, 1),
        "per_fold": fold_results,
    }
    print(f"\n  MLP: macro-F1 {summary['mean_macro_f1']:.4f} "
          f"+/- {summary['std_macro_f1']:.4f} | "
          f"{summary['parameter_count']} parameters | "
          f"{summary['wall_seconds']/60:.1f} min")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=["overfit", "repro"])
    ap.add_argument("--features", choices=["full", "compact"], default="full")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-edge", action="store_true")
    args = ap.parse_args()

    print(banner())
    device = resolve_device(args.device)
    print(f"  device       {device}\n")
    df, cols = load(exclude_edge=not args.keep_edge, feature_set=args.features)

    if args.demo == "overfit":
        sys.exit(demo_overfit(df, cols, device))
    if args.demo == "repro":
        sys.exit(demo_repro(df, cols, device))

    summary = run_nested(df, cols, device)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    out = ARTIFACT_DIR / "results_mlp.json"
    out.write_text(json.dumps(
        {"generated": datetime.now(timezone.utc).isoformat(), "commit": commit,
         "host": platform.node(), "device": str(device), "features": cols,
         "seed": SEED, "outer_fold_seeds": list(OUTER_FOLD_SEEDS),
         "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
         "patience": PATIENCE, "grad_clip": GRAD_CLIP, "dropout": DROPOUT,
         "model": summary}, indent=2, default=str))
    print(f"Wrote {out.name}, oof_predictions_mlp.csv, mlp_training_curves.csv")


if __name__ == "__main__":
    main()
