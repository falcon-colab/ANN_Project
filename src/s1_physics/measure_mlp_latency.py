"""
src/s1_physics/measure_mlp_latency.py

The one Section 18 metric the MLP is missing: batch-one inference latency.

train_mlp.py times training but not inference, so job 2305 produced latency for
the three classical models and nothing for the network. Retraining to get it
would waste 50 GPU minutes and, because the network is only reproducible within
a device, would also produce a second set of slightly different weights. This
script instead loads the five checkpoints that job 2305 already wrote to
ARTIFACT_DIR/models/mlp_fold{k}.pt and times them under exactly the protocol
train_classical.py uses: batch size 1, 100 warm-up inferences, 1,000 timed
inferences, data loading excluded, repeated five times and averaged, with mean
and 95th percentile reported per repeat.

Two details specific to timing a GPU:

  * torch.cuda.synchronize() is called after every inference. CUDA launches are
    asynchronous, so without it the loop measures the time to enqueue work, not
    the time to finish it, which understates latency by an order of magnitude.
  * the input tensor is created before the timer starts and reused, so the
    measurement excludes host-to-device transfer of new data, matching
    "data-loading time excluded" in the protocol.

Batch-one latency for a fixed-size input does not depend on which segment is
fed in, so the inputs are a fixed stratified sample drawn with seed 2026. The
preprocessing applied to them is taken from each checkpoint's stored scaler, so
each fold's model sees inputs on the scale it was trained with.

Usage on Atlas, inside a job on the same node type as the training run:

    python src/s1_physics/measure_mlp_latency.py               # cuda if present
    python src/s1_physics/measure_mlp_latency.py --also-cpu    # both devices

Writes results_mlp_latency.json next to the other results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))

from config import ARTIFACT_DIR, banner                      # noqa: E402
from feature_transforms import HEAVY_TAILED, signed_log1p    # noqa: E402
from train_mlp import MLP, resolve_device                    # noqa: E402

N_WARMUP = 100
N_TIMED = 1000
N_REPEATS = 5
SAMPLE_ROWS = 1000
SEED = 2026


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Equal numbers per class where possible, so the sample is not all drones."""
    rng = np.random.default_rng(seed)
    per_class = max(1, n // df["class_id"].nunique())
    parts = []
    for _, grp in df.groupby("class_id", sort=True):
        take = min(per_class, len(grp))
        idx = rng.choice(len(grp), size=take, replace=False)
        parts.append(grp.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def prepare_inputs(df: pd.DataFrame, features: list[str],
                   mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Reproduce FeaturePreprocessor.transform from the checkpoint's scaler."""
    X = df[features].to_numpy(dtype=float)
    cols = [i for i, c in enumerate(features) if c in HEAVY_TAILED]
    if cols:
        X = X.copy()
        X[:, cols] = signed_log1p(X[:, cols])
    return (X - mean) / scale


@torch.inference_mode()
def time_one(model: torch.nn.Module, X: np.ndarray, device: torch.device,
             n_warmup: int, n_timed: int) -> np.ndarray:
    """One pass of the protocol. Returns per-inference times in milliseconds."""
    cuda = device.type == "cuda"
    rows = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = rows.shape[0]

    for i in range(n_warmup):
        model(rows[i % n].unsqueeze(0))
        if cuda:
            torch.cuda.synchronize()

    times = np.empty(n_timed, dtype=float)
    for i in range(n_timed):
        x = rows[i % n].unsqueeze(0)            # view, no copy, no host transfer
        t0 = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        times[i] = (time.perf_counter() - t0) * 1e3
    return times


def load_fold(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    features = list(ckpt["features"])
    model = MLP(len(features)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()                                # dropout off, running BN stats
    return model, features, np.asarray(ckpt["scaler_mean"]), \
        np.asarray(ckpt["scaler_scale"]), ckpt


def measure(device: torch.device, args) -> dict:
    feats_df = pd.read_csv(ARTIFACT_DIR / "s1_features.csv")
    sample = stratified_sample(feats_df, SAMPLE_ROWS, SEED)
    model_dir = ARTIFACT_DIR / "models"
    paths = sorted(model_dir.glob("mlp_fold*.pt"))
    if not paths:
        raise FileNotFoundError(f"no mlp_fold*.pt in {model_dir}")

    per_fold = []
    for path in paths:
        model, features, mean, scale, ckpt = load_fold(path, device)
        X = prepare_inputs(sample, features, mean, scale)
        means, p95s = [], []
        for r in range(args.repeats):
            t = time_one(model, X, device, args.warmup, args.timed)
            means.append(float(t.mean()))
            p95s.append(float(np.percentile(t, 95)))
            print(f"  {path.name} repeat {r + 1}: mean {means[-1]:.4f} ms, "
                  f"p95 {p95s[-1]:.4f} ms", flush=True)
        per_fold.append({
            "checkpoint": path.name,
            "outer_fold": int(ckpt.get("outer_fold", -1)),
            "parameter_count": sum(p.numel() for p in model.parameters()
                                   if p.requires_grad),
            "mean_latency_ms": float(np.mean(means)),
            "p95_latency_ms": float(np.mean(p95s)),
            "latency_repeats": args.repeats,
            "latency_mean_spread_ms": float(max(means) - min(means)),
            "latency_per_repeat_ms": [round(m, 4) for m in means],
        })

    fold_means = [f["mean_latency_ms"] for f in per_fold]
    return {
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device)
                        if device.type == "cuda" else "cpu"),
        "n_warmup": args.warmup,
        "n_timed": args.timed,
        "repeats": args.repeats,
        "mean_latency_ms": float(np.mean(fold_means)),
        "p95_latency_ms": float(np.mean([f["p95_latency_ms"] for f in per_fold])),
        "fold_spread_ms": float(max(fold_means) - min(fold_means)),
        "worst_repeat_spread_ms": float(max(f["latency_mean_spread_ms"]
                                            for f in per_fold)),
        "per_fold": per_fold,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--also-cpu", action="store_true",
                    help="measure on the CPU as well, for a fair comparison "
                         "with the classical models, which are CPU-timed")
    ap.add_argument("--warmup", type=int, default=N_WARMUP)
    ap.add_argument("--timed", type=int, default=N_TIMED)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()

    print(banner(), "\n")
    torch.manual_seed(SEED)

    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "protocol": f"batch 1, {args.warmup} warm-up, {args.timed} timed, "
                       f"{args.repeats} repeats, data loading excluded",
           "devices": {}}

    devices = [resolve_device(args.device)]
    if args.also_cpu and devices[0].type != "cpu":
        devices.append(torch.device("cpu"))

    for dev in devices:
        print(f"=== batch-one latency on {dev} ===")
        res = measure(dev, args)
        out["devices"][str(dev)] = res
        print(f"  {dev}: mean {res['mean_latency_ms']:.4f} ms, "
              f"p95 {res['p95_latency_ms']:.4f} ms, "
              f"fold spread {res['fold_spread_ms']:.4f} ms, "
              f"worst repeat spread {res['worst_repeat_spread_ms']:.4f} ms\n")

    path = ARTIFACT_DIR / "results_mlp_latency.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
