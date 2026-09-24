"""run_baselines.py - Simple baselines for Section 5 of the data paper.

Run LOCALLY, pointed at whichever sibling project the --task needs (both are
expected one directory up from this file):

    ../drought_forecast_regression   (--task regression)
    ../drought_forecast_binary       (--task classification)

Trains persistence / linear / logistic / random-forest baselines per pixel,
using the same chronological train/val/test split as the main framework.
Not the full ConvLSTM pipeline - just proof the datasets are learnable.

By default sweeps a reduced (p, q) grid (p=[3,6,12], q=[1,3,6,12], 12
combinations) - smaller than the real grid search's full grid
(experiments/grid_search.py: p_values=[3,6,9,12], q_values=[1,3,6,9,12]),
since the goal here is just to show the datasets are learnable across
horizons, not to tune hyperparameters. Pass --p/--q to restrict further.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, r2_score, recall_score, roc_auc_score,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BINARY_PROJECT = Path(__file__).resolve().parent.parent / "drought_forecast_binary"
REGRESSION_PROJECT = Path(__file__).resolve().parent.parent / "drought_forecast_regression"

P_VALUES = [3, 6, 12]
Q_VALUES = [1, 3, 6, 12]


def load_region_data(region: str, task: str, base_dir: str = None):
    """Load the region's raw data + SPI once; reused for every (p, q) pair."""
    project = REGRESSION_PROJECT if task == "regression" else BINARY_PROJECT
    sys.path.insert(0, str(project))
    from config import ExperimentConfig, get_paths, get_data_path
    import config.paths as _paths_mod
    from data import load_region_timeseries, load_spi_cache

    if base_dir:
        _paths_mod.get_output_path = lambda: Path(base_dir)

    config = ExperimentConfig()
    config.region = region
    paths = get_paths(region)
    out = load_region_timeseries(get_data_path(), config)
    spi, _ = load_spi_cache(config.spi.scale, paths["spi_cache_dir"])

    time_idx = np.array([y * 12 + (m - 1) for y, m in zip(out["years"], out["months"])])
    split = config.split
    train_end = split.ym_to_int(split.train_gs[1])
    val_end = split.ym_to_int(split.val_gs[1])

    return {
        "data": out["data"],             # (T, H, W, C)
        "valid_mask": out["valid_mask"],  # (H, W)
        "spi": spi,
        "time_idx": time_idx,
        "train_end": train_end,
        "val_end": val_end,
        "threshold": None if task == "regression" else config.spi.threshold,
    }


def build_flat_dataset(region_data: dict, p: int, q: int, task: str):
    """
    Build a flat (N_samples, N_features) / (N_samples,) array pair for sklearn.

    Each sample is one pixel-month: features are the p months of the 7
    climate variables at that pixel (flattened, p*7), plus the last
    observed SPI value at that pixel (for the persistence baseline).
    Target is SPI (continuous) or SPI<=threshold (binary) at t+p+q-1.
    """
    data = region_data["data"]
    valid_mask = region_data["valid_mask"]
    spi = region_data["spi"]
    time_idx = region_data["time_idx"]
    train_end = region_data["train_end"]
    val_end = region_data["val_end"]
    threshold = region_data["threshold"]

    T = data.shape[0]
    X_list, y_list, persist_list, split_list = [], [], [], []
    ys_idx, xs_idx = np.where(valid_mask)

    for t in range(T - p - q + 1):
        target_idx = t + p + q - 1
        if target_idx >= T:
            continue

        window = data[t:t + p]           # (p, H, W, C)
        last_spi = spi[t + p - 1]        # (H, W) - SPI at the end of the input window
        target_spi = spi[target_idx]     # (H, W)

        this_time = time_idx[target_idx]
        split_name = "train" if this_time <= train_end else ("val" if this_time <= val_end else "test")

        for yy, xx in zip(ys_idx, xs_idx):
            feat = window[:, yy, xx, :].reshape(-1)
            if not np.all(np.isfinite(feat)):
                continue
            tgt = target_spi[yy, xx]
            persist_val = last_spi[yy, xx]
            if not (np.isfinite(tgt) and np.isfinite(persist_val)):
                continue

            X_list.append(feat)
            y_list.append(tgt if task == "regression" else float(tgt <= threshold))
            persist_list.append(persist_val if task == "regression" else float(persist_val <= threshold))
            split_list.append(split_name)

    return (np.array(X_list), np.array(y_list), np.array(persist_list), np.array(split_list))


def run_regression_baselines(X, y, persist, split_arr):
    train_mask, test_mask = split_arr == "train", split_arr == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    persist_test = persist[test_mask]

    def reg_metrics(pred, obs):
        err = pred - obs
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        obar = obs.mean()
        denom = np.sum((np.abs(pred - obar) + np.abs(obs - obar)) ** 2)
        wi = 1.0 if denom < 1e-12 else float(np.clip(1 - np.sum(err ** 2) / denom, 0, 1))
        return {"wi": wi, "rmse": rmse, "mae": mae, "r2": r2_score(obs, pred)}

    results = {"persistence": reg_metrics(persist_test, y_test)}

    lr = LinearRegression().fit(X_train, y_train)
    results["linear_regression"] = reg_metrics(lr.predict(X_test), y_test)

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    results["random_forest"] = reg_metrics(rf.predict(X_test), y_test)

    return results


def run_classification_baselines(X, y, persist, split_arr):
    train_mask, test_mask = split_arr == "train", split_arr == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    persist_test = persist[test_mask]

    def clf_metrics(pred, proba, obs):
        return {
            "f1": f1_score(obs, pred),
            "auc_roc": roc_auc_score(obs, proba) if len(np.unique(obs)) > 1 else float("nan"),
            "precision": precision_score(obs, pred, zero_division=0),
            "recall": recall_score(obs, pred, zero_division=0),
        }

    results = {"persistence": clf_metrics(persist_test, persist_test, y_test)}

    logit = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
    proba = logit.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = clf_metrics((proba >= 0.5).astype(int), proba, y_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = clf_metrics((proba >= 0.5).astype(int), proba, y_test)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="Centro-Oeste")
    parser.add_argument("--p", type=int, default=None, nargs="+",
                         help="Restrict to these p value(s) instead of the full grid %s." % P_VALUES)
    parser.add_argument("--q", type=int, default=None, nargs="+",
                         help="Restrict to these q value(s) instead of the full grid %s." % Q_VALUES)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--base-dir", type=str, default=None,
                         help="Override the project's outputs/ directory "
                              "(where spi_cache/, grid_search/, etc. live).")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    p_values = args.p or P_VALUES
    q_values = args.q or Q_VALUES

    print(f"Loading region data: region={args.region}, task={args.task}")
    region_data = load_region_data(args.region, args.task, args.base_dir)

    all_results = []
    for p in p_values:
        for q in q_values:
            print(f"\n=== p={p}, q={q} ===")
            X, y, persist, split_arr = build_flat_dataset(region_data, p, q, args.task)
            print(f"  N={len(y)}  train={np.sum(split_arr == 'train')}  "
                  f"val={np.sum(split_arr == 'val')}  test={np.sum(split_arr == 'test')}")

            results = (run_regression_baselines(X, y, persist, split_arr) if args.task == "regression"
                       else run_classification_baselines(X, y, persist, split_arr))

            for name, metrics in results.items():
                print(f"  {name}:")
                for k, v in metrics.items():
                    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

            all_results.append({"p": p, "q": q, "n_samples": int(len(y)), "results": results})

    out_path = args.out or f"baselines_{args.task}_{args.region}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
