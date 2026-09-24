"""compute_dataset_statistics.py - Populate Section 4 of the data paper with real numbers.

Run this LOCALLY, from a Python environment that has both sibling projects'
dependencies installed. Both are expected one directory up from this file:

    ../drought_forecast_regression   (continuous SPI)
    ../drought_forecast_binary       (binary drought event)

Usage:
    python compute_dataset_statistics.py --task regression --region Sul
    python compute_dataset_statistics.py --task regression --all-regions
    python compute_dataset_statistics.py --task binary --region Sul

Both projects define top-level modules with the same names (`config`,
`data`, ...), so this script puts only ONE project on sys.path at a time,
selected by --task, to avoid import collisions. If you want both tasks in
one run, run this script twice (once per --task) rather than editing it
to import both simultaneously.
"""

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BINARY_PROJECT = Path(__file__).resolve().parent.parent / "drought_forecast_binary"
REGRESSION_PROJECT = Path(__file__).resolve().parent.parent / "drought_forecast_regression"

REGIONS = ["Sul", "Sudeste", "Nordeste", "Centro-Oeste", "Norte"]
P_VALUES = [3, 6, 12]
Q_VALUES = [1, 3, 6, 12]
KAPPA_VALUES = [-1.0, -1.5, -2.0]  # binary framework only


def compute_regression_stats(region: str, base_dir: str = None) -> dict:
    """Continuous SPI dataset stats: target_mean, target_std, sample counts per (p,q)."""
    sys.path.insert(0, str(REGRESSION_PROJECT))
    from config import ExperimentConfig, get_paths, get_data_path
    import config.paths as _paths_mod
    from data import load_region_timeseries, load_spi_cache
    from data.dataset import ClimateDataset

    if base_dir:
        _paths_mod.get_output_path = lambda: Path(base_dir)

    config = ExperimentConfig()
    config.region = region
    paths = get_paths(region)
    out = load_region_timeseries(get_data_path(), config)
    spi, _ = load_spi_cache(config.spi.scale, paths["spi_cache_dir"])

    region_stats = {"region": region, "per_pq": {}}

    for p in P_VALUES:
        for q in Q_VALUES:
            ds = ClimateDataset(
                data=out["data"],
                spi=spi,
                months=out["months"],
                p=p,
                q=q,
                valid_mask=out["valid_mask"],
                mode="regression",
                verbose=False,
            )
            info = ds.get_data_info()
            region_stats["per_pq"][f"p{p}_q{q}"] = {
                "num_samples": info["num_samples"],
                "target_mean": info.get("target_mean"),
                "target_std": info.get("target_std"),
            }

    return region_stats


def compute_binary_stats(region: str, base_dir: str = None) -> dict:
    """Binary drought dataset stats: drought_ratio, drought_prevalence per (p,q,kappa)."""
    sys.path.insert(0, str(BINARY_PROJECT))
    from config import ExperimentConfig, get_paths, get_data_path
    import config.paths as _paths_mod
    from data import load_region_timeseries, load_spi_cache
    from data.dataset import ClimateDataset  # classification-mode version in this project

    if base_dir:
        _paths_mod.get_output_path = lambda: Path(base_dir)

    config = ExperimentConfig()
    config.region = region
    paths = get_paths(region)
    out = load_region_timeseries(get_data_path(), config)
    spi, _ = load_spi_cache(config.spi.scale, paths["spi_cache_dir"])

    region_stats = {"region": region, "per_pqk": {}}

    for kappa in KAPPA_VALUES:
        config.spi.threshold = kappa
        for p in P_VALUES:
            for q in Q_VALUES:
                ds = ClimateDataset(
                    data=out["data"],
                    spi=spi,
                    months=out["months"],
                    p=p,
                    q=q,
                    valid_mask=out["valid_mask"],
                    mode="classification",
                    spi_threshold=kappa,
                    verbose=False,
                )
                info = ds.get_data_info()
                region_stats["per_pqk"][f"p{p}_q{q}_k{kappa}"] = {
                    "num_samples": info.get("num_samples"),
                    # get_data_info() doesn't expose these; they're only
                    # set as instance attributes on ClimateDataset.
                    "drought_ratio": ds.drought_ratio,
                    "drought_prevalence": ds.drought_prevalence,
                }

    return region_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--all-regions", action="store_true")
    parser.add_argument("--task", choices=["regression", "binary"], default="regression")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--base-dir", type=str, default=None,
                         help="Override the project's outputs/ directory "
                              "(where spi_cache/, grid_search/, etc. live).")
    args = parser.parse_args()

    regions = REGIONS if args.all_regions else [args.region or "Sul"]
    out_path = args.out or f"dataset_statistics_{args.task}.json"

    results = []
    for region in regions:
        print(f"Computing statistics for {region} ({args.task})...")
        if args.task == "regression":
            results.append(compute_regression_stats(region, args.base_dir))
        else:
            results.append(compute_binary_stats(region, args.base_dir))

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
