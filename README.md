# Two Multivariate Spatiotemporal Climate Datasets for Drought Forecasting in Brazil

Statistical characterization and baseline validation for the two machine-learning-ready datasets used across this thesis: a **binary drought-event** dataset (`drought_forecast_binary`) and a **continuous SPI-regression** dataset (`drought_forecast_regression`). Both are derived from the same 45-year (1980–2024) TerraClimate record, accessed via the Google Earth Engine Python API, covering Brazil's five macro-regions, and share the same spatial validity mask, monthly-anomaly representation, and chronological train/validation/test partition — differing only in their prediction target.

This repository does **not** implement a data pipeline of its own. It is a thin analysis layer that characterizes datasets built and cached by the two sibling projects, and validates them with simple baselines before any deep-learning model is trained on them.

---

## How this project relates to its two sibling projects

Neither raw-data acquisition nor SPI computation lives here. Both scripts below add exactly **one** of `../drought_forecast_binary` or `../drought_forecast_regression` to `sys.path` at runtime — selected via `--task` — and reuse that project's own `config`/`data` modules to load the already-preprocessed climate cubes and cached SPI. Because the two sibling projects define top-level modules with identical names (`config`, `data`, ...), only one can be on `sys.path` per process; if you need both, run the script twice (once per `--task`), rather than editing it to import both simultaneously.

**Prerequisite:** the target sibling project must already have a populated `outputs/<region>/spi_cache/spi_scale_3.pkl` (i.e. its own `precompute-spi` step must have run first — see that project's README).

---

## Project Structure

```
.
├── compute_dataset_statistics.py   # SPI distribution / drought prevalence stats per (p,q[,kappa])
├── run_baselines.py                # Persistence / linear-or-logistic / random-forest baselines per pixel
├── summarize_all_regions.py        # One-off script: prints the cross-region summary tables used in the paper
├── dataset_statistics_regression.json
├── dataset_statistics_binary.json
├── baselines_regression_<Region>.json
├── baselines_classification_<Region>.json
```

---

## Methodology Notes

### Two different notions of "sample count"

`compute_dataset_statistics.py` reports `num_samples` from `ClimateDataset.get_data_info()`, which counts valid **timesteps** (whole-region months with a complete input window and a valid target month) — e.g. 537 for South at `p=3, q=1`. `run_baselines.py` instead builds one row per **pixel-month**, which is orders of magnitude larger (881,754 for the same region and `(p,q)`) since every valid pixel within each valid timestep becomes its own training sample. Both are correct; they just answer different questions ("how many distinct forecast instants?" vs. "how many training examples for a per-pixel model?").

### Leakage-free splitting, reimplemented independently

`run_baselines.py` does not reuse the sibling project's `ClimateDataset` for sample construction — it rebuilds the sliding window in plain NumPy, but reads the exact same calendar boundaries from that project's own `SplitConfig` (train ≤ 2019-12, validation ≤ 2022-12, test = 2023–2024). A sample's train/val/test membership is decided by its **target month's** absolute time, never by its input window's start, so a window may look back across a split boundary without leaking future information into training.

### `drought_prevalence` vs. `drought_ratio`

For the binary dataset, two related but distinct statistics are reported per region and severity threshold κ:
- **`drought_prevalence`** — fraction of all pixel-months in the *entire* SPI series below κ. Depends only on κ, so it is constant across every `(p, q)` for a fixed threshold.
- **`drought_ratio`** — fraction of *valid dataset samples* whose target month contains at least one drought pixel anywhere in the region. Varies slightly with `(p, q)`, since the set of eligible target months changes.

`summarize_all_regions.py` also prints the theoretical prevalence expected under a standard-normal SPI, Φ(κ) — Φ(−1.0) ≈ 15.87%, Φ(−1.5) ≈ 6.68%, Φ(−2.0) ≈ 2.28% — alongside the empirical value, to show how closely the zero-inflated-Gamma SPI construction tracks a true standard normal at each severity level.

### Baselines

Three model families are trained per pixel, using the reduced `(p, q)` grid below:
- **Persistence** — no fitting; the last observed SPI in the input window is the prediction (or, for classification, whether that value already falls below κ).
- **Linear / Logistic Regression** (`class_weight="balanced"` for classification).
- **Random Forest** (`n_estimators=100, max_depth=12`; `class_weight="balanced"` for the classifier).

Metrics: WI (Willmott's Index of Agreement — the same formula and role used as the primary model-selection metric in the deep-learning frameworks, so these numbers are directly comparable), RMSE, MAE, R² for the regression task; F1, AUC-ROC, precision, recall for classification.

### Reduced `(p, q)` grid

By default, `run_baselines.py` sweeps `p ∈ {3, 6, 12}`, `q ∈ {1, 3, 6, 12}` (12 combinations) — smaller than the deep-learning frameworks' full grid search (`p ∈ {3, 6, 9, 12}`, `q ∈ {1, 3, 6, 9, 12}`, 20 combinations), since the goal here is only to show that the datasets carry learnable signal across horizons, not to tune hyperparameters. Pass `--p`/`--q` to restrict further.

---

## Quick Start

```bash
# Dataset statistics (SPI distribution, drought prevalence)
python compute_dataset_statistics.py --task regression --region Sul
python compute_dataset_statistics.py --task binary --all-regions

# Simple baselines per pixel
python run_baselines.py --task regression --region Sul
python run_baselines.py --task classification --region Sul --p 12 --q 1

# Cross-region summary tables (reads the JSON files produced above)
python summarize_all_regions.py
```

`--base-dir` on the first two scripts overrides the sibling project's `outputs/` directory, if you keep it somewhere other than that project's own root.

---

## Requirements

This project's own direct dependencies:

```
numpy
scipy
scikit-learn
```

Running either script also requires, at import time, a working installation of whichever sibling project `--task` points at (`torch`, `pandas`, `rasterio`, etc.) — see `drought_forecast_binary/README.md` or `drought_forecast_regression/README.md` for that project's own requirements.
