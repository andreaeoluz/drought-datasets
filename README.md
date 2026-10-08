# Two Multivariate Spatiotemporal Climate Datasets for Drought Forecasting in Brazil

Code and results for the paper **"Two Multivariate Spatiotemporal Climate Datasets for Drought Forecasting in Brazil: Construction and Validation"**.

The paper describes the two machine-learning-ready datasets used by the transfer-learning frameworks: a **binary drought-event** dataset (SPI-3 ≤ κ, κ ∈ {−1.0, −1.5, −2.0}) and a **continuous SPI-3** dataset. Both come from the same monthly TerraClimate record (1980–2024) for Brazil's five macro-regions and share the validity mask, input representation and chronological partition; they differ only in the target. This repository characterizes both datasets and validates them with simple baselines.

---

## Related repositories

| Repository | Paper |
|---|---|
| [drought_forecast_binary](https://github.com/andreaeoluz/drought_forecast_binary) | Binary rare-drought forecasting with TL |
| [spi-forecast-continuous](https://github.com/andreaeoluz/spi-forecast-continuous) | Continuous SPI forecasting with TL |
| [drought-datasets](https://github.com/andreaeoluz/drought-datasets) (this one) | The two datasets used by the TL frameworks |
| [spi-forecast-ml-dl](https://github.com/andreaeoluz/spi-forecast-ml-dl) | ML vs. DL comparison for SPI forecasting |

---

## Method in brief

- **Data.** Monthly TerraClimate rasters (1980–2024), seven climate variables plus SPI-3, downsampled 3×3 (5×5 in the North) and masked to pixels with ≥ 70% valid training observations.
- **Split.** Training 1980–2019, validation 2020–2022, test 2023–2024; the SPI Gamma fit uses the training period only.
- **Samples.** One sample per `p`-month input window and target instant `q`, with p ∈ {3, 6, 9, 12} and q ∈ {1, 3, 6, 9, 12}.
- **Statistics.** SPI distribution and drought prevalence per region, split and (p, q, κ).
- **Baselines.** Per-pixel persistence, linear/logistic regression and random forest (regression: WI, R², RMSE, MAE; classification: AUC-ROC, F1, precision, recall).

The datasets are built by the two framework repositories; the scripts here import one of them (`--task`) to load its preprocessed climate cubes and SPI cache.

---

## Repository structure

```
├── compute_dataset_statistics.py   # SPI distribution and drought prevalence per (p, q[, κ]) -> dataset_statistics_*.json
├── run_baselines.py                # Persistence / linear or logistic / random-forest baselines -> baselines_*.json
├── summarize_all_regions.py        # Cross-region numbers and tables from the JSON files
├── plot_drought_prevalence.py      # Drought prevalence figure -> images/drought_prevalence.*
└── plot_south_baselines.py         # South baseline skill vs. q -> images/south_baselines_vs_q.*
```

---

## Installation and data

The scripts expect both framework repositories next to this one, with these folder names:

```bash
git clone https://github.com/andreaeoluz/drought-datasets.git
git clone https://github.com/andreaeoluz/drought_forecast_binary.git drought_forecast_binary
git clone https://github.com/andreaeoluz/spi-forecast-continuous.git drought_forecast_regression
pip install -r drought_forecast_regression/requirements.txt
```

Each framework must have run its `precompute-spi` step for the regions of interest (`outputs/<region>/spi_cache/spi_scale_3.pkl`); see their READMEs for the raw data location. `--base-dir` points the scripts to a different `outputs/` directory.

---

## Reproducing the paper

```bash
cd drought-datasets

# Dataset statistics (all regions, full (p, q) grid)
python compute_dataset_statistics.py --task regression --all-regions
python compute_dataset_statistics.py --task binary --all-regions

# Baselines (full (p, q) grid by default), one run per region
python run_baselines.py --task regression --region Sul
python run_baselines.py --task classification --region Sul

# Cross-region numbers and figures
python summarize_all_regions.py
python plot_drought_prevalence.py
python plot_south_baselines.py
```

---

## Paper figures and tables

| Paper element | Produced by |
|---|---|
| Dataset statistics (SPI distribution, sample counts) | `compute_dataset_statistics.py` → `summarize_all_regions.py` |
| Drought prevalence figure (`drought_prevalence`) | `plot_drought_prevalence.py` |
| Baseline tables (appendix) and cross-region baseline summary | `run_baselines.py` → `summarize_all_regions.py` |
| South baselines vs. q (`south_baselines_vs_q`) | `plot_south_baselines.py` |
| Downsampling factor (spatial preprocessing) | `analyze_downsampling_events.py` in [drought_forecast_binary](https://github.com/andreaeoluz/drought_forecast_binary) |
| Study-area map | `generate_study_area_map.py` in [spi-forecast-continuous](https://github.com/andreaeoluz/spi-forecast-continuous) |

---

## Included results

| File | Content |
|---|---|
| `dataset_statistics_regression.json` | Continuous dataset statistics, all regions and (p, q) |
| `dataset_statistics_binary.json` | Binary dataset statistics, all regions, (p, q) and κ |
| `baselines_regression_<Region>.json` | Regression baselines per region and (p, q) |
| `baselines_classification_<Region>.json` | Classification baselines per region, (p, q) and κ |

---

## License

MIT — see [LICENSE](LICENSE).
