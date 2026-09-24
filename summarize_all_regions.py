#!/usr/bin/env python3
"""One-off summary of the 5-region dataset_statistics_*.json and
baselines_*_*.json outputs, to source the cross-region tables/prose in
paper_datasets_short.tex."""
import json
from scipy.stats import norm

REGIONS = ["Sul", "Sudeste", "Nordeste", "Centro-Oeste", "Norte"]
REGION_LABEL = {"Sul": "South", "Sudeste": "Southeast", "Nordeste": "Northeast",
                "Centro-Oeste": "Center-West", "Norte": "North"}
KAPPAS = [-1.0, -1.5, -2.0]

print("=" * 100)
print("1) REGRESSION DATASET STATS: target_mean/std per region")
print("=" * 100)
reg_stats = json.load(open("dataset_statistics_regression.json"))
by_region = {r["region"]: r for r in reg_stats}
for reg in REGIONS:
    pq = by_region[reg]["per_pq"]["p3_q1"]
    print(f"{REGION_LABEL[reg]:14s} mean={pq['target_mean']:+.6f}  std={pq['target_std']:.6f}")

print()
print("=" * 100)
print("2) BINARY DATASET STATS: drought_prevalence, Phi(kappa), drought_ratio range")
print("=" * 100)
bin_stats = json.load(open("dataset_statistics_binary.json"))
by_region_b = {r["region"]: r for r in bin_stats}
for reg in REGIONS:
    pqk = by_region_b[reg]["per_pqk"]
    for k in KAPPAS:
        keys = [key for key in pqk if key.endswith(f"k{k}")]
        prev = pqk[keys[0]]["drought_prevalence"]
        ratios = [pqk[key]["drought_ratio"] for key in keys]
        phi = norm.cdf(k)
        print(f"{REGION_LABEL[reg]:14s} kappa={k:5.1f}  prevalence={prev*100:6.2f}%  "
              f"Phi={phi*100:6.2f}%  ratio=[{min(ratios)*100:5.1f}%, {max(ratios)*100:5.1f}%]")

print()
print("=" * 100)
print("3) REGRESSION BASELINES at p=12, q=1 (all regions)")
print("=" * 100)
for reg in REGIONS:
    rows = json.load(open(f"baselines_regression_{reg}.json"))
    row = next(r for r in rows if r["p"] == 12 and r["q"] == 1)
    res = row["results"]
    print(f"{REGION_LABEL[reg]:14s} N={row['n_samples']:>9,d}  "
          f"persist(WI={res['persistence']['wi']:.3f},RMSE={res['persistence']['rmse']:.3f},R2={res['persistence']['r2']:+.3f})  "
          f"linreg(WI={res['linear_regression']['wi']:.3f},RMSE={res['linear_regression']['rmse']:.3f},R2={res['linear_regression']['r2']:+.3f})  "
          f"RF(WI={res['random_forest']['wi']:.3f},RMSE={res['random_forest']['rmse']:.3f},R2={res['random_forest']['r2']:+.3f})")

print()
print("=" * 100)
print("4) REGRESSION BASELINES degradation: mean R2 across q in {3,6,12} at p=12")
print("=" * 100)
for reg in REGIONS:
    rows = json.load(open(f"baselines_regression_{reg}.json"))
    for method in ["persistence", "random_forest"]:
        vals = [r["results"][method]["r2"] for r in rows if r["p"] == 12 and r["q"] in (3, 6, 12)]
        print(f"{REGION_LABEL[reg]:14s} {method:16s} mean_R2(q=3,6,12) = {sum(vals)/len(vals):+.3f}   individual={[round(v,3) for v in vals]}")

print()
print("=" * 100)
print("5) CLASSIFICATION BASELINES at p=12, q=1, kappa=-2.0 (all regions)")
print("=" * 100)
for reg in REGIONS:
    rows = json.load(open(f"baselines_classification_{reg}.json"))
    row = next(r for r in rows if r["p"] == 12 and r["q"] == 1)
    res = row["results"]
    print(f"{REGION_LABEL[reg]:14s} N={row['n_samples']:>9,d}  "
          f"persist(F1={res['persistence']['f1']:.3f},AUC={res['persistence']['auc_roc']:.3f})  "
          f"logit(F1={res['logistic_regression']['f1']:.3f},AUC={res['logistic_regression']['auc_roc']:.3f})  "
          f"RF(F1={res['random_forest']['f1']:.3f},AUC={res['random_forest']['auc_roc']:.3f})")

print()
print("=" * 100)
print("6) CLASSIFICATION BASELINES degradation: mean AUC across q in {3,6,12} at p=12 (logistic)")
print("=" * 100)
for reg in REGIONS:
    rows = json.load(open(f"baselines_classification_{reg}.json"))
    vals = [r["results"]["logistic_regression"]["auc_roc"] for r in rows if r["p"] == 12 and r["q"] in (3, 6, 12)]
    print(f"{REGION_LABEL[reg]:14s} mean_AUC(q=3,6,12) logistic = {sum(vals)/len(vals):.3f}   individual={[round(v,3) for v in vals]}")

print()
print("=" * 100)
print("7) Sample counts (should be region-invariant) for p in {3,6,12}, q in {1,3,6,12}")
print("=" * 100)
rows = json.load(open("baselines_regression_Sul.json"))
for r in rows:
    print(f"p={r['p']:2d} q={r['q']:2d}  N={r['n_samples']:>9,d}")
