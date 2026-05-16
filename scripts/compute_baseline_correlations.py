"""Compute Spearman rho with 95% CI between baseline metrics and endpoints.

Usage:
    python scripts/compute_baseline_correlations.py \\
      --baselines reports/baseline_results.csv \\
      --harmonized data/processed/harmonized_antibody_dev.parquet \\
      --out reports/baseline_correlations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ab_benchmark.eval.bootstrap import bootstrap_spearman, fisher_z_spearman


ENDPOINTS = ["tm_onset_c", "hic_rt", "ac_sins", "bvp_score", "psr_score", "expression_mgl"]

# Baseline metrics we expect to correlate with Jain endpoints.
# (empty list = use all metrics from that baseline)
BASELINE_METRIC_FILTER: dict[str, list[str]] = {
    "tap": [
        "tap_cdr_mean_hydrophobicity",
        "tap_cdr_net_pos_count",
        "tap_cdr_net_neg_count",
        "tap_fv_charge_asymmetry",
        "tap_risk_flag_count",
        "tap_h3_length",
    ],
    "developability_index": [
        "di_seq_proxy",
        "di_sfvcsp_seq",
        "di_mean_hydrophobicity",
    ],
    "camsol_intrinsic": [
        "camsol_intrinsic_mean",
        "camsol_intrinsic_min",
        "camsol_intrinsic_frac_negative",
    ],
}


def build_baseline_wide(baselines_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot baseline long-format to one column per (baseline, metric)."""
    rows = baselines_long[baselines_long["available"] & (baselines_long["metric"] != "")]
    pivot = rows.pivot_table(
        index=["ab_id_canonical", "source"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def build_endpoint_wide(harmonized_long: pd.DataFrame) -> pd.DataFrame:
    """One row per (ab_id_canonical, source); one column per endpoint."""
    ep_rows = harmonized_long[harmonized_long["endpoint_kind"].notna()]
    ep_rows = ep_rows[ep_rows["assay_detail"] != "duplicate skipped"]
    pivot = ep_rows.pivot_table(
        index=["ab_id_canonical", "source"],
        columns="endpoint_kind",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def compute_correlations(baselines_wide: pd.DataFrame, endpoints_wide: pd.DataFrame) -> pd.DataFrame:
    merged = endpoints_wide.merge(baselines_wide, on=["ab_id_canonical", "source"], how="inner")

    rows = []
    for baseline_name, metrics in BASELINE_METRIC_FILTER.items():
        # Only include metrics that actually exist in the baseline pivot.
        metrics = [m for m in metrics if m in merged.columns]
        for metric in metrics:
            for ep in ENDPOINTS:
                if ep not in merged.columns:
                    continue
                x = merged[metric].to_numpy()
                y = merged[ep].to_numpy()
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 10:
                    continue
                ci_boot = bootstrap_spearman(x[mask], y[mask], n_boot=2000, random_state=0)
                ci_fz = fisher_z_spearman(x[mask], y[mask])
                rows.append({
                    "baseline": baseline_name,
                    "metric": metric,
                    "endpoint": ep,
                    "n": int(mask.sum()),
                    "rho": ci_boot.point,
                    "ci_low_bootstrap": ci_boot.low,
                    "ci_high_bootstrap": ci_boot.high,
                    "ci_low_fisher_z": ci_fz.low,
                    "ci_high_fisher_z": ci_fz.high,
                })
    df = pd.DataFrame(rows)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", type=Path, default=Path("reports/baseline_results.csv"))
    ap.add_argument("--harmonized", type=Path, default=Path("data/processed/harmonized_antibody_dev.parquet"))
    ap.add_argument("--out", type=Path, default=Path("reports/baseline_correlations.csv"))
    args = ap.parse_args()

    baselines_long = pd.read_csv(args.baselines)
    harmonized_long = pd.read_parquet(args.harmonized)

    baselines_wide = build_baseline_wide(baselines_long)
    endpoints_wide = build_endpoint_wide(harmonized_long)

    df = compute_correlations(baselines_wide, endpoints_wide)
    df.sort_values(["endpoint", "baseline", "metric"], inplace=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
