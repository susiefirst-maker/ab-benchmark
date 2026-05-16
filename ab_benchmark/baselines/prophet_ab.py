"""PROPHET-Ab ridge baseline — wrapper around the ProtePilot benchmark.

Reference:
    Johnson et al. 2024 — PROPHET-Ab: predictive developability models
    from ESM-2 PLM embeddings trained on clinical antibody biophysical
    assays. (Referenced in ProtePilot's benchmarks/prophet_ab_benchmark.py.)

In the ab-benchmark Phase 0, this wrapper does NOT retrain the PROPHET-Ab
ridge; it provides the canonical interface that Phase 1 will populate.
The actual model training consumes ESM-2 t12 VH+VL (960-dim) embeddings
against harmonized endpoints (e.g. tm_onset_c, hic_rt) using ridge
regression with grouped-repeated CV.

Availability in Phase 0:
    - Returns available=False with reason "requires ESM-2 embeddings (Phase 1)"
    - A cached predictions TSV at $PROPHET_AB_CACHE_TSV can override this
      (columns: ab_id, endpoint, prediction), in which case the wrapper
      reads the prediction directly.

When available, we report per endpoint trained:
    prophet_ab_pred_tm_onset_c
    prophet_ab_pred_hic_rt
    prophet_ab_pred_ac_sins
    prophet_ab_pred_bvp_score
    prophet_ab_pred_psr_score
    prophet_ab_pred_expression_mgl
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

from ab_benchmark.baselines.base import BaselineResult, ok, unavailable
from ab_benchmark.schema import AntibodyRecord, EndpointKind

VERSION = "0.1.0-cache-only"


@lru_cache(maxsize=1)
def _check_cache() -> tuple[bool, str, pd.DataFrame | None]:
    cache = os.environ.get("PROPHET_AB_CACHE_TSV")
    if not cache:
        return False, (
            "PROPHET-Ab ridge not yet trained in Phase 0 — needs ESM-2 embeddings "
            "from Phase 1. Set PROPHET_AB_CACHE_TSV once predictions are cached."
        ), None
    if not os.path.exists(cache):
        return False, f"PROPHET_AB_CACHE_TSV missing: {cache}", None
    try:
        df = pd.read_csv(cache, sep="\t")
    except Exception as e:
        return False, f"could not read PROPHET-Ab cache: {e}", None
    required = {"ab_id", "endpoint", "prediction"}
    if not required.issubset(df.columns):
        return False, f"PROPHET-Ab cache missing columns {required - set(df.columns)}", None
    return True, "ok", df


def compute_prophet_ab(record: AntibodyRecord) -> BaselineResult:
    available, reason, df = _check_cache()
    if not available or df is None:
        return unavailable("prophet_ab", VERSION, record, reason)

    sub = df[df["ab_id"] == record.ab_id]
    if sub.empty:
        return unavailable("prophet_ab", VERSION, record, f"no cached predictions for {record.ab_id!r}")

    metrics: dict[str, float] = {}
    for _, row in sub.iterrows():
        endpoint = str(row["endpoint"])
        # Only accept endpoints we know.
        try:
            _ = EndpointKind(endpoint)
        except ValueError:
            continue
        metrics[f"prophet_ab_pred_{endpoint}"] = float(row["prediction"])

    if not metrics:
        return unavailable(
            "prophet_ab", VERSION, record,
            "cache had rows but no recognized endpoints",
        )
    return ok("prophet_ab", VERSION, record, metrics)
