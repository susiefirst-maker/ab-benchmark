"""DynaMine backbone-flexibility wrapper (stub for Phase 0).

Reference:
    Cilia, Pancsa, Tompa, Lenaerts, Vranken. "From protein sequence to
    dynamics and disorder with DynaMine." Nature Communications 4 (2013): 2741.
    DOI: 10.1038/ncomms3741

DynaMine predicts per-residue backbone dynamics (S² order parameter) from
sequence. There is no pip package; it is a web API at
https://bio2byte.be/dynamine/. For Phase 0 we provide this stub so the
`run_all` pipeline reports DynaMine uniformly with a clear reason. When
a batch-computed DynaMine file is placed at
$DYNAMINE_CACHE_TSV (per-antibody S² values), this wrapper becomes available.

When available, we report:
    dynamine_s2_mean             — mean S² over VH+VL
    dynamine_s2_min              — minimum S² (most flexible residue)
    dynamine_s2_cdr_h3_mean      — mean S² over the CDR-H3 region
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

from ab_benchmark.baselines.base import BaselineResult, ok, unavailable
from ab_benchmark.schema import AntibodyRecord
from ab_benchmark.seqprops import extract_cdrs

VERSION = "0.1.0-cache-only"


@lru_cache(maxsize=1)
def _check_availability() -> tuple[bool, str, pd.DataFrame | None]:
    cache = os.environ.get("DYNAMINE_CACHE_TSV")
    if not cache:
        return False, (
            "no DynaMine data (set DYNAMINE_CACHE_TSV to a batch-computed TSV, "
            "or fetch per-antibody predictions from https://bio2byte.be/dynamine/)"
        ), None
    if not os.path.exists(cache):
        return False, f"DYNAMINE_CACHE_TSV points to missing file: {cache}", None
    try:
        df = pd.read_csv(cache, sep="\t")
    except Exception as e:
        return False, f"could not read DynaMine cache: {e}", None
    required = {"ab_id", "residue_index", "s2"}
    if not required.issubset(df.columns):
        return False, f"DynaMine cache missing columns {required - set(df.columns)}", None
    return True, "ok", df


def compute_dynamine(record: AntibodyRecord) -> BaselineResult:
    available, reason, df = _check_availability()
    if not available or df is None:
        return unavailable("dynamine", VERSION, record, reason)

    sub = df[df["ab_id"] == record.ab_id]
    if sub.empty:
        return unavailable("dynamine", VERSION, record, f"no cached DynaMine rows for {record.ab_id!r}")

    metrics = {
        "dynamine_s2_mean": float(sub["s2"].mean()),
        "dynamine_s2_min": float(sub["s2"].min()),
    }

    # CDR-H3 specific mean, if we can identify the H3 residues.
    cdrs = extract_cdrs(record.vh, record.vl)
    if "h3" in cdrs and "region" in sub.columns:
        h3_rows = sub[sub["region"] == "h3"]
        if not h3_rows.empty:
            metrics["dynamine_s2_cdr_h3_mean"] = float(h3_rows["s2"].mean())

    return ok("dynamine", VERSION, record, metrics)
