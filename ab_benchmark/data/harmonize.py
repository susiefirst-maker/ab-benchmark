"""Harmonize AntibodyRecord lists into a single long-format DataFrame.

Canonical schema (long format, one row per (antibody, source, endpoint)):

    ab_id                  source-native identifier
    ab_id_canonical        normalized for cross-source matching
    source                 SourceDataset value
    vh                     heavy-chain variable region AA (may be empty)
    vl                     light-chain variable region AA
    v_gene_heavy           lazily filled (empty in Phase 0)
    v_gene_light
    cdr_h3
    cdr_h3_length
    endpoint_kind          EndpointKind value (e.g. 'tm_onset_c')
    value                  float
    unit                   string
    assay_detail           free-text

SAbDab-Thera rows, which have no endpoints, contribute one "metadata-only"
row per antibody with endpoint_kind=None, value=NaN. This keeps provenance
explicit — it is always obvious which rows carry measurements and which
carry only sequence/status.

The wide format (`to_wide`) pivots endpoint_kind → columns, producing one
row per (ab_id_canonical, source) for ML training.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from ab_benchmark.schema import (
    AntibodyRecord,
    UNITS,
)


# --- canonical ID -----------------------------------------------------------

_INN_SUFFIX_RE = re.compile(
    r"(mab|umab|omab|ximab|zumab)\b", re.IGNORECASE
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


def canonicalize_ab_id(raw: str) -> str:
    """Normalize an antibody name for cross-source matching.

    Strategy: lowercase, strip whitespace, remove non-alphanumeric characters.
    INN suffix variation (-mab, -umab, -ximab, -zumab, -omab) is preserved
    because it encodes humanization status — stripping it would fuse
    distinct molecules. Examples:

        'Trastuzumab' → 'trastuzumab'
        'Adalimumab'  → 'adalimumab'
        'mAb 12-34'   → 'mab1234'
        'Ab001'       → 'ab001'
    """
    if not raw:
        return ""
    s = raw.strip().lower()
    s = _NON_ALNUM_RE.sub("", s)
    return s


# --- harmonization ----------------------------------------------------------

LONG_COLUMNS = [
    "ab_id",
    "ab_id_canonical",
    "source",
    "vh",
    "vl",
    "v_gene_heavy",
    "v_gene_light",
    "cdr_h3",
    "cdr_h3_length",
    "endpoint_kind",
    "value",
    "unit",
    "assay_detail",
]


def records_to_long_df(records: Iterable[AntibodyRecord]) -> pd.DataFrame:
    """Convert AntibodyRecord list to a long-format DataFrame.

    Records with no endpoints (e.g. SAbDab-Thera sequence-only) yield a
    single metadata row with endpoint_kind=None, value=NaN. This makes
    provenance explicit and keeps every antibody present in the harmonized
    set even when measurements are absent.

    Duplicates: if the same (ab_id_canonical, source, endpoint_kind) appears
    twice, the FIRST occurrence wins and a warning is recorded as a second
    row with `assay_detail='duplicate skipped'`. We do not silently merge
    — measurements from different assay runs should be preserved.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str, str | None]] = set()

    for r in records:
        canon = canonicalize_ab_id(r.ab_id)
        base = {
            "ab_id": r.ab_id,
            "ab_id_canonical": canon,
            "source": r.source.value,
            "vh": r.vh,
            "vl": r.vl,
            "v_gene_heavy": r.v_gene_heavy,
            "v_gene_light": r.v_gene_light,
            "cdr_h3": r.cdr_h3,
            "cdr_h3_length": r.cdr_h3_length,
        }

        if not r.endpoints:
            rows.append({
                **base,
                "endpoint_kind": None,
                "value": np.nan,
                "unit": "",
                "assay_detail": "metadata_only",
            })
            continue

        for e in r.endpoints:
            key = (canon, r.source.value, e.kind.value)
            if key in seen:
                rows.append({
                    **base,
                    "endpoint_kind": e.kind.value,
                    "value": e.value,
                    "unit": e.unit,
                    "assay_detail": "duplicate skipped",
                })
                continue
            seen.add(key)

            expected_unit = UNITS.get(e.kind, "")
            assay = e.assay_detail
            if expected_unit and e.unit != expected_unit:
                assay = f"{assay} [UNIT MISMATCH expected={expected_unit} got={e.unit}]".strip()

            rows.append({
                **base,
                "endpoint_kind": e.kind.value,
                "value": e.value,
                "unit": e.unit,
                "assay_detail": assay,
            })

    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    return df


def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long → wide: one row per (ab_id_canonical, source), one column per endpoint.

    Non-endpoint columns (vh, vl, v_gene_*, cdr_*) are deduplicated per
    antibody-source; if they differ across rows for the same antibody (they
    shouldn't), the first non-empty value wins.

    Metadata-only rows (endpoint_kind=None) are preserved as a row with all
    endpoint columns NaN.
    """
    if long_df.empty:
        return long_df.copy()

    # Sequence/metadata columns — one value per (ab_id_canonical, source).
    meta_cols = ["ab_id", "vh", "vl", "v_gene_heavy", "v_gene_light", "cdr_h3", "cdr_h3_length"]
    meta = (
        long_df.groupby(["ab_id_canonical", "source"], as_index=False)[meta_cols]
        .agg(lambda s: _first_non_empty(s))
    )

    # Endpoint pivot.
    endpoint_rows = long_df[long_df["endpoint_kind"].notna()].copy()
    if endpoint_rows.empty:
        pivot = pd.DataFrame()
    else:
        # Drop duplicates-marked rows so they don't pollute the pivot.
        endpoint_rows = endpoint_rows[endpoint_rows["assay_detail"] != "duplicate skipped"]
        pivot = endpoint_rows.pivot_table(
            index=["ab_id_canonical", "source"],
            columns="endpoint_kind",
            values="value",
            aggfunc="first",
        ).reset_index()
        pivot.columns.name = None

    if pivot.empty:
        return meta

    wide = meta.merge(pivot, on=["ab_id_canonical", "source"], how="left")
    return wide


def _first_non_empty(series: pd.Series):
    for v in series:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return series.iloc[0] if len(series) else None


# --- summary ---------------------------------------------------------------


def harmonization_summary(long_df: pd.DataFrame) -> dict:
    """Summarize the harmonized long DataFrame.

    Returns per-source record counts, per-endpoint measurement counts, and
    cross-source overlap by ab_id_canonical.
    """
    out: dict = {}

    # Records per source (one entry per row — long format, so multiple per AB).
    out["rows_by_source"] = long_df["source"].value_counts().to_dict()

    # Unique antibodies per source (canonical IDs).
    out["unique_antibodies_by_source"] = (
        long_df.groupby("source")["ab_id_canonical"].nunique().to_dict()
    )

    # Measurements per endpoint kind.
    endpoint_rows = long_df[long_df["endpoint_kind"].notna()]
    endpoint_rows = endpoint_rows[endpoint_rows["assay_detail"] != "duplicate skipped"]
    out["measurements_by_endpoint"] = (
        endpoint_rows["endpoint_kind"].value_counts().to_dict()
    )

    # Total unique antibodies + cross-source overlap.
    total_unique = long_df["ab_id_canonical"].nunique()
    out["total_unique_antibodies"] = total_unique

    overlap = (
        long_df.groupby("ab_id_canonical")["source"].nunique().reset_index(name="n_sources")
    )
    out["antibodies_in_multiple_sources"] = int((overlap["n_sources"] > 1).sum())

    return out
