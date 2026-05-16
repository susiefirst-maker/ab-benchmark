"""Dataset loaders — one function per source.

Each loader returns a list[AntibodyRecord]. Missing source files raise
FileNotFoundError with clear instructions for where to obtain the data.
Un-implemented sources raise NotImplementedError with a pointer to the
download URL or DOI, so the harmonizer can skip them cleanly.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ab_benchmark.schema import (
    AntibodyRecord,
    DevelopabilityEndpoint,
    EndpointKind,
    SourceDataset,
)

# --- Jain 2017 (PNAS) -------------------------------------------------------

# Column name → (EndpointKind, unit)
# Only columns that map cleanly to EndpointKind are promoted; the rest go
# into `extras` so nothing is silently lost.
_JAIN_2017_ENDPOINT_MAP: dict[str, tuple[EndpointKind, str]] = {
    "HEK Titer (mg/L)": (EndpointKind.EXPRESSION_MGL, "mg/L"),
    "Fab Tm by DSF (°C)": (EndpointKind.TM_ONSET_C, "°C"),
    "HIC Retention Time (Min)a": (EndpointKind.HIC_RT, "min"),
    "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average": (
        EndpointKind.AC_SINS,
        "nm",
    ),
    "BVP ELISA": (EndpointKind.BVP_SCORE, "AU"),
    "Poly-Specificity Reagent (PSR) SMP Score (0-1)": (EndpointKind.PSR_SCORE, "score"),
}

# Columns kept as extras (preserved, not dropped).
_JAIN_2017_EXTRAS_COLS = [
    "SGAC-SINS AS100 ((NH4)2SO4 mM)",
    "SMAC Retention Time (Min)a",
    "Slope for Accelerated Stability",
    "CIC Retention Time (Min)",
    "CSI-BLI Delta Response (nm)",
    "ELISA",
]


def load_jain_2017(path: str | Path) -> list[AntibodyRecord]:
    """Load the Jain et al. 2017 PNAS developability dataset.

    Expected file: the cleaned combined CSV with columns {Name, VH, VL,
    plus 12 biophysical endpoint columns}. The file shipped with ProtePilot
    (data/Jain137_Cleaned_Training_Data.csv) matches this schema.

    DOI: 10.1073/pnas.1616408114
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Jain 2017 file not found at {path}. "
            "Obtain from PNAS supplementary (DOI 10.1073/pnas.1616408114) or "
            "use ProtePilot's data/Jain137_Cleaned_Training_Data.csv."
        )

    df = pd.read_csv(path)
    records: list[AntibodyRecord] = []

    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        if not name:
            continue
        vh = str(row.get("VH", "")).strip().upper()
        vl = str(row.get("VL", "")).strip().upper()

        endpoints: list[DevelopabilityEndpoint] = []
        for col, (kind, unit) in _JAIN_2017_ENDPOINT_MAP.items():
            if col not in df.columns:
                continue
            val = row[col]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            endpoints.append(
                DevelopabilityEndpoint(
                    kind=kind,
                    value=fv,
                    unit=unit,
                    assay_detail="Jain 2017 combined table",
                )
            )

        extras = {
            col: row[col]
            for col in _JAIN_2017_EXTRAS_COLS
            if col in df.columns and not _is_missing(row[col])
        }

        try:
            records.append(
                AntibodyRecord(
                    ab_id=name,
                    source=SourceDataset.JAIN_2017,
                    vh=vh if vh else "",
                    vl=vl if vl else "",
                    endpoints=endpoints,
                    extras=extras,
                )
            )
        except ValueError as e:
            # Non-standard residue in sequence — record without sequence.
            # (Alternative would be to skip; preserving the endpoints keeps n up.)
            records.append(
                AntibodyRecord(
                    ab_id=name,
                    source=SourceDataset.JAIN_2017,
                    endpoints=endpoints,
                    extras={**extras, "sequence_error": str(e)},
                )
            )

    return records


# --- SAbDab-Thera -----------------------------------------------------------


def load_sabdab_thera(path: str | Path) -> list[AntibodyRecord]:
    """Load Thera-SAbDab therapeutic antibody sequences + status annotations.

    Expected file: the standard TheraSAbDab_SeqStruc_OnlineDownload.csv
    from OPIG (https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab).
    Columns include Therapeutic, Format, HC, LC, Target, Highest_Clin_Trial,
    Est. Status, Year Proposed, Year Recommended, structure IDs.

    This source contributes *sequences and therapeutic metadata* but no
    biophysical endpoints. It is useful for augmenting the harmonized set
    with clinical-stage antibodies that have no public biophysical data,
    and for framework-diversity analyses.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"SAbDab-Thera file not found at {path}. "
            "Download from https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab "
            "(TheraSAbDab_SeqStruc_OnlineDownload.csv)."
        )

    df = pd.read_csv(path)
    records: list[AntibodyRecord] = []

    for _, row in df.iterrows():
        name = str(row.get("Therapeutic", "")).strip()
        if not name:
            continue

        # Only whole mAbs with sequences — bispecifics and empties skipped for now.
        hc = str(row.get("HC", "")).strip().upper()
        lc = str(row.get("LC", "")).strip().upper()
        if _is_missing(hc) or _is_missing(lc) or hc == "NA" or lc == "NA":
            continue

        # Trim to variable region heuristically — take first 120 AA (approx VH/VL).
        # Strictly: run ANARCI later to get proper variable-region boundaries.
        # For now we store the full chain and let a downstream numbering step
        # trim to variable region.
        extras = {
            "therapeutic_format": str(row.get("Format", "")),
            "highest_clin_trial": str(row.get("Highest_Clin_Trial (Feb '25)", "")),
            "est_status": str(row.get("Est. Status", "")),
            "target": str(row.get("Target", "")),
            "year_proposed": _safe_int(row.get("Year Proposed")),
            "year_recommended": _safe_int(row.get("Year Recommended")),
            "struct_100si": str(row.get("100% SI Structure", "")),
            "struct_99si": str(row.get("99% SI Structure", "")),
        }

        try:
            records.append(
                AntibodyRecord(
                    ab_id=name,
                    source=SourceDataset.SABDAB_THERA,
                    vh=hc,
                    vl=lc,
                    extras=extras,
                )
            )
        except ValueError:
            # Non-standard residue (e.g. ambiguous X) — skip rather than
            # silently create a record without sequence, because SAbDab's
            # only value here is the sequence itself.
            continue

    return records


# --- Shehata 2019 (stub) ----------------------------------------------------


def load_shehata_2019(path: str | Path) -> list[AntibodyRecord]:
    """Load the Shehata et al. 2019 mAbs developability dataset.

    Not currently implemented. Obtain from:
    Shehata et al. 2019 mAbs. Supplementary tables contain ~100 antibodies
    with solubility and viscosity measurements.
    """
    raise NotImplementedError(
        "Shehata 2019 loader not yet implemented. "
        "Supplementary source: mAbs (Taylor & Francis), Shehata et al. 2019."
    )


# --- Bailly 2020 (stub) -----------------------------------------------------


def load_bailly_2020(path: str | Path) -> list[AntibodyRecord]:
    """Load the Bailly et al. 2020 mAbs developability panel.

    Not currently implemented.
    """
    raise NotImplementedError(
        "Bailly 2020 loader not yet implemented. "
        "Supplementary source: mAbs (Taylor & Francis), Bailly et al. 2020."
    )


# --- PROPHET-Ab (deferred) --------------------------------------------------


def load_prophet_ab(path: str | Path) -> list[AntibodyRecord]:
    """PROPHET-Ab benchmark.

    PROPHET-Ab is treated as an already-running named baseline inside
    ProtePilot (benchmarks/prophet_ab_benchmark.py), not re-ingested as raw
    per-antibody rows here.
    """
    raise NotImplementedError(
        "PROPHET-Ab loader intentionally deferred. The PROPHET-Ab ridge baseline "
        "is wired up via ab_benchmark.baselines.prophet_ab (Phase 0 Step 5) which "
        "calls into ProtePilot's existing benchmarks/prophet_ab_benchmark.py."
    )


# --- helpers ----------------------------------------------------------------


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str):
        s = v.strip()
        return s == "" or s.lower() in {"na", "n/a", "none", "nan"}
    return False


def _safe_int(v) -> int | None:
    if _is_missing(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# --- registry ---------------------------------------------------------------

LOADER_REGISTRY: dict[SourceDataset, callable] = {
    SourceDataset.JAIN_2017: load_jain_2017,
    SourceDataset.SABDAB_THERA: load_sabdab_thera,
    SourceDataset.SHEHATA_2019: load_shehata_2019,
    SourceDataset.BAILLY_2020: load_bailly_2020,
    SourceDataset.PROPHET_AB: load_prophet_ab,
}
