"""Canonical antibody record schema for the harmonized developability benchmark.

Every dataset loader produces a list of AntibodyRecord instances. Harmonization
merges them into a single table, resolving duplicates by (source, ab_id).

Design notes:
- VH/VL sequences are stored as plain strings; numbering (IMGT/Kabat/Chothia)
  is computed lazily in ab_benchmark.data.numbering.
- `endpoints` is a list of DevelopabilityEndpoint so that different sources
  can contribute different assays for the same antibody; missing values are
  represented as absence from the list rather than NaN.
- `extras` is an untyped bag for source-specific metadata that does not fit
  the schema (e.g. PROPHET-Ab fold-ID, SAbDab therapeutic-target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceDataset(str, Enum):
    """Source dataset identifier. Used to preserve provenance through harmonization."""

    JAIN_2017 = "jain_2017"
    SHEHATA_2019 = "shehata_2019"
    BAILLY_2020 = "bailly_2020"
    PROPHET_AB = "prophet_ab"
    SABDAB_THERA = "sabdab_thera"


class EndpointKind(str, Enum):
    """Canonical endpoint names. Units are defined per kind (see UNITS below)."""

    HMW_PCT = "hmw_pct"              # SEC %HMW
    TM_ONSET_C = "tm_onset_c"        # DSC Tm onset, °C
    KD_DLS = "kd_dls"                # DLS interaction parameter, mL/g
    HIC_RT = "hic_rt"                # HIC retention time, minutes
    AC_SINS = "ac_sins"              # AC-SINS score, nm shift
    BVP_SCORE = "bvp_score"          # baculovirus particle polyreactivity, AU
    PSR_SCORE = "psr_score"          # Poly-Specificity Reagent SMP score, 0-1
    SOLUBILITY = "solubility"        # generic solubility, mg/mL
    VISCOSITY_CP = "viscosity_cp"    # cP
    EXPRESSION_MGL = "expression_mgl"  # mg/L titer
    CLEARANCE_ML_DAY_KG = "clearance"  # in vivo clearance, mL/day/kg


UNITS: dict[EndpointKind, str] = {
    EndpointKind.HMW_PCT: "%",
    EndpointKind.TM_ONSET_C: "°C",
    EndpointKind.KD_DLS: "mL/g",
    EndpointKind.HIC_RT: "min",
    EndpointKind.AC_SINS: "nm",
    EndpointKind.BVP_SCORE: "AU",
    EndpointKind.PSR_SCORE: "score",
    EndpointKind.SOLUBILITY: "mg/mL",
    EndpointKind.VISCOSITY_CP: "cP",
    EndpointKind.EXPRESSION_MGL: "mg/L",
    EndpointKind.CLEARANCE_ML_DAY_KG: "mL/day/kg",
}


@dataclass(frozen=True)
class DevelopabilityEndpoint:
    """One measured biophysical property."""

    kind: EndpointKind
    value: float
    unit: str                # must match UNITS[kind] after harmonization
    assay_detail: str = ""   # free-text: assay protocol, buffer, temperature, etc.


@dataclass
class AntibodyRecord:
    """Canonical harmonized record for one antibody from one source dataset.

    If the same antibody appears in multiple sources, it yields multiple
    records; the harmonizer de-duplicates on (ab_id_canonical) and merges
    endpoint lists.
    """

    # --- identity ---
    ab_id: str                           # source-native ID
    source: SourceDataset
    ab_id_canonical: str = ""            # filled by harmonizer (e.g. INN or lowercased)

    # --- sequence ---
    vh: str = ""                         # heavy-chain variable region AA sequence
    vl: str = ""                         # light-chain variable region AA sequence

    # --- annotations (filled lazily) ---
    v_gene_heavy: str = ""               # e.g. "IGHV3-23"
    v_gene_light: str = ""
    cdr_h3: str = ""                     # IMGT-defined CDR-H3
    cdr_h3_length: int = 0

    # --- measurements ---
    endpoints: list[DevelopabilityEndpoint] = field(default_factory=list)

    # --- source-specific bag ---
    extras: dict[str, Any] = field(default_factory=dict)

    # --- validators ---

    def __post_init__(self) -> None:
        if self.vh and not _is_aa(self.vh):
            raise ValueError(f"VH contains non-standard amino acids for {self.ab_id!r}")
        if self.vl and not _is_aa(self.vl):
            raise ValueError(f"VL contains non-standard amino acids for {self.ab_id!r}")
        if self.cdr_h3_length and self.cdr_h3 and len(self.cdr_h3) != self.cdr_h3_length:
            raise ValueError(
                f"cdr_h3_length={self.cdr_h3_length} != len(cdr_h3)={len(self.cdr_h3)} "
                f"for {self.ab_id!r}"
            )

    def has_endpoint(self, kind: EndpointKind) -> bool:
        return any(e.kind is kind for e in self.endpoints)

    def get_endpoint(self, kind: EndpointKind) -> DevelopabilityEndpoint | None:
        for e in self.endpoints:
            if e.kind is kind:
                return e
        return None


_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _is_aa(seq: str) -> bool:
    """True if seq contains only standard 20 amino acids (uppercase)."""
    return bool(seq) and set(seq).issubset(_STANDARD_AA)
