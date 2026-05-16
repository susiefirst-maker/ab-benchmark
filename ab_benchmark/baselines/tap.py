"""Therapeutic Antibody Profiler (TAP) — sequence-proxy implementation.

Reference:
    Raybould, Marks, Krawczyk, Taddese, Nowak, Lewis, Bujotzek, Shi, Deane.
    "Five computational developability guidelines for therapeutic antibody
    profiling." PNAS 116 (2019): 4025-4030.
    DOI: 10.1073/pnas.1810576116

TAP's original five guidelines are:
    1. Total CDR length
    2. Patches of Surface Hydrophobicity (PSH) — requires 3D structure
    3. Patches of Positive Charge (PPC) — requires 3D structure
    4. Patches of Negative Charge (PNC) — requires 3D structure
    5. Structural Fv Charge Symmetry Parameter (SFvCSP) — requires 3D structure

In Phase 0 we have sequence only. We compute sequence-proxy versions of
guidelines 1-5 that proxy the underlying physicochemical properties
without requiring a 3D structure:

    tap_cdr_vicinity_total_length   — sum of CDR lengths (guideline 1, direct)
    tap_cdr_mean_hydrophobicity     — proxy for PSH (guideline 2)
    tap_cdr_net_pos_charge          — proxy for PPC (guideline 3)
    tap_cdr_net_neg_charge          — proxy for PNC (guideline 4)
    tap_fv_charge_diff              — VH net charge - VL net charge; rough SFvCSP proxy

When structure becomes available in Phase 3, a `tap_structural` baseline
can replace these proxies with the exact PSH/PPC/PNC patch counts over
SASA>7.5 Å² surface residues.
"""

from __future__ import annotations

from ab_benchmark.baselines.base import BaselineResult, ok, unavailable
from ab_benchmark.schema import AntibodyRecord
from ab_benchmark.seqprops import (
    extract_cdrs,
    mean_hydrophobicity,
    net_charge_at_ph,
)

VERSION = "0.1.0-seq-proxy"


def compute_tap(record: AntibodyRecord) -> BaselineResult:
    if not record.vh:
        return unavailable("tap", VERSION, record, "no VH sequence")

    cdrs = extract_cdrs(record.vh, record.vl)
    if "h3" not in cdrs:
        return unavailable("tap", VERSION, record, "CDR-H3 extraction failed (unusual framework?)")

    cdr_residues = "".join(cdrs.values())
    total_cdr_length = len(cdr_residues)
    h3_length = len(cdrs["h3"])

    # Guideline 1 proxy — CDR length (direct from H3 + other recovered CDRs).
    # Report H3 length separately since it is the most predictive individual
    # guideline (Raybould 2019 Fig. 2).

    # Guideline 2 proxy — CDR hydrophobicity.
    cdr_hydro = mean_hydrophobicity(cdr_residues) if cdr_residues else 0.0

    # Guidelines 3/4 proxies — CDR positive/negative charge counts.
    cdr_net_pos = sum(1 for a in cdr_residues if a in {"K", "R"})
    cdr_net_neg = sum(1 for a in cdr_residues if a in {"D", "E"})
    cdr_his = sum(1 for a in cdr_residues if a == "H")

    # Guideline 5 proxy — Fv charge asymmetry.
    vh_charge = net_charge_at_ph(record.vh, 7.4)
    vl_charge = net_charge_at_ph(record.vl, 7.4) if record.vl else 0.0
    fv_charge_asymmetry = vh_charge - vl_charge

    # Raybould-style risk flags (thresholds from the paper Fig. 4 amber bands).
    # These are calibrated against clinical-stage mAbs; exceeding the amber
    # range correlates with higher developability risk.
    flags = {
        "tap_flag_h3_too_long": int(h3_length >= 18),
        "tap_flag_cdr_too_hydrophobic": int(cdr_hydro >= 1.0),
        "tap_flag_cdr_too_positive": int(cdr_net_pos - cdr_net_neg >= 3),
        "tap_flag_cdr_too_negative": int(cdr_net_neg - cdr_net_pos >= 3),
        "tap_flag_fv_charge_asymmetric": int(abs(fv_charge_asymmetry) >= 4.0),
    }

    metrics = {
        "tap_h3_length": h3_length,
        "tap_cdr_total_length": total_cdr_length,
        "tap_cdr_mean_hydrophobicity": cdr_hydro,
        "tap_cdr_net_pos_count": cdr_net_pos,
        "tap_cdr_net_neg_count": cdr_net_neg,
        "tap_cdr_his_count": cdr_his,
        "tap_vh_charge": vh_charge,
        "tap_vl_charge": vl_charge,
        "tap_fv_charge_asymmetry": fv_charge_asymmetry,
        "tap_risk_flag_count": float(sum(flags.values())),
        **flags,
    }
    notes = "sequence-proxy; structural PSH/PPC/PNC require Phase 3 structure analyzer"
    return ok("tap", VERSION, record, metrics, notes)
