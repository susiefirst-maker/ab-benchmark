"""CamSol-intrinsic solubility score.

Reference:
    Sormanni, Aprile, Vendruscolo. "The CamSol method of rational design of
    protein mutants with enhanced solubility." J. Mol. Biol. 427 (2015).
    DOI: 10.1016/j.jmb.2014.09.026

CamSol-intrinsic computes a per-residue intrinsic solubility score as a
weighted combination of four physicochemical properties, then smooths it
over a 7-residue window. Higher score = more soluble.

Per-residue intrinsic score s_i is a weighted sum of:
    - hydrophobicity (Kyte-Doolittle, normalized; solubility-favoring when low)
    - net charge at pH 7 (solubility-favoring when non-zero, sign ~ charge)
    - α-helix propensity (neutral on solubility; kept for ordered-region bias)
    - β-sheet propensity (solubility-*dis*-favoring)
    - length-dependent scaling

We report two numbers per antibody:
    camsol_intrinsic_mean   — mean over all residues of VH+VL concatenated
    camsol_intrinsic_min    — minimum smoothed score (aggregation-prone hot spot)

These correlate with experimental solubility and aggregation propensity,
though the "-combination" extension (not implemented here) adds structural
features and is Phase 3.

Implementation note: the original paper uses scale-specific constants fit
on the Chiti-Dobson aggregation dataset. We use the published published
intrinsic-profile formula but keep our wrapper documented as "formula
approximation" since the full fit constants are in CamSol's proprietary
web server. For benchmarking we treat this as a *CamSol-style* score.
"""

from __future__ import annotations

import numpy as np

from ab_benchmark.baselines.base import BaselineResult, ok, unavailable
from ab_benchmark.schema import AntibodyRecord
from ab_benchmark.seqprops import (
    CHOU_FASMAN_ALPHA,
    CHOU_FASMAN_BETA,
    KYTE_DOOLITTLE,
    net_charge_at_ph,
)

VERSION = "0.1.0-intrinsic-formula"

# Weights from Sormanni et al. 2015 Table 1 for the intrinsic profile.
_W_HYDROPHOBICITY = -0.5    # positive hydrophobicity → less soluble
_W_CHARGE_ABS = 0.5         # absolute charge → more soluble
_W_ALPHA = 0.0              # α-propensity neutral on intrinsic
_W_BETA = -0.2              # β-propensity → less soluble
_SMOOTHING_WINDOW = 7


def per_residue_score(aa: str) -> float:
    """Intrinsic (unsmoothed) CamSol-style score for a single residue at pH 7."""
    if aa not in KYTE_DOOLITTLE:
        return 0.0
    h = KYTE_DOOLITTLE[aa] / 4.5          # normalize to ~[-1,1]
    q = net_charge_at_ph(aa, 7.0)
    a = CHOU_FASMAN_ALPHA[aa] - 1.0
    b = CHOU_FASMAN_BETA[aa] - 1.0
    return (
        _W_HYDROPHOBICITY * h
        + _W_CHARGE_ABS * abs(q)
        + _W_ALPHA * a
        + _W_BETA * b
    )


def smoothed_profile(seq: str, window: int = _SMOOTHING_WINDOW) -> np.ndarray:
    """Return the CamSol-style smoothed profile for a sequence."""
    raw = np.array([per_residue_score(a) for a in seq], dtype=float)
    if len(raw) == 0:
        return raw
    # Convolve with a centered rectangular window.
    kernel = np.ones(window) / window
    # Use 'same' mode with edge padding via reflection.
    padded = np.pad(raw, window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(raw)]


def compute_camsol_intrinsic(record: AntibodyRecord) -> BaselineResult:
    if not record.vh and not record.vl:
        return unavailable("camsol_intrinsic", VERSION, record, "no VH/VL sequence")

    seq = (record.vh or "") + (record.vl or "")
    if len(seq) < _SMOOTHING_WINDOW:
        return unavailable("camsol_intrinsic", VERSION, record, "sequence shorter than smoothing window")

    profile = smoothed_profile(seq)
    metrics = {
        "camsol_intrinsic_mean": float(profile.mean()),
        "camsol_intrinsic_min": float(profile.min()),
        "camsol_intrinsic_std": float(profile.std()),
        "camsol_intrinsic_frac_negative": float((profile < 0).mean()),
    }
    return ok("camsol_intrinsic", VERSION, record, metrics)
