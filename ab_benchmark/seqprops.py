"""Amino-acid physicochemical scales, CDR extraction, charge/hydrophobicity utilities.

Scales used below are the canonical ones cited by the baseline papers:

- Kyte-Doolittle hydrophobicity (Kyte & Doolittle 1982, J. Mol. Biol.)
- Chou-Fasman α-helix and β-sheet propensities (Chou & Fasman 1974)
- Side-chain pKa values for net-charge estimation (Lehninger, Biochemistry)

CDR extraction in Phase 0 is a regex-based heuristic that works for
conventional therapeutic IgG variable regions. When it fails (unusual
framework, multi-domain, missing cysteines) we return an empty dict and
flag the antibody. ANARCI-based IMGT numbering is a Phase 1 upgrade.
"""

from __future__ import annotations

import re

# --- scales ---------------------------------------------------------------

KYTE_DOOLITTLE: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Chou-Fasman α-helix propensity P_alpha.
CHOU_FASMAN_ALPHA: dict[str, float] = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

# Chou-Fasman β-sheet propensity P_beta.
CHOU_FASMAN_BETA: dict[str, float] = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Side-chain pKa values for charged residues.
PKA_SIDECHAIN: dict[str, float] = {
    "D": 3.65,
    "E": 4.25,
    "H": 6.0,
    "K": 10.53,
    "R": 12.48,
    "C": 8.33,
    "Y": 10.07,
}
PKA_N_TERM = 8.0
PKA_C_TERM = 3.1


# --- net charge ------------------------------------------------------------


def net_charge_at_ph(seq: str, ph: float = 7.4) -> float:
    """Approximate net charge of a peptide at the given pH (Henderson-Hasselbalch)."""
    q = 0.0
    # N-terminus positive.
    q += _positive_fraction(PKA_N_TERM, ph)
    # C-terminus negative.
    q -= _negative_fraction(PKA_C_TERM, ph)
    for aa in seq:
        pka = PKA_SIDECHAIN.get(aa)
        if pka is None:
            continue
        if aa in {"K", "R", "H"}:
            q += _positive_fraction(pka, ph)
        else:  # D, E, C, Y
            q -= _negative_fraction(pka, ph)
    return q


def _positive_fraction(pka: float, ph: float) -> float:
    return 1.0 / (1.0 + 10 ** (ph - pka))


def _negative_fraction(pka: float, ph: float) -> float:
    return 1.0 / (1.0 + 10 ** (pka - ph))


# --- hydrophobicity --------------------------------------------------------


def mean_hydrophobicity(seq: str) -> float:
    """Mean Kyte-Doolittle hydrophobicity of the sequence. Unknown residues are skipped."""
    vals = [KYTE_DOOLITTLE[a] for a in seq if a in KYTE_DOOLITTLE]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


# --- CDR extraction (heuristic) --------------------------------------------

# CDR-H3: bounded by the second conserved Cys (~Kabat pos 92) and the
# WGxG / WGxR motif of framework 4. We use findall() and keep the
# rightmost match, since early Cys→WGxG matches can be spurious framework
# artifacts. Covers ~95% of therapeutic IgG VH sequences.
_HCDR3_RE = re.compile(r"C([A-Z]{3,30}?)W[GT][A-Z]G")

# CDR-L3: second conserved Cys → F/W G.G of FR-L4. Rightmost wins.
_LCDR3_RE = re.compile(r"C([A-Z]{3,30}?)[FW]G[A-Z]G")

# CDR-H1 Kabat pos 26-35; anchored on conserved Cys22 and WVR/WIR motif ~36.
_HCDR1_RE = re.compile(r"C[A-Z]{4}([A-Z]{5,10})W[IV]R")

# CDR-H2 Kabat pos 50-65; anchored on WVR motif preceding and ~14 chars gap.
_HCDR2_RE = re.compile(r"W[IV]R[A-Z]{14}([A-Z]{15,20})")


def extract_cdrs(vh: str, vl: str = "") -> dict[str, str]:
    """Extract CDRs from VH/VL using regex heuristics.

    Returns a dict with keys h1, h2, h3, l3 when each is found. H3/L3 are
    the most reliable (flanked by conserved residues); H1/H2 use Kabat
    position anchors and fail more often on unusual frameworks.

    For H3/L3 we scan the whole chain and keep the *rightmost* match,
    because early Cys→WGxG pairs can be framework artifacts (e.g. a spurious
    WGxG in framework-3).
    """
    out: dict[str, str] = {}

    h3_matches = list(_HCDR3_RE.finditer(vh))
    if h3_matches:
        out["h3"] = h3_matches[-1].group(1)

    m = _HCDR1_RE.search(vh)
    if m:
        out["h1"] = m.group(1)

    m = _HCDR2_RE.search(vh)
    if m:
        out["h2"] = m.group(1)

    if vl:
        l3_matches = list(_LCDR3_RE.finditer(vl))
        if l3_matches:
            out["l3"] = l3_matches[-1].group(1)

    return out
