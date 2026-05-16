"""BioPhi / OASis humanness wrapper.

Reference:
    Prihoda, Maamary, Waight, Juan, Fayadat-Dilman, Svozil, Bitton.
    "BioPhi: A platform for antibody design, humanization, and humanness
    evaluation based on natural antibody repertoires and deep learning."
    mAbs 14 (2022): 2020203. DOI: 10.1080/19420862.2021.2020203

BioPhi's OASis humanness score reports the fraction of 9-mer peptides in
a variable region that appear with sufficient prevalence in the Observed
Antibody Space (OAS) human repertoire. It requires:

    1. `biophi` Python package installed (pip install biophi)
    2. The OASis 9-mer peptide database (~1-5 GB download)
    3. The path to that DB passed via BIOPHI_OASIS_DB environment variable

If any of those are missing we return available=False with a reason.
Availability is cached at import time so repeated calls are cheap.

When available, we report:
    biophi_oasis_identity        — fraction of 9-mers passing the prevalence cutoff
    biophi_oasis_germline        — closest human germline match (V-gene)
    biophi_oasis_prevalence_mean — mean prevalence across 9-mers
"""

from __future__ import annotations

import os
from functools import lru_cache

from ab_benchmark.baselines.base import BaselineResult, ok, unavailable
from ab_benchmark.schema import AntibodyRecord

VERSION = "0.1.0-detection-only"


@lru_cache(maxsize=1)
def _check_availability() -> tuple[bool, str]:
    """Returns (available, reason)."""
    try:
        import biophi  # noqa: F401
    except ImportError:
        return False, "biophi package not installed (pip install biophi)"
    db_path = os.environ.get("BIOPHI_OASIS_DB")
    if not db_path:
        return False, "BIOPHI_OASIS_DB env var not set (path to OASis 9-mer DB)"
    if not os.path.exists(db_path):
        return False, f"BIOPHI_OASIS_DB points to missing path: {db_path}"
    return True, "ok"


def compute_biophi_oasis(record: AntibodyRecord) -> BaselineResult:
    available, reason = _check_availability()
    if not available:
        return unavailable("biophi_oasis", VERSION, record, reason)

    if not record.vh or not record.vl:
        return unavailable("biophi_oasis", VERSION, record, "need both VH and VL")

    # Lazy import — only reached when biophi is installed.
    from biophi.humanization.methods.humanness import get_oasis_humanness_score  # type: ignore

    try:
        score = get_oasis_humanness_score(
            record.vh, record.vl, db_path=os.environ["BIOPHI_OASIS_DB"]
        )
    except Exception as e:
        return unavailable("biophi_oasis", VERSION, record, f"biophi runtime error: {e}")

    metrics = {
        "biophi_oasis_identity": float(score.get("identity", float("nan"))),
        "biophi_oasis_prevalence_mean": float(score.get("prevalence_mean", float("nan"))),
    }
    notes = ""
    germline = score.get("germline")
    if germline:
        notes = f"closest germline: {germline}"
    return ok("biophi_oasis", VERSION, record, metrics, notes)
