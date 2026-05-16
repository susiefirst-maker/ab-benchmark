"""CABS-flex coarse-grained flexibility wrapper (stub for Phase 0).

Reference:
    Kuriata, Iglesias, Pujols, Kurcinski, Kmiecik, Ventura. "Aggrescan3D
    standalone package for structure-based prediction of protein
    aggregation properties." Bioinformatics 35 (2019): 3834-3835.
    Kuriata, Gierut, Oleniecki, Ciemny, Kolinski, Kurcinski, Kmiecik.
    "CABS-flex 2.0: a web server for fast simulations of flexibility of
    protein structures." Nucleic Acids Research 46 (2018): W338-W343.
    DOI: 10.1093/nar/gky356

CABS-flex is a coarse-grained structure-based flexibility predictor.
It requires a 3D structure (PDB file) and a non-trivial install
(MODELLER + HMMER + its own Python dependencies). Typical runtime is
10-30 min per antibody. In Phase 0 we don't have structures yet, so
this wrapper returns available=False. The wrapper becomes real once:

    1. cabs_flex Python package or binary is on PATH
    2. CABS_FLEX_STRUCTURE_DIR points to a directory with {ab_id}.pdb files

When available, we report:
    cabs_flex_rmsf_mean      — mean per-residue RMSF (Å)
    cabs_flex_rmsf_cdr_h3    — CDR-H3 region RMSF (Å)
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache

from ab_benchmark.baselines.base import BaselineResult, unavailable
from ab_benchmark.schema import AntibodyRecord

VERSION = "0.1.0-stub"


@lru_cache(maxsize=1)
def _check_availability() -> tuple[bool, str]:
    if shutil.which("cabsflex") is None:
        try:
            import cabs_flex  # noqa: F401
        except ImportError:
            return False, (
                "CABS-flex not installed (see https://bitbucket.org/lcbio/cabsflex). "
                "Phase 0 uses stub; enable once structures available in Phase 3."
            )
    struct_dir = os.environ.get("CABS_FLEX_STRUCTURE_DIR")
    if not struct_dir:
        return False, "CABS_FLEX_STRUCTURE_DIR env var not set"
    if not os.path.isdir(struct_dir):
        return False, f"CABS_FLEX_STRUCTURE_DIR missing: {struct_dir}"
    return True, "ok"


def compute_cabs_flex(record: AntibodyRecord) -> BaselineResult:
    available, reason = _check_availability()
    if not available:
        return unavailable("cabs_flex", VERSION, record, reason)

    # If we get here, CABS-flex install + structure dir exist. Phase 3
    # integration: call cabsflex on {ab_id}.pdb, parse the RMSF output.
    # Returning unavailable even in the "detected" path, because Phase 0
    # explicitly defers structure-based flexibility to Phase 3.
    return unavailable(
        "cabs_flex", VERSION, record,
        "CABS-flex detected but Phase 0 defers structure-based flexibility to Phase 3"
    )
