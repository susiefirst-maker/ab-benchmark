"""Baseline calculators.

Every baseline is a function (AntibodyRecord) -> BaselineResult. The
registry maps short names to their compute functions so `run_all` can
iterate without hard-coding imports.
"""

from ab_benchmark.baselines.base import BaselineFn, BaselineResult, ok, unavailable
from ab_benchmark.baselines.biophi import compute_biophi_oasis
from ab_benchmark.baselines.cabs_flex import compute_cabs_flex
from ab_benchmark.baselines.camsol import compute_camsol_intrinsic
from ab_benchmark.baselines.developability_index import compute_developability_index
from ab_benchmark.baselines.dynamine import compute_dynamine
from ab_benchmark.baselines.prophet_ab import compute_prophet_ab
from ab_benchmark.baselines.tap import compute_tap

BASELINE_REGISTRY: dict[str, BaselineFn] = {
    "tap": compute_tap,
    "developability_index": compute_developability_index,
    "camsol_intrinsic": compute_camsol_intrinsic,
    "biophi_oasis": compute_biophi_oasis,
    "dynamine": compute_dynamine,
    "cabs_flex": compute_cabs_flex,
    "prophet_ab": compute_prophet_ab,
}

__all__ = [
    "BaselineFn",
    "BaselineResult",
    "BASELINE_REGISTRY",
    "compute_biophi_oasis",
    "compute_cabs_flex",
    "compute_camsol_intrinsic",
    "compute_developability_index",
    "compute_dynamine",
    "compute_prophet_ab",
    "compute_tap",
    "ok",
    "unavailable",
]
