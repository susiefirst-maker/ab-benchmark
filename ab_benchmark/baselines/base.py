"""Baseline contract: single shape for every baseline result.

Every baseline function takes an AntibodyRecord and returns a BaselineResult.
If the baseline cannot run (missing structure, missing install, missing DB),
it returns BaselineResult(available=False, notes=...) rather than raising.
This lets `run_all` produce a complete report with honest "unavailable"
cells instead of silent failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ab_benchmark.schema import AntibodyRecord


@dataclass
class BaselineResult:
    """One baseline's output for one antibody."""

    baseline: str          # e.g. "tap", "camsol_intrinsic"
    version: str           # implementation version string
    ab_id: str             # source-native antibody ID
    ab_id_canonical: str
    available: bool        # whether compute produced values
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""        # explanation of unavailability or edge cases


# Every baseline is a function with this signature.
BaselineFn = Callable[[AntibodyRecord], BaselineResult]


# Conveniences for writing unavailability results.


def unavailable(baseline: str, version: str, record: AntibodyRecord, reason: str) -> BaselineResult:
    """Uniform "cannot run" result with the reason preserved."""
    from ab_benchmark.data.harmonize import canonicalize_ab_id

    return BaselineResult(
        baseline=baseline,
        version=version,
        ab_id=record.ab_id,
        ab_id_canonical=canonicalize_ab_id(record.ab_id),
        available=False,
        metrics={},
        notes=reason,
    )


def ok(baseline: str, version: str, record: AntibodyRecord, metrics: dict[str, float], notes: str = "") -> BaselineResult:
    from ab_benchmark.data.harmonize import canonicalize_ab_id

    return BaselineResult(
        baseline=baseline,
        version=version,
        ab_id=record.ab_id,
        ab_id_canonical=canonicalize_ab_id(record.ab_id),
        available=True,
        metrics=metrics,
        notes=notes,
    )
