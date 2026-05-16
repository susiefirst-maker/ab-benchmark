"""Bootstrap 95% confidence intervals for ranking and regression metrics.

Non-parametric percentile bootstrap. For n < 30 we also report the
Fisher-z back-transformed interval for Spearman ρ, which is more stable
than the percentile method at small n (and matches the calculation the
reviewers did on n=27 test sets).

Reference for Fisher-z:
    For Spearman ρ, z = artanh(ρ) has approximate SE = 1 / sqrt(n-3).
    95% interval on z is z ± 1.96·SE; back-transform via tanh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class CI:
    point: float
    low: float
    high: float
    n: int
    method: str  # 'bootstrap' or 'fisher_z'

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.low, self.high

    def __repr__(self) -> str:
        return f"{self.point:+.3f} [{self.low:+.3f}, {self.high:+.3f}] n={self.n} ({self.method})"


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> CI:
    """Percentile bootstrap 95% CI on Spearman ρ."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"), n=n, method="bootstrap")

    rho_hat = spearmanr(x, y).statistic
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = spearmanr(x[idx], y[idx]).statistic
        stats[i] = s if not np.isnan(s) else rho_hat

    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return CI(point=float(rho_hat), low=low, high=high, n=n, method="bootstrap")


def fisher_z_spearman(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> CI:
    """Fisher-z back-transformed 95% CI on Spearman ρ. Requires n ≥ 4."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"), n=n, method="fisher_z")

    rho_hat = spearmanr(x, y).statistic
    if np.isnan(rho_hat):
        return CI(point=float("nan"), low=float("nan"), high=float("nan"), n=n, method="fisher_z")
    # Clip away from exact ±1 to avoid arctanh blowup.
    r = float(np.clip(rho_hat, -0.9999, 0.9999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    return CI(point=float(rho_hat), low=float(np.tanh(z_lo)), high=float(np.tanh(z_hi)),
              n=n, method="fisher_z")


def bootstrap_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> CI:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"), n=n, method="bootstrap")

    rmse_hat = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = np.sqrt(((y_true[idx] - y_pred[idx]) ** 2).mean())

    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return CI(point=rmse_hat, low=low, high=high, n=n, method="bootstrap")


def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> CI:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return CI(point=float("nan"), low=float("nan"), high=float("nan"), n=n, method="bootstrap")

    r2_hat = _r2(y_true, y_pred)
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = _r2(y_true[idx], y_pred[idx])

    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return CI(point=r2_hat, low=low, high=high, n=n, method="bootstrap")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
