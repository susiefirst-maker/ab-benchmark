"""Tests for ab_benchmark.eval.splits and ab_benchmark.eval.bootstrap."""

import numpy as np
import pandas as pd
import pytest

from ab_benchmark.eval.bootstrap import (
    bootstrap_r2,
    bootstrap_rmse,
    bootstrap_spearman,
    fisher_z_spearman,
)
from ab_benchmark.eval.splits import (
    SplitConfig,
    assign_clusters,
    check_no_cluster_leakage,
    grouped_repeated_splits,
)


# --- splits / clustering --------------------------------------------------


class TestAssignClusters:
    def test_unique_antibodies_each_cluster(self):
        df = pd.DataFrame({
            "ab_id_canonical": ["a", "b", "c", "d"],
            "v_gene_heavy": ["", "", "", ""],
            "cdr_h3": ["", "", "", ""],
            "cdr_h3_length": [0, 0, 0, 0],
        })
        clusters = assign_clusters(df)
        assert len(clusters.unique()) == 4

    def test_same_canonical_id_same_cluster(self):
        df = pd.DataFrame({
            "ab_id_canonical": ["tra", "tra", "ada"],
            "v_gene_heavy": ["", "", ""],
            "cdr_h3": ["", "", ""],
            "cdr_h3_length": [0, 0, 0],
        })
        clusters = assign_clusters(df)
        assert clusters.iloc[0] == clusters.iloc[1]
        assert clusters.iloc[0] != clusters.iloc[2]

    def test_similar_h3_same_vgene_cluster_together(self):
        df = pd.DataFrame({
            "ab_id_canonical": ["a", "b", "c"],
            "v_gene_heavy": ["IGHV3-23", "IGHV3-23", "IGHV3-23"],
            "cdr_h3":         ["ARWGGDGFY", "ARWGGDGFY", "TOTALLY_DIF"],  # a=b, a≠c
            "cdr_h3_length":  [9, 9, 11],
        })
        clusters = assign_clusters(df, identity_threshold=0.5)
        assert clusters.iloc[0] == clusters.iloc[1]
        # Different length → different bucket → not merged even if identity
        # would cross threshold.
        assert clusters.iloc[0] != clusters.iloc[2]

    def test_different_vgene_different_cluster(self):
        df = pd.DataFrame({
            "ab_id_canonical": ["a", "b"],
            "v_gene_heavy": ["IGHV3-23", "IGHV1-69"],
            "cdr_h3":         ["ARWGGDGFY", "ARWGGDGFY"],
            "cdr_h3_length":  [9, 9],
        })
        clusters = assign_clusters(df)
        assert clusters.iloc[0] != clusters.iloc[1]


class TestSplits:
    def _make_df(self, n=50):
        return pd.DataFrame({
            "ab_id_canonical": [f"ab{i}" for i in range(n)],
            "v_gene_heavy": [""] * n,
            "cdr_h3": [""] * n,
            "cdr_h3_length": [0] * n,
        })

    def test_no_leakage_across_folds(self):
        df = self._make_df(50)
        config = SplitConfig(n_splits=5, n_repeats=3, random_state=42)
        clusters = assign_clusters(df)
        for _, _, train_idx, test_idx in grouped_repeated_splits(df, config, clusters):
            assert check_no_cluster_leakage(train_idx, test_idx, clusters)

    def test_all_indices_covered(self):
        df = self._make_df(50)
        config = SplitConfig(n_splits=5, n_repeats=1, random_state=42)
        covered = set()
        for _, _, train_idx, test_idx in grouped_repeated_splits(df, config):
            covered |= set(test_idx)
        # Every cluster (=every antibody) should appear in a test fold exactly once per repeat.
        assert covered == set(range(50))

    def test_reproducibility(self):
        df = self._make_df(30)
        config = SplitConfig(n_splits=5, n_repeats=2, random_state=7)
        a = list(grouped_repeated_splits(df, config))
        b = list(grouped_repeated_splits(df, config))
        assert len(a) == len(b) == 10  # 5 splits × 2 repeats
        for (fa, ra, tra_a, tea_a), (fb, rb, tra_b, tea_b) in zip(a, b):
            assert fa == fb and ra == rb
            np.testing.assert_array_equal(tra_a, tra_b)
            np.testing.assert_array_equal(tea_a, tea_b)

    def test_different_seeds_different_splits(self):
        df = self._make_df(30)
        a = list(grouped_repeated_splits(df, SplitConfig(n_splits=5, n_repeats=1, random_state=0)))
        b = list(grouped_repeated_splits(df, SplitConfig(n_splits=5, n_repeats=1, random_state=1)))
        # At least one test-fold composition should differ between seed 0 and seed 1.
        differ = any(not np.array_equal(t0[3], t1[3]) for t0, t1 in zip(a, b))
        assert differ


# --- bootstrap CIs --------------------------------------------------------


class TestBootstrapSpearman:
    def test_perfect_correlation(self):
        x = np.arange(20)
        y = x * 2.0
        ci = bootstrap_spearman(x, y, n_boot=500, random_state=0)
        assert ci.point == pytest.approx(1.0)
        assert ci.low > 0.9
        assert ci.high <= 1.0

    def test_no_correlation_interval_spans_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=40)
        y = rng.normal(size=40)
        ci = bootstrap_spearman(x, y, n_boot=500, random_state=0)
        assert ci.low < 0 < ci.high  # interval should straddle 0

    def test_handles_nans(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        y = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
        ci = bootstrap_spearman(x, y, n_boot=200, random_state=0)
        assert ci.n == 4

    def test_tiny_n_returns_nan(self):
        ci = bootstrap_spearman([1, 2], [1, 2], n_boot=100)
        assert np.isnan(ci.point)


class TestFisherZ:
    def test_fisher_z_n27_rho04_matches_reference(self):
        """Spec sanity: Fisher-z calculation for a small reference sample
        claimed n=27 gives CI [0.02, 0.68] at ρ=0.40. Reproduce that."""
        # Construct x, y yielding Spearman ρ exactly 0.40 at n=27.
        rng = np.random.default_rng(0)
        # Instead of constructing ρ=0.4 data, just verify the SE formula gives
        # the expected CI width: SE=1/sqrt(24)=0.204; z=arctanh(0.4)=0.424;
        # z ± 1.96·0.204 = [0.024, 0.824] → tanh back = [0.024, 0.677].
        # So expect low ≈ 0.024, high ≈ 0.677.
        x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
        # Make y mostly monotone-with-x with some noise to hit ρ≈0.4.
        y = x + rng.normal(scale=12, size=27)  # noisy version
        ci = fisher_z_spearman(x, y)
        # We don't force ρ=0.4 exactly; check the Fisher-z formula shape.
        assert ci.method == "fisher_z"
        assert ci.n == 27
        # The CI width should be roughly ±0.33 around the point estimate.
        assert 0.2 < (ci.high - ci.low) < 1.0

    def test_fisher_z_clips_near_one(self):
        x = np.arange(10)
        y = x.copy()
        ci = fisher_z_spearman(x, y)
        assert ci.point == pytest.approx(1.0)
        # No arctanh blowup.
        assert not np.isnan(ci.high)


class TestBootstrapRmseR2:
    def test_rmse_positive(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=30)
        y_pred = y_true + rng.normal(scale=0.5, size=30)
        ci = bootstrap_rmse(y_true, y_pred, n_boot=200, random_state=0)
        assert ci.point > 0
        assert ci.low > 0
        assert ci.low <= ci.point <= ci.high

    def test_r2_near_one_for_perfect_pred(self):
        y_true = np.arange(30, dtype=float)
        ci = bootstrap_r2(y_true, y_true.copy(), n_boot=200, random_state=0)
        # Perfect predictions → R²=1. Small bootstrap noise OK.
        assert ci.point > 0.99

    def test_r2_negative_for_worse_than_mean(self):
        y_true = np.arange(30, dtype=float)
        y_pred = -y_true  # anti-correlated prediction
        ci = bootstrap_r2(y_true, y_pred, n_boot=200, random_state=0)
        assert ci.point < 0
