"""Grouped-repeated cross-validation for leakage-resistant evaluation.

Leakage risk when evaluating antibody developability models:
    - Same antibody with measurements across sources → trivial memorization.
    - Highly similar CDR-H3 sequences (same clone lineage) → optimistic bias.
    - Shared V-gene germline can encode developability-relevant features
      that the model learns as "germline fingerprint."

Mitigation: cluster antibodies before splitting. Two antibodies land in
the same cluster (and therefore the same CV fold) if ANY of:

    1. Same ab_id_canonical (identity across sources).
    2. Same V-gene-heavy family AND same CDR-H3 length AND
       normalized CDR-H3 sequence similarity >= identity_threshold.

Clusters are assigned to folds via `GroupKFold` so no cluster is split.
Repeated CV means we run K-fold N times with different cluster-to-fold
assignments (via seed), giving a bootstrap-like distribution of fold
performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class SplitConfig:
    n_splits: int = 5
    n_repeats: int = 5
    identity_threshold: float = 0.5
    random_state: int = 0


# --- similarity ------------------------------------------------------------


def _identity(a: str, b: str) -> float:
    """Fraction of matching residues between two equal-length sequences. 0.0 if lengths differ."""
    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


# --- clustering ------------------------------------------------------------


def assign_clusters(df: pd.DataFrame, identity_threshold: float = 0.5) -> pd.Series:
    """Assign a cluster ID to every row.

    Required input columns: `ab_id_canonical`, `v_gene_heavy` (may be empty),
    `cdr_h3` (may be empty), `cdr_h3_length`.

    Strategy: first collapse on `ab_id_canonical` (identity), then on
    (v_gene_heavy, cdr_h3_length, cdr_h3-similarity) via single-linkage
    agglomeration. Antibodies with missing CDR-H3 are treated as their
    own cluster — this is the safe choice because we cannot guarantee
    leakage-free grouping without H3.

    Returns: pandas Series of integer cluster IDs indexed like `df`.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    cluster = np.arange(n)  # every row starts as its own cluster

    # Step 1 — identity on canonical ID.
    groups = df.groupby("ab_id_canonical").indices
    for _, idxs in groups.items():
        root = idxs[0]
        for i in idxs[1:]:
            cluster[i] = cluster[root]

    # Step 2 — bucket by (v_gene_heavy, cdr_h3_length), then cluster
    # within each bucket by CDR-H3 identity threshold.
    have_h3 = df["cdr_h3"].fillna("").astype(str) != ""
    bucket_df = df[have_h3].copy()
    if not bucket_df.empty and "cdr_h3_length" in bucket_df.columns:
        bucket_df["_v_gene"] = bucket_df["v_gene_heavy"].fillna("").astype(str)
        bucket_df["_bucket"] = list(
            zip(bucket_df["_v_gene"], bucket_df["cdr_h3_length"].astype(int))
        )
        for bucket_key, bucket_rows in bucket_df.groupby("_bucket"):
            idxs = bucket_rows.index.tolist()
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a = bucket_df.at[idxs[i], "cdr_h3"]
                    b = bucket_df.at[idxs[j], "cdr_h3"]
                    if _identity(a, b) >= identity_threshold:
                        # Union clusters of idxs[i] and idxs[j].
                        _union(cluster, idxs[i], idxs[j])

    # Canonicalize cluster labels to 0..n_clusters-1 for cleaner downstream use.
    _, renumbered = np.unique(cluster, return_inverse=True)
    return pd.Series(renumbered, index=df.index, name="cluster")


def _union(parent: np.ndarray, a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _find(parent: np.ndarray, a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


# --- splits ----------------------------------------------------------------


def grouped_repeated_splits(
    df: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    clusters: pd.Series | None = None,
):
    """Yield (fold_idx, repeat_idx, train_idx, test_idx) tuples.

    One iteration yields config.n_splits * config.n_repeats splits total.
    Clusters are computed from df if not pre-supplied. Deterministic given
    config.random_state.

    Each repeat uses a different random cluster-to-fold assignment.
    """
    if clusters is None:
        clusters = assign_clusters(df, identity_threshold=config.identity_threshold)

    rng = np.random.default_rng(config.random_state)

    for repeat_idx in range(config.n_repeats):
        # Shuffle cluster IDs for this repeat — GroupKFold assigns clusters
        # to folds in order, so shuffling changes which clusters go to which folds.
        unique_clusters = np.unique(clusters.to_numpy())
        shuffled = unique_clusters.copy()
        rng.shuffle(shuffled)
        remap = {old: new for new, old in enumerate(shuffled)}
        shuffled_labels = clusters.map(remap).to_numpy()

        gkf = GroupKFold(n_splits=config.n_splits)
        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(X=np.zeros(len(df)), y=np.zeros(len(df)), groups=shuffled_labels)
        ):
            yield fold_idx, repeat_idx, train_idx, test_idx


# --- helper ----------------------------------------------------------------


def check_no_cluster_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    clusters: pd.Series,
) -> bool:
    """Assert no cluster appears in both train and test. Returns True if clean."""
    train_clusters = set(clusters.iloc[train_idx].unique())
    test_clusters = set(clusters.iloc[test_idx].unique())
    return not (train_clusters & test_clusters)
