# ab-benchmark

> Shared antibody developability benchmark: harmonized datasets, developability baseline proxies, leakage-resistant split utilities, and bootstrap 95% confidence intervals.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)

## Key results

| Metric | Value |
|--------|-------|
| Harmonized antibodies | 1,243 in the committed local run; active loaders currently cover Jain 2017 + SAbDab-Thera |
| Spearman correlation pairs | 72 (with bootstrap 95% CIs + Fisher-z transform) |
| TAP coverage | 1,233/1,243 (99.2%) |
| Developability Index coverage | 1,243/1,243 (100%) |
| CamSol-intrinsic coverage | 1,241/1,243 (99.8%) |
| CV strategy | 5-fold GroupKFold (V-gene x CDR-H3 length x 50% identity) |

## Problem

Existing antibody developability tools report single-point metrics on narrow datasets with random train/test splits. This package provides a conservative baseline audit and shared schema for TAP/DI/CamSol-style metrics, with grouped-CV utilities available for downstream modeling.

## Approach

Active sources are Jain 2017 and SAbDab-Thera. Baseline families include TAP/DI/CamSol sequence proxies plus wrappers for BioPhi/OASis, DynaMine, CABS-flex, and PROPHET-Ab where cached/external artifacts are available. Correlation outputs use Spearman correlations with bootstrap 95% CIs and Fisher-z transform for small-sample correction.

**Explicit corrections vs. common misconceptions:** TAP (Raybould 2019) uses modeled structure-derived surface patches -- it is **not** sequence-only. Developability Index (Lauer 2012) is explicitly structure/patch-based. This repo's current TAP/DI implementations are sequence-proxy audit metrics, not replacements for full structure-patch calculations. CamSol-intrinsic is sequence-based; CamSol-combination uses structure.

## Quick start

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    python -m ab_benchmark.baselines.run_all \
      --config configs/datasets.yaml \
      --out reports/baseline_results.csv

## Reproduction

    # 1. Build harmonized dataset
    python -m ab_benchmark.data.build_harmonized \
      --config configs/datasets.yaml \
      --out data/processed/harmonized_antibody_dev.parquet
    # 2. Run all baselines
    python -m ab_benchmark.baselines.run_all \
      --config configs/datasets.yaml \
      --out reports/baseline_results.csv
    # 3. Compute correlations with bootstrap CIs
    python scripts/compute_baseline_correlations.py \
      --baselines reports/baseline_results.csv \
      --harmonized data/processed/harmonized_antibody_dev.parquet \
      --out reports/baseline_correlations.csv
    # 4. Run tests
    pytest tests/ --cov=ab_benchmark

## Datasets harmonized

| Source | Antibodies | Endpoints |
|--------|-----------|-----------|
| Jain et al. 2017 PNAS | ~137 | %HMW, Tm, kD, HIC-RT, AC-SINS, BVP |
| SAbDab-Thera annotations | active if CSV configured | therapeutic annotations |

Default `configs/datasets.yaml` assumes this repo is checked out next to `../ProtePilot`. Edit the paths or point them at local raw downloads before reproducing.

## Citation

    @software{wu2026abbenchmark,
      author = {Wu, Di},
      title  = {ab-benchmark: Antibody Developability Baseline Benchmark},
      year   = {2026},
      url    = {https://github.com/diwuhub/ab-benchmark}
    }

## References

- Jain et al. 2017 PNAS (DOI: 10.1073/pnas.1616408114)
- Raybould et al. 2019 PNAS -- TAP
- Lauer et al. 2012 J. Pharm. Sci. -- Developability Index
- Sormanni et al. 2015 J. Mol. Biol. -- CamSol-intrinsic
- Prihoda et al. 2022 -- BioPhi
- Cilia et al. 2013 Nat. Commun. -- DynaMine
- Johnson et al. 2024 -- PROPHET-Ab

## License

MIT. See LICENSE.
