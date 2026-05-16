"""Tests for the four wrapper baselines (BioPhi, DynaMine, CABS-flex, PROPHET-Ab).

None of these are expected to be available in a fresh Phase 0 checkout —
their purpose is to report unavailability uniformly so `run_all` can
produce a complete table. These tests verify the unavailability reason
is meaningful and the BaselineResult shape is correct.
"""

import pandas as pd
import pytest

from ab_benchmark.baselines import BASELINE_REGISTRY
from ab_benchmark.baselines.biophi import compute_biophi_oasis
from ab_benchmark.baselines.cabs_flex import compute_cabs_flex
from ab_benchmark.baselines.dynamine import compute_dynamine
from ab_benchmark.baselines.prophet_ab import compute_prophet_ab
from ab_benchmark.schema import AntibodyRecord, SourceDataset


TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


@pytest.fixture
def trastuzumab():
    return AntibodyRecord(
        ab_id="Trastuzumab",
        source=SourceDataset.JAIN_2017,
        vh=TRASTUZUMAB_VH,
        vl=TRASTUZUMAB_VL,
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear baseline availability caches so env-var edits take effect per test."""
    from ab_benchmark.baselines.biophi import _check_availability as biophi_chk
    from ab_benchmark.baselines.cabs_flex import _check_availability as cabs_chk
    from ab_benchmark.baselines.dynamine import _check_availability as dyn_chk
    from ab_benchmark.baselines.prophet_ab import _check_cache as prop_chk
    biophi_chk.cache_clear()
    cabs_chk.cache_clear()
    dyn_chk.cache_clear()
    prop_chk.cache_clear()


class TestBioPhi:
    def test_unavailable_without_env(self, trastuzumab, monkeypatch):
        monkeypatch.delenv("BIOPHI_OASIS_DB", raising=False)
        res = compute_biophi_oasis(trastuzumab)
        assert not res.available
        # Either missing package or missing DB — both acceptable for Phase 0.
        assert res.notes

    def test_shape(self, trastuzumab, monkeypatch):
        monkeypatch.delenv("BIOPHI_OASIS_DB", raising=False)
        res = compute_biophi_oasis(trastuzumab)
        assert res.baseline == "biophi_oasis"
        assert res.ab_id == "Trastuzumab"
        assert res.ab_id_canonical == "trastuzumab"


class TestDynaMine:
    def test_unavailable_without_cache(self, trastuzumab, monkeypatch):
        monkeypatch.delenv("DYNAMINE_CACHE_TSV", raising=False)
        res = compute_dynamine(trastuzumab)
        assert not res.available
        assert "DynaMine" in res.notes or "DYNAMINE" in res.notes

    def test_reads_cache(self, trastuzumab, monkeypatch, tmp_path):
        tsv = tmp_path / "dynamine_cache.tsv"
        pd.DataFrame({
            "ab_id": ["Trastuzumab"] * 5,
            "residue_index": [1, 2, 3, 4, 5],
            "s2": [0.9, 0.85, 0.8, 0.7, 0.6],
        }).to_csv(tsv, sep="\t", index=False)
        monkeypatch.setenv("DYNAMINE_CACHE_TSV", str(tsv))
        res = compute_dynamine(trastuzumab)
        assert res.available
        assert res.metrics["dynamine_s2_mean"] == pytest.approx(0.77, abs=0.01)
        assert res.metrics["dynamine_s2_min"] == 0.6


class TestCABSflex:
    def test_unavailable_in_phase_0(self, trastuzumab, monkeypatch):
        monkeypatch.delenv("CABS_FLEX_STRUCTURE_DIR", raising=False)
        res = compute_cabs_flex(trastuzumab)
        assert not res.available
        assert "CABS" in res.notes or "cabs" in res.notes.lower() or "structure" in res.notes.lower()


class TestPROPHETAb:
    def test_unavailable_without_cache(self, trastuzumab, monkeypatch):
        monkeypatch.delenv("PROPHET_AB_CACHE_TSV", raising=False)
        res = compute_prophet_ab(trastuzumab)
        assert not res.available
        assert "Phase 1" in res.notes or "ESM" in res.notes

    def test_reads_cache_with_known_endpoints(self, trastuzumab, monkeypatch, tmp_path):
        tsv = tmp_path / "prophet_cache.tsv"
        pd.DataFrame({
            "ab_id": ["Trastuzumab", "Trastuzumab"],
            "endpoint": ["tm_onset_c", "hic_rt"],
            "prediction": [74.8, 9.3],
        }).to_csv(tsv, sep="\t", index=False)
        monkeypatch.setenv("PROPHET_AB_CACHE_TSV", str(tsv))
        res = compute_prophet_ab(trastuzumab)
        assert res.available
        assert res.metrics["prophet_ab_pred_tm_onset_c"] == 74.8
        assert res.metrics["prophet_ab_pred_hic_rt"] == 9.3

    def test_ignores_unknown_endpoints(self, trastuzumab, monkeypatch, tmp_path):
        tsv = tmp_path / "prophet_cache.tsv"
        pd.DataFrame({
            "ab_id": ["Trastuzumab"],
            "endpoint": ["fictional_endpoint"],
            "prediction": [42.0],
        }).to_csv(tsv, sep="\t", index=False)
        monkeypatch.setenv("PROPHET_AB_CACHE_TSV", str(tsv))
        res = compute_prophet_ab(trastuzumab)
        assert not res.available
        assert "no recognized" in res.notes


class TestRegistry:
    def test_all_seven_baselines_registered(self):
        expected = {
            "tap",
            "developability_index",
            "camsol_intrinsic",
            "biophi_oasis",
            "dynamine",
            "cabs_flex",
            "prophet_ab",
        }
        assert set(BASELINE_REGISTRY) == expected
