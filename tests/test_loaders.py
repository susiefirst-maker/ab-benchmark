"""Tests for ab_benchmark.data.loaders."""

import os
from pathlib import Path

import pytest

from ab_benchmark.data.loaders import (
    LOADER_REGISTRY,
    load_bailly_2020,
    load_jain_2017,
    load_prophet_ab,
    load_sabdab_thera,
    load_shehata_2019,
)
from ab_benchmark.schema import EndpointKind, SourceDataset

# Optional sibling data files used for smoke tests when available.
_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "ProtePilot" / "data"
JAIN_2017_CSV = Path(os.environ.get("AB_BENCHMARK_JAIN_CSV", _DEFAULT_DATA / "Jain137_Cleaned_Training_Data.csv"))
SABDAB_CSV = Path(os.environ.get("AB_BENCHMARK_SABDAB_CSV", _DEFAULT_DATA / "TheraSAbDab_SeqStruc_OnlineDownload.csv"))


class TestJain2017:
    @pytest.mark.skipif(not JAIN_2017_CSV.exists(), reason="Jain 2017 file not available")
    def test_loads_real_file(self):
        records = load_jain_2017(JAIN_2017_CSV)
        assert 130 <= len(records) <= 140, f"expected ~137 antibodies, got {len(records)}"
        assert all(r.source is SourceDataset.JAIN_2017 for r in records)

    @pytest.mark.skipif(not JAIN_2017_CSV.exists(), reason="Jain 2017 file not available")
    def test_all_records_have_at_least_some_endpoints(self):
        records = load_jain_2017(JAIN_2017_CSV)
        with_endpoints = [r for r in records if r.endpoints]
        # Not every antibody has every endpoint, but most should have something.
        assert len(with_endpoints) >= 130

    @pytest.mark.skipif(not JAIN_2017_CSV.exists(), reason="Jain 2017 file not available")
    def test_tm_values_are_physical(self):
        records = load_jain_2017(JAIN_2017_CSV)
        tms = [
            e.value for r in records for e in r.endpoints if e.kind is EndpointKind.TM_ONSET_C
        ]
        assert len(tms) > 100, "expected >100 Tm measurements"
        assert all(50 < t < 100 for t in tms), f"Tm outside physical range: {[t for t in tms if not 50 < t < 100][:5]}"

    @pytest.mark.skipif(not JAIN_2017_CSV.exists(), reason="Jain 2017 file not available")
    def test_sequences_present(self):
        records = load_jain_2017(JAIN_2017_CSV)
        with_vh = [r for r in records if r.vh]
        assert len(with_vh) >= 130, "most records should have VH sequence"

    @pytest.mark.skipif(not JAIN_2017_CSV.exists(), reason="Jain 2017 file not available")
    def test_known_antibody_present(self):
        records = load_jain_2017(JAIN_2017_CSV)
        names = {r.ab_id.lower() for r in records}
        # Trastuzumab or a common therapeutic should be in the Jain set.
        known = {"abituzumab", "trastuzumab", "bevacizumab", "adalimumab"}
        assert names & known, f"no known therapeutic found; got {list(names)[:5]}"

    @pytest.mark.skipif(not JAIN_2017_CSV.exists(), reason="Jain 2017 file not available")
    def test_extras_preserved(self):
        records = load_jain_2017(JAIN_2017_CSV)
        with_extras = [r for r in records if r.extras]
        assert len(with_extras) > 0, "expected non-endpoint columns in extras"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Jain 2017"):
            load_jain_2017(tmp_path / "does_not_exist.csv")


class TestSabDabThera:
    @pytest.mark.skipif(not SABDAB_CSV.exists(), reason="SAbDab-Thera file not available")
    def test_loads_real_file(self):
        records = load_sabdab_thera(SABDAB_CSV)
        assert len(records) > 200, f"expected >200 therapeutic antibodies, got {len(records)}"
        assert all(r.source is SourceDataset.SABDAB_THERA for r in records)

    @pytest.mark.skipif(not SABDAB_CSV.exists(), reason="SAbDab-Thera file not available")
    def test_no_endpoints(self):
        # SAbDab-Thera carries status/target/year but no biophysical assays.
        records = load_sabdab_thera(SABDAB_CSV)
        assert all(not r.endpoints for r in records)

    @pytest.mark.skipif(not SABDAB_CSV.exists(), reason="SAbDab-Thera file not available")
    def test_extras_have_clinical_metadata(self):
        records = load_sabdab_thera(SABDAB_CSV)
        r = records[0]
        assert "highest_clin_trial" in r.extras
        assert "target" in r.extras
        assert "therapeutic_format" in r.extras

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SAbDab-Thera"):
            load_sabdab_thera(tmp_path / "does_not_exist.csv")


class TestStubs:
    def test_shehata_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError, match="Shehata 2019"):
            load_shehata_2019(tmp_path / "whatever.csv")

    def test_bailly_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError, match="Bailly 2020"):
            load_bailly_2020(tmp_path / "whatever.csv")

    def test_prophet_ab_deferred(self, tmp_path):
        with pytest.raises(NotImplementedError, match="PROPHET-Ab"):
            load_prophet_ab(tmp_path / "whatever.csv")


class TestRegistry:
    def test_registry_has_all_sources(self):
        for src in SourceDataset:
            assert src in LOADER_REGISTRY, f"{src} missing from LOADER_REGISTRY"
