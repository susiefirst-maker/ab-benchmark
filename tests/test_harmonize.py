"""Tests for ab_benchmark.data.harmonize."""

import os
from pathlib import Path

import pandas as pd
import pytest

from ab_benchmark.data.harmonize import (
    canonicalize_ab_id,
    harmonization_summary,
    records_to_long_df,
    to_wide,
)
from ab_benchmark.schema import (
    AntibodyRecord,
    DevelopabilityEndpoint,
    EndpointKind,
    SourceDataset,
)


class TestCanonicalize:
    def test_trastuzumab_variants_normalize(self):
        assert canonicalize_ab_id("Trastuzumab") == "trastuzumab"
        assert canonicalize_ab_id("trastuzumab") == "trastuzumab"
        assert canonicalize_ab_id("  Trastuzumab  ") == "trastuzumab"

    def test_strips_non_alnum(self):
        assert canonicalize_ab_id("mAb 12-34") == "mab1234"
        assert canonicalize_ab_id("Ab/001") == "ab001"

    def test_preserves_inn_suffix(self):
        # Different humanization → different canonical.
        assert canonicalize_ab_id("abituximab") != canonicalize_ab_id("abituzumab")
        assert canonicalize_ab_id("adalimumab") != canonicalize_ab_id("adalibizumab")

    def test_empty(self):
        assert canonicalize_ab_id("") == ""
        assert canonicalize_ab_id("   ") == ""


class TestRecordsToLong:
    def test_empty_input(self):
        df = records_to_long_df([])
        assert len(df) == 0
        assert list(df.columns)[:3] == ["ab_id", "ab_id_canonical", "source"]

    def test_metadata_only_record_yields_one_row(self):
        r = AntibodyRecord(ab_id="Ab1", source=SourceDataset.SABDAB_THERA, vh="EVQL", vl="DIQM")
        df = records_to_long_df([r])
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["endpoint_kind"])
        assert df.iloc[0]["assay_detail"] == "metadata_only"
        assert df.iloc[0]["vh"] == "EVQL"

    def test_multiple_endpoints_yield_multiple_rows(self):
        r = AntibodyRecord(
            ab_id="Ab1",
            source=SourceDataset.JAIN_2017,
            endpoints=[
                DevelopabilityEndpoint(EndpointKind.TM_ONSET_C, 75.5, "°C"),
                DevelopabilityEndpoint(EndpointKind.HIC_RT, 9.2, "min"),
            ],
        )
        df = records_to_long_df([r])
        assert len(df) == 2
        assert set(df["endpoint_kind"]) == {"tm_onset_c", "hic_rt"}

    def test_unit_mismatch_flagged_in_assay_detail(self):
        r = AntibodyRecord(
            ab_id="Ab1",
            source=SourceDataset.JAIN_2017,
            endpoints=[DevelopabilityEndpoint(EndpointKind.TM_ONSET_C, 75.5, "Kelvin")],
        )
        df = records_to_long_df([r])
        assert "UNIT MISMATCH" in df.iloc[0]["assay_detail"]

    def test_duplicate_endpoints_marked(self):
        r1 = AntibodyRecord(
            ab_id="Ab1",
            source=SourceDataset.JAIN_2017,
            endpoints=[DevelopabilityEndpoint(EndpointKind.TM_ONSET_C, 75.5, "°C")],
        )
        r2 = AntibodyRecord(
            ab_id="Ab1",
            source=SourceDataset.JAIN_2017,
            endpoints=[DevelopabilityEndpoint(EndpointKind.TM_ONSET_C, 80.0, "°C")],
        )
        df = records_to_long_df([r1, r2])
        assert len(df) == 2
        duplicates = df[df["assay_detail"] == "duplicate skipped"]
        assert len(duplicates) == 1
        assert duplicates.iloc[0]["value"] == 80.0


class TestToWide:
    def test_empty(self):
        df = to_wide(pd.DataFrame(columns=["ab_id_canonical", "source", "endpoint_kind", "value"]))
        assert len(df) == 0

    def test_pivots_endpoints(self):
        r = AntibodyRecord(
            ab_id="Ab1",
            source=SourceDataset.JAIN_2017,
            vh="EVQL",
            vl="DIQM",
            endpoints=[
                DevelopabilityEndpoint(EndpointKind.TM_ONSET_C, 75.5, "°C"),
                DevelopabilityEndpoint(EndpointKind.HIC_RT, 9.2, "min"),
            ],
        )
        long_df = records_to_long_df([r])
        wide = to_wide(long_df)
        assert len(wide) == 1
        assert wide.iloc[0]["tm_onset_c"] == 75.5
        assert wide.iloc[0]["hic_rt"] == 9.2
        assert wide.iloc[0]["vh"] == "EVQL"

    def test_metadata_only_record_preserved(self):
        r = AntibodyRecord(ab_id="Ab1", source=SourceDataset.SABDAB_THERA, vh="EVQL", vl="DIQM")
        long_df = records_to_long_df([r])
        wide = to_wide(long_df)
        assert len(wide) == 1
        assert wide.iloc[0]["vh"] == "EVQL"

    def test_two_records_same_ab_different_sources_two_wide_rows(self):
        r1 = AntibodyRecord(
            ab_id="Trastuzumab",
            source=SourceDataset.JAIN_2017,
            endpoints=[DevelopabilityEndpoint(EndpointKind.TM_ONSET_C, 75.5, "°C")],
        )
        r2 = AntibodyRecord(
            ab_id="Trastuzumab",
            source=SourceDataset.SABDAB_THERA,
            vh="EVQL", vl="DIQM",
        )
        long_df = records_to_long_df([r1, r2])
        wide = to_wide(long_df)
        assert len(wide) == 2
        assert set(wide["source"]) == {"jain_2017", "sabdab_thera"}


class TestSummary:
    def test_summary_shape(self):
        r1 = AntibodyRecord(
            ab_id="Trastuzumab",
            source=SourceDataset.JAIN_2017,
            endpoints=[DevelopabilityEndpoint(EndpointKind.TM_ONSET_C, 75.5, "°C")],
        )
        r2 = AntibodyRecord(
            ab_id="Trastuzumab",
            source=SourceDataset.SABDAB_THERA,
            vh="EVQL", vl="DIQM",
        )
        r3 = AntibodyRecord(
            ab_id="Adalimumab",
            source=SourceDataset.JAIN_2017,
            endpoints=[DevelopabilityEndpoint(EndpointKind.HIC_RT, 9.0, "min")],
        )
        long_df = records_to_long_df([r1, r2, r3])
        s = harmonization_summary(long_df)
        assert s["total_unique_antibodies"] == 2
        assert s["antibodies_in_multiple_sources"] == 1  # only Trastuzumab
        assert s["measurements_by_endpoint"] == {"tm_onset_c": 1, "hic_rt": 1}
        assert s["unique_antibodies_by_source"] == {"jain_2017": 2, "sabdab_thera": 1}


class TestPipelineIntegration:
    """End-to-end integration — builds the actual parquet from real files."""

    _default_data = Path(__file__).resolve().parents[2] / "ProtePilot" / "data"
    JAIN = Path(os.environ.get("AB_BENCHMARK_JAIN_CSV", _default_data / "Jain137_Cleaned_Training_Data.csv"))
    SABDAB = Path(os.environ.get("AB_BENCHMARK_SABDAB_CSV", _default_data / "TheraSAbDab_SeqStruc_OnlineDownload.csv"))

    @pytest.mark.skipif(
        not JAIN.exists() or not SABDAB.exists(),
        reason="real data files not available",
    )
    def test_real_harmonization_meets_phase0_targets(self, tmp_path):
        from ab_benchmark.data.loaders import load_jain_2017, load_sabdab_thera

        records = load_jain_2017(self.JAIN) + load_sabdab_thera(self.SABDAB)
        long_df = records_to_long_df(records)
        summary = harmonization_summary(long_df)

        assert summary["total_unique_antibodies"] >= 400, summary

        # Jain's 6 endpoints should each have >= 130 measurements.
        for endpoint in ["tm_onset_c", "hic_rt", "ac_sins", "bvp_score", "psr_score", "expression_mgl"]:
            assert summary["measurements_by_endpoint"].get(endpoint, 0) >= 130, (
                f"{endpoint}: {summary['measurements_by_endpoint']}"
            )

    @pytest.mark.skipif(
        not JAIN.exists() or not SABDAB.exists(),
        reason="real data files not available",
    )
    def test_wide_view_on_real_data(self, tmp_path):
        from ab_benchmark.data.loaders import load_jain_2017, load_sabdab_thera

        records = load_jain_2017(self.JAIN) + load_sabdab_thera(self.SABDAB)
        long_df = records_to_long_df(records)
        wide = to_wide(long_df)

        # Wide: one row per (ab_id_canonical, source).
        assert len(wide) >= 1200
        # Every Jain row has all 6 endpoints populated.
        jain_rows = wide[wide["source"] == "jain_2017"]
        for endpoint in ["tm_onset_c", "hic_rt", "ac_sins", "bvp_score", "psr_score", "expression_mgl"]:
            n_valued = jain_rows[endpoint].notna().sum()
            assert n_valued >= 130, f"{endpoint} in Jain rows: {n_valued} non-null"
