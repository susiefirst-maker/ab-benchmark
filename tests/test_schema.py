"""Tests for ab_benchmark.schema."""

import pytest

from ab_benchmark.schema import (
    AntibodyRecord,
    DevelopabilityEndpoint,
    EndpointKind,
    SourceDataset,
    UNITS,
)

TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


class TestAntibodyRecord:
    def test_minimal_record(self):
        r = AntibodyRecord(ab_id="Ab001", source=SourceDataset.JAIN_2017)
        assert r.ab_id == "Ab001"
        assert r.source is SourceDataset.JAIN_2017
        assert r.endpoints == []
        assert r.extras == {}

    def test_with_sequences(self):
        r = AntibodyRecord(
            ab_id="Trastuzumab",
            source=SourceDataset.JAIN_2017,
            vh=TRASTUZUMAB_VH,
            vl=TRASTUZUMAB_VL,
        )
        assert len(r.vh) > 100
        assert len(r.vl) > 100

    def test_rejects_non_standard_aa_in_vh(self):
        with pytest.raises(ValueError, match="non-standard amino acids"):
            AntibodyRecord(
                ab_id="bad",
                source=SourceDataset.JAIN_2017,
                vh="EVQLVBXZ",
            )

    def test_rejects_non_standard_aa_in_vl(self):
        with pytest.raises(ValueError, match="non-standard amino acids"):
            AntibodyRecord(
                ab_id="bad",
                source=SourceDataset.JAIN_2017,
                vl="DIQMTQJ",
            )

    def test_rejects_cdr_length_mismatch(self):
        with pytest.raises(ValueError, match="cdr_h3_length"):
            AntibodyRecord(
                ab_id="bad",
                source=SourceDataset.JAIN_2017,
                cdr_h3="ARGDY",
                cdr_h3_length=99,
            )

    def test_allows_cdr_without_length(self):
        r = AntibodyRecord(
            ab_id="ok",
            source=SourceDataset.JAIN_2017,
            cdr_h3="ARGDY",
        )
        assert r.cdr_h3 == "ARGDY"

    def test_has_endpoint_and_get_endpoint(self):
        e = DevelopabilityEndpoint(kind=EndpointKind.HMW_PCT, value=1.2, unit="%")
        r = AntibodyRecord(
            ab_id="Ab001",
            source=SourceDataset.JAIN_2017,
            endpoints=[e],
        )
        assert r.has_endpoint(EndpointKind.HMW_PCT)
        assert not r.has_endpoint(EndpointKind.TM_ONSET_C)
        assert r.get_endpoint(EndpointKind.HMW_PCT) is e
        assert r.get_endpoint(EndpointKind.TM_ONSET_C) is None


class TestEnums:
    def test_source_dataset_values(self):
        assert SourceDataset.JAIN_2017.value == "jain_2017"
        assert SourceDataset.PROPHET_AB.value == "prophet_ab"
        assert len(list(SourceDataset)) == 5

    def test_endpoint_kinds_have_units(self):
        for kind in EndpointKind:
            assert kind in UNITS, f"{kind!r} missing from UNITS table"

    def test_endpoint_frozen(self):
        e = DevelopabilityEndpoint(kind=EndpointKind.TM_ONSET_C, value=70.0, unit="°C")
        with pytest.raises(Exception):
            e.value = 80.0  # frozen dataclass → FrozenInstanceError
