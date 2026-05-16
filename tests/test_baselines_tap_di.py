"""Tests for TAP and Developability Index baselines."""

from ab_benchmark.baselines.developability_index import compute_developability_index
from ab_benchmark.baselines.tap import compute_tap
from ab_benchmark.schema import AntibodyRecord, SourceDataset

TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


def _trastuzumab() -> AntibodyRecord:
    return AntibodyRecord(
        ab_id="Trastuzumab",
        source=SourceDataset.JAIN_2017,
        vh=TRASTUZUMAB_VH,
        vl=TRASTUZUMAB_VL,
    )


class TestTAP:
    def test_trastuzumab_runs(self):
        res = compute_tap(_trastuzumab())
        assert res.available
        assert res.baseline == "tap"

    def test_trastuzumab_h3_length_reasonable(self):
        res = compute_tap(_trastuzumab())
        # Trastuzumab H3 is ~13 residues by Kabat/IMGT.
        assert 8 <= res.metrics["tap_h3_length"] <= 25

    def test_trastuzumab_not_flagged_excessively(self):
        # Trastuzumab is a well-developed clinical mAb; it should not
        # trip most TAP risk flags.
        res = compute_tap(_trastuzumab())
        assert res.metrics["tap_risk_flag_count"] <= 2

    def test_expected_metric_keys(self):
        res = compute_tap(_trastuzumab())
        expected = {
            "tap_h3_length",
            "tap_cdr_total_length",
            "tap_cdr_mean_hydrophobicity",
            "tap_cdr_net_pos_count",
            "tap_cdr_net_neg_count",
            "tap_cdr_his_count",
            "tap_vh_charge",
            "tap_vl_charge",
            "tap_fv_charge_asymmetry",
            "tap_risk_flag_count",
        }
        assert expected.issubset(set(res.metrics))

    def test_no_vh_returns_unavailable(self):
        r = AntibodyRecord(ab_id="Ab1", source=SourceDataset.JAIN_2017)
        res = compute_tap(r)
        assert not res.available
        assert "no VH" in res.notes

    def test_unusual_framework_returns_unavailable(self):
        # 40 random residues — no Cys→WGxG motif.
        r = AntibodyRecord(
            ab_id="Bad1",
            source=SourceDataset.JAIN_2017,
            vh="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        )
        res = compute_tap(r)
        assert not res.available
        assert "CDR-H3" in res.notes


class TestDevelopabilityIndex:
    def test_trastuzumab_runs(self):
        res = compute_developability_index(_trastuzumab())
        assert res.available
        assert res.baseline == "developability_index"

    def test_expected_metric_keys(self):
        res = compute_developability_index(_trastuzumab())
        expected = {
            "di_sfvcsp_seq",
            "di_vh_charge_ph6",
            "di_vl_charge_ph6",
            "di_mean_hydrophobicity",
            "di_hydro_patch_length",
            "di_seq_proxy",
        }
        assert expected.issubset(set(res.metrics))

    def test_sfvcsp_sign_sensible(self):
        # For trastuzumab at pH 6.0, VH is positively charged (many K/R)
        # and VL is close to neutral or slightly positive — SFvCSP should
        # be a finite value in a reasonable range.
        res = compute_developability_index(_trastuzumab())
        sfvcsp = res.metrics["di_sfvcsp_seq"]
        assert -100 < sfvcsp < 100

    def test_missing_vl_returns_unavailable(self):
        r = AntibodyRecord(ab_id="Ab1", source=SourceDataset.JAIN_2017, vh=TRASTUZUMAB_VH)
        res = compute_developability_index(r)
        assert not res.available
        assert "requires both VH and VL" in res.notes
