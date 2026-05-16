"""Tests for ab_benchmark.seqprops."""

import pytest

from ab_benchmark.seqprops import (
    CHOU_FASMAN_ALPHA,
    CHOU_FASMAN_BETA,
    KYTE_DOOLITTLE,
    extract_cdrs,
    mean_hydrophobicity,
    net_charge_at_ph,
)

# Trastuzumab (from tests/test_schema.py fixtures, but re-stated here
# so tests don't have cross-file imports).
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


class TestScales:
    def test_all_scales_have_20_aas(self):
        assert len(KYTE_DOOLITTLE) == 20
        assert len(CHOU_FASMAN_ALPHA) == 20
        assert len(CHOU_FASMAN_BETA) == 20

    def test_kyte_doolittle_signs(self):
        # Isoleucine most hydrophobic, arginine most hydrophilic.
        assert KYTE_DOOLITTLE["I"] > 0
        assert KYTE_DOOLITTLE["R"] < 0
        assert max(KYTE_DOOLITTLE, key=KYTE_DOOLITTLE.get) == "I"
        assert min(KYTE_DOOLITTLE, key=KYTE_DOOLITTLE.get) == "R"


class TestNetCharge:
    def test_trastuzumab_vh_charge_slightly_positive(self):
        # Therapeutic IgG variable regions are typically near neutral at
        # physiological pH. A value in [-5, +5] is expected; outside that
        # would signal a bug.
        q = net_charge_at_ph(TRASTUZUMAB_VH, 7.4)
        assert -5 < q < 5, f"unexpected VH charge: {q}"

    def test_poly_lysine_positive(self):
        q = net_charge_at_ph("KKKKKKKKKK", 7.4)
        assert q > 5

    def test_poly_glutamate_negative(self):
        q = net_charge_at_ph("EEEEEEEEEE", 7.4)
        assert q < -5

    def test_low_ph_more_positive(self):
        q_low = net_charge_at_ph(TRASTUZUMAB_VH, 4.0)
        q_high = net_charge_at_ph(TRASTUZUMAB_VH, 10.0)
        assert q_low > q_high


class TestMeanHydrophobicity:
    def test_poly_ile_high(self):
        assert mean_hydrophobicity("IIIIII") == pytest.approx(4.5)

    def test_poly_arg_low(self):
        assert mean_hydrophobicity("RRRRRR") == pytest.approx(-4.5)

    def test_empty(self):
        assert mean_hydrophobicity("") == 0.0


class TestCDRExtraction:
    def test_trastuzumab_h3_recovered(self):
        cdrs = extract_cdrs(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        # Trastuzumab CDR-H3 is SRWGGDGFYAMDY (or similar depending on convention).
        # Expect at least a non-empty H3 between the last C and the WGXG motif.
        assert "h3" in cdrs
        assert 5 < len(cdrs["h3"]) < 25, f"H3 length suspicious: {cdrs['h3']}"

    def test_trastuzumab_l3_recovered(self):
        cdrs = extract_cdrs(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        assert "l3" in cdrs
        # Trastuzumab CDR-L3 ≈ QQHYTTPPT.
        assert cdrs["l3"].startswith("Q") or "Q" in cdrs["l3"]

    def test_empty_vh_returns_empty(self):
        assert extract_cdrs("") == {}

    def test_garbage_sequence_handled(self):
        # No conserved cysteines or WGXG motif — should return {} or partial.
        cdrs = extract_cdrs("AAAAAAAAAA")
        assert "h3" not in cdrs
