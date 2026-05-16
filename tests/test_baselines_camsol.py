"""Tests for ab_benchmark.baselines.camsol."""

import numpy as np

from ab_benchmark.baselines.camsol import (
    compute_camsol_intrinsic,
    per_residue_score,
    smoothed_profile,
)
from ab_benchmark.schema import AntibodyRecord, SourceDataset

TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


class TestPerResidue:
    def test_charged_aas_get_nonzero_score(self):
        # K and R are charged and hydrophilic → positive CamSol contribution.
        assert per_residue_score("K") > 0
        assert per_residue_score("R") > 0

    def test_hydrophobic_aas_get_lower_score(self):
        # Ile and Leu are hydrophobic → negative hydrophobicity term dominates.
        assert per_residue_score("I") < 0
        assert per_residue_score("V") < 0

    def test_unknown_residue_returns_zero(self):
        assert per_residue_score("X") == 0.0


class TestSmoothedProfile:
    def test_length_preserved(self):
        prof = smoothed_profile(TRASTUZUMAB_VH)
        assert len(prof) == len(TRASTUZUMAB_VH)

    def test_empty_returns_empty(self):
        prof = smoothed_profile("")
        assert len(prof) == 0

    def test_reduces_variance(self):
        # Smoothing should reduce stddev vs the raw per-residue series.
        raw = np.array([per_residue_score(a) for a in TRASTUZUMAB_VH])
        sm = smoothed_profile(TRASTUZUMAB_VH)
        assert sm.std() < raw.std()


class TestCompute:
    def test_trastuzumab_returns_available(self):
        r = AntibodyRecord(
            ab_id="Trastuzumab",
            source=SourceDataset.JAIN_2017,
            vh=TRASTUZUMAB_VH,
            vl=TRASTUZUMAB_VL,
        )
        res = compute_camsol_intrinsic(r)
        assert res.available
        assert res.baseline == "camsol_intrinsic"
        assert set(res.metrics) >= {
            "camsol_intrinsic_mean",
            "camsol_intrinsic_min",
            "camsol_intrinsic_std",
            "camsol_intrinsic_frac_negative",
        }

    def test_min_lower_than_mean(self):
        r = AntibodyRecord(
            ab_id="Trastuzumab",
            source=SourceDataset.JAIN_2017,
            vh=TRASTUZUMAB_VH,
            vl=TRASTUZUMAB_VL,
        )
        res = compute_camsol_intrinsic(r)
        assert res.metrics["camsol_intrinsic_min"] <= res.metrics["camsol_intrinsic_mean"]

    def test_no_sequence_returns_unavailable(self):
        r = AntibodyRecord(ab_id="Ab1", source=SourceDataset.JAIN_2017)
        res = compute_camsol_intrinsic(r)
        assert not res.available
        assert "no VH/VL sequence" in res.notes
