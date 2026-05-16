"""Developability Index (DI) — sequence-proxy implementation.

Reference:
    Lauer, Sathish, Ramachandra, Agaskar, Ramagubathan, Cain, Smiley, Singh,
    Stewart, Lang, Tannenbaum, Ulmer, Wang, Bjelke, Sharma, Bos, Li, Prahlada,
    Bhowmik, Yu, Szczechura, Schultz, Baroni, Li, Pietrzak, Keating, Boyce,
    Krishnamurthy, Ma, Ishii, Gao, Patel, DeFlavio, Ramos, Ye, McGowan,
    Mennie, Amin, Cohen, Tabrizi, Chaparro-Riggers, Berezovskaya, Giessmann,
    Pollard, Chwistek, Wray, Singh, Jelkmann, Szymanski, Kantonen, Piazza,
    Smith, Kellison, Martella, Gagnon, Smith, Goose, Bhattacharyya, Chaitanya,
    Nair, Krishnamurthi, Matta, Pasupulati, Dumanli, Wang, Shergill, Zhang,
    Gajewska, Lanza, Gupta, Thayumanavan, Holster, Hovland, Arvelo, Carino,
    Donaldson, Lane, Tong, Stewart, Zhang, Russ, Bieber, Stoll, Lasser.
    "Developability Index: A Rapid In Silico Tool for the Screening of
    Antibody Aggregation Propensity." J. Pharm. Sci. 101 (2012): 102-115.
    DOI: 10.1002/jps.22758

DI = β · SAP  −  SFvCSP

where SAP (Spatial Aggregation Propensity; Chennamsetty 2009) is a
surface-exposed hydrophobicity patch score and SFvCSP is the product of
net VH and VL charges at pH 6.0 (negative → asymmetric → favored).

Both SAP and SFvCSP in the original formulation require a 3D structure.
In Phase 0 (sequence only), we compute:

    di_seq_proxy               — −SFvCSP_seq + β·hydrophobicity_mean
    di_sfvcsp_seq              — VH_charge_pH6 * VL_charge_pH6 (no structure)
    di_mean_hydrophobicity     — mean Kyte-Doolittle over VH+VL as SAP proxy
    di_hydro_patch_length      — longest run of hydrophobicity > +1.0 (SAP proxy)

When the Phase 3 structure analyzer is available, DI can be upgraded to
the full SAP-based version.
"""

from __future__ import annotations

from ab_benchmark.baselines.base import BaselineResult, ok, unavailable
from ab_benchmark.schema import AntibodyRecord
from ab_benchmark.seqprops import KYTE_DOOLITTLE, net_charge_at_ph

VERSION = "0.1.0-seq-proxy"

# Lauer 2012 uses β=0.815 for the SAP coefficient. Kept verbatim even
# though our SAP proxy is sequence-based (so β does not literally apply).
_BETA = 0.815


def compute_developability_index(record: AntibodyRecord) -> BaselineResult:
    if not record.vh or not record.vl:
        return unavailable(
            "developability_index", VERSION, record, "DI requires both VH and VL sequences"
        )

    vh_charge_ph6 = net_charge_at_ph(record.vh, 6.0)
    vl_charge_ph6 = net_charge_at_ph(record.vl, 6.0)
    sfvcsp_seq = vh_charge_ph6 * vl_charge_ph6

    fv = record.vh + record.vl
    hydro_vals = [KYTE_DOOLITTLE.get(a, 0.0) for a in fv]
    mean_hydro = sum(hydro_vals) / len(hydro_vals) if hydro_vals else 0.0

    # Longest run of hydrophobicity > 1.0 as a simple aggregation-prone
    # "patch length" proxy (rough SAP replacement).
    patch_length = _longest_run_above(hydro_vals, threshold=1.0)

    di_seq_proxy = _BETA * mean_hydro - sfvcsp_seq

    metrics = {
        "di_sfvcsp_seq": sfvcsp_seq,
        "di_vh_charge_ph6": vh_charge_ph6,
        "di_vl_charge_ph6": vl_charge_ph6,
        "di_mean_hydrophobicity": mean_hydro,
        "di_hydro_patch_length": float(patch_length),
        "di_seq_proxy": di_seq_proxy,
    }
    notes = "sequence-proxy; full SAP requires Phase 3 structure (Chennamsetty 2009)"
    return ok("developability_index", VERSION, record, metrics, notes)


def _longest_run_above(xs: list[float], threshold: float) -> int:
    best = cur = 0
    for x in xs:
        if x > threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
