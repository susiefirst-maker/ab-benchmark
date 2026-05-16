"""Run every baseline over the harmonized dataset.

Usage:

    python -m ab_benchmark.baselines.run_all \\
      --config configs/datasets.yaml \\
      --out reports/baseline_results.csv

Each baseline is called once per record loaded from the configured sources.
Output is a long-format CSV:

    ab_id, ab_id_canonical, source, baseline, version, available, metric, value, notes

Plus a JSON summary at `{out}.summary.json` with per-baseline coverage
counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ab_benchmark.baselines import BASELINE_REGISTRY
from ab_benchmark.data.build_harmonized import _expand, _load_config
from ab_benchmark.data.loaders import LOADER_REGISTRY
from ab_benchmark.schema import AntibodyRecord, SourceDataset


def _load_all_records(config_path: Path) -> list[AntibodyRecord]:
    config = _load_config(config_path)
    records: list[AntibodyRecord] = []
    for src_key, cfg in config.get("sources", {}).items():
        try:
            source = SourceDataset(src_key)
        except ValueError:
            continue
        if not cfg.get("enabled", False):
            continue
        path = _expand(cfg.get("path"))
        if not path:
            continue
        loader = LOADER_REGISTRY[source]
        try:
            records.extend(loader(path))
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"[warn] {src_key}: {e}", file=sys.stderr)
            continue
    return records


def run(config_path: Path, out_path: Path) -> dict:
    records = _load_all_records(config_path)
    if not records:
        raise RuntimeError("No records loaded; check configs/datasets.yaml.")

    rows: list[dict] = []
    coverage: dict[str, dict[str, int]] = {}

    for baseline_name, fn in BASELINE_REGISTRY.items():
        n_ok = 0
        n_unavail = 0
        first_unavail_reason = ""

        for r in records:
            result = fn(r)
            if result.available:
                n_ok += 1
                for metric, value in result.metrics.items():
                    rows.append({
                        "ab_id": result.ab_id,
                        "ab_id_canonical": result.ab_id_canonical,
                        "source": r.source.value,
                        "baseline": result.baseline,
                        "version": result.version,
                        "available": True,
                        "metric": metric,
                        "value": value,
                        "notes": result.notes,
                    })
            else:
                n_unavail += 1
                if not first_unavail_reason:
                    first_unavail_reason = result.notes
                rows.append({
                    "ab_id": result.ab_id,
                    "ab_id_canonical": result.ab_id_canonical,
                    "source": r.source.value,
                    "baseline": result.baseline,
                    "version": result.version,
                    "available": False,
                    "metric": "",
                    "value": float("nan"),
                    "notes": result.notes,
                })
        coverage[baseline_name] = {
            "ok": n_ok,
            "unavailable": n_unavail,
            "total": len(records),
            "first_unavail_reason": first_unavail_reason,
        }

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = {
        "n_records": len(records),
        "coverage": coverage,
        "output_csv": str(out_path),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def _print(summary: dict) -> None:
    print("=" * 60)
    print("Baseline coverage")
    print("=" * 60)
    print(f"Records: {summary['n_records']}")
    print()
    print(f"{'baseline':24s}  {'ok':>5s}  {'unavail':>8s}  reason")
    for name, c in summary["coverage"].items():
        reason = c["first_unavail_reason"][:60]
        print(f"  {name:22s}  {c['ok']:5d}  {c['unavailable']:8d}  {reason}")
    print()
    print(f"Written:  {summary['output_csv']}")
    print(f"          {summary['summary_path']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    ap.add_argument("--out", type=Path, default=Path("reports/baseline_results.csv"))
    args = ap.parse_args(argv)

    summary = run(args.config, args.out)
    _print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
