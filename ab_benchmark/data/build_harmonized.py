"""Build the harmonized antibody developability dataset.

Usage:

    python -m ab_benchmark.data.build_harmonized \\
      --config configs/datasets.yaml \\
      --out data/processed/harmonized_antibody_dev.parquet

Reads the dataset config, calls each enabled loader, harmonizes into the
canonical long-format schema, writes parquet + a CSV summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from ab_benchmark.data.harmonize import (
    harmonization_summary,
    records_to_long_df,
    to_wide,
)
from ab_benchmark.data.loaders import LOADER_REGISTRY
from ab_benchmark.schema import SourceDataset


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.expandvars(os.path.expanduser(path))


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build(config_path: Path, out_path: Path, wide_out_path: Path | None = None) -> dict:
    """Run the harmonization pipeline. Returns the summary dict.

    Writes:
      out_path          long-format parquet
      wide_out_path     wide-format parquet (optional)
      {out_path}.summary.json   harmonization summary
    """
    config = _load_config(config_path)
    sources = config.get("sources", {})

    all_records = []
    per_source_info = {}

    for src_key, cfg in sources.items():
        try:
            source = SourceDataset(src_key)
        except ValueError:
            print(f"[warn] unknown source key {src_key!r} — skipped", file=sys.stderr)
            continue

        if not cfg.get("enabled", False):
            per_source_info[src_key] = {"status": "disabled", "n_records": 0}
            continue

        path = _expand(cfg.get("path"))
        if not path:
            per_source_info[src_key] = {"status": "no-path", "n_records": 0}
            continue

        loader = LOADER_REGISTRY[source]
        try:
            records = loader(path)
        except NotImplementedError as e:
            per_source_info[src_key] = {"status": "not-implemented", "note": str(e), "n_records": 0}
            continue
        except FileNotFoundError as e:
            per_source_info[src_key] = {"status": "file-missing", "note": str(e), "n_records": 0}
            continue

        per_source_info[src_key] = {"status": "ok", "n_records": len(records), "path": path}
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No records loaded from any enabled source. Check configs/datasets.yaml."
        )

    long_df = records_to_long_df(all_records)
    summary = harmonization_summary(long_df)
    summary["per_source"] = per_source_info

    # Write parquet (long).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_path, index=False)

    # Write wide parquet alongside if requested.
    if wide_out_path:
        wide_df = to_wide(long_df)
        wide_out_path.parent.mkdir(parents=True, exist_ok=True)
        wide_df.to_parquet(wide_out_path, index=False)
        summary["wide_rows"] = len(wide_df)
        summary["wide_path"] = str(wide_out_path)

    # Write summary JSON sidecar.
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    summary["long_rows"] = len(long_df)
    summary["long_path"] = str(out_path)
    summary["summary_path"] = str(summary_path)
    return summary


def _print_summary(summary: dict) -> None:
    print("=" * 60)
    print("Harmonization summary")
    print("=" * 60)
    print(f"Long rows:       {summary.get('long_rows', '?')}")
    if "wide_rows" in summary:
        print(f"Wide rows:       {summary['wide_rows']}")
    print(f"Total unique antibodies (canonical): {summary['total_unique_antibodies']}")
    print(f"Antibodies in ≥2 sources:            {summary['antibodies_in_multiple_sources']}")
    print()
    print("Per-source status:")
    for src, info in summary.get("per_source", {}).items():
        status = info.get("status", "?")
        n = info.get("n_records", 0)
        print(f"  {src:18s}  {status:18s}  n={n}")
    print()
    print("Measurements per endpoint:")
    for ep, n in sorted(summary.get("measurements_by_endpoint", {}).items()):
        print(f"  {ep:20s}  n={n}")
    print()
    print(f"Written:  {summary.get('long_path')}")
    if "wide_path" in summary:
        print(f"          {summary['wide_path']}")
    print(f"          {summary.get('summary_path')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"),
                    help="YAML dataset config (default: configs/datasets.yaml).")
    ap.add_argument("--out", type=Path, default=Path("data/processed/harmonized_antibody_dev.parquet"),
                    help="Output path for the long-format parquet.")
    ap.add_argument("--wide-out", type=Path, default=Path("data/processed/harmonized_antibody_dev_wide.parquet"),
                    help="Output path for the wide-format parquet (set empty to skip).")
    args = ap.parse_args(argv)

    wide_out = args.wide_out if str(args.wide_out) else None
    summary = build(args.config, args.out, wide_out_path=wide_out)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
