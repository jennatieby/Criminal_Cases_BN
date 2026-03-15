#!/usr/bin/env python3
"""
compare_extraction_methods.py

Compare three extraction methods in one summary:
  1) Rule-based only (baseline ontology)
  2) Rule-based + actus reus / mens rea enrichment
  3) Rule-based + general enrichment

Uses the 947-case cleaned dataset (data/interim/uk_cases_full.cleaned.csv) by default
when running extraction (--run-extraction).

Usage:
  If you already ran extraction three times with the three ontologies:
    python code/compare_extraction_methods.py

  To run extraction for all three then compare (same input CSV for each):
    python code/compare_extraction_methods.py --run-extraction

  Custom node CSV paths:
    python code/compare_extraction_methods.py --baseline data/processed/nodes_baseline.csv --actus-mens data/processed/nodes_actus_mens.csv --general data/processed/nodes_general_enriched.csv

Output:
  outputs/compare_extraction_methods/summary.csv   (one row per method, columns = metrics)
  Printed table to stdout
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "compare_extraction_methods"

# Default node CSVs produced by running extraction with each ontology
DEFAULT_BASELINE = DATA / "nodes_baseline.csv"
DEFAULT_ACTUS_MENS = DATA / "nodes_actus_mens.csv"
DEFAULT_GENERAL = DATA / "nodes_general_enriched.csv"

# Actus reus + mens rea labels (for focused comparison)
ACTUS_MENS_LABELS = [
    "actus_reus_killing",
    "unlawful_killing",
    "causation",
    "mens_rea_intent_to_kill",
    "mens_rea_intent_gbh",
]


def run_extraction(input_csv: Path, ontology_path: Path | None, output_csv: Path) -> bool:
    cmd = [
        sys.executable,
        str(ROOT / "code" / "extract_nodes_from_cases.py"),
        "--input", str(input_csv),
        "--output-csv", str(output_csv),
    ]
    if ontology_path and ontology_path.exists():
        cmd.extend(["--ontology", str(ontology_path)])
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


def summarize_nodes(path: Path, case_ids_subset: set | None = None) -> dict:
    """Compute metrics for a node CSV. If case_ids_subset is set, only include those cases (for like-for-like comparison)."""
    if not path.exists():
        return {"method": path.stem, "error": "file not found"}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return {"method": path.stem, "total_nodes": 0, "n_cases": 0, "n_cases_with_nodes": 0, "n_cases_with_2_plus_nodes": 0}

    if case_ids_subset is not None and "case_id" in df.columns:
        df = df[df["case_id"].astype(str).isin({str(c) for c in case_ids_subset})].copy()

    n_cases = df["case_id"].nunique() if "case_id" in df.columns else 0
    counts = df.groupby("case_id")["node_id"].nunique() if "case_id" in df.columns and "node_id" in df.columns else pd.Series(dtype=int)
    n_cases_with_nodes = counts.size
    n_cases_with_2_plus_nodes = int((counts >= 2).sum()) if counts.size else 0

    out = {
        "method": path.stem,
        "total_nodes": len(df),
        "n_cases": n_cases,
        "n_cases_with_nodes": n_cases_with_nodes,
        "n_cases_with_2_plus_nodes": n_cases_with_2_plus_nodes,
    }

    if "label" in df.columns:
        for lbl in ACTUS_MENS_LABELS:
            out[f"label_{lbl}"] = int((df["label"] == lbl).sum())
        out["unique_labels"] = df["label"].nunique()

    return out


def load_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty or "case_id" not in df.columns:
        return set()
    return set(df["case_id"].astype(str).unique())


def main():
    p = argparse.ArgumentParser(description="Compare three extraction methods in one summary.")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="Node CSV for rule-based only.")
    p.add_argument("--actus-mens", type=Path, default=DEFAULT_ACTUS_MENS, help="Node CSV for actus/mens enriched.")
    p.add_argument("--general", type=Path, default=DEFAULT_GENERAL, help="Node CSV for general enriched.")
    p.add_argument("--output", type=Path, default=OUT_DIR / "summary.csv", help="Output summary CSV.")
    p.add_argument(
        "--run-extraction",
        action="store_true",
        help="Run extract_nodes_from_cases for all three methods first (same input CSV).",
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv",
        help="Input case CSV when using --run-extraction (default: 947-case cleaned dataset).",
    )
    args = p.parse_args()

    if args.run_extraction:
        input_csv = args.input_csv.resolve()
        if not input_csv.exists():
            print(f"Input CSV not found: {input_csv}")
            sys.exit(1)
        input_df = pd.read_csv(input_csv, encoding="utf-8-sig")
        n_input_cases = len(input_df)
        print(f"Input dataset: {input_csv.name} ({n_input_cases} cases)")
        onto_dir = ROOT / "rules"
        print("Running extraction: rule-based only ...")
        if not run_extraction(input_csv, None, args.baseline.resolve()):
            print("Baseline extraction failed.")
            sys.exit(1)
        print("Running extraction: actus/mens enriched ...")
        if not run_extraction(input_csv, onto_dir / "ontology_actus_mens_enriched.yml", args.actus_mens.resolve()):
            print("Actus/mens extraction failed.")
            sys.exit(1)
        print("Running extraction: general enriched ...")
        if not run_extraction(input_csv, onto_dir / "ontology_general_enriched.yml", args.general.resolve()):
            print("General enriched extraction failed.")
            sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    methods = [
        ("rule_based_only", args.baseline),
        ("rule_plus_actus_mens_enriched", args.actus_mens),
        ("rule_plus_general_enriched", args.general),
    ]

    case_ids_per_method = {name: load_case_ids(path) for name, path in methods}
    common_case_ids = set()
    if case_ids_per_method:
        common_case_ids = set.intersection(*case_ids_per_method.values())

    n_per_method = {name: len(cids) for name, cids in case_ids_per_method.items()}
    if len(set(n_per_method.values())) > 1:
        print("Case count differs across method outputs:", n_per_method)
        print(f"Comparing on {len(common_case_ids)} cases present in all three.")
    elif common_case_ids:
        print(f"All three outputs share the same case set: {len(common_case_ids)} cases.")

    rows = []
    for name, path in methods:
        row = summarize_nodes(path, case_ids_subset=common_case_ids if common_case_ids else None)
        row["method"] = name
        if common_case_ids:
            row["n_cases_compared"] = len(common_case_ids)
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Reorder columns: method, dataset size, totals, then actus/mens label counts
    base_cols = ["method", "n_cases_compared", "total_nodes", "n_cases", "n_cases_with_nodes", "n_cases_with_2_plus_nodes"]
    base_cols = [c for c in base_cols if c in summary.columns]
    if "unique_labels" in summary.columns:
        base_cols.append("unique_labels")
    label_cols = [c for c in summary.columns if c.startswith("label_") and c in summary.columns]
    summary = summary[[c for c in base_cols + label_cols if c in summary.columns]]

    summary.to_csv(args.output, index=False)
    print(f"Summary saved -> {args.output}")
    print()
    if common_case_ids:
        print(f"Comparing on {len(common_case_ids)} cases (present in all three method outputs).")
    else:
        print("No common case set (one or more node files missing or empty).")
    print("(n_cases_with_2_plus_nodes = cases with >=2 nodes, i.e. edge-bearing)")
    print()
    print(summary.to_string(index=False))
    print()
    print("Methods: rule_based_only | rule_plus_actus_mens_enriched | rule_plus_general_enriched")


if __name__ == "__main__":
    main()
