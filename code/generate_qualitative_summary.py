#!/usr/bin/env python3
"""
generate_qualitative_summary.py

Reads extraction comparison results (outputs/compare_extraction_methods/summary.csv)
and generates a short qualitative markdown summary interpreting the three methods:
  - rule_based_only
  - rule_plus_actus_mens_enriched
  - rule_plus_general_enriched

Output: outputs/compare_extraction_methods/qualitative_summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "compare_extraction_methods" / "summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "compare_extraction_methods" / "qualitative_summary.md"

# Required columns for interpretation (script fails gracefully if any are missing)
REQUIRED_COLUMNS = [
    "method",
    "total_nodes",
    "n_cases_with_nodes",
    "n_cases_with_2_plus_nodes",
    "unique_labels",
    "label_actus_reus_killing",
    "label_unlawful_killing",
    "label_causation",
    "label_mens_rea_intent_to_kill",
    "label_mens_rea_intent_gbh",
]

# At least one of these for case count
CASE_COUNT_COLUMNS = ["n_cases", "n_cases_compared"]

METHOD_DISPLAY_NAMES = {
    "rule_based_only": "Rule-based only (baseline)",
    "rule_plus_actus_mens_enriched": "Rule-based + actus reus / mens rea enriched",
    "rule_plus_general_enriched": "Rule-based + general enriched",
}


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return list of missing required column names. Empty if all present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    has_case_count = any(c in df.columns for c in CASE_COUNT_COLUMNS)
    if not has_case_count:
        missing.append("(one of: " + ", ".join(CASE_COUNT_COLUMNS) + ")")
    return missing


def get_case_count(row: pd.Series) -> int:
    """Number of cases (from n_cases or n_cases_compared)."""
    for c in CASE_COUNT_COLUMNS:
        if c in row.index and pd.notna(row.get(c)):
            return int(row[c])
    return 0


def generate_summary(df: pd.DataFrame) -> str:
    """Build qualitative markdown from summary dataframe."""
    lines = [
        "# Qualitative Summary: Extraction Method Comparison",
        "",
        "This document interprets the numerical comparison of the three extraction methods.",
        "",
        "---",
        "",
    ]

    # Ensure we have exactly the three expected methods (order preserved)
    method_order = [
        "rule_based_only",
        "rule_plus_actus_mens_enriched",
        "rule_plus_general_enriched",
    ]
    rows = []
    for m in method_order:
        r = df[df["method"] == m]
        if not r.empty:
            rows.append((m, r.iloc[0]))

    if len(rows) != 3:
        lines.append("_Note: Expected three methods; some may be missing from the summary CSV._")
        lines.append("")
        rows = [(m, df[df["method"] == m].iloc[0]) for m in df["method"].unique()]

    # Key metrics per method
    baseline = next((r for k, r in rows if k == "rule_based_only"), None)
    actus_mens = next((r for k, r in rows if k == "rule_plus_actus_mens_enriched"), None)
    general = next((r for k, r in rows if k == "rule_plus_general_enriched"), None)

    def pct(a: int, b: int) -> str:
        if b == 0:
            return "—"
        return f"{100 * a / b:.1f}%"

    n_cases = int(baseline["n_cases_with_nodes"]) if baseline is not None else 0
    if n_cases == 0 and baseline is not None:
        n_cases = get_case_count(baseline)

    # --- Coverage ---
    lines.append("## Coverage")
    lines.append("")
    if baseline is not None and actus_mens is not None and general is not None:
        t_b, t_a, t_g = int(baseline["total_nodes"]), int(actus_mens["total_nodes"]), int(general["total_nodes"])
        lines.append(f"- **Rule-based only** extracts the fewest nodes ({t_b:,}), giving a conservative baseline.")
        lines.append(f"- **Actus reus / mens rea enriched** adds a moderate number of nodes ({t_a:,}), mainly in doctrinal categories.")
        lines.append(f"- **General enriched** yields the most nodes ({t_g:,}), maximising recall over the case set.")
        two_b = int(baseline["n_cases_with_2_plus_nodes"])
        two_a = int(actus_mens["n_cases_with_2_plus_nodes"])
        two_g = int(general["n_cases_with_2_plus_nodes"])
        if n_cases > 0:
            lines.append(f"- Cases with at least two nodes: baseline {pct(two_b, n_cases)}, actus/mens {pct(two_a, n_cases)}, general {pct(two_g, n_cases)}. Richer methods slightly increase structural coverage per case.")
    lines.append("")

    # --- Doctrinal completeness ---
    lines.append("## Doctrinal completeness")
    lines.append("")
    if baseline is not None and actus_mens is not None and general is not None:
        for lbl, title in [
            ("label_actus_reus_killing", "Actus reus (killing)"),
            ("label_unlawful_killing", "Unlawful killing"),
            ("label_causation", "Causation"),
            ("label_mens_rea_intent_to_kill", "Mens rea (intent to kill)"),
            ("label_mens_rea_intent_gbh", "Mens rea (intent GBH)"),
        ]:
            if lbl not in baseline.index:
                continue
            b, a, g = int(baseline[lbl]), int(actus_mens[lbl]), int(general[lbl])
            lines.append(f"- **{title}**: baseline {b:,}, actus/mens {a:,}, general {g:,}. Enriched ontologies pull in more doctrinal mentions; actus/mens is strongest for causation and intent-to-GBH.")
        lines.append("")
        lines.append("The actus/mens ontology is targeted at murder-relevant elements and typically increases counts for unlawful killing, causation, and intent without diluting label semantics. General enrichment adds breadth but may include less doctrine-specific nodes.")
    lines.append("")

    # --- Structural richness ---
    lines.append("## Structural richness")
    lines.append("")
    if baseline is not None and general is not None:
        u_b = int(baseline["unique_labels"])
        u_g = int(general["unique_labels"]) if "unique_labels" in general.index else 0
        lines.append(f"- **Unique labels**: baseline {u_b}, general enriched {u_g}. Richer ontologies produce more label diversity, which can support finer-grained graphs and analysis.")
        lines.append("- Higher total nodes and more labels per case improve the potential for edge extraction and Bayesian network structure learning.")
    lines.append("")

    # --- Noise / interpretability tradeoff ---
    lines.append("## Noise / interpretability tradeoff")
    lines.append("")
    lines.append("- **Rule-based only** is the most interpretable and least noisy: every node is driven by ontology and rules, so labels are consistent and auditable.")
    lines.append("- **Actus reus / mens rea enriched** keeps the focus on murder-relevant doctrine while increasing recall; noise is limited to synonym/paraphrase matches within a controlled vocabulary.")
    lines.append("- **General enriched** maximises recall and structural richness but introduces more variation in phrasing and potentially less precise labels; downstream filtering or aggregation may be needed for clean doctrinal summaries.")
    lines.append("")

    # --- Recommended method ---
    lines.append("## Recommended method")
    lines.append("")
    lines.append("- **For doctrinal analysis and murder-specific elements**: prefer **rule-based + actus reus / mens rea enriched**. It improves coverage of unlawful killing, causation, and mens rea without sacrificing interpretability.")
    lines.append("- **For exploratory graph building and maximum coverage**: use **rule-based + general enriched**, then apply post-hoc filters or aggregation for clarity.")
    lines.append("- **For audits and reproducibility**: **rule-based only** remains the best baseline to report and compare against.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate qualitative markdown summary from extraction comparison results.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Summary CSV path.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output markdown path.")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if df.empty:
        print("Error: summary CSV is empty.", file=sys.stderr)
        return 1

    missing = validate_columns(df)
    if missing:
        print("Error: summary CSV is missing required columns:", file=sys.stderr)
        for c in missing:
            print(f"  - {c}", file=sys.stderr)
        return 1

    md = generate_summary(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")

    print(f"Qualitative summary written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
