#!/usr/bin/env python3
"""
build_case_node_matrix.py

Build a binary case×DAG-node matrix using:
- extraction output (nodes CSV with columns including: case_id, label)
- mapping table (CSV with: extracted_label, dag_node, confidence, notes)

Matrix requirements:
- Rows = case_id
- Columns = DAG node names (from design_bn_dag.py)
- Cell = 1 if at least one mapped extracted label for that DAG node appears in that case, else 0
- Final column: Verdict encoded as Murder=2, Manslaughter=1, Not Guilty=0

Verdict source:
By default, Verdict is inferred from extracted labels mapped to the DAG node "Verdict":
  - If a case has any label in {"murder"} -> 2
  - Else if any label starting with "manslaughter_" -> 1
  - Else if label == "lawful_killing_self_defence" -> 0
  - Else -> NaN (unknown) and reported

Outputs:
1) case_node_matrix.csv (repo root)
2) outputs/case_node_matrix_summary.txt
3) outputs/case_node_cooccurrence_heatmap.png
4) outputs/structurally_empty_cases.txt

Run:
  python code/build_case_node_matrix.py
  python code/build_case_node_matrix.py --nodes data/processed/nodes_actus_mens.csv --mapping outputs/label_to_dag_mapping.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NODES = ROOT / "data" / "processed" / "nodes_actus_mens.csv"
DEFAULT_MAPPING = ROOT / "outputs" / "label_to_dag_mapping.csv"

OUT_MATRIX = ROOT / "case_node_matrix.csv"
OUT_SUMMARY = ROOT / "outputs" / "case_node_matrix_summary.txt"
OUT_HEATMAP = ROOT / "outputs" / "case_node_cooccurrence_heatmap.png"
OUT_EMPTY = ROOT / "outputs" / "structurally_empty_cases.txt"


def load_dag_nodes() -> list[str]:
    # Single source of truth: the DAG node list used in design_bn_dag.py (kept in sync manually).
    return [
        "VoluntaryAct",
        "DeathOfHumanBeing",
        "FactualCausation",
        "LegalCausation",
        "UnlawfulKilling",
        "IntentToKill",
        "IntentToCauseGBH",
        "MensReaIntent",
        "MaliceAforethought",
        "LossOfControl",
        "DiminishedResponsibility",
        "Defences",
        "Verdict",
    ]


def infer_verdict_from_labels(labels: set[str]) -> float:
    """
    Infer verdict from extracted labels (not DAG nodes).
    Returns 2 (Murder), 1 (Manslaughter), 0 (Not Guilty), or np.nan (unknown).
    """
    if "murder" in labels:
        return 2.0
    if any(lab.startswith("manslaughter_") for lab in labels):
        return 1.0
    if "lawful_killing_self_defence" in labels:
        return 0.0
    return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="Build binary case×DAG-node matrix from extraction + mapping table.")
    p.add_argument("--nodes", type=Path, default=DEFAULT_NODES, help="Extraction nodes CSV (must contain case_id,label).")
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="Mapping CSV (extracted_label -> dag_node).")
    args = p.parse_args()

    if not args.nodes.exists():
        raise FileNotFoundError(f"Nodes CSV not found: {args.nodes}")
    if not args.mapping.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping}")

    nodes = pd.read_csv(args.nodes, encoding="utf-8-sig", usecols=["case_id", "label"])
    mapping = pd.read_csv(args.mapping, encoding="utf-8-sig")

    required_map_cols = {"extracted_label", "dag_node", "confidence", "notes"}
    missing = sorted(required_map_cols - set(mapping.columns))
    if missing:
        raise ValueError(f"Mapping CSV missing columns: {missing}")

    # Keep only mapped labels
    mapping = mapping[mapping["dag_node"].notna()].copy()
    mapping["extracted_label"] = mapping["extracted_label"].astype(str).str.strip()
    mapping["dag_node"] = mapping["dag_node"].astype(str).str.strip()

    label_to_node = dict(zip(mapping["extracted_label"], mapping["dag_node"]))
    nodes["label"] = nodes["label"].astype(str).str.strip()

    # Map extracted labels to DAG nodes (UNMAPPED are ignored for matrix construction)
    nodes["dag_node"] = nodes["label"].map(label_to_node)
    nodes_mapped = nodes[(nodes["dag_node"].notna()) & (nodes["dag_node"] != "UNMAPPED")].copy()

    dag_nodes = load_dag_nodes()
    dag_cols = [n for n in dag_nodes if n != "Verdict"]

    # Binary matrix: case_id × dag_node (excluding Verdict)
    present = (
        nodes_mapped[nodes_mapped["dag_node"].isin(dag_cols)]
        .drop_duplicates(subset=["case_id", "dag_node"])
        .assign(value=1)
        .pivot(index="case_id", columns="dag_node", values="value")
        .fillna(0)
        .astype(int)
    )

    # Ensure all expected columns exist (even if no hits)
    for col in dag_cols:
        if col not in present.columns:
            present[col] = 0
    present = present[dag_cols]

    # Verdict per case (inferred from extracted labels)
    labels_by_case = nodes.groupby("case_id")["label"].apply(lambda s: set(s.dropna().astype(str))).to_dict()
    verdict_series = pd.Series({cid: infer_verdict_from_labels(labs) for cid, labs in labels_by_case.items()}, name="Verdict")

    # Align verdict with matrix index; include cases that appear in nodes but have no mapped nodes
    all_cases = sorted(set(nodes["case_id"].astype(str)))
    present = present.reindex(all_cases, fill_value=0)
    verdict_series = verdict_series.reindex(all_cases)

    matrix = present.copy()
    matrix["Verdict"] = verdict_series

    # Save matrix
    matrix.to_csv(OUT_MATRIX, index=True, encoding="utf-8")

    # Summary stats
    n_cases = len(matrix)
    prevalence = (matrix[dag_cols].sum(axis=0) / max(1, n_cases) * 100.0).sort_values(ascending=False)
    verdict_dist = matrix["Verdict"].value_counts(dropna=False).sort_index()

    # Co-occurrence heatmap (binary)
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        co = (matrix[dag_cols].T @ matrix[dag_cols]).astype(int)
        plt.figure(figsize=(12, 10))
        sns.heatmap(co, cmap="Blues", square=True)
        plt.title("Node co-occurrence (counts across cases)")
        OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT_HEATMAP, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        # Heatmap is optional; stats + matrix still produced
        OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
        OUT_HEATMAP.write_text("Heatmap not generated (missing seaborn/matplotlib).", encoding="utf-8")

    # Structurally empty cases: all doctrinal nodes = 0 (exclude Verdict column entirely)
    empty_mask = matrix[dag_cols].sum(axis=1) == 0
    empty_cases = matrix.index[empty_mask].tolist()
    OUT_EMPTY.parent.mkdir(parents=True, exist_ok=True)
    OUT_EMPTY.write_text("\n".join(empty_cases) + ("\n" if empty_cases else ""), encoding="utf-8")

    # Write summary file
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"Input nodes CSV: {args.nodes}")
    lines.append(f"Input mapping CSV: {args.mapping}")
    lines.append(f"Cases: {n_cases}")
    lines.append("")
    lines.append("Node prevalence (% of cases with node=1):")
    for node, pct in prevalence.items():
        lines.append(f"  {node}: {pct:.1f}%")
    lines.append("")
    lines.append("Verdict distribution (encoded: Murder=2, Manslaughter=1, Not Guilty=0; NaN=unknown):")
    for k, v in verdict_dist.items():
        lines.append(f"  {k}: {int(v)}")
    unknown_n = int(matrix["Verdict"].isna().sum())
    lines.append(f"Unknown verdicts (NaN): {unknown_n}")
    lines.append("")
    lines.append(f"Structurally empty cases (all doctrinal nodes 0): {len(empty_cases)}")
    text = "\n".join(lines) + "\n"
    OUT_SUMMARY.write_text(text, encoding="utf-8")

    print(f"Matrix written to: {OUT_MATRIX}")
    print(f"Summary written to: {OUT_SUMMARY}")
    print(f"Heatmap written to: {OUT_HEATMAP}")
    print(f"Structurally empty cases written to: {OUT_EMPTY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
