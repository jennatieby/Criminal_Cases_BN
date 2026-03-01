#!/usr/bin/env python3
"""
generate_quality_report.py — Data quality assessment before parameterising the BN

Produces CSVs you can open in Excel to inspect:
  • Nodes and edges per case
  • Which ontology labels / BN template edges are present vs missing per case
  • Global stats (coverage %, never-seen labels/edges)

Inputs:
  data/processed/nodes.csv
  data/processed/edges.csv
  outputs/bn_edges_aggregated.csv (BN template)
  rules/ontology.yml (canonical labels)

Outputs (in outputs/quality/):
  case_summary.csv           — one row per case: counts, coverage %
  label_coverage.csv         — long: case_id, label, count (pivot in Excel for case×label)
  label_matrix_wide.csv      — wide: case_id × label with counts (direct matrix)
  edge_coverage.csv          — long: case_id, src_label, dst_label, count
  missing_labels_per_case.csv — case_id, label (ontology labels missing in that case)
  missing_edges_per_case.csv   — case_id, src_label, dst_label (template edges missing)
  global_label_stats.csv     — label, cases_with, pct, total_occurrences
  global_edge_stats.csv      — src, dst, cases_with, pct
  never_seen_labels.csv      — labels in ontology that appear in 0 cases
  never_seen_edges.csv       — template edges that appear in 0 cases

Run from repo root:
  python code/generate_quality_report.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
QUALITY = OUTPUTS / "quality"
RULES = ROOT / "rules"

NODES_CSV = DATA / "nodes.csv"
EDGES_CSV = DATA / "edges.csv"
BN_EDGES_CSV = OUTPUTS / "bn_edges_aggregated.csv"
ONTO_PATH = RULES / "ontology.yml"

QUALITY.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not NODES_CSV.exists():
        raise FileNotFoundError(f"Missing {NODES_CSV} — run extract_nodes_from_cases.py first.")
    if not EDGES_CSV.exists():
        raise FileNotFoundError(f"Missing {EDGES_CSV} — run build_edges_between_nodes.py first.")

    nodes = pd.read_csv(NODES_CSV)
    edges = pd.read_csv(EDGES_CSV)

    # Resolve node_id -> label
    id2label = dict(zip(nodes["node_id"], nodes["label"]))

    # Label coverage per case (long)
    label_counts = (
        nodes.groupby(["case_id", "label", "type"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    # Edge coverage at label level: map node edges to (case, src_label, dst_label)
    edges_with_labels = edges.copy()
    edges_with_labels["src_label"] = edges_with_labels["src_node_id"].map(id2label)
    edges_with_labels["dst_label"] = edges_with_labels["dst_node_id"].map(id2label)
    edges_labeled = edges_with_labels.dropna(subset=["src_label", "dst_label"])

    edge_counts = (
        edges_labeled.groupby(["case_id", "src_label", "dst_label"])
        .agg(count=("score", "count"))
        .reset_index()
    )

    # Reference: expected labels (ontology) and expected edges (BN template)
    if ONTO_PATH.exists():
        with open(ONTO_PATH, "r", encoding="utf-8") as f:
            onto = yaml.safe_load(f) or {}
        all_labels = list(onto.get("labels", {}).keys())
    else:
        all_labels = sorted(set(nodes["label"].dropna().unique()) | set(label_counts["label"].unique()))

    if BN_EDGES_CSV.exists():
        bn_edges = pd.read_csv(BN_EDGES_CSV)
        template_edges = set(
            zip(bn_edges["src_label"], bn_edges["dst_label"])
        )
    else:
        template_edges = set(
            zip(edge_counts["src_label"], edge_counts["dst_label"])
        )

    cases = sorted(nodes["case_id"].unique())
    n_cases = len(cases)

    # ---- 1. case_summary.csv ----
    case_nodes = nodes.groupby("case_id").agg(
        num_nodes=("node_id", "nunique"),
        num_unique_labels=("label", "nunique"),
    ).reset_index()
    case_edges = edges.groupby("case_id").size().reset_index(name="num_edges")
    case_edge_labels = edge_counts.groupby("case_id").size().reset_index(name="num_label_edges")
    case_summary = (
        case_nodes
        .merge(case_edges, on="case_id", how="left")
        .merge(case_edge_labels, on="case_id", how="left")
    )
    case_summary["num_edges"] = case_summary["num_edges"].fillna(0).astype(int)
    case_summary["num_label_edges"] = case_summary["num_label_edges"].fillna(0).astype(int)
    case_summary["labels_present_pct"] = (
        case_summary["num_unique_labels"] / len(all_labels) * 100
    ).round(1)
    case_summary.to_csv(QUALITY / "case_summary.csv", index=False)
    print(f"  → {QUALITY.name}/case_summary.csv")

    # ---- 2. label_coverage.csv (long) ----
    label_counts.to_csv(QUALITY / "label_coverage.csv", index=False)
    print(f"  → {QUALITY.name}/label_coverage.csv")

    # ---- 3. label_matrix_wide.csv ----
    pivot = label_counts.pivot_table(
        index="case_id", columns="label", values="count", fill_value=0
    )
    # Ensure all expected labels appear as columns (0 if missing)
    for lbl in all_labels:
        if lbl not in pivot.columns:
            pivot[lbl] = 0
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    pivot.to_csv(QUALITY / "label_matrix_wide.csv")
    print(f"  → {QUALITY.name}/label_matrix_wide.csv")

    # ---- 4. edge_coverage.csv ----
    edge_counts.to_csv(QUALITY / "edge_coverage.csv", index=False)
    print(f"  → {QUALITY.name}/edge_coverage.csv")

    # ---- 5. missing_labels_per_case.csv ----
    labels_per_case = set(zip(label_counts["case_id"], label_counts["label"]))
    missing_labels = []
    for cid in cases:
        for lbl in all_labels:
            if (cid, lbl) not in labels_per_case:
                missing_labels.append({"case_id": cid, "label": lbl})
    if missing_labels:
        pd.DataFrame(missing_labels).to_csv(
            QUALITY / "missing_labels_per_case.csv", index=False
        )
        print(f"  → {QUALITY.name}/missing_labels_per_case.csv ({len(missing_labels)} rows)")
    else:
        print(f"  → {QUALITY.name}/missing_labels_per_case.csv (none missing)")

    # ---- 6. missing_edges_per_case.csv ----
    edges_per_case = set(
        zip(
            edge_counts["case_id"],
            edge_counts["src_label"],
            edge_counts["dst_label"],
        )
    )
    missing_edges = []
    for cid in cases:
        for (src, dst) in template_edges:
            if (cid, src, dst) not in edges_per_case:
                missing_edges.append({
                    "case_id": cid,
                    "src_label": src,
                    "dst_label": dst,
                })
    if missing_edges:
        pd.DataFrame(missing_edges).to_csv(
            QUALITY / "missing_edges_per_case.csv", index=False
        )
        print(f"  → {QUALITY.name}/missing_edges_per_case.csv ({len(missing_edges)} rows)")
    else:
        print(f"  → {QUALITY.name}/missing_edges_per_case.csv (none missing)")

    # ---- 7. global_label_stats.csv ----
    glb = (
        label_counts.groupby("label", as_index=False)
        .agg(
            total_occurrences=("count", "sum"),
            cases_with=("case_id", "nunique"),
        )
    )
    glb["pct_cases"] = (glb["cases_with"] / n_cases * 100).round(1)
    glb = glb.sort_values("cases_with", ascending=False)
    glb.to_csv(QUALITY / "global_label_stats.csv", index=False)
    print(f"  → {QUALITY.name}/global_label_stats.csv")

    # ---- 8. global_edge_stats.csv ----
    geb = (
        edge_counts.groupby(["src_label", "dst_label"], as_index=False)
        .agg(cases_with=("case_id", "nunique"))
    )
    geb["pct_cases"] = (geb["cases_with"] / n_cases * 100).round(1)
    geb = geb.sort_values("cases_with", ascending=False)
    geb.to_csv(QUALITY / "global_edge_stats.csv", index=False)
    print(f"  → {QUALITY.name}/global_edge_stats.csv")

    # ---- 9. never_seen_labels.csv ----
    seen_labels = set(label_counts["label"].unique())
    never_seen = [lbl for lbl in all_labels if lbl not in seen_labels]
    if never_seen:
        pd.DataFrame({"label": never_seen}).to_csv(
            QUALITY / "never_seen_labels.csv", index=False
        )
        print(f"  → {QUALITY.name}/never_seen_labels.csv ({len(never_seen)} labels)")
    else:
        print(f"  → {QUALITY.name}/never_seen_labels.csv (all labels seen)")

    # ---- 10. never_seen_edges.csv ----
    seen_edges = set(
        zip(edge_counts["src_label"], edge_counts["dst_label"])
    )
    never_seen_edges = [(s, d) for (s, d) in template_edges if (s, d) not in seen_edges]
    if never_seen_edges:
        pd.DataFrame(
            never_seen_edges,
            columns=["src_label", "dst_label"],
        ).to_csv(QUALITY / "never_seen_edges.csv", index=False)
        print(f"  → {QUALITY.name}/never_seen_edges.csv ({len(never_seen_edges)} edges)")
    else:
        print(f"  → {QUALITY.name}/never_seen_edges.csv (all template edges seen)")

    print(f"\n✅ Quality report complete: {QUALITY}")


if __name__ == "__main__":
    main()
