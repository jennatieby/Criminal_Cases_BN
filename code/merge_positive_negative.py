#!/usr/bin/env python3
"""
merge_positive_negative.py

Merge positive (BAILII) and negative (CCRC) nodes and edges into unified CSVs
with a provenance column. Use for "all cases" or "filter by provenance" analyses.

Usage:
  python code/merge_positive_negative.py

Outputs:
  data/processed/nodes_unified.csv
  data/processed/edges_unified.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

DEFAULT_POS_NODES = DATA / "nodes.csv"
DEFAULT_NEG_NODES = DATA / "negative_nodes.csv"
DEFAULT_POS_EDGES = DATA / "edges.csv"
DEFAULT_NEG_EDGES = DATA / "negative_edges.csv"
OUT_NODES = DATA / "nodes_unified.csv"
OUT_EDGES = DATA / "edges_unified.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Merge positive and negative nodes/edges into unified CSVs.")
    p.add_argument("--pos-nodes", type=Path, default=DEFAULT_POS_NODES, help="Positive nodes CSV.")
    p.add_argument("--neg-nodes", type=Path, default=DEFAULT_NEG_NODES, help="Negative nodes CSV.")
    p.add_argument("--pos-edges", type=Path, default=DEFAULT_POS_EDGES, help="Positive edges CSV.")
    p.add_argument("--neg-edges", type=Path, default=DEFAULT_NEG_EDGES, help="Negative edges CSV.")
    p.add_argument("--out-nodes", type=Path, default=OUT_NODES, help="Output unified nodes CSV.")
    p.add_argument("--out-edges", type=Path, default=OUT_EDGES, help="Output unified edges CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    # Nodes
    pos_n = pd.read_csv(args.pos_nodes, encoding="utf-8-sig")
    neg_n = pd.read_csv(args.neg_nodes, encoding="utf-8-sig")
    if "provenance" not in pos_n.columns:
        pos_n["provenance"] = "real"
    else:
        pos_n["provenance"] = pos_n["provenance"].fillna("real").astype(str).replace("", "real")
    if "provenance" not in neg_n.columns:
        neg_n["provenance"] = "negative"
    else:
        neg_n["provenance"] = neg_n["provenance"].fillna("negative").astype(str).replace("", "negative")
    nodes_unified = pd.concat([pos_n, neg_n], ignore_index=True)
    nodes_unified.to_csv(args.out_nodes, index=False)
    print(f"Nodes unified: {len(nodes_unified):,} rows -> {args.out_nodes}")
    print(f"  real:    {len(nodes_unified[nodes_unified['provenance'] == 'real']):,}")
    print(f"  negative: {len(nodes_unified[nodes_unified['provenance'] == 'negative']):,}")

    # Edges
    pos_e = pd.read_csv(args.pos_edges, encoding="utf-8-sig")
    neg_e = pd.read_csv(args.neg_edges, encoding="utf-8-sig")
    if "provenance" not in pos_e.columns:
        pos_e["provenance"] = "real"
    else:
        pos_e["provenance"] = pos_e["provenance"].fillna("real").astype(str).replace("", "real")
    if "provenance" not in neg_e.columns:
        neg_e["provenance"] = "negative"
    else:
        neg_e["provenance"] = neg_e["provenance"].fillna("negative").astype(str).replace("", "negative")
    edges_unified = pd.concat([pos_e, neg_e], ignore_index=True)
    edges_unified.to_csv(args.out_edges, index=False)
    print(f"Edges unified: {len(edges_unified):,} rows -> {args.out_edges}")
    print(f"  real:    {len(edges_unified[edges_unified['provenance'] == 'real']):,}")
    print(f"  negative: {len(edges_unified[edges_unified['provenance'] == 'negative']):,}")
    print(f"  cases (unique case_id): {edges_unified['case_id'].nunique():,}")


if __name__ == "__main__":
    main()
