#!/usr/bin/env python
"""
generate_bn_template.py: Build BN template (DAG of labels) from nodes/edges.
Positive: python code/generate_bn_template.py
Negative: python code/generate_bn_template.py --nodes data/processed/negative_nodes.csv --edges data/processed/negative_edges.csv --prefix negative_
Unified (less greedy): python code/generate_bn_template.py --nodes data/processed/nodes_unified.csv --edges data/processed/edges_unified.csv --prefix unified_ --min-mean-score 0.2
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTD = ROOT / "outputs"
OUTD.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Build BN template from nodes and edges.")
    p.add_argument("--nodes", type=Path, default=DATA / "nodes.csv", help="Input nodes CSV.")
    p.add_argument("--edges", type=Path, default=DATA / "edges.csv", help="Input edges CSV.")
    p.add_argument("--prefix", type=str, default="", help="Output filename prefix (e.g. negative_).")
    p.add_argument("--min-support", type=int, default=1,
                  help="Min number of cases an edge must appear in (default: 1).")
    p.add_argument("--min-mean-score", type=float, default=0.35,
                  help="Min mean score for an edge to be kept (default: 0.35). Lower = less greedy.")
    return p.parse_args()


def main():
    args = parse_args()
    nodes_path = args.nodes.resolve()
    edges_path = args.edges.resolve()
    prefix = (args.prefix or "").strip()
    OUT_GRAPHML = OUTD / f"{prefix}bn_template.graphml"
    OUT_GML = OUTD / f"{prefix}bn_template.gml"
    OUT_EDGES = OUTD / f"{prefix}bn_edges_aggregated.csv"
    OUT_NODES = OUTD / f"{prefix}bn_nodes_summary.csv"

    nodes_df = pd.read_csv(nodes_path, encoding="utf-8-sig")
    edges_df = pd.read_csv(edges_path, encoding="utf-8-sig")
    if nodes_df.empty or edges_df.empty:
        raise SystemExit("Nodes or edges CSV is empty.")

    # Hygiene
    for c in ["case_id","node_id","label","type","source","polarity","sent_text","provenance"]:
        if c in nodes_df.columns:
            nodes_df[c] = nodes_df[c].fillna("")
    for c in ["case_id","src_node_id","dst_node_id","score","provenance","rationale"]:
        if c in edges_df.columns:
            edges_df[c] = edges_df[c].fillna("")

    # Per-node summary
    node_meta = (
        nodes_df[["case_id","node_id","label","type","provenance"]]
        .drop_duplicates()
    )

    node_summary = (
        node_meta.groupby(["label","type"], as_index=False)
                 .size()
                 .rename(columns={"size":"occurrences"})
                 .sort_values("occurrences", ascending=False)
    )
    node_summary.to_csv(OUT_NODES, index=False)

    id2label = dict(node_meta[["node_id","label"]].itertuples(index=False))
    label2type = (
        node_meta.groupby("label")["type"].agg(lambda s: s.value_counts().idxmax())
    )

    edges_labeled = edges_df.copy()
    edges_labeled["src_label"] = edges_labeled["src_node_id"].map(id2label)
    edges_labeled["dst_label"] = edges_labeled["dst_node_id"].map(id2label)
    edges_labeled = edges_labeled.dropna(subset=["src_label","dst_label"]).copy()
    edges_labeled["score"] = pd.to_numeric(edges_labeled["score"], errors="coerce").fillna(0.0)

    agg = (
        edges_labeled
          .groupby(["src_label","dst_label"], as_index=False)
          .agg(
              support_cases=("case_id", lambda s: s.nunique()),
              obs_count=("case_id", "count"),
              mean_score=("score", "mean"),
              max_score=("score", "max"),
          )
    )
    agg["src_type"] = agg["src_label"].map(label2type).fillna("narrative")
    agg["dst_type"] = agg["dst_label"].map(label2type).fillna("narrative")

    TOTAL_CASES_WITH_EDGES = edges_df["case_id"].nunique()
    min_case_support = args.min_support
    min_mean_score = args.min_mean_score

    print(f"[diag] edge-bearing cases: {TOTAL_CASES_WITH_EDGES}")
    print(f"[diag] raw labeled edge pairs: {len(agg)}")
    print(f"[diag] pruning: support>={min_case_support}, mean_score>={min_mean_score}")

    filtered = agg[
        (agg["support_cases"] >= min_case_support) &
        (agg["mean_score"]   >= min_mean_score)
    ].copy()

    if filtered.empty:
        print("[diag] filtered is empty; relaxing thresholds...")
        min_mean_score = 0.0
        filtered = agg[
            (agg["support_cases"] >= min_case_support) &
            (agg["mean_score"]   >= min_mean_score)
        ].copy()

    if filtered.empty:
        print("[diag] still empty after relax; using fallback top-K")
        tmp = agg.copy().sort_values(
            ["support_cases","mean_score","max_score"], ascending=[False, False, False]
        )
        K = min(100, max(10, len(tmp)//50))
        filtered = tmp.head(K).copy()

    filtered = filtered.sort_values(
        ["support_cases","mean_score","max_score"], ascending=[False, False, False]
    ).reset_index(drop=True)

    print(f"[diag] kept {len(filtered)} label-level edges after pruning/fallback")
    filtered.to_csv(OUT_EDGES, index=False)

    G = nx.DiGraph()
    labels = sorted(set(filtered["src_label"]).union(set(filtered["dst_label"])))

    def label_occ(lbl: str) -> int:
        sub = node_summary[node_summary["label"] == lbl]
        return int(sub["occurrences"].sum()) if not sub.empty else 0

    for lbl in labels:
        t = label2type.get(lbl, "narrative")
        occ = label_occ(lbl)
        G.add_node(lbl, type=t, occurrences=occ)

    edge_collapse = (
        filtered.groupby(["src_label","dst_label"], as_index=False)
                .agg(
                    support_cases=("support_cases","max"),
                    obs_count=("obs_count","sum"),
                    mean_score=("mean_score","mean"),
                    max_score=("max_score","max"),
                )
                .sort_values(["support_cases","mean_score"], ascending=[False, False])
    )

    for r in edge_collapse.itertuples(index=False):
        G.add_edge(
            r.src_label, r.dst_label,
            support_cases=int(r.support_cases),
            obs_count=int(r.obs_count),
            mean_score=float(r.mean_score),
            max_score=float(r.max_score),
            weight=float(r.mean_score),
        )

    def break_cycles_greedily(Gin: nx.DiGraph) -> nx.DiGraph:
        H = Gin.copy()
        while True:
            try:
                cycle = nx.find_cycle(H, orientation="original")
            except nx.NetworkXNoCycle:
                break
            min_e = None
            min_w = 1e9
            for u, v, _ in cycle:
                w = H[u][v].get("weight", 0.0)
                if w < min_w:
                    min_w = w
                    min_e = (u, v)
            if min_e is None:
                break
            H.remove_edge(*min_e)
        return H

    H = break_cycles_greedily(G)

    nx.write_graphml(H, OUT_GRAPHML)
    nx.write_gml(H, OUT_GML)

    print(f"BN template built with {H.number_of_nodes()} nodes and {H.number_of_edges()} edges")
    print(f"   Saved graph -> {OUT_GRAPHML.name}")
    print(f"   Edge audit  -> {OUT_EDGES.name}")
    print(f"   Node summary-> {OUT_NODES.name}")


if __name__ == "__main__":
    main()