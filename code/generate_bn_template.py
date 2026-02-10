#!/usr/bin/env python
"""
generate_bn_template.py — Step 4 of the Legal BN Pipeline
---------------------------------------------------------
Purpose:
  • Aggregate edges across cases at the *concept (label)* level to build a BN-ready DAG.
  • Prune weak/rare edges (by cross-case support and mean score).
  • Break cycles greedily so the final graph is a DAG (BN requirement).
  • Export GraphML/GML plus CSV summaries for auditability.

Inputs:
  data/processed/nodes.csv
  data/processed/edges.csv

Outputs:
  outputs/bn_template.graphml
  outputs/bn_template.gml
  outputs/bn_edges_aggregated.csv
  outputs/bn_nodes_summary.csv

Run:
  conda activate legalnlp
  cd /Users/jennatieby/Desktop/Criminal_Cases_BN
  python code/generate_bn_template.py
  # or: make bn
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTD = ROOT / "outputs"
OUTD.mkdir(parents=True, exist_ok=True)

NODES_CSV = DATA / "nodes.csv"
EDGES_CSV = DATA / "edges.csv"

OUT_GRAPHML = OUTD / "bn_template.graphml"
OUT_GML     = OUTD / "bn_template.gml"
OUT_EDGES   = OUTD / "bn_edges_aggregated.csv"
OUT_NODES   = OUTD / "bn_nodes_summary.csv"

# ---------------- Load ----------------
nodes_df = pd.read_csv(NODES_CSV)
edges_df = pd.read_csv(EDGES_CSV)

if nodes_df.empty or edges_df.empty:
    raise SystemExit("nodes.csv or edges.csv is empty — complete Steps 2–3 first.")

# Hygiene
for c in ["case_id","node_id","label","type","source","polarity","sent_text","provenance"]:
    if c in nodes_df.columns:
        nodes_df[c] = nodes_df[c].fillna("")
for c in ["case_id","src_node_id","dst_node_id","score","provenance","rationale"]:
    if c in edges_df.columns:
        edges_df[c] = edges_df[c].fillna("")

# ---------------- Per-node summary ----------------
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

# Maps
id2label = dict(node_meta[["node_id","label"]].itertuples(index=False))
id2type  = dict(node_meta[["node_id","type"]].itertuples(index=False))

# ---------------- Aggregate edges across cases at the LABEL level ----------------
edges_labeled = edges_df.copy()
edges_labeled["src_label"] = edges_labeled["src_node_id"].map(id2label)
edges_labeled["dst_label"] = edges_labeled["dst_node_id"].map(id2label)
edges_labeled = edges_labeled.dropna(subset=["src_label","dst_label"]).copy()
edges_labeled["score"] = pd.to_numeric(edges_labeled["score"], errors="coerce").fillna(0.0)

# type per label = most common observed type
label2type = (
    node_meta.groupby("label")["type"].agg(lambda s: s.value_counts().idxmax())
)

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

# ---------------- Pruning rules (diagnostics + gentle defaults) ----------------
TOTAL_CASES_WITH_EDGES = edges_df["case_id"].nunique()

# Start gentle:
min_case_support = 1          # was max(2, ceil(2% * cases))
min_mean_score   = 0.35       # was 0.45

print(f"[diag] edge-bearing cases: {TOTAL_CASES_WITH_EDGES}")
print(f"[diag] raw labeled edge pairs: {len(agg)}")
print(f"[diag] pruning: support>={min_case_support}, mean_score>={min_mean_score}")

filtered = agg[
    (agg["support_cases"] >= min_case_support) &
    (agg["mean_score"]   >= min_mean_score)
].copy()

# If still empty, progressively relax and/or fallback to top-K by support
if filtered.empty:
    print("[diag] filtered is empty; relaxing thresholds...")
    min_case_support = 1
    min_mean_score   = 0.0
    filtered = agg[
        (agg["support_cases"] >= min_case_support) &
        (agg["mean_score"]   >= min_mean_score)
    ].copy()

if filtered.empty:
    print("[diag] still empty after relax; using fallback top-K by support then score")
    tmp = agg.copy().sort_values(
        ["support_cases","mean_score","max_score"], ascending=[False, False, False]
    )
    K = min(100, max(10, len(tmp)//50))  # pick up to 100 best edges, at least 10
    filtered = tmp.head(K).copy()

filtered = filtered.sort_values(
    ["support_cases","mean_score","max_score"], ascending=[False, False, False]
).reset_index(drop=True)

print(f"[diag] kept {len(filtered)} label-level edges after pruning/fallback")

# save aggregated, pruned edges (for audit)
filtered.to_csv(OUT_EDGES, index=False)

# ---------------- Build directed graph (labels as nodes) ----------------
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

# ---------------- Make DAG (break cycles greedily) ----------------
def break_cycles_greedily(Gin: nx.DiGraph) -> nx.DiGraph:
    H = Gin.copy()
    while True:
        try:
            cycle = nx.find_cycle(H, orientation="original")
        except nx.NetworkXNoCycle:
            break
        # remove the weakest edge in the cycle
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

# ---------------- Export ----------------
nx.write_graphml(H, OUT_GRAPHML)
nx.write_gml(H, OUT_GML)

print(f"✅ BN template built with {H.number_of_nodes()} nodes and {H.number_of_edges()} edges")
print(f"   • Saved graph → {OUT_GRAPHML.name} and {OUT_GML.name}")
print(f"   • Edge audit  → {OUT_EDGES.name}")
print(f"   • Node summary→ {OUT_NODES.name}")