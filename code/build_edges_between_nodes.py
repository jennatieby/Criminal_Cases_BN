#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import itertools, math

# ---------- Paths (robust to where you run from) ----------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NODES_CSV = DATA / "processed" / "nodes.csv"
EDGES_CSV = DATA / "processed" / "edges.csv"
INV_CSV   = DATA / "processed" / "node_inventory.csv"
EDGES_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------- Load nodes ----------
df = pd.read_csv(NODES_CSV)
if df.empty:
    raise SystemExit("nodes.csv is empty — run Step 2 first.")

# Normalize a few fields
for col in ["type","source","provenance","section","sent_text"]:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Create a simple sentence id per case by text (stable hashing)
df["sent_id"] = (
    df["case_id"].astype(str) + "::" + df["sent_text"].astype(str)
).astype("category").cat.codes

# ---------- Heuristics ----------
# Base directional priors by node types (src_type, dst_type)
TYPE_PRIOR = {
    ("narrative","evidence"): 0.60,
    ("narrative","legal_facts"): 0.65,
    ("evidence","legal_facts"): 0.55,
    ("evidence","narrative"): 0.35,
    ("legal_facts","narrative"): 0.20,
    ("legal_facts","evidence"): 0.25,
}

def base_prior(t_src: str, t_dst: str) -> float:
    return TYPE_PRIOR.get((t_src, t_dst), 0.30)

def polarity_penalty(p: float) -> float:
    # p is node polarity: 1 asserted, 0 negated, -1 uncertain
    if p == 0:
        return 0.60
    if p == -1:
        return 0.75
    return 1.00

def source_adjustment(s: str) -> float:
    s = (s or "").lower()
    if s == "court_fact":  # judicial finding
        return +0.10
    if s in {"pros","prosecution_case","prosecution"}:
        return +0.03
    if s in {"def","defence_case","defense"}:
        return +0.00
    if s == "uncertain":
        return -0.05
    return 0.00

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ---------- Build edges ----------
records = []        # raw observations before aggregation
inventory = []      # node counts per case/label

# inventory for QC
for (cid, lab, prov) in df[["case_id","label","provenance"]].itertuples(index=False):
    inventory.append((cid, lab, prov))

# group by case
for case_id, g in df.groupby("case_id", sort=False):
    # section-aware buckets for proximity
    by_section = defaultdict(list)
    for r in g.itertuples(index=False):
        by_section[r.section].append(r)

    # 1) Same-sentence co-occurrence (strongest)
    for sid, sg in g.groupby("sent_id", sort=False):
        # all ordered pairs within sentence
        rows = list(sg.itertuples(index=False))
        for a, b in itertools.permutations(rows, 2):
            if a.node_id == b.node_id:
                continue
            # base score
            s = base_prior(a.type, b.type)
            # same sentence boost
            s += 0.25
            # polarity penalties (both ends matter)
            s *= polarity_penalty(a.polarity) * polarity_penalty(b.polarity)
            # source tweaks (src + dst influences)
            s += 0.5 * source_adjustment(a.source) + 0.25 * source_adjustment(b.source)
            score = clamp01(s)

            rationale = "same_sentence|type_prior"
            records.append({
                "case_id": case_id,
                "src_node_id": a.node_id,
                "dst_node_id": b.node_id,
                "score": score,
                "rationale": rationale,
                "provenance": g["provenance"].iloc[0] if "provenance" in g else "real"
            })

    # 2) Same-section proximity (weaker, if different sentence)
    for sect, rows in by_section.items():
        if not rows:
            continue
        # order by appearance to create near-neighbor links
        # (we don't have true positions, so we use occurrence order)
        for a, b in itertools.permutations(rows, 2):
            if a.node_id == b.node_id or a.sent_id == b.sent_id:
                continue
            s = base_prior(a.type, b.type)
            s += 0.10  # section proximity
            s *= polarity_penalty(a.polarity) * polarity_penalty(b.polarity)
            s += 0.5 * source_adjustment(a.source) + 0.25 * source_adjustment(b.source)
            score = clamp01(s)
            rationale = "same_section|type_prior"
            records.append({
                "case_id": case_id,
                "src_node_id": a.node_id,
                "dst_node_id": b.node_id,
                "score": score,
                "rationale": rationale,
                "provenance": g["provenance"].iloc[0] if "provenance" in g else "real"
            })

# ---------- Aggregate duplicate edges (case, src, dst) ----------
bucket = defaultdict(lambda: {"sum":0.0, "cnt":0, "rationales":Counter(), "prov":None})
for r in records:
    key = (r["case_id"], r["src_node_id"], r["dst_node_id"])
    b = bucket[key]
    b["sum"] += r["score"]
    b["cnt"] += 1
    b["rationales"][r["rationale"]] += 1
    b["prov"] = r["provenance"]

final_edges = []
for (case_id, src, dst), info in bucket.items():
    avg = info["sum"]/info["cnt"]
    # small bump for repeated evidence of the same link
    final_score = clamp01(avg + math.log1p(info["cnt"]) / 12.0)
    rationale = "|".join([k for k,_ in info["rationales"].most_common(3)])
    final_edges.append({
        "case_id": case_id,
        "src_node_id": src,
        "dst_node_id": dst,
        "score": round(final_score, 3),
        "count": info["cnt"],
        "rationale": rationale,
        "provenance": info["prov"] or "real"
    })

edges_df = pd.DataFrame(final_edges).sort_values(
    ["case_id","score"], ascending=[True, False]
)

# ---------- Save outputs ----------
edges_df.to_csv(EDGES_CSV, index=False)

inv_df = (
    pd.DataFrame(inventory, columns=["case_id","label","provenance"])
      .groupby(["label","provenance"], as_index=False)
      .size()
      .rename(columns={"size":"count"})
      .sort_values("count", ascending=False)
)
inv_df.to_csv(INV_CSV, index=False)

print(f"✅ Built {len(edges_df):,} edges across {edges_df['case_id'].nunique()} cases → {EDGES_CSV}")
print(f"📦 Node inventory saved → {INV_CSV}")