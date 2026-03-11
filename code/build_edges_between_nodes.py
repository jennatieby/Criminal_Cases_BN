#!/usr/bin/env python
"""
build_edges_between_nodes.py

Builds directed edges between nodes (same case) using co-occurrence and type heuristics.
Edges: same_sentence (strongest), same_section, same_case cross-section (weakest).
Supports positive (BAILII) and negative (CCRC) runs via CLI arguments.

Usage:
  Positive (defaults):
    python code/build_edges_between_nodes.py
  Negative:
    python code/build_edges_between_nodes.py --input data/processed/negative_nodes.csv --output-edges data/processed/negative_edges.csv --output-inventory data/processed/negative_node_inventory.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import itertools, math

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_args():
    p = argparse.ArgumentParser(description="Build edges between nodes from a nodes CSV.")
    p.add_argument("--input", type=Path, default=DATA / "processed" / "nodes.csv",
                   help="Input nodes CSV (from extract_nodes_from_cases.py).")
    p.add_argument("--output-edges", type=Path, default=DATA / "processed" / "edges.csv",
                   help="Output CSV for edges.")
    p.add_argument("--output-inventory", type=Path, default=None,
                   help="Output CSV for node inventory. If omitted, derived from output-edges.")
    args = p.parse_args()
    if args.output_inventory is None:
        stem = args.output_edges.stem.replace("edges", "node_inventory")
        args.output_inventory = args.output_edges.parent / (stem + ".csv")
    return args


# ---------- Heuristics (type priors, polarity, source) ----------
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
    if p == 0:
        return 0.60
    if p == -1:
        return 0.75
    return 1.00

def source_adjustment(s: str) -> float:
    s = (s or "").lower()
    if s == "court_fact":
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


def main():
    args = parse_args()
    input_path = args.input.resolve()
    edges_path = args.output_edges.resolve()
    inv_path = args.output_inventory.resolve()
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if df.empty:
        raise SystemExit(f"Input nodes CSV is empty: {input_path}")

    # Normalize a few fields
    for col in ["type","source","provenance","section","sent_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Create a simple sentence id per case by text (stable hashing)
    df["sent_id"] = (
        df["case_id"].astype(str) + "::" + df["sent_text"].astype(str)
    ).astype("category").cat.codes

    # ---------- Build edges ----------
    records = []
    inventory = []
    for (cid, lab, prov) in df[["case_id","label","provenance"]].itertuples(index=False):
        inventory.append((cid, lab, prov))

    for case_id, g in df.groupby("case_id", sort=False):
        by_section = defaultdict(list)
        for r in g.itertuples(index=False):
            by_section[r.section].append(r)

        for sid, sg in g.groupby("sent_id", sort=False):
            rows = list(sg.itertuples(index=False))
            for a, b in itertools.permutations(rows, 2):
                if a.node_id == b.node_id:
                    continue
                s = base_prior(a.type, b.type)
                s += 0.25
                s *= polarity_penalty(a.polarity) * polarity_penalty(b.polarity)
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

        for sect, rows in by_section.items():
            if not rows:
                continue
            for a, b in itertools.permutations(rows, 2):
                if a.node_id == b.node_id or a.sent_id == b.sent_id:
                    continue
                s = base_prior(a.type, b.type)
                s += 0.10
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

        # 3) Same-case cross-section (weak): link nodes in different sections so cases with 2+ nodes always get at least one edge
        sections_list = list(by_section.keys())
        for i, sect1 in enumerate(sections_list):
            for sect2 in sections_list[i + 1:]:
                for a in by_section[sect1]:
                    for b in by_section[sect2]:
                        if a.node_id == b.node_id:
                            continue
                        s = base_prior(a.type, b.type)
                        s += 0.05
                        s *= polarity_penalty(a.polarity) * polarity_penalty(b.polarity)
                        s += 0.5 * source_adjustment(a.source) + 0.25 * source_adjustment(b.source)
                        score = clamp01(s)
                        rationale = "same_case|type_prior"
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

    edges_df.to_csv(edges_path, index=False)
    inv_df = (
        pd.DataFrame(inventory, columns=["case_id","label","provenance"])
        .groupby(["label","provenance"], as_index=False)
        .size()
        .rename(columns={"size":"count"})
        .sort_values("count", ascending=False)
    )
    inv_df.to_csv(inv_path, index=False)
    print(f"Built {len(edges_df):,} edges across {edges_df['case_id'].nunique()} cases -> {edges_path}")
    print(f"Node inventory saved -> {inv_path}")


if __name__ == "__main__":
    main()