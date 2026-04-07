#!/usr/bin/env python3
"""
Evaluate BN consistency on full-corpus FULL_EVIDENCE posteriors.

Inputs:
  - outputs/scenario_results_full.csv
  - case_node_matrix.csv

Method:
1) Group cases by structural similarity (Jaccard on binary doctrinal vectors excluding Verdict).
   - Primary threshold: 0.60.
   - If fewer than 20 groups are found, also run threshold 0.50 and report both.
2) Within each group, compute pairwise Jensen-Shannon divergence across
   [P(Murder), P(Manslaughter), P(Not Guilty)].
3) Outputs:
   - outputs/consistency_results.csv (full pairwise rows + group stats + threshold used)
   - outputs/consistency_plot.png (bar chart of group mean JS with inconsistency threshold line)
   - console summary including group counts, size distribution, inconsistency counts,
     total similar-case pairs, mean JS across all pairs, and overall MAP-match consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "outputs" / "scenario_results_full.csv"
MATRIX = ROOT / "case_node_matrix.csv"

OUT_RESULTS = ROOT / "outputs" / "consistency_results.csv"
OUT_PLOT = ROOT / "outputs" / "consistency_plot.png"
OUT_GROUPS = ROOT / "outputs" / "consistency_groups_summary.csv"

INCONSISTENT_MEAN_JS = 0.1


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence (natural log)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")

    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = np.clip(m, eps, 1.0)

    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity for binary vectors."""
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    return float(inter / union) if union else 1.0


def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Connected components for an undirected graph."""
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    seen: set[str] = set()
    comps: list[list[str]] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        seen.add(n)
        comp: list[str] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def run_threshold(
    X: pd.DataFrame,
    post: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    ids = list(X.index)
    edges: list[tuple[str, str]] = []
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(ids)):
        ai = X.loc[ids[i]].to_numpy()
        for j in range(i + 1, len(ids)):
            bj = X.loc[ids[j]].to_numpy()
            sim = jaccard(ai, bj)
            if sim >= threshold:
                edges.append((ids[i], ids[j]))
                pairs.append((ids[i], ids[j], sim))

    comps = [c for c in connected_components(ids, edges) if len(c) >= 2]
    group_sizes = sorted([len(c) for c in comps], reverse=True)
    group_size_map: dict[int, int] = {gi + 1: len(comp) for gi, comp in enumerate(comps)}

    if not comps or not pairs:
        empty_pairs = pd.DataFrame(
            [],
            columns=[
                "threshold",
                "group_id",
                "group_label",
                "case_id_a",
                "case_id_b",
                "jaccard",
                "js_divergence",
                "map_a",
                "map_b",
                "map_match",
                "group_mean_js",
                "group_max_js",
                "group_flag",
            ],
        )
        empty_gstats = pd.DataFrame(
            [],
            columns=[
                "threshold",
                "group_id",
                "group_label",
                "group_size",
                "group_mean_js",
                "group_max_js",
                "n_pairs",
                "group_flag",
                "overall_consistency",
                "mean_js_all_pairs",
                "total_similar_pairs",
            ],
        )
        return empty_pairs, empty_gstats, group_sizes

    group_of: dict[str, int] = {}
    for gi, comp in enumerate(comps, start=1):
        for cid in comp:
            group_of[cid] = gi

    rows: list[dict[str, object]] = []
    for a, b, sim in pairs:
        ga = group_of.get(a)
        gb = group_of.get(b)
        if ga is None or gb is None or ga != gb:
            continue
        pa = post.loc[a][["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]].to_numpy(dtype=float)
        pb = post.loc[b][["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]].to_numpy(dtype=float)
        js = js_divergence(pa, pb)
        map_a = str(post.loc[a]["MAP_verdict"])
        map_b = str(post.loc[b]["MAP_verdict"])
        rows.append(
            {
                "threshold": threshold,
                "group_id": ga,
                "group_label": f"t={threshold:.2f}|g={ga}",
                "case_id_a": a,
                "case_id_b": b,
                "jaccard": float(sim),
                "js_divergence": float(js),
                "map_a": map_a,
                "map_b": map_b,
                "map_match": bool(map_a == map_b),
            }
        )

    res = pd.DataFrame(rows)
    gstats = (
        res.groupby(["threshold", "group_id", "group_label"])["js_divergence"]
        .agg(group_mean_js="mean", group_max_js="max", n_pairs="count")
        .reset_index()
    )
    gstats["group_size"] = gstats["group_id"].map(group_size_map).astype(int)
    gstats["group_flag"] = np.where(gstats["group_mean_js"] > INCONSISTENT_MEAN_JS, "INCONSISTENT", "OK")
    overall_consistency = float(res["map_match"].mean()) if len(res) else float("nan")
    mean_js_all_pairs = float(res["js_divergence"].mean()) if len(res) else float("nan")
    total_similar_pairs = int(len(res))
    gstats["overall_consistency"] = overall_consistency
    gstats["mean_js_all_pairs"] = mean_js_all_pairs
    gstats["total_similar_pairs"] = total_similar_pairs
    res = res.merge(
        gstats[["threshold", "group_id", "group_mean_js", "group_max_js", "group_flag"]],
        on=["threshold", "group_id"],
        how="left",
    )
    return res, gstats, group_sizes


def print_summary_for_threshold(threshold: float, res: pd.DataFrame, gstats: pd.DataFrame, group_sizes: list[int]) -> None:
    n_groups = int(len(gstats))
    total_pairs = int(len(res))
    n_inconsistent = int((gstats["group_flag"] == "INCONSISTENT").sum()) if n_groups else 0
    n_ok = int((gstats["group_flag"] == "OK").sum()) if n_groups else 0
    mean_js_all = float(res["js_divergence"].mean()) if total_pairs else float("nan")
    overall_consistency = float(res["map_match"].mean()) if total_pairs else float("nan")

    print(f"\n=== Threshold {threshold:.2f} ===")
    print(f"Groups found: {n_groups}")
    print(f"Group size distribution (desc): {group_sizes[:30]}{' ...' if len(group_sizes) > 30 else ''}")
    print(f"Total similar-case pairs found: {total_pairs}")
    print(f"Groups flagged INCONSISTENT vs OK: {n_inconsistent} vs {n_ok}")
    print(f"Mean JS divergence across all pairs: {mean_js_all:.6f}" if total_pairs else "Mean JS divergence across all pairs: NaN")
    print(
        f"Overall consistency score (MAP verdict match among similar-case pairs): {overall_consistency:.6f}"
        if total_pairs
        else "Overall consistency score (MAP verdict match among similar-case pairs): NaN"
    )
    if n_groups:
        show = gstats[["group_label", "n_pairs", "group_mean_js", "group_max_js", "group_flag"]].copy()
        print("\nPer-group JS summary:")
        print(show.to_string(index=False))


def main() -> None:
    if not SCENARIOS.exists():
        raise FileNotFoundError(f"Missing: {SCENARIOS}")
    if not MATRIX.exists():
        raise FileNotFoundError(f"Missing: {MATRIX}")

    scen = pd.read_csv(SCENARIOS, encoding="utf-8-sig")
    if "scenario" in scen.columns:
        full = scen[scen["scenario"] == "FULL_EVIDENCE"].copy()
    else:
        full = scen.copy()
    if full.empty:
        raise ValueError(f"No FULL_EVIDENCE rows found in {SCENARIOS}")

    mat = pd.read_csv(MATRIX, encoding="utf-8-sig")
    if "case_id" not in mat.columns:
        mat = mat.rename(columns={mat.columns[0]: "case_id"})

    full["case_id"] = full["case_id"].astype(str)
    mat["case_id"] = mat["case_id"].astype(str)

    required_prob_cols = ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)", "MAP_verdict"]
    missing = [c for c in required_prob_cols if c not in full.columns]
    if missing:
        raise ValueError(f"{SCENARIOS} missing required columns: {missing}")

    case_ids = sorted(set(full["case_id"]))
    sub = mat[mat["case_id"].isin(case_ids)].copy()
    if sub.empty:
        raise ValueError("No overlapping case_ids between inference results and case_node_matrix.csv")

    doctrinal_cols = [c for c in sub.columns if c not in {"case_id", "Verdict"}]
    X = sub.set_index("case_id")[doctrinal_cols].fillna(0).astype(int)
    post = full.set_index("case_id")[required_prob_cols].copy()
    # align index order
    post = post.loc[X.index]

    thresholds = [0.60, 0.50]
    res60, gstats60, sizes60 = run_threshold(X, post, 0.60)
    res50, gstats50, sizes50 = run_threshold(X, post, 0.50)
    all_res = pd.concat([res60, res50], ignore_index=True)
    all_gstats = pd.concat([gstats60, gstats50], ignore_index=True)

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    all_res.to_csv(OUT_RESULTS, index=False, encoding="utf-8")
    all_gstats = all_gstats[
        [
            "threshold",
            "group_id",
            "group_label",
            "group_size",
            "n_pairs",
            "group_mean_js",
            "group_max_js",
            "group_flag",
            "overall_consistency",
            "mean_js_all_pairs",
            "total_similar_pairs",
        ]
    ].sort_values(["threshold", "group_id"])
    all_gstats.to_csv(OUT_GROUPS, index=False, encoding="utf-8")

    # Plot all analyzed groups (single or both thresholds)
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(10, len(all_gstats) * 0.4), 6))
    labels = all_gstats["group_label"].astype(str).tolist()
    vals = all_gstats["group_mean_js"].to_numpy(dtype=float)
    plt.bar(labels, vals)
    plt.axhline(
        INCONSISTENT_MEAN_JS,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"inconsistent threshold ({INCONSISTENT_MEAN_JS})",
    )
    plt.xlabel("Group")
    plt.ylabel("Mean Jensen-Shannon divergence")
    plt.title("Within-group posterior variance (FULL_EVIDENCE)")
    if len(labels) > 20:
        plt.xticks([])
    else:
        plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    plt.close()

    print(f"Evaluated FULL_EVIDENCE cases: {len(X)}")
    print_summary_for_threshold(0.60, res60, gstats60, sizes60)
    if 0.50 in thresholds:
        print_summary_for_threshold(0.50, res50, gstats50, sizes50)

    print(f"\nThresholds analyzed: {thresholds}")
    print(f"Wrote full results: {OUT_RESULTS}")
    print(f"Wrote group summary: {OUT_GROUPS}")
    print(f"Wrote plot: {OUT_PLOT}")

    code_dir = Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import generate_results_figures as grf

    grf.setup_style()
    cons_reload = pd.read_csv(OUT_RESULTS, encoding="utf-8-sig")
    matrix_reload = pd.read_csv(MATRIX, encoding="utf-8-sig")
    grf.OUT_DIR.mkdir(parents=True, exist_ok=True)
    grf.fig5_consistency_js(cons_reload, matrix_reload)
    grf.table3_consistency_summary(cons_reload)
    print(f"Wrote {grf.OUT_DIR / 'fig5_consistency_js.png'}")
    print(f"Wrote {grf.OUT_DIR / 'table3_consistency_summary.csv'}")
    print(f"Wrote {grf.OUT_DIR / 'table3_consistency_summary.png'}")


if __name__ == "__main__":
    main()

