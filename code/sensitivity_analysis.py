#!/usr/bin/env python3
"""
Sensitivity analysis for the train-fitted Bayesian network.

Inputs:
  - outputs/homicide_bn_train.bif
  - outputs/scenario_results_full.csv (reference: mean P(Murder) on full corpus)

Outputs:
  - outputs/sensitivity_single_node.csv
  - outputs/sensitivity_cpt_perturbation.csv
  - outputs/sensitivity_summary.csv
  - outputs/figures/fig_sensitivity_analysis.png

Run:
  pip install pgmpy pandas matplotlib
  python code/sensitivity_analysis.py
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.readwrite import BIFReader
except Exception as e:
    raise SystemExit("Install pgmpy: pip install pgmpy") from e


ROOT = Path(__file__).resolve().parents[1]
BIF_PATH = ROOT / "outputs" / "homicide_bn_train.bif"
SCENARIO_CSV = ROOT / "outputs" / "scenario_results_full.csv"

OUT_SINGLE = ROOT / "outputs" / "sensitivity_single_node.csv"
OUT_CPT = ROOT / "outputs" / "sensitivity_cpt_perturbation.csv"
OUT_SUMMARY = ROOT / "outputs" / "sensitivity_summary.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "fig_sensitivity_analysis.png"

VERDICT_ORDER = ["Murder", "Manslaughter", "Not_Guilty"]
MULTIPLIERS = (1.1, 0.9, 1.2, 0.8)


def load_model_from_bif(bif_path: Path) -> DiscreteBayesianNetwork:
    txt = bif_path.read_text(encoding="utf-8", errors="ignore")
    if "Not Guilty" in txt:
        txt = txt.replace("Not Guilty", "Not_Guilty")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bif", delete=False, encoding="utf-8") as f:
            f.write(txt)
            tmp = Path(f.name)
        try:
            return BIFReader(str(tmp)).get_model()
        finally:
            tmp.unlink(missing_ok=True)
    return BIFReader(str(bif_path)).get_model()


def p_murder(model: DiscreteBayesianNetwork, evidence: dict[str, str] | None = None) -> float:
    infer = VariableElimination(model)
    q = infer.query(variables=["Verdict"], evidence=evidence or {}, show_progress=False)
    states = list(q.state_names["Verdict"])
    probs = dict(zip(states, q.values))
    if "Not Guilty" in probs and "Not_Guilty" not in probs:
        probs["Not_Guilty"] = probs["Not Guilty"]
    return float(probs.get("Murder", 0.0))


def non_verdict_nodes(model: DiscreteBayesianNetwork) -> list[str]:
    nodes = [n for n in model.nodes() if n != "Verdict"]
    return sorted(nodes)


def cpd_to_2d(cpd: TabularCPD) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return (matrix n_child x n_configs, original_shape)."""
    arr = np.asarray(cpd.values, dtype=float)
    orig_shape = arr.shape
    if arr.ndim == 1:
        mat = arr.reshape(-1, 1)
    else:
        mat = arr.reshape(arr.shape[0], -1)
    return mat, orig_shape


def rebuild_cpd(cpd: TabularCPD, mat2d: np.ndarray) -> TabularCPD:
    """Build TabularCPD from 2D probability table (child states x parent configs)."""
    if len(cpd.variables) == 1:
        return TabularCPD(
            variable=cpd.variable,
            variable_card=cpd.variable_card,
            values=mat2d,
            state_names=cpd.state_names,
        )
    return TabularCPD(
        variable=cpd.variable,
        variable_card=cpd.variable_card,
        values=mat2d,
        evidence=cpd.variables[1:],
        evidence_card=list(cpd.cardinality[1:]),
        state_names=cpd.state_names,
    )


def clone_model_replace_cpd(
    base: DiscreteBayesianNetwork, target: str, new_cpd: TabularCPD
) -> DiscreteBayesianNetwork:
    m = DiscreteBayesianNetwork(base.edges())
    for v in sorted(base.nodes()):
        if v == target:
            m.add_cpds(new_cpd)
        else:
            m.add_cpds(copy.deepcopy(base.get_cpds(v)))
    if not m.check_model():
        raise RuntimeError(f"Invalid model after replacing CPD for {target}")
    return m


def run_single_node_sensitivity(model: DiscreteBayesianNetwork) -> pd.DataFrame:
    rows = []
    for node in non_verdict_nodes(model):
        p0 = p_murder(model, {node: "0"})
        p1 = p_murder(model, {node: "1"})
        delta = p1 - p0
        direction = "INCREASES" if delta > 1e-12 else ("DECREASES" if delta < -1e-12 else "NEUTRAL")
        rows.append(
            {
                "node": node,
                "P_Murder_given_node_0": p0,
                "P_Murder_given_node_1": p1,
                "delta_P_Murder": delta,
                "direction_vs_node0": direction,
            }
        )
    df = pd.DataFrame(rows)
    df["abs_delta"] = df["delta_P_Murder"].abs()
    return df.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def run_cpt_perturbation(model: DiscreteBayesianNetwork, baseline_p: float) -> pd.DataFrame:
    """
    For each CPT cell (conditional column x child row), try +/-10% and +/-20% multipliers
    on that probability, clip, renormalize the column, measure |Δ P(Murder)|.
    """
    detail_rows: list[dict] = []
    mean_abs_by_node: dict[str, list[float]] = {}

    # Perturb every CPT including Verdict (directly moves the murder marginal).
    for node in sorted(model.nodes()):
        cpd = model.get_cpds(node)
        mat2d, _ = cpd_to_2d(cpd)
        child_states = list(cpd.state_names[cpd.variable])
        n_r, n_c = mat2d.shape
        abs_deltas_this_node: list[float] = []

        for c in range(n_c):
            for r in range(n_r):
                base_p_cell = float(mat2d[r, c])
                best_abs = 0.0
                best_mult = None
                best_p_pert = baseline_p

                for mult in MULTIPLIERS:
                    col = mat2d[:, c].astype(float).copy()
                    col[r] = float(np.clip(col[r] * mult, 0.0, 1.0))
                    s = float(col.sum())
                    if s <= 1e-15:
                        continue
                    col = col / s
                    new_mat = mat2d.copy()
                    new_mat[:, c] = col
                    try:
                        new_cpd = rebuild_cpd(cpd, new_mat)
                        m2 = clone_model_replace_cpd(model, node, new_cpd)
                        p_new = p_murder(m2)
                    except Exception:
                        continue
                    ad = abs(p_new - baseline_p)
                    abs_deltas_this_node.append(ad)
                    if ad > best_abs:
                        best_abs = ad
                        best_mult = mult
                        best_p_pert = p_new

                detail_rows.append(
                    {
                        "cpt_node": node,
                        "parent_config_index": c,
                        "child_state_index": r,
                        "child_state_label": child_states[r] if r < len(child_states) else str(r),
                        "baseline_prob": base_p_cell,
                        "best_multiplier": best_mult,
                        "P_Murder_baseline": baseline_p,
                        "P_Murder_perturbed": best_p_pert,
                        "max_abs_delta_P_Murder": best_abs,
                    }
                )

        mean_abs_by_node[node] = abs_deltas_this_node

    df = pd.DataFrame(detail_rows)
    # Per-node mean absolute change across all perturbation trials (all cells x multipliers attempted)
    per_node_mean = []
    for node in sorted(mean_abs_by_node.keys()):
        vals = mean_abs_by_node[node]
        per_node_mean.append(
            {
                "cpt_node": node,
                "n_perturbation_trials": len(vals),
                "mean_abs_delta_P_Murder": float(np.mean(vals)) if vals else 0.0,
            }
        )
    df_mean = pd.DataFrame(per_node_mean)

    # Merge mean onto detail for convenience (repeat per row) — user asked separate reporting;
    # keep detail file + we add summary rows in OUT_SUMMARY. Store mean in CPT file as second table?
    # Simpler: append summary rows at end of OUT_CPT with cpt_node starting with __MEAN__
    # Better: single CSV with all detail + separate section via a column `row_type`.

    df["row_type"] = "cpt_entry"
    df_mean["row_type"] = "node_mean_abs_delta"
    # Align columns without overwriting df_mean's summary stats (n_perturbation_trials, mean_*).
    df_pad = df.copy()
    df_mean_pad = df_mean.copy()
    for col in df_mean_pad.columns:
        if col not in df_pad.columns:
            df_pad[col] = np.nan
    for col in df_pad.columns:
        if col not in df_mean_pad.columns:
            df_mean_pad[col] = np.nan
    combined = pd.concat([df_pad, df_mean_pad], ignore_index=True)
    return combined


def run_pathway_sensitivity(model: DiscreteBayesianNetwork) -> pd.DataFrame:
    pathways = [
        ("ACTUS_REUS", "UnlawfulKilling"),
        ("MENS_REA", "MaliceAforethought"),
        ("DEFENCES", "Defences"),
    ]
    rows = []
    for label, node in pathways:
        if node not in model.nodes():
            continue
        p0 = p_murder(model, {node: "0"})
        p1 = p_murder(model, {node: "1"})
        rows.append(
            {
                "pathway": label,
                "node": node,
                "P_Murder_node_0": p0,
                "P_Murder_node_1": p1,
                "delta_P_Murder": p1 - p0,
            }
        )
    return pd.DataFrame(rows)


def load_scenario_mean_p_murder() -> float | None:
    if not SCENARIO_CSV.exists():
        return None
    df = pd.read_csv(SCENARIO_CSV, encoding="utf-8-sig")
    if "scenario" in df.columns:
        df = df[df["scenario"] == "FULL_EVIDENCE"]
    if df.empty or "P(Murder)" not in df.columns:
        return None
    return float(df["P(Murder)"].astype(float).mean())


def plot_single_node_bars(df_single: pd.DataFrame, out_path: Path) -> None:
    d = df_single.sort_values("abs_delta", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(d))))
    y = np.arange(len(d))
    colors = ["#1e3a5f" if x >= 0 else "#8b2942" for x in d["delta_P_Murder"]]
    ax.barh(y, d["delta_P_Murder"].values, color=colors, edgecolor="#334155", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d["node"].values)
    ax.axvline(0, color="#64748b", linewidth=1)
    ax.set_xlabel("Δ P(Murder) when node toggled 0 → 1 (else marginal / no other evidence)")
    ax.set_title("Single-node sensitivity (train BN): |Δ| ranked")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    if not BIF_PATH.exists():
        raise FileNotFoundError(f"Missing BIF: {BIF_PATH}")

    model = load_model_from_bif(BIF_PATH)
    baseline_p = p_murder(model, {})
    scenario_mean = load_scenario_mean_p_murder()

    # 1) Single-node
    df_single = run_single_node_sensitivity(model)
    OUT_SINGLE.parent.mkdir(parents=True, exist_ok=True)
    df_single.to_csv(OUT_SINGLE, index=False, encoding="utf-8")

    print("=== 1) Single-node sensitivity (ranked by |Δ P(Murder)|) ===")
    print(
        df_single[
            ["node", "P_Murder_given_node_0", "P_Murder_given_node_1", "delta_P_Murder", "direction_vs_node0"]
        ].to_string(index=False)
    )

    # 2) CPT perturbation
    df_cpt = run_cpt_perturbation(model, baseline_p)
    df_cpt.to_csv(OUT_CPT, index=False, encoding="utf-8")

    detail = df_cpt[df_cpt["row_type"] == "cpt_entry"].copy()
    means = df_cpt[df_cpt["row_type"] == "node_mean_abs_delta"].copy()

    top = detail.sort_values("max_abs_delta_P_Murder", ascending=False).head(15)
    print("\n=== 2) CPT perturbation: top 15 entries by max |Δ P(Murder)| (over ±10%/±20% multipliers) ===")
    print(
        top[
            [
                "cpt_node",
                "parent_config_index",
                "child_state_label",
                "baseline_prob",
                "best_multiplier",
                "max_abs_delta_P_Murder",
            ]
        ].to_string(index=False)
    )

    print("\n=== 2b) Mean |Δ P(Murder)| per CPT (over all perturbation trials) ===")
    print(means.sort_values("mean_abs_delta_P_Murder", ascending=False).to_string(index=False))

    # 3) Pathway
    df_path = run_pathway_sensitivity(model)
    print("\n=== 3) Pathway sensitivity (single parent evidence, else marginal) ===")
    print(df_path.to_string(index=False))

    # 4) Combined summary CSV + figure
    summary_parts: list[pd.DataFrame] = []

    meta = pd.DataFrame(
        [
            {"analysis": "metadata", "key": "BIF", "value": str(BIF_PATH)},
            {"analysis": "metadata", "key": "P_Murder_unconditional", "value": baseline_p},
            {
                "analysis": "metadata",
                "key": "mean_P_Murder_scenario_full_corpus",
                "value": scenario_mean if scenario_mean is not None else np.nan,
            },
        ]
    )
    summary_parts.append(meta)

    s1 = df_single.copy()
    s1.insert(0, "analysis", "single_node")
    summary_parts.append(s1)

    s3 = df_path.copy()
    s3.insert(0, "analysis", "pathway")
    summary_parts.append(s3)

    # Top CPT hits + means in summary
    top20 = detail.sort_values("max_abs_delta_P_Murder", ascending=False).head(20).copy()
    top20.insert(0, "analysis", "cpt_top_entry")
    summary_parts.append(top20)

    m2 = means.copy()
    m2.insert(0, "analysis", "cpt_node_mean")
    summary_parts.append(m2)

    # Align columns — use outer concat
    summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")

    plot_single_node_bars(df_single, FIG_PATH)

    print(f"\nSaved: {OUT_SINGLE}")
    print(f"Saved: {OUT_CPT}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {FIG_PATH}")


if __name__ == "__main__":
    main()
