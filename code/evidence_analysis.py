#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader

ROOT = Path(__file__).resolve().parents[1]
BIF_PATH = ROOT / "outputs" / "homicide_bn_train.bif"
MATRIX_PATH = ROOT / "case_node_matrix.csv"
OUT_DIR = ROOT / "outputs"

OUT_LR = OUT_DIR / "likelihood_ratios_verdict.csv"
OUT_CR = OUT_DIR / "characteristic_ratios.csv"
OUT_INC = OUT_DIR / "incremental_effects.csv"
OUT_MI = OUT_DIR / "mutual_information.csv"
OUT_SUM = OUT_DIR / "evidence_analysis_summary.csv"

NON_VERDICT_NODES = [
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
]

PAIR_LIST = [
    ("UnlawfulKilling", "MaliceAforethought"),
    ("UnlawfulKilling", "Defences"),
    ("MaliceAforethought", "Defences"),
    ("IntentToKill", "MaliceAforethought"),
]

VERDICT_LABEL = "Murder"
EPS = 1e-15


def load_model(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if "Not Guilty" in txt:
        txt = txt.replace("Not Guilty", "Not_Guilty")
        tmp = path.with_suffix(".tmp_clean.bif")
        tmp.write_text(txt, encoding="utf-8")
        try:
            return BIFReader(str(tmp)).get_model()
        finally:
            tmp.unlink(missing_ok=True)
    return BIFReader(str(path)).get_model()


def get_state_names(model, var: str) -> list[str]:
    cpd = model.get_cpds(var)
    return list(cpd.state_names[var])


def prob_from_query(q, var: str, state: str) -> float:
    states = list(q.state_names[var])
    vals = q.values
    return float(vals[states.index(state)])


def p_murder(infer: VariableElimination, evidence: dict[str, str] | None = None) -> float:
    q = infer.query(variables=["Verdict"], evidence=evidence or {}, show_progress=False)
    return prob_from_query(q, "Verdict", VERDICT_LABEL)


def joint_prob_2var(infer: VariableElimination, x: str, y: str) -> tuple[np.ndarray, list[str], list[str]]:
    q = infer.query(variables=[x, y], show_progress=False)
    states_x = list(q.state_names[x])
    states_y = list(q.state_names[y])
    arr = np.asarray(q.values, dtype=float)
    return arr, states_x, states_y


def marginal_prob_from_cpt(cpd, wanted_state: str = "1") -> float:
    # Root nodes: unconditional is directly in CPT.
    evidence = cpd.get_evidence()
    if not evidence:
        states = list(cpd.state_names[cpd.variable])
        vals = np.asarray(cpd.values, dtype=float).reshape(-1)
        return float(vals[states.index(wanted_state)])
    # Non-root nodes have conditional CPTs; unconditional marginal is not directly in CPT.
    # Return NaN here; caller should compute from model joint/marginal distribution.
    return float("nan")


def safe_div(a: float, b: float) -> float:
    if abs(b) <= EPS:
        return float("nan")
    return float(a / b)


def main() -> None:
    if not BIF_PATH.exists():
        raise FileNotFoundError(f"Missing model: {BIF_PATH}")
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"Missing matrix: {MATRIX_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load once
    model = load_model(BIF_PATH)
    infer = VariableElimination(model)
    _ = pd.read_csv(MATRIX_PATH, encoding="utf-8-sig")  # Loaded as requested (for setup parity)

    # 1) Baseline
    p_base = p_murder(infer)
    print(f"Baseline P(V=Murder): {p_base:.6f}")
    print("Baseline done")

    # 2) Likelihood ratios P(V|node) / P(V)
    lr_rows = []
    for node in NON_VERDICT_NODES:
        p1 = p_murder(infer, {node: "1"})
        p0 = p_murder(infer, {node: "0"})
        lr_rows.append(
            {
                "node": node,
                "LR_present": safe_div(p1, p_base),
                "LR_absent": safe_div(p0, p_base),
            }
        )
    df_lr = pd.DataFrame(lr_rows)
    df_lr.to_csv(OUT_LR, index=False, encoding="utf-8")
    print("Likelihood ratios done")

    # 3) Characteristic ratios: P(node=1|Murder) / P(node=1)
    cr_rows = []
    for node in NON_VERDICT_NODES:
        arr, node_states, verdict_states = joint_prob_2var(infer, node, "Verdict")
        i1 = node_states.index("1")
        iv = verdict_states.index(VERDICT_LABEL)
        p_node_and_v = float(arr[i1, iv])
        p_v = float(arr[:, iv].sum())
        p_node = float(arr[i1, :].sum())
        p_node_given_v = safe_div(p_node_and_v, p_v)

        # Read direct marginal from CPT if root; otherwise use joint-derived marginal.
        cpd = model.get_cpds(node)
        p_node_cpt = marginal_prob_from_cpt(cpd, wanted_state="1")
        p_node_marginal = p_node_cpt if not np.isnan(p_node_cpt) else p_node

        cr_rows.append(
            {
                "node": node,
                "P_node_given_murder": p_node_given_v,
                "P_node_marginal": p_node_marginal,
                "CR": safe_div(p_node_given_v, p_node_marginal),
            }
        )
    df_cr = pd.DataFrame(cr_rows)
    df_cr.to_csv(OUT_CR, index=False, encoding="utf-8")
    print("Characteristic ratios done")

    # 4) Incremental effects (VE with evidence)
    inc_rows = []
    for e1, e2 in PAIR_LIST:
        a = p_murder(infer, {e1: "1", e2: "1"})
        b = p_murder(infer, {e1: "1"})
        c = p_murder(infer, {e2: "1"})
        d = p_base
        inc_rows.append(
            {
                "E1": e1,
                "E2": e2,
                "P_both": a,
                "P_E1_only": b,
                "P_E2_only": c,
                "P_baseline": d,
                "joint_LR": safe_div(a, d),
                "incremental_given_E1": safe_div(a, b),
                "incremental_given_E2": safe_div(a, c),
            }
        )
    df_inc = pd.DataFrame(inc_rows)
    df_inc.to_csv(OUT_INC, index=False, encoding="utf-8")
    print("Incremental effects done")

    # 5) Mutual information I(node; Verdict)
    mi_rows = []
    for node in NON_VERDICT_NODES:
        arr, node_states, verdict_states = joint_prob_2var(infer, node, "Verdict")
        p_node = arr.sum(axis=1)
        p_ver = arr.sum(axis=0)
        mi = 0.0
        for i, _ in enumerate(node_states):
            for j, _ in enumerate(verdict_states):
                p_xy = float(arr[i, j])
                if p_xy <= 0:
                    continue
                denom = float(p_node[i] * p_ver[j])
                if denom <= 0:
                    continue
                mi += p_xy * np.log2(p_xy / denom)
        mi_rows.append({"node": node, "mutual_information": float(mi)})
    df_mi = pd.DataFrame(mi_rows).sort_values("mutual_information", ascending=False).reset_index(drop=True)
    df_mi.to_csv(OUT_MI, index=False, encoding="utf-8")
    print("Mutual information done")

    # 6) Summary table with MI rank + LR_present + CR
    df_sum = (
        df_mi.merge(df_lr[["node", "LR_present"]], on="node", how="left")
        .merge(df_cr[["node", "CR"]], on="node", how="left")
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )
    df_sum.to_csv(OUT_SUM, index=False, encoding="utf-8")

    print("\n=== Evidence analysis summary (ranked by MI) ===")
    print(df_sum.to_string(index=False))
    print(f"\nSaved: {OUT_LR}")
    print(f"Saved: {OUT_CR}")
    print(f"Saved: {OUT_INC}")
    print(f"Saved: {OUT_MI}")
    print(f"Saved: {OUT_SUM}")


if __name__ == "__main__":
    main()
