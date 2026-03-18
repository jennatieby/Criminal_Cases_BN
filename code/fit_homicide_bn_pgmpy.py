#!/usr/bin/env python3
"""
fit_homicide_bn_pgmpy.py

Fit a Bayesian network (pgmpy) for homicide legal reasoning using:
- a binary case-node matrix at case_node_matrix.csv (DAG node columns + Verdict)
- a hand-defined edge dictionary (DAG structure)

Steps:
1) Instantiate BayesianNetwork using the edge dictionary
2) Fit CPTs using MaximumLikelihoodEstimator (baseline)
3) Apply Bayesian smoothing (BDeu/Dirichlet) with equivalent_sample_size=5
4) For each node, print its CPT and flag any parent-config columns with <10 observations as LOW_SUPPORT
5) Save the fitted model to homicide_bn.bif
6) Sanity check: query P(Verdict=Murder) unconditionally vs corpus base rate

Run:
  pip install pgmpy
  python code/fit_homicide_bn_pgmpy.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.readwrite import BIFWriter
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install pgmpy\n"
        f"Import error: {e}"
    )


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "case_node_matrix.csv"
OUT_BIF = ROOT / "homicide_bn.bif"

# Use the exact edge dictionary provided by the user
EDGES = {
    "VoluntaryAct": ["FactualCausation"],
    "DeathOfHumanBeing": ["UnlawfulKilling"],
    "FactualCausation": ["UnlawfulKilling", "LegalCausation"],
    "LegalCausation": ["UnlawfulKilling"],
    "UnlawfulKilling": ["Verdict"],
    "IntentToKill": ["MensReaIntent"],
    "IntentToCauseGBH": ["MensReaIntent"],
    "MensReaIntent": ["MaliceAforethought"],
    "MaliceAforethought": ["Verdict"],
    "LossOfControl": ["Defences"],
    "DiminishedResponsibility": ["Defences"],
    "Defences": ["Verdict"],
}

# IMPORTANT: Avoid spaces in BIF state names (BIFReader treats them as separators).
VERDICT_LABELS = {0: "Not_Guilty", 1: "Manslaughter", 2: "Murder"}


def edges_to_tuples(edges: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(p, c) for p, kids in edges.items() for c in kids]


def required_nodes(edges: dict[str, list[str]]) -> list[str]:
    s = set(edges.keys())
    for kids in edges.values():
        s.update(kids)
    return sorted(s)


def load_data() -> pd.DataFrame:
    if not MATRIX.exists():
        raise FileNotFoundError(f"Missing matrix: {MATRIX}")
    df = pd.read_csv(MATRIX, encoding="utf-8-sig")
    if "case_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "case_id"})

    nodes = required_nodes(EDGES)
    missing_cols = [c for c in nodes if c not in df.columns]
    if missing_cols:
        raise ValueError(f"case_node_matrix.csv missing required columns: {missing_cols}")

    # Keep only model columns (drop case_id)
    d = df[nodes].copy()

    # Normalize binary columns to {0,1} ints; Verdict to {0,1,2}
    for col in nodes:
        if col == "Verdict":
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Drop rows with missing verdict (pgmpy can't fit with NaNs)
    before = len(d)
    d = d.dropna(subset=["Verdict"]).copy()
    dropped = before - len(d)
    if dropped:
        print(f"[warn] Dropped {dropped} rows with NaN Verdict")

    # Enforce integer states
    for col in nodes:
        d[col] = d[col].round().astype(int)

    # Map to string states so pgmpy prints nicely and query uses labels
    for col in nodes:
        if col == "Verdict":
            d[col] = d[col].map(VERDICT_LABELS).astype(str)
        else:
            d[col] = d[col].map({0: "0", 1: "1"}).astype(str)

    return d


def parent_support_counts(data: pd.DataFrame, node: str, parents: list[str]) -> dict[tuple[str, ...], int]:
    """
    Return counts per parent-configuration tuple (in parents order).
    If no parents, returns {(): len(data)}.
    """
    if not parents:
        return {(): int(len(data))}
    grp = data.groupby(parents, dropna=False).size()
    out: dict[tuple[str, ...], int] = {}
    for idx, cnt in grp.items():
        if not isinstance(idx, tuple):
            idx = (idx,)
        out[tuple(str(x) for x in idx)] = int(cnt)
    return out


def print_cpd_with_support(model: DiscreteBayesianNetwork, data: pd.DataFrame, node: str, low_support_n: int = 10) -> None:
    cpd = model.get_cpds(node)
    parents = list(model.get_parents(node))
    parents_sorted = list(cpd.variables[1:])  # evidence order used by pgmpy in this CPD

    print("\n" + "=" * 80)
    print(f"NODE: {node}")
    print(f"PARENTS: {parents_sorted if parents_sorted else '[]'}")
    print(cpd)

    # Support counts per parent config
    support = parent_support_counts(data, node=node, parents=parents_sorted)
    if not parents_sorted:
        n = support.get((), 0)
        flag = " LOW_SUPPORT" if n < low_support_n else ""
        print(f"SUPPORT: {n}{flag}")
        return

    # Enumerate CPD columns in the same order as cpd.get_evidence() / cpd.cardinality
    ev_cards = cpd.cardinality[1:]
    ev_states = [cpd.state_names[p] for p in parents_sorted]

    # Build cartesian product of evidence states in the same order pgmpy uses
    # pgmpy CPD columns correspond to product over evidence states with last evidence varying fastest
    combos: list[tuple[str, ...]] = [()]
    for states in ev_states:
        combos = [c + (s,) for c in combos for s in states]

    # Print support per config and flag
    print("SUPPORT PER PARENT CONFIG:")
    for conf in combos:
        n = support.get(tuple(conf), 0)
        flag = " LOW_SUPPORT" if n < low_support_n else ""
        conf_str = ", ".join(f"{p}={v}" for p, v in zip(parents_sorted, conf))
        print(f"  ({conf_str}): {n}{flag}")


def main() -> None:
    data = load_data()

    # 1) Build structure
    model = DiscreteBayesianNetwork(edges_to_tuples(EDGES))

    # 2) Fit CPTs using MLE (baseline)
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # 3) Apply Bayesian parameter smoothing (BDeu / Dirichlet) with equivalent_sample_size=5
    # Re-fit CPDs using BayesianEstimator with BDeu prior.
    model.fit(
        data,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5,
    )

    # 4) Print CPT + support flags
    for node in model.nodes():
        print_cpd_with_support(model, data, node=node, low_support_n=10)

    # 5) Save to BIF
    writer = BIFWriter(model)
    writer.write_bif(str(OUT_BIF))
    print("\n" + "=" * 80)
    print(f"Wrote model to: {OUT_BIF}")

    # 6) Sanity check: P(Verdict=Murder) unconditional vs base rate
    infer = VariableElimination(model)
    q = infer.query(variables=["Verdict"], show_progress=False)
    p_murder = float(q.values[q.state_names["Verdict"].index("Murder")])
    base_rate = float((data["Verdict"] == "Murder").mean())
    print("\nSanity check:")
    print(f"  Model P(Verdict=Murder): {p_murder:.4f}")
    print(f"  Corpus base rate Murder:  {base_rate:.4f}")


if __name__ == "__main__":
    main()

