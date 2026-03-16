#!/usr/bin/env python3
"""
design_bn_dag.py

Hand-crafted Bayesian network DAG for modelling murder under the law of
England and Wales (Coke's definition: unlawful killing of a human being
under the King's peace with malice aforethought).

This script:
- Defines the DAG as a parent -> children edge dictionary.
- Builds a NetworkX DiGraph and validates that it is acyclic.
- Prints the edge dictionary and parents of key nodes.
- Renders a simple visualisation using matplotlib.

Run from repo root:
    python code/design_bn_dag.py
"""

from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# DAG structure (revised)
# ---------------------------------------------------------------------------

EDGES = {
    # Actus reus chain
    "VoluntaryAct": ["FactualCausation"],
    # DeathOfHumanBeing is the condition against which factual causation is assessed
    "DeathOfHumanBeing": ["FactualCausation", "UnlawfulKilling"],
    "FactualCausation": ["UnlawfulKilling", "LegalCausation"],
    "LegalCausation": ["UnlawfulKilling"],
    # UnlawfulKilling now only feeds Verdict (parallel to MaliceAforethought)
    "UnlawfulKilling": ["Verdict"],

    # Mens rea chain
    "IntentToKill": ["MensReaIntent"],
    "IntentToCauseGBH": ["MensReaIntent"],
    "MensReaIntent": ["MaliceAforethought"],
    # MaliceAforethought is purely mental; no UnlawfulKilling parent
    "MaliceAforethought": ["Verdict"],

    # Partial defences
    "LossOfControl": ["Defences"],
    "DiminishedResponsibility": ["Defences"],
    "Defences": ["Verdict"],
}


def build_dag() -> nx.DiGraph:
    """Build the NetworkX DiGraph from EDGES and validate acyclicity."""
    g = nx.DiGraph()
    for parent, children in EDGES.items():
        for child in children:
            g.add_edge(parent, child)

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("The constructed graph is not acyclic")
    return g


def print_structure(g: nx.DiGraph) -> None:
    """Print edges and parents of key nodes."""
    print("Revised edges (parent -> children):")
    for parent, children in EDGES.items():
        print(f"  {parent} -> {children}")

    print("\nGraph is a DAG (acyclic).")
    print("\nParents of MaliceAforethought:", list(g.predecessors("MaliceAforethought")))
    print("Parents of Verdict:", list(g.predecessors("Verdict")))


def draw_dag(g: nx.DiGraph) -> None:
    """Render a DAG visualisation using a fixed manual hierarchical layout."""
    plt.figure(figsize=(18, 12))

    # Manually specified positions (top-down hierarchy)
    pos = {
        "VoluntaryAct":             (-4, 4),
        "DeathOfHumanBeing":        (-2, 4),
        "IntentToKill":             (2, 4),
        "IntentToCauseGBH":         (4, 4),
        "LossOfControl":            (6, 4),
        "DiminishedResponsibility": (8, 4),
        "FactualCausation":         (-3, 2),
        "MensReaIntent":            (3, 2),
        "LegalCausation":           (-4, 0),
        "MaliceAforethought":       (3, 0),
        "UnlawfulKilling":          (-2, -2),
        "Defences":                 (2, -2),
        "Verdict":                  (0, -4),
    }

    nx.draw(
        g,
        pos,
        with_labels=True,
        node_size=3500,
        node_color="#ddeeff",
        font_size=9,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
    )

    plt.title("Bayesian Network: Murder under English and Welsh Law")
    plt.show()


def main() -> None:
    g = build_dag()
    print_structure(g)
    draw_dag(g)


if __name__ == "__main__":
    main()

