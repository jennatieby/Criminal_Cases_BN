#!/usr/bin/env python3
"""
CASE_00895 network diagram: observed binary states, full vs partial causation posteriors.
Output: outputs/figures/fig_case_network_example.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "case_node_matrix.csv"
SCEN = ROOT / "outputs" / "scenario_results_full.csv"
OUT = ROOT / "outputs" / "figures" / "fig_case_network_example.png"

CASE_ID = "CASE_00895"

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

POS = {
    "VoluntaryAct": (-4, 4),
    "DeathOfHumanBeing": (-2, 4),
    "IntentToKill": (2, 4),
    "IntentToCauseGBH": (4, 4),
    "LossOfControl": (6, 4),
    "DiminishedResponsibility": (8, 4),
    "FactualCausation": (-3, 2),
    "MensReaIntent": (3, 2),
    "LegalCausation": (-4, 0),
    "MaliceAforethought": (3, 0),
    "UnlawfulKilling": (-2, -2),
    "Defences": (2, -2),
    "Verdict": (0, -4),
}

NON_VERDICT = [
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

REMOVED = {"FactualCausation", "LegalCausation", "UnlawfulKilling"}

NAVY = "#1f3b6d"
LIGHT_BLUE = "#dce9f8"
RED_FILL = "#ef4444"
RED_EDGE = "#8b1d1d"


def main() -> None:
    mat = pd.read_csv(MATRIX, encoding="utf-8-sig")
    if "case_id" not in mat.columns:
        mat = mat.rename(columns={mat.columns[0]: "case_id"})
    mat["case_id"] = mat["case_id"].astype(str)
    row = mat[mat["case_id"] == CASE_ID]
    if row.empty:
        raise ValueError(f"{CASE_ID} not in matrix")
    row = row.iloc[0]

    scen = pd.read_csv(SCEN, encoding="utf-8-sig")
    scen["case_id"] = scen["case_id"].astype(str)
    full = scen[(scen["case_id"] == CASE_ID) & (scen["scenario"] == "FULL_EVIDENCE")].iloc[0]
    part = scen[(scen["case_id"] == CASE_ID) & (scen["scenario"] == "PARTIAL_EVIDENCE_NO_CAUSATION")].iloc[0]

    pm_f, pms_f, png_f = float(full["P(Murder)"]), float(full["P(Manslaughter)"]), float(full["P(Not Guilty)"])
    pm_p, pms_p, png_p = float(part["P(Murder)"]), float(part["P(Manslaughter)"]), float(part["P(Not Guilty)"])
    delta_m = float(part["delta_P(Murder)"])

    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")
    ax.set_title(
        f"{CASE_ID}: disputed causation scenario (ΔP(Murder) = {delta_m:+.4f}; largest positive in sample)",
        pad=14,
        fontsize=12,
    )

    all_edges = [(p, c) for p, cs in EDGES.items() for c in cs]
    for u, v in all_edges:
        x1, y1 = POS[u]
        x2, y2 = POS[v]
        color = "#7f8a99"
        linestyle = "-"
        lw = 2.0
        if u in REMOVED or v in REMOVED:
            color = "#d62728"
            linestyle = "--"
            lw = 2.4
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=linestyle, shrinkA=12, shrinkB=12),
        )

    for n in NON_VERDICT:
        x, y = POS[n]
        val = int(round(float(row[n])))
        removed = n in REMOVED

        if removed:
            ec = RED_EDGE
            lw = 2.2
            if val == 1:
                fc = RED_FILL
            else:
                fc = "#ffffff"
        else:
            ec = NAVY
            lw = 1.7
            fc = LIGHT_BLUE if val == 1 else "#ffffff"

        circ = plt.Circle((x, y), 0.55, facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(circ)
        ax.text(x, y, n, ha="center", va="center", fontsize=8)
        ax.text(x, y - 0.72, f"={val}", ha="center", va="top", fontsize=7, color="#334155")

    vx, vy = POS["Verdict"]
    circ_v = plt.Circle((vx, vy), 0.75, facecolor="#e0e7ef", edgecolor=NAVY, linewidth=2.0)
    ax.add_patch(circ_v)
    ax.text(vx, vy + 0.15, "Verdict", ha="center", va="center", fontsize=10, fontweight="bold")

    verdict_box = (
        f"FULL evidence:\n"
        f"  P(Murder)={pm_f:.3f}  P(Manslaughter)={pms_f:.3f}\n"
        f"  P(Not Guilty)={png_f:.3f}\n\n"
        f"PARTIAL (causation removed):\n"
        f"  P(Murder)={pm_p:.3f}  P(Manslaughter)={pms_p:.3f}\n"
        f"  P(Not Guilty)={png_p:.3f}"
    )
    ax.text(
        vx,
        vy - 0.42,
        verdict_box,
        ha="center",
        va="top",
        fontsize=8,
        family="monospace",
        linespacing=1.25,
    )

    handles = [
        Patch(facecolor=LIGHT_BLUE, edgecolor=NAVY, label="observed = 1 (present)"),
        Patch(facecolor="#ffffff", edgecolor=NAVY, label="observed = 0 (absent)"),
        Patch(facecolor="#ffffff", edgecolor=RED_EDGE, linewidth=2, label="removed under disputed causation (0/1 as observed)"),
        Line2D([0], [0], color="#d62728", linestyle="--", lw=2.4, label="edges touching removed causation subgraph"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True)
    ax.set_xlim(-5.5, 9.5)
    ax.set_ylim(-6.2, 5.2)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
