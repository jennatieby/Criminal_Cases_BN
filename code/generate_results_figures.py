#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import itertools
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"

SCEN_FULL = ROOT / "outputs" / "scenario_results_full.csv"
SCEN_FALLBACK = ROOT / "outputs" / "scenario_results.csv"
CONSISTENCY = ROOT / "outputs" / "consistency_results.csv"
CALIBRATION = ROOT / "outputs" / "calibration_results.csv"
MATRIX = ROOT / "case_node_matrix.csv"
BIF = ROOT / "outputs" / "homicide_bn_train.bif"

ORDERED_LABELS = ["Murder", "Manslaughter", "Not Guilty"]
NAVY = "#1f3b6d"
LIGHT_BLUE = "#dce9f8"

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


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_table_png(df: pd.DataFrame, out_path: Path, title: str, red_rows: Iterable[int] | None = None, green_all_rows: bool = False) -> None:
    fig_h = max(2.5, 0.45 * (len(df) + 2))
    fig_w = max(8.0, 1.2 * len(df.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    # header style
    for c in range(len(df.columns)):
        cell = table[0, c]
        cell.set_facecolor("#dce9f8")
        cell.set_text_props(weight="bold", color="#102a43")

    red_rows = set(red_rows or [])
    for r in range(len(df)):
        row_color = None
        if r in red_rows:
            row_color = "#fde2e2"
        elif green_all_rows:
            row_color = "#e6f7ec"
        if row_color:
            for c in range(len(df.columns)):
                table[r + 1, c].set_facecolor(row_color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def normalize_verdict_label(x: object) -> str:
    s = str(x)
    if s == "Not_Guilty":
        return "Not Guilty"
    return s


def parse_verdict_cpt_from_bif(bif_path: Path) -> pd.DataFrame:
    txt = bif_path.read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace("Not Guilty", "Not_Guilty")
    m = re.search(r"probability\s*\(\s*Verdict\s*\|\s*([^)]+?)\s*\)\s*\{(.*?)\}", txt, flags=re.S)
    if not m:
        raise ValueError("Could not locate Verdict CPT block in BIF.")
    parent_order = [p.strip() for p in m.group(1).split(",")]
    block = m.group(2)
    rows = re.findall(r"\(\s*([01])\s*,\s*([01])\s*,\s*([01])\s*\)\s*([0-9eE\.\-\+]+)\s*,\s*([0-9eE\.\-\+]+)\s*,\s*([0-9eE\.\-\+]+)\s*;", block)
    if len(rows) != 8:
        raise ValueError("Unexpected number of Verdict CPT rows.")
    if set(parent_order) != {"UnlawfulKilling", "MaliceAforethought", "Defences"}:
        raise ValueError(f"Unexpected Verdict parent set in BIF: {parent_order}")

    data = []
    for a, b, c, p1, p2, p3 in rows:
        # Verdict states in this BIF are: Manslaughter, Murder, Not_Guilty.
        p_mans = float(p1)
        p_murder = float(p2)
        p_ng = float(p3)
        vals = {parent_order[0]: int(a), parent_order[1]: int(b), parent_order[2]: int(c)}
        data.append(
            {
                "UnlawfulKilling": vals["UnlawfulKilling"],
                "MaliceAforethought": vals["MaliceAforethought"],
                "Defences": vals["Defences"],
                "P(Murder)": p_murder,
                "P(Manslaughter)": p_mans,
                "P(Not Guilty)": p_ng,
            }
        )
    return pd.DataFrame(data)


def get_scenario_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    full_src = pd.read_csv(SCEN_FULL, encoding="utf-8-sig")
    if "scenario" in full_src.columns:
        full = full_src[full_src["scenario"] == "FULL_EVIDENCE"].copy()
    else:
        full = full_src.copy()
    if full.empty:
        raise ValueError("scenario_results_full.csv has no FULL_EVIDENCE rows.")

    need = {
        "PARTIAL_EVIDENCE_NO_CAUSATION",
        "COUNTERFACTUAL_BASELINE",
        "COUNTERFACTUAL_NO_INTENT_STRONG",
        "FULL_EVIDENCE",
    }
    if "scenario" in full_src.columns and need.issubset(set(full_src["scenario"].dropna().unique())):
        scen_all = full_src
    elif SCEN_FALLBACK.exists():
        scen_all = pd.read_csv(SCEN_FALLBACK, encoding="utf-8-sig")
    else:
        scen_all = full.copy()
    return full, scen_all


def fig1_confusion_matrix(df_full: pd.DataFrame) -> None:
    d = df_full.copy()
    d["actual"] = d["actual_verdict"].map(normalize_verdict_label)
    d["pred"] = d["MAP_verdict"].map(normalize_verdict_label)
    cm = pd.crosstab(
        pd.Categorical(d["actual"], categories=ORDERED_LABELS, ordered=True),
        pd.Categorical(d["pred"], categories=ORDERED_LABELS, ordered=True),
        dropna=False,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(3), ORDERED_LABELS)
    ax.set_yticks(range(3), ORDERED_LABELS)
    ax.set_xlabel("MAP_verdict")
    ax.set_ylabel("actual_verdict")
    n = len(d)
    ax.set_title(f"Confusion Matrix: MAP Verdict vs Actual Verdict (n={n})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", color="#0b1f3a", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = OUT_DIR / "fig1_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def table1_verdict_cpt() -> None:
    df = parse_verdict_cpt_from_bif(BIF)
    df = df.sort_values("P(Murder)", ascending=False).reset_index(drop=True)
    csv_out = OUT_DIR / "table1_verdict_cpt.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

    png_df = df.copy()
    for c in ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]:
        png_df[c] = png_df[c].map(lambda x: f"{x:.4f}")
    save_table_png(
        png_df,
        OUT_DIR / "table1_verdict_cpt.png",
        "Verdict CPT: Murder Posterior Across All Parent Configurations",
        green_all_rows=True,
    )


def draw_dsep_figure(title: str, out_name: str, red_nodes: set[str], orange_nodes: set[str]) -> None:
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_title(title, pad=14)
    ax.axis("off")

    all_edges = [(p, c) for p, cs in EDGES.items() for c in cs]
    for u, v in all_edges:
        x1, y1 = POS[u]
        x2, y2 = POS[v]
        color = "#7f8a99"
        linestyle = "-"
        lw = 2.0
        if u in orange_nodes or v in orange_nodes:
            color = "#f59e0b"
            linestyle = "--"
            lw = 2.4
        if u in red_nodes or v in red_nodes:
            color = "#d62728"
            linestyle = "--"
            lw = 2.4
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=linestyle, shrinkA=12, shrinkB=12),
        )

    for n, (x, y) in POS.items():
        if n in red_nodes:
            fc = "#ef4444"
            ec = "#8b1d1d"
        elif n in orange_nodes:
            fc = "#f59e0b"
            ec = "#8a5800"
        else:
            fc = LIGHT_BLUE
            ec = NAVY
        circ = plt.Circle((x, y), 0.55, facecolor=fc, edgecolor=ec, linewidth=1.7)
        ax.add_patch(circ)
        ax.text(x, y, n, ha="center", va="center", fontsize=9)

    handles = [
        Patch(facecolor="#ef4444", edgecolor="#8b1d1d", label="red nodes = removed from evidence / excluded mediator"),
        Patch(facecolor="#f59e0b", edgecolor="#8a5800", label="orange nodes = intent nodes forced to 0"),
        Line2D([0], [0], color="#d62728", linestyle="--", lw=2.4, label="dashed red = blocked/affected pathway"),
        Line2D([0], [0], color="#f59e0b", linestyle="--", lw=2.4, label="dashed orange = manipulated intent pathway"),
        Line2D([0], [0], color="#7f8a99", linestyle="-", lw=2.0, label="solid grey = other edges"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)
    ax.set_xlim(-5.5, 9.5)
    ax.set_ylim(-5.5, 5.2)

    out = OUT_DIR / out_name
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def table2_causation_deltas(df_all: pd.DataFrame) -> None:
    part = df_all[df_all["scenario"] == "PARTIAL_EVIDENCE_NO_CAUSATION"].copy()
    full = df_all[df_all["scenario"] == "FULL_EVIDENCE"].copy()
    if part.empty or full.empty:
        raise ValueError("Missing PARTIAL_EVIDENCE_NO_CAUSATION or FULL_EVIDENCE rows for Table 2.")
    merged = part.merge(
        full[["case_id", "P(Murder)", "P(Manslaughter)"]].rename(
            columns={"P(Murder)": "P(Murder)_full", "P(Manslaughter)": "P(Manslaughter)_full"}
        ),
        on="case_id",
        how="left",
    )
    out = pd.DataFrame(
        {
            "case_id": merged["case_id"],
            "P(Murder)_full": merged["P(Murder)_full"],
            "P(Murder)_partial": merged["P(Murder)"],
            "delta_P(Murder)": merged["delta_P(Murder)"],
            "P(Manslaughter)_full": merged["P(Manslaughter)_full"],
            "P(Manslaughter)_partial": merged["P(Manslaughter)"],
            "delta_P(Manslaughter)": merged["delta_P(Manslaughter)"],
            "actual_verdict": merged["actual_verdict"].map(normalize_verdict_label),
        }
    )
    out["abs_delta"] = out["delta_P(Murder)"].abs()
    out = out.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"]).reset_index(drop=True)

    csv_out = OUT_DIR / "table2_causation_deltas.csv"
    out.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

    png_df = out.copy()
    for c in ["P(Murder)_full", "P(Murder)_partial", "delta_P(Murder)", "P(Manslaughter)_full", "P(Manslaughter)_partial", "delta_P(Manslaughter)"]:
        png_df[c] = png_df[c].map(lambda x: f"{x:.4f}")
    save_table_png(
        png_df,
        OUT_DIR / "table2_causation_deltas.png",
        "Verdict Posterior Shifts: Disputed Causation Scenario (n=20)",
    )


def fig4_counterfactual_bars(df_all: pd.DataFrame) -> None:
    b = df_all[df_all["scenario"] == "COUNTERFACTUAL_BASELINE"].copy()
    c = df_all[df_all["scenario"] == "COUNTERFACTUAL_NO_INTENT_STRONG"].copy()
    if b.empty or c.empty:
        raise ValueError("Missing COUNTERFACTUAL_BASELINE or COUNTERFACTUAL_NO_INTENT_STRONG rows for Figure 4.")
    m = b.merge(c, on="case_id", suffixes=("_base", "_no"))
    probs = ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]
    means_base = [m[f"{p}_base"].mean() for p in probs]
    means_no = [m[f"{p}_no"].mean() for p in probs]
    std_base = [m[f"{p}_base"].std(ddof=1) for p in probs]
    std_no = [m[f"{p}_no"].std(ddof=1) for p in probs]

    x = np.arange(len(probs))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, means_base, width, yerr=std_base, color=NAVY, alpha=0.95, label="Baseline")
    ax.bar(x + width / 2, means_no, width, yerr=std_no, color="#cbd5e1", edgecolor="#64748b", label="No-intent")
    ax.set_xticks(x, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Mean Verdict Posteriors: Counterfactual Intent Manipulation (n=10)")
    ax.legend()
    delta_m = means_no[0] - means_base[0]
    ax.text(0.02, 0.95, f"Mean delta P(Murder): {delta_m:.4f}", transform=ax.transAxes, va="top", fontsize=10, color="#102a43")
    fig.tight_layout()
    out = OUT_DIR / "fig4_counterfactual_bars.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def fig5_consistency_js(cons: pd.DataFrame, matrix: pd.DataFrame) -> None:
    c = cons[np.isclose(cons["threshold"].astype(float), 0.6)].copy()
    if c.empty:
        raise ValueError("No threshold=0.60 rows in consistency_results.csv")
    g = c.groupby("group_id").agg(mean_JS=("js_divergence", "mean"), n_pairs=("js_divergence", "size")).reset_index()

    # derive group size from unique case IDs in pairs
    size_map = {}
    for gid, gg in c.groupby("group_id"):
        ids = set(gg["case_id_a"]).union(set(gg["case_id_b"]))
        size_map[gid] = len(ids)
    g["group_size"] = g["group_id"].map(size_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(g["group_id"].astype(str), g["mean_JS"], color="#2d5b95")
    ax.axhline(0.10, color="#d62728", linestyle="--", linewidth=1.6)
    ax.set_ylabel("Mean Jensen-Shannon Divergence")
    ax.set_xlabel("Group ID")
    ax.set_title("Within-Group Jensen-Shannon Divergence (Jaccard threshold = 0.60)")
    for rect, n in zip(bars, g["group_size"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.001, f"n={int(n)}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig5_consistency_js.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def table3_consistency_summary(cons: pd.DataFrame) -> None:
    grp = cons.groupby(["threshold", "group_id"]).agg(mean_JS=("js_divergence", "mean"), max_JS=("js_divergence", "max"), n_pairs=("js_divergence", "size")).reset_index()
    size_rows = []
    for (thr, gid), gg in cons.groupby(["threshold", "group_id"]):
        ids = set(gg["case_id_a"]).union(set(gg["case_id_b"]))
        size_rows.append({"threshold": thr, "group_id": gid, "group_size": len(ids)})
    sizes = pd.DataFrame(size_rows)
    out = grp.merge(sizes, on=["threshold", "group_id"], how="left")
    out["flag"] = np.where(out["mean_JS"] > 0.1, "INCONSISTENT", "OK")
    out = out[["threshold", "group_id", "group_size", "n_pairs", "mean_JS", "max_JS", "flag"]].sort_values(["threshold", "group_id"]).reset_index(drop=True)

    csv_out = OUT_DIR / "table3_consistency_summary.csv"
    out.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

    png_df = out.copy()
    png_df["mean_JS"] = png_df["mean_JS"].map(lambda x: f"{x:.6f}")
    png_df["max_JS"] = png_df["max_JS"].map(lambda x: f"{x:.6f}")
    red_rows = list(np.where(out["flag"] == "INCONSISTENT")[0])
    save_table_png(
        png_df,
        OUT_DIR / "table3_consistency_summary.png",
        "Consistency Evaluation: Group Summary",
        red_rows=red_rows,
    )


def reliability_plot_for_class(bin_df: pd.DataFrame, class_name: str, title: str, out_name: str) -> None:
    d = bin_df[bin_df["class"] == class_name].copy()
    d = d[d["n_cases"].fillna(0) > 0].copy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.5)
    ax.plot(d["mean_pred"], d["obs_freq"], marker="o", color=NAVY, linewidth=2)
    for _, r in d.iterrows():
        ax.annotate(f"n={int(r['n_cases'])}", (r["mean_pred"], r["obs_freq"]), textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def table4_brier_ece(cal: pd.DataFrame) -> None:
    m = cal[cal["row_type"] == "metric"].copy()
    if m.empty:
        raise ValueError("calibration_results.csv missing metric rows for table 4.")

    def val(metric: str) -> float:
        sub = m[m["metric"] == metric]
        return float(sub["value"].iloc[0]) if len(sub) else float("nan")

    def interp(metric: str) -> str:
        sub = m[m["metric"] == metric]
        return str(sub["interpretation"].iloc[0]) if len(sub) and pd.notna(sub["interpretation"].iloc[0]) else ""

    out = pd.DataFrame(
        [
            {"class": "Murder", "brier_score": val("brier_murder"), "interpretation": "moderate"},
            {"class": "Manslaughter", "brier_score": val("brier_manslaughter"), "interpretation": "moderate"},
            {"class": "Not Guilty", "brier_score": val("brier_not_guilty"), "interpretation": "moderate"},
            {"class": "Overall multiclass", "brier_score": val("brier_multiclass_overall"), "interpretation": "poor"},
            {"class": "ECE Murder", "brier_score": val("ece_murder"), "interpretation": interp("ece_murder") or "—"},
        ]
    )
    csv_out = OUT_DIR / "table4_brier_ece.csv"
    out.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")
    png_df = out.copy()
    png_df["brier_score"] = png_df["brier_score"].map(lambda x: f"{x:.4f}")
    save_table_png(
        png_df,
        OUT_DIR / "table4_brier_ece.png",
        "Calibration Metrics Summary",
    )


def summary_files() -> list[Path]:
    return sorted(
        [
            OUT_DIR / "fig1_confusion_matrix.png",
            OUT_DIR / "table1_verdict_cpt.csv",
            OUT_DIR / "table1_verdict_cpt.png",
            OUT_DIR / "fig2_dsep_causation.png",
            OUT_DIR / "fig3_dsep_intent.png",
            OUT_DIR / "table2_causation_deltas.csv",
            OUT_DIR / "table2_causation_deltas.png",
            OUT_DIR / "fig4_counterfactual_bars.png",
            OUT_DIR / "fig5_consistency_js.png",
            OUT_DIR / "table3_consistency_summary.csv",
            OUT_DIR / "table3_consistency_summary.png",
            OUT_DIR / "fig6_reliability_murder.png",
            OUT_DIR / "fig7_reliability_manslaughter.png",
            OUT_DIR / "fig8_reliability_notguilty.png",
            OUT_DIR / "table4_brier_ece.csv",
            OUT_DIR / "table4_brier_ece.png",
        ]
    )


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_full, df_all = get_scenario_df()
    cons = pd.read_csv(CONSISTENCY, encoding="utf-8-sig")
    cal = pd.read_csv(CALIBRATION, encoding="utf-8-sig")
    matrix = pd.read_csv(MATRIX, encoding="utf-8-sig")

    fig1_confusion_matrix(df_full)
    table1_verdict_cpt()

    draw_dsep_figure(
        title="D-Separation: Disputed Causation Scenario",
        out_name="fig2_dsep_causation.png",
        red_nodes={"FactualCausation", "LegalCausation", "UnlawfulKilling"},
        orange_nodes=set(),
    )
    draw_dsep_figure(
        title="D-Separation: Counterfactual Intent Scenario",
        out_name="fig3_dsep_intent.png",
        red_nodes={"MensReaIntent", "MaliceAforethought"},
        orange_nodes={"IntentToKill", "IntentToCauseGBH"},
    )

    table2_causation_deltas(df_all)
    fig4_counterfactual_bars(df_all)

    fig5_consistency_js(cons, matrix)
    table3_consistency_summary(cons)

    bins = cal[cal["row_type"] == "reliability_bin"].copy()
    reliability_plot_for_class(bins, "Murder", "Reliability Diagram: Murder Class", "fig6_reliability_murder.png")
    reliability_plot_for_class(bins, "Manslaughter", "Reliability Diagram: Manslaughter Class", "fig7_reliability_manslaughter.png")
    reliability_plot_for_class(bins, "Not_Guilty", "Reliability Diagram: Not Guilty Class", "fig8_reliability_notguilty.png")

    table4_brier_ece(cal)

    print("\nCreated files summary:")
    created = summary_files()
    for p in created:
        if p.exists():
            kb = p.stat().st_size / 1024.0
            print(f"- {p} ({kb:.1f} KB)")
        else:
            print(f"- MISSING: {p}")


if __name__ == "__main__":
    main()

