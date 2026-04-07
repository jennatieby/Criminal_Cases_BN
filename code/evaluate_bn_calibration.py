#!/usr/bin/env python3
"""
Evaluate calibration of full-corpus BN inference results.

Input:
  - outputs/scenario_results_full.csv (FULL_EVIDENCE rows)

Outputs:
  - outputs/reliability_diagram.png
  - outputs/reliability_diagram_manslaughter.png
  - outputs/reliability_diagram_notguilty.png
  - outputs/calibration_results.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "outputs" / "scenario_results_full.csv"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_MURDER_PLOT = FIG_DIR / "fig6_reliability_murder.png"
OUT_MANSLAUGHTER_PLOT = FIG_DIR / "fig7_reliability_manslaughter.png"
OUT_NOTGUILTY_PLOT = FIG_DIR / "fig8_reliability_notguilty.png"
OUT_RESULTS = ROOT / "outputs" / "calibration_results.csv"

N_BINS = 10


def canonical_label(x: object) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s in {"Murder", "murder"}:
        return "Murder"
    if s in {"Manslaughter", "manslaughter"}:
        return "Manslaughter"
    if s in {"Not_Guilty", "Not Guilty", "not_guilty", "not guilty"}:
        return "Not_Guilty"
    return None


def build_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "actual_verdict" in out.columns:
        labels = out["actual_verdict"].map(canonical_label)
    elif "Verdict" in out.columns:
        # numeric fallback if present
        mapper = {2: "Murder", 1: "Manslaughter", 0: "Not_Guilty"}
        labels = out["Verdict"].map(lambda v: mapper.get(int(round(float(v)))) if pd.notna(v) else None)
    else:
        raise ValueError("Missing verdict truth source: expected 'actual_verdict' (or 'Verdict').")

    out["actual_label"] = labels
    out = out[out["actual_label"].notna()].copy()
    if out.empty:
        raise ValueError("No rows with valid actual verdict labels.")

    out["y_Murder"] = (out["actual_label"] == "Murder").astype(int)
    out["y_Manslaughter"] = (out["actual_label"] == "Manslaughter").astype(int)
    out["y_Not_Guilty"] = (out["actual_label"] == "Not_Guilty").astype(int)
    return out


def make_bin_stats(p: pd.Series, y: pd.Series, class_name: str) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    # include 1.0 in last bin
    binned = pd.cut(p, bins=bins, include_lowest=True, right=True)
    tmp = pd.DataFrame({"p": p, "y": y, "bin": binned})
    g = tmp.groupby("bin", observed=False)
    stats = g.agg(
        mean_pred=("p", "mean"),
        obs_freq=("y", "mean"),
        n_cases=("y", "size"),
    ).reset_index()

    # readable bin labels
    labels = []
    for i in range(N_BINS):
        lo = bins[i]
        hi = bins[i + 1]
        labels.append(f"{lo:.1f}-{hi:.1f}")
    stats["bin_label"] = labels
    stats["class"] = class_name
    return stats[["class", "bin_label", "mean_pred", "obs_freq", "n_cases"]]


def plot_reliability(stats: pd.DataFrame, class_name: str, out_path: Path) -> None:
    s = stats.copy()
    s = s[s["n_cases"] > 0].copy()
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(s["mean_pred"], s["obs_freq"], marker="o", linewidth=1.5, label=class_name)
    for _, r in s.iterrows():
        plt.annotate(
            f"n={int(r['n_cases'])}",
            (float(r["mean_pred"]), float(r["obs_freq"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(f"Mean predicted P({class_name})")
    plt.ylabel(f"Observed frequency of {class_name}")
    plt.title(f"Reliability Diagram: {class_name}")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def brier_binary(p: pd.Series, y: pd.Series) -> float:
    return float(np.mean((p.to_numpy(dtype=float) - y.to_numpy(dtype=float)) ** 2))


def brier_multiclass(p_mat: np.ndarray, y_mat: np.ndarray) -> float:
    # average squared error across classes and samples
    return float(np.mean(np.sum((p_mat - y_mat) ** 2, axis=1)))


def compute_ece(stats: pd.DataFrame) -> float:
    # weighted absolute calibration gap
    n_total = float(stats["n_cases"].sum())
    if n_total <= 0:
        return float("nan")
    ece = float(np.sum((stats["n_cases"] / n_total) * np.abs(stats["mean_pred"] - stats["obs_freq"])))
    return ece


def ece_label(ece: float) -> str:
    if np.isnan(ece):
        return "unknown"
    if ece < 0.05:
        return "well calibrated"
    if ece <= 0.15:
        return "moderate"
    return "poorly calibrated"


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input: {IN_CSV}")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    if "scenario" in df.columns:
        df = df[df["scenario"] == "FULL_EVIDENCE"].copy()
    if df.empty:
        raise ValueError("No FULL_EVIDENCE rows found in scenario results.")

    required_prob_cols = ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]
    missing = [c for c in required_prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")

    df = build_truth_columns(df)

    p_m = df["P(Murder)"].astype(float)
    p_ms = df["P(Manslaughter)"].astype(float)
    p_ng = df["P(Not Guilty)"].astype(float)

    y_m = df["y_Murder"].astype(int)
    y_ms = df["y_Manslaughter"].astype(int)
    y_ng = df["y_Not_Guilty"].astype(int)

    stats_m = make_bin_stats(p_m, y_m, "Murder")
    stats_ms = make_bin_stats(p_ms, y_ms, "Manslaughter")
    stats_ng = make_bin_stats(p_ng, y_ng, "Not_Guilty")

    plot_reliability(stats_m, "Murder", OUT_MURDER_PLOT)
    plot_reliability(stats_ms, "Manslaughter", OUT_MANSLAUGHTER_PLOT)
    plot_reliability(stats_ng, "Not_Guilty", OUT_NOTGUILTY_PLOT)

    brier_m = brier_binary(p_m, y_m)
    brier_ms = brier_binary(p_ms, y_ms)
    brier_ng = brier_binary(p_ng, y_ng)

    p_mat = np.column_stack([p_m.to_numpy(), p_ms.to_numpy(), p_ng.to_numpy()])
    y_mat = np.column_stack([y_m.to_numpy(), y_ms.to_numpy(), y_ng.to_numpy()])
    brier_multi = brier_multiclass(p_mat, y_mat)

    ece_m = compute_ece(stats_m)
    ece_m_label = ece_label(ece_m)

    metrics = pd.DataFrame(
        [
            {"row_type": "metric", "metric": "brier_murder", "value": brier_m, "class": "Murder", "interpretation": ""},
            {"row_type": "metric", "metric": "brier_manslaughter", "value": brier_ms, "class": "Manslaughter", "interpretation": ""},
            {"row_type": "metric", "metric": "brier_not_guilty", "value": brier_ng, "class": "Not_Guilty", "interpretation": ""},
            {"row_type": "metric", "metric": "brier_multiclass_overall", "value": brier_multi, "class": "overall", "interpretation": ""},
            {"row_type": "metric", "metric": "ece_murder", "value": ece_m, "class": "Murder", "interpretation": ece_m_label},
        ]
    )

    # Also save bin-level reliability stats in the same CSV (long format)
    all_bins = pd.concat([stats_m, stats_ms, stats_ng], ignore_index=True)
    all_bins = all_bins.rename(columns={"class": "class_name"})
    all_bins["row_type"] = "reliability_bin"
    all_bins["metric"] = ""
    all_bins["value"] = np.nan
    all_bins["class"] = all_bins["class_name"]
    all_bins["interpretation"] = ""
    all_bins = all_bins[
        ["row_type", "metric", "value", "class", "interpretation", "bin_label", "mean_pred", "obs_freq", "n_cases"]
    ]

    metrics["bin_label"] = ""
    metrics["mean_pred"] = np.nan
    metrics["obs_freq"] = np.nan
    metrics["n_cases"] = np.nan
    metrics = metrics[
        ["row_type", "metric", "value", "class", "interpretation", "bin_label", "mean_pred", "obs_freq", "n_cases"]
    ]

    out = pd.concat([metrics, all_bins], ignore_index=True)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_RESULTS, index=False, encoding="utf-8")

    print("Brier scores:")
    print(
        pd.DataFrame(
            {
                "class": ["Murder", "Manslaughter", "Not_Guilty", "overall(multiclass)"],
                "brier_score": [brier_m, brier_ms, brier_ng, brier_multi],
            }
        ).to_string(index=False)
    )
    print(f"\nMurder ECE (10 bins): {ece_m:.6f} -> {ece_m_label}")
    print(f"\nWrote: {OUT_MURDER_PLOT}")
    print(f"Wrote: {OUT_MANSLAUGHTER_PLOT}")
    print(f"Wrote: {OUT_NOTGUILTY_PLOT}")
    print(f"Wrote: {OUT_RESULTS}")


if __name__ == "__main__":
    main()

