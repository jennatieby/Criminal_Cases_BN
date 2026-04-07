#!/usr/bin/env python3
"""
Before/after comparison: HEAD case_node_matrix (980 cases) vs current matrix + pipeline outputs.

Before inference metrics: same procedure as the live pipeline (80/20 stratified train fit,
VariableElimination FULL_EVIDENCE on all rows of that matrix).

Run after the main pipeline so outputs/scenario_results_full.csv and consistency_results.csv exist.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import train_test_split_evaluation as tte
from evaluate_bn_consistency import run_threshold
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


ROOT = Path(__file__).resolve().parents[1]
OUT_COMPARISON = ROOT / "outputs" / "pipeline_rerun_comparison.csv"
MATRIX_CUR = ROOT / "case_node_matrix.csv"
SCEN_AFTER = ROOT / "outputs" / "scenario_results_full.csv"
CONS_AFTER = ROOT / "outputs" / "consistency_results.csv"


def matrix_from_git_head() -> Path | None:
    r = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:case_node_matrix.csv"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    d = Path(tempfile.mkdtemp(prefix="matrix_head_"))
    p = d / "case_node_matrix.csv"
    p.write_text(r.stdout, encoding="utf-8")
    return p


def verdict_counts(raw: pd.DataFrame) -> dict[str, int]:
    vc = raw["Verdict"].value_counts().sort_index()
    labels = {0: "Not_Guilty", 1: "Manslaughter", 2: "Murder"}
    return {f"verdict_{int(k)}_{labels.get(int(k), k)}": int(v) for k, v in vc.items()}


def fit_and_infer_full_corpus(matrix_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (raw_all_rows_with_verdict, inference_results DataFrame)."""
    raw = pd.read_csv(matrix_path, encoding="utf-8-sig")
    if "case_id" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "case_id"})
    raw = raw.dropna(subset=["Verdict"]).copy()
    raw["Verdict"] = pd.to_numeric(raw["Verdict"], errors="coerce")
    raw = raw.dropna(subset=["Verdict"]).copy()
    raw["Verdict"] = raw["Verdict"].round().astype(int)

    train_idx, _ = tte.train_test_split(
        np.arange(len(raw)),
        test_size=0.2,
        random_state=42,
        stratify=raw["Verdict"].values,
    )
    train_df = raw.iloc[train_idx].reset_index(drop=True)

    train_fit = tte.prepare_fit_frame(train_df)
    model = DiscreteBayesianNetwork(tte.edges_to_tuples(tte.EDGES))
    model.fit(train_fit, estimator=MaximumLikelihoodEstimator)
    model.fit(
        train_fit,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5,
    )
    infer = VariableElimination(model)
    model_vars = [v for v in model.nodes() if v != "Verdict"]

    rows_out = []
    for _, row in raw.iterrows():
        ev = tte.encode_evidence_row(row, model_vars)
        res = tte.query_verdict(infer, ev)
        code = int(round(float(row["Verdict"])))
        actual = tte.VERDICT_LABELS[code]
        rows_out.append(
            {
                "case_id": str(row["case_id"]),
                "P(Murder)": res["P(Murder)"],
                "P(Manslaughter)": res["P(Manslaughter)"],
                "P(Not Guilty)": res["P(Not Guilty)"],
                "MAP_verdict": res["MAP_verdict"],
                "actual_verdict": actual,
                "correct": bool(res["MAP_verdict"] == actual),
            }
        )
    return raw, pd.DataFrame(rows_out)


def mean_js_at_threshold(matrix_path: Path, inf_df: pd.DataFrame, threshold: float) -> float:
    mat = pd.read_csv(matrix_path, encoding="utf-8-sig")
    if "case_id" not in mat.columns:
        mat = mat.rename(columns={mat.columns[0]: "case_id"})
    mat["case_id"] = mat["case_id"].astype(str)
    inf_df = inf_df.copy()
    inf_df["case_id"] = inf_df["case_id"].astype(str)

    doctrinal_cols = [c for c in mat.columns if c not in {"case_id", "Verdict"}]
    sub = mat[mat["case_id"].isin(inf_df["case_id"])].copy()
    X = sub.set_index("case_id")[doctrinal_cols].fillna(0).astype(int)
    post = inf_df.set_index("case_id")[
        ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)", "MAP_verdict"]
    ].copy()
    post = post.loc[X.index]
    res, _, _ = run_threshold(X, post, threshold)
    if res.empty:
        return float("nan")
    return float(res["js_divergence"].mean())


def collect_metrics(matrix_path: Path) -> dict[str, float | int | str | bool]:
    raw, inf = fit_and_infer_full_corpus(matrix_path)
    p_m = inf["P(Murder)"].to_numpy(dtype=float)
    y_m = (inf["actual_verdict"] == "Murder").astype(int).to_numpy()
    acc = float(inf["correct"].mean())
    b_m = tte.brier_binary(p_m, y_m.astype(float))
    ece = tte.ece_murder_10bins(p_m, y_m.astype(float))
    map_dom = bool((inf["MAP_verdict"] == "Murder").all())
    mean_js = mean_js_at_threshold(matrix_path, inf, 0.60)

    out: dict[str, float | int | str | bool] = {
        "n_cases": int(len(raw)),
        "overall_accuracy_FULL_EVIDENCE": acc,
        "brier_murder": b_m,
        "ece_murder": ece,
        "MAP_all_Murder": map_dom,
        "mean_JS_pairs_t0.60": mean_js,
    }
    out.update(verdict_counts(raw))
    return out


def main() -> None:
    before_path = matrix_from_git_head()
    before: dict[str, float | int | str | bool] = {}
    if before_path is None:
        before = {"n_cases": "", "note": "git HEAD case_node_matrix.csv unavailable"}
    else:
        try:
            before = collect_metrics(before_path)
        except Exception as e:
            before = {"error": str(e)}

    scen = pd.read_csv(SCEN_AFTER, encoding="utf-8-sig")
    full = scen[scen["scenario"] == "FULL_EVIDENCE"].copy()
    acc_after = float(full["correct"].mean()) if "correct" in full.columns and len(full) else float("nan")
    p_m = full["P(Murder)"].astype(float)
    y_m = full["actual_verdict"].apply(lambda x: 1 if str(x).strip() == "Murder" else 0).astype(int)
    b_after = tte.brier_binary(p_m.to_numpy(), y_m.to_numpy().astype(float))
    ece_after = tte.ece_murder_10bins(p_m.to_numpy(), y_m.to_numpy().astype(float))
    map_after = bool((full["MAP_verdict"] == "Murder").all()) if len(full) else False

    cons = pd.read_csv(CONS_AFTER, encoding="utf-8-sig")
    c60 = cons[np.isclose(cons["threshold"].astype(float), 0.60)]
    js_after = float(c60["js_divergence"].mean()) if len(c60) else float("nan")

    raw_cur = pd.read_csv(MATRIX_CUR, encoding="utf-8-sig")
    if "case_id" not in raw_cur.columns:
        raw_cur = raw_cur.rename(columns={raw_cur.columns[0]: "case_id"})
    raw_cur = raw_cur.dropna(subset=["Verdict"]).copy()
    raw_cur["Verdict"] = pd.to_numeric(raw_cur["Verdict"], errors="coerce")
    raw_cur = raw_cur.dropna(subset=["Verdict"]).copy()
    raw_cur["Verdict"] = raw_cur["Verdict"].round().astype(int)
    v_after = verdict_counts(raw_cur)

    rows = [
        {"metric": "total_cases", "before": before.get("n_cases", ""), "after": int(len(raw_cur))},
        {
            "metric": "verdict_distribution",
            "before": str({k: v for k, v in before.items() if str(k).startswith("verdict_")}) if before_path else "",
            "after": str(v_after),
        },
        {"metric": "overall_accuracy_FULL_EVIDENCE", "before": before.get("overall_accuracy_FULL_EVIDENCE", ""), "after": acc_after},
        {"metric": "brier_murder", "before": before.get("brier_murder", ""), "after": b_after},
        {"metric": "ece_murder", "before": before.get("ece_murder", ""), "after": ece_after},
        {"metric": "MAP_dominance_all_Murder", "before": before.get("MAP_all_Murder", ""), "after": map_after},
        {"metric": "mean_JS_divergence_t0.60", "before": before.get("mean_JS_pairs_t0.60", ""), "after": js_after},
    ]

    df = pd.DataFrame(rows)
    OUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_COMPARISON, index=False, encoding="utf-8")

    print("=== Pipeline rerun: before (git HEAD matrix) vs after (current) ===\n")
    print(df.to_string(index=False))
    print(f"\nSaved {OUT_COMPARISON}")


if __name__ == "__main__":
    main()
