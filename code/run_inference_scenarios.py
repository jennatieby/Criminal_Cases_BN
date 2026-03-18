#!/usr/bin/env python3
"""
run_inference_scenarios.py

Given:
- a fitted Bayesian network in BIF format (homicide_bn.bif)
- a binary case-node matrix (case_node_matrix.csv)

Run three inference test scenarios using pgmpy's VariableElimination:
1) FULL EVIDENCE: use all observed node values as evidence for 20 cases, query P(Verdict|evidence),
   compare MAP verdict to actual verdict, report accuracy.
2) PARTIAL EVIDENCE (disputed element): for the same 20 cases, remove the causation node from evidence
   and re-run inference; report posterior shifts vs full evidence.
   (We treat \"Causation\" as the model's LegalCausation node.)
3) COUNTERFACTUAL: for 10 Murder cases, flip IntentToKill and IntentToCauseGBH to 0 in evidence and re-run
   inference; check whether probability mass shifts away from Murder.

Outputs:
- outputs/scenario_results.csv
- Prints a summary table with accuracy per scenario and mean posterior probabilities.

Run:
  pip install pgmpy
  python code/run_inference_scenarios.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pgmpy.inference import VariableElimination
    from pgmpy.readwrite import BIFReader
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install pgmpy\n"
        f"Import error: {e}"
    )


ROOT = Path(__file__).resolve().parents[1]
BIF_PATH = ROOT / "homicide_bn.bif"
MATRIX_PATH = ROOT / "case_node_matrix.csv"
OUT_CSV = ROOT / "outputs" / "scenario_results.csv"
OUT_CLEAN_BIF = ROOT / "outputs" / "homicide_bn_clean.bif"

CAUSE_NODE = "LegalCausation"  # interpret \"Causation\" in prompt as this DAG node

# Support both legacy BIF state naming ("Not Guilty") and safe naming ("Not_Guilty")
VERDICT_CODE_TO_LABEL = {2: "Murder", 1: "Manslaughter", 0: "Not_Guilty"}
VERDICT_LABELS = ["Murder", "Manslaughter", "Not_Guilty"]


def load_model():
    if not BIF_PATH.exists():
        raise FileNotFoundError(f"Missing BIF model: {BIF_PATH}")
    # BIFReader splits states on whitespace; fix legacy state name "Not Guilty" -> "Not_Guilty"
    bif_text = BIF_PATH.read_text(encoding="utf-8", errors="ignore")
    if "Not Guilty" in bif_text:
        bif_text = bif_text.replace("Not Guilty", "Not_Guilty")
        OUT_CLEAN_BIF.parent.mkdir(parents=True, exist_ok=True)
        OUT_CLEAN_BIF.write_text(bif_text, encoding="utf-8")
        reader = BIFReader(str(OUT_CLEAN_BIF))
    else:
        reader = BIFReader(str(BIF_PATH))
    return reader.get_model()


def load_matrix() -> pd.DataFrame:
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"Missing matrix: {MATRIX_PATH}")
    df = pd.read_csv(MATRIX_PATH, encoding="utf-8-sig")
    if "case_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "case_id"})
    df["case_id"] = df["case_id"].astype(str)
    return df


def encode_evidence_row(row: pd.Series, evidence_vars: list[str]) -> dict[str, str]:
    """
    pgmpy model states in BIF are '0'/'1' for binary nodes.
    We pass evidence values as strings.
    """
    ev = {}
    for v in evidence_vars:
        val = row.get(v)
        if pd.isna(val):
            continue
        # nodes are 0/1 (may be float in CSV)
        ev[v] = str(int(round(float(val))))
    return ev


def verdict_label_from_row(row: pd.Series) -> str | None:
    val = row.get("Verdict")
    if pd.isna(val):
        return None
    try:
        return VERDICT_CODE_TO_LABEL[int(round(float(val)))]
    except Exception:
        return None


def query_verdict(infer: VariableElimination, evidence: dict[str, str]) -> dict:
    q = infer.query(variables=["Verdict"], evidence=evidence, show_progress=False)
    states = q.state_names["Verdict"]
    probs = dict(zip(states, q.values))
    # normalize to our expected labels ordering (if model uses different order, handle safely)
    # tolerate either Not_Guilty or Not Guilty in model states
    if "Not Guilty" in probs and "Not_Guilty" not in probs:
        probs["Not_Guilty"] = probs["Not Guilty"]
    out = {k: float(probs.get(k, 0.0)) for k in VERDICT_LABELS}
    map_verdict = max(out.items(), key=lambda kv: kv[1])[0]
    return {
        "P(Murder)": out["Murder"],
        "P(Manslaughter)": out["Manslaughter"],
        "P(Not Guilty)": out["Not_Guilty"],
        "MAP_verdict": map_verdict,
    }


def main() -> None:
    model = load_model()
    infer = VariableElimination(model)
    df = load_matrix()

    # Evidence variables = all model variables except Verdict
    model_vars = [v for v in model.nodes() if v != "Verdict"]
    missing = [v for v in model_vars + ["Verdict"] if v not in df.columns]
    if missing:
        raise ValueError(f"Matrix missing required columns for model: {missing}")

    # Deterministic sampling for repeatability
    rng = np.random.default_rng(42)

    # 20 cases for full/partial evidence
    sample20 = df.sample(n=min(20, len(df)), random_state=42).copy()

    results: list[dict] = []

    # Cache full-evidence posteriors for delta comparisons in partial scenario
    full_post_by_case: dict[str, dict] = {}

    # Scenario 1: FULL EVIDENCE
    for _, row in sample20.iterrows():
        case_id = row["case_id"]
        actual = verdict_label_from_row(row)
        evidence = encode_evidence_row(row, model_vars)
        res = query_verdict(infer, evidence)
        full_post_by_case[case_id] = res
        results.append(
            {
                "case_id": case_id,
                "scenario": "FULL_EVIDENCE",
                **{k: res[k] for k in ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]},
                "MAP_verdict": res["MAP_verdict"],
                "actual_verdict": actual,
                "correct": bool(actual is not None and res["MAP_verdict"] == actual),
            }
        )

    # Scenario 2: PARTIAL EVIDENCE (disputed causation)
    # Important: if we keep UnlawfulKilling in evidence, LegalCausation becomes irrelevant to Verdict
    # because UnlawfulKilling is a direct parent of Verdict and a descendant of LegalCausation.
    # To actually test disputed causation, we remove LegalCausation AND the downstream actus reus
    # conclusion it feeds (UnlawfulKilling). We also remove FactualCausation to keep the whole
    # causation subgraph unobserved.
    for _, row in sample20.iterrows():
        case_id = row["case_id"]
        actual = verdict_label_from_row(row)
        ev_vars = [v for v in model_vars if v not in {CAUSE_NODE, "FactualCausation", "UnlawfulKilling"}]
        evidence = encode_evidence_row(row, ev_vars)
        res = query_verdict(infer, evidence)

        # record shifts vs full evidence
        base = full_post_by_case.get(case_id, {})
        d_m = float(res["P(Murder)"] - base.get("P(Murder)", 0.0))
        d_ms = float(res["P(Manslaughter)"] - base.get("P(Manslaughter)", 0.0))
        d_ng = float(res["P(Not Guilty)"] - base.get("P(Not Guilty)", 0.0))

        results.append(
            {
                "case_id": case_id,
                "scenario": "PARTIAL_EVIDENCE_NO_CAUSATION",
                **{k: res[k] for k in ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]},
                "MAP_verdict": res["MAP_verdict"],
                "actual_verdict": actual,
                "correct": bool(actual is not None and res["MAP_verdict"] == actual),
                "delta_P(Murder)": d_m,
                "delta_P(Manslaughter)": d_ms,
                "delta_P(Not Guilty)": d_ng,
            }
        )

    # Scenario 3: COUNTERFACTUAL (10 murder cases; flip intent nodes to 0)
    murder_cases = df[df["Verdict"].round().astype(int) == 2].copy()
    if len(murder_cases) > 0:
        sample10 = murder_cases.sample(n=min(10, len(murder_cases)), random_state=42).copy()
        for _, row in sample10.iterrows():
            case_id = row["case_id"]
            actual = verdict_label_from_row(row)
            # Counterfactual baseline evidence set:
            # Do NOT condition on MaliceAforethought or MensReaIntent.
            # Condition on the remaining observed nodes:
            base_vars = [
                "VoluntaryAct",
                "DeathOfHumanBeing",
                "FactualCausation",
                "LegalCausation",
                "UnlawfulKilling",
                "LossOfControl",
                "DiminishedResponsibility",
                "Defences",
                # plus the two intent inputs at observed values for baseline
                "IntentToKill",
                "IntentToCauseGBH",
            ]
            evidence_base = encode_evidence_row(row, base_vars)
            base_res = query_verdict(infer, evidence_base)
            results.append(
                {
                    "case_id": case_id,
                    "scenario": "COUNTERFACTUAL_BASELINE",
                    **{k: base_res[k] for k in ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]},
                    "MAP_verdict": base_res["MAP_verdict"],
                    "actual_verdict": actual,
                    "correct": bool(actual is not None and base_res["MAP_verdict"] == actual),
                }
            )

            # Counterfactual: force intents to 0 (same evidence set otherwise)
            evidence_cf = dict(evidence_base)
            evidence_cf["IntentToKill"] = "0"
            evidence_cf["IntentToCauseGBH"] = "0"
            cf_res = query_verdict(infer, evidence_cf)
            p_murder_delta = float(cf_res["P(Murder)"] - base_res["P(Murder)"])
            counterfactual_shifted = bool(p_murder_delta < 0)
            results.append(
                {
                    "case_id": case_id,
                    "scenario": "COUNTERFACTUAL_NO_INTENT",
                    **{k: cf_res[k] for k in ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]},
                    "MAP_verdict": cf_res["MAP_verdict"],
                    "actual_verdict": actual,
                    "correct": bool(actual is not None and cf_res["MAP_verdict"] == actual),
                    "counterfactual_shifted": counterfactual_shifted,
                    "p_murder_delta": p_murder_delta,
                    "p_murder_baseline": float(base_res["P(Murder)"]),
                }
            )

            # Scenario 4: COUNTERFACTUAL_NO_INTENT_STRONG
            # Exclude UnlawfulKilling, MensReaIntent, MaliceAforethought from evidence entirely.
            # Condition only on actus-reus background + defences; compare observed intents vs forced no-intent.
            strong_vars = [
                "VoluntaryAct",
                "DeathOfHumanBeing",
                "FactualCausation",
                "LegalCausation",
                "LossOfControl",
                "DiminishedResponsibility",
                "Defences",
                "IntentToKill",
                "IntentToCauseGBH",
            ]
            strong_base_ev = encode_evidence_row(row, strong_vars)
            strong_base_res = query_verdict(infer, strong_base_ev)

            strong_cf_ev = dict(strong_base_ev)
            strong_cf_ev["IntentToKill"] = "0"
            strong_cf_ev["IntentToCauseGBH"] = "0"
            strong_cf_res = query_verdict(infer, strong_cf_ev)

            strong_delta = float(strong_cf_res["P(Murder)"] - strong_base_res["P(Murder)"])
            strong_shifted = bool(strong_delta < 0)

            results.append(
                {
                    "case_id": case_id,
                    "scenario": "COUNTERFACTUAL_NO_INTENT_STRONG",
                    **{k: strong_cf_res[k] for k in ["P(Murder)", "P(Manslaughter)", "P(Not Guilty)"]},
                    "MAP_verdict": strong_cf_res["MAP_verdict"],
                    "actual_verdict": actual,
                    "correct": bool(actual is not None and strong_cf_res["MAP_verdict"] == actual),
                    "counterfactual_shifted": strong_shifted,
                    "p_murder_delta": strong_delta,
                    "p_murder_baseline": float(strong_base_res["P(Murder)"]),
                }
            )

    out = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Summary: accuracy and mean posterior probabilities per scenario
    def _acc(x: pd.Series) -> float:
        # treat None as incorrect
        if x.empty:
            return float("nan")
        return float(x.fillna(False).mean())

    summary = (
        out.groupby("scenario")
        .agg(
            n_cases=("case_id", "count"),
            accuracy=("correct", _acc),
            mean_p_murder=("P(Murder)", "mean"),
            mean_p_manslaughter=("P(Manslaughter)", "mean"),
            mean_p_not_guilty=("P(Not Guilty)", "mean"),
        )
        .reset_index()
    )

    print(f"Wrote results to: {OUT_CSV}")
    print("\nSummary (accuracy + mean posteriors):")
    print(summary.to_string(index=False))

    # Counterfactual diagnostics: compare NO_INTENT to COUNTERFACTUAL_BASELINE
    cf = out[out["scenario"] == "COUNTERFACTUAL_NO_INTENT"].copy()
    base = out[out["scenario"] == "COUNTERFACTUAL_BASELINE"].copy()
    if not cf.empty and not base.empty:
        shifted_prop = float(cf["counterfactual_shifted"].fillna(False).mean())
        mean_delta = float(cf["p_murder_delta"].mean())
        mean_p_cf = float(cf["P(Murder)"].mean())
        mean_p_base = float(base["P(Murder)"].mean())
        print("\nCounterfactual diagnostics (NO_INTENT vs BASELINE):")
        print(f"  Proportion shifted: {shifted_prop:.2f}")
        print(f"  Mean p_murder_delta (no_intent - baseline): {mean_delta:.4f}")
        print(f"  Mean P(Murder) baseline:  {mean_p_base:.4f}")
        print(f"  Mean P(Murder) no_intent: {mean_p_cf:.4f}")

    # Strong counterfactual diagnostics (NO_INTENT_STRONG): use per-row baseline stored in p_murder_baseline
    strong = out[out["scenario"] == "COUNTERFACTUAL_NO_INTENT_STRONG"].copy()
    if not strong.empty:
        shifted_prop = float(strong["counterfactual_shifted"].fillna(False).mean())
        mean_delta = float(strong["p_murder_delta"].mean())
        mean_p_cf = float(strong["P(Murder)"].mean())
        mean_p_base = float(strong["p_murder_baseline"].mean())
        print("\nCounterfactual diagnostics (NO_INTENT_STRONG):")
        print(f"  Proportion shifted: {shifted_prop:.2f}")
        print(f"  Mean p_murder_delta (no_intent - baseline): {mean_delta:.4f}")
        print(f"  Mean P(Murder) baseline:  {mean_p_base:.4f}")
        print(f"  Mean P(Murder) no_intent: {mean_p_cf:.4f}")
        print("\nPer-case deltas (NO_INTENT_STRONG):")
        print(strong[["case_id", "p_murder_baseline", "P(Murder)", "p_murder_delta", "counterfactual_shifted"]].to_string(index=False))


if __name__ == "__main__":
    main()

