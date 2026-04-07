#!/usr/bin/env python3
"""
Train-test split and out-of-sample evaluation for the homicide Bayesian network.

1. Stratified 80/20 split on Verdict -> data/processed/train_set.csv, test_set.csv
2. Fit DiscreteBayesianNetwork on train only (BayesianEstimator BDeu, equivalent_sample_size=5)
3. VariableElimination FULL_EVIDENCE on test set -> outputs/test_set_results.csv
4. Metrics: accuracy, per-class accuracy, confusion matrix PNG, Brier, ECE Murder, reliability plot
5. Compare to fixed in-sample reference metrics

Run:
  pip install pgmpy scikit-learn matplotlib
  python code/train_test_split_evaluation.py

Use existing split + refitted model (no re-fit, no overwrite of CSVs/BIF):
  python code/train_test_split_evaluation.py --eval-only
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise SystemExit("Install scikit-learn: pip install scikit-learn") from e

try:
    from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.readwrite import BIFReader, BIFWriter
except Exception as e:
    raise SystemExit("Install pgmpy: pip install pgmpy") from e


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "case_node_matrix.csv"
PROCESSED = ROOT / "data" / "processed"
OUT_BIF = ROOT / "outputs" / "homicide_bn_train.bif"
OUT_RESULTS = ROOT / "outputs" / "test_set_results.csv"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_CONF = FIG_DIR / "fig_test_confusion_matrix.png"
FIG_REL = FIG_DIR / "fig_test_reliability_murder.png"

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

VERDICT_LABELS = {0: "Not_Guilty", 1: "Manslaughter", 2: "Murder"}
VERDICT_ORDER = ["Murder", "Manslaughter", "Not_Guilty"]
DISPLAY_LABELS = {"Not_Guilty": "Not Guilty", "Manslaughter": "Manslaughter", "Murder": "Murder"}

# In-sample (full corpus) reference metrics for comparison
IN_SAMPLE = {
    "accuracy": 0.7255,
    "brier_murder": 0.196,
    "ece_murder": 0.0019,
}


def edges_to_tuples(edges: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(p, c) for p, kids in edges.items() for c in kids]


def required_nodes(edges: dict[str, list[str]]) -> list[str]:
    s = set(edges.keys())
    for kids in edges.values():
        s.update(kids)
    return sorted(s)


def load_matrix_raw() -> pd.DataFrame:
    if not MATRIX.exists():
        raise FileNotFoundError(f"Missing: {MATRIX}")
    df = pd.read_csv(MATRIX, encoding="utf-8-sig")
    if "case_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "case_id"})
    return df


def prepare_fit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows ready for pgmpy fit: string states, no case_id."""
    nodes = required_nodes(EDGES)
    missing = [c for c in nodes if c not in df.columns]
    if missing:
        raise ValueError(f"Matrix missing columns: {missing}")
    d = df[nodes].copy()
    for col in nodes:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["Verdict"]).copy()
    for col in nodes:
        d[col] = d[col].round().astype(int)
    for col in nodes:
        if col == "Verdict":
            d[col] = d[col].map(VERDICT_LABELS).astype(str)
        else:
            d[col] = d[col].map({0: "0", 1: "1"}).astype(str)
    return d


def encode_evidence_row(row: pd.Series, evidence_vars: list[str]) -> dict[str, str]:
    ev = {}
    for v in evidence_vars:
        val = row.get(v)
        if pd.isna(val):
            continue
        ev[v] = str(int(round(float(val))))
    return ev


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


def query_verdict(infer: VariableElimination, evidence: dict[str, str]) -> dict:
    q = infer.query(variables=["Verdict"], evidence=evidence, show_progress=False)
    states = q.state_names["Verdict"]
    probs = dict(zip(states, q.values))
    if "Not Guilty" in probs and "Not_Guilty" not in probs:
        probs["Not_Guilty"] = probs["Not Guilty"]
    out = {k: float(probs.get(k, 0.0)) for k in VERDICT_ORDER}
    map_v = max(out.items(), key=lambda kv: kv[1])[0]
    return {
        "P(Murder)": out["Murder"],
        "P(Manslaughter)": out["Manslaughter"],
        "P(Not Guilty)": out["Not_Guilty"],
        "MAP_verdict": map_v,
    }


def verdict_distribution_for_row(row: pd.Series) -> dict[str, float]:
    """One-hot actual verdict for Brier."""
    code = int(round(float(row["Verdict"])))
    lab = VERDICT_LABELS[code]
    return {
        "Murder": 1.0 if lab == "Murder" else 0.0,
        "Manslaughter": 1.0 if lab == "Manslaughter" else 0.0,
        "Not_Guilty": 1.0 if lab == "Not_Guilty" else 0.0,
    }


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_multiclass(p_mat: np.ndarray, y_mat: np.ndarray) -> float:
    return float(np.mean(np.sum((p_mat - y_mat) ** 2, axis=1)))


def ece_murder_10bins(p_murder: np.ndarray, y_murder: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    total = len(p_murder)
    if total == 0:
        return float("nan")
    ece = 0.0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        if i < 9:
            mask = (p_murder >= lo) & (p_murder < hi)
        else:
            mask = (p_murder >= lo) & (p_murder <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p = float(p_murder[mask].mean())
        obs = float(y_murder[mask].mean())
        ece += (n / total) * abs(mean_p - obs)
    return ece


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("MAP verdict (test)")
    ax.set_ylabel("Actual verdict")
    ax.set_title("Test set confusion matrix (stratified hold-out)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="#0b1f3a", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_reliability_murder(p_murder: np.ndarray, y_murder: np.ndarray, out_path: Path) -> None:
    bins = np.linspace(0.0, 1.0, 11)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1.5, label="Perfect calibration")
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        if i < 9:
            mask = (p_murder >= lo) & (p_murder < hi)
        else:
            mask = (p_murder >= lo) & (p_murder <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mp = float(p_murder[mask].mean())
        of = float(y_murder[mask].mean())
        ax.plot(mp, of, "o", color="#1f3b6d", markersize=8)
        ax.annotate(f"n={n}", (mp, of), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(Murder)")
    ax.set_ylabel("Observed frequency (Murder)")
    ax.set_title("Reliability diagram: Murder class (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def class_distribution(df: pd.DataFrame, name: str) -> None:
    vc = df["Verdict"].value_counts().sort_index()
    print(f"\n{name} size: {len(df)}")
    print(f"{name} Verdict distribution (code -> count):")
    for code, cnt in vc.items():
        lab = VERDICT_LABELS.get(int(code), str(code))
        pct = 100.0 * cnt / len(df)
        print(f"  {code} ({lab}): {int(cnt)} ({pct:.1f}%)")


def run_test_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bif_path: Path,
    *,
    write_results: bool = True,
) -> pd.DataFrame:
    """Load BN from BIF, infer on test_df, optional CSV write, metrics + plots."""
    if not bif_path.exists():
        raise FileNotFoundError(f"Missing BIF: {bif_path}")

    model_loaded = load_model_from_bif(bif_path)
    infer = VariableElimination(model_loaded)
    model_vars = [v for v in model_loaded.nodes() if v != "Verdict"]

    rows_out = []
    for _, row in test_df.iterrows():
        ev = encode_evidence_row(row, model_vars)
        res = query_verdict(infer, ev)
        code = int(round(float(row["Verdict"])))
        actual = VERDICT_LABELS[code]
        correct = res["MAP_verdict"] == actual
        rows_out.append(
            {
                "case_id": row["case_id"],
                "P(Murder)": res["P(Murder)"],
                "P(Manslaughter)": res["P(Manslaughter)"],
                "P(Not Guilty)": res["P(Not Guilty)"],
                "MAP_verdict": res["MAP_verdict"],
                "actual_verdict": actual,
                "correct": bool(correct),
            }
        )
    results = pd.DataFrame(rows_out)
    if write_results:
        OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUT_RESULTS, index=False, encoding="utf-8")
        print(f"Saved {OUT_RESULTS}")

    # Metrics
    y_true = results["actual_verdict"].values
    y_pred = results["MAP_verdict"].values
    acc = float((y_true == y_pred).mean())

    print("\n--- Test set metrics ---")
    print(f"Overall accuracy: {acc:.4f}")
    print("\nAccuracy by actual class:")
    for lab in ["Murder", "Manslaughter", "Not_Guilty"]:
        mask = results["actual_verdict"] == lab
        if mask.sum() == 0:
            print(f"  {DISPLAY_LABELS[lab]}: N/A (no cases)")
        else:
            sub = results[mask]
            a = float(sub["correct"].mean())
            print(f"  {DISPLAY_LABELS[lab]}: {a:.4f} (n={int(mask.sum())})")

    # Confusion matrix: rows actual, cols predicted (Murder, Manslaughter, Not_Guilty order)
    label_list = ["Murder", "Manslaughter", "Not_Guilty"]
    cm = np.zeros((3, 3), dtype=int)
    for a, p in zip(y_true, y_pred):
        i = label_list.index(a)
        j = label_list.index(p)
        cm[i, j] += 1
    print("\nConfusion matrix (rows=actual, cols=MAP predicted):")
    print("              Murder  Manslaughter  Not Guilty")
    for i, lab in enumerate(["Murder", "Manslaughter", "Not Guilty"]):
        print(f"  {lab:12s} {cm[i, 0]:6d}  {cm[i, 1]:12d}  {cm[i, 2]:10d}")
    plot_confusion_matrix(cm, ["Murder", "Manslaughter", "Not Guilty"], FIG_CONF)

    p_m = results["P(Murder)"].to_numpy(dtype=float)
    p_ms = results["P(Manslaughter)"].to_numpy(dtype=float)
    p_ng = results["P(Not Guilty)"].to_numpy(dtype=float)
    y_m = (results["actual_verdict"] == "Murder").astype(int).to_numpy()
    y_ms = (results["actual_verdict"] == "Manslaughter").astype(int).to_numpy()
    y_ng = (results["actual_verdict"] == "Not_Guilty").astype(int).to_numpy()
    p_mat = np.column_stack([p_m, p_ms, p_ng])
    y_mat = np.column_stack([y_m, y_ms, y_ng])

    b_m = brier_binary(p_m, y_m.astype(float))
    b_ms = brier_binary(p_ms, y_ms.astype(float))
    b_ng = brier_binary(p_ng, y_ng.astype(float))
    b_multi = brier_multiclass(p_mat, y_mat.astype(float))
    ece = ece_murder_10bins(p_m, y_m.astype(float))

    print(f"\nPer-class Brier: Murder={b_m:.4f}, Manslaughter={b_ms:.4f}, Not Guilty={b_ng:.4f}")
    print(f"Multiclass Brier: {b_multi:.4f}")
    print(f"ECE (Murder, 10 bins): {ece:.4f}")

    plot_reliability_murder(p_m, y_m.astype(float), FIG_REL)

    # Comparison table
    print("\n--- In-sample vs test-set comparison ---")
    print(f"{'metric':<22} | {'in-sample':>10} | {'test-set':>10} | {'difference':>12}")
    print("-" * 62)
    print(f"{'accuracy':<22} | {IN_SAMPLE['accuracy']:>10.4f} | {acc:>10.4f} | {acc - IN_SAMPLE['accuracy']:>+12.4f}")
    print(f"{'Brier (Murder)':<22} | {IN_SAMPLE['brier_murder']:>10.4f} | {b_m:>10.4f} | {b_m - IN_SAMPLE['brier_murder']:>+12.4f}")
    print(f"{'ECE (Murder)':<22} | {IN_SAMPLE['ece_murder']:>10.4f} | {ece:>10.4f} | {ece - IN_SAMPLE['ece_murder']:>+12.4f}")

    # Final summary
    all_murder_map = bool((results["MAP_verdict"] == "Murder").all())
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Train: n={len(train_df)} — class counts: {train_df['Verdict'].value_counts().sort_index().to_dict()}")
    print(f"Test:  n={len(test_df)} — class counts: {test_df['Verdict'].value_counts().sort_index().to_dict()}")
    print(f"\nMAP dominance on unseen test data: every prediction is Murder? {all_murder_map}")
    if not all_murder_map:
        print("  MAP verdict counts:", results["MAP_verdict"].value_counts().to_dict())
    calib_note = (
        "ECE(Murder) rises vs in-sample but stays <0.05 — modest OOS miscalibration; see reliability plot."
        if ece < 0.05
        else "ECE(Murder) >= 0.05 on test — review fig_test_reliability_murder.png for OOS calibration."
    )
    print(f"\nCalibration (out-of-sample): {calib_note}")
    print("=" * 70)

    return results


def main() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10})

    parser = argparse.ArgumentParser(description="Train/test BN evaluation.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Use existing data/processed/train_set.csv, test_set.csv and --bif model; skip split and refit.",
    )
    parser.add_argument(
        "--bif",
        type=Path,
        default=OUT_BIF,
        help=f"BIF model path (default: {OUT_BIF})",
    )
    args = parser.parse_args()

    if args.eval_only:
        train_path = PROCESSED / "train_set.csv"
        test_path = PROCESSED / "test_set.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing {test_path}")
        train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        test_df = pd.read_csv(test_path, encoding="utf-8-sig")
        if "case_id" not in train_df.columns:
            train_df = train_df.rename(columns={train_df.columns[0]: "case_id"})
        if "case_id" not in test_df.columns:
            test_df = test_df.rename(columns={test_df.columns[0]: "case_id"})
        print(f"Loaded {train_path} (n={len(train_df)})")
        print(f"Loaded {test_path} (n={len(test_df)})")
        print(f"Using model: {args.bif.resolve()}")
        print("(eval-only: not overwriting train/test CSVs or BIF)\n")
        class_distribution(train_df, "Train")
        class_distribution(test_df, "Test")
        run_test_evaluation(train_df, test_df, args.bif, write_results=True)
        return

    raw = load_matrix_raw()
    raw = raw.dropna(subset=["Verdict"]).copy()
    raw["Verdict"] = pd.to_numeric(raw["Verdict"], errors="coerce")
    raw = raw.dropna(subset=["Verdict"]).copy()
    raw["Verdict"] = raw["Verdict"].round().astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(raw)),
        test_size=0.2,
        random_state=42,
        stratify=raw["Verdict"].values,
    )
    train_df = raw.iloc[train_idx].reset_index(drop=True)
    test_df = raw.iloc[test_idx].reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED / "train_set.csv", index=False, encoding="utf-8")
    test_df.to_csv(PROCESSED / "test_set.csv", index=False, encoding="utf-8")
    print("Saved data/processed/train_set.csv")
    print("Saved data/processed/test_set.csv")

    class_distribution(train_df, "Train")
    class_distribution(test_df, "Test")

    train_fit = prepare_fit_frame(train_df)
    model = DiscreteBayesianNetwork(edges_to_tuples(EDGES))
    model.fit(train_fit, estimator=MaximumLikelihoodEstimator)
    model.fit(
        train_fit,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5,
    )
    OUT_BIF.parent.mkdir(parents=True, exist_ok=True)
    BIFWriter(model).write_bif(str(OUT_BIF))
    print(f"\nSaved refitted model: {OUT_BIF}")

    run_test_evaluation(train_df, test_df, OUT_BIF, write_results=True)


if __name__ == "__main__":
    main()
