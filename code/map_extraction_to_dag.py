#!/usr/bin/env python3
"""
map_extraction_to_dag.py

Maps extracted node labels (from the extraction pipeline) to the formal DAG nodes
defined in design_bn_dag.py for the murder BN under English and Welsh law.

- Loads extraction output (nodes CSV).
- Builds a mapping: extracted_label -> (dag_node, confidence, notes).
- Outputs a mapping table (CSV) and a summary (counts per DAG node, DAG nodes with no coverage).

Run from repo root:
    python code/map_extraction_to_dag.py
    python code/map_extraction_to_dag.py --input data/processed/nodes_actus_mens.csv --output outputs/mapping_actus_mens.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "nodes_baseline.csv"
DEFAULT_OUTPUT_TABLE = ROOT / "outputs" / "label_to_dag_mapping.csv"
DEFAULT_OUTPUT_SUMMARY = ROOT / "outputs" / "label_to_dag_summary.txt"

# All DAG nodes (must match design_bn_dag.py EDGES)
DAG_NODES = [
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
    "Verdict",
]

# Canonical mapping: extracted_label -> (dag_node, confidence, notes)
# confidence: high | medium | low | unmapped
LABEL_TO_DAG: dict[str, tuple[str, str, str]] = {
    # Actus reus / causation
    "actus_reus_killing": ("UnlawfulKilling", "high", "physical element of killing; extraction does not split factual/legal causation"),
    "unlawful_killing": ("UnlawfulKilling", "high", "direct match"),
    "causation": ("LegalCausation", "medium", "extraction does not distinguish factual vs legal causation; mapped to LegalCausation"),
    # Mens rea
    "mens_rea_intent_to_kill": ("IntentToKill", "high", "direct match"),
    "mens_rea_intent_gbh": ("IntentToCauseGBH", "high", "direct match"),
    # Partial defences
    "loss_of_control": ("LossOfControl", "high", "direct match"),
    "diminished_responsibility": ("DiminishedResponsibility", "high", "direct match"),
    # Defences (aggregate node)
    "self_defence_legal": ("Defences", "high", "self-defence"),
    "insanity": ("Defences", "high", "M'Naghten-type defence"),
    "duress_not_available_murder": ("Defences", "low", "duress not available for murder; doctrinal note"),
    # Verdict / outcome
    "murder": ("Verdict", "high", "conviction outcome"),
    "manslaughter_diminished_resp": ("Verdict", "high", "verdict variant"),
    "manslaughter_loss_of_control": ("Verdict", "high", "verdict variant"),
    "manslaughter_unlawful_act": ("Verdict", "high", "verdict variant"),
    "manslaughter_gross_negligence": ("Verdict", "high", "verdict variant"),
    "lawful_killing_self_defence": ("Verdict", "high", "acquittal outcome"),

    # ----- VoluntaryAct: physical acts, bodily movements, conduct -----
    "assault_method_stabbing": ("VoluntaryAct", "medium", "stabbing is voluntary conduct causing harm"),
    "assault_method_shooting": ("VoluntaryAct", "medium", "shooting is voluntary conduct causing harm"),
    "assault_method_blunt_force": ("VoluntaryAct", "medium", "blunt-force attack is voluntary conduct"),
    "strangulation": ("VoluntaryAct", "medium", "strangulation is voluntary bodily act"),
    "poisoning": ("VoluntaryAct", "medium", "administering poison is voluntary act"),
    "abduction": ("VoluntaryAct", "medium", "abduction involves voluntary restraint/removal"),
    "tied_restraint": ("VoluntaryAct", "medium", "tying/restraining is voluntary conduct"),
    "weapon_present": ("VoluntaryAct", "low", "carrying/using weapon implies voluntary conduct"),
    "weapon_firearm": ("VoluntaryAct", "low", "firearm use implies voluntary conduct"),
    "weapon_sharp": ("VoluntaryAct", "low", "use of sharp weapon implies voluntary conduct"),
    "weapon_blunt": ("VoluntaryAct", "low", "use of blunt weapon implies voluntary conduct"),
    "weapon_petrol_fire": ("VoluntaryAct", "low", "use of petrol/fire implies voluntary conduct"),
    "concealment_actions": ("VoluntaryAct", "low", "post-offence concealment is voluntary conduct"),
    "disposal_of_weapon": ("VoluntaryAct", "low", "disposing of weapon is voluntary act"),
    "flight_after_offence": ("VoluntaryAct", "low", "flight is voluntary conduct (post-offence)"),
    "post_offence_burial": ("VoluntaryAct", "low", "burial is voluntary post-offence conduct"),

    # ----- DeathOfHumanBeing: death, deceased, cause of death -----
    "time_of_death_window_narrow": ("DeathOfHumanBeing", "medium", "narrow time-of-death window evidences that death occurred"),
    "injury_defensive_wounds": ("DeathOfHumanBeing", "low", "defensive wounds evidence fatal attack and thus death of human being"),

    # ----- FactualCausation: but-for, factual cause, direct cause -----
    "wound_sharp_force": ("FactualCausation", "low", "sharp-force wounds evidence defendant's act was factual cause of harm/death"),
    "wound_blunt_force": ("FactualCausation", "low", "blunt-force wounds evidence defendant's act was factual cause of harm/death"),

    # ----- MensReaIntent: intent, mental state, knowledge, foresight (catch-all) -----
    "prior_threats": ("MensReaIntent", "medium", "prior threats evidence intent or state of mind"),
    "motive_financial_gain": ("MensReaIntent", "low", "motive as evidence of mental state / intent"),
    "motive_jealousy_revenge": ("MensReaIntent", "low", "motive as evidence of mental state / intent"),
    "confession_or_admission": ("MensReaIntent", "low", "admissions can evidence intent or state of mind"),
    "boasting_admissions": ("MensReaIntent", "low", "boasting about offence can evidence intent/state of mind"),
    "search_history_weapons_or_poison": ("MensReaIntent", "low", "search history can evidence foresight or intent to cause harm"),
    "prior_conflict": ("MensReaIntent", "low", "prior conflict can contextualise intent"),
    "prior_assault": ("MensReaIntent", "low", "prior assault can evidence intent or propensity"),

    # ----- MaliceAforethought: malice, aforethought, premeditation -----
    "planning_premeditation": ("MaliceAforethought", "medium", "premeditation is a facet of malice aforethought under Coke's definition"),
}

UNMAPPED = "UNMAPPED"


def load_extraction_labels(path: Path) -> list[str]:
    """Load unique 'label' values from extraction nodes CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Extraction output not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["label"])
    return df["label"].dropna().astype(str).str.strip().unique().tolist()


def build_mapping_table(extracted_labels: list[str]) -> pd.DataFrame:
    """Build mapping table: extracted_label, dag_node, confidence, notes."""
    rows = []
    for lab in sorted(extracted_labels):
        if lab in LABEL_TO_DAG:
            dag_node, conf, notes = LABEL_TO_DAG[lab]
            rows.append({"extracted_label": lab, "dag_node": dag_node, "confidence": conf, "notes": notes})
        else:
            rows.append({"extracted_label": lab, "dag_node": UNMAPPED, "confidence": "unmapped", "notes": "no corresponding DAG node or ambiguous"})
    return pd.DataFrame(rows)


def write_mapping_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")


def summarise(df: pd.DataFrame, out_path: Path) -> str:
    """Return summary text: counts per DAG node, and DAG nodes with no coverage."""
    mapped = df[df["dag_node"] != UNMAPPED]
    counts = mapped.groupby("dag_node")["extracted_label"].nunique()
    lines = [
        "Summary: Extracted labels -> DAG nodes",
        "======================================",
        "",
        "Unique extracted labels mapping to each DAG node:",
        "",
    ]
    for node in DAG_NODES:
        n = int(counts.get(node, 0))
        lines.append(f"  {node}: {n}")
    lines.append("")
    no_coverage = [n for n in DAG_NODES if counts.get(n, 0) == 0]
    lines.append("DAG nodes with no coverage in extraction output:")
    if no_coverage:
        for n in no_coverage:
            lines.append(f"  - {n}")
    else:
        lines.append("  (none)")
    lines.append("")
    n_unmapped = int((df["dag_node"] == UNMAPPED).sum())
    lines.append(f"Total unique extracted labels: {len(df)}")
    lines.append(f"Mapped: {len(df) - n_unmapped}")
    lines.append(f"Unmapped: {n_unmapped}")
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    p = argparse.ArgumentParser(description="Map extraction labels to murder BN DAG nodes.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Extraction nodes CSV.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_TABLE, help="Output mapping table CSV.")
    p.add_argument("--summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY, help="Output summary text file.")
    args = p.parse_args()

    try:
        labels = load_extraction_labels(args.input)
    except Exception as e:
        print(f"Error loading extraction output: {e}", file=sys.stderr)
        return 1

    if not labels:
        print("No labels found in extraction output.", file=sys.stderr)
        return 1

    table = build_mapping_table(labels)
    write_mapping_table(table, args.output)
    summary_text = summarise(table, args.summary)

    print(f"Mapping table written to: {args.output}")
    print(f"Summary written to: {args.summary}")
    print()
    print(summary_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
