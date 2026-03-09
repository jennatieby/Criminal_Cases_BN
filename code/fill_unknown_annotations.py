#!/usr/bin/env python3
"""
fill_unknown_annotations.py

Post-process llm_annotations.csv: fill act_type, means_or_weapon, mental_state,
legal_concept, and evidence_type from keyword rules when the LLM returned "unknown".

Run after llm_annotate_sentences.py. Overwrites the CSV in place (or set OUT_CSV to a new path).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "outputs" / "ai_sample" / "llm_annotations.csv"
OUT_CSV = ROOT / "outputs" / "ai_sample" / "llm_annotations.csv"  # same file, overwrite


# Keyword rules: (regex or substring, value). Sentence is lowercased for matching.
# Order matters: first match wins for that field.
ACT_TYPE_RULES = [
    (r"\bstabb(?:ed|ing|er)\b|\bknife\s+attack\b", "stabbing"),
    (r"\bshot\b|\bshoot(?:ing|er)\b|\bdischarged\s+(?:a\s+)?(?:gun|firearm)\b", "shooting"),
    (r"\bstrangl(?:ed|ing)\b|\bthrottl", "strangulation"),
    (r"\bblunt\s+force\b|\bbeat(?:en)?\b|\bbatter", "blunt_force"),
    (r"\bpoison", "poisoning"),
    (r"\bset\s+alight\b|\bpetrol\b|\bburned\s+alive\b", "fire"),
    (r"\bassault\b|\battack\b", "assault"),
    (r"\bconceal(?:ed|ment)\b|\bdispos(?:ed|al)\s+of\b|\bcleaned\s+scene\b", "post_offence_conduct"),
]

MEANS_WEAPON_RULES = [
    (r"\bknife\b|\bbladed\b|\bsharp\s+instrument\b", "knife"),
    (r"\bgun\b|\bfirearm\b|\bpistol\b|\bshotgun\b", "firearm"),
    (r"\bblunt\s+instrument\b|\bbat\b|\bhammer\b", "blunt_instrument"),
    (r"\bbare\s+hands\b|\bhands\s+around\b", "bare_hands"),
    (r"\bpoison\b|\btoxic\b|\bantifreeze\b", "poison"),
    (r"\bpetrol\b|\baccelerant\b", "petrol"),
    (r"\bvehicle\b|\bcar\s+driven\b", "vehicle"),
]

MENTAL_STATE_RULES = [
    (r"\bintent\s+to\s+kill\b|\bintended\s+to\s+kill\b|\bmurderous\s+intent\b|\bpurpose\s+to\s+kill\b", "intent_to_kill"),
    (r"\bintent\s+to\s+cause\s+gbh\b|\bserious\s+harm\s+intent\b", "intent_to_cause_GBH"),
    (r"\bdiminished\s+responsibility\b|\babnormality\s+of\s+mental\b", "diminished_responsibility"),
    (r"\bloss\s+of\s+(?:self[- ])?control\b|\bqualifying\s+trigger\b|\bprovocation\b", "provocation_loss_of_control"),
    (r"\bself[- ]defen[cs]e\b|\breasonable\s+force\b|\bhonestly\s+believed\s+.*attack\b", "self_defence"),
    (r"\breckless\b|\brecklessness\b", "recklessness"),
]

LEGAL_CONCEPT_RULES = [
    (r"\bactus\s+reus\b|\bcaused\s+the\s+death\b|\bcaused\s+death\b", "actus_reus"),
    (r"\bmens\s+rea\b|\bintent(?:ion)?\s+to\s+kill\b", "mens_rea"),
    (r"\bself[- ]defen[cs]e\b|\breasonable\s+force\b", "self_defence"),
    (r"\bdiminished\s+responsibility\b", "diminished_responsibility"),
    (r"\bjoint\s+enterprise\b|\baiding\s+and\s+abetting\b|\bparty\s+to\b", "joint_enterprise"),
    (r"\bevidence\b|\bwitness(?:es)?\s+(?:said|stated)\b|\bforensic\b", "evidence"),
    (r"\bsentenc(?:ing|ed)\b|\bminimum\s+term\b|\btariff\b", "sentencing_factor"),
    (r"\bunlawful\s+killing\b|\bhomicide\b", "unlawful_killing"),
]

EVIDENCE_TYPE_RULES = [
    (r"\bcctv\b|\bvideo\s+footage\b|\bcaught\s+on\s+camera\b", "cctv"),
    (r"\bdna\b|\bforensic\b|\bblood\s+spatter\b|\bfingerprint", "forensic"),
    (r"\bwitness(?:es)?\s+(?:said|stated|saw|heard)\b|\beyewitness\b", "witness"),
    (r"\bconfess(?:ed|ion)\b|\badmitted\b|\btold\s+police\b", "confession"),
    (r"\btext\s+message\b|\bwhatsapp\b|\bphone\s+record\b|\bcell\s+site\b", "digital"),
]


def apply_rules(sentence: str, rules: list[tuple], default: str = "unknown") -> str:
    s = sentence.lower()
    for pattern, value in rules:
        if re.search(pattern, s, re.I):
            return value
    return default


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Ensure string columns
    for c in ["act_type", "means_or_weapon", "mental_state", "legal_concept", "evidence_type"]:
        if c not in df.columns:
            continue
        df[c] = df[c].astype(str).str.strip()

    sentence_col = "sentence"
    if sentence_col not in df.columns:
        raise SystemExit(f"Expected column '{sentence_col}' in {IN_CSV}")

    filled = {c: 0 for c in ["act_type", "means_or_weapon", "mental_state", "legal_concept", "evidence_type"]}

    for i, row in df.iterrows():
        sent = str(row.get(sentence_col, ""))
        for col, rules in [
            ("act_type", ACT_TYPE_RULES),
            ("means_or_weapon", MEANS_WEAPON_RULES),
            ("mental_state", MENTAL_STATE_RULES),
            ("legal_concept", LEGAL_CONCEPT_RULES),
            ("evidence_type", EVIDENCE_TYPE_RULES),
        ]:
            if col not in df.columns:
                continue
            val = str(row[col]).strip().lower()
            if val in ("unknown", "nan", ""):
                new_val = apply_rules(sent, rules)
                if new_val != "unknown":
                    df.at[i, col] = new_val
                    filled[col] += 1

    df.to_csv(OUT_CSV, index=False)
    print(f"Updated {OUT_CSV}")
    for col, count in filled.items():
        print(f"   {col}: filled {count} previously 'unknown' rows")


if __name__ == "__main__":
    main()
