#!/usr/bin/env python3
"""
llm_enrich_ontology_synonyms.py

Use an LLM to suggest extra synonyms for ontology labels; merge with rule-based
ontology and write an enriched YAML. Keeps the same labels and types; only
synonyms are expanded.

Two modes:
  1) --focus actus_mens: enrich only actus reus + mens rea labels (actus_reus_killing,
     unlawful_killing, causation, mens_rea_intent_to_kill, mens_rea_intent_gbh).
  2) --focus all: enrich synonyms for all labels.

Output:
  rules/ontology_actus_mens_enriched.yml   (when --focus actus_mens)
  rules/ontology_general_enriched.yml      (when --focus all)

Then run extraction with the enriched ontology:
  python code/extract_nodes_from_cases.py --ontology rules/ontology_actus_mens_enriched.yml ...

Requires: OPENAI_API_KEY set; pip install openai pyyaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(f"Install openai: pip install openai\nImport error: {e}")

ROOT = Path(__file__).resolve().parents[1]
ONTO_PATH = ROOT / "rules" / "ontology.yml"
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Actus reus + mens rea labels (Method 1: focused enrichment)
ACTUS_MENS_LABELS = [
    "actus_reus_killing",
    "unlawful_killing",
    "causation",
    "mens_rea_intent_to_kill",
    "mens_rea_intent_gbh",
]


def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    return OpenAI()


def _prompt(label: str, type_hint: str, current_synonyms: list[str], focus: str) -> str:
    focus_note = (
        "Focus on actus reus and mens rea: phrases that describe the physical act of killing, unlawful homicide, causation, or intent to kill/cause serious harm."
        if focus == "actus_mens"
        else "Suggest phrases that commonly appear in UK criminal homicide judgments and express this concept."
    )
    syns = ", ".join(current_synonyms[:15]) if current_synonyms else "(none)"
    return f"""You are helping build a phrase list for concept extraction from UK criminal (homicide) judgments.

Label: {label}
Type: {type_hint}
Current synonyms: {syns}

{focus_note}
Suggest 10-20 additional short phrases (2-6 words typical) that could appear in judgments and mean the same concept. Include legal and colloquial variants. One phrase per line. Do not number or bullet. Output only the phrases, nothing else."""


def _parse_phrases(text: str) -> list[str]:
    out = []
    for line in text.strip().splitlines():
        line = line.strip().strip(".-)").strip()
        if not line or len(line) > 120:
            continue
        out.append(line)
    return out


def _merge_synonyms(existing: list[str], new_phrases: list[str], label: str) -> list[str]:
    seen = {label.replace("_", " ").lower()}
    result = []
    for s in (existing or []):
        if isinstance(s, str) and s.strip():
            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(s.strip())
    for s in new_phrases:
        if not s or not isinstance(s, str):
            continue
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(s.strip())
    return result


def enrich_label(client: OpenAI, onto: dict, label: str, focus: str) -> list[str] | None:
    labels_cfg = onto.get("labels", {})
    if label not in labels_cfg:
        return None
    meta = labels_cfg[label]
    type_hint = meta.get("type", "legal_facts")
    current = meta.get("synonyms") or []
    current = [x for x in current if isinstance(x, str)]

    prompt = _prompt(label, type_hint, current, focus)
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You output only phrase lists, one phrase per line. No explanations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = (r.choices[0].message.content or "").strip()
        phrases = _parse_phrases(content)
        return _merge_synonyms(current, phrases, label)
    except Exception as e:
        print(f"  [skip {label}] {e}")
        return None


def main():
    p = argparse.ArgumentParser(description="Enrich ontology synonyms via LLM.")
    p.add_argument(
        "--focus",
        choices=["actus_mens", "all"],
        default="actus_mens",
        help="actus_mens: only actus reus + mens rea labels; all: every label.",
    )
    p.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="Input ontology YAML (default: rules/ontology.yml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path. Default: rules/ontology_actus_mens_enriched.yml or ontology_general_enriched.yml.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print labels that would be enriched; do not call LLM or write.",
    )
    args = p.parse_args()

    in_path = (args.ontology or ONTO_PATH).resolve()
    if not in_path.exists():
        raise SystemExit(f"Ontology not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        onto = yaml.safe_load(f)

    labels_cfg = onto.get("labels", {})
    if not labels_cfg:
        raise SystemExit("Ontology has no 'labels'.")

    if args.focus == "actus_mens":
        to_enrich = [l for l in ACTUS_MENS_LABELS if l in labels_cfg]
    else:
        to_enrich = list(labels_cfg.keys())

    if args.dry_run:
        print(f"Would enrich {len(to_enrich)} labels (focus={args.focus}): {to_enrich[:20]}...")
        return

    if not to_enrich:
        print("No labels to enrich.")
        return

    out_path = args.output
    if out_path is None:
        stem = "ontology_actus_mens_enriched" if args.focus == "actus_mens" else "ontology_general_enriched"
        out_path = ROOT / "rules" / f"{stem}.yml"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = _client()
    updated = 0
    for i, label in enumerate(to_enrich):
        print(f"[{i+1}/{len(to_enrich)}] {label} ...")
        merged = enrich_label(client, onto, label, args.focus)
        if merged is not None:
            onto["labels"][label]["synonyms"] = merged
            updated += 1
        time.sleep(0.3)

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(onto, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Enriched {updated} labels -> {out_path}")


if __name__ == "__main__":
    main()
