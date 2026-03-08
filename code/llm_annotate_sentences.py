#!/usr/bin/env python3
"""
llm_annotate_sentences.py

Annotate candidate sentences with structured fields using an LLM.

Input:
  outputs/ai_sample/candidate_sentences.csv

Output:
  outputs/ai_sample/llm_annotations.csv

Run:
  export OPENAI_API_KEY="sk-..."
  python code/llm_annotate_sentences.py
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import time

import pandas as pd

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install openai\n"
        f"Import error: {e}"
    )


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "outputs" / "ai_sample" / "candidate_sentences.csv"
OUT_CSV = ROOT / "outputs" / "ai_sample" / "llm_annotations.csv"

# Use gpt-4o-mini (current name); override with OPENAI_MODEL env if needed
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()


SYSTEM_PROMPT = """You are annotating sentences from UK criminal appellate homicide judgments.

For EACH input sentence, respond with a SINGLE JSON object with these keys:
- is_homicide_context: boolean (true/false)
- offence_type: string; one of: murder, manslaughter, attempted_murder, other, unknown
- phase: string; one of: facts, charge, verdict, sentencing, procedural, other, unknown
- actors: array of short strings like defendant, victim, co_defendant, witness, judge, police, unknown
- act_type: short string (e.g. stabbing, shooting, strangulation, assault, post_offence_conduct, other, unknown)
- means_or_weapon: short string (e.g. knife, firearm, bare_hands, vehicle, unknown)
- mental_state: short string (e.g. intent_to_kill, intent_to_cause_GBH, recklessness, provocation_loss_of_control, diminished_responsibility, self_defence, unknown)
- legal_concept: short string (e.g. actus_reus, mens_rea, self_defence, diminished_responsibility, joint_enterprise, evidence, sentencing_factor, other, unknown)
- evidence_type: short string (e.g. witness, forensic, cctv, confession, digital, none, unknown)
- summary_label: 3-6 word descriptive label you would use as an ontology node name, in lower_snake_case.

Be conservative: if information is not clearly present in the sentence, use "unknown" or an empty list.
Return ONLY the JSON object, no prose. No markdown code blocks.
"""


def _quote_bare_json_values(text: str) -> str:
    """Quote bare identifiers used as JSON values (e.g. offence_type: unknown -> offence_type: \"unknown\")."""
    # Match : <optional space> <bare word> followed by , or } or ]
    # Do not match if already quoted (": ") or number/true/false/null
    return re.sub(
        r':\s*([a-zA-Z_][a-zA-Z0-9_]*)(\s*[,}\]])',
        r': "\1"\2',
        text,
    )


def parse_llm_json(content: str) -> dict | None:
    """Parse JSON from LLM response; strip markdown code fences and extract first {...}."""
    if not content or not content.strip():
        return None
    text = content.strip()
    # Strip ```json ... ``` or ``` ... ```
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    # Fix unquoted string values (model often returns offence_type: unknown instead of "unknown")
    text = _quote_bare_json_values(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


DEFAULT_ANNOTATION = {
    "is_homicide_context": None,
    "offence_type": "unknown",
    "phase": "unknown",
    "actors": [],
    "act_type": "unknown",
    "means_or_weapon": "unknown",
    "mental_state": "unknown",
    "legal_concept": "unknown",
    "evidence_type": "unknown",
    "summary_label": "unknown",
}


def annotate_sentence(sentence: str) -> dict:
    msg = f"Sentence: {sentence}"
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                temperature=0.1,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = parse_llm_json(content)
            if parsed is not None:
                for k, v in DEFAULT_ANNOTATION.items():
                    if k not in parsed:
                        parsed[k] = v
                return parsed
            if attempt == 1:
                print(
                    f"[warn] invalid JSON (first 200 chars): {repr(content[:200])}",
                    flush=True,
                )
        except Exception as e:
            print(f"[warn] attempt {attempt}/3 failed: {e}", flush=True)
        time.sleep(2 * attempt)

    return dict(DEFAULT_ANNOTATION)


def main() -> None:
    print(f"[start] model={MODEL}", flush=True)
    print(f"[start] reading {IN_CSV}", flush=True)

    if not IN_CSV.exists():
        raise SystemExit(f"Missing input file: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    print(f"[start] loaded {len(df)} sentences", flush=True)

    ann_rows: list[dict] = []

    for i, row in df.iterrows():
        sent = str(row.get("sentence", "")).strip()
        if not sent:
            continue

        case_id = row.get("case_id", "")
        url = row.get("URL", "")

        if (i % 25) == 0:
            print(f"[progress] {i+1}/{len(df)} case_id={case_id}", flush=True)

        ann = annotate_sentence(sent)
        ann_rows.append(
            {
                "case_id": case_id,
                "URL": url,
                "sentence": sent,
                **ann,
            }
        )

        time.sleep(0.2)

        if (i + 1) % 100 == 0:
            pd.DataFrame(ann_rows).to_csv(OUT_CSV, index=False)
            print(f"[checkpoint] wrote {len(ann_rows)} rows → {OUT_CSV}", flush=True)

    pd.DataFrame(ann_rows).to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote annotations to {OUT_CSV} ({len(ann_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
