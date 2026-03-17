#!/usr/bin/env python3
"""
extract_case_metadata.py

Extract per-case metadata from cleaned case corpora using OpenAI:
1) VERDICT at original trial (Murder | Manslaughter | Not Guilty | Unknown)
2) DEATH ESTABLISHED (Yes | No | Unclear)

Inputs (defaults):
  - data/interim/uk_cases_full.cleaned.csv         (positive / BAILII)
  - data/interim/negative_cases_cleaned.csv        (negative / CCRC)

Output:
  - outputs/case_metadata.csv
      columns: case_id, verdict_raw, verdict_encoded, death_established
  - outputs/metadata_extraction_errors.csv (logs failures)

Then:
  - joins outputs/case_metadata.csv onto case_node_matrix.csv on case_id
  - overwrites Verdict column with verdict_encoded
  - overwrites DeathOfHumanBeing column with death_established
  - writes updated case_node_matrix.csv and prints updated stats

Run:
  export OPENAI_API_KEY="sk-..."
  python code/extract_case_metadata.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install openai\n"
        f"Import error: {e}"
    )


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POS = ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv"
DEFAULT_NEG = ROOT / "data" / "interim" / "negative_cases_cleaned.csv"

OUT_META = ROOT / "outputs" / "case_metadata.csv"
OUT_ERRORS = ROOT / "outputs" / "metadata_extraction_errors.csv"

MATRIX_PATH = ROOT / "case_node_matrix.csv"

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()

BATCH_SIZE = 10
BATCH_SLEEP_S = 1.0


PROMPT_TEMPLATE = """You are a legal analyst reviewing appellate homicide judgments
from England and Wales. Read the following judgment and answer
two questions.

Question 1 - VERDICT: What was the final verdict for the primary
defendant on the homicide charge at the original trial?
Answer with ONLY one of: Murder, Manslaughter, Not Guilty, Unknown

Question 2 - DEATH ESTABLISHED: Does the judgment confirm or
treat as established that a human being died as a result of the
defendant's actions? This may be stated explicitly or treated as
an undisputed fact from the original trial.
Answer with ONLY one of: Yes, No, Unclear

Respond in this exact JSON format with no other text:
{{
  'verdict': '...',
  'death_established': '...'
}}

Judgment text: {case_text}
"""


ALLOWED_VERDICTS = {"Murder", "Manslaughter", "Not Guilty", "Unknown"}
ALLOWED_DEATH = {"Yes", "No", "Unclear"}


def encode_verdict(v: str) -> float:
    if v == "Murder":
        return 2.0
    if v == "Manslaughter":
        return 1.0
    if v == "Not Guilty":
        return 0.0
    return float("nan")


def encode_death(d: str) -> float:
    if d == "Yes":
        return 1.0
    if d == "No":
        return 0.0
    return float("nan")


def _extract_first_braced_object(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()


def parse_response(content: str) -> dict | None:
    """
    Parse the model response. The prompt requests single-quoted JSON; tolerate:
    - single quotes
    - markdown fences
    - extra text (we extract first {...})
    """
    if not content or not content.strip():
        return None
    text = content.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    braced = _extract_first_braced_object(text) or text
    # 1) Try strict JSON
    try:
        return json.loads(braced)
    except Exception:
        pass
    # 2) Try Python literal (handles single quotes)
    try:
        obj = ast.literal_eval(braced)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 3) Naive quote fix: replace single quotes with double quotes
    try:
        fixed = braced.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        return None


def call_llm(case_text: str) -> tuple[str, str, str]:
    """
    Returns (verdict_raw, death_raw, raw_response_text).
    Raises on repeated failure.
    """
    msg = PROMPT_TEMPLATE.format(case_text=case_text)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": msg}],
                temperature=0.1,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = parse_response(content)
            if not parsed:
                raise ValueError(f"Could not parse JSON from response: {content[:200]!r}")
            verdict = str(parsed.get("verdict", "")).strip()
            death = str(parsed.get("death_established", "")).strip()
            if verdict not in ALLOWED_VERDICTS:
                verdict = "Unknown"
            if death not in ALLOWED_DEATH:
                death = "Unclear"
            return verdict, death, content
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)
    assert last_err is not None
    raise last_err


def ensure_error_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["case_id", "url", "source_csv", "error", "raw_response"],
        )
        w.writeheader()


def append_error(path: Path, row: dict) -> None:
    ensure_error_log_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["case_id", "url", "source_csv", "error", "raw_response"],
        )
        w.writerow(row)


def load_cases(csv_path: Path, prefix: str) -> pd.DataFrame:
    """
    Load a cleaned corpus CSV and add case_id consistent with extraction pipeline:
      CASE_00000... for positive, NEG_00000... for negative.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "CleanText" not in df.columns:
        raise ValueError(f"{csv_path} missing CleanText column; columns={list(df.columns)}")
    # mimic extract_nodes_from_cases.py behaviour when case_id is absent
    if "case_id" not in df.columns:
        df["case_id"] = [f"{prefix}{i:05d}" for i in range(len(df))]
    if "URL" not in df.columns:
        df["URL"] = ""
    return df[["case_id", "URL", "CleanText"]].copy()


def print_distributions(meta_df: pd.DataFrame) -> None:
    print("\nVerdict distribution (raw):")
    print(meta_df["verdict_raw"].value_counts(dropna=False).to_string())
    print("\nDeath established distribution (raw):")
    print(meta_df["death_established_raw"].value_counts(dropna=False).to_string())

    n_unknown = int((meta_df["verdict_raw"] == "Unknown").sum())
    n_unclear = int((meta_df["death_established_raw"] == "Unclear").sum())
    print(f"\nUnknown verdicts: {n_unknown}")
    print(f"Unclear death flags: {n_unclear}")


def join_onto_matrix(meta_df: pd.DataFrame) -> None:
    if not MATRIX_PATH.exists():
        print(f"\n[warn] {MATRIX_PATH} not found; skipping join/update.")
        return
    mat = pd.read_csv(MATRIX_PATH, encoding="utf-8-sig")
    if "case_id" not in mat.columns:
        # case_id is likely the first column; handle both formats
        if mat.columns[0] != "case_id":
            mat = mat.rename(columns={mat.columns[0]: "case_id"})
    mat["case_id"] = mat["case_id"].astype(str)

    m = meta_df[["case_id", "verdict_encoded", "death_established"]].copy()
    m["case_id"] = m["case_id"].astype(str)

    merged = mat.merge(m, on="case_id", how="left", suffixes=("", "_meta"))

    # Overwrite Verdict and DeathOfHumanBeing if metadata present
    if "Verdict" in merged.columns:
        merged["Verdict"] = merged["verdict_encoded"].combine_first(merged["Verdict"])
    else:
        merged["Verdict"] = merged["verdict_encoded"]

    if "DeathOfHumanBeing" in merged.columns:
        merged["DeathOfHumanBeing"] = merged["death_established"].combine_first(merged["DeathOfHumanBeing"])
    else:
        merged["DeathOfHumanBeing"] = merged["death_established"]

    merged = merged.drop(columns=["verdict_encoded", "death_established"])
    merged.to_csv(MATRIX_PATH, index=False, encoding="utf-8")

    # Updated stats
    print("\nUpdated node prevalence (% of cases with node=1):")
    node_cols = [c for c in merged.columns if c not in {"case_id", "Verdict"}]
    # Only compute prevalence for binary-ish columns
    prev = (merged[node_cols].fillna(0).astype(float).gt(0).sum(axis=0) / max(1, len(merged)) * 100.0).sort_values(ascending=False)
    for node, pct in prev.items():
        print(f"  {node}: {pct:.1f}%")

    print("\nUpdated verdict distribution (encoded):")
    print(merged["Verdict"].value_counts(dropna=False).sort_index().to_string())


def main() -> int:
    p = argparse.ArgumentParser(description="Extract verdict + death-established metadata via OpenAI.")
    p.add_argument("--pos", type=Path, default=DEFAULT_POS, help="Positive cleaned CSV (BAILII).")
    p.add_argument("--neg", type=Path, default=DEFAULT_NEG, help="Negative cleaned CSV (CCRC).")
    args = p.parse_args()

    OUT_META.parent.mkdir(parents=True, exist_ok=True)

    pos_df = load_cases(args.pos, prefix="CASE_")
    neg_df = load_cases(args.neg, prefix="NEG_")
    all_df = pd.concat(
        [
            pos_df.assign(source_csv=str(args.pos)),
            neg_df.assign(source_csv=str(args.neg)),
        ],
        ignore_index=True,
    )

    # Resume support: skip case_ids already processed
    done: set[str] = set()
    if OUT_META.exists():
        try:
            prev = pd.read_csv(OUT_META, encoding="utf-8-sig", usecols=["case_id"])
            done = set(prev["case_id"].dropna().astype(str))
            print(f"[resume] found {len(done)} previously processed case_ids in {OUT_META}")
        except Exception:
            pass

    rows_out: list[dict] = []
    if OUT_META.exists():
        # We will append; keep rows_out for new ones only
        pass
    else:
        rows_out = []

    to_process = all_df[~all_df["case_id"].astype(str).isin(done)].copy()
    print(f"[start] total cases={len(all_df)} to_process={len(to_process)} model={MODEL}")

    # Process in batches of 10 with a 1s sleep between batches
    for start in range(0, len(to_process), BATCH_SIZE):
        batch = to_process.iloc[start : start + BATCH_SIZE]
        batch_rows: list[dict] = []

        for _, row in batch.iterrows():
            case_id = str(row["case_id"])
            url = str(row.get("URL", ""))
            text = str(row.get("CleanText", "")).strip()
            if not text:
                # Treat as failure
                append_error(
                    OUT_ERRORS,
                    {"case_id": case_id, "url": url, "source_csv": row["source_csv"], "error": "empty CleanText", "raw_response": ""},
                )
                continue

            try:
                verdict_raw, death_raw, raw_resp = call_llm(text)
                batch_rows.append(
                    {
                        "case_id": case_id,
                        "verdict_raw": verdict_raw,
                        "verdict_encoded": encode_verdict(verdict_raw),
                        "death_established_raw": death_raw,
                        "death_established": encode_death(death_raw),
                    }
                )
            except Exception as e:
                append_error(
                    OUT_ERRORS,
                    {"case_id": case_id, "url": url, "source_csv": row["source_csv"], "error": str(e), "raw_response": ""},
                )

        if batch_rows:
            out_df = pd.DataFrame(batch_rows)
            if OUT_META.exists():
                out_df.to_csv(OUT_META, mode="a", header=False, index=False, encoding="utf-8")
            else:
                out_df.to_csv(OUT_META, index=False, encoding="utf-8")
            rows_out.extend(batch_rows)

        print(f"[progress] processed {min(start + BATCH_SIZE, len(to_process))}/{len(to_process)}", flush=True)
        time.sleep(BATCH_SLEEP_S)

    # Load full metadata for distributions + join (includes resumed rows)
    meta_df = pd.read_csv(OUT_META, encoding="utf-8-sig")
    print_distributions(meta_df)

    # Join onto case_node_matrix.csv and overwrite columns
    join_onto_matrix(meta_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

