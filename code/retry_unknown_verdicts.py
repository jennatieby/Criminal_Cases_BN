#!/usr/bin/env python3
"""
retry_unknown_verdicts.py

Retry metadata extraction for cases with unknown verdicts (Verdict = NaN) in
case_node_matrix.csv, using a longer case-text excerpt and a more permissive prompt.

Workflow:
1) Load case_node_matrix.csv and find case_ids where Verdict is NaN.
2) Load original cleaned corpora (positive + negative) and get CleanText for those case_ids.
3) Call OpenAI in batches of 10 with 1s sleeps between batches.
4) If verdict can be resolved, overwrite Verdict in the matrix.
5) If still NaN after retry, remove those rows from the matrix and save the cleaned matrix.
6) Print how many were resolved vs removed.

Outputs:
  - outputs/metadata_retry_results.csv
  - outputs/metadata_retry_errors.csv
  - case_node_matrix.csv (overwritten with cleaned matrix)

Run:
  export OPENAI_API_KEY="sk-..."
  python code/retry_unknown_verdicts.py
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

MATRIX_PATH = ROOT / "case_node_matrix.csv"

POS_CSV = ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv"
NEG_CSV = ROOT / "data" / "interim" / "negative_cases_cleaned.csv"

OUT_RESULTS = ROOT / "outputs" / "metadata_retry_results.csv"
OUT_ERRORS = ROOT / "outputs" / "metadata_retry_errors.csv"

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()

BATCH_SIZE = 10
BATCH_SLEEP_S = 1.0

ALLOWED_VERDICTS = {"Murder", "Manslaughter", "Not Guilty", "Unknown"}


PROMPT_TEMPLATE = """You are a legal analyst reviewing appellate homicide judgments from England and Wales.

Your task is to infer the ORIGINAL TRIAL verdict for the primary defendant on the homicide charge.

Be permissive in using signals, for example:
- \"the appellant was convicted\" (infer conviction type if stated nearby)
- \"convicted of murder\" / \"convicted of manslaughter\"
- \"conviction was quashed\" (use what the conviction was BEFORE it was quashed)
- \"substituted verdict of manslaughter\" (treat original conviction as murder unless clearly otherwise)
- \"pleaded guilty to manslaughter\" (manslaughter)
- \"acquitted\" / \"not guilty\" (not guilty)

If the judgment does not clearly establish the trial verdict, answer Unknown.

Answer with ONLY one of: Murder, Manslaughter, Not Guilty, Unknown

Respond in this exact JSON format with no other text:
{{
  'verdict': '...'
}}

Judgment excerpt (may include start and end of judgment):
{case_text}
"""


def _extract_first_braced_object(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()


def parse_response(content: str) -> dict | None:
    if not content or not content.strip():
        return None
    text = content.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    braced = _extract_first_braced_object(text) or text
    try:
        return json.loads(braced)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(braced)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        fixed = braced.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        return None


def call_llm(case_text: str) -> tuple[str, str]:
    """Return (verdict_raw, raw_response_text)."""
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
            if verdict not in ALLOWED_VERDICTS:
                verdict = "Unknown"
            return verdict, content
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)
    assert last_err is not None
    raise last_err


def encode_verdict(v: str) -> float:
    if v == "Murder":
        return 2.0
    if v == "Manslaughter":
        return 1.0
    if v == "Not Guilty":
        return 0.0
    return float("nan")


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
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "CleanText" not in df.columns:
        raise ValueError(f"{csv_path} missing CleanText column; columns={list(df.columns)}")
    if "case_id" not in df.columns:
        df["case_id"] = [f"{prefix}{i:05d}" for i in range(len(df))]
    if "URL" not in df.columns:
        df["URL"] = ""
    return df[["case_id", "URL", "CleanText"]].copy()


def make_long_excerpt(text: str, head_chars: int = 12000, tail_chars: int = 8000) -> str:
    t = (text or "").strip()
    if len(t) <= head_chars + tail_chars + 200:
        return t
    head = t[:head_chars].rstrip()
    tail = t[-tail_chars:].lstrip()
    return head + "\n\n[... TEXT TRUNCATED ...]\n\n" + tail


def main() -> int:
    p = argparse.ArgumentParser(description="Retry NaN verdict cases and drop unresolved from matrix.")
    p.add_argument("--matrix", type=Path, default=MATRIX_PATH, help="Path to case_node_matrix.csv")
    p.add_argument("--pos", type=Path, default=POS_CSV, help="Positive cleaned CSV")
    p.add_argument("--neg", type=Path, default=NEG_CSV, help="Negative cleaned CSV")
    args = p.parse_args()

    if not args.matrix.exists():
        raise FileNotFoundError(f"Matrix not found: {args.matrix}")

    mat = pd.read_csv(args.matrix, encoding="utf-8-sig")
    if "case_id" not in mat.columns:
        mat = mat.rename(columns={mat.columns[0]: "case_id"})
    mat["case_id"] = mat["case_id"].astype(str)

    nan_cases = mat.loc[pd.isna(mat["Verdict"]), "case_id"].astype(str).tolist()
    if not nan_cases:
        print("No NaN verdict cases found in case_node_matrix.csv. Nothing to retry.")
        return 0

    print(f"[start] NaN verdict cases: {len(nan_cases)}")

    pos_df = load_cases(args.pos, prefix="CASE_").assign(source_csv=str(args.pos))
    neg_df = load_cases(args.neg, prefix="NEG_").assign(source_csv=str(args.neg))
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Build lookup
    lookup = all_df.set_index("case_id")[["URL", "CleanText", "source_csv"]].to_dict(orient="index")

    rows = []
    resolved = 0
    still_nan = []

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(nan_cases), BATCH_SIZE):
        batch_ids = nan_cases[start : start + BATCH_SIZE]
        batch_rows = []

        for case_id in batch_ids:
            item = lookup.get(case_id)
            if not item:
                still_nan.append(case_id)
                append_error(
                    OUT_ERRORS,
                    {"case_id": case_id, "url": "", "source_csv": "", "error": "case_id not found in corpora", "raw_response": ""},
                )
                continue

            url = str(item.get("URL", ""))
            src = str(item.get("source_csv", ""))
            text = make_long_excerpt(str(item.get("CleanText", "")))
            if not text.strip():
                still_nan.append(case_id)
                append_error(
                    OUT_ERRORS,
                    {"case_id": case_id, "url": url, "source_csv": src, "error": "empty CleanText", "raw_response": ""},
                )
                continue

            try:
                verdict_raw, raw_resp = call_llm(text)
                verdict_enc = encode_verdict(verdict_raw)
                if np.isnan(verdict_enc):
                    still_nan.append(case_id)
                else:
                    resolved += 1
                batch_rows.append(
                    {
                        "case_id": case_id,
                        "url": url,
                        "source_csv": src,
                        "verdict_raw": verdict_raw,
                        "verdict_encoded": verdict_enc,
                    }
                )
            except Exception as e:
                still_nan.append(case_id)
                append_error(
                    OUT_ERRORS,
                    {"case_id": case_id, "url": url, "source_csv": src, "error": str(e), "raw_response": ""},
                )

        if batch_rows:
            pd.DataFrame(batch_rows).to_csv(
                OUT_RESULTS,
                mode="a" if OUT_RESULTS.exists() else "w",
                header=not OUT_RESULTS.exists(),
                index=False,
                encoding="utf-8",
            )
            rows.extend(batch_rows)

        print(f"[progress] {min(start + BATCH_SIZE, len(nan_cases))}/{len(nan_cases)}", flush=True)
        time.sleep(BATCH_SLEEP_S)

    # Apply resolved verdicts to matrix
    if rows:
        upd = pd.DataFrame(rows)[["case_id", "verdict_encoded"]].copy()
        upd["case_id"] = upd["case_id"].astype(str)
        mat = mat.merge(upd, on="case_id", how="left", suffixes=("", "_new"))
        mat["Verdict"] = mat["verdict_encoded"].combine_first(mat["Verdict"])
        mat = mat.drop(columns=["verdict_encoded"])

    # Remove still-NaN verdict cases
    before = len(mat)
    mat_clean = mat[~pd.isna(mat["Verdict"])].copy()
    removed = before - len(mat_clean)

    mat_clean.to_csv(args.matrix, index=False, encoding="utf-8")

    print(f"\nResolved verdicts: {resolved}")
    print(f"Removed unresolved cases: {removed}")
    print(f"Cleaned matrix saved to: {args.matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

