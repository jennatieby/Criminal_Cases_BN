#!/usr/bin/env python3
"""
audit_bailii_outcomes.py

Audit BAILII appellate judgments (data/interim/uk_cases_full.cleaned.csv) to
extract:
  1) TRIAL VERDICT: Murder | Manslaughter | Not Guilty | Unknown
  2) APPEAL OUTCOME: UPHELD | QUASHED | REDUCED | RETRIAL | SENTENCE | UNKNOWN

Uses GPT-4o with a strict JSON response contract, processed in batches of 10
with a 1 second sleep between batches.

Resume capability:
  - if outputs/bailii_outcome_audit.csv already contains a case_id, skip it.

Outputs:
  - outputs/bailii_outcome_audit.csv
  - outputs/bailii_audit_errors.csv

Run:
  export OPENAI_API_KEY="sk-..."
  python code/audit_bailii_outcomes.py
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install openai\n"
        f"Import error: {e}"
    )


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv"

OUT_CSV = ROOT / "outputs" / "bailii_outcome_audit.csv"
OUT_ERRORS = ROOT / "outputs" / "bailii_audit_errors.csv"

# Requirement: GPT-4o
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

BATCH_SIZE = 10
BATCH_SLEEP_S = 1.0
MAX_CASE_TEXT_CHARS = int(os.environ.get("MAX_CASE_TEXT_CHARS", "12000"))
MAX_API_RETRIES = int(os.environ.get("MAX_API_RETRIES", "3"))

ALLOWED_TRIAL_VERDICTS = {"Murder", "Manslaughter", "Not Guilty", "Unknown"}
ALLOWED_APPEAL_OUTCOMES = {"UPHELD", "QUASHED", "REDUCED", "RETRIAL", "SENTENCE", "UNKNOWN"}

PROMPT_TEMPLATE = (
    "You are a legal analyst reviewing an appellate homicide judgment "
    "from England and Wales. Read the following judgment and answer "
    "two questions about the primary defendant and the homicide charge.\n\n"
    "Question 1 - TRIAL VERDICT: What was the original trial verdict for the primary "
    "defendant on the homicide charge at the original trial?\n"
    "Answer with ONLY one of: Murder, Manslaughter, Not Guilty, Unknown\n\n"
    "Question 2 - APPEAL OUTCOME: What was the outcome of this appeal for the primary "
    "defendant on the homicide charge?\n"
    "Answer with ONLY one of: "
    "UPHELD - conviction upheld, original verdict stands; "
    "QUASHED - conviction quashed or acquittal entered; "
    "REDUCED - verdict reduced (e.g. murder to manslaughter); "
    "RETRIAL - retrial ordered; "
    "SENTENCE - appeal concerned sentence only not conviction; "
    "UNKNOWN - cannot be determined\n\n"
    "Respond in this exact JSON format with no other text:\n"
    "{{\n"
    "  'trial_verdict': '...',\n"
    "  'appeal_outcome': '...'\n"
    "}}\n\n"
    "Judgment text: {case_text}"
)


def _extract_first_braced_object(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()


def parse_response(content: str) -> dict | None:
    """
    Parse a model response into a Python dict.
    Tolerates:
      - code fences
      - single-quoted JSON
      - extra text (we extract the first { ... } object)
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
        obj = json.loads(braced)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Try Python literal (handles single quotes)
    try:
        obj = ast.literal_eval(braced)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) Naive single-quote replacement fallback
    try:
        fixed = braced.replace("'", '"')
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def canonical_trial_verdict(v: str) -> str:
    s = str(v).strip()
    if not s:
        return "Unknown"
    # normalize common variations
    s_low = s.lower()
    if "murder" in s_low:
        return "Murder"
    if "manslaughter" in s_low:
        return "Manslaughter"
    if "not" in s_low and "guilty" in s_low:
        return "Not Guilty"
    if s in ALLOWED_TRIAL_VERDICTS:
        return s
    return "Unknown"


def canonical_appeal_outcome(v: str) -> str:
    s = str(v).strip()
    if not s:
        return "UNKNOWN"
    s_up = s.upper()
    # normalize common variants like "Upheld" / "Quashed"
    if "UPHELD" in s_up:
        return "UPHELD"
    if "QUASH" in s_up:
        return "QUASHED"
    if "REDUC" in s_up:
        return "REDUCED"
    if "RETRIAL" in s_up:
        return "RETRIAL"
    if "SENTENCE" in s_up:
        return "SENTENCE"
    if "UNKNOWN" in s_up:
        return "UNKNOWN"
    if s_up in ALLOWED_APPEAL_OUTCOMES:
        return s_up
    return "UNKNOWN"


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


def load_existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=["case_id"])
    except Exception:
        return set()
    if df.empty:
        return set()
    return set(df["case_id"].astype(str).unique())


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {IN_CSV}")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    if "CleanText" not in df.columns:
        raise ValueError(f"{IN_CSV} missing CleanText column. Columns={list(df.columns)}")
    if "URL" not in df.columns:
        # Still run, but we won't have stable URLs for error logs.
        df["URL"] = ""

    # Stable case_id scheme consistent with other scripts when case_id absent.
    if "case_id" not in df.columns:
        df["case_id"] = [f"CASE_{i:05d}" for i in range(len(df))]
    else:
        df["case_id"] = df["case_id"].astype(str)

    done_case_ids = load_existing_case_ids(OUT_CSV)

    # Prepare rows to process.
    df["case_id"] = df["case_id"].astype(str)
    to_process = df[~df["case_id"].isin(done_case_ids)].copy()

    print(f"Input cases: {len(df):,}")
    print(f"Already processed: {len(done_case_ids):,}")
    print(f"Remaining: {len(to_process):,}")

    out_rows: list[dict] = []

    # OpenAI client
    client = OpenAI()

    for start in range(0, len(to_process), BATCH_SIZE):
        batch = to_process.iloc[start : start + BATCH_SIZE]
        batch_rows: list[dict] = []

        for _, row in batch.iterrows():
            case_id = str(row["case_id"])
            url = str(row.get("URL", "")) if "URL" in row else ""
            case_text = str(row["CleanText"] or "").strip()
            # Guard against very large inputs triggering token/TPM limits.
            # (We keep the prompt contract the same; only shorten the text.)
            case_text = " ".join(case_text.split())
            if len(case_text) > MAX_CASE_TEXT_CHARS:
                case_text = case_text[:MAX_CASE_TEXT_CHARS]

            if not case_text:
                append_error(
                    OUT_ERRORS,
                    {
                        "case_id": case_id,
                        "url": url,
                        "source_csv": str(IN_CSV),
                        "error": "empty CleanText",
                        "raw_response": "",
                    },
                )
                continue

            prompt = PROMPT_TEMPLATE.format(case_text=case_text)

            try:
                last_err: Exception | None = None
                for attempt in range(1, MAX_API_RETRIES + 1):
                    try:
                        resp = client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1,
                        )
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        msg = str(e)
                        # Retry for rate limiting; for "request too large" we already truncated,
                        # but retrying won't hurt in case of transient limits.
                        if "429" in msg or "rate_limit_exceeded" in msg:
                            time.sleep(2.0 * attempt)
                            continue
                        if "Request too large" in msg or "tokens per min" in msg:
                            # If we still hit size limits, shorten further and retry.
                            if MAX_CASE_TEXT_CHARS > 3000:
                                max_chars_now = int(MAX_CASE_TEXT_CHARS * 0.75)
                            else:
                                max_chars_now = MAX_CASE_TEXT_CHARS
                            short = case_text[:max_chars_now]
                            prompt = PROMPT_TEMPLATE.format(case_text=short)
                            continue
                        raise
                if last_err is not None:
                    raise last_err
                content = (resp.choices[0].message.content or "").strip()

                parsed = parse_response(content)
                if not parsed:
                    raise ValueError(f"Could not parse JSON from response: {content[:200]!r}")

                trial_raw = parsed.get("trial_verdict", "Unknown")
                appeal_raw = parsed.get("appeal_outcome", "UNKNOWN")

                trial_verdict = canonical_trial_verdict(trial_raw)
                appeal_outcome = canonical_appeal_outcome(appeal_raw)

                if trial_verdict not in ALLOWED_TRIAL_VERDICTS:
                    trial_verdict = "Unknown"
                if appeal_outcome not in ALLOWED_APPEAL_OUTCOMES:
                    appeal_outcome = "UNKNOWN"

                batch_rows.append(
                    {
                        "case_id": case_id,
                        "trial_verdict": trial_verdict,
                        "appeal_outcome": appeal_outcome,
                    }
                )
            except Exception as e:
                append_error(
                    OUT_ERRORS,
                    {
                        "case_id": case_id,
                        "url": url,
                        "source_csv": str(IN_CSV),
                        "error": str(e),
                        "raw_response": "",
                    },
                )
                # Requirement: "Save full results" even on failures.
                # We record an explicit UNKNOWN outcome so downstream summaries/crosstabs
                # can account for every case_id.
                batch_rows.append(
                    {"case_id": case_id, "trial_verdict": "Unknown", "appeal_outcome": "UNKNOWN"}
                )

        if batch_rows:
            out_df = pd.DataFrame(batch_rows)
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            if OUT_CSV.exists():
                out_df.to_csv(OUT_CSV, mode="a", header=False, index=False, encoding="utf-8")
            else:
                out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
            out_rows.extend(batch_rows)

        processed_so_far = min(start + BATCH_SIZE, len(to_process))
        print(f"[progress] {processed_so_far}/{len(to_process)} in current run", flush=True)

        # Sleep between batches (requirement).
        time.sleep(BATCH_SLEEP_S)

    # Load full results and print summary.
    if not OUT_CSV.exists():
        print(f"No output created; expected: {OUT_CSV}")
        return

    results = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    if results.empty:
        print("No results found in output CSV.")
        return

    # If multiple runs happened concurrently, the output CSV may contain duplicate case_id rows.
    # Deduplicate for accurate summary statistics.
    before_n = len(results)
    results = results.drop_duplicates(subset=["case_id"], keep="last")
    after_n = len(results)
    if after_n != before_n:
        results.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Distributions
    def dist_table(col: str) -> pd.DataFrame:
        vc = results[col].value_counts(dropna=False)
        pct = (vc / len(results) * 100.0).round(2)
        return pd.DataFrame({"count": vc.astype(int), "percent": pct})

    trial_dist = dist_table("trial_verdict").sort_index()
    appeal_dist = dist_table("appeal_outcome").sort_index()

    print("\n=== Summary ===")
    print("\nTrial verdict distribution:")
    print(trial_dist.to_string())

    print("\nAppeal outcome distribution:")
    print(appeal_dist.to_string())

    print("\nCross-tab: trial_verdict vs appeal_outcome")
    ct = pd.crosstab(results["trial_verdict"], results["appeal_outcome"])
    print(ct.to_string())

    upheld_mask = results["appeal_outcome"] == "UPHELD"
    quashed_reduced_mask = results["appeal_outcome"].isin({"QUASHED", "REDUCED"})

    n_upheld = int(upheld_mask.sum())
    n_quashed_reduced = int(quashed_reduced_mask.sum())

    print("\nKey question (UPHELD vs QUASHED/REDUCED):")
    print(f"  UPHELD cases: {n_upheld}")
    print(f"  QUASHED/REDUCED cases (combined): {n_quashed_reduced}")

    if n_quashed_reduced > 0:
        verdicts_qr = (
            results.loc[quashed_reduced_mask, "trial_verdict"]
            .value_counts(dropna=False)
            .to_dict()
        )
    else:
        verdicts_qr = {}

    verdicts_up = (
        results.loc[upheld_mask, "trial_verdict"]
        .value_counts(dropna=False)
        .to_dict()
    )

    print("  Original trial verdicts within UPHELD:")
    print(verdicts_up)
    print("  Original trial verdicts within QUASHED/REDUCED:")
    print(verdicts_qr)


if __name__ == "__main__":
    main()

