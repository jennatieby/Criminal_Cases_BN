#!/usr/bin/env python3

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"

IN_CSV = DATA_RAW / "negative_cases_full.csv"
OUT_CSV = DATA_INTERIM / "negative_cases_cleaned.csv"

DATA_INTERIM.mkdir(parents=True, exist_ok=True)


def clean_ccrc(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text

    pub_idx = t.find("Published:")
    if pub_idx == -1:
        for phrase in [" was found guilty", " was convicted", " was convicted of"]:
            idx = t.find(phrase)
            if idx != -1:
                start = t.rfind(".", 0, idx)
                if start != -1:
                    pub_idx = start + 1
                else:
                    pub_idx = 0
                break
        else:
            pub_idx = 0
    if pub_idx > 0:
        t = t[pub_idx:].strip()

    for marker in ["Contact us", "CCRC 23 Stephenson", "info@ccrc.gov.uk"]:
        idx = t.find(marker)
        if idx != -1:
            t = t[:idx].strip()
            break

    t = re.sub(r"Go back to Home\s*", " ", t, flags=re.I)
    t = re.sub(r"©\s*Copyright,?\s*Criminal Cases Review Commission\s*\d{4}\.?\s*", " ", t, flags=re.I)
    t = re.sub(r"Skip to content\s*", " ", t, flags=re.I)
    t = re.sub(
        r"Cookies on Criminal Cases Review Commission.*?View cookies\s*",
        " ",
        t,
        flags=re.I | re.DOTALL,
    )

    return t


def clean_text_generic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    return t


def clean_negative_text(text: str) -> str:
    t = clean_ccrc(text)
    return clean_text_generic(t)


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input file: {IN_CSV}. Run scrape_negative_cases.py first.")

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    text_col = None
    for c in ["CaseText", "text", "content", "body"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise SystemExit(f"No text column found in {IN_CSV}. Columns: {list(df.columns)}")

    df["CleanText"] = df[text_col].fillna("").astype(str).apply(clean_negative_text)

    min_len = 100
    before = len(df)
    df = df[df["CleanText"].str.len() >= min_len]
    dropped = before - len(df)
    if dropped:
        print(f"[info] Dropped {dropped} rows with CleanText length < {min_len}")

    df.to_csv(OUT_CSV, index=False)
    print(f"Cleaned {len(df)} cases -> {OUT_CSV}")


if __name__ == "__main__":
    main()
