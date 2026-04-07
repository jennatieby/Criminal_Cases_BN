#!/usr/bin/env python3
"""
filter_ni_cases.py

Remove Northern Ireland jurisdiction cases from the negative dataset.

1) Load data/interim/negative_cases_cleaned.csv
2) Flag rows whose case text and/or available metadata include Northern Ireland indicators:
   - "Northern Ireland" (case-insensitive)
   - "\bNI\b" (abbreviation; restricted to court/title-like columns when possible)
   - "NICA" (Northern Ireland Court of Appeal citation)
   - "NIHC" (Northern Ireland High Court citation)
   - "Crown Court in Northern Ireland"
   - "Belfast Crown Court"
   - Other Northern Ireland court identifiers (via the above patterns)
3) Print total cases, number flagged, and flagged case_ids with the matching indicator(s)
4) Save:
   - data/interim/negative_cases_cleaned_ew.csv  (NI cases removed)
   - data/interim/negative_cases_ni_removed.csv   (NI cases only, for inspection)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "interim" / "negative_cases_cleaned.csv"
OUT_EW = ROOT / "data" / "interim" / "negative_cases_cleaned_ew.csv"
OUT_NI = ROOT / "data" / "interim" / "negative_cases_ni_removed.csv"


def _string_like_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable for substring/regex matching."""
    cols: list[str] = []
    for c in df.columns:
        # Only consider object/string columns; numeric columns are not useful for jurisdiction keywords.
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            cols.append(c)
    return cols


def _case_identifier_series(df: pd.DataFrame) -> pd.Series:
    """
    Choose an ID for printing.
    Prefer `case_id`, else use a likely identifying field (`URL`), else fall back to index.
    """
    for candidate in ["case_id", "CaseID", "caseid", "id", "ID"]:
        if candidate in df.columns:
            return df[candidate].astype(str)
    if "URL" in df.columns:
        return df["URL"].astype(str)
    return df.index.astype(str)


def _regex_contains_any(
    df: pd.DataFrame, columns: Iterable[str], pattern: str, case_sensitive: bool = False
) -> pd.Series:
    """Vectorized row mask: does any selected column contain the regex pattern."""
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(pattern, flags=flags)
    mask = pd.Series(False, index=df.index)
    for c in columns:
        # Cast to string; fill NaNs to keep the mask stable.
        s = df[c].fillna("").astype(str)
        mask |= s.str.contains(compiled, na=False)
    return mask


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    total_cases = len(df)

    id_series = _case_identifier_series(df)

    # Columns to scan for jurisdiction identifiers.
    string_cols = _string_like_columns(df)
    if not string_cols:
        raise ValueError("No string/object columns found to search for NI indicators.")

    # When an indicator is specifically about "court name or case title",
    # prefer columns whose names suggest that context.
    colnames_lower = {c: c.lower() for c in df.columns}
    court_title_cols = [
        c
        for c in string_cols
        if any(k in colnames_lower[c] for k in ["court", "title", "citation", "case_name", "case name"])
    ]
    if not court_title_cols:
        court_title_cols = string_cols

    indicators: list[tuple[str, str, str]] = [
        # label, regex, column_pool
        ("Northern Ireland (phrase)", r"northern\s*ireland", "any"),
        # Abbreviation "NI" is very broad; restrict to court/title-like columns when possible.
        ("NI (abbreviation)", r"\bNI\b", "court_title_only"),
        ("NICA citation", r"\bNICA\b", "any"),
        ("NIHC citation", r"\bNIHC\b", "any"),
        ("Northern Ireland High Court", r"northern\s+ireland\s+high\s*court", "any"),
        ("Northern Ireland Court of Appeal", r"northern\s+ireland\s+court\s+of\s+appeal", "any"),
        ("Crown Court in Northern Ireland", r"crown\s*court\s+in\s+northern\s+ireland", "court_title_only"),
        ("Belfast Crown Court", r"belfast\s+crown\s*court", "court_title_only"),
    ]

    indicator_masks: dict[str, pd.Series] = {}
    overall_mask = pd.Series(False, index=df.index)

    for label, regex, pool in indicators:
        cols = string_cols if pool == "any" else court_title_cols
        mask = _regex_contains_any(df, cols, regex, case_sensitive=False)
        indicator_masks[label] = mask
        overall_mask |= mask

    removed_df = df[overall_mask].copy()
    ew_df = df[~overall_mask].copy()

    flagged_n = len(removed_df)
    print(f"Total cases in negative dataset: {total_cases:,}")
    print(f"NI-flagged cases: {flagged_n:,}")

    if flagged_n == 0:
        OUT_EW.parent.mkdir(parents=True, exist_ok=True)
        ew_df.to_csv(OUT_EW, index=False, encoding="utf-8")
        removed_df.to_csv(OUT_NI, index=False, encoding="utf-8")
        print(f"Saved: {OUT_EW}")
        print(f"Saved: {OUT_NI}")
        return

    # Print flagged case identifiers + indicator(s).
    # Keep this readable even if many rows are flagged.
    max_to_print = 200
    flagged_indices = removed_df.index.tolist()
    print("\nFlagged case_ids (with matching indicator(s)):")
    for k, idx in enumerate(flagged_indices[:max_to_print], start=1):
        case_id = str(id_series.loc[idx])
        hits = [label for label, m in indicator_masks.items() if bool(m.loc[idx])]
        hits_str = "; ".join(hits) if hits else "(no indicator label found)"
        print(f"{k:>4}. {case_id} -> {hits_str}")
    if flagged_n > max_to_print:
        print(f"... (showing first {max_to_print} of {flagged_n:,} flagged cases)")

    # Save outputs.
    OUT_EW.parent.mkdir(parents=True, exist_ok=True)
    ew_df.to_csv(OUT_EW, index=False, encoding="utf-8")
    removed_df.to_csv(OUT_NI, index=False, encoding="utf-8")
    print(f"\nSaved EW-only filtered dataset: {OUT_EW}")
    print(f"Saved NI-removed cases for inspection: {OUT_NI}")


if __name__ == "__main__":
    main()

