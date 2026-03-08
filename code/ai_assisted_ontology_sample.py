#!/usr/bin/env python3
"""
Ran into an issue with extremely greedy extraction of nodes from cases.
Using AI to sample cases and extract sentences to create an ontology to compare to the original one. 

Sample 30–50 cases and automatically extract:
- homicide-relevant sentences
- candidate phrases (noun chunks, key verbs)

Outputs (in outputs/ai_sample/):
- cases_sampled.csv           (which cases were picked)
- candidate_sentences.csv     (one row per relevant sentence)
- candidate_phrases.csv       (aggregated phrases across cases)
"""

from pathlib import Path
import random
import re

import pandas as pd
import spacy

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv"
OUT_DIR = ROOT / "outputs" / "ai_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Config ----------------
N_CASES = 40  # adjust to 30–50 as you like
MAX_CHARS_PER_CASE = 30000  # truncate very long texts for speed

HOMICIDE_KEYWORDS = [
    "murder",
    "manslaughter",
    "homicide",
    "attempted murder",
    "killed",
    "killing",
    "stabbed",
    "stabbing",
    "shot",
    "shooting",
    "strangled",
    "fatal injury",
    "cause the death",
]

# compile once
HOMICIDE_RE = re.compile(
    r"|".join(re.escape(k.lower()) for k in HOMICIDE_KEYWORDS),
    flags=re.I,
)

# ---------------- Load spaCy ----------------
nlp = spacy.load("en_core_web_sm", disable=["ner"])  # we mostly want syntax + sents


def case_is_likely_homicide(text: str) -> bool:
    """Simple heuristic: keyword in first N chars."""
    if not isinstance(text, str):
        return False
    head = text[:8000].lower()
    return bool(HOMICIDE_RE.search(head))


def main() -> None:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Missing cleaned CSV: {CLEAN_CSV}")

    df = pd.read_csv(CLEAN_CSV)
    text_col = None
    for cand in ["CleanText", "CaseText", "text"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise ValueError(f"No text column found in {CLEAN_CSV}")

    # Add a simple case_id if missing
    if "case_id" not in df.columns:
        df["case_id"] = [f"CLEAN_{i:05d}" for i in range(len(df))]

    # Filter to likely homicide cases first
    df["is_homicide_like"] = df[text_col].astype(str).apply(case_is_likely_homicide)
    homicide_df = df[df["is_homicide_like"]].copy()
    if homicide_df.empty:
        homicide_df = df.copy()  # fallback: sample from all

    # Sample N_CASES (or fewer if not enough)
    sample_df = homicide_df.sample(
        n=min(N_CASES, len(homicide_df)),
        random_state=42,
        replace=False,
    ).reset_index(drop=True)

    # Save which cases we sampled
    sample_df[["case_id", "URL"]].to_csv(
        OUT_DIR / "cases_sampled.csv", index=False
    )

    sent_rows = []
    phrase_rows = []

    for _, row in sample_df.iterrows():
        case_id = row["case_id"]
        url = row.get("URL", "")
        raw = str(row.get(text_col) or "")

        if not raw.strip():
            continue

        # Truncate for speed
        text = raw[:MAX_CHARS_PER_CASE]

        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            # only keep homicide-relevant sentences
            if not HOMICIDE_RE.search(s.lower()):
                continue

            sent_rows.append(
                {
                    "case_id": case_id,
                    "URL": url,
                    "sentence": s,
                }
            )

            # extract noun chunks and root verb as candidate phrases
            for nc in sent.noun_chunks:
                phrase_rows.append(
                    {
                        "case_id": case_id,
                        "URL": url,
                        "sentence": s,
                        "phrase_type": "noun_chunk",
                        "phrase": nc.text.strip(),
                    }
                )

            # main verb (if any)
            root_verbs = [t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"]
            for rv in root_verbs:
                phrase_rows.append(
                    {
                        "case_id": case_id,
                        "URL": url,
                        "sentence": s,
                        "phrase_type": "root_verb",
                        "phrase": rv.lemma_.strip(),
                    }
                )

    # Save candidate sentences
    pd.DataFrame(sent_rows).to_csv(
        OUT_DIR / "candidate_sentences.csv", index=False
    )

    # Save candidate phrases
    phrases_df = pd.DataFrame(phrase_rows)
    if not phrases_df.empty:
        # aggregate phrase frequencies
        agg = (
            phrases_df.groupby(["phrase_type", "phrase"], as_index=False)
            .agg(
                cases_with=("case_id", "nunique"),
                occurrences=("sentence", "count"),
            )
            .sort_values(["phrase_type", "occurrences"], ascending=[True, False])
        )
        agg.to_csv(OUT_DIR / "candidate_phrases.csv", index=False)
    else:
        # still write empty file so you know it ran
        phrases_df.to_csv(OUT_DIR / "candidate_phrases.csv", index=False)

    print(f"✅ Sampled {len(sample_df)} cases → {OUT_DIR}")


if __name__ == "__main__":
    main()