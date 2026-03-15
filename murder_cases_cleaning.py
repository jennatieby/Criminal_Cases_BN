import re
import pandas as pd
from pathlib import Path

"""
BAILII case text cleaning pipeline.

Reads data/raw/uk_cases_full.csv, cleans CaseText, writes data/interim/uk_cases_full.cleaned.csv.
Drops rows where cleaned text length <= 300.

Cleaning steps (in order):
1. Strip header up to the copyright symbol only: in the first 3000 chars, remove from start up to and
   including "Copyright (c)" or "Crown Copyright ©" (the symbol). The rest of the line is kept so the
   sentence that follows the symbol is not removed.
2. Slice to judgment start: keep only from the first judge line (e.g. THE LORD CHIEF JUSTICE:,
   MR JUSTICE X:) or "HTML VERSION OF JUDGMENT" onward (search limited to first 3000 chars).
3. Remove bracketed nav links ([ Home ], [ Databases ], etc.) and "You are here"/URL lines.
4. Remove meta lines (Cite as:, Neutral Citation Number:, Case No:) and court boilerplate
   (IN THE COURT OF APPEAL...DIVISION, Royal Courts of Justice..., Computer Aided Transcription,
   Official Shorthand Writers, HTML VERSION OF JUDGMENT, Crown Copyright © line only up to 80 chars).
5. Always remove the last BAILII footer: find the last occurrence of "BAILII:" in the text; if it
   starts the policy footer (e.g. "BAILII: Copyright Policy | Disclaimers | Privacy Policy |
   Feedback | Donate to BAILII URL: https://..."), remove from that "BAILII:" to the end of the text.
6. Collapse spaces/tabs and multiple newlines, then strip.
"""

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "uk_cases_full.csv"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_INTERIM.mkdir(parents=True, exist_ok=True)

INPUT = DATA_RAW
OUTPUT = DATA_INTERIM / "uk_cases_full.cleaned.csv"


def find_judgment_start(text: str) -> int:
    """
    Find the start of the judgment body in BAILII text. Return character offset.
    Judgment typically starts with a judge name (LORD JUSTICE, MR JUSTICE, etc.)
    or after 'HTML VERSION OF JUDGMENT'. If no marker found, return 0.
    Only search the first 3000 chars (typical header length) so we do not slice from mid-document.
    """
    if not text or not isinstance(text, str):
        return 0
    head = text[:3000]
    for pattern in [
        r"(?m)^\s*THE\s+LORD\s+CHIEF\s+JUSTICE\s*:",
        r"(?m)^\s*LORD\s+JUSTICE\s+\w+\s*:",
        r"(?m)^\s*LADY\s+JUSTICE\s+\w+\s*:",
        r"(?m)^\s*MR\s+JUSTICE\s+\w+\s*:",
        r"(?m)^\s*MRS\s+JUSTICE\s+\w+\s*:",
        r"\bHTML\s+VERSION\s+OF\s+JUDGMENT\b",
    ]:
        m = re.search(pattern, head, re.I)
        if m:
            return m.start()
    return 0


def clean_bailii_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text

    # Remove header prefix up to and including the copyright symbol only: "(c)" or "©"
    # Only look in the first 3000 chars. Do not remove the rest of the line (judgment often starts right after the symbol).
    header_zone = t[:3000]
    m = re.search(r"(?s)^.*?(?:Copyright\s*\([cC]\)|Crown Copyright\s*©)", header_zone)
    if m:
        t = t[m.end() :].lstrip()

    # Keep only from judgment start onward (avoids header/nav; prevents greedy regexes from deleting body)
    start = find_judgment_start(t)
    if start > 0:
        t = t[start:]

    # remove bracketed nav links
    t = re.sub(r"\[ *Home *\].*?\[ *DONATE *\]", " ", t, flags=re.I)
    t = re.sub(r"\[ *Home *\].*?$", " ", t, flags=re.I|re.M)
    t = re.sub(r"\[ *Databases *\].*?\[ *Help *\]", " ", t, flags=re.I)
    t = re.sub(r"\[ *[A-Za-z ]+\ *\]", " ", t)       # all square-bracket nav links

    # remove "You are here" and URL lines
    t = re.sub(r"You are here:[^\n]*", " ", t, flags=re.I)
    t = re.sub(r"URL:\s*https?://\S+", " ", t, flags=re.I)

    # remove meta lines: Cite as, Neutral Citation, Case No (line-limited)
    t = re.sub(r"Cite as:[^\n]*", " ", t, flags=re.I)
    t = re.sub(r"Neutral Citation Number:[^\n]*", " ", t, flags=re.I)
    t = re.sub(r"Case No:[^\n]*", " ", t, flags=re.I)

    # remove uppercase court boilerplate blocks
    t = re.sub(r"IN THE COURT OF APPEAL[^\n]*DIVISION", " ", t, flags=re.I)
    t = re.sub(r"Royal Courts? of Justice[^\n]*\d{4}", " ", t, flags=re.I)
    t = re.sub(r"Computer Aided Transcription[^\n]*", " ", t, flags=re.I)
    t = re.sub(r"Official Shorthand Writers[^\n]*", " ", t, flags=re.I)
    t = re.sub(r"HTML VERSION OF JUDGMENT", " ", t, flags=re.I)
    # Crown Copyright line only: match date part (e.g. "Tuesday 17 June 2008"); do not eat following judge line
    t = re.sub(r"Crown Copyright ©[^\n]{0,80}", " ", t, flags=re.I)

    # Always remove the last sentence/block starting with "BAILII:" (footer: Copyright Policy | Disclaimers | URL etc.)
    last_bailii = t.rfind("BAILII:")
    if last_bailii >= 0:
        tail = t[last_bailii:]
        if re.search(r"BAILII:\s*Copyright\s+Policy", tail, re.I):
            t = t[:last_bailii].rstrip()

    # collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = t.strip()

    return t


# Load your scraped CSV
df = pd.read_csv(INPUT)
text_col = "CaseText" if "CaseText" in df.columns else "text"
df["CleanText"] = df[text_col].apply(clean_bailii_text)

# Drop empty or trivially short texts (after diagnostics, 300 is safer)
df = df[df["CleanText"].str.len() > 300]

# Save cleaned file
df.to_csv(OUTPUT, index=False)
print(f"Cleaned file saved to: {OUTPUT}")
print(df[["URL", "CleanText"]].head(2))