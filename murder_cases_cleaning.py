import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "uk_cases_full.csv"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_INTERIM.mkdir(parents=True, exist_ok=True)

INPUT = DATA_RAW
OUTPUT = DATA_INTERIM / "uk_cases_full.cleaned.csv"

def clean_bailii_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text

    # remove bracketed nav links
    t = re.sub(r"\[ *Home *\].*?\[ *DONATE *\]", " ", t, flags=re.I)
    t = re.sub(r"\[ *Home *\].*?$", " ", t, flags=re.I|re.M)
    t = re.sub(r"\[ *Databases *\].*?\[ *Help *\]", " ", t, flags=re.I)
    t = re.sub(r"\[ *[A-Za-z ]+\ *\]", " ", t)       # all square-bracket nav links

    # remove "You are here" and URL lines
    t = re.sub(r"You are here:.*?(?:EWCA|EWHC).*?(html|rtf|pdf)", " ", t, flags=re.I)
    t = re.sub(r"URL:\s*https?://\S+", " ", t, flags=re.I)

    # remove meta lines: Cite as, Neutral Citation, Case No
    t = re.sub(r"Cite as:.*?(EWCA|EWHC).*?\d{4}", " ", t, flags=re.I)
    t = re.sub(r"Neutral Citation Number:.*", " ", t, flags=re.I)
    t = re.sub(r"Case No:.*", " ", t, flags=re.I)

    # remove uppercase court boilerplate blocks
    t = re.sub(r"IN THE COURT OF APPEAL.*?DIVISION", " ", t, flags=re.I)
    t = re.sub(r"Royal Courts? of Justice.*?(London)?.*?\d{4}", " ", t, flags=re.I)
    t = re.sub(r"Computer Aided Transcription.*?Company", " ", t, flags=re.I|re.S)
    t = re.sub(r"Official Shorthand Writers.*?(Company)?", " ", t, flags=re.I)
    t = re.sub(r"HTML VERSION OF JUDGMENT", " ", t, flags=re.I)

    # remove BAILII footer
    t = re.sub(r"BAILII: Copyright Policy.*$", " ", t, flags=re.I|re.S)

    # drop repetitive “Crown Copyright ©” and variants
    t = re.sub(r"Crown Copyright ©", " ", t, flags=re.I)

    # collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = t.strip()

    return t


# Load your scraped CSV
df = pd.read_csv(INPUT)
text_col = "CaseText" if "CaseText" in df.columns else "text"
df["CleanText"] = df[text_col].apply(clean_bailii_text)

# Drop empty or trivially short texts
df = df[df["CleanText"].str.len() > 1000]

# Save cleaned file
df.to_csv(OUTPUT, index=False)
print(f"✅ Cleaned file saved to: {OUTPUT}")
print(df[["URL", "CleanText"]].head(2))