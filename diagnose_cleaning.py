"""Diagnose why cleaning drops so many cases."""
import pandas as pd
import re

# Get raw text of one dropped case to inspect
def _inspect_raw():
    df = pd.read_csv("data/raw/uk_cases_full.csv")
    text_col = "CaseText" if "CaseText" in df.columns else "text"
    raw = str(df[text_col].iloc[0])
    # Find key patterns
    for name, pat in [
        ("BAILII: Copyright Policy", "BAILII: Copyright Policy"),
        ("You are here:", "You are here:"),
        ("IN THE COURT", "IN THE COURT"),
    ]:
        pos = raw.find(pat)
        print(f"  First '{name}' at position {pos} (of {len(raw)})")
    # What's around position 400?
    print(f"\n  Raw chars 350-800: {repr(raw[350:800])}")

_inspect_raw()
print()

def clean_bailii_text(text):
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r"\[ *Home *\].*?\[ *DONATE *\]", " ", t, flags=re.I)
    t = re.sub(r"\[ *Home *\].*?$", " ", t, flags=re.I|re.M)
    t = re.sub(r"\[ *Databases *\].*?\[ *Help *\]", " ", t, flags=re.I)
    t = re.sub(r"\[ *[A-Za-z ]+\ *\]", " ", t)
    t = re.sub(r"You are here:.*?(?:EWCA|EWHC).*?(html|rtf|pdf)", " ", t, flags=re.I)
    t = re.sub(r"URL:\s*https?://\S+", " ", t, flags=re.I)
    t = re.sub(r"Cite as:.*?(EWCA|EWHC).*?\d{4}", " ", t, flags=re.I)
    t = re.sub(r"Neutral Citation Number:.*", " ", t, flags=re.I)
    t = re.sub(r"Case No:.*", " ", t, flags=re.I)
    t = re.sub(r"IN THE COURT OF APPEAL.*?DIVISION", " ", t, flags=re.I)
    t = re.sub(r"Royal Courts? of Justice.*?(London)?.*?\d{4}", " ", t, flags=re.I)
    t = re.sub(r"Computer Aided Transcription.*?Company", " ", t, flags=re.I|re.S)
    t = re.sub(r"Official Shorthand Writers.*?(Company)?", " ", t, flags=re.I)
    t = re.sub(r"HTML VERSION OF JUDGMENT", " ", t, flags=re.I)
    t = re.sub(r"BAILII: Copyright Policy.*$", " ", t, flags=re.I|re.S)
    t = re.sub(r"Crown Copyright ©", " ", t, flags=re.I)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

df = pd.read_csv("data/raw/uk_cases_full.csv")
text_col = "CaseText" if "CaseText" in df.columns else "text"
df["CleanText"] = df[text_col].apply(clean_bailii_text)
df["raw_len"] = df[text_col].fillna("").astype(str).str.len()
df["clean_len"] = df["CleanText"].fillna("").astype(str).str.len().astype(int)

print("=== LENGTH STATS ===")
print(f"Raw - min: {df['raw_len'].min()}, max: {df['raw_len'].max()}, mean: {df['raw_len'].mean():.0f}")
print(f"Clean - min: {df['clean_len'].min()}, max: {df['clean_len'].max()}, mean: {df['clean_len'].mean():.0f}")
print(f"Rows with raw_len > 1000: {(df['raw_len'] > 1000).sum()}")
print(f"Rows with clean_len > 1000: {(df['clean_len'] > 1000).sum()}")
print(f"DROPPED (clean <= 1000): {(df['clean_len'] <= 1000).sum()}")
print(f"Empty clean (len=0): {(df['clean_len'] == 0).sum()}")
print(f"NaN CaseText: {df[text_col].isna().sum()}")

dropped = df[df["clean_len"] <= 1000].head(5)
print("\n=== SAMPLE DROPPED (first 5) ===")
for i, r in dropped.iterrows():
    raw_preview = str(r[text_col])[:150] if pd.notna(r[text_col]) else "(NaN)"
    clean_preview = str(r["CleanText"])[:200] if r["clean_len"] > 0 else "(empty)"
    print(f"\n--- Row {i} ---")
    print(f"  raw_len={r['raw_len']}, clean_len={r['clean_len']}")
    print(f"  RAW start: {repr(raw_preview)}...")
    print(f"  CLEAN: {repr(clean_preview)}...")
