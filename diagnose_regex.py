"""Test each regex to find which one nukes the content."""
import pandas as pd
import re

df = pd.read_csv("data/raw/uk_cases_full.csv")
text_col = "CaseText"
raw = str(df[text_col].iloc[0])

# Apply regexes one by one, measure length after each
t = raw
steps = []

def apply(pat, flags=0, name=None):
    global t
    before = len(t)
    t = re.sub(pat, " ", t, flags=flags)
    after = len(t)
    steps.append((name or pat[:40], before, after, before - after))

# Same order as murder_cases_cleaning.py
apply(r"\[ *Home *\].*?\[ *DONATE *\]", re.I, "nav1")
apply(r"\[ *Home *\].*?$", re.I|re.M, "nav2")
apply(r"\[ *Databases *\].*?\[ *Help *\]", re.I, "nav3")
apply(r"\[ *[A-Za-z ]+\ *\]", 0, "nav4")
apply(r"You are here:.*?(?:EWCA|EWHC).*?(html|rtf|pdf)", re.I, "you_are_here")
apply(r"URL:\s*https?://\S+", re.I, "url")
apply(r"Cite as:.*?(EWCA|EWHC).*?\d{4}", re.I, "cite_as")
apply(r"Neutral Citation Number:.*", re.I, "neutral_citation")
apply(r"Case No:.*", re.I, "case_no")
apply(r"IN THE COURT OF APPEAL.*?DIVISION", re.I, "court_header")
apply(r"Royal Courts? of Justice.*?(London)?.*?\d{4}", re.I, "royal_courts")
apply(r"Computer Aided Transcription.*?Company", re.I|re.S, "transcription")  # SUSPECT
apply(r"Official Shorthand Writers.*?(Company)?", re.I, "shorthand")
apply(r"HTML VERSION OF JUDGMENT", re.I, "html_version")
apply(r"BAILII: Copyright Policy.*$", re.I|re.S, "bailii_footer")
apply(r"Crown Copyright ©", re.I, "crown_copyright")
t = re.sub(r"[ \t]+", " ", t)
t = re.sub(r"\n{2,}", "\n", t)
t = t.strip()

print("Regex impact (chars removed):")
for name, before, after, removed in steps:
    if removed > 100:
        print(f"  *** {name}: -{removed} chars (BEFORE: {before} -> AFTER: {after})")
    else:
        print(f"  {name}: -{removed} chars")
print(f"\nFinal length: {len(t)}")
print(f"First 500 chars: {repr(t[:500])}")
