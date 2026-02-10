from pathlib import Path

import pandas as pd, re

ROOT = Path(__file__).resolve().parent
CLEANED_CSV = ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv"

df = pd.read_csv(CLEANED_CSV)

# show before/after for a random case
i = 5
raw = df.loc[i, "CaseText"]
clean = df.loc[i, "CleanText"]

print("Before:", len(raw), "chars")
print("After:", len(clean), "chars")
print("\nFirst 400 chars of clean:")
print(clean[:400])

# look for the start of substantive judgment
pattern = re.compile(r"(THE\s+HONOURABLE|MR\s+JUSTICE|LORD\s+JUSTICE)", re.I)
print("\nContains judge start:", bool(pattern.search(clean)))