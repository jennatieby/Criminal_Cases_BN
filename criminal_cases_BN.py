from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep


ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

INDEX_CSV = ROOT / "uk_cases_index.csv"
OUT_CSV = DATA_RAW / "uk_cases_full.csv"

# Load your case index
df = pd.read_csv(INDEX_CSV)

texts = []
for i, row in df.iterrows():
    url = row["URL"]
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        texts.append(text)
        print(f"Scraped {i+1}/{len(df)}: {url}")
        sleep(1.5)  # polite delay
    except Exception as e:
        print(f"Error on {url}: {e}")
        texts.append("")

# Add scraped text as a new column
df["CaseText"] = texts

# Save to CSV
df.to_csv(OUT_CSV, index=False)
print(f"all cases saved to {OUT_CSV}")
