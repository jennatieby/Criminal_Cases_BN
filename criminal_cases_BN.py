import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep

# Load your case index
df = pd.read_csv("uk_cases_index.csv")

texts = []
for i, row in df.iterrows():
    url = row['URL']
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
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
df.to_csv("uk_cases_full.csv", index=False)
print("all cases saved to uk_cases_full.csv")