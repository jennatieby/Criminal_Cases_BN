#!/usr/bin/env python3
"""
scrape_negative_cases.py

Scrape negative-instance case pages (e.g. CCRC decisions — overturned/quashed).
Reads URLs from data/raw/negative_cases_raw.csv, saves full text to
data/raw/negative_cases_full.csv. Logs failed URLs to data/raw/negative_cases_failed_urls.csv.

Run from repo root:
  python scrape_negative_cases.py
"""

from pathlib import Path
import csv
import time
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

IN_CSV = DATA_RAW / "negative_cases_raw.csv"
OUT_CSV = DATA_RAW / "negative_cases_full.csv"
FAILED_CSV = DATA_RAW / "negative_cases_failed_urls.csv"

USER_AGENT = "Mozilla/5.0 (compatible; CriminalCasesBN/1.0; +https://github.com/jennatieby/Criminal_Cases_BN)"
HEADERS = {"User-Agent": USER_AGENT}


def check_robots_txt(sample_url: str) -> None:
    try:
        parsed = urlparse(sample_url)
        if not parsed.scheme or not parsed.netloc:
            return
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        r = requests.get(robots_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            print(f"[info] Fetched robots.txt from {robots_url} (len={len(r.text)})")
        else:
            print(f"[info] robots.txt not available or non-200 from {robots_url} (status={r.status_code})")
    except Exception as e:
        print(f"[warn] Could not fetch robots.txt: {e}")


def fetch_with_retries(url: str, max_retries: int = 3, base_delay: float = 1.5) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            print(f"[warn] Attempt {attempt}/{max_retries} failed for {url}: {e}")
            if attempt == max_retries:
                return ""
            time.sleep(base_delay * attempt)
    return ""


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input file: {IN_CSV}")

    # Support Excel-exported CSV (BOM, URL column name)
    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    url_col = "URL"
    if url_col not in df.columns:
        # Try first column
        url_col = df.columns[0]
    df = df.rename(columns={url_col: "URL"})
    df = df.dropna(subset=["URL"])
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"].str.startswith("http")]

    if df.empty:
        raise SystemExit(f"No valid URLs found in {IN_CSV}")

    print(f"[start] Loaded {len(df)} URLs from {IN_CSV}")
    first_url = df["URL"].iloc[0]
    check_robots_txt(first_url)

    texts = []
    failures = []

    for i, row in df.iterrows():
        url = row["URL"]
        text = fetch_with_retries(url)
        if text:
            print(f"Scraped {i+1}/{len(df)}: {url[:60]}...")
        else:
            print(f"[error] Giving up on {url} after retries.")
            failures.append({"index": i, "URL": url})
        texts.append(text)
        time.sleep(1.5)

    df["CaseText"] = texts
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")

    if failures:
        with FAILED_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "URL"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"[warn] {len(failures)} URLs failed; logged to {FAILED_CSV}")
    else:
        print("[info] All URLs scraped successfully.")


if __name__ == "__main__":
    main()
