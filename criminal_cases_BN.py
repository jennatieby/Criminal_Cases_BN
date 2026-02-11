from pathlib import Path
from time import sleep
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

INDEX_CSV = ROOT / "uk_cases_index.csv"
OUT_CSV = DATA_RAW / "uk_cases_full.csv"
FAILED_CSV = DATA_RAW / "failed_urls.csv"

USER_AGENT = "Mozilla/5.0 (compatible; CriminalCasesBN/1.0; +https://github.com/jennatieby/Criminal_Cases_BN)"
HEADERS = {"User-Agent": USER_AGENT}


def check_robots_txt(sample_url: str) -> None:
    """
    Best-effort robots.txt fetch for transparency.
    Does not enforce policy, but surfaces it in logs.
    """
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
    """
    Fetch a URL with simple retry + exponential backoff.
    Returns the page text on success, or empty string on failure.
    """
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
            # exponential backoff
            delay = base_delay * attempt
            sleep(delay)


def main() -> None:
    # Load your case index
    df = pd.read_csv(INDEX_CSV)

    if df.empty:
        print(f"[error] Case index {INDEX_CSV} is empty.")
        return

    # Best-effort robots.txt check using the first URL
    first_url = df["URL"].iloc[0]
    check_robots_txt(first_url)

    texts = []
    failures = []

    for i, row in df.iterrows():
        url = row["URL"]
        text = fetch_with_retries(url)
        if text:
            print(f"Scraped {i+1}/{len(df)}: {url}")
        else:
            print(f"[error] Giving up on {url} after retries.")
            failures.append({"index": i, "URL": url})
        texts.append(text)
        # polite base delay between URLs (independent of retry backoff)
        sleep(1.5)

    # Add scraped text as a new column
    df["CaseText"] = texts

    # Save to CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"all cases saved to {OUT_CSV}")

    # Log failures (if any) so they can be re-run later
    if failures:
        import csv

        with FAILED_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "URL"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"[warn] {len(failures)} URLs failed; logged to {FAILED_CSV}")
    else:
        print("[info] All URLs scraped successfully; no failures logged.")


if __name__ == "__main__":
    main()
