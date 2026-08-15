"""Download historical Premier League match data from football-data.co.uk.

Each season's file (E0.csv) has full-time/half-time results, match stats
(shots, corners, cards, etc. from 2000/01 on), and betting odds.
"""
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matches"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"


def season_codes(start_year: int, end_year: int) -> list[str]:
    """e.g. start_year=1993 -> '9394', ..., up to end_year."""
    codes = []
    for y in range(start_year, end_year + 1):
        a = str(y)[-2:]
        b = str(y + 1)[-2:]
        codes.append(f"{a}{b}")
    return codes


def download_season(code: str) -> Path | None:
    out_path = RAW_DIR / f"E0_{code}.csv"
    if out_path.exists():
        return out_path
    url = BASE_URL.format(season=code)
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200 or len(resp.content) < 500:
        print(f"  skip {code}: not available ({resp.status_code})")
        return None
    out_path.write_bytes(resp.content)
    return out_path


def main(start_year: int = 1993, end_year: int = 2025) -> None:
    downloaded = []
    for code in season_codes(start_year, end_year):
        path = download_season(code)
        if path:
            downloaded.append(path)
            print(f"  ok: {path.name}")
        time.sleep(0.3)
    print(f"\nDownloaded {len(downloaded)} season files to {RAW_DIR}")


if __name__ == "__main__":
    main()
