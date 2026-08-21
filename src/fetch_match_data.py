"""Download historical match data from football-data.co.uk, for every
league this project tracks.

Each season's file has full-time/half-time results, match stats (shots,
corners, cards, etc. from 2000/01 on), and betting odds. Column layout is
identical across leagues, so downstream processing (build_datasets.py)
doesn't need to know which league a file came from beyond its code prefix.
"""
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matches"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# football-data.co.uk's per-league file codes
LEAGUES = {
    "Premier League": "E0",
    "La Liga": "SP1",
}


def season_codes(start_year: int, end_year: int) -> list[str]:
    """e.g. start_year=1993 -> '9394', ..., up to end_year."""
    codes = []
    for y in range(start_year, end_year + 1):
        a = str(y)[-2:]
        b = str(y + 1)[-2:]
        codes.append(f"{a}{b}")
    return codes


def download_season(league_code: str, season_code: str) -> Path | None:
    out_path = RAW_DIR / f"{league_code}_{season_code}.csv"
    if out_path.exists():
        return out_path
    url = BASE_URL.format(season=season_code, code=league_code)
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200 or len(resp.content) < 500:
        print(f"  skip {season_code}: not available ({resp.status_code})")
        return None
    out_path.write_bytes(resp.content)
    return out_path


def fetch_league(league_code: str, start_year: int = 1993, end_year: int = 2026) -> list[Path]:
    downloaded = []
    for code in season_codes(start_year, end_year):
        path = download_season(league_code, code)
        if path:
            downloaded.append(path)
            print(f"  ok: {path.name}")
        time.sleep(0.3)
    return downloaded


def main() -> None:
    for league_name, league_code in LEAGUES.items():
        print(f"Fetching {league_name} ({league_code})...")
        downloaded = fetch_league(league_code)
        print(f"  {len(downloaded)} season files\n")


if __name__ == "__main__":
    main()
