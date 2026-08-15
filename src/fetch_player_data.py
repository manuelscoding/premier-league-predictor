"""Download player, team, and fixture data from the Fantasy Premier League API.

No API key required. Endpoints:
  - bootstrap-static: players, teams, positions, gameweeks (current season snapshot)
  - element-summary/{id}: per-player gameweek history + past-season history
  - fixtures: full fixture list with results
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "fpl"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://fantasy.premierleague.com/api"


def fetch_bootstrap() -> dict:
    resp = requests.get(f"{BASE}/bootstrap-static/", timeout=20)
    resp.raise_for_status()
    data = resp.json()
    (RAW_DIR / "bootstrap-static.json").write_text(json.dumps(data))
    return data


def fetch_fixtures() -> list:
    resp = requests.get(f"{BASE}/fixtures/", timeout=20)
    resp.raise_for_status()
    data = resp.json()
    (RAW_DIR / "fixtures.json").write_text(json.dumps(data))
    return data


def fetch_player_summaries(player_ids: list[int]) -> None:
    out_dir = RAW_DIR / "element-summary"
    out_dir.mkdir(exist_ok=True)
    for i, pid in enumerate(player_ids):
        out_path = out_dir / f"{pid}.json"
        if out_path.exists():
            continue
        resp = requests.get(f"{BASE}/element-summary/{pid}/", timeout=20)
        if resp.status_code == 200:
            out_path.write_text(resp.text)
        if i % 50 == 0:
            print(f"  {i}/{len(player_ids)} player summaries fetched")
        time.sleep(0.05)


def main() -> None:
    print("Fetching bootstrap-static (players, teams, positions)...")
    bootstrap = fetch_bootstrap()
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    positions = pd.DataFrame(bootstrap["element_types"])
    print(f"  {len(players)} players, {len(teams)} teams, {len(positions)} positions")

    print("Fetching fixtures...")
    fixtures = fetch_fixtures()
    print(f"  {len(fixtures)} fixtures")

    print("Fetching per-player gameweek history (this may take a couple minutes)...")
    fetch_player_summaries(players["id"].tolist())
    print("Done.")


if __name__ == "__main__":
    main()
