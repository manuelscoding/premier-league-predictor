"""Pull current Premier League player market values from the
transfermarkt-datasets project (github.com/dcaribou/transfermarkt-datasets)
— a community-maintained, weekly-refreshed CSV dump of Transfermarkt data,
hosted publicly with no auth required. No scraping of Transfermarkt itself:
this reuses someone else's already-solved scraping problem, which is far
more reliable than us hitting Transfermarkt directly (it's considerably
more aggressive about bot detection than FBref).

Runs fine in the main venv (just pandas + requests) — unlike FBref, there's
no Selenium dependency here.
"""
import io
from pathlib import Path

import pandas as pd
import requests

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PLAYERS_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
PREMIER_LEAGUE_CODE = "GB1"  # transfermarkt-datasets' code for the Premier League


def main() -> None:
    print(f"Downloading {PLAYERS_URL} ...")
    # pandas' default urllib fetch gets a 403 from this CDN (looks like a
    # missing/blocked User-Agent); a plain requests.get works fine.
    resp = requests.get(PLAYERS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content), compression="gzip")
    print(f"  {len(df)} total players across all leagues/eras")

    epl = df[df["current_club_domestic_competition_id"] == PREMIER_LEAGUE_CODE].copy()
    latest_season = epl["last_season"].max()
    epl = epl[epl["last_season"] == latest_season]
    print(f"  {len(epl)} current Premier League players (last_season={latest_season})")

    out = epl[["player_id", "name", "current_club_name", "position",
               "sub_position", "market_value_in_eur", "highest_market_value_in_eur"]]
    out_path = PROCESSED_DIR / "player_market_values.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
