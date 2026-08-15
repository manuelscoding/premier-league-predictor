"""Consolidate per-player FPL history (element-summary/*.json) into a long
player-season table, joined with current position/team from bootstrap-static.
"""
import json
from pathlib import Path

import pandas as pd

RAW_FPL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "fpl"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
]


def load_players() -> pd.DataFrame:
    bootstrap = json.loads((RAW_FPL_DIR / "bootstrap-static.json").read_text())
    players = pd.DataFrame(bootstrap["elements"])
    positions = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name"]]
    positions.columns = ["element_type", "position"]
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]]
    teams.columns = ["team", "team_name"]

    players = players.merge(positions, on="element_type").merge(teams, on="team")
    players["full_name"] = players["first_name"] + " " + players["second_name"]
    return players[["id", "full_name", "position", "team_name", "now_cost"]]


def load_player_season_history() -> pd.DataFrame:
    summary_dir = RAW_FPL_DIR / "element-summary"
    rows = []
    for path in sorted(summary_dir.glob("*.json")):
        pid = int(path.stem)
        data = json.loads(path.read_text())
        for season in data.get("history_past", []):
            row = {"id": pid, "season_name": season["season_name"]}
            for col in NUMERIC_COLS:
                row[col] = season.get(col)
            rows.append(row)
    hist = pd.DataFrame(rows)
    for col in NUMERIC_COLS:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist


def build_lag_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Predict season N totals from season N-1 totals (+ 2-season averages)."""
    hist = hist.sort_values(["id", "season_name"]).copy()
    targets = ["total_points", "goals_scored", "assists", "minutes",
               "clean_sheets", "saves", "goals_conceded"]

    df = hist.copy()
    for col in NUMERIC_COLS:
        df[f"prev_{col}"] = df.groupby("id")[col].shift(1)
    for col in targets:
        df[f"avg2_{col}"] = (
            df.groupby("id")[col].shift(1).rolling(2, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
    df["seasons_played"] = df.groupby("id").cumcount()
    return df


def main() -> None:
    print("Loading current player/position/team info...")
    players = load_players()
    print(f"  {len(players)} current players")

    print("Loading per-player season history...")
    hist = load_player_season_history()
    print(f"  {len(hist)} player-season rows from {hist['id'].nunique()} players")

    print("Building lag features...")
    df = build_lag_features(hist)
    df = df.merge(players[["id", "full_name", "position", "team_name"]], on="id", how="left")

    df.to_csv(PROCESSED_DIR / "player_season_history.csv", index=False)
    print(f"Saved {len(df)} rows to {PROCESSED_DIR / 'player_season_history.csv'}")

    print("\nPosition breakdown of player-season rows:")
    print(df["position"].value_counts())
    print("\nSeasons-played distribution (rows with at least 1 prior season):")
    print(df["seasons_played"].value_counts().sort_index())


if __name__ == "__main__":
    main()
