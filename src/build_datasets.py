"""Consolidate raw football-data.co.uk season CSVs into clean, analysis-ready
datasets: one match-level table across all seasons, and one team-season
standings table (used as classification features for champion prediction).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_MATCHES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matches"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CORE_COLS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
    "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
]


def season_label(code: str) -> str:
    """'2324' -> '2023-24'"""
    a, b = code[:2], code[2:]
    century = "19" if a in ("93", "94", "95", "96", "97", "98", "99") else "20"
    return f"{century}{a}-{b}"


def load_all_matches() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_MATCHES_DIR.glob("E0_*.csv")):
        code = re.search(r"E0_(\d{4})", path.name).group(1)
        df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip", engine="python")
        df = df[[c for c in CORE_COLS if c in df.columns]].copy()
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["Season"] = season_label(code)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(["Season", "Date"]).reset_index(drop=True)
    return matches


def build_standings(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team-season: points, GD, wins/draws/losses, champion flag."""
    rows = []
    for season, sdf in matches.groupby("Season"):
        teams = pd.unique(sdf[["HomeTeam", "AwayTeam"]].values.ravel())
        stats = {t: {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0} for t in teams}
        for _, m in sdf.iterrows():
            h, a, hg, ag = m["HomeTeam"], m["AwayTeam"], m["FTHG"], m["FTAG"]
            stats[h]["GF"] += hg
            stats[h]["GA"] += ag
            stats[a]["GF"] += ag
            stats[a]["GA"] += hg
            if hg > ag:
                stats[h]["W"] += 1
                stats[a]["L"] += 1
            elif hg < ag:
                stats[a]["W"] += 1
                stats[h]["L"] += 1
            else:
                stats[h]["D"] += 1
                stats[a]["D"] += 1
        for team, s in stats.items():
            played = s["W"] + s["D"] + s["L"]
            points = s["W"] * 3 + s["D"]
            rows.append({
                "Season": season, "Team": team, "Played": played,
                "Wins": s["W"], "Draws": s["D"], "Losses": s["L"],
                "GF": s["GF"], "GA": s["GA"], "GD": s["GF"] - s["GA"],
                "Points": points,
            })
    standings = pd.DataFrame(rows)
    standings["Rank"] = standings.groupby("Season")["Points"] \
        .rank(method="first", ascending=False)
    # break ties by GD for rank ordering (approximate; good enough for features)
    standings = standings.sort_values(["Season", "Points", "GD"], ascending=[True, False, False])
    standings["Rank"] = standings.groupby("Season").cumcount() + 1
    standings["Champion"] = (standings["Rank"] == 1).astype(int)
    standings["Top4"] = (standings["Rank"] <= 4).astype(int)

    # prior-season features (form carried into next season)
    standings = standings.sort_values(["Team", "Season"])
    standings["Prev_Points"] = standings.groupby("Team")["Points"].shift(1)
    standings["Prev_Rank"] = standings.groupby("Team")["Rank"].shift(1)
    standings["Prev_GD"] = standings.groupby("Team")["GD"].shift(1)
    standings["Prev_Champion"] = standings.groupby("Team")["Champion"].shift(1)

    return standings.sort_values(["Season", "Rank"]).reset_index(drop=True)


def main() -> None:
    print("Loading and consolidating match files...")
    matches = load_all_matches()
    print(f"  {len(matches)} matches across {matches['Season'].nunique()} seasons")
    matches.to_csv(PROCESSED_DIR / "matches_all_seasons.csv", index=False)

    print("Building team-season standings...")
    standings = build_standings(matches)
    standings.to_csv(PROCESSED_DIR / "standings_all_seasons.csv", index=False)
    print(f"  {len(standings)} team-season rows")

    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
