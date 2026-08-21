"""Consolidate raw football-data.co.uk season CSVs into clean, analysis-ready
datasets: one match-level table across all seasons/leagues, and one
team-season standings table (used as classification features for champion
prediction).
"""
import re
from pathlib import Path

import pandas as pd

from fetch_match_data import LEAGUES

RAW_MATCHES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matches"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CODE_TO_LEAGUE = {code: name for name, code in LEAGUES.items()}

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
    for path in sorted(RAW_MATCHES_DIR.glob("*_*.csv")):
        m = re.match(r"([A-Z0-9]+)_(\d{4})\.csv$", path.name)
        if not m or m.group(1) not in CODE_TO_LEAGUE:
            continue
        league_code, season_code = m.group(1), m.group(2)
        # the default C engine handles on_bad_lines fine and is dramatically
        # faster than engine="python" on these wide, odds-heavy files —
        # that combination was slow enough to look like a hang under load
        df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        df = df[[c for c in CORE_COLS if c in df.columns]].copy()
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df["Season"] = season_label(season_code)
        df["League"] = CODE_TO_LEAGUE[league_code]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(["League", "Season", "Date"]).reset_index(drop=True)
    return matches


def build_standings(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per league-team-season: points, GD, wins/draws/losses, champion flag."""
    rows = []
    for (league, season), sdf in matches.groupby(["League", "Season"]):
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
                "League": league, "Season": season, "Team": team, "Played": played,
                "Wins": s["W"], "Draws": s["D"], "Losses": s["L"],
                "GF": s["GF"], "GA": s["GA"], "GD": s["GF"] - s["GA"],
                "Points": points,
            })
    standings = pd.DataFrame(rows)

    # break ties by GD for rank ordering (approximate; good enough for features)
    standings = standings.sort_values(["League", "Season", "Points", "GD"], ascending=[True, True, False, False])
    standings["Rank"] = standings.groupby(["League", "Season"]).cumcount() + 1
    standings["Champion"] = (standings["Rank"] == 1).astype(int)
    standings["Top4"] = (standings["Rank"] <= 4).astype(int)

    # prior-season features (form carried into next season)
    standings = standings.sort_values(["League", "Team", "Season"])
    grp = standings.groupby(["League", "Team"])
    standings["Prev_Points"] = grp["Points"].shift(1)
    standings["Prev_Rank"] = grp["Rank"].shift(1)
    standings["Prev_GD"] = grp["GD"].shift(1)
    standings["Prev_Champion"] = grp["Champion"].shift(1)

    return standings.sort_values(["League", "Season", "Rank"]).reset_index(drop=True)


def main() -> None:
    print("Loading and consolidating match files...")
    matches = load_all_matches()
    for league, ldf in matches.groupby("League"):
        print(f"  {league}: {len(ldf)} matches across {ldf['Season'].nunique()} seasons")
    matches.to_csv(PROCESSED_DIR / "matches_all_seasons.csv", index=False)

    print("Building team-season standings...")
    standings = build_standings(matches)
    standings.to_csv(PROCESSED_DIR / "standings_all_seasons.csv", index=False)
    print(f"  {len(standings)} team-season rows")

    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
