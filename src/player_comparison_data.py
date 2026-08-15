"""Consolidate FPL (goals/xG/xA/ICT) and FBref (shots/tackles/interceptions)
into one per-player-season table of per-90 stats, for percentile-based
player-comparison pizza charts.

FPL and FBref use different name conventions for the same player (FPL often
keeps a full legal name, e.g. "Bruno Borges Fernandes", while FBref uses the
commonly-known name, e.g. "Bruno Fernandes"). We match on normalized
(first name token, last name token, season) rather than exact full-name
equality, which handles most of these cases without needing a hand-built
alias table.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CATEGORIES = {
    "goals_p90": "Goals",
    "xg_p90": "Expected Goals (xG)",
    "shots_p90": "Shots",
    "threat_p90": "Threat",
    "creativity_p90": "Creativity",
    "xa_p90": "Expected Assists (xA)",
    "tackles_won_p90": "Tackles Won",
    "interceptions_p90": "Interceptions",
    "influence_p90": "Influence",
}
CATEGORY_GROUP = {
    "goals_p90": "Attacking", "xg_p90": "Attacking", "shots_p90": "Attacking",
    "threat_p90": "Creative", "creativity_p90": "Creative", "xa_p90": "Creative",
    "tackles_won_p90": "Defensive", "interceptions_p90": "Defensive", "influence_p90": "Defensive",
}
MIN_MINUTES = 900  # ~10 full matches, filters out small-sample noise


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return name.lower().replace(".", "").replace("'", "").replace("-", " ").strip()


def name_key(name: str) -> tuple[str, str]:
    tokens = normalize_name(name).split()
    if not tokens:
        return ("", "")
    return (tokens[0], tokens[-1])


def load_fpl(player_hist_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = player_hist_df if player_hist_df is not None else pd.read_csv(PROCESSED_DIR / "player_season_history.csv")
    df = df[df["minutes"] >= MIN_MINUTES].copy()
    df["season_short"] = df["season_name"].str.replace("/", "").str[2:]  # '2024/25' -> '2425'
    df["name_key"] = df["full_name"].map(name_key)
    return df


def load_fbref() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "player_advanced_stats.csv", dtype={"season": str})
    df["name_key"] = df["player"].map(name_key)
    return df


def build_comparison_dataset(player_hist_df: pd.DataFrame | None = None,
                              fbref_df: pd.DataFrame | None = None) -> pd.DataFrame:
    fpl = load_fpl(player_hist_df)
    fbref = fbref_df if fbref_df is not None else load_fbref()

    merged = fpl.merge(
        fbref[["name_key", "season", "shots_p90", "tackles_won_p90", "interceptions_p90"]],
        left_on=["name_key", "season_short"], right_on=["name_key", "season"], how="left",
    )

    p90 = 90 / merged["minutes"].clip(lower=1)
    merged["goals_p90"] = merged["goals_scored"] * p90
    merged["xg_p90"] = merged["expected_goals"] * p90
    merged["threat_p90"] = merged["threat"] * p90
    merged["creativity_p90"] = merged["creativity"] * p90
    merged["xa_p90"] = merged["expected_assists"] * p90
    merged["influence_p90"] = merged["influence"] * p90
    # shots/tackles_won/interceptions already per-90 from FBref (NaN where no match found)

    label_cols = ["id", "full_name", "position", "team_name", "season_name", "minutes"]
    out = merged[label_cols + list(CATEGORIES.keys())].dropna(subset=["full_name"])
    # keep only player-seasons with every category present (mostly seasons
    # outside FBref's coverage window, or players we couldn't name-match) —
    # a partial pizza chart is more confusing than a shorter player list
    return out.dropna(subset=list(CATEGORIES.keys()))


def compute_percentiles(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Adds a `{col}_pct` column (0-100) ranking each player-season against
    others at the same position across all collected seasons."""
    pool = df[df["position"] == position].copy()
    for col in CATEGORIES:
        pool[f"{col}_pct"] = pool[col].rank(pct=True, na_option="keep") * 100
    return pool


def player_label(row: pd.Series) -> str:
    season = row["season_name"]
    return f"{row['full_name']} — {row['team_name']} {season}"


if __name__ == "__main__":
    df = build_comparison_dataset()
    matched = df["shots_p90"].notna().mean()
    print(f"{len(df)} player-seasons (>= {MIN_MINUTES} min)")
    print(f"FBref match rate (shots/tackles/interceptions): {matched:.1%}")
    df.to_csv(PROCESSED_DIR / "player_comparison_dataset.csv", index=False)
    print(f"Saved to {PROCESSED_DIR / 'player_comparison_dataset.csv'}")
