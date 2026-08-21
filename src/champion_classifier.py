"""Classifier that predicts league-champion probability from each team's
trailing form (features known BEFORE a season starts: prior points, rank,
goal difference, rolling multi-season averages). Complements the Poisson
season simulator with a pure historical-trend view.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from team_strength_model import FPL_TO_FD

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_FPL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "fpl"

FEATURES = [
    "Prev_Points", "Prev_Rank", "Prev_GD", "Prev_Champion", "Prev_Top4",
    "Avg_Points_L3", "Points_Trend", "Seasons_In_Data", "Is_New_To_Data",
]


def drop_incomplete_latest_season(standings: pd.DataFrame, min_played: int = 10) -> pd.DataFrame:
    """Excludes the most recent season if it's barely underway — otherwise a
    team that's played just 1-2 games this season would have its trailing-
    form features built from that tiny partial sample instead of last
    season's real final total, corrupting exactly the teams that happen to
    have already played while leaving everyone else on last season's numbers.
    Only matters once a league actually has some current-season results on
    file (this was a non-issue for the Premier League before now, since
    football-data.co.uk had no 2026-27 file yet; La Liga's does)."""
    if standings.empty:
        return standings
    latest_season = standings["Season"].max()
    latest_played = standings.loc[standings["Season"] == latest_season, "Played"].max()
    if latest_played < min_played:
        return standings[standings["Season"] != latest_season]
    return standings


def engineer_features(standings: pd.DataFrame) -> pd.DataFrame:
    df = standings.sort_values(["Team", "Season"]).copy()
    df["Prev_Top4"] = df.groupby("Team")["Top4"].shift(1)
    df["Avg_Points_L3"] = (
        df.groupby("Team")["Points"].shift(1).rolling(3, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df["Points_Trend"] = df.groupby("Team")["Points"].shift(1) - df.groupby("Team")["Points"].shift(2)
    df["Seasons_In_Data"] = df.groupby("Team").cumcount()
    df["Is_New_To_Data"] = (df["Seasons_In_Data"] == 0).astype(int)

    # impute "no history yet" rows (newly promoted / first season in dataset)
    # with mid-table-ish defaults rather than dropping them
    fill_defaults = {
        "Prev_Points": 45, "Prev_Rank": 17, "Prev_GD": -10,
        "Prev_Champion": 0, "Prev_Top4": 0, "Avg_Points_L3": 45, "Points_Trend": 0,
    }
    for col, val in fill_defaults.items():
        df[col] = df[col].fillna(val)

    return df


def train(df: pd.DataFrame, test_seasons: list[str]):
    train_df = df[~df["Season"].isin(test_seasons)]
    test_df = df[df["Season"].isin(test_seasons)]

    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42,
    )
    clf.fit(train_df[FEATURES], train_df["Champion"])
    return clf, train_df, test_df


def evaluate_top1_accuracy(clf, df: pd.DataFrame) -> float:
    """Per season: does the model's single highest-probability pick match the
    actual champion? A stricter, more interpretable metric than AUC alone."""
    hits = 0
    seasons = df["Season"].unique()
    for season in seasons:
        sdf = df[df["Season"] == season].copy()
        sdf["pred_proba"] = clf.predict_proba(sdf[FEATURES])[:, 1]
        picked = sdf.loc[sdf["pred_proba"].idxmax(), "Team"]
        actual = sdf.loc[sdf["Champion"] == 1, "Team"]
        if len(actual) and picked == actual.iloc[0]:
            hits += 1
    return hits / len(seasons)


def main(league: str = "Premier League") -> None:
    standings = pd.read_csv(PROCESSED_DIR / "standings_all_seasons.csv")
    standings = standings[standings["League"] == league]
    df = engineer_features(drop_incomplete_latest_season(standings))
    df = df[df["Season"] >= "1995-96"]  # drop first couple seasons lost to shift()

    all_seasons = sorted(df["Season"].unique())
    test_seasons = all_seasons[-8:]  # last 8 seasons held out
    clf, train_df, test_df = train(df, test_seasons)

    auc = roc_auc_score(test_df["Champion"], clf.predict_proba(test_df[FEATURES])[:, 1])
    top1_test = evaluate_top1_accuracy(clf, test_df)
    top1_all = evaluate_top1_accuracy(clf, df)
    print(f"Held-out seasons: {test_seasons[0]}..{test_seasons[-1]}")
    print(f"  ROC-AUC (champion vs rest): {auc:.3f}")
    print(f"  Top-1 pick accuracy on held-out seasons: {top1_test:.1%}")
    print(f"  Top-1 pick accuracy across all seasons (in-sample + held-out): {top1_all:.1%}")

    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.round(3).to_string())

    # refit on ALL historical data, predict the upcoming season
    clf_full = train_full_classifier(df)

    bootstrap = json.loads((RAW_FPL_DIR / "bootstrap-static.json").read_text())
    current_roster = load_current_roster(bootstrap)
    preview = predict_champion_probabilities(clf_full, df, current_roster)
    print("\nTop 10 predicted title contenders for next season (renormalized):")
    print(preview.round(3).to_string(index=False))
    preview.to_csv(PROCESSED_DIR / "champion_classifier_predictions.csv", index=False)


def load_current_roster(bootstrap: dict | None = None) -> list[str]:
    if bootstrap is None:
        bootstrap = json.loads((RAW_FPL_DIR / "bootstrap-static.json").read_text())
    fpl_names = [t["name"] for t in bootstrap["teams"]]
    return [FPL_TO_FD.get(n, n) for n in fpl_names]


def predict_champion_probabilities(clf, df: pd.DataFrame, current_roster: list[str]) -> pd.DataFrame:
    """Top-10 renormalized champion probabilities for the current roster."""
    next_rows = build_next_season_features(df, current_roster)
    next_rows["proba_champion"] = clf.predict_proba(next_rows[FEATURES])[:, 1]
    preview = next_rows[["Team", "proba_champion"]].sort_values(
        "proba_champion", ascending=False
    ).head(10).copy()
    preview["proba_champion"] = preview["proba_champion"] / preview["proba_champion"].sum()
    return preview


def train_full_classifier(df: pd.DataFrame) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42,
    ).fit(df[FEATURES], df["Champion"])


def build_next_season_features(df: pd.DataFrame, current_roster: list[str]) -> pd.DataFrame:
    """One row per current-roster team, features derived from each team's
    most recent seasons in the historical dataset. Teams with no history
    (freshly promoted, never seen before) fall back to bottom-table defaults."""
    df_sorted = df.sort_values("Season")
    points_by_team = df_sorted.groupby("Team")["Points"]
    latest = df_sorted.groupby("Team").tail(1).set_index("Team")

    fallback = {
        "Prev_Points": 38, "Prev_Rank": 18, "Prev_GD": -25,
        "Prev_Champion": 0, "Prev_Top4": 0, "Avg_Points_L3": 38,
        "Points_Trend": 0, "Seasons_In_Data": 0, "Is_New_To_Data": 1,
    }

    rows = []
    for team in current_roster:
        if team in latest.index:
            l = latest.loc[team]
            pts_series = points_by_team.get_group(team)
            trend = (pts_series.iloc[-1] - pts_series.iloc[-2]) if len(pts_series) > 1 else 0
            rows.append({
                "Team": team,
                "Prev_Points": l["Points"], "Prev_Rank": l["Rank"], "Prev_GD": l["GD"],
                "Prev_Champion": l["Champion"], "Prev_Top4": l["Top4"],
                "Avg_Points_L3": pts_series.tail(3).mean(), "Points_Trend": trend,
                "Seasons_In_Data": l["Seasons_In_Data"] + 1, "Is_New_To_Data": 0,
            })
        else:
            rows.append({"Team": team, **fallback})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
