"""Per-position regression models predicting next-season player stats from
trailing FPL history (goals, assists, minutes, points, etc.).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_FPL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "fpl"

POSITION_TARGETS = {
    "Goalkeeper": ["minutes", "clean_sheets", "saves", "goals_conceded"],
    "Defender": ["minutes", "clean_sheets", "goals_scored", "assists"],
    "Midfielder": ["minutes", "goals_scored", "assists"],
    "Forward": ["minutes", "goals_scored", "assists"],
}

FEATURE_COLS = (
    [f"prev_{c}" for c in [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "saves", "bonus", "bps", "ict_index",
        "expected_goals", "expected_assists",
    ]]
    + [f"avg2_{c}" for c in [
        "total_points", "goals_scored", "assists", "minutes", "clean_sheets",
        "saves", "goals_conceded",
    ]]
    + ["seasons_played"]
)


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "player_season_history.csv")
    return df[df["seasons_played"] > 0].copy()  # need at least 1 prior season


def train_position_models(df: pd.DataFrame, with_cv: bool = True, verbose: bool = True) -> dict:
    """with_cv=False skips the 5-fold R2 diagnostic (the slow part) for
    faster live/interactive training where only the fitted models matter."""
    models = {}
    for position, targets in POSITION_TARGETS.items():
        pdf = df[df["position"] == position].copy()
        pdf[FEATURE_COLS] = pdf[FEATURE_COLS].fillna(0)
        models[position] = {}
        if verbose:
            print(f"\n{position} (n={len(pdf)}):")
        for target in targets:
            valid = pdf.dropna(subset=[target])
            if len(valid) < 20:
                continue
            X, y = valid[FEATURE_COLS], valid[target]
            model = RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42,
            )
            if with_cv:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                if verbose:
                    print(f"  {target:18s} R2 = {scores.mean():.3f} (+/- {scores.std():.3f})")
            model.fit(X, y)
            models[position][target] = model
    return models


def refresh_current_meta(df: pd.DataFrame, bootstrap: dict) -> pd.DataFrame:
    """Overwrite each player's position/team_name/full_name with the live
    bootstrap snapshot, so a mid-season transfer shows up without waiting for
    the next scheduled historical rebuild."""
    positions = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name"]]
    positions.columns = ["element_type", "position"]
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]]
    teams.columns = ["team", "team_name"]
    current = pd.DataFrame(bootstrap["elements"])
    current = current.merge(positions, on="element_type").merge(teams, on="team")
    current["full_name"] = current["first_name"] + " " + current["second_name"]
    current = current[["id", "full_name", "position", "team_name"]]

    df = df.drop(columns=["full_name", "position", "team_name"]).merge(current, on="id", how="inner")
    return df


def predict_upcoming_season(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Use each current player's most recent season as 'prev' features to
    predict their upcoming-season totals."""
    latest = df.sort_values("season_name").groupby("id").tail(1).copy()

    # blend this season's actual value with its own avg2 (already an avg of
    # the *prior* two seasons) to approximate a rolling avg2 for next season,
    # computed before we overwrite/drop the old lag columns below
    raw_targets = ["total_points", "goals_scored", "assists", "minutes",
                   "clean_sheets", "saves", "goals_conceded"]
    new_avg2 = {
        f"avg2_{tgt}": latest[[tgt, f"avg2_{tgt}"]].mean(axis=1, skipna=True)
        for tgt in raw_targets
    }

    # this season's actual stats become next season's lag features; drop the
    # existing prev_*/avg2_* columns first so the rename below can't collide
    old_lag_cols = [c for c in latest.columns if c.startswith("prev_") or c.startswith("avg2_")]
    rename_map = {c: f"prev_{c}" for c in [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "saves", "bonus", "bps", "ict_index",
        "expected_goals", "expected_assists",
    ]}
    feat = latest.drop(columns=old_lag_cols).rename(columns=rename_map)
    for col, series in new_avg2.items():
        feat[col] = series
    feat["seasons_played"] = feat["seasons_played"] + 1
    feat[FEATURE_COLS] = feat[FEATURE_COLS].fillna(0)

    results = []
    for position, targets in POSITION_TARGETS.items():
        pdf = feat[feat["position"] == position].copy()
        if pdf.empty:
            continue
        out = pdf[["id", "full_name", "position", "team_name"]].copy()
        for target in targets:
            if target in models.get(position, {}):
                out[f"pred_{target}"] = models[position][target].predict(pdf[FEATURE_COLS]).round(1)
        results.append(out)
    return pd.concat(results, ignore_index=True)


def main() -> None:
    df = load_training_data()
    print(f"Training rows (>=1 prior season): {len(df)}")
    models = train_position_models(df)

    full_df = pd.read_csv(PROCESSED_DIR / "player_season_history.csv")
    predictions = predict_upcoming_season(models, full_df)
    predictions.to_csv(PROCESSED_DIR / "player_predictions_next_season.csv", index=False)

    print(f"\nSaved predictions for {len(predictions)} players.")
    sort_priority = ["pred_goals_scored", "pred_clean_sheets", "pred_assists", "pred_saves"]
    for position in POSITION_TARGETS:
        top = predictions[predictions["position"] == position]
        pred_cols = [c for c in top.columns if c.startswith("pred_") and top[c].notna().any()]
        sort_col = next((c for c in sort_priority if c in pred_cols), pred_cols[0] if pred_cols else None)
        if sort_col:
            top = top.sort_values(sort_col, ascending=False).head(5)
            print(f"\nTop 5 predicted {position}s by {sort_col.replace('pred_', '')}:")
            print(top.drop(columns=["id"]).to_string(index=False))


if __name__ == "__main__":
    main()
