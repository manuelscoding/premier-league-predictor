"""Poisson attack/defense team-strength model + Monte Carlo season simulator.

Classic football rating approach (Maher 1982 / Dixon-Coles style):
    home_goals ~ Poisson(exp(home_adv + attack[home] - defense[away]))
    away_goals ~ Poisson(exp(attack[away] - defense[home]))

Fit by weighted Poisson GLM on recent match history (more recent matches
weighted higher via exponential time-decay), then used to simulate an
entire season many times to get title / top-4 / relegation probabilities.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# football-data.co.uk name -> FPL API name (only where they differ)
FD_TO_FPL = {
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
}
FPL_TO_FD = {v: k for k, v in FD_TO_FPL.items()}


def load_recent_matches(seasons: list[str], league: str = "Premier League") -> pd.DataFrame:
    matches = pd.read_csv(PROCESSED_DIR / "matches_all_seasons.csv", parse_dates=["Date"])
    return matches[(matches["Season"].isin(seasons)) & (matches["League"] == league)].copy()


def load_league_matches(league: str) -> pd.DataFrame:
    matches = pd.read_csv(PROCESSED_DIR / "matches_all_seasons.csv", parse_dates=["Date"])
    return matches[matches["League"] == league].copy()


def generate_current_season_fixtures(league_matches: pd.DataFrame, min_teams: int = 18) -> pd.DataFrame:
    """Builds a full double round-robin fixture list (every team home+away
    against every other) for leagues with no live fixtures API — used for
    La Liga instead of the FPL-fixtures path used for the Premier League.

    Team roster comes from the most recent season with enough matches
    played to have seen (most of) the full league; if the latest season is
    too early in its life (or hasn't started) to reveal all ~20 teams, we
    fall back to the prior season's roster — a preseason simulation using
    last year's line-up, same idea as the promoted-team strength fallback.
    Already-played fixtures in the latest season are marked Finished with
    their actual score; everything else is left for the Monte Carlo sim.
    """
    seasons = sorted(league_matches["Season"].unique())
    latest_season = seasons[-1]
    latest = league_matches[league_matches["Season"] == latest_season]

    teams = sorted(set(latest["HomeTeam"]) | set(latest["AwayTeam"]))
    roster_season = latest_season
    if len(teams) < min_teams and len(seasons) > 1:
        roster_season = seasons[-2]
        teams = sorted(set(league_matches.loc[league_matches["Season"] == roster_season, "HomeTeam"])
                        | set(league_matches.loc[league_matches["Season"] == roster_season, "AwayTeam"]))

    pairs = [(h, a) for h in teams for a in teams if h != a]
    fixtures = pd.DataFrame(pairs, columns=["HomeTeam", "AwayTeam"])

    played = latest[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].rename(
        columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"}
    )
    played["Finished"] = True
    fixtures = fixtures.merge(played, on=["HomeTeam", "AwayTeam"], how="left")
    # merge leaves this as object dtype (True/False/NaN mixed), where `~`
    # does a Python bitwise-invert per element (~True == -2) instead of
    # boolean negation — cast back to real bool after filling the NaNs
    fixtures["Finished"] = fixtures["Finished"].fillna(False).astype(bool)
    fixtures.attrs["season"] = latest_season
    fixtures.attrs["roster_season"] = roster_season
    return fixtures


def _decay_weights(dates: pd.Series, half_life_days: float = 380.0) -> np.ndarray:
    """Exponential recency weighting: matches ~1 season old get half weight."""
    age_days = (dates.max() - dates).dt.days.clip(lower=0)
    return 0.5 ** (age_days / half_life_days)


def fit_team_strengths(matches: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Returns (team_strength_df, fitted GLM model)."""
    home = matches.rename(columns={"HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals"})
    home["is_home"] = 1
    away = matches.rename(columns={"AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals"})
    away["is_home"] = 0
    long_df = pd.concat([
        home[["team", "opponent", "goals", "is_home", "Date"]],
        away[["team", "opponent", "goals", "is_home", "Date"]],
    ], ignore_index=True)
    long_df["weight"] = _decay_weights(long_df["Date"])

    model = smf.glm(
        formula="goals ~ is_home + C(team) + C(opponent)",
        data=long_df,
        family=sm.families.Poisson(),
        var_weights=long_df["weight"],
    ).fit()

    teams = sorted(pd.unique(matches[["HomeTeam", "AwayTeam"]].values.ravel()))
    attack, defense = {}, {}
    base_team = teams[0]  # reference level absorbed into intercept
    for t in teams:
        key_att = f"C(team)[T.{t}]"
        key_def = f"C(opponent)[T.{t}]"
        attack[t] = model.params.get(key_att, 0.0)
        # opponent coefficient is on the *conceding* side; more positive = worse defense
        defense[t] = model.params.get(key_def, 0.0)

    strengths = pd.DataFrame({
        "Team": teams,
        "Attack": [attack[t] for t in teams],
        "Defense": [defense[t] for t in teams],
    })
    strengths["Attack_Rank"] = strengths["Attack"].rank(ascending=False)
    strengths["Defense_Rank"] = strengths["Defense"].rank(ascending=True)
    return strengths.sort_values("Attack", ascending=False).reset_index(drop=True), model


def estimate_promoted_team_strength(strengths: pd.DataFrame) -> dict:
    """Newly promoted teams have no top-flight history; approximate with the
    average attack/defense of last season's bottom-3 (typical relegation form)."""
    bottom3 = strengths.nsmallest(3, "Attack")
    return {
        "Attack": bottom3["Attack"].mean() - 0.15,  # slightly below relegated teams
        "Defense": bottom3["Defense"].mean() + 0.15,
    }


def predict_score_rates(home: str, away: str, strengths: pd.DataFrame, model,
                         home_adv: float, intercept: float, fallback: dict) -> tuple[float, float]:
    s = strengths.set_index("Team")
    def get(team, col):
        if team in s.index:
            return s.loc[team, col]
        return fallback[col]

    # Defense coefficient is the raw C(opponent) effect: higher = leakier
    # defense (opponent concedes more), so it's ADDED, not subtracted.
    lam_home = np.exp(intercept + home_adv + get(home, "Attack") + get(away, "Defense"))
    lam_away = np.exp(intercept + get(away, "Attack") + get(home, "Defense"))
    return lam_home, lam_away


def simulate_season(fixtures: pd.DataFrame, strengths: pd.DataFrame, model,
                     n_sims: int = 10000, seed: int = 42) -> pd.DataFrame:
    """fixtures: DataFrame with columns HomeTeam, AwayTeam (FPL-mapped to FD names).
    Returns per-team simulation summary: title/top4/relegation probabilities,
    expected points, expected rank.
    """
    rng = np.random.default_rng(seed)
    intercept = model.params["Intercept"]
    home_adv = model.params["is_home"]
    fallback = estimate_promoted_team_strength(strengths)

    teams = sorted(pd.unique(fixtures[["HomeTeam", "AwayTeam"]].values.ravel()))
    lam_home = np.zeros(len(fixtures))
    lam_away = np.zeros(len(fixtures))
    for i, row in enumerate(fixtures.itertuples()):
        lh, la = predict_score_rates(row.HomeTeam, row.AwayTeam, strengths, model,
                                      home_adv, intercept, fallback)
        lam_home[i], lam_away[i] = lh, la

    n_matches = len(fixtures)
    home_goals = rng.poisson(lam_home, size=(n_sims, n_matches))
    away_goals = rng.poisson(lam_away, size=(n_sims, n_matches))

    team_idx = {t: i for i, t in enumerate(teams)}
    home_ids = fixtures["HomeTeam"].map(team_idx).to_numpy()
    away_ids = fixtures["AwayTeam"].map(team_idx).to_numpy()

    points = np.zeros((n_sims, len(teams)))
    gd = np.zeros((n_sims, len(teams)))

    home_win = home_goals > away_goals
    away_win = away_goals > home_goals
    draw = home_goals == away_goals

    for m in range(n_matches):
        h, a = home_ids[m], away_ids[m]
        points[:, h] += np.where(home_win[:, m], 3, np.where(draw[:, m], 1, 0))
        points[:, a] += np.where(away_win[:, m], 3, np.where(draw[:, m], 1, 0))
        gd[:, h] += home_goals[:, m] - away_goals[:, m]
        gd[:, a] += away_goals[:, m] - home_goals[:, m]

    # rank each simulation by points then GD (tie-break) with tiny random jitter
    # to break residual ties, matching real league tie-break unpredictability
    jitter = rng.uniform(0, 1e-6, size=points.shape)
    sort_key = points * 1e6 + gd * 1e-2 + jitter
    ranks = (-sort_key).argsort(axis=1).argsort(axis=1) + 1

    results = pd.DataFrame({
        "Team": teams,
        "Title_Prob": (ranks == 1).mean(axis=0),
        "Top4_Prob": (ranks <= 4).mean(axis=0),
        "Relegation_Prob": (ranks >= len(teams) - 2).mean(axis=0),
        "Expected_Points": points.mean(axis=0),
        "Expected_Rank": ranks.mean(axis=0),
    })
    return results.sort_values("Title_Prob", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    matches = load_recent_matches(["2023-24", "2024-25", "2025-26"])
    strengths, model = fit_team_strengths(matches)
    print(strengths.to_string(index=False))
    strengths.to_csv(PROCESSED_DIR / "team_strengths.csv", index=False)
