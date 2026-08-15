"""End-to-end: fit team strengths from recent PL seasons, load the upcoming
season's full fixture list from the FPL API, and Monte Carlo simulate it to
get title / top-4 / relegation probabilities.

Every function accepts already-fetched `bootstrap`/`fixtures_raw` dicts so a
live caller (e.g. the dashboard) can pass freshly-pulled API data; falling
back to locally cached JSON files when run standalone as a script.
"""
import json
from pathlib import Path

import pandas as pd

from team_strength_model import (
    FPL_TO_FD, fit_team_strengths, load_recent_matches, simulate_season,
)

RAW_FPL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "fpl"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def fixtures_to_df(bootstrap: dict, fixtures_raw: list) -> pd.DataFrame:
    id_to_name = {t["id"]: t["name"] for t in bootstrap["teams"]}
    rows = []
    for f in fixtures_raw:
        home_fpl = id_to_name[f["team_h"]]
        away_fpl = id_to_name[f["team_a"]]
        rows.append({
            "GW": f["event"],
            "HomeTeam": FPL_TO_FD.get(home_fpl, home_fpl),
            "AwayTeam": FPL_TO_FD.get(away_fpl, away_fpl),
            "Finished": f["finished"],
            "HomeGoals": f["team_h_score"],
            "AwayGoals": f["team_a_score"],
        })
    return pd.DataFrame(rows)


def load_upcoming_fixtures(bootstrap: dict | None = None, fixtures_raw: list | None = None) -> pd.DataFrame:
    if bootstrap is None:
        bootstrap = json.loads((RAW_FPL_DIR / "bootstrap-static.json").read_text())
    if fixtures_raw is None:
        fixtures_raw = json.loads((RAW_FPL_DIR / "fixtures.json").read_text())
    return fixtures_to_df(bootstrap, fixtures_raw)


def run_simulation(matches: pd.DataFrame, bootstrap: dict, fixtures_raw: list,
                    n_sims: int = 20000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit team strengths on `matches` and simulate the season implied by
    `fixtures_raw`. Returns (sim_results, team_strengths)."""
    strengths, model = fit_team_strengths(matches)

    fixtures = fixtures_to_df(bootstrap, fixtures_raw)
    remaining = fixtures[~fixtures["Finished"]].copy()
    played = fixtures[fixtures["Finished"]].copy()

    sim_results = simulate_season(remaining, strengths, model, n_sims=n_sims)

    if len(played):
        banked = {}
        for r in played.itertuples():
            banked.setdefault(r.HomeTeam, 0)
            banked.setdefault(r.AwayTeam, 0)
            if r.HomeGoals > r.AwayGoals:
                banked[r.HomeTeam] += 3
            elif r.HomeGoals < r.AwayGoals:
                banked[r.AwayTeam] += 3
            else:
                banked[r.HomeTeam] += 1
                banked[r.AwayTeam] += 1
        sim_results["Banked_Points"] = sim_results["Team"].map(banked).fillna(0)
        sim_results["Expected_Points"] += sim_results["Banked_Points"]

    sim_results = sim_results.sort_values("Title_Prob", ascending=False).reset_index(drop=True)
    return sim_results, strengths


def main(n_sims: int = 20000) -> None:
    print("Fitting team strengths on 2023-24 / 2024-25 / 2025-26 matches...")
    matches = load_recent_matches(["2023-24", "2024-25", "2025-26"])

    print("Loading upcoming season fixtures from FPL API data...")
    bootstrap = json.loads((RAW_FPL_DIR / "bootstrap-static.json").read_text())
    fixtures_raw = json.loads((RAW_FPL_DIR / "fixtures.json").read_text())

    print(f"Simulating {n_sims} seasons (already-played results are locked in)...")
    sim_results, strengths = run_simulation(matches, bootstrap, fixtures_raw, n_sims=n_sims)

    strengths.to_csv(PROCESSED_DIR / "team_strengths.csv", index=False)
    sim_results.to_csv(PROCESSED_DIR / "season_simulation.csv", index=False)
    print("\n" + sim_results.round(3).to_string(index=False))
    print(f"\nSaved to {PROCESSED_DIR / 'season_simulation.csv'}")


if __name__ == "__main__":
    main()
