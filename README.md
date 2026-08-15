# Premier League Predictor

Predicts the Premier League title race and projects individual player stats
by position, using two complementary approaches for the league winner and a
per-position regression suite for players.

## Data sources

- **[football-data.co.uk](https://www.football-data.co.uk/englandm.php)** — match results
  for every PL season since 1993-94 (goals, shots, cards, etc.)
- **[Fantasy Premier League API](https://fantasy.premierleague.com/api/bootstrap-static/)** —
  current squads/positions, fixtures, and per-player season history (no key required)

## Models

1. **Poisson team-strength + Monte Carlo simulation** (`src/team_strength_model.py`,
   `src/simulate_current_season.py`) — fits attack/defense ratings per team via a
   weighted Poisson GLM on the last 3 seasons (recency-weighted), then simulates the
   full season 20,000× to get title / top-4 / relegation probabilities.
2. **Historical-trend classifier** (`src/champion_classifier.py`) — gradient-boosted
   classifier predicting champion probability from trailing form (prior points, rank,
   goal difference, multi-season averages) — a purely history-based second opinion,
   independent of the current fixture list.
3. **Player stat projections** (`src/player_stats_model.py`) — random-forest
   regressors trained separately per position (GK/DEF/MID/FWD) predicting next-season
   points, minutes, goals, assists, and position-specific stats (clean sheets, saves)
   from each player's trailing FPL history.

## Pipeline

```bash
source venv/bin/activate
python src/fetch_match_data.py        # historical match CSVs -> data/raw/matches
python src/fetch_player_data.py       # FPL players/fixtures -> data/raw/fpl
python src/build_datasets.py          # -> data/processed/{matches,standings}_all_seasons.csv
python src/simulate_current_season.py # -> data/processed/season_simulation.csv
python src/champion_classifier.py     # -> data/processed/champion_classifier_predictions.csv
python src/build_player_dataset.py    # -> data/processed/player_season_history.csv
python src/player_stats_model.py      # -> data/processed/player_predictions_next_season.csv

streamlit run dashboard/app.py
```

## Known limitations

- Newly promoted teams with no top-flight history (e.g. a team promoted straight
  into this dataset's gap) fall back to a bottom-table strength estimate — a coarse
  approximation of real promoted-team form.
- Player models use lagged season totals, not per-90 rates, so a big change in
  playing time year-over-year is a major source of error (most visible in the
  Forward goals R² during cross-validation).
- The Poisson model doesn't yet use a Dixon-Coles low-score correlation adjustment
  or separate home/away defensive splits — a reasonable next iteration.
