# Premier League Predictor

Predicts the Premier League title race, projects individual player stats by
position, and lets you compare players head-to-head on Opta-style percentile
"pizza" charts.

**Live app:** https://premier-league-predictor-6vr4eaepwxqzyzugx7vwtz.streamlit.app/

## Data sources

- **[football-data.co.uk](https://www.football-data.co.uk/englandm.php)** — match results
  for every PL season since 1993-94 (goals, shots, cards, etc.)
- **[Fantasy Premier League API](https://fantasy.premierleague.com/api/bootstrap-static/)** —
  current squads/positions, fixtures, and per-player season history (no key required)
- **[FBref](https://fbref.com/)** (via [`soccerdata`](https://github.com/probberechts/soccerdata)) —
  shot volume and defensive-action counts (tackles won, interceptions) that
  FPL doesn't track, for the player-comparison pizza charts
- **[transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)** —
  current player market values, as a public weekly-refreshed CSV dump rather
  than scraping Transfermarkt directly (see [Market value refresh](#market-value-refresh-every-6-months) below)

## Live architecture

The dashboard computes everything itself on a rolling cache (default: every
6 hours) rather than reading pre-baked prediction files:

- **Fetched live, every cache cycle:** current fixtures/results and squad
  info, pulled directly from the FPL API inside `dashboard/app.py`.
- **Committed to the repo, refreshed periodically:** the historical corpus
  (`data/processed/matches_all_seasons.csv`, `standings_all_seasons.csv`,
  `player_season_history.csv`, `player_advanced_stats.csv`) — this is what
  actually takes real fetch time to build, so it's checked into git and
  rebuilt on a schedule rather than per-request.

A nightly GitHub Action (`.github/workflows/refresh-data.yml`) rebuilds the
football-data.co.uk / FPL half of that corpus and commits the result. The
FBref half (`player_advanced_stats.csv`) is **not** in that job — see
[FBref refresh](#fbref-refresh-manual) below for why.

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
   minutes, goals, assists, and position-specific stats (clean sheets, saves) from
   each player's trailing FPL history. Deliberately excludes fantasy points —
   this is a stats tool, not an FPL-team-picker.
4. **Player comparison pizza charts** (`src/player_comparison_data.py`,
   `src/pizza_chart.py`) — percentile rank (0-100) against same-position peers
   across the last 4 seasons (≥900 minutes played), across 9 per-90 categories
   grouped Attacking / Creative / Defensive:

   | Attacking | Creative | Defensive |
   |---|---|---|
   | Goals | Threat (ICT) | Tackles Won |
   | Expected Goals (xG) | Creativity (ICT) | Interceptions |
   | Shots | Expected Assists (xA) | Influence (ICT) |

   Goals/xG/Threat/Creativity/xA/Influence come from FPL; Shots/Tackles Won/
   Interceptions come from FBref, matched to FPL players by normalized
   (first name, last name, season) since the two sources spell names
   differently (e.g. FPL's "Bruno Borges Fernandes" vs FBref's "Bruno
   Fernandes"). About 85% of in-range player-seasons match cleanly; the rest
   are dropped from the comparison pool rather than shown with holes.

   Each player's current Transfermarkt market value is also shown (Player
   Projections table, and under each chart title in the comparison view) —
   matched by name with the same 3-stage fallback (exact name → unique
   surname → unique first name) as a similar name-mismatch problem, ~80%
   match rate.

## Pipeline

```bash
source venv/bin/activate
python src/fetch_match_data.py        # historical match CSVs -> data/raw/matches
python src/fetch_player_data.py       # FPL players/fixtures -> data/raw/fpl
python src/build_datasets.py          # -> data/processed/{matches,standings}_all_seasons.csv
python src/build_player_dataset.py    # -> data/processed/player_season_history.csv

streamlit run dashboard/app.py        # everything else computes live on load
```

`simulate_current_season.py`, `champion_classifier.py`, and
`player_stats_model.py` can still be run standalone (they'll print a
preview and write their own CSVs), but the dashboard no longer reads those
files — it calls the same functions directly with live data.

### FBref refresh (manual)

`src/fetch_fbref_data.py` uses the `soccerdata` package, which pulls in a
Selenium-based browser to get past FBref's bot protection. That dependency
conflicts with Streamlit's, so it lives in a **separate virtualenv**
(`venv-fbref`, gitignored) and is never installed alongside the dashboard:

```bash
python3 -m venv venv-fbref
source venv-fbref/bin/activate
pip install soccerdata pandas
python src/fetch_fbref_data.py   # -> data/processed/player_advanced_stats.csv
```

This isn't wired into the nightly GitHub Action on purpose: GitHub-hosted
runner IPs get challenged by Cloudflare far more aggressively than a
residential IP, so an automated nightly scrape would be flaky. FBref's
per-90 numbers also don't move much game-to-game, so a manual refresh every
few weeks (or once a new season's worth of data is worth pulling) is
plenty — just commit the regenerated `player_advanced_stats.csv`.

### Market value refresh (every 6 months)

Unlike FBref, `src/fetch_market_values.py` needs no scraping and no special
environment — it's a plain HTTP download of a public CSV
(`transfermarkt-datasets`' `players.csv.gz`), filtered to current Premier
League players. It runs fine in the main `venv` and is wired into its own
scheduled workflow (`.github/workflows/refresh-market-values.yml`, cron on
Jan 1 and Jul 1) rather than the nightly job, since market values move far
slower than match results — roughly the summer/winter transfer-window
cadence.

## Known limitations

- Newly promoted teams with no top-flight history fall back to a
  bottom-table strength estimate — a coarse approximation of real
  promoted-team form.
- Player stat models use lagged season totals, not per-90 rates, so a big
  change in playing time year-over-year is a major source of error (most
  visible in the Forward goals R² during cross-validation).
- The Poisson model doesn't yet use a Dixon-Coles low-score correlation
  adjustment or separate home/away defensive splits — a reasonable next
  iteration.
- FBref no longer publishes "Aerials Won" or a literal "Possession Won"
  stat for outfield players (those columns disappeared from their site at
  some point), and its "possession"/"passing" tables (touches, take-ons,
  key passes) render via client-side JS that a static scrape can't
  reliably capture. The pizza chart substitutes Threat/Creativity/xA/
  Influence from FPL's ICT index instead of inventing numbers — see the
  categories table above.
