"""Premier League predictor dashboard: title race, predicted table, and
player stat projections by position.

Predictions are computed live on each cache refresh (default: every 6 hours)
from committed historical data + a fresh pull of current FPL fixtures/roster,
so the dashboard tracks new results and gameweeks without a manual rebuild.

Run with: streamlit run dashboard/app.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from champion_classifier import (  # noqa: E402
    engineer_features, predict_champion_probabilities, train_full_classifier,
)
from player_stats_model import (  # noqa: E402
    predict_upcoming_season, refresh_current_meta, train_position_models,
)
from simulate_current_season import run_simulation  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FPL_BASE = "https://fantasy.premierleague.com/api"
REFRESH_TTL_SECONDS = 6 * 3600

# palette pulled from the portfolio site's design tokens
INK = "#0D141D"
INK_2 = "#161F2A"
LINE = "#2B3A4C"
BONE = "#ECE7DB"
BONE_DIM = "#9FA8B3"
MARIGOLD = "#E9A13B"

st.set_page_config(page_title="Premier League Predictor", layout="wide", page_icon="⚽")

st.markdown(f"""
<style>
.stApp {{ background-color: {INK}; color: {BONE}; }}
section[data-testid="stSidebar"] {{ background-color: {INK_2}; }}
h1, h2, h3, h4 {{ color: {BONE}; font-family: 'JetBrains Mono', monospace; }}
[data-testid="stMetricValue"] {{ color: {MARIGOLD}; }}
[data-testid="stMetricLabel"] {{ color: {BONE_DIM}; }}
.stDataFrame {{ border: 1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=REFRESH_TTL_SECONDS, show_spinner="Fetching live Premier League data...")
def fetch_live_fpl():
    bootstrap = requests.get(f"{FPL_BASE}/bootstrap-static/", timeout=20).json()
    fixtures_raw = requests.get(f"{FPL_BASE}/fixtures/", timeout=20).json()
    return bootstrap, fixtures_raw, datetime.now(timezone.utc)


@st.cache_data(show_spinner=False)
def load_historical():
    """Committed base data (refreshed nightly by CI, not on every visit)."""
    all_matches = pd.read_csv(PROCESSED_DIR / "matches_all_seasons.csv", parse_dates=["Date"])
    last3_seasons = sorted(all_matches["Season"].unique())[-3:]
    matches = all_matches[all_matches["Season"].isin(last3_seasons)]
    standings = pd.read_csv(PROCESSED_DIR / "standings_all_seasons.csv")
    player_hist = pd.read_csv(PROCESSED_DIR / "player_season_history.csv")
    return matches, standings, player_hist


@st.cache_data(ttl=REFRESH_TTL_SECONDS, show_spinner="Recomputing predictions from live data...")
def compute_predictions():
    bootstrap, fixtures_raw, fetched_at = fetch_live_fpl()
    matches, standings, player_hist = load_historical()

    sim_results, strengths = run_simulation(matches, bootstrap, fixtures_raw, n_sims=20000)

    champ_df = engineer_features(standings)
    champ_df = champ_df[champ_df["Season"] >= "1995-96"]
    clf_full = train_full_classifier(champ_df)
    champ_clf = predict_champion_probabilities(clf_full, champ_df, bootstrap)

    train_df = player_hist[player_hist["seasons_played"] > 0]
    models = train_position_models(train_df, with_cv=False, verbose=False)
    player_hist_live = refresh_current_meta(player_hist, bootstrap)
    players_pred = predict_upcoming_season(models, player_hist_live)

    return sim_results, strengths, champ_clf, players_pred, fetched_at


sim, strengths, champ_clf, players, fetched_at = compute_predictions()

st.title("Premier League Predictor")
st.caption("Monte Carlo season simulation + historical-trend classifier for the title race, "
           "and per-position regression models for player stat projections.")
st.caption(f"🔄 Live data as of {fetched_at.strftime('%Y-%m-%d %H:%M UTC')} "
           f"— refreshes automatically every {REFRESH_TTL_SECONDS // 3600} hours.")

tab1, tab2, tab3 = st.tabs(["Title Race", "Predicted Table", "Player Projections"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Title probability — Monte Carlo season simulation")
        top = sim.head(10)
        fig = go.Figure(go.Bar(
            x=top["Title_Prob"] * 100, y=top["Team"], orientation="h",
            marker_color=MARIGOLD,
        ))
        fig.update_layout(
            plot_bgcolor=INK, paper_bgcolor=INK, font_color=BONE,
            xaxis_title="Title probability (%)", yaxis=dict(autorange="reversed"),
            height=420, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Historical-trend classifier")
        st.caption("Champion probability from trailing form alone (no simulation).")
        st.dataframe(
            champ_clf.head(8).style.format({"proba_champion": "{:.1%}"}),
            hide_index=True, use_container_width=True,
        )

    favorite = sim.iloc[0]
    st.metric("Simulation favorite", favorite["Team"], f"{favorite['Title_Prob']:.1%} title probability")

with tab2:
    st.subheader("Full predicted table (expected points, rank, top-4 / relegation odds)")
    display_cols = ["Team", "Expected_Points", "Expected_Rank", "Title_Prob", "Top4_Prob", "Relegation_Prob"]
    st.dataframe(
        sim[display_cols].style.format({
            "Expected_Points": "{:.1f}", "Expected_Rank": "{:.1f}",
            "Title_Prob": "{:.1%}", "Top4_Prob": "{:.1%}", "Relegation_Prob": "{:.1%}",
        }).background_gradient(subset=["Expected_Points"], cmap="YlOrBr"),
        hide_index=True, use_container_width=True, height=740,
    )

    with st.expander("Underlying team attack/defense strengths (Poisson model)"):
        st.dataframe(strengths, hide_index=True, use_container_width=True)

with tab3:
    position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])
    pdf = players[players["position"] == position].copy()
    # only keep stat columns that actually apply to this position (e.g. no
    # saves/clean_sheets for a Midfielder, no clean_sheets for a Forward) —
    # the source data has the union of every position's columns, NaN elsewhere
    pred_cols = [c for c in pdf.columns if c.startswith("pred_") and pdf[c].notna().any()]
    sort_col = "pred_total_points" if "pred_total_points" in pred_cols else pred_cols[0]
    pdf = pdf.sort_values(sort_col, ascending=False).head(25)

    st.subheader(f"Top predicted {position}s for next season")
    show_cols = ["full_name", "team_name"] + pred_cols
    rename = {c: c.replace("pred_", "").replace("_", " ").title() for c in pred_cols}
    rename.update({"full_name": "Player", "team_name": "Team"})
    st.dataframe(
        pdf[show_cols].rename(columns=rename),
        hide_index=True, use_container_width=True, height=700,
    )
