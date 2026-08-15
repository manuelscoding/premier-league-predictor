"""Scrape shot volume and defensive-action counts from FBref for recent
Premier League seasons, to complement FPL's ICT/xG/xA data in the
player-comparison pizza charts.

Requires the `soccerdata` package, which pulls in a Selenium-based browser
(needed to get past FBref's bot protection) and conflicts with Streamlit's
dependencies. Run this with the separate `venv-fbref` environment, never the
main `venv` used by the dashboard:

    source venv-fbref/bin/activate
    python src/fetch_fbref_data.py

Scope note: FBref's "possession" and "passing" tables (touches, take-ons,
carries, key passes) render those cells via client-side JS after page load
(empty `<td class="iz" data-stat="touches"/>` in the raw DOM) — a static/
headless fetch can't reliably capture them without a much heavier
scroll-and-wait Selenium routine, so this script only pulls the tables that
come back fully server-rendered: shooting (Sh) and defense (TklW, Int).
Goals, xG, xA, and creativity/threat come from the FPL data we already
collect in build_player_dataset.py, so this is additive, not a replacement.
"""
from pathlib import Path

import pandas as pd
import soccerdata as sd
from lxml import etree, html
from soccerdata.fbref import FBREF_API, _parse_table

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

LEAGUE = "ENG-Premier League"
# soccerdata season codes, e.g. '2425' = 2024-25.
SEASONS = ["2223", "2324", "2425", "2526"]

STANDARD_SPECS = {"player": "player", "team": "team", "pos": "pos", "90s": ("Playing Time", "90s")}
SHOOTING_SPECS = {"player": "player", "team": "team", "shots": ("Standard", "Sh")}
DEFENSE_SPECS = {
    "player": "Player", "team": "Squad",
    "tackles_won": ("Tackles", "TklW"), "interceptions": "Int",
}


def fetch_custom_stat_type(fbref: sd.FBref, lkey: str, skey: str, season_row, page: str) -> pd.DataFrame:
    filepath = fbref.data_dir / f"players_{lkey}_{skey}_{page}.html"
    url = (
        FBREF_API
        + "/".join(season_row.url.split("/")[:-1])
        + f"/{page}/"
        + season_row.url.split("/")[-1]
    )
    reader = fbref.get(url, filepath)
    tree = html.parse(reader)
    for elem in tree.xpath("//td[@data-stat='comp_level']//span"):
        elem.getparent().remove(elem)
    (el,) = tree.xpath(f"//comment()[contains(.,'div_stats_{page}')]")
    parser = etree.HTMLParser(recover=True)
    (html_table,) = etree.fromstring(el.text, parser).xpath(f"//table[contains(@id, 'stats_{page}')]")
    return _parse_table(html_table)


def get_col(df: pd.DataFrame, spec) -> pd.Series:
    if isinstance(spec, tuple):
        outer, inner = spec
        matches = [c for c in df.columns if c[0] == outer and c[1] == inner]
    else:
        matches = [c for c in df.columns if c[0] == spec] or [c for c in df.columns if c[1] == spec]
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 match for {spec!r}, got {matches}")
    return df[matches[0]]


def extract(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    return pd.DataFrame({name: get_col(df, spec) for name, spec in specs.items()})


def fetch_season(fbref: sd.FBref, lkey: str, skey: str, season_row) -> pd.DataFrame:
    standard = fbref.read_player_season_stats(stat_type="standard")
    standard = standard[(standard.index.get_level_values("league") == lkey)
                         & (standard.index.get_level_values("season") == skey)].reset_index()
    shooting = fbref.read_player_season_stats(stat_type="shooting")
    shooting = shooting[(shooting.index.get_level_values("league") == lkey)
                         & (shooting.index.get_level_values("season") == skey)].reset_index()
    defense = fetch_custom_stat_type(fbref, lkey, skey, season_row, "defense")

    base = extract(standard, STANDARD_SPECS)
    shots = extract(shooting, SHOOTING_SPECS)
    def_df = extract(defense, DEFENSE_SPECS)

    merged = base.merge(shots, on=["player", "team"], how="left")
    merged = merged.merge(def_df, on=["player", "team"], how="left")

    merged["league"] = lkey
    merged["season"] = skey
    for col in ["90s", "shots", "tackles_won", "interceptions"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def main() -> None:
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASONS)
    seasons = fbref.read_seasons()

    frames = []
    for (lkey, skey), season_row in seasons.iterrows():
        print(f"Fetching {lkey} {skey}...")
        df = fetch_season(fbref, lkey, skey, season_row)
        print(f"  {len(df)} players")
        frames.append(df)

    all_players = pd.concat(frames, ignore_index=True)
    all_players = all_players[all_players["90s"] > 0].copy()

    for col in ["shots", "tackles_won", "interceptions"]:
        all_players[f"{col}_p90"] = all_players[col] / all_players["90s"]

    out_path = PROCESSED_DIR / "player_advanced_stats.csv"
    all_players.to_csv(out_path, index=False)
    print(f"\nSaved {len(all_players)} player-seasons to {out_path}")


if __name__ == "__main__":
    main()
