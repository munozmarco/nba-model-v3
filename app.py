import os
import math
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime
import joblib
import streamlit as st

# ========= CONFIG / KEYS =========

# In Streamlit Cloud you'll set these as secrets / env vars
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"
ODDS_SPORT = "basketball_nba"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "american"
RAPIDAPI_HOST = "nba-injury-reports.p.rapidapi.com"

# ========= TEAM NAME MAPS =========

TEAM_NAME_MAP = {
    "Atlanta Hawks": "Hawks",
    "Boston Celtics": "Celtics",
    "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets",
    "Chicago Bulls": "Bulls",
    "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks",
    "Denver Nuggets": "Nuggets",
    "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors",
    "Houston Rockets": "Rockets",
    "Indiana Pacers": "Pacers",
    "Los Angeles Clippers": "Clippers",
    "LA Clippers": "Clippers",
    "Los Angeles Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat",
    "Milwaukee Bucks": "Bucks",
    "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans",
    "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder",
    "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers",
    "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers",
    "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs",
    "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz",
    "Washington Wizards": "Wizards",
}

INJURY_TEAM_NAME_MAP = TEAM_NAME_MAP.copy()

def normalize_team_name(api_name: str) -> str:
    if api_name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[api_name]
    raise ValueError(f"Unknown team name from odds API: {api_name!r}")

def normalize_injury_team(api_name: str) -> str:
    if api_name in INJURY_TEAM_NAME_MAP:
        return INJURY_TEAM_NAME_MAP[api_name]
    raise ValueError(f"Unknown injury team name: {api_name!r}")

# ========= MATH HELPERS =========

def american_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)

def prob_to_american(p):
    if p <= 0 or p >= 1 or pd.isna(p):
        return np.nan
    if p >= 0.5:
        return -int(round(100 * p / (1.0 - p)))
    else:
        return int(round(100 * (1.0 - p) / p))

# ========= LOAD MODELS + DATA =========

@st.cache_resource
def load_models_and_data():
    df_pre = pd.read_csv("df_preprocessed.csv")
    df_pre["gameDateTimeEst"] = pd.to_datetime(df_pre["gameDateTimeEst"], utc=True, format="mixed")

    rf_v3 = joblib.load("rf_v3.pkl")
    cb_v3 = joblib.load("cb_v3.pkl")
    meta_model_v3 = joblib.load("meta_model_v3.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    rf_feature_medians = joblib.load("rf_feature_medians.pkl")

    return df_pre, rf_v3, cb_v3, meta_model_v3, feature_cols, rf_feature_medians

df_preprocessed, rf_v3, cb_v3, meta_model_v3, feature_cols, rf_feature_medians = load_models_and_data()

# ========= INJURY LOADER (SAFE) =========

STATUS_WEIGHTS = {
    "out": 1.0,
    "doubtful": 0.7,
    "questionable": 0.4,
    "probable": 0.2,
}

def _raw_fetch_injuries_for_date(date_str: str):
    if not RAPIDAPI_KEY:
        return []
    url = f"https://{RAPIDAPI_HOST}/injuries/{date_str}"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code in (404, 429):
        return []
    resp.raise_for_status()
    return resp.json()

def _build_team_injury_df_from_raw(raw):
    rows = []
    for item in raw:
        api_team = item.get("team") or item.get("team_name") or item.get("teamName")
        player = item.get("player") or item.get("player_name") or item.get("name")
        status = (item.get("status") or "").lower()
        is_star = bool(item.get("star") or item.get("isStar") or False)

        if not api_team or not player or not status:
            continue

        try:
            team = normalize_injury_team(api_team)
        except ValueError:
            continue

        if (
            "out" not in status
            and "doubt" not in status
            and "question" not in status
            and "probable" not in status
        ):
            continue

        base_weight = STATUS_WEIGHTS.get(status.split()[0], 0.3)
        if is_star:
            base_weight *= 1.5

        rows.append(
            {
                "teamName": team,
                "player": player,
                "status": status,
                "is_star": is_star,
                "impact": base_weight,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["teamName", "numPlayersOut", "starOut", "injuryImpact"])

    inj_df = pd.DataFrame(rows)

    team_agg = (
        inj_df.groupby("teamName")
        .agg(
            numPlayersOut=("player", "nunique"),
            starOut=("is_star", lambda x: int(x.any())),
            injuryImpact=("impact", "sum"),
        )
        .reset_index()
    )

    return team_agg

def build_team_injury_df_safe(date_str: str):
    today_str = date.today().isoformat()
    if date_str != today_str:
        # basic plan usually only useful for today; return empty
        return pd.DataFrame(columns=["teamName", "numPlayersOut", "starOut", "injuryImpact"])
    try:
        raw = _raw_fetch_injuries_for_date(today_str)
        return _build_team_injury_df_from_raw(raw)
    except Exception:
        return pd.DataFrame(columns=["teamName", "numPlayersOut", "starOut", "injuryImpact"])

# ========= ODDS API =========

def pick_bookmaker(game_obj, preferred=("draftkings", "fanduel", "betmgm", "pointsbetus")):
    bms = game_obj.get("bookmakers", [])
    if not bms:
        return None
    for key in preferred:
        for bm in bms:
            if bm.get("key") == key:
                return bm
    return bms[0]

def extract_market(bm, market_key):
    for m in bm.get("markets", []):
        if m.get("key") == market_key:
            return m
    return None

def fetch_raw_odds():
    if not ODDS_API_KEY:
        return []
    url = f"{ODDS_API_BASE_URL}/{ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def build_odds_df_from_raw_games(raw_games, target_date=None, tz="America/Chicago"):
    rows = []

    if target_date is not None:
        target_date = pd.to_datetime(target_date).date()

    for g in raw_games:
        game_id = g.get("id")

        # Odds API gives commence_time in UTC; convert to local tz for date filtering
        commence_utc = pd.to_datetime(g.get("commence_time"), utc=True)
        commence_local = commence_utc.tz_convert(tz)

        # Only keep games whose LOCAL date matches the selected date
        if target_date is not None and commence_local.date() != target_date:
            continue

        home_full = g.get("home_team")
        away_full = g.get("away_team")

        try:
            home_team = normalize_team_name(home_full)
            away_team = normalize_team_name(away_full)
        except ValueError:
            continue

        bm = pick_bookmaker(g)
        if bm is None:
            continue

        m_h2h = extract_market(bm, "h2h")
        m_spreads = extract_market(bm, "spreads")
        m_totals = extract_market(bm, "totals")

        ml_home = ml_away = np.nan
        if m_h2h is not None:
            for o in m_h2h.get("outcomes", []):
                name = normalize_team_name(o["name"])
                price = o["price"]
                if name == home_team:
                    ml_home = price
                elif name == away_team:
                    ml_away = price

        spread_home = spread_away = np.nan
        if m_spreads is not None:
            for o in m_spreads.get("outcomes", []):
                name = normalize_team_name(o["name"])
                point = o.get("point")
                if name == home_team:
                    spread_home = point
                elif name == away_team:
                    spread_away = point

        total_points = np.nan
        if m_totals is not None and m_totals.get("outcomes"):
            total_points = m_totals["outcomes"][0].get("point")

        # Use local time as the game datetime for display / filtering
        rows.append({
            "api_game_id": game_id,
            "gameDateTimeEst": commence_local,   # local timezone
            "teamName": home_team,
            "opponentTeamName": away_team,
            "home": 1,
            "teamMoneyline": ml_home,
            "closingSpread_team": spread_home,
            "closingTotal": total_points,
        })
        rows.append({
            "api_game_id": game_id,
            "gameDateTimeEst": commence_local,
            "teamName": away_team,
            "opponentTeamName": home_team,
            "home": 0,
            "teamMoneyline": ml_away,
            "closingSpread_team": spread_away,
            "closingTotal": total_points,
        })

    odds_df = pd.DataFrame(rows)
    if not odds_df.empty:
        odds_df = odds_df.sort_values("gameDateTimeEst").reset_index(drop=True)
    return odds_df


# ========= MODEL PREDICTION HELPERS =========

def get_team_state(team_name, as_of_date):
    games = df_preprocessed[
        (df_preprocessed["teamName"] == team_name)
        & (df_preprocessed["gameDateTimeEst"] < as_of_date)
    ]
    if games.empty:
        raise ValueError(f"No historical games for team {team_name} before {as_of_date}")
    return games.iloc[-1]

def predict_side_prob_v3(team_name, opponent_name, home_flag, game_datetime,
                         injuryImpact_team=0.0, injuryImpact_opp=0.0):
    as_of_date = pd.to_datetime(game_datetime)
    if as_of_date.tzinfo is None:
        as_of_date = as_of_date.tz_localize("UTC")
    else:
        as_of_date = as_of_date.tz_convert("UTC")

    team_state = get_team_state(team_name, as_of_date)
    opp_state = get_team_state(opponent_name, as_of_date)

    feat_row = {
        "home": int(home_flag),
        "teamWinPct": float(team_state["teamWinPct"]),
        "teamRollingWinPct5": float(team_state["teamRollingWinPct5"]),
        "oppRollingWinPct5": float(team_state["oppRollingWinPct5"]),
        "teamOff_5": float(team_state["teamOff_5"]),
        "teamDef_5": float(team_state["teamDef_5"]),
        "teamMargin_5": float(team_state["teamMargin_5"]),
        "oppOff_5": float(team_state["oppOff_5"]),
        "oppDef_5": float(team_state["oppDef_5"]),
        "oppMargin_5": float(team_state["oppMargin_5"]),
        "teamElo_pre": float(team_state["teamElo_pre"]),
        "oppElo_pre": float(team_state["oppElo_pre"]),
        "teamEffElo_pre": float(team_state["teamEffElo_pre"]),
        "oppEffElo_pre": float(team_state["oppEffElo_pre"]),
        "teamSOS10": float(team_state["teamSOS10"]),
        "travelKm": float(team_state["travelKm"]),
        "daysRest": float(team_state["daysRest"]),
        "gamesLast7d": float(team_state["gamesLast7d"]),
    }

    X_match = pd.DataFrame([feat_row])[feature_cols]
    X_match = X_match.fillna(rf_feature_medians)

    # Injury adjustment on Elo
    ELO_PENALTY_PER_IMPACT = 10.0
    team_penalty = injuryImpact_team * ELO_PENALTY_PER_IMPACT
    opp_penalty = injuryImpact_opp * ELO_PENALTY_PER_IMPACT

    X_match.loc[:, "teamEffElo_pre"] = X_match["teamEffElo_pre"] - team_penalty
    X_match.loc[:, "oppEffElo_pre"] = X_match["oppEffElo_pre"] - opp_penalty

    rf_prob = float(rf_v3.predict_proba(X_match)[0][1])
    cb_prob = float(cb_v3.predict_proba(X_match)[0][1])

    elo_diff = float((X_match["teamEffElo_pre"] - X_match["oppEffElo_pre"]).iloc[0])
    elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    meta_input = np.array([[rf_prob, cb_prob, elo_prob]])
    final_prob = float(meta_model_v3.predict_proba(meta_input)[0][1])

    return final_prob

def add_v3_moneyline_edges(odds_df):
    df_o = odds_df.copy()
    df_o["gameDateTimeEst"] = pd.to_datetime(df_o["gameDateTimeEst"], utc=True, format="mixed")

    # Ensure injury columns exist
    for col in [
        "injuryImpact_team",
        "injuryImpact_opp",
    ]:
        if col not in df_o.columns:
            df_o[col] = 0.0

    model_probs = []
    for idx, row in df_o.iterrows():
        try:
            p = predict_side_prob_v3(
                team_name=row["teamName"],
                opponent_name=row["opponentTeamName"],
                home_flag=row["home"],
                game_datetime=row["gameDateTimeEst"],
                injuryImpact_team=row.get("injuryImpact_team", 0.0),
                injuryImpact_opp=row.get("injuryImpact_opp", 0.0),
            )
            model_probs.append(p)
        except Exception as e:
            model_probs.append(np.nan)

    df_o["modelWinProb_v3"] = model_probs
    df_o["marketProb"] = df_o["teamMoneyline"].apply(american_to_prob)
    df_o["fairMoneyline_v3"] = df_o["modelWinProb_v3"].apply(prob_to_american)
    df_o["edgeProb"] = df_o["modelWinProb_v3"] - df_o["marketProb"]
    df_o["edgePct"] = df_o["edgeProb"] * 100.0

    return df_o

# ========= STREAMLIT UI =========

st.title("NBA Model v3 – Daily Edges")

st.write("This app loads your offline-trained v3 model, pulls live odds, and shows model vs market edges.")

today = date.today()
pick_date = st.date_input("Date to score", value=today)

if st.button("Score games for selected date"):
    if not ODDS_API_KEY:
        st.error("ODDS_API_KEY is not set. Configure it as an environment variable / Streamlit secret.")
    else:
        with st.spinner("Fetching odds..."):
            try:
                raw_games = fetch_raw_odds()
            except Exception as e:
                st.error(f"Error fetching odds: {e}")
                st.stop()

        odds_df = build_odds_df_from_raw_games(raw_games, target_date=pick_date.isoformat())

        if odds_df.empty:
            st.warning("No games found for that date.")
            st.stop()

        st.subheader("Raw odds")
        st.dataframe(odds_df)

        # Injuries
        with st.spinner("Fetching injuries (if available)..."):
            inj_df = build_team_injury_df_safe(pick_date.isoformat())

        if inj_df.empty:
            odds_df["injuryImpact_team"] = 0.0
            odds_df["injuryImpact_opp"] = 0.0
        else:
            odds_df = odds_df.merge(inj_df, on="teamName", how="left")
            odds_df = odds_df.rename(
                columns={
                    "injuryImpact": "injuryImpact_team",
                }
            )
            inj_opp = inj_df.rename(
                columns={
                    "teamName": "opponentTeamName",
                    "injuryImpact": "injuryImpact_opp",
                }
            )
            odds_df = odds_df.merge(inj_opp, on="opponentTeamName", how="left")
            for col in ["injuryImpact_team", "injuryImpact_opp"]:
                odds_df[col] = odds_df[col].fillna(0.0)

        with st.spinner("Scoring games with v3..."):
            scored = add_v3_moneyline_edges(odds_df)

        st.subheader("Model vs Market – Moneyline Edges")
        show_cols = [
            "gameDateTimeEst",
            "teamName",
            "opponentTeamName",
            "home",
            "teamMoneyline",
            "marketProb",
            "modelWinProb_v3",
            "fairMoneyline_v3",
            "edgePct",
            "injuryImpact_team",
            "injuryImpact_opp",
        ]
        scored_display = scored[show_cols].sort_values("edgePct", ascending=False)
        st.dataframe(scored_display.reset_index(drop=True))
