import os
import math
import requests
import numpy as np
import pandas as pd
from datetime import date
import joblib
import streamlit as st
from sklearn.linear_model import LinearRegression

# Auto-refresh (optional dependency)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="NBA Model v3",
    layout="wide",  # looks good on desktop & still okay on mobile
)

# ---------- API KEYS / CONFIG ----------

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"
ODDS_SPORT = "basketball_nba"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "american"

RAPIDAPI_HOST = "nba-injury-reports.p.rapidapi.com"

LOCAL_TZ = "America/Chicago"

# ---------- TEAM NAME MAPPING ----------

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
    raise ValueError(f"Unknown team from odds API: {api_name!r}")


def normalize_injury_team(api_name: str) -> str:
    if api_name in INJURY_TEAM_NAME_MAP:
        return INJURY_TEAM_NAME_MAP[api_name]
    raise ValueError(f"Unknown team from injury API: {api_name!r}")


# ---------- MATH HELPERS ----------

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


# ---------- LOAD DATA + MODELS ----------

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


# ---------- SIMPLE SPREAD MODEL (trained once on app start) ----------

@st.cache_resource
def train_spread_model():
    df = df_preprocessed.dropna(subset=["teamEffElo_pre", "oppEffElo_pre", "margin"])
    if df.empty:
        return None

    X = (df["teamEffElo_pre"] - df["oppEffElo_pre"]).values.reshape(-1, 1)
    y = df["margin"].values

    lr = LinearRegression()
    lr.fit(X, y)
    return lr


spread_model_v1 = train_spread_model()


# ---------- INJURIES (PLAYER-LEVEL + TEAM AGG) ----------

STATUS_WEIGHTS = {
    "out": 1.0,
    "doubtful": 0.7,
    "questionable": 0.4,
    "probable": 0.2,
}


@st.cache_data(ttl=86400)
def fetch_injuries_raw_and_agg(date_str: str):
    """Hit the injury API at most once/day, return raw player table + team aggregate."""
    if not RAPIDAPI_KEY:
        return pd.DataFrame(), pd.DataFrame()

    url = f"https://{RAPIDAPI_HOST}/injuries/{date_str}"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }

    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code in (404, 429):
            return pd.DataFrame(), pd.DataFrame()
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for item in data:
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

        # Only count if some kind of injury designation
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
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.DataFrame(rows)

    team_agg = (
        raw_df.groupby("teamName")
        .agg(
            numPlayersOut=("player", "nunique"),
            starOut=("is_star", lambda x: int(x.any())),
            injuryImpact=("impact", "sum"),
        )
        .reset_index()
    )

    return raw_df, team_agg


# ---------- ODDS API HELPERS ----------

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


@st.cache_data(ttl=60)
def fetch_raw_odds():
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
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


def build_odds_df_from_raw_games(raw_games, target_date=None, tz=LOCAL_TZ):
    rows = []

    if target_date is not None:
        target_date = pd.to_datetime(target_date).date()

    for g in raw_games:
        game_id = g.get("id")
        commence_utc = pd.to_datetime(g.get("commence_time"), utc=True)
        commence_local = commence_utc.tz_convert(tz)

        # Filter by LOCAL calendar date
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

        # Home row
        rows.append({
            "api_game_id": game_id,
            "gameDateTimeEst": commence_local,
            "teamName": home_team,
            "opponentTeamName": away_team,
            "home": 1,
            "teamMoneyline": ml_home,
            "closingSpread_team": spread_home,
            "closingTotal": total_points,
        })
        # Away row
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


# ---------- MODEL FEATURE BUILDER + PREDICTION ----------

def get_team_state(team_name, as_of_date):
    games = df_preprocessed[
        (df_preprocessed["teamName"] == team_name)
        & (df_preprocessed["gameDateTimeEst"] < as_of_date)
    ]
    if games.empty:
        raise ValueError(f"No historical games for {team_name} before {as_of_date}")
    return games.iloc[-1]


def build_feature_row(team_name, opponent_name, home_flag, game_datetime,
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

    # Injury → adjust effective Elo
    ELO_PENALTY_PER_IMPACT = 10.0
    team_penalty = injuryImpact_team * ELO_PENALTY_PER_IMPACT
    opp_penalty = injuryImpact_opp * ELO_PENALTY_PER_IMPACT

    X_match.loc[:, "teamEffElo_pre"] = X_match["teamEffElo_pre"] - team_penalty
    X_match.loc[:, "oppEffElo_pre"] = X_match["oppEffElo_pre"] - opp_penalty

    # Elo diff AFTER injury adjustment
    elo_diff = float((X_match["teamEffElo_pre"] - X_match["oppEffElo_pre"]).iloc[0])

    return X_match, elo_diff


def predict_side_prob_v3(team_name, opponent_name, home_flag, game_datetime,
                         injuryImpact_team=0.0, injuryImpact_opp=0.0,
                         return_elo_diff=False):
    X_match, elo_diff = build_feature_row(
        team_name, opponent_name, home_flag, game_datetime,
        injuryImpact_team, injuryImpact_opp
    )

    rf_prob = float(rf_v3.predict_proba(X_match)[0][1])
    cb_prob = float(cb_v3.predict_proba(X_match)[0][1])

    # Elo-based prob
    elo_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    meta_input = np.array([[rf_prob, cb_prob, elo_prob]])
    final_prob = float(meta_model_v3.predict_proba(meta_input)[0][1])

    if return_elo_diff:
        return final_prob, elo_diff
    return final_prob


def add_v3_moneyline_and_spread_edges(odds_df):
    df_o = odds_df.copy()
    df_o["gameDateTimeEst"] = pd.to_datetime(df_o["gameDateTimeEst"])

    # Ensure injury columns exist
    for col in ["injuryImpact_team", "injuryImpact_opp"]:
        if col not in df_o.columns:
            df_o[col] = 0.0

    model_probs = []
    elo_diffs = []

    for _, row in df_o.iterrows():
        try:
            p, ediff = predict_side_prob_v3(
                team_name=row["teamName"],
                opponent_name=row["opponentTeamName"],
                home_flag=row["home"],
                game_datetime=row["gameDateTimeEst"],
                injuryImpact_team=row.get("injuryImpact_team", 0.0),
                injuryImpact_opp=row.get("injuryImpact_opp", 0.0),
                return_elo_diff=True,
            )
            model_probs.append(p)
            elo_diffs.append(ediff)
        except Exception:
            model_probs.append(np.nan)
            elo_diffs.append(np.nan)

    df_o["modelWinProb_v3"] = model_probs
    df_o["marketProb"] = df_o["teamMoneyline"].apply(american_to_prob)
    df_o["fairMoneyline_v3"] = df_o["modelWinProb_v3"].apply(prob_to_american)
    df_o["edgeProb"] = df_o["modelWinProb_v3"] - df_o["marketProb"]
    df_o["edgePct"] = df_o["edgeProb"] * 100.0

    # Spread model (if trained)
    df_o["eloDiff"] = elo_diffs
    if spread_model_v1 is not None:
        model_spreads = []
        for d in df_o["eloDiff"]:
            if pd.isna(d):
                model_spreads.append(np.nan)
            else:
                model_spreads.append(float(spread_model_v1.predict([[d]])[0]))
        df_o["modelSpread_v3"] = model_spreads
        df_o["spreadEdgePts"] = df_o["modelSpread_v3"] - df_o["closingSpread_team"]
    else:
        df_o["modelSpread_v3"] = np.nan
        df_o["spreadEdgePts"] = np.nan

    return df_o


# ---------- STYLING HELPERS (EV HIGHLIGHTING) ----------

def style_edge_col(s, threshold=0.0):
    styles = []
    for v in s:
        if pd.isna(v):
            styles.append("")
        elif v >= threshold:
            styles.append("background-color: rgba(0, 200, 0, 0.25);")
        elif v <= -threshold:
            styles.append("background-color: rgba(255, 0, 0, 0.25);")
        else:
            styles.append("")
    return styles


# ---------- UI ----------

st.title("NBA Model v3 – Daily Edges")

st.caption(
    "Uses offline-trained models + live odds. "
    "This is for education / modeling only, not betting advice."
)

# Sidebar controls
st.sidebar.header("Controls")

today = date.today()
picked_date = st.sidebar.date_input("Date", value=today)

min_edge = st.sidebar.slider("Min moneyline edge (%)", 0.0, 25.0, 5.0, 0.5)
min_spread_edge = st.sidebar.slider("Min spread edge (points)", 0.0, 25.0, 2.0, 0.5)

only_plus_ev = st.sidebar.checkbox("Show only +EV sides", value=True)
home_only = st.sidebar.checkbox("Home teams only", value=False)
team_filter_text = st.sidebar.text_input("Filter by team (optional)", value="").strip()

# Auto-refresh
st.sidebar.subheader("Auto-refresh")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 2 minutes", value=False)
if auto_refresh and st_autorefresh is not None:
    st_autorefresh(interval=120_000, key="odds_autorefresh")
elif auto_refresh and st_autorefresh is None and auto_refresh:
    st.sidebar.warning("Install 'streamlit-autorefresh' in requirements.txt to enable auto-refresh.")

# Main button
run_button = st.button("Score games for selected date")

if run_button:
    if not ODDS_API_KEY:
        st.error("ODDS_API_KEY is not set (configure in Streamlit secrets).")
        st.stop()

    # Fetch odds
    with st.spinner("Fetching odds..."):
        try:
            raw_games = fetch_raw_odds()
        except Exception as e:
            st.error(f"Error fetching odds: {e}")
            st.stop()

    odds_df = build_odds_df_from_raw_games(raw_games, target_date=picked_date.isoformat())

    if odds_df.empty:
        st.warning("No games found for that date.")
        st.stop()

    # Injuries
    with st.spinner("Fetching injuries (if available)..."):
        raw_inj_df, team_inj_df = fetch_injuries_raw_and_agg(picked_date.isoformat())

    # Merge injuries into odds_df
    if team_inj_df.empty:
        odds_df["injuryImpact_team"] = 0.0
        odds_df["injuryImpact_opp"] = 0.0
    else:
        odds_df = odds_df.merge(team_inj_df, on="teamName", how="left")
        odds_df = odds_df.rename(columns={"injuryImpact": "injuryImpact_team"})
        inj_opp = team_inj_df.rename(
            columns={"teamName": "opponentTeamName", "injuryImpact": "injuryImpact_opp"}
        )
        odds_df = odds_df.merge(inj_opp, on="opponentTeamName", how="left")
        for col in ["injuryImpact_team", "injuryImpact_opp"]:
            odds_df[col] = odds_df[col].fillna(0.0)

    with st.spinner("Scoring games with v3 + spread model..."):
        scored = add_v3_moneyline_and_spread_edges(odds_df)

    # ---------- FILTERS ----------

    view = scored.copy()

    if home_only:
        view = view[view["home"] == 1]

    if team_filter_text:
        mask = view["teamName"].str.contains(team_filter_text, case=False) | \
               view["opponentTeamName"].str.contains(team_filter_text, case=False)
        view = view[mask]

    if only_plus_ev:
        view_ml = view[view["edgePct"] >= min_edge].copy()
    else:
        view_ml = view[view["edgePct"].abs() >= min_edge].copy()

    # For spreads, we usually care about absolute edge
    view_spread = view[view["spreadEdgePts"].abs() >= min_spread_edge].copy()

    # ---------- MONEYLINE TABLE (EV HIGHLIGHT) ----------

    st.subheader("Moneyline Edges (v3)")

    if view_ml.empty:
        st.info("No moneyline sides meet the current filters.")
    else:
        ml_cols = [
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
        ml_display = (
            view_ml[ml_cols]
            .sort_values("edgePct", ascending=False)
            .reset_index(drop=True)
        )

        styled_ml = ml_display.style.apply(
            style_edge_col,
            subset=["edgePct"],
            threshold=min_edge,
        )

        st.dataframe(styled_ml, use_container_width=True)

    # ---------- SPREAD TABLE (EV HIGHLIGHT) ----------

    st.subheader("Spread Edges (simple Elo-based model)")

    if spread_model_v1 is None:
        st.info("Spread model is not available (could not train from df_preprocessed).")
    elif view_spread.empty:
        st.info("No spread edges meet the current filters.")
    else:
        sp_cols = [
            "gameDateTimeEst",
            "teamName",
            "opponentTeamName",
            "home",
            "closingSpread_team",
            "modelSpread_v3",
            "spreadEdgePts",
        ]
        sp_display = (
            view_spread[sp_cols]
            .sort_values("spreadEdgePts", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True)
        )

        styled_sp = sp_display.style.apply(
            style_edge_col,
            subset=["spreadEdgePts"],
            threshold=min_spread_edge,
        )

        st.dataframe(styled_sp, use_container_width=True)

    # ---------- PLAYER-LEVEL INJURIES ----------

    st.subheader("Injury Report (by team)")

    if raw_inj_df.empty:
        st.caption("No injury data available from the API for this date (or key/plan limits).")
    else:
        teams_in_slate = sorted(set(view["teamName"]).union(set(view["opponentTeamName"])))
        for t in teams_in_slate:
            t_inj = raw_inj_df[raw_inj_df["teamName"] == t]
            if t_inj.empty:
                continue
            with st.expander(f"{t} injuries ({len(t_inj)})", expanded=False):
                show_cols = ["player", "status", "is_star", "impact"]
                st.table(t_inj[show_cols].reset_index(drop=True))

else:
    st.info("Pick a date and click 'Score games for selected date' to run the model.")
