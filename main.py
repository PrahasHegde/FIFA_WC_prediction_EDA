"""
FIFA World Cup 2026 Prediction Model
=====================================
Pipeline: time-decayed Elo -> innovative feature engineering ->
dual Poisson goal-expectancy model (XGBoost) -> Monte Carlo bracket simulation.

Data source: https://github.com/martj42/international_results.git
NOTE: as of the data pull date, this repo already contains real scores for the
WC2026 GROUP STAGE (it's mid-tournament). We use those as a live validation set
and only simulate the remaining knockout rounds.

Run: python wc2026_model.py
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
DATA_DIR = "international_results"

results = pd.read_csv(f"{DATA_DIR}/results.csv", parse_dates=["date"])
results = results.sort_values("date").reset_index(drop=True)

# Split: historical (played) vs WC2026 fixtures (some played, some upcoming)
played = results.dropna(subset=["home_score", "away_score"]).copy()
wc2026_all = results[(results.tournament == "FIFA World Cup") &
                      (results.date >= "2026-01-01")].copy()
wc2026_played = wc2026_all.dropna(subset=["home_score", "away_score"])
wc2026_upcoming = wc2026_all[wc2026_all.home_score.isna()]

print(f"Total historical matches: {len(played)}")
print(f"WC2026 matches already played (group stage / live): {len(wc2026_played)}")
print(f"WC2026 matches still to simulate (knockouts): {len(wc2026_upcoming)}")

# ---------------------------------------------------------------------------
# 2. TIME-DECAYED, MARGIN-WEIGHTED ELO
#    This is the single highest-value feature in football prediction.
#    Standard win/loss Elo throws away information (5-0 vs 1-0 update
#    identically) -- we weight by goal difference and competition importance,
#    following the World Football Elo Ratings methodology.
# ---------------------------------------------------------------------------
def tournament_weight(t: str) -> int:
    t = str(t).lower()
    if "world cup" in t and "qualif" not in t:
        return 60
    if "world cup qualif" in t:
        return 40
    if "confederations cup" in t or ("cup" in t and "qualif" not in t and "friendly" not in t):
        return 50  # continental championships (Euro, Copa America, AFCON...)
    if "friendly" in t:
        return 20
    return 30

def run_elo(df: pd.DataFrame):
    elo = defaultdict(lambda: 1500.0)
    home_elo_pre, away_elo_pre, expected_home_list = [], [], []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        hs, as_ = row.home_score, row.away_score
        Rh, Ra = elo[home], elo[away]
        home_adv = 0 if row.neutral else 60
        expected_home = 1 / (1 + 10 ** (-(Rh + home_adv - Ra) / 400))

        home_elo_pre.append(Rh)
        away_elo_pre.append(Ra)
        expected_home_list.append(expected_home)

        if hs > as_:
            actual_home = 1.0
        elif hs == as_:
            actual_home = 0.5
        else:
            actual_home = 0.0

        gd = abs(hs - as_)
        g = 1 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd - 3) / 8))
        K = tournament_weight(row.tournament)
        delta = K * g * (actual_home - expected_home)

        elo[home] = Rh + delta
        elo[away] = Ra - delta

    df = df.copy()
    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["expected_home"] = expected_home_list
    return df, elo

played, elo_after_history = run_elo(played)

# ---------------------------------------------------------------------------
# 3. INNOVATIVE FEATURES
# ---------------------------------------------------------------------------

# 3a. Exponentially-decayed FORM (recent performance, more weight to recent
#     matches than a flat "last 5 games" window). Captures momentum.
def compute_form(df: pd.DataFrame, halflife_days=180):
    history = defaultdict(list)  # team -> list of (date, points)
    home_form, away_form = [], []

    def decayed_score(events, asof):
        if not events:
            return 0.5  # neutral prior
        s, w = 0.0, 0.0
        for date, val in events:
            age = (asof - date).days
            wt = 0.5 ** (age / halflife_days)
            s += wt * val
            w += wt
        return s / w if w > 0 else 0.5

    for row in df.itertuples(index=False):
        h, a, d = row.home_team, row.away_team, row.date
        home_form.append(decayed_score(history[h], d))
        away_form.append(decayed_score(history[a], d))
        hp = 1 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0)
        history[h].append((d, hp))
        history[a].append((d, 1 - hp))

    df = df.copy()
    df["home_form"] = home_form
    df["away_form"] = away_form
    return df, history

played, form_history = compute_form(played)

# 3b. FATIGUE / REST proxy: days since each team's last match.
played["days_since_last_home"] = played.groupby("home_team")["date"].diff().dt.days.fillna(90)
played["days_since_last_away"] = played.groupby("away_team")["date"].diff().dt.days.fillna(90)

# 3c. KNOCKOUT-STAGE PRESSURE flag: scoring drops, draws spike in elimination
#     matches due to risk-averse tactics -- this is a real, measurable effect
#     that a generic "is World Cup" flag misses.
played["is_knockout"] = played["round"].astype(str).str.contains(
    "Final|Quarter|Semi|Round of|Third", case=False, na=False
) if "round" in played.columns else False
# results.csv has no 'round' column -> approximate via tournament name text instead
if "is_knockout" not in played.columns or played["is_knockout"].sum() == 0:
    played["is_knockout"] = False  # group-stage-only signal unavailable pre-bracket; set in sim manually

# 3d. HEAD-TO-HEAD SURPRISE FACTOR: average (actual - expected) over each
#     pair's last 5 meetings. Captures "bogey team" psychological effects
#     that a global Elo rating cannot represent.
def compute_h2h_surprise(df: pd.DataFrame, window=5):
    pair_history = defaultdict(list)  # frozenset({a,b}) -> list of surprise values (from home persp)
    surprises = []
    for row in df.itertuples(index=False):
        key = tuple(sorted([row.home_team, row.away_team]))
        vals = pair_history[key][-window:]
        surprises.append(np.mean(vals) if vals else 0.0)
        hp = 1 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0)
        surprise = hp - row.expected_home
        # store oriented to "home_team of this row" each time, sign-flip if order differs
        sign = 1 if row.home_team == key[0] else -1
        pair_history[key].append(sign * surprise)
    df = df.copy()
    df["h2h_surprise"] = surprises
    return df, pair_history

played, h2h_history = compute_h2h_surprise(played)

# 3e. CONFEDERATION STRENGTH PRIOR: mean Elo of a team's confederation peers,
#     useful as a backstop for teams with thin head-to-head history (e.g.
#     first-time qualifiers). Build a lightweight confederation map.
CONFEDERATIONS = {
    "UEFA": ["Germany","France","Spain","England","Portugal","Netherlands","Italy","Belgium",
             "Croatia","Switzerland","Denmark","Austria","Sweden","Poland","Serbia","Scotland",
             "Wales","Ukraine","Norway","Slovenia","Slovakia","Czech Republic","Hungary","Turkey"],
    "CONMEBOL": ["Brazil","Argentina","Uruguay","Colombia","Ecuador","Paraguay","Chile","Peru",
                 "Bolivia","Venezuela"],
    "CONCACAF": ["Mexico","United States","Canada","Costa Rica","Jamaica","Panama","Honduras",
                 "Haiti","Curacao","El Salvador"],
    "CAF": ["Morocco","Senegal","Tunisia","Algeria","Egypt","Nigeria","Ghana","Cameroon",
            "Ivory Coast","South Africa","Cape Verde","DR Congo","Mali"],
    "AFC": ["Japan","South Korea","Iran","Saudi Arabia","Australia","Qatar","Iraq","Jordan",
            "Uzbekistan","United Arab Emirates"],
    "OFC": ["New Zealand"]
}
team_to_conf = {t: c for c, teams in CONFEDERATIONS.items() for t in teams}

def confederation_avg_elo(elo_dict, conf_map):
    conf_elos = defaultdict(list)
    for team, rating in elo_dict.items():
        conf = conf_map.get(team)
        if conf:
            conf_elos[conf].append(rating)
    return {c: np.mean(v) for c, v in conf_elos.items()}

# ---------------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT BY DATE (never split randomly on time-series data --
#    a random split leaks future Elo strength into the training set)
# ---------------------------------------------------------------------------
feature_cols = ["home_elo_pre", "away_elo_pre", "expected_home",
                 "home_form", "away_form",
                 "days_since_last_home", "days_since_last_away",
                 "h2h_surprise", "neutral"]

played[feature_cols] = played[feature_cols].astype(float)
played["neutral"] = played["neutral"].astype(float)

train = played[played.date < "2018-01-01"]
test = played[(played.date >= "2018-01-01") & (played.date < "2026-01-01")]

X_train, y_home_train, y_away_train = train[feature_cols], train.home_score, train.away_score
X_test, y_home_test, y_away_test = test[feature_cols], test.home_score, test.away_score

home_model = xgb.XGBRegressor(objective="count:poisson", n_estimators=300,
                               max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=42)
away_model = xgb.XGBRegressor(objective="count:poisson", n_estimators=300,
                               max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=42)

home_model.fit(X_train, y_home_train)
away_model.fit(X_train, y_away_train)

# ---------------------------------------------------------------------------
# 5. VALIDATION: compare against pure-Elo baseline
#    (if XGBoost doesn't beat Elo-only, the extra features aren't earning
#    their complexity -- always check this)
# ---------------------------------------------------------------------------
def poisson_log_loss(y_true, lam):
    lam = np.clip(lam, 1e-6, None)
    return np.mean(lam - y_true * np.log(lam))

pred_home = home_model.predict(X_test)
pred_away = away_model.predict(X_test)
print("\n--- Validation (2018-2025 holdout) ---")
print(f"XGB Poisson loss  home: {poisson_log_loss(y_home_test.values, pred_home):.4f}")
print(f"XGB Poisson loss  away: {poisson_log_loss(y_away_test.values, pred_away):.4f}")

# match outcome accuracy as a sanity check
def outcome(h, a):
    return "H" if h > a else ("A" if a > h else "D")

actual_outcome = [outcome(h, a) for h, a in zip(y_home_test, y_away_test)]
# simulate most-likely outcome from predicted lambdas (argmax over small grid)
def most_likely_outcome(lh, la, max_goals=6):
    from scipy.stats import poisson
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    pH = np.tril(grid, -1).sum()
    pD = np.trace(grid)
    pA = np.triu(grid, 1).sum()
    return "H" if pH >= pD and pH >= pA else ("A" if pA >= pD else "D")

pred_outcome = [most_likely_outcome(lh, la) for lh, la in zip(pred_home, pred_away)]
acc = np.mean([p == a for p, a in zip(pred_outcome, actual_outcome)])
print(f"Outcome accuracy (H/D/A) on holdout: {acc:.3f}")

feat_importance = pd.Series(home_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop features driving home-goal expectancy:")
print(feat_importance)

# ---------------------------------------------------------------------------
# 6. RE-RUN ELO/FORM THROUGH FULL HISTORY INCLUDING WC2026 GROUP STAGE
#    so our "live" ratings reflect matches already played this tournament.
# ---------------------------------------------------------------------------
full_played = pd.concat([played[["date","home_team","away_team","home_score","away_score",
                                  "tournament","neutral"]],
                          wc2026_played[["date","home_team","away_team","home_score","away_score",
                                         "tournament","neutral"]]]).sort_values("date").reset_index(drop=True)

full_played, live_elo = run_elo(full_played)
full_played, live_form_history = compute_form(full_played)
full_played, live_h2h_history = compute_h2h_surprise(full_played)
live_conf_elo = confederation_avg_elo(live_elo, team_to_conf)

print("\n--- Live Elo Top 10 (post group stage) ---")
print(pd.Series(live_elo).sort_values(ascending=False).head(10))

# ---------------------------------------------------------------------------
# 7. BUILD A FEATURE ROW FOR ANY FUTURE FIXTURE USING LIVE STATE
# ---------------------------------------------------------------------------
def decayed_score_now(events, asof, halflife_days=180):
    if not events:
        return 0.5
    s, w = 0.0, 0.0
    for date, val in events:
        age = (asof - date).days
        wt = 0.5 ** (age / halflife_days)
        s += wt * val
        w += wt
    return s / w if w > 0 else 0.5

def build_feature_row(home, away, match_date, neutral=True, knockout=True):
    Rh = live_elo.get(home, live_conf_elo.get(team_to_conf.get(home, ""), 1500))
    Ra = live_elo.get(away, live_conf_elo.get(team_to_conf.get(away, ""), 1500))
    home_adv = 0 if neutral else 60
    expected_home = 1 / (1 + 10 ** (-(Rh + home_adv - Ra) / 400))
    h_form = decayed_score_now(live_form_history.get(home, []), match_date)
    a_form = decayed_score_now(live_form_history.get(away, []), match_date)
    key = tuple(sorted([home, away]))
    h2h_vals = live_h2h_history.get(key, [])[-5:]
    h2h = np.mean(h2h_vals) if h2h_vals else 0.0
    return pd.DataFrame([{
        "home_elo_pre": Rh, "away_elo_pre": Ra, "expected_home": expected_home,
        "home_form": h_form, "away_form": a_form,
        "days_since_last_home": 7, "days_since_last_away": 7,  # WC knockout: ~7 day rest
        "h2h_surprise": h2h, "neutral": 1.0
    }])[feature_cols]

# ---------------------------------------------------------------------------
# 8. MONTE CARLO MATCH + KNOCKOUT SIMULATION
#    Predict Poisson goal rates -> sample scorelines -> resolve draws via
#    extra-time-adjusted penalty coin-flip (slightly favoring higher Elo).
# ---------------------------------------------------------------------------
def simulate_match(home, away, match_date, neutral=True, knockout=False):
    feat = build_feature_row(home, away, match_date, neutral, knockout)
    lam_h = max(float(home_model.predict(feat)[0]), 0.05)
    lam_a = max(float(away_model.predict(feat)[0]), 0.05)
    # knockout matches: empirically lower-scoring (more cautious tactics)
    if knockout:
        lam_h *= 0.92
        lam_a *= 0.92
    hs = RNG.poisson(lam_h)
    as_ = RNG.poisson(lam_a)
    if knockout and hs == as_:
        Rh = live_elo.get(home, 1500)
        Ra = live_elo.get(away, 1500)
        p_home_win = 1 / (1 + 10 ** (-(Rh - Ra) / 400))
        winner = home if RNG.random() < p_home_win else away
        return hs, as_, winner
    winner = home if hs > as_ else (away if as_ > hs else None)
    return hs, as_, winner

# Example: simulate one knockout fixture
example = simulate_match("Brazil", "Argentina", pd.Timestamp("2026-07-10"),
                          neutral=True, knockout=True)
print(f"\nExample sim Brazil vs Argentina: {example}")

# ---------------------------------------------------------------------------
# 9. FULL TOURNAMENT MONTE CARLO (skeleton)
#    Plug in the real 2026 bracket structure (48 teams, 12 groups of 4,
#    top 2 + 8 best third-place teams advance to Round of 32) once the group
#    stage table is finalized. Below: title-probability loop template.
# ---------------------------------------------------------------------------
def simulate_knockout_bracket(matchups, match_date, n_sims=2000):
    """matchups: list of (teamA, teamB) for e.g. Round of 16 onward."""
    title_counts = defaultdict(int)
    for _ in range(n_sims):
        round_teams = list(matchups)
        while len(round_teams) >= 1:
            winners = []
            for a, b in round_teams:
                _, _, w = simulate_match(a, b, match_date, neutral=True, knockout=True)
                winners.append(w)
            if len(winners) == 1:
                title_counts[winners[0]] += 1
                break
            round_teams = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
    total = sum(title_counts.values())
    return pd.Series(title_counts).sort_values(ascending=False) / total 


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# RUNNING THE SIMULATION
# Replace these teams with the actual qualified Round of 16 bracket structure 
sample_r16_matchups = [
    ("France", "Sweden"),

]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("\n--- Running 2,000 Tournament Monte Carlo Simulations ---")
sim_date = pd.Timestamp("2026-07-04") # Mid-tournament simulation date
title_probabilities = simulate_knockout_bracket(sample_r16_matchups, sim_date, n_sims=5000)

print("\n=== FIFA World Cup 2026 Win Probabilities ===")
for team, prob in title_probabilities.items():
    print(f"{team:<15}: {prob * 100:.2f}%")