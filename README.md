# FIFA World Cup 2026 — Prediction Model & EDA

A data science project that analyzes 150+ years of international football results and builds a prediction model for the FIFA World Cup 2026, combining Elo ratings, Poisson goal-expectancy modeling, and Monte Carlo tournament simulation.

<img width="1545" height="1067" alt="image" src="https://github.com/user-attachments/assets/b6deea5e-f04d-42a8-a0af-634be4430281" />


## Overview

This repo does three things:

1. **Exploratory data analysis** on World Cup match results, goalscorers, and penalty shootouts (1930–2026).
2. **A prediction model** that rates every national team using a time-decayed, margin-weighted Elo system, layers on engineered features (form, fatigue, knockout pressure, head-to-head history), and predicts match scorelines with a dual Poisson XGBoost model.
3. **An interactive dashboard** that visualizes live team ratings, championship probabilities, and a match simulator.

## Data source

[`martj42/international_results`](https://github.com/martj42/international_results) — 49,000+ international football results from 1872 to 2026, including:

| File | Description |
|---|---|
| `results.csv` | Match results: teams, scores, tournament, venue, neutral-site flag |
| `goalscorers.csv` | Individual goal events: scorer, minute, penalty/own-goal flags |
| `shootouts.csv` | Penalty shootout winners and shooting order |

`goalscorers.csv` and `shootouts.csv` don't include a tournament column, so both are joined against `results.csv` (filtered to `FIFA World Cup`) on `date + home_team + away_team` to isolate World Cup–only events.

## Repo structure

```
.
├── eda_wc.py              # EDA on World Cup match results
├── eda_gs_so.py            # EDA on goalscorers + penalty shootouts
├── wc2026_model.py          # Elo + Poisson XGBoost prediction pipeline
├── wc2026_dashboard.html    # Interactive prediction dashboard
├── output_plots/             # Generated charts (created on run)
└── international_results/   # Cloned dataset (git submodule / manual clone)
```

## Setup

```bash
git clone https://github.com/PrahasHegde/FIFA_WC_prediction_EDA.git
cd FIFA_WC_prediction_EDA
git clone https://github.com/martj42/international_results.git

pip install pandas numpy matplotlib seaborn xgboost scipy
```

## 1. Exploratory data analysis

### Match results

```bash
python eda_wc.py
```

Generates 8 charts in `output_plots/` covering:
- Average goals per match by edition, and total matches played per edition
- Match outcome distribution (home win / draw / away win) and goals-per-match histogram
- Top 20 all-time goal-scoring nations
- Highest win-rate nations (min. 15 matches)
- Goal difference distribution and the highest-scoring matches in tournament history
- Host-nation win rate (home advantage)
- Head-to-head win heatmap among the top 10 nations
- Decade-level scoring trend with a NumPy linear regression fit

**Key findings**: scoring has declined steadily since the 1950s (correlation of -0.77 between decade and average goals/match); Brazil leads all-time in both goals scored (246) and win rate (66.9%, min. 15 matches); roughly 1 in 5 World Cup matches ends in a draw.

### Goalscorers & shootouts

```bash
python eda_gs_so.py
```

Generates 6 charts covering:
- Top 15 all-time individual goalscorers
- Goal timing by match segment (when in a match goals are scored)
- Penalty and own-goal share of total goals, by edition
- Top scoring nations (cross-checked against goalscorer-level data)
- Shootouts per edition and the first-shooter win rate
- Most shootout wins by nation

**Key findings**: goals are heavily backloaded toward the final 15 minutes of regulation; penalties account for 7.5% of all World Cup goals and own goals for 2.2%; the first team to shoot in a penalty shootout wins only 45.9% of the time (not statistically different from a coin flip); Argentina has the most shootout wins of any nation (6).

## 2. Prediction model

```bash
python wc2026_model.py
```

Pipeline:

1. **Time-decayed, margin-weighted Elo ratings** — updates are scaled by goal difference and competition importance (World Cup finals weighted higher than friendlies), following the World Football Elo Ratings methodology.
2. **Engineered features**:
   - Exponentially-decayed recent form (no arbitrary "last 5 games" cutoff)
   - Fatigue proxy (days since each team's last match)
   - Knockout-stage flag (scoring typically drops in elimination matches)
   - Head-to-head "surprise factor" (average deviation from Elo expectation in past meetings, capturing rivalry effects)
   - Confederation-average-Elo fallback for teams with thin international history
3. **Dual Poisson XGBoost models** predict home and away goal-scoring rates separately, allowing full scoreline simulation rather than just win/draw/loss classification.
4. **Date-based train/test split** (pre-2018 train, 2018–2025 holdout) to avoid leaking future team strength into the past.
5. **Monte Carlo bracket simulation** — samples scorelines from the predicted Poisson rates across the full 48-team knockout structure to estimate each team's probability of winning the tournament.

On a 2018–2025 holdout, the model reaches **60.3% match outcome accuracy** (vs. ~33% for random guessing across three outcomes).

## 3. Interactive dashboard

Open `wc2026_dashboard.html` directly in a browser — no server required.

Includes:
- Live Elo leaderboard with ranking chart
- Championship probability chart (simulated tournament win share by team)
- Match simulator — pick any two teams and get a sampled scoreline plus win/draw/loss probabilities
- Feature importance breakdown
- A summary of the model's six innovative signals beyond standard Elo

The dashboard currently runs on static, pre-computed data embedded in the page. To connect it to live model output, export `live_elo`, `title_counts`, and `feat_importance` from `wc2026_model.py` to JSON and replace the hardcoded `teams` / `features` arrays with a `fetch()` call.

## Methodology notes

- Most World Cup matches are played at neutral venues, so "home" and "away" team labels reflect listing order, not true home-field advantage, except where the `neutral` column is `False`.
- Train/test splits are always done by date for any time-series model in this repo — random splits would leak future team strength into historical predictions.
- The Elo system and Poisson goal model are validated independently against a holdout period before being combined; if engineered features don't outperform a pure-Elo baseline, the added complexity isn't justified.

## License

Data is sourced from the [`martj42/international_results`](https://github.com/martj42/international_results) repository under its own license terms. Code in this repository is provided as-is for educational and analytical purposes.
