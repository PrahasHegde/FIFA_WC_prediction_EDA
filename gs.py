"""
FIFA World Cup — Goalscorers & Penalty Shootouts EDA
Data: https://github.com/martj42/international_results.git
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

OUT = "output_plots"
DATA = "international_results"

import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD + FILTER TO WORLD CUP ONLY
#    goalscorers.csv and shootouts.csv have no tournament column, so we
#    inner-join on (date, home_team, away_team) against results.csv filtered
#    to FIFA World Cup matches.
# ---------------------------------------------------------------------------
results = pd.read_csv(f"{DATA}/results.csv", parse_dates=["date"])
goalscorers = pd.read_csv(f"{DATA}/goalscorers.csv", parse_dates=["date"])
shootouts = pd.read_csv(f"{DATA}/shootouts.csv", parse_dates=["date"])

wc_keys = results[results.tournament == "FIFA World Cup"][["date", "home_team", "away_team"]]

gs = goalscorers.merge(wc_keys, on=["date", "home_team", "away_team"])
so = shootouts.merge(wc_keys, on=["date", "home_team", "away_team"])

gs["year"] = gs.date.dt.year
so["year"] = so.date.dt.year

print(f"World Cup goal events: {len(gs)}")
print(f"World Cup penalty shootouts: {len(so)}")

# ===========================================================================
# GOALSCORERS ANALYSIS
# ===========================================================================

# ---------------------------------------------------------------------------
# 2. TOP ALL-TIME WORLD CUP GOALSCORERS (individuals)
# ---------------------------------------------------------------------------
scorer_goals = gs[~gs.own_goal].groupby("scorer").size().sort_values(ascending=False)
top15 = scorer_goals.head(15)

fig, ax = plt.subplots(figsize=(9, 6.5))
sns.barplot(x=top15.values, y=top15.index, color="#1B3B6F", ax=ax)
ax.set_title("Top 15 all-time World Cup goalscorers")
ax.set_xlabel("Goals"); ax.set_ylabel("")
for i, v in enumerate(top15.values):
    ax.text(v + 0.15, i, str(v), va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/eda_gs_01_top_scorers.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. GOAL TIMING — WHEN DO WORLD CUP GOALS HAPPEN?
#    Minute can exceed 90 (stoppage/extra time) -- keep raw values, bin into
#    standard match-segment buckets used by football analysts.
# ---------------------------------------------------------------------------
bins = [0, 15, 30, 45, 60, 75, 90, 105, 120, 130]
labels = ["1-15", "16-30", "31-45(+)", "46-60", "61-75", "76-90(+)", "90-105(ET)", "106-120(ET)", "120+"]
gs["segment"] = pd.cut(gs.minute, bins=bins, labels=labels, include_lowest=True)

fig, ax = plt.subplots(figsize=(10, 5))
seg_counts = gs.segment.value_counts().reindex(labels)
sns.barplot(x=seg_counts.index, y=seg_counts.values, color="#0E7A4D", ax=ax)
ax.set_title("World Cup goals by match segment")
ax.set_xlabel("Minute range"); ax.set_ylabel("Goals scored")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT}/eda_gs_02_goal_timing.png")
plt.close()

# ---------------------------------------------------------------------------
# 4. PENALTY GOALS & OWN GOALS SHARE OVER TIME
# ---------------------------------------------------------------------------
yearly = gs.groupby("year").agg(
    total=("scorer", "size"),
    penalties=("penalty", "sum"),
    own_goals=("own_goal", "sum")
)
yearly["pen_pct"] = yearly.penalties / yearly.total * 100
yearly["og_pct"] = yearly.own_goals / yearly.total * 100

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(yearly.index, yearly.pen_pct, marker="o", color="#1B3B6F", label="Penalty goals %")
ax.plot(yearly.index, yearly.og_pct, marker="s", color="#C8202F", label="Own goals %")
ax.set_title("Share of goals from penalties vs own goals, by edition")
ax.set_xlabel("Year"); ax.set_ylabel("% of all goals")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/eda_gs_03_penalty_owngoal_trend.png")
plt.close()

overall_pen_pct = gs.penalty.mean() * 100
overall_og_pct = gs.own_goal.mean() * 100

# ---------------------------------------------------------------------------
# 5. GOALS PER TEAM (attacking output, World Cup only) — cross-check vs
#    results.csv totals, using goalscorers data this time
# ---------------------------------------------------------------------------
team_goals = gs[~gs.own_goal].groupby("team").size().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6.5))
sns.barplot(x=team_goals.values, y=team_goals.index, color="#1B3B6F", ax=ax)
ax.set_title("Top 15 nations by World Cup goals scored (goalscorers.csv)")
ax.set_xlabel("Goals"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_gs_04_team_goals.png")
plt.close()

# ===========================================================================
# SHOOTOUTS ANALYSIS
# ===========================================================================

# ---------------------------------------------------------------------------
# 6. SHOOTOUTS PER EDITION + FIRST-SHOOTER ADVANTAGE
# ---------------------------------------------------------------------------
so["won_as_first_shooter"] = so.winner == so.first_shooter
first_shooter_data = so.dropna(subset=["first_shooter"])
fs_win_rate = first_shooter_data.won_as_first_shooter.mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
so_by_year = so.groupby("year").size()
sns.barplot(x=so_by_year.index, y=so_by_year.values, color="#1B3B6F", ax=axes[0])
axes[0].set_title("World Cup matches decided by penalty shootout")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Shootouts")
plt.setp(axes[0].get_xticklabels(), rotation=45)

pie_vals = [fs_win_rate, 100 - fs_win_rate]
axes[1].pie(pie_vals, labels=["Won as first shooter", "Lost as first shooter"],
            autopct="%1.1f%%", colors=["#0E7A4D", "#C8202F"],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}, startangle=90)
axes[1].set_title(f"Outcome for the team shooting first\n(n={len(first_shooter_data)} shootouts with data)")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_so_01_shootouts_and_first_shooter.png")
plt.close()

# ---------------------------------------------------------------------------
# 7. SHOOTOUT WINS BY NATION
# ---------------------------------------------------------------------------
so_wins = so.winner.value_counts().head(12)
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.barplot(x=so_wins.values, y=so_wins.index, color="#0E7A4D", ax=ax)
ax.set_title("Most World Cup penalty shootout wins, by nation")
ax.set_xlabel("Shootouts won"); ax.set_ylabel("")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f"{OUT}/eda_so_02_shootout_wins_by_nation.png")
plt.close()

# ---------------------------------------------------------------------------
# 8. NUMPY: BINOMIAL TEST -- IS THE FIRST-SHOOTER ADVANTAGE REAL OR NOISE?
# ---------------------------------------------------------------------------
n = len(first_shooter_data)
k = int(first_shooter_data.won_as_first_shooter.sum())
# standard error under a fair-coin null (p=0.5), normal approximation
se = np.sqrt(0.25 / n)
z = (k / n - 0.5) / se
print(f"\nFirst-shooter win rate: {k}/{n} = {fs_win_rate:.1f}%  (z-score vs 50% baseline: {z:.2f})")

print("\nAll goalscorer/shootout EDA figures saved to", OUT)
print("\n--- Key numbers ---")
print(f"Total WC goals analyzed: {len(gs[~gs.own_goal])}")
print(f"Top scorer: {scorer_goals.index[0]} ({scorer_goals.iloc[0]} goals)")
print(f"Penalty share of all goals: {overall_pen_pct:.1f}%")
print(f"Own-goal share of all goals: {overall_og_pct:.1f}%")
print(f"Total WC shootouts: {len(so)}")
print(f"Most shootout wins: {so_wins.index[0]} ({so_wins.iloc[0]})")