"""
FIFA World Cup — Exploratory Data Analysis
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

import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD + FILTER TO FIFA WORLD CUP FINALS MATCHES
# ---------------------------------------------------------------------------
df = pd.read_csv("international_results/results.csv", parse_dates=["date"])
wc = df[df.tournament == "FIFA World Cup"].copy()
wc["year"] = wc.date.dt.year
wc_played = wc.dropna(subset=["home_score", "away_score"]).copy()
wc_played["home_score"] = wc_played.home_score.astype(int)
wc_played["away_score"] = wc_played.away_score.astype(int)
wc_played["total_goals"] = wc_played.home_score + wc_played.away_score
wc_played["goal_diff"] = wc_played.home_score - wc_played.away_score

def outcome(row):
    if row.home_score > row.away_score: return "Home win"
    if row.home_score < row.away_score: return "Away win"
    return "Draw"
wc_played["outcome"] = wc_played.apply(outcome, axis=1)

print(f"Total FIFA World Cup matches (1930-2026): {len(wc)}")
print(f"Played (scored) matches: {len(wc_played)}")
print(f"Editions covered: {sorted(wc_played.year.unique())}")

# ---------------------------------------------------------------------------
# 2. GOALS PER EDITION OVER TIME
# ---------------------------------------------------------------------------
by_year = wc_played.groupby("year").agg(
    matches=("total_goals", "size"),
    total_goals=("total_goals", "sum"),
    avg_goals=("total_goals", "mean")
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.lineplot(data=by_year, x="year", y="avg_goals", marker="o", ax=axes[0], color="#1B3B6F")
axes[0].set_title("Average goals per match, by World Cup edition")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Avg goals / match")
axes[0].axhline(by_year.avg_goals.mean(), ls="--", color="#C8202F", lw=1, label=f"All-time avg = {by_year.avg_goals.mean():.2f}")
axes[0].legend()

sns.barplot(data=by_year, x="year", y="matches", ax=axes[1], color="#0E7A4D")
axes[1].set_title("Matches played per edition")
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Matches")
axes[1].tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.savefig(f"{OUT}/eda_01_goals_over_time.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. OUTCOME DISTRIBUTION (home win / draw / away win)
#    Note: most WC matches are at neutral venues, so "home" is nominal
#    (whoever is listed first), not a true home-advantage signal.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
outcome_counts = wc_played.outcome.value_counts()
colors = {"Home win": "#1B3B6F", "Draw": "#9CA9B8", "Away win": "#C8202F"}
axes[0].pie(outcome_counts.values, labels=outcome_counts.index, autopct="%1.1f%%",
            colors=[colors[k] for k in outcome_counts.index], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0].set_title("Match outcome distribution")

sns.histplot(wc_played.total_goals, bins=range(0, 12), discrete=True, ax=axes[1], color="#1B3B6F")
axes[1].set_title("Distribution of total goals per match")
axes[1].set_xlabel("Total goals"); axes[1].set_ylabel("Matches")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_02_outcomes_and_goals.png")
plt.close()

# ---------------------------------------------------------------------------
# 4. TOP GOAL-SCORING NATIONS (all-time, World Cup only)
# ---------------------------------------------------------------------------
home_goals = wc_played.groupby("home_team").home_score.sum()
away_goals = wc_played.groupby("away_team").away_score.sum()
total_goals_by_team = home_goals.add(away_goals, fill_value=0).sort_values(ascending=False)

home_matches = wc_played.home_team.value_counts()
away_matches = wc_played.away_team.value_counts()
matches_by_team = home_matches.add(away_matches, fill_value=0)

top20 = total_goals_by_team.head(20)

fig, ax = plt.subplots(figsize=(9, 7))
sns.barplot(x=top20.values, y=top20.index, palette="Blues_r", ax=ax)
ax.set_title("Top 20 all-time World Cup goal-scoring nations")
ax.set_xlabel("Total goals scored"); ax.set_ylabel("")
for i, v in enumerate(top20.values):
    ax.text(v + 1, i, str(int(v)), va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/eda_03_top_scoring_nations.png")
plt.close()

# ---------------------------------------------------------------------------
# 5. WINS BY NATION (most successful teams by win count)
# ---------------------------------------------------------------------------
home_wins = wc_played[wc_played.outcome == "Home win"].home_team.value_counts()
away_wins = wc_played[wc_played.outcome == "Away win"].away_team.value_counts()
total_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)
win_rate = (total_wins / matches_by_team).fillna(0)
summary = pd.DataFrame({"matches": matches_by_team, "wins": total_wins.reindex(matches_by_team.index).fillna(0),
                         "win_rate": win_rate.reindex(matches_by_team.index).fillna(0)})
qualified = summary[summary.matches >= 15].sort_values("win_rate", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=qualified.win_rate * 100, y=qualified.index, palette="Greens_r", ax=ax)
ax.set_title("Highest World Cup win rate (min. 15 matches played)")
ax.set_xlabel("Win rate (%)"); ax.set_ylabel("")
for i, v in enumerate(qualified.win_rate * 100):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/eda_04_win_rate_leaders.png")
plt.close()

# ---------------------------------------------------------------------------
# 6. GOAL DIFFERENCE DISTRIBUTION + BLOWOUTS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(wc_played.goal_diff, bins=range(-10, 11), discrete=True, ax=axes[0], color="#C8202F")
axes[0].set_title("Goal difference distribution (home - away)")
axes[0].set_xlabel("Goal difference"); axes[0].set_ylabel("Matches")
axes[0].axvline(0, color="black", lw=1)

blowouts = wc_played.reindex(wc_played.total_goals.sort_values(ascending=False).index).head(10)
blow_labels = blowouts.home_team + " " + blowouts.home_score.astype(str) + "-" + \
               blowouts.away_score.astype(str) + " " + blowouts.away_team + " (" + blowouts.year.astype(str) + ")"
sns.barplot(x=blowouts.total_goals.values, y=blow_labels.values, palette="Reds_r", ax=axes[1])
axes[1].set_title("Highest-scoring World Cup matches")
axes[1].set_xlabel("Total goals")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_05_goal_diff_and_blowouts.png")
plt.close()

# ---------------------------------------------------------------------------
# 7. HOST NATION PERFORMANCE (host advantage)
# ---------------------------------------------------------------------------
hosts = wc_played[wc_played.neutral == False].copy()
host_summary = hosts.groupby("home_team").agg(
    matches=("outcome", "size"),
    wins=("outcome", lambda s: (s == "Home win").sum())
)
host_summary["win_rate"] = host_summary.wins / host_summary.matches
host_summary = host_summary.sort_values("win_rate", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=host_summary.win_rate * 100, y=host_summary.index, palette="Blues_r", ax=ax)
ax.set_title("Win rate when playing as host nation (non-neutral matches)")
ax.set_xlabel("Win rate (%)"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_06_host_advantage.png")
plt.close()

# ---------------------------------------------------------------------------
# 8. HEAD-TO-HEAD HEATMAP — TOP 10 NATIONS
# ---------------------------------------------------------------------------
top10_teams = total_goals_by_team.head(10).index.tolist()
h2h = pd.DataFrame(0, index=top10_teams, columns=top10_teams, dtype=int)
for row in wc_played.itertuples():
    if row.home_team in top10_teams and row.away_team in top10_teams:
        if row.home_score > row.away_score: h2h.loc[row.home_team, row.away_team] += 1
        elif row.away_score > row.home_score: h2h.loc[row.away_team, row.home_team] += 1

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(h2h, annot=True, fmt="d", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Wins (row over column)"})
ax.set_title("Head-to-head wins among top 10 World Cup nations")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_07_h2h_heatmap.png")
plt.close()

# ---------------------------------------------------------------------------
# 9. NUMPY-BASED CORRELATION: goals vs match era (decade)
# ---------------------------------------------------------------------------
wc_played["decade"] = (wc_played.year // 10) * 10
decade_avg = wc_played.groupby("decade").total_goals.mean()
years_arr = np.array(decade_avg.index, dtype=float)
goals_arr = np.array(decade_avg.values, dtype=float)
corr = np.corrcoef(years_arr, goals_arr)[0, 1]
slope, intercept = np.polyfit(years_arr, goals_arr, 1)

fig, ax = plt.subplots(figsize=(9, 5))
sns.regplot(x=years_arr, y=goals_arr, ax=ax, color="#1B3B6F",
            line_kws={"color": "#C8202F"}, scatter_kws={"s": 60})
ax.set_title(f"Avg goals/match by decade  (corr={corr:.2f}, slope={slope:.3f} goals/decade)")
ax.set_xlabel("Decade"); ax.set_ylabel("Avg goals per match")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_08_decade_trend.png")
plt.close()

print("\nAll EDA figures saved to", OUT)
print("\n--- Key numbers ---")
print(f"Overall average goals/match: {wc_played.total_goals.mean():.2f}")
print(f"Draw rate: {(wc_played.outcome=='Draw').mean()*100:.1f}%")
print(f"Goals/match correlation with decade: {corr:.2f}")
print(f"Top scorer nation: {total_goals_by_team.index[0]} ({total_goals_by_team.iloc[0]:.0f} goals)")
print(f"Best win-rate nation (min 15 matches): {qualified.index[0]} ({qualified.win_rate.iloc[0]*100:.1f}%)")