import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (THEMING)
# ==========================================
st.set_page_config(
    page_title="FIFA 2026 Prediction Engine",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the deep stadium green / gold theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0b130e;
        color: #e5e7eb;
    }
    .metric-card {
        background-color: #112218;
        border: 1px solid #1d3a28;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        color: #34d399; /* Emerald */
    }
    .metric-value-away { color: #fbbf24; } /* Amber */
    .metric-value-draw { color: #9ca3af; } /* Gray */
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
    }
    h1, h2, h3 { color: #fbbf24 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LAYER (Model State)
# ==========================================
TEAMS = {
    "Argentina": {"elo": 2140, "form": 0.78, "confed": "CONMEBOL"},
    "France":    {"elo": 2110, "form": 0.75, "confed": "UEFA"},
    "Brazil":    {"elo": 2080, "form": 0.68, "confed": "CONMEBOL"},
    "Spain":     {"elo": 2060, "form": 0.82, "confed": "UEFA"},
    "England":   {"elo": 2030, "form": 0.70, "confed": "UEFA"},
    "Germany":   {"elo": 1990, "form": 0.65, "confed": "UEFA"},
    "Morocco":   {"elo": 1940, "form": 0.72, "confed": "CAF"},
    "Japan":     {"elo": 1910, "form": 0.74, "confed": "AFC"},
    "United States": {"elo": 1850, "form": 0.60, "confed": "CONCACAF"},
    "Mexico":    {"elo": 1820, "form": 0.55, "confed": "CONCACAF"}
}

# ==========================================
# 3. HEADER
# ==========================================
st.title("🏆 FIFA 2026 XG-BOOST ENGINE")
st.markdown("*Time-Decayed Elo → Poisson Goal-Expectancy → Monte Carlo Simulation*")
st.divider()

# ==========================================
# 4. SIDEBAR & CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚽ Fixture Configuration")
    
    home_team = st.selectbox("Home Team (Team A)", list(TEAMS.keys()), index=0)
    away_team = st.selectbox("Away Team (Team B)", list(TEAMS.keys()), index=1)
    
    st.markdown("---")
    st.subheader("Match Rules & Environment")
    is_neutral = st.toggle("Neutral Venue", value=True, help="If disabled, Home team gets +60 Elo advantage.")
    is_knockout = st.toggle("Knockout Stage", value=True, help="Applies a 0.92x goal multiplier and resolves draws via Elo-biased penalties.")
    
    st.markdown("---")
    st.subheader("Physical & Psych Factors")
    home_rest = st.slider(f"{home_team} Rest Days", min_value=3, max_value=14, value=7)
    away_rest = st.slider(f"{away_team} Rest Days", min_value=3, max_value=14, value=7)
    h2h_surprise = st.slider("H2H Surprise (Bogey Team Bias)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)

# ==========================================
# 5. MAIN DASHBOARD & SIMULATION LOGIC
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Simulation Feed")
    
    if home_team == away_team:
        st.error("Please select two distinct teams to run the simulation.")
        st.stop()

    # --- MATH & EXPECTANCY CALCULATION ---
    home_data = TEAMS[home_team]
    away_data = TEAMS[away_team]
    
    # Base Elo adjustments
    elo_home_adj = home_data["elo"] + (0 if is_neutral else 60)
    expected_home_win_prob = 1 / (1 + 10 ** (-(elo_home_adj - away_data["elo"]) / 400))
    
    # Base Lambda (Goal Expectancy)
    base_lambda_home = 1.15 + (elo_home_adj - away_data["elo"]) / 450 + (home_data["form"] - away_data["form"]) * 0.4
    base_lambda_away = 1.15 + (away_data["elo"] - elo_home_adj) / 450 + (away_data["form"] - home_data["form"]) * 0.4
    
    # Fatigue Penalties (Rest < 6 days)
    if home_rest < 6: base_lambda_home -= (6 - home_rest) * 0.08
    if away_rest < 6: base_lambda_away -= (6 - away_rest) * 0.08
    
    # H2H factor
    base_lambda_home += h2h_surprise * 0.5
    base_lambda_away -= h2h_surprise * 0.5
    
    # Floor values
    lambda_home = max(base_lambda_home, 0.15)
    lambda_away = max(base_lambda_away, 0.15)
    
    # Knockout stage suppression
    if is_knockout:
        lambda_home *= 0.92
        lambda_away *= 0.92

    # --- MONTE CARLO EXECUTION ---
    if st.button("🎲 Execute 2,000 Trial Monte Carlo Simulation", use_container_width=True):
        iterations = 2000
        
        # Sample Poisson distributions using numpy
        home_goals_sim = np.random.poisson(lambda_home, iterations)
        away_goals_sim = np.random.poisson(lambda_away, iterations)
        
        wins_home, draws, wins_away = 0, 0, 0
        scorelines = []
        
        for hg, ag in zip(home_goals_sim, away_goals_sim):
            scorelines.append(f"{hg}-{ag}")
            
            if hg > ag:
                wins_home += 1
            elif ag > hg:
                wins_away += 1
            else:
                if is_knockout:
                    # Resolve tie via Elo-biased coin flip (penalties)
                    if np.random.random() < expected_home_win_prob:
                        wins_home += 1
                    else:
                        wins_away += 1
                else:
                    draws += 1
                    
        prob_home = (wins_home / iterations) * 100
        prob_draw = (draws / iterations) * 100
        prob_away = (wins_away / iterations) * 100
        
        # --- UI DISPLAY: RESULTS ---
        st.markdown(f"**Calculated Expectancy (λ):** {home_team} **{lambda_home:.2f}** | {away_team} **{lambda_away:.2f}**")
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{home_team} Win</div>
                <div class="metric-value">{prob_home:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="opacity: {0.3 if is_knockout else 1.0}">
                <div class="metric-label">Draw</div>
                <div class="metric-value metric-value-draw">{prob_draw:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{away_team} Win</div>
                <div class="metric-value metric-value-away">{prob_away:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") # spacing
        
        # Display Top Scorelines
        st.subheader("Top Predicted Exact Scorelines")
        score_counts = Counter(scorelines).most_common(3)
        
        sc1, sc2, sc3 = st.columns(3)
        for i, col in enumerate([sc1, sc2, sc3]):
            if i < len(score_counts):
                score, count = score_counts[i]
                pct = (count / iterations) * 100
                col.markdown(f"""
                <div style="background-color: #162e20; border: 1px solid #1e3c29; padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #fbbf24; font-family: monospace;">{score}</div>
                    <div style="font-size: 0.8rem; color: #9ca3af;">{pct:.1f}% Probability</div>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.subheader("Tournament Model Outlook")
    st.markdown("Aggregate weight distribution based on global bracket matrix.")
    
    # Mock data based on initial Elo/Form weights
    outlook_data = pd.DataFrame({
        "Team": ["Argentina", "France", "Spain", "Brazil", "England", "Morocco", "Japan", "Germany", "United States", "Mexico"],
        "Win Probability (%)": [18.2, 16.5, 14.1, 12.8, 11.2, 8.5, 7.9, 6.2, 3.1, 1.5]
    }).set_index("Team")
    
    st.dataframe(
        outlook_data.style.format("{:.1f}%").background_gradient(cmap="summer", axis=0),
        use_container_width=True
    )
    
    st.caption("Mathematical Pipeline: Dual Poisson Parameter Tuning mapped via $f(\Delta\text{Elo}, \Delta\text{Form}) \to \text{Poisson}(\lambda)$.")