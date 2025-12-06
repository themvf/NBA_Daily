#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the tournament GPP Score calculation with actual prediction data.
"""

import sqlite3
import pandas as pd
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add streamlit_app functions (we'll import the logic)
sys.path.insert(0, r'C:\Docs\_AI Python Projects\NBA_Daily')

conn = sqlite3.connect('nba_stats.db')

# Get most recent predictions
query = """
SELECT
    player_name,
    team_name,
    opponent_name,
    opponent_id,
    projected_ppg,
    proj_ceiling,
    proj_floor,
    season_avg_ppg,
    recent_avg_5,
    opponent_def_rating
FROM predictions
WHERE game_date = (SELECT MAX(game_date) FROM predictions)
  AND proj_ceiling >= 30
ORDER BY proj_ceiling DESC
LIMIT 15
"""

df = pd.read_sql_query(query, conn)

print("\n" + "="*120)
print("TOURNAMENT GPP SCORE TEST")
print("="*120)
print(f"\nDate: {pd.read_sql_query('SELECT MAX(game_date) FROM predictions', conn).iloc[0, 0]}")
print(f"Total predictions with ceiling >= 30: {len(df)}")
print("\n" + "="*120)

# Import the tournament scoring function from streamlit_app
# We'll manually implement it here to avoid import issues
def get_opponent_defense_ceiling_factor(opp_team_id: int, conn) -> float:
    """Calculate opponent's defense variance ceiling factor."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pts
        FROM team_game_logs
        WHERE opp_team_id = ? AND season = '2024-25'
    """, (opp_team_id,))

    pts_against = [row[0] for row in cursor.fetchall() if row[0] is not None]

    if len(pts_against) < 10:
        return 1.0

    import numpy as np
    avg_allowed = np.mean(pts_against)
    p90_allowed = np.percentile(pts_against, 90)
    ceiling_factor = p90_allowed / avg_allowed if avg_allowed > 0 else 1.0

    return ceiling_factor


def calculate_tournament_gpp_score(row, conn):
    """Calculate GPP Score for a player."""

    # Component 1: Ceiling Base Score (0-50 points)
    ceiling = row['proj_ceiling']
    if ceiling >= 50:
        ceiling_base = 50
    elif ceiling >= 45:
        ceiling_base = 46
    elif ceiling >= 40:
        ceiling_base = 42
    elif ceiling >= 37:
        ceiling_base = 38
    elif ceiling >= 35:
        ceiling_base = 35
    elif ceiling >= 32:
        ceiling_base = 30
    elif ceiling >= 30:
        ceiling_base = 25
    else:
        ceiling_base = 20

    # Component 2: Hot Streak Bonus (0-20 points)
    hot_streak_bonus = 0
    if pd.notna(row['recent_avg_5']) and row['season_avg_ppg'] > 0:
        hot_streak_ratio = row['recent_avg_5'] / row['season_avg_ppg']

        if hot_streak_ratio >= 1.25:
            hot_streak_bonus = 20
        elif hot_streak_ratio >= 1.15:
            hot_streak_bonus = 15
        elif hot_streak_ratio >= 1.10:
            hot_streak_bonus = 12
        elif hot_streak_ratio >= 1.05:
            hot_streak_bonus = 10
        elif hot_streak_ratio >= 0.95:
            hot_streak_bonus = 5
        else:
            hot_streak_bonus = 0  # Cold streak

    # Component 3: Defense Variance Bonus (0-15 points)
    variance_bonus = 0
    if pd.notna(row['opponent_id']):
        ceiling_factor = get_opponent_defense_ceiling_factor(int(row['opponent_id']), conn)

        if ceiling_factor >= 1.15:
            variance_bonus = 15
        elif ceiling_factor >= 1.12:
            variance_bonus = 10
        elif ceiling_factor >= 1.10:
            variance_bonus = 6
        elif ceiling_factor >= 1.08:
            variance_bonus = 3

    # Component 4: Defense Quality (±5 points)
    defense_adjustment = 0
    if pd.notna(row['opponent_def_rating']):
        league_avg = 112.0
        def_diff = row['opponent_def_rating'] - league_avg

        if def_diff >= 6:
            defense_adjustment = 5
        elif def_diff >= 3:
            defense_adjustment = 3
        elif def_diff <= -6:
            defense_adjustment = -5
        elif def_diff <= -3:
            defense_adjustment = -3

    # Total GPP Score
    gpp_score = ceiling_base + hot_streak_bonus + variance_bonus + defense_adjustment

    # Grade
    if gpp_score >= 85:
        grade = "GPP Lock"
    elif gpp_score >= 75:
        grade = "Core Play"
    elif gpp_score >= 65:
        grade = "Strong"
    elif gpp_score >= 55:
        grade = "Playable"
    else:
        grade = "Punt/Fade"

    return {
        'gpp_score': gpp_score,
        'grade': grade,
        'ceiling_base': ceiling_base,
        'hot_streak': hot_streak_bonus,
        'variance': variance_bonus,
        'defense': defense_adjustment
    }


# Calculate GPP scores
print("\nCalculating GPP Scores...\n")

results = []
for idx, row in df.iterrows():
    score_data = calculate_tournament_gpp_score(row, conn)
    results.append({
        'Player': row['player_name'],
        'Team': row['team_name'],
        'vs': row['opponent_name'],
        'Ceiling': row['proj_ceiling'],
        'L5': row['recent_avg_5'] if pd.notna(row['recent_avg_5']) else 0,
        'Proj': row['projected_ppg'],
        'GPP Score': score_data['gpp_score'],
        'Grade': score_data['grade'],
        'Ceil': score_data['ceiling_base'],
        'Hot': score_data['hot_streak'],
        'Var': score_data['variance'],
        'Def': score_data['defense']
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('GPP Score', ascending=False)

# Display results
print(f"{'Player':<25} {'Team':<6} {'vs':<6} {'Ceil':>5} {'L5':>5} {'Proj':>5} {'GPP':>5} {'Grade':<18} | Breakdown")
print("="*120)

for idx, row in results_df.iterrows():
    print(f"{row['Player']:<25} {row['Team']:<6} {row['vs']:<6} "
          f"{row['Ceiling']:5.1f} {row['L5']:5.1f} {row['Proj']:5.1f} "
          f"{row['GPP Score']:5.0f} {row['Grade']:<18} | "
          f"C:{row['Ceil']:>2} + H:{row['Hot']:>2} + V:{row['Var']:>2} + D:{row['Def']:>+2}")

print("="*120)

# Summary stats
print(f"\n📊 GPP Score Summary:")
print(f"  - Avg GPP Score: {results_df['GPP Score'].mean():.1f}")
print(f"  - Max GPP Score: {results_df['GPP Score'].max():.1f}")
print(f"  - Min GPP Score: {results_df['GPP Score'].min():.1f}")
print(f"  - GPP Lock (85+): {(results_df['GPP Score'] >= 85).sum()} players")
print(f"  - Core Play (75+): {(results_df['GPP Score'] >= 75).sum()} players")
print(f"  - Strong (65+): {(results_df['GPP Score'] >= 65).sum()} players")

# Component breakdown
print(f"\n📈 Component Breakdown:")
print(f"  - Avg Ceiling Base: {results_df['Ceil'].mean():.1f}")
print(f"  - Avg Hot Streak: {results_df['Hot'].mean():.1f}")
print(f"  - Avg Variance Bonus: {results_df['Var'].mean():.1f}")
print(f"  - Avg Defense Adj: {results_df['Def'].mean():.1f}")

conn.close()

print("\n✅ GPP Score test complete!\n")
