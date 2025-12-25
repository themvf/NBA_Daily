#!/usr/bin/env python3
"""
Opponent Ceiling Allowance Analytics

Tracks which teams allow high-scoring individual performances.
This is different from raw defensive rating - some teams suppress averages
but allow spikes (volatile defense), while others are consistently tight.

Key Metrics:
- % of games allowing 35+ point performances
- % of games allowing 40+ point performances
- % of games allowing 45+ point performances
- Ceiling volatility score (0-100)

Usage:
    import opponent_ceiling_analytics as oca

    ceiling_data = oca.get_team_ceiling_profile(conn, team_id, season='2025-26')
    volatility_score = ceiling_data['ceiling_volatility']
"""

import sqlite3
from typing import Dict, Optional
import pandas as pd


def get_team_ceiling_profile(
    conn: sqlite3.Connection,
    team_id: int,
    season: str = '2025-26',
    season_type: str = 'Regular Season',
    min_games: int = 5
) -> Optional[Dict]:
    """
    Get ceiling allowance profile for a team.

    Args:
        conn: Database connection
        team_id: Team ID to analyze
        season: Season (e.g., '2025-26')
        season_type: 'Regular Season', 'Playoffs', etc.
        min_games: Minimum games required for reliable data

    Returns:
        Dictionary with ceiling metrics, or None if insufficient data
    """
    query = """
        SELECT
            COUNT(*) as total_games,
            SUM(CASE WHEN points >= 35 THEN 1 ELSE 0 END) as games_35plus,
            SUM(CASE WHEN points >= 40 THEN 1 ELSE 0 END) as games_40plus,
            SUM(CASE WHEN points >= 45 THEN 1 ELSE 0 END) as games_45plus,
            MAX(points) as max_allowed,
            AVG(points) as avg_allowed,
            (AVG(points * points) - AVG(points) * AVG(points)) as variance
        FROM player_game_logs p
        WHERE p.opponent_team_id = ?
          AND p.season = ?
          AND p.season_type = ?
          AND p.points IS NOT NULL
    """

    cursor = conn.cursor()
    cursor.execute(query, (team_id, season, season_type))
    result = cursor.fetchone()

    if not result or result[0] < min_games:
        return None

    total_games = result[0]
    games_35plus = result[1] or 0
    games_40plus = result[2] or 0
    games_45plus = result[3] or 0
    max_allowed = result[4] or 0
    avg_allowed = result[5] or 0
    variance = result[6] or 0

    # Calculate percentages
    pct_35plus = (games_35plus / total_games) * 100
    pct_40plus = (games_40plus / total_games) * 100
    pct_45plus = (games_45plus / total_games) * 100

    # Calculate ceiling volatility score (0-100)
    # Factors:
    # - Higher % of 35+ games = higher score
    # - Higher % of 40+ games = higher score (weighted more)
    # - Higher % of 45+ games = higher score (weighted most)
    # - Higher variance = higher score

    volatility_score = 0.0

    # 35+ point games (baseline)
    volatility_score += pct_35plus * 0.3

    # 40+ point games (2x weight - these are DFS targets)
    volatility_score += pct_40plus * 0.6

    # 45+ point games (3x weight - nuclear outcomes)
    volatility_score += pct_45plus * 0.9

    # Variance contribution (normalized)
    # Typical variance range: 20-80, normalize to 0-20 contribution
    if variance > 0:
        variance_contribution = min(20, (variance ** 0.5) * 2)
        volatility_score += variance_contribution

    # Clamp to 0-100
    volatility_score = max(0, min(100, volatility_score))

    # Classify ceiling allowance
    if volatility_score >= 70:
        tier = "Elite Ceiling Spot"
        description = "Frequently allows explosive games"
    elif volatility_score >= 55:
        tier = "High Ceiling Spot"
        description = "Above average ceiling potential"
    elif volatility_score >= 40:
        tier = "Average Ceiling"
        description = "Moderate ceiling potential"
    elif volatility_score >= 25:
        tier = "Low Ceiling Spot"
        description = "Below average ceiling potential"
    else:
        tier = "Ceiling Suppressor"
        description = "Rarely allows big games"

    return {
        'team_id': team_id,
        'season': season,
        'total_games': total_games,
        'pct_35plus': round(pct_35plus, 1),
        'pct_40plus': round(pct_40plus, 1),
        'pct_45plus': round(pct_45plus, 1),
        'max_allowed': max_allowed,
        'avg_allowed': round(avg_allowed, 1),
        'ceiling_volatility': round(volatility_score, 1),
        'tier': tier,
        'description': description
    }


def get_all_teams_ceiling_rankings(
    conn: sqlite3.Connection,
    season: str = '2025-26',
    season_type: str = 'Regular Season',
    min_games: int = 5
) -> pd.DataFrame:
    """
    Get ceiling allowance rankings for all teams.

    Returns:
        DataFrame with teams ranked by ceiling volatility
    """
    # Get all team IDs
    teams_query = "SELECT DISTINCT team_id, team_name FROM teams ORDER BY team_name"
    teams_df = pd.read_sql_query(teams_query, conn)

    results = []
    for _, team in teams_df.iterrows():
        profile = get_team_ceiling_profile(
            conn,
            team['team_id'],
            season,
            season_type,
            min_games
        )

        if profile:
            profile['team_name'] = team['team_name']
            results.append(profile)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by ceiling volatility (descending)
    df = df.sort_values('ceiling_volatility', ascending=False)

    # Add rank
    df['rank'] = range(1, len(df) + 1)

    # Reorder columns for readability
    cols = ['rank', 'team_name', 'ceiling_volatility', 'tier',
            'pct_35plus', 'pct_40plus', 'pct_45plus',
            'avg_allowed', 'max_allowed', 'total_games']

    return df[cols]


if __name__ == "__main__":
    # Example usage
    import sys

    db_path = "nba_stats.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    conn = sqlite3.connect(db_path)

    print("=" * 70)
    print("OPPONENT CEILING ALLOWANCE RANKINGS - 2025-26 Season")
    print("=" * 70)

    rankings = get_all_teams_ceiling_rankings(conn, season='2025-26')

    if not rankings.empty:
        print("\nTop 10 Ceiling Spots (Best for DFS Tournaments):")
        print("-" * 70)
        print(rankings.head(10).to_string(index=False))

        print("\n\nBottom 10 Ceiling Suppressors (Avoid for Tournaments):")
        print("-" * 70)
        print(rankings.tail(10).to_string(index=False))
    else:
        print("\nNo data available yet for 2025-26 season")

    conn.close()
