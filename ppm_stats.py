#!/usr/bin/env python3
"""
PPM (Points Per Minute) Statistics Module

Calculates offensive and defensive PPM stats for teams and players.
Used by prediction_tracker.py and streamlit_app.py for improved projections.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_game_ppm(conn, game_id, team_id):
    """Calculate team's PPM for a specific game."""
    query = """
        SELECT
            SUM(points) as total_points,
            SUM(CAST(minutes AS REAL)) as total_minutes
        FROM player_game_logs
        WHERE game_id = ?
          AND team_id = ?
    """

    cursor = conn.cursor()
    cursor.execute(query, (game_id, team_id))
    result = cursor.fetchone()

    if result and result[0] and result[1] and result[1] > 0:
        return result[0] / result[1]
    return None


def get_team_offensive_ppm_stats(conn, season='2025-26') -> pd.DataFrame:
    """
    Calculate offensive PPM stats for all teams.

    Returns DataFrame with:
    - team_id, team_name
    - avg_off_ppm: Average offensive PPM
    - std_off_ppm: Standard deviation
    - p90_off_ppm: 90th percentile (ceiling games)
    - ceiling_factor: P90 / Avg
    - off_ppm_grade: A+ to D- grade
    """

    # Get all games for each team (as offense)
    query = """
        SELECT
            tgl.team_id,
            t.full_name as team_name,
            tgl.game_id,
            tgl.pts as pts_scored
        FROM team_game_logs tgl
        JOIN teams t ON tgl.team_id = t.team_id
        WHERE tgl.season = ?
        ORDER BY tgl.team_id, tgl.game_date
    """

    df = pd.read_sql_query(query, conn, params=[season])

    # Calculate PPM for each game
    ppm_data = []
    for idx, row in df.iterrows():
        game_ppm = calculate_game_ppm(conn, row['game_id'], row['team_id'])
        if game_ppm is not None:
            ppm_data.append({
                'team_id': row['team_id'],
                'team_name': row['team_name'],
                'game_id': row['game_id'],
                'offensive_ppm': game_ppm,
                'pts_scored': row['pts_scored']
            })

    ppm_df = pd.DataFrame(ppm_data)

    # Aggregate by team
    team_stats = ppm_df.groupby(['team_id', 'team_name']).agg(
        games=('game_id', 'count'),
        avg_off_ppm=('offensive_ppm', 'mean'),
        std_off_ppm=('offensive_ppm', 'std'),
        min_off_ppm=('offensive_ppm', 'min'),
        max_off_ppm=('offensive_ppm', 'max'),
        p90_off_ppm=('offensive_ppm', lambda x: x.quantile(0.90)),
        p10_off_ppm=('offensive_ppm', lambda x: x.quantile(0.10)),
        avg_pts_scored=('pts_scored', 'mean')
    ).reset_index()

    # Calculate ceiling factor
    team_stats['ceiling_factor'] = team_stats['p90_off_ppm'] / team_stats['avg_off_ppm']

    # Calculate coefficient of variation
    team_stats['cv_off_ppm'] = team_stats['std_off_ppm'] / team_stats['avg_off_ppm']

    # Assign grades (league avg ~0.483)
    def get_off_ppm_grade(ppm):
        if pd.isna(ppm):
            return 'N/A'
        elif ppm >= 0.515:  # Elite offense
            return 'A+'
        elif ppm >= 0.500:  # Excellent
            return 'A'
        elif ppm >= 0.485:  # Good (above average)
            return 'B'
        elif ppm >= 0.470:  # Average
            return 'C'
        elif ppm >= 0.455:  # Poor
            return 'D'
        else:  # Very poor
            return 'D-'

    team_stats['off_ppm_grade'] = team_stats['avg_off_ppm'].apply(get_off_ppm_grade)

    return team_stats


def get_team_defensive_ppm_stats(conn, season='2025-26') -> pd.DataFrame:
    """
    Calculate defensive PPM stats for all teams.

    Returns DataFrame with:
    - team_id, team_name
    - avg_def_ppm: Average defensive PPM allowed
    - std_def_ppm: Standard deviation
    - p90_def_ppm: 90th percentile (worst defensive games)
    - ceiling_factor: P90 / Avg (high = allows blowups)
    - def_ppm_grade: A+ to D- grade
    """

    # Get all games for each team (as defender)
    query = """
        SELECT
            tgl.opp_team_id as def_team_id,
            t.full_name as def_team_name,
            tgl.game_id,
            tgl.team_id as off_team_id,
            tgl.pts as pts_allowed
        FROM team_game_logs tgl
        JOIN teams t ON tgl.opp_team_id = t.team_id
        WHERE tgl.season = ?
        ORDER BY tgl.opp_team_id, tgl.game_date
    """

    df = pd.read_sql_query(query, conn, params=[season])

    # Calculate opponent PPM for each game
    ppm_data = []
    for idx, row in df.iterrows():
        game_ppm = calculate_game_ppm(conn, row['game_id'], row['off_team_id'])
        if game_ppm is not None:
            ppm_data.append({
                'def_team_id': row['def_team_id'],
                'def_team_name': row['def_team_name'],
                'game_id': row['game_id'],
                'opponent_ppm': game_ppm,
                'pts_allowed': row['pts_allowed']
            })

    ppm_df = pd.DataFrame(ppm_data)

    # Aggregate by team
    team_stats = ppm_df.groupby(['def_team_id', 'def_team_name']).agg(
        games=('game_id', 'count'),
        avg_def_ppm=('opponent_ppm', 'mean'),
        std_def_ppm=('opponent_ppm', 'std'),
        min_def_ppm=('opponent_ppm', 'min'),
        max_def_ppm=('opponent_ppm', 'max'),
        p90_def_ppm=('opponent_ppm', lambda x: x.quantile(0.90)),
        p10_def_ppm=('opponent_ppm', lambda x: x.quantile(0.10)),
        avg_pts_allowed=('pts_allowed', 'mean')
    ).reset_index()

    # Calculate ceiling factor
    team_stats['ceiling_factor'] = team_stats['p90_def_ppm'] / team_stats['avg_def_ppm']

    # Calculate coefficient of variation
    team_stats['cv_def_ppm'] = team_stats['std_def_ppm'] / team_stats['avg_def_ppm']

    # Assign grades (lower is better for defense)
    def get_def_ppm_grade(ppm):
        if pd.isna(ppm):
            return 'N/A'
        elif ppm < 0.455:  # Elite defense
            return 'A+'
        elif ppm < 0.470:  # Excellent
            return 'A'
        elif ppm < 0.485:  # Good (above average)
            return 'B'
        elif ppm < 0.500:  # Average
            return 'C'
        elif ppm < 0.515:  # Poor
            return 'D'
        else:  # Very poor (target for DFS)
            return 'D-'

    team_stats['def_ppm_grade'] = team_stats['avg_def_ppm'].apply(get_def_ppm_grade)

    # Rename columns for clarity
    team_stats = team_stats.rename(columns={
        'def_team_id': 'team_id',
        'def_team_name': 'team_name'
    })

    return team_stats


def get_player_ppm_stats(conn, player_name, season='2025-26', min_minutes=10) -> Optional[Dict]:
    """
    Calculate PPM consistency stats for a specific player.

    Returns:
    - avg_ppm: Average points per minute
    - std_ppm: Standard deviation
    - cv_ppm: Coefficient of variation
    - consistency_score: 0-100 score (higher = more consistent)
    - ppm_grade: A+ to D grade
    - games_counted: Number of games with min_minutes+
    """

    query = """
        SELECT
            points / CAST(minutes AS REAL) as ppm
        FROM player_game_logs
        WHERE player_name = ?
          AND season = ?
          AND CAST(minutes AS REAL) >= ?
    """

    df = pd.read_sql_query(query, conn, params=[player_name, season, min_minutes])

    if len(df) < 5:
        return None

    avg_ppm = df['ppm'].mean()
    std_ppm = df['ppm'].std()
    cv_ppm = std_ppm / avg_ppm if avg_ppm > 0 else 999
    consistency_score = 100 * (1 - min(cv_ppm, 1.0))

    # Assign grade
    if consistency_score >= 85:
        grade = 'A+'
    elif consistency_score >= 75:
        grade = 'A'
    elif consistency_score >= 65:
        grade = 'B'
    elif consistency_score >= 55:
        grade = 'C'
    else:
        grade = 'D'

    return {
        'avg_ppm': avg_ppm,
        'std_ppm': std_ppm,
        'cv_ppm': cv_ppm,
        'consistency_score': consistency_score,
        'ppm_grade': grade,
        'games_counted': len(df)
    }


def get_league_avg_ppm(conn, season='2025-26') -> float:
    """Get league average PPM for the season."""
    query = """
        SELECT AVG(points / CAST(minutes AS REAL)) as league_avg_ppm
        FROM player_game_logs
        WHERE season = ?
          AND CAST(minutes AS REAL) >= 10
    """

    result = pd.read_sql_query(query, conn, params=[season])
    return result['league_avg_ppm'].iloc[0] if not result.empty else 0.483


if __name__ == '__main__':
    # Test the module
    conn = sqlite3.connect('nba_stats.db')

    print("Testing PPM Stats Module...")
    print("\n1. Offensive PPM Stats (top 5):")
    off_stats = get_team_offensive_ppm_stats(conn)
    print(off_stats.nlargest(5, 'avg_off_ppm')[['team_name', 'avg_off_ppm', 'off_ppm_grade']])

    print("\n2. Defensive PPM Stats (top 5 - best defenses):")
    def_stats = get_team_defensive_ppm_stats(conn)
    print(def_stats.nsmallest(5, 'avg_def_ppm')[['team_name', 'avg_def_ppm', 'def_ppm_grade']])

    print("\n3. Player PPM Stats (sample):")
    player_stats = get_player_ppm_stats(conn, 'Jaylen Brown')
    if player_stats:
        print(f"Jaylen Brown: {player_stats['avg_ppm']:.3f} PPM, Grade: {player_stats['ppm_grade']}")

    print(f"\n4. League Average PPM: {get_league_avg_ppm(conn):.3f}")

    conn.close()
    print("\n✓ PPM Stats Module test complete!")
