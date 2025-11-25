#!/usr/bin/env python3
"""Defense type analytics for NBA_Daily.

Analyzes how players perform against different types of defenses:
- Fast pace vs Slow pace teams
- Strong defense vs Weak defense teams
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import sqlite3


@dataclass
class DefenseTypePerformance:
    """Stats showing how a player performs against a specific defense type."""
    defense_type: str  # e.g., "Fast Pace", "Slow Pace", "Elite Defense", "Weak Defense"
    games_played: int
    avg_pts: float
    avg_fg3m: float
    avg_usg: float
    avg_minutes: float

    # Comparison to player's overall average
    pts_vs_average: float  # Delta from season average
    fg3m_vs_average: float

    # Teams in this category
    sample_teams: List[str]  # Up to 3 sample team names


def categorize_teams_by_defense(
    conn: sqlite3.Connection,
    season: str,
    season_type: str
) -> Dict[int, Dict[str, any]]:
    """
    Categorize all teams by their defensive characteristics.

    Returns:
        Dict mapping team_id to defense info:
        {
            team_id: {
                'team_name': str,
                'def_rating': float,  # Points allowed per game
                'pace': float,  # Estimated possessions per game
                'pace_category': 'Fast' | 'Slow' | 'Average',
                'defense_category': 'Elite' | 'Strong' | 'Average' | 'Weak'
            }
        }
    """
    # Calculate defensive rating (points allowed per game) and pace for each team
    query = """
    WITH team_stats AS (
        SELECT
            team_id,
            MAX(team_name) as team_name,
            COUNT(*) as games,
            AVG(CAST(opp_pts AS FLOAT)) as avg_pts_allowed,
            -- Estimate pace using possessions formula
            -- Poss ≈ FGA + 0.44*FTA - OREB + TOV
            AVG(
                COALESCE(CAST(fga AS FLOAT), 0.0) +
                0.44 * COALESCE(CAST(fta AS FLOAT), 0.0) -
                COALESCE(CAST(oreb AS FLOAT), 0.0) +
                COALESCE(CAST(tov AS FLOAT), 0.0)
            ) as team_pace,
            AVG(
                COALESCE(CAST(opp_fga AS FLOAT), 0.0) +
                0.44 * COALESCE(CAST(opp_fta AS FLOAT), 0.0) -
                COALESCE(CAST(opp_oreb AS FLOAT), 0.0) +
                COALESCE(CAST(opp_tov AS FLOAT), 0.0)
            ) as opp_pace
        FROM team_game_logs
        WHERE season = ?
          AND season_type = ?
          AND opp_pts IS NOT NULL
        GROUP BY team_id
        HAVING COUNT(*) >= 5
    )
    SELECT
        team_id,
        team_name,
        avg_pts_allowed as def_rating,
        (team_pace + opp_pace) / 2.0 as pace
    FROM team_stats
    WHERE avg_pts_allowed IS NOT NULL
      AND team_pace IS NOT NULL
      AND opp_pace IS NOT NULL
    """

    df = pd.read_sql_query(query, conn, params=[season, season_type])

    if df.empty:
        return {}

    # Filter out rows with NULL values
    df = df.dropna(subset=['def_rating', 'pace'])

    if df.empty:
        return {}

    # Calculate percentiles for categorization
    def_rating_33 = df['def_rating'].quantile(0.33)
    def_rating_67 = df['def_rating'].quantile(0.67)
    pace_33 = df['pace'].quantile(0.33)
    pace_67 = df['pace'].quantile(0.67)

    result = {}

    for _, row in df.iterrows():
        team_id = int(row['team_id'])
        def_rating = row['def_rating']
        pace = row['pace']

        # Skip if still somehow NULL (defensive programming)
        if pd.isna(def_rating) or pd.isna(pace):
            continue

        # Categorize pace (higher = faster)
        if pace >= pace_67:
            pace_cat = 'Fast'
        elif pace <= pace_33:
            pace_cat = 'Slow'
        else:
            pace_cat = 'Average'

        # Categorize defense (lower points allowed = better defense)
        if def_rating <= def_rating_33:
            def_cat = 'Elite'
        elif def_rating <= def_rating_67:
            def_cat = 'Strong'
        else:
            def_cat = 'Weak'

        result[team_id] = {
            'team_name': row['team_name'],
            'def_rating': def_rating,
            'pace': pace,
            'pace_category': pace_cat,
            'defense_category': def_cat
        }

    return result


def calculate_defense_type_splits(
    conn: sqlite3.Connection,
    player_id: int,
    season: str,
    season_type: str,
    min_games: int = 2
) -> List[DefenseTypePerformance]:
    """
    Calculate how a player performs against different defense types.

    Args:
        conn: Database connection
        player_id: The player to analyze
        season: Season to analyze (e.g., '2025-26')
        season_type: 'Regular Season' or 'Playoffs'
        min_games: Minimum games against a defense type to include

    Returns:
        List of DefenseTypePerformance objects for each defense type
    """
    # Get team defense categorizations
    team_defense_map = categorize_teams_by_defense(conn, season, season_type)

    if not team_defense_map:
        return []

    # Get all player game logs
    player_games_query = """
    SELECT
        game_id,
        matchup,
        points,
        fg3m,
        usg_pct,
        minutes
    FROM player_game_logs
    WHERE player_id = ?
      AND season = ?
      AND season_type = ?
    """

    player_games = pd.read_sql_query(
        player_games_query,
        conn,
        params=[player_id, season, season_type]
    )

    if player_games.empty:
        return []

    # Convert minutes from 'MM:SS' format to decimal
    player_games['minutes_decimal'] = player_games['minutes'].apply(_parse_minutes)

    # Calculate player's overall averages
    season_avg_pts = player_games['points'].mean()
    season_avg_fg3m = player_games['fg3m'].mean()

    # Extract opponent team from matchup field
    player_games['opponent_abbr'] = player_games['matchup'].apply(_extract_opponent)

    # Get team abbreviation to ID mapping
    abbr_to_id = {}
    for team_id, info in team_defense_map.items():
        # Try to find abbreviation from team_game_logs
        cursor = conn.cursor()
        cursor.execute("""
            SELECT team_abbreviation FROM team_game_logs
            WHERE team_id = ? LIMIT 1
        """, [team_id])
        result = cursor.fetchone()
        if result and result[0]:
            abbr_to_id[result[0]] = team_id

    # Map each game to defense categories
    def get_defense_info(opponent_abbr):
        team_id = abbr_to_id.get(opponent_abbr)
        if team_id and team_id in team_defense_map:
            return team_defense_map[team_id]
        return None

    player_games['defense_info'] = player_games['opponent_abbr'].apply(get_defense_info)

    # Group by defense types and calculate stats
    results = []

    # Pace-based splits
    for pace_cat in ['Fast', 'Slow']:
        games_vs_pace = player_games[
            player_games['defense_info'].apply(
                lambda x: x is not None and x['pace_category'] == pace_cat
            )
        ]

        if len(games_vs_pace) >= min_games:
            avg_pts = games_vs_pace['points'].mean()
            avg_fg3m = games_vs_pace['fg3m'].mean()
            avg_usg = games_vs_pace['usg_pct'].mean()
            avg_minutes = games_vs_pace['minutes_decimal'].mean()

            # Get sample teams
            sample_teams = list(games_vs_pace['defense_info'].apply(
                lambda x: x['team_name'] if x else None
            ).dropna().unique()[:3])

            results.append(DefenseTypePerformance(
                defense_type=f"{pace_cat} Pace",
                games_played=len(games_vs_pace),
                avg_pts=avg_pts,
                avg_fg3m=avg_fg3m,
                avg_usg=avg_usg,
                avg_minutes=avg_minutes,
                pts_vs_average=avg_pts - season_avg_pts,
                fg3m_vs_average=avg_fg3m - season_avg_fg3m,
                sample_teams=sample_teams
            ))

    # Defense quality splits
    for def_cat in ['Elite', 'Weak']:
        games_vs_def = player_games[
            player_games['defense_info'].apply(
                lambda x: x is not None and x['defense_category'] == def_cat
            )
        ]

        if len(games_vs_def) >= min_games:
            avg_pts = games_vs_def['points'].mean()
            avg_fg3m = games_vs_def['fg3m'].mean()
            avg_usg = games_vs_def['usg_pct'].mean()
            avg_minutes = games_vs_def['minutes_decimal'].mean()

            # Get sample teams
            sample_teams = list(games_vs_def['defense_info'].apply(
                lambda x: x['team_name'] if x else None
            ).dropna().unique()[:3])

            results.append(DefenseTypePerformance(
                defense_type=f"{def_cat} Defense",
                games_played=len(games_vs_def),
                avg_pts=avg_pts,
                avg_fg3m=avg_fg3m,
                avg_usg=avg_usg,
                avg_minutes=avg_minutes,
                pts_vs_average=avg_pts - season_avg_pts,
                fg3m_vs_average=avg_fg3m - season_avg_fg3m,
                sample_teams=sample_teams
            ))

    return results


# Helper functions

def _parse_minutes(minutes_str: str) -> float:
    """Convert 'MM:SS' format to decimal minutes."""
    if pd.isna(minutes_str) or minutes_str is None:
        return 0.0

    try:
        if ':' in str(minutes_str):
            parts = str(minutes_str).split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + (seconds / 60.0)
        else:
            return float(minutes_str)
    except (ValueError, IndexError):
        return 0.0


def _extract_opponent(matchup: str) -> str:
    """Extract opponent abbreviation from matchup string.

    Examples:
        'LAL vs. BOS' -> 'BOS'
        'LAL @ BOS' -> 'BOS'
    """
    if pd.isna(matchup) or matchup is None:
        return ''

    matchup = str(matchup)

    if 'vs.' in matchup:
        return matchup.split('vs.')[-1].strip()
    elif '@' in matchup:
        return matchup.split('@')[-1].strip()
    else:
        return ''
