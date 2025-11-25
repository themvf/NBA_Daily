#!/usr/bin/env python3
"""Player correlation analytics for NBA_Daily.

Analyzes how players perform with specific teammates (positive correlations for DFS stacks)
and against specific opposing teams/players (matchup advantages).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import sqlite3


@dataclass
class TeammateCorrelation:
    """Stats showing how a player performs with a specific teammate."""
    teammate_id: int
    teammate_name: str
    games_together: int
    games_apart: int

    # Player's performance WITH this teammate
    avg_pts_together: float
    avg_fg3m_together: float
    avg_usg_together: float
    avg_minutes_together: float

    # Player's performance WITHOUT this teammate
    avg_pts_apart: float
    avg_fg3m_apart: float
    avg_usg_apart: float
    avg_minutes_apart: float

    # Deltas (together - apart)
    pts_delta: float
    fg3m_delta: float
    usg_delta: float
    minutes_delta: float

    # Correlation strength (0-100)
    correlation_score: float


@dataclass
class OpponentCorrelation:
    """Stats showing how a player performs against a specific opponent team."""
    opponent_team_id: int
    opponent_team_name: str
    games_vs_opponent: int

    # Player's performance vs this opponent
    avg_pts_vs: float
    avg_fg3m_vs: float
    avg_usg_vs: float
    avg_minutes_vs: float

    # Player's performance vs all other teams
    avg_pts_vs_others: float
    avg_fg3m_vs_others: float
    avg_usg_vs_others: float
    avg_minutes_vs_others: float

    # Deltas (vs opponent - vs others)
    pts_delta: float
    fg3m_delta: float
    usg_delta: float
    minutes_delta: float

    # Matchup advantage score (0-100)
    matchup_score: float


def calculate_teammate_correlations(
    conn: sqlite3.Connection,
    player_id: int,
    season: str,
    season_type: str,
    min_games_together: int = 3
) -> List[TeammateCorrelation]:
    """
    Calculate how TEAMMATES perform when the selected player is in/out of the lineup.

    This analyzes the impact of the selected player on their teammates' performance:
    - WITH player: teammate stats when player is in the lineup
    - WITHOUT player: teammate stats when player is absent

    Args:
        conn: Database connection
        player_id: The player whose impact on teammates we're analyzing
        season: Season to analyze (e.g., '2025-26')
        season_type: 'Regular Season' or 'Playoffs'
        min_games_together: Minimum games together to include in analysis

    Returns:
        List of TeammateCorrelation objects showing how each teammate performs
        with vs without the selected player, sorted by correlation_score (descending)
    """
    # Get all games the selected player participated in
    player_games_query = """
    SELECT game_id, team_id
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

    player_game_ids = set(player_games['game_id'].values)

    # Get the player's team ID (most recent)
    team_id = int(player_games['team_id'].iloc[-1])  # Convert numpy.int64 to int

    # Get all team games (to identify games where player was absent)
    team_games_query = """
    SELECT DISTINCT game_id
    FROM team_game_logs
    WHERE team_id = ?
      AND season = ?
      AND season_type = ?
    """
    team_games = pd.read_sql_query(
        team_games_query,
        conn,
        params=[team_id, season, season_type]
    )

    all_team_game_ids = set(team_games['game_id'].values)
    games_without_player = all_team_game_ids - player_game_ids

    # Need at least 1 game where player was absent
    if len(games_without_player) < 1:
        return []

    # Get all teammates (players who played for the same team, excluding the selected player)
    teammates_query = """
    SELECT DISTINCT player_id, player_name
    FROM player_game_logs
    WHERE team_id = ?
      AND season = ?
      AND season_type = ?
      AND player_id != ?
    ORDER BY player_name
    """
    teammates = pd.read_sql_query(
        teammates_query,
        conn,
        params=[team_id, season, season_type, player_id]
    )

    correlations = []

    for _, teammate_row in teammates.iterrows():
        teammate_id = teammate_row['player_id']
        teammate_name = teammate_row['player_name']

        # Get ALL games where this teammate played
        teammate_games_query = """
        SELECT game_id, points, fg3m, usg_pct, minutes
        FROM player_game_logs
        WHERE player_id = ?
          AND season = ?
          AND season_type = ?
        """
        teammate_all_games = pd.read_sql_query(
            teammate_games_query,
            conn,
            params=[teammate_id, season, season_type]
        )

        if teammate_all_games.empty:
            continue

        # Convert minutes from 'MM:SS' format to decimal
        teammate_all_games['minutes_decimal'] = teammate_all_games['minutes'].apply(_parse_minutes)

        # Split into WITH and WITHOUT selected player
        games_with_player = teammate_all_games[
            teammate_all_games['game_id'].isin(player_game_ids)
        ]
        games_without_player_df = teammate_all_games[
            teammate_all_games['game_id'].isin(games_without_player)
        ]

        # Need minimum games in each scenario
        if len(games_with_player) < min_games_together or len(games_without_player_df) < 1:
            continue

        # Calculate teammate stats WITH selected player
        avg_pts_together = games_with_player['points'].mean()
        avg_fg3m_together = games_with_player['fg3m'].mean()
        avg_usg_together = games_with_player['usg_pct'].mean()
        avg_minutes_together = games_with_player['minutes_decimal'].mean()

        # Calculate teammate stats WITHOUT selected player
        avg_pts_apart = games_without_player_df['points'].mean()
        avg_fg3m_apart = games_without_player_df['fg3m'].mean()
        avg_usg_apart = games_without_player_df['usg_pct'].mean()
        avg_minutes_apart = games_without_player_df['minutes_decimal'].mean()

        # Calculate deltas (positive = teammate performs better WITH selected player)
        pts_delta = avg_pts_together - avg_pts_apart
        fg3m_delta = avg_fg3m_together - avg_fg3m_apart
        usg_delta = avg_usg_together - avg_usg_apart
        minutes_delta = avg_minutes_together - avg_minutes_apart

        # Calculate correlation score (0-100)
        correlation_score = _calculate_correlation_score(
            pts_delta, fg3m_delta, minutes_delta, len(games_with_player)
        )

        correlations.append(TeammateCorrelation(
            teammate_id=teammate_id,
            teammate_name=teammate_name,
            games_together=len(games_with_player),
            games_apart=len(games_without_player_df),
            avg_pts_together=avg_pts_together,
            avg_fg3m_together=avg_fg3m_together,
            avg_usg_together=avg_usg_together,
            avg_minutes_together=avg_minutes_together,
            avg_pts_apart=avg_pts_apart,
            avg_fg3m_apart=avg_fg3m_apart,
            avg_usg_apart=avg_usg_apart,
            avg_minutes_apart=avg_minutes_apart,
            pts_delta=pts_delta,
            fg3m_delta=fg3m_delta,
            usg_delta=usg_delta,
            minutes_delta=minutes_delta,
            correlation_score=correlation_score
        ))

    # Sort by correlation score (highest = best synergy)
    correlations.sort(key=lambda x: x.correlation_score, reverse=True)

    return correlations


def calculate_opponent_correlations(
    conn: sqlite3.Connection,
    player_id: int,
    season: str,
    season_type: str,
    min_games_vs: int = 1
) -> List[OpponentCorrelation]:
    """
    Calculate how a player performs against each opponent team.

    Args:
        conn: Database connection
        player_id: The player to analyze
        season: Season to analyze (e.g., '2025-26')
        season_type: 'Regular Season' or 'Playoffs'
        min_games_vs: Minimum games against opponent to include

    Returns:
        List of OpponentCorrelation objects, sorted by matchup_score (descending)
    """
    # Get all games the player participated in
    player_games_query = """
    SELECT game_id, points, fg3m, usg_pct, minutes, matchup
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

    # Extract opponent from matchup field (format: "LAL vs. BOS" or "LAL @ BOS")
    player_games['opponent_abbr'] = player_games['matchup'].apply(_extract_opponent)

    # Get opponent team names
    teams_query = """
    SELECT DISTINCT team_id, team_abbreviation, team_name
    FROM player_game_logs
    WHERE season = ?
      AND season_type = ?
    """
    teams = pd.read_sql_query(teams_query, conn, params=[season, season_type])

    # Create mapping from abbreviation to team info
    team_map = {row['team_abbreviation']: (row['team_id'], row['team_name'])
                for _, row in teams.iterrows()}

    correlations = []

    # Group by opponent
    for opponent_abbr, games_vs in player_games.groupby('opponent_abbr'):
        if opponent_abbr not in team_map:
            continue

        opponent_team_id, opponent_team_name = team_map[opponent_abbr]

        games_vs_opponent = len(games_vs)

        if games_vs_opponent < min_games_vs:
            continue

        # Stats vs this opponent
        avg_pts_vs = games_vs['points'].mean()
        avg_fg3m_vs = games_vs['fg3m'].mean()
        avg_usg_vs = games_vs['usg_pct'].mean()
        avg_minutes_vs = games_vs['minutes_decimal'].mean()

        # Stats vs all OTHER opponents
        games_vs_others = player_games[player_games['opponent_abbr'] != opponent_abbr]

        if games_vs_others.empty:
            # Only played this opponent (unlikely but handle it)
            avg_pts_vs_others = avg_pts_vs
            avg_fg3m_vs_others = avg_fg3m_vs
            avg_usg_vs_others = avg_usg_vs
            avg_minutes_vs_others = avg_minutes_vs
        else:
            avg_pts_vs_others = games_vs_others['points'].mean()
            avg_fg3m_vs_others = games_vs_others['fg3m'].mean()
            avg_usg_vs_others = games_vs_others['usg_pct'].mean()
            avg_minutes_vs_others = games_vs_others['minutes_decimal'].mean()

        # Calculate deltas
        pts_delta = avg_pts_vs - avg_pts_vs_others
        fg3m_delta = avg_fg3m_vs - avg_fg3m_vs_others
        usg_delta = avg_usg_vs - avg_usg_vs_others
        minutes_delta = avg_minutes_vs - avg_minutes_vs_others

        # Calculate matchup advantage score (0-100)
        matchup_score = _calculate_matchup_score(
            pts_delta, fg3m_delta, games_vs_opponent
        )

        correlations.append(OpponentCorrelation(
            opponent_team_id=opponent_team_id,
            opponent_team_name=opponent_team_name,
            games_vs_opponent=games_vs_opponent,
            avg_pts_vs=avg_pts_vs,
            avg_fg3m_vs=avg_fg3m_vs,
            avg_usg_vs=avg_usg_vs,
            avg_minutes_vs=avg_minutes_vs,
            avg_pts_vs_others=avg_pts_vs_others,
            avg_fg3m_vs_others=avg_fg3m_vs_others,
            avg_usg_vs_others=avg_usg_vs_others,
            avg_minutes_vs_others=avg_minutes_vs_others,
            pts_delta=pts_delta,
            fg3m_delta=fg3m_delta,
            usg_delta=usg_delta,
            minutes_delta=minutes_delta,
            matchup_score=matchup_score
        ))

    # Sort by matchup score (highest = best matchup)
    correlations.sort(key=lambda x: x.matchup_score, reverse=True)

    return correlations


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
            # Already a number
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


def _calculate_correlation_score(
    pts_delta: float,
    fg3m_delta: float,
    minutes_delta: float,
    games_together: int
) -> float:
    """
    Calculate correlation score (0-100) for teammate synergy.

    Higher score = better performance with this teammate.
    Considers points boost, 3PM boost, and sample size.
    """
    # Base score from points delta (most important)
    # +5 pts = +50 score, +10 pts = +100 score
    pts_score = pts_delta * 10

    # Bonus from 3PM delta
    # +1 three = +20 score
    fg3m_score = fg3m_delta * 20

    # Small penalty if minutes drop significantly (reduced role)
    minutes_penalty = 0
    if minutes_delta < -5:
        minutes_penalty = abs(minutes_delta) * 2

    # Sample size confidence boost (more games = more reliable)
    confidence_boost = min(games_together * 2, 20)  # Cap at +20

    raw_score = pts_score + fg3m_score + confidence_boost - minutes_penalty

    # Normalize to 0-100 range, centered at 50
    score = 50 + raw_score

    return max(0, min(100, score))


def _calculate_matchup_score(
    pts_delta: float,
    fg3m_delta: float,
    games_vs: int
) -> float:
    """
    Calculate matchup advantage score (0-100).

    Higher score = better performance vs this opponent.
    """
    # Base score from points delta
    # +5 pts = +50 score, +10 pts = +100 score
    pts_score = pts_delta * 10

    # Bonus from 3PM delta
    # +1 three = +20 score
    fg3m_score = fg3m_delta * 20

    # Sample size confidence (more games = more reliable)
    confidence_boost = min(games_vs * 3, 15)  # Cap at +15

    raw_score = pts_score + fg3m_score + confidence_boost

    # Normalize to 0-100 range, centered at 50
    score = 50 + raw_score

    return max(0, min(100, score))
