#!/usr/bin/env python3
"""
Injury Impact Analytics Module

Analyzes how player absences affect:
1. Team performance (win rate, scoring, defense)
2. Teammate usage and scoring redistribution
3. Opponent performance changes

Uses game log data to infer absences rather than relying on injury reports.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PlayerAbsenceStats:
    """Statistics comparing team/player performance with and without a specific player."""

    player_name: str
    player_id: int
    team_name: str
    season: str

    # Game counts
    games_played: int
    games_absent: int
    total_team_games: int

    # Team performance with player
    team_wins_with: int
    team_losses_with: int
    team_win_pct_with: float
    team_avg_pts_with: float
    team_avg_pts_allowed_with: float

    # Team performance without player
    team_wins_without: int
    team_losses_without: int
    team_win_pct_without: float
    team_avg_pts_without: float
    team_avg_pts_allowed_without: float

    # Impact deltas
    win_pct_delta: float
    offensive_rating_delta: float
    defensive_rating_delta: float


@dataclass
class TeammateImpact:
    """How a teammate's stats change when the key player is absent."""

    teammate_name: str
    teammate_id: int

    # With key player present
    avg_pts_with: float
    avg_usg_with: float
    avg_minutes_with: float

    # With key player absent
    avg_pts_without: float
    avg_usg_without: float
    avg_minutes_without: float

    # Deltas
    pts_delta: float
    usg_delta: float
    minutes_delta: float

    # Sample sizes
    games_together: int
    games_apart: int


@dataclass
class OpponentImpact:
    """How opponents perform differently when a key player is absent."""

    # Opponent stats when key player present
    avg_opp_pts_with: float
    avg_opp_fg3m_with: float
    avg_opp_fg_pct_with: float

    # Opponent stats when key player absent
    avg_opp_pts_without: float
    avg_opp_fg3m_without: float
    avg_opp_fg_pct_without: float

    # Deltas (positive means opponents perform better without the player)
    opp_pts_delta: float
    opp_fg3m_delta: float
    opp_fg_pct_delta: float


def get_player_absences(
    conn: sqlite3.Connection,
    player_id: int,
    season: str = "2025-26",
    season_type: str = "Regular Season",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify games where a player was absent vs. present.

    Returns:
        (games_with_player, games_without_player) - Two DataFrames of team game logs
    """
    # Get all team games for the player's team(s) during the season
    team_games_query = """
        SELECT DISTINCT tgl.*
        FROM team_game_logs tgl
        WHERE tgl.season = ?
          AND tgl.season_type = ?
          AND tgl.team_id IN (
              SELECT DISTINCT team_id
              FROM player_game_logs
              WHERE player_id = ? AND season = ? AND season_type = ?
          )
        ORDER BY tgl.game_date
    """

    team_games = pd.read_sql_query(
        team_games_query,
        conn,
        params=[season, season_type, player_id, season, season_type]
    )

    # Get games where the player actually played
    player_games_query = """
        SELECT game_id, player_id, player_name, team_id, points, minutes, usg_pct
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

    # Split team games into with/without player
    player_game_ids = set(player_games['game_id'].values)

    games_with_player = team_games[team_games['game_id'].isin(player_game_ids)].copy()
    games_without_player = team_games[~team_games['game_id'].isin(player_game_ids)].copy()

    return games_with_player, games_without_player


def calculate_team_impact(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    season: str = "2025-26",
    season_type: str = "Regular Season",
) -> Optional[PlayerAbsenceStats]:
    """
    Calculate how team performance changes when a player is absent.
    """
    games_with, games_without = get_player_absences(conn, player_id, season, season_type)

    if games_with.empty:
        return None

    # Get team name from first game
    team_name = games_with.iloc[0]['team_name'] if not games_with.empty else "Unknown"

    # Calculate team stats WITH player
    if not games_with.empty:
        # Determine wins/losses (assuming pts > opp_pts means win)
        games_with['is_win'] = games_with['pts'] > games_with['opp_pts']
        wins_with = games_with['is_win'].sum()
        losses_with = len(games_with) - wins_with
        win_pct_with = wins_with / len(games_with) if len(games_with) > 0 else 0.0
        avg_pts_with = games_with['pts'].mean()
        avg_pts_allowed_with = games_with['opp_pts'].mean()
    else:
        wins_with = losses_with = 0
        win_pct_with = avg_pts_with = avg_pts_allowed_with = 0.0

    # Calculate team stats WITHOUT player
    if not games_without.empty:
        games_without['is_win'] = games_without['pts'] > games_without['opp_pts']
        wins_without = games_without['is_win'].sum()
        losses_without = len(games_without) - wins_without
        win_pct_without = wins_without / len(games_without) if len(games_without) > 0 else 0.0
        avg_pts_without = games_without['pts'].mean()
        avg_pts_allowed_without = games_without['opp_pts'].mean()
    else:
        wins_without = losses_without = 0
        win_pct_without = avg_pts_without = avg_pts_allowed_without = 0.0

    # Calculate deltas
    win_pct_delta = win_pct_with - win_pct_without
    offensive_rating_delta = avg_pts_with - avg_pts_without
    defensive_rating_delta = avg_pts_allowed_with - avg_pts_allowed_without

    return PlayerAbsenceStats(
        player_name=player_name,
        player_id=player_id,
        team_name=team_name,
        season=season,
        games_played=len(games_with),
        games_absent=len(games_without),
        total_team_games=len(games_with) + len(games_without),
        team_wins_with=wins_with,
        team_losses_with=losses_with,
        team_win_pct_with=win_pct_with,
        team_avg_pts_with=avg_pts_with,
        team_avg_pts_allowed_with=avg_pts_allowed_with,
        team_wins_without=wins_without,
        team_losses_without=losses_without,
        team_win_pct_without=win_pct_without,
        team_avg_pts_without=avg_pts_without,
        team_avg_pts_allowed_without=avg_pts_allowed_without,
        win_pct_delta=win_pct_delta,
        offensive_rating_delta=offensive_rating_delta,
        defensive_rating_delta=defensive_rating_delta,
    )


def calculate_teammate_redistribution(
    conn: sqlite3.Connection,
    player_id: int,
    season: str = "2025-26",
    season_type: str = "Regular Season",
    min_games: int = 3,
) -> List[TeammateImpact]:
    """
    Analyze how teammates' usage and scoring change when key player is absent.
    """
    games_with, games_without = get_player_absences(conn, player_id, season, season_type)

    if games_with.empty or games_without.empty:
        return []

    # Get the team ID
    team_id = games_with.iloc[0]['team_id']

    # Get all teammate game logs for games WITH the key player
    with_game_ids = tuple(games_with['game_id'].values)

    query_with = f"""
        SELECT player_id, player_name,
               AVG(points) as avg_pts,
               AVG(CAST(usg_pct AS REAL)) as avg_usg,
               AVG(CAST(minutes AS REAL)) as avg_minutes,
               COUNT(*) as game_count
        FROM player_game_logs
        WHERE team_id = ?
          AND season = ?
          AND season_type = ?
          AND player_id != ?
          AND game_id IN ({','.join('?' * len(with_game_ids))})
          AND CAST(minutes AS REAL) > 0
        GROUP BY player_id, player_name
        HAVING COUNT(*) >= ?
    """

    teammates_with = pd.read_sql_query(
        query_with,
        conn,
        params=[team_id, season, season_type, player_id] + list(with_game_ids) + [min_games]
    )

    # Get all teammate game logs for games WITHOUT the key player
    if not games_without.empty:
        without_game_ids = tuple(games_without['game_id'].values)

        query_without = f"""
            SELECT player_id, player_name,
                   AVG(points) as avg_pts,
                   AVG(CAST(usg_pct AS REAL)) as avg_usg,
                   AVG(CAST(minutes AS REAL)) as avg_minutes,
                   COUNT(*) as game_count
            FROM player_game_logs
            WHERE team_id = ?
              AND season = ?
              AND season_type = ?
              AND player_id != ?
              AND game_id IN ({','.join('?' * len(without_game_ids))})
              AND CAST(minutes AS REAL) > 0
            GROUP BY player_id, player_name
            HAVING COUNT(*) >= ?
        """

        teammates_without = pd.read_sql_query(
            query_without,
            conn,
            params=[team_id, season, season_type, player_id] + list(without_game_ids) + [min_games]
        )
    else:
        teammates_without = pd.DataFrame()

    # Merge and calculate deltas
    teammate_impacts = []

    if not teammates_without.empty:
        merged = teammates_with.merge(
            teammates_without,
            on='player_id',
            suffixes=('_with', '_without'),
            how='inner'
        )

        for _, row in merged.iterrows():
            impact = TeammateImpact(
                teammate_name=row['player_name_with'],
                teammate_id=row['player_id'],
                avg_pts_with=row['avg_pts_with'],
                avg_usg_with=row['avg_usg_with'] if pd.notna(row['avg_usg_with']) else 0.0,
                avg_minutes_with=row['avg_minutes_with'] if pd.notna(row['avg_minutes_with']) else 0.0,
                avg_pts_without=row['avg_pts_without'],
                avg_usg_without=row['avg_usg_without'] if pd.notna(row['avg_usg_without']) else 0.0,
                avg_minutes_without=row['avg_minutes_without'] if pd.notna(row['avg_minutes_without']) else 0.0,
                pts_delta=row['avg_pts_without'] - row['avg_pts_with'],
                usg_delta=(row['avg_usg_without'] if pd.notna(row['avg_usg_without']) else 0.0) -
                         (row['avg_usg_with'] if pd.notna(row['avg_usg_with']) else 0.0),
                minutes_delta=(row['avg_minutes_without'] if pd.notna(row['avg_minutes_without']) else 0.0) -
                             (row['avg_minutes_with'] if pd.notna(row['avg_minutes_with']) else 0.0),
                games_together=int(row['game_count_with']),
                games_apart=int(row['game_count_without']),
            )
            teammate_impacts.append(impact)

    # Sort by absolute points delta (biggest changes first)
    teammate_impacts.sort(key=lambda x: abs(x.pts_delta), reverse=True)

    return teammate_impacts


def calculate_opponent_impact(
    conn: sqlite3.Connection,
    player_id: int,
    season: str = "2025-26",
    season_type: str = "Regular Season",
) -> Optional[OpponentImpact]:
    """
    Analyze how opponents perform when key player is absent.
    """
    games_with, games_without = get_player_absences(conn, player_id, season, season_type)

    if games_with.empty:
        return None

    # Calculate opponent stats WITH player
    if not games_with.empty:
        avg_opp_pts_with = games_with['opp_pts'].mean()
        avg_opp_fg3m_with = games_with['opp_fg3m'].mean()

        # Calculate opponent FG% (if data available)
        if 'opp_fga' in games_with.columns:
            games_with_fga = games_with[games_with['opp_fga'] > 0].copy()
            if not games_with_fga.empty:
                avg_opp_fg_pct_with = (games_with_fga['opp_pts'].sum() /
                                       (2 * games_with_fga['opp_fga'].sum()) * 100)
            else:
                avg_opp_fg_pct_with = 0.0
        else:
            avg_opp_fg_pct_with = 0.0
    else:
        avg_opp_pts_with = avg_opp_fg3m_with = avg_opp_fg_pct_with = 0.0

    # Calculate opponent stats WITHOUT player
    if not games_without.empty:
        avg_opp_pts_without = games_without['opp_pts'].mean()
        avg_opp_fg3m_without = games_without['opp_fg3m'].mean()

        if 'opp_fga' in games_without.columns:
            games_without_fga = games_without[games_without['opp_fga'] > 0].copy()
            if not games_without_fga.empty:
                avg_opp_fg_pct_without = (games_without_fga['opp_pts'].sum() /
                                          (2 * games_without_fga['opp_fga'].sum()) * 100)
            else:
                avg_opp_fg_pct_without = 0.0
        else:
            avg_opp_fg_pct_without = 0.0
    else:
        avg_opp_pts_without = avg_opp_fg3m_without = avg_opp_fg_pct_without = 0.0

    return OpponentImpact(
        avg_opp_pts_with=avg_opp_pts_with,
        avg_opp_fg3m_with=avg_opp_fg3m_with,
        avg_opp_fg_pct_with=avg_opp_fg_pct_with,
        avg_opp_pts_without=avg_opp_pts_without,
        avg_opp_fg3m_without=avg_opp_fg3m_without,
        avg_opp_fg_pct_without=avg_opp_fg_pct_without,
        opp_pts_delta=avg_opp_pts_without - avg_opp_pts_with,
        opp_fg3m_delta=avg_opp_fg3m_without - avg_opp_fg3m_with,
        opp_fg_pct_delta=avg_opp_fg_pct_without - avg_opp_fg_pct_with,
    )


def get_significant_players(
    conn: sqlite3.Connection,
    season: str = "2025-26",
    season_type: str = "Regular Season",
    min_games: int = 10,
    min_avg_points: float = 10.0,
) -> pd.DataFrame:
    """
    Get list of significant players (those whose absence would be impactful).

    Returns DataFrame with player_id, player_name, team_name, games_played, avg_points, avg_usg
    """
    query = """
        SELECT
            pst.player_id,
            pst.player_name,
            pst.team_name,
            pst.games_played,
            ROUND(pst.points / CAST(pst.games_played AS REAL), 1) as avg_points,
            ROUND(pst.usg_pct, 1) as avg_usg
        FROM player_season_totals pst
        WHERE pst.season = ?
          AND pst.season_type = ?
          AND pst.games_played >= ?
          AND (pst.points / CAST(pst.games_played AS REAL)) >= ?
        ORDER BY (pst.points / CAST(pst.games_played AS REAL)) DESC
    """

    return pd.read_sql_query(
        query,
        conn,
        params=[season, season_type, min_games, min_avg_points]
    )
