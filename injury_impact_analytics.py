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
    avg_reb_with: float = 0.0
    avg_ast_with: float = 0.0
    avg_stl_with: float = 0.0
    avg_blk_with: float = 0.0
    avg_fg3m_with: float = 0.0

    # With key player absent
    avg_pts_without: float = 0.0
    avg_usg_without: float = 0.0
    avg_minutes_without: float = 0.0
    avg_reb_without: float = 0.0
    avg_ast_without: float = 0.0
    avg_stl_without: float = 0.0
    avg_blk_without: float = 0.0
    avg_fg3m_without: float = 0.0

    # Deltas (positive = teammate does BETTER when key player is OUT)
    pts_delta: float = 0.0
    usg_delta: float = 0.0
    minutes_delta: float = 0.0
    reb_delta: float = 0.0      # Rebounds increase when rebounder is out
    ast_delta: float = 0.0      # Assists increase when playmaker is out
    stl_delta: float = 0.0      # Steals redistribution
    blk_delta: float = 0.0      # Blocks redistribution
    fg3m_delta: float = 0.0     # 3-pointers (volume opportunity)

    # Sample sizes
    games_together: int = 0
    games_apart: int = 0


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
        # Filter out games with null scores and convert to numeric
        games_with_clean = games_with.copy()
        games_with_clean['pts'] = pd.to_numeric(games_with_clean['pts'], errors='coerce')
        games_with_clean['opp_pts'] = pd.to_numeric(games_with_clean['opp_pts'], errors='coerce')
        games_with_clean = games_with_clean.dropna(subset=['pts', 'opp_pts'])

        if not games_with_clean.empty:
            # Determine wins/losses (assuming pts > opp_pts means win)
            games_with_clean['is_win'] = games_with_clean['pts'] > games_with_clean['opp_pts']
            wins_with = games_with_clean['is_win'].sum()
            losses_with = len(games_with_clean) - wins_with
            win_pct_with = wins_with / len(games_with_clean) if len(games_with_clean) > 0 else 0.0
            avg_pts_with = games_with_clean['pts'].mean()
            avg_pts_allowed_with = games_with_clean['opp_pts'].mean()
        else:
            wins_with = losses_with = 0
            win_pct_with = avg_pts_with = avg_pts_allowed_with = 0.0
    else:
        wins_with = losses_with = 0
        win_pct_with = avg_pts_with = avg_pts_allowed_with = 0.0

    # Calculate team stats WITHOUT player
    if not games_without.empty:
        # Filter out games with null scores and convert to numeric
        games_without_clean = games_without.copy()
        games_without_clean['pts'] = pd.to_numeric(games_without_clean['pts'], errors='coerce')
        games_without_clean['opp_pts'] = pd.to_numeric(games_without_clean['opp_pts'], errors='coerce')
        games_without_clean = games_without_clean.dropna(subset=['pts', 'opp_pts'])

        if not games_without_clean.empty:
            games_without_clean['is_win'] = games_without_clean['pts'] > games_without_clean['opp_pts']
            wins_without = games_without_clean['is_win'].sum()
            losses_without = len(games_without_clean) - wins_without
            win_pct_without = wins_without / len(games_without_clean) if len(games_without_clean) > 0 else 0.0
            avg_pts_without = games_without_clean['pts'].mean()
            avg_pts_allowed_without = games_without_clean['opp_pts'].mean()
        else:
            wins_without = losses_without = 0
            win_pct_without = avg_pts_without = avg_pts_allowed_without = 0.0
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


def _get_available_stat_columns(conn: sqlite3.Connection) -> set:
    """Get which stat columns are available in the player_game_logs table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(player_game_logs)")
    existing = {row[1] for row in cursor.fetchall()}
    return existing


def _build_teammate_query(game_ids: tuple, existing_columns: set) -> str:
    """
    Build a dynamic query based on available columns.
    Only queries columns that exist in the database.
    """
    # Base columns that should always exist
    select_parts = [
        "player_id",
        "player_name",
        "AVG(points) as avg_pts",
        "AVG(CAST(usg_pct AS REAL)) as avg_usg",
        "AVG(CAST(minutes AS REAL)) as avg_minutes",
    ]

    # Optional stat columns
    optional_stats = [
        ('rebounds', 'avg_reb'),
        ('assists', 'avg_ast'),
        ('steals', 'avg_stl'),
        ('blocks', 'avg_blk'),
        ('fg3m', 'avg_fg3m'),
    ]

    for col, alias in optional_stats:
        if col in existing_columns:
            select_parts.append(f"AVG(COALESCE({col}, 0)) as {alias}")
        else:
            select_parts.append(f"0.0 as {alias}")

    select_parts.append("COUNT(*) as game_count")

    return f"""
        SELECT {', '.join(select_parts)}
        FROM player_game_logs
        WHERE team_id = ?
          AND season = ?
          AND season_type = ?
          AND player_id != ?
          AND game_id IN ({','.join('?' * len(game_ids))})
          AND CAST(minutes AS REAL) > 0
        GROUP BY player_id, player_name
        HAVING COUNT(*) >= ?
    """


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

    # Get which columns exist in the database
    existing_columns = _get_available_stat_columns(conn)

    # Get the team ID (convert to native int for SQL compatibility)
    team_id = int(games_with.iloc[0]['team_id'])

    # Get all teammate game logs for games WITH the key player
    with_game_ids = tuple(games_with['game_id'].values)
    query_with = _build_teammate_query(with_game_ids, existing_columns)

    teammates_with = pd.read_sql_query(
        query_with,
        conn,
        params=[team_id, season, season_type, player_id] + list(with_game_ids) + [min_games]
    )

    # Get all teammate game logs for games WITHOUT the key player
    if not games_without.empty:
        without_game_ids = tuple(games_without['game_id'].values)
        query_without = _build_teammate_query(without_game_ids, existing_columns)

        teammates_without = pd.read_sql_query(
            query_without,
            conn,
            params=[team_id, season, season_type, player_id] + list(without_game_ids) + [min_games]
        )
    else:
        teammates_without = pd.DataFrame()

    # Merge and calculate deltas
    teammate_impacts = []

    if not teammates_without.empty and not teammates_with.empty:
        merged = teammates_with.merge(
            teammates_without,
            on='player_id',
            suffixes=('_with', '_without'),
            how='inner'
        )

        for _, row in merged.iterrows():
            # Helper function for safe numeric extraction
            def safe_float(val, default=0.0):
                return float(val) if pd.notna(val) else default

            impact = TeammateImpact(
                teammate_name=row['player_name_with'],
                teammate_id=row['player_id'],
                # With key player present
                avg_pts_with=safe_float(row['avg_pts_with']),
                avg_usg_with=safe_float(row['avg_usg_with']),
                avg_minutes_with=safe_float(row['avg_minutes_with']),
                avg_reb_with=safe_float(row.get('avg_reb_with', 0)),
                avg_ast_with=safe_float(row.get('avg_ast_with', 0)),
                avg_stl_with=safe_float(row.get('avg_stl_with', 0)),
                avg_blk_with=safe_float(row.get('avg_blk_with', 0)),
                avg_fg3m_with=safe_float(row.get('avg_fg3m_with', 0)),
                # With key player absent
                avg_pts_without=safe_float(row['avg_pts_without']),
                avg_usg_without=safe_float(row['avg_usg_without']),
                avg_minutes_without=safe_float(row['avg_minutes_without']),
                avg_reb_without=safe_float(row.get('avg_reb_without', 0)),
                avg_ast_without=safe_float(row.get('avg_ast_without', 0)),
                avg_stl_without=safe_float(row.get('avg_stl_without', 0)),
                avg_blk_without=safe_float(row.get('avg_blk_without', 0)),
                avg_fg3m_without=safe_float(row.get('avg_fg3m_without', 0)),
                # Deltas (positive = teammate does better when key player OUT)
                pts_delta=safe_float(row['avg_pts_without']) - safe_float(row['avg_pts_with']),
                usg_delta=safe_float(row['avg_usg_without']) - safe_float(row['avg_usg_with']),
                minutes_delta=safe_float(row['avg_minutes_without']) - safe_float(row['avg_minutes_with']),
                reb_delta=safe_float(row.get('avg_reb_without', 0)) - safe_float(row.get('avg_reb_with', 0)),
                ast_delta=safe_float(row.get('avg_ast_without', 0)) - safe_float(row.get('avg_ast_with', 0)),
                stl_delta=safe_float(row.get('avg_stl_without', 0)) - safe_float(row.get('avg_stl_with', 0)),
                blk_delta=safe_float(row.get('avg_blk_without', 0)) - safe_float(row.get('avg_blk_with', 0)),
                fg3m_delta=safe_float(row.get('avg_fg3m_without', 0)) - safe_float(row.get('avg_fg3m_with', 0)),
                # Sample sizes
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
        # Convert to numeric and handle nulls
        games_with_clean = games_with.copy()
        games_with_clean['opp_pts'] = pd.to_numeric(games_with_clean['opp_pts'], errors='coerce')
        games_with_clean['opp_fg3m'] = pd.to_numeric(games_with_clean['opp_fg3m'], errors='coerce')

        avg_opp_pts_with = games_with_clean['opp_pts'].mean() if not games_with_clean['opp_pts'].isna().all() else 0.0
        avg_opp_fg3m_with = games_with_clean['opp_fg3m'].mean() if not games_with_clean['opp_fg3m'].isna().all() else 0.0

        # Calculate opponent FG% (if data available)
        if 'opp_fga' in games_with_clean.columns:
            games_with_clean['opp_fga'] = pd.to_numeric(games_with_clean['opp_fga'], errors='coerce')
            games_with_fga = games_with_clean[(games_with_clean['opp_fga'] > 0) & (games_with_clean['opp_fga'].notna())].copy()
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
        # Convert to numeric and handle nulls
        games_without_clean = games_without.copy()
        games_without_clean['opp_pts'] = pd.to_numeric(games_without_clean['opp_pts'], errors='coerce')
        games_without_clean['opp_fg3m'] = pd.to_numeric(games_without_clean['opp_fg3m'], errors='coerce')

        avg_opp_pts_without = games_without_clean['opp_pts'].mean() if not games_without_clean['opp_pts'].isna().all() else 0.0
        avg_opp_fg3m_without = games_without_clean['opp_fg3m'].mean() if not games_without_clean['opp_fg3m'].isna().all() else 0.0

        if 'opp_fga' in games_without_clean.columns:
            games_without_clean['opp_fga'] = pd.to_numeric(games_without_clean['opp_fga'], errors='coerce')
            games_without_fga = games_without_clean[(games_without_clean['opp_fga'] > 0) & (games_without_clean['opp_fga'].notna())].copy()
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


# =============================================================================
# STAT ADJUSTMENT FUNCTIONS - Apply injury impacts to projections
# =============================================================================

@dataclass
class InjuryStatBoost:
    """
    Stat boosts for a player when specific teammates are out.

    All values are ADDITIVE deltas to apply to projections.
    Example: pts_boost = +3.2 means add 3.2 to projected points.
    """

    player_id: int
    player_name: str

    # Total stat boosts from all injured teammates
    pts_boost: float = 0.0
    reb_boost: float = 0.0
    ast_boost: float = 0.0
    stl_boost: float = 0.0
    blk_boost: float = 0.0
    fg3m_boost: float = 0.0
    minutes_boost: float = 0.0
    usg_boost: float = 0.0

    # Tracking
    injured_teammates: List[str] = None
    confidence: float = 0.0  # Based on sample size
    total_games_data: int = 0

    def __post_init__(self):
        if self.injured_teammates is None:
            self.injured_teammates = []

    def to_fpts_boost(self) -> float:
        """
        Convert stat boosts to estimated FanDuel fantasy points boost.

        FanDuel scoring:
        - Points: 1 FPTS each
        - Rebounds: 1.2 FPTS each
        - Assists: 1.5 FPTS each
        - Steals: 3 FPTS each
        - Blocks: 3 FPTS each
        - 3PM: Bonus already in points
        """
        return (
            self.pts_boost * 1.0 +
            self.reb_boost * 1.2 +
            self.ast_boost * 1.5 +
            self.stl_boost * 3.0 +
            self.blk_boost * 3.0
        )


def get_injury_stat_boosts(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    injured_player_ids: List[int],
    season: str = "2025-26",
    season_type: str = "Regular Season",
    min_games_apart: int = 3,
) -> InjuryStatBoost:
    """
    Calculate stat boosts for a player when specific teammates are injured/out.

    This aggregates the historical stat deltas across all injured teammates
    to provide a total projection boost.

    Args:
        conn: Database connection
        player_id: The player receiving the boost
        player_name: Player's name for display
        injured_player_ids: List of player_ids who are OUT
        season: NBA season
        season_type: Regular Season or Playoffs
        min_games_apart: Minimum sample size for reliability

    Returns:
        InjuryStatBoost with total additive adjustments
    """
    boost = InjuryStatBoost(
        player_id=player_id,
        player_name=player_name,
    )

    if not injured_player_ids:
        return boost

    # For each injured teammate, get the redistribution data
    total_games = 0
    injured_names = []

    for injured_id in injured_player_ids:
        impacts = calculate_teammate_redistribution(
            conn, injured_id, season, season_type, min_games=min_games_apart
        )

        # Find this player in the impacts
        for impact in impacts:
            if impact.teammate_id == player_id:
                # Add this injured player's impact
                boost.pts_boost += impact.pts_delta
                boost.reb_boost += impact.reb_delta
                boost.ast_boost += impact.ast_delta
                boost.stl_boost += impact.stl_delta
                boost.blk_boost += impact.blk_delta
                boost.fg3m_boost += impact.fg3m_delta
                boost.minutes_boost += impact.minutes_delta
                boost.usg_boost += impact.usg_delta
                total_games += impact.games_apart

                # Get injured player name from query
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT full_name FROM players WHERE player_id = ?",
                    [injured_id]
                )
                result = cursor.fetchone()
                if result:
                    injured_names.append(result[0])
                break

    boost.injured_teammates = injured_names
    boost.total_games_data = total_games

    # Calculate confidence based on sample size
    # 10+ games apart = high confidence, 3-5 games = low confidence
    if total_games >= 10:
        boost.confidence = 1.0
    elif total_games >= 5:
        boost.confidence = 0.7
    elif total_games >= min_games_apart:
        boost.confidence = 0.5
    else:
        boost.confidence = 0.3

    return boost


def get_team_injured_players(
    conn: sqlite3.Connection,
    team_id: int,
    game_date: str,
    season: str = "2025-26",
) -> List[Dict]:
    """
    Get list of injured/out players for a team on a given date.

    Uses injury_reports table if available, otherwise returns empty list.
    You can also manually specify injured players.

    Returns:
        List of dicts with player_id, player_name, injury_status
    """
    cursor = conn.cursor()

    # Check if injury_reports table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='injury_reports'
    """)

    if not cursor.fetchone():
        return []

    # Query injury reports for this date and team
    cursor.execute("""
        SELECT player_id, player_name, status, injury_type
        FROM injury_reports
        WHERE team_id = ?
          AND date(report_date) = date(?)
          AND status IN ('Out', 'Doubtful')
    """, [team_id, game_date])

    injured = []
    for row in cursor.fetchall():
        injured.append({
            'player_id': row[0],
            'player_name': row[1],
            'status': row[2],
            'injury_type': row[3],
        })

    return injured


def apply_injury_boosts_to_projection(
    base_projection: Dict[str, float],
    injury_boost: InjuryStatBoost,
    apply_confidence_scaling: bool = True,
) -> Dict[str, float]:
    """
    Apply injury stat boosts to a base projection.

    Args:
        base_projection: Dict with keys like 'pts', 'reb', 'ast', 'stl', 'blk', 'fg3m'
        injury_boost: The calculated InjuryStatBoost
        apply_confidence_scaling: Scale boosts by confidence level

    Returns:
        Updated projection dict
    """
    scale = injury_boost.confidence if apply_confidence_scaling else 1.0

    adjusted = base_projection.copy()

    # Apply scaled boosts
    if 'pts' in adjusted:
        adjusted['pts'] = adjusted['pts'] + (injury_boost.pts_boost * scale)
    if 'reb' in adjusted:
        adjusted['reb'] = adjusted['reb'] + (injury_boost.reb_boost * scale)
    if 'ast' in adjusted:
        adjusted['ast'] = adjusted['ast'] + (injury_boost.ast_boost * scale)
    if 'stl' in adjusted:
        adjusted['stl'] = adjusted['stl'] + (injury_boost.stl_boost * scale)
    if 'blk' in adjusted:
        adjusted['blk'] = adjusted['blk'] + (injury_boost.blk_boost * scale)
    if 'fg3m' in adjusted:
        adjusted['fg3m'] = adjusted['fg3m'] + (injury_boost.fg3m_boost * scale)
    if 'minutes' in adjusted:
        adjusted['minutes'] = adjusted['minutes'] + (injury_boost.minutes_boost * scale)

    return adjusted


def get_slate_injury_boosts(
    conn: sqlite3.Connection,
    game_date: str,
    player_ids: Optional[List[int]] = None,
    injured_player_ids: Optional[Dict[int, List[int]]] = None,
    season: str = "2025-26",
    season_type: str = "Regular Season",
) -> Dict[int, InjuryStatBoost]:
    """
    Calculate injury boosts for all players on a slate.

    This is the main entry point for integrating injury impacts into the
    prediction pipeline.

    Args:
        conn: Database connection
        game_date: Game date (YYYY-MM-DD)
        player_ids: Optional list of specific player IDs to process
        injured_player_ids: Optional dict mapping team_id -> [injured_player_ids]
                           If not provided, will query injury_reports table
        season: NBA season
        season_type: Season type

    Returns:
        Dict mapping player_id -> InjuryStatBoost
    """
    cursor = conn.cursor()
    boosts = {}

    # Get all players in predictions for this date if not specified
    if player_ids is None:
        cursor.execute("""
            SELECT DISTINCT player_id, player_name, team_id
            FROM predictions
            WHERE date(game_date) = date(?)
        """, [game_date])
        players = cursor.fetchall()
    else:
        # Get player info for specified IDs
        if not player_ids:
            return boosts
        placeholders = ','.join('?' * len(player_ids))
        cursor.execute(f"""
            SELECT player_id, player_name, team_id
            FROM predictions
            WHERE player_id IN ({placeholders})
              AND date(game_date) = date(?)
        """, player_ids + [game_date])
        players = cursor.fetchall()

    # Get injured players by team if not provided
    if injured_player_ids is None:
        injured_player_ids = {}
        # Get unique team IDs
        team_ids = set(p[2] for p in players)
        for team_id in team_ids:
            injured = get_team_injured_players(conn, team_id, game_date, season)
            if injured:
                injured_player_ids[team_id] = [i['player_id'] for i in injured]

    # Calculate boosts for each player
    for player_id, player_name, team_id in players:
        team_injured = injured_player_ids.get(team_id, [])

        if team_injured:
            boost = get_injury_stat_boosts(
                conn,
                player_id,
                player_name,
                team_injured,
                season,
                season_type,
            )
            if boost.to_fpts_boost() != 0:
                boosts[player_id] = boost

    return boosts


def format_injury_boost_summary(boost: InjuryStatBoost) -> str:
    """
    Format an InjuryStatBoost for display in UI.

    Returns a string like:
    "+3.2 PTS, +1.5 REB, +0.8 AST (Ja Morant out) [High confidence]"
    """
    parts = []

    if abs(boost.pts_boost) >= 0.5:
        parts.append(f"{boost.pts_boost:+.1f} PTS")
    if abs(boost.reb_boost) >= 0.5:
        parts.append(f"{boost.reb_boost:+.1f} REB")
    if abs(boost.ast_boost) >= 0.5:
        parts.append(f"{boost.ast_boost:+.1f} AST")
    if abs(boost.stl_boost) >= 0.2:
        parts.append(f"{boost.stl_boost:+.1f} STL")
    if abs(boost.blk_boost) >= 0.2:
        parts.append(f"{boost.blk_boost:+.1f} BLK")

    if not parts:
        return ""

    stat_str = ", ".join(parts)

    # Add injured players
    if boost.injured_teammates:
        injured_str = ", ".join(boost.injured_teammates[:2])  # Max 2 names
        if len(boost.injured_teammates) > 2:
            injured_str += f" +{len(boost.injured_teammates) - 2}"
        stat_str += f" ({injured_str} out)"

    # Add confidence indicator
    if boost.confidence >= 0.8:
        stat_str += " [High]"
    elif boost.confidence >= 0.5:
        stat_str += " [Med]"
    else:
        stat_str += " [Low]"

    return stat_str
