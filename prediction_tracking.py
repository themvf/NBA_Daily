#!/usr/bin/env python3
"""Prediction tracking and validation system for NBA_Daily.

This module tracks daily projections vs actual performance to:
1. Log predictions before games happen
2. Compare predictions to actual results
3. Calculate accuracy metrics (MAE, RMSE, hit rate)
4. Identify which factors improve/hurt predictions
5. Enable continuous model improvement
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Dict
import pandas as pd
import sqlite3
from pathlib import Path


@dataclass
class Prediction:
    """A single player prediction for a specific game."""
    prediction_id: Optional[int]  # Auto-increment primary key
    prediction_date: str  # Date prediction was made (YYYY-MM-DD)
    game_date: str  # Date game will be played (YYYY-MM-DD)
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    opponent_id: int
    opponent_name: str

    # Projection values
    projected_ppg: float
    proj_confidence: float
    proj_floor: float
    proj_ceiling: float

    # Factors used in projection
    season_avg_ppg: float
    recent_avg_3: Optional[float]
    recent_avg_5: Optional[float]
    vs_opponent_avg: Optional[float]
    vs_opponent_games: int

    # Analytics indicators
    analytics_used: str  # e.g., "🎯⚡" for correlation + pace split

    # Matchup metadata
    opponent_def_rating: Optional[float]
    opponent_pace: Optional[float]
    dfs_score: float
    dfs_grade: str

    # Actual performance (filled in after game)
    actual_ppg: Optional[float] = None
    actual_fg3m: Optional[float] = None
    actual_minutes: Optional[float] = None
    did_play: Optional[bool] = None  # False if DNP/injury

    # Accuracy metrics (calculated after game)
    error: Optional[float] = None  # actual - projected
    abs_error: Optional[float] = None  # |actual - projected|
    hit_floor_ceiling: Optional[bool] = None  # Was actual within range?

    # Opponent injury impact (filled in during prediction)
    opponent_injury_detected: bool = False  # Was opponent injury boost applied?
    opponent_injury_boost_projection: float = 0.0  # Projection boost % (e.g., 0.05 = +5%)
    opponent_injury_boost_ceiling: float = 0.0  # Ceiling boost % (e.g., 0.12 = +12%)
    opponent_injured_player_ids: Optional[str] = None  # Comma-separated IDs
    opponent_injury_impact_score: float = 0.0  # Total impact score (0-2.0+)

    created_at: Optional[str] = None  # Timestamp when logged


@dataclass
class AccuracyMetrics:
    """Aggregate accuracy metrics for a set of predictions."""
    total_predictions: int
    predictions_with_actuals: int

    # Error metrics
    mean_error: float  # Average (actual - projected), shows bias
    mean_absolute_error: float  # Average |actual - projected|
    rmse: float  # Root mean squared error

    # Hit rates
    hit_rate_floor_ceiling: float  # % predictions within floor-ceiling range
    over_projections: int  # Count where projected > actual
    under_projections: int  # Count where projected < actual

    # By confidence level
    high_confidence_mae: Optional[float] = None  # MAE for confidence >= 70%
    low_confidence_mae: Optional[float] = None  # MAE for confidence < 50%

    # By analytics quality
    correlation_used_mae: Optional[float] = None  # MAE when 🎯 used
    generic_only_mae: Optional[float] = None  # MAE when 📊 used


def create_predictions_table(conn: sqlite3.Connection) -> None:
    """Create the predictions tracking table if it doesn't exist."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT NOT NULL,
            game_date TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            opponent_id INTEGER NOT NULL,
            opponent_name TEXT NOT NULL,

            -- Projection values
            projected_ppg REAL NOT NULL,
            proj_confidence REAL NOT NULL,
            proj_floor REAL NOT NULL,
            proj_ceiling REAL NOT NULL,

            -- Factors used
            season_avg_ppg REAL NOT NULL,
            recent_avg_3 REAL,
            recent_avg_5 REAL,
            vs_opponent_avg REAL,
            vs_opponent_games INTEGER NOT NULL,

            -- Analytics metadata
            analytics_used TEXT NOT NULL,
            opponent_def_rating REAL,
            opponent_pace REAL,
            dfs_score REAL NOT NULL,
            dfs_grade TEXT NOT NULL,

            -- Actual performance (populated after game)
            actual_ppg REAL,
            actual_fg3m REAL,
            actual_minutes REAL,
            did_play BOOLEAN,

            -- Accuracy metrics
            error REAL,
            abs_error REAL,
            hit_floor_ceiling BOOLEAN,

            -- Injury adjustment tracking (OUR team's injuries)
            injury_adjusted BOOLEAN DEFAULT 0,
            injury_adjustment_amount REAL DEFAULT 0.0,
            injured_player_ids TEXT DEFAULT NULL,
            original_projected_ppg REAL DEFAULT NULL,
            original_proj_floor REAL DEFAULT NULL,
            original_proj_ceiling REAL DEFAULT NULL,
            original_proj_confidence REAL DEFAULT NULL,

            -- Opponent injury impact tracking (OPPONENT team's injuries)
            opponent_injury_detected BOOLEAN DEFAULT 0,
            opponent_injury_boost_projection REAL DEFAULT 0.0,
            opponent_injury_boost_ceiling REAL DEFAULT 0.0,
            opponent_injured_player_ids TEXT DEFAULT NULL,
            opponent_injury_impact_score REAL DEFAULT 0.0,

            -- Refresh audit trail
            last_refreshed_at TEXT DEFAULT NULL,
            refresh_count INTEGER DEFAULT 0,
            refresh_reason TEXT DEFAULT NULL,

            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

            -- Indexes for common queries
            UNIQUE(player_id, game_date)
        )
    """)

    # Create indexes for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_game_date
        ON predictions(game_date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_player_id
        ON predictions(player_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_actual_ppg
        ON predictions(actual_ppg)
    """)

    conn.commit()


def upgrade_predictions_table_for_injuries(conn: sqlite3.Connection) -> None:
    """Add injury adjustment fields to existing predictions table if they don't exist."""
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define new columns to add
    new_columns = {
        'injury_adjusted': 'BOOLEAN DEFAULT 0',
        'injury_adjustment_amount': 'REAL DEFAULT 0.0',
        'injured_player_ids': 'TEXT DEFAULT NULL',
        'original_projected_ppg': 'REAL DEFAULT NULL',
        'original_proj_floor': 'REAL DEFAULT NULL',
        'original_proj_ceiling': 'REAL DEFAULT NULL',
        'original_proj_confidence': 'REAL DEFAULT NULL',
    }

    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")

    conn.commit()


def upgrade_predictions_table_for_refresh(conn: sqlite3.Connection) -> None:
    """Add refresh audit trail fields to existing predictions table if they don't exist."""
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define new columns to add
    new_columns = {
        'last_refreshed_at': 'TEXT DEFAULT NULL',
        'refresh_count': 'INTEGER DEFAULT 0',
        'refresh_reason': 'TEXT DEFAULT NULL',
    }

    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")

    conn.commit()


def upgrade_predictions_table_for_opponent_injury(conn: sqlite3.Connection) -> None:
    """Add opponent injury impact fields to existing predictions table if they don't exist."""
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define new columns to add
    new_columns = {
        'opponent_injury_detected': 'BOOLEAN DEFAULT 0',
        'opponent_injury_boost_projection': 'REAL DEFAULT 0.0',
        'opponent_injury_boost_ceiling': 'REAL DEFAULT 0.0',
        'opponent_injured_player_ids': 'TEXT DEFAULT NULL',
        'opponent_injury_impact_score': 'REAL DEFAULT 0.0',
    }

    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"Added opponent injury column: {col_name}")

    conn.commit()


def get_predictions_for_date(
    conn: sqlite3.Connection,
    game_date: str
) -> List[Dict]:
    """
    Get all predictions for a specific date as a list of dictionaries.

    Args:
        conn: Database connection
        game_date: Date in YYYY-MM-DD format

    Returns:
        List of prediction dictionaries with all fields
    """
    query = """
        SELECT
            prediction_id, player_id, player_name, team_id, team_name,
            opponent_id, opponent_name, projected_ppg, proj_confidence,
            proj_floor, proj_ceiling, season_avg_ppg,
            injury_adjusted, injury_adjustment_amount, injured_player_ids,
            original_projected_ppg, original_proj_floor,
            original_proj_ceiling, original_proj_confidence
        FROM predictions
        WHERE game_date = ?
    """

    df = pd.read_sql_query(query, conn, params=[game_date])
    return df.to_dict('records')


def log_prediction(conn: sqlite3.Connection, pred: Prediction) -> int:
    """
    Log a prediction to the database.

    Returns:
        prediction_id of the inserted/updated record
    """
    cursor = conn.cursor()

    # Use INSERT OR REPLACE to handle re-runs on same day
    cursor.execute("""
        INSERT OR REPLACE INTO predictions (
            prediction_date, game_date, player_id, player_name,
            team_id, team_name, opponent_id, opponent_name,
            projected_ppg, proj_confidence, proj_floor, proj_ceiling,
            season_avg_ppg, recent_avg_3, recent_avg_5,
            vs_opponent_avg, vs_opponent_games,
            analytics_used, opponent_def_rating, opponent_pace,
            dfs_score, dfs_grade,
            opponent_injury_detected, opponent_injury_boost_projection,
            opponent_injury_boost_ceiling, opponent_injured_player_ids,
            opponent_injury_impact_score,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        pred.prediction_date, pred.game_date, pred.player_id, pred.player_name,
        pred.team_id, pred.team_name, pred.opponent_id, pred.opponent_name,
        pred.projected_ppg, pred.proj_confidence, pred.proj_floor, pred.proj_ceiling,
        pred.season_avg_ppg, pred.recent_avg_3, pred.recent_avg_5,
        pred.vs_opponent_avg, pred.vs_opponent_games,
        pred.analytics_used, pred.opponent_def_rating, pred.opponent_pace,
        pred.dfs_score, pred.dfs_grade,
        pred.opponent_injury_detected, pred.opponent_injury_boost_projection,
        pred.opponent_injury_boost_ceiling, pred.opponent_injured_player_ids,
        pred.opponent_injury_impact_score
    ))

    conn.commit()
    return cursor.lastrowid


def update_actuals(
    conn: sqlite3.Connection,
    game_date: str,
    player_id: int,
    actual_ppg: float,
    actual_fg3m: float,
    actual_minutes: float,
    did_play: bool = True
) -> bool:
    """
    Update a prediction with actual performance data.

    Returns:
        True if prediction was found and updated, False otherwise
    """
    cursor = conn.cursor()

    # First, check if prediction exists
    cursor.execute("""
        SELECT prediction_id, projected_ppg, proj_floor, proj_ceiling
        FROM predictions
        WHERE game_date = ? AND player_id = ?
    """, (game_date, player_id))

    result = cursor.fetchone()
    if not result:
        return False

    prediction_id, projected_ppg, proj_floor, proj_ceiling = result

    # Calculate accuracy metrics
    error = actual_ppg - projected_ppg
    abs_error = abs(error)
    hit_floor_ceiling = proj_floor <= actual_ppg <= proj_ceiling

    # Update the record
    cursor.execute("""
        UPDATE predictions
        SET actual_ppg = ?,
            actual_fg3m = ?,
            actual_minutes = ?,
            did_play = ?,
            error = ?,
            abs_error = ?,
            hit_floor_ceiling = ?
        WHERE prediction_id = ?
    """, (
        actual_ppg, actual_fg3m, actual_minutes, did_play,
        error, abs_error, hit_floor_ceiling,
        prediction_id
    ))

    conn.commit()
    return True


def bulk_update_actuals_from_game_logs(
    conn: sqlite3.Connection,
    game_date: str
) -> int:
    """
    Update all predictions for a specific game date using player_game_logs.

    Returns:
        Number of predictions updated
    """
    # Get all player game logs for this date
    query = """
        SELECT player_id, points, fg3m, minutes
        FROM player_game_logs
        WHERE game_date = ?
    """

    game_logs = pd.read_sql_query(query, conn, params=[game_date])

    if game_logs.empty:
        return 0

    updated_count = 0

    for _, log in game_logs.iterrows():
        player_id = log['player_id']
        points = log['points']
        fg3m = log['fg3m']
        minutes_str = log['minutes']

        # Parse minutes from MM:SS format
        try:
            if ':' in str(minutes_str):
                parts = str(minutes_str).split(':')
                minutes = int(parts[0]) + int(parts[1]) / 60.0
            else:
                minutes = float(minutes_str) if minutes_str else 0.0
        except (ValueError, IndexError):
            minutes = 0.0

        # Update prediction
        success = update_actuals(
            conn, game_date, player_id,
            actual_ppg=points,
            actual_fg3m=fg3m,
            actual_minutes=minutes,
            did_play=minutes > 0
        )

        if success:
            updated_count += 1

    return updated_count


def calculate_accuracy_metrics(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_confidence: Optional[float] = None
) -> AccuracyMetrics:
    """
    Calculate accuracy metrics for predictions in a date range.

    Args:
        start_date: Start date (YYYY-MM-DD), inclusive
        end_date: End date (YYYY-MM-DD), inclusive
        min_confidence: Only include predictions with confidence >= this value
    """
    query = """
        SELECT
            projected_ppg, actual_ppg, error, abs_error,
            hit_floor_ceiling, proj_confidence, analytics_used
        FROM predictions
        WHERE actual_ppg IS NOT NULL
    """

    params = []

    if start_date:
        query += " AND game_date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND game_date <= ?"
        params.append(end_date)

    if min_confidence is not None:
        query += " AND proj_confidence >= ?"
        params.append(min_confidence)

    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return AccuracyMetrics(
            total_predictions=0,
            predictions_with_actuals=0,
            mean_error=0.0,
            mean_absolute_error=0.0,
            rmse=0.0,
            hit_rate_floor_ceiling=0.0,
            over_projections=0,
            under_projections=0
        )

    # Calculate basic metrics
    mean_error = df['error'].mean()
    mae = df['abs_error'].mean()
    rmse = (df['error'] ** 2).mean() ** 0.5
    hit_rate = df['hit_floor_ceiling'].mean()
    over = (df['error'] < 0).sum()
    under = (df['error'] > 0).sum()

    # Calculate by confidence level
    high_conf_df = df[df['proj_confidence'] >= 0.70]
    low_conf_df = df[df['proj_confidence'] < 0.50]

    high_conf_mae = high_conf_df['abs_error'].mean() if not high_conf_df.empty else None
    low_conf_mae = low_conf_df['abs_error'].mean() if not low_conf_df.empty else None

    # Calculate by analytics quality
    correlation_df = df[df['analytics_used'].str.contains('🎯', na=False)]
    generic_df = df[df['analytics_used'] == '📊']

    correlation_mae = correlation_df['abs_error'].mean() if not correlation_df.empty else None
    generic_mae = generic_df['abs_error'].mean() if not generic_df.empty else None

    # Get total count including those without actuals
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE game_date >= ? AND game_date <= ?",
                   (start_date or '2000-01-01', end_date or '2099-12-31'))
    total_count = cursor.fetchone()[0]

    return AccuracyMetrics(
        total_predictions=total_count,
        predictions_with_actuals=len(df),
        mean_error=mean_error,
        mean_absolute_error=mae,
        rmse=rmse,
        hit_rate_floor_ceiling=hit_rate,
        over_projections=over,
        under_projections=under,
        high_confidence_mae=high_conf_mae,
        low_confidence_mae=low_conf_mae,
        correlation_used_mae=correlation_mae,
        generic_only_mae=generic_mae
    )


def get_predictions_vs_actuals(
    conn: sqlite3.Connection,
    game_date: str,
    min_dfs_score: Optional[float] = None
) -> pd.DataFrame:
    """
    Get predictions vs actuals for a specific game date.

    Returns DataFrame with columns for analysis and display.
    """
    query = """
        SELECT
            player_name,
            team_name,
            opponent_name,
            projected_ppg,
            actual_ppg,
            error,
            abs_error,
            proj_confidence,
            proj_floor,
            proj_ceiling,
            hit_floor_ceiling,
            dfs_score,
            dfs_grade,
            analytics_used,
            season_avg_ppg,
            did_play
        FROM predictions
        WHERE game_date = ?
    """

    params = [game_date]

    if min_dfs_score is not None:
        query += " AND dfs_score >= ?"
        params.append(min_dfs_score)

    query += " ORDER BY dfs_score DESC, projected_ppg DESC"

    df = pd.read_sql_query(query, conn, params=params)

    return df


def get_best_worst_predictions(
    conn: sqlite3.Connection,
    n: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Get the best and worst predictions (by absolute error).

    Returns:
        Dict with keys 'best' and 'worst', each containing a DataFrame
    """
    query = """
        SELECT
            game_date,
            player_name,
            team_name,
            opponent_name,
            projected_ppg,
            actual_ppg,
            error,
            abs_error,
            proj_confidence,
            dfs_score,
            analytics_used
        FROM predictions
        WHERE actual_ppg IS NOT NULL
    """

    params = []

    if start_date:
        query += " AND game_date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND game_date <= ?"
        params.append(end_date)

    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return {'best': pd.DataFrame(), 'worst': pd.DataFrame()}

    # Sort by absolute error
    df_sorted = df.sort_values('abs_error')

    return {
        'best': df_sorted.head(n),
        'worst': df_sorted.tail(n)
    }
