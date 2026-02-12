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


def ensure_prediction_sim_columns(conn: sqlite3.Connection) -> None:
    """
    Add Monte Carlo simulation columns to predictions table.

    These columns store the simulation outputs that are used for ranking:
    - p_top3: Probability of finishing in top 3 scorers
    - p_top1: Probability of being the #1 scorer
    - top_scorer_score: Heuristic score for top scorer potential
    - sim_*: Metadata for reproducibility

    CRITICAL: Without these columns, backtest falls back to projected_ppg ranking,
    which causes star burial issues (e.g., Murray #76, Shai #41).
    """
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define simulation output columns
    sim_columns = {
        # Core ranking fields (used by backtest)
        'p_top3': 'REAL DEFAULT NULL',
        'p_top1': 'REAL DEFAULT NULL',
        'top_scorer_score': 'REAL DEFAULT NULL',
        # Simulation metadata (for reproducibility)
        'sim_sigma': 'REAL DEFAULT NULL',           # Variance used in simulation
        'sim_tier': 'TEXT DEFAULT NULL',            # Player tier (STAR/ROLE/BENCH)
        'sim_n': 'INTEGER DEFAULT NULL',            # Number of simulations
        'sim_seed': 'INTEGER DEFAULT NULL',         # Random seed (if set)
        'sim_version': 'TEXT DEFAULT NULL',         # Algorithm version
        'sim_created_at': 'TEXT DEFAULT NULL',      # When simulation was run
        'sim_status': 'TEXT DEFAULT NULL',          # OK/FAILED/PENDING
    }

    # Add missing columns
    added = []
    for col_name, col_type in sim_columns.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
                added.append(col_name)
            except Exception as e:
                print(f"Warning: Could not add column {col_name}: {e}")

    if added:
        print(f"Added simulation columns: {', '.join(added)}")

    # Create indexes for common ranking queries
    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_p_top3
            ON predictions(game_date, p_top3)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_top_scorer_score
            ON predictions(game_date, top_scorer_score)
        """)
    except Exception as e:
        print(f"Warning: Could not create indexes: {e}")

    conn.commit()


def persist_sim_results(
    conn: sqlite3.Connection,
    game_date: str,
    df: 'pd.DataFrame',
    sim_n: int = 10000,
    sim_seed: int = None,
    sim_version: str = 'v1.0'
) -> int:
    """
    Persist simulation results back into predictions table.

    Args:
        conn: Database connection
        game_date: Game date (YYYY-MM-DD)
        df: DataFrame with columns: player_id, p_top3, p_top1, top_scorer_score,
            optionally: scoring_stddev, tier
        sim_n: Number of simulations run
        sim_seed: Random seed used (if any)
        sim_version: Algorithm version string

    Returns:
        Number of rows updated
    """
    cursor = conn.cursor()

    rows = []
    for _, r in df.iterrows():
        rows.append((
            float(r.get('p_top3', 0) or 0),
            float(r.get('p_top1', 0) or 0),
            float(r.get('top_scorer_score', 0) or 0),
            float(r.get('scoring_stddev', 0) or 0) if 'scoring_stddev' in r else None,
            str(r.get('tier', '')) if 'tier' in r else None,
            sim_n,
            sim_seed,
            sim_version,
            'OK',  # sim_status
            game_date,
            int(r['player_id']),
        ))

    cursor.executemany("""
        UPDATE predictions
        SET p_top3 = ?,
            p_top1 = ?,
            top_scorer_score = ?,
            sim_sigma = ?,
            sim_tier = ?,
            sim_n = ?,
            sim_seed = ?,
            sim_version = ?,
            sim_status = ?,
            sim_created_at = datetime('now')
        WHERE game_date = ?
          AND player_id = ?
    """, rows)

    conn.commit()
    return cursor.rowcount


def upgrade_predictions_table_for_fanduel(conn: sqlite3.Connection) -> None:
    """Add FanDuel odds fields to existing predictions table if they don't exist."""
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define new columns to add
    new_columns = {
        'fanduel_ou': 'REAL DEFAULT NULL',
        'fanduel_over_odds': 'INTEGER DEFAULT NULL',
        'fanduel_under_odds': 'INTEGER DEFAULT NULL',
        'fanduel_fetched_at': 'TEXT DEFAULT NULL',
        'odds_event_id': 'TEXT DEFAULT NULL',
    }

    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"Added FanDuel column: {col_name}")

    conn.commit()


def upgrade_predictions_table_for_fanduel_comparison(conn: sqlite3.Connection) -> None:
    """Add FanDuel comparison tracking columns to predictions table."""
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define new comparison columns
    new_columns = {
        # Over/Under tracking
        'our_ou_call': 'TEXT DEFAULT NULL',           # 'over' or 'under'
        'actual_ou_result': 'TEXT DEFAULT NULL',      # 'over' or 'under'
        'ou_call_correct': 'INTEGER DEFAULT NULL',    # 1 if correct, 0 if not
        # Who was closer
        'fanduel_error': 'REAL DEFAULT NULL',         # |actual - fanduel_ou|
        'we_were_closer': 'INTEGER DEFAULT NULL',     # 1 if we were closer
        'closer_margin': 'REAL DEFAULT NULL',         # How much closer (positive = us)
    }

    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"Added FanDuel comparison column: {col_name}")

    conn.commit()


def upgrade_predictions_table_for_minutes(conn: sqlite3.Connection) -> None:
    """Add projected minutes and tier columns to predictions table if they don't exist."""
    cursor = conn.cursor()

    # Check which columns already exist
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define new minutes/tier columns
    new_columns = {
        'proj_minutes': 'REAL DEFAULT NULL',           # Projected minutes
        'l5_minutes_avg': 'REAL DEFAULT NULL',         # Last 5 games avg minutes
        'l5_minutes_stddev': 'REAL DEFAULT NULL',      # Last 5 games minutes stddev
        'season_minutes_avg': 'REAL DEFAULT NULL',     # Season avg minutes
        'starter_est': 'INTEGER DEFAULT NULL',         # 1 if estimated starter
        'tier': 'TEXT DEFAULT NULL',                   # 'STAR', 'ROLE', 'SIXTH_MAN', 'BENCH'
        'minutes_confidence': 'REAL DEFAULT NULL',     # Confidence in minutes projection
        'role_change': 'INTEGER DEFAULT NULL',         # 1 if recent role change detected
    }

    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"Added minutes/tier column: {col_name}")

    conn.commit()


def create_backtest_table(conn: sqlite3.Connection) -> None:
    """Create backtest_daily_results table for storing backtest metrics."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_daily_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date TEXT NOT NULL,
            strategy_name TEXT NOT NULL,

            -- Our picks (with diagnostics)
            picked1_id INTEGER,
            picked1_name TEXT,
            picked1_pts REAL,
            picked1_finish INTEGER,
            picked1_pred_rank INTEGER,
            picked2_id INTEGER,
            picked2_name TEXT,
            picked2_pts REAL,
            picked2_finish INTEGER,
            picked2_pred_rank INTEGER,
            picked3_id INTEGER,
            picked3_name TEXT,
            picked3_pts REAL,
            picked3_finish INTEGER,
            picked3_pred_rank INTEGER,

            -- Actual top 3
            actual1_id INTEGER,
            actual1_name TEXT,
            actual1_points REAL,
            actual2_id INTEGER,
            actual2_name TEXT,
            actual2_points REAL,
            actual3_id INTEGER,
            actual3_name TEXT,
            actual3_points REAL,

            -- Overlap metrics (tournament-style)
            overlap INTEGER,
            hit_any INTEGER,
            hit_2plus INTEGER,
            hit_exact INTEGER,
            hit_1 INTEGER,
            tie_friendly_overlap INTEGER,

            -- Closeness metrics
            closest_miss REAL,

            -- Ranking quality metrics
            pred_rank_a1 INTEGER,
            pred_rank_a2 INTEGER,
            pred_rank_a3 INTEGER,
            avg_rank_actual_top3 REAL,
            best_rank_actual_top3 INTEGER,

            -- Context
            n_pred_players INTEGER,
            actual_3rd_points REAL,
            ties_at_3rd INTEGER,

            created_at TEXT DEFAULT (datetime('now')),

            UNIQUE(slate_date, strategy_name)
        )
    """)

    # Add columns if they don't exist (for schema upgrades)
    new_columns = [
        ('picked1_pts', 'REAL'),
        ('picked1_finish', 'INTEGER'),
        ('picked1_pred_rank', 'INTEGER'),
        ('picked1_proj', 'REAL'),
        ('picked1_p_top1', 'REAL'),
        ('picked2_pts', 'REAL'),
        ('picked2_finish', 'INTEGER'),
        ('picked2_pred_rank', 'INTEGER'),
        ('picked2_proj', 'REAL'),
        ('picked2_p_top1', 'REAL'),
        ('picked3_pts', 'REAL'),
        ('picked3_finish', 'INTEGER'),
        ('picked3_pred_rank', 'INTEGER'),
        ('picked3_proj', 'REAL'),
        ('picked3_p_top1', 'REAL'),
        ('closest_miss', 'REAL'),
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE backtest_daily_results ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_backtest_date
        ON backtest_daily_results(slate_date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_backtest_strategy
        ON backtest_daily_results(strategy_name)
    """)

    conn.commit()


def calculate_fanduel_comparison_metrics(conn: sqlite3.Connection, game_date: Optional[str] = None) -> int:
    """
    Calculate FanDuel comparison metrics for predictions with actuals.

    Calculates:
    - our_ou_call: Did we project over or under FanDuel's line?
    - actual_ou_result: Did the actual score go over or under?
    - ou_call_correct: Did our call match reality?
    - fanduel_error: How far off was FanDuel?
    - we_were_closer: Did we beat FanDuel's prediction?
    - closer_margin: By how much?

    Args:
        conn: Database connection
        game_date: Optional specific date (YYYY-MM-DD). If None, processes all.

    Returns:
        Number of predictions updated
    """
    cursor = conn.cursor()

    # Ensure comparison columns exist first
    upgrade_predictions_table_for_fanduel_comparison(conn)

    # Check which columns exist to build appropriate WHERE clause
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Build WHERE clause based on available columns
    where_parts = ["actual_ppg IS NOT NULL", "fanduel_ou IS NOT NULL"]

    # Only filter on comparison columns if they exist
    if 'ou_call_correct' in existing_columns and 'fanduel_error' in existing_columns:
        where_parts.append("(ou_call_correct IS NULL OR fanduel_error IS NULL)")

    if game_date:
        where_parts.append(f"game_date = '{game_date}'")

    where_clause = "WHERE " + " AND ".join(where_parts)

    # Get predictions that need calculation
    cursor.execute(f"""
        SELECT prediction_id, projected_ppg, fanduel_ou, actual_ppg
        FROM predictions
        {where_clause}
    """)

    rows = cursor.fetchall()
    updated_count = 0

    for prediction_id, projected_ppg, fanduel_ou, actual_ppg in rows:
        # Calculate our O/U call (what we implied vs FanDuel line)
        our_ou_call = 'over' if projected_ppg > fanduel_ou else 'under'

        # Calculate actual result vs FanDuel line
        actual_ou_result = 'over' if actual_ppg > fanduel_ou else 'under'

        # Did our call match reality?
        ou_call_correct = 1 if our_ou_call == actual_ou_result else 0

        # Calculate errors
        our_error = abs(actual_ppg - projected_ppg)
        fanduel_error = abs(actual_ppg - fanduel_ou)

        # Who was closer?
        we_were_closer = 1 if our_error < fanduel_error else 0
        closer_margin = fanduel_error - our_error  # Positive = we were closer

        # Update the prediction
        cursor.execute("""
            UPDATE predictions
            SET our_ou_call = ?,
                actual_ou_result = ?,
                ou_call_correct = ?,
                fanduel_error = ?,
                we_were_closer = ?,
                closer_margin = ?
            WHERE prediction_id = ?
        """, (our_ou_call, actual_ou_result, ou_call_correct,
              fanduel_error, we_were_closer, closer_margin, prediction_id))

        updated_count += 1

    conn.commit()
    return updated_count


def get_overall_model_performance(conn: sqlite3.Connection,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> Dict:
    """
    Get overall model performance stats for ALL predictions with actuals.

    This includes predictions without FanDuel lines.

    Returns:
        Dict with keys: total_predictions, with_actuals, avg_error, hit_rate,
                       over_projected, under_projected, over_pct
    """
    cursor = conn.cursor()

    where_clauses = ["actual_ppg IS NOT NULL"]
    if start_date:
        where_clauses.append(f"game_date >= '{start_date}'")
    if end_date:
        where_clauses.append(f"game_date <= '{end_date}'")

    where_sql = " AND ".join(where_clauses)

    # Get total predictions in date range (with and without actuals)
    total_where = []
    if start_date:
        total_where.append(f"game_date >= '{start_date}'")
    if end_date:
        total_where.append(f"game_date <= '{end_date}'")
    total_where_sql = " AND ".join(total_where) if total_where else "1=1"

    cursor.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT(actual_ppg) as with_actuals
        FROM predictions
        WHERE {total_where_sql}
    """)
    counts = cursor.fetchone()
    total_predictions = counts[0] if counts else 0
    with_actuals = counts[1] if counts else 0

    # Get performance metrics for predictions with actuals
    cursor.execute(f"""
        SELECT
            AVG(abs_error) as avg_error,
            AVG(CASE WHEN hit_floor_ceiling = 1 THEN 1.0 ELSE 0.0 END) as hit_rate,
            SUM(CASE WHEN error < 0 THEN 1 ELSE 0 END) as over_projected,
            SUM(CASE WHEN error > 0 THEN 1 ELSE 0 END) as under_projected,
            SUM(CASE WHEN error = 0 THEN 1 ELSE 0 END) as exact
        FROM predictions
        WHERE {where_sql}
    """)

    row = cursor.fetchone()

    if row and with_actuals > 0:
        avg_error = row[0] or 0
        hit_rate = (row[1] or 0) * 100
        over_projected = row[2] or 0
        under_projected = row[3] or 0
        exact = row[4] or 0
        over_pct = (over_projected / with_actuals * 100) if with_actuals > 0 else 0
    else:
        avg_error = 0
        hit_rate = 0
        over_projected = 0
        under_projected = 0
        exact = 0
        over_pct = 0

    return {
        'total_predictions': total_predictions,
        'with_actuals': with_actuals,
        'avg_error': avg_error,
        'hit_rate': hit_rate,
        'over_projected': over_projected,
        'under_projected': under_projected,
        'exact': exact,
        'over_pct': over_pct
    }


def get_fanduel_comparison_summary(conn: sqlite3.Connection,
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None) -> Dict:
    """
    Get summary statistics for Model vs FanDuel comparison.

    Returns:
        Dict with keys: total_compared, ou_accuracy_pct, we_closer_pct,
                       our_avg_error, fd_avg_error, by_player stats
    """
    cursor = conn.cursor()

    where_clauses = ["ou_call_correct IS NOT NULL"]
    if start_date:
        where_clauses.append(f"game_date >= '{start_date}'")
    if end_date:
        where_clauses.append(f"game_date <= '{end_date}'")

    where_sql = " AND ".join(where_clauses)

    # Overall stats
    cursor.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(ou_call_correct) as correct_calls,
            SUM(we_were_closer) as times_closer,
            AVG(abs_error) as our_avg_error,
            AVG(fanduel_error) as fd_avg_error
        FROM predictions
        WHERE {where_sql}
    """)

    row = cursor.fetchone()
    total = row[0] or 0

    if total == 0:
        return {
            'total_compared': 0,
            'ou_accuracy_pct': 0,
            'we_closer_pct': 0,
            'our_avg_error': 0,
            'fd_avg_error': 0,
            'by_player': []
        }

    # Per-player breakdown (min 3 games)
    cursor.execute(f"""
        SELECT
            player_name,
            COUNT(*) as games,
            SUM(ou_call_correct) as correct,
            SUM(we_were_closer) as closer,
            AVG(abs_error) as our_mae,
            AVG(fanduel_error) as fd_mae
        FROM predictions
        WHERE {where_sql}
        GROUP BY player_id, player_name
        HAVING COUNT(*) >= 3
        ORDER BY SUM(we_were_closer) * 1.0 / COUNT(*) DESC
    """)

    by_player = []
    for player_row in cursor.fetchall():
        by_player.append({
            'player_name': player_row[0],
            'games': player_row[1],
            'ou_correct': player_row[2],
            'times_closer': player_row[3],
            'our_mae': player_row[4],
            'fd_mae': player_row[5],
        })

    return {
        'total_compared': total,
        'ou_accuracy_pct': (row[1] / total * 100) if total > 0 else 0,
        'we_closer_pct': (row[2] / total * 100) if total > 0 else 0,
        'our_avg_error': row[3] or 0,
        'fd_avg_error': row[4] or 0,
        'by_player': by_player
    }


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

    Uses INSERT ... ON CONFLICT DO UPDATE so that re-running predictions
    updates only the projection columns while preserving Vegas odds data
    (fanduel_ou, vegas_implied_fpts, etc.) that was written separately.

    Returns:
        prediction_id of the inserted/updated record
    """
    cursor = conn.cursor()

    # ON CONFLICT updates only prediction columns, preserving Vegas/FanDuel data
    cursor.execute("""
        INSERT INTO predictions (
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
        ON CONFLICT(player_id, game_date) DO UPDATE SET
            prediction_date = excluded.prediction_date,
            player_name = excluded.player_name,
            team_id = excluded.team_id,
            team_name = excluded.team_name,
            opponent_id = excluded.opponent_id,
            opponent_name = excluded.opponent_name,
            projected_ppg = excluded.projected_ppg,
            proj_confidence = excluded.proj_confidence,
            proj_floor = excluded.proj_floor,
            proj_ceiling = excluded.proj_ceiling,
            season_avg_ppg = excluded.season_avg_ppg,
            recent_avg_3 = excluded.recent_avg_3,
            recent_avg_5 = excluded.recent_avg_5,
            vs_opponent_avg = excluded.vs_opponent_avg,
            vs_opponent_games = excluded.vs_opponent_games,
            analytics_used = excluded.analytics_used,
            opponent_def_rating = excluded.opponent_def_rating,
            opponent_pace = excluded.opponent_pace,
            dfs_score = excluded.dfs_score,
            dfs_grade = excluded.dfs_grade,
            opponent_injury_detected = excluded.opponent_injury_detected,
            opponent_injury_boost_projection = excluded.opponent_injury_boost_projection,
            opponent_injury_boost_ceiling = excluded.opponent_injury_boost_ceiling,
            opponent_injured_player_ids = excluded.opponent_injured_player_ids,
            opponent_injury_impact_score = excluded.opponent_injury_impact_score,
            last_refreshed_at = CURRENT_TIMESTAMP,
            refresh_count = COALESCE(predictions.refresh_count, 0) + 1
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
