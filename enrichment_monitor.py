#!/usr/bin/env python3
"""
Enrichment Monitoring Module

Tracks enrichment effects over time to validate hypotheses and detect drift.
Creates audit logs, weekly summaries, and alerts for enrichment health.

Tables Created:
- enrichment_audit_log: Per-prediction enrichment factors and outcomes
- enrichment_weekly_summary: Aggregated weekly metrics
- enrichment_alerts: Active alerts for drift/issues

Usage:
    # Daily job (after actuals are populated):
    python enrichment_monitor.py --daily

    # Weekly summary:
    python enrichment_monitor.py --weekly

    # Check for alerts:
    python enrichment_monitor.py --check-alerts
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

def ensure_enrichment_columns(conn: sqlite3.Connection):
    """Ensure predictions table has enrichment columns (migration for cloud DB)."""
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(predictions)")
    existing_cols = {col[1] for col in cursor.fetchall()}

    # Enrichment columns to add
    enrichment_cols = [
        ('days_rest', 'INTEGER DEFAULT NULL'),
        ('rest_multiplier', 'REAL DEFAULT 1.0'),
        ('is_b2b', 'INTEGER DEFAULT 0'),
        ('game_script_tier', 'TEXT DEFAULT "neutral"'),
        ('blowout_risk', 'REAL DEFAULT 0.0'),
        ('minutes_adjustment', 'REAL DEFAULT 0.0'),
        ('role_tier', 'TEXT DEFAULT NULL'),
        ('position_matchup_factor', 'REAL DEFAULT 1.0'),
    ]

    for col_name, col_def in enrichment_cols:
        if col_name not in existing_cols:
            try:
                cursor.execute(f'ALTER TABLE predictions ADD COLUMN {col_name} {col_def}')
            except Exception:
                pass  # Column might already exist

    conn.commit()


def ensure_monitoring_tables(conn: sqlite3.Connection):
    """Create monitoring tables if they don't exist."""
    cursor = conn.cursor()

    # First ensure predictions has enrichment columns
    ensure_enrichment_columns(conn)

    # Per-prediction audit log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enrichment_audit_log (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT,

            -- Enrichment factors applied
            days_rest INTEGER,
            rest_multiplier REAL,
            is_b2b INTEGER,
            game_script_tier TEXT,
            blowout_risk REAL,
            minutes_adjustment REAL,
            role_tier TEXT,
            position_matchup_factor REAL,

            -- Prediction vs Actual
            projected_ppg REAL,
            actual_ppg REAL,
            prediction_error REAL,
            abs_error REAL,

            -- Projection components
            base_projection REAL,
            enriched_projection REAL,
            enrichment_delta REAL,

            -- Outcome flags
            ceiling_hit INTEGER,
            was_top10 INTEGER,
            actual_minutes REAL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(game_date, player_id)
        )
    """)

    # Weekly summary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enrichment_weekly_summary (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_ending TEXT NOT NULL UNIQUE,

            -- Sample sizes
            total_predictions INTEGER,
            b2b_predictions INTEGER,
            blowout_predictions INTEGER,
            close_game_predictions INTEGER,

            -- Rest/B2B metrics
            b2b_mean_error REAL,
            non_b2b_mean_error REAL,
            b2b_effect_observed REAL,
            rest_multiplier_mae REAL,

            -- Game Script metrics
            blowout_star_mae REAL,
            blowout_bench_mae REAL,
            close_game_star_mae REAL,
            game_script_mae_impact REAL,

            -- Role metrics
            star_mae REAL,
            starter_mae REAL,
            rotation_mae REAL,
            bench_mae REAL,
            star_count INTEGER,
            starter_count INTEGER,
            rotation_count INTEGER,
            bench_count INTEGER,

            -- Position Defense metrics
            guard_mae REAL,
            forward_mae REAL,
            center_mae REAL,
            pos_factor_correlation REAL,
            pos_factor_mae_impact REAL,

            -- Overall
            overall_mae REAL,
            overall_rmse REAL,
            overall_bias REAL,
            ceiling_hit_rate REAL,
            top10_capture_rate REAL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enrichment_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            metric_name TEXT,
            metric_value REAL,
            threshold REAL,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT
        )
    """)

    conn.commit()


# =============================================================================
# ALERT DEFINITIONS
# =============================================================================

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AlertThreshold:
    """Defines an alert condition."""
    name: str
    category: str
    metric: str
    operator: str  # 'gt', 'lt', 'abs_gt'
    threshold: float
    severity: AlertSeverity
    message_template: str


ALERT_THRESHOLDS = [
    # B2B effect flip
    AlertThreshold(
        name="b2b_effect_flip",
        category="rest",
        metric="b2b_effect_observed",
        operator="gt",
        threshold=0.0,
        severity=AlertSeverity.HIGH,
        message_template="B2B effect flipped positive ({value:.1%}) - players on B2B outperforming"
    ),

    # B2B effect too weak
    AlertThreshold(
        name="b2b_effect_weak",
        category="rest",
        metric="b2b_effect_observed",
        operator="gt",
        threshold=-0.03,
        severity=AlertSeverity.MEDIUM,
        message_template="B2B effect weaker than expected ({value:.1%} vs expected -8%)"
    ),

    # Blowout overcorrection
    AlertThreshold(
        name="blowout_star_overcorrection",
        category="game_script",
        metric="blowout_star_mae",
        operator="gt",
        threshold=6.0,
        severity=AlertSeverity.MEDIUM,
        message_template="Blowout star MAE too high ({value:.1f}) - may be overcorrecting"
    ),

    # Role tier drift
    AlertThreshold(
        name="star_underperformance",
        category="roles",
        metric="star_mae",
        operator="gt",
        threshold=5.0,
        severity=AlertSeverity.MEDIUM,
        message_template="STAR tier MAE elevated ({value:.1f}) - check tier classification"
    ),

    # Position defense degradation
    AlertThreshold(
        name="pos_factor_degradation",
        category="position_defense",
        metric="pos_factor_mae_impact",
        operator="gt",
        threshold=0.0,
        severity=AlertSeverity.MEDIUM,
        message_template="Position defense factor increasing MAE ({value:.2f})"
    ),

    # Overall MAE spike
    AlertThreshold(
        name="overall_mae_spike",
        category="overall",
        metric="overall_mae",
        operator="gt",
        threshold=5.5,
        severity=AlertSeverity.HIGH,
        message_template="Overall MAE spiked to {value:.2f} (threshold: 5.5)"
    ),

    # Ceiling hit rate drop
    AlertThreshold(
        name="ceiling_hit_drop",
        category="overall",
        metric="ceiling_hit_rate",
        operator="lt",
        threshold=0.60,
        severity=AlertSeverity.MEDIUM,
        message_template="Ceiling hit rate dropped to {value:.1%} (threshold: 60%)"
    ),
]


# =============================================================================
# DAILY AUDIT LOG POPULATION
# =============================================================================

def populate_daily_audit_log(
    conn: sqlite3.Connection,
    game_date: str,
    verbose: bool = False
) -> int:
    """
    Populate enrichment audit log for a specific date.

    Requires: Predictions with actuals already populated.

    Args:
        conn: Database connection
        game_date: Game date (YYYY-MM-DD)
        verbose: Print progress

    Returns:
        Number of records inserted
    """
    ensure_monitoring_tables(conn)
    cursor = conn.cursor()

    # Get predictions with actuals
    cursor.execute("""
        SELECT
            p.game_date,
            p.player_id,
            p.player_name,
            p.days_rest,
            p.rest_multiplier,
            p.is_b2b,
            p.game_script_tier,
            p.blowout_risk,
            p.minutes_adjustment,
            p.role_tier,
            p.position_matchup_factor,
            p.projected_ppg,
            p.actual_ppg,
            p.proj_floor,
            p.proj_ceiling,
            p.actual_minutes,
            p.season_avg_ppg
        FROM predictions p
        WHERE date(p.game_date) = date(?)
          AND p.actual_ppg IS NOT NULL
    """, [game_date])

    predictions = cursor.fetchall()

    if verbose:
        print(f"Processing {len(predictions)} predictions for {game_date}")

    # Get top 10 actual scorers for this date
    cursor.execute("""
        SELECT player_id FROM predictions
        WHERE date(game_date) = date(?) AND actual_ppg IS NOT NULL
        ORDER BY actual_ppg DESC
        LIMIT 10
    """, [game_date])
    top10_ids = {row[0] for row in cursor.fetchall()}

    count = 0
    for row in predictions:
        (game_dt, player_id, player_name, days_rest, rest_mult, is_b2b,
         game_script, blowout_risk, min_adj, role_tier, pos_factor,
         projected, actual, floor, ceiling, actual_min, season_avg) = row

        # Calculate metrics
        prediction_error = (actual - projected) if actual and projected else None
        abs_error = abs(prediction_error) if prediction_error else None
        ceiling_hit = 1 if (floor and ceiling and actual and floor <= actual <= ceiling) else 0
        was_top10 = 1 if player_id in top10_ids else 0

        # Calculate base projection (before enrichments)
        # This is an approximation - ideally we'd store this separately
        base_projection = season_avg if season_avg else projected
        enriched_projection = projected
        enrichment_delta = (projected - base_projection) if base_projection else 0

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO enrichment_audit_log (
                    game_date, player_id, player_name,
                    days_rest, rest_multiplier, is_b2b,
                    game_script_tier, blowout_risk, minutes_adjustment,
                    role_tier, position_matchup_factor,
                    projected_ppg, actual_ppg, prediction_error, abs_error,
                    base_projection, enriched_projection, enrichment_delta,
                    ceiling_hit, was_top10, actual_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_dt, player_id, player_name,
                days_rest, rest_mult, is_b2b,
                game_script, blowout_risk, min_adj,
                role_tier, pos_factor,
                projected, actual, prediction_error, abs_error,
                base_projection, enriched_projection, enrichment_delta,
                ceiling_hit, was_top10, actual_min
            ))
            count += 1
        except Exception as e:
            if verbose:
                print(f"Error inserting player {player_id}: {e}")

    conn.commit()

    if verbose:
        print(f"Inserted {count} audit log records for {game_date}")

    return count


def backfill_audit_log(
    conn: sqlite3.Connection,
    days: int = 30,
    verbose: bool = False
) -> int:
    """Backfill audit log for recent days."""
    total = 0
    for i in range(days):
        game_date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
        count = populate_daily_audit_log(conn, game_date, verbose=verbose)
        total += count
    return total


# =============================================================================
# WEEKLY SUMMARY CALCULATION
# =============================================================================

def snap_to_week_ending_sunday(date_str: str) -> str:
    """
    Snap a date to the Sunday ending that calendar week.

    Calendar weeks run Monday-Sunday:
    - Monday 2026-01-20 → Sunday 2026-01-26
    - Tuesday 2026-01-21 → Sunday 2026-01-26
    - Sunday 2026-01-26 → Sunday 2026-01-26

    Returns the Sunday date as YYYY-MM-DD string.
    """
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    # weekday(): Monday=0, Sunday=6
    days_until_sunday = (6 - dt.weekday()) % 7
    if days_until_sunday == 0 and dt.weekday() != 6:
        # It's not Sunday, so we need to go forward to next Sunday
        days_until_sunday = 7
    # Actually, simpler: always go to the end of the current week
    # If today is Sunday (weekday=6), days_until_sunday = 0
    # If today is Monday (weekday=0), days_until_sunday = 6
    days_until_sunday = (6 - dt.weekday()) % 7
    week_end_dt = dt + timedelta(days=days_until_sunday)
    return week_end_dt.strftime('%Y-%m-%d')


def get_week_start_from_sunday(week_ending_sunday: str) -> str:
    """Given a Sunday week-ending date, return the Monday start."""
    dt = datetime.strptime(week_ending_sunday, '%Y-%m-%d')
    # Go back 6 days to get Monday
    return (dt - timedelta(days=6)).strftime('%Y-%m-%d')


def calculate_weekly_summary(
    conn: sqlite3.Connection,
    week_ending: str = None,
    verbose: bool = False
) -> Dict:
    """
    Calculate weekly enrichment summary metrics for a TRUE CALENDAR WEEK.

    Calendar weeks run Monday through Sunday. The week_ending parameter
    will be snapped to the Sunday ending that week.

    Args:
        conn: Database connection
        week_ending: Any date in the week (will be snapped to Sunday)
        verbose: Print progress

    Returns:
        Dict with all summary metrics
    """
    if week_ending is None:
        week_ending = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Snap to true calendar week ending Sunday
    week_ending = snap_to_week_ending_sunday(week_ending)
    week_start = get_week_start_from_sunday(week_ending)

    if verbose:
        print(f"Calculating summary for week {week_start} to {week_ending}")

    # Load audit log for the week
    query = """
        SELECT * FROM enrichment_audit_log
        WHERE date(game_date) BETWEEN date(?) AND date(?)
    """
    df = pd.read_sql_query(query, conn, params=[week_start, week_ending])

    if df.empty:
        if verbose:
            print("No audit data for this week")
        return {}

    # Calculate summary metrics
    summary = {
        'week_ending': week_ending,
        'total_predictions': len(df),
    }

    # B2B metrics
    b2b_df = df[df['is_b2b'] == 1]
    non_b2b_df = df[df['is_b2b'] == 0]

    summary['b2b_predictions'] = len(b2b_df)
    summary['b2b_mean_error'] = b2b_df['prediction_error'].mean() if len(b2b_df) > 0 else None
    summary['non_b2b_mean_error'] = non_b2b_df['prediction_error'].mean() if len(non_b2b_df) > 0 else None

    # B2B effect observed
    if len(b2b_df) > 0 and len(non_b2b_df) > 0:
        b2b_actual_mean = b2b_df['actual_ppg'].mean()
        non_b2b_actual_mean = non_b2b_df['actual_ppg'].mean()
        summary['b2b_effect_observed'] = (b2b_actual_mean - non_b2b_actual_mean) / non_b2b_actual_mean if non_b2b_actual_mean else None
    else:
        summary['b2b_effect_observed'] = None

    summary['rest_multiplier_mae'] = df['abs_error'].mean()

    # Game script metrics
    blowout_df = df[df['game_script_tier'] == 'blowout']
    close_df = df[df['game_script_tier'] == 'close_game']

    summary['blowout_predictions'] = len(blowout_df)
    summary['close_game_predictions'] = len(close_df)

    # Blowout star/bench performance
    blowout_star = blowout_df[blowout_df['role_tier'] == 'STAR']
    blowout_bench = blowout_df[blowout_df['role_tier'] == 'BENCH']

    summary['blowout_star_mae'] = blowout_star['abs_error'].mean() if len(blowout_star) > 0 else None
    summary['blowout_bench_mae'] = blowout_bench['abs_error'].mean() if len(blowout_bench) > 0 else None

    close_star = close_df[close_df['role_tier'] == 'STAR']
    summary['close_game_star_mae'] = close_star['abs_error'].mean() if len(close_star) > 0 else None

    # Game script MAE impact (compare to overall)
    overall_mae = df['abs_error'].mean()
    script_df = df[df['game_script_tier'].isin(['blowout', 'close_game'])]
    if len(script_df) > 0:
        script_mae = script_df['abs_error'].mean()
        summary['game_script_mae_impact'] = script_mae - overall_mae
    else:
        summary['game_script_mae_impact'] = None

    # Role metrics
    for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
        role_df = df[df['role_tier'] == role]
        summary[f'{role.lower()}_mae'] = role_df['abs_error'].mean() if len(role_df) > 0 else None
        summary[f'{role.lower()}_count'] = len(role_df)

    # Position defense metrics (need player position from another table)
    # For now, use position matchup factor correlation
    if 'position_matchup_factor' in df.columns:
        valid_pos = df[df['position_matchup_factor'].notna()]
        if len(valid_pos) > 0:
            summary['pos_factor_correlation'] = valid_pos['position_matchup_factor'].corr(valid_pos['actual_ppg'])
            # Compare factor groups
            high_factor = valid_pos[valid_pos['position_matchup_factor'] > 1.0]
            low_factor = valid_pos[valid_pos['position_matchup_factor'] <= 1.0]
            if len(high_factor) > 0 and len(low_factor) > 0:
                summary['pos_factor_mae_impact'] = high_factor['abs_error'].mean() - low_factor['abs_error'].mean()
            else:
                summary['pos_factor_mae_impact'] = None
        else:
            summary['pos_factor_correlation'] = None
            summary['pos_factor_mae_impact'] = None
    else:
        summary['pos_factor_correlation'] = None
        summary['pos_factor_mae_impact'] = None

    # Overall metrics
    summary['overall_mae'] = df['abs_error'].mean()
    summary['overall_rmse'] = np.sqrt((df['prediction_error'] ** 2).mean())
    summary['overall_bias'] = df['prediction_error'].mean()
    summary['ceiling_hit_rate'] = df['ceiling_hit'].mean() if 'ceiling_hit' in df.columns else None
    summary['top10_capture_rate'] = df['was_top10'].mean() if 'was_top10' in df.columns else None

    # Store summary
    ensure_monitoring_tables(conn)
    cursor = conn.cursor()

    # Build insert statement dynamically
    cols = [k for k in summary.keys() if k != 'week_ending']
    placeholders = ', '.join(['?' for _ in cols])
    col_names = ', '.join(cols)

    cursor.execute(f"""
        INSERT OR REPLACE INTO enrichment_weekly_summary
        (week_ending, {col_names})
        VALUES (?, {placeholders})
    """, [week_ending] + [summary[k] for k in cols])

    conn.commit()

    if verbose:
        print(f"Summary stored for week ending {week_ending}")
        print(f"  Total predictions: {summary['total_predictions']}")
        print(f"  Overall MAE: {summary['overall_mae']:.2f}")
        print(f"  B2B effect observed: {summary['b2b_effect_observed']:.1%}" if summary['b2b_effect_observed'] else "  B2B effect: N/A")

    return summary


# =============================================================================
# ALERT CHECKING
# =============================================================================

def check_alerts(
    conn: sqlite3.Connection,
    week_ending: str = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Check for alert conditions based on latest summary.

    Args:
        conn: Database connection
        week_ending: Week to check (defaults to most recent)
        verbose: Print progress

    Returns:
        List of triggered alerts
    """
    ensure_monitoring_tables(conn)
    cursor = conn.cursor()

    # Get latest summary
    if week_ending:
        cursor.execute("""
            SELECT * FROM enrichment_weekly_summary
            WHERE week_ending = ?
        """, [week_ending])
    else:
        cursor.execute("""
            SELECT * FROM enrichment_weekly_summary
            ORDER BY week_ending DESC LIMIT 1
        """)

    row = cursor.fetchone()
    if not row:
        if verbose:
            print("No summary data to check")
        return []

    # Convert to dict
    columns = [desc[0] for desc in cursor.description]
    summary = dict(zip(columns, row))

    triggered_alerts = []

    for threshold in ALERT_THRESHOLDS:
        metric_value = summary.get(threshold.metric)

        if metric_value is None:
            continue

        triggered = False
        if threshold.operator == 'gt':
            triggered = metric_value > threshold.threshold
        elif threshold.operator == 'lt':
            triggered = metric_value < threshold.threshold
        elif threshold.operator == 'abs_gt':
            triggered = abs(metric_value) > threshold.threshold

        if triggered:
            alert = {
                'alert_type': threshold.name,
                'severity': threshold.severity.value,
                'category': threshold.category,
                'message': threshold.message_template.format(value=metric_value),
                'metric_name': threshold.metric,
                'metric_value': metric_value,
                'threshold': threshold.threshold,
            }
            triggered_alerts.append(alert)

            # Store alert
            cursor.execute("""
                INSERT INTO enrichment_alerts
                (alert_type, severity, category, message, metric_name, metric_value, threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert['alert_type'], alert['severity'], alert['category'],
                alert['message'], alert['metric_name'], alert['metric_value'],
                alert['threshold']
            ))

            if verbose:
                print(f"[{alert['severity']}] {alert['message']}")

    conn.commit()

    if verbose and not triggered_alerts:
        print("No alerts triggered")

    return triggered_alerts


def get_active_alerts(conn: sqlite3.Connection) -> List[Dict]:
    """Get all active (unresolved) alerts."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT alert_type, severity, category, message, metric_value, created_at
        FROM enrichment_alerts
        WHERE is_active = 1
        ORDER BY created_at DESC
    """)

    return [
        {
            'alert_type': row[0],
            'severity': row[1],
            'category': row[2],
            'message': row[3],
            'metric_value': row[4],
            'created_at': row[5],
        }
        for row in cursor.fetchall()
    ]


def resolve_alert(conn: sqlite3.Connection, alert_type: str):
    """Resolve an alert by type."""
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE enrichment_alerts
        SET is_active = 0, resolved_at = CURRENT_TIMESTAMP
        WHERE alert_type = ? AND is_active = 1
    """, [alert_type])
    conn.commit()


# =============================================================================
# HYPOTHESIS VALIDATION
# =============================================================================

def validate_hypothesis(
    conn: sqlite3.Connection,
    hypothesis_id: str,
    days: int = 30,
    verbose: bool = False
) -> Dict:
    """
    Validate a specific hypothesis against recent data.

    Args:
        conn: Database connection
        hypothesis_id: Hypothesis identifier (e.g., 'R1', 'G1')
        days: Days of data to analyze
        verbose: Print progress

    Returns:
        Dict with validation results
    """
    # This would implement specific statistical tests for each hypothesis
    # For now, return placeholder structure

    return {
        'hypothesis_id': hypothesis_id,
        'days_analyzed': days,
        'sample_size': 0,
        'test_statistic': None,
        'p_value': None,
        'effect_size': None,
        'conclusion': 'NOT_IMPLEMENTED',
        'details': 'Hypothesis validation not yet implemented'
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enrichment monitoring')
    parser.add_argument('--daily', type=str, help='Run daily audit for date (YYYY-MM-DD)')
    parser.add_argument('--weekly', type=str, help='Calculate weekly summary for week ending (YYYY-MM-DD)')
    parser.add_argument('--check-alerts', action='store_true', help='Check for alerts')
    parser.add_argument('--show-alerts', action='store_true', help='Show active alerts')
    parser.add_argument('--backfill', type=int, help='Backfill audit log for N days')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_monitoring_tables(conn)

    if args.daily:
        count = populate_daily_audit_log(conn, args.daily, verbose=True)
        print(f"Populated {count} audit records for {args.daily}")

    elif args.weekly:
        summary = calculate_weekly_summary(conn, args.weekly, verbose=True)
        if summary:
            print(f"\nWeekly Summary for {args.weekly}:")
            for k, v in summary.items():
                if v is not None:
                    if isinstance(v, float):
                        print(f"  {k}: {v:.3f}")
                    else:
                        print(f"  {k}: {v}")

    elif args.check_alerts:
        alerts = check_alerts(conn, verbose=True)
        print(f"\n{len(alerts)} alert(s) triggered")

    elif args.show_alerts:
        alerts = get_active_alerts(conn)
        if alerts:
            print("Active Alerts:")
            for a in alerts:
                print(f"  [{a['severity']}] {a['message']} (created: {a['created_at']})")
        else:
            print("No active alerts")

    elif args.backfill:
        count = backfill_audit_log(conn, args.backfill, verbose=True)
        print(f"Backfilled {count} total audit records")

    else:
        print("Use --daily, --weekly, --check-alerts, --show-alerts, or --backfill")

    conn.close()
