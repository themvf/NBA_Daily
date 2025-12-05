#!/usr/bin/env python3
"""
Prediction Refresh System for NBA_Daily.

This module handles refreshing predictions when players are marked as OUT after
initial predictions have been generated. It removes OUT players' predictions
and recalculates teammate projections with injury adjustments.

Key Features:
- Delete predictions for newly OUT players
- Apply injury adjustments to boost teammates
- Maintain audit trail of refresh operations
- Transaction-safe with rollback on errors
"""

import sqlite3
from typing import Dict, List
from datetime import date
import injury_adjustment as ia


def refresh_predictions_for_date(
    game_date: str,
    conn: sqlite3.Connection
) -> Dict[str, any]:
    """
    Refresh predictions for a specific date by removing OUT players and applying injury adjustments.

    This function is designed for the workflow where:
    1. User generates predictions in the morning (all players healthy)
    2. Later in day, injury news comes out
    3. User marks player(s) as OUT
    4. User clicks "Refresh Predictions" button
    5. System removes OUT players and redistributes their points to teammates

    Args:
        game_date: Date in YYYY-MM-DD format
        conn: Database connection

    Returns:
        Dictionary with refresh summary:
        {
            'removed': int,  # Number of predictions deleted (OUT players)
            'adjusted': int,  # Number of predictions adjusted (teammates)
            'skipped': int,  # Number skipped (insufficient data)
            'affected_players': List[str],  # Names of OUT players
            'error': str | None  # Error message if any
        }
    """
    cursor = conn.cursor()

    try:
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")

        # Step 1: Get active injuries for this date
        active_injuries = ia.get_active_injuries(conn, check_return_dates=True)
        injured_player_ids = [inj['player_id'] for inj in active_injuries]
        injured_player_names = [inj['player_name'] for inj in active_injuries]

        if not injured_player_ids:
            # No injuries, nothing to refresh
            conn.rollback()
            return {
                'removed': 0,
                'adjusted': 0,
                'skipped': 0,
                'affected_players': [],
                'error': None
            }

        # Step 2: Check which OUT players have predictions for this date
        placeholders = ','.join('?' * len(injured_player_ids))
        cursor.execute(f"""
            SELECT player_id, player_name, team_name, projected_ppg
            FROM predictions
            WHERE game_date = ?
              AND player_id IN ({placeholders})
        """, [game_date] + injured_player_ids)

        out_players_with_predictions = cursor.fetchall()

        if not out_players_with_predictions:
            # OUT players don't have predictions (already filtered), nothing to do
            conn.rollback()
            return {
                'removed': 0,
                'adjusted': 0,
                'skipped': 0,
                'affected_players': injured_player_names,
                'error': None
            }

        # Step 3: Delete predictions for OUT players
        cursor.execute(f"""
            DELETE FROM predictions
            WHERE game_date = ?
              AND player_id IN ({placeholders})
        """, [game_date] + injured_player_ids)

        removed_count = cursor.rowcount

        # Update refresh audit trail for remaining predictions
        refresh_reason = f"Removed {removed_count} OUT player(s): {', '.join(injured_player_names)}"

        # Step 4: Apply injury adjustments to remaining predictions (teammates)
        # This redistributes the OUT players' expected points to teammates
        try:
            adjusted, skipped, _ = ia.apply_injury_adjustments(
                injured_player_ids,
                game_date,
                conn,
                min_historical_games=3,
                max_adjustment_pct=0.25
            )
        except Exception as e:
            # If injury adjustment fails, still commit the deletions
            print(f"Warning: Injury adjustment failed: {e}")
            adjusted = 0
            skipped = 0

        # Step 5: Update refresh metadata for affected predictions
        cursor.execute("""
            UPDATE predictions
            SET last_refreshed_at = CURRENT_TIMESTAMP,
                refresh_count = refresh_count + 1,
                refresh_reason = ?
            WHERE game_date = ?
        """, (refresh_reason, game_date))

        # Commit transaction
        conn.commit()

        return {
            'removed': removed_count,
            'adjusted': adjusted,
            'skipped': skipped,
            'affected_players': injured_player_names,
            'error': None
        }

    except Exception as e:
        # Rollback on any error
        conn.rollback()
        return {
            'removed': 0,
            'adjusted': 0,
            'skipped': 0,
            'affected_players': [],
            'error': str(e)
        }


def get_refresh_status(
    game_date: str,
    conn: sqlite3.Connection
) -> Dict[str, any]:
    """
    Get status of predictions for a date to show if refresh is needed.

    Args:
        game_date: Date in YYYY-MM-DD format
        conn: Database connection

    Returns:
        Dictionary with status information:
        {
            'predictions_count': int,  # Total predictions for date
            'out_players_with_predictions': List[Dict],  # OUT players still having predictions
            'needs_refresh': bool,  # True if OUT players have predictions
            'last_refreshed': str | None,  # Timestamp of last refresh
            'refresh_count': int  # Number of times refreshed
        }
    """
    cursor = conn.cursor()

    # Get total predictions count
    cursor.execute("""
        SELECT COUNT(*) FROM predictions WHERE game_date = ?
    """, (game_date,))
    pred_count = cursor.fetchone()[0]

    if pred_count == 0:
        return {
            'predictions_count': 0,
            'out_players_with_predictions': [],
            'needs_refresh': False,
            'last_refreshed': None,
            'refresh_count': 0
        }

    # Get active injuries
    active_injuries = ia.get_active_injuries(conn, check_return_dates=True)
    injured_player_ids = [inj['player_id'] for inj in active_injuries]

    if not injured_player_ids:
        # No injuries, no refresh needed
        return {
            'predictions_count': pred_count,
            'out_players_with_predictions': [],
            'needs_refresh': False,
            'last_refreshed': None,
            'refresh_count': 0
        }

    # Check which OUT players have predictions
    placeholders = ','.join('?' * len(injured_player_ids))
    cursor.execute(f"""
        SELECT player_id, player_name, team_name, projected_ppg
        FROM predictions
        WHERE game_date = ?
          AND player_id IN ({placeholders})
    """, [game_date] + injured_player_ids)

    out_with_preds = cursor.fetchall()

    # Get refresh metadata
    cursor.execute("""
        SELECT MAX(last_refreshed_at), MAX(refresh_count)
        FROM predictions
        WHERE game_date = ?
    """, (game_date,))

    last_refreshed, refresh_count = cursor.fetchone()

    return {
        'predictions_count': pred_count,
        'out_players_with_predictions': [
            {
                'player_id': row[0],
                'player_name': row[1],
                'team_name': row[2],
                'projected_ppg': row[3]
            }
            for row in out_with_preds
        ],
        'needs_refresh': len(out_with_preds) > 0,
        'last_refreshed': last_refreshed,
        'refresh_count': refresh_count or 0
    }


def reset_refresh_metadata(
    game_date: str,
    conn: sqlite3.Connection
) -> int:
    """
    Reset refresh metadata for a specific date.

    Useful for testing or if you want to clear the refresh history.

    Args:
        game_date: Date in YYYY-MM-DD format
        conn: Database connection

    Returns:
        Number of predictions updated
    """
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE predictions
        SET last_refreshed_at = NULL,
            refresh_count = 0,
            refresh_reason = NULL
        WHERE game_date = ?
    """, (game_date,))

    conn.commit()
    return cursor.rowcount
