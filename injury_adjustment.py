#!/usr/bin/env python3
"""
Injury-aware prediction adjustment system for NBA_Daily.

This module allows users to manually mark players as injured/out and automatically
adjusts teammate predictions based on historical injury impact data.

Key features:
- Manual injury detection (user marks players as OUT)
- Historical PPG boost calculation using injury_impact_analytics
- Additive adjustments for multiple injuries (with 25% cap)
- Idempotent re-adjustments (always uses original baseline)
- Full audit trail (preserves original predictions)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sqlite3
import pandas as pd
import json

# Import existing injury impact analytics module
import injury_impact_analytics as iia


@dataclass
class AdjustmentRecord:
    """Record of a single prediction adjustment."""
    player_id: int
    player_name: str
    team_name: str
    original_ppg: float
    adjusted_ppg: float
    adjustment_amount: float
    injured_players: List[str]  # Names of injured players causing boost
    reason: str


def apply_injury_adjustments(
    injured_player_ids: List[int],
    game_date: str,
    conn: sqlite3.Connection,
    min_historical_games: int = 3,
    max_adjustment_pct: float = 0.25
) -> Tuple[int, int, List[AdjustmentRecord]]:
    """
    Adjust teammate predictions based on injured players.

    Args:
        injured_player_ids: List of player IDs who are OUT
        game_date: Game date in YYYY-MM-DD format
        conn: Database connection
        min_historical_games: Minimum absences required for reliable data
        max_adjustment_pct: Maximum % increase allowed (0.25 = 25%)

    Returns:
        Tuple of (adjusted_count, skipped_count, adjustment_records)
    """
    cursor = conn.cursor()

    adjusted_count = 0
    skipped_count = 0
    adjustment_records = []

    # Get all predictions for this date
    cursor.execute("""
        SELECT prediction_id, player_id, player_name, team_id, team_name,
               projected_ppg, proj_floor, proj_ceiling, proj_confidence,
               injury_adjusted, original_projected_ppg, original_proj_floor,
               original_proj_ceiling, original_proj_confidence
        FROM predictions
        WHERE game_date = ?
    """, (game_date,))

    predictions = cursor.fetchall()

    if not predictions:
        return 0, 0, []

    # Calculate teammate adjustments for each injured player
    teammate_adjustments = {}  # {teammate_id: total_ppg_boost}
    teammate_historical_games = {}  # {teammate_id: max_games_apart}
    injured_player_names = {}  # {player_id: player_name}

    for injured_id in injured_player_ids:
        # Get injured player's name
        cursor.execute("SELECT player_name FROM predictions WHERE player_id = ? LIMIT 1", (injured_id,))
        result = cursor.fetchone()
        if result:
            injured_player_names[injured_id] = result[0]

        # Calculate teammate redistribution
        try:
            impacts = iia.calculate_teammate_redistribution(
                conn,
                injured_id,
                season="2025-26",
                min_games=min_historical_games
            )

            # Accumulate adjustments for teammates who benefited
            for impact in impacts:
                if impact.pts_delta > 0 and impact.games_apart >= min_historical_games:
                    teammate_id = impact.teammate_id

                    if teammate_id not in teammate_adjustments:
                        teammate_adjustments[teammate_id] = 0.0
                        teammate_historical_games[teammate_id] = 0

                    teammate_adjustments[teammate_id] += impact.pts_delta
                    teammate_historical_games[teammate_id] = max(
                        teammate_historical_games[teammate_id],
                        impact.games_apart
                    )

        except Exception as e:
            print(f"Warning: Could not calculate impact for player {injured_id}: {e}")
            continue

    # Apply adjustments to predictions
    for pred in predictions:
        (prediction_id, player_id, player_name, team_id, team_name,
         current_projected_ppg, current_floor, current_ceiling, current_confidence,
         already_adjusted, orig_ppg, orig_floor, orig_ceiling, orig_conf) = pred

        # Skip if this player is one of the injured players
        if player_id in injured_player_ids:
            skipped_count += 1
            continue

        # Check if this player has a teammate adjustment
        if player_id not in teammate_adjustments:
            skipped_count += 1
            continue

        # Determine baseline values (use original if already adjusted before)
        if already_adjusted and orig_ppg is not None:
            baseline_ppg = orig_ppg
            baseline_floor = orig_floor
            baseline_ceiling = orig_ceiling
            baseline_confidence = orig_conf
        else:
            baseline_ppg = current_projected_ppg
            baseline_floor = current_floor
            baseline_ceiling = current_ceiling
            baseline_confidence = current_confidence

        # Calculate adjustment with cap
        raw_adjustment = teammate_adjustments[player_id]
        max_allowed = baseline_ppg * max_adjustment_pct
        capped_adjustment = min(raw_adjustment, max_allowed)

        # Calculate new projection values
        new_projected_ppg = baseline_ppg + capped_adjustment

        # Scale floor/ceiling proportionally
        adjustment_ratio = new_projected_ppg / baseline_ppg
        new_floor = baseline_floor * adjustment_ratio
        new_ceiling = baseline_ceiling * adjustment_ratio

        # Adjust confidence based on data quality
        historical_games = teammate_historical_games[player_id]
        if historical_games >= 5:
            confidence_adjustment = 0.05  # +5% for high-quality data
        elif historical_games >= 3:
            confidence_adjustment = 0.0  # No change for moderate data
        else:
            confidence_adjustment = -0.03  # -3% for sparse data

        new_confidence = min(0.95, max(0.30, baseline_confidence + confidence_adjustment))

        # Create injured player IDs JSON array
        injured_ids_json = json.dumps(injured_player_ids)

        # Update database
        if not already_adjusted:
            # First time adjusting - store originals
            cursor.execute("""
                UPDATE predictions
                SET projected_ppg = ?,
                    proj_floor = ?,
                    proj_ceiling = ?,
                    proj_confidence = ?,
                    injury_adjusted = 1,
                    injury_adjustment_amount = ?,
                    injured_player_ids = ?,
                    original_projected_ppg = ?,
                    original_proj_floor = ?,
                    original_proj_ceiling = ?,
                    original_proj_confidence = ?
                WHERE prediction_id = ?
            """, (
                new_projected_ppg, new_floor, new_ceiling, new_confidence,
                capped_adjustment, injured_ids_json,
                baseline_ppg, baseline_floor, baseline_ceiling, baseline_confidence,
                prediction_id
            ))
        else:
            # Re-adjusting - just update current values, keep originals
            cursor.execute("""
                UPDATE predictions
                SET projected_ppg = ?,
                    proj_floor = ?,
                    proj_ceiling = ?,
                    proj_confidence = ?,
                    injury_adjustment_amount = ?,
                    injured_player_ids = ?
                WHERE prediction_id = ?
            """, (
                new_projected_ppg, new_floor, new_ceiling, new_confidence,
                capped_adjustment, injured_ids_json,
                prediction_id
            ))

        adjusted_count += 1

        # Build injured players list for this record
        injured_names = [injured_player_names.get(iid, f"Player {iid}")
                        for iid in injured_player_ids]

        adjustment_records.append(AdjustmentRecord(
            player_id=player_id,
            player_name=player_name,
            team_name=team_name,
            original_ppg=baseline_ppg,
            adjusted_ppg=new_projected_ppg,
            adjustment_amount=capped_adjustment,
            injured_players=injured_names,
            reason=f"Boost from {len(injured_player_ids)} teammate(s) out"
        ))

    conn.commit()
    return adjusted_count, skipped_count, adjustment_records


def preview_adjustments(
    injured_player_ids: List[int],
    game_date: str,
    conn: sqlite3.Connection,
    min_historical_games: int = 3
) -> pd.DataFrame:
    """
    Generate preview of adjustments WITHOUT applying them.

    Args:
        injured_player_ids: List of player IDs who are OUT
        game_date: Game date in YYYY-MM-DD format
        conn: Database connection
        min_historical_games: Minimum absences required

    Returns:
        DataFrame with columns: Player, Team, Current Projection, Adjustment,
        New Projection, Injured Players
    """
    cursor = conn.cursor()

    # Get injured player names
    injured_player_names = {}
    for injured_id in injured_player_ids:
        cursor.execute("SELECT player_name FROM predictions WHERE player_id = ? LIMIT 1", (injured_id,))
        result = cursor.fetchone()
        if result:
            injured_player_names[injured_id] = result[0]

    # Calculate teammate adjustments
    teammate_adjustments = {}

    for injured_id in injured_player_ids:
        try:
            impacts = iia.calculate_teammate_redistribution(
                conn,
                injured_id,
                season="2025-26",
                min_games=min_historical_games
            )

            for impact in impacts:
                if impact.pts_delta > 0 and impact.games_apart >= min_historical_games:
                    if impact.teammate_id not in teammate_adjustments:
                        teammate_adjustments[impact.teammate_id] = 0.0
                    teammate_adjustments[impact.teammate_id] += impact.pts_delta

        except Exception as e:
            print(f"Warning: Could not calculate impact for player {injured_id}: {e}")
            continue

    if not teammate_adjustments:
        return pd.DataFrame()

    # Get predictions for affected teammates
    cursor.execute("""
        SELECT player_id, player_name, team_name, projected_ppg,
               injury_adjusted, original_projected_ppg
        FROM predictions
        WHERE game_date = ?
          AND player_id IN ({})
    """.format(','.join('?' * len(teammate_adjustments))),
    [game_date] + list(teammate_adjustments.keys()))

    predictions = cursor.fetchall()

    preview_data = []
    for pred in predictions:
        player_id, player_name, team_name, current_ppg, already_adjusted, orig_ppg = pred

        # Use original if already adjusted, otherwise use current
        baseline_ppg = orig_ppg if already_adjusted and orig_ppg is not None else current_ppg

        raw_adjustment = teammate_adjustments[player_id]
        max_allowed = baseline_ppg * 0.25
        capped_adjustment = min(raw_adjustment, max_allowed)

        new_ppg = baseline_ppg + capped_adjustment

        injured_names = ', '.join([injured_player_names.get(iid, f"Player {iid}")
                                  for iid in injured_player_ids])

        preview_data.append({
            'Player': player_name,
            'Team': team_name,
            'Current Projection': f"{baseline_ppg:.1f}",
            'Adjustment': f"+{capped_adjustment:.1f}",
            'New Projection': f"{new_ppg:.1f}",
            'Injured Players': injured_names
        })

    return pd.DataFrame(preview_data)


def reset_adjustments(
    game_date: str,
    conn: sqlite3.Connection
) -> int:
    """
    Revert all predictions for a date back to original values.

    Args:
        game_date: Game date in YYYY-MM-DD format
        conn: Database connection

    Returns:
        Number of predictions reset
    """
    cursor = conn.cursor()

    # Restore original values
    cursor.execute("""
        UPDATE predictions
        SET projected_ppg = original_projected_ppg,
            proj_floor = original_proj_floor,
            proj_ceiling = original_proj_ceiling,
            proj_confidence = original_proj_confidence,
            injury_adjusted = 0,
            injury_adjustment_amount = 0.0,
            injured_player_ids = NULL
        WHERE game_date = ?
          AND injury_adjusted = 1
          AND original_projected_ppg IS NOT NULL
    """, (game_date,))

    reset_count = cursor.rowcount
    conn.commit()

    return reset_count


def get_adjusted_predictions_summary(
    game_date: str,
    conn: sqlite3.Connection
) -> pd.DataFrame:
    """
    Get before/after comparison for adjusted predictions.

    Args:
        game_date: Game date in YYYY-MM-DD format
        conn: Database connection

    Returns:
        DataFrame with Original and Adjusted columns for comparison
    """
    query = """
        SELECT
            player_name as Player,
            team_name as Team,
            original_projected_ppg,
            projected_ppg,
            original_proj_floor,
            original_proj_ceiling,
            proj_floor,
            proj_ceiling,
            injury_adjustment_amount,
            injured_player_ids
        FROM predictions
        WHERE game_date = ?
          AND injury_adjusted = 1
        ORDER BY injury_adjustment_amount DESC
    """

    df = pd.read_sql_query(query, conn, params=[game_date])

    if df.empty:
        return pd.DataFrame()

    # Format for display
    df['Original PPG'] = df['original_projected_ppg'].apply(lambda x: f"{x:.1f}")
    df['Adjusted PPG'] = df['projected_ppg'].apply(lambda x: f"{x:.1f}")
    df['Original Range'] = df.apply(
        lambda row: f"{row['original_proj_floor']:.1f}-{row['original_proj_ceiling']:.1f}",
        axis=1
    )
    df['New Range'] = df.apply(
        lambda row: f"{row['proj_floor']:.1f}-{row['proj_ceiling']:.1f}",
        axis=1
    )
    df['Boost'] = df['injury_adjustment_amount'].apply(lambda x: f"+{x:.1f}")

    # Parse injured player IDs (JSON)
    def format_injured_players(ids_json):
        try:
            ids = json.loads(ids_json) if ids_json else []
            return f"{len(ids)} player(s)"
        except:
            return "N/A"

    df['Injured Players'] = df['injured_player_ids'].apply(format_injured_players)

    # Select final columns
    return df[['Player', 'Team', 'Original PPG', 'Adjusted PPG',
               'Original Range', 'New Range', 'Boost', 'Injured Players']]
