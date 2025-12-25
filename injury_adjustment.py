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
from datetime import date, datetime
import sqlite3
import pandas as pd
import json

# Import existing injury impact analytics module
import injury_impact_analytics as iia


@dataclass
class InjuryRecord:
    """Record of a player on the injury list."""
    injury_id: Optional[int]
    player_id: int
    player_name: str
    team_name: str
    injury_date: str
    expected_return_date: Optional[str]
    status: str  # 'active', 'returned', 'questionable'
    notes: Optional[str]


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
    max_adjustment_pct: float = 0.25,
    status_aware: bool = True
) -> Tuple[int, int, List[AdjustmentRecord]]:
    """
    Adjust teammate predictions based on injured players with status-aware scaling.

    When status_aware=True, adjustments are scaled based on injury status:
    - OUT (0.0 play prob): 100% adjustment (player excluded from predictions)
    - DOUBTFUL (0.25 play prob): 100% adjustment (player excluded)
    - QUESTIONABLE (0.5 play prob): 50% adjustment (player included but risky)
    - PROBABLE (0.8 play prob): 20% adjustment (player included, minimal boost)

    Args:
        injured_player_ids: List of player IDs who are injured
        game_date: Game date in YYYY-MM-DD format
        conn: Database connection
        min_historical_games: Minimum absences required for reliable data
        max_adjustment_pct: Maximum % increase allowed (0.25 = 25%)
        status_aware: If True, scale adjustments based on injury status play probability

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
    injury_status_multipliers = {}  # {player_id: adjustment_multiplier}

    # Fetch injury statuses if status_aware mode is enabled
    if status_aware:
        try:
            import injury_config as config
            # Get statuses for all injured players
            placeholders = ','.join('?' * len(injured_player_ids))
            cursor.execute(f"""
                SELECT player_id, status, confidence
                FROM injury_list
                WHERE player_id IN ({placeholders})
            """, injured_player_ids)

            for row in cursor.fetchall():
                player_id, status, confidence = row
                multiplier = config.get_adjustment_multiplier(status)
                injury_status_multipliers[player_id] = multiplier
        except Exception as e:
            print(f"Warning: Could not load injury statuses, using full adjustments: {e}")
            # Fallback: treat all as full adjustment (1.0)
            injury_status_multipliers = {pid: 1.0 for pid in injured_player_ids}
    else:
        # Status-aware disabled: treat all as full adjustment
        injury_status_multipliers = {pid: 1.0 for pid in injured_player_ids}

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
            # Scale by injury status multiplier (1.0 for OUT, 0.5 for QUESTIONABLE, etc.)
            status_multiplier = injury_status_multipliers.get(injured_id, 1.0)

            for impact in impacts:
                if impact.pts_delta > 0 and impact.games_apart >= min_historical_games:
                    teammate_id = impact.teammate_id

                    if teammate_id not in teammate_adjustments:
                        teammate_adjustments[teammate_id] = 0.0
                        teammate_historical_games[teammate_id] = 0

                    # Scale adjustment by injury status
                    scaled_delta = impact.pts_delta * status_multiplier
                    teammate_adjustments[teammate_id] += scaled_delta
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


# ============================================================================
# PERSISTENT INJURY LIST MANAGEMENT
# ============================================================================

def create_injury_list_table(conn: sqlite3.Connection) -> None:
    """Create the injury_list table if it doesn't exist, and add new columns if missing."""
    cursor = conn.cursor()

    # Create table with full schema (for new installations)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS injury_list (
            injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_name TEXT NOT NULL,
            injury_date TEXT NOT NULL,
            expected_return_date TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            injury_type TEXT,
            source TEXT DEFAULT 'manual',
            confidence REAL DEFAULT 1.0,
            last_fetched_at TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(player_id, status)
        )
    """)

    # For existing tables, check if columns exist before adding
    # Get current column names
    cursor.execute("PRAGMA table_info(injury_list)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Columns we want to ensure exist
    new_columns = [
        ("injury_type", "TEXT"),
        ("source", "TEXT DEFAULT 'manual'"),
        ("confidence", "REAL DEFAULT 1.0"),
        ("last_fetched_at", "TEXT")
    ]

    for col_name, col_def in new_columns:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE injury_list ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                # Column already exists (race condition), skip
                pass

    # Create index for quick lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_injury_list_status
        ON injury_list(status)
    """)

    conn.commit()


def create_injury_fetch_lock_table(conn: sqlite3.Connection) -> None:
    """Create the injury_fetch_lock table if it doesn't exist."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS injury_fetch_lock (
            lock_id INTEGER PRIMARY KEY CHECK (lock_id = 1),
            locked INTEGER DEFAULT 0,
            locked_at TEXT,
            locked_by TEXT
        )
    """)

    # Insert initial lock record
    cursor.execute("""
        INSERT OR IGNORE INTO injury_fetch_lock (lock_id, locked)
        VALUES (1, 0)
    """)

    conn.commit()


def create_player_aliases_table(conn: sqlite3.Connection) -> None:
    """Create the player_aliases table if it doesn't exist."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_aliases (
            alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias_name TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(alias_name, source)
        )
    """)

    # Create index for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_player_aliases_name
        ON player_aliases(alias_name)
    """)

    conn.commit()


def add_to_injury_list(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_name: str,
    status: str = 'out',
    injury_type: Optional[str] = None,
    expected_return_date: Optional[str] = None,
    notes: Optional[str] = None,
    source: str = 'manual'
) -> int:
    """
    Add a player to the injury list with proper UPSERT.

    Uses ON CONFLICT DO UPDATE to preserve existing fields like created_at
    when updating an existing injury record.

    Args:
        conn: Database connection
        player_id: Player ID
        player_name: Player full name
        team_name: Team name
        status: Injury status ('out', 'doubtful', 'questionable', 'probable', 'day-to-day')
        injury_type: Description of injury (e.g., 'Ankle Sprain', 'Rest')
        expected_return_date: Expected return date (YYYY-MM-DD) or None for indefinite
        notes: Optional notes about the injury
        source: Data source ('manual' or 'automated')

    Returns:
        injury_id of the created/updated record
    """
    cursor = conn.cursor()
    today = date.today().strftime('%Y-%m-%d')

    # First, check if player already has an injury record
    cursor.execute("SELECT injury_id FROM injury_list WHERE player_id = ?", (player_id,))
    existing = cursor.fetchone()

    if existing:
        # Update existing record
        cursor.execute("""
            UPDATE injury_list
            SET player_name = ?,
                team_name = ?,
                status = ?,
                injury_type = ?,
                expected_return_date = ?,
                notes = COALESCE(?, notes),
                source = ?,
                confidence = 1.0,
                updated_at = CURRENT_TIMESTAMP
            WHERE player_id = ?
        """, (player_name, team_name, status, injury_type, expected_return_date,
              notes, source, player_id))
        conn.commit()
        return existing[0]
    else:
        # Insert new record
        cursor.execute("""
            INSERT INTO injury_list (
                player_id, player_name, team_name, injury_date,
                expected_return_date, status, injury_type, notes,
                source, confidence, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, CURRENT_TIMESTAMP)
        """, (player_id, player_name, team_name, today, expected_return_date,
              status, injury_type, notes, source))
        conn.commit()
        return cursor.lastrowid


def remove_from_injury_list(
    conn: sqlite3.Connection,
    player_id: int
) -> bool:
    """
    Mark a player as returned (status = 'returned').

    Updates any non-returned injury status to 'returned', indicating the player
    is no longer injured and available to play.

    Args:
        conn: Database connection
        player_id: Player ID to mark as returned

    Returns:
        True if player was found and updated, False otherwise
    """
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE injury_list
        SET status = 'returned',
            updated_at = CURRENT_TIMESTAMP
        WHERE player_id = ? AND status != 'returned'
    """, (player_id,))

    conn.commit()
    return cursor.rowcount > 0


def get_active_injuries(
    conn: sqlite3.Connection,
    check_return_dates: bool = True,
    status_filter: Optional[List[str]] = None
) -> List[Dict]:
    """
    Get currently injured players filtered by status.

    By default, returns players whose injury status indicates they should be
    excluded from predictions (play_probability < PREDICTION_EXCLUSION_THRESHOLD).

    Args:
        conn: Database connection
        check_return_dates: If True, filter out players past their return date
        status_filter: List of statuses to include. If None, uses default filter
                      based on PREDICTION_EXCLUSION_THRESHOLD from injury_config.
                      Default filters: ['out', 'doubtful'] (play_prob < 0.3)

    Returns:
        List of injury records as dictionaries with fields:
        - injury_id, player_id, player_name, team_name
        - injury_date, expected_return_date, status
        - injury_type, source, confidence, notes
    """
    # Default filter: statuses where play probability < 0.3 (out, doubtful)
    if status_filter is None:
        try:
            import injury_config as config
            # Filter statuses below exclusion threshold
            status_filter = [
                status for status, prob in config.STATUS_PLAY_PROBABILITY.items()
                if prob < config.PREDICTION_EXCLUSION_THRESHOLD
            ]
        except ImportError:
            # Fallback if config not available
            status_filter = ['out', 'doubtful']

    # Handle empty status filter - return empty list immediately
    if not status_filter:
        return []

    # Build query with status filter
    placeholders = ','.join('?' * len(status_filter))

    # Try new schema first (with injury_type, source, confidence)
    try:
        query = f"""
            SELECT injury_id, player_id, player_name, team_name,
                   injury_date, expected_return_date, status,
                   injury_type, source, confidence, notes
            FROM injury_list
            WHERE status IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=tuple(status_filter))
    except Exception:
        # Fallback to old schema (without new columns and only 'active' status)
        query = """
            SELECT injury_id, player_id, player_name, team_name,
                   injury_date, expected_return_date, status, notes
            FROM injury_list
            WHERE status = 'active'
        """
        df = pd.read_sql_query(query, conn)

        # Add missing columns with default values for backward compatibility
        df['injury_type'] = None
        df['source'] = 'manual'
        df['confidence'] = 1.0
        # Map old 'active' status to new 'out' status for consistency
        df['status'] = df['status'].replace('active', 'out')

    if df.empty:
        return []

    # Filter by return date if requested
    if check_return_dates:
        today = date.today().strftime('%Y-%m-%d')
        # Keep players with no return date (indefinite) or return date >= today
        df = df[(df['expected_return_date'].isna()) | (df['expected_return_date'] >= today)]

    return df.to_dict('records')


def update_injury_return_date(
    conn: sqlite3.Connection,
    player_id: int,
    new_return_date: Optional[str]
) -> bool:
    """
    Update the expected return date for an injured player.

    Args:
        conn: Database connection
        player_id: Player ID
        new_return_date: New return date (YYYY-MM-DD) or None for indefinite

    Returns:
        True if updated successfully
    """
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE injury_list
        SET expected_return_date = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE player_id = ? AND status != 'returned'
    """, (new_return_date, player_id))

    conn.commit()
    return cursor.rowcount > 0


def get_injury_list_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Get formatted summary of injury list for display.

    Returns:
        DataFrame with columns: Player, Team, Injury Date, Return Date, Status, Notes
    """
    query = """
        SELECT
            player_name as Player,
            team_name as Team,
            injury_date as "Injury Date",
            COALESCE(expected_return_date, 'Indefinite') as "Return Date",
            status as Status,
            COALESCE(notes, '') as Notes,
            player_id
        FROM injury_list
        WHERE status IN ('active', 'questionable')
        ORDER BY injury_date DESC
    """

    df = pd.read_sql_query(query, conn)
    return df
