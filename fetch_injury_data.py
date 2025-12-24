#!/usr/bin/env python3
"""
Automated Injury Data Fetcher for NBA_Daily.

Fetches current injury reports from balldontlie.io API and syncs to injury_list table.
Implements 3-tier name matching with persistent alias storage and atomic fetch locking.

Usage:
    import fetch_injury_data
    updated, new, skipped, errors = fetch_injury_data.fetch_current_injuries(conn)
"""

import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz
import time

# Import configuration
import injury_config as config

# Constants
BASE_URL = "https://api.balldontlie.io/v1"


def acquire_fetch_lock(conn: sqlite3.Connection, lock_by: str = "fetch_injury_data") -> bool:
    """
    Atomically acquire fetch lock to prevent concurrent fetches.

    Uses BEGIN IMMEDIATE transaction + compare-and-swap UPDATE to ensure
    only one process can fetch at a time. Respects cooldown period.

    Args:
        conn: Database connection
        lock_by: Identifier for who acquired the lock

    Returns:
        True if lock acquired, False if already locked or in cooldown
    """
    cursor = conn.cursor()

    try:
        # Use BEGIN IMMEDIATE for atomic lock acquisition
        conn.execute("BEGIN IMMEDIATE")

        # Try to acquire lock (compare-and-swap)
        cursor.execute(f"""
            UPDATE injury_fetch_lock
            SET locked = 1,
                locked_at = CURRENT_TIMESTAMP,
                locked_by = ?
            WHERE lock_id = 1
              AND (locked = 0 OR locked_at < datetime('now', '-{config.AUTO_FETCH_COOLDOWN_MINUTES} minutes'))
        """, (lock_by,))

        # Check if we acquired the lock (changes() returns rows affected)
        acquired = cursor.rowcount > 0

        if acquired:
            conn.commit()
            return True
        else:
            conn.rollback()
            return False

    except Exception as e:
        conn.rollback()
        print(f"Error acquiring lock: {e}")
        return False


def release_fetch_lock(conn: sqlite3.Connection) -> None:
    """
    Release the fetch lock.

    Args:
        conn: Database connection
    """
    try:
        conn.execute("""
            UPDATE injury_fetch_lock
            SET locked = 0,
                locked_at = NULL,
                locked_by = NULL
            WHERE lock_id = 1
        """)
        conn.commit()
    except Exception as e:
        print(f"Error releasing lock: {e}")


def map_team_id_to_abbrev(team_id: int, conn: sqlite3.Connection) -> str:
    """
    Map balldontlie.io team_id to NBA team abbreviation.

    Args:
        team_id: Team ID from balldontlie API
        conn: Database connection

    Returns:
        Team abbreviation (e.g., "LAL") or "UNK" if not found
    """
    # balldontlie.io team IDs (v1 API)
    # Source: https://docs.balldontlie.io/#teams
    TEAM_ID_MAP = {
        1: 'ATL',   # Atlanta Hawks
        2: 'BOS',   # Boston Celtics
        3: 'BKN',   # Brooklyn Nets
        4: 'CHA',   # Charlotte Hornets
        5: 'CHI',   # Chicago Bulls
        6: 'CLE',   # Cleveland Cavaliers
        7: 'DAL',   # Dallas Mavericks
        8: 'DEN',   # Denver Nuggets
        9: 'DET',   # Detroit Pistons
        10: 'GSW',  # Golden State Warriors
        11: 'HOU',  # Houston Rockets
        12: 'IND',  # Indiana Pacers
        13: 'LAC',  # LA Clippers
        14: 'LAL',  # Los Angeles Lakers
        15: 'MEM',  # Memphis Grizzlies
        16: 'MIA',  # Miami Heat
        17: 'MIL',  # Milwaukee Bucks
        18: 'MIN',  # Minnesota Timberwolves
        19: 'NOP',  # New Orleans Pelicans
        20: 'NYK',  # New York Knicks
        21: 'OKC',  # Oklahoma City Thunder
        22: 'ORL',  # Orlando Magic
        23: 'PHI',  # Philadelphia 76ers
        24: 'PHX',  # Phoenix Suns
        25: 'POR',  # Portland Trail Blazers
        26: 'SAC',  # Sacramento Kings
        27: 'SAS',  # San Antonio Spurs
        28: 'TOR',  # Toronto Raptors
        29: 'UTA',  # Utah Jazz
        30: 'WAS',  # Washington Wizards
    }

    return TEAM_ID_MAP.get(team_id, 'UNK')


def get_api_key(conn: sqlite3.Connection) -> Optional[str]:
    """
    Get balldontlie API key from Streamlit secrets or environment.

    Args:
        conn: Database connection (unused, kept for interface consistency)

    Returns:
        API key string or None if not found
    """
    try:
        import streamlit as st
        return st.secrets["balldontlie"]["API_KEY"]
    except:
        pass

    # Fallback: try reading from secrets.toml directly
    try:
        import toml
        import os
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            if "balldontlie" in secrets and "API_KEY" in secrets["balldontlie"]:
                return secrets["balldontlie"]["API_KEY"]
    except:
        pass

    # Fallback: environment variable
    import os
    return os.environ.get("BALLDONTLIE_API_KEY")


def fetch_injuries_from_api(api_key: str) -> List[Dict]:
    """
    Fetch injury data from balldontlie.io API with retry logic.

    Args:
        api_key: balldontlie.io API key

    Returns:
        List of injury records from API

    Raises:
        Exception: If API call fails after retries
    """
    headers = {"Authorization": api_key}

    for attempt in range(config.API_RETRY_COUNT):
        try:
            response = requests.get(
                f"{BASE_URL}/player_injuries",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            injuries = data.get('data', [])

            print(f"Fetched {len(injuries)} injuries from balldontlie.io")
            return injuries

        except requests.exceptions.RequestException as e:
            if attempt < config.API_RETRY_COUNT - 1:
                # Exponential backoff
                delay = config.API_RETRY_BASE_DELAY * (2 ** attempt)
                print(f"API call failed (attempt {attempt + 1}/{config.API_RETRY_COUNT}), "
                      f"retrying in {delay}s: {e}")
                time.sleep(delay)
            else:
                # Final attempt failed
                raise Exception(f"API call failed after {config.API_RETRY_COUNT} attempts: {e}")


def map_player_name_to_id(
    first_name: str,
    last_name: str,
    balldontlie_id: int,
    team_id: Optional[int],
    conn: sqlite3.Connection
) -> Tuple[Optional[int], float, str]:
    """
    Map player name to player_id using 3-tier fallback strategy.

    Tier 1: Check player_aliases table (persistent fuzzy match results)
    Tier 2: Exact match on players.full_name
    Tier 3: Fuzzy match with threshold → save to player_aliases

    Args:
        first_name: Player first name from API
        last_name: Player last name from API
        balldontlie_id: Player ID from balldontlie.io
        team_id: Team ID from balldontlie.io (optional)
        conn: Database connection

    Returns:
        Tuple of (player_id, confidence, match_method)
        - player_id: Matched player_id or None if not found
        - confidence: Match confidence (0.0-1.0)
        - match_method: "alias" | "exact" | "fuzzy" | "not_found"
    """
    cursor = conn.cursor()
    full_name = f"{first_name} {last_name}".strip()

    # Tier 1: Check alias table
    cursor.execute("""
        SELECT player_id, confidence
        FROM player_aliases
        WHERE alias_name = ? AND source = 'balldontlie'
        ORDER BY confidence DESC
        LIMIT 1
    """, (full_name,))

    result = cursor.fetchone()
    if result:
        player_id, confidence = result
        return player_id, confidence, "alias"

    # Tier 2: Exact match
    cursor.execute("""
        SELECT player_id
        FROM players
        WHERE full_name = ? AND is_active = 1
        LIMIT 1
    """, (full_name,))

    result = cursor.fetchone()
    if result:
        player_id = result[0]

        # Save to alias table for future
        cursor.execute("""
            INSERT OR IGNORE INTO player_aliases (alias_name, player_id, source, confidence)
            VALUES (?, ?, 'balldontlie', 1.0)
        """, (full_name, player_id))
        conn.commit()

        return player_id, 1.0, "exact"

    # Tier 3: Fuzzy match
    cursor.execute("""
        SELECT player_id, full_name
        FROM players
        WHERE is_active = 1
    """)

    best_match = None
    best_score = 0
    best_player_id = None

    for row in cursor.fetchall():
        db_player_id, db_full_name = row
        score = fuzz.ratio(full_name.lower(), db_full_name.lower())

        if score > best_score:
            best_score = score
            best_match = db_full_name
            best_player_id = db_player_id

    if best_score >= config.FUZZY_MATCH_THRESHOLD:
        confidence = best_score / 100.0

        # Save to alias table
        cursor.execute("""
            INSERT OR IGNORE INTO player_aliases (alias_name, player_id, source, confidence)
            VALUES (?, ?, 'balldontlie', ?)
        """, (full_name, best_player_id, confidence))
        conn.commit()

        return best_player_id, confidence, "fuzzy"

    # Not found
    return None, 0.0, "not_found"


def upsert_injury(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_name: str,
    status: str,
    injury_type: Optional[str],
    return_date: Optional[str],
    description: Optional[str],
    confidence: float
) -> bool:
    """
    Upsert injury record with manual override protection.

    Uses proper UPSERT (ON CONFLICT DO UPDATE) to preserve manual fields.
    Manual entries within 24 hours are not overwritten by automated updates.

    Args:
        conn: Database connection
        player_id: Player ID
        player_name: Player full name
        team_name: Team abbreviation
        status: Injury status (from config.BALLDONTLIE_STATUS_MAP)
        injury_type: Injury description
        return_date: Expected return date string
        description: Full injury description
        confidence: Match confidence (0.0-1.0)

    Returns:
        True if upserted successfully, False otherwise
    """
    cursor = conn.cursor()

    try:
        # Check if player already has an injury record
        cursor.execute("""
            SELECT injury_id, source, updated_at
            FROM injury_list
            WHERE player_id = ?
        """, (player_id,))
        existing = cursor.fetchone()

        if existing:
            injury_id, existing_source, updated_at = existing

            # Check manual override rule
            cursor.execute("""
                SELECT datetime(?, '-24 hours') < datetime(?)
            """, (updated_at, updated_at))
            is_recent_manual = (
                existing_source == 'manual' and
                cursor.fetchone()[0] == 1
            )

            if is_recent_manual:
                # Manual override active - don't update status, just refresh metadata
                cursor.execute("""
                    UPDATE injury_list
                    SET team_name = ?,
                        injury_type = ?,
                        confidence = ?,
                        last_fetched_at = CURRENT_TIMESTAMP,
                        expected_return_date = COALESCE(?, expected_return_date),
                        notes = COALESCE(?, notes),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE player_id = ?
                """, (team_name, injury_type, confidence, return_date, description, player_id))
            else:
                # Normal update - automated source can update everything
                cursor.execute("""
                    UPDATE injury_list
                    SET player_name = ?,
                        team_name = ?,
                        status = ?,
                        injury_type = ?,
                        confidence = ?,
                        last_fetched_at = CURRENT_TIMESTAMP,
                        expected_return_date = COALESCE(?, expected_return_date),
                        notes = COALESCE(?, notes),
                        source = 'automated',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE player_id = ?
                """, (player_name, team_name, status, injury_type, confidence,
                      return_date, description, player_id))
        else:
            # Insert new record
            cursor.execute("""
                INSERT INTO injury_list (
                    player_id, player_name, team_name, status, injury_type,
                    source, confidence, last_fetched_at, injury_date,
                    expected_return_date, notes, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'automated', ?, CURRENT_TIMESTAMP, DATE('now'), ?, ?, CURRENT_TIMESTAMP)
            """, (player_id, player_name, team_name, status, injury_type,
                  confidence, return_date, description))

        conn.commit()
        return True

    except Exception as e:
        print(f"Error upserting injury for {player_name}: {e}")
        conn.rollback()
        return False


def log_fetch_error(
    conn: sqlite3.Connection,
    player_name: str,
    team_name: Optional[str],
    error_message: str
) -> None:
    """
    Log fetch error to injury_fetch_errors table.

    Args:
        conn: Database connection
        player_name: Player name that failed
        team_name: Team name (optional)
        error_message: Error description
    """
    try:
        conn.execute("""
            INSERT INTO injury_fetch_errors (player_name, team_name, error_message)
            VALUES (?, ?, ?)
        """, (player_name, team_name, error_message))
        conn.commit()
    except Exception as e:
        print(f"Error logging fetch error: {e}")


def fetch_current_injuries(conn: sqlite3.Connection) -> Tuple[int, int, int, List[str]]:
    """
    Main entry point: Fetch injury data from balldontlie.io and sync to database.

    Implements:
    - Atomic fetch lock (prevents concurrent fetches)
    - 3-tier name matching (alias → exact → fuzzy)
    - Proper UPSERT with manual override protection
    - Error logging to database
    - Cooldown enforcement

    Args:
        conn: Database connection

    Returns:
        Tuple of (updated_count, new_count, skipped_count, error_messages)
        - updated_count: Number of existing injuries updated
        - new_count: Number of new injuries added
        - skipped_count: Number of players that couldn't be matched
        - error_messages: List of error message strings
    """
    # Check if automation is enabled
    if not config.INJURY_AUTOMATION_ENABLED:
        return (0, 0, 0, ["Injury automation is disabled in config"])

    # Try to acquire lock
    if not acquire_fetch_lock(conn):
        return (0, 0, 0, [f"Fetch skipped: cooldown active or another fetch in progress"])

    error_messages = []
    updated_count = 0
    new_count = 0
    skipped_count = 0

    try:
        # Get API key
        api_key = get_api_key(conn)
        if not api_key:
            error_messages.append("No API key found. Check Streamlit secrets or environment.")
            return (0, 0, 0, error_messages)

        # Fetch injuries from API
        injuries = fetch_injuries_from_api(api_key)

        if not injuries:
            return (0, 0, 0, ["No injuries returned from API (may be normal if no active injuries)"])

        # Process each injury
        for injury_record in injuries:
            player_data = injury_record.get('player', {})
            first_name = player_data.get('first_name', '')
            last_name = player_data.get('last_name', '')
            full_name = f"{first_name} {last_name}".strip()
            balldontlie_id = player_data.get('id')
            team_id = player_data.get('team_id')

            # Map status
            api_status = injury_record.get('status', '')
            status = config.BALLDONTLIE_STATUS_MAP.get(api_status, api_status.lower())

            # Get injury details
            return_date_str = injury_record.get('return_date')
            description = injury_record.get('description', '')

            # Extract injury type from description (first sentence)
            injury_type = description.split('.')[0] if description else None

            # Map player name to player_id
            player_id, confidence, match_method = map_player_name_to_id(
                first_name, last_name, balldontlie_id, team_id, conn
            )

            if player_id is None:
                # Could not match player
                skipped_count += 1
                error_msg = f"Player not found: {full_name} (balldontlie ID: {balldontlie_id})"
                error_messages.append(error_msg)
                log_fetch_error(conn, full_name, None, error_msg)
                continue

            # Map team_id to abbreviation
            team_name = map_team_id_to_abbrev(team_id, conn) if team_id else "UNK"

            # Check if this is an update or new record
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM injury_list WHERE player_id = ?", (player_id,))
            is_update = cursor.fetchone() is not None

            # Upsert injury
            success = upsert_injury(
                conn, player_id, full_name, team_name, status,
                injury_type, return_date_str, description, confidence
            )

            if success:
                if is_update:
                    updated_count += 1
                else:
                    new_count += 1
            else:
                error_messages.append(f"Failed to upsert injury for {full_name}")

        return (updated_count, new_count, skipped_count, error_messages)

    except Exception as e:
        error_messages.append(f"Fatal error during fetch: {e}")
        return (0, 0, 0, error_messages)

    finally:
        # Always release lock
        release_fetch_lock(conn)


if __name__ == "__main__":
    # Test the module
    print("Testing fetch_injury_data module...")
    print("This requires a database connection.")
    print("Run from your main application, not standalone.")
