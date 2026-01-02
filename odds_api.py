#!/usr/bin/env python3
"""
The Odds API Integration for NBA_Daily.

Fetches FanDuel player points over/under lines and compares them with our projections.
Uses The Odds API (theoddsapi.com) with a 500 requests/month budget.

Usage:
    import odds_api
    result = odds_api.fetch_fanduel_lines_for_date(conn, game_date)
"""

import sqlite3
import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from zoneinfo import ZoneInfo
from rapidfuzz import fuzz
import time
import os

# Constants
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKET = "player_points"
BOOKMAKER = "fanduel"
MAX_API_REQUESTS_PER_MONTH = 500
BUDGET_WARNING_THRESHOLD = 0.80  # Warn at 80%
BUDGET_BLOCK_THRESHOLD = 0.95   # Block at 95%
FUZZY_MATCH_THRESHOLD = 85      # Minimum similarity score


def get_api_key() -> Optional[str]:
    """
    Get The Odds API key from Streamlit secrets.

    Follows the same 3-tier pattern as fetch_injury_data.py:
    1. st.secrets (Streamlit Cloud)
    2. secrets.toml file (local development)
    3. Environment variable (fallback)

    Returns:
        API key string or None if not configured
    """
    # Tier 1: Streamlit secrets (production)
    try:
        import streamlit as st
        return st.secrets["theoddsapi"]["API_KEY"]
    except Exception:
        pass

    # Tier 2: Local secrets.toml file
    try:
        import toml
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            if "theoddsapi" in secrets and "API_KEY" in secrets["theoddsapi"]:
                return secrets["theoddsapi"]["API_KEY"]
    except Exception:
        pass

    # Tier 3: Environment variable
    return os.environ.get("THEODDSAPI_KEY")


def create_odds_tables(conn: sqlite3.Connection) -> None:
    """Create tables for odds tracking if they don't exist."""
    cursor = conn.cursor()

    # Fetch log for API budget tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_fetch_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_date TEXT NOT NULL,
            game_date TEXT NOT NULL,
            events_fetched INTEGER DEFAULT 0,
            players_matched INTEGER DEFAULT 0,
            api_requests_used INTEGER DEFAULT 0,
            remaining_requests INTEGER DEFAULT NULL,
            error_message TEXT DEFAULT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Player name aliases for FanDuel name mapping
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_player_aliases (
            alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fanduel_name TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            confidence REAL DEFAULT 1.0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(fanduel_name)
        )
    """)

    # Index for fast lookup
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_odds_fetch_log_date
        ON odds_fetch_log(game_date)
    """)

    conn.commit()


def upgrade_predictions_for_fanduel(conn: sqlite3.Connection) -> None:
    """Add FanDuel columns to predictions table if they don't exist."""
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # New columns to add
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


def get_monthly_api_usage(conn: sqlite3.Connection) -> int:
    """
    Get total API requests used this month.

    Returns:
        Number of API requests used in current calendar month
    """
    cursor = conn.cursor()

    # Get first day of current month
    today = date.today()
    first_of_month = today.replace(day=1).strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT COALESCE(SUM(api_requests_used), 0)
        FROM odds_fetch_log
        WHERE fetch_date >= ?
    """, (first_of_month,))

    return cursor.fetchone()[0]


def should_fetch_odds(conn: sqlite3.Connection, game_date: date) -> Tuple[bool, str]:
    """
    Determine if we should fetch odds for this date.

    Checks:
    1. API key is configured
    2. Monthly budget not exceeded
    3. Haven't already fetched for this date

    Returns:
        (should_fetch: bool, reason: str)
    """
    # Check API key
    api_key = get_api_key()
    if not api_key:
        return False, "API key not configured (add [theoddsapi] section to secrets.toml)"

    # Check monthly budget
    monthly_usage = get_monthly_api_usage(conn)
    if monthly_usage >= MAX_API_REQUESTS_PER_MONTH * BUDGET_BLOCK_THRESHOLD:
        return False, f"API budget nearly exhausted ({monthly_usage}/{MAX_API_REQUESTS_PER_MONTH})"

    # Check if already fetched today for this game date
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM odds_fetch_log
        WHERE game_date = ?
        AND fetch_date = ?
        AND error_message IS NULL
    """, (str(game_date), str(date.today())))

    if cursor.fetchone()[0] > 0:
        return False, "Already fetched odds for this date today"

    # Budget warning (but still allow)
    if monthly_usage >= MAX_API_REQUESTS_PER_MONTH * BUDGET_WARNING_THRESHOLD:
        return True, f"Warning: API budget at {monthly_usage}/{MAX_API_REQUESTS_PER_MONTH}"

    return True, "OK"


def get_nba_events(api_key: str) -> Tuple[List[Dict], int]:
    """
    Fetch list of upcoming NBA events from The Odds API.

    Returns:
        (events_list, requests_used)
    """
    url = f"{BASE_URL}/sports/{SPORT}/events"
    params = {"apiKey": api_key}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    # Get remaining requests from headers
    remaining = int(response.headers.get("x-requests-remaining", 0))

    return response.json(), 1


def get_player_props_for_event(
    api_key: str,
    event_id: str
) -> Tuple[Dict[str, Dict], int]:
    """
    Fetch player points props for a specific event.

    Args:
        api_key: The Odds API key
        event_id: Event ID from get_nba_events

    Returns:
        (player_lines dict, requests_used)
        player_lines: {player_name: {ou: float, over_odds: int, under_odds: int}}
    """
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": MARKET,
        "bookmakers": BOOKMAKER,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    player_lines = {}

    # Parse the response to extract player lines
    if "bookmakers" in data:
        for bookmaker in data["bookmakers"]:
            if bookmaker.get("key") == BOOKMAKER:
                for market in bookmaker.get("markets", []):
                    if market.get("key") == MARKET:
                        for outcome in market.get("outcomes", []):
                            player_name = outcome.get("description", "")
                            point = outcome.get("point")
                            price = outcome.get("price")
                            outcome_name = outcome.get("name", "").lower()

                            if player_name and point is not None:
                                if player_name not in player_lines:
                                    player_lines[player_name] = {
                                        "ou": point,
                                        "over_odds": None,
                                        "under_odds": None,
                                    }

                                if outcome_name == "over":
                                    player_lines[player_name]["over_odds"] = price
                                elif outcome_name == "under":
                                    player_lines[player_name]["under_odds"] = price

    return player_lines, 1


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    # Remove suffixes like Jr., III, etc.
    name = name.lower().strip()
    name = name.replace(".", "").replace(",", "")
    name = name.replace(" jr", "").replace(" sr", "")
    name = name.replace(" iii", "").replace(" ii", "").replace(" iv", "")
    return name


def map_player_name_to_id(
    fanduel_name: str,
    conn: sqlite3.Connection
) -> Tuple[Optional[int], float, str]:
    """
    Map FanDuel player name to database player_id.

    3-tier matching strategy:
    1. Check odds_player_aliases table
    2. Exact match on players.full_name
    3. Fuzzy match with rapidfuzz

    Returns:
        (player_id, confidence, match_method)
    """
    cursor = conn.cursor()

    # Tier 1: Check aliases table
    cursor.execute("""
        SELECT player_id, confidence
        FROM odds_player_aliases
        WHERE fanduel_name = ?
    """, (fanduel_name,))

    alias_result = cursor.fetchone()
    if alias_result:
        return alias_result[0], alias_result[1], "alias"

    # Tier 2: Exact match
    cursor.execute("""
        SELECT player_id FROM players
        WHERE LOWER(full_name) = LOWER(?)
    """, (fanduel_name,))

    exact_result = cursor.fetchone()
    if exact_result:
        # Store in aliases for future
        _store_alias(conn, fanduel_name, exact_result[0], 1.0)
        return exact_result[0], 1.0, "exact"

    # Tier 3: Fuzzy match
    cursor.execute("SELECT player_id, full_name FROM players WHERE is_active = 1")
    all_players = cursor.fetchall()

    normalized_fanduel = normalize_name(fanduel_name)
    best_match = None
    best_score = 0

    for player_id, full_name in all_players:
        normalized_db = normalize_name(full_name)
        score = fuzz.ratio(normalized_fanduel, normalized_db)

        if score > best_score:
            best_score = score
            best_match = (player_id, full_name)

    if best_match and best_score >= FUZZY_MATCH_THRESHOLD:
        confidence = best_score / 100.0
        _store_alias(conn, fanduel_name, best_match[0], confidence)
        return best_match[0], confidence, "fuzzy"

    return None, 0.0, "no_match"


def _store_alias(
    conn: sqlite3.Connection,
    fanduel_name: str,
    player_id: int,
    confidence: float
) -> None:
    """Store a player name alias for future lookups."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO odds_player_aliases
        (fanduel_name, player_id, confidence, created_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, (fanduel_name, player_id, confidence))
    conn.commit()


def update_prediction_with_odds(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    fanduel_ou: float,
    fanduel_over_odds: Optional[int],
    fanduel_under_odds: Optional[int],
    event_id: Optional[str] = None
) -> bool:
    """
    Update a prediction record with FanDuel odds data.

    Returns:
        True if updated, False if prediction not found
    """
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE predictions
        SET fanduel_ou = ?,
            fanduel_over_odds = ?,
            fanduel_under_odds = ?,
            fanduel_fetched_at = CURRENT_TIMESTAMP,
            odds_event_id = ?
        WHERE player_id = ? AND game_date = ?
    """, (fanduel_ou, fanduel_over_odds, fanduel_under_odds, event_id, player_id, game_date))

    conn.commit()
    return cursor.rowcount > 0


def log_fetch_attempt(
    conn: sqlite3.Connection,
    game_date: date,
    events_fetched: int,
    players_matched: int,
    api_requests_used: int,
    remaining_requests: Optional[int] = None,
    error_message: Optional[str] = None
) -> None:
    """Log an API fetch attempt for budget tracking."""
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO odds_fetch_log
        (fetch_date, game_date, events_fetched, players_matched,
         api_requests_used, remaining_requests, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        str(date.today()),
        str(game_date),
        events_fetched,
        players_matched,
        api_requests_used,
        remaining_requests,
        error_message
    ))

    conn.commit()


def fetch_fanduel_lines_for_date(
    conn: sqlite3.Connection,
    game_date: date
) -> Dict[str, Any]:
    """
    Fetch FanDuel player points lines for all games on a date.

    This is the main entry point for fetching odds.

    Args:
        conn: Database connection
        game_date: Date to fetch odds for

    Returns:
        Dict with keys:
            - success: bool
            - lines: Dict[player_id, {ou, over_odds, under_odds}]
            - players_matched: int
            - events_fetched: int
            - api_requests_used: int
            - error: Optional[str]
    """
    result = {
        "success": False,
        "lines": {},
        "players_matched": 0,
        "events_fetched": 0,
        "api_requests_used": 0,
        "error": None,
    }

    # Ensure tables exist
    create_odds_tables(conn)
    upgrade_predictions_for_fanduel(conn)

    # Check if we should fetch
    should_fetch, reason = should_fetch_odds(conn, game_date)
    if not should_fetch:
        result["error"] = reason
        return result

    api_key = get_api_key()
    if not api_key:
        result["error"] = "API key not configured"
        return result

    total_requests = 0
    remaining = None

    try:
        # Step 1: Get today's events
        events, req_count = get_nba_events(api_key)
        total_requests += req_count

        # Filter to events on the target date (convert UTC to Eastern time)
        game_date_str = game_date.strftime("%Y-%m-%d")
        eastern = ZoneInfo("America/New_York")

        todays_events = []
        for e in events:
            commence_str = e.get("commence_time", "")
            if commence_str:
                try:
                    # Parse UTC timestamp: "2026-01-01T23:00:00Z"
                    utc_dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
                    # Convert to Eastern time
                    eastern_dt = utc_dt.astimezone(eastern)
                    # Compare date in Eastern timezone
                    if eastern_dt.strftime("%Y-%m-%d") == game_date_str:
                        todays_events.append(e)
                except ValueError:
                    continue

        result["events_fetched"] = len(todays_events)

        if not todays_events:
            result["error"] = f"No NBA events found for {game_date_str}"
            log_fetch_attempt(conn, game_date, 0, 0, total_requests, error_message=result["error"])
            return result

        # Step 2: Fetch player props for each event
        all_player_lines = {}

        for event in todays_events:
            event_id = event.get("id")
            if not event_id:
                continue

            try:
                player_lines, req_count = get_player_props_for_event(api_key, event_id)
                total_requests += req_count

                # Map and store lines
                for fanduel_name, line_data in player_lines.items():
                    player_id, confidence, method = map_player_name_to_id(fanduel_name, conn)

                    if player_id:
                        # Update prediction record
                        updated = update_prediction_with_odds(
                            conn=conn,
                            player_id=player_id,
                            game_date=game_date_str,
                            fanduel_ou=line_data["ou"],
                            fanduel_over_odds=line_data["over_odds"],
                            fanduel_under_odds=line_data["under_odds"],
                            event_id=event_id
                        )

                        if updated:
                            result["lines"][player_id] = line_data
                            result["players_matched"] += 1

                # Small delay between requests to be nice to API
                time.sleep(0.5)

            except requests.RequestException as e:
                print(f"Warning: Failed to fetch props for event {event_id}: {e}")
                continue

        result["success"] = True
        result["api_requests_used"] = total_requests

        # Log successful fetch
        log_fetch_attempt(
            conn, game_date,
            result["events_fetched"],
            result["players_matched"],
            total_requests,
            remaining
        )

        # Auto-backup to S3 after successful fetch (prevents data loss on reboot)
        if result["players_matched"] > 0:
            auto_backup_to_s3()

    except requests.RequestException as e:
        result["error"] = f"API request failed: {str(e)}"
        log_fetch_attempt(conn, game_date, 0, 0, total_requests, error_message=result["error"])

    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        log_fetch_attempt(conn, game_date, 0, 0, total_requests, error_message=result["error"])

    return result


def get_comparison_data(
    conn: sqlite3.Connection,
    game_date: str
) -> List[Dict]:
    """
    Get predictions with FanDuel comparison data for a date.

    Returns list of dicts with all comparison fields.
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            player_name,
            team_name,
            opponent_name,
            projected_ppg,
            proj_floor,
            proj_ceiling,
            proj_confidence,
            fanduel_ou,
            fanduel_over_odds,
            fanduel_under_odds,
            fanduel_fetched_at,
            dfs_score,
            dfs_grade
        FROM predictions
        WHERE game_date = ?
        ORDER BY dfs_score DESC
    """, (game_date,))

    columns = [
        "player_name", "team_name", "opponent_name",
        "projected_ppg", "proj_floor", "proj_ceiling", "proj_confidence",
        "fanduel_ou", "fanduel_over_odds", "fanduel_under_odds",
        "fanduel_fetched_at", "dfs_score", "dfs_grade"
    ]

    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def archive_fanduel_lines_to_s3(conn: sqlite3.Connection, game_date: date) -> Dict[str, Any]:
    """
    Export FanDuel lines for a date to CSV and upload to S3.

    Creates: s3://bucket/odds/fanduel_lines_YYYY-MM-DD.csv

    Args:
        conn: Database connection
        game_date: Date to archive

    Returns:
        Dict with success status, rows archived, and S3 key
    """
    try:
        import pandas as pd
        from s3_storage import S3PredictionStorage
    except ImportError as e:
        return {'success': False, 'error': f'Missing dependency: {e}'}

    # Query lines for the date
    query = """
        SELECT
            game_date,
            player_id,
            player_name,
            team_name,
            opponent_name,
            fanduel_ou,
            fanduel_over_odds,
            fanduel_under_odds,
            fanduel_fetched_at,
            projected_ppg,
            proj_floor,
            proj_ceiling
        FROM predictions
        WHERE game_date = ? AND fanduel_ou IS NOT NULL
        ORDER BY fanduel_ou DESC
    """

    try:
        df = pd.read_sql_query(query, conn, params=[str(game_date)])
    except Exception as e:
        return {'success': False, 'error': f'Query failed: {e}'}

    if df.empty:
        return {'success': False, 'error': 'No FanDuel lines found for this date'}

    # Convert to CSV
    csv_buffer = df.to_csv(index=False)

    # Upload to S3
    try:
        storage = S3PredictionStorage()
        if not storage.is_connected():
            return {'success': False, 'error': 'S3 not configured'}

        s3_key = f"odds/fanduel_lines_{game_date}.csv"
        storage.s3.put_object(
            Bucket=storage.bucket,
            Key=s3_key,
            Body=csv_buffer.encode('utf-8'),
            ContentType='text/csv',
            Metadata={
                'game_date': str(game_date),
                'rows': str(len(df)),
                'archived_at': datetime.now().isoformat()
            }
        )

        return {
            'success': True,
            'rows': len(df),
            's3_key': s3_key,
            'message': f'Archived {len(df)} lines to s3://{storage.bucket}/{s3_key}'
        }

    except Exception as e:
        return {'success': False, 'error': f'S3 upload failed: {e}'}


def auto_backup_to_s3(db_path: str = 'nba_stats.db') -> bool:
    """
    Trigger S3 backup of the database.

    Called after successful FanDuel fetch to ensure data isn't lost.
    """
    try:
        from s3_storage import S3PredictionStorage
        from pathlib import Path

        storage = S3PredictionStorage()
        if storage.is_connected():
            success, message = storage.upload_database(Path(db_path))
            if success:
                print(f"Auto-backup to S3: {message}")
            return success
        return False
    except Exception as e:
        print(f"Auto-backup failed: {e}")
        return False


if __name__ == "__main__":
    # Simple test
    print("The Odds API Integration Module")
    print("================================")

    api_key = get_api_key()
    if api_key:
        print(f"API key configured: {api_key[:8]}...")
    else:
        print("API key NOT configured")
        print("Add [theoddsapi] section to .streamlit/secrets.toml:")
        print('  API_KEY = "your-key-here"')
