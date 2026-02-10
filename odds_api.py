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
MARKET = "player_points"  # Legacy single market (for compatibility)
BOOKMAKER = "fanduel"
MAX_API_REQUESTS_PER_MONTH = 500
BUDGET_WARNING_THRESHOLD = 0.80  # Warn at 80%
BUDGET_BLOCK_THRESHOLD = 0.95   # Block at 95%
FUZZY_MATCH_THRESHOLD = 85      # Minimum similarity score

# Extended player prop markets for DFS stat analysis
# All fetched in a single request per event (efficient!)
EXTENDED_PROP_MARKETS = [
    "player_points",                    # 1 FPTS per point
    "player_rebounds",                  # 1.25 FPTS per rebound
    "player_assists",                   # 1.5 FPTS per assist
    "player_threes",                    # 0.5 FPTS bonus per 3PM
    "player_steals",                    # 2 FPTS per steal
    "player_blocks",                    # 2 FPTS per block
    "player_points_rebounds_assists",   # PRA combo - fantasy proxy!
]

# Market key to database column mapping
MARKET_COLUMN_MAP = {
    "player_points": "fanduel_ou",           # Points - existing column
    "player_rebounds": "fanduel_reb_ou",
    "player_assists": "fanduel_ast_ou",
    "player_threes": "fanduel_3pm_ou",
    "player_steals": "fanduel_stl_ou",
    "player_blocks": "fanduel_blk_ou",
    "player_points_rebounds_assists": "fanduel_pra_ou",  # Fantasy proxy!
}

# DraftKings DFS scoring weights (for converting stat lines to FPTS)
DFS_WEIGHTS = {
    "points": 1.0,
    "rebounds": 1.25,
    "assists": 1.5,
    "threes": 0.5,  # Bonus only (points already counted)
    "steals": 2.0,
    "blocks": 2.0,
}


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

    # Core columns (existing)
    new_columns = {
        'fanduel_ou': 'REAL DEFAULT NULL',
        'fanduel_over_odds': 'INTEGER DEFAULT NULL',
        'fanduel_under_odds': 'INTEGER DEFAULT NULL',
        'fanduel_fetched_at': 'TEXT DEFAULT NULL',
        'odds_event_id': 'TEXT DEFAULT NULL',
        # Extended stat prop columns
        'fanduel_reb_ou': 'REAL DEFAULT NULL',      # Rebounds O/U
        'fanduel_ast_ou': 'REAL DEFAULT NULL',      # Assists O/U
        'fanduel_3pm_ou': 'REAL DEFAULT NULL',      # 3-pointers made O/U
        'fanduel_stl_ou': 'REAL DEFAULT NULL',      # Steals O/U
        'fanduel_blk_ou': 'REAL DEFAULT NULL',      # Blocks O/U
        'fanduel_pra_ou': 'REAL DEFAULT NULL',      # PRA combo (fantasy proxy!)
        # Vegas-implied fantasy points
        'vegas_implied_fpts': 'REAL DEFAULT NULL',  # Calculated from all props
        'vegas_vs_proj_diff': 'REAL DEFAULT NULL',  # Difference from our projection
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

    # Get first day of current month (use Eastern time for Streamlit Cloud compatibility)
    today = datetime.now(ZoneInfo("America/New_York")).date()
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
    """, (str(game_date), str(datetime.now(ZoneInfo("America/New_York")).date())))

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


def get_game_odds_bulk(api_key: str, game_date: date = None) -> Tuple[List[Dict], int]:
    """
    Fetch game-level odds (spreads, totals) for NBA games.

    This is a BULK endpoint - 1 request returns all upcoming games with odds.
    Date filtering is done locally in extract_game_odds_from_response() since
    the commenceTimeFrom/To API params require a paid subscription tier.

    Returns:
        (games_with_odds_list, requests_used)
        Each game contains: home_team, away_team, commence_time, bookmakers with spreads/totals
    """
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json(), 1


def extract_game_odds_from_response(games_data: List[Dict], target_date: str) -> List[Dict]:
    """
    Extract spread/total from bulk game odds response.

    Extracts ALL games from the API response (no date filtering).
    Each game's date is derived from its commence_time (UTC → Eastern).
    The caller or DB query handles date filtering downstream.

    Args:
        games_data: Response from get_game_odds_bulk
        target_date: Date string (YYYY-MM-DD) — used as fallback if commence_time missing

    Returns:
        List of dicts with: game_id, home_team, away_team, spread, total, etc.
    """
    from zoneinfo import ZoneInfo

    # Team name mapping (same as vegas_odds.py)
    TEAM_NAME_TO_ABBR = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
    }

    eastern = ZoneInfo("America/New_York")
    extracted = []

    for game in games_data:
        # Derive game date from commence_time (UTC → Eastern)
        commence_str = game.get("commence_time", "")
        game_date_str = target_date  # fallback
        if commence_str:
            try:
                utc_dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
                eastern_dt = utc_dt.astimezone(eastern)
                game_date_str = eastern_dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass  # Use target_date as fallback

        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        home_abbr = TEAM_NAME_TO_ABBR.get(home_team, home_team[:3].upper())
        away_abbr = TEAM_NAME_TO_ABBR.get(away_team, away_team[:3].upper())

        # Extract odds from bookmakers (prefer FanDuel)
        bookmakers = game.get("bookmakers", [])
        preferred_order = ["fanduel", "draftkings", "betmgm", "caesars"]

        selected_book = None
        for preferred in preferred_order:
            for book in bookmakers:
                if book.get("key", "").lower() == preferred:
                    selected_book = book
                    break
            if selected_book:
                break

        if not selected_book and bookmakers:
            selected_book = bookmakers[0]

        if not selected_book:
            continue

        spread = None
        total = None

        for market in selected_book.get("markets", []):
            market_key = market.get("key")
            outcomes = market.get("outcomes", [])

            if market_key == "spreads":
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        spread = outcome.get("point")
                        break

            elif market_key == "totals":
                for outcome in outcomes:
                    if outcome.get("name") == "Over":
                        total = outcome.get("point")
                        break

        if spread is None or total is None:
            continue

        game_id = f"{game_date_str}_{away_abbr}_{home_abbr}"

        extracted.append({
            "game_id": game_id,
            "game_date": game_date_str,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "spread": spread,
            "total": total,
            "commence_time": commence_str,
        })

    return extracted


def store_game_odds_from_api(conn: sqlite3.Connection, games: List[Dict]) -> int:
    """
    Store game odds in game_odds table (for Tournament Strategy reuse).

    Args:
        conn: Database connection
        games: List of game dicts from extract_game_odds_from_response

    Returns:
        Number of games stored
    """
    # Import vegas_odds functions for environment calculation
    try:
        from vegas_odds import (
            ensure_game_odds_table,
            GameOdds,
            compute_game_environment,
            store_game_odds
        )
    except ImportError:
        print("Warning: vegas_odds module not available, skipping game odds storage")
        return 0

    ensure_game_odds_table(conn)
    stored = 0

    for game in games:
        # Create GameOdds object
        odds = GameOdds(
            game_id=game["game_id"],
            game_date=game["game_date"],
            home_team=game["home_team"],
            away_team=game["away_team"],
            spread=game["spread"],
            total=game["total"],
            home_ml=0,  # Not fetched in bulk
            away_ml=0,
            commence_time=game["commence_time"],
        )

        # Compute environment metrics
        env = compute_game_environment(odds)

        # Store in database
        store_game_odds(conn, odds, env)
        stored += 1

    return stored


def get_player_props_for_event(
    api_key: str,
    event_id: str,
    extended_markets: bool = False
) -> Tuple[Dict[str, Dict], int]:
    """
    Fetch player props for a specific event.

    Args:
        api_key: The Odds API key
        event_id: Event ID from get_nba_events
        extended_markets: If True, fetch all stat props (reb, ast, 3pm, stl, blk, PRA)

    Returns:
        (player_lines dict, requests_used)
        player_lines: {player_name: {
            ou: float,              # Points O/U
            over_odds: int,
            under_odds: int,
            reb_ou: float,          # Rebounds O/U (if extended)
            ast_ou: float,          # Assists O/U (if extended)
            fg3m_ou: float,         # 3PM O/U (if extended)
            stl_ou: float,          # Steals O/U (if extended)
            blk_ou: float,          # Blocks O/U (if extended)
            pra_ou: float,          # PRA combo (if extended)
        }}
    """
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"

    # Determine which markets to fetch
    if extended_markets:
        markets = ",".join(EXTENDED_PROP_MARKETS)
    else:
        markets = MARKET

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "bookmakers": BOOKMAKER,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    player_lines = {}

    # Parse the response to extract player lines for all markets
    if "bookmakers" in data:
        for bookmaker in data["bookmakers"]:
            if bookmaker.get("key") == BOOKMAKER:
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")

                    for outcome in market.get("outcomes", []):
                        player_name = outcome.get("description", "")
                        point = outcome.get("point")
                        price = outcome.get("price")
                        outcome_name = outcome.get("name", "").lower()

                        if not player_name or point is None:
                            continue

                        # Initialize player entry if needed
                        if player_name not in player_lines:
                            player_lines[player_name] = {
                                "ou": None,
                                "over_odds": None,
                                "under_odds": None,
                                "reb_ou": None,
                                "ast_ou": None,
                                "fg3m_ou": None,
                                "stl_ou": None,
                                "blk_ou": None,
                                "pra_ou": None,
                            }

                        # Map market to appropriate field
                        if market_key == "player_points":
                            player_lines[player_name]["ou"] = point
                            if outcome_name == "over":
                                player_lines[player_name]["over_odds"] = price
                            elif outcome_name == "under":
                                player_lines[player_name]["under_odds"] = price

                        elif market_key == "player_rebounds":
                            if outcome_name == "over":  # Use Over line as the O/U
                                player_lines[player_name]["reb_ou"] = point

                        elif market_key == "player_assists":
                            if outcome_name == "over":
                                player_lines[player_name]["ast_ou"] = point

                        elif market_key == "player_threes":
                            if outcome_name == "over":
                                player_lines[player_name]["fg3m_ou"] = point

                        elif market_key == "player_steals":
                            if outcome_name == "over":
                                player_lines[player_name]["stl_ou"] = point

                        elif market_key == "player_blocks":
                            if outcome_name == "over":
                                player_lines[player_name]["blk_ou"] = point

                        elif market_key == "player_points_rebounds_assists":
                            if outcome_name == "over":
                                player_lines[player_name]["pra_ou"] = point

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
    event_id: Optional[str] = None,
    # Extended stat props
    reb_ou: Optional[float] = None,
    ast_ou: Optional[float] = None,
    fg3m_ou: Optional[float] = None,
    stl_ou: Optional[float] = None,
    blk_ou: Optional[float] = None,
    pra_ou: Optional[float] = None,
) -> bool:
    """
    Update a prediction record with FanDuel odds data (including extended stats).

    Returns:
        True if updated, False if prediction not found
    """
    cursor = conn.cursor()

    # Calculate Vegas-implied FPTS if we have enough data
    vegas_implied_fpts = None
    if fanduel_ou is not None:
        vegas_implied_fpts = calculate_vegas_implied_fpts(
            points_ou=fanduel_ou,
            reb_ou=reb_ou,
            ast_ou=ast_ou,
            fg3m_ou=fg3m_ou,
            stl_ou=stl_ou,
            blk_ou=blk_ou,
            pra_ou=pra_ou
        )

    cursor.execute("""
        UPDATE predictions
        SET fanduel_ou = ?,
            fanduel_over_odds = ?,
            fanduel_under_odds = ?,
            fanduel_fetched_at = CURRENT_TIMESTAMP,
            odds_event_id = ?,
            fanduel_reb_ou = ?,
            fanduel_ast_ou = ?,
            fanduel_3pm_ou = ?,
            fanduel_stl_ou = ?,
            fanduel_blk_ou = ?,
            fanduel_pra_ou = ?,
            vegas_implied_fpts = ?
        WHERE player_id = ? AND game_date = ?
    """, (
        fanduel_ou, fanduel_over_odds, fanduel_under_odds, event_id,
        reb_ou, ast_ou, fg3m_ou, stl_ou, blk_ou, pra_ou,
        vegas_implied_fpts,
        player_id, game_date
    ))

    conn.commit()
    return cursor.rowcount > 0


def calculate_vegas_implied_fpts(
    points_ou: Optional[float] = None,
    reb_ou: Optional[float] = None,
    ast_ou: Optional[float] = None,
    fg3m_ou: Optional[float] = None,
    stl_ou: Optional[float] = None,
    blk_ou: Optional[float] = None,
    pra_ou: Optional[float] = None,
    turnovers_estimate: float = 1.5,  # League avg ~1.5 TOV
) -> Optional[float]:
    """
    Calculate Vegas-implied fantasy points from player props.

    DraftKings DFS Scoring:
    - Points: 1 FPTS each
    - Rebounds: 1.25 FPTS each
    - Assists: 1.5 FPTS each
    - 3PM: 0.5 FPTS bonus (points already counted)
    - Steals: 2 FPTS each
    - Blocks: 2 FPTS each
    - Turnovers: -0.5 FPTS each

    Strategy:
    1. If PRA combo is available, use it (most accurate single proxy)
    2. Otherwise, sum individual stat props × weights
    3. Apply adjustments for 3PM bonus and turnovers

    Args:
        points_ou: Points O/U line
        reb_ou: Rebounds O/U line
        ast_ou: Assists O/U line
        fg3m_ou: 3-pointers made O/U line
        stl_ou: Steals O/U line
        blk_ou: Blocks O/U line
        pra_ou: Points+Rebounds+Assists combo O/U
        turnovers_estimate: Estimated turnovers (default 1.5)

    Returns:
        Estimated fantasy points, or None if insufficient data
    """
    if points_ou is None:
        return None

    fpts = 0.0

    # Method 1: Use PRA combo if available (most reliable)
    if pra_ou is not None:
        # PRA = P + R + A, but DFS weights are 1, 1.2, 1.5
        # We need to estimate the breakdown
        # Average NBA breakdown: 60% P, 22% R, 18% A
        estimated_pts = pra_ou * 0.60
        estimated_reb = pra_ou * 0.22
        estimated_ast = pra_ou * 0.18

        # But use individual lines if available to refine
        if reb_ou is not None:
            estimated_reb = reb_ou
        if ast_ou is not None:
            estimated_ast = ast_ou
        if points_ou is not None:
            estimated_pts = points_ou

        fpts = (
            estimated_pts * DFS_WEIGHTS["points"] +
            estimated_reb * DFS_WEIGHTS["rebounds"] +
            estimated_ast * DFS_WEIGHTS["assists"]
        )

    # Method 2: Build from individual props
    else:
        fpts = points_ou * DFS_WEIGHTS["points"]

        if reb_ou is not None:
            fpts += reb_ou * DFS_WEIGHTS["rebounds"]
        else:
            # Estimate rebounds from points (very rough: ~0.2 R per P)
            fpts += (points_ou * 0.2) * DFS_WEIGHTS["rebounds"]

        if ast_ou is not None:
            fpts += ast_ou * DFS_WEIGHTS["assists"]
        else:
            # Estimate assists from points (rough: ~0.15 A per P)
            fpts += (points_ou * 0.15) * DFS_WEIGHTS["assists"]

    # Add 3PM bonus (0.5 FPTS per 3 made, on top of the 3 points)
    if fg3m_ou is not None:
        fpts += fg3m_ou * DFS_WEIGHTS["threes"]
    else:
        # Estimate 3PM from points (~0.1 per point for average player)
        fpts += (points_ou * 0.1) * DFS_WEIGHTS["threes"]

    # Add steals and blocks
    if stl_ou is not None:
        fpts += stl_ou * DFS_WEIGHTS["steals"]
    else:
        # League avg ~0.8 steals
        fpts += 0.8 * DFS_WEIGHTS["steals"]

    if blk_ou is not None:
        fpts += blk_ou * DFS_WEIGHTS["blocks"]
    else:
        # League avg ~0.5 blocks
        fpts += 0.5 * DFS_WEIGHTS["blocks"]

    # Subtract turnovers
    fpts -= turnovers_estimate * 0.5

    return round(fpts, 1)


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
        str(datetime.now(ZoneInfo("America/New_York")).date()),
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
    game_date: date,
    force: bool = False,
    extended_markets: bool = True  # NEW: Fetch all stat props by default
) -> Dict[str, Any]:
    """
    Fetch FanDuel player props for all games on a date.

    This is the main entry point for fetching odds.

    Args:
        conn: Database connection
        game_date: Date to fetch odds for
        force: If True, bypass the "already fetched" safeguard (for injury updates)
        extended_markets: If True (default), fetch all stat props (reb, ast, 3pm, stl, blk, PRA)
                         This is efficient - all markets are fetched in a single request per event

    Returns:
        Dict with keys:
            - success: bool
            - lines: Dict[player_id, {ou, over_odds, under_odds, reb_ou, ast_ou, ...}]
            - players_matched: int
            - events_fetched: int
            - api_requests_used: int
            - extended_stats_found: int (count of players with extended stat props)
            - error: Optional[str]
    """
    result = {
        "success": False,
        "lines": {},
        "players_matched": 0,
        "events_fetched": 0,
        "api_requests_used": 0,
        "extended_stats_found": 0,
        "error": None,
    }

    # Ensure tables exist
    create_odds_tables(conn)
    upgrade_predictions_for_fanduel(conn)

    # Check if we should fetch (unless force=True)
    if not force:
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

        # Step 1.5: ALSO fetch game-level odds (spreads/totals) for Tournament Strategy
        # This is a BULK endpoint - 1 request returns ALL games, very efficient!
        try:
            games_data, req_count = get_game_odds_bulk(api_key, game_date=game_date)
            total_requests += req_count

            # Diagnostic: log what the API returned
            odds_debug = {
                "bulk_games_returned": len(games_data) if games_data else 0,
                "target_date": game_date_str,
            }
            if games_data:
                # Sample first game's structure for debugging
                sample = games_data[0]
                odds_debug["sample_commence"] = sample.get("commence_time", "N/A")
                odds_debug["sample_home"] = sample.get("home_team", "N/A")
                odds_debug["sample_bookmakers"] = len(sample.get("bookmakers", []))
                # Check all commence times
                from zoneinfo import ZoneInfo as _ZI
                _eastern = _ZI("America/New_York")
                _dates_seen = set()
                for g in games_data:
                    ct = g.get("commence_time", "")
                    if ct:
                        try:
                            _utc = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                            _et = _utc.astimezone(_eastern)
                            _dates_seen.add(_et.strftime("%Y-%m-%d"))
                        except Exception:
                            _dates_seen.add(f"parse_err:{ct}")
                odds_debug["eastern_dates_in_response"] = sorted(_dates_seen)

            # Extract all games (no date filtering — match by teams downstream)
            game_odds_list = extract_game_odds_from_response(games_data, game_date_str)
            odds_debug["extracted_total"] = len(game_odds_list)
            # Count how many match the target date
            target_date_count = sum(1 for g in game_odds_list if g.get("game_date") == game_date_str)
            odds_debug["extracted_for_target_date"] = target_date_count
            if game_odds_list:
                odds_debug["sample_extracted"] = {
                    "game_id": game_odds_list[0].get("game_id"),
                    "game_date": game_odds_list[0].get("game_date"),
                    "spread": game_odds_list[0].get("spread"),
                    "total": game_odds_list[0].get("total"),
                }

            games_stored = store_game_odds_from_api(conn, game_odds_list)
            result["game_odds_stored"] = games_stored
            result["odds_debug"] = odds_debug
            print(f"[odds_api] Stored {games_stored} game odds | debug: {odds_debug}")

        except Exception as e:
            # Non-fatal: player props are still useful even if game odds fail
            import traceback, re as _re
            result["game_odds_stored"] = 0
            # Sanitize API key from error messages before exposing to UI
            err_str = _re.sub(r'apiKey=[a-f0-9]+', 'apiKey=***', str(e))
            tb_str = _re.sub(r'apiKey=[a-f0-9]+', 'apiKey=***', traceback.format_exc())
            result["odds_debug"] = {"error": err_str, "traceback": tb_str}
            print(f"[odds_api] Warning: Failed to fetch game odds: {err_str}")

        # Step 2: Fetch player props for each event
        all_player_lines = {}

        for event in todays_events:
            event_id = event.get("id")
            if not event_id:
                continue

            try:
                player_lines, req_count = get_player_props_for_event(
                    api_key, event_id, extended_markets=extended_markets
                )
                total_requests += req_count

                # Map and store lines
                for fanduel_name, line_data in player_lines.items():
                    player_id, confidence, method = map_player_name_to_id(fanduel_name, conn)

                    if player_id:
                        # Update prediction record with all available props
                        updated = update_prediction_with_odds(
                            conn=conn,
                            player_id=player_id,
                            game_date=game_date_str,
                            fanduel_ou=line_data.get("ou"),
                            fanduel_over_odds=line_data.get("over_odds"),
                            fanduel_under_odds=line_data.get("under_odds"),
                            event_id=event_id,
                            # Extended stat props
                            reb_ou=line_data.get("reb_ou"),
                            ast_ou=line_data.get("ast_ou"),
                            fg3m_ou=line_data.get("fg3m_ou"),
                            stl_ou=line_data.get("stl_ou"),
                            blk_ou=line_data.get("blk_ou"),
                            pra_ou=line_data.get("pra_ou"),
                        )

                        if updated:
                            result["lines"][player_id] = line_data
                            # Track if extended stats were found
                            if any([
                                line_data.get("reb_ou"),
                                line_data.get("ast_ou"),
                                line_data.get("fg3m_ou"),
                                line_data.get("stl_ou"),
                                line_data.get("blk_ou"),
                                line_data.get("pra_ou"),
                            ]):
                                result["extended_stats_found"] += 1
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


# =============================================================================
# Vegas Ensemble Integration (for Projection Blending)
# =============================================================================

def get_vegas_ensemble_projection(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    our_projection: float,
    vegas_weight: float = 0.30,
    dynamic_weighting: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    Blend our projection with FanDuel Over/Under line for ensemble prediction.

    Vegas lines capture market consensus including:
    - Injury news that may not be in our model yet
    - Lineup changes and rotation adjustments
    - Sharp money movements from professional bettors

    Formula: ensemble = (1 - vegas_weight) * our_projection + vegas_weight * vegas_ou

    Dynamic weighting adjusts vegas_weight based on:
    - Divergence: Increase Vegas weight when our projection differs significantly
    - Recent injury returns: Vegas may have better info on returning players
    - New team situations: Vegas better at pricing trades/acquisitions

    Args:
        conn: Database connection
        player_id: Player's database ID
        game_date: Game date (YYYY-MM-DD)
        our_projection: Our model's PPG projection
        vegas_weight: Base weight for Vegas line (default 0.30 = 30%)
        dynamic_weighting: If True, adjust weight based on divergence

    Returns:
        Tuple of (ensemble_projection, analytics_dict)
        analytics_dict contains:
        - vegas_ou: The FanDuel O/U line
        - weight_used: Actual weight applied
        - divergence_pct: How much our projection differs from Vegas
        - adjustment: Points added/subtracted from our projection
    """
    cursor = conn.cursor()
    analytics = {
        'vegas_ou': None,
        'weight_used': 0.0,
        'divergence_pct': 0.0,
        'adjustment': 0.0,
        'blended': False,
        'reason': 'No Vegas data'
    }

    # Query for FanDuel O/U line
    cursor.execute("""
        SELECT fanduel_ou, fanduel_over_odds, fanduel_under_odds
        FROM predictions
        WHERE player_id = ? AND game_date = ?
        AND fanduel_ou IS NOT NULL
    """, (player_id, game_date))

    result = cursor.fetchone()

    if not result or result[0] is None:
        return our_projection, analytics

    vegas_ou = result[0]
    over_odds = result[1]
    under_odds = result[2]

    analytics['vegas_ou'] = vegas_ou

    # Calculate divergence
    if our_projection > 0:
        divergence_pct = abs(vegas_ou - our_projection) / our_projection * 100
    else:
        divergence_pct = 0

    analytics['divergence_pct'] = round(divergence_pct, 1)

    # Dynamic weighting
    effective_weight = vegas_weight

    if dynamic_weighting:
        # Increase Vegas weight when divergence is high (>20%)
        # This suggests Vegas has information we don't
        if divergence_pct > 30:
            effective_weight = min(0.50, vegas_weight + 0.15)  # Max 50% Vegas
            analytics['reason'] = 'High divergence - trusting Vegas more'
        elif divergence_pct > 20:
            effective_weight = min(0.45, vegas_weight + 0.10)
            analytics['reason'] = 'Moderate divergence - increased Vegas weight'
        else:
            analytics['reason'] = 'Normal blending'

        # If odds are heavily skewed to one side, Vegas has strong conviction
        if over_odds and under_odds:
            odds_skew = abs(over_odds - under_odds)
            if odds_skew > 30:  # Significant skew
                effective_weight = min(0.50, effective_weight + 0.05)
                analytics['reason'] += ' + Vegas has conviction'

    analytics['weight_used'] = round(effective_weight, 2)

    # Calculate ensemble projection
    ensemble = (1 - effective_weight) * our_projection + effective_weight * vegas_ou
    ensemble = round(ensemble, 1)

    analytics['adjustment'] = round(ensemble - our_projection, 1)
    analytics['blended'] = True

    return ensemble, analytics


def get_vegas_game_environment(
    conn: sqlite3.Connection,
    game_date: str,
    team_abbrev: str
) -> Dict[str, Any]:
    """
    Get game environment factors derived from Vegas odds.

    This provides context about the game that affects player scoring:
    - Pace score: High-total games mean more possessions and scoring chances
    - Blowout risk: Large spreads suggest stars may rest in 4th quarter
    - OT probability: Close games have higher overtime likelihood
    - Stack score: Indicates quality of same-game stacking

    Args:
        conn: Database connection
        game_date: Game date (YYYY-MM-DD)
        team_abbrev: Team abbreviation (e.g., 'LAL', 'BOS')

    Returns:
        Dict with environment factors:
        - pace_score: game_total / 228 (league avg) - >1 means faster pace
        - blowout_risk: 0-1 scale, higher = more risk of blowout
        - ot_probability: 0-1 scale, probability of overtime
        - stack_score: 0-1 scale, quality for same-game stacking
        - volatility_multiplier: Factor to expand projection ranges
        - spread: Point spread (negative = favorite)
        - total: Game total points (O/U)
        - is_home: Whether the team is home
    """
    cursor = conn.cursor()

    result = {
        'pace_score': 1.0,
        'blowout_risk': 0.0,
        'ot_probability': 0.06,  # League base rate ~6%
        'stack_score': 0.5,
        'volatility_multiplier': 1.0,
        'spread': None,
        'total': None,
        'is_home': None,
        'found': False
    }

    # Query game_odds table
    cursor.execute("""
        SELECT home_team, away_team, spread, total, blowout_risk,
               ot_probability, stack_score, volatility_multiplier
        FROM game_odds
        WHERE date(game_date) = date(?)
        AND (home_team = ? OR away_team = ?)
    """, (game_date, team_abbrev, team_abbrev))

    row = cursor.fetchone()

    if not row:
        return result

    home_team, away_team, spread, total, blowout_risk, ot_prob, stack_score, volatility = row

    result['found'] = True
    result['is_home'] = (team_abbrev == home_team)
    result['spread'] = spread
    result['total'] = total

    # Use stored values if available, otherwise calculate
    if total:
        result['pace_score'] = round(total / 228.0, 3)  # 228 is league avg total

    if blowout_risk is not None:
        result['blowout_risk'] = blowout_risk
    elif spread is not None:
        # Calculate blowout risk from spread
        abs_spread = abs(spread)
        if abs_spread >= 12:
            result['blowout_risk'] = min(0.85, 0.6 + (abs_spread - 12) * 0.03)
        elif abs_spread >= 8:
            result['blowout_risk'] = 0.4 + (abs_spread - 8) * 0.05
        else:
            result['blowout_risk'] = max(0.1, abs_spread * 0.05)

    if ot_prob is not None:
        result['ot_probability'] = ot_prob
    elif spread is not None:
        # Calculate OT probability from spread
        abs_spread = abs(spread)
        if abs_spread <= 3.5:
            result['ot_probability'] = 0.09  # 9% for very close games
        elif abs_spread <= 6:
            result['ot_probability'] = 0.07
        else:
            result['ot_probability'] = 0.05  # Below base rate for blowouts

    if stack_score is not None:
        result['stack_score'] = stack_score
    elif spread is not None and total is not None:
        # Calculate stack score from spread + total
        abs_spread = abs(spread)
        if abs_spread <= 4 and total >= 228:
            result['stack_score'] = 0.85  # Ideal stacking game
        elif abs_spread <= 6 and total >= 220:
            result['stack_score'] = 0.70
        elif abs_spread <= 8:
            result['stack_score'] = 0.50
        else:
            result['stack_score'] = 0.30  # Avoid stacking blowouts

    if volatility is not None:
        result['volatility_multiplier'] = volatility
    elif spread is not None:
        # Calculate volatility multiplier
        volatility = 1.0
        abs_spread = abs(spread)
        if abs_spread <= 4:
            volatility += 0.10  # Close games are more volatile
        if total and total >= 235:
            volatility += 0.05  # High-scoring games are volatile

        result['volatility_multiplier'] = round(volatility, 2)

    return result


def get_vegas_adjusted_projection(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_abbrev: str,
    game_date: str,
    base_projection: float,
    base_floor: float,
    base_ceiling: float,
    season_avg: float,
    is_star: bool = False
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Apply full Vegas adjustments to a player's projection.

    This is the main entry point for Vegas integration into the projection pipeline.
    It applies:
    1. Ensemble blending with FanDuel O/U (30% weight)
    2. Game environment adjustments (blowout risk, pace)
    3. Volatility adjustments to floor/ceiling ranges

    Args:
        conn: Database connection
        player_id: Player's database ID
        player_name: Player name (for logging)
        team_abbrev: Team abbreviation
        game_date: Game date (YYYY-MM-DD)
        base_projection: Our model's base projection
        base_floor: Our model's floor estimate
        base_ceiling: Our model's ceiling estimate
        season_avg: Player's season scoring average
        is_star: Whether player is a star (season avg >= 22)

    Returns:
        Tuple of (adjusted_projection, adjusted_floor, adjusted_ceiling, analytics_dict)
    """
    analytics = {
        'vegas_applied': False,
        'ensemble': {},
        'environment': {},
        'adjustments': []
    }

    # Step 1: Ensemble blend with FanDuel O/U
    projection, ensemble_data = get_vegas_ensemble_projection(
        conn, player_id, game_date, base_projection
    )
    analytics['ensemble'] = ensemble_data

    if ensemble_data['blended']:
        analytics['vegas_applied'] = True
        analytics['adjustments'].append(f"Vegas blend: {ensemble_data['adjustment']:+.1f}")

    # Step 2: Get game environment
    environment = get_vegas_game_environment(conn, game_date, team_abbrev)
    analytics['environment'] = environment

    # Step 3: Apply game environment adjustments
    floor = base_floor
    ceiling = base_ceiling

    if environment['found']:
        analytics['vegas_applied'] = True

        # Blowout risk adjustment for stars
        if environment['blowout_risk'] >= 0.6 and is_star:
            # Stars get benched in blowouts - reduce projection
            blowout_penalty = 0.95  # 5% reduction
            projection = projection * blowout_penalty
            ceiling = ceiling * 0.92  # Ceiling reduced more (less upside)
            analytics['adjustments'].append(f"Blowout risk ({environment['blowout_risk']:.0%}): -5%")

        # High pace game boost
        if environment['pace_score'] > 1.05:  # >5% faster than average
            pace_bonus = 1 + (environment['pace_score'] - 1) * 0.5  # Half the pace diff
            projection = projection * pace_bonus
            ceiling = ceiling * (1 + (environment['pace_score'] - 1) * 0.8)  # Ceiling benefits more
            analytics['adjustments'].append(f"High pace ({environment['pace_score']:.2f}): +{(pace_bonus-1)*100:.1f}%")

        # Volatility adjustment to ranges
        if environment['volatility_multiplier'] != 1.0:
            # Expand ranges based on volatility
            mid = (floor + ceiling) / 2
            half_range = (ceiling - floor) / 2
            new_half_range = half_range * environment['volatility_multiplier']
            floor = mid - new_half_range
            ceiling = mid + new_half_range
            analytics['adjustments'].append(f"Volatility: x{environment['volatility_multiplier']:.2f}")

        # OT probability boost to ceiling
        if environment['ot_probability'] > 0.08:
            ot_ceiling_boost = 1 + (environment['ot_probability'] - 0.06) * 2
            ceiling = ceiling * ot_ceiling_boost
            analytics['adjustments'].append(f"OT likelihood ({environment['ot_probability']:.0%}): ceiling +{(ot_ceiling_boost-1)*100:.1f}%")

    # Ensure reasonable bounds
    floor = max(0, round(floor, 1))
    ceiling = round(ceiling, 1)
    projection = round(projection, 1)

    # Ensure projection is within range
    if projection < floor:
        projection = floor
    if projection > ceiling:
        projection = (projection + ceiling) / 2

    return projection, floor, ceiling, analytics


# =============================================================================
# Mispricing Detection for DFS Edge
# =============================================================================

def detect_stat_mispricings(
    conn: sqlite3.Connection,
    game_date: str,
    min_divergence_pct: float = 15.0,
    include_marginal_players: bool = True
) -> List[Dict[str, Any]]:
    """
    Detect players where Vegas props significantly differ from our projections.

    This is the key value of extended prop markets - finding edge cases where:
    1. Vegas has info we don't (injury news, lineup changes)
    2. Market is inefficient on peripheral stats (rebounds, assists for mid-tier players)
    3. PRA combo reveals fantasy upside not captured in individual stat models

    Args:
        conn: Database connection
        game_date: Game date (YYYY-MM-DD)
        min_divergence_pct: Minimum % difference to flag (default 15%)
        include_marginal_players: Include players with salary < $5000

    Returns:
        List of mispricing dicts, sorted by edge potential:
        {
            player_name, salary, position,
            our_proj, vegas_implied, divergence_pct,
            stat_mispricings: [{stat, our_value, vegas_value, edge}],
            recommendation: "BOOST" | "FADE" | "WATCH"
        }
    """
    cursor = conn.cursor()

    # Query players with Vegas data
    cursor.execute("""
        SELECT
            p.player_name,
            p.team_name,
            p.projected_ppg,
            p.proj_floor,
            p.proj_ceiling,
            p.proj_rebounds,
            p.proj_assists,
            p.proj_fg3m,
            p.proj_steals,
            p.proj_blocks,
            p.fanduel_ou,
            p.fanduel_reb_ou,
            p.fanduel_ast_ou,
            p.fanduel_3pm_ou,
            p.fanduel_stl_ou,
            p.fanduel_blk_ou,
            p.fanduel_pra_ou,
            p.vegas_implied_fpts,
            p.dfs_salary,
            p.position
        FROM predictions p
        WHERE p.game_date = ?
        AND p.fanduel_ou IS NOT NULL
    """, (game_date,))

    columns = [
        'player_name', 'team_name', 'projected_ppg', 'proj_floor', 'proj_ceiling',
        'proj_rebounds', 'proj_assists', 'proj_fg3m', 'proj_steals', 'proj_blocks',
        'fanduel_ou', 'fanduel_reb_ou', 'fanduel_ast_ou', 'fanduel_3pm_ou',
        'fanduel_stl_ou', 'fanduel_blk_ou', 'fanduel_pra_ou', 'vegas_implied_fpts',
        'dfs_salary', 'position'
    ]

    mispricings = []

    for row in cursor.fetchall():
        data = dict(zip(columns, row))

        # Skip low-salary players if not including marginal
        salary = data.get('dfs_salary') or 0
        if not include_marginal_players and salary < 5000:
            continue

        our_fpts = data.get('projected_ppg') or 0
        vegas_fpts = data.get('vegas_implied_fpts')

        # Calculate overall divergence
        if our_fpts > 0 and vegas_fpts:
            overall_divergence = ((vegas_fpts - our_fpts) / our_fpts) * 100
        else:
            overall_divergence = 0

        # Check individual stat mispricings
        stat_mispricings = []

        stat_checks = [
            ('points', data.get('projected_ppg'), data.get('fanduel_ou'), DFS_WEIGHTS["points"]),
            ('rebounds', data.get('proj_rebounds'), data.get('fanduel_reb_ou'), DFS_WEIGHTS["rebounds"]),
            ('assists', data.get('proj_assists'), data.get('fanduel_ast_ou'), DFS_WEIGHTS["assists"]),
            ('3pm', data.get('proj_fg3m'), data.get('fanduel_3pm_ou'), DFS_WEIGHTS["threes"]),
            ('steals', data.get('proj_steals'), data.get('fanduel_stl_ou'), DFS_WEIGHTS["steals"]),
            ('blocks', data.get('proj_blocks'), data.get('fanduel_blk_ou'), DFS_WEIGHTS["blocks"]),
        ]

        for stat_name, our_val, vegas_val, fpts_weight in stat_checks:
            if our_val and vegas_val and our_val > 0:
                divergence = ((vegas_val - our_val) / our_val) * 100
                edge_fpts = (vegas_val - our_val) * fpts_weight

                if abs(divergence) >= min_divergence_pct:
                    stat_mispricings.append({
                        'stat': stat_name,
                        'our_value': round(our_val, 1),
                        'vegas_value': round(vegas_val, 1),
                        'divergence_pct': round(divergence, 1),
                        'edge_fpts': round(edge_fpts, 1),
                        'direction': 'OVER' if divergence > 0 else 'UNDER'
                    })

        # Only include if there's meaningful divergence
        if abs(overall_divergence) >= min_divergence_pct or stat_mispricings:
            # Determine recommendation
            if overall_divergence >= 15:
                recommendation = "BOOST"  # Vegas sees more upside
            elif overall_divergence <= -15:
                recommendation = "FADE"   # Vegas sees less upside
            else:
                recommendation = "WATCH"  # Individual stat edges

            mispricings.append({
                'player_name': data['player_name'],
                'team': data['team_name'],
                'salary': salary,
                'position': data.get('position', ''),
                'our_proj_fpts': round(our_fpts, 1),
                'vegas_implied_fpts': round(vegas_fpts, 1) if vegas_fpts else None,
                'overall_divergence_pct': round(overall_divergence, 1),
                'stat_mispricings': stat_mispricings,
                'recommendation': recommendation,
                # Additional context
                'proj_floor': data.get('proj_floor'),
                'proj_ceiling': data.get('proj_ceiling'),
                'pra_ou': data.get('fanduel_pra_ou'),
            })

    # Sort by absolute divergence (biggest edges first)
    mispricings.sort(key=lambda x: abs(x['overall_divergence_pct']), reverse=True)

    return mispricings


def get_vegas_stat_blend(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    stat_name: str,
    our_projection: float,
    vegas_weight: float = 0.35
) -> Tuple[float, Dict[str, Any]]:
    """
    Blend our stat projection with Vegas O/U for a specific stat.

    This is the stat-level equivalent of get_vegas_ensemble_projection().
    Particularly valuable for:
    - Rebounds for stretch 4s (Vegas may price rest differently)
    - Assists for backup PGs (matchup-dependent)
    - Steals/blocks for defensive specialists

    Args:
        conn: Database connection
        player_id: Player's database ID
        game_date: Game date (YYYY-MM-DD)
        stat_name: One of 'rebounds', 'assists', 'fg3m', 'steals', 'blocks'
        our_projection: Our model's projection for this stat
        vegas_weight: Weight for Vegas line (default 35% - slightly higher than points)

    Returns:
        (blended_projection, analytics_dict)
    """
    # Map stat name to database column
    stat_column_map = {
        'rebounds': 'fanduel_reb_ou',
        'assists': 'fanduel_ast_ou',
        'fg3m': 'fanduel_3pm_ou',
        'threes': 'fanduel_3pm_ou',
        'steals': 'fanduel_stl_ou',
        'blocks': 'fanduel_blk_ou',
    }

    column = stat_column_map.get(stat_name.lower())
    if not column:
        return our_projection, {'blended': False, 'reason': f'Unknown stat: {stat_name}'}

    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT {column} FROM predictions
        WHERE player_id = ? AND game_date = ?
    """, (player_id, game_date))

    result = cursor.fetchone()

    analytics = {
        'stat': stat_name,
        'our_projection': our_projection,
        'vegas_ou': None,
        'weight_used': 0.0,
        'adjustment': 0.0,
        'blended': False,
        'reason': 'No Vegas data'
    }

    if not result or result[0] is None:
        return our_projection, analytics

    vegas_ou = result[0]
    analytics['vegas_ou'] = vegas_ou

    # Calculate divergence
    if our_projection > 0:
        divergence_pct = abs(vegas_ou - our_projection) / our_projection * 100
    else:
        divergence_pct = 0

    # Dynamic weighting - trust Vegas more for peripheral stats
    effective_weight = vegas_weight

    # Higher weight for stats with larger divergence
    if divergence_pct > 25:
        effective_weight = min(0.50, vegas_weight + 0.10)
        analytics['reason'] = 'Large divergence - trusting Vegas more'
    elif divergence_pct > 15:
        effective_weight = min(0.45, vegas_weight + 0.05)
        analytics['reason'] = 'Moderate divergence'
    else:
        analytics['reason'] = 'Normal blending'

    analytics['weight_used'] = round(effective_weight, 2)

    # Blend
    blended = (1 - effective_weight) * our_projection + effective_weight * vegas_ou
    blended = round(blended, 1)

    analytics['adjustment'] = round(blended - our_projection, 1)
    analytics['blended'] = True

    return blended, analytics


def format_mispricing_summary(mispricings: List[Dict]) -> str:
    """
    Format mispricings for display in Streamlit UI.

    Returns a markdown-formatted string.
    """
    if not mispricings:
        return "No significant mispricings detected."

    lines = ["### 🎯 Vegas Mispricing Alerts\n"]

    boost_players = [m for m in mispricings if m['recommendation'] == 'BOOST']
    fade_players = [m for m in mispricings if m['recommendation'] == 'FADE']

    if boost_players:
        lines.append("**🚀 BOOST (Vegas sees more upside):**")
        for p in boost_players[:5]:
            stat_edges = ', '.join([
                f"{s['stat']}: {s['direction']} ({s['edge_fpts']:+.1f} FPTS)"
                for s in p['stat_mispricings'][:2]
            ])
            lines.append(
                f"- **{p['player_name']}** (${p['salary']:,}): "
                f"Our {p['our_proj_fpts']} → Vegas {p['vegas_implied_fpts']} "
                f"({p['overall_divergence_pct']:+.0f}%)"
            )
            if stat_edges:
                lines.append(f"  - {stat_edges}")

    if fade_players:
        lines.append("\n**📉 FADE (Vegas sees less upside):**")
        for p in fade_players[:5]:
            lines.append(
                f"- **{p['player_name']}** (${p['salary']:,}): "
                f"Our {p['our_proj_fpts']} → Vegas {p['vegas_implied_fpts']} "
                f"({p['overall_divergence_pct']:+.0f}%)"
            )

    return '\n'.join(lines)


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
