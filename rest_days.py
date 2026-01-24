#!/usr/bin/env python3
"""
Rest Days & Back-to-Back Detection

Calculates days since a player's last game and applies fatigue multipliers
to improve prediction accuracy.

Key Insight:
- Players on 2nd night of back-to-back score 8-10% lower on average
- Players with 3+ days rest often have a small boost (fresh legs)
- This is one of the most overlooked factors in DFS projections

Multipliers:
- B2B (1 day rest): 0.92 (-8% fatigue penalty)
- 2 days rest: 1.02 (+2% bonus)
- 3+ days rest: 1.05 (+5% well-rested bonus)
- Default/unknown: 1.0

Usage:
    from rest_days import calculate_rest_factors_for_slate, get_rest_multiplier

    factors = calculate_rest_factors_for_slate(conn, '2026-01-20')
    for player_id, data in factors.items():
        print(f"Player {player_id}: {data['days_rest']} days rest, multiplier={data['multiplier']}")
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd


# =============================================================================
# REST MULTIPLIERS
# =============================================================================

# Research-based multipliers for rest impact on scoring
REST_MULTIPLIERS = {
    0: 1.00,  # Same day (unusual, shouldn't happen)
    1: 0.92,  # Back-to-back: 8% penalty
    2: 1.02,  # 2 days rest: 2% bonus
    3: 1.05,  # 3 days rest: 5% bonus (well-rested)
    4: 1.05,  # 4+ days: same as 3 (diminishing returns)
}


def get_rest_multiplier(days_rest: int) -> float:
    """
    Return fatigue multiplier based on days since last game.

    Args:
        days_rest: Number of days since player's last game
                   1 = back-to-back, 2 = one day off, etc.

    Returns:
        Multiplier to apply to projection (e.g., 0.92 for B2B)

    Examples:
        >>> get_rest_multiplier(1)  # B2B
        0.92
        >>> get_rest_multiplier(3)  # Well rested
        1.05
    """
    if days_rest <= 0:
        return 1.0  # Unknown or same-day (shouldn't happen)
    elif days_rest == 1:
        return REST_MULTIPLIERS[1]  # B2B
    elif days_rest == 2:
        return REST_MULTIPLIERS[2]  # Normal rest
    else:
        return REST_MULTIPLIERS[3]  # Well rested (3+ days)


def is_back_to_back(days_rest: int) -> bool:
    """Check if player is on second night of back-to-back."""
    return days_rest == 1


# =============================================================================
# DAYS REST CALCULATION
# =============================================================================

def calculate_days_rest(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    season: str = None
) -> Optional[int]:
    """
    Calculate days since player's last game.

    Args:
        conn: Database connection
        player_id: Player ID to look up
        game_date: Target game date (YYYY-MM-DD format)
        season: Optional season filter (e.g., "2025-26")

    Returns:
        Days since last game (1 = B2B), or None if no previous game found

    Example:
        >>> days = calculate_days_rest(conn, 203507, '2026-01-20')
        >>> print(f"Days rest: {days}")  # e.g., 2
    """
    cursor = conn.cursor()

    # Find the most recent game before game_date
    if season:
        cursor.execute("""
            SELECT MAX(game_date) FROM player_game_logs
            WHERE player_id = ?
              AND game_date < ?
              AND season = ?
              AND points IS NOT NULL
        """, [player_id, game_date, season])
    else:
        cursor.execute("""
            SELECT MAX(game_date) FROM player_game_logs
            WHERE player_id = ?
              AND game_date < ?
              AND points IS NOT NULL
        """, [player_id, game_date])

    result = cursor.fetchone()

    if result[0] is None:
        return None  # No previous game found

    last_game_date = result[0]

    # Calculate days difference
    target = datetime.strptime(game_date, '%Y-%m-%d')
    last_game = datetime.strptime(last_game_date, '%Y-%m-%d')
    days_rest = (target - last_game).days

    return days_rest


def calculate_rest_factor(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    season: str = None
) -> Dict:
    """
    Calculate complete rest factor data for a player.

    Args:
        conn: Database connection
        player_id: Player ID
        game_date: Target game date
        season: Optional season filter

    Returns:
        Dict with:
            - days_rest: int or None
            - multiplier: float (1.0 default)
            - is_b2b: bool
            - last_game_date: str or None
            - impact_description: str
    """
    days_rest = calculate_days_rest(conn, player_id, game_date, season)

    if days_rest is None:
        return {
            'days_rest': None,
            'multiplier': 1.0,
            'is_b2b': False,
            'last_game_date': None,
            'impact_description': 'No recent games found'
        }

    multiplier = get_rest_multiplier(days_rest)
    is_b2b = is_back_to_back(days_rest)

    # Generate description
    if is_b2b:
        description = f"B2B (-{int((1-multiplier)*100)}% fatigue)"
    elif days_rest == 2:
        description = f"1 day rest (+{int((multiplier-1)*100)}%)"
    elif days_rest >= 3:
        description = f"{days_rest-1} days rest (+{int((multiplier-1)*100)}%)"
    else:
        description = "Normal rest"

    # Get last game date for reference
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(game_date) FROM player_game_logs
        WHERE player_id = ? AND game_date < ? AND points IS NOT NULL
    """, [player_id, game_date])
    last_game_date = cursor.fetchone()[0]

    return {
        'days_rest': days_rest,
        'multiplier': multiplier,
        'is_b2b': is_b2b,
        'last_game_date': last_game_date,
        'impact_description': description
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def _parse_date(date_str: str) -> datetime:
    """Parse date string, handling various formats including timestamps."""
    if date_str is None:
        return None
    # Handle dates with timestamp suffix (e.g., "2026-01-01T00:00:00")
    date_str = date_str.split('T')[0]
    return datetime.strptime(date_str, '%Y-%m-%d')


def calculate_rest_factors_for_slate(
    conn: sqlite3.Connection,
    game_date: str,
    player_ids: list = None,
    season: str = None
) -> Dict[int, Dict]:
    """
    Calculate rest factors for all players on a slate.

    This is more efficient than calling calculate_rest_factor() for each player
    because it uses a single query to get all last game dates.

    Args:
        conn: Database connection
        game_date: Target game date
        player_ids: Optional list of player IDs (if None, uses predictions table)
        season: Optional season filter

    Returns:
        Dict mapping player_id -> rest factor data

    Example:
        >>> factors = calculate_rest_factors_for_slate(conn, '2026-01-20')
        >>> for pid, data in factors.items():
        ...     print(f"{pid}: {data['days_rest']} days, mult={data['multiplier']}")
    """
    cursor = conn.cursor()

    # Normalize game_date (handle timestamp suffix)
    game_date = game_date.split('T')[0]

    # If no player list provided, get from predictions table
    if player_ids is None:
        cursor.execute("""
            SELECT DISTINCT player_id FROM predictions
            WHERE date(game_date) = date(?)
        """, [game_date])
        player_ids = [row[0] for row in cursor.fetchall()]

    if not player_ids:
        return {}

    # Build query to get last game for all players at once
    placeholders = ','.join(['?' for _ in player_ids])

    if season:
        cursor.execute(f"""
            SELECT player_id, MAX(date(game_date)) as last_game
            FROM player_game_logs
            WHERE player_id IN ({placeholders})
              AND date(game_date) < date(?)
              AND season = ?
              AND points IS NOT NULL
            GROUP BY player_id
        """, player_ids + [game_date, season])
    else:
        cursor.execute(f"""
            SELECT player_id, MAX(date(game_date)) as last_game
            FROM player_game_logs
            WHERE player_id IN ({placeholders})
              AND date(game_date) < date(?)
              AND points IS NOT NULL
            GROUP BY player_id
        """, player_ids + [game_date])

    last_games = {row[0]: row[1] for row in cursor.fetchall()}

    # Calculate rest factors for each player
    target_date = _parse_date(game_date)
    results = {}

    for player_id in player_ids:
        last_game_date = last_games.get(player_id)

        if last_game_date is None:
            results[player_id] = {
                'days_rest': None,
                'multiplier': 1.0,
                'is_b2b': False,
                'last_game_date': None,
                'impact_description': 'No recent games'
            }
        else:
            last_game = _parse_date(last_game_date)
            days_rest = (target_date - last_game).days
            multiplier = get_rest_multiplier(days_rest)
            is_b2b = is_back_to_back(days_rest)

            if is_b2b:
                description = f"B2B (-{int((1-multiplier)*100)}%)"
            elif days_rest == 2:
                description = f"1 day rest (+{int((multiplier-1)*100)}%)"
            elif days_rest >= 3:
                description = f"{days_rest-1} days rest (+{int((multiplier-1)*100)}%)"
            else:
                description = "Normal rest"

            results[player_id] = {
                'days_rest': days_rest,
                'multiplier': multiplier,
                'is_b2b': is_b2b,
                'last_game_date': last_game_date,
                'impact_description': description
            }

    return results


# =============================================================================
# TEAM-LEVEL ANALYSIS
# =============================================================================

def get_team_rest_status(
    conn: sqlite3.Connection,
    team_id: int,
    game_date: str
) -> Dict:
    """
    Get rest status for an entire team.

    Useful for understanding if a team is on a B2B collectively
    (which affects game pace and scoring environment).

    Args:
        conn: Database connection
        team_id: Team ID
        game_date: Target game date

    Returns:
        Dict with team-level rest info:
            - team_on_b2b: bool
            - last_team_game: str
            - days_since_last_game: int
    """
    cursor = conn.cursor()

    # Find team's last game from team_game_logs
    cursor.execute("""
        SELECT MAX(game_date) FROM team_game_logs
        WHERE team_id = ?
          AND game_date < ?
    """, [team_id, game_date])

    result = cursor.fetchone()
    last_game_date = result[0] if result else None

    if last_game_date is None:
        return {
            'team_on_b2b': False,
            'last_team_game': None,
            'days_since_last_game': None
        }

    target = datetime.strptime(game_date, '%Y-%m-%d')
    last_game = datetime.strptime(last_game_date, '%Y-%m-%d')
    days_rest = (target - last_game).days

    return {
        'team_on_b2b': days_rest == 1,
        'last_team_game': last_game_date,
        'days_since_last_game': days_rest
    }


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate rest days for players')
    parser.add_argument('--date', type=str, help='Game date (YYYY-MM-DD)')
    parser.add_argument('--player', type=int, help='Specific player ID')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    if args.date:
        if args.player:
            # Single player
            factor = calculate_rest_factor(conn, args.player, args.date)
            print(f"Player {args.player} on {args.date}:")
            print(f"  Days rest: {factor['days_rest']}")
            print(f"  Multiplier: {factor['multiplier']}")
            print(f"  B2B: {factor['is_b2b']}")
            print(f"  Impact: {factor['impact_description']}")
        else:
            # All players on slate
            factors = calculate_rest_factors_for_slate(conn, args.date)

            b2b_count = sum(1 for f in factors.values() if f['is_b2b'])
            rested_count = sum(1 for f in factors.values()
                              if f['days_rest'] and f['days_rest'] >= 3)

            print(f"Rest factors for {args.date}:")
            print(f"  Total players: {len(factors)}")
            print(f"  On B2B: {b2b_count}")
            print(f"  Well rested (3+ days): {rested_count}")

            # Show B2B players
            if b2b_count > 0:
                print("\nPlayers on B2B:")
                for pid, data in factors.items():
                    if data['is_b2b']:
                        print(f"  Player {pid}: {data['impact_description']}")
    else:
        print("Usage: python rest_days.py --date 2026-01-20 [--player 203507]")

    conn.close()
