#!/usr/bin/env python3
"""
Depth Chart & Role Tier Classification

Classifies players into role tiers (STAR, STARTER, ROTATION, BENCH) based on
their season statistics. This enables smarter injury impact analysis.

Key Insight:
- When a STAR is injured, the next STARTER gets a big usage boost
- When a STARTER is injured, ROTATION players get elevated
- Bench players rarely benefit significantly from injuries
- This hierarchy matters for predicting injury ripple effects

Role Tiers:
- STAR: Top 2-3 players per team, high minutes + high scoring
- STARTER: Regular starters, solid minutes
- ROTATION: Consistent bench players, 15-25 minutes
- BENCH: Deep bench, <15 minutes typically

Usage:
    from depth_chart import refresh_all_player_roles, get_player_role

    refresh_all_player_roles(conn, '2025-26')
    role = get_player_role(conn, player_id, team_id, '2025-26')
    print(f"Role tier: {role}")
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd


# =============================================================================
# TABLE CREATION
# =============================================================================

def ensure_player_roles_table(conn: sqlite3.Connection):
    """Create player_roles table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_roles (
            player_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            season TEXT NOT NULL,
            role_tier TEXT NOT NULL,
            avg_minutes REAL,
            games_played INTEGER,
            games_started INTEGER,
            season_ppg REAL,
            usage_pct REAL,
            calculated_at TEXT NOT NULL,
            PRIMARY KEY (player_id, team_id, season)
        )
    """)
    conn.commit()


# =============================================================================
# ROLE TIER THRESHOLDS
# =============================================================================

# Thresholds for role classification (based on per-game stats)
ROLE_THRESHOLDS = {
    'STAR': {
        'min_minutes': 30.0,      # 30+ minutes per game
        'min_ppg': 18.0,          # 18+ PPG
        'min_usage': 24.0,        # 24%+ usage rate (optional)
    },
    'STARTER': {
        'min_minutes': 25.0,      # 25+ minutes per game
        'min_ppg': 10.0,          # 10+ PPG
    },
    'ROTATION': {
        'min_minutes': 15.0,      # 15+ minutes per game
        'min_games_pct': 0.3,     # Played in 30%+ of team games
    },
    # BENCH is the default if none of the above match
}


def classify_role_tier(
    avg_minutes: float,
    ppg: float,
    games_played: int,
    team_games: int = None,
    usage_pct: float = None,
    avg_fpts: float = None,
) -> str:
    """
    Classify player into role tier based on statistics.

    Classification logic (in order):
    1. STAR: 30+ minutes AND 18+ PPG (or high usage, or 40+ DK FPTS)
    2. STARTER: 25+ minutes AND 10+ PPG
    3. ROTATION: 15+ minutes AND plays regularly
    4. BENCH: Everyone else

    Args:
        avg_minutes: Average minutes per game
        ppg: Points per game
        games_played: Number of games played
        team_games: Total team games (optional, for games% calc)
        usage_pct: Usage percentage (optional)
        avg_fpts: Average DK fantasy points (optional, catches multi-category stars)

    Returns:
        Role tier string: 'STAR', 'STARTER', 'ROTATION', or 'BENCH'
    """
    if avg_minutes is None or ppg is None:
        return 'BENCH'

    # STAR: High minutes + high scoring
    star_thresh = ROLE_THRESHOLDS['STAR']
    if avg_minutes >= star_thresh['min_minutes'] and ppg >= star_thresh['min_ppg']:
        return 'STAR'

    # Also STAR if high usage + good minutes (usage-based stars)
    if usage_pct and usage_pct >= star_thresh['min_usage'] and avg_minutes >= 28:
        return 'STAR'

    # Also STAR if high DK FPTS (multi-category producers like Wembanyama)
    if avg_fpts and avg_fpts >= 40.0 and avg_minutes >= 28:
        return 'STAR'

    # STARTER: Solid minutes + scoring
    starter_thresh = ROLE_THRESHOLDS['STARTER']
    if avg_minutes >= starter_thresh['min_minutes'] and ppg >= starter_thresh['min_ppg']:
        return 'STARTER'

    # ROTATION: Consistent bench player
    rotation_thresh = ROLE_THRESHOLDS['ROTATION']
    if avg_minutes >= rotation_thresh['min_minutes']:
        # Check if they play regularly
        if team_games is not None:
            games_pct = games_played / team_games if team_games > 0 else 0
            if games_pct >= rotation_thresh['min_games_pct']:
                return 'ROTATION'
        else:
            # No team games info, use absolute game count
            if games_played >= 10:
                return 'ROTATION'

    # Default to BENCH
    return 'BENCH'


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def calculate_team_depth_chart(
    conn: sqlite3.Connection,
    team_id: int,
    season: str
) -> List[Dict]:
    """
    Calculate full depth chart for a team.

    Args:
        conn: Database connection
        team_id: Team ID
        season: Season string (e.g., "2025-26")

    Returns:
        List of player dicts sorted by avg_minutes (descending):
        [{player_id, player_name, role_tier, avg_minutes, ppg, position}, ...]
    """
    cursor = conn.cursor()

    # Get team's total games played (approximation from top player)
    cursor.execute("""
        SELECT MAX(games_played) FROM player_season_totals
        WHERE team_id = ? AND season = ? AND season_type = 'Regular Season'
    """, [team_id, season])
    result = cursor.fetchone()
    team_games = result[0] if result and result[0] else 40  # Default estimate

    # Get all players on team with their stats
    cursor.execute("""
        SELECT
            pst.player_id,
            pst.player_name,
            pst.games_played,
            pst.minutes,
            pst.points,
            pst.usg_pct,
            tr.position
        FROM player_season_totals pst
        LEFT JOIN team_rosters tr ON pst.player_id = tr.player_id
            AND tr.team_id = pst.team_id AND tr.season = pst.season
        WHERE pst.team_id = ? AND pst.season = ?
          AND pst.season_type = 'Regular Season'
          AND pst.games_played > 0
        ORDER BY pst.minutes DESC
    """, [team_id, season])

    depth_chart = []
    for row in cursor.fetchall():
        player_id, name, games, total_min, total_pts, usg, position = row

        # Calculate per-game stats
        avg_minutes = total_min / games if games > 0 else 0
        ppg = total_pts / games if games > 0 else 0

        # Classify role
        role_tier = classify_role_tier(
            avg_minutes=avg_minutes,
            ppg=ppg,
            games_played=games,
            team_games=team_games,
            usage_pct=usg
        )

        depth_chart.append({
            'player_id': player_id,
            'player_name': name,
            'role_tier': role_tier,
            'avg_minutes': round(avg_minutes, 1),
            'ppg': round(ppg, 1),
            'games_played': games,
            'usage_pct': usg,
            'position': position
        })

    return depth_chart


def refresh_team_player_roles(
    conn: sqlite3.Connection,
    team_id: int,
    season: str
) -> int:
    """
    Refresh player_roles table for a specific team.

    Args:
        conn: Database connection
        team_id: Team ID
        season: Season string

    Returns:
        Number of roles updated
    """
    depth_chart = calculate_team_depth_chart(conn, team_id, season)

    cursor = conn.cursor()
    now = datetime.now().isoformat()
    count = 0

    for player in depth_chart:
        cursor.execute("""
            INSERT OR REPLACE INTO player_roles
            (player_id, team_id, season, role_tier, avg_minutes,
             games_played, games_started, season_ppg, usage_pct, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player['player_id'],
            team_id,
            season,
            player['role_tier'],
            player['avg_minutes'],
            player['games_played'],
            None,  # games_started not available
            player['ppg'],
            player['usage_pct'],
            now
        ))
        count += 1

    conn.commit()
    return count


def refresh_all_player_roles(conn: sqlite3.Connection, season: str) -> int:
    """
    Refresh player_roles table for all teams in a season.

    This should be called during daily_update.py after fetching stats.

    Args:
        conn: Database connection
        season: Season string (e.g., "2025-26")

    Returns:
        Total number of player roles updated
    """
    # Ensure table exists before inserting
    ensure_player_roles_table(conn)

    cursor = conn.cursor()

    # Get all teams with data for this season
    cursor.execute("""
        SELECT DISTINCT team_id FROM player_season_totals
        WHERE season = ? AND season_type = 'Regular Season'
    """, [season])
    team_ids = [row[0] for row in cursor.fetchall()]

    total_updated = 0
    for team_id in team_ids:
        count = refresh_team_player_roles(conn, team_id, season)
        total_updated += count

    return total_updated


def get_player_role(
    conn: sqlite3.Connection,
    player_id: int,
    team_id: int = None,
    season: str = None
) -> Optional[str]:
    """
    Get a player's role tier from the database.

    Args:
        conn: Database connection
        player_id: Player ID
        team_id: Optional team ID filter
        season: Optional season filter

    Returns:
        Role tier string or None if not found
    """
    cursor = conn.cursor()

    if season and team_id:
        cursor.execute("""
            SELECT role_tier FROM player_roles
            WHERE player_id = ? AND team_id = ? AND season = ?
        """, [player_id, team_id, season])
    elif season:
        cursor.execute("""
            SELECT role_tier FROM player_roles
            WHERE player_id = ? AND season = ?
            ORDER BY calculated_at DESC LIMIT 1
        """, [player_id, season])
    else:
        cursor.execute("""
            SELECT role_tier FROM player_roles
            WHERE player_id = ?
            ORDER BY calculated_at DESC LIMIT 1
        """, [player_id])

    result = cursor.fetchone()
    return result[0] if result else None


def get_player_roles_for_slate(
    conn: sqlite3.Connection,
    game_date: str,
    season: str = None
) -> Dict[int, str]:
    """
    Get role tiers for all players with predictions on a date.

    Args:
        conn: Database connection
        game_date: Game date
        season: Season string (optional, derived if not provided)

    Returns:
        Dict mapping player_id -> role_tier
    """
    cursor = conn.cursor()

    if not season:
        # Derive season from game_date
        year = int(game_date.split('-')[0])
        month = int(game_date.split('-')[1])
        if month >= 10:
            season = f"{year}-{str(year+1)[-2:]}"
        else:
            season = f"{year-1}-{str(year)[-2:]}"

    # Get player IDs from predictions
    cursor.execute("""
        SELECT DISTINCT player_id FROM predictions
        WHERE date(game_date) = date(?)
    """, [game_date])
    player_ids = [row[0] for row in cursor.fetchall()]

    if not player_ids:
        return {}

    # Get roles for these players
    placeholders = ','.join(['?' for _ in player_ids])
    cursor.execute(f"""
        SELECT player_id, role_tier FROM player_roles
        WHERE player_id IN ({placeholders}) AND season = ?
    """, player_ids + [season])

    roles = {row[0]: row[1] for row in cursor.fetchall()}

    # Default missing players to STARTER
    for pid in player_ids:
        if pid not in roles:
            roles[pid] = 'STARTER'

    return roles


# =============================================================================
# INJURY BENEFICIARIES
# =============================================================================

def get_injury_beneficiaries(
    conn: sqlite3.Connection,
    injured_player_id: int,
    team_id: int,
    season: str
) -> List[Dict]:
    """
    Identify teammates who benefit most when a player is injured.

    Uses role tiers to determine who gets the usage boost:
    - If STAR out: Other STARs and STARTERs get boost
    - If STARTER out: ROTATION players get elevated
    - If ROTATION out: Other ROTATION/BENCH may benefit

    Args:
        conn: Database connection
        injured_player_id: ID of injured player
        team_id: Team ID
        season: Season string

    Returns:
        List of beneficiary dicts:
        [{player_id, player_name, role_tier, expected_boost_pct}, ...]
    """
    cursor = conn.cursor()

    # Get injured player's role
    cursor.execute("""
        SELECT role_tier, season_ppg FROM player_roles
        WHERE player_id = ? AND team_id = ? AND season = ?
    """, [injured_player_id, team_id, season])
    result = cursor.fetchone()

    if not result:
        return []

    injured_role, injured_ppg = result
    injured_ppg = injured_ppg or 0

    # Get all teammates and their roles
    cursor.execute("""
        SELECT player_id, role_tier, avg_minutes, season_ppg
        FROM player_roles
        WHERE team_id = ? AND season = ? AND player_id != ?
        ORDER BY avg_minutes DESC
    """, [team_id, season, injured_player_id])

    teammates = cursor.fetchall()

    # Calculate beneficiary boosts based on hierarchy
    beneficiaries = []

    # How much production to redistribute (injured player's PPG)
    production_to_redistribute = injured_ppg

    if injured_role == 'STAR':
        # STARs and STARTERs benefit most
        boost_order = ['STAR', 'STARTER', 'ROTATION']
        boost_weights = {'STAR': 0.40, 'STARTER': 0.30, 'ROTATION': 0.20}
    elif injured_role == 'STARTER':
        # ROTATION players get elevated
        boost_order = ['STARTER', 'ROTATION', 'BENCH']
        boost_weights = {'STARTER': 0.25, 'ROTATION': 0.35, 'BENCH': 0.15}
    else:
        # Minor redistribution for bench injuries
        boost_order = ['ROTATION', 'BENCH']
        boost_weights = {'ROTATION': 0.20, 'BENCH': 0.30}

    # Count players in each tier
    tier_counts = {}
    for pid, role, mins, ppg in teammates:
        tier_counts[role] = tier_counts.get(role, 0) + 1

    for pid, role, mins, ppg in teammates:
        if role in boost_weights:
            weight = boost_weights[role]
            tier_count = max(1, tier_counts.get(role, 1))

            # Boost is proportional to weight, split among tier members
            boost_pct = (production_to_redistribute * weight) / tier_count

            if boost_pct > 0.5:  # Minimum 0.5 PPG boost to include
                # Get player name
                cursor.execute("SELECT player_name FROM player_roles WHERE player_id = ?", [pid])
                name_result = cursor.fetchone()
                name = name_result[0] if name_result else f"Player {pid}"

                beneficiaries.append({
                    'player_id': pid,
                    'player_name': name,
                    'role_tier': role,
                    'expected_boost_ppg': round(boost_pct, 1),
                    'expected_boost_pct': round(boost_pct / max(1, ppg or 10) * 100, 0)
                })

    # Sort by expected boost (descending)
    beneficiaries.sort(key=lambda x: x['expected_boost_ppg'], reverse=True)

    return beneficiaries[:5]  # Top 5 beneficiaries


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Depth chart and role analysis')
    parser.add_argument('--team', type=int, help='Team ID to analyze')
    parser.add_argument('--season', type=str, default='2025-26', help='Season')
    parser.add_argument('--refresh', action='store_true', help='Refresh all roles')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    if args.refresh:
        print(f"Refreshing player roles for {args.season}...")
        count = refresh_all_player_roles(conn, args.season)
        print(f"Updated {count} player roles")

    elif args.team:
        depth_chart = calculate_team_depth_chart(conn, args.team, args.season)
        print(f"\nDepth Chart for Team {args.team} ({args.season}):")
        print("-" * 70)
        for p in depth_chart:
            print(f"  {p['role_tier']:8} | {p['avg_minutes']:5.1f} min | "
                  f"{p['ppg']:5.1f} ppg | {p['player_name']}")

    else:
        # Show summary stats
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role_tier, COUNT(*) FROM player_roles
            WHERE season = ?
            GROUP BY role_tier
        """, [args.season])
        print(f"Player roles for {args.season}:")
        for role, count in cursor.fetchall():
            print(f"  {role}: {count}")

    conn.close()
