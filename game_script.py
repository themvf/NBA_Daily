#!/usr/bin/env python3
"""
Game Script Prediction Module

Uses Vegas spread to predict game flow and adjust player minutes/projections.

Key Insight:
- Large spreads (>10) predict blowouts where starters may rest in 4th quarter
- Bench players get "garbage time" minutes in blowouts
- Close games (<3 point spread) mean heavy minutes for closers/stars
- This affects both floor (minutes) and ceiling (opportunity)

Game Script Tiers:
- BLOWOUT (|spread| > 10): Starters may lose 3-5 minutes, bench gains 3-5
- COMFORTABLE (|spread| 5-10): Normal game flow
- TOSSUP (|spread| 3-5): Competitive, normal minutes
- CLOSE_GAME (|spread| < 3): Heavy minutes for stars/closers

Usage:
    from game_script import get_game_script_adjustment

    adj = get_game_script_adjustment(spread=-8.5, role_tier='STAR', is_home=True)
    print(f"Minutes adj: {adj['minutes_adj']}, PPG adj: {adj['ppg_adj']}")
"""

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


# =============================================================================
# GAME SCRIPT TIERS
# =============================================================================

class GameScriptTier(Enum):
    BLOWOUT = "blowout"           # |spread| > 10
    COMFORTABLE = "comfortable"   # |spread| 5-10
    TOSSUP = "tossup"             # |spread| 3-5
    CLOSE_GAME = "close_game"     # |spread| < 3
    UNKNOWN = "unknown"           # No spread available


def classify_game_script(spread: float) -> GameScriptTier:
    """
    Classify expected game script from Vegas spread.

    The spread indicates how lopsided the game is expected to be.
    Larger absolute spreads = more likely blowout = less starter minutes.

    Args:
        spread: Vegas point spread (negative = home favorite)
                e.g., -8.5 means home team favored by 8.5

    Returns:
        GameScriptTier enum value
    """
    if spread is None:
        return GameScriptTier.UNKNOWN

    abs_spread = abs(spread)

    if abs_spread > 10:
        return GameScriptTier.BLOWOUT
    elif abs_spread >= 5:
        return GameScriptTier.COMFORTABLE
    elif abs_spread >= 3:
        return GameScriptTier.TOSSUP
    else:
        return GameScriptTier.CLOSE_GAME


# =============================================================================
# MINUTES ADJUSTMENTS BY ROLE
# =============================================================================

# Minutes adjustment matrix: [game_script_tier][role_tier][is_favorite]
# Positive = more minutes, Negative = fewer minutes
MINUTES_ADJ_MATRIX = {
    GameScriptTier.BLOWOUT: {
        'STAR': {True: -4.0, False: 1.0},       # Stars on winning team rest; losers keep trying
        'STARTER': {True: -3.0, False: 0.5},    # Starters also rest if winning big
        'ROTATION': {True: 4.0, False: 2.0},    # Rotation gets garbage time either way
        'BENCH': {True: 5.0, False: 3.0},       # Deep bench gets most run in blowouts
    },
    GameScriptTier.COMFORTABLE: {
        'STAR': {True: -1.0, False: 0.0},
        'STARTER': {True: -0.5, False: 0.0},
        'ROTATION': {True: 1.5, False: 1.0},
        'BENCH': {True: 2.0, False: 1.5},
    },
    GameScriptTier.TOSSUP: {
        'STAR': {True: 0.0, False: 0.0},
        'STARTER': {True: 0.0, False: 0.0},
        'ROTATION': {True: 0.0, False: 0.0},
        'BENCH': {True: 0.0, False: 0.0},
    },
    GameScriptTier.CLOSE_GAME: {
        'STAR': {True: 2.0, False: 2.0},        # Stars play heavy in close games
        'STARTER': {True: 1.5, False: 1.5},     # Starters also play more
        'ROTATION': {True: -1.0, False: -1.0},  # Rotation squeezed out
        'BENCH': {True: -2.0, False: -2.0},     # Deep bench doesn't see floor
    },
    GameScriptTier.UNKNOWN: {
        'STAR': {True: 0.0, False: 0.0},
        'STARTER': {True: 0.0, False: 0.0},
        'ROTATION': {True: 0.0, False: 0.0},
        'BENCH': {True: 0.0, False: 0.0},
    },
}

# Points per minute conversion (approximate)
PPM_CONVERSION = 0.55  # Average is ~0.5 PPM, higher for scorers


def get_minutes_adjustment(
    spread: float,
    role_tier: str,
    is_home: bool = True
) -> Dict:
    """
    Calculate minutes adjustment based on game script.

    Args:
        spread: Vegas spread (negative = home favorite)
        role_tier: Player's role ('STAR', 'STARTER', 'ROTATION', 'BENCH')
        is_home: Whether player is on home team

    Returns:
        Dict with:
            - minutes_adj: Expected minutes change (+/-)
            - ppg_adj: Expected PPG impact from minutes
            - game_script_tier: Classification
            - is_favorite: Whether player's team is favored
            - explanation: Human-readable description
    """
    tier = classify_game_script(spread)

    # Determine if player's team is the favorite
    if spread is None:
        is_favorite = True  # Default assumption
    elif is_home:
        is_favorite = spread < 0  # Negative spread = home favorite
    else:
        is_favorite = spread > 0  # Positive spread = away favorite

    # Normalize role_tier
    role = role_tier.upper() if role_tier else 'STARTER'
    if role not in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
        role = 'STARTER'  # Default

    # Look up minutes adjustment
    tier_matrix = MINUTES_ADJ_MATRIX.get(tier, MINUTES_ADJ_MATRIX[GameScriptTier.UNKNOWN])
    role_matrix = tier_matrix.get(role, tier_matrix.get('STARTER'))
    minutes_adj = role_matrix.get(is_favorite, 0.0)

    # Convert to PPG impact
    ppg_adj = minutes_adj * PPM_CONVERSION

    # Generate explanation
    if tier == GameScriptTier.BLOWOUT:
        if is_favorite:
            if role in ['STAR', 'STARTER']:
                explanation = f"Blowout risk: {role.lower()} may rest (-{abs(minutes_adj):.0f} min)"
            else:
                explanation = f"Blowout upside: garbage time (+{minutes_adj:.0f} min)"
        else:
            explanation = f"Blowout risk: team losing, {role.lower()} plays through"
    elif tier == GameScriptTier.CLOSE_GAME:
        if role in ['STAR', 'STARTER']:
            explanation = f"Close game: heavy minutes (+{minutes_adj:.0f} min)"
        else:
            explanation = f"Close game: {role.lower()} squeezed out"
    elif tier == GameScriptTier.COMFORTABLE:
        explanation = "Comfortable lead expected: normal minutes"
    else:
        explanation = "Competitive game: standard minutes"

    return {
        'minutes_adj': round(minutes_adj, 1),
        'ppg_adj': round(ppg_adj, 2),
        'game_script_tier': tier.value,
        'is_favorite': is_favorite,
        'explanation': explanation
    }


# =============================================================================
# FLOOR/CEILING ADJUSTMENTS
# =============================================================================

def adjust_projection_for_game_script(
    projection: float,
    floor: float,
    ceiling: float,
    spread: float,
    role_tier: str,
    is_home: bool = True
) -> Tuple[float, float, float, Dict]:
    """
    Apply game script adjustments to player projection.

    This adjusts the projection, floor, and ceiling based on expected
    game flow (blowout vs close game).

    Args:
        projection: Base projected PPG
        floor: Projection floor
        ceiling: Projection ceiling
        spread: Vegas spread
        role_tier: Player's role tier
        is_home: Whether player is home team

    Returns:
        Tuple of (adjusted_proj, adjusted_floor, adjusted_ceiling, breakdown)
    """
    adj = get_minutes_adjustment(spread, role_tier, is_home)

    # Apply PPG adjustment
    adj_projection = projection + adj['ppg_adj']

    # Floor is more affected by negative adjustments (risk of sitting)
    if adj['ppg_adj'] < 0:
        adj_floor = floor + (adj['ppg_adj'] * 1.5)  # 1.5x impact on floor
    else:
        adj_floor = floor + (adj['ppg_adj'] * 0.5)  # Less impact on floor

    # Ceiling is more affected by positive adjustments (opportunity)
    if adj['ppg_adj'] > 0:
        adj_ceiling = ceiling + (adj['ppg_adj'] * 1.5)  # 1.5x impact on ceiling
    else:
        adj_ceiling = ceiling + (adj['ppg_adj'] * 0.5)  # Less impact on ceiling

    # Ensure floor doesn't exceed projection and ceiling
    adj_floor = max(0, min(adj_floor, adj_projection - 2))
    adj_ceiling = max(adj_ceiling, adj_projection + 2)

    breakdown = {
        **adj,
        'original_projection': projection,
        'original_floor': floor,
        'original_ceiling': ceiling,
        'adjusted_projection': round(adj_projection, 1),
        'adjusted_floor': round(adj_floor, 1),
        'adjusted_ceiling': round(adj_ceiling, 1),
    }

    return (
        round(adj_projection, 1),
        round(adj_floor, 1),
        round(adj_ceiling, 1),
        breakdown
    )


# =============================================================================
# DATABASE INTEGRATION
# =============================================================================

def get_game_spread(
    conn: sqlite3.Connection,
    game_date: str,
    home_team: str = None,
    away_team: str = None,
    team_abbrev: str = None
) -> Optional[float]:
    """
    Look up Vegas spread for a game.

    Args:
        conn: Database connection
        game_date: Game date (YYYY-MM-DD)
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        team_abbrev: Team abbreviation (will check both home/away)

    Returns:
        Spread (negative = home favorite) or None
    """
    cursor = conn.cursor()

    if home_team and away_team:
        cursor.execute("""
            SELECT spread FROM game_odds
            WHERE date(game_date) = date(?)
              AND home_team = ?
              AND away_team = ?
        """, [game_date, home_team, away_team])
    elif team_abbrev:
        cursor.execute("""
            SELECT spread, home_team FROM game_odds
            WHERE date(game_date) = date(?)
              AND (home_team = ? OR away_team = ?)
        """, [game_date, team_abbrev, team_abbrev])
    else:
        return None

    result = cursor.fetchone()
    return result[0] if result else None


def get_game_scripts_for_slate(
    conn: sqlite3.Connection,
    game_date: str
) -> Dict[str, Dict]:
    """
    Get game script data for all games on a slate.

    Args:
        conn: Database connection
        game_date: Game date

    Returns:
        Dict mapping game_id -> game script data
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT game_id, home_team, away_team, spread, blowout_risk
        FROM game_odds
        WHERE date(game_date) = date(?)
    """, [game_date])

    results = {}
    for row in cursor.fetchall():
        game_id, home, away, spread, blowout_risk = row
        tier = classify_game_script(spread)

        results[game_id] = {
            'home_team': home,
            'away_team': away,
            'spread': spread,
            'blowout_risk': blowout_risk,
            'game_script_tier': tier.value,
            'is_competitive': tier in [GameScriptTier.CLOSE_GAME, GameScriptTier.TOSSUP]
        }

    return results


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Game script analysis')
    parser.add_argument('--spread', type=float, help='Vegas spread (e.g., -8.5)')
    parser.add_argument('--role', type=str, default='STARTER',
                        help='Role tier (STAR, STARTER, ROTATION, BENCH)')
    parser.add_argument('--home', action='store_true', help='Player is on home team')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    parser.add_argument('--date', type=str, help='Check spreads for date')
    args = parser.parse_args()

    if args.spread is not None:
        adj = get_minutes_adjustment(args.spread, args.role, args.home)
        print(f"Spread: {args.spread:+.1f}")
        print(f"Role: {args.role}")
        print(f"Home team: {args.home}")
        print(f"\nGame Script: {adj['game_script_tier']}")
        print(f"Is Favorite: {adj['is_favorite']}")
        print(f"Minutes Adj: {adj['minutes_adj']:+.1f}")
        print(f"PPG Adj: {adj['ppg_adj']:+.2f}")
        print(f"Explanation: {adj['explanation']}")

    elif args.date:
        conn = sqlite3.connect(args.db)
        scripts = get_game_scripts_for_slate(conn, args.date)

        print(f"Game scripts for {args.date}:")
        for game_id, data in scripts.items():
            spread_str = f"{data['spread']:+.1f}" if data['spread'] else "N/A"
            print(f"  {data['away_team']} @ {data['home_team']}: "
                  f"spread={spread_str}, tier={data['game_script_tier']}")
        conn.close()

    else:
        # Demo with sample spreads
        print("Game Script Examples:")
        print("=" * 60)
        for spread in [-12, -8, -4, -2, 2, 6, 11]:
            for role in ['STAR', 'BENCH']:
                adj = get_minutes_adjustment(spread, role, is_home=True)
                print(f"Spread {spread:+3d} | {role:8} | "
                      f"{adj['game_script_tier']:11} | "
                      f"Min: {adj['minutes_adj']:+5.1f} | "
                      f"PPG: {adj['ppg_adj']:+5.2f}")
            print()
