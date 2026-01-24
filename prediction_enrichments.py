#!/usr/bin/env python3
"""
Prediction Enrichments Module

Applies additional factors to predictions after they're generated:
- Rest days & B2B adjustments
- Game script (blowout/close game) adjustments
- Role tier classification
- Position-specific matchup factors

This is applied as a post-processing step to keep the main prediction
pipeline clean and maintain backwards compatibility.

Usage:
    from prediction_enrichments import apply_enrichments_to_predictions

    # After generating predictions:
    apply_enrichments_to_predictions(conn, game_date)
"""

import sqlite3
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

# Import our enrichment modules
from rest_days import calculate_rest_factors_for_slate, get_rest_multiplier
from game_script import get_minutes_adjustment, classify_game_script
from depth_chart import get_player_roles_for_slate
from position_ppm_stats import get_position_matchup_factor


def apply_enrichments_to_predictions(
    conn: sqlite3.Connection,
    game_date: str,
    apply_adjustments: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Apply all enrichment factors to predictions for a date.

    This updates the predictions table with:
    - days_rest, rest_multiplier, is_b2b
    - game_script_tier, blowout_risk, minutes_adjustment
    - role_tier
    - position_matchup_factor

    If apply_adjustments=True, also adjusts projected_ppg with rest/position factors.

    Args:
        conn: Database connection
        game_date: Game date (YYYY-MM-DD)
        apply_adjustments: Whether to modify projections (default True)
        verbose: Print progress info

    Returns:
        Dict with enrichment stats:
            - players_enriched: int
            - b2b_count: int
            - well_rested_count: int
            - blowout_games: int
            - close_games: int
    """
    cursor = conn.cursor()
    game_date = game_date.split('T')[0]  # Normalize date format

    stats = {
        'players_enriched': 0,
        'b2b_count': 0,
        'well_rested_count': 0,
        'blowout_games': 0,
        'close_games': 0,
        'errors': []
    }

    if verbose:
        print(f"Applying enrichments to predictions for {game_date}...")

    # ===========================================================================
    # Step 1: Calculate rest factors for all players
    # ===========================================================================
    try:
        rest_factors = calculate_rest_factors_for_slate(conn, game_date)
        if verbose:
            print(f"  Rest factors calculated for {len(rest_factors)} players")
    except Exception as e:
        rest_factors = {}
        stats['errors'].append(f"Rest factors: {e}")

    # ===========================================================================
    # Step 2: Get role tiers for all players
    # ===========================================================================
    try:
        role_tiers = get_player_roles_for_slate(conn, game_date)
        if verbose:
            print(f"  Role tiers loaded for {len(role_tiers)} players")
    except Exception as e:
        role_tiers = {}
        stats['errors'].append(f"Role tiers: {e}")

    # ===========================================================================
    # Step 3: Get game script data (spreads) for all games
    # ===========================================================================
    game_scripts = {}
    try:
        cursor.execute("""
            SELECT game_id, home_team, away_team, spread, blowout_risk
            FROM game_odds
            WHERE date(game_date) = date(?)
        """, [game_date])

        for row in cursor.fetchall():
            game_id, home, away, spread, blowout = row
            tier = classify_game_script(spread)
            game_scripts[game_id] = {
                'home_team': home,
                'away_team': away,
                'spread': spread,
                'blowout_risk': blowout,
                'tier': tier.value
            }

            if tier.value == 'blowout':
                stats['blowout_games'] += 1
            elif tier.value == 'close_game':
                stats['close_games'] += 1

        if verbose:
            print(f"  Game scripts loaded for {len(game_scripts)} games")
    except Exception as e:
        stats['errors'].append(f"Game scripts: {e}")

    # ===========================================================================
    # Step 4: Get all predictions for this date
    # ===========================================================================
    cursor.execute("""
        SELECT prediction_id, player_id, team_id, team_name, opponent_id,
               projected_ppg, proj_floor, proj_ceiling
        FROM predictions
        WHERE date(game_date) = date(?)
    """, [game_date])

    predictions = cursor.fetchall()
    if verbose:
        print(f"  Processing {len(predictions)} predictions...")

    # ===========================================================================
    # Step 5: Update each prediction with enrichments
    # ===========================================================================
    for pred in predictions:
        pred_id, player_id, team_id, team_name, opponent_id, proj, floor, ceiling = pred

        try:
            # Get rest factor
            rest_data = rest_factors.get(player_id, {})
            days_rest = rest_data.get('days_rest')
            rest_mult = rest_data.get('multiplier', 1.0)
            is_b2b = 1 if rest_data.get('is_b2b', False) else 0

            if is_b2b:
                stats['b2b_count'] += 1
            elif days_rest and days_rest >= 3:
                stats['well_rested_count'] += 1

            # Get role tier
            role_tier = role_tiers.get(player_id, 'STARTER')

            # Get game script (find game for this team)
            game_script_tier = 'neutral'
            blowout_risk = 0.0
            minutes_adj = 0.0
            spread = None

            for gid, gdata in game_scripts.items():
                # Check if this team is in this game
                team_abbrev = _get_team_abbrev(conn, team_id)
                if team_abbrev in [gdata['home_team'], gdata['away_team']]:
                    spread = gdata['spread']
                    game_script_tier = gdata['tier']
                    blowout_risk = gdata['blowout_risk'] or 0.0

                    # Calculate minutes adjustment
                    is_home = (team_abbrev == gdata['home_team'])
                    adj = get_minutes_adjustment(spread, role_tier, is_home)
                    minutes_adj = adj['ppg_adj']
                    break

            # Get position matchup factor
            player_position = _get_player_position(conn, player_id)
            pos_factor = get_position_matchup_factor(conn, opponent_id, player_position)

            # ===========================================================================
            # Apply adjustments to projection if enabled
            # ===========================================================================
            adjusted_proj = proj
            adjusted_floor = floor
            adjusted_ceiling = ceiling

            if apply_adjustments:
                # Apply rest multiplier
                adjusted_proj = proj * rest_mult
                adjusted_floor = floor * rest_mult
                adjusted_ceiling = ceiling * rest_mult

                # Apply position matchup factor
                adjusted_proj *= pos_factor
                adjusted_floor *= pos_factor
                adjusted_ceiling *= pos_factor

                # Apply game script adjustment (additive)
                adjusted_proj += minutes_adj
                adjusted_floor += minutes_adj * 0.5  # Less impact on floor
                adjusted_ceiling += minutes_adj * 1.5  # More impact on ceiling

                # Round to 1 decimal
                adjusted_proj = round(adjusted_proj, 1)
                adjusted_floor = round(max(0, adjusted_floor), 1)
                adjusted_ceiling = round(adjusted_ceiling, 1)

            # ===========================================================================
            # Update the prediction record
            # ===========================================================================
            cursor.execute("""
                UPDATE predictions SET
                    days_rest = ?,
                    rest_multiplier = ?,
                    is_b2b = ?,
                    game_script_tier = ?,
                    blowout_risk = ?,
                    minutes_adjustment = ?,
                    role_tier = ?,
                    position_matchup_factor = ?,
                    projected_ppg = ?,
                    proj_floor = ?,
                    proj_ceiling = ?
                WHERE prediction_id = ?
            """, (
                days_rest,
                round(rest_mult, 3),
                is_b2b,
                game_script_tier,
                round(blowout_risk, 3),
                round(minutes_adj, 2),
                role_tier,
                round(pos_factor, 3),
                adjusted_proj,
                adjusted_floor,
                adjusted_ceiling,
                pred_id
            ))

            stats['players_enriched'] += 1

        except Exception as e:
            stats['errors'].append(f"Player {player_id}: {e}")

    conn.commit()

    if verbose:
        print(f"\n  Enrichments applied:")
        print(f"    - Players enriched: {stats['players_enriched']}")
        print(f"    - On B2B: {stats['b2b_count']}")
        print(f"    - Well rested (3+ days): {stats['well_rested_count']}")
        print(f"    - Blowout games: {stats['blowout_games']}")
        print(f"    - Close games: {stats['close_games']}")
        if stats['errors']:
            print(f"    - Errors: {len(stats['errors'])}")

    return stats


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_team_abbrev_cache = {}

def _get_team_abbrev(conn: sqlite3.Connection, team_id: int) -> str:
    """Get team abbreviation from team_id."""
    if team_id in _team_abbrev_cache:
        return _team_abbrev_cache[team_id]

    cursor = conn.cursor()
    cursor.execute("SELECT abbreviation FROM teams WHERE team_id = ?", [team_id])
    result = cursor.fetchone()
    abbrev = result[0] if result else ""
    _team_abbrev_cache[team_id] = abbrev
    return abbrev


_player_position_cache = {}

def _get_player_position(conn: sqlite3.Connection, player_id: int) -> str:
    """Get player position."""
    if player_id in _player_position_cache:
        return _player_position_cache[player_id]

    cursor = conn.cursor()
    cursor.execute("SELECT position FROM players WHERE player_id = ?", [player_id])
    result = cursor.fetchone()
    position = result[0] if result else "Guard"
    _player_position_cache[player_id] = position
    return position


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Apply enrichments to predictions')
    parser.add_argument('--date', type=str, required=True, help='Game date (YYYY-MM-DD)')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    parser.add_argument('--no-adjust', action='store_true', help="Don't modify projections")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    stats = apply_enrichments_to_predictions(
        conn,
        args.date,
        apply_adjustments=not args.no_adjust,
        verbose=True
    )

    print("\nDone!")
    conn.close()
