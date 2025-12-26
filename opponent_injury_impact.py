#!/usr/bin/env python3
"""
Opponent Injury Impact Analysis

Tracks how opponent team injuries affect our player's scoring opportunities.

Key Concept: When opponent's primary defender or rim protector is OUT,
our offensive players face easier matchups → increase ceiling & projection.

Usage:
    import opponent_injury_impact as oii

    impact = oii.calculate_opponent_injury_impact(
        conn,
        player_position='C',
        opponent_team_id=1610612750,  # MIN
        game_date='2025-12-25'
    )

    if impact['has_significant_injuries']:
        ceiling_boost = impact['ceiling_boost_pct']  # e.g., 0.12 = +12%
        projection_boost = impact['projection_boost_pct']  # e.g., 0.05 = +5%
"""

import sqlite3
from typing import Dict, List, Optional
from datetime import datetime


# Position matchup matrix - who guards who
DEFENSIVE_MATCHUPS = {
    'C': ['C'],  # Centers guard centers
    'PF': ['PF', 'C'],  # Power forwards guard PF/C
    'SF': ['SF', 'PF'],  # Small forwards guard SF/PF
    'SG': ['SG', 'SF'],  # Shooting guards guard SG/SF
    'PG': ['PG', 'SG'],  # Point guards guard PG/SG
}

# Position importance for defensive impact
POSITION_DEFENSIVE_IMPORTANCE = {
    'C': 1.0,  # Rim protectors have highest impact
    'PF': 0.8,  # Secondary rim protection
    'SF': 0.6,  # Perimeter defense
    'SG': 0.7,  # Perimeter defense
    'PG': 0.6,  # Perimeter defense
}


def get_opponent_injuries(
    conn: sqlite3.Connection,
    opponent_team_id: int,
    game_date: Optional[str] = None
) -> List[Dict]:
    """
    Get current injuries for opponent team.

    Args:
        conn: Database connection
        opponent_team_id: Opponent team ID
        game_date: Optional game date (YYYY-MM-DD)

    Returns:
        List of injury records with player details
    """
    query = """
        SELECT
            i.player_id,
            i.player_name,
            i.team_name,
            i.status,
            i.injury_type,
            i.expected_return_date,
            i.confidence,
            p.position
        FROM injury_list i
        LEFT JOIN players p ON i.player_id = p.player_id
        WHERE i.team_name = (SELECT team_name FROM teams WHERE team_id = ?)
          AND i.status IN ('out', 'doubtful')
    """

    # Add date filter if provided
    if game_date:
        query += " AND (i.expected_return_date IS NULL OR i.expected_return_date >= ?)"
        params = (opponent_team_id, game_date)
    else:
        params = (opponent_team_id,)

    cursor = conn.cursor()
    cursor.execute(query, params)

    injuries = []
    for row in cursor.fetchall():
        injuries.append({
            'player_id': row[0],
            'player_name': row[1],
            'team_name': row[2],
            'status': row[3],
            'injury_type': row[4],
            'expected_return': row[5],
            'confidence': row[6],
            'position': row[7]
        })

    return injuries


def get_player_importance(
    conn: sqlite3.Connection,
    player_id: int,
    season: str = '2025-26'
) -> Dict:
    """
    Get player's importance metrics (minutes, PPG, usage).

    Higher importance = bigger impact when absent.
    """
    query = """
        SELECT
            AVG(CAST(minutes AS REAL)) as avg_minutes,
            AVG(points) as avg_points,
            AVG(usg_pct) as avg_usage,
            COUNT(*) as games_played
        FROM player_game_logs
        WHERE player_id = ?
          AND season = ?
          AND CAST(minutes AS REAL) > 0
    """

    cursor = conn.cursor()
    cursor.execute(query, (player_id, season))
    result = cursor.fetchone()

    if not result or result[3] == 0:
        return None

    avg_minutes, avg_points, avg_usage, games = result

    # Calculate importance score (0-1)
    # Based on: minutes (40%), scoring (30%), usage (30%)
    minutes_score = min(1.0, (avg_minutes or 0) / 36.0)  # 36 min = full score
    points_score = min(1.0, (avg_points or 0) / 25.0)    # 25 PPG = full score
    usage_score = min(1.0, (avg_usage or 0.15) / 0.30)   # 30% usage = full score

    importance = (
        minutes_score * 0.40 +
        points_score * 0.30 +
        usage_score * 0.30
    )

    return {
        'avg_minutes': avg_minutes,
        'avg_points': avg_points,
        'avg_usage': avg_usage,
        'games_played': games,
        'importance_score': round(importance, 3)
    }


def calculate_opponent_injury_impact(
    conn: sqlite3.Connection,
    player_position: Optional[str],
    opponent_team_id: int,
    game_date: Optional[str] = None,
    season: str = '2025-26'
) -> Dict:
    """
    Calculate how opponent injuries affect our player's ceiling/projection.

    Args:
        conn: Database connection
        player_position: Our player's position (C, PF, SF, SG, PG)
        opponent_team_id: Opponent team ID
        game_date: Game date (YYYY-MM-DD)
        season: Season for historical data

    Returns:
        Dictionary with impact analysis
    """
    # Get opponent injuries
    injuries = get_opponent_injuries(conn, opponent_team_id, game_date)

    if not injuries:
        return {
            'has_significant_injuries': False,
            'ceiling_boost_pct': 0.0,
            'projection_boost_pct': 0.0,
            'injuries': [],
            'reason': 'No opponent injuries'
        }

    # Calculate cumulative impact
    total_impact = 0.0
    significant_injuries = []

    for injury in injuries:
        # Get player importance
        importance = get_player_importance(conn, injury['player_id'], season)

        if not importance:
            continue

        # FILTER 1: Only significant players (20+ MPG, key rotation)
        # Importance threshold: 0.45 = ~20 MPG, 12 PPG, 18% usage
        if importance['importance_score'] < 0.45:
            continue  # Skip bench players

        # FILTER 2: Direct positional matchup ONLY
        # Only count injuries that directly affect our player's matchup
        is_defensive_matchup = False
        if player_position and injury['position']:
            matchup_positions = DEFENSIVE_MATCHUPS.get(player_position, [])
            is_defensive_matchup = injury['position'] in matchup_positions

        # Skip if NOT a direct matchup
        if not is_defensive_matchup:
            continue

        # Calculate impact based on:
        # 1. Player importance (0-1)
        # 2. Position defensive importance (0-1)
        # 3. Direct matchup multiplier (1.5x)

        position_factor = POSITION_DEFENSIVE_IMPORTANCE.get(
            injury['position'],
            0.5
        )

        matchup_multiplier = 1.5  # Always 1.5x since we filtered for direct matchups only

        # Status factor (OUT = 1.0, DOUBTFUL = 0.5)
        status_factor = 1.0 if injury['status'] == 'out' else 0.5

        # Combined impact
        impact = (
            importance['importance_score'] *
            position_factor *
            matchup_multiplier *
            status_factor
        )

        total_impact += impact

        # Track for display (all direct matchup injuries are significant)
        significant_injuries.append({
            'player_name': injury['player_name'],
            'position': injury['position'],
            'status': injury['status'],
            'importance': importance['importance_score'],
            'impact': round(impact, 3),
            'is_direct_matchup': is_defensive_matchup
        })

    # Convert impact to boost percentages
    # Scale: 0.0-0.3 impact → 0-15% ceiling boost, 0-8% projection boost
    ceiling_boost = min(0.15, total_impact * 0.50)  # Max 15% ceiling boost
    projection_boost = min(0.08, total_impact * 0.27)  # Max 8% projection boost

    # Only flag as significant if we have direct matchup injuries
    # (not just any random opponent injury)
    has_significant = len(significant_injuries) > 0

    return {
        'has_significant_injuries': has_significant,
        'ceiling_boost_pct': round(ceiling_boost, 4),
        'projection_boost_pct': round(projection_boost, 4),
        'total_impact_score': round(total_impact, 3),
        'injuries': significant_injuries,
        'all_injuries': injuries,
        'reason': f"{len(injuries)} opponent injuries, {len(significant_injuries)} direct matchups"
    }


def format_opponent_injury_summary(impact: Dict) -> str:
    """
    Format opponent injury impact as readable string.

    Usage in UI:
        summary = format_opponent_injury_summary(impact)
        st.info(summary)
    """
    if not impact['has_significant_injuries']:
        return ""

    summary_parts = []

    # Header
    ceiling_pct = impact['ceiling_boost_pct'] * 100
    proj_pct = impact['projection_boost_pct'] * 100

    summary_parts.append(
        f"🚑 **Opponent Injuries Boost:** "
        f"+{ceiling_pct:.1f}% ceiling, +{proj_pct:.1f}% projection"
    )

    # List significant injuries
    for inj in impact['injuries']:
        matchup = " (DIRECT MATCHUP)" if inj['is_direct_matchup'] else ""
        summary_parts.append(
            f"  - {inj['player_name']} ({inj['position']}) - "
            f"{inj['status'].upper()}{matchup}"
        )

    return "\n".join(summary_parts)


if __name__ == "__main__":
    # Example usage
    import sys

    db_path = "nba_stats.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    conn = sqlite3.connect(db_path)

    # Example: Check Jokić vs MIN on Christmas
    print("=" * 70)
    print("OPPONENT INJURY IMPACT ANALYSIS")
    print("=" * 70)

    # Get Minnesota Timberwolves team_id
    cursor = conn.cursor()
    cursor.execute("SELECT team_id FROM teams WHERE team_name LIKE '%Minnesota%'")
    min_team = cursor.fetchone()

    if min_team:
        min_team_id = min_team[0]

        # Check impact for a center (Jokić) vs MIN
        impact = calculate_opponent_injury_impact(
            conn,
            player_position='C',
            opponent_team_id=min_team_id,
            game_date='2025-12-25'
        )

        print(f"\nPlayer Position: Center (e.g., Jokić)")
        print(f"Opponent: Minnesota Timberwolves")
        print(f"Game Date: 2025-12-25")
        print("-" * 70)
        print(f"Has Significant Injuries: {impact['has_significant_injuries']}")
        print(f"Ceiling Boost: +{impact['ceiling_boost_pct']*100:.1f}%")
        print(f"Projection Boost: +{impact['projection_boost_pct']*100:.1f}%")
        print(f"Total Impact Score: {impact['total_impact_score']:.3f}")

        if impact['injuries']:
            print("\nSignificant Injuries:")
            for inj in impact['injuries']:
                print(f"  - {inj['player_name']} ({inj['position']}) - {inj['status']}")
                print(f"    Importance: {inj['importance']:.3f}, Impact: {inj['impact']:.3f}")
                if inj['is_direct_matchup']:
                    print("    ⚠️ DIRECT DEFENSIVE MATCHUP")

        print("\n" + format_opponent_injury_summary(impact))

    conn.close()
