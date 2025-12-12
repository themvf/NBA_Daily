#!/usr/bin/env python3
"""
Position-Specific PPM Statistics Module with Bayesian Blending

This module calculates position-specific defensive PPM and blends it with
team-level defensive PPM using Bayesian weighting to avoid overfitting.

Key Features:
- Position-specific defensive PPM (vs Guards/Forwards/Centers)
- Bayesian blending with sample size weighting
- Season progress adjustment (early season = more conservative)
- Exploit detection (position-specific weaknesses)

Example Use Case:
- Zubac @ Cleveland: CLE is B grade overall (0.481 PPM) but D grade vs Centers (0.550 PPM)
- Blended PPM accounts for this position-specific weakness
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


def get_position_defensive_ppm(
    conn: sqlite3.Connection,
    team_id: int,
    position: str,
    season: str = '2025-26'
) -> Optional[Dict]:
    """
    Calculate team's defensive PPM allowed vs specific position.

    Args:
        conn: Database connection
        team_id: Defensive team's ID
        position: "Guard", "Forward", or "Center"
        season: Season string (e.g., "2025-26")

    Returns:
        Dict with:
        - avg_def_ppm_vs_position: Average PPM allowed to this position
        - std_def_ppm: Standard deviation
        - games_count: Number of games with data vs this position
        - min_ppm, max_ppm, p90_ppm: Distribution stats
        - None if insufficient data (<3 games)
    """
    # Get all games where this team played defense against the position
    query = """
        SELECT
            pgl.game_id,
            pgl.player_name,
            pgl.points,
            CAST(pgl.minutes AS REAL) as minutes,
            pgl.points / CAST(pgl.minutes AS REAL) as ppm
        FROM player_game_logs pgl
        JOIN players p ON pgl.player_id = p.player_id
        JOIN team_game_logs tgl ON pgl.game_id = tgl.game_id AND pgl.team_id = tgl.team_id
        WHERE tgl.opp_team_id = ?
          AND p.position = ?
          AND pgl.season = ?
          AND CAST(pgl.minutes AS REAL) >= 10
    """

    df = pd.read_sql_query(query, conn, params=[team_id, position, season])

    if len(df) < 3:
        return None

    # Calculate aggregate stats
    avg_ppm = df['ppm'].mean()
    std_ppm = df['ppm'].std()
    games_count = df['game_id'].nunique()

    return {
        'avg_def_ppm_vs_position': avg_ppm,
        'std_def_ppm': std_ppm,
        'games_count': games_count,
        'min_ppm': df['ppm'].min(),
        'max_ppm': df['ppm'].max(),
        'p90_ppm': df['ppm'].quantile(0.90)
    }


def calculate_blended_defensive_ppm(
    team_overall_ppm: float,
    position_specific_ppm: Optional[float],
    games_vs_position: int,
    season_progress: float
) -> Tuple[float, float, str]:
    """
    Bayesian blend of team and position-specific PPM.

    Formula:
    1. Base weight: 10% (start conservative)
    2. Sample size adjustment: Increase weight based on games played
       - 0-5 games: Stay at 10%
       - 5-10 games: Linear increase to 30%
       - 10+ games: Linear increase to 60% (max)
    3. Season progress adjustment: Multiply by season progress (0.5 to 1.0)
       - Early season (0-25%): 0.5x weight (more conservative)
       - Mid season (25-75%): 0.75x weight
       - Late season (75%+): 1.0x weight (full confidence)

    Args:
        team_overall_ppm: Team's overall defensive PPM allowed
        position_specific_ppm: PPM allowed vs specific position (None if no data)
        games_vs_position: Number of games with data vs this position
        season_progress: Fraction of season completed (0.0 to 1.0)

    Returns:
        Tuple of (blended_ppm, position_weight, blend_strategy):
        - blended_ppm: Weighted average of team and position PPM
        - position_weight: Final weight applied to position PPM (0.0 to 0.6)
        - blend_strategy: Description of blending approach used
    """
    # If no position-specific data, return team overall
    if position_specific_ppm is None or games_vs_position < 3:
        return team_overall_ppm, 0.0, "team_only (insufficient position data)"

    # Base weight: 10%
    base_weight = 0.10

    # Sample size adjustment (0 to 0.60 max)
    if games_vs_position <= 5:
        sample_size_weight = base_weight  # Stay at 10%
    elif games_vs_position <= 10:
        # Linear from 10% to 30% over 5-10 games
        progress = (games_vs_position - 5) / 5.0
        sample_size_weight = base_weight + (progress * (0.30 - base_weight))
    else:
        # Linear from 30% to 60% over 10-20 games
        progress = min((games_vs_position - 10) / 10.0, 1.0)
        sample_size_weight = 0.30 + (progress * (0.60 - 0.30))

    # Season progress adjustment (0.5 to 1.0 multiplier)
    if season_progress < 0.25:
        season_multiplier = 0.5  # Early season: very conservative
    elif season_progress < 0.75:
        season_multiplier = 0.75  # Mid season: moderate confidence
    else:
        season_multiplier = 1.0  # Late season: full confidence

    # Final position weight
    position_weight = sample_size_weight * season_multiplier

    # Blended PPM
    blended_ppm = (position_specific_ppm * position_weight) + (team_overall_ppm * (1 - position_weight))

    # Blend strategy description
    blend_strategy = f"blended ({position_weight:.0%} position, {1-position_weight:.0%} team)"

    return blended_ppm, position_weight, blend_strategy


def get_blended_def_ppm_for_matchup(
    conn: sqlite3.Connection,
    def_team_id: int,
    off_player_position: str,
    team_overall_def_ppm: float,
    season: str = '2025-26',
    current_date: Optional[str] = None
) -> Dict:
    """
    Get blended defensive PPM for specific matchup with full context.

    Args:
        conn: Database connection
        def_team_id: Defensive team's ID
        off_player_position: Offensive player's position ("Guard", "Forward", "Center")
        team_overall_def_ppm: Team's overall defensive PPM (from ppm_stats.py)
        season: Season string
        current_date: Date string for season progress calculation (YYYY-MM-DD)

    Returns:
        Dict with:
        - blended_def_ppm: Final blended PPM value to use for projection
        - team_overall_ppm: Team's overall defensive PPM
        - position_specific_ppm: PPM allowed vs this position (None if no data)
        - position_weight: Weight applied to position data (0.0 to 0.6)
        - blend_strategy: Description of blending approach
        - games_vs_position: Number of games with position data
        - exploit_detected: Boolean indicating position-specific weakness
        - exploit_severity: "none", "minor", "moderate", "severe"
    """
    # Get position-specific data
    position_data = get_position_defensive_ppm(conn, def_team_id, off_player_position, season)

    # Calculate season progress (assume season: Oct 22, 2025 - Apr 13, 2026 = 173 days)
    if current_date:
        try:
            season_start = datetime(2025, 10, 22)
            current = datetime.strptime(current_date, '%Y-%m-%d')
            days_elapsed = (current - season_start).days
            season_progress = min(max(days_elapsed / 173.0, 0.0), 1.0)
        except:
            season_progress = 0.30  # Default: ~30% through season
    else:
        season_progress = 0.30

    # Extract position-specific PPM
    position_ppm = position_data['avg_def_ppm_vs_position'] if position_data else None
    games_vs_position = position_data['games_count'] if position_data else 0

    # Calculate blended PPM
    blended_ppm, position_weight, blend_strategy = calculate_blended_defensive_ppm(
        team_overall_ppm=team_overall_def_ppm,
        position_specific_ppm=position_ppm,
        games_vs_position=games_vs_position,
        season_progress=season_progress
    )

    # Exploit detection
    # Compare position-specific to team overall (if available)
    exploit_detected = False
    exploit_severity = "none"

    if position_ppm is not None and games_vs_position >= 5:
        ppm_diff = position_ppm - team_overall_def_ppm
        ppm_diff_pct = ppm_diff / team_overall_def_ppm

        if ppm_diff_pct >= 0.08:  # 8%+ worse vs position
            exploit_detected = True
            if ppm_diff_pct >= 0.15:
                exploit_severity = "severe"  # 15%+ worse (Zubac @ CLE scenario)
            elif ppm_diff_pct >= 0.12:
                exploit_severity = "moderate"  # 12-15% worse
            else:
                exploit_severity = "minor"  # 8-12% worse

    return {
        'blended_def_ppm': blended_ppm,
        'team_overall_ppm': team_overall_def_ppm,
        'position_specific_ppm': position_ppm,
        'position_weight': position_weight,
        'blend_strategy': blend_strategy,
        'games_vs_position': games_vs_position,
        'season_progress': season_progress,
        'exploit_detected': exploit_detected,
        'exploit_severity': exploit_severity,
        'ppm_diff': position_ppm - team_overall_def_ppm if position_ppm else 0.0
    }


def analyze_all_position_matchups(
    conn: sqlite3.Connection,
    team_id: int,
    team_overall_def_ppm: float,
    season: str = '2025-26'
) -> pd.DataFrame:
    """
    Analyze team's defensive performance vs all three positions.

    Useful for identifying position-specific exploits.

    Args:
        conn: Database connection
        team_id: Defensive team's ID
        team_overall_def_ppm: Team's overall defensive PPM
        season: Season string

    Returns:
        DataFrame with one row per position (Guard, Forward, Center):
        - position
        - avg_def_ppm_vs_position
        - games_count
        - diff_from_team (position PPM - team PPM)
        - diff_pct (percentage difference)
        - exploit_severity
    """
    positions = ['Guard', 'Forward', 'Center']
    results = []

    for position in positions:
        position_data = get_position_defensive_ppm(conn, team_id, position, season)

        if position_data:
            avg_ppm = position_data['avg_def_ppm_vs_position']
            games = position_data['games_count']
            diff = avg_ppm - team_overall_def_ppm
            diff_pct = diff / team_overall_def_ppm

            # Exploit severity
            if games >= 5 and diff_pct >= 0.15:
                severity = "severe"
            elif games >= 5 and diff_pct >= 0.12:
                severity = "moderate"
            elif games >= 5 and diff_pct >= 0.08:
                severity = "minor"
            else:
                severity = "none"

            results.append({
                'position': position,
                'avg_def_ppm_vs_position': avg_ppm,
                'std_def_ppm': position_data['std_def_ppm'],
                'games_count': games,
                'diff_from_team': diff,
                'diff_pct': diff_pct,
                'exploit_severity': severity
            })
        else:
            results.append({
                'position': position,
                'avg_def_ppm_vs_position': None,
                'std_def_ppm': None,
                'games_count': 0,
                'diff_from_team': 0.0,
                'diff_pct': 0.0,
                'exploit_severity': 'insufficient_data'
            })

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Test the module
    conn = sqlite3.connect('nba_stats.db')

    # Import ppm_stats to get team overall PPM
    import ppm_stats

    print("Position-Specific PPM Analysis with Bayesian Blending")
    print("=" * 80)

    # Get team defensive PPM stats
    def_ppm_df = ppm_stats.get_team_defensive_ppm_stats(conn)

    # Test Case: Cleveland Cavaliers vs Centers (Zubac scenario)
    # Find Cleveland's team_id
    cle_row = def_ppm_df[def_ppm_df['team_name'].str.contains('Cleveland', case=False, na=False)]

    if not cle_row.empty:
        cle_team_id = cle_row['team_id'].iloc[0]
        cle_team_name = cle_row['team_name'].iloc[0]
        cle_overall_ppm = cle_row['avg_def_ppm'].iloc[0]

        print(f"\nTest Case: {cle_team_name}")
        print(f"Overall Defensive PPM: {cle_overall_ppm:.3f}")
        print("-" * 80)

        # Analyze all position matchups
        position_analysis = analyze_all_position_matchups(conn, cle_team_id, cle_overall_ppm)
        print("\nPosition-Specific Breakdown:")
        print(position_analysis.to_string(index=False))

        # Test blended matchup for Centers
        print("\n" + "=" * 80)
        print("Blended PPM Calculation for Center Matchup:")
        print("-" * 80)

        blended_result = get_blended_def_ppm_for_matchup(
            conn=conn,
            def_team_id=cle_team_id,
            off_player_position='Center',
            team_overall_def_ppm=cle_overall_ppm,
            season='2025-26',
            current_date='2025-11-23'  # Zubac game date
        )

        print(f"Team Overall PPM: {blended_result['team_overall_ppm']:.3f}")
        print(f"Position-Specific PPM (vs Centers): {blended_result['position_specific_ppm']:.3f}" if blended_result['position_specific_ppm'] else "Position-Specific PPM: No data")
        print(f"Blended PPM: {blended_result['blended_def_ppm']:.3f}")
        print(f"Position Weight: {blended_result['position_weight']:.1%}")
        print(f"Blend Strategy: {blended_result['blend_strategy']}")
        print(f"Games vs Centers: {blended_result['games_vs_position']}")
        print(f"Season Progress: {blended_result['season_progress']:.1%}")
        print(f"Exploit Detected: {'YES' if blended_result['exploit_detected'] else 'No'}")
        print(f"Exploit Severity: {blended_result['exploit_severity']}")
        print(f"PPM Difference: {blended_result['ppm_diff']:+.3f} ({blended_result['ppm_diff']/blended_result['team_overall_ppm']*100:+.1f}%)")

        # Show projection impact
        print("\n" + "=" * 80)
        print("Projection Impact Example (Zubac @ CLE):")
        print("-" * 80)

        league_avg_ppm = ppm_stats.get_league_avg_ppm(conn)
        zubac_season_ppm = 0.726  # From previous analysis
        expected_minutes = 35

        # Old method (team overall only)
        old_boost = cle_overall_ppm / league_avg_ppm
        old_projection = expected_minutes * zubac_season_ppm * old_boost

        # New method (blended)
        new_boost = blended_result['blended_def_ppm'] / league_avg_ppm
        new_projection = expected_minutes * zubac_season_ppm * new_boost

        print(f"League Avg PPM: {league_avg_ppm:.3f}")
        print(f"Zubac Season PPM: {zubac_season_ppm:.3f}")
        print(f"Expected Minutes: {expected_minutes}")
        print()
        print(f"OLD (Team Overall): {cle_overall_ppm:.3f} PPM -> Boost {old_boost:.3f} -> {old_projection:.1f} PPG")
        print(f"NEW (Blended):      {blended_result['blended_def_ppm']:.3f} PPM -> Boost {new_boost:.3f} -> {new_projection:.1f} PPG")
        print(f"Improvement: {new_projection - old_projection:+.1f} PPG ({(new_projection - old_projection)/old_projection*100:+.1f}%)")
        print(f"Actual Result: 33.0 PPG")

    else:
        print("Cleveland not found in database")

    conn.close()
    print("\n" + "=" * 80)
    print("Position-Specific PPM Module test complete!")
    print("=" * 80 + "\n")
