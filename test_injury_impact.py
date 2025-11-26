#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify injury impact analytics functionality.
"""

import sqlite3
import sys

# Ensure UTF-8 output for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import injury_impact_analytics as iia

def test_injury_impact():
    """Test the injury impact analytics module."""
    db_path = Path(__file__).parent / "nba_stats.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Please ensure nba_stats.db exists in the current directory.")
        return

    conn = sqlite3.connect(db_path)
    print("[OK] Connected to database")

    # Test getting significant players
    print("\n1. Testing get_significant_players()...")
    significant_players = iia.get_significant_players(
        conn,
        season="2025-26",
        season_type="Regular Season",
        min_games=10,
        min_avg_points=10.0,
    )

    if significant_players.empty:
        print("  [WARN] No significant players found for 2025-26 season")
        print("  Trying 2024-25 season instead...")

        significant_players = iia.get_significant_players(
            conn,
            season="2024-25",
            season_type="Regular Season",
            min_games=10,
            min_avg_points=10.0,
        )

        if significant_players.empty:
            print("  [WARN] No significant players found for 2024-25 either")
            print("  Database may not have enough game log data yet.")
            conn.close()
            return
        else:
            test_season = "2024-25"
    else:
        test_season = "2025-26"

    print(f"  [OK] Found {len(significant_players)} significant players in {test_season}")
    print(f"  Top 5 players by PPG:")
    for idx, row in significant_players.head(5).iterrows():
        print(f"    - {row['player_name']} ({row['team_name']}): {row['avg_points']} PPG")

    # Test team impact analysis for the top scorer
    if len(significant_players) > 0:
        top_player = significant_players.iloc[0]
        player_id = top_player['player_id']
        player_name = top_player['player_name']

        print(f"\n2. Testing calculate_team_impact() for {player_name}...")
        team_impact = iia.calculate_team_impact(
            conn,
            player_id,
            player_name,
            test_season,
            "Regular Season",
        )

        if team_impact:
            print(f"  [OK] Team Impact Analysis:")
            print(f"    Team: {team_impact.team_name}")
            print(f"    Games played: {team_impact.games_played}")
            print(f"    Games absent: {team_impact.games_absent}")
            print(f"    Win % with player: {team_impact.team_win_pct_with:.1%}")
            print(f"    Win % without player: {team_impact.team_win_pct_without:.1%}")
            print(f"    Win % delta: {team_impact.win_pct_delta:+.1%}")
            print(f"    PPG with player: {team_impact.team_avg_pts_with:.1f}")
            print(f"    PPG without player: {team_impact.team_avg_pts_without:.1f}")
            print(f"    Offensive delta: {team_impact.offensive_rating_delta:+.1f}")
        else:
            print(f"  [WARN] No team impact data available for {player_name}")

        # Test teammate redistribution
        print(f"\n3. Testing calculate_teammate_redistribution() for {player_name}...")
        teammate_impacts = iia.calculate_teammate_redistribution(
            conn,
            player_id,
            test_season,
            "Regular Season",
            min_games=3,
        )

        if teammate_impacts:
            print(f"  [OK] Found {len(teammate_impacts)} teammates with sufficient data")
            print(f"  Top 3 most affected teammates:")
            for tm in teammate_impacts[:3]:
                print(f"    - {tm.teammate_name}:")
                print(f"      PPG delta: {tm.pts_delta:+.1f} ({tm.avg_pts_with:.1f} -> {tm.avg_pts_without:.1f})")
                print(f"      Usage delta: {tm.usg_delta:+.1f}%")
        else:
            print(f"  [WARN] No teammate redistribution data available (need 3+ games in each scenario)")

        # Test opponent impact
        print(f"\n4. Testing calculate_opponent_impact() for {player_name}...")
        opponent_impact = iia.calculate_opponent_impact(
            conn,
            player_id,
            test_season,
            "Regular Season",
        )

        if opponent_impact:
            print(f"  [OK] Opponent Impact Analysis:")
            print(f"    Opp PPG with player: {opponent_impact.avg_opp_pts_with:.1f}")
            print(f"    Opp PPG without player: {opponent_impact.avg_opp_pts_without:.1f}")
            print(f"    Opp PPG delta: {opponent_impact.opp_pts_delta:+.1f}")
            print(f"    Opp 3PM with player: {opponent_impact.avg_opp_fg3m_with:.1f}")
            print(f"    Opp 3PM without player: {opponent_impact.avg_opp_fg3m_without:.1f}")
            print(f"    Opp 3PM delta: {opponent_impact.opp_fg3m_delta:+.1f}")
        else:
            print(f"  [WARN] No opponent impact data available for {player_name}")

    conn.close()
    print("\n[SUCCESS] All tests completed successfully!")


if __name__ == "__main__":
    test_injury_impact()
