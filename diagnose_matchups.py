#!/usr/bin/env python3
"""Diagnose why 'Vs This Team' shows N/A"""
import sqlite3
import sys

conn = sqlite3.connect("nba_stats.db")

# Check 1: How many team game logs have opponent IDs?
print("=== Database Quality Check ===\n")
cursor = conn.execute("""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN opp_team_id IS NOT NULL THEN 1 ELSE 0 END) as with_opp
    FROM team_game_logs
    WHERE season = '2025-26'
""")
total, with_opp = cursor.fetchone()
print(f"Team game logs (2025-26): {total} total, {with_opp} with opponent ID")
print(f"Missing opponent IDs: {total - with_opp}")

# Check 2: How many player-team matchups exist?
print("\n=== Player vs Team Matchups ===\n")
cursor = conn.execute("""
    SELECT COUNT(DISTINCT p.player_id || '-' || t.opp_team_id) as unique_matchups,
           COUNT(*) as total_games
    FROM player_game_logs p
    JOIN team_game_logs t ON p.game_id = t.game_id AND p.team_id = t.team_id
    WHERE p.season = '2025-26'
      AND t.opp_team_id IS NOT NULL
      AND p.points IS NOT NULL
""")
matchups, games = cursor.fetchone()
print(f"Unique player-team combinations: {matchups}")
print(f"Total games tracked: {games}")

# Check 3: How many have 2+ games (minimum for matchup rating)?
cursor = conn.execute("""
    SELECT COUNT(*) as matchups_with_2plus
    FROM (
        SELECT p.player_id, t.opp_team_id, COUNT(*) as games
        FROM player_game_logs p
        JOIN team_game_logs t ON p.game_id = t.game_id AND p.team_id = t.team_id
        WHERE p.season = '2025-26'
          AND t.opp_team_id IS NOT NULL
          AND p.points IS NOT NULL
        GROUP BY p.player_id, t.opp_team_id
        HAVING games >= 2
    )
""")
matchups_2plus = cursor.fetchone()[0]
print(f"Player-team matchups with 2+ games: {matchups_2plus}")

# Check 4: Sample some actual matchups
print("\n=== Sample Player-Team Matchups ===\n")
cursor = conn.execute("""
    SELECT p.player_name,
           t.opp_team_abbreviation,
           COUNT(*) as games,
           ROUND(AVG(p.points), 1) as avg_pts,
           ROUND(AVG(p.fg3m), 1) as avg_3pm
    FROM player_game_logs p
    JOIN team_game_logs t ON p.game_id = t.game_id AND p.team_id = t.team_id
    WHERE p.season = '2025-26'
      AND t.opp_team_id IS NOT NULL
      AND p.points IS NOT NULL
    GROUP BY p.player_id, t.opp_team_id
    HAVING games >= 2
    ORDER BY avg_pts DESC
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"  {row[0]:20} vs {row[1]:3}: {row[2]} games, {row[3]} PPG, {row[4]} 3PM")

conn.close()

print("\n=== DIAGNOSIS ===")
if matchups_2plus == 0:
    print("❌ ISSUE: No player-team matchups with 2+ games found")
    print("   → Early in season - players haven't faced most opponents twice yet")
    print("   → This is EXPECTED behavior in November/December")
    print("   → Matchup ratings will improve as season progresses")
elif matchups_2plus < 50:
    print("⚠️  LIMITED DATA: Only a few matchups with 2+ games")
    print("   → Most players will show N/A for 'Vs This Team'")
    print("   → Algorithm correctly falling back to defense style")
else:
    print("✅ DATA AVAILABLE: Matchups with history exist")
    print("   → Issue may be in display logic, not data availability")
    print("   → Check build_player_vs_team_history() function")
