#!/usr/bin/env python3
"""
Quick debug script to check confidence calculation inputs
"""
import sqlite3
import sys

db_path = "nba_stats.db"

# Check a sample of players
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Checking player vs team history...")
cursor.execute("""
    SELECT COUNT(*) as total_rows,
           COUNT(CASE WHEN opp_team_id IS NOT NULL THEN 1 END) as with_opponent,
           COUNT(DISTINCT player_id) as unique_players
    FROM player_game_logs
    WHERE season = '2025-26' AND season_type = 'Regular Season'
""")
row = cursor.fetchone()
print(f"Total player game logs: {row[0]}")
print(f"Rows with opponent: {row[1]}")
print(f"Unique players: {row[2]}")

print("\nChecking team matchups (games between specific teams)...")
cursor.execute("""
    SELECT
        player_name,
        COUNT(*) as games,
        AVG(points) as avg_pts,
        opp_team_id
    FROM player_game_logs p
    JOIN team_game_logs t ON p.game_id = t.game_id AND p.team_id = t.team_id
    WHERE p.season = '2025-26'
      AND p.season_type = 'Regular Season'
      AND t.opp_team_id IS NOT NULL
      AND p.points IS NOT NULL
    GROUP BY p.player_id, t.opp_team_id
    HAVING games >= 2
    LIMIT 10
""")
matchups = cursor.fetchall()
print(f"Found {len(matchups)} player-team matchups with 2+ games:")
for m in matchups:
    print(f"  {m[0]}: {m[1]} games vs team {m[3]}, avg {m[2]:.1f} PPG")

print("\nChecking defense stats availability...")
cursor.execute("""
    SELECT COUNT(*)
    FROM team_game_logs
    WHERE season = '2025-26'
      AND season_type = 'Regular Season'
      AND opp_fga IS NOT NULL
""")
def_count = cursor.fetchone()[0]
print(f"Team games with defense stats: {def_count}")

conn.close()
