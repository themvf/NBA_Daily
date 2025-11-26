#!/usr/bin/env python3
"""Check database schema and understand confidence issue"""
import sqlite3

conn = sqlite3.connect("nba_stats.db")
cursor = conn.cursor()

print("=== Checking team_game_logs schema ===")
cursor.execute("PRAGMA table_info(team_game_logs)")
columns = cursor.fetchall()
print("Columns in team_game_logs:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

has_opp_team_id = any(col[1] == 'opp_team_id' for col in columns)
print(f"\nHas opp_team_id column: {has_opp_team_id}")

if has_opp_team_id:
    print("\n=== Checking opponent data quality ===")
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(opp_team_id) as with_opp,
            COUNT(CASE WHEN opp_team_id IS NOT NULL THEN 1 END) as not_null_opp
        FROM team_game_logs
        WHERE season = '2025-26'
    """)
    stats = cursor.fetchone()
    print(f"Total 2025-26 team game logs: {stats[0]}")
    print(f"With opponent ID: {stats[1] or stats[2]}")

    if stats[1] > 0 or stats[2] > 0:
        print("\n=== Sample player vs team matchups ===")
        cursor.execute("""
            SELECT
                p.player_name,
                t.opp_team_abbreviation,
                COUNT(*) as games,
                AVG(p.points) as avg_pts
            FROM player_game_logs p
            JOIN team_game_logs t
                ON p.game_id = t.game_id
                AND p.team_id = t.team_id
            WHERE p.season = '2025-26'
                AND t.opp_team_id IS NOT NULL
                AND p.points IS NOT NULL
            GROUP BY p.player_id, t.opp_team_id
            HAVING games >= 2
            LIMIT 5
        """)
        matchups = cursor.fetchall()
        if matchups:
            for m in matchups:
                print(f"  {m[0]} vs {m[1]}: {m[2]} games, {m[3]:.1f} PPG avg")
        else:
            print("  No player-team matchups with 2+ games found")

conn.close()
