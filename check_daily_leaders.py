#!/usr/bin/env python3
"""
Quick diagnostic script to check why Daily Leaders page is empty.
"""

import sqlite3
import sys
from pathlib import Path

db_path = "nba_2026.db"
if len(sys.argv) > 1:
    db_path = sys.argv[1]

print(f"Checking database: {db_path}")
print("=" * 70)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check 1: Does player_game_logs table exist?
print("\n1. Checking if player_game_logs table exists...")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_game_logs'")
table_exists = cursor.fetchone()
if table_exists:
    print("   [OK] player_game_logs table EXISTS")
else:
    print("   [ERROR] player_game_logs table MISSING!")
    conn.close()
    sys.exit(1)

# Check 2: How many total rows?
print("\n2. Checking total rows in player_game_logs...")
cursor.execute("SELECT COUNT(*) FROM player_game_logs")
total_rows = cursor.fetchone()[0]
print(f"   Total rows: {total_rows:,}")

if total_rows == 0:
    print("   [ERROR] Table is EMPTY! No game logs at all.")
    conn.close()
    sys.exit(1)

# Check 3: What seasons are available?
print("\n3. Checking available seasons...")
cursor.execute("""
    SELECT season, COUNT(*) as games, MIN(game_date) as first_game, MAX(game_date) as last_game
    FROM player_game_logs
    GROUP BY season
    ORDER BY season DESC
""")
seasons = cursor.fetchall()
for season, games, first_date, last_date in seasons:
    print(f"   Season {season}: {games:,} games ({first_date} to {last_date})")

# Check 4: What's the query that Daily Leaders uses?
print("\n4. Running the ACTUAL Daily Leaders query (top 10 scorers per day)...")
cursor.execute("""
    WITH ranked AS (
        SELECT
            player_id,
            player_name,
            team_name,
            game_date,
            matchup,
            points,
            minutes,
            ROW_NUMBER() OVER (
                PARTITION BY game_date
                ORDER BY points DESC
            ) AS rn
        FROM player_game_logs
        WHERE season = ?
          AND season_type = ?
          AND points IS NOT NULL
    )
    SELECT
        game_date,
        player_name,
        team_name,
        points,
        COUNT(*) OVER() as total_results
    FROM ranked
    WHERE rn <= 10
    ORDER BY game_date DESC, points DESC
    LIMIT 30
""", ("2025-26", "Regular Season"))

results = cursor.fetchall()
if results:
    print(f"   [OK] Query returned {results[0][4] if results else 0} results")
    print("\n   Sample (most recent 10 scorers):")
    for i, (game_date, player, team, pts, _) in enumerate(results[:10], 1):
        print(f"   {i:2d}. {game_date} - {player:25s} ({team:20s}): {pts} pts")
else:
    print("   [ERROR] Query returned ZERO results for season '2025-26', season_type 'Regular Season'")
    print("\n   Trying 2024-25 season instead...")

    cursor.execute("""
        WITH ranked AS (
            SELECT
                player_id,
                player_name,
                team_name,
                game_date,
                matchup,
                points,
                minutes,
                ROW_NUMBER() OVER (
                    PARTITION BY game_date
                    ORDER BY points DESC
                ) AS rn
            FROM player_game_logs
            WHERE season = ?
              AND season_type = ?
              AND points IS NOT NULL
        )
        SELECT
            game_date,
            player_name,
            team_name,
            points,
            COUNT(*) OVER() as total_results
        FROM ranked
        WHERE rn <= 10
        ORDER BY game_date DESC, points DESC
        LIMIT 30
    """, ("2024-25", "Regular Season"))

    results_2024 = cursor.fetchall()
    if results_2024:
        print(f"   [OK] 2024-25 season HAS data! {results_2024[0][4] if results_2024 else 0} results")
        print("\n   Sample (most recent 10 scorers from 2024-25):")
        for i, (game_date, player, team, pts, _) in enumerate(results_2024[:10], 1):
            print(f"   {i:2d}. {game_date} - {player:25s} ({team:20s}): {pts} pts")
    else:
        print("   [ERROR] Even 2024-25 season has no data!")

# Check 5: What about games from December 2025?
print("\n5. Checking for games specifically in December 2025...")
cursor.execute("""
    SELECT game_date, COUNT(DISTINCT player_id) as players, AVG(points) as avg_pts
    FROM player_game_logs
    WHERE game_date >= '2025-12-01' AND game_date <= '2025-12-31'
    GROUP BY game_date
    ORDER BY game_date DESC
""")
dec_games = cursor.fetchall()
if dec_games:
    print(f"   [OK] Found {len(dec_games)} game dates in December 2025:")
    for game_date, players, avg_pts in dec_games[:10]:
        print(f"      {game_date}: {players} players (avg {avg_pts:.1f} pts)")
else:
    print("   [ERROR] NO games found in December 2025")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("-" * 70)

if total_rows == 0:
    print("[ERROR] Your database has NO game logs at all!")
    print("   You need to run the database builder to populate player_game_logs.")
elif not results:
    print("[ERROR] Your database has no games for season '2025-26'")
    print(f"   Available seasons: {', '.join([s[0] for s in seasons])}")
    print("\n   SOLUTION: Change 'Standings season for context' in Today's Games")
    print(f"   from '2025-26' to '{seasons[0][0]}' (your most recent season)")
else:
    print("[OK] Daily Leaders query should work!")
    print("   If the page still shows 'no data', try refreshing the Streamlit app.")

conn.close()
