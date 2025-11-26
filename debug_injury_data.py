#!/usr/bin/env python3
"""Debug script to check injury impact data quality."""

import sqlite3
from pathlib import Path
import pandas as pd

db_path = Path(__file__).parent / "nba_stats.db"
conn = sqlite3.connect(db_path)

# Let's check a specific player - James Harden
player_name = "James Harden"

# Get player ID
player_query = """
SELECT player_id, player_name, team_name, games_played
FROM player_season_totals
WHERE player_name LIKE ?
  AND season = '2025-26'
  AND season_type = 'Regular Season'
"""

players = pd.read_sql_query(player_query, conn, params=[f"%{player_name}%"])
print("Players matching search:")
print(players)
print()

if not players.empty:
    player_id = players.iloc[0]['player_id']
    print(f"Analyzing player_id: {player_id}")
    print()

    # Get team ID
    team_query = """
    SELECT DISTINCT team_id
    FROM player_game_logs
    WHERE player_id = ? AND season = '2025-26' AND season_type = 'Regular Season'
    """
    team_df = pd.read_sql_query(team_query, conn, params=[player_id])

    if not team_df.empty:
        team_id = team_df.iloc[0]['team_id']
        print(f"Team ID: {team_id}")
        print()

        # Get ALL team games
        all_team_games = """
        SELECT game_id, game_date, pts, opp_pts, team_name
        FROM team_game_logs
        WHERE team_id = ?
          AND season = '2025-26'
          AND season_type = 'Regular Season'
        ORDER BY game_date
        """
        team_games = pd.read_sql_query(all_team_games, conn, params=[team_id])
        print(f"Total team games: {len(team_games)}")
        print("Sample team games:")
        print(team_games.head(10))
        print()

        # Get games WHERE PLAYER PLAYED
        player_games_query = """
        SELECT game_id, game_date, points, minutes
        FROM player_game_logs
        WHERE player_id = ?
          AND season = '2025-26'
          AND season_type = 'Regular Season'
        ORDER BY game_date
        """
        player_games = pd.read_sql_query(player_games_query, conn, params=[player_id])
        print(f"Games player participated in: {len(player_games)}")
        print()

        # Identify games player MISSED
        player_game_ids = set(player_games['game_id'].values)
        team_game_ids = set(team_games['game_id'].values)
        missed_game_ids = team_game_ids - player_game_ids

        print(f"Games player MISSED: {len(missed_game_ids)}")

        if missed_game_ids:
            missed_games = team_games[team_games['game_id'].isin(missed_game_ids)]
            print("Missed games details:")
            print(missed_games[['game_id', 'game_date', 'pts', 'opp_pts', 'team_name']])
            print()

            # Check for NULL values
            print("Checking for NULL/NaN values in missed games:")
            print(f"  pts is NULL: {missed_games['pts'].isna().sum()} games")
            print(f"  opp_pts is NULL: {missed_games['opp_pts'].isna().sum()} games")
            print()

            # Check actual values
            print("Actual pts values in missed games:")
            print(missed_games['pts'].values)
            print("Actual opp_pts values in missed games:")
            print(missed_games['opp_pts'].values)

conn.close()
