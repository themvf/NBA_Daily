#!/usr/bin/env python3
"""Simple debug to check James Harden's game data."""

import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "nba_stats.db"
conn = sqlite3.connect(db_path)

# Check James Harden's player game logs
query = """
SELECT COUNT(*) as game_count, team_id, team_name
FROM player_game_logs
WHERE player_id = 201935
  AND season = '2025-26'
  AND season_type = 'Regular Season'
GROUP BY team_id, team_name
"""

cursor = conn.cursor()
cursor.execute(query)
results = cursor.fetchall()

print("James Harden's game logs:")
for row in results:
    print(f"  Games: {row[0]}, Team ID: {row[1]}, Team: {row[2]}")
print()

# Now check team game logs for that team
if results:
    team_id = results[0][1]
    print(f"Checking all games for team_id {team_id}:")

    team_query = """
    SELECT game_id, game_date, pts, opp_pts
    FROM team_game_logs
    WHERE team_id = ?
      AND season = '2025-26'
      AND season_type = 'Regular Season'
    ORDER BY game_date
    LIMIT 20
    """

    cursor.execute(team_query, [team_id])
    team_games = cursor.fetchall()

    print(f"Total team games found: {len(team_games)}")
    for game in team_games:
        game_id, date, pts, opp_pts = game
        print(f"  {date}: pts={pts}, opp_pts={opp_pts}, game_id={game_id}")

conn.close()
