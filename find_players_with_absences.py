#!/usr/bin/env python3
"""Find players who have missed games."""

import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "nba_stats.db"
conn = sqlite3.connect(db_path)

# Find players who haven't played in all their team's games
query = """
WITH team_game_counts AS (
    SELECT team_id, COUNT(DISTINCT game_id) as total_games
    FROM team_game_logs
    WHERE season = '2025-26' AND season_type = 'Regular Season'
    GROUP BY team_id
),
player_game_counts AS (
    SELECT
        pgl.player_id,
        pgl.player_name,
        pgl.team_id,
        COUNT(DISTINCT pgl.game_id) as games_played,
        pst.points / CAST(pst.games_played AS REAL) as avg_ppg
    FROM player_game_logs pgl
    JOIN player_season_totals pst ON pgl.player_id = pst.player_id
        AND pgl.season = pst.season
        AND pgl.season_type = pst.season_type
    WHERE pgl.season = '2025-26' AND pgl.season_type = 'Regular Season'
        AND pst.games_played >= 10
        AND pst.points / CAST(pst.games_played AS REAL) >= 15.0
    GROUP BY pgl.player_id, pgl.player_name, pgl.team_id
)
SELECT
    pgc.player_name,
    pgc.games_played,
    tgc.total_games,
    (tgc.total_games - pgc.games_played) as games_missed,
    ROUND(pgc.avg_ppg, 1) as ppg
FROM player_game_counts pgc
JOIN team_game_counts tgc ON pgc.team_id = tgc.team_id
WHERE tgc.total_games > pgc.games_played
ORDER BY (tgc.total_games - pgc.games_played) DESC, pgc.avg_ppg DESC
LIMIT 10
"""

import pandas as pd
df = pd.read_sql_query(query, conn)

if df.empty:
    print("No players with absences found for 2025-26 season yet.")
    print("The season may have just started, or all top players have played every game.")
else:
    print("Players who have missed games (min 15 PPG, min 10 games played):")
    print(df.to_string(index=False))

conn.close()
