#!/usr/bin/env python3
"""
Analyze opponent defense variance to identify teams that allow huge scoring games.
This is critical for tournament strategy - teams with high variance are better
targets for ceiling plays.
"""

import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('nba_stats.db')

# Get all points scored against each opponent
# Join with teams table to get opponent names
query = """
SELECT
    tgl.opp_team_id,
    t.full_name as opp_team_name,
    tgl.pts as pts_scored_against_them
FROM team_game_logs tgl
JOIN teams t ON tgl.opp_team_id = t.team_id
WHERE tgl.season = '2024-25'
ORDER BY tgl.opp_team_id, tgl.game_date
"""

df = pd.read_sql_query(query, conn)

# Calculate variance metrics by opponent
defense_variance = df.groupby(['opp_team_id', 'opp_team_name'])['pts_scored_against_them'].agg([
    ('games', 'count'),
    ('avg_allowed', 'mean'),
    ('max_allowed', 'max'),
    ('min_allowed', 'min'),
    ('std_dev', 'std'),
    ('p90_allowed', lambda x: x.quantile(0.90)),  # 90th percentile
    ('p10_allowed', lambda x: x.quantile(0.10))   # 10th percentile
]).reset_index()

# Calculate range and ceiling factor
defense_variance['pts_range'] = defense_variance['max_allowed'] - defense_variance['min_allowed']
defense_variance['ceiling_factor'] = defense_variance['p90_allowed'] / defense_variance['avg_allowed']

# Filter to teams with enough games
defense_variance = defense_variance[defense_variance['games'] >= 15]

# Sort by ceiling factor (teams that allow big games)
defense_variance = defense_variance.sort_values('ceiling_factor', ascending=False)

print('\nOpponent Defense Variance Analysis:')
print('(Higher ceiling factor = more likely to allow explosive scoring games)')
print('='*120)
print(f'{"Team":<25} {"Games":>6} {"Avg":>6} {"Max":>6} {"90th%":>6} {"StdDev":>7} {"Range":>7} {"Ceiling":>8}  {"Notes":<25}')
print('='*120)

for _, row in defense_variance.head(20).iterrows():
    team = row['opp_team_name'] if pd.notna(row['opp_team_name']) else 'Unknown'
    notes = ''

    # Categorize teams for tournament strategy
    if row['ceiling_factor'] >= 1.15:
        notes = 'ELITE CEILING SPOT'
    elif row['ceiling_factor'] >= 1.12:
        notes = 'Good ceiling matchup'
    elif row['ceiling_factor'] >= 1.10:
        notes = 'Above average variance'
    else:
        notes = 'Average defense'

    print(f'{team:<25} {int(row["games"]):6} {row["avg_allowed"]:6.1f} {row["max_allowed"]:6.1f} '
          f'{row["p90_allowed"]:6.1f} {row["std_dev"]:7.1f} {row["pts_range"]:7.1f} '
          f'{row["ceiling_factor"]:8.2f}  {notes:<25}')

print('='*120)
print('\nKey Insights:')
print(f'  - Ceiling Factor: (90th percentile points allowed) / (avg points allowed)')
print(f'  - 1.15+ = Elite ceiling spot (allow 15%+ more points in big games)')
print(f'  - 1.12+ = Good ceiling matchup')
print(f'  - 1.10- = Average/tight defense (less upside)')

# Save for integration into ceiling calculation
print('\n\nTeams with highest ceiling factors (best tournament matchups):')
print('='*80)
for _, row in defense_variance.head(10).iterrows():
    print(f'  {row["opp_team_name"]:<30} Factor: {row["ceiling_factor"]:.3f}  (90th%: {row["p90_allowed"]:.1f} vs Avg: {row["avg_allowed"]:.1f})')

conn.close()
