#!/usr/bin/env python3
"""Check what fields TeamGameLogs API actually returns."""

from nba_api.stats.endpoints import teamgamelogs

# Get a sample of team game logs
logs = teamgamelogs.TeamGameLogs(
    season_nullable="2025-26",
    season_type_nullable="Regular Season"
)

df = logs.get_data_frames()[0]

print("Available columns in TeamGameLogs:")
print(df.columns.tolist())
print()

print("Sample row (first game):")
if not df.empty:
    first_row = df.iloc[0]
    for col in df.columns:
        print(f"  {col}: {first_row[col]}")
