#!/usr/bin/env python3
"""
Analyze Defensive Points Per Minute (PPM) Allowed by Each Team

This script calculates how many points per minute each team's defense allows,
providing a pace-normalized defensive metric that's more accurate than raw
points allowed or defensive rating.

Key Metrics:
- Avg Def PPM Allowed: Mean opponent PPM across all games
- Def PPM Std Dev: Consistency of defensive performance
- P90 Def PPM: 90th percentile (worst defensive games)
- Ceiling Factor: P90 / Avg (high = allows occasional blowups)
- Def PPM Grade: A+ to D- based on PPM allowed
"""

import sqlite3
import pandas as pd
import numpy as np

def calculate_game_ppm(conn, game_id, team_id):
    """Calculate opponent's PPM for a specific game."""
    # Get opponent's total points and total minutes played
    query = """
        SELECT
            SUM(points) as total_points,
            SUM(CAST(minutes AS REAL)) as total_minutes
        FROM player_game_logs
        WHERE game_id = ?
          AND team_id = ?
    """

    cursor = conn.cursor()
    cursor.execute(query, (game_id, team_id))
    result = cursor.fetchone()

    if result and result[0] and result[1] and result[1] > 0:
        return result[0] / result[1]  # PPM
    return None


def get_defensive_ppm_stats(conn, season='2025-26'):
    """Calculate defensive PPM stats for all teams."""

    # Get all games for each team (as defender)
    query = """
        SELECT
            tgl.opp_team_id as def_team_id,
            t.full_name as def_team_name,
            tgl.game_id,
            tgl.team_id as off_team_id,
            tgl.pts as pts_allowed
        FROM team_game_logs tgl
        JOIN teams t ON tgl.opp_team_id = t.team_id
        WHERE tgl.season = ?
        ORDER BY tgl.opp_team_id, tgl.game_date
    """

    df = pd.read_sql_query(query, conn, params=[season])

    # Calculate PPM for each game
    print("Calculating opponent PPM for each game...")
    ppm_data = []

    for idx, row in df.iterrows():
        game_ppm = calculate_game_ppm(conn, row['game_id'], row['off_team_id'])
        if game_ppm is not None:
            ppm_data.append({
                'def_team_id': row['def_team_id'],
                'def_team_name': row['def_team_name'],
                'game_id': row['game_id'],
                'opponent_ppm': game_ppm,
                'pts_allowed': row['pts_allowed']
            })

    ppm_df = pd.DataFrame(ppm_data)

    # Aggregate by team
    team_stats = ppm_df.groupby(['def_team_id', 'def_team_name']).agg(
        games=('game_id', 'count'),
        avg_def_ppm=('opponent_ppm', 'mean'),
        std_def_ppm=('opponent_ppm', 'std'),
        min_def_ppm=('opponent_ppm', 'min'),
        max_def_ppm=('opponent_ppm', 'max'),
        p90_def_ppm=('opponent_ppm', lambda x: x.quantile(0.90)),
        p10_def_ppm=('opponent_ppm', lambda x: x.quantile(0.10)),
        avg_pts_allowed=('pts_allowed', 'mean')
    ).reset_index()

    # Calculate ceiling factor
    team_stats['ceiling_factor'] = team_stats['p90_def_ppm'] / team_stats['avg_def_ppm']

    # Calculate coefficient of variation
    team_stats['cv_def_ppm'] = team_stats['std_def_ppm'] / team_stats['avg_def_ppm']

    # Assign grades based on actual league distribution
    # League average is ~0.483, range is ~0.438-0.529
    def get_def_ppm_grade(ppm):
        if pd.isna(ppm):
            return 'N/A'
        elif ppm < 0.455:  # Elite (top ~10%)
            return 'A+'
        elif ppm < 0.470:  # Excellent (top 25%)
            return 'A'
        elif ppm < 0.485:  # Good (above average)
            return 'B'
        elif ppm < 0.500:  # Average
            return 'C'
        elif ppm < 0.515:  # Poor
            return 'D'
        else:  # Very Poor (bottom 10%)
            return 'D-'

    team_stats['def_ppm_grade'] = team_stats['avg_def_ppm'].apply(get_def_ppm_grade)

    # Sort by avg_def_ppm (best defense first)
    team_stats = team_stats.sort_values('avg_def_ppm')

    return team_stats


def main():
    conn = sqlite3.connect('nba_stats.db')

    print("=" * 100)
    print("DEFENSIVE PPM ANALYSIS - 2025-26 Season")
    print("=" * 100)
    print("\nCalculating defensive PPM stats for all teams...\n")

    stats = get_defensive_ppm_stats(conn)

    # Calculate league average
    league_avg_ppm = stats['avg_def_ppm'].mean()

    print(f"League Average Defensive PPM: {league_avg_ppm:.3f}\n")
    print("=" * 100)
    print(f"{'Rank':<5} {'Team':<30} {'Games':<7} {'Avg PPM':<9} {'Std':<7} {'P90':<7} {'Ceiling':<9} {'Grade':<7} {'Pts/G':<7}")
    print("=" * 100)

    for idx, row in stats.iterrows():
        rank = idx + 1

        # Status based on grade (removed emojis for Windows compatibility)
        if row['def_ppm_grade'] in ['A+', 'A']:
            status = 'Elite'
        elif row['def_ppm_grade'] == 'B':
            status = 'Good'
        elif row['def_ppm_grade'] == 'C':
            status = 'Average'
        else:
            status = 'Target'

        print(f"{rank:<5} {row['def_team_name']:<30} {int(row['games']):<7} "
              f"{row['avg_def_ppm']:<9.3f} {row['std_def_ppm']:<7.3f} "
              f"{row['p90_def_ppm']:<7.3f} {row['ceiling_factor']:<9.3f} "
              f"{row['def_ppm_grade']:<7} {row['avg_pts_allowed']:<7.1f}")

    print("=" * 100)

    # Show top 5 elite defenses
    print("\nELITE DEFENSES (A+/A Grade):")
    print("-" * 80)
    elite = stats[stats['def_ppm_grade'].isin(['A+', 'A'])]
    for _, row in elite.head(5).iterrows():
        print(f"  {row['def_team_name']:<30} {row['avg_def_ppm']:.3f} PPM allowed  ({row['def_ppm_grade']} grade)")

    # Show bottom 5 target defenses
    print("\nTARGET DEFENSES FOR DFS (D/D- Grade):")
    print("-" * 80)
    targets = stats[stats['def_ppm_grade'].isin(['D', 'D-'])]
    for _, row in targets.tail(5).iterrows():
        print(f"  {row['def_team_name']:<30} {row['avg_def_ppm']:.3f} PPM allowed  ({row['def_ppm_grade']} grade)")

    # Show highest ceiling factors (best for GPP)
    print("\nHIGHEST CEILING FACTORS (Best Tournament Matchups):")
    print("-" * 80)
    high_ceiling = stats.nlargest(5, 'ceiling_factor')
    for _, row in high_ceiling.iterrows():
        print(f"  {row['def_team_name']:<30} Factor: {row['ceiling_factor']:.3f}  "
              f"(P90: {row['p90_def_ppm']:.3f} vs Avg: {row['avg_def_ppm']:.3f})")

    # Show most consistent defenses (lowest CV)
    print("\nMOST CONSISTENT DEFENSES (Lowest Variance):")
    print("-" * 80)
    consistent = stats.nsmallest(5, 'cv_def_ppm')
    for _, row in consistent.iterrows():
        print(f"  {row['def_team_name']:<30} CV: {row['cv_def_ppm']:.3f}  "
              f"(Std: {row['std_def_ppm']:.3f}, Avg: {row['avg_def_ppm']:.3f})")

    # Export to CSV for further analysis
    output_file = 'defensive_ppm_stats.csv'
    stats.to_csv(output_file, index=False)
    print(f"\nFull stats exported to: {output_file}")

    conn.close()
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
