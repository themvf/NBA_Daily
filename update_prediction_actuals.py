#!/usr/bin/env python3
"""Update predictions with actual performance data from completed games."""

import sqlite3
from pathlib import Path
from datetime import date, timedelta
import prediction_tracking as pt
import sys


def main():
    if len(sys.argv) > 1:
        # Use date from command line (YYYY-MM-DD)
        game_date = sys.argv[1]
    else:
        # Default to yesterday
        yesterday = date.today() - timedelta(days=1)
        game_date = str(yesterday)

    db_path = Path(__file__).parent / "nba_stats.db"
    conn = sqlite3.connect(db_path)

    print(f"Updating predictions for game date: {game_date}")
    print("-" * 60)

    # Update actuals from game logs
    updated_count = pt.bulk_update_actuals_from_game_logs(conn, game_date)

    print(f"[OK] Updated {updated_count} predictions with actual performance")

    # Show some statistics
    cursor = conn.cursor()

    # Get predictions for this date
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(actual_ppg) as with_actuals,
            AVG(abs_error) as avg_error,
            AVG(CASE WHEN hit_floor_ceiling = 1 THEN 1.0 ELSE 0.0 END) as hit_rate
        FROM predictions
        WHERE game_date = ?
    """, (game_date,))

    result = cursor.fetchone()
    if result:
        total, with_actuals, avg_error, hit_rate = result
        print(f"\nStatistics for {game_date}:")
        print(f"  Total predictions: {total}")
        print(f"  With actuals: {with_actuals}")
        if with_actuals and avg_error:
            print(f"  Average error (MAE): {avg_error:.2f} PPG")
            print(f"  Hit rate (in range): {hit_rate:.1%}")

    # Show top 5 best and worst predictions
    cursor.execute("""
        SELECT player_name, projected_ppg, actual_ppg, error, abs_error
        FROM predictions
        WHERE game_date = ? AND actual_ppg IS NOT NULL
        ORDER BY abs_error ASC
        LIMIT 5
    """, (game_date,))

    best = cursor.fetchall()
    if best:
        print(f"\nBest predictions (lowest error):")
        for player_name, proj, actual, error, abs_err in best:
            print(f"  {player_name}: {proj:.1f} proj, {actual:.1f} actual ({error:+.1f}, {abs_err:.1f} MAE)")

    cursor.execute("""
        SELECT player_name, projected_ppg, actual_ppg, error, abs_error
        FROM predictions
        WHERE game_date = ? AND actual_ppg IS NOT NULL
        ORDER BY abs_error DESC
        LIMIT 5
    """, (game_date,))

    worst = cursor.fetchall()
    if worst:
        print(f"\nWorst predictions (highest error):")
        for player_name, proj, actual, error, abs_err in worst:
            print(f"  {player_name}: {proj:.1f} proj, {actual:.1f} actual ({error:+.1f}, {abs_err:.1f} MAE)")

    conn.close()


if __name__ == "__main__":
    main()
