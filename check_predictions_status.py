"""
Check the status of predictions vs actual game data.

This script helps identify:
1. Which dates have predictions
2. Which dates have actual game results
3. Which predictions are ready to score
4. Which dates need game data to be fetched
"""

import sqlite3
from datetime import datetime, timedelta
import pandas as pd

DB_PATH = "nba_stats.db"


def check_prediction_status():
    """Check prediction scoring readiness."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("="*70)
    print("PREDICTION SCORING STATUS CHECKER")
    print("="*70)
    print()

    # Get all prediction dates
    cursor.execute('''
        SELECT game_date, COUNT(*) as pred_count
        FROM predictions
        GROUP BY game_date
        ORDER BY game_date DESC
    ''')

    pred_dates = cursor.fetchall()

    if not pred_dates:
        print("No predictions found in database.")
        conn.close()
        return

    print(f"Found predictions for {len(pred_dates)} date(s):")
    print()

    scoreable_dates = []
    needs_data_dates = []

    for game_date, pred_count in pred_dates:
        # Check if game logs exist for this date (handle timestamp format)
        cursor.execute('''
            SELECT COUNT(DISTINCT game_id) as games, COUNT(*) as player_logs
            FROM player_game_logs
            WHERE game_date LIKE ?
        ''', (f'{game_date}%',))

        result = cursor.fetchone()
        games, player_logs = result if result else (0, 0)

        # Check how many predictions already have actual scores
        cursor.execute('''
            SELECT
                SUM(CASE WHEN did_play = 1 THEN 1 ELSE 0 END) as scored,
                SUM(CASE WHEN did_play = 0 THEN 1 ELSE 0 END) as dnp,
                SUM(CASE WHEN did_play IS NULL THEN 1 ELSE 0 END) as unscored
            FROM predictions
            WHERE game_date = ?
        ''', (game_date,))

        scored, dnp, unscored = cursor.fetchone()
        scored = scored or 0
        dnp = dnp or 0
        unscored = unscored or 0

        print(f"Date: {game_date}")
        print(f"  Predictions: {pred_count}")
        print(f"  Game Logs: {games} games, {player_logs} player logs")
        print(f"  Scoring Status: {scored} scored, {dnp} DNP, {unscored} not yet scored")

        if games > 0 and unscored > 0:
            scoreable_dates.append(game_date)
            print(f"  STATUS: [READY TO SCORE]")
        elif games > 0 and unscored == 0:
            print(f"  STATUS: [ALREADY SCORED]")
        else:
            needs_data_dates.append(game_date)
            print(f"  STATUS: [NEEDS GAME DATA] - No game logs found")

        print()

    # Summary and next steps
    print("="*70)
    print("SUMMARY & NEXT STEPS")
    print("="*70)
    print()

    if scoreable_dates:
        print(f"[READY TO SCORE] {len(scoreable_dates)} date(s):")
        for date in scoreable_dates:
            print(f"  - {date}")
        print()
        print("Action: Run `python score_predictions.py` or use date-specific scoring:")
        for date in scoreable_dates:
            print(f"  python -c \"from score_predictions import score_predictions_for_date; score_predictions_for_date('{date}')\"")
        print()

    if needs_data_dates:
        print(f"[NEEDS DATA] {len(needs_data_dates)} date(s) missing game logs:")
        for date in needs_data_dates:
            print(f"  - {date}")
        print()
        print("Action: Fetch game data for these dates using one of:")
        print("  1. Run nba_to_sqlite.py (if it fetches game logs)")
        print("  2. Use the NBA API to fetch LeagueGameLog data")
        print("  3. Check if game was postponed/cancelled")
        print()

    conn.close()


if __name__ == "__main__":
    check_prediction_status()
