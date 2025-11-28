"""
Score and evaluate predictions against actual game results.

This script:
1. Fetches actual game results from player_game_logs
2. Updates predictions table with actual values
3. Calculates error metrics
4. Generates accuracy report
"""

import sqlite3
from datetime import datetime, timedelta
import pandas as pd

DB_PATH = "nba_stats.db"


def score_predictions_for_date(game_date: str):
    """
    Score all predictions for a given game date.

    Args:
        game_date: Date string in format 'YYYY-MM-DD'
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"{'='*70}")
    print(f"SCORING PREDICTIONS FOR {game_date}")
    print(f"{'='*70}\n")

    # Get all predictions for this date
    cursor.execute('''
        SELECT
            prediction_id,
            player_id,
            player_name,
            team_name,
            opponent_name,
            projected_ppg,
            proj_floor,
            proj_ceiling,
            proj_confidence
        FROM predictions
        WHERE game_date = ?
    ''', (game_date,))

    predictions = cursor.fetchall()
    total_predictions = len(predictions)

    if total_predictions == 0:
        print(f"No predictions found for {game_date}")
        conn.close()
        return

    print(f"Found {total_predictions} predictions to score\n")

    scored_count = 0
    dnp_count = 0  # Did not play

    for pred in predictions:
        pred_id, player_id, player_name, team, opponent, proj_ppg, proj_floor, proj_ceiling, confidence = pred

        # Fetch actual game log
        # NOTE: game_date stored as timestamp "2025-11-25T00:00:00" so use LIKE for matching
        cursor.execute('''
            SELECT
                points,
                fg3m,
                minutes,
                game_id
            FROM player_game_logs
            WHERE player_id = ?
              AND game_date LIKE ?
        ''', (player_id, f'{game_date}%'))

        actual = cursor.fetchone()

        if actual:
            actual_ppg, actual_fg3m, actual_min_str, game_id = actual

            # Parse minutes (format like "34:25" or NULL)
            actual_min = None
            if actual_min_str:
                try:
                    parts = actual_min_str.split(':')
                    actual_min = float(parts[0]) + float(parts[1])/60 if len(parts) == 2 else float(parts[0])
                except:
                    actual_min = 0

            did_play = actual_min is not None and actual_min > 0

            if did_play:
                # Calculate errors
                error = actual_ppg - proj_ppg if actual_ppg is not None and proj_ppg is not None else None
                abs_error = abs(error) if error is not None else None

                # Check if actual fell within floor-ceiling range
                hit_floor_ceiling = False
                if actual_ppg is not None and proj_floor is not None and proj_ceiling is not None:
                    hit_floor_ceiling = proj_floor <= actual_ppg <= proj_ceiling

                # Update prediction with actuals
                cursor.execute('''
                    UPDATE predictions
                    SET actual_ppg = ?,
                        actual_fg3m = ?,
                        actual_minutes = ?,
                        did_play = 1,
                        error = ?,
                        abs_error = ?,
                        hit_floor_ceiling = ?
                    WHERE prediction_id = ?
                ''', (actual_ppg, actual_fg3m, actual_min, error, abs_error, hit_floor_ceiling, pred_id))

                scored_count += 1
            else:
                # Player was inactive (DNP)
                cursor.execute('''
                    UPDATE predictions
                    SET actual_ppg = 0,
                        actual_minutes = 0,
                        did_play = 0
                    WHERE prediction_id = ?
                ''', (pred_id,))

                dnp_count += 1
        else:
            # No game log found - player might not have played
            cursor.execute('''
                UPDATE predictions
                SET did_play = 0,
                    actual_ppg = 0,
                    actual_minutes = 0
                WHERE prediction_id = ?
            ''', (pred_id,))

            dnp_count += 1

    conn.commit()

    print(f"[SCORED] {scored_count} predictions")
    print(f"[DNP] {dnp_count} players did not play (DNP/inactive)")
    print(f"[TOTAL] {scored_count + dnp_count}/{total_predictions}\n")

    # Generate accuracy report
    generate_accuracy_report(conn, game_date)

    conn.close()


def generate_accuracy_report(conn, game_date: str):
    """Generate detailed accuracy report for predictions."""

    print(f"{'-'*70}")
    print(f"ACCURACY REPORT FOR {game_date}")
    print(f"{'-'*70}\n")

    # Overall statistics
    query = '''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN did_play = 1 THEN 1 ELSE 0 END) as played,
            AVG(CASE WHEN did_play = 1 THEN abs_error ELSE NULL END) as mae,
            AVG(CASE WHEN did_play = 1 THEN error ELSE NULL END) as mean_error,
            SUM(CASE WHEN hit_floor_ceiling = 1 THEN 1 ELSE 0 END) as within_range,
            AVG(CASE WHEN did_play = 1 THEN proj_confidence ELSE NULL END) as avg_confidence
        FROM predictions
        WHERE game_date = ?
    '''

    df = pd.read_sql_query(query, conn, params=(game_date,))

    total = df['total'].iloc[0]
    played = df['played'].iloc[0] or 0
    mae = df['mae'].iloc[0]
    mean_error = df['mean_error'].iloc[0]
    within_range = df['within_range'].iloc[0] or 0
    avg_conf = df['avg_confidence'].iloc[0]

    print(f"Overall Statistics:")
    print(f"   Total Predictions: {total}")
    print(f"   Players Who Played: {played}/{total} ({played/total*100:.1f}%)")

    if played > 0:
        print(f"\nPrediction Accuracy (for players who played):")
        print(f"   Mean Absolute Error (MAE): {mae:.2f} PPG")
        print(f"   Mean Error (bias): {mean_error:+.2f} PPG")
        if mean_error > 0:
            print(f"      -> Predictions tend to UNDER-project by {abs(mean_error):.2f} PPG")
        elif mean_error < 0:
            print(f"      -> Predictions tend to OVER-project by {abs(mean_error):.2f} PPG")
        else:
            print(f"      -> Predictions are unbiased")

        print(f"\nFloor-Ceiling Accuracy:")
        print(f"   Predictions within range: {within_range}/{played} ({within_range/played*100:.1f}%)")
        print(f"   Average Confidence: {avg_conf*100:.1f}%")

    # Top 5 most accurate predictions
    print(f"\nTop 5 Most Accurate Predictions:")
    print(f"{'-'*70}")

    accurate_query = '''
        SELECT
            player_name,
            team_name,
            projected_ppg,
            actual_ppg,
            abs_error,
            proj_confidence
        FROM predictions
        WHERE game_date = ?
          AND did_play = 1
        ORDER BY abs_error ASC
        LIMIT 5
    '''

    accurate_df = pd.read_sql_query(accurate_query, conn, params=(game_date,))

    for idx, row in accurate_df.iterrows():
        print(f"{idx+1}. {row['player_name']:20s} ({row['team_name']:20s})")
        print(f"   Projected: {row['projected_ppg']:.1f} | Actual: {row['actual_ppg']:.1f} | Error: {row['abs_error']:.2f} | Conf: {row['proj_confidence']*100:.0f}%")

    # Top 5 biggest misses
    print(f"\nTop 5 Biggest Misses:")
    print(f"{'-'*70}")

    misses_query = '''
        SELECT
            player_name,
            team_name,
            projected_ppg,
            actual_ppg,
            error,
            abs_error
        FROM predictions
        WHERE game_date = ?
          AND did_play = 1
        ORDER BY abs_error DESC
        LIMIT 5
    '''

    misses_df = pd.read_sql_query(misses_query, conn, params=(game_date,))

    for idx, row in misses_df.iterrows():
        direction = "UNDER" if row['error'] > 0 else "OVER"
        print(f"{idx+1}. {row['player_name']:20s} ({row['team_name']:20s})")
        print(f"   Projected: {row['projected_ppg']:.1f} | Actual: {row['actual_ppg']:.1f} | {direction}-projected by {abs(row['error']):.1f}")

    print(f"\n{'-'*70}\n")


def score_yesterday():
    """Score predictions from yesterday's games."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    score_predictions_for_date(yesterday)


def score_date_range(start_date: str, end_date: str):
    """Score predictions for a date range."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        score_predictions_for_date(date_str)
        current += timedelta(days=1)


def score_all_unscored():
    """Score all predictions that don't have actual results yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n" + "="*70)
    print("SCORING ALL UNSCORED PREDICTIONS")
    print("="*70 + "\n")

    # Find all dates with unscored predictions
    cursor.execute('''
        SELECT DISTINCT game_date
        FROM predictions
        WHERE did_play IS NULL OR actual_ppg IS NULL
        ORDER BY game_date DESC
    ''')

    unscored_dates = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not unscored_dates:
        print("No unscored predictions found. All predictions are up to date!")
        return

    print(f"Found {len(unscored_dates)} date(s) with unscored predictions:")
    for date in unscored_dates:
        print(f"  - {date}")
    print()

    # Score each date
    for date in unscored_dates:
        score_predictions_for_date(date)
        print()  # Add spacing between dates

    print("="*70)
    print("ALL UNSCORED PREDICTIONS PROCESSED")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NBA PREDICTION SCORER")
    print("="*70 + "\n")

    # Score yesterday's predictions
    score_yesterday()

    print("\n[SUCCESS] Scoring complete! Check the predictions table in nba_stats.db")
    print("   Or view results in the Prediction Log tab of the Streamlit app.\n")
