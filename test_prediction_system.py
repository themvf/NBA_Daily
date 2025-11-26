#!/usr/bin/env python3
"""Test the complete prediction tracking system."""

import sqlite3
from pathlib import Path
from datetime import date, timedelta
import prediction_tracking as pt


def main():
    db_path = Path(__file__).parent / "nba_stats.db"
    conn = sqlite3.connect(db_path)

    print("=" * 60)
    print("Testing Prediction Tracking System")
    print("=" * 60)

    # Test 1: Create table (should be idempotent)
    print("\n1. Creating/verifying predictions table...")
    try:
        pt.create_predictions_table(conn)
        print("   [OK] Table ready")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return

    # Test 2: Create a test prediction
    print("\n2. Creating test prediction...")
    try:
        test_pred = pt.Prediction(
            prediction_id=None,
            prediction_date=str(date.today()),
            game_date=str(date.today() - timedelta(days=1)),  # Yesterday
            player_id=201935,  # James Harden
            player_name="James Harden",
            team_id=1610612756,
            team_name="Phoenix Suns",
            opponent_id=1610612737,
            opponent_name="Atlanta Hawks",
            projected_ppg=28.5,
            proj_confidence=0.75,
            proj_floor=22.0,
            proj_ceiling=35.0,
            season_avg_ppg=27.3,
            recent_avg_3=29.0,
            recent_avg_5=28.0,
            vs_opponent_avg=35.0,
            vs_opponent_games=1,
            analytics_used="🎯⚡",
            opponent_def_rating=112.5,
            opponent_pace=99.8,
            dfs_score=85.0,
            dfs_grade="A"
        )

        pred_id = pt.log_prediction(conn, test_pred)
        print(f"   [OK] Logged prediction with ID: {pred_id}")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Test 3: Query predictions
    print("\n3. Querying predictions...")
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM predictions
        """)
        count = cursor.fetchone()[0]
        print(f"   [OK] Total predictions in database: {count}")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Test 4: Update actuals (if game logs exist for yesterday)
    yesterday = str(date.today() - timedelta(days=1))
    print(f"\n4. Checking for game logs from {yesterday}...")
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM player_game_logs
            WHERE game_date = ?
        """, (yesterday,))
        game_log_count = cursor.fetchone()[0]

        if game_log_count > 0:
            print(f"   [OK] Found {game_log_count} game logs")
            print(f"   Updating predictions with actuals...")

            updated = pt.bulk_update_actuals_from_game_logs(conn, yesterday)
            print(f"   [OK] Updated {updated} predictions")

            # Show accuracy metrics
            metrics = pt.calculate_accuracy_metrics(conn, yesterday, yesterday)
            if metrics.predictions_with_actuals > 0:
                print(f"\n   Accuracy Metrics for {yesterday}:")
                print(f"     Predictions with actuals: {metrics.predictions_with_actuals}")
                print(f"     Mean Absolute Error: {metrics.mean_absolute_error:.2f} PPG")
                print(f"     RMSE: {metrics.rmse:.2f} PPG")
                print(f"     Hit Rate (in range): {metrics.hit_rate_floor_ceiling:.1%}")
                print(f"     Over-projections: {metrics.over_projections}")
                print(f"     Under-projections: {metrics.under_projections}")
        else:
            print(f"   [INFO] No game logs found for {yesterday}")
            print(f"   This is normal if games haven't been fetched yet")

    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Get best/worst predictions
    print("\n5. Getting best/worst predictions...")
    try:
        results = pt.get_best_worst_predictions(conn, n=3)

        if not results['best'].empty:
            print("\n   Best predictions:")
            for _, row in results['best'].iterrows():
                print(f"     {row['player_name']}: Proj {row['projected_ppg']:.1f}, "
                      f"Actual {row['actual_ppg']:.1f} (MAE: {row['abs_error']:.1f})")

        if not results['worst'].empty:
            print("\n   Worst predictions:")
            for _, row in results['worst'].iterrows():
                print(f"     {row['player_name']}: Proj {row['projected_ppg']:.1f}, "
                      f"Actual {row['actual_ppg']:.1f} (MAE: {row['abs_error']:.1f})")

        if results['best'].empty and results['worst'].empty:
            print("   [INFO] No predictions with actuals yet")

    except Exception as e:
        print(f"   [ERROR] {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
