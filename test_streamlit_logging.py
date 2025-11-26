#!/usr/bin/env python3
"""Test if prediction logging code works in Streamlit context."""

import sqlite3
from pathlib import Path
from datetime import date
import prediction_tracking as pt

def test_logging():
    """Simulate what happens in Streamlit when viewing Today's Games."""

    db_path = Path(__file__).parent / "nba_stats.db"

    # This mimics what Streamlit does
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    print("Testing prediction logging...")
    print(f"Database: {db_path}")
    print(f"Database exists: {db_path.exists()}")

    # Ensure table exists
    try:
        pt.create_predictions_table(conn)
        print("[OK] Predictions table ready")
    except Exception as e:
        print(f"[ERROR] Table creation failed: {e}")
        return

    # Create a test prediction (simulating what Today's Games does)
    try:
        test_pred = pt.Prediction(
            prediction_id=None,
            prediction_date=str(date.today()),
            game_date=str(date.today()),
            player_id=2544,  # LeBron
            player_name="LeBron James",
            team_id=1610612747,
            team_name="Los Angeles Lakers",
            opponent_id=1610612751,
            opponent_name="Brooklyn Nets",
            projected_ppg=24.5,
            proj_confidence=0.80,
            proj_floor=18.0,
            proj_ceiling=31.0,
            season_avg_ppg=23.8,
            recent_avg_3=25.0,
            recent_avg_5=24.2,
            vs_opponent_avg=26.0,
            vs_opponent_games=2,
            analytics_used="🎯",
            opponent_def_rating=110.5,
            opponent_pace=98.5,
            dfs_score=88.0,
            dfs_grade="A"
        )

        pred_id = pt.log_prediction(conn, test_pred)
        print(f"[OK] Successfully logged prediction with ID: {pred_id}")

        # Verify it was logged
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE game_date = ?", (str(date.today()),))
        count = cursor.fetchone()[0]
        print(f"[OK] Total predictions for today: {count}")

    except Exception as e:
        print(f"[ERROR] Logging failed: {e}")
        import traceback
        traceback.print_exc()

    conn.close()

if __name__ == "__main__":
    test_logging()
