#!/usr/bin/env python3
"""Quick check to verify prediction logging is working."""

import sqlite3
from pathlib import Path
from datetime import date

def main():
    db_path = Path(__file__).parent / "nba_stats.db"

    if not db_path.exists():
        print("[ERROR] Database not found at:", db_path)
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 60)
    print("Prediction Logging Status Check")
    print("=" * 60)

    # Check if predictions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if not cursor.fetchone():
        print("[ERROR] Predictions table does not exist")
        print("Run: python init_predictions_table.py")
        conn.close()
        return

    print("[OK] Predictions table exists")

    # Check total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    print(f"[INFO] Total predictions in database: {total}")

    # Check predictions by date
    cursor.execute("""
        SELECT game_date, COUNT(*), MIN(created_at), MAX(created_at)
        FROM predictions
        GROUP BY game_date
        ORDER BY game_date DESC
        LIMIT 10
    """)

    dates = cursor.fetchall()
    if dates:
        print(f"\nPredictions by date (last 10 dates):")
        for game_date, count, first, last in dates:
            print(f"  {game_date}: {count} predictions")
            print(f"    First logged: {first}")
            print(f"    Last logged:  {last}")
    else:
        print("\n[INFO] No predictions logged yet")

    # Check today's predictions
    today = str(date.today())
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE game_date = ?", (today,))
    today_count = cursor.fetchone()[0]

    print(f"\n[INFO] Predictions for today ({today}): {today_count}")

    if today_count > 0:
        # Show sample of today's predictions
        cursor.execute("""
            SELECT player_name, team_name, opponent_name, projected_ppg,
                   analytics_used, dfs_score
            FROM predictions
            WHERE game_date = ?
            ORDER BY dfs_score DESC
            LIMIT 5
        """, (today,))

        print("\nTop 5 projections for today:")
        for player, team, opp, proj, analytics, dfs in cursor.fetchall():
            print(f"  {player} ({team} vs {opp})")
            print(f"    Projected: {proj:.1f} PPG | DFS Score: {dfs:.1f} | Analytics: {analytics}")

    # Check for predictions with actuals
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_ppg IS NOT NULL")
    with_actuals = cursor.fetchone()[0]
    print(f"\n[INFO] Predictions with actual results: {with_actuals}")

    if with_actuals > 0:
        cursor.execute("""
            SELECT AVG(abs_error), AVG(CASE WHEN hit_floor_ceiling THEN 1.0 ELSE 0.0 END)
            FROM predictions
            WHERE actual_ppg IS NOT NULL
        """)
        mae, hit_rate = cursor.fetchone()
        print(f"  Average error (MAE): {mae:.2f} PPG")
        print(f"  Hit rate: {hit_rate:.1%}")

    print("\n" + "=" * 60)

    if today_count == 0:
        print("STATUS: Waiting for predictions to be logged")
        print("ACTION: View the 'Today's Games (Scoreboard)' tab in Streamlit")
        print("        Predictions will log automatically as the page loads")
    else:
        print("STATUS: Predictions are being logged successfully!")
        print("ACTION: View the 'Prediction Log' tab to analyze accuracy")

    print("=" * 60)

    conn.close()

if __name__ == "__main__":
    main()
