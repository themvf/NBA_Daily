#!/usr/bin/env python3
"""Export predictions to CSV for backup and analysis."""

import sqlite3
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import sys


def export_all_predictions(db_path: Path, output_dir: Path = None) -> str:
    """Export all predictions to a CSV file."""
    if output_dir is None:
        output_dir = db_path.parent / "prediction_exports"

    output_dir.mkdir(exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"predictions_all_{timestamp}.csv"

    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            prediction_id,
            prediction_date,
            game_date,
            player_id,
            player_name,
            team_name,
            opponent_name,
            projected_ppg,
            proj_confidence,
            proj_floor,
            proj_ceiling,
            season_avg_ppg,
            recent_avg_3,
            recent_avg_5,
            vs_opponent_avg,
            vs_opponent_games,
            analytics_used,
            opponent_def_rating,
            opponent_pace,
            dfs_score,
            dfs_grade,
            actual_ppg,
            actual_fg3m,
            actual_minutes,
            did_play,
            error,
            abs_error,
            hit_floor_ceiling,
            created_at
        FROM predictions
        ORDER BY game_date DESC, dfs_score DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df.to_csv(output_file, index=False)
    return str(output_file)


def export_by_date_range(
    db_path: Path,
    start_date: str,
    end_date: str,
    output_dir: Path = None
) -> str:
    """Export predictions for a specific date range."""
    if output_dir is None:
        output_dir = db_path.parent / "prediction_exports"

    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"predictions_{start_date}_to_{end_date}.csv"

    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            prediction_id,
            prediction_date,
            game_date,
            player_name,
            team_name,
            opponent_name,
            projected_ppg,
            proj_confidence,
            proj_floor,
            proj_ceiling,
            season_avg_ppg,
            analytics_used,
            dfs_score,
            dfs_grade,
            actual_ppg,
            error,
            abs_error,
            hit_floor_ceiling,
            created_at
        FROM predictions
        WHERE game_date >= ? AND game_date <= ?
        ORDER BY game_date DESC, dfs_score DESC
    """

    df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    conn.close()

    df.to_csv(output_file, index=False)
    return str(output_file)


def export_summary_stats(db_path: Path, output_dir: Path = None) -> str:
    """Export accuracy summary statistics by date."""
    if output_dir is None:
        output_dir = db_path.parent / "prediction_exports"

    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"accuracy_summary_{timestamp}.csv"

    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            game_date,
            COUNT(*) as total_predictions,
            COUNT(actual_ppg) as with_actuals,
            AVG(CASE WHEN actual_ppg IS NOT NULL THEN abs_error END) as mae,
            AVG(CASE WHEN actual_ppg IS NOT NULL THEN error END) as mean_error,
            AVG(CASE WHEN hit_floor_ceiling = 1 THEN 1.0 ELSE 0.0 END) as hit_rate,
            SUM(CASE WHEN error < 0 THEN 1 ELSE 0 END) as over_projections,
            SUM(CASE WHEN error > 0 THEN 1 ELSE 0 END) as under_projections,
            -- Breakdown by analytics type
            AVG(CASE WHEN analytics_used LIKE '%🎯%' AND actual_ppg IS NOT NULL THEN abs_error END) as mae_correlation,
            AVG(CASE WHEN analytics_used = '📊' AND actual_ppg IS NOT NULL THEN abs_error END) as mae_generic,
            -- Breakdown by confidence
            AVG(CASE WHEN proj_confidence >= 0.7 AND actual_ppg IS NOT NULL THEN abs_error END) as mae_high_conf,
            AVG(CASE WHEN proj_confidence < 0.5 AND actual_ppg IS NOT NULL THEN abs_error END) as mae_low_conf
        FROM predictions
        GROUP BY game_date
        ORDER BY game_date DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df.to_csv(output_file, index=False)
    return str(output_file)


def main():
    db_path = Path(__file__).parent / "nba_stats.db"

    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=" * 60)
    print("Prediction Export Tool")
    print("=" * 60)

    # Check if there are predictions to export
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        print("[INFO] No predictions to export yet")
        return

    print(f"[INFO] Found {count} predictions to export")
    print()

    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "date-range" and len(sys.argv) >= 4:
            start_date = sys.argv[2]
            end_date = sys.argv[3]
            print(f"Exporting date range: {start_date} to {end_date}")
            output_file = export_by_date_range(db_path, start_date, end_date)
            print(f"[OK] Exported to: {output_file}")

        elif command == "summary":
            print("Exporting accuracy summary statistics")
            output_file = export_summary_stats(db_path)
            print(f"[OK] Exported to: {output_file}")

        else:
            print(f"[ERROR] Unknown command: {command}")
            print_usage()

    else:
        # Default: export all predictions
        print("Exporting all predictions...")
        output_file = export_all_predictions(db_path)
        print(f"[OK] Exported to: {output_file}")

        # Also export summary
        print("\nExporting accuracy summary...")
        summary_file = export_summary_stats(db_path)
        print(f"[OK] Exported to: {summary_file}")

    print("\n" + "=" * 60)


def print_usage():
    print("""
Usage:
  python export_predictions.py                    # Export all predictions + summary
  python export_predictions.py date-range START END  # Export specific date range
  python export_predictions.py summary             # Export summary stats only

Examples:
  python export_predictions.py
  python export_predictions.py date-range 2025-11-20 2025-11-26
  python export_predictions.py summary
""")


if __name__ == "__main__":
    main()
