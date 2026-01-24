#!/usr/bin/env python3
"""
One-command daily update script for NBA predictions.

This script automates the entire daily workflow:
1. Fetches latest NBA game data
2. Scores yesterday's predictions
3. Uploads to S3 for Streamlit Cloud

Usage:
    python daily_update.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3


def run_command(command, description):
    """Run a shell command and report results."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def check_database_status():
    """Check if there are predictions that need scoring."""
    try:
        conn = sqlite3.connect('nba_stats.db')
        cursor = conn.cursor()

        # Get yesterday's date
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Check for predictions
        cursor.execute(
            'SELECT COUNT(*) FROM predictions WHERE game_date = ?',
            (yesterday,)
        )
        pred_count = cursor.fetchone()[0]

        # Check for scored predictions
        cursor.execute(
            'SELECT COUNT(*) FROM predictions WHERE game_date = ? AND actual_ppg IS NOT NULL',
            (yesterday,)
        )
        scored_count = cursor.fetchone()[0]

        conn.close()

        return pred_count, scored_count, yesterday

    except Exception as e:
        print(f"Error checking database: {e}")
        return 0, 0, None


def upload_to_s3():
    """Upload database to S3 using the s3_storage module."""
    try:
        # Import here to avoid dependency issues if AWS not configured
        import s3_storage

        storage = s3_storage.S3PredictionStorage()

        if not storage.is_connected():
            print("\nWARNING: S3 not configured. Skipping upload.")
            print("The database has been updated locally.")
            print("To enable S3 sync, configure AWS credentials in .streamlit/secrets.toml")
            return False

        db_path = Path("nba_stats.db")
        success, message = storage.upload_database(db_path)

        if success:
            print(f"\n[SUCCESS] {message}")
            print("\n[NEXT STEP] Restart your Streamlit Cloud app to see the updates!")
            return True
        else:
            print(f"\n[ERROR] S3 upload failed: {message}")
            return False

    except ImportError:
        print("\nWARNING: s3_storage module not found. Skipping S3 upload.")
        return False
    except Exception as e:
        print(f"\nERROR during S3 upload: {e}")
        return False


def main():
    """Run the complete daily update workflow."""
    print("\n" + "="*70)
    print("NBA DAILY PREDICTION UPDATE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check current status
    pred_count, scored_count, yesterday = check_database_status()

    if pred_count > 0:
        print(f"\nFound {pred_count} predictions for {yesterday}")
        print(f"Currently scored: {scored_count}/{pred_count}")

    # Step 0.5: Auto-fetch current injury data
    print(f"\n{'='*70}")
    print("STEP 0.5: Auto-Fetching Current Injury Reports")
    print(f"{'='*70}")

    try:
        import fetch_injury_data
        conn = sqlite3.connect('nba_stats.db')
        updated, new, skipped, errors = fetch_injury_data.fetch_current_injuries(conn)
        conn.close()

        print(f"\n✅ Injury data updated:")
        print(f"   - Updated: {updated}")
        print(f"   - New: {new}")
        print(f"   - Skipped: {skipped}")

        if errors:
            print(f"\n⚠️ Warnings during fetch:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more")
    except Exception as e:
        print(f"\n⚠️ Could not auto-fetch injuries: {e}")
        print("Continuing with manual injury data...")

    # Step 1: Fetch latest game data
    step1_success = run_command(
        'python nba_to_sqlite.py --season "2025-26" --season-type "Regular Season" --no-include-rosters',
        "STEP 1: Fetching Latest NBA Game Data"
    )

    if not step1_success:
        print("\n❌ Failed to fetch game data. Aborting.")
        sys.exit(1)

    # Step 1.5: Refresh enrichment tables (role tiers, position defense)
    print(f"\n{'='*70}")
    print("STEP 1.5: Refreshing Enrichment Tables")
    print(f"{'='*70}")

    try:
        conn = sqlite3.connect('nba_stats.db')

        # Refresh player roles (STAR/STARTER/ROTATION/BENCH)
        from depth_chart import refresh_all_player_roles
        role_count = refresh_all_player_roles(conn, '2025-26')
        print(f"  - Player roles refreshed: {role_count}")

        # Refresh position-specific defense data
        from position_ppm_stats import refresh_team_position_defense
        pos_def_count = refresh_team_position_defense(conn, '2025-26')
        print(f"  - Position defense records: {pos_def_count}")

        conn.close()
        print("\n[OK] Enrichment tables updated")

    except Exception as e:
        print(f"\n[WARNING] Could not refresh enrichments: {e}")
        print("Continuing without enrichment updates...")

    # Step 2: Score predictions
    step2_success = run_command(
        'python -c "from score_predictions import score_all_unscored; score_all_unscored()"',
        "STEP 2: Scoring All Unscored Predictions"
    )

    if not step2_success:
        print("\n⚠️ Prediction scoring had issues, but continuing...")

    # Step 3: Upload to S3
    print(f"\n{'='*70}")
    print("STEP 3: Uploading to S3")
    print(f"{'='*70}")

    s3_success = upload_to_s3()

    # Final summary
    print("\n" + "="*70)
    print("DAILY UPDATE COMPLETE")
    print("="*70)

    new_pred_count, new_scored_count, _ = check_database_status()

    print(f"\nFinal Status for {yesterday}:")
    print(f"  - Total predictions: {new_pred_count}")
    print(f"  - Scored predictions: {new_scored_count}")
    print(f"  - Scoring rate: {(new_scored_count/new_pred_count*100) if new_pred_count > 0 else 0:.1f}%")

    if s3_success:
        print(f"\n[SUCCESS] All steps completed successfully!")
        print(f"\n[ACTION REQUIRED]")
        print(f"   Restart your Streamlit Cloud app to see the updates")
    else:
        print(f"\n[SUCCESS] Local updates complete!")
        print(f"[WARNING] S3 upload skipped (not configured or failed)")
        print(f"\n   Your local database is up to date.")
        print(f"   To sync to cloud, configure AWS credentials or use the app's sync button.")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
