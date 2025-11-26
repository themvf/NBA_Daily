#!/usr/bin/env python3
"""Initialize the predictions tracking table in the database."""

import sqlite3
from pathlib import Path
import prediction_tracking as pt

def main():
    db_path = Path(__file__).parent / "nba_stats.db"
    conn = sqlite3.connect(db_path)

    print("Creating predictions table...")
    pt.create_predictions_table(conn)
    print("[OK] Predictions table created successfully!")

    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM sqlite_master
        WHERE type='table' AND name='predictions'
    """)
    count = cursor.fetchone()[0]

    if count == 1:
        print(f"[OK] Verified: predictions table exists in {db_path}")

        # Show table info
        cursor.execute("PRAGMA table_info(predictions)")
        columns = cursor.fetchall()
        print(f"\nTable has {len(columns)} columns:")
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            print(f"  - {col_name} ({col_type})")

    conn.close()

if __name__ == "__main__":
    main()
