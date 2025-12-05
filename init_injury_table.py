#!/usr/bin/env python3
"""Initialize the injury_list table and upgrade predictions table in the database."""

import sqlite3
import injury_adjustment as ia
import prediction_tracking as pt

def init_injury_table(db_path='nba_stats.db'):
    """Create the injury_list table and add refresh audit columns if they don't exist."""
    conn = sqlite3.connect(db_path)

    print(f"Initializing injury tracking in {db_path}...")

    # Create the injury_list table
    ia.create_injury_list_table(conn)

    # Verify it was created
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='injury_list'")
    if cursor.fetchone():
        print("SUCCESS: injury_list table created successfully!")

        # Show table schema
        cursor.execute("PRAGMA table_info(injury_list)")
        columns = cursor.fetchall()
        print("\ninjury_list table schema:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    else:
        print("ERROR: Failed to create injury_list table")

    # Add refresh audit columns to predictions table
    print("\nUpgrading predictions table with refresh audit columns...")
    pt.upgrade_predictions_table_for_refresh(conn)
    print("SUCCESS: predictions table upgraded!")

    conn.close()

if __name__ == "__main__":
    init_injury_table()
