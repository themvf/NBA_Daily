#!/usr/bin/env python3
"""Initialize the injury_list table in the database."""

import sqlite3
import injury_adjustment as ia

def init_injury_table(db_path='nba_stats.db'):
    """Create the injury_list table if it doesn't exist."""
    conn = sqlite3.connect(db_path)

    print(f"Initializing injury_list table in {db_path}...")

    # Create the table
    ia.create_injury_list_table(conn)

    # Verify it was created
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='injury_list'")
    if cursor.fetchone():
        print("SUCCESS: injury_list table created successfully!")

        # Show table schema
        cursor.execute("PRAGMA table_info(injury_list)")
        columns = cursor.fetchall()
        print("\nTable schema:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    else:
        print("ERROR: Failed to create injury_list table")

    conn.close()

if __name__ == "__main__":
    init_injury_table()
