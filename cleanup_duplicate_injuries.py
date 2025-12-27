#!/usr/bin/env python3
"""
Clean up duplicate injury records for same player with different statuses.

Issue: Database allows multiple statuses per player (e.g., Giannis has "returned" AND "active")
This confuses the upsert logic and prevents API updates.

Solution: Keep only the most recent record per player.
"""

import sqlite3
from datetime import datetime

db_path = "nba_stats.db"

print("=" * 70)
print("CLEANING UP DUPLICATE INJURY RECORDS")
print("=" * 70)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find players with multiple status records
print("\n1. Finding players with multiple injury statuses...")
cursor.execute("""
    SELECT player_id, player_name, COUNT(*) as record_count
    FROM injury_list
    GROUP BY player_id
    HAVING COUNT(*) > 1
    ORDER BY record_count DESC
""")

duplicates = cursor.fetchall()

if not duplicates:
    print("   ✓ No duplicates found. Database is clean!")
    conn.close()
    exit(0)

print(f"\n   Found {len(duplicates)} players with multiple records:")
for player_id, player_name, count in duplicates:
    print(f"   - {player_name}: {count} records")

# Show details for each duplicate
print("\n2. Showing all records for duplicate players:")
for player_id, player_name, count in duplicates:
    print(f"\n   {player_name}:")
    cursor.execute("""
        SELECT injury_id, status, updated_at, source
        FROM injury_list
        WHERE player_id = ?
        ORDER BY updated_at DESC
    """, (player_id,))

    records = cursor.fetchall()
    for injury_id, status, updated_at, source in records:
        print(f"      ID {injury_id}: {status:15s} | {source:10s} | {updated_at}")

# Ask for confirmation
print("\n3. Cleanup strategy:")
print("   For each player, KEEP the most recent record and DELETE older ones.")
print()
response = input("   Proceed with cleanup? (yes/no): ").strip().lower()

if response != 'yes':
    print("\n   Cleanup cancelled.")
    conn.close()
    exit(0)

# Perform cleanup
print("\n4. Cleaning up...")
deleted_total = 0

for player_id, player_name, count in duplicates:
    # Get all records for this player, ordered by date (newest first)
    cursor.execute("""
        SELECT injury_id
        FROM injury_list
        WHERE player_id = ?
        ORDER BY updated_at DESC
    """, (player_id,))

    all_ids = [row[0] for row in cursor.fetchall()]

    # Keep first (most recent), delete rest
    keep_id = all_ids[0]
    delete_ids = all_ids[1:]

    if delete_ids:
        placeholders = ','.join('?' * len(delete_ids))
        cursor.execute(f"""
            DELETE FROM injury_list
            WHERE injury_id IN ({placeholders})
        """, delete_ids)

        deleted_count = cursor.rowcount
        deleted_total += deleted_count
        print(f"   {player_name}: Kept ID {keep_id}, deleted {deleted_count} old record(s)")

conn.commit()

print(f"\n   Total records deleted: {deleted_total}")

# Verify cleanup
print("\n5. Verifying cleanup...")
cursor.execute("""
    SELECT player_id, COUNT(*) as record_count
    FROM injury_list
    GROUP BY player_id
    HAVING COUNT(*) > 1
""")

remaining_dupes = cursor.fetchall()

if remaining_dupes:
    print(f"   ⚠️ WARNING: {len(remaining_dupes)} players still have duplicates!")
else:
    print("   ✓ All duplicates cleaned up successfully!")

conn.close()

print("\n" + "=" * 70)
print("DONE")
print("Now go to Injury Admin → Fetch Now to get current injury data.")
print("=" * 70)
