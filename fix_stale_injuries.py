#!/usr/bin/env python3
"""
Fix stale manual injury entries by converting them to automated source.

This allows the automated fetch system to update them with current data.
"""

import sqlite3
from datetime import datetime

db_path = "nba_stats.db"

print("=" * 70)
print("FIXING STALE MANUAL INJURY ENTRIES")
print("=" * 70)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find manual entries older than 24 hours
print("\n1. Finding stale manual entries (older than 24 hours)...")
cursor.execute("""
    SELECT player_id, player_name, team_name, updated_at
    FROM injury_list
    WHERE source = 'manual'
      AND datetime(updated_at, '+24 hours') < datetime('now')
""")

stale_entries = cursor.fetchall()

if not stale_entries:
    print("   No stale manual entries found. All good!")
    conn.close()
    exit(0)

print(f"\n   Found {len(stale_entries)} stale manual entries:")
for player_id, player_name, team_name, updated_at in stale_entries:
    safe_name = player_name.encode('ascii', 'replace').decode('ascii')
    print(f"   - {safe_name} ({team_name}): updated {updated_at}")

# Convert to automated source
print("\n2. Converting to 'automated' source to allow updates...")
cursor.execute("""
    UPDATE injury_list
    SET source = 'automated',
        updated_at = CURRENT_TIMESTAMP
    WHERE source = 'manual'
      AND datetime(updated_at, '+24 hours') < datetime('now')
""")

updated_count = cursor.rowcount
conn.commit()

print(f"   Converted {updated_count} entries to 'automated' source")

print("\n3. Triggering injury fetch to update with current data...")
try:
    import fetch_injury_data
    updated, new, skipped, errors = fetch_injury_data.fetch_current_injuries(conn)

    print(f"\n   Fetch complete:")
    print(f"   - Updated: {updated}")
    print(f"   - New: {new}")
    print(f"   - Skipped: {skipped}")

    if errors:
        print(f"\n   Errors:")
        for error in errors[:5]:
            print(f"   - {error}")

except Exception as e:
    print(f"   Error during fetch: {e}")
    print("   You can manually trigger a fetch from the Injury Admin tab")

conn.close()

print("\n" + "=" * 70)
print("DONE")
print("Stale manual entries converted. They will now be updated by automated fetches.")
print("=" * 70)
