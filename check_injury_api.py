#!/usr/bin/env python3
"""
Check injury API status and database state.
"""

import sqlite3
import requests
from datetime import datetime
import json

db_path = "nba_stats.db"

print("=" * 70)
print("INJURY DATA DIAGNOSTIC")
print("=" * 70)

# Check 1: Database state
print("\n1. Checking injury_list table for known injured players...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT player_name, team_name, status, injury_type, source, updated_at, last_fetched_at
    FROM injury_list
    WHERE player_name IN ('Giannis Antetokounmpo', 'Jalen Suggs', 'Franz Wagner')
    ORDER BY updated_at DESC
""")

results = cursor.fetchall()
if results:
    for row in results:
        print(f"\n   {row[0]} ({row[1]})")
        print(f"      Status: {row[2]}")
        print(f"      Injury: {row[3]}")
        print(f"      Source: {row[4]}")
        print(f"      Updated: {row[5]}")
        print(f"      Last Fetched: {row[6]}")
else:
    print("   ❌ NO RECORDS FOUND for these players")

# Check 2: All injuries in database
print("\n2. All injuries currently in database:")
cursor.execute("""
    SELECT player_name, team_name, status, injury_type, updated_at
    FROM injury_list
    ORDER BY updated_at DESC
    LIMIT 20
""")

all_injuries = cursor.fetchall()
if all_injuries:
    print(f"   Found {len(all_injuries)} injuries (showing first 20):")
    for row in all_injuries:
        # Safely encode special characters
        player_name = row[0].encode('ascii', 'replace').decode('ascii')
        injury_desc = str(row[3])[:60].encode('ascii', 'replace').decode('ascii') if row[3] else 'None'
        print(f"   - {player_name} ({row[1]}): {row[2]} - {injury_desc} (updated: {row[4]})")
else:
    print("   ❌ NO INJURIES IN DATABASE AT ALL")

# Check 3: Last fetch time
print("\n3. Checking last API fetch time...")
cursor.execute("""
    SELECT locked, locked_at, locked_by
    FROM injury_fetch_lock
    WHERE lock_id = 1
""")

lock_info = cursor.fetchone()
if lock_info:
    locked, locked_at, locked_by = lock_info
    print(f"   Lock status: {'LOCKED' if locked else 'UNLOCKED'}")
    if locked_at:
        print(f"   Last fetch: {locked_at}")
        print(f"   Locked by: {locked_by}")
else:
    print("   ❌ No lock table found")

# Check 4: Test API directly
print("\n4. Testing balldontlie.io API directly...")

# Try to get API key
api_key = None
try:
    import toml
    import os
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        if "balldontlie" in secrets and "API_KEY" in secrets["balldontlie"]:
            api_key = secrets["balldontlie"]["API_KEY"]
except:
    pass

if not api_key:
    import os
    api_key = os.environ.get("BALLDONTLIE_API_KEY")

if api_key:
    print(f"   API key found: {api_key[:10]}...")

    try:
        headers = {"Authorization": api_key}
        response = requests.get(
            "https://api.balldontlie.io/v1/player_injuries",
            headers=headers,
            timeout=30
        )

        print(f"   API Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            injuries = data.get('data', [])
            print(f"   Total injuries from API: {len(injuries)}")

            # Check for specific players
            for player_name in ['Giannis Antetokounmpo', 'Jalen Suggs', 'Franz Wagner']:
                found = False
                for injury in injuries:
                    player = injury.get('player', {})
                    full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                    if full_name == player_name:
                        found = True
                        print(f"\n   ✓ {player_name} FOUND in API:")
                        print(f"      Status: {injury.get('status')}")
                        print(f"      Description: {injury.get('description')}")
                        print(f"      Return: {injury.get('return_date')}")
                        break

                if not found:
                    print(f"\n   ✗ {player_name} NOT in API response")

            # Show first 5 injuries from API
            print(f"\n   First 5 injuries from API:")
            for i, injury in enumerate(injuries[:5], 1):
                player = injury.get('player', {})
                full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                print(f"   {i}. {full_name}: {injury.get('status')} - {injury.get('description', 'N/A')[:50]}")

        else:
            print(f"   ❌ API Error: {response.text[:200]}")

    except Exception as e:
        print(f"   ❌ API call failed: {e}")
else:
    print("   ❌ NO API KEY FOUND")
    print("   Check .streamlit/secrets.toml or BALLDONTLIE_API_KEY environment variable")

conn.close()

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("-" * 70)
print("If players show as injured but aren't in the database:")
print("1. API may not be reporting them (check API output above)")
print("2. Fetch lock cooldown may be preventing updates")
print("3. Player name matching may be failing")
print("4. Manual overrides may be blocking updates")
print("\nIf API shows injuries but database doesn't:")
print("- Run fetch manually or check Injury Admin tab in app")
print("=" * 70)
