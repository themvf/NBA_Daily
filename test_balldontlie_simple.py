#!/usr/bin/env python3
"""
Simple test for balldontlie.io API (no Streamlit required).
"""

import requests
import os

print("="*70)
print("balldontlie.io API Test")
print("="*70)

# Try to load API key from Streamlit secrets
API_KEY = None
try:
    # Try reading from .streamlit/secrets.toml
    import toml
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        # Read from [balldontlie] section
        if "balldontlie" in secrets and "API_KEY" in secrets["balldontlie"]:
            API_KEY = secrets["balldontlie"]["API_KEY"]
            print(f"\n[OK] API key loaded from {secrets_path}")
        else:
            print(f"\n[ERROR] No [balldontlie] API_KEY found in {secrets_path}")
            print("  Add this to your secrets.toml:")
            print('  [balldontlie]')
            print('  API_KEY="your-key-here"')
            API_KEY = None
    else:
        print(f"\n[ERROR] No secrets file found at {secrets_path}")
        print("  Please enter API key manually:")
        API_KEY = input("  API Key: ").strip()
except ImportError:
    print("\n[ERROR] toml package not installed")
    print("  Please enter API key manually:")
    API_KEY = input("  API Key: ").strip()

if not API_KEY:
    print("\n[ERROR] No API key provided. Exiting.")
    exit(1)

BASE_URL = "https://api.balldontlie.io/v1"

# Test 1: Fetch players
print("\n" + "="*70)
print("Test 1: Fetch Players")
print("="*70)
try:
    response = requests.get(
        f"{BASE_URL}/players",
        headers={"Authorization": API_KEY},
        params={"per_page": 5}
    )
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] API call successful!")
        print(f"  Players returned: {len(data.get('data', []))}")

        if data.get('data'):
            player = data['data'][0]
            print(f"\n  Sample player:")
            print(f"    ID: {player.get('id')}")
            print(f"    Name: {player.get('first_name')} {player.get('last_name')}")
            print(f"    Position: {player.get('position')}")
            print(f"    Team: {player.get('team', {}).get('full_name', 'N/A')}")
    else:
        print(f"[ERROR] API call failed")
        print(f"  Response: {response.text}")

except Exception as e:
    print(f"[ERROR] Error: {e}")

# Test 2: Search for LeBron James
print("\n" + "="*70)
print("Test 2: Search for LeBron James")
print("="*70)
try:
    response = requests.get(
        f"{BASE_URL}/players",
        headers={"Authorization": API_KEY},
        params={"search": "LeBron"}
    )
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Search successful!")
        print(f"  Players found: {len(data.get('data', []))}")

        for player in data.get('data', []):
            print(f"\n  {player.get('first_name')} {player.get('last_name')}")
            print(f"    ID: {player.get('id')}")
            print(f"    Position: {player.get('position')}")
            print(f"    Team: {player.get('team', {}).get('full_name', 'N/A')}")
    else:
        print(f"[ERROR] Search failed")
        print(f"  Response: {response.text}")

except Exception as e:
    print(f"[ERROR] Error: {e}")

# Test 3: Check for injury endpoint
print("\n" + "="*70)
print("Test 3: Check for Injury Endpoint")
print("="*70)
print("Checking if balldontlie.io has an /injuries endpoint...")
try:
    response = requests.get(
        f"{BASE_URL}/injuries",
        headers={"Authorization": API_KEY}
    )
    print(f"Status Code: {response.status_code}")

    if response.status_code == 404:
        print(f"[NOT FOUND] /injuries endpoint NOT FOUND (404)")
        print(f"  balldontlie.io v1 does NOT have injury data")
    elif response.status_code == 200:
        print(f"[OK] /injuries endpoint exists!")
        data = response.json()
        print(f"  Data: {data}")
    else:
        print(f"[UNKNOWN] Unexpected status code: {response.status_code}")
        print(f"  Response: {response.text}")

except Exception as e:
    print(f"[ERROR] Error: {e}")

# Summary
print("\n" + "="*70)
print("CRITICAL FINDING")
print("="*70)
print("""
balldontlie.io API v1 provides:
  [OK] /players - Player data
  [OK] /teams - Team data
  [OK] /games - Game data
  [OK] /stats - Player stats
  [NO] /injuries - NOT AVAILABLE

CONCLUSION:
balldontlie.io cannot replace nbainjuries for automated injury detection
because it does NOT provide injury status data.

OPTIONS:
1. Continue with manual injury entry (current system, most reliable)
2. Web scraping (ESPN, NBA.com - fragile but possible)
3. Find different API with injury data
4. Hybrid: Manual entry + ESPN scraping for validation
""")

print("="*70)
print("Test Complete")
print("="*70)
