#!/usr/bin/env python3
"""
Test script for balldontlie.io API.

This script tests the balldontlie.io API to:
1. Verify API key works
2. Understand the injury data format
3. Test player name matching
4. Validate data completeness

Run: streamlit run test_balldontlie_api.py
"""

import streamlit as st
import requests
from datetime import datetime

st.title("🏀 balldontlie.io API Test")

# Get API key from Streamlit secrets
try:
    api_key = st.secrets["API_Key"]
    st.success("✅ API key loaded from Streamlit secrets")
except Exception as e:
    st.error(f"❌ Could not load API key: {e}")
    st.stop()

# API endpoint
BASE_URL = "https://api.balldontlie.io/v1"

st.divider()

# Test 1: Fetch all players
st.subheader("1️⃣ Test: Fetch Players")
if st.button("Fetch Players", key="fetch_players"):
    with st.spinner("Fetching players..."):
        try:
            response = requests.get(
                f"{BASE_URL}/players",
                headers={"Authorization": api_key},
                params={"per_page": 10}
            )
            response.raise_for_status()
            data = response.json()

            st.success(f"✅ API call successful! (Status {response.status_code})")
            st.write(f"**Players returned:** {len(data.get('data', []))}")

            if data.get('data'):
                st.write("**Sample player data:**")
                st.json(data['data'][0])

        except Exception as e:
            st.error(f"❌ API call failed: {e}")

st.divider()

# Test 2: Fetch teams
st.subheader("2️⃣ Test: Fetch Teams")
if st.button("Fetch Teams", key="fetch_teams"):
    with st.spinner("Fetching teams..."):
        try:
            response = requests.get(
                f"{BASE_URL}/teams",
                headers={"Authorization": api_key}
            )
            response.raise_for_status()
            data = response.json()

            st.success(f"✅ API call successful! (Status {response.status_code})")
            st.write(f"**Teams returned:** {len(data.get('data', []))}")

            if data.get('data'):
                st.write("**Team data structure:**")
                st.json(data['data'][0])

        except Exception as e:
            st.error(f"❌ API call failed: {e}")

st.divider()

# Test 3: Search for specific player
st.subheader("3️⃣ Test: Search Player")
player_search = st.text_input("Enter player name (e.g., 'LeBron James')", value="LeBron James")
if st.button("Search Player", key="search_player"):
    with st.spinner(f"Searching for {player_search}..."):
        try:
            # balldontlie.io uses 'search' parameter for player name
            response = requests.get(
                f"{BASE_URL}/players",
                headers={"Authorization": api_key},
                params={"search": player_search}
            )
            response.raise_for_status()
            data = response.json()

            st.success(f"✅ Search successful! (Status {response.status_code})")
            st.write(f"**Players found:** {len(data.get('data', []))}")

            if data.get('data'):
                for player in data['data']:
                    st.write(f"**ID:** {player.get('id')}")
                    st.write(f"**Name:** {player.get('first_name')} {player.get('last_name')}")
                    st.write(f"**Team:** {player.get('team', {}).get('full_name', 'N/A')}")
                    st.write(f"**Position:** {player.get('position', 'N/A')}")
                    st.divider()
            else:
                st.warning("No players found")

        except Exception as e:
            st.error(f"❌ Search failed: {e}")

st.divider()

# Test 4: Check API documentation
st.subheader("4️⃣ API Endpoints Available")
st.markdown("""
According to balldontlie.io documentation:

**Available Endpoints:**
- `/players` - Get all NBA players
- `/teams` - Get all NBA teams
- `/games` - Get game data
- `/stats` - Get player stats
- `/season_averages` - Get season averages

**Note:** balldontlie.io v1 does NOT have a dedicated `/injuries` endpoint.

**Alternative Approach:**
- Use `/players` with filters
- Cross-reference with our injury_list table
- Or use a different API for injury data
- Or continue with manual entry + consider ESPN scraping
""")

st.divider()

# Test 5: Rate limiting info
st.subheader("5️⃣ Rate Limiting")
st.info("""
**balldontlie.io rate limits:**
- Free tier: 30 requests per minute
- Paid tier: Higher limits
- Returns 429 status code when limit exceeded
""")

st.divider()

st.success("✅ Test script ready! Click buttons above to test API calls.")
