#!/usr/bin/env python3
"""
Test script for nbainjuries API.

This script tests the nbainjuries package to:
1. Verify API connection works
2. Understand the data format returned
3. Test date/time parameters
4. Validate Java dependency

Run: python test_nbainjuries_api.py
"""

from datetime import datetime
from nbainjuries import injury

print("="*70)
print("NBA Injuries API Test")
print("="*70)

# Test 1: Fetch current injury data
print("\n1. Fetching current injury data...")
try:
    # Get injury data for current date
    injury_data = injury.get_reportdata(datetime.now())

    print(f"✅ API call successful!")
    print(f"   Data type: {type(injury_data)}")

    # Check if it's a list, dict, or DataFrame
    if hasattr(injury_data, 'shape'):
        print(f"   Shape: {injury_data.shape}")
        print(f"   Columns: {list(injury_data.columns)}")
        print(f"\n   First 3 rows:")
        print(injury_data.head(3))
    elif isinstance(injury_data, list):
        print(f"   List length: {len(injury_data)}")
        if len(injury_data) > 0:
            print(f"   First item: {injury_data[0]}")
    elif isinstance(injury_data, dict):
        print(f"   Dict keys: {list(injury_data.keys())}")

except Exception as e:
    print(f"❌ API call failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Test 2: Try different parameters
print("\n2. Testing different parameters...")
try:
    # Try with return_df=True
    injury_df = injury.get_reportdata(datetime.now(), return_df=True)
    print(f"✅ DataFrame mode works!")
    print(f"   Shape: {injury_df.shape}")
    print(f"   Columns: {list(injury_df.columns)}")

    # Show sample data
    if len(injury_df) > 0:
        print(f"\n   Sample injury records:")
        print("-" * 70)
        for idx, row in injury_df.head(5).iterrows():
            print(f"   Player: {row.get('Relinquished', 'N/A')}")
            print(f"   Team: {row.get('Team', 'N/A')}")
            print(f"   Status: {row.get('Status', 'N/A')}")
            print(f"   Notes: {row.get('Notes', 'N/A')}")
            print("-" * 70)
    else:
        print("   No injury data returned (this may be normal if no active injuries)")

except Exception as e:
    print(f"❌ DataFrame mode failed: {e}")

# Test 3: Check available attributes/methods
print("\n3. Checking available methods...")
try:
    injury_methods = [m for m in dir(injury) if not m.startswith('_')]
    print(f"Available methods in nbainjuries.injury module:")
    for method in injury_methods:
        print(f"   - {method}")
except Exception as e:
    print(f"❌ Could not inspect module: {e}")

# Test 4: Data format analysis
print("\n4. Analyzing data format...")
try:
    injury_df = injury.get_reportdata(datetime.now(), return_df=True)

    if len(injury_df) > 0:
        print(f"Column data types:")
        for col in injury_df.columns:
            print(f"   {col}: {injury_df[col].dtype}")

        print(f"\nSample values:")
        for col in injury_df.columns:
            sample_val = injury_df[col].iloc[0] if len(injury_df) > 0 else "N/A"
            print(f"   {col}: {sample_val}")
    else:
        print("   No data to analyze (may indicate no current injuries)")

except Exception as e:
    print(f"❌ Analysis failed: {e}")

print("\n" + "="*70)
print("Test Complete")
print("="*70)
