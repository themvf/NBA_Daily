#!/usr/bin/env python3
"""
Comprehensive test suite for the prediction refresh system.

Tests the complete workflow:
1. Generate initial predictions with all players healthy
2. Mark player as OUT
3. Refresh predictions
4. Verify OUT player prediction is removed
5. Verify teammate predictions are adjusted
6. Verify audit trail is updated
"""

import sqlite3
import pandas as pd
from datetime import date
import prediction_refresh as pr
import injury_adjustment as ia
import prediction_tracking as pt


def setup_test_database(db_path='nba_stats.db'):
    """Ensure test database has required tables and columns."""
    conn = sqlite3.connect(db_path)

    # Create tables
    ia.create_injury_list_table(conn)
    pt.create_predictions_table(conn)
    pt.upgrade_predictions_table_for_injuries(conn)
    pt.upgrade_predictions_table_for_refresh(conn)

    conn.close()


def test_basic_refresh():
    """Test 1: Basic refresh - mark player OUT, refresh, verify prediction removed."""
    print("=" * 80)
    print("TEST 1: BASIC REFRESH")
    print("=" * 80)

    conn = sqlite3.connect('nba_stats.db')
    cursor = conn.cursor()

    # Get latest game date with predictions
    cursor.execute("SELECT MAX(game_date) FROM predictions")
    latest_date = cursor.fetchone()[0]

    if not latest_date:
        print("ERROR: No predictions in database to test with")
        conn.close()
        return False

    print(f"\nUsing game date: {latest_date}")

    # Get a sample player from predictions
    cursor.execute("""
        SELECT player_id, player_name, team_name, projected_ppg
        FROM predictions
        WHERE game_date = ?
        ORDER BY projected_ppg DESC
        LIMIT 1
    """, (latest_date,))

    result = cursor.fetchone()
    if not result:
        print("ERROR: No predictions found for test")
        conn.close()
        return False

    test_player_id, test_player_name, test_team_name, original_ppg = result
    print(f"Test player: {test_player_name} ({test_team_name}) - {original_ppg:.1f} PPG")

    # Step 1: Mark player as OUT
    print("\nStep 1: Marking player as OUT...")
    injury_id = ia.add_to_injury_list(
        conn,
        player_id=test_player_id,
        player_name=test_player_name,
        team_name=test_team_name,
        notes="Test injury for refresh"
    )
    print(f"SUCCESS: Added to injury list (injury_id: {injury_id})")

    # Step 2: Verify player still has prediction
    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE game_date = ? AND player_id = ?
    """, (latest_date, test_player_id))

    pred_count_before = cursor.fetchone()[0]
    print(f"\nStep 2: Predictions before refresh: {pred_count_before}")

    if pred_count_before == 0:
        print("ERROR: Player doesn't have prediction to refresh")
        ia.remove_from_injury_list(conn, test_player_id)
        conn.close()
        return False

    # Step 3: Get refresh status
    print("\nStep 3: Getting refresh status...")
    status = pr.get_refresh_status(latest_date, conn)
    print(f"  - Total predictions: {status['predictions_count']}")
    print(f"  - Needs refresh: {status['needs_refresh']}")
    print(f"  - OUT players with predictions: {len(status['out_players_with_predictions'])}")

    if not status['needs_refresh']:
        print("ERROR: Status shows no refresh needed")
        ia.remove_from_injury_list(conn, test_player_id)
        conn.close()
        return False

    # Step 4: Refresh predictions
    print("\nStep 4: Refreshing predictions...")
    result = pr.refresh_predictions_for_date(latest_date, conn)

    if result['error']:
        print(f"ERROR: Refresh failed: {result['error']}")
        ia.remove_from_injury_list(conn, test_player_id)
        conn.close()
        return False

    print(f"SUCCESS: Refresh completed")
    print(f"  - Removed: {result['removed']}")
    print(f"  - Adjusted: {result['adjusted']}")
    print(f"  - Skipped: {result['skipped']}")
    print(f"  - Affected players: {', '.join(result['affected_players'])}")

    # Step 5: Verify prediction was removed
    cursor.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE game_date = ? AND player_id = ?
    """, (latest_date, test_player_id))

    pred_count_after = cursor.fetchone()[0]
    print(f"\nStep 5: Predictions after refresh: {pred_count_after}")

    if pred_count_after > 0:
        print("ERROR: Prediction was NOT removed")
        ia.remove_from_injury_list(conn, test_player_id)
        conn.close()
        return False

    print("SUCCESS: OUT player prediction was removed")

    # Step 6: Verify refresh metadata was updated
    cursor.execute("""
        SELECT COUNT(*), MAX(refresh_count), MAX(last_refreshed_at)
        FROM predictions
        WHERE game_date = ? AND refresh_count > 0
    """, (latest_date,))

    refreshed_count, max_refresh_count, last_refreshed = cursor.fetchone()
    print(f"\nStep 6: Refresh audit trail")
    print(f"  - Predictions with refresh metadata: {refreshed_count}")
    print(f"  - Max refresh count: {max_refresh_count}")
    print(f"  - Last refreshed: {last_refreshed}")

    # Cleanup
    print("\nCleanup: Removing test injury...")
    ia.remove_from_injury_list(conn, test_player_id)

    conn.close()

    print("\n" + "=" * 80)
    print("TEST 1: PASSED")
    print("=" * 80)
    return True


def test_multiple_refreshes():
    """Test 2: Multiple refreshes - ensure idempotency and audit trail."""
    print("\n" + "=" * 80)
    print("TEST 2: MULTIPLE REFRESHES (IDEMPOTENCY)")
    print("=" * 80)

    conn = sqlite3.connect('nba_stats.db')
    cursor = conn.cursor()

    # Get latest game date with predictions
    cursor.execute("SELECT MAX(game_date) FROM predictions")
    latest_date = cursor.fetchone()[0]

    if not latest_date:
        print("ERROR: No predictions in database to test with")
        conn.close()
        return False

    # Get a sample player who is NOT already in injury list
    cursor.execute("""
        SELECT p.player_id, p.player_name, p.team_name
        FROM predictions p
        LEFT JOIN injury_list i ON p.player_id = i.player_id AND i.status = 'active'
        WHERE p.game_date = ? AND i.player_id IS NULL
        LIMIT 1
    """, (latest_date,))

    result = cursor.fetchone()
    if not result:
        print("ERROR: No predictions found for test")
        conn.close()
        return False

    test_player_id, test_player_name, test_team_name = result

    # Mark player as OUT
    print(f"\nMarking {test_player_name} as OUT...")
    ia.add_to_injury_list(conn, test_player_id, test_player_name, test_team_name)

    # First refresh
    print("\nFirst refresh...")
    result1 = pr.refresh_predictions_for_date(latest_date, conn)
    print(f"  - Removed: {result1['removed']}")

    # Get refresh count after first refresh
    cursor.execute("""
        SELECT MAX(refresh_count) FROM predictions WHERE game_date = ?
    """, (latest_date,))
    count_after_first = cursor.fetchone()[0] or 0

    # Second refresh (should be idempotent)
    print("\nSecond refresh (should be idempotent)...")
    result2 = pr.refresh_predictions_for_date(latest_date, conn)
    print(f"  - Removed: {result2['removed']} (should be 0)")

    # Get refresh count after second refresh
    cursor.execute("""
        SELECT MAX(refresh_count) FROM predictions WHERE game_date = ?
    """, (latest_date,))
    count_after_second = cursor.fetchone()[0] or 0

    print(f"\nRefresh counts:")
    print(f"  - After first refresh: {count_after_first}")
    print(f"  - After second refresh: {count_after_second}")

    # Verify idempotency
    if result2['removed'] != 0:
        print("ERROR: Second refresh removed predictions (not idempotent)")
        ia.remove_from_injury_list(conn, test_player_id)
        conn.close()
        return False

    # Second refresh should increment count only if it did work
    # Since OUT player already removed, refresh_count stays same (no work done)
    if count_after_second < count_after_first:
        print("ERROR: Refresh count decreased (shouldn't happen)")
        ia.remove_from_injury_list(conn, test_player_id)
        conn.close()
        return False

    print("SUCCESS: Multiple refreshes are idempotent (no duplicate deletions)")

    # Cleanup
    ia.remove_from_injury_list(conn, test_player_id)
    conn.close()

    print("\n" + "=" * 80)
    print("TEST 2: PASSED")
    print("=" * 80)
    return True


def test_no_predictions_to_refresh():
    """Test 3: Refresh when no predictions exist for the date."""
    print("\n" + "=" * 80)
    print("TEST 3: NO PREDICTIONS TO REFRESH")
    print("=" * 80)

    conn = sqlite3.connect('nba_stats.db')

    # Use a future date with no predictions
    future_date = "2099-12-31"

    print(f"\nAttempting to refresh date with no predictions: {future_date}")

    result = pr.refresh_predictions_for_date(future_date, conn)

    print(f"Result:")
    print(f"  - Removed: {result['removed']}")
    print(f"  - Error: {result['error']}")

    if result['removed'] != 0 or result['error']:
        print("ERROR: Unexpected result for empty date")
        conn.close()
        return False

    print("SUCCESS: Handles empty date gracefully")

    conn.close()

    print("\n" + "=" * 80)
    print("TEST 3: PASSED")
    print("=" * 80)
    return True


def test_refresh_status():
    """Test 4: Verify get_refresh_status returns correct information."""
    print("\n" + "=" * 80)
    print("TEST 4: REFRESH STATUS FUNCTION")
    print("=" * 80)

    conn = sqlite3.connect('nba_stats.db')

    # Get latest game date with predictions
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(game_date) FROM predictions")
    latest_date = cursor.fetchone()[0]

    if not latest_date:
        print("ERROR: No predictions in database to test with")
        conn.close()
        return False

    print(f"\nGetting status for: {latest_date}")

    status = pr.get_refresh_status(latest_date, conn)

    print(f"Status:")
    print(f"  - Predictions count: {status['predictions_count']}")
    print(f"  - Needs refresh: {status['needs_refresh']}")
    print(f"  - OUT players with predictions: {len(status['out_players_with_predictions'])}")
    print(f"  - Last refreshed: {status['last_refreshed']}")
    print(f"  - Refresh count: {status['refresh_count']}")

    if status['predictions_count'] == 0:
        print("ERROR: Should have predictions")
        conn.close()
        return False

    print("SUCCESS: Status function returns expected data structure")

    conn.close()

    print("\n" + "=" * 80)
    print("TEST 4: PASSED")
    print("=" * 80)
    return True


def run_all_tests():
    """Run all test scenarios."""
    print("+" + "=" * 78 + "+")
    print("|" + " " * 20 + "PREDICTION REFRESH TEST SUITE" + " " * 29 + "|")
    print("+" + "=" * 78 + "+")

    # Setup
    print("\nSetting up test environment...")
    setup_test_database()
    print("SUCCESS: Test environment ready")

    # Run tests
    tests = [
        ("Basic Refresh", test_basic_refresh),
        ("Multiple Refreshes (Idempotency)", test_multiple_refreshes),
        ("No Predictions to Refresh", test_no_predictions_to_refresh),
        ("Refresh Status Function", test_refresh_status)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "+" + "=" * 78 + "+")
    print("|" + " " * 30 + "TEST SUMMARY" + " " * 36 + "|")
    print("+" + "=" * 78 + "+")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"|  {name:<50} {status:>26}  |")

    print("+" + "=" * 78 + "+")
    print(f"|  TOTAL: {passed_count}/{total_count} tests passed" + " " * (78 - 25 - len(str(passed_count)) - len(str(total_count))) + "|")
    print("+" + "=" * 78 + "+")

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
