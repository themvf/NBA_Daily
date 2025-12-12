#!/usr/bin/env python3
"""
Fetch Player Positions from NBA API and Update Database

This script fetches position data for all active players using the NBA API's
CommonPlayerInfo endpoint and updates the players table.

Position Normalization:
- Guard, Point Guard, Shooting Guard → "Guard"
- Forward, Small Forward, Power Forward → "Forward"
- Center, Forward-Center → "Center"
- Guard-Forward → "Guard" (guards have priority)
"""

import sqlite3
import time
from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd


def normalize_position(position: str) -> str:
    """
    Normalize NBA API positions to Guard/Forward/Center.

    Args:
        position: Raw position from NBA API (e.g., "Point Guard", "Forward-Center")

    Returns:
        Normalized position: "Guard", "Forward", or "Center"
    """
    if not position:
        return "Unknown"

    pos_lower = position.lower()

    # Guard positions
    if 'guard' in pos_lower:
        return "Guard"

    # Center positions (check before forward to catch "Forward-Center")
    if 'center' in pos_lower:
        return "Center"

    # Forward positions
    if 'forward' in pos_lower:
        return "Forward"

    # Default to Unknown if can't determine
    return "Unknown"


def fetch_and_update_positions(conn, limit=None, delay=0.6):
    """
    Fetch positions for all players and update the database.

    Args:
        conn: Database connection
        limit: Optional limit for testing (fetches only N players)
        delay: Delay between API calls in seconds (default 0.6 = 100 requests/min)
    """
    cursor = conn.cursor()

    # Get all unique players from player_game_logs who don't have positions yet
    query = """
        SELECT DISTINCT pgl.player_id, pgl.player_name
        FROM player_game_logs pgl
        LEFT JOIN players p ON pgl.player_id = p.player_id
        WHERE p.position IS NULL OR p.position = ''
        ORDER BY pgl.player_name
    """

    if limit:
        query += f" LIMIT {limit}"

    players_df = pd.read_sql_query(query, conn)

    if players_df.empty:
        print("All players already have position data!")
        return

    print(f"Found {len(players_df)} players without position data")
    print(f"Fetching positions from NBA API (delay: {delay}s between calls)...\n")

    success_count = 0
    error_count = 0

    for idx, row in players_df.iterrows():
        player_id = row['player_id']
        player_name = row['player_name']

        try:
            # Fetch player info from NBA API
            player_info = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id,
                timeout=30
            )

            df = player_info.get_data_frames()[0]

            if not df.empty and 'POSITION' in df.columns:
                raw_position = df['POSITION'].iloc[0]
                normalized_position = normalize_position(raw_position)

                # Update database
                cursor.execute(
                    "UPDATE players SET position = ? WHERE player_id = ?",
                    (normalized_position, player_id)
                )
                conn.commit()

                success_count += 1
                print(f"[{idx+1}/{len(players_df)}] {player_name}: {raw_position} -> {normalized_position}")
            else:
                print(f"[{idx+1}/{len(players_df)}] {player_name}: No position data found")
                error_count += 1

            # Rate limiting
            time.sleep(delay)

        except Exception as e:
            print(f"[{idx+1}/{len(players_df)}] {player_name}: ERROR - {str(e)}")
            error_count += 1
            time.sleep(delay * 2)  # Double delay after error

    print(f"\n=== SUMMARY ===")
    print(f"Successfully updated: {success_count} players")
    print(f"Errors: {error_count} players")
    print(f"Total: {len(players_df)} players")


def show_position_distribution(conn):
    """Display current position distribution in database."""
    query = """
        SELECT
            position,
            COUNT(*) as player_count
        FROM players
        WHERE position IS NOT NULL AND position != ''
        GROUP BY position
        ORDER BY player_count DESC
    """

    df = pd.read_sql_query(query, conn)

    if not df.empty:
        print("\n=== POSITION DISTRIBUTION ===")
        print(df.to_string(index=False))
    else:
        print("\nNo position data in database yet.")


if __name__ == '__main__':
    import sys
    import io

    # Fix Unicode encoding for Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("NBA Player Position Fetcher")
    print("=" * 60)

    conn = sqlite3.connect('nba_stats.db')

    # Show current distribution
    show_position_distribution(conn)

    # Check for command-line argument
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'full':
            print("\n--- FULL UPDATE: Fetching all players ---\n")
            fetch_and_update_positions(conn, limit=None, delay=0.6)
        elif mode == 'test':
            print("\n--- TEST MODE: Fetching 10 players ---\n")
            fetch_and_update_positions(conn, limit=10, delay=0.6)
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python fetch_player_positions.py [test|full]")
    else:
        # Interactive mode
        print("\nOptions:")
        print("  1. Test mode (fetch 10 players)")
        print("  2. Full update (all players)")
        print("  3. Skip update, show current data")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            print("\n--- TEST MODE: Fetching 10 players ---\n")
            fetch_and_update_positions(conn, limit=10, delay=0.6)
        elif choice == '2':
            confirm = input("\nThis will fetch data for all players. Continue? (yes/no): ").strip().lower()
            if confirm == 'yes':
                print("\n--- FULL UPDATE: Fetching all players ---\n")
                fetch_and_update_positions(conn, limit=None, delay=0.6)
            else:
                print("Cancelled.")
        else:
            print("Skipping update.")

    # Show final distribution
    show_position_distribution(conn)

    conn.close()
    print("\nDone!")
