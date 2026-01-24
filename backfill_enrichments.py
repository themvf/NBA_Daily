#!/usr/bin/env python3
"""
Backfill Enrichments for Historical Predictions

Populates enrichment columns (days_rest, rest_multiplier, is_b2b,
game_script_tier, role_tier, position_matchup_factor) for existing
predictions that were created before the enrichment system.

Usage:
    python backfill_enrichments.py
    python backfill_enrichments.py --db nba_stats.db --days 30
    python backfill_enrichments.py --no-adjust  # Don't modify projections
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List
import argparse

from depth_chart import refresh_all_player_roles
from position_ppm_stats import refresh_team_position_defense
from prediction_enrichments import apply_enrichments_to_predictions


def get_prediction_dates(conn: sqlite3.Connection, days: int = None) -> List[str]:
    """Get all unique game dates with predictions."""
    cursor = conn.cursor()

    if days:
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT DISTINCT date(game_date) as gd
            FROM predictions
            WHERE date(game_date) >= date(?)
            ORDER BY gd
        """, [cutoff])
    else:
        cursor.execute("""
            SELECT DISTINCT date(game_date) as gd
            FROM predictions
            ORDER BY gd
        """)

    return [row[0] for row in cursor.fetchall()]


def backfill_enrichments(
    conn: sqlite3.Connection,
    days: int = None,
    apply_adjustments: bool = False,
    verbose: bool = True
) -> dict:
    """
    Backfill enrichment data for historical predictions.

    Args:
        conn: Database connection
        days: Only process predictions from last N days (None = all)
        apply_adjustments: Whether to modify projected_ppg values
        verbose: Print progress

    Returns:
        Dict with backfill statistics
    """
    stats = {
        'dates_processed': 0,
        'total_enriched': 0,
        'total_b2b': 0,
        'total_well_rested': 0,
        'errors': []
    }

    # ==========================================================================
    # Step 1: Ensure player roles are populated
    # ==========================================================================
    if verbose:
        print("Step 1: Refreshing player roles table...")

    try:
        roles_count = refresh_all_player_roles(conn, '2025-26')
        if verbose:
            print(f"  -> {roles_count} player roles refreshed")
    except Exception as e:
        if verbose:
            print(f"  -> Warning: Could not refresh roles: {e}")
        stats['errors'].append(f"Role refresh: {e}")

    # ==========================================================================
    # Step 2: Ensure team position defense is populated
    # ==========================================================================
    if verbose:
        print("\nStep 2: Refreshing team position defense ratings...")

    try:
        defense_count = refresh_team_position_defense(conn, '2025-26')
        if verbose:
            print(f"  -> {defense_count} team position defense records refreshed")
    except Exception as e:
        if verbose:
            print(f"  -> Warning: Could not refresh position defense: {e}")
        stats['errors'].append(f"Position defense refresh: {e}")

    # ==========================================================================
    # Step 3: Get all prediction dates
    # ==========================================================================
    if verbose:
        print("\nStep 3: Getting prediction dates...")

    dates = get_prediction_dates(conn, days)

    if not dates:
        if verbose:
            print("  -> No predictions found to backfill")
        return stats

    if verbose:
        print(f"  -> Found {len(dates)} dates with predictions")
        print(f"  -> Date range: {dates[0]} to {dates[-1]}")

    # ==========================================================================
    # Step 4: Process each date
    # ==========================================================================
    if verbose:
        print(f"\nStep 4: Processing predictions (apply_adjustments={apply_adjustments})...")
        print("-" * 60)

    for i, game_date in enumerate(dates):
        try:
            result = apply_enrichments_to_predictions(
                conn,
                game_date,
                apply_adjustments=apply_adjustments,
                verbose=False
            )

            stats['dates_processed'] += 1
            stats['total_enriched'] += result['players_enriched']
            stats['total_b2b'] += result['b2b_count']
            stats['total_well_rested'] += result['well_rested_count']

            if verbose:
                pct = (i + 1) / len(dates) * 100
                print(f"  [{pct:5.1f}%] {game_date}: {result['players_enriched']} players, "
                      f"{result['b2b_count']} B2B, {result['well_rested_count']} rested")

                if result['errors']:
                    for err in result['errors'][:3]:  # Show first 3 errors
                        print(f"          Warning: {err}")

        except Exception as e:
            stats['errors'].append(f"{game_date}: {e}")
            if verbose:
                print(f"  [ERROR] {game_date}: {e}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    if verbose:
        print("-" * 60)
        print(f"\nBackfill Complete!")
        print(f"  Dates processed: {stats['dates_processed']}")
        print(f"  Total predictions enriched: {stats['total_enriched']}")
        print(f"  Total B2B situations: {stats['total_b2b']}")
        print(f"  Total well-rested (3+ days): {stats['total_well_rested']}")
        if stats['errors']:
            print(f"  Errors encountered: {len(stats['errors'])}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Backfill enrichments for historical predictions')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    parser.add_argument('--days', type=int, default=None,
                        help='Only process last N days (default: all)')
    parser.add_argument('--no-adjust', action='store_true',
                        help="Don't modify projected_ppg values, only populate enrichment columns")
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    print("=" * 60)
    print("ENRICHMENT BACKFILL")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Days to process: {'All' if args.days is None else args.days}")
    print(f"Adjust projections: {not args.no_adjust}")
    print("=" * 60)
    print()

    stats = backfill_enrichments(
        conn,
        days=args.days,
        apply_adjustments=not args.no_adjust,
        verbose=not args.quiet
    )

    conn.close()

    return 0 if not stats['errors'] else 1


if __name__ == "__main__":
    exit(main())
