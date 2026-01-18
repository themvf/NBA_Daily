#!/usr/bin/env python3
"""
Backfill simulation probabilities for historical predictions.

This script runs Monte Carlo simulation for all dates in the predictions table
that don't have p_top3/p_top1 values, and persists the results.

Usage:
    # Backfill all dates missing simulation
    python backfill_sim_probs.py

    # Backfill specific date range
    python backfill_sim_probs.py --from 2025-12-01 --to 2025-12-31

    # Backfill with custom simulation count
    python backfill_sim_probs.py --sim-n 5000

    # Dry run (show what would be done)
    python backfill_sim_probs.py --dry-run
"""

import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import prediction_tracking as pt


def get_dates_needing_backfill(conn: sqlite3.Connection, date_from: str = None, date_to: str = None) -> list:
    """
    Get list of dates that have predictions but missing simulation values.

    Returns list of (game_date, player_count, sim_count) tuples.
    """
    query = """
        SELECT
            game_date,
            COUNT(*) as total_players,
            SUM(CASE WHEN p_top3 IS NOT NULL THEN 1 ELSE 0 END) as sim_players
        FROM predictions
        WHERE 1=1
    """
    params = []

    if date_from:
        query += " AND game_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND game_date <= ?"
        params.append(date_to)

    query += """
        GROUP BY game_date
        HAVING sim_players < total_players
        ORDER BY game_date
    """

    cursor = conn.execute(query, params)
    return cursor.fetchall()


def backfill_date(conn: sqlite3.Connection, game_date: str, sim_n: int = 10000, verbose: bool = True) -> dict:
    """
    Run simulation for a single date and persist results.

    Returns dict with status and stats.
    """
    try:
        # Import ranking module
        import top3_ranking as t3r

        # Create ranker
        ranker = t3r.Top3Ranker(conn)

        # Run simulation
        if verbose:
            print(f"  Running simulation ({sim_n} iterations)...", end=" ", flush=True)

        sim_df = ranker.simulate_top3_probability(game_date, n_simulations=sim_n)

        if sim_df.empty:
            if verbose:
                print("No predictions found")
            return {'status': 'skip', 'reason': 'no_predictions', 'count': 0}

        # Also calculate top_scorer_score
        if verbose:
            print("Calculating TopScorerScore...", end=" ", flush=True)

        score_df = ranker.rank_by_top_scorer_score(game_date, include_components=False)

        # Merge top_scorer_score into sim_df
        if not score_df.empty and 'top_scorer_score' in score_df.columns:
            sim_df = sim_df.merge(
                score_df[['player_id', 'top_scorer_score']],
                on='player_id',
                how='left'
            )

        # Persist results
        if verbose:
            print("Persisting...", end=" ", flush=True)

        updated = pt.persist_sim_results(
            conn,
            game_date,
            sim_df,
            sim_n=sim_n,
            sim_seed=None,
            sim_version='backfill-v1.0'
        )

        if verbose:
            print(f"OK ({updated} rows)")

        return {'status': 'ok', 'count': updated}

    except Exception as e:
        if verbose:
            print(f"FAILED: {e}")
        return {'status': 'failed', 'error': str(e), 'count': 0}


def main():
    parser = argparse.ArgumentParser(description='Backfill simulation probabilities')
    parser.add_argument('--db', default='nba_stats.db', help='Database path')
    parser.add_argument('--from', dest='date_from', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', dest='date_to', help='End date (YYYY-MM-DD)')
    parser.add_argument('--sim-n', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    # Connect to database
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))

    # Ensure simulation columns exist
    print("=" * 60)
    print("BACKFILL SIMULATION PROBABILITIES")
    print("=" * 60)
    print(f"\nDatabase: {db_path}")
    print(f"Simulations per date: {args.sim_n:,}")
    print()

    print("Ensuring simulation columns exist...")
    pt.ensure_prediction_sim_columns(conn)
    print()

    # Get dates needing backfill
    print("Finding dates needing backfill...")
    dates = get_dates_needing_backfill(conn, args.date_from, args.date_to)

    if not dates:
        print("No dates need backfill - all predictions have simulation values!")
        conn.close()
        return

    total_players = sum(row[1] for row in dates)
    sim_players = sum(row[2] for row in dates)
    missing_players = total_players - sim_players

    print(f"\nFound {len(dates)} dates needing backfill:")
    print(f"  Total predictions: {total_players:,}")
    print(f"  Already simulated: {sim_players:,}")
    print(f"  Missing simulation: {missing_players:,}")
    print()

    if args.dry_run:
        print("DRY RUN - would process these dates:")
        for game_date, total, simmed in dates[:20]:
            missing = total - simmed
            print(f"  {game_date}: {missing}/{total} missing")
        if len(dates) > 20:
            print(f"  ... and {len(dates) - 20} more dates")
        conn.close()
        return

    # Process each date
    print("Processing dates...")
    print("-" * 60)

    stats = {'ok': 0, 'skip': 0, 'failed': 0, 'total_rows': 0}

    for i, (game_date, total, simmed) in enumerate(dates, 1):
        missing = total - simmed
        if not args.quiet:
            print(f"[{i}/{len(dates)}] {game_date} ({missing} players):", end=" ")

        result = backfill_date(conn, game_date, sim_n=args.sim_n, verbose=not args.quiet)

        stats[result['status']] += 1
        stats['total_rows'] += result.get('count', 0)

    # Summary
    print()
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Dates processed: {len(dates)}")
    print(f"  Success: {stats['ok']}")
    print(f"  Skipped: {stats['skip']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total rows updated: {stats['total_rows']:,}")

    # Verify
    print()
    print("Verification:")
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN p_top3 IS NOT NULL THEN 1 ELSE 0 END) as with_sim
        FROM predictions
    """)
    total, with_sim = cursor.fetchone()
    pct = (with_sim / total * 100) if total > 0 else 0
    print(f"  {with_sim:,}/{total:,} predictions now have simulation values ({pct:.1f}%)")

    conn.close()


if __name__ == "__main__":
    main()
