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

    # Verify only (run sanity checks without backfilling)
    python backfill_sim_probs.py --verify-only

    # Backfill + verify
    python backfill_sim_probs.py --verify
"""

import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import prediction_tracking as pt


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_coverage(conn: sqlite3.Connection, date_from: str = None, date_to: str = None) -> dict:
    """
    Check A: Coverage should be near 100%.

    Returns dict with coverage stats and pass/fail status.
    """
    query = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN p_top3 IS NOT NULL THEN 1 ELSE 0 END) as with_sim,
            SUM(CASE WHEN p_top3 IS NULL THEN 1 ELSE 0 END) as missing_sim
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

    cursor = conn.execute(query, params)
    total, with_sim, missing_sim = cursor.fetchone()

    coverage_pct = (with_sim / total * 100) if total > 0 else 0

    return {
        'total': total,
        'with_sim': with_sim,
        'missing_sim': missing_sim,
        'coverage_pct': coverage_pct,
        'passed': coverage_pct >= 99.0,  # Pass if >= 99% coverage
        'status': 'PASS' if coverage_pct >= 99.0 else ('WARN' if coverage_pct >= 90 else 'FAIL')
    }


def verify_probability_sanity(conn: sqlite3.Connection, date_from: str = None, date_to: str = None) -> dict:
    """
    Check B: Probabilities should be sane.

    For each slate:
    - sum(p_top1) should be ~1.0 (close, not exact if filtered)
    - max(p_top1) should be 5-25% (if 80%+, variance model is broken)

    Returns dict with sanity stats and issues found.
    """
    query = """
        SELECT
            game_date,
            COUNT(*) as n_players,
            SUM(p_top1) as sum_p_top1,
            MAX(p_top1) as max_p_top1,
            MIN(p_top1) as min_p_top1,
            AVG(p_top1) as avg_p_top1
        FROM predictions
        WHERE p_top1 IS NOT NULL
    """
    params = []
    if date_from:
        query += " AND game_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND game_date <= ?"
        params.append(date_to)

    query += " GROUP BY game_date ORDER BY game_date"

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    issues = []
    slates_checked = 0
    slates_with_issues = 0

    for game_date, n_players, sum_p1, max_p1, min_p1, avg_p1 in rows:
        slates_checked += 1
        slate_issues = []

        # Check 1: sum(p_top1) should be close to 1.0 (within 0.1)
        if sum_p1 is not None and abs(sum_p1 - 1.0) > 0.15:
            slate_issues.append(f"sum(p_top1)={sum_p1:.3f} (expected ~1.0)")

        # Check 2: max(p_top1) should be reasonable (5-35%)
        # Very high (>50%) suggests broken variance
        if max_p1 is not None and max_p1 > 0.50:
            slate_issues.append(f"max(p_top1)={max_p1:.1%} (too high, variance may be broken)")

        # Check 3: max(p_top1) shouldn't be too low either (<3% with many players)
        if max_p1 is not None and max_p1 < 0.03 and n_players > 30:
            slate_issues.append(f"max(p_top1)={max_p1:.1%} (too low for {n_players} players)")

        if slate_issues:
            slates_with_issues += 1
            issues.append({
                'game_date': game_date,
                'n_players': n_players,
                'sum_p_top1': sum_p1,
                'max_p_top1': max_p1,
                'issues': slate_issues
            })

    passed = slates_with_issues == 0

    return {
        'slates_checked': slates_checked,
        'slates_with_issues': slates_with_issues,
        'issues': issues[:10],  # Limit to first 10 issues
        'passed': passed,
        'status': 'PASS' if passed else ('WARN' if slates_with_issues <= 2 else 'FAIL')
    }


def verify_sorting_fix(conn: sqlite3.Connection, sample_dates: int = 3) -> dict:
    """
    Check C: Sorting should now reflect probabilities.

    For sample dates, check that:
    - Stars (season_avg >= 25) have p_top3 values (not NULL)
    - Their ranks by p_top3 are reasonable (not #76 for a star)

    Returns dict with verification results.
    """
    # Get some recent dates with simulation data
    cursor = conn.execute("""
        SELECT DISTINCT game_date
        FROM predictions
        WHERE p_top3 IS NOT NULL
        ORDER BY game_date DESC
        LIMIT ?
    """, [sample_dates])
    dates = [row[0] for row in cursor.fetchall()]

    if not dates:
        return {
            'dates_checked': 0,
            'stars_checked': 0,
            'stars_missing_sim': 0,
            'stars_buried': 0,
            'passed': False,
            'status': 'FAIL',
            'details': []
        }

    stars_checked = 0
    stars_missing_sim = 0
    stars_buried = 0  # Stars ranked worse than #20 by p_top3
    details = []

    for game_date in dates:
        # Get stars (season_avg >= 25) for this date
        cursor = conn.execute("""
            SELECT
                player_name,
                season_avg_ppg,
                projected_ppg,
                p_top3,
                p_top1,
                RANK() OVER (ORDER BY COALESCE(p_top3, 0) DESC) as sim_rank
            FROM predictions
            WHERE game_date = ? AND season_avg_ppg >= 25
            ORDER BY season_avg_ppg DESC
        """, [game_date])

        stars = cursor.fetchall()

        for player_name, season_avg, proj, p_top3, p_top1, sim_rank in stars:
            stars_checked += 1

            if p_top3 is None:
                stars_missing_sim += 1
                details.append({
                    'date': game_date,
                    'player': player_name,
                    'issue': 'MISSING SIM',
                    'season_avg': season_avg
                })
            elif sim_rank > 20:
                stars_buried += 1
                details.append({
                    'date': game_date,
                    'player': player_name,
                    'issue': f'BURIED (rank #{sim_rank})',
                    'season_avg': season_avg,
                    'p_top3': p_top3
                })

    passed = stars_missing_sim == 0 and stars_buried == 0

    return {
        'dates_checked': len(dates),
        'stars_checked': stars_checked,
        'stars_missing_sim': stars_missing_sim,
        'stars_buried': stars_buried,
        'passed': passed,
        'status': 'PASS' if passed else 'FAIL',
        'details': details[:10]
    }


def run_all_verifications(conn: sqlite3.Connection, date_from: str = None, date_to: str = None) -> dict:
    """
    Run all three verification checks and return combined results.
    """
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS")
    print("=" * 60)

    results = {}
    all_passed = True

    # Check A: Coverage
    print("\n[A] COVERAGE CHECK")
    print("-" * 40)
    coverage = verify_coverage(conn, date_from, date_to)
    results['coverage'] = coverage
    print(f"  Total predictions: {coverage['total']:,}")
    print(f"  With simulation: {coverage['with_sim']:,}")
    print(f"  Missing simulation: {coverage['missing_sim']:,}")
    print(f"  Coverage: {coverage['coverage_pct']:.1f}%")
    print(f"  Status: [{coverage['status']}]")
    if coverage['status'] == 'FAIL':
        all_passed = False

    # Check B: Probability Sanity
    print("\n[B] PROBABILITY SANITY CHECK")
    print("-" * 40)
    sanity = verify_probability_sanity(conn, date_from, date_to)
    results['sanity'] = sanity
    print(f"  Slates checked: {sanity['slates_checked']}")
    print(f"  Slates with issues: {sanity['slates_with_issues']}")
    if sanity['issues']:
        print("  Issues found:")
        for issue in sanity['issues'][:5]:
            print(f"    {issue['game_date']}: {', '.join(issue['issues'])}")
    print(f"  Status: [{sanity['status']}]")
    if sanity['status'] == 'FAIL':
        all_passed = False

    # Check C: Sorting Fix
    print("\n[C] STAR BURIAL CHECK")
    print("-" * 40)
    sorting = verify_sorting_fix(conn)
    results['sorting'] = sorting
    print(f"  Dates checked: {sorting['dates_checked']}")
    print(f"  Stars checked: {sorting['stars_checked']}")
    print(f"  Stars missing sim: {sorting['stars_missing_sim']}")
    print(f"  Stars buried (rank > 20): {sorting['stars_buried']}")
    if sorting['details']:
        print("  Issues found:")
        for detail in sorting['details'][:5]:
            print(f"    {detail['date']} - {detail['player']}: {detail['issue']}")
    print(f"  Status: [{sorting['status']}]")
    if sorting['status'] == 'FAIL':
        all_passed = False

    # Overall
    print("\n" + "=" * 60)
    overall_status = 'PASS' if all_passed else 'FAIL'
    print(f"OVERALL VERIFICATION: [{overall_status}]")
    print("=" * 60)

    results['overall_passed'] = all_passed
    return results


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
    parser.add_argument('--verify', action='store_true', help='Run verification after backfill')
    parser.add_argument('--verify-only', action='store_true', help='Only run verification (no backfill)')
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

    # Verify-only mode
    if args.verify_only:
        run_all_verifications(conn, args.date_from, args.date_to)
        conn.close()
        return

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

    # Quick coverage check
    print()
    print("Quick Coverage Check:")
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN p_top3 IS NOT NULL THEN 1 ELSE 0 END) as with_sim
        FROM predictions
    """)
    total, with_sim = cursor.fetchone()
    pct = (with_sim / total * 100) if total > 0 else 0
    print(f"  {with_sim:,}/{total:,} predictions now have simulation values ({pct:.1f}%)")

    # Run full verification if requested
    if args.verify:
        run_all_verifications(conn, args.date_from, args.date_to)

    conn.close()


if __name__ == "__main__":
    main()
