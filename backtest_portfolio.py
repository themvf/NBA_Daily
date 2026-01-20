#!/usr/bin/env python3
"""
Portfolio Backtest System for 20-Lineup Tournament Strategy

Evaluates how well our 20-lineup portfolio would have performed historically.

Key Concept:
- Contest winner has the HIGHEST COMBINED SUM of 3 players' points
- Our portfolio "wins" if ANY of our 20 lineups achieves the optimal 3-player sum

Metrics:
- won_slate: Did any lineup achieve the optimal 3-player sum?
- sum_shortfall: How many points short was our best lineup?
- rank_percentile: Where did our best lineup rank among all possible lineups?
- hit_2_of_top3_any: Did any lineup have 2+ of the actual top 3 scorers?

Usage:
    python backtest_portfolio.py --days 30
    python backtest_portfolio.py --from 2026-01-01 --to 2026-01-15
"""

import sqlite3
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path
import json
import warnings

# Import from our project
from lineup_optimizer import (
    TournamentLineupOptimizer,
    PlayerPool,
    OptimizerConfig,
    create_player_pool_from_predictions
)
from correlation_model import (
    PlayerCorrelationModel,
    PlayerSlateInfo,
    CorrelatedSimResult
)


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

def create_portfolio_tables(conn: sqlite3.Connection) -> None:
    """Create tables for portfolio storage and backtest results."""
    cursor = conn.cursor()

    # Table to store generated portfolios (20 lineups per date)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_lineups (
            lineup_id INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date TEXT NOT NULL,
            lineup_index INTEGER NOT NULL,
            bucket TEXT NOT NULL,

            player1_id INTEGER NOT NULL,
            player1_name TEXT,
            player2_id INTEGER NOT NULL,
            player2_name TEXT,
            player3_id INTEGER NOT NULL,
            player3_name TEXT,

            estimated_win_prob REAL,
            stack_game TEXT,

            created_at TEXT DEFAULT (datetime('now')),

            UNIQUE(slate_date, lineup_index)
        )
    """)

    # Table for portfolio backtest results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_portfolio_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date TEXT NOT NULL UNIQUE,

            -- Best lineup from portfolio
            best_lineup_ids TEXT,
            best_lineup_names TEXT,
            best_lineup_sum REAL,
            best_lineup_rank INTEGER,

            -- Optimal lineup (brute force from actuals)
            optimal_lineup_ids TEXT,
            optimal_lineup_names TEXT,
            optimal_lineup_sum REAL,

            -- Win metrics
            won_slate INTEGER,
            sum_shortfall REAL,
            rank_percentile REAL,

            -- Coverage metrics
            hit_2_of_top3_any INTEGER,
            hit_all_top3_any INTEGER,
            unique_players_used INTEGER,

            -- Estimated vs actual
            estimated_portfolio_win_prob REAL,

            -- Context
            n_games INTEGER,
            n_players_on_slate INTEGER,
            total_possible_lineups INTEGER,

            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_portfolio_date
        ON portfolio_lineups(slate_date)
    """)

    conn.commit()


# =============================================================================
# LINEUP COMPUTATION
# =============================================================================

def compute_all_lineup_sums(
    actuals: pd.DataFrame,
    lineup_size: int = 3
) -> List[Tuple[List[int], float]]:
    """
    Compute all possible 3-player lineup sums from actual scores.

    Returns list of (player_ids, sum) tuples, sorted by sum descending.

    WARNING: This is O(C(n,3)) where n is number of players.
    For 100 players, that's ~160,000 combinations.
    """
    player_ids = actuals['player_id'].tolist()
    player_points = dict(zip(actuals['player_id'], actuals['actual_ppg']))

    all_lineups = []
    for combo in combinations(player_ids, lineup_size):
        lineup_sum = sum(player_points.get(pid, 0) for pid in combo)
        all_lineups.append((list(combo), lineup_sum))

    # Sort by sum descending
    all_lineups.sort(key=lambda x: x[1], reverse=True)

    return all_lineups


def compute_optimal_lineup(actuals: pd.DataFrame) -> Tuple[List[int], float]:
    """
    Compute the optimal 3-player lineup from actual scores.

    With independent scores, the optimal lineup is simply the top 3 scorers.
    This is O(n) instead of O(n^3).
    """
    sorted_actuals = actuals.nlargest(3, 'actual_ppg')

    optimal_ids = sorted_actuals['player_id'].tolist()
    optimal_sum = sorted_actuals['actual_ppg'].sum()

    return optimal_ids, optimal_sum


def evaluate_portfolio_against_actuals(
    lineups: List[List[int]],
    actuals: pd.DataFrame,
    compute_all_ranks: bool = False
) -> Dict:
    """
    Evaluate a portfolio of lineups against actual scores.

    Args:
        lineups: List of lineups, each lineup is a list of player_ids
        actuals: DataFrame with player_id, player_name, actual_ppg
        compute_all_ranks: If True, compute full ranking (expensive)

    Returns:
        Dict with evaluation metrics
    """
    # Build lookup
    points_lookup = dict(zip(actuals['player_id'], actuals['actual_ppg']))
    name_lookup = dict(zip(actuals['player_id'], actuals['player_name']))

    # Get actual top 3 scorers
    top3_actual = set(actuals.nlargest(3, 'actual_ppg')['player_id'].tolist())

    # Compute optimal lineup
    optimal_ids, optimal_sum = compute_optimal_lineup(actuals)
    optimal_names = [name_lookup.get(pid, str(pid)) for pid in optimal_ids]

    # Evaluate each lineup in portfolio
    best_sum = -np.inf
    best_lineup_ids = []
    best_lineup_names = []
    hit_2_of_top3_any = 0
    hit_all_top3_any = 0

    for lineup in lineups:
        lineup_sum = sum(points_lookup.get(pid, 0) for pid in lineup)
        lineup_overlap = len(set(lineup) & top3_actual)

        if lineup_sum > best_sum:
            best_sum = lineup_sum
            best_lineup_ids = lineup
            best_lineup_names = [name_lookup.get(pid, str(pid)) for pid in lineup]

        if lineup_overlap >= 2:
            hit_2_of_top3_any = 1
        if lineup_overlap >= 3:
            hit_all_top3_any = 1

    # Did we win? (best lineup sum equals optimal sum)
    won_slate = 1 if best_sum >= optimal_sum - 0.01 else 0
    sum_shortfall = optimal_sum - best_sum

    # Compute rank of our best lineup among ALL possible lineups
    if compute_all_ranks:
        all_lineups = compute_all_lineup_sums(actuals)
        total_possible = len(all_lineups)

        # Find rank of our best lineup
        best_rank = 1
        for i, (_, lineup_sum) in enumerate(all_lineups, 1):
            if lineup_sum <= best_sum:
                best_rank = i
                break

        rank_percentile = best_rank / total_possible
    else:
        total_possible = None
        best_rank = None
        rank_percentile = None

    # Count unique players across portfolio
    unique_players = set()
    for lineup in lineups:
        unique_players.update(lineup)

    return {
        'best_lineup_ids': best_lineup_ids,
        'best_lineup_names': best_lineup_names,
        'best_lineup_sum': best_sum,
        'best_lineup_rank': best_rank,
        'optimal_lineup_ids': optimal_ids,
        'optimal_lineup_names': optimal_names,
        'optimal_lineup_sum': optimal_sum,
        'won_slate': won_slate,
        'sum_shortfall': sum_shortfall,
        'rank_percentile': rank_percentile,
        'hit_2_of_top3_any': hit_2_of_top3_any,
        'hit_all_top3_any': hit_all_top3_any,
        'unique_players_used': len(unique_players),
        'total_possible_lineups': total_possible
    }


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def get_actuals_for_date(conn: sqlite3.Connection, game_date: str) -> pd.DataFrame:
    """
    Load actual scoring results from player_game_logs.

    Uses player_game_logs as ground truth, NOT predictions table.
    """
    query = """
        SELECT
            player_id,
            player_name,
            points as actual_ppg,
            minutes,
            team_abbreviation as team
        FROM player_game_logs
        WHERE game_date = ?
          AND points IS NOT NULL
          AND minutes > 0
        ORDER BY points DESC
    """
    df = pd.read_sql_query(query, conn, params=[game_date])

    # Fallback to predictions if needed
    if df.empty:
        fallback_query = """
            SELECT
                player_id,
                player_name,
                actual_ppg,
                actual_minutes as minutes,
                team_name as team
            FROM predictions
            WHERE game_date = ?
              AND actual_ppg IS NOT NULL
            ORDER BY actual_ppg DESC
        """
        df = pd.read_sql_query(fallback_query, conn, params=[game_date])

    return df


def get_predictions_for_date(conn: sqlite3.Connection, game_date: str) -> pd.DataFrame:
    """Load predictions for a date to re-generate lineups."""
    query = """
        SELECT
            player_id,
            player_name,
            team_name as team_abbreviation,
            game_id,
            projected_ppg,
            proj_ceiling,
            proj_floor,
            p_top1,
            p_top3,
            season_avg_ppg,
            injury_status,
            injury_adjusted
        FROM predictions
        WHERE game_date = ?
          AND projected_ppg IS NOT NULL
        ORDER BY projected_ppg DESC
    """
    return pd.read_sql_query(query, conn, params=[game_date])


def get_stored_portfolio(conn: sqlite3.Connection, game_date: str) -> Optional[List[List[int]]]:
    """Retrieve stored portfolio lineups for a date, if available."""
    query = """
        SELECT player1_id, player2_id, player3_id
        FROM portfolio_lineups
        WHERE slate_date = ?
        ORDER BY lineup_index
    """
    df = pd.read_sql_query(query, conn, params=[game_date])

    if df.empty:
        return None

    lineups = []
    for _, row in df.iterrows():
        lineups.append([row['player1_id'], row['player2_id'], row['player3_id']])

    return lineups


def store_portfolio(
    conn: sqlite3.Connection,
    game_date: str,
    lineups: List,  # List of Lineup objects
    estimated_win_prob: float
) -> None:
    """Store generated portfolio lineups."""
    cursor = conn.cursor()

    # Delete existing entries for this date
    cursor.execute("DELETE FROM portfolio_lineups WHERE slate_date = ?", [game_date])

    for i, lineup in enumerate(lineups):
        player_ids = lineup.player_ids()
        player_names = lineup.player_names()

        cursor.execute("""
            INSERT INTO portfolio_lineups
            (slate_date, lineup_index, bucket, player1_id, player1_name,
             player2_id, player2_name, player3_id, player3_name,
             estimated_win_prob, stack_game)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            game_date, i, lineup.bucket,
            player_ids[0], player_names[0] if player_names else '',
            player_ids[1], player_names[1] if len(player_names) > 1 else '',
            player_ids[2], player_names[2] if len(player_names) > 2 else '',
            lineup.win_probability,
            lineup.stack_game
        ])

    conn.commit()


def backtest_portfolio_date(
    conn: sqlite3.Connection,
    game_date: str,
    regenerate: bool = False,
    compute_full_rank: bool = True
) -> Optional[Dict]:
    """
    Backtest portfolio for a single date.

    Args:
        conn: Database connection
        game_date: Date to backtest (YYYY-MM-DD)
        regenerate: If True, regenerate lineups even if stored
        compute_full_rank: If True, compute rank among all possible lineups

    Returns:
        Dict with backtest metrics, or None if insufficient data
    """
    # Load actuals
    actuals = get_actuals_for_date(conn, game_date)
    if actuals.empty or len(actuals) < 10:
        return None

    # Try to get stored portfolio, or regenerate
    lineups = None
    estimated_win_prob = 0.0

    if not regenerate:
        lineups = get_stored_portfolio(conn, game_date)

    if lineups is None:
        # Need to regenerate from predictions
        predictions = get_predictions_for_date(conn, game_date)
        if predictions.empty or len(predictions) < 10:
            return None

        # Build player pool
        player_pool = create_player_pool_from_predictions(predictions)

        if len(player_pool) < 10:
            return None

        # Build game environments (basic - could enhance with stored Vegas data)
        game_ids = predictions['game_id'].dropna().unique()
        game_envs = {gid: {'stack_score': 0.6, 'ot_probability': 0.06} for gid in game_ids}

        # Run optimizer
        optimizer = TournamentLineupOptimizer(
            player_pool=player_pool,
            game_environments=game_envs
        )

        result = optimizer.optimize()
        lineups = [l.player_ids() for l in result.lineups]
        estimated_win_prob = result.total_win_probability

        # Store for future use
        store_portfolio(conn, game_date, result.lineups, estimated_win_prob)

    # Evaluate against actuals
    eval_result = evaluate_portfolio_against_actuals(
        lineups, actuals, compute_all_ranks=compute_full_rank
    )

    # Add context
    n_games = len(actuals['team'].unique()) // 2  # Approximate
    eval_result['estimated_portfolio_win_prob'] = estimated_win_prob
    eval_result['n_games'] = n_games
    eval_result['n_players_on_slate'] = len(actuals)
    eval_result['slate_date'] = game_date

    return eval_result


def store_backtest_result(conn: sqlite3.Connection, result: Dict) -> None:
    """Store backtest result in the database."""
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO backtest_portfolio_results (
            slate_date,
            best_lineup_ids, best_lineup_names, best_lineup_sum, best_lineup_rank,
            optimal_lineup_ids, optimal_lineup_names, optimal_lineup_sum,
            won_slate, sum_shortfall, rank_percentile,
            hit_2_of_top3_any, hit_all_top3_any, unique_players_used,
            estimated_portfolio_win_prob,
            n_games, n_players_on_slate, total_possible_lineups
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        result['slate_date'],
        json.dumps(result['best_lineup_ids']),
        json.dumps(result['best_lineup_names']),
        result['best_lineup_sum'],
        result.get('best_lineup_rank'),
        json.dumps(result['optimal_lineup_ids']),
        json.dumps(result['optimal_lineup_names']),
        result['optimal_lineup_sum'],
        result['won_slate'],
        result['sum_shortfall'],
        result.get('rank_percentile'),
        result['hit_2_of_top3_any'],
        result['hit_all_top3_any'],
        result['unique_players_used'],
        result.get('estimated_portfolio_win_prob', 0),
        result.get('n_games', 0),
        result.get('n_players_on_slate', 0),
        result.get('total_possible_lineups')
    ])

    conn.commit()


def run_portfolio_backtest(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    regenerate: bool = False,
    compute_full_rank: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run portfolio backtest over a date range.

    Returns DataFrame with results for each date.
    """
    # Ensure tables exist
    create_portfolio_tables(conn)

    # Get available dates in range
    query = """
        SELECT DISTINCT game_date
        FROM predictions
        WHERE game_date BETWEEN ? AND ?
          AND actual_ppg IS NOT NULL
        ORDER BY game_date
    """
    dates_df = pd.read_sql_query(query, conn, params=[date_from, date_to])
    dates = dates_df['game_date'].tolist()

    if verbose:
        print(f"Running portfolio backtest for {len(dates)} dates...")

    results = []
    wins = 0

    for i, game_date in enumerate(dates, 1):
        if verbose:
            print(f"  [{i}/{len(dates)}] {game_date}...", end=' ')

        try:
            result = backtest_portfolio_date(
                conn, game_date, regenerate=regenerate, compute_full_rank=compute_full_rank
            )

            if result:
                store_backtest_result(conn, result)
                results.append(result)
                wins += result['won_slate']

                if verbose:
                    status = "WON" if result['won_slate'] else f"LOST (shortfall: {result['sum_shortfall']:.1f})"
                    print(status)
            else:
                if verbose:
                    print("SKIPPED (insufficient data)")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    # Summary
    if verbose and results:
        n = len(results)
        print(f"\n{'='*60}")
        print(f"PORTFOLIO BACKTEST SUMMARY ({len(dates)} dates)")
        print(f"{'='*60}")
        print(f"Slates won: {wins}/{n} ({100*wins/n:.1f}%)")
        print(f"Avg shortfall: {np.mean([r['sum_shortfall'] for r in results]):.2f} pts")
        print(f"Avg hit_2_of_top3: {np.mean([r['hit_2_of_top3_any'] for r in results]):.1%}")

        if results[0].get('rank_percentile') is not None:
            avg_pct = np.mean([r['rank_percentile'] for r in results if r.get('rank_percentile')])
            print(f"Avg rank percentile: {avg_pct:.2%} (lower is better)")

    return pd.DataFrame(results)


def get_backtest_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get summary statistics from stored backtest results."""
    query = """
        SELECT
            COUNT(*) as n_slates,
            SUM(won_slate) as wins,
            AVG(won_slate) * 100 as win_rate_pct,
            AVG(sum_shortfall) as avg_shortfall,
            AVG(rank_percentile) as avg_rank_percentile,
            AVG(hit_2_of_top3_any) * 100 as hit_2_of_top3_pct,
            AVG(hit_all_top3_any) * 100 as hit_all_top3_pct,
            AVG(unique_players_used) as avg_unique_players
        FROM backtest_portfolio_results
    """
    return pd.read_sql_query(query, conn)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Portfolio Backtest System')
    parser.add_argument('--db', default='nba_daily.db', help='Database path')
    parser.add_argument('--days', type=int, help='Number of days back to test')
    parser.add_argument('--from', dest='date_from', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', dest='date_to', help='End date (YYYY-MM-DD)')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate lineups')
    parser.add_argument('--no-rank', action='store_true', help='Skip full rank computation')
    parser.add_argument('--summary', action='store_true', help='Show summary only')

    args = parser.parse_args()

    # Determine database path
    db_path = Path(args.db)
    if not db_path.exists():
        # Try in project directory
        project_dir = Path(__file__).parent
        db_path = project_dir / args.db

    if not db_path.exists():
        print(f"Database not found: {args.db}")
        return

    conn = sqlite3.connect(str(db_path))

    try:
        # Ensure tables exist
        create_portfolio_tables(conn)

        if args.summary:
            summary = get_backtest_summary(conn)
            print("\nPortfolio Backtest Summary:")
            print(summary.to_string(index=False))
            return

        # Determine date range
        if args.days:
            date_to = datetime.now().strftime('%Y-%m-%d')
            date_from = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        elif args.date_from and args.date_to:
            date_from = args.date_from
            date_to = args.date_to
        else:
            print("Please specify --days or --from/--to")
            return

        # Run backtest
        results = run_portfolio_backtest(
            conn,
            date_from,
            date_to,
            regenerate=args.regenerate,
            compute_full_rank=not args.no_rank
        )

        if not results.empty:
            print(f"\nResults saved to database.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
