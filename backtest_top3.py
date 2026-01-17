#!/usr/bin/env python3
"""
Top-3 Scorer Backtest Engine

Evaluates ranking strategy performance against actual game results.
Answers two key questions:
1. Did my picked 3 overlap the actual top 3? (tournament-style)
2. Were my projections accurate at ranking? (ranking quality)

Usage:
    python backtest_top3.py --days 30 --strategy sim_p_top3
    python backtest_top3.py --from 2026-01-01 --to 2026-01-15 --strategy sim_p_top3
    python backtest_top3.py --days 30 --compare-all
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

STRATEGIES = {
    'projection_only': {
        'ranking_field': 'projected_ppg',
        'description': 'Baseline: rank by projected PPG'
    },
    'ceiling_only': {
        'ranking_field': 'proj_ceiling',
        'description': 'Rank by scoring ceiling'
    },
    'top_scorer_score': {
        'ranking_field': 'top_scorer_score',
        'description': 'TSS heuristic formula'
    },
    'sim_p_top3': {
        'ranking_field': 'p_top3',
        'description': 'Monte Carlo P(top-3)'
    },
    'sim_p_first': {
        'ranking_field': 'p_top1',
        'description': 'Monte Carlo P(#1)'
    }
}


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def create_backtest_table(conn: sqlite3.Connection) -> None:
    """Create backtest_daily_results table if not exists."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_daily_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date TEXT NOT NULL,
            strategy_name TEXT NOT NULL,

            -- Our picks
            picked1_id INTEGER,
            picked1_name TEXT,
            picked2_id INTEGER,
            picked2_name TEXT,
            picked3_id INTEGER,
            picked3_name TEXT,

            -- Actual top 3
            actual1_id INTEGER,
            actual1_name TEXT,
            actual1_points REAL,
            actual2_id INTEGER,
            actual2_name TEXT,
            actual2_points REAL,
            actual3_id INTEGER,
            actual3_name TEXT,
            actual3_points REAL,

            -- Overlap metrics (tournament-style)
            overlap INTEGER,
            hit_any INTEGER,
            hit_2plus INTEGER,
            hit_exact INTEGER,
            hit_1 INTEGER,
            tie_friendly_overlap INTEGER,

            -- Ranking quality metrics
            pred_rank_a1 INTEGER,
            pred_rank_a2 INTEGER,
            pred_rank_a3 INTEGER,
            avg_rank_actual_top3 REAL,
            best_rank_actual_top3 INTEGER,

            -- Context
            n_pred_players INTEGER,
            actual_3rd_points REAL,
            ties_at_3rd INTEGER,

            created_at TEXT DEFAULT (datetime('now')),

            UNIQUE(slate_date, strategy_name)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_backtest_date
        ON backtest_daily_results(slate_date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_backtest_strategy
        ON backtest_daily_results(strategy_name)
    """)

    conn.commit()


def get_available_dates(conn: sqlite3.Connection) -> List[str]:
    """Get dates that have predictions with actual results."""
    query = """
        SELECT DISTINCT game_date
        FROM predictions
        WHERE actual_ppg IS NOT NULL
        ORDER BY game_date DESC
    """
    df = pd.read_sql_query(query, conn)
    return df['game_date'].tolist()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions(conn: sqlite3.Connection, game_date: str) -> pd.DataFrame:
    """
    Load predictions for a specific date.

    Returns DataFrame with all prediction columns needed for ranking.
    Handles missing columns gracefully (simulation fields may not exist in historical data).
    """
    # First, check which columns exist in the predictions table
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(predictions)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Build query with only existing columns
    base_columns = ['player_id', 'player_name', 'team_name', 'projected_ppg',
                    'proj_ceiling', 'proj_floor', 'actual_ppg', 'dfs_score']
    optional_columns = ['top_scorer_score', 'p_top3', 'p_top1']

    select_cols = ['p.' + col for col in base_columns if col in existing_columns]
    for col in optional_columns:
        if col in existing_columns:
            select_cols.append(f'p.{col}')

    query = f"""
        SELECT {', '.join(select_cols)}
        FROM predictions p
        WHERE p.game_date = ?
          AND p.actual_ppg IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=[game_date])

    # Add missing optional columns with default values
    for col in optional_columns:
        if col not in df.columns:
            df[col] = 0.0

    # Fill NaN ranking fields with 0
    for col in ['top_scorer_score', 'p_top3', 'p_top1']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def load_actuals(conn: sqlite3.Connection, game_date: str) -> pd.DataFrame:
    """
    Load actual scoring results for a specific date.

    Returns DataFrame with player_id, player_name, actual_ppg, sorted by points DESC.
    """
    query = """
        SELECT
            player_id,
            player_name,
            actual_ppg
        FROM predictions
        WHERE game_date = ?
          AND actual_ppg IS NOT NULL
        ORDER BY actual_ppg DESC
    """
    return pd.read_sql_query(query, conn, params=[game_date])


# =============================================================================
# TRUTH COMPUTATION
# =============================================================================

def break_ties(df: pd.DataFrame, points_col: str = 'actual_ppg') -> pd.DataFrame:
    """
    Deterministic tie-breaking for actual top 3:
    1. Higher points wins
    2. Lower player_id as final deterministic breaker
    """
    return df.sort_values(
        [points_col, 'player_id'],
        ascending=[False, True]
    ).reset_index(drop=True)


def compute_actual_top3(actuals: pd.DataFrame) -> Dict:
    """
    Compute the actual top 3 scorers for a date.

    Returns dict with:
    - top3: list of (player_id, player_name, points)
    - actual_3rd_points: points of #3 player
    - ties_at_3rd: True if there's a tie at the #3 cutoff
    """
    if actuals.empty or len(actuals) < 3:
        return {
            'top3': [],
            'actual_3rd_points': 0,
            'ties_at_3rd': False
        }

    # Apply deterministic tie-breaking
    sorted_df = break_ties(actuals)

    # Get top 3
    top3_df = sorted_df.head(3)
    actual_3rd_points = top3_df.iloc[2]['actual_ppg']

    # Check for ties at #3 position
    ties_at_3rd = (sorted_df['actual_ppg'] == actual_3rd_points).sum() > 1

    top3 = [
        (row['player_id'], row['player_name'], row['actual_ppg'])
        for _, row in top3_df.iterrows()
    ]

    return {
        'top3': top3,
        'actual_3rd_points': actual_3rd_points,
        'ties_at_3rd': ties_at_3rd
    }


# =============================================================================
# RANKING
# =============================================================================

def rank_predictions(preds: pd.DataFrame, ranking_field: str) -> pd.DataFrame:
    """
    Rank predictions by the specified field.

    Adds 'pred_rank' column (1 = highest ranked).
    Returns sorted DataFrame.
    """
    if ranking_field not in preds.columns:
        # Fallback to projected_ppg if field doesn't exist
        ranking_field = 'projected_ppg'

    # Sort descending by ranking field
    ranked = preds.sort_values(ranking_field, ascending=False).reset_index(drop=True)

    # Assign ranks (1-indexed)
    ranked['pred_rank'] = range(1, len(ranked) + 1)

    return ranked


def select_picked3(preds_ranked: pd.DataFrame) -> List[Tuple]:
    """
    Select top 3 picks from ranked predictions.

    Returns list of (player_id, player_name) tuples.
    """
    if len(preds_ranked) < 3:
        return [(row['player_id'], row['player_name']) for _, row in preds_ranked.iterrows()]

    top3 = preds_ranked.head(3)
    return [(row['player_id'], row['player_name']) for _, row in top3.iterrows()]


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_overlap(
    picked3: List[Tuple],
    actual_top3: Dict,
    actuals_df: pd.DataFrame
) -> Dict:
    """
    Calculate overlap metrics between picked and actual top 3.

    Returns dict with:
    - overlap: 0-3 count of correct picks
    - hit_any: 1 if overlap >= 1
    - hit_2plus: 1 if overlap >= 2
    - hit_exact: 1 if overlap == 3
    - hit_1: 1 if actual #1 is in our picks
    - tie_friendly_overlap: overlap counting ties at #3
    """
    if not actual_top3['top3']:
        return {
            'overlap': 0,
            'hit_any': 0,
            'hit_2plus': 0,
            'hit_exact': 0,
            'hit_1': 0,
            'tie_friendly_overlap': 0
        }

    picked_ids = set(p[0] for p in picked3)
    actual_ids = set(a[0] for a in actual_top3['top3'])

    # Basic overlap
    overlap = len(picked_ids.intersection(actual_ids))

    # Actual #1 scorer
    actual_1_id = actual_top3['top3'][0][0]
    hit_1 = 1 if actual_1_id in picked_ids else 0

    # Tie-friendly overlap: count picks that scored >= actual_3rd_points
    actual_3rd_points = actual_top3['actual_3rd_points']
    tie_friendly = 0
    for player_id, _ in picked3:
        player_row = actuals_df[actuals_df['player_id'] == player_id]
        if not player_row.empty:
            if player_row.iloc[0]['actual_ppg'] >= actual_3rd_points:
                tie_friendly += 1

    return {
        'overlap': overlap,
        'hit_any': 1 if overlap >= 1 else 0,
        'hit_2plus': 1 if overlap >= 2 else 0,
        'hit_exact': 1 if overlap == 3 else 0,
        'hit_1': hit_1,
        'tie_friendly_overlap': tie_friendly
    }


def score_ranking(
    preds_ranked: pd.DataFrame,
    actual_top3: Dict
) -> Dict:
    """
    Calculate ranking quality metrics.

    Returns dict with:
    - pred_rank_a1: our rank for actual #1 scorer
    - pred_rank_a2: our rank for actual #2 scorer
    - pred_rank_a3: our rank for actual #3 scorer
    - avg_rank_actual_top3: mean of above
    - best_rank_actual_top3: min of above
    """
    if not actual_top3['top3']:
        return {
            'pred_rank_a1': None,
            'pred_rank_a2': None,
            'pred_rank_a3': None,
            'avg_rank_actual_top3': None,
            'best_rank_actual_top3': None
        }

    # Build rank lookup
    rank_lookup = {
        row['player_id']: row['pred_rank']
        for _, row in preds_ranked.iterrows()
    }

    n_players = len(preds_ranked)
    ranks = []

    for i, (player_id, _, _) in enumerate(actual_top3['top3']):
        # If player not in predictions, assign rank = n_players + 1
        rank = rank_lookup.get(player_id, n_players + 1)
        ranks.append(rank)

    pred_rank_a1 = ranks[0] if len(ranks) > 0 else None
    pred_rank_a2 = ranks[1] if len(ranks) > 1 else None
    pred_rank_a3 = ranks[2] if len(ranks) > 2 else None

    valid_ranks = [r for r in ranks if r is not None]

    return {
        'pred_rank_a1': pred_rank_a1,
        'pred_rank_a2': pred_rank_a2,
        'pred_rank_a3': pred_rank_a3,
        'avg_rank_actual_top3': np.mean(valid_ranks) if valid_ranks else None,
        'best_rank_actual_top3': min(valid_ranks) if valid_ranks else None
    }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def backtest_date(
    conn: sqlite3.Connection,
    game_date: str,
    strategy: str
) -> Optional[Dict]:
    """
    Run backtest for a single date and strategy.

    Returns complete result row dict, or None if insufficient data.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGIES.keys())}")

    ranking_field = STRATEGIES[strategy]['ranking_field']

    # Load data
    preds = load_predictions(conn, game_date)
    actuals = load_actuals(conn, game_date)

    if preds.empty or actuals.empty or len(actuals) < 3:
        return None

    # Compute actual top 3
    actual_top3 = compute_actual_top3(actuals)

    if not actual_top3['top3']:
        return None

    # Rank predictions and select picks
    preds_ranked = rank_predictions(preds, ranking_field)
    picked3 = select_picked3(preds_ranked)

    # Score
    overlap_metrics = score_overlap(picked3, actual_top3, actuals)
    ranking_metrics = score_ranking(preds_ranked, actual_top3)

    # Build actual finish rank lookup (1st, 2nd, ... by actual points)
    actuals_sorted = break_ties(actuals)
    actuals_sorted['finish_rank'] = range(1, len(actuals_sorted) + 1)
    finish_rank_lookup = dict(zip(actuals_sorted['player_id'], actuals_sorted['finish_rank']))
    actual_pts_lookup = dict(zip(actuals_sorted['player_id'], actuals_sorted['actual_ppg']))

    # Build pred rank lookup
    pred_rank_lookup = dict(zip(preds_ranked['player_id'], preds_ranked['pred_rank']))

    # Compute diagnostic metrics for each pick
    pick_diagnostics = []
    pick_points = []
    for player_id, player_name in picked3:
        pts = actual_pts_lookup.get(player_id, 0)
        finish = finish_rank_lookup.get(player_id, len(actuals) + 1)
        pred = pred_rank_lookup.get(player_id, len(preds) + 1)
        pick_diagnostics.append({
            'pts': pts,
            'finish': finish,
            'pred_rank': pred
        })
        pick_points.append(pts)

    # Closest miss: how far short was our best pick from the #3 threshold?
    actual_3rd = actual_top3['actual_3rd_points']
    closest_miss = max(pick_points) - actual_3rd if pick_points else -999

    # Build result row
    result = {
        'slate_date': game_date,
        'strategy_name': strategy,

        # Our picks with diagnostics
        'picked1_id': picked3[0][0] if len(picked3) > 0 else None,
        'picked1_name': picked3[0][1] if len(picked3) > 0 else None,
        'picked1_pts': pick_diagnostics[0]['pts'] if len(pick_diagnostics) > 0 else None,
        'picked1_finish': pick_diagnostics[0]['finish'] if len(pick_diagnostics) > 0 else None,
        'picked1_pred_rank': pick_diagnostics[0]['pred_rank'] if len(pick_diagnostics) > 0 else None,

        'picked2_id': picked3[1][0] if len(picked3) > 1 else None,
        'picked2_name': picked3[1][1] if len(picked3) > 1 else None,
        'picked2_pts': pick_diagnostics[1]['pts'] if len(pick_diagnostics) > 1 else None,
        'picked2_finish': pick_diagnostics[1]['finish'] if len(pick_diagnostics) > 1 else None,
        'picked2_pred_rank': pick_diagnostics[1]['pred_rank'] if len(pick_diagnostics) > 1 else None,

        'picked3_id': picked3[2][0] if len(picked3) > 2 else None,
        'picked3_name': picked3[2][1] if len(picked3) > 2 else None,
        'picked3_pts': pick_diagnostics[2]['pts'] if len(pick_diagnostics) > 2 else None,
        'picked3_finish': pick_diagnostics[2]['finish'] if len(pick_diagnostics) > 2 else None,
        'picked3_pred_rank': pick_diagnostics[2]['pred_rank'] if len(pick_diagnostics) > 2 else None,

        # Actual top 3 with our predicted ranks
        'actual1_id': actual_top3['top3'][0][0],
        'actual1_name': actual_top3['top3'][0][1],
        'actual1_points': actual_top3['top3'][0][2],
        'actual2_id': actual_top3['top3'][1][0],
        'actual2_name': actual_top3['top3'][1][1],
        'actual2_points': actual_top3['top3'][1][2],
        'actual3_id': actual_top3['top3'][2][0],
        'actual3_name': actual_top3['top3'][2][1],
        'actual3_points': actual_top3['top3'][2][2],

        # Overlap metrics
        **overlap_metrics,

        # Ranking metrics
        **ranking_metrics,

        # Closeness metrics
        'closest_miss': closest_miss,

        # Context
        'n_pred_players': len(preds),
        'actual_3rd_points': actual_top3['actual_3rd_points'],
        'ties_at_3rd': 1 if actual_top3['ties_at_3rd'] else 0
    }

    return result


def compute_baseline_overlap(conn: sqlite3.Connection, game_date: str) -> Dict:
    """
    Compute baseline overlap for projection_only and ceiling_only strategies.

    Returns dict with baseline overlaps to compare against the main strategy.
    """
    baselines = {}

    for baseline_strategy in ['projection_only', 'ceiling_only']:
        result = backtest_date(conn, game_date, baseline_strategy)
        if result:
            baselines[baseline_strategy] = {
                'overlap': result['overlap'],
                'tie_friendly_overlap': result['tie_friendly_overlap'],
                'pred_rank_a1': result['pred_rank_a1']
            }
        else:
            baselines[baseline_strategy] = {
                'overlap': None,
                'tie_friendly_overlap': None,
                'pred_rank_a1': None
            }

    return baselines


def get_drilldown_context(conn: sqlite3.Connection, game_date: str, strategy: str, top_n: int = 15) -> Dict:
    """
    Get detailed context for drilldown analysis of a specific date.

    Returns:
    - our_top_ranked: Our top N ranked players with their predictions
    - actual_top_scorers: Actual top N scorers with our predicted ranks
    - slate_stats: Overall slate statistics
    """
    if strategy not in STRATEGIES:
        strategy = 'projection_only'

    ranking_field = STRATEGIES[strategy]['ranking_field']

    # Load data
    preds = load_predictions(conn, game_date)
    actuals = load_actuals(conn, game_date)

    if preds.empty or actuals.empty:
        return {'our_top_ranked': [], 'actual_top_scorers': [], 'slate_stats': {}}

    # Rank predictions
    preds_ranked = rank_predictions(preds, ranking_field)

    # Build finish rank lookup
    actuals_sorted = break_ties(actuals)
    actuals_sorted['finish_rank'] = range(1, len(actuals_sorted) + 1)
    finish_rank_lookup = dict(zip(actuals_sorted['player_id'], actuals_sorted['finish_rank']))
    actual_pts_lookup = dict(zip(actuals_sorted['player_id'], actuals_sorted['actual_ppg']))

    # Our top N ranked players
    our_top = []
    for _, row in preds_ranked.head(top_n).iterrows():
        player_id = row['player_id']
        our_top.append({
            'rank': int(row['pred_rank']),
            'name': row['player_name'],
            'proj_ppg': row.get('projected_ppg', 0),
            'ceiling': row.get('proj_ceiling', 0),
            'actual_pts': actual_pts_lookup.get(player_id, 0),
            'finish_rank': finish_rank_lookup.get(player_id, 999),
            'p_top3': row.get('p_top3', 0),
            'p_top1': row.get('p_top1', 0),
        })

    # Build pred rank lookup
    pred_rank_lookup = dict(zip(preds_ranked['player_id'], preds_ranked['pred_rank']))

    # Actual top N scorers
    actual_top = []
    for _, row in actuals_sorted.head(top_n).iterrows():
        player_id = row['player_id']
        pred_row = preds_ranked[preds_ranked['player_id'] == player_id]
        actual_top.append({
            'finish_rank': int(row['finish_rank']),
            'name': row['player_name'],
            'actual_pts': row['actual_ppg'],
            'our_pred_rank': int(pred_rank_lookup.get(player_id, len(preds) + 1)),
            'proj_ppg': pred_row['projected_ppg'].iloc[0] if not pred_row.empty else 0,
            'ceiling': pred_row['proj_ceiling'].iloc[0] if not pred_row.empty else 0,
        })

    # Slate stats
    slate_stats = {
        'total_players': len(actuals),
        'avg_points': actuals['actual_ppg'].mean(),
        'max_points': actuals['actual_ppg'].max(),
        'top3_threshold': actuals_sorted.iloc[2]['actual_ppg'] if len(actuals_sorted) >= 3 else 0,
    }

    return {
        'our_top_ranked': our_top,
        'actual_top_scorers': actual_top,
        'slate_stats': slate_stats
    }


def store_result(conn: sqlite3.Connection, result: Dict) -> None:
    """Store a backtest result row in the database."""
    cursor = conn.cursor()

    # Use INSERT OR REPLACE to handle UNIQUE constraint
    columns = list(result.keys())
    placeholders = ', '.join(['?' for _ in columns])
    col_names = ', '.join(columns)

    cursor.execute(f"""
        INSERT OR REPLACE INTO backtest_daily_results ({col_names})
        VALUES ({placeholders})
    """, list(result.values()))

    conn.commit()


def run_backtest(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    strategy: str,
    store_results: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run backtest over a date range for a single strategy.

    Returns DataFrame with all daily results.
    """
    # Get available dates in range
    available_dates = get_available_dates(conn)
    dates_in_range = [
        d for d in available_dates
        if date_from <= d <= date_to
    ]

    if verbose:
        print(f"Running backtest for strategy '{strategy}'")
        print(f"Date range: {date_from} to {date_to}")
        print(f"Dates with data: {len(dates_in_range)}")

    results = []

    for game_date in sorted(dates_in_range):
        result = backtest_date(conn, game_date, strategy)

        if result:
            if store_results:
                store_result(conn, result)
            results.append(result)

            if verbose:
                print(f"  {game_date}: overlap={result['overlap']}/3, "
                      f"hit_1={result['hit_1']}, rank_a1={result['pred_rank_a1']}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def run_compare_all(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    store_results: bool = True,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run backtest for all strategies and return comparison.
    """
    all_results = {}

    for strategy in STRATEGIES:
        if verbose:
            print(f"\n{'='*60}")
        results_df = run_backtest(
            conn, date_from, date_to, strategy,
            store_results=store_results, verbose=verbose
        )
        all_results[strategy] = results_df

    return all_results


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary_stats(results_df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics across all backtest dates.
    """
    if results_df.empty:
        return {
            'n_slates': 0,
            'hit_1_rate': 0,
            'hit_any_rate': 0,
            'hit_2plus_rate': 0,
            'hit_exact_rate': 0,
            'avg_overlap': 0,
            'avg_rank_a1': None,
            'avg_rank_top3': None
        }

    n = len(results_df)

    return {
        'n_slates': n,
        'hit_1_rate': results_df['hit_1'].mean(),
        'hit_any_rate': results_df['hit_any'].mean(),
        'hit_2plus_rate': results_df['hit_2plus'].mean(),
        'hit_exact_rate': results_df['hit_exact'].mean(),
        'avg_overlap': results_df['overlap'].mean(),
        'avg_rank_a1': results_df['pred_rank_a1'].mean(),
        'avg_rank_top3': results_df['avg_rank_actual_top3'].mean()
    }


def print_summary(strategy: str, stats: Dict) -> None:
    """Print formatted summary statistics."""
    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY: {strategy}")
    print(f"{'='*60}")
    print(f"Slates analyzed:     {stats['n_slates']}")
    print(f"Hit #1 rate:         {stats['hit_1_rate']*100:.1f}%")
    print(f"Hit any rate:        {stats['hit_any_rate']*100:.1f}%")
    print(f"Hit 2+ rate:         {stats['hit_2plus_rate']*100:.1f}%")
    print(f"Hit exact (3/3):     {stats['hit_exact_rate']*100:.1f}%")
    print(f"Avg overlap:         {stats['avg_overlap']:.2f}/3")
    print(f"Avg rank of #1:      {stats['avg_rank_a1']:.1f}" if stats['avg_rank_a1'] else "Avg rank of #1:      N/A")
    print(f"Avg rank of top 3:   {stats['avg_rank_top3']:.1f}" if stats['avg_rank_top3'] else "Avg rank of top 3:   N/A")


def print_comparison(all_stats: Dict[str, Dict]) -> None:
    """Print comparison table of all strategies."""
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")

    # Header
    print(f"{'Strategy':<20} {'Slates':>8} {'Hit#1':>8} {'HitAny':>8} {'Overlap':>10} {'Rank#1':>8}")
    print("-" * 80)

    # Sort by avg_overlap descending
    sorted_strategies = sorted(
        all_stats.items(),
        key=lambda x: x[1].get('avg_overlap', 0),
        reverse=True
    )

    for strategy, stats in sorted_strategies:
        hit1 = f"{stats['hit_1_rate']*100:.1f}%" if stats['n_slates'] > 0 else "N/A"
        hitany = f"{stats['hit_any_rate']*100:.1f}%" if stats['n_slates'] > 0 else "N/A"
        overlap = f"{stats['avg_overlap']:.2f}/3" if stats['n_slates'] > 0 else "N/A"
        rank1 = f"{stats['avg_rank_a1']:.1f}" if stats.get('avg_rank_a1') else "N/A"

        print(f"{strategy:<20} {stats['n_slates']:>8} {hit1:>8} {hitany:>8} {overlap:>10} {rank1:>8}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Top-3 Scorer Backtest Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_top3.py --days 30 --strategy sim_p_top3
  python backtest_top3.py --from 2026-01-01 --to 2026-01-15 --strategy projection_only
  python backtest_top3.py --days 30 --compare-all
        """
    )

    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest (default: 30)')
    parser.add_argument('--from', dest='date_from', type=str,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', dest='date_to', type=str,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, default='sim_p_top3',
                        choices=list(STRATEGIES.keys()),
                        help='Ranking strategy to evaluate')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all strategies')
    parser.add_argument('--output', type=str,
                        help='Output CSV file path')
    parser.add_argument('--db', type=str,
                        default='nba_stats.db',
                        help='Database file path')
    parser.add_argument('--no-store', action='store_true',
                        help='Do not store results in database')

    args = parser.parse_args()

    # Determine date range
    if args.date_to:
        date_to = args.date_to
    else:
        date_to = datetime.now().strftime('%Y-%m-%d')

    if args.date_from:
        date_from = args.date_from
    else:
        date_from = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        # Try in script directory
        script_dir = Path(__file__).parent
        db_path = script_dir / 'nba_daily.db'

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1

    # Connect and ensure table exists
    conn = sqlite3.connect(str(db_path))
    create_backtest_table(conn)

    store_results = not args.no_store

    if args.compare_all:
        # Run all strategies
        all_results = run_compare_all(conn, date_from, date_to,
                                      store_results=store_results)

        # Compute and display comparison
        all_stats = {
            strategy: compute_summary_stats(df)
            for strategy, df in all_results.items()
        }
        print_comparison(all_stats)

        # Save to CSV if requested
        if args.output:
            # Combine all results
            combined = pd.concat(all_results.values(), ignore_index=True)
            combined.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

    else:
        # Single strategy
        results_df = run_backtest(conn, date_from, date_to, args.strategy,
                                  store_results=store_results)

        stats = compute_summary_stats(results_df)
        print_summary(args.strategy, stats)

        # Save to CSV if requested
        if args.output and not results_df.empty:
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

    conn.close()
    return 0


if __name__ == '__main__':
    exit(main())
