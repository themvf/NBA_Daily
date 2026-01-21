#!/usr/bin/env python3
"""
Scenario Preset A/B Test for Tournament Strategy

Compares the performance of different scenario presets (BALANCED, SHOOTOUT,
CLOSE_GAMES, CHAOS, etc.) across historical slates to determine which strategies
work best under different conditions.

Key Questions Answered:
- Which preset has the highest overall win rate?
- Does SHOOTOUT outperform BALANCED on high-total slates?
- Does CHAOS work better on small slates?
- Which preset minimizes shortfall when it doesn't win?

Usage:
    python preset_ab_test.py --days 30
    python preset_ab_test.py --from 2025-12-01 --to 2026-01-15
    python preset_ab_test.py --summary
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path
from dataclasses import dataclass, field
import json
import warnings

# Project imports
from scenario_presets import PRESETS, ScenarioPreset, apply_scenario_preset
from lineup_optimizer import (
    TournamentLineupOptimizer,
    OptimizerConfig,
    create_player_pool_from_predictions
)
from backtest_portfolio import (
    get_actuals_for_date,
    get_predictions_for_date,
    evaluate_portfolio_against_actuals,
    compute_optimal_lineup
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PresetSlateResult:
    """Results for a single preset on a single slate."""
    preset_name: str
    slate_date: str
    won_slate: int
    sum_shortfall: float
    best_lineup_rank: Optional[int]
    hit_2_of_top3: int
    hit_all_top3: int
    unique_players: int
    estimated_win_prob: float
    # Slate context
    n_games: int
    n_players: int
    avg_total: float
    avg_spread: float


@dataclass
class PresetABResult:
    """Aggregated results for a preset across all test dates."""
    preset_name: str
    preset_icon: str
    n_slates: int
    wins: int
    win_rate: float
    avg_shortfall: float
    median_shortfall: float
    hit_2_of_top3_rate: float
    avg_unique_players: float
    avg_estimated_win_prob: float
    # By slate type
    win_rate_small_slate: Optional[float] = None   # <=4 games
    win_rate_large_slate: Optional[float] = None   # >7 games
    win_rate_high_total: Optional[float] = None    # avg total >= 230
    win_rate_tight_spread: Optional[float] = None  # avg spread <= 4


@dataclass
class PresetABTestSummary:
    """Complete A/B test summary across all presets."""
    test_dates: List[str]
    n_dates: int
    results: Dict[str, PresetABResult]
    best_overall: str
    best_small_slate: Optional[str] = None
    best_large_slate: Optional[str] = None
    best_high_total: Optional[str] = None
    best_tight_spread: Optional[str] = None


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

def create_preset_ab_tables(conn: sqlite3.Connection) -> None:
    """Create tables for preset A/B test results."""
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preset_ab_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            slate_date TEXT NOT NULL,
            preset_name TEXT NOT NULL,

            -- Portfolio metrics
            won_slate INTEGER,
            sum_shortfall REAL,
            best_lineup_rank INTEGER,
            hit_2_of_top3 INTEGER,
            hit_all_top3 INTEGER,
            unique_players INTEGER,
            estimated_win_prob REAL,

            -- Slate context
            n_games INTEGER,
            n_players INTEGER,
            avg_total REAL,
            avg_spread REAL,

            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(slate_date, preset_name)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_preset_ab_date
        ON preset_ab_results(slate_date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_preset_ab_preset
        ON preset_ab_results(preset_name)
    """)

    conn.commit()


def store_preset_slate_result(conn: sqlite3.Connection, result: PresetSlateResult) -> None:
    """Store a single preset/slate result."""
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO preset_ab_results (
            slate_date, preset_name,
            won_slate, sum_shortfall, best_lineup_rank,
            hit_2_of_top3, hit_all_top3, unique_players, estimated_win_prob,
            n_games, n_players, avg_total, avg_spread
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        result.slate_date, result.preset_name,
        result.won_slate, result.sum_shortfall, result.best_lineup_rank,
        result.hit_2_of_top3, result.hit_all_top3, result.unique_players,
        result.estimated_win_prob,
        result.n_games, result.n_players, result.avg_total, result.avg_spread
    ])

    conn.commit()


# =============================================================================
# SLATE CONTEXT COMPUTATION
# =============================================================================

def compute_slate_context(
    conn: sqlite3.Connection,
    game_date: str,
    predictions: pd.DataFrame
) -> Dict:
    """
    Compute slate context (avg total, avg spread, n_games) for categorization.

    Uses vegas_odds table if available, otherwise estimates from predictions.
    """
    context = {
        'n_games': 0,
        'n_players': len(predictions),
        'avg_total': 220.0,  # default
        'avg_spread': 5.0,   # default
    }

    # Get unique games
    game_ids = predictions['game_id'].dropna().unique()
    context['n_games'] = len(game_ids)

    if context['n_games'] == 0:
        return context

    # Try to get Vegas data
    try:
        query = """
            SELECT game_id, over_under, spread
            FROM vegas_odds
            WHERE game_id IN ({})
        """.format(','.join(['?' for _ in game_ids]))

        vegas_df = pd.read_sql_query(query, conn, params=list(game_ids))

        if not vegas_df.empty:
            context['avg_total'] = vegas_df['over_under'].mean()
            context['avg_spread'] = vegas_df['spread'].abs().mean()
            return context
    except Exception:
        pass

    # Fallback: estimate from projections
    # High-scoring slates tend to have higher average projected points
    avg_proj = predictions['projected_ppg'].mean()
    if avg_proj > 25:
        context['avg_total'] = 235.0
    elif avg_proj > 22:
        context['avg_total'] = 225.0
    else:
        context['avg_total'] = 215.0

    return context


def build_game_environments_from_context(
    predictions: pd.DataFrame,
    context: Dict
) -> Dict[str, dict]:
    """Build game environments dict for optimizer from slate context."""
    game_ids = predictions['game_id'].dropna().unique()

    # Use context to set default stack_score
    avg_total = context.get('avg_total', 220)
    avg_spread = context.get('avg_spread', 5)

    # Compute default stack score
    # Higher totals + lower spreads = better stacking
    total_factor = (avg_total - 210) / 30  # 0 at 210, 1 at 240
    spread_factor = (10 - avg_spread) / 10  # 1 at spread=0, 0 at spread=10
    default_stack_score = 0.5 + 0.25 * total_factor + 0.25 * spread_factor
    default_stack_score = max(0.3, min(0.9, default_stack_score))

    return {
        gid: {
            'stack_score': default_stack_score,
            'ot_probability': 0.06,
            'implied_total': avg_total,
            'spread': avg_spread
        }
        for gid in game_ids
    }


# =============================================================================
# PRESET EVALUATION
# =============================================================================

def evaluate_preset_on_slate(
    conn: sqlite3.Connection,
    game_date: str,
    preset_name: str,
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    context: Dict,
    compute_full_rank: bool = False
) -> Optional[PresetSlateResult]:
    """
    Evaluate a single preset on a single slate.

    Args:
        conn: Database connection
        game_date: Slate date
        preset_name: Name of preset to test
        predictions: Predictions DataFrame
        actuals: Actuals DataFrame
        context: Slate context dict
        compute_full_rank: Whether to compute full ranking (expensive)

    Returns:
        PresetSlateResult or None if error
    """
    try:
        # Get preset and apply to base config
        preset = PRESETS.get(preset_name)
        if preset is None:
            return None

        base_config = OptimizerConfig()
        adjusted_config = apply_scenario_preset(base_config, preset)

        # Build player pool
        player_pool = create_player_pool_from_predictions(predictions)
        if len(player_pool) < 10:
            return None

        # Build game environments
        game_envs = build_game_environments_from_context(predictions, context)

        # Run optimizer with adjusted config
        optimizer = TournamentLineupOptimizer(
            player_pool=player_pool,
            game_environments=game_envs,
            config=adjusted_config
        )

        result = optimizer.optimize()

        if not result.lineups:
            return None

        # Extract lineup player IDs
        lineups = [l.player_ids() for l in result.lineups]

        # Evaluate against actuals
        eval_result = evaluate_portfolio_against_actuals(
            lineups, actuals, compute_all_ranks=compute_full_rank
        )

        return PresetSlateResult(
            preset_name=preset_name,
            slate_date=game_date,
            won_slate=eval_result['won_slate'],
            sum_shortfall=eval_result['sum_shortfall'],
            best_lineup_rank=eval_result.get('best_lineup_rank'),
            hit_2_of_top3=eval_result['hit_2_of_top3_any'],
            hit_all_top3=eval_result['hit_all_top3_any'],
            unique_players=eval_result['unique_players_used'],
            estimated_win_prob=result.total_win_probability,
            n_games=context['n_games'],
            n_players=context['n_players'],
            avg_total=context['avg_total'],
            avg_spread=context['avg_spread']
        )

    except Exception as e:
        warnings.warn(f"Error evaluating {preset_name} on {game_date}: {e}")
        return None


def evaluate_all_presets_on_slate(
    conn: sqlite3.Connection,
    game_date: str,
    presets_to_test: List[str] = None,
    compute_full_rank: bool = False,
    verbose: bool = True
) -> List[PresetSlateResult]:
    """
    Evaluate all presets on a single slate.

    Returns list of PresetSlateResult for each preset.
    """
    if presets_to_test is None:
        presets_to_test = list(PRESETS.keys())

    # Load data once
    predictions = get_predictions_for_date(conn, game_date)
    if predictions.empty or len(predictions) < 10:
        return []

    actuals = get_actuals_for_date(conn, game_date)
    if actuals.empty or len(actuals) < 10:
        return []

    # Compute slate context
    context = compute_slate_context(conn, game_date, predictions)

    results = []
    for preset_name in presets_to_test:
        result = evaluate_preset_on_slate(
            conn, game_date, preset_name,
            predictions, actuals, context,
            compute_full_rank=compute_full_rank
        )
        if result:
            results.append(result)
            store_preset_slate_result(conn, result)

    return results


# =============================================================================
# A/B TEST RUNNER
# =============================================================================

def run_preset_ab_test(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    presets_to_test: List[str] = None,
    compute_full_rank: bool = False,
    verbose: bool = True
) -> PresetABTestSummary:
    """
    Run A/B test comparing scenario presets over a date range.

    Args:
        conn: Database connection
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        presets_to_test: List of preset names to test (default: all)
        compute_full_rank: Whether to compute full ranking
        verbose: Print progress

    Returns:
        PresetABTestSummary with aggregated results
    """
    # Ensure tables exist
    create_preset_ab_tables(conn)

    if presets_to_test is None:
        presets_to_test = list(PRESETS.keys())

    # Get available dates
    query = """
        SELECT DISTINCT game_date
        FROM predictions
        WHERE actual_ppg IS NOT NULL
          AND game_date BETWEEN ? AND ?
        ORDER BY game_date
    """
    dates_df = pd.read_sql_query(query, conn, params=[date_from, date_to])
    test_dates = dates_df['game_date'].tolist()

    if verbose:
        print(f"Running preset A/B test on {len(test_dates)} dates...")
        print(f"Presets: {', '.join(presets_to_test)}")
        print()

    # Collect all results
    all_results: List[PresetSlateResult] = []

    for i, game_date in enumerate(test_dates, 1):
        if verbose:
            print(f"[{i}/{len(test_dates)}] {game_date}...", end=' ')

        slate_results = evaluate_all_presets_on_slate(
            conn, game_date, presets_to_test,
            compute_full_rank=compute_full_rank,
            verbose=False
        )

        all_results.extend(slate_results)

        if verbose:
            if slate_results:
                winners = [r.preset_name for r in slate_results if r.won_slate]
                if winners:
                    print(f"Winners: {', '.join(winners)}")
                else:
                    shortfalls = {r.preset_name: r.sum_shortfall for r in slate_results}
                    best = min(shortfalls, key=shortfalls.get)
                    print(f"Best: {best} (shortfall: {shortfalls[best]:.1f})")
            else:
                print("SKIPPED")

    # Aggregate results by preset
    aggregated = aggregate_preset_results(all_results)

    # Determine best presets by category
    best_overall = max(aggregated.keys(), key=lambda k: aggregated[k].win_rate) if aggregated else None

    # Find best by slate type
    best_small = find_best_for_condition(
        aggregated, lambda r: r.win_rate_small_slate, "small slate")
    best_large = find_best_for_condition(
        aggregated, lambda r: r.win_rate_large_slate, "large slate")
    best_high_total = find_best_for_condition(
        aggregated, lambda r: r.win_rate_high_total, "high total")
    best_tight = find_best_for_condition(
        aggregated, lambda r: r.win_rate_tight_spread, "tight spread")

    summary = PresetABTestSummary(
        test_dates=test_dates,
        n_dates=len(test_dates),
        results=aggregated,
        best_overall=best_overall,
        best_small_slate=best_small,
        best_large_slate=best_large,
        best_high_total=best_high_total,
        best_tight_spread=best_tight
    )

    if verbose:
        print_ab_test_summary(summary)

    return summary


def aggregate_preset_results(results: List[PresetSlateResult]) -> Dict[str, PresetABResult]:
    """Aggregate per-slate results into per-preset summary."""
    if not results:
        return {}

    # Group by preset
    by_preset: Dict[str, List[PresetSlateResult]] = {}
    for r in results:
        if r.preset_name not in by_preset:
            by_preset[r.preset_name] = []
        by_preset[r.preset_name].append(r)

    aggregated = {}

    for preset_name, preset_results in by_preset.items():
        n = len(preset_results)
        wins = sum(r.won_slate for r in preset_results)
        shortfalls = [r.sum_shortfall for r in preset_results]

        # Get preset icon
        preset = PRESETS.get(preset_name)
        icon = preset.icon if preset else ''

        # Calculate by slate type
        small_results = [r for r in preset_results if r.n_games <= 4]
        large_results = [r for r in preset_results if r.n_games > 7]
        high_total_results = [r for r in preset_results if r.avg_total >= 230]
        tight_spread_results = [r for r in preset_results if r.avg_spread <= 4]

        aggregated[preset_name] = PresetABResult(
            preset_name=preset_name,
            preset_icon=icon,
            n_slates=n,
            wins=wins,
            win_rate=wins / n if n > 0 else 0,
            avg_shortfall=np.mean(shortfalls) if shortfalls else 0,
            median_shortfall=np.median(shortfalls) if shortfalls else 0,
            hit_2_of_top3_rate=np.mean([r.hit_2_of_top3 for r in preset_results]),
            avg_unique_players=np.mean([r.unique_players for r in preset_results]),
            avg_estimated_win_prob=np.mean([r.estimated_win_prob for r in preset_results]),
            win_rate_small_slate=(
                sum(r.won_slate for r in small_results) / len(small_results)
                if small_results else None
            ),
            win_rate_large_slate=(
                sum(r.won_slate for r in large_results) / len(large_results)
                if large_results else None
            ),
            win_rate_high_total=(
                sum(r.won_slate for r in high_total_results) / len(high_total_results)
                if high_total_results else None
            ),
            win_rate_tight_spread=(
                sum(r.won_slate for r in tight_spread_results) / len(tight_spread_results)
                if tight_spread_results else None
            ),
        )

    return aggregated


def find_best_for_condition(
    aggregated: Dict[str, PresetABResult],
    getter,
    condition_name: str
) -> Optional[str]:
    """Find best preset for a given condition."""
    valid = {k: getter(v) for k, v in aggregated.items() if getter(v) is not None}
    if not valid:
        return None
    return max(valid.keys(), key=lambda k: valid[k])


# =============================================================================
# REPORTING
# =============================================================================

def print_ab_test_summary(summary: PresetABTestSummary) -> None:
    """Print formatted A/B test summary."""
    print()
    print("=" * 70)
    print(f"PRESET A/B TEST SUMMARY ({summary.n_dates} slates)")
    print("=" * 70)

    # Sort by win rate
    sorted_presets = sorted(
        summary.results.items(),
        key=lambda x: x[1].win_rate,
        reverse=True
    )

    print()
    print(f"{'Preset':<15} {'Slates':<8} {'Wins':<6} {'Win%':<8} {'Avg Short':<10} {'Hit 2/3':<8}")
    print("-" * 70)

    for preset_name, result in sorted_presets:
        # Handle emojis safely for Windows console
        try:
            print(f"{preset_name:<15} {result.n_slates:<8} "
                  f"{result.wins:<6} {result.win_rate*100:>5.1f}%   "
                  f"{result.avg_shortfall:>8.1f}  {result.hit_2_of_top3_rate*100:>5.1f}%")
        except UnicodeEncodeError:
            print(f"{preset_name:<15} {result.n_slates:<8} "
                  f"{result.wins:<6} {result.win_rate*100:>5.1f}%   "
                  f"{result.avg_shortfall:>8.1f}  {result.hit_2_of_top3_rate*100:>5.1f}%")

    print()
    print("BY SLATE TYPE:")
    print("-" * 70)

    print(f"  Best overall:     {summary.best_overall or 'N/A'}")
    print(f"  Best small slate: {summary.best_small_slate or 'N/A'} (<=4 games)")
    print(f"  Best large slate: {summary.best_large_slate or 'N/A'} (>7 games)")
    print(f"  Best high total:  {summary.best_high_total or 'N/A'} (avg total >= 230)")
    print(f"  Best tight spread:{summary.best_tight_spread or 'N/A'} (avg spread <= 4)")

    print()


def get_stored_ab_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get summary from stored A/B test results."""
    query = """
        SELECT
            preset_name,
            COUNT(*) as n_slates,
            SUM(won_slate) as wins,
            AVG(won_slate) * 100 as win_rate_pct,
            AVG(sum_shortfall) as avg_shortfall,
            AVG(hit_2_of_top3) * 100 as hit_2_of_top3_pct,
            AVG(unique_players) as avg_unique_players
        FROM preset_ab_results
        GROUP BY preset_name
        ORDER BY win_rate_pct DESC
    """
    return pd.read_sql_query(query, conn)


def get_ab_results_by_slate_type(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get win rates by preset and slate type."""
    query = """
        SELECT
            preset_name,
            CASE
                WHEN n_games <= 4 THEN 'small'
                WHEN n_games > 7 THEN 'large'
                ELSE 'medium'
            END as slate_size,
            COUNT(*) as n_slates,
            AVG(won_slate) * 100 as win_rate_pct
        FROM preset_ab_results
        GROUP BY preset_name, slate_size
        ORDER BY preset_name, slate_size
    """
    return pd.read_sql_query(query, conn)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Scenario Preset A/B Test')
    parser.add_argument('--db', default='nba_stats.db', help='Database path')
    parser.add_argument('--days', type=int, help='Number of days back to test')
    parser.add_argument('--from', dest='date_from', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', dest='date_to', help='End date (YYYY-MM-DD)')
    parser.add_argument('--presets', nargs='+', help='Specific presets to test')
    parser.add_argument('--summary', action='store_true', help='Show stored summary only')
    parser.add_argument('--by-type', action='store_true', help='Show results by slate type')
    parser.add_argument('--full-rank', action='store_true', help='Compute full ranking (slow)')

    args = parser.parse_args()

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        project_dir = Path(__file__).parent
        db_path = project_dir / args.db

    if not db_path.exists():
        print(f"Database not found: {args.db}")
        return

    conn = sqlite3.connect(str(db_path))

    try:
        create_preset_ab_tables(conn)

        if args.summary:
            summary = get_stored_ab_summary(conn)
            print("\nPreset A/B Test Summary (from stored results):")
            print(summary.to_string(index=False))
            return

        if args.by_type:
            by_type = get_ab_results_by_slate_type(conn)
            print("\nWin Rates by Preset and Slate Type:")
            print(by_type.to_string(index=False))
            return

        # Determine date range
        if args.days:
            date_to = datetime.now().strftime('%Y-%m-%d')
            date_from = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        elif args.date_from and args.date_to:
            date_from = args.date_from
            date_to = args.date_to
        else:
            print("Please specify --days or --from/--to, or use --summary")
            return

        # Run A/B test
        summary = run_preset_ab_test(
            conn,
            date_from,
            date_to,
            presets_to_test=args.presets,
            compute_full_rank=args.full_rank
        )

        print(f"\nResults saved to database.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
