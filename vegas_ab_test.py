#!/usr/bin/env python3
"""
Vegas A/B Test for Tournament Strategy

Tests whether incorporating Vegas team totals improves our ability to
identify top-3 scorers and win tournament lineups.

Methodology:
1. Split historical dates into test set
2. Generate rankings/lineups WITH and WITHOUT Vegas inputs
3. Compare hit rates and portfolio win rates
4. Calculate statistical significance via bootstrap confidence intervals

Usage:
    python vegas_ab_test.py --days 60
    python vegas_ab_test.py --from 2025-12-01 --to 2026-01-15
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path
from dataclasses import dataclass
import warnings


@dataclass
class ABTestResult:
    """Results from a single A/B comparison."""
    metric_name: str
    with_vegas: float
    without_vegas: float
    delta: float
    delta_pct: float
    n_samples: int
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    significant: bool  # Is the CI entirely positive or negative?


@dataclass
class VegasABTestSummary:
    """Complete A/B test summary."""
    test_dates: List[str]
    n_dates: int
    results: Dict[str, ABTestResult]
    recommendation: str
    details: Dict


# =============================================================================
# DATA LOADING
# =============================================================================

def get_available_test_dates(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str
) -> List[str]:
    """Get dates that have both predictions and actuals."""
    query = """
        SELECT DISTINCT game_date
        FROM predictions
        WHERE actual_ppg IS NOT NULL
          AND projected_ppg IS NOT NULL
          AND game_date BETWEEN ? AND ?
        ORDER BY game_date
    """
    df = pd.read_sql_query(query, conn, params=[date_from, date_to])
    return df['game_date'].tolist()


def get_predictions_for_date(conn: sqlite3.Connection, game_date: str) -> pd.DataFrame:
    """Load predictions with Vegas data for a specific date."""
    query = """
        SELECT
            p.player_id,
            p.player_name,
            p.team_name,
            p.opponent_name,
            p.game_id,
            p.projected_ppg,
            p.proj_ceiling,
            p.proj_floor,
            p.p_top1,
            p.p_top3,
            p.season_avg_ppg,
            p.actual_ppg,
            v.over_under as vegas_total,
            v.spread as vegas_spread,
            v.home_implied_total,
            v.away_implied_total
        FROM predictions p
        LEFT JOIN vegas_odds v ON p.game_id = v.game_id
        WHERE p.game_date = ?
          AND p.projected_ppg IS NOT NULL
        ORDER BY p.projected_ppg DESC
    """
    return pd.read_sql_query(query, conn, params=[game_date])


def get_actuals_for_date(conn: sqlite3.Connection, game_date: str) -> pd.DataFrame:
    """Load actual scores from player_game_logs."""
    query = """
        SELECT
            player_id,
            player_name,
            points as actual_ppg,
            minutes
        FROM player_game_logs
        WHERE game_date = ?
          AND points IS NOT NULL
          AND minutes > 0
        ORDER BY points DESC
    """
    return pd.read_sql_query(query, conn, params=[game_date])


# =============================================================================
# RANKING FUNCTIONS (With/Without Vegas)
# =============================================================================

def rank_without_vegas(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Rank players using only our projection model (no Vegas).

    Uses p_top3 if available, otherwise projected_ppg.
    """
    df = predictions.copy()

    # Use p_top3 if available and valid
    if 'p_top3' in df.columns and df['p_top3'].notna().any():
        df['rank_score'] = df['p_top3'].fillna(0)
    else:
        # Fallback to projected_ppg
        df['rank_score'] = df['projected_ppg'].fillna(0)

    df = df.sort_values('rank_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


def rank_with_vegas(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Rank players using projection model + Vegas game environment.

    Vegas boost formula:
    - Higher game total = higher scoring environment = boost
    - Close spread = more competitive = more playing time for stars

    Adjusted score = base_score * vegas_factor
    """
    df = predictions.copy()

    # Calculate base score
    if 'p_top3' in df.columns and df['p_top3'].notna().any():
        df['base_score'] = df['p_top3'].fillna(0)
    else:
        df['base_score'] = df['projected_ppg'].fillna(0)

    # Vegas adjustment factor
    df['vegas_factor'] = 1.0  # Default

    # Game total boost (higher total = more scoring opportunity)
    # Average NBA game total is ~225. Boost games > 230, penalize < 220.
    if 'vegas_total' in df.columns:
        df['vegas_total'] = df['vegas_total'].fillna(225)
        # Scale: +/- 5% for every 10 points above/below 225
        df['total_boost'] = (df['vegas_total'] - 225) / 200  # ~2.5% per 5 pts

    else:
        df['total_boost'] = 0.0

    # Spread factor (close games = stars play more minutes)
    # Blowout penalty: if spread > 10, reduce expectations slightly
    if 'vegas_spread' in df.columns:
        df['vegas_spread'] = df['vegas_spread'].fillna(0).abs()
        # Slight penalty for blowouts, bonus for close games
        df['spread_factor'] = np.where(
            df['vegas_spread'] > 10,
            -0.02,  # -2% for potential blowouts
            np.where(
                df['vegas_spread'] < 4,
                0.01,  # +1% for close games (more clutch scoring)
                0.0
            )
        )
    else:
        df['spread_factor'] = 0.0

    # Apply Vegas factor
    df['vegas_factor'] = 1.0 + df['total_boost'] + df['spread_factor']
    df['rank_score'] = df['base_score'] * df['vegas_factor']

    # Sort and rank
    df = df.sort_values('rank_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_ranking(
    ranking: pd.DataFrame,
    actuals: pd.DataFrame,
    top_n: int = 3
) -> Dict:
    """
    Evaluate a ranking against actual results.

    Metrics:
    - hit_1: Did we pick the #1 scorer?
    - hit_any: Did any of our top-3 picks finish in actual top-3?
    - hit_2plus: Did 2+ of our picks finish in actual top-3?
    - hit_exact: Did all 3 picks match actual top-3?
    - avg_rank_of_picks: Average actual rank of our picks
    """
    # Get actual top 3
    actual_top3 = set(actuals.head(3)['player_id'].tolist())
    actual_top1 = actuals.iloc[0]['player_id'] if len(actuals) > 0 else None

    # Get our picks
    our_picks = set(ranking.head(top_n)['player_id'].tolist())
    our_pick_1 = ranking.iloc[0]['player_id'] if len(ranking) > 0 else None

    # Calculate overlap
    overlap = len(our_picks & actual_top3)

    # Build actual rank lookup
    actuals_ranked = actuals.copy().reset_index(drop=True)
    actuals_ranked['actual_rank'] = range(1, len(actuals_ranked) + 1)
    rank_lookup = dict(zip(actuals_ranked['player_id'], actuals_ranked['actual_rank']))

    # Get ranks of our picks
    pick_ranks = [rank_lookup.get(pid, len(actuals) + 1) for pid in our_picks]

    return {
        'hit_1': 1 if our_pick_1 == actual_top1 else 0,
        'hit_any': 1 if overlap >= 1 else 0,
        'hit_2plus': 1 if overlap >= 2 else 0,
        'hit_exact': 1 if overlap == 3 else 0,
        'overlap_count': overlap,
        'avg_rank_of_picks': np.mean(pick_ranks) if pick_ranks else 999,
        'best_pick_rank': min(pick_ranks) if pick_ranks else 999,
    }


def evaluate_portfolio_sum(
    ranking: pd.DataFrame,
    actuals: pd.DataFrame,
    n_lineups: int = 5
) -> Dict:
    """
    Evaluate lineup sum performance (tournament win condition).

    Creates simple lineups from top N ranked players and checks
    if any lineup achieves the optimal 3-player sum.
    """
    # Build lineups from top ranked players
    top_players = ranking.head(n_lineups * 3)['player_id'].tolist()

    if len(top_players) < 3:
        return {'won_slate': 0, 'best_sum': 0, 'optimal_sum': 0, 'shortfall': 999}

    # Create simple lineups (top 3, next 3, etc.)
    lineups = []
    for i in range(0, min(len(top_players), n_lineups * 3), 3):
        if i + 3 <= len(top_players):
            lineups.append(top_players[i:i+3])

    if not lineups:
        return {'won_slate': 0, 'best_sum': 0, 'optimal_sum': 0, 'shortfall': 999}

    # Build actual scores lookup
    score_lookup = dict(zip(actuals['player_id'], actuals['actual_ppg']))

    # Calculate each lineup's sum
    best_sum = 0
    for lineup in lineups:
        lineup_sum = sum(score_lookup.get(pid, 0) for pid in lineup)
        best_sum = max(best_sum, lineup_sum)

    # Calculate optimal sum (top 3 actual scorers)
    optimal_sum = actuals.head(3)['actual_ppg'].sum()

    return {
        'won_slate': 1 if best_sum >= optimal_sum - 0.01 else 0,
        'best_sum': best_sum,
        'optimal_sum': optimal_sum,
        'shortfall': optimal_sum - best_sum,
    }


# =============================================================================
# A/B TEST EXECUTION
# =============================================================================

def run_ab_test_for_date(
    conn: sqlite3.Connection,
    game_date: str
) -> Optional[Dict]:
    """Run A/B test for a single date."""
    predictions = get_predictions_for_date(conn, game_date)
    actuals = get_actuals_for_date(conn, game_date)

    if predictions.empty or actuals.empty or len(predictions) < 10:
        return None

    # Rank with and without Vegas
    ranking_no_vegas = rank_without_vegas(predictions)
    ranking_with_vegas = rank_with_vegas(predictions)

    # Evaluate both
    eval_no_vegas = evaluate_ranking(ranking_no_vegas, actuals)
    eval_with_vegas = evaluate_ranking(ranking_with_vegas, actuals)

    # Portfolio evaluation
    portfolio_no_vegas = evaluate_portfolio_sum(ranking_no_vegas, actuals)
    portfolio_with_vegas = evaluate_portfolio_sum(ranking_with_vegas, actuals)

    return {
        'game_date': game_date,
        'no_vegas': {**eval_no_vegas, **portfolio_no_vegas},
        'with_vegas': {**eval_with_vegas, **portfolio_with_vegas},
    }


def run_full_ab_test(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    verbose: bool = True
) -> VegasABTestSummary:
    """
    Run complete A/B test across date range.

    Returns summary with all metrics and statistical significance.
    """
    dates = get_available_test_dates(conn, date_from, date_to)

    if verbose:
        print(f"Running Vegas A/B Test for {len(dates)} dates...")

    # Collect results
    all_results = []
    for i, game_date in enumerate(dates, 1):
        if verbose and i % 10 == 0:
            print(f"  [{i}/{len(dates)}] Processing {game_date}...")

        result = run_ab_test_for_date(conn, game_date)
        if result:
            all_results.append(result)

    if not all_results:
        return VegasABTestSummary(
            test_dates=[],
            n_dates=0,
            results={},
            recommendation="INSUFFICIENT DATA",
            details={'error': 'No valid test dates found'}
        )

    # Aggregate metrics
    metrics = ['hit_1', 'hit_any', 'hit_2plus', 'hit_exact', 'won_slate']
    ab_results = {}

    for metric in metrics:
        no_vegas_values = [r['no_vegas'][metric] for r in all_results]
        with_vegas_values = [r['with_vegas'][metric] for r in all_results]

        no_vegas_rate = np.mean(no_vegas_values)
        with_vegas_rate = np.mean(with_vegas_values)
        delta = with_vegas_rate - no_vegas_rate

        # Bootstrap confidence interval for delta
        ci_lower, ci_upper = bootstrap_ci(
            no_vegas_values, with_vegas_values, n_bootstrap=1000
        )

        significant = (ci_lower > 0) or (ci_upper < 0)
        delta_pct = (delta / no_vegas_rate * 100) if no_vegas_rate > 0 else 0

        ab_results[metric] = ABTestResult(
            metric_name=metric,
            with_vegas=with_vegas_rate,
            without_vegas=no_vegas_rate,
            delta=delta,
            delta_pct=delta_pct,
            n_samples=len(all_results),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=significant
        )

    # Determine recommendation
    hit_any_result = ab_results['hit_any']
    won_slate_result = ab_results['won_slate']

    if hit_any_result.significant and hit_any_result.delta > 0:
        recommendation = "VEGAS HELPS: Significant improvement in hit_any rate"
    elif won_slate_result.significant and won_slate_result.delta > 0:
        recommendation = "VEGAS HELPS: Significant improvement in portfolio wins"
    elif hit_any_result.delta > 0.02 or won_slate_result.delta > 0.02:
        recommendation = "VEGAS MAY HELP: Positive but not statistically significant"
    elif abs(hit_any_result.delta) < 0.01 and abs(won_slate_result.delta) < 0.01:
        recommendation = "VEGAS NEUTRAL: No meaningful difference detected"
    else:
        recommendation = "VEGAS UNCERTAIN: Mixed or negative results"

    return VegasABTestSummary(
        test_dates=[r['game_date'] for r in all_results],
        n_dates=len(all_results),
        results=ab_results,
        recommendation=recommendation,
        details={
            'avg_shortfall_no_vegas': np.mean([r['no_vegas']['shortfall'] for r in all_results]),
            'avg_shortfall_with_vegas': np.mean([r['with_vegas']['shortfall'] for r in all_results]),
        }
    )


def bootstrap_ci(
    no_vegas_values: List[float],
    with_vegas_values: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for the difference in means.

    Returns (ci_lower, ci_upper) for the delta (with_vegas - no_vegas).
    """
    n = len(no_vegas_values)
    deltas = []

    np.random.seed(42)

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        nv_sample = [no_vegas_values[i] for i in indices]
        wv_sample = [with_vegas_values[i] for i in indices]

        delta = np.mean(wv_sample) - np.mean(nv_sample)
        deltas.append(delta)

    alpha = 1 - ci_level
    ci_lower = np.percentile(deltas, alpha / 2 * 100)
    ci_upper = np.percentile(deltas, (1 - alpha / 2) * 100)

    return ci_lower, ci_upper


# =============================================================================
# REPORTING
# =============================================================================

def print_ab_test_report(summary: VegasABTestSummary) -> None:
    """Print formatted A/B test report."""
    print("\n" + "=" * 70)
    print("VEGAS A/B TEST RESULTS")
    print("=" * 70)

    print(f"\nTest dates: {summary.n_dates}")
    print(f"Recommendation: {summary.recommendation}")

    print(f"\n{'-'*70}")
    print(f"{'Metric':<15} {'No Vegas':<12} {'With Vegas':<12} {'Delta':<10} {'95% CI':<20} {'Sig?'}")
    print(f"{'-'*70}")

    for metric_name in ['hit_1', 'hit_any', 'hit_2plus', 'won_slate']:
        r = summary.results.get(metric_name)
        if r:
            ci_str = f"[{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]"
            sig_str = "YES *" if r.significant else "no"
            print(f"{r.metric_name:<15} {r.without_vegas:<12.1%} {r.with_vegas:<12.1%} "
                  f"{r.delta:+.1%}      {ci_str:<20} {sig_str}")

    print(f"\n{'-'*70}")
    print(f"Avg shortfall (no Vegas): {summary.details.get('avg_shortfall_no_vegas', 0):.1f} pts")
    print(f"Avg shortfall (with Vegas): {summary.details.get('avg_shortfall_with_vegas', 0):.1f} pts")


def save_ab_test_results(
    conn: sqlite3.Connection,
    summary: VegasABTestSummary
) -> None:
    """Save A/B test results to database."""
    cursor = conn.cursor()

    # Create table if needed
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vegas_ab_test_results (
            test_id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_date TEXT NOT NULL,
            n_test_dates INTEGER,
            metric_name TEXT,
            without_vegas REAL,
            with_vegas REAL,
            delta REAL,
            ci_lower REAL,
            ci_upper REAL,
            significant INTEGER,
            recommendation TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    test_date = str(datetime.now().date())

    for metric_name, result in summary.results.items():
        cursor.execute("""
            INSERT INTO vegas_ab_test_results
            (test_date, n_test_dates, metric_name, without_vegas, with_vegas,
             delta, ci_lower, ci_upper, significant, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            test_date,
            summary.n_dates,
            metric_name,
            result.without_vegas,
            result.with_vegas,
            result.delta,
            result.ci_lower,
            result.ci_upper,
            1 if result.significant else 0,
            summary.recommendation
        ])

    conn.commit()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Vegas A/B Test')
    parser.add_argument('--db', default='nba_daily.db', help='Database path')
    parser.add_argument('--days', type=int, help='Number of days to test')
    parser.add_argument('--from', dest='date_from', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', dest='date_to', help='End date (YYYY-MM-DD)')
    parser.add_argument('--save', action='store_true', help='Save results to database')

    args = parser.parse_args()

    # Determine database path
    db_path = Path(args.db)
    if not db_path.exists():
        project_dir = Path(__file__).parent
        db_path = project_dir / args.db

    if not db_path.exists():
        print(f"Database not found: {args.db}")
        return

    conn = sqlite3.connect(str(db_path))

    try:
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

        # Run test
        summary = run_full_ab_test(conn, date_from, date_to)

        # Print report
        print_ab_test_report(summary)

        # Save if requested
        if args.save:
            save_ab_test_results(conn, summary)
            print("\nResults saved to database.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
