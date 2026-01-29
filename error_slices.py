#!/usr/bin/env python3
"""
Error Slices Analysis ("Where We Were Wrong")

Breaks down prediction errors by various buckets to identify
which assumptions are failing and where to focus improvements.

Features:
- Bootstrap confidence intervals for ΔMAE
- N threshold filtering (only trust buckets with sufficient sample)
- Significance flags ("real" if CI excludes 0)

Slices:
- Role tier (STAR / STARTER / ROTATION / BENCH)
- Spread bucket (0-3, 4-7, 8-12, 13+)
- Rest bucket (B2B, 1 day, 2 days, 3+ days)
- Interaction slices (Role x Spread, Role x Rest)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def bootstrap_mae_ci_stratified(
    slice_df: pd.DataFrame,
    baseline_mae: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Tuple[float, float, bool]:
    """
    Compute STRATIFIED bootstrap confidence interval for MAE difference vs baseline.

    IMPORTANT: Resamples by DATE (slate), not by individual rows.
    This accounts for within-slate correlation (players on same date are correlated).

    Args:
        slice_df: DataFrame with 'abs_error' and 'game_date' columns
        baseline_mae: Overall MAE to compare against
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 95%)

    Returns:
        (ci_lower, ci_upper, is_significant)
        is_significant is True if CI excludes 0
    """
    if len(slice_df) < 10:
        return (np.nan, np.nan, False)

    # Group errors by date
    if 'game_date' not in slice_df.columns:
        # Fallback to simple bootstrap if no date column
        return bootstrap_mae_ci_simple(slice_df['abs_error'].values, baseline_mae, n_bootstrap, ci_level)

    dates = slice_df['game_date'].unique()
    if len(dates) < 3:
        # Not enough dates for stratified bootstrap
        return bootstrap_mae_ci_simple(slice_df['abs_error'].values, baseline_mae, n_bootstrap, ci_level)

    # Group by date
    date_errors = {d: slice_df[slice_df['game_date'] == d]['abs_error'].values for d in dates}

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        # Resample DATES with replacement (stratified bootstrap)
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)

        # Collect all errors from sampled dates
        sampled_errors = []
        for d in sampled_dates:
            sampled_errors.extend(date_errors[d])

        if len(sampled_errors) > 0:
            sample_mae = np.mean(sampled_errors)
            bootstrap_deltas.append(sample_mae - baseline_mae)

    if not bootstrap_deltas:
        return (np.nan, np.nan, False)

    # Compute percentiles
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_deltas, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_deltas, (1 - alpha / 2) * 100)

    # Significant if CI excludes 0
    is_significant = (ci_lower > 0) or (ci_upper < 0)

    return (ci_lower, ci_upper, is_significant)


def bootstrap_mae_ci_simple(
    errors: np.ndarray,
    baseline_mae: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Tuple[float, float, bool]:
    """
    Simple (non-stratified) bootstrap for fallback cases.
    """
    if len(errors) < 10:
        return (np.nan, np.nan, False)

    rng = np.random.default_rng(42)
    bootstrap_deltas = []

    for _ in range(n_bootstrap):
        sample = rng.choice(errors, size=len(errors), replace=True)
        sample_mae = np.mean(sample)
        bootstrap_deltas.append(sample_mae - baseline_mae)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_deltas, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_deltas, (1 - alpha / 2) * 100)
    is_significant = (ci_lower > 0) or (ci_upper < 0)

    return (ci_lower, ci_upper, is_significant)


def compute_slice_stats(
    slice_df: pd.DataFrame,
    overall_mae: float,
    overall_n: int,
    slice_name: str,
    min_n_threshold: int = 50,
    min_pct_threshold: float = 2.0,
    min_dates_threshold: int = 5
) -> Optional[Dict]:
    """
    Compute statistics for a single slice with STRATIFIED confidence intervals.

    Uses date-stratified bootstrap to account for within-slate correlation.

    Args:
        slice_df: DataFrame for this slice (must have 'abs_error', 'game_date')
        overall_mae: Overall MAE baseline
        overall_n: Total sample size
        slice_name: Name of the slice
        min_n_threshold: Minimum N to be considered "trustworthy"
        min_pct_threshold: Minimum % of volume to be considered "trustworthy"
        min_dates_threshold: Minimum unique dates for trustworthy significance

    Returns:
        Dict with slice statistics, or None if insufficient data
    """
    n = len(slice_df)
    if n < 10:  # Absolute minimum to even show
        return None

    errors = slice_df['abs_error'].values
    mae = np.mean(errors)
    bias = slice_df['prediction_error'].mean()
    delta_mae = mae - overall_mae
    pct_volume = n / overall_n * 100

    # Count unique dates for this slice
    n_dates = slice_df['game_date'].nunique() if 'game_date' in slice_df.columns else 0

    # STRATIFIED Bootstrap CI for delta MAE (resamples by date)
    ci_lower, ci_upper, is_significant = bootstrap_mae_ci_stratified(slice_df, overall_mae)

    # Trustworthy if (N >= threshold OR pct >= threshold) AND enough dates
    has_volume = (n >= min_n_threshold) or (pct_volume >= min_pct_threshold)
    has_dates = n_dates >= min_dates_threshold
    is_trustworthy = has_volume and has_dates

    # Only flag as "real fix target" if significant AND trustworthy
    is_fix_target = is_significant and is_trustworthy and delta_mae > 0.3

    return {
        'slice': slice_name,
        'n': n,
        'n_dates': n_dates,
        'pct_volume': pct_volume,
        'mae': mae,
        'bias': bias,
        'delta_mae': delta_mae,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'is_significant': is_significant,
        'is_trustworthy': is_trustworthy,
        'is_fix_target': is_fix_target,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def calculate_error_slices(
    conn: sqlite3.Connection,
    days: int = 30,
    min_n_threshold: int = 50,
    min_pct_threshold: float = 2.0
) -> Dict:
    """
    Calculate MAE and other metrics broken down by various slices.

    Args:
        conn: Database connection
        days: Number of days to analyze
        min_n_threshold: Minimum N for a slice to be "trustworthy"
        min_pct_threshold: Minimum % volume for a slice to be "trustworthy"

    Returns:
        Dict with slice breakdowns including CIs and significance flags
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # Query enriched predictions
    query = """
        SELECT
            p.player_id,
            p.player_name,
            p.game_date,
            p.projected_ppg,
            p.actual_ppg,
            p.proj_minutes,
            p.actual_minutes,
            p.role_tier,
            p.game_script_tier,
            p.days_rest,
            p.is_b2b,
            p.blowout_risk,
            p.position_matchup_factor,
            ea.abs_error,
            ea.prediction_error,
            ea.was_top10
        FROM predictions p
        LEFT JOIN enrichment_audit_log ea
            ON p.player_id = ea.player_id AND date(p.game_date) = date(ea.game_date)
        WHERE p.actual_ppg IS NOT NULL
          AND date(p.game_date) >= date(?)
    """

    df = pd.read_sql_query(query, conn, params=[cutoff])

    if df.empty:
        return {'has_data': False, 'message': 'No predictions found'}

    # Calculate error if not already present
    if df['abs_error'].isna().all():
        df['prediction_error'] = df['actual_ppg'] - df['projected_ppg']
        df['abs_error'] = np.abs(df['prediction_error'])

    # Overall baseline
    overall_mae = df['abs_error'].mean()
    overall_bias = df['prediction_error'].mean()
    overall_n = len(df)

    results = {
        'has_data': True,
        'days_analyzed': days,
        'min_n_threshold': min_n_threshold,
        'min_pct_threshold': min_pct_threshold,
        'overall': {
            'n': overall_n,
            'mae': overall_mae,
            'bias': overall_bias,
        }
    }

    # =========================================================================
    # A. MAE by Role Tier
    # =========================================================================
    role_slices = []
    for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
        role_df = df[df['role_tier'] == role]
        stats = compute_slice_stats(role_df, overall_mae, overall_n, role,
                                    min_n_threshold, min_pct_threshold)
        if stats:
            role_slices.append(stats)

    results['by_role'] = sorted(role_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # B. MAE by Game Script (Spread Proxy)
    # =========================================================================
    script_slices = []
    script_labels = {
        'blowout': 'Blowout (13+ spread)',
        'neutral': 'Neutral (4-12 spread)',
        'close_game': 'Close Game (0-3 spread)',
    }

    for script, label in script_labels.items():
        script_df = df[df['game_script_tier'] == script]
        stats = compute_slice_stats(script_df, overall_mae, overall_n, label,
                                    min_n_threshold, min_pct_threshold)
        if stats:
            stats['script_key'] = script
            script_slices.append(stats)

    results['by_game_script'] = sorted(script_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # C. MAE by Rest Bucket
    # =========================================================================
    rest_slices = []

    # B2B
    b2b_df = df[df['is_b2b'] == 1]
    stats = compute_slice_stats(b2b_df, overall_mae, overall_n, 'Back-to-Back',
                                min_n_threshold, min_pct_threshold)
    if stats:
        rest_slices.append(stats)

    # 1 day rest
    rest1_df = df[(df['is_b2b'] == 0) & (df['days_rest'] == 1)]
    stats = compute_slice_stats(rest1_df, overall_mae, overall_n, '1 Day Rest',
                                min_n_threshold, min_pct_threshold)
    if stats:
        rest_slices.append(stats)

    # 2 days rest
    rest2_df = df[df['days_rest'] == 2]
    stats = compute_slice_stats(rest2_df, overall_mae, overall_n, '2 Days Rest',
                                min_n_threshold, min_pct_threshold)
    if stats:
        rest_slices.append(stats)

    # 3+ days rest
    rest3_df = df[df['days_rest'] >= 3]
    stats = compute_slice_stats(rest3_df, overall_mae, overall_n, '3+ Days Rest',
                                min_n_threshold, min_pct_threshold)
    if stats:
        rest_slices.append(stats)

    results['by_rest'] = sorted(rest_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # D. Interaction Slices (Role x Game Script)
    # =========================================================================
    interaction_slices = []

    for role in ['STAR', 'STARTER', 'ROTATION']:
        for script in ['blowout', 'close_game']:
            int_df = df[(df['role_tier'] == role) & (df['game_script_tier'] == script)]
            slice_name = f"{role} in {script.replace('_', ' ').title()}"
            stats = compute_slice_stats(int_df, overall_mae, overall_n, slice_name,
                                        min_n_threshold, min_pct_threshold)
            if stats:
                stats['role'] = role
                stats['script'] = script
                interaction_slices.append(stats)

    results['interactions_role_script'] = sorted(interaction_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # E. Interaction Slices (Role x Rest)
    # =========================================================================
    role_rest_slices = []

    for role in ['STAR', 'STARTER']:
        # Role on B2B
        int_df = df[(df['role_tier'] == role) & (df['is_b2b'] == 1)]
        slice_name = f"{role} on B2B"
        stats = compute_slice_stats(int_df, overall_mae, overall_n, slice_name,
                                    min_n_threshold, min_pct_threshold)
        if stats:
            stats['role'] = role
            stats['rest'] = 'b2b'
            role_rest_slices.append(stats)

        # Role well-rested (3+)
        int_df = df[(df['role_tier'] == role) & (df['days_rest'] >= 3)]
        slice_name = f"{role} Well-Rested (3+)"
        stats = compute_slice_stats(int_df, overall_mae, overall_n, slice_name,
                                    min_n_threshold, min_pct_threshold)
        if stats:
            stats['role'] = role
            stats['rest'] = '3+'
            role_rest_slices.append(stats)

    results['interactions_role_rest'] = sorted(role_rest_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # F. Fix Targets (statistically significant AND trustworthy)
    # =========================================================================
    all_slices = (
        results['by_role'] +
        results['by_game_script'] +
        results['by_rest'] +
        results['interactions_role_script'] +
        results['interactions_role_rest']
    )

    # Only include slices that pass the statistical bar
    results['fix_targets'] = [
        s for s in all_slices
        if s.get('is_fix_target', False)
    ]

    # Best buckets (significantly better than baseline)
    results['best_buckets'] = [
        s for s in all_slices
        if s.get('is_significant', False) and s['delta_mae'] < -0.3 and s.get('is_trustworthy', False)
    ]

    # Count how many slices are trustworthy vs not
    results['trustworthy_count'] = sum(1 for s in all_slices if s.get('is_trustworthy', False))
    results['total_slices'] = len(all_slices)

    return results


def format_slice_table(slices: List[Dict], title: str) -> str:
    """Format a slice list as a text table with CIs."""
    if not slices:
        return f"{title}: No data\n"

    lines = [
        title,
        "-" * 80,
        f"{'Slice':<25} {'N':>6} {'Vol%':>6} {'MAE':>6} {'ΔMAE':>7} {'95% CI':>15} {'Sig?':>5}",
        "-" * 80,
    ]

    for s in slices:
        ci_str = f"[{s['ci_lower']:+.2f}, {s['ci_upper']:+.2f}]" if not np.isnan(s['ci_lower']) else "N/A"
        sig_str = "YES" if s.get('is_significant') else "no"
        trust_marker = "" if s.get('is_trustworthy') else " *"

        lines.append(
            f"{s['slice']:<25} {s['n']:>6} {s['pct_volume']:>5.1f}% "
            f"{s['mae']:>6.2f} {s['delta_mae']:>+6.2f} {ci_str:>15} {sig_str:>5}{trust_marker}"
        )

    lines.append("")
    lines.append("* = low sample size, interpret with caution")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Error slices analysis')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    results = calculate_error_slices(conn, days=args.days)

    if results['has_data']:
        print("=" * 80)
        print("ERROR SLICES ANALYSIS (with statistical significance)")
        print("=" * 80)
        print(f"Sample: {results['overall']['n']} predictions over {results['days_analyzed']} days")
        print(f"Overall MAE: {results['overall']['mae']:.2f}, Bias: {results['overall']['bias']:+.2f}")
        print(f"Trustworthy slices: {results['trustworthy_count']}/{results['total_slices']}")
        print()

        print(format_slice_table(results['by_role'], "BY ROLE TIER"))
        print(format_slice_table(results['by_game_script'], "BY GAME SCRIPT"))
        print(format_slice_table(results['by_rest'], "BY REST"))

        print("=" * 80)
        print("FIX TARGETS (statistically significant + sufficient sample)")
        print("=" * 80)
        if results['fix_targets']:
            for s in results['fix_targets']:
                print(f"  🎯 {s['slice']}: {s['pct_volume']:.1f}% volume, +{s['delta_mae']:.2f} MAE worse")
                print(f"     95% CI: [{s['ci_lower']:+.2f}, {s['ci_upper']:+.2f}]")
        else:
            print("  No statistically significant fix targets found")

        print()
        print("BEST BUCKETS (significantly better than baseline)")
        print("-" * 80)
        if results['best_buckets']:
            for s in results['best_buckets']:
                print(f"  ✅ {s['slice']}: {s['pct_volume']:.1f}% volume, {s['delta_mae']:.2f} MAE better")
        else:
            print("  No significantly better buckets found")
    else:
        print(results.get('message', 'No data'))

    conn.close()
