#!/usr/bin/env python3
"""
Minutes Counterfactual Backtest

Quantifies how much prediction error comes from "bad minutes" vs "bad per-minute scoring."

Method (per player-game):
- P0 (current): your normal projected points
- P1 (minutes-perfect): projected_ppm * actual_minutes
- P2 (rate-perfect): actual_ppm * projected_minutes

If MAE(P1) << MAE(P0) -> minutes modeling is the big lever
If MAE(P2) << MAE(P0) -> rate/usage/matchup modeling is the big lever
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


def run_counterfactual_analysis(
    conn: sqlite3.Connection,
    days: int = 30,
    min_minutes: float = 10.0
) -> Dict:
    """
    Run the minutes counterfactual backtest.

    Args:
        conn: Database connection
        days: Number of days to analyze
        min_minutes: Minimum actual minutes to include (filters DNPs/garbage time)

    Returns:
        Dict with analysis results
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # Query predictions with both projected and actual data
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
            p.season_avg_ppg,
            p.l5_minutes_avg
        FROM predictions p
        WHERE p.actual_ppg IS NOT NULL
          AND p.actual_minutes IS NOT NULL
          AND p.proj_minutes IS NOT NULL
          AND p.proj_minutes > 0
          AND p.actual_minutes >= ?
          AND date(p.game_date) >= date(?)
    """

    df = pd.read_sql_query(query, conn, params=[min_minutes, cutoff])

    if df.empty:
        return {
            'has_data': False,
            'message': 'No predictions with minutes data found'
        }

    # Calculate PPM (points per minute)
    df['projected_ppm'] = df['projected_ppg'] / df['proj_minutes']
    df['actual_ppm'] = df['actual_ppg'] / df['actual_minutes']

    # Calculate counterfactual predictions
    # P0: Current prediction (what we predicted)
    df['P0'] = df['projected_ppg']

    # P1: Minutes-perfect (use actual minutes, keep projected rate)
    df['P1'] = df['projected_ppm'] * df['actual_minutes']

    # P2: Rate-perfect (use actual rate, keep projected minutes)
    df['P2'] = df['actual_ppm'] * df['proj_minutes']

    # Calculate errors
    df['error_P0'] = df['actual_ppg'] - df['P0']
    df['error_P1'] = df['actual_ppg'] - df['P1']
    df['error_P2'] = df['actual_ppg'] - df['P2']

    df['abs_error_P0'] = np.abs(df['error_P0'])
    df['abs_error_P1'] = np.abs(df['error_P1'])
    df['abs_error_P2'] = np.abs(df['error_P2'])

    # Also track minutes error directly
    df['minutes_error'] = df['actual_minutes'] - df['proj_minutes']
    df['abs_minutes_error'] = np.abs(df['minutes_error'])

    # =========================================================================
    # Overall Results
    # =========================================================================
    overall = {
        'n': len(df),
        'mae_P0': df['abs_error_P0'].mean(),
        'mae_P1': df['abs_error_P1'].mean(),
        'mae_P2': df['abs_error_P2'].mean(),
        'bias_P0': df['error_P0'].mean(),
        'bias_P1': df['error_P1'].mean(),
        'bias_P2': df['error_P2'].mean(),
        'minutes_mae': df['abs_minutes_error'].mean(),
        'minutes_bias': df['minutes_error'].mean(),
    }

    # Calculate improvement percentages
    overall['improvement_P1'] = (overall['mae_P0'] - overall['mae_P1']) / overall['mae_P0'] * 100
    overall['improvement_P2'] = (overall['mae_P0'] - overall['mae_P2']) / overall['mae_P0'] * 100

    # Determine primary lever
    if overall['improvement_P1'] > overall['improvement_P2'] + 5:
        overall['primary_lever'] = 'MINUTES'
        overall['lever_explanation'] = f"Minutes modeling is the bigger lever ({overall['improvement_P1']:.0f}% vs {overall['improvement_P2']:.0f}% improvement)"
    elif overall['improvement_P2'] > overall['improvement_P1'] + 5:
        overall['primary_lever'] = 'RATE'
        overall['lever_explanation'] = f"Rate/matchup modeling is the bigger lever ({overall['improvement_P2']:.0f}% vs {overall['improvement_P1']:.0f}% improvement)"
    else:
        overall['primary_lever'] = 'BOTH'
        overall['lever_explanation'] = f"Both levers contribute similarly ({overall['improvement_P1']:.0f}% vs {overall['improvement_P2']:.0f}%)"

    # =========================================================================
    # Breakdown by Role Tier
    # =========================================================================
    role_tiers = ['STAR', 'STARTER', 'ROTATION', 'BENCH']
    by_role = {}

    for role in role_tiers:
        role_df = df[df['role_tier'] == role]
        if len(role_df) >= 10:
            by_role[role] = {
                'n': len(role_df),
                'mae_P0': role_df['abs_error_P0'].mean(),
                'mae_P1': role_df['abs_error_P1'].mean(),
                'mae_P2': role_df['abs_error_P2'].mean(),
                'improvement_P1': (role_df['abs_error_P0'].mean() - role_df['abs_error_P1'].mean()) / role_df['abs_error_P0'].mean() * 100,
                'improvement_P2': (role_df['abs_error_P0'].mean() - role_df['abs_error_P2'].mean()) / role_df['abs_error_P0'].mean() * 100,
                'minutes_mae': role_df['abs_minutes_error'].mean(),
            }

    # =========================================================================
    # Breakdown by Game Script (Spread Proxy)
    # =========================================================================
    by_game_script = {}

    for script in ['blowout', 'neutral', 'close_game']:
        script_df = df[df['game_script_tier'] == script]
        if len(script_df) >= 10:
            by_game_script[script] = {
                'n': len(script_df),
                'mae_P0': script_df['abs_error_P0'].mean(),
                'mae_P1': script_df['abs_error_P1'].mean(),
                'mae_P2': script_df['abs_error_P2'].mean(),
                'improvement_P1': (script_df['abs_error_P0'].mean() - script_df['abs_error_P1'].mean()) / script_df['abs_error_P0'].mean() * 100,
                'improvement_P2': (script_df['abs_error_P0'].mean() - script_df['abs_error_P2'].mean()) / script_df['abs_error_P0'].mean() * 100,
                'minutes_mae': script_df['abs_minutes_error'].mean(),
            }

    # =========================================================================
    # Worst Misses Analysis
    # =========================================================================
    # Find cases where minutes error was the culprit
    df['minutes_was_culprit'] = df['abs_error_P1'] < df['abs_error_P0'] * 0.5
    df['rate_was_culprit'] = df['abs_error_P2'] < df['abs_error_P0'] * 0.5

    worst_minutes_misses = df.nlargest(10, 'abs_minutes_error')[
        ['player_name', 'game_date', 'proj_minutes', 'actual_minutes', 'minutes_error', 'role_tier']
    ].to_dict('records')

    # =========================================================================
    # Package Results
    # =========================================================================
    return {
        'has_data': True,
        'days_analyzed': days,
        'min_minutes_filter': min_minutes,
        'overall': overall,
        'by_role': by_role,
        'by_game_script': by_game_script,
        'worst_minutes_misses': worst_minutes_misses,
        'pct_minutes_culprit': df['minutes_was_culprit'].mean() * 100,
        'pct_rate_culprit': df['rate_was_culprit'].mean() * 100,
    }


def format_counterfactual_summary(results: Dict) -> str:
    """Format results as a text summary."""
    if not results.get('has_data'):
        return results.get('message', 'No data')

    o = results['overall']
    lines = [
        "=" * 60,
        "MINUTES COUNTERFACTUAL BACKTEST",
        "=" * 60,
        f"Sample: {o['n']} predictions over {results['days_analyzed']} days",
        f"Filter: actual_minutes >= {results['min_minutes_filter']}",
        "",
        "OVERALL RESULTS:",
        "-" * 40,
        f"  P0 (Current):        MAE = {o['mae_P0']:.2f}  (baseline)",
        f"  P1 (Minutes-Perfect): MAE = {o['mae_P1']:.2f}  ({o['improvement_P1']:+.0f}% improvement)",
        f"  P2 (Rate-Perfect):    MAE = {o['mae_P2']:.2f}  ({o['improvement_P2']:+.0f}% improvement)",
        "",
        f"  Minutes MAE: {o['minutes_mae']:.1f} min  (bias: {o['minutes_bias']:+.1f})",
        "",
        f"PRIMARY LEVER: {o['primary_lever']}",
        f"  {o['lever_explanation']}",
        "",
    ]

    if results['by_role']:
        lines.append("BY ROLE TIER:")
        lines.append("-" * 40)
        for role, data in results['by_role'].items():
            lines.append(f"  {role}: N={data['n']}, MAE {data['mae_P0']:.1f} -> "
                        f"P1: {data['improvement_P1']:+.0f}%, P2: {data['improvement_P2']:+.0f}%")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minutes counterfactual backtest')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    results = run_counterfactual_analysis(conn, days=args.days)
    print(format_counterfactual_summary(results))
    conn.close()
