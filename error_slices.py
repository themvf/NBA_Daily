#!/usr/bin/env python3
"""
Error Slices Analysis ("Where We Were Wrong")

Breaks down prediction errors by various buckets to identify
which assumptions are failing and where to focus improvements.

Slices:
- Role tier (STAR / STARTER / ROTATION / BENCH)
- Spread bucket (0-3, 4-7, 8-12, 13+)
- Rest bucket (B2B, 1 day, 2 days, 3+ days)
- Interaction slices (Role x Spread, Role x Rest)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


def calculate_error_slices(
    conn: sqlite3.Connection,
    days: int = 30,
    min_sample: int = 10
) -> Dict:
    """
    Calculate MAE and other metrics broken down by various slices.

    Args:
        conn: Database connection
        days: Number of days to analyze
        min_sample: Minimum sample size to include a slice

    Returns:
        Dict with slice breakdowns
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
        if len(role_df) >= min_sample:
            mae = role_df['abs_error'].mean()
            role_slices.append({
                'slice': role,
                'n': len(role_df),
                'pct_volume': len(role_df) / overall_n * 100,
                'mae': mae,
                'bias': role_df['prediction_error'].mean(),
                'delta_mae': mae - overall_mae,
                'top10_rate': role_df['was_top10'].mean() * 100 if 'was_top10' in role_df.columns and not role_df['was_top10'].isna().all() else None,
            })

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
        if len(script_df) >= min_sample:
            mae = script_df['abs_error'].mean()
            script_slices.append({
                'slice': label,
                'script_key': script,
                'n': len(script_df),
                'pct_volume': len(script_df) / overall_n * 100,
                'mae': mae,
                'bias': script_df['prediction_error'].mean(),
                'delta_mae': mae - overall_mae,
            })

    results['by_game_script'] = sorted(script_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # C. MAE by Rest Bucket
    # =========================================================================
    rest_slices = []

    # B2B
    b2b_df = df[df['is_b2b'] == 1]
    if len(b2b_df) >= min_sample:
        mae = b2b_df['abs_error'].mean()
        rest_slices.append({
            'slice': 'Back-to-Back',
            'n': len(b2b_df),
            'pct_volume': len(b2b_df) / overall_n * 100,
            'mae': mae,
            'bias': b2b_df['prediction_error'].mean(),
            'delta_mae': mae - overall_mae,
        })

    # 1 day rest
    rest1_df = df[(df['is_b2b'] == 0) & (df['days_rest'] == 1)]
    if len(rest1_df) >= min_sample:
        mae = rest1_df['abs_error'].mean()
        rest_slices.append({
            'slice': '1 Day Rest',
            'n': len(rest1_df),
            'pct_volume': len(rest1_df) / overall_n * 100,
            'mae': mae,
            'bias': rest1_df['prediction_error'].mean(),
            'delta_mae': mae - overall_mae,
        })

    # 2 days rest
    rest2_df = df[df['days_rest'] == 2]
    if len(rest2_df) >= min_sample:
        mae = rest2_df['abs_error'].mean()
        rest_slices.append({
            'slice': '2 Days Rest',
            'n': len(rest2_df),
            'pct_volume': len(rest2_df) / overall_n * 100,
            'mae': mae,
            'bias': rest2_df['prediction_error'].mean(),
            'delta_mae': mae - overall_mae,
        })

    # 3+ days rest
    rest3_df = df[df['days_rest'] >= 3]
    if len(rest3_df) >= min_sample:
        mae = rest3_df['abs_error'].mean()
        rest_slices.append({
            'slice': '3+ Days Rest',
            'n': len(rest3_df),
            'pct_volume': len(rest3_df) / overall_n * 100,
            'mae': mae,
            'bias': rest3_df['prediction_error'].mean(),
            'delta_mae': mae - overall_mae,
        })

    results['by_rest'] = sorted(rest_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # D. Interaction Slices (Role x Game Script)
    # =========================================================================
    interaction_slices = []

    for role in ['STAR', 'STARTER', 'ROTATION']:
        for script in ['blowout', 'close_game']:
            int_df = df[(df['role_tier'] == role) & (df['game_script_tier'] == script)]
            if len(int_df) >= min_sample:
                mae = int_df['abs_error'].mean()
                interaction_slices.append({
                    'slice': f"{role} in {script.replace('_', ' ').title()}",
                    'role': role,
                    'script': script,
                    'n': len(int_df),
                    'pct_volume': len(int_df) / overall_n * 100,
                    'mae': mae,
                    'bias': int_df['prediction_error'].mean(),
                    'delta_mae': mae - overall_mae,
                })

    results['interactions_role_script'] = sorted(interaction_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # E. Interaction Slices (Role x Rest)
    # =========================================================================
    role_rest_slices = []

    for role in ['STAR', 'STARTER']:
        # Role on B2B
        int_df = df[(df['role_tier'] == role) & (df['is_b2b'] == 1)]
        if len(int_df) >= min_sample:
            mae = int_df['abs_error'].mean()
            role_rest_slices.append({
                'slice': f"{role} on B2B",
                'role': role,
                'rest': 'b2b',
                'n': len(int_df),
                'pct_volume': len(int_df) / overall_n * 100,
                'mae': mae,
                'bias': int_df['prediction_error'].mean(),
                'delta_mae': mae - overall_mae,
            })

        # Role well-rested (3+)
        int_df = df[(df['role_tier'] == role) & (df['days_rest'] >= 3)]
        if len(int_df) >= min_sample:
            mae = int_df['abs_error'].mean()
            role_rest_slices.append({
                'slice': f"{role} Well-Rested (3+)",
                'role': role,
                'rest': '3+',
                'n': len(int_df),
                'pct_volume': len(int_df) / overall_n * 100,
                'mae': mae,
                'bias': int_df['prediction_error'].mean(),
                'delta_mae': mae - overall_mae,
            })

    results['interactions_role_rest'] = sorted(role_rest_slices, key=lambda x: x['delta_mae'], reverse=True)

    # =========================================================================
    # F. Worst Buckets Summary (actionable callouts)
    # =========================================================================
    all_slices = (
        results['by_role'] +
        results['by_game_script'] +
        results['by_rest'] +
        results['interactions_role_script'] +
        results['interactions_role_rest']
    )

    # Filter to buckets with meaningful volume and significant delta
    actionable = [
        s for s in all_slices
        if s['pct_volume'] >= 5 and abs(s['delta_mae']) >= 0.5
    ]

    results['worst_buckets'] = sorted(actionable, key=lambda x: x['delta_mae'], reverse=True)[:5]
    results['best_buckets'] = sorted(actionable, key=lambda x: x['delta_mae'])[:5]

    return results


def format_slice_table(slices: List[Dict], title: str) -> str:
    """Format a slice list as a text table."""
    if not slices:
        return f"{title}: No data\n"

    lines = [
        title,
        "-" * 60,
        f"{'Slice':<25} {'N':>6} {'Vol%':>6} {'MAE':>6} {'ΔMAE':>7} {'Bias':>7}",
        "-" * 60,
    ]

    for s in slices:
        lines.append(
            f"{s['slice']:<25} {s['n']:>6} {s['pct_volume']:>5.1f}% "
            f"{s['mae']:>6.2f} {s['delta_mae']:>+6.2f} {s['bias']:>+6.2f}"
        )

    lines.append("")
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
        print("=" * 60)
        print("ERROR SLICES ANALYSIS")
        print("=" * 60)
        print(f"Sample: {results['overall']['n']} predictions over {results['days_analyzed']} days")
        print(f"Overall MAE: {results['overall']['mae']:.2f}, Bias: {results['overall']['bias']:+.2f}")
        print()

        print(format_slice_table(results['by_role'], "BY ROLE TIER"))
        print(format_slice_table(results['by_game_script'], "BY GAME SCRIPT"))
        print(format_slice_table(results['by_rest'], "BY REST"))
        print(format_slice_table(results['interactions_role_script'], "ROLE x GAME SCRIPT"))
        print(format_slice_table(results['interactions_role_rest'], "ROLE x REST"))

        print("=" * 60)
        print("WORST BUCKETS (Fix Targets)")
        print("=" * 60)
        for s in results['worst_buckets']:
            print(f"  {s['slice']}: {s['pct_volume']:.1f}% volume, +{s['delta_mae']:.2f} MAE worse")

        print()
        print("BEST BUCKETS")
        print("-" * 60)
        for s in results['best_buckets']:
            print(f"  {s['slice']}: {s['pct_volume']:.1f}% volume, {s['delta_mae']:.2f} MAE better")
    else:
        print(results.get('message', 'No data'))

    conn.close()
