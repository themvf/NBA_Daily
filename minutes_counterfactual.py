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

Minutes Miss Classification:
- DNP/Late Scratch: proj_minutes > 0, actual = 0
- Blowout Pull: spread large AND minutes << expected
- Foul Trouble: minutes unexpectedly low (moderate)
- OT Boost: actual minutes >> expected
- Rotation Shift: everything else (coach decision / depth chart change)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta


# =============================================================================
# MISS CLASSIFICATION
# =============================================================================

def classify_minutes_miss(
    proj_minutes: float,
    actual_minutes: float,
    role_tier: str,
    game_script: str,
    blowout_risk: float = 0.0
) -> Tuple[str, str]:
    """
    Classify the type of minutes miss for actionable insights.

    Args:
        proj_minutes: Projected minutes
        actual_minutes: Actual minutes played
        role_tier: Player's role (STAR, STARTER, etc.)
        game_script: Game script tier (blowout, close_game, neutral)
        blowout_risk: Pre-game blowout probability

    Returns:
        (category, explanation)
    """
    minutes_diff = actual_minutes - proj_minutes
    minutes_pct_diff = minutes_diff / proj_minutes if proj_minutes > 0 else 0

    # DNP / Late Scratch
    if actual_minutes == 0 and proj_minutes > 0:
        return ('DNP', 'Player did not play (injury, rest, coach decision)')

    # OT Boost (actual >> expected)
    if minutes_pct_diff > 0.20 and actual_minutes >= 40:
        return ('OT_BOOST', 'Likely overtime - minutes inflated beyond normal')

    # Blowout Pull (spread large OR game_script is blowout, minutes down significantly)
    if game_script == 'blowout' or blowout_risk > 0.5:
        if minutes_pct_diff < -0.15:
            if role_tier in ['STAR', 'STARTER']:
                return ('BLOWOUT_PULL', 'Star/starter pulled early due to blowout')
            else:
                return ('BLOWOUT_ROTATION', 'Bench minutes affected by blowout')

    # Foul Trouble (moderate minutes reduction, not extreme)
    if -0.30 < minutes_pct_diff < -0.15 and actual_minutes > 15:
        return ('FOUL_TROUBLE', 'Possible foul trouble - moderate minutes reduction')

    # Significant underperformance in minutes (not blowout)
    if minutes_pct_diff < -0.25:
        return ('ROTATION_SHIFT', 'Rotation shift or coach decision - significant minutes loss')

    # Significant overperformance in minutes (not OT)
    if minutes_pct_diff > 0.20 and actual_minutes < 40:
        return ('ROTATION_BOOST', 'Unexpected minutes boost - rotation opportunity')

    # Minor variance (within 15%)
    if abs(minutes_pct_diff) <= 0.15:
        return ('NORMAL', 'Minutes within expected range')

    return ('OTHER', 'Unclassified minutes variance')


def analyze_miss_distribution(misses: List[Dict]) -> Dict:
    """
    Analyze the distribution of miss types.

    Returns aggregated counts and insights.
    """
    if not misses:
        return {}

    categories = {}
    for miss in misses:
        cat = miss.get('miss_category', 'OTHER')
        if cat not in categories:
            categories[cat] = {
                'count': 0,
                'total_minutes_error': 0,
                'examples': []
            }
        categories[cat]['count'] += 1
        categories[cat]['total_minutes_error'] += abs(miss.get('minutes_error', 0))
        if len(categories[cat]['examples']) < 3:
            categories[cat]['examples'].append({
                'player': miss.get('player_name', ''),
                'date': miss.get('game_date', ''),
                'proj': miss.get('proj_minutes', 0),
                'actual': miss.get('actual_minutes', 0),
            })

    # Calculate percentages
    total = sum(c['count'] for c in categories.values())
    for cat in categories:
        categories[cat]['pct'] = categories[cat]['count'] / total * 100 if total > 0 else 0
        categories[cat]['avg_minutes_error'] = (
            categories[cat]['total_minutes_error'] / categories[cat]['count']
            if categories[cat]['count'] > 0 else 0
        )

    return categories


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_counterfactual_analysis(
    conn: sqlite3.Connection,
    days: int = 30,
    min_minutes: float = 0.0  # Changed to 0 to capture DNPs
) -> Dict:
    """
    Run the minutes counterfactual backtest with miss classification.

    Args:
        conn: Database connection
        days: Number of days to analyze
        min_minutes: Minimum actual minutes to include in MAE calc (0 = include DNPs)

    Returns:
        Dict with analysis results including miss classification
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # Query predictions with both projected and actual data
    # Include DNPs (actual_minutes = 0) for classification
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
            p.blowout_risk,
            p.season_avg_ppg,
            p.l5_minutes_avg
        FROM predictions p
        WHERE p.actual_ppg IS NOT NULL
          AND p.actual_minutes IS NOT NULL
          AND p.proj_minutes IS NOT NULL
          AND p.proj_minutes > 0
          AND date(p.game_date) >= date(?)
    """

    df = pd.read_sql_query(query, conn, params=[cutoff])

    if df.empty:
        return {
            'has_data': False,
            'message': 'No predictions with minutes data found'
        }

    # ==========================================================================
    # Classify all minutes misses
    # ==========================================================================
    all_misses = []
    for _, row in df.iterrows():
        miss_cat, miss_explain = classify_minutes_miss(
            row['proj_minutes'],
            row['actual_minutes'],
            row['role_tier'] or 'STARTER',
            row['game_script_tier'] or 'neutral',
            row['blowout_risk'] or 0.0
        )
        all_misses.append({
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'game_date': row['game_date'],
            'proj_minutes': row['proj_minutes'],
            'actual_minutes': row['actual_minutes'],
            'minutes_error': row['actual_minutes'] - row['proj_minutes'],
            'role_tier': row['role_tier'],
            'game_script_tier': row['game_script_tier'],
            'miss_category': miss_cat,
            'miss_explanation': miss_explain,
        })

    miss_distribution = analyze_miss_distribution(all_misses)

    # ==========================================================================
    # Filter for MAE calculations (exclude DNPs for fair comparison)
    # ==========================================================================
    df_for_mae = df[df['actual_minutes'] >= max(min_minutes, 10.0)].copy()

    if df_for_mae.empty:
        return {
            'has_data': True,
            'message': 'Not enough data with actual minutes >= threshold',
            'miss_distribution': miss_distribution,
            'all_misses': all_misses,
        }

    # Calculate PPM (points per minute)
    df_for_mae['projected_ppm'] = df_for_mae['projected_ppg'] / df_for_mae['proj_minutes']
    df_for_mae['actual_ppm'] = df_for_mae['actual_ppg'] / df_for_mae['actual_minutes']

    # Calculate counterfactual predictions
    df_for_mae['P0'] = df_for_mae['projected_ppg']
    df_for_mae['P1'] = df_for_mae['projected_ppm'] * df_for_mae['actual_minutes']
    df_for_mae['P2'] = df_for_mae['actual_ppm'] * df_for_mae['proj_minutes']

    # Calculate errors
    df_for_mae['error_P0'] = df_for_mae['actual_ppg'] - df_for_mae['P0']
    df_for_mae['error_P1'] = df_for_mae['actual_ppg'] - df_for_mae['P1']
    df_for_mae['error_P2'] = df_for_mae['actual_ppg'] - df_for_mae['P2']

    df_for_mae['abs_error_P0'] = np.abs(df_for_mae['error_P0'])
    df_for_mae['abs_error_P1'] = np.abs(df_for_mae['error_P1'])
    df_for_mae['abs_error_P2'] = np.abs(df_for_mae['error_P2'])

    df_for_mae['minutes_error'] = df_for_mae['actual_minutes'] - df_for_mae['proj_minutes']
    df_for_mae['abs_minutes_error'] = np.abs(df_for_mae['minutes_error'])

    # ==========================================================================
    # Overall Results
    # ==========================================================================
    overall = {
        'n': len(df_for_mae),
        'mae_P0': df_for_mae['abs_error_P0'].mean(),
        'mae_P1': df_for_mae['abs_error_P1'].mean(),
        'mae_P2': df_for_mae['abs_error_P2'].mean(),
        'bias_P0': df_for_mae['error_P0'].mean(),
        'bias_P1': df_for_mae['error_P1'].mean(),
        'bias_P2': df_for_mae['error_P2'].mean(),
        'minutes_mae': df_for_mae['abs_minutes_error'].mean(),
        'minutes_bias': df_for_mae['minutes_error'].mean(),
    }

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

    # ==========================================================================
    # Breakdown by Role Tier
    # ==========================================================================
    role_tiers = ['STAR', 'STARTER', 'ROTATION', 'BENCH']
    by_role = {}

    for role in role_tiers:
        role_df = df_for_mae[df_for_mae['role_tier'] == role]
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

    # ==========================================================================
    # Breakdown by Game Script
    # ==========================================================================
    by_game_script = {}

    for script in ['blowout', 'neutral', 'close_game']:
        script_df = df_for_mae[df_for_mae['game_script_tier'] == script]
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

    # ==========================================================================
    # Worst Minutes Misses (for display)
    # ==========================================================================
    worst_misses = sorted(all_misses, key=lambda x: abs(x['minutes_error']), reverse=True)[:15]

    # ==========================================================================
    # Investment Recommendations based on miss distribution
    # ==========================================================================
    recommendations = []

    if miss_distribution:
        # Check DNP rate
        dnp_pct = miss_distribution.get('DNP', {}).get('pct', 0)
        if dnp_pct > 5:
            recommendations.append({
                'area': 'Injury/Availability Data',
                'priority': 'HIGH',
                'reason': f'{dnp_pct:.1f}% of predictions were DNPs - improve injury tracking'
            })

        # Check blowout impact
        blowout_pct = miss_distribution.get('BLOWOUT_PULL', {}).get('pct', 0) + miss_distribution.get('BLOWOUT_ROTATION', {}).get('pct', 0)
        if blowout_pct > 10:
            recommendations.append({
                'area': 'Blowout Minutes Model',
                'priority': 'HIGH',
                'reason': f'{blowout_pct:.1f}% of misses are blowout-related - improve game script adjustments'
            })

        # Check rotation shifts
        rotation_pct = miss_distribution.get('ROTATION_SHIFT', {}).get('pct', 0) + miss_distribution.get('ROTATION_BOOST', {}).get('pct', 0)
        if rotation_pct > 15:
            recommendations.append({
                'area': 'Depth Chart / Rotation Updates',
                'priority': 'MEDIUM',
                'reason': f'{rotation_pct:.1f}% are rotation shifts - update depth charts more frequently'
            })

        # Check OT impact
        ot_pct = miss_distribution.get('OT_BOOST', {}).get('pct', 0)
        if ot_pct > 3:
            recommendations.append({
                'area': 'OT Probability Model',
                'priority': 'LOW',
                'reason': f'{ot_pct:.1f}% are OT boosts - limited improvement available'
            })

    # ==========================================================================
    # Package Results
    # ==========================================================================
    return {
        'has_data': True,
        'days_analyzed': days,
        'min_minutes_filter': min_minutes,
        'overall': overall,
        'by_role': by_role,
        'by_game_script': by_game_script,
        'worst_minutes_misses': worst_misses,
        'miss_distribution': miss_distribution,
        'recommendations': recommendations,
        'total_predictions': len(df),
        'dnp_count': len([m for m in all_misses if m['miss_category'] == 'DNP']),
    }


def format_counterfactual_summary(results: Dict) -> str:
    """Format results as a text summary."""
    if not results.get('has_data'):
        return results.get('message', 'No data')

    o = results['overall']
    lines = [
        "=" * 70,
        "MINUTES COUNTERFACTUAL BACKTEST",
        "=" * 70,
        f"Sample: {o['n']} predictions over {results['days_analyzed']} days",
        f"Filter: actual_minutes >= {results['min_minutes_filter']}",
        "",
        "OVERALL RESULTS:",
        "-" * 50,
        f"  P0 (Current):         MAE = {o['mae_P0']:.2f}  (baseline)",
        f"  P1 (Minutes-Perfect): MAE = {o['mae_P1']:.2f}  ({o['improvement_P1']:+.0f}% improvement)",
        f"  P2 (Rate-Perfect):    MAE = {o['mae_P2']:.2f}  ({o['improvement_P2']:+.0f}% improvement)",
        "",
        f"  Minutes MAE: {o['minutes_mae']:.1f} min  (bias: {o['minutes_bias']:+.1f})",
        "",
        f"PRIMARY LEVER: {o['primary_lever']}",
        f"  {o['lever_explanation']}",
        "",
    ]

    if results.get('miss_distribution'):
        lines.append("MISS CLASSIFICATION:")
        lines.append("-" * 50)
        for cat, data in sorted(results['miss_distribution'].items(), key=lambda x: -x[1]['count']):
            lines.append(f"  {cat}: {data['count']} ({data['pct']:.1f}%) - avg error: {data['avg_minutes_error']:.1f} min")

    if results.get('recommendations'):
        lines.append("")
        lines.append("INVESTMENT RECOMMENDATIONS:")
        lines.append("-" * 50)
        for rec in results['recommendations']:
            lines.append(f"  [{rec['priority']}] {rec['area']}")
            lines.append(f"         {rec['reason']}")

    lines.append("")
    lines.append("=" * 70)

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
