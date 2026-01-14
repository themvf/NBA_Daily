#!/usr/bin/env python3
"""
Enhanced Prediction Evaluation Metrics

Goes beyond simple MAE to provide deeper insights into prediction quality:
- Spearman rank correlation (are we ordering players correctly?)
- Top-10% miss rate (how badly do we miss on outliers?)
- Ceiling coverage rate (% of actuals within ceiling range)
- Floor coverage rate (% of actuals above floor)
- Calibration score (are our confidence intervals well-calibrated?)

Usage:
    import prediction_evaluation_metrics as pem

    metrics = pem.calculate_enhanced_metrics(conn, game_date='2025-12-25')
    print(f"Spearman correlation: {metrics['spearman_correlation']:.3f}")
"""

import sqlite3
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def calculate_enhanced_metrics(
    conn: sqlite3.Connection,
    game_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_actual_ppg: float = 0.0
) -> Dict:
    """
    Calculate comprehensive prediction evaluation metrics.

    Args:
        conn: Database connection
        game_date: Single date to analyze (YYYY-MM-DD)
        start_date: Start of date range (YYYY-MM-DD)
        end_date: End of date range (YYYY-MM-DD)
        min_actual_ppg: Minimum actual PPG to include (filter DNPs)

    Returns:
        Dictionary with comprehensive metrics
    """
    # Build query based on date parameters
    if game_date:
        date_filter = "game_date = ?"
        params = (game_date,)
    elif start_date and end_date:
        date_filter = "game_date BETWEEN ? AND ?"
        params = (start_date, end_date)
    elif start_date:
        date_filter = "game_date >= ?"
        params = (start_date,)
    else:
        date_filter = "1=1"
        params = ()

    query = f"""
        SELECT
            player_name,
            projected_ppg,
            actual_ppg,
            proj_floor,
            proj_ceiling,
            proj_confidence,
            analytics_used,
            game_date
        FROM predictions
        WHERE {date_filter}
          AND actual_ppg IS NOT NULL
          AND actual_ppg >= ?
        ORDER BY game_date, player_name
    """

    params = params + (min_actual_ppg,)
    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return {
            'error': 'No predictions found for specified criteria',
            'total_predictions': 0
        }

    # Basic calculations
    df['error'] = df['projected_ppg'] - df['actual_ppg']
    df['abs_error'] = df['error'].abs()
    df['within_5'] = df['abs_error'] <= 5.0
    df['within_10'] = df['abs_error'] <= 10.0
    df['within_floor_ceiling'] = (
        (df['actual_ppg'] >= df['proj_floor']) &
        (df['actual_ppg'] <= df['proj_ceiling'])
    )
    df['above_floor'] = df['actual_ppg'] >= df['proj_floor']
    df['below_ceiling'] = df['actual_ppg'] <= df['proj_ceiling']

    # Core metrics
    total_predictions = len(df)
    mae = df['abs_error'].mean()
    rmse = np.sqrt((df['error'] ** 2).mean())
    bias = df['error'].mean()
    median_error = df['error'].median()

    # Hit rates
    hit_rate_5 = (df['within_5'].sum() / total_predictions) * 100
    hit_rate_10 = (df['within_10'].sum() / total_predictions) * 100
    floor_ceiling_rate = (df['within_floor_ceiling'].sum() / total_predictions) * 100
    floor_coverage = (df['above_floor'].sum() / total_predictions) * 100
    ceiling_coverage = (df['below_ceiling'].sum() / total_predictions) * 100

    # Spearman rank correlation
    # Measures if we're ordering players correctly (crucial for DFS)
    spearman_corr, spearman_pval = stats.spearmanr(
        df['projected_ppg'],
        df['actual_ppg']
    )

    # Pearson correlation (for completeness)
    pearson_corr, pearson_pval = stats.pearsonr(
        df['projected_ppg'],
        df['actual_ppg']
    )

    # Top 10% miss rate
    # How badly do we miss on the biggest outliers?
    top_10_pct = int(total_predictions * 0.10)
    if top_10_pct > 0:
        worst_misses = df.nlargest(top_10_pct, 'abs_error')
        top_10_miss_rate = worst_misses['abs_error'].mean()
        top_10_max_miss = worst_misses['abs_error'].max()
    else:
        top_10_miss_rate = None
        top_10_max_miss = None

    # Bottom 10% accuracy (best predictions)
    bottom_10_pct = int(total_predictions * 0.10)
    if bottom_10_pct > 0:
        best_predictions = df.nsmallest(bottom_10_pct, 'abs_error')
        bottom_10_mae = best_predictions['abs_error'].mean()
    else:
        bottom_10_mae = None

    # Calibration score
    # Are our confidence intervals well-calibrated?
    # High confidence should = tighter ranges that hit more often
    if 'proj_confidence' in df.columns and df['proj_confidence'].notna().any():
        # Group by confidence buckets
        df['confidence_bucket'] = pd.cut(
            df['proj_confidence'],
            bins=[0, 0.7, 0.8, 0.9, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        calibration_by_conf = df.groupby('confidence_bucket', observed=True).agg({
            'within_floor_ceiling': 'mean',
            'abs_error': 'mean',
            'player_name': 'count'
        }).round(3)

        calibration_score = calibration_by_conf.to_dict()
    else:
        calibration_score = None

    # Over/under projection analysis
    over_projections = (df['error'] > 0).sum()
    under_projections = (df['error'] < 0).sum()
    exact_projections = (df['error'] == 0).sum()

    over_pct = (over_projections / total_predictions) * 100
    under_pct = (under_projections / total_predictions) * 100

    # Analytics type breakdown
    analytics_breakdown = df.groupby('analytics_used').agg({
        'abs_error': ['mean', 'count'],
        'within_floor_ceiling': 'mean'
    }).round(2)
    analytics_breakdown.columns = ['MAE', 'Count', 'Floor_Ceiling_Rate']

    # Build comprehensive results
    results = {
        # Basic stats
        'total_predictions': total_predictions,
        'date_range': f"{df['game_date'].min()} to {df['game_date'].max()}",

        # Core accuracy metrics
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'bias': round(bias, 2),
        'median_error': round(median_error, 2),

        # Hit rates
        'hit_rate_within_5': round(hit_rate_5, 1),
        'hit_rate_within_10': round(hit_rate_10, 1),
        'floor_ceiling_hit_rate': round(floor_ceiling_rate, 1),
        'floor_coverage': round(floor_coverage, 1),
        'ceiling_coverage': round(ceiling_coverage, 1),

        # Correlation metrics
        'spearman_correlation': round(spearman_corr, 3),
        'spearman_pvalue': spearman_pval,
        'pearson_correlation': round(pearson_corr, 3),
        'pearson_pvalue': pearson_pval,

        # Outlier analysis
        'top_10_pct_miss_rate': round(top_10_miss_rate, 2) if top_10_miss_rate else None,
        'top_10_pct_max_miss': round(top_10_max_miss, 2) if top_10_max_miss else None,
        'bottom_10_pct_mae': round(bottom_10_mae, 2) if bottom_10_mae else None,

        # Over/under balance
        'over_projections': over_projections,
        'under_projections': under_projections,
        'over_pct': round(over_pct, 1),
        'under_pct': round(under_pct, 1),

        # Detailed breakdowns
        'calibration_score': calibration_score,
        'analytics_breakdown': analytics_breakdown.to_dict() if not analytics_breakdown.empty else None
    }

    return results


def get_metrics_by_player_tier(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate metrics broken down by player scoring tier.

    Helps identify if model performs differently for stars vs role players.

    Returns:
        DataFrame with metrics per tier
    """
    # Get predictions with actuals
    if start_date and end_date:
        date_filter = "game_date BETWEEN ? AND ?"
        params = (start_date, end_date)
    elif start_date:
        date_filter = "game_date >= ?"
        params = (start_date,)
    else:
        date_filter = "1=1"
        params = ()

    query = f"""
        SELECT
            projected_ppg,
            actual_ppg,
            proj_floor,
            proj_ceiling
        FROM predictions
        WHERE {date_filter}
          AND actual_ppg IS NOT NULL
          AND actual_ppg > 0
    """

    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    # Calculate errors
    df['error'] = df['projected_ppg'] - df['actual_ppg']
    df['abs_error'] = df['error'].abs()
    df['within_floor_ceiling'] = (
        (df['actual_ppg'] >= df['proj_floor']) &
        (df['actual_ppg'] <= df['proj_ceiling'])
    )

    # Define player tiers based on ACTUAL PPG
    df['tier'] = pd.cut(
        df['actual_ppg'],
        bins=[0, 10, 15, 20, 25, 100],
        labels=['Role (0-10)', 'Bench (10-15)', 'Starter (15-20)', 'Star (20-25)', 'Superstar (25+)']
    )

    # Calculate metrics per tier
    tier_metrics = df.groupby('tier', observed=True).agg({
        'abs_error': ['mean', 'std', 'count'],
        'error': 'mean',
        'within_floor_ceiling': 'mean',
        'projected_ppg': 'mean',
        'actual_ppg': 'mean'
    }).round(2)

    tier_metrics.columns = ['MAE', 'StdDev', 'Count', 'Bias', 'Floor_Ceiling_Rate', 'Avg_Proj', 'Avg_Actual']

    return tier_metrics.reset_index()


def get_worst_predictions(
    conn: sqlite3.Connection,
    game_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20
) -> pd.DataFrame:
    """
    Get the worst predictions (largest misses).

    Useful for identifying systematic blind spots.
    """
    # Build query
    if game_date:
        date_filter = "game_date = ?"
        params = (game_date,)
    elif start_date and end_date:
        date_filter = "game_date BETWEEN ? AND ?"
        params = (start_date, end_date)
    elif start_date:
        date_filter = "game_date >= ?"
        params = (start_date,)
    else:
        date_filter = "1=1"
        params = ()

    query = f"""
        SELECT
            game_date,
            player_name,
            projected_ppg,
            actual_ppg,
            (projected_ppg - actual_ppg) as error,
            ABS(projected_ppg - actual_ppg) as abs_error,
            proj_floor,
            proj_ceiling,
            analytics_used
        FROM predictions
        WHERE {date_filter}
          AND actual_ppg IS NOT NULL
        ORDER BY abs_error DESC
        LIMIT ?
    """

    params = params + (limit,)
    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    # Round for readability
    df['projected_ppg'] = df['projected_ppg'].round(1)
    df['actual_ppg'] = df['actual_ppg'].round(1)
    df['error'] = df['error'].round(1)
    df['abs_error'] = df['abs_error'].round(1)
    df['proj_floor'] = df['proj_floor'].round(1)
    df['proj_ceiling'] = df['proj_ceiling'].round(1)

    return df


def print_metrics_report(metrics: Dict):
    """Pretty print metrics report to console."""
    print("=" * 70)
    print("ENHANCED PREDICTION EVALUATION METRICS")
    print("=" * 70)

    print(f"\nDataset: {metrics['total_predictions']} predictions")
    print(f"Date Range: {metrics['date_range']}")

    print("\n--- CORE ACCURACY METRICS ---")
    print(f"MAE (Mean Absolute Error):    {metrics['mae']:.2f} PPG")
    print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.2f} PPG")
    print(f"Bias (Systematic Error):       {metrics['bias']:+.2f} PPG")
    print(f"Median Error:                  {metrics['median_error']:+.2f} PPG")

    print("\n--- HIT RATES ---")
    print(f"Within ±5 PPG:                 {metrics['hit_rate_within_5']:.1f}%")
    print(f"Within ±10 PPG:                {metrics['hit_rate_within_10']:.1f}%")
    print(f"Within Floor-Ceiling Range:    {metrics['floor_ceiling_hit_rate']:.1f}%")
    print(f"Above Floor:                   {metrics['floor_coverage']:.1f}%")
    print(f"Below Ceiling:                 {metrics['ceiling_coverage']:.1f}%")

    print("\n--- RANK-ORDERING QUALITY ---")
    print(f"Spearman Correlation:          {metrics['spearman_correlation']:.3f}")
    print(f"  (How well we rank players relative to each other)")
    print(f"Pearson Correlation:           {metrics['pearson_correlation']:.3f}")

    print("\n--- OUTLIER ANALYSIS ---")
    if metrics['top_10_pct_miss_rate']:
        print(f"Top 10% Worst Misses (MAE):    {metrics['top_10_pct_miss_rate']:.2f} PPG")
        print(f"Top 10% Max Miss:              {metrics['top_10_pct_max_miss']:.2f} PPG")
    if metrics['bottom_10_pct_mae']:
        print(f"Bottom 10% Best Preds (MAE):   {metrics['bottom_10_pct_mae']:.2f} PPG")

    print("\n--- OVER/UNDER BALANCE ---")
    print(f"Over-projections:              {metrics['over_projections']} ({metrics['over_pct']:.1f}%)")
    print(f"Under-projections:             {metrics['under_projections']} ({metrics['under_pct']:.1f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    db_path = "nba_stats.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    conn = sqlite3.connect(db_path)

    # Calculate metrics for recent predictions
    print("Calculating metrics for last 30 days...")
    from datetime import date, timedelta

    end_date = date.today().strftime('%Y-%m-%d')
    start_date = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    metrics = calculate_enhanced_metrics(
        conn,
        start_date=start_date,
        end_date=end_date,
        min_actual_ppg=1.0  # Filter DNPs
    )

    if 'error' not in metrics:
        print_metrics_report(metrics)

        print("\n\nMETRICS BY PLAYER TIER:")
        print("-" * 70)
        tier_metrics = get_metrics_by_player_tier(conn, start_date, end_date)
        if not tier_metrics.empty:
            print(tier_metrics.to_string(index=False))
    else:
        print(f"ERROR: {metrics['error']}")

    conn.close()
