#!/usr/bin/env python3
"""
Ablation Backtesting Module

Compares model variants with different enrichment combinations to measure
the impact of each enrichment on prediction accuracy.

Variants tested:
- Baseline: No enrichments applied
- +Rest: Only rest/B2B adjustments
- +GameScript: Only game script adjustments
- +Roles: Role tier information (for segmentation)
- +PosDefense: Position-specific matchup factors
- Full: All enrichments combined

Metrics reported:
- MAE, RMSE, Bias (overall)
- Calibration by projection bucket
- Segment breakdowns
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """Results from a single backtest variant."""
    variant_name: str
    date_range: Tuple[str, str]
    n_predictions: int

    # Overall metrics
    mae: float
    rmse: float
    bias: float
    r_squared: float

    # Calibration by bucket
    calibration_buckets: Dict[str, Dict]

    # Segment breakdowns
    segment_results: Dict[str, Dict]

    # Tail accuracy
    ceiling_hit_rate: float
    top_decile_capture: float

    # Comparison to baseline
    mae_vs_baseline: Optional[float] = None
    mae_improvement_pct: Optional[float] = None


@dataclass
class AblationResults:
    """Complete ablation study results."""
    study_date: str
    date_range: Tuple[str, str]
    variants: Dict[str, BacktestResult] = field(default_factory=dict)
    best_variant: Optional[str] = None
    summary: Optional[str] = None


class AblationBacktester:
    """
    Runs ablation studies comparing enrichment variants.

    Usage:
        backtest = AblationBacktester(conn)
        results = backtest.run_full_ablation('2025-12-01', '2025-12-31')
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_predictions_with_actuals(
        self,
        date_from: str,
        date_to: str
    ) -> pd.DataFrame:
        """Load predictions with actual outcomes for date range."""
        query = """
            SELECT
                p.game_date,
                p.player_id,
                p.player_name,
                p.projected_ppg,
                p.actual_ppg,
                p.proj_floor,
                p.proj_ceiling,
                p.season_avg_ppg,
                p.days_rest,
                p.rest_multiplier,
                p.is_b2b,
                p.game_script_tier,
                p.blowout_risk,
                p.minutes_adjustment,
                p.role_tier,
                p.position_matchup_factor,
                p.opponent_def_rating
            FROM predictions p
            WHERE date(p.game_date) BETWEEN date(?) AND date(?)
              AND p.actual_ppg IS NOT NULL
              AND p.projected_ppg IS NOT NULL
        """
        return pd.read_sql_query(query, self.conn, params=[date_from, date_to])

    def calculate_baseline_projection(self, row: pd.Series) -> float:
        """
        Calculate what projection would be without enrichments.

        This reverses the enrichment adjustments to get the base projection.
        """
        proj = row['projected_ppg']
        rest_mult = row.get('rest_multiplier', 1.0) or 1.0
        pos_factor = row.get('position_matchup_factor', 1.0) or 1.0
        min_adj = row.get('minutes_adjustment', 0.0) or 0.0

        # Reverse: proj = base * rest_mult * pos_factor + min_adj
        # So: base = (proj - min_adj) / (rest_mult * pos_factor)
        if rest_mult * pos_factor != 0:
            base = (proj - min_adj) / (rest_mult * pos_factor)
        else:
            base = proj

        return base

    def apply_enrichment_variant(
        self,
        df: pd.DataFrame,
        variant: str
    ) -> pd.DataFrame:
        """
        Apply a specific enrichment variant to get adjusted projections.

        Variants:
        - 'baseline': No enrichments
        - '+rest': Only rest adjustments
        - '+gamescript': Only game script adjustments
        - '+roles': No adjustments, but use role for segmentation
        - '+posdefense': Only position defense factor
        - 'full': All enrichments
        """
        df = df.copy()

        # First, calculate baseline projection for all
        df['base_projection'] = df.apply(self.calculate_baseline_projection, axis=1)

        if variant == 'baseline':
            df['variant_projection'] = df['base_projection']

        elif variant == '+rest':
            df['variant_projection'] = df['base_projection'] * df['rest_multiplier'].fillna(1.0)

        elif variant == '+gamescript':
            df['variant_projection'] = df['base_projection'] + df['minutes_adjustment'].fillna(0.0)

        elif variant == '+roles':
            # Role doesn't change projection, just used for segmentation
            df['variant_projection'] = df['base_projection']

        elif variant == '+posdefense':
            df['variant_projection'] = df['base_projection'] * df['position_matchup_factor'].fillna(1.0)

        elif variant == 'full':
            # All enrichments combined
            df['variant_projection'] = df['projected_ppg']  # Already has all applied

        else:
            df['variant_projection'] = df['projected_ppg']

        return df

    def calculate_metrics(
        self,
        df: pd.DataFrame,
        projection_col: str = 'variant_projection'
    ) -> Dict:
        """Calculate prediction accuracy metrics."""
        df = df.copy()
        df['error'] = df['actual_ppg'] - df[projection_col]
        df['abs_error'] = df['error'].abs()
        df['squared_error'] = df['error'] ** 2

        # Ceiling hit
        df['ceiling_hit'] = (
            (df['actual_ppg'] >= df['proj_floor']) &
            (df['actual_ppg'] <= df['proj_ceiling'])
        ).astype(int)

        # Top decile
        top_decile_threshold = df['actual_ppg'].quantile(0.90)
        df['is_top_decile'] = (df['actual_ppg'] >= top_decile_threshold).astype(int)
        df['predicted_top_decile'] = (df[projection_col] >= df[projection_col].quantile(0.80)).astype(int)

        metrics = {
            'n': len(df),
            'mae': df['abs_error'].mean(),
            'rmse': np.sqrt(df['squared_error'].mean()),
            'bias': df['error'].mean(),
            'r_squared': 1 - (df['squared_error'].sum() / ((df['actual_ppg'] - df['actual_ppg'].mean()) ** 2).sum()),
            'ceiling_hit_rate': df['ceiling_hit'].mean(),
            'top_decile_capture': (df['is_top_decile'] & df['predicted_top_decile']).sum() / df['is_top_decile'].sum() if df['is_top_decile'].sum() > 0 else 0,
        }

        return metrics

    def calculate_calibration_buckets(
        self,
        df: pd.DataFrame,
        projection_col: str = 'variant_projection'
    ) -> Dict[str, Dict]:
        """Calculate calibration by projection bucket."""
        buckets = {
            '5-10': (5, 10),
            '10-15': (10, 15),
            '15-20': (15, 20),
            '20-25': (20, 25),
            '25-30': (25, 30),
            '30+': (30, 100),
        }

        results = {}
        for bucket_name, (low, high) in buckets.items():
            bucket_df = df[(df[projection_col] >= low) & (df[projection_col] < high)]
            if len(bucket_df) > 0:
                results[bucket_name] = {
                    'count': len(bucket_df),
                    'predicted_mean': bucket_df[projection_col].mean(),
                    'actual_mean': bucket_df['actual_ppg'].mean(),
                    'bias': bucket_df['actual_ppg'].mean() - bucket_df[projection_col].mean(),
                    'mae': (bucket_df['actual_ppg'] - bucket_df[projection_col]).abs().mean(),
                }
            else:
                results[bucket_name] = {
                    'count': 0,
                    'predicted_mean': None,
                    'actual_mean': None,
                    'bias': None,
                    'mae': None,
                }

        return results

    def calculate_segment_results(
        self,
        df: pd.DataFrame,
        projection_col: str = 'variant_projection'
    ) -> Dict[str, Dict]:
        """Calculate metrics by segment (role, game script, rest status)."""
        segments = {}

        # By role tier
        for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
            role_df = df[df['role_tier'] == role]
            if len(role_df) > 0:
                segments[f'role_{role}'] = {
                    'count': len(role_df),
                    'mae': (role_df['actual_ppg'] - role_df[projection_col]).abs().mean(),
                    'bias': (role_df['actual_ppg'] - role_df[projection_col]).mean(),
                }

        # By game script
        for script in ['blowout', 'close_game', 'tossup', 'comfortable']:
            script_df = df[df['game_script_tier'] == script]
            if len(script_df) > 0:
                segments[f'script_{script}'] = {
                    'count': len(script_df),
                    'mae': (script_df['actual_ppg'] - script_df[projection_col]).abs().mean(),
                    'bias': (script_df['actual_ppg'] - script_df[projection_col]).mean(),
                }

        # By rest status
        b2b_df = df[df['is_b2b'] == 1]
        if len(b2b_df) > 0:
            segments['rest_b2b'] = {
                'count': len(b2b_df),
                'mae': (b2b_df['actual_ppg'] - b2b_df[projection_col]).abs().mean(),
                'bias': (b2b_df['actual_ppg'] - b2b_df[projection_col]).mean(),
            }

        rested_df = df[(df['days_rest'].notna()) & (df['days_rest'] >= 3)]
        if len(rested_df) > 0:
            segments['rest_wellrested'] = {
                'count': len(rested_df),
                'mae': (rested_df['actual_ppg'] - rested_df[projection_col]).abs().mean(),
                'bias': (rested_df['actual_ppg'] - rested_df[projection_col]).mean(),
            }

        return segments

    def run_variant_backtest(
        self,
        df: pd.DataFrame,
        variant: str
    ) -> BacktestResult:
        """Run backtest for a single variant."""
        df_variant = self.apply_enrichment_variant(df, variant)
        metrics = self.calculate_metrics(df_variant)
        calibration = self.calculate_calibration_buckets(df_variant)
        segments = self.calculate_segment_results(df_variant)

        return BacktestResult(
            variant_name=variant,
            date_range=('', ''),  # Will be set by caller
            n_predictions=metrics['n'],
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            bias=metrics['bias'],
            r_squared=metrics['r_squared'],
            calibration_buckets=calibration,
            segment_results=segments,
            ceiling_hit_rate=metrics['ceiling_hit_rate'],
            top_decile_capture=metrics['top_decile_capture'],
        )

    def run_full_ablation(
        self,
        date_from: str,
        date_to: str,
        verbose: bool = False
    ) -> AblationResults:
        """
        Run complete ablation study across all variants.

        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            verbose: Print progress

        Returns:
            AblationResults with all variant comparisons
        """
        if verbose:
            print(f"Running ablation study from {date_from} to {date_to}...")

        # Load data
        df = self.get_predictions_with_actuals(date_from, date_to)
        if df.empty:
            print("No prediction data found for date range")
            return AblationResults(
                study_date=datetime.now().strftime('%Y-%m-%d'),
                date_range=(date_from, date_to),
            )

        if verbose:
            print(f"Loaded {len(df)} predictions with actuals")

        variants = ['baseline', '+rest', '+gamescript', '+roles', '+posdefense', 'full']
        results = AblationResults(
            study_date=datetime.now().strftime('%Y-%m-%d'),
            date_range=(date_from, date_to),
        )

        baseline_mae = None

        for variant in variants:
            if verbose:
                print(f"  Testing variant: {variant}...")

            result = self.run_variant_backtest(df, variant)
            result.date_range = (date_from, date_to)

            if variant == 'baseline':
                baseline_mae = result.mae
            elif baseline_mae:
                result.mae_vs_baseline = result.mae - baseline_mae
                result.mae_improvement_pct = (baseline_mae - result.mae) / baseline_mae * 100

            results.variants[variant] = result

            if verbose:
                improvement = f"({result.mae_improvement_pct:+.1f}%)" if result.mae_improvement_pct else ""
                print(f"    MAE: {result.mae:.3f} {improvement}")

        # Find best variant
        best_variant = min(results.variants.keys(), key=lambda v: results.variants[v].mae)
        results.best_variant = best_variant

        # Generate summary
        results.summary = self._generate_summary(results)

        return results

    def _generate_summary(self, results: AblationResults) -> str:
        """Generate text summary of ablation results."""
        lines = [
            f"# Ablation Study Summary",
            f"",
            f"Date Range: {results.date_range[0]} to {results.date_range[1]}",
            f"Study Date: {results.study_date}",
            f"",
            f"## Variant Comparison",
            f"",
            f"| Variant | MAE | vs Baseline | RMSE | Bias |",
            f"|---------|-----|-------------|------|------|",
        ]

        for name, result in results.variants.items():
            vs_baseline = f"{result.mae_improvement_pct:+.1f}%" if result.mae_improvement_pct else "-"
            lines.append(
                f"| {name} | {result.mae:.3f} | {vs_baseline} | "
                f"{result.rmse:.3f} | {result.bias:+.3f} |"
            )

        lines.extend([
            f"",
            f"**Best Variant:** {results.best_variant}",
            f"",
            f"## Key Findings",
            f"",
        ])

        # Add findings based on results
        full_result = results.variants.get('full')
        baseline_result = results.variants.get('baseline')

        if full_result and baseline_result:
            improvement = baseline_result.mae - full_result.mae
            pct = improvement / baseline_result.mae * 100
            lines.append(f"- Full model reduces MAE by {improvement:.3f} ({pct:.1f}%) vs baseline")

        return '\n'.join(lines)


def run_ablation_study(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    verbose: bool = True
) -> AblationResults:
    """
    Convenience function to run ablation study.

    Args:
        conn: Database connection
        date_from: Start date
        date_to: End date
        verbose: Print progress

    Returns:
        AblationResults
    """
    backtest = AblationBacktester(conn)
    return backtest.run_full_ablation(date_from, date_to, verbose=verbose)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run enrichment ablation study')
    parser.add_argument('--from', dest='date_from', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', dest='date_to', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    parser.add_argument('--output', type=str, help='Output file for summary')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    results = run_ablation_study(conn, args.date_from, args.date_to, verbose=True)

    print("\n" + results.summary)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(results.summary)
        print(f"\nSummary saved to {args.output}")

    conn.close()
