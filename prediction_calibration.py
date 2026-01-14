#!/usr/bin/env python3
"""
Prediction Calibration Module

Provides multiple calibration strategies to correct systematic biases:
1. Constant bias correction: proj_cal = proj - bias
2. Linear calibration: proj_cal = a + b * proj
3. By-tier calibration: different (a, b) per scorer tier
4. Rolling window: recalculate params on recent data

Usage:
    from prediction_calibration import PredictionCalibrator

    calibrator = PredictionCalibrator(conn)
    params = calibrator.fit_linear(days_back=30)
    calibrated = calibrator.apply_linear(raw_projection, params)
"""

import sqlite3
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class CalibrationParams:
    """Stores calibration parameters with metadata."""
    method: str  # 'constant', 'linear', 'by_tier'
    intercept: float  # 'a' in a + b*x
    slope: float  # 'b' in a + b*x (1.0 for constant)
    n_samples: int
    r_squared: float
    fitted_date: str
    days_back: int
    tier_params: Optional[Dict[str, Tuple[float, float]]] = None  # For by-tier


class PredictionCalibrator:
    """
    Calibrates predictions using historical performance data.

    The core insight: if actual = a + b * projected (empirically),
    then calibrated = a + b * projected will have zero expected bias.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _get_historical_data(
        self,
        days_back: int = 30,
        min_actual_ppg: float = 0.0
    ) -> pd.DataFrame:
        """Load historical predictions with actuals for calibration."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        query = """
            SELECT
                projected_ppg,
                actual_ppg,
                proj_confidence,
                team_name,
                game_date
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND actual_ppg >= ?
              AND game_date >= ?
              AND game_date <= ?
        """

        df = pd.read_sql_query(
            query, self.conn,
            params=[min_actual_ppg, str(start_date), str(end_date)]
        )
        return df

    # =========================================================================
    # Constant Bias Correction
    # =========================================================================

    def fit_constant_bias(self, days_back: int = 30) -> CalibrationParams:
        """
        Calculate constant bias: mean(projected - actual).

        Calibration: proj_calibrated = proj - bias
        """
        df = self._get_historical_data(days_back)

        if df.empty:
            return CalibrationParams(
                method='constant',
                intercept=0.0,
                slope=1.0,
                n_samples=0,
                r_squared=0.0,
                fitted_date=str(datetime.now().date()),
                days_back=days_back
            )

        bias = (df['projected_ppg'] - df['actual_ppg']).mean()

        # R² for reference (how much variance explained)
        ss_res = ((df['actual_ppg'] - (df['projected_ppg'] - bias)) ** 2).sum()
        ss_tot = ((df['actual_ppg'] - df['actual_ppg'].mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return CalibrationParams(
            method='constant',
            intercept=-bias,  # Stored as additive: proj + intercept
            slope=1.0,
            n_samples=len(df),
            r_squared=r_squared,
            fitted_date=str(datetime.now().date()),
            days_back=days_back
        )

    def apply_constant_bias(
        self,
        projection: float,
        params: CalibrationParams
    ) -> float:
        """Apply constant bias correction."""
        return projection + params.intercept

    # =========================================================================
    # Linear Calibration (a + b * proj)
    # =========================================================================

    def fit_linear(self, days_back: int = 30) -> CalibrationParams:
        """
        Fit linear calibration: actual = a + b * projected.

        This fixes both intercept (bias) and slope (compression) issues.
        """
        df = self._get_historical_data(days_back)

        if len(df) < 10:
            # Not enough data - return identity
            return CalibrationParams(
                method='linear',
                intercept=0.0,
                slope=1.0,
                n_samples=len(df),
                r_squared=0.0,
                fitted_date=str(datetime.now().date()),
                days_back=days_back
            )

        # Linear regression: actual = a + b * projected
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['projected_ppg'],
            df['actual_ppg']
        )

        return CalibrationParams(
            method='linear',
            intercept=intercept,
            slope=slope,
            n_samples=len(df),
            r_squared=r_value ** 2,
            fitted_date=str(datetime.now().date()),
            days_back=days_back
        )

    def apply_linear(
        self,
        projection: float,
        params: CalibrationParams
    ) -> float:
        """Apply linear calibration: calibrated = a + b * projection."""
        return params.intercept + params.slope * projection

    # =========================================================================
    # By-Tier Calibration
    # =========================================================================

    def fit_by_tier(self, days_back: int = 30) -> CalibrationParams:
        """
        Fit separate linear calibrations for each scorer tier.

        Tiers:
        - Bench (0-12 pts projected)
        - Role (12-18 pts projected)
        - Starter (18-25 pts projected)
        - Star (25+ pts projected)
        """
        df = self._get_historical_data(days_back)

        if len(df) < 20:
            return self.fit_linear(days_back)  # Fall back to global

        # Define tiers
        def get_tier(ppg):
            if ppg < 12:
                return 'Bench'
            elif ppg < 18:
                return 'Role'
            elif ppg < 25:
                return 'Starter'
            else:
                return 'Star'

        df['tier'] = df['projected_ppg'].apply(get_tier)

        tier_params = {}
        for tier in ['Bench', 'Role', 'Starter', 'Star']:
            tier_df = df[df['tier'] == tier]

            if len(tier_df) >= 5:
                slope, intercept, r_value, _, _ = stats.linregress(
                    tier_df['projected_ppg'],
                    tier_df['actual_ppg']
                )
                tier_params[tier] = (intercept, slope)
            else:
                # Not enough data - use global params
                tier_params[tier] = (0.0, 1.0)

        # Overall R² (weighted)
        df['calibrated'] = df.apply(
            lambda row: tier_params[row['tier']][0] + tier_params[row['tier']][1] * row['projected_ppg'],
            axis=1
        )
        ss_res = ((df['actual_ppg'] - df['calibrated']) ** 2).sum()
        ss_tot = ((df['actual_ppg'] - df['actual_ppg'].mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return CalibrationParams(
            method='by_tier',
            intercept=0.0,  # Not used directly
            slope=1.0,  # Not used directly
            n_samples=len(df),
            r_squared=r_squared,
            fitted_date=str(datetime.now().date()),
            days_back=days_back,
            tier_params=tier_params
        )

    def apply_by_tier(
        self,
        projection: float,
        params: CalibrationParams
    ) -> float:
        """Apply tier-specific calibration."""
        if params.tier_params is None:
            return projection

        # Determine tier
        if projection < 12:
            tier = 'Bench'
        elif projection < 18:
            tier = 'Role'
        elif projection < 25:
            tier = 'Starter'
        else:
            tier = 'Star'

        a, b = params.tier_params.get(tier, (0.0, 1.0))
        return a + b * projection

    # =========================================================================
    # Unified Apply Function
    # =========================================================================

    def apply(self, projection: float, params: CalibrationParams) -> float:
        """Apply calibration based on method type."""
        if params.method == 'constant':
            return self.apply_constant_bias(projection, params)
        elif params.method == 'linear':
            return self.apply_linear(projection, params)
        elif params.method == 'by_tier':
            return self.apply_by_tier(projection, params)
        else:
            return projection

    def apply_batch(
        self,
        projections: List[float],
        params: CalibrationParams
    ) -> List[float]:
        """Apply calibration to a batch of projections."""
        return [self.apply(p, params) for p in projections]

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_calibration_summary(self, days_back: int = 30) -> Dict:
        """
        Get comprehensive calibration diagnostics.

        Returns metrics before and after each calibration method.
        """
        df = self._get_historical_data(days_back)

        if df.empty:
            return {'error': 'No data available'}

        # Calculate errors for raw predictions
        df['error'] = df['projected_ppg'] - df['actual_ppg']
        df['abs_error'] = df['error'].abs()

        raw_metrics = {
            'mae': df['abs_error'].mean(),
            'bias': df['error'].mean(),
            'rmse': np.sqrt((df['error'] ** 2).mean()),
            'n_samples': len(df)
        }

        # Fit all calibration methods
        constant_params = self.fit_constant_bias(days_back)
        linear_params = self.fit_linear(days_back)
        tier_params = self.fit_by_tier(days_back)

        # Calculate calibrated errors
        results = {'raw': raw_metrics}

        for name, params in [
            ('constant', constant_params),
            ('linear', linear_params),
            ('by_tier', tier_params)
        ]:
            df[f'cal_{name}'] = df['projected_ppg'].apply(
                lambda x: self.apply(x, params)
            )
            df[f'error_{name}'] = df[f'cal_{name}'] - df['actual_ppg']
            df[f'abs_error_{name}'] = df[f'error_{name}'].abs()

            results[name] = {
                'mae': df[f'abs_error_{name}'].mean(),
                'bias': df[f'error_{name}'].mean(),
                'rmse': np.sqrt((df[f'error_{name}'] ** 2).mean()),
                'r_squared': params.r_squared,
                'params': {
                    'intercept': params.intercept,
                    'slope': params.slope,
                    'tier_params': params.tier_params
                }
            }

        return results

    def get_fanduel_diagnostic(self, days_back: int = 30) -> pd.DataFrame:
        """
        Diagnostic for FanDuel comparison: shows avg(our_error - fd_error)
        by team with count and std_dev.

        This helps identify if 0% win rates are real or measurement issues.
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        query = """
            SELECT
                team_name,
                abs_error as our_error,
                fanduel_error as fd_error,
                (abs_error - fanduel_error) as error_diff,
                we_were_closer
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND fanduel_ou IS NOT NULL
              AND game_date >= ?
              AND game_date <= ?
        """

        df = pd.read_sql_query(
            query, self.conn,
            params=[str(start_date), str(end_date)]
        )

        if df.empty:
            return pd.DataFrame()

        # Aggregate by team
        result = df.groupby('team_name').agg({
            'error_diff': ['mean', 'std', 'count'],
            'we_were_closer': 'sum',
            'our_error': 'mean',
            'fd_error': 'mean'
        }).round(3)

        result.columns = [
            'avg_error_diff', 'std_error_diff', 'n_games',
            'times_closer', 'our_mae', 'fd_mae'
        ]
        result['win_rate'] = (result['times_closer'] / result['n_games'] * 100).round(1)
        result = result.reset_index()
        result = result.sort_values('avg_error_diff')

        return result


# =========================================================================
# Convenience Functions
# =========================================================================

def quick_calibration_check(conn: sqlite3.Connection, days_back: int = 30) -> None:
    """
    Print a quick calibration summary to console.

    Usage:
        from prediction_calibration import quick_calibration_check
        quick_calibration_check(conn, days_back=30)
    """
    calibrator = PredictionCalibrator(conn)
    summary = calibrator.get_calibration_summary(days_back)

    if 'error' in summary:
        print(f"Error: {summary['error']}")
        return

    print(f"\n{'='*60}")
    print(f"CALIBRATION SUMMARY (last {days_back} days)")
    print(f"{'='*60}")
    print(f"Samples: {summary['raw']['n_samples']}")
    print()

    print(f"{'Method':<15} {'MAE':<10} {'Bias':<10} {'RMSE':<10} {'R²':<10}")
    print(f"{'-'*55}")

    for method in ['raw', 'constant', 'linear', 'by_tier']:
        m = summary[method]
        r2 = m.get('r_squared', 'N/A')
        r2_str = f"{r2:.3f}" if isinstance(r2, float) else r2
        print(f"{method:<15} {m['mae']:<10.2f} {m['bias']:<+10.2f} {m['rmse']:<10.2f} {r2_str:<10}")

    print()

    # Show best method
    methods = ['constant', 'linear', 'by_tier']
    best = min(methods, key=lambda x: summary[x]['mae'])
    improvement = summary['raw']['mae'] - summary[best]['mae']

    print(f"Best method: {best}")
    print(f"MAE improvement: {improvement:.2f} pts ({improvement/summary['raw']['mae']*100:.1f}%)")

    if best == 'linear':
        params = summary['linear']['params']
        print(f"Formula: calibrated = {params['intercept']:.2f} + {params['slope']:.3f} * projected")
