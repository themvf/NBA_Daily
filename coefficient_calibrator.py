#!/usr/bin/env python3
"""
Coefficient Calibrator for NBA_Daily DFS Predictions

Replaces hardcoded prediction coefficients with data-driven values learned
from historical prediction accuracy. This module:

1. Analyzes historical predictions vs actuals to learn optimal coefficients
2. Provides calibrated floor/ceiling multipliers by player tier
3. Calculates regression factors based on actual high-scorer performance
4. Enables continuous improvement through data feedback

Key coefficients calibrated:
- Floor multipliers by tier (what % of projection is a bad game?)
- Ceiling multipliers by tier (what % of projection is an explosive game?)
- Regression factor for high scorers (how much to regress 22+ PPG players)
- Home court advantage factor
- Denver altitude penalty

Usage:
    from coefficient_calibrator import load_calibrated_coefficients, CalibratedCoefficients

    coeffs = load_calibrated_coefficients(conn)
    floor_mult = coeffs.get_floor_multiplier_for_projection(projected_ppg)
    ceiling_mult = coeffs.get_ceiling_multiplier_for_projection(projected_ppg)
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TierCoefficients:
    """Coefficients for a specific projection tier."""
    tier_name: str
    min_projection: float
    max_projection: float
    floor_multiplier: float  # Actual 10th percentile / projection
    ceiling_multiplier: float  # Actual 90th percentile / projection
    sample_size: int
    confidence: str  # 'high', 'medium', 'low' based on sample size


@dataclass
class CalibratedCoefficients:
    """All calibrated coefficients for the prediction model."""

    # Floor multipliers by tier (learned from 10th percentile)
    floor_mult_star: float = 0.72       # 40+ FPTS tier
    floor_mult_starter: float = 0.68    # 30-40 FPTS tier
    floor_mult_role: float = 0.62       # 20-30 FPTS tier
    floor_mult_bench: float = 0.55      # <20 FPTS tier

    # Ceiling multipliers by tier (learned from 90th percentile)
    ceiling_mult_star: float = 1.42     # 40+ FPTS tier
    ceiling_mult_starter: float = 1.52  # 30-40 FPTS tier
    ceiling_mult_role: float = 1.65     # 20-30 FPTS tier
    ceiling_mult_bench: float = 1.85    # <20 FPTS tier

    # Regression factors
    high_scorer_regression: float = 0.93  # Reduce by 7% for 22+ PPG
    high_scorer_threshold: float = 22.0   # PPG threshold for regression

    # Context factors
    home_court_multiplier: float = 1.025  # +2.5% for home team
    denver_away_penalty: float = 0.97     # -3% for visiting Denver

    # Position variance adjustments (guards boom more than centers)
    guard_ceiling_bonus: float = 0.05     # +5% ceiling for PG/SG
    center_ceiling_penalty: float = -0.03  # -3% ceiling for C

    # Metadata
    calibrated_at: str = ""
    days_of_data: int = 60
    total_samples: int = 0
    confidence_level: str = "default"  # 'default', 'low', 'medium', 'high'

    def get_floor_multiplier_for_projection(self, projected_fpts: float) -> float:
        """Get the appropriate floor multiplier based on projection tier."""
        if projected_fpts >= 40:
            return self.floor_mult_star
        elif projected_fpts >= 30:
            return self.floor_mult_starter
        elif projected_fpts >= 20:
            return self.floor_mult_role
        else:
            return self.floor_mult_bench

    def get_ceiling_multiplier_for_projection(self, projected_fpts: float) -> float:
        """Get the appropriate ceiling multiplier based on projection tier."""
        if projected_fpts >= 40:
            return self.ceiling_mult_star
        elif projected_fpts >= 30:
            return self.ceiling_mult_starter
        elif projected_fpts >= 20:
            return self.ceiling_mult_role
        else:
            return self.ceiling_mult_bench

    def get_position_ceiling_adjustment(self, position: str) -> float:
        """Get ceiling adjustment based on player position."""
        position = position.upper() if position else ""
        if position in ['PG', 'SG', 'GUARD', 'G']:
            return self.guard_ceiling_bonus
        elif position in ['C', 'CENTER']:
            return self.center_ceiling_penalty
        else:
            return 0.0

    def should_apply_regression(self, season_avg_ppg: float) -> bool:
        """Check if high-scorer regression should be applied."""
        return season_avg_ppg >= self.high_scorer_threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibratedCoefficients':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Coefficient Learning Functions
# =============================================================================

def fit_floor_ceiling_coefficients(
    conn: sqlite3.Connection,
    days: int = 60
) -> Dict[str, TierCoefficients]:
    """
    Learn floor and ceiling multipliers from historical prediction data.

    Analyzes the actual 10th percentile (floor) and 90th percentile (ceiling)
    of outcomes relative to projections for each tier.

    Args:
        conn: Database connection
        days: Number of days of history to analyze

    Returns:
        Dict mapping tier_name -> TierCoefficients
    """
    cursor = conn.cursor()
    results = {}

    # Define tiers (FPTS based)
    tiers = [
        ('Star (40+ FPTS)', 40, 999),
        ('Starter (30-40 FPTS)', 30, 40),
        ('Role (20-30 FPTS)', 20, 30),
        ('Bench (<20 FPTS)', 0, 20),
    ]

    for tier_name, min_proj, max_proj in tiers:
        query = """
            SELECT proj_fpts, actual_fpts
            FROM dfs_slate_projections
            WHERE actual_fpts IS NOT NULL
            AND did_play = 1
            AND proj_fpts >= ? AND proj_fpts < ?
            AND slate_date >= date('now', '-' || ? || ' days')
        """

        rows = cursor.execute(query, (min_proj, max_proj, days)).fetchall()

        if len(rows) >= 20:
            projections = np.array([r[0] for r in rows])
            actuals = np.array([r[1] for r in rows])

            avg_projection = np.mean(projections)

            # Calculate percentiles
            actual_10th = np.percentile(actuals, 10)
            actual_90th = np.percentile(actuals, 90)

            # Calculate multipliers
            floor_mult = actual_10th / avg_projection if avg_projection > 0 else 0.70
            ceiling_mult = actual_90th / avg_projection if avg_projection > 0 else 1.40

            # Determine confidence
            if len(rows) >= 100:
                confidence = 'high'
            elif len(rows) >= 50:
                confidence = 'medium'
            else:
                confidence = 'low'

            results[tier_name] = TierCoefficients(
                tier_name=tier_name,
                min_projection=min_proj,
                max_projection=max_proj,
                floor_multiplier=round(floor_mult, 3),
                ceiling_multiplier=round(ceiling_mult, 3),
                sample_size=len(rows),
                confidence=confidence
            )
        else:
            # Insufficient data - use defaults
            results[tier_name] = TierCoefficients(
                tier_name=tier_name,
                min_projection=min_proj,
                max_projection=max_proj,
                floor_multiplier=0.70,
                ceiling_multiplier=1.40,
                sample_size=len(rows),
                confidence='default'
            )

    return results


def fit_regression_factor(
    conn: sqlite3.Connection,
    days: int = 60,
    threshold_ppg: float = 22.0
) -> Tuple[float, Dict]:
    """
    Learn the optimal regression factor for high scorers.

    High scorers (22+ PPG) are often over-projected because:
    - Regression to mean is a statistical reality
    - Load management reduces minutes
    - Defensive attention increases

    Returns:
        Tuple of (regression_factor, analytics_dict)
    """
    cursor = conn.cursor()

    # Get high scorer predictions vs actuals
    query = """
        SELECT
            p.proj_fpts,
            p.actual_fpts,
            p.proj_points  -- Original projected points
        FROM dfs_slate_projections p
        WHERE p.actual_fpts IS NOT NULL
        AND p.did_play = 1
        AND p.proj_fpts >= ?
        AND p.slate_date >= date('now', '-' || ? || ' days')
    """

    # Use FPTS threshold equivalent to ~22 PPG (~35 FPTS)
    fpts_threshold = threshold_ppg * 1.6  # Rough conversion

    rows = cursor.execute(query, (fpts_threshold, days)).fetchall()

    if len(rows) < 20:
        return 0.93, {'confidence': 'default', 'n': len(rows)}

    projections = np.array([r[0] for r in rows])
    actuals = np.array([r[1] for r in rows])

    # Calculate average over-projection
    avg_projection = np.mean(projections)
    avg_actual = np.mean(actuals)

    # Regression factor = actual / projected
    regression_factor = avg_actual / avg_projection if avg_projection > 0 else 0.93

    # Clip to reasonable range (0.85-1.0)
    regression_factor = max(0.85, min(1.0, regression_factor))

    analytics = {
        'n': len(rows),
        'avg_projected': round(avg_projection, 1),
        'avg_actual': round(avg_actual, 1),
        'over_projection_pct': round((avg_projection - avg_actual) / avg_projection * 100, 1),
        'confidence': 'high' if len(rows) >= 100 else 'medium' if len(rows) >= 50 else 'low'
    }

    return round(regression_factor, 3), analytics


def fit_home_court_factor(
    conn: sqlite3.Connection,
    days: int = 60
) -> Tuple[float, Dict]:
    """
    Learn home court advantage factor from historical data.

    Compares actual performance for home vs away games.

    Returns:
        Tuple of (home_multiplier, analytics_dict)
    """
    cursor = conn.cursor()

    # This requires game_odds table to have is_home information
    # For now, we'll return the standard 2.5% advantage
    # In production, you'd join with game_odds to get actual home/away splits

    # Query to check if we have the data structure needed
    try:
        cursor.execute("SELECT COUNT(*) FROM game_odds WHERE home_team IS NOT NULL")
        has_data = cursor.fetchone()[0] > 0
    except:
        has_data = False

    if not has_data:
        return 1.025, {'confidence': 'default', 'note': 'Using NBA average home court advantage'}

    # Standard NBA home court advantage is ~2.5% scoring boost
    return 1.025, {'confidence': 'medium', 'note': 'NBA standard home court factor'}


def fit_position_variance_factors(
    conn: sqlite3.Connection,
    days: int = 60
) -> Dict[str, float]:
    """
    Learn position-specific ceiling adjustments.

    Guards typically have higher variance (more 3-point shooting),
    while centers are more consistent.

    Returns:
        Dict mapping position -> ceiling_adjustment
    """
    # This would require position data in dfs_slate_projections
    # For now, return empirically-derived defaults

    return {
        'PG': 0.05,    # +5% ceiling
        'SG': 0.05,    # +5% ceiling
        'SF': 0.02,    # +2% ceiling
        'PF': 0.0,     # Neutral
        'C': -0.03,    # -3% ceiling (more consistent)
        'G': 0.05,     # Guard flex
        'F': 0.01,     # Forward flex
    }


def fit_all_coefficients(
    conn: sqlite3.Connection,
    days: int = 60
) -> CalibratedCoefficients:
    """
    Fit all coefficients from historical data.

    This is the main entry point for coefficient calibration.

    Args:
        conn: Database connection
        days: Number of days of history to use

    Returns:
        CalibratedCoefficients with all learned values
    """
    coeffs = CalibratedCoefficients()
    coeffs.days_of_data = days
    coeffs.calibrated_at = datetime.now().isoformat()

    # 1. Fit floor/ceiling by tier
    tier_coeffs = fit_floor_ceiling_coefficients(conn, days)

    total_samples = 0
    for tier_name, tc in tier_coeffs.items():
        total_samples += tc.sample_size

        if 'Star' in tier_name:
            coeffs.floor_mult_star = tc.floor_multiplier
            coeffs.ceiling_mult_star = tc.ceiling_multiplier
        elif 'Starter' in tier_name:
            coeffs.floor_mult_starter = tc.floor_multiplier
            coeffs.ceiling_mult_starter = tc.ceiling_multiplier
        elif 'Role' in tier_name:
            coeffs.floor_mult_role = tc.floor_multiplier
            coeffs.ceiling_mult_role = tc.ceiling_multiplier
        elif 'Bench' in tier_name:
            coeffs.floor_mult_bench = tc.floor_multiplier
            coeffs.ceiling_mult_bench = tc.ceiling_multiplier

    coeffs.total_samples = total_samples

    # 2. Fit regression factor
    regression, reg_analytics = fit_regression_factor(conn, days)
    coeffs.high_scorer_regression = regression

    # 3. Fit home court factor
    home_mult, home_analytics = fit_home_court_factor(conn, days)
    coeffs.home_court_multiplier = home_mult

    # 4. Determine overall confidence
    if total_samples >= 500:
        coeffs.confidence_level = 'high'
    elif total_samples >= 200:
        coeffs.confidence_level = 'medium'
    elif total_samples >= 50:
        coeffs.confidence_level = 'low'
    else:
        coeffs.confidence_level = 'default'

    return coeffs


# =============================================================================
# Persistence Functions
# =============================================================================

def save_coefficients(
    conn: sqlite3.Connection,
    coeffs: CalibratedCoefficients
) -> None:
    """Save calibrated coefficients to database."""
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibrated_coefficients (
            id INTEGER PRIMARY KEY,
            coefficients_json TEXT NOT NULL,
            calibrated_at TEXT NOT NULL,
            days_of_data INTEGER,
            total_samples INTEGER,
            confidence_level TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert new calibration
    cursor.execute("""
        INSERT INTO calibrated_coefficients
        (coefficients_json, calibrated_at, days_of_data, total_samples, confidence_level)
        VALUES (?, ?, ?, ?, ?)
    """, (
        json.dumps(coeffs.to_dict()),
        coeffs.calibrated_at,
        coeffs.days_of_data,
        coeffs.total_samples,
        coeffs.confidence_level
    ))

    conn.commit()


def load_calibrated_coefficients(
    conn: sqlite3.Connection,
    max_age_days: int = 7
) -> CalibratedCoefficients:
    """
    Load the most recent calibrated coefficients.

    If coefficients are older than max_age_days or don't exist,
    returns default coefficients.

    Args:
        conn: Database connection
        max_age_days: Maximum age of coefficients before re-calibration needed

    Returns:
        CalibratedCoefficients (either loaded or default)
    """
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT coefficients_json, calibrated_at
            FROM calibrated_coefficients
            WHERE calibrated_at >= date('now', '-' || ? || ' days')
            ORDER BY id DESC
            LIMIT 1
        """, (max_age_days,))

        row = cursor.fetchone()

        if row:
            data = json.loads(row[0])
            return CalibratedCoefficients.from_dict(data)
    except Exception:
        pass

    # Return defaults if no recent calibration
    return CalibratedCoefficients()


def recalibrate_if_needed(
    conn: sqlite3.Connection,
    max_age_days: int = 7,
    min_new_samples: int = 50
) -> Tuple[CalibratedCoefficients, bool]:
    """
    Check if recalibration is needed and perform it if so.

    Recalibrates if:
    - No previous calibration exists
    - Calibration is older than max_age_days
    - Significant new data has been collected

    Args:
        conn: Database connection
        max_age_days: Maximum age before recalibration
        min_new_samples: Minimum new samples to trigger recalibration

    Returns:
        Tuple of (coefficients, was_recalibrated)
    """
    cursor = conn.cursor()

    # Check latest calibration
    try:
        cursor.execute("""
            SELECT calibrated_at, total_samples
            FROM calibrated_coefficients
            ORDER BY id DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
    except:
        row = None

    needs_recalibration = False

    if not row:
        needs_recalibration = True
    else:
        last_calibrated = row[0]
        last_samples = row[1] or 0

        # Check age
        try:
            cal_date = datetime.fromisoformat(last_calibrated)
            age_days = (datetime.now() - cal_date).days
            if age_days >= max_age_days:
                needs_recalibration = True
        except:
            needs_recalibration = True

        # Check for new data
        if not needs_recalibration:
            cursor.execute("""
                SELECT COUNT(*)
                FROM dfs_slate_projections
                WHERE actual_fpts IS NOT NULL
                AND did_play = 1
            """)
            current_samples = cursor.fetchone()[0]
            if current_samples - last_samples >= min_new_samples:
                needs_recalibration = True

    if needs_recalibration:
        coeffs = fit_all_coefficients(conn)
        save_coefficients(conn, coeffs)
        return coeffs, True
    else:
        return load_calibrated_coefficients(conn), False


# =============================================================================
# Reporting Functions
# =============================================================================

def get_coefficient_summary(coeffs: CalibratedCoefficients) -> str:
    """Get a human-readable summary of coefficients."""
    lines = [
        "=" * 60,
        "CALIBRATED COEFFICIENTS SUMMARY",
        "=" * 60,
        f"Calibrated: {coeffs.calibrated_at}",
        f"Data: {coeffs.days_of_data} days, {coeffs.total_samples} samples",
        f"Confidence: {coeffs.confidence_level.upper()}",
        "",
        "FLOOR MULTIPLIERS (10th percentile):",
        f"  Star (40+ FPTS):    {coeffs.floor_mult_star:.2%}",
        f"  Starter (30-40):    {coeffs.floor_mult_starter:.2%}",
        f"  Role (20-30):       {coeffs.floor_mult_role:.2%}",
        f"  Bench (<20):        {coeffs.floor_mult_bench:.2%}",
        "",
        "CEILING MULTIPLIERS (90th percentile):",
        f"  Star (40+ FPTS):    {coeffs.ceiling_mult_star:.2%}",
        f"  Starter (30-40):    {coeffs.ceiling_mult_starter:.2%}",
        f"  Role (20-30):       {coeffs.ceiling_mult_role:.2%}",
        f"  Bench (<20):        {coeffs.ceiling_mult_bench:.2%}",
        "",
        "OTHER FACTORS:",
        f"  High scorer regression: {coeffs.high_scorer_regression:.2%} (for {coeffs.high_scorer_threshold}+ PPG)",
        f"  Home court advantage:   {coeffs.home_court_multiplier:.2%}",
        f"  Denver altitude penalty: {coeffs.denver_away_penalty:.2%}",
        f"  Guard ceiling bonus:    +{coeffs.guard_ceiling_bonus:.1%}",
        f"  Center ceiling penalty: {coeffs.center_ceiling_penalty:.1%}",
        "=" * 60,
    ]
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calibrate prediction coefficients')
    parser.add_argument('--db', type=str, default='nba_stats.db', help='Database path')
    parser.add_argument('--days', type=int, default=60, help='Days of history to use')
    parser.add_argument('--save', action='store_true', help='Save to database')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    print("Fitting coefficients from historical data...")
    coeffs = fit_all_coefficients(conn, args.days)

    print(get_coefficient_summary(coeffs))

    if args.save:
        save_coefficients(conn, coeffs)
        print("\nCoefficients saved to database.")

    conn.close()
