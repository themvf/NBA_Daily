#!/usr/bin/env python3
"""
Top 3 Probability Ranking System

Provides ranking algorithms optimized for picking daily top 3 scorers:
1. TopScorerScore - Fast heuristic formula
2. Monte Carlo simulation - P(top 3) estimation via simulation

Usage:
    from top3_ranking import Top3Ranker

    ranker = Top3Ranker(conn)
    rankings = ranker.rank_by_top_scorer_score(game_date)
    # or
    rankings = ranker.rank_by_simulation(game_date, n_simulations=10000)
"""

import sqlite3
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_get(row: dict, key: str, default: float = 0.0) -> float:
    """
    Safely get numeric value, handling None, NaN, and string edge cases.

    Handles:
    - None values
    - NaN (float)
    - Empty strings
    - String representations of None/NaN
    - Numeric strings (converts to float)
    """
    val = row.get(key, None)
    if val is None:
        return default

    # Handle string edge cases from SQLite
    if isinstance(val, str):
        val = val.strip()
        if val in ('', 'nan', 'None', 'null', 'NaN'):
            return default
        try:
            val = float(val)
        except ValueError:
            return default

    # Handle NaN
    if isinstance(val, float) and math.isnan(val):
        return default

    return float(val)


def calculate_projected_minutes(player_data: dict) -> Tuple[float, float]:
    """
    Calculate projected minutes using weighted average + role guardrails + context.

    Returns:
        Tuple of (proj_minutes, minutes_confidence)

    Formula:
        base_min = 0.55 * L5 + 0.25 * L10 + 0.20 * season
        + role guardrails (clamp by starter/rotation/bench)
        + context adjustments (B2B, blowout, injury boost)
    """
    # Step A: Weighted baseline
    l5_min = _safe_get(player_data, 'avg_minutes_last5') or _safe_get(player_data, 'avg_minutes', 0)
    l10_min = _safe_get(player_data, 'avg_minutes_last10') or l5_min
    season_min = _safe_get(player_data, 'avg_minutes', 0)

    if l5_min == 0 and season_min == 0:
        # No minutes data at all
        return 0.0, 0.0

    base_min = 0.55 * l5_min + 0.25 * l10_min + 0.20 * season_min

    # Step B: Role guardrails (prevent insane values)
    starts_last_5 = _safe_get(player_data, 'starts_last_5', 0)
    starter_est = (l5_min >= 28) or (starts_last_5 >= 3)

    if starter_est:
        base_min = max(28, min(40, base_min))  # Clamp starters to [28, 40]
    elif l5_min >= 18:
        base_min = max(18, min(32, base_min))  # Rotation: [18, 32]
    else:
        base_min = max(0, min(22, base_min))   # Deep bench: [0, 22]

    # Step C: Context adjustments
    proj_min = base_min

    # Back-to-back penalty
    if player_data.get('back_to_back'):
        proj_min -= 1.5

    # Blowout risk - use softer approach that mostly affects minutes TAIL, not mean
    # Key insight: Stars still produce in 3 quarters; don't kill their mean projection
    # Use sigmoid for probability: p_blowout = 1 / (1 + exp(-(spread-8)/3))
    spread = abs(_safe_get(player_data, 'vegas_spread', 0))
    if spread >= 8:
        # Calculate blowout probability (smooth sigmoid)
        p_blowout = 1 / (1 + math.exp(-(spread - 8) / 3))
        # Calculate minutes ceiling reduction (stars lose less because they produce in 3Q)
        minutes_ceiling_reduction = p_blowout * 4.0  # Max -4 minutes at high spreads

        # Apply reduction scaled by role (starters/stars impacted less)
        if starter_est:
            # Starters still get ~32-34 mins even in blowouts; reduce less
            proj_min -= minutes_ceiling_reduction * 0.5
        else:
            # Bench players might get MORE minutes in blowouts, but projection stays conservative
            proj_min -= minutes_ceiling_reduction * 0.3

    # Injury beneficiary boost
    injury_minutes_boost = _safe_get(player_data, 'injury_minutes_delta', 0)
    if injury_minutes_boost == 0:
        # Fallback: use injury_adjustment_amount as proxy
        injury_adj = _safe_get(player_data, 'injury_adjustment_amount', 0)
        if injury_adj > 0:
            injury_minutes_boost = injury_adj * 0.5  # Rough conversion
    proj_min += min(injury_minutes_boost, 6.0)  # Cap at +6

    # Questionable/minutes restriction cap
    injury_status = player_data.get('injury_status')
    if injury_status and str(injury_status).lower() in ('questionable', 'doubtful'):
        proj_min = min(proj_min, 26)

    # Step D: Confidence (based on L5 volatility)
    l5_stddev = _safe_get(player_data, 'l5_minutes_stddev', 0)
    if l5_stddev > 8:
        minutes_confidence = 0.4  # Very volatile
    elif l5_stddev > 5:
        minutes_confidence = 0.6  # Somewhat volatile
    else:
        minutes_confidence = 0.85  # Stable

    return max(0, proj_min), minutes_confidence


def classify_tier(proj_minutes: float, starts_last_5: int, proj_ceiling: float,
                  usg_pct: Optional[float] = None) -> str:
    """
    Non-circular tier classification based on minutes and role, NOT proj_ppg.

    Tiers:
        STAR: High minutes + starter + high offensive control
        SIXTH_MAN: Bench but live for top-3 (tournament darts)
        ROLE: Starter or solid rotation player
        BENCH: Minutes-limited (unlikely top scorer)

    Key insight: Tier drives variance estimation -> affects P(top-3).
    Using proj_ppg here would create circular logic.

    Primary drivers:
    1. Projected minutes (most important)
    2. Starter status (from starts_last_5)
    3. Usage % (if available)
    4. Ceiling (tie-breaker only)

    Returns: 'STAR', 'ROLE', 'SIXTH_MAN', 'BENCH'
    """
    # STAR: High minutes + starter + either high usage or high ceiling
    if proj_minutes >= 34 and starts_last_5 >= 3:
        if (usg_pct and usg_pct >= 28) or proj_ceiling >= 45:
            return 'STAR'
        # High minutes starter but not elite usage/ceiling -> still ROLE
        return 'ROLE'

    # SIXTH_MAN: Bench minutes (24-30) + NOT starter + high offensive potential
    if 24 <= proj_minutes < 30 and starts_last_5 <= 2:
        if (usg_pct and usg_pct >= 26) or proj_ceiling >= 40:
            return 'SIXTH_MAN'

    # ROLE: Solid rotation player (24+ minutes OR 20+ with starter status)
    if proj_minutes >= 24:
        return 'ROLE'
    if proj_minutes >= 20 and starts_last_5 >= 2:
        return 'ROLE'

    # BENCH: Minutes-limited
    return 'BENCH'


def detect_role_change(l5_min: float, season_min: float) -> bool:
    """
    Detect if player's role has recently changed (minutes spike).

    A +6 minute jump from season average is significant.
    """
    if season_min <= 0:
        return False
    return l5_min >= season_min + 6


def get_minutes_band(proj_minutes: float) -> str:
    """Categorize projected minutes into bands for calibration lookup."""
    if proj_minutes >= 36:
        return '36-40'
    elif proj_minutes >= 32:
        return '32-36'
    elif proj_minutes >= 28:
        return '28-32'
    elif proj_minutes >= 24:
        return '24-28'
    elif proj_minutes >= 20:
        return '20-24'
    else:
        return '<20'


def calculate_historical_stddev(conn: sqlite3.Connection, days_back: int = 60) -> pd.DataFrame:
    """
    Calculate actual scoring stddev by tier and minutes band from historical data.

    This provides calibrated variance estimates based on real prediction residuals,
    rather than guessing from ceiling-floor spread.

    Returns DataFrame with columns:
    - tier, minutes_band, residual_std, mean_actual, sample_size

    Note: Requires predictions table to have tier and actual_ppg populated.
    """
    query = f"""
        SELECT
            p.tier,
            p.proj_minutes,
            p.projected_ppg,
            p.actual_ppg,
            p.actual_ppg - p.projected_ppg as residual
        FROM predictions p
        WHERE p.actual_ppg IS NOT NULL
          AND p.tier IS NOT NULL
          AND p.proj_minutes IS NOT NULL
          AND p.game_date >= date('now', '-{days_back} days')
    """

    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        # Table might not have required columns yet
        return pd.DataFrame()

    if df.empty or len(df) < 20:
        return pd.DataFrame()

    # Add minutes band
    df['minutes_band'] = df['proj_minutes'].apply(get_minutes_band)

    # Group by tier and minutes band
    result = df.groupby(['tier', 'minutes_band']).agg(
        residual_std=('residual', 'std'),
        mean_actual=('actual_ppg', 'mean'),
        mean_projected=('projected_ppg', 'mean'),
        sample_size=('residual', 'count')
    ).reset_index()

    return result


# Global cache for calibration table (refreshed once per session)
_CALIBRATION_TABLE: Optional[pd.DataFrame] = None
_CALIBRATION_CONN_ID: Optional[int] = None


def get_calibration_table(conn: sqlite3.Connection, days_back: int = 60) -> pd.DataFrame:
    """Get cached calibration table, computing if needed."""
    global _CALIBRATION_TABLE, _CALIBRATION_CONN_ID

    conn_id = id(conn)
    if _CALIBRATION_TABLE is None or _CALIBRATION_CONN_ID != conn_id:
        _CALIBRATION_TABLE = calculate_historical_stddev(conn, days_back)
        _CALIBRATION_CONN_ID = conn_id

    return _CALIBRATION_TABLE


def get_calibrated_stddev(tier: str, proj_minutes: float,
                          calibration_table: pd.DataFrame,
                          fallback_std: float = 6.0) -> Tuple[float, bool]:
    """
    Look up calibrated stddev from historical data.

    Returns:
        Tuple of (stddev, is_calibrated)
        - stddev: The calibrated or fallback standard deviation
        - is_calibrated: True if from real data, False if fallback

    Fallback hierarchy:
    1. Exact tier + minutes band match (sample >= 20)
    2. Same tier, adjacent minutes band (sample >= 10)
    3. Same minutes band, any tier (sample >= 20)
    4. Default fallback value
    """
    if calibration_table is None or calibration_table.empty:
        return fallback_std, False

    minutes_band = get_minutes_band(proj_minutes)

    # Try exact match
    row = calibration_table[
        (calibration_table['tier'] == tier) &
        (calibration_table['minutes_band'] == minutes_band)
    ]

    if len(row) > 0 and row['sample_size'].iloc[0] >= 20:
        return row['residual_std'].iloc[0], True

    # Try same tier, any minutes band with enough samples
    tier_rows = calibration_table[
        (calibration_table['tier'] == tier) &
        (calibration_table['sample_size'] >= 10)
    ]
    if len(tier_rows) > 0:
        # Use weighted average by sample size
        weights = tier_rows['sample_size']
        avg_std = (tier_rows['residual_std'] * weights).sum() / weights.sum()
        return avg_std, True

    # Try same minutes band, any tier
    band_rows = calibration_table[
        (calibration_table['minutes_band'] == minutes_band) &
        (calibration_table['sample_size'] >= 20)
    ]
    if len(band_rows) > 0:
        return band_rows['residual_std'].mean(), True

    # Fallback
    return fallback_std, False


def estimate_scoring_stddev(player: dict,
                            calibration_table: Optional[pd.DataFrame] = None) -> Tuple[float, bool]:
    """
    Estimate scoring variance, preferring calibrated historical data.

    Higher variance = better for top-3 probability (tail event).

    Hierarchy:
    1. Calibrated stddev from historical residuals (if available + sufficient samples)
    2. Ceiling-floor spread / 3.29 (99% CI estimate)
    3. 25% of projected ppg as last resort

    Additional adjustments:
    - Minutes volatility (uncertain minutes = higher variance)
    - Injury beneficiary status (fill-in role = higher variance)

    Returns:
        Tuple of (stddev, is_calibrated)
    """
    tier = player.get('tier', 'ROLE')
    proj_minutes = _safe_get(player, 'proj_minutes', 30)
    ceiling = _safe_get(player, 'proj_ceiling', 0)
    floor = _safe_get(player, 'proj_floor', 0)
    proj_ppg = _safe_get(player, 'projected_ppg', 20)

    # Try calibrated stddev first
    is_calibrated = False
    if calibration_table is not None and not calibration_table.empty:
        # Calculate ceiling-floor fallback for get_calibrated_stddev
        if ceiling and floor and ceiling > floor:
            fallback = (ceiling - floor) / 3.29
        else:
            fallback = proj_ppg * 0.25

        base_std, is_calibrated = get_calibrated_stddev(
            tier, proj_minutes, calibration_table, fallback_std=max(fallback, 4.0)
        )
    else:
        # No calibration table - use ceiling-floor estimate
        if ceiling and floor and ceiling > floor:
            base_std = (ceiling - floor) / 3.29  # 99% confidence interval
        else:
            base_std = proj_ppg * 0.25  # 25% of projection as fallback

    # Adjust for minutes volatility (uncertain minutes = higher variance)
    minutes_conf = _safe_get(player, 'minutes_confidence', 0.7)
    if minutes_conf < 0.5:
        base_std *= 1.3  # More volatile

    # Adjust for injury beneficiary (temporary role = higher variance)
    injury_adjusted = player.get('injury_adjusted')
    role_change = player.get('role_change')
    if injury_adjusted and not role_change:
        base_std *= 1.2  # Fill-in role is volatile

    return max(base_std, 2.0), is_calibrated


@dataclass
class RiskFlags:
    """Risk indicators for a player."""
    questionable: bool = False
    minutes_restriction: bool = False
    blowout_risk: bool = False
    back_to_back: bool = False

    def total_penalty(self) -> float:
        """Calculate total risk penalty."""
        penalty = 0.0
        if self.questionable:
            penalty += 5.0
        if self.minutes_restriction:
            penalty += 8.0
        if self.blowout_risk:
            penalty += 3.0
        if self.back_to_back:
            penalty += 2.0
        return penalty

    def flags_string(self) -> str:
        """Return emoji flags for display."""
        flags = []
        if self.questionable:
            flags.append("⚠️")
        if self.minutes_restriction:
            flags.append("🕐")
        if self.blowout_risk:
            flags.append("📉")
        if self.back_to_back:
            flags.append("😴")
        return "".join(flags) if flags else ""


class Top3Ranker:
    """
    Ranks players by probability of finishing in daily top 3.

    Two ranking modes:
    1. TopScorerScore: Fast heuristic optimized for upside
    2. Simulation: Monte Carlo estimation of P(top 3)
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._calibration_params = None

    # =========================================================================
    # Data Loading
    # =========================================================================

    def get_predictions_for_date(self, game_date: str) -> pd.DataFrame:
        """Load predictions for a specific date with all needed columns."""
        query = """
            SELECT
                p.prediction_id,
                p.player_id,
                p.player_name,
                p.team_name,
                p.opponent_name,
                p.opponent_def_rating,
                p.projected_ppg,
                p.proj_floor,
                p.proj_ceiling,
                p.proj_confidence,
                p.season_avg_ppg,
                p.recent_avg_3,
                p.recent_avg_5,
                p.dfs_score,
                p.dfs_grade,
                p.injury_adjusted,
                p.injury_adjustment_amount,
                p.opponent_injury_detected,
                p.opponent_injury_boost_projection,
                p.actual_ppg,
                il.status as injury_status
            FROM predictions p
            LEFT JOIN injury_list il ON p.player_id = il.player_id
                AND il.status IN ('questionable', 'doubtful', 'out')
            WHERE p.game_date = ?
        """
        return pd.read_sql_query(query, self.conn, params=[game_date])

    def get_calibration_params(self, days_back: int = 30) -> Tuple[float, float]:
        """
        Get linear calibration parameters (a, b) for: calibrated = a + b * projected.

        Returns cached params if available.
        """
        if self._calibration_params is not None:
            return self._calibration_params

        try:
            import prediction_calibration as pcal
            calibrator = pcal.PredictionCalibrator(self.conn)
            params = calibrator.fit_linear(days_back)
            self._calibration_params = (params.intercept, params.slope)
        except Exception:
            # Fallback: no calibration
            self._calibration_params = (0.0, 1.0)

        return self._calibration_params

    # =========================================================================
    # Risk Detection
    # =========================================================================

    def detect_risk_flags(
        self,
        player_id: int,
        game_date: str,
        injury_status: Optional[str] = None,
        vegas_spread: Optional[float] = None
    ) -> RiskFlags:
        """Detect risk factors for a player."""
        flags = RiskFlags()

        # Questionable/Doubtful status
        if injury_status and injury_status.lower() in ('questionable', 'doubtful'):
            flags.questionable = True

        # Minutes restriction (first game back after injury)
        # Check if player missed recent games
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM player_game_logs
            WHERE player_id = ?
              AND game_date >= date(?, '-7 days')
              AND game_date < ?
              AND points IS NOT NULL
        """, [player_id, game_date, game_date])
        recent_games = cursor.fetchone()[0]

        # If player played 0-2 games in last 7 days and has injury history, flag as restricted
        if recent_games <= 2 and injury_status:
            flags.minutes_restriction = True

        # Blowout risk (if vegas spread provided)
        if vegas_spread is not None and abs(vegas_spread) > 10:
            flags.blowout_risk = True

        # Back-to-back detection
        cursor.execute("""
            SELECT COUNT(*) FROM player_game_logs
            WHERE player_id = ?
              AND game_date = date(?, '-1 day')
              AND points IS NOT NULL
        """, [player_id, game_date])
        played_yesterday = cursor.fetchone()[0] > 0

        if played_yesterday:
            flags.back_to_back = True

        return flags

    # =========================================================================
    # TopScorerScore Heuristic
    # =========================================================================

    def calculate_top_scorer_score(
        self,
        row: pd.Series,
        cal_intercept: float,
        cal_slope: float
    ) -> Tuple[float, Dict]:
        """
        Calculate TopScorerScore optimized for top-3 identification.

        Philosophy: Focus on SUSTAINABLE factors, not chasing misleading hot streaks.

        Priority order (most to least important):
        1. Calibrated projection - foundation (~60% of score)
        2. Ceiling/upside - tournament edge
        3. Today's matchup quality - what matters NOW
        4. Role sustainability - GATED hot streak (ValidityFactor pattern)
        5. Star power - established scorers are more reliable

        Returns:
            Tuple of (score, component_breakdown)
        """
        # Convert to dict for safe .get() access (sqlite3.Row doesn't always support .get())
        if hasattr(row, 'to_dict'):
            row = row.to_dict()
        else:
            row = dict(row)

        components = {}

        # =================================================================
        # 1. CALIBRATED BASE (Foundation - ~60% of score)
        # =================================================================
        # This is our best estimate of true scoring ability TODAY
        # Already accounts for matchup, recent form via projection model
        calibrated_ppg = cal_intercept + cal_slope * row['projected_ppg']
        components['calibrated_base'] = calibrated_ppg

        # =================================================================
        # 2. CEILING BONUS (Tournament Edge - up to +8)
        # =================================================================
        # High ceiling = high variance = good for tournaments
        # But we scale by projection to avoid overrating low-volume players
        ceiling_upside = row['proj_ceiling'] - row['projected_ppg']
        # Only give full bonus if projection is high enough (20+ PPG)
        proj_factor = min(row['projected_ppg'] / 25.0, 1.0)  # Scales 0-1 for 0-25 PPG
        ceiling_bonus = min(ceiling_upside * 0.5 * proj_factor, 8.0)
        components['ceiling_bonus'] = ceiling_bonus

        # =================================================================
        # 3. SUSTAINABLE ROLE SCORE (ValidityFactor pattern)
        # =================================================================
        # Formula: RoleScore = RawHotBonus * ValidityFactor
        # ValidityFactor (0-1) crushes fake streaks
        #
        # Key insight: Don't add hot bonuses conditionally - multiply by validity
        # This prevents noise from competing with ceiling/matchup
        role_score = 0.0

        season_avg = row.get('season_avg_ppg') or 0
        if season_avg > 0:
            # Calculate raw hot bonus (what we WOULD give if trend is valid)
            recent_avg = row.get('recent_avg_5') or season_avg
            hot_ratio = recent_avg / season_avg

            # Raw bonus: +4 max for 20%+ above season (before validity gate)
            if hot_ratio > 1.05:
                raw_hot_bonus = min((hot_ratio - 1.0) * 20, 4.0)  # Cap at +4
            elif hot_ratio < 0.90:
                raw_hot_bonus = max((hot_ratio - 1.0) * 15, -3.0)  # Cold penalty
            else:
                raw_hot_bonus = 0.0

            # =============================================================
            # VALIDITY FACTOR (0.0 to 1.0) - gates the hot bonus
            # =============================================================
            validity_factor = 1.0  # Start at full validity

            # Check 1: STAR RETURNS DISCOUNT (most important)
            # If player's recent spike was during teammate absence, but teammate is back
            is_injury_beneficiary_today = (
                row.get('injury_adjusted') and
                row.get('injury_adjustment_amount') and
                row.get('injury_adjustment_amount') > 0
            )

            # If elevated stats but NO injury boost today = role likely contracting
            if hot_ratio > 1.10 and not is_injury_beneficiary_today:
                # Spike without ongoing injury context = very suspicious
                # Star teammate might have been out, now back
                validity_factor *= 0.25  # Crush to 25%

            # Check 2: MINUTES STABILITY
            # Only trust hot streaks if minutes support it
            # Note: We don't have L5 minutes in current schema, so we use
            # proj_confidence as a proxy (high confidence = stable minutes expectation)
            proj_confidence = row.get('proj_confidence') or 0.5
            if proj_confidence < 0.6 and hot_ratio > 1.10:
                # Low confidence in minutes + elevated stats = suspect
                validity_factor *= 0.5

            # Check 3: IF injury boost IS active, trust the streak more
            if is_injury_beneficiary_today and hot_ratio > 1.05:
                # Teammate still out, recent play elevated = makes sense
                validity_factor = max(validity_factor, 0.8)  # At least 80%

            # Apply validity factor to raw bonus
            role_score = raw_hot_bonus * validity_factor

        components['role_sustainability'] = round(role_score, 2)

        # =================================================================
        # 4. TODAY'S MATCHUP QUALITY (up to +5)
        # =================================================================
        # Opponent defense rating matters for TODAY, not what happened last week
        matchup_bonus = 0.0
        opp_def = row.get('opponent_def_rating')
        if opp_def:
            # League average is ~112. Higher = worse defense = bonus
            if opp_def >= 116:
                matchup_bonus = 5.0  # Elite matchup (bad defense)
            elif opp_def >= 114:
                matchup_bonus = 3.0  # Good matchup
            elif opp_def >= 112:
                matchup_bonus = 1.0  # Neutral
            elif opp_def <= 108:
                matchup_bonus = -3.0  # Tough matchup (elite defense)
            elif opp_def <= 110:
                matchup_bonus = -1.5  # Difficult matchup
        components['matchup_today'] = matchup_bonus

        # =================================================================
        # 5. INJURY BENEFICIARY (Only if teammate STILL out - up to +4)
        # =================================================================
        injury_boost = 0.0
        if row.get('injury_adjusted') and row.get('injury_adjustment_amount'):
            # Teammate is confirmed OUT today
            injury_boost = min(row['injury_adjustment_amount'] * 0.4, 4.0)

        # Opponent star out = easier scoring
        if row.get('opponent_injury_detected') and row.get('opponent_injury_boost_projection'):
            injury_boost += min(row['opponent_injury_boost_projection'] * 8, 2.0)

        components['injury_opportunity'] = injury_boost

        # =================================================================
        # 6. STAR POWER BONUS (Established scorers - up to +3)
        # =================================================================
        # Stars with 25+ season avg are more likely to have big games
        # They get more shots in close games, clutch situations
        star_bonus = 0.0
        if row['season_avg_ppg'] >= 28:
            star_bonus = 3.0  # Elite star
        elif row['season_avg_ppg'] >= 25:
            star_bonus = 2.0  # All-star level
        elif row['season_avg_ppg'] >= 22:
            star_bonus = 1.0  # Borderline star
        components['star_power'] = star_bonus

        # =================================================================
        # 7. RISK PENALTIES (up to -10)
        # =================================================================
        risk_penalty = 0.0

        # Questionable/Doubtful = might not play or limited minutes
        if row.get('injury_status') and row['injury_status'] in ('questionable', 'doubtful'):
            risk_penalty += 5.0

        # Blowout risk = stars get benched in 4th quarter
        if row.get('blowout_risk'):
            risk_penalty += 3.0

        # Back-to-back = fatigue, sometimes rest
        if row.get('back_to_back'):
            risk_penalty += 2.0

        components['risk_penalty'] = -risk_penalty

        # =================================================================
        # TOTAL SCORE
        # =================================================================
        total = (
            calibrated_ppg +      # Foundation (~60%)
            ceiling_bonus +        # Tournament edge (+0-8)
            role_score +           # Sustainable role (+/- 5)
            matchup_bonus +        # Today's matchup (+/- 5)
            injury_boost +         # Opportunity (+0-6)
            star_bonus +           # Star power (+0-3)
            risk_penalty           # Risks (-0-10)
        )

        return total, components

    def rank_by_top_scorer_score(
        self,
        game_date: str,
        include_components: bool = False
    ) -> pd.DataFrame:
        """
        Rank players by TopScorerScore for a given date.

        Returns DataFrame sorted by score descending with rank column.
        """
        df = self.get_predictions_for_date(game_date)

        if df.empty:
            return df

        # Get calibration params
        cal_intercept, cal_slope = self.get_calibration_params()

        # Calculate scores
        scores = []
        all_components = []

        for _, row in df.iterrows():
            score, components = self.calculate_top_scorer_score(
                row, cal_intercept, cal_slope
            )
            scores.append(score)
            all_components.append(components)

        df['top_scorer_score'] = scores

        if include_components:
            components_df = pd.DataFrame(all_components)
            df = pd.concat([df, components_df], axis=1)

        # Sort and rank
        df = df.sort_values('top_scorer_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)

        return df

    # =========================================================================
    # Monte Carlo Simulation
    # =========================================================================

    def simulate_top3_probability(
        self,
        game_date: str,
        n_simulations: int = 5000,
        use_calibration: bool = True,
        include_minutes: bool = True
    ) -> pd.DataFrame:
        """
        Estimate P(top 3) for each player via Monte Carlo simulation.

        Key insight: Top-3 is a TAIL EVENT. Higher variance = better probability
        of reaching the tail, even with lower mean.

        Distribution model:
        - Mean = calibrated projection
        - Std = estimate_scoring_stddev() incorporating:
          * Ceiling-floor spread (primary)
          * Minutes volatility (more uncertain = higher variance)
          * Injury beneficiary status (temporary role = higher variance)

        Returns DataFrame with player info and p_top3 column.
        """
        df = self.get_predictions_for_date(game_date)

        if df.empty:
            return df

        # Get calibration if enabled
        if use_calibration:
            cal_intercept, cal_slope = self.get_calibration_params()
        else:
            cal_intercept, cal_slope = 0.0, 1.0

        # Get historical stddev calibration table
        calibration_table = get_calibration_table(self.conn, days_back=60)

        # Prepare player data with improved variance estimation
        player_ids = df['player_id'].tolist()
        means = []
        stds = []
        tiers = []
        is_calibrated_flags = []

        for _, row in df.iterrows():
            # Convert row to dict for helper functions
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

            # Calculate projected minutes and tier FIRST (needed for stddev lookup)
            if include_minutes:
                proj_min, min_conf = calculate_projected_minutes(row_dict)
                row_dict['proj_minutes'] = proj_min
                row_dict['minutes_confidence'] = min_conf

                tier = classify_tier(
                    proj_min,
                    int(_safe_get(row_dict, 'starts_last_5', 0)),
                    _safe_get(row_dict, 'proj_ceiling', 0),
                    _safe_get(row_dict, 'avg_usg_last5')
                )
                row_dict['tier'] = tier
            else:
                tier = 'ROLE'
                proj_min = 30.0
                row_dict['tier'] = tier
                row_dict['proj_minutes'] = proj_min

            # Calculate mean (calibrated projection)
            mean = cal_intercept + cal_slope * _safe_get(row_dict, 'projected_ppg', 20)

            # Calculate variance using calibrated historical data when available
            std, is_calibrated = estimate_scoring_stddev(row_dict, calibration_table)

            means.append(mean)
            stds.append(std)
            tiers.append(tier)
            is_calibrated_flags.append(is_calibrated)

        means = np.array(means)
        stds = np.array(stds)

        # Build game and team groupings for correlated simulation
        # Game key = (team, opponent) - used for game-level factors
        # Team key = team - used for usage cannibalization
        player_teams = df['team_name'].tolist()
        player_opponents = df['opponent_name'].tolist()
        player_implied_totals = [
            _safe_get(row.to_dict() if hasattr(row, 'to_dict') else dict(row),
                      'implied_total', 220)
            for _, row in df.iterrows()
        ]

        # Create indices for quick team/game lookup
        team_to_indices = {}
        game_to_indices = {}
        for idx, (team, opp) in enumerate(zip(player_teams, player_opponents)):
            # Team grouping (for usage constraint)
            if team not in team_to_indices:
                team_to_indices[team] = []
            team_to_indices[team].append(idx)

            # Game grouping (for game factor)
            game_key = tuple(sorted([team, opp]))  # Normalize so MIA-BOS = BOS-MIA
            if game_key not in game_to_indices:
                game_to_indices[game_key] = []
            game_to_indices[game_key].append(idx)

        # Run simulation with correlation
        top3_counts = np.zeros(len(player_ids))
        top1_counts = np.zeros(len(player_ids))  # Track #1 scorer too

        for _ in range(n_simulations):
            # Step 1: Draw game factors (positive correlation within games)
            # Each game gets a random factor affecting all players
            # Factor drawn from normal(1.0, 0.08) = ±8% typical swing
            # Scaled by implied total (high-pace games have higher variance)
            game_factors = {}
            for game_key, indices in game_to_indices.items():
                # Use first player's implied total as game proxy
                implied_total = player_implied_totals[indices[0]]
                pace_factor = implied_total / 220  # Normalize to ~1.0
                game_factors[game_key] = np.random.normal(pace_factor, 0.08)

            # Step 2: Sample base points with game factor applied
            sampled_points = np.zeros(len(player_ids))
            for idx in range(len(player_ids)):
                team = player_teams[idx]
                opp = player_opponents[idx]
                game_key = tuple(sorted([team, opp]))
                gf = game_factors.get(game_key, 1.0)

                # Sample from normal distribution with game factor applied to mean
                base_mean = means[idx] * gf
                sampled = max(0, np.random.normal(base_mean, stds[idx]))
                sampled_points[idx] = sampled

            # Step 3: Apply team usage constraint (negative correlation)
            # When one teammate spikes (>1.2x mean), suppress others slightly
            for team, indices in team_to_indices.items():
                if len(indices) < 2:
                    continue  # No teammates to cannibalize

                for i, idx in enumerate(indices):
                    if sampled_points[idx] > means[idx] * 1.2:
                        # This player spiked - suppress teammates by 5%
                        for j, teammate_idx in enumerate(indices):
                            if i != j:
                                sampled_points[teammate_idx] *= 0.95

            # Step 4: Find top scorers
            sorted_indices = np.argsort(sampled_points)
            top3_indices = sorted_indices[-3:]
            top1_idx = sorted_indices[-1]

            # Increment counters
            for idx in top3_indices:
                top3_counts[idx] += 1
            top1_counts[top1_idx] += 1

        # Convert to probabilities
        p_top3 = top3_counts / n_simulations
        p_top1 = top1_counts / n_simulations

        # Add to dataframe
        df['p_top3'] = p_top3
        df['p_top3_pct'] = (p_top3 * 100).round(1)
        df['p_top1'] = p_top1
        df['p_top1_pct'] = (p_top1 * 100).round(1)
        df['scoring_stddev'] = stds
        df['tier'] = tiers
        df['stddev_calibrated'] = is_calibrated_flags

        # Sort by top-3 probability
        df = df.sort_values('p_top3', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)

        return df

    # =========================================================================
    # Combined Ranking
    # =========================================================================

    def rank_players(
        self,
        game_date: str,
        method: str = 'top_scorer_score',
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Rank players using specified method.

        Args:
            game_date: Date to rank for
            method: 'top_scorer_score', 'simulation', or 'both'
            n_simulations: Number of simulations (if using simulation)

        Returns:
            DataFrame with rankings
        """
        if method == 'top_scorer_score':
            return self.rank_by_top_scorer_score(game_date, include_components=True)
        elif method == 'simulation':
            return self.simulate_top3_probability(game_date, n_simulations)
        elif method == 'both':
            # Get both rankings and merge
            score_df = self.rank_by_top_scorer_score(game_date, include_components=True)
            sim_df = self.simulate_top3_probability(game_date, n_simulations)

            # Merge simulation results
            score_df = score_df.merge(
                sim_df[['player_id', 'p_top3', 'p_top3_pct']],
                on='player_id',
                how='left'
            )

            # Add simulation rank
            score_df['sim_rank'] = score_df['p_top3'].rank(ascending=False, method='min')

            return score_df
        else:
            raise ValueError(f"Unknown method: {method}")

    # =========================================================================
    # Utility Functions
    # =========================================================================

    def get_top_n(
        self,
        game_date: str,
        n: int = 3,
        method: str = 'top_scorer_score'
    ) -> List[Dict]:
        """
        Get top N players for a date.

        Returns list of dicts with player info and score.
        """
        df = self.rank_players(game_date, method=method)

        if df.empty:
            return []

        top_n = df.head(n)

        results = []
        for _, row in top_n.iterrows():
            results.append({
                'rank': int(row['rank']),
                'player_id': row['player_id'],
                'player_name': row['player_name'],
                'team': row['team_name'],
                'opponent': row['opponent_name'],
                'projected_ppg': row['projected_ppg'],
                'top_scorer_score': row.get('top_scorer_score'),
                'p_top3_pct': row.get('p_top3_pct'),
                'ceiling': row['proj_ceiling'],
            })

        return results


# =========================================================================
# Convenience Functions
# =========================================================================

def quick_top3_ranking(conn: sqlite3.Connection, game_date: str) -> None:
    """Print quick top 3 ranking to console."""
    ranker = Top3Ranker(conn)
    top3 = ranker.get_top_n(game_date, n=3, method='top_scorer_score')

    print(f"\n{'='*60}")
    print(f"TOP 3 SCORER PICKS FOR {game_date}")
    print(f"{'='*60}")

    if not top3:
        print("No predictions found for this date.")
        return

    for player in top3:
        print(f"\n{player['rank']}. {player['player_name']} ({player['team']})")
        print(f"   vs {player['opponent']}")
        print(f"   Projected: {player['projected_ppg']:.1f} PPG")
        print(f"   Ceiling: {player['ceiling']:.1f} PPG")
        print(f"   TopScorerScore: {player['top_scorer_score']:.1f}")
        if player.get('p_top3_pct'):
            print(f"   P(Top 3): {player['p_top3_pct']:.1f}%")
