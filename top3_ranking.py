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

    # Blowout risk (spread-based, not binary)
    spread = abs(_safe_get(player_data, 'vegas_spread', 0))
    if spread >= 12:
        proj_min -= min((spread - 10) * 0.3, 4.0)  # Up to -4 for 24+ spread

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


def classify_tier(proj_minutes: float, proj_ppg: float, proj_ceiling: float,
                  usg_pct: Optional[float] = None) -> str:
    """
    Classify player into tier for tournament strategy.

    Tiers:
        STAR: High minutes + high offensive control (top scorer candidates)
        SIXTH_MAN: Bench but live for top-3 (tournament darts)
        ROLE: Starter or solid rotation player
        BENCH: Minutes-limited (unlikely top scorer)

    Returns: 'STAR', 'ROLE', 'SIXTH_MAN', 'BENCH'
    """
    # STAR: High minutes AND high offensive control
    if proj_minutes >= 34:
        if (usg_pct and usg_pct >= 28) or proj_ppg >= 24 or proj_ceiling >= 45:
            return 'STAR'

    # SIXTH_MAN: Bench minutes but live for top-3 (tournament dart)
    if 24 <= proj_minutes < 30:
        if (usg_pct and usg_pct >= 26) or proj_ceiling >= 38:
            return 'SIXTH_MAN'

    # ROLE: Starter or solid rotation
    if proj_minutes >= 24 or (proj_minutes >= 20 and proj_ppg >= 15):
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


def estimate_scoring_stddev(player: dict) -> float:
    """
    Estimate scoring variance from available data.

    Higher variance = better for top-3 probability (tail event).

    Uses ceiling-floor spread as primary signal, adjusted for:
    - Minutes volatility (uncertain minutes = higher variance)
    - Injury beneficiary status (fill-in role = higher variance)
    """
    # Primary: use ceiling-floor spread
    ceiling = _safe_get(player, 'proj_ceiling', 0)
    floor = _safe_get(player, 'proj_floor', 0)
    proj_ppg = _safe_get(player, 'projected_ppg', 20)

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

    return max(base_std, 2.0)


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

        # Prepare player data with improved variance estimation
        player_ids = df['player_id'].tolist()
        means = []
        stds = []
        tiers = []

        for _, row in df.iterrows():
            # Convert row to dict for helper functions
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

            # Calculate mean (calibrated projection)
            mean = cal_intercept + cal_slope * _safe_get(row_dict, 'projected_ppg', 20)

            # Calculate variance using improved estimation
            std = estimate_scoring_stddev(row_dict)

            # Calculate projected minutes and tier if enabled
            if include_minutes:
                proj_min, min_conf = calculate_projected_minutes(row_dict)
                row_dict['proj_minutes'] = proj_min
                row_dict['minutes_confidence'] = min_conf

                tier = classify_tier(
                    proj_min,
                    _safe_get(row_dict, 'projected_ppg', 0),
                    _safe_get(row_dict, 'proj_ceiling', 0),
                    _safe_get(row_dict, 'avg_usg_last5')
                )
            else:
                tier = 'ROLE'
                proj_min = 30.0

            means.append(mean)
            stds.append(std)
            tiers.append(tier)

        means = np.array(means)
        stds = np.array(stds)

        # Run simulation (vectorized for speed)
        top3_counts = np.zeros(len(player_ids))
        top1_counts = np.zeros(len(player_ids))  # Track #1 scorer too

        for _ in range(n_simulations):
            # Sample points for all players (truncated at 0)
            sampled_points = np.maximum(0, np.random.normal(means, stds))

            # Find indices of top 3 (highest last)
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
