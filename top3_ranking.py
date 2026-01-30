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


# =============================================================================
# MIXTURE MODEL FOR TAIL PROBABILITY
# =============================================================================
# Two-component mixture: typical game + spike game
# P(points) = (1-w)*N(μ, σ_typical) + w*N(μ+Δ, σ_spike)
#
# This captures "nuclear" outcomes better than single Normal.
# Key insight: role players don't just have high σ - they have a SPIKE MODE
# where everything clicks (hot shooting, extra minutes, OT).
# =============================================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi]."""
    return max(lo, min(hi, x))


def _sigmoid(z: float) -> float:
    """Standard sigmoid function."""
    # Clamp to avoid overflow
    z = _clamp(z, -20, 20)
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class MixtureParams:
    """Parameters for the two-component mixture model."""
    mu: float                # Mean projection
    sigma_typical: float     # Typical game stddev (your current σ)
    spike_weight: float      # w: probability of spike game (0.05 - 0.25)
    spike_shift: float       # Δ: how much higher the spike mean is
    sigma_spike: float       # Stddev in spike mode (slightly wider)

    # Debug info
    r3: float = 0.0          # Normalized 3PA rate
    u: float = 0.0           # Normalized usage
    vmin: float = 0.0        # Normalized minutes volatility
    role_up: float = 0.0     # Normalized role expansion

    def expected_value(self) -> float:
        """E[X] for mixture = (1-w)*μ + w*(μ+Δ)."""
        return self.mu + self.spike_weight * self.spike_shift


def calculate_spike_weight(
    fg3a: float,
    fga: float,
    fta: float,
    minutes: float,
    usg_pct: float,
    l5_minutes_stddev: float,
    avg_minutes_last5: float,
    season_avg_minutes: float,
    injury_adjusted: bool,
    opponent_pace: float = 100.0,
    vegas_spread: float = 0.0,
    minutes_confidence: float = 0.7,
    sigma_typical_multiplier: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate spike weight w using bounded logistic.

    Features (all normalized to [0,1]):
    - r3: 3PA rate (high shooters spike more)
    - u: usage % (high usage = more opportunity to spike)
    - vmin: minutes volatility (unstable = can get extra minutes)
    - role_up: recent minutes increase (expanded role = spike potential)
    - inj: injury beneficiary (teammate out = opportunity)
    - pace: game pace (fast games = more possessions = more variance)
    - ft: FTA per minute (drive-heavy players spike via FTs)
    - close_game: probability game stays close (amplifies spike chance)

    REFINEMENTS:
    - role_up is halved if σ_typical already got a ≥1.15x boost (double-count protection)
    - FTA signal added (drive-heavy players can spike via free throws)
    - Close-game amplifier (tight games = more crunch time = spike opportunity)

    Returns:
        (spike_weight, feature_dict for debugging)
    """
    # Feature normalization (all in [0,1])

    # r3: 3PA rate. 0.22→0.40 maps to 0→1
    three_pa_rate = fg3a / max(fga, 1.0) if fga > 0 else 0.0
    r3 = _clamp((three_pa_rate - 0.22) / 0.18, 0, 1)

    # u: usage. 18→30 maps to 0→1
    u = _clamp((usg_pct - 18) / 12, 0, 1) if usg_pct else 0.0

    # vmin: minutes volatility. 3→7 stddev maps to 0→1
    vmin = _clamp((l5_minutes_stddev - 3.0) / 4.0, 0, 1) if l5_minutes_stddev else 0.0

    # role_up: minutes increase. +0→+8 maps to 0→1
    minutes_delta = avg_minutes_last5 - season_avg_minutes if (avg_minutes_last5 and season_avg_minutes) else 0.0
    role_up_raw = _clamp(minutes_delta / 8.0, 0, 1)

    # DOUBLE-COUNT PROTECTION:
    # If σ_typical already got a ≥1.15x multiplier from role expansion,
    # halve role_up's contribution to the mixture model
    if sigma_typical_multiplier >= 1.15:
        role_up = role_up_raw * 0.5
    else:
        role_up = role_up_raw

    # inj: injury beneficiary (binary)
    inj = 1.0 if injury_adjusted else 0.0

    # pace: opponent pace. 97→103 maps to 0→1
    pace = _clamp((opponent_pace - 97) / 6, 0, 1) if opponent_pace else 0.5

    # ft: FTA per minute (drive-heavy players spike via free throws)
    # 0.10→0.28 FTA/min maps to 0→1
    fta_per_min = fta / max(minutes, 1.0) if minutes > 0 else 0.0
    ft = _clamp((fta_per_min - 0.10) / 0.18, 0, 1)

    # Logistic combination (with FT signal)
    z = (-2.1
         + 1.2 * r3
         + 0.9 * u
         + 0.7 * vmin
         + 0.7 * role_up
         + 0.6 * inj
         + 0.4 * pace
         + 0.6 * ft)  # FTA signal - catches drive-heavy spikes

    w_raw = _sigmoid(z)

    # Final weight: floor at 5%, cap at 25%
    w = _clamp(0.05 + 0.20 * w_raw, 0.05, 0.25)

    # CLOSE-GAME AMPLIFIER:
    # Spike nights are more likely to become top-15 when player stays on floor in crunch time
    # close_prob = sigmoid(1.2 - 0.25*|spread|)
    # Then: w = w * (0.9 + 0.3*close_prob), capped
    if vegas_spread is not None:
        close_prob = _sigmoid(1.2 - 0.25 * abs(vegas_spread))
        w = w * (0.9 + 0.3 * close_prob)
        w = _clamp(w, 0.05, 0.25)  # Re-clamp after amplifier

    # Guardrail: if low confidence + not injury + no role_up, cap at 12%
    # Prevents "random volatile bench" from dominating
    if minutes_confidence < 0.3 and not injury_adjusted and role_up < 0.2:
        w = min(w, 0.12)

    features = {
        'r3': r3,
        'u': u,
        'vmin': vmin,
        'role_up': role_up,
        'role_up_raw': role_up_raw,
        'inj': inj,
        'pace': pace,
        'ft': ft,
        'close_prob': close_prob if vegas_spread is not None else 0.5,
        'z': z,
        'w_raw': w_raw,
    }

    return w, features


def calculate_spike_shift(
    sigma_typical: float,
    r3: float,
    u: float,
    role_up: float,
    avg_fga_last5: float = None,
    season_fga: float = None,
    sigma_typical_multiplier: float = 1.0
) -> float:
    """
    Calculate spike shift Δ (how much higher the spike mean is).

    Scales with:
    - σ_typical (higher variance = bigger spikes possible)
    - 3PA rate (shooters can get hot)
    - Usage (high usage = more shots to convert)
    - Role expansion (more minutes = more points)
    - Shot volume increase (optional)

    REFINEMENTS:
    - Volume gate: low-FGA players can't get huge Δ even with high r3
    - role_up excluded from Δ if σ already got role expansion boost (double-count protection)

    Returns:
        Spike shift Δ in points (capped at 5-14)
    """
    # Optional shot volume increase
    shot = 0.0
    if avg_fga_last5 and season_fga and season_fga > 0:
        shot = _clamp((avg_fga_last5 - season_fga) / 6.0, 0, 1)

    # DOUBLE-COUNT PROTECTION:
    # If σ_typical already got role expansion boost, exclude role_up from Δ
    # (keep it only in spike weight w)
    role_up_for_delta = role_up if sigma_typical_multiplier < 1.15 else 0.0

    # Base delta calculation
    delta_raw = (4.0
                 + 0.45 * sigma_typical
                 + 3.0 * r3
                 + 2.0 * u
                 + 2.0 * role_up_for_delta
                 + 1.0 * shot)

    # VOLUME GATE:
    # A guy with 4 FGA/g shouldn't get +12 Δ
    # vol_gate = clamp((avg_fga_last5 - 6) / 10, 0, 1) → 6-16 FGA maps 0-1
    # Multiply: Δ = Δ * (0.6 + 0.4*vol_gate)
    if avg_fga_last5 is not None:
        vol_gate = _clamp((avg_fga_last5 - 6) / 10, 0, 1)
        delta = delta_raw * (0.6 + 0.4 * vol_gate)
    else:
        delta = delta_raw

    return _clamp(delta, 5.0, 14.0)


def calculate_sigma_spike(
    sigma_typical: float,
    r3: float,
    vmin: float
) -> float:
    """
    Calculate σ for spike component (slightly wider than typical).

    Spike games have extra variance from:
    - Hot shooting streaks (higher for 3PA shooters)
    - Minutes uncertainty (higher for volatile players)

    Returns:
        σ_spike (capped at 1.10x - 1.60x of typical)
    """
    multiplier = 1.15 + 0.20 * r3 + 0.10 * vmin
    sigma_spike = sigma_typical * multiplier

    # Clamp to reasonable range
    return _clamp(sigma_spike, sigma_typical * 1.10, sigma_typical * 1.60)


def build_mixture_params(
    player: dict,
    sigma_typical: float,
    mu: float,
    sigma_typical_multiplier: float = 1.0
) -> MixtureParams:
    """
    Build complete mixture model parameters for a player.

    Args:
        player: Player data dict with all features
        sigma_typical: Your current σ (after all multipliers)
        mu: Calibrated mean projection
        sigma_typical_multiplier: How much σ was boosted (for double-count protection)

    Returns:
        MixtureParams dataclass with all mixture components
    """
    # Extract features with safe defaults
    fg3a = _safe_get(player, 'avg_fg3a_last5') or _safe_get(player, 'avg_fg3a', 0)
    fga = _safe_get(player, 'avg_fga_last5') or _safe_get(player, 'avg_fga', 0)
    fta = _safe_get(player, 'avg_fta_last5') or _safe_get(player, 'avg_fta', 0)
    minutes = _safe_get(player, 'avg_minutes_last5') or _safe_get(player, 'avg_minutes', 30)
    usg_pct = _safe_get(player, 'usg_pct') or _safe_get(player, 'avg_usg_last5', 0)
    l5_minutes_stddev = _safe_get(player, 'l5_minutes_stddev', 4.0)
    avg_minutes_last5 = _safe_get(player, 'avg_minutes_last5', 0)
    season_avg_minutes = _safe_get(player, 'avg_minutes', 0)
    injury_adjusted = bool(player.get('injury_adjusted'))
    opponent_pace = _safe_get(player, 'opponent_pace', 100.0)
    vegas_spread = _safe_get(player, 'vegas_spread', 0.0)
    minutes_confidence = _safe_get(player, 'minutes_confidence', 0.7)

    # Calculate spike weight (with all refinements)
    w, features = calculate_spike_weight(
        fg3a=fg3a,
        fga=fga,
        fta=fta,
        minutes=minutes,
        usg_pct=usg_pct,
        l5_minutes_stddev=l5_minutes_stddev,
        avg_minutes_last5=avg_minutes_last5,
        season_avg_minutes=season_avg_minutes,
        injury_adjusted=injury_adjusted,
        opponent_pace=opponent_pace,
        vegas_spread=vegas_spread,
        minutes_confidence=minutes_confidence,
        sigma_typical_multiplier=sigma_typical_multiplier
    )

    # Calculate spike shift (with volume gate + double-count protection)
    delta = calculate_spike_shift(
        sigma_typical=sigma_typical,
        r3=features['r3'],
        u=features['u'],
        role_up=features['role_up'],
        avg_fga_last5=_safe_get(player, 'avg_fga_last5'),
        season_fga=_safe_get(player, 'season_fga'),
        sigma_typical_multiplier=sigma_typical_multiplier
    )

    # Calculate spike sigma
    sigma_spike = calculate_sigma_spike(
        sigma_typical=sigma_typical,
        r3=features['r3'],
        vmin=features['vmin']
    )

    return MixtureParams(
        mu=mu,
        sigma_typical=sigma_typical,
        spike_weight=w,
        spike_shift=delta,
        sigma_spike=sigma_spike,
        r3=features['r3'],
        u=features['u'],
        vmin=features['vmin'],
        role_up=features['role_up']
    )


def mixture_tail_probability(
    params: MixtureParams,
    threshold: float,
    p_cap: float = None,
    is_top3_threshold: bool = False
) -> float:
    """
    Calculate P(points ≥ T) using the two-component mixture.

    P = (1-w) * P_typical(X ≥ T) + w * P_spike(X ≥ T)

    Where:
    - P_typical uses N(μ, σ_typical)
    - P_spike uses N(μ + Δ, σ_spike)

    REFINEMENT: P_cap prevents hallucinating "55% chance to score 32"
    - For top-15 threshold: P_cap = 0.35
    - For top-3 threshold: P_cap = 0.20

    Args:
        params: MixtureParams with all model parameters
        threshold: Target threshold T (e.g., 32 points)
        p_cap: Manual cap on probability (overrides auto-detection)
        is_top3_threshold: If True, uses stricter cap (0.20 vs 0.35)

    Returns:
        Probability of exceeding threshold (0 to 1), capped
    """
    from scipy import stats

    w = params.spike_weight
    mu = params.mu

    # Typical component: P(X ≥ T) where X ~ N(μ, σ_typical)
    z_typical = (threshold - mu) / params.sigma_typical
    p_typical = 1.0 - stats.norm.cdf(z_typical)

    # Spike component: P(X ≥ T) where X ~ N(μ + Δ, σ_spike)
    z_spike = (threshold - (mu + params.spike_shift)) / params.sigma_spike
    p_spike = 1.0 - stats.norm.cdf(z_spike)

    # Mixture probability
    p_mixture = (1 - w) * p_typical + w * p_spike

    # P_CAP: prevent hallucinating extreme probabilities
    # Even the best player shouldn't have >35% chance of hitting top-15 threshold
    if p_cap is None:
        p_cap = 0.20 if is_top3_threshold else 0.35

    return min(p_mixture, p_cap)


def calculate_slate_threshold(
    projections: List[float],
    percentile: float = 92,
    offset: float = 2.0
) -> float:
    """
    Calculate slate-specific threshold T for "top scorer" probability.

    T = percentile(projections, pct) + offset

    This adapts to slate strength:
    - Big slates with stars: higher T
    - Small/slow slates: lower T

    Args:
        projections: List of all player projections on slate
        percentile: Which percentile to use (92 for top-15, 97 for top-3)
        offset: Additional points above percentile

    Returns:
        Threshold T in points
    """
    if not projections:
        return 30.0  # Fallback

    pct_value = np.percentile(projections, percentile)
    return pct_value + offset


# =============================================================================
# CALIBRATION SANITY CHECKS
# =============================================================================

def calculate_calibration_metrics(
    conn: sqlite3.Connection,
    days_back: int = 30,
    threshold_percentile: float = 92
) -> Dict[str, float]:
    """
    Calculate calibration metrics to detect tail drift.

    Key metrics:
    1. predicted_hits vs realized_hits (should match across slates)
    2. Brier score for P(X ≥ T)
    3. Calibration by decile

    If predicted_hits >> realized_hits → mixture is too "fat"
    If predicted_hits << realized_hits → mixture is too conservative

    Args:
        conn: Database connection
        days_back: How many days of history to analyze
        threshold_percentile: Percentile for threshold T

    Returns:
        Dict with calibration metrics
    """
    query = f"""
        SELECT
            game_date,
            player_id,
            player_name,
            projected_ppg,
            actual_ppg,
            p_threshold,
            threshold_T
        FROM predictions
        WHERE actual_ppg IS NOT NULL
          AND p_threshold IS NOT NULL
          AND game_date >= date('now', '-{days_back} days')
        ORDER BY game_date, p_threshold DESC
    """

    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        return {'error': 'Query failed or columns missing'}

    if df.empty or len(df) < 50:
        return {'error': 'Not enough data', 'n': len(df)}

    # Group by date to calculate per-slate metrics
    results = {
        'n_slates': 0,
        'n_predictions': len(df),
        'total_predicted_hits': 0.0,
        'total_realized_hits': 0,
        'brier_scores': [],
        'calibration_by_decile': {},
    }

    for game_date, slate_df in df.groupby('game_date'):
        results['n_slates'] += 1

        # Get threshold for this slate
        T = slate_df['threshold_T'].iloc[0] if 'threshold_T' in slate_df.columns else 30.0

        # Predicted hits = sum of all P(≥T) for ranked pool
        predicted_hits = slate_df['p_threshold'].sum()
        results['total_predicted_hits'] += predicted_hits

        # Realized hits = count of players who actually hit T
        realized_hits = (slate_df['actual_ppg'] >= T).sum()
        results['total_realized_hits'] += realized_hits

        # Brier score for each prediction
        for _, row in slate_df.iterrows():
            actual_hit = 1.0 if row['actual_ppg'] >= T else 0.0
            brier = (row['p_threshold'] - actual_hit) ** 2
            results['brier_scores'].append(brier)

    # Aggregate metrics
    results['predicted_hits_per_slate'] = results['total_predicted_hits'] / max(results['n_slates'], 1)
    results['realized_hits_per_slate'] = results['total_realized_hits'] / max(results['n_slates'], 1)
    results['hit_ratio'] = results['total_predicted_hits'] / max(results['total_realized_hits'], 1)
    results['mean_brier'] = np.mean(results['brier_scores']) if results['brier_scores'] else 0.0

    # Calibration check: predicted/realized ratio should be ~1.0
    if results['hit_ratio'] > 1.3:
        results['diagnosis'] = 'TAIL_TOO_FAT: predicted hits >> realized hits. Reduce intercept or spike weight.'
    elif results['hit_ratio'] < 0.7:
        results['diagnosis'] = 'TAIL_TOO_THIN: predicted hits << realized hits. Increase intercept or spike weight.'
    else:
        results['diagnosis'] = 'CALIBRATED: predicted/realized ratio within acceptable range.'

    # Calibration by decile (for top deciles)
    df_sorted = df.sort_values('p_threshold', ascending=False)
    n_per_decile = len(df_sorted) // 10

    for decile in range(1, 4):  # Top 3 deciles
        start_idx = (decile - 1) * n_per_decile
        end_idx = decile * n_per_decile
        decile_df = df_sorted.iloc[start_idx:end_idx]

        if len(decile_df) > 0:
            T = decile_df['threshold_T'].iloc[0] if 'threshold_T' in decile_df.columns else 30.0
            predicted_rate = decile_df['p_threshold'].mean()
            realized_rate = (decile_df['actual_ppg'] >= T).mean()
            results['calibration_by_decile'][f'decile_{decile}'] = {
                'predicted_rate': predicted_rate,
                'realized_rate': realized_rate,
                'n': len(decile_df),
            }

    # Clean up for return
    del results['brier_scores']

    return results


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

    TOURNAMENT STRATEGY INSIGHT:
    For tournaments, variance is GOOD. A 6th man with high σ can pop off
    and beat stars on any given night. We need to capture this by:
    1. Using tier-based σ multipliers (SIXTH_MAN gets bonus)
    2. Detecting role changes (L5 minutes >> season = expanded role)
    3. Accounting for game script uncertainty

    Hierarchy:
    1. Calibrated stddev from historical residuals (if available + sufficient samples)
    2. Ceiling-floor spread / 3.29 (99% CI estimate)
    3. 25% of projected ppg as last resort

    Additional adjustments:
    - Tier-based multipliers (SIXTH_MAN, BENCH have higher variance)
    - Role change detection (recent minutes spike = higher variance)
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

    # =========================================================================
    # TIER-BASED VARIANCE MULTIPLIERS
    # =========================================================================
    # Key insight: Different player tiers have inherently different variance
    # SIXTH_MAN: Can ghost (12 pts) or pop off (35 pts) - highest variance
    # BENCH: Limited ceiling but can DNP - high variance
    # STAR: More consistent usage, but game script dependent
    # ROLE: Most predictable minutes and usage
    tier_multipliers = {
        'SIXTH_MAN': 1.35,  # 6th men are tournament gold - can pop off
        'BENCH': 1.25,      # High variance due to minutes uncertainty
        'STAR': 1.10,       # Stars have variance from game script (blowouts)
        'ROLE': 1.00,       # Baseline - most predictable
    }
    tier_mult = tier_multipliers.get(tier, 1.0)
    base_std *= tier_mult

    # =========================================================================
    # ROLE CHANGE DETECTION
    # =========================================================================
    # If L5 minutes >> season average, player's role is expanding
    # This could be permanent (trade, injury) or temporary (hot streak)
    # Either way, it means higher variance as role stabilizes
    l5_minutes = _safe_get(player, 'avg_minutes_last5', 0)
    season_minutes = _safe_get(player, 'avg_minutes', 0)

    if season_minutes > 0 and l5_minutes > 0:
        minutes_ratio = l5_minutes / season_minutes
        if minutes_ratio >= 1.25:
            # +25% minutes = major role change
            base_std *= 1.30
        elif minutes_ratio >= 1.15:
            # +15% minutes = moderate role expansion
            base_std *= 1.15
        elif minutes_ratio <= 0.85:
            # -15% minutes = role contraction (also high variance)
            base_std *= 1.20

    # =========================================================================
    # EXISTING ADJUSTMENTS
    # =========================================================================
    # Adjust for minutes volatility (uncertain minutes = higher variance)
    minutes_conf = _safe_get(player, 'minutes_confidence', 0.7)
    if minutes_conf < 0.5:
        base_std *= 1.3  # More volatile

    # Adjust for injury beneficiary (temporary role = higher variance)
    injury_adjusted = player.get('injury_adjusted')
    role_change = player.get('role_change')
    if injury_adjusted and not role_change:
        base_std *= 1.2  # Fill-in role is volatile

    # Cap at reasonable maximum (3x base projection as 99th percentile)
    max_std = proj_ppg * 0.5 if proj_ppg > 10 else 6.0
    base_std = min(base_std, max_std)

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
        cal_slope: float,
        calibration_table: Optional[pd.DataFrame] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate TopScorerScore optimized for TOP-3 IDENTIFICATION using top-tail probability.

        TOURNAMENT STRATEGY CORE INSIGHT:
        ================================
        For tournaments where you win by having a player in the top 3 scorers,
        you should NOT rank by mean projection alone. You should rank by
        P(player scores >= threshold) or equivalently by top-tail probability.

        Mathematical approach: TopScorerScore = Proj + k * σ

        Where:
        - Proj = calibrated mean projection
        - σ = scoring standard deviation (higher = more upside potential)
        - k = aggressiveness factor (1.0-1.5 depending on contest type)

        This formula naturally:
        1. Rewards high-ceiling players (higher σ)
        2. Keeps stars at the top (higher Proj)
        3. Finds "nuclear" role players who can pop off (high σ relative to Proj)

        Additional modifiers (smaller magnitude):
        - Matchup quality (+/- 3 pts)
        - Injury opportunity (+0-4 pts)
        - Risk penalties (-0-5 pts)

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
        # 1. CALIBRATED BASE PROJECTION (μ)
        # =================================================================
        calibrated_ppg = cal_intercept + cal_slope * row['projected_ppg']
        components['calibrated_base'] = round(calibrated_ppg, 1)

        # =================================================================
        # 2. ESTIMATE SCORING VARIANCE (σ)
        # =================================================================
        # This is the KEY for tournament strategy:
        # Higher σ = more likely to hit the top tail
        #
        # We need to estimate σ using:
        # - Tier-based variance (SIXTH_MAN > STAR > ROLE > BENCH)
        # - Role change detection (L5 minutes vs season)
        # - Ceiling-floor spread
        # - Minutes volatility

        # Build player dict for estimate_scoring_stddev
        player_dict = {
            'tier': row.get('tier', 'ROLE'),
            'proj_minutes': row.get('proj_minutes', 30),
            'proj_ceiling': row.get('proj_ceiling', 0),
            'proj_floor': row.get('proj_floor', 0),
            'projected_ppg': row.get('projected_ppg', 20),
            'minutes_confidence': row.get('proj_confidence', 0.7),
            'injury_adjusted': row.get('injury_adjusted'),
            'role_change': row.get('role_change'),
            'avg_minutes_last5': row.get('avg_minutes_last5'),
            'avg_minutes': row.get('avg_minutes'),
        }

        # Calculate σ using the enhanced estimate_scoring_stddev function
        sim_sigma, is_calibrated = estimate_scoring_stddev(player_dict, calibration_table)
        components['sim_sigma'] = round(sim_sigma, 2)
        components['sigma_calibrated'] = is_calibrated

        # =================================================================
        # 3. TOP-TAIL PROBABILITY SCORE (Proj + k*σ)
        # =================================================================
        # The aggressiveness factor k determines how much we value variance:
        # - k = 0.8: Conservative (cash games)
        # - k = 1.0: Balanced
        # - k = 1.2: Aggressive (GPPs)
        # - k = 1.5: Very aggressive (winner-take-all)
        #
        # For top-3 scorer contests, we use k = 1.2 (aggressive)
        K_AGGRESSIVENESS = 1.2

        top_tail_score = calibrated_ppg + K_AGGRESSIVENESS * sim_sigma
        components['top_tail_score'] = round(top_tail_score, 1)
        components['k_factor'] = K_AGGRESSIVENESS

        # =================================================================
        # 4. ROLE TREND OVERRIDE (last 3-5 games)
        # =================================================================
        # If player's minutes have spiked 15%+ in L5, their role is expanding
        # This gets ADDED to top_tail_score, not just baked into σ
        role_trend_bonus = 0.0
        season_avg = row.get('season_avg_ppg') or 0

        if season_avg > 0:
            recent_avg = row.get('recent_avg_5') or season_avg
            hot_ratio = recent_avg / season_avg

            # Check if this is a VALID trend (injury-backed or minutes-backed)
            is_injury_beneficiary_today = (
                row.get('injury_adjusted') and
                row.get('injury_adjustment_amount') and
                row.get('injury_adjustment_amount') > 0
            )

            # Minutes-based role expansion check
            l5_minutes = row.get('avg_minutes_last5') or row.get('avg_minutes') or 0
            season_minutes = row.get('avg_minutes') or 0
            minutes_ratio = l5_minutes / season_minutes if season_minutes > 0 else 1.0

            # TRUST hot streaks if backed by role expansion
            if hot_ratio > 1.10:
                if is_injury_beneficiary_today:
                    # Teammate out = role expansion is REAL
                    role_trend_bonus = min((hot_ratio - 1.0) * 15, 3.0)
                elif minutes_ratio >= 1.15:
                    # Minutes up 15%+ = role expansion is REAL
                    role_trend_bonus = min((hot_ratio - 1.0) * 12, 2.5)
                else:
                    # Elevated stats without role backing = noise, small bonus only
                    role_trend_bonus = min((hot_ratio - 1.0) * 5, 1.0)
            elif hot_ratio < 0.85:
                # Cold streak - apply penalty
                role_trend_bonus = max((hot_ratio - 1.0) * 10, -2.0)

        components['role_trend'] = round(role_trend_bonus, 2)

        # =================================================================
        # 5. TODAY'S MATCHUP QUALITY (+/- 3)
        # =================================================================
        matchup_bonus = 0.0
        opp_def = row.get('opponent_def_rating')
        if opp_def:
            # League average is ~112. Higher = worse defense = bonus
            if opp_def >= 116:
                matchup_bonus = 3.0  # Elite matchup (bad defense)
            elif opp_def >= 114:
                matchup_bonus = 2.0  # Good matchup
            elif opp_def >= 112:
                matchup_bonus = 0.5  # Neutral
            elif opp_def <= 108:
                matchup_bonus = -2.5  # Tough matchup (elite defense)
            elif opp_def <= 110:
                matchup_bonus = -1.0  # Difficult matchup
        components['matchup_today'] = matchup_bonus

        # =================================================================
        # 6. INJURY BENEFICIARY (teammate STILL out - up to +4)
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
        # 7. RISK PENALTIES (up to -5)
        # =================================================================
        # NOTE: Reduced penalties vs old formula because:
        # - Questionable players have UPSIDE if they play (captured in σ)
        # - Blowout risk is already in minutes projection
        risk_penalty = 0.0

        # Questionable/Doubtful = might not play (but if they do, high ceiling)
        # Penalty reduced because σ already increased for these players
        if row.get('injury_status') and row['injury_status'] in ('questionable', 'doubtful'):
            risk_penalty += 2.0  # Reduced from 5.0

        # Blowout risk = stars get benched in 4th quarter
        if row.get('blowout_risk'):
            risk_penalty += 2.0  # Reduced from 3.0

        # Back-to-back = fatigue, sometimes rest
        if row.get('back_to_back'):
            risk_penalty += 1.0  # Reduced from 2.0

        components['risk_penalty'] = -risk_penalty

        # =================================================================
        # TOTAL SCORE (Top-Tail Formula)
        # =================================================================
        # TopScorerScore = Proj + k*σ + modifiers
        #
        # This naturally:
        # 1. Keeps stars at top (high Proj)
        # 2. Rewards high-ceiling players (high σ)
        # 3. Finds role player "nuclear" outcomes (high σ/Proj ratio)
        total = (
            top_tail_score +       # Core: Proj + k*σ (~85% of score)
            role_trend_bonus +     # Valid hot streaks (+/- 3)
            matchup_bonus +        # Today's matchup (+/- 3)
            injury_boost +         # Opportunity (+0-6)
            risk_penalty           # Risks (-0-5)
        )

        return total, components

    def rank_by_top_scorer_score(
        self,
        game_date: str,
        include_components: bool = False,
        k_aggressiveness: float = 1.2
    ) -> pd.DataFrame:
        """
        Rank players by TopScorerScore for a given date.

        Uses the top-tail probability formula: Score = Proj + k*σ + modifiers

        Args:
            game_date: Date to rank for
            include_components: If True, includes score breakdown columns
            k_aggressiveness: How much to weight variance (σ)
                - 0.8: Conservative (cash games)
                - 1.0: Balanced
                - 1.2: Aggressive (GPPs) [default]
                - 1.5: Very aggressive (winner-take-all)

        Returns DataFrame sorted by score descending with rank column.
        """
        df = self.get_predictions_for_date(game_date)

        if df.empty:
            return df

        # Get calibration params and table
        cal_intercept, cal_slope = self.get_calibration_params()
        calibration_table = get_calibration_table(self.conn, days_back=60)

        # Calculate scores
        scores = []
        all_components = []

        for _, row in df.iterrows():
            score, components = self.calculate_top_scorer_score(
                row, cal_intercept, cal_slope, calibration_table
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
    # Mixture Model Tail Probability Ranking
    # =========================================================================

    def rank_by_mixture_probability(
        self,
        game_date: str,
        threshold_percentile: float = 92,
        threshold_offset: float = 2.0,
        include_components: bool = False
    ) -> pd.DataFrame:
        """
        Rank players by P(points ≥ T) using two-component mixture model.

        This is the mathematically correct way to rank for "top-N scorer" contests.

        Model:
            P(points) = (1-w)*N(μ, σ_typical) + w*N(μ+Δ, σ_spike)

        Where:
        - w = spike_weight (5-25%, depends on 3PA rate, usage, role volatility)
        - Δ = spike_shift (5-14 points, depends on σ + shooting profile)
        - σ_spike = σ_typical * (1.15 to 1.60)

        The ranking score is P(points ≥ T) where T is slate-adaptive.

        Args:
            game_date: Date to rank for
            threshold_percentile: Percentile for threshold (92 for top-15, 97 for top-3)
            threshold_offset: Points to add above percentile
            include_components: If True, includes mixture model breakdown

        Returns:
            DataFrame sorted by P(≥T) descending with rank column
        """
        df = self.get_predictions_for_date(game_date)

        if df.empty:
            return df

        # Get calibration params
        cal_intercept, cal_slope = self.get_calibration_params()
        calibration_table = get_calibration_table(self.conn, days_back=60)

        # First pass: calculate means and σ for all players (needed for threshold)
        player_data_list = []

        for _, row in df.iterrows():
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

            # Calibrated mean
            mu = cal_intercept + cal_slope * _safe_get(row_dict, 'projected_ppg', 20)

            # Build player dict for σ estimation
            player_dict = {
                'tier': row_dict.get('tier', 'ROLE'),
                'proj_minutes': row_dict.get('proj_minutes', 30),
                'proj_ceiling': row_dict.get('proj_ceiling', 0),
                'proj_floor': row_dict.get('proj_floor', 0),
                'projected_ppg': row_dict.get('projected_ppg', 20),
                'minutes_confidence': row_dict.get('proj_confidence', 0.7),
                'injury_adjusted': row_dict.get('injury_adjusted'),
                'role_change': row_dict.get('role_change'),
                'avg_minutes_last5': row_dict.get('avg_minutes_last5'),
                'avg_minutes': row_dict.get('avg_minutes'),
                # Additional fields for mixture model
                'avg_fg3a_last5': row_dict.get('avg_fg3a_last5'),
                'avg_fg3a': row_dict.get('avg_fg3a'),
                'avg_fga_last5': row_dict.get('avg_fga_last5'),
                'avg_fga': row_dict.get('avg_fga'),
                'usg_pct': row_dict.get('usg_pct'),
                'avg_usg_last5': row_dict.get('avg_usg_last5'),
                'l5_minutes_stddev': row_dict.get('l5_minutes_stddev', 4.0),
                'opponent_pace': row_dict.get('opponent_pace', 100.0),
            }

            # Estimate σ_typical (your current σ)
            sigma_typical, _ = estimate_scoring_stddev(player_dict, calibration_table)

            player_data_list.append({
                'mu': mu,
                'sigma_typical': sigma_typical,
                'player_dict': player_dict,
                'row_dict': row_dict,
            })

        # Calculate slate threshold T
        all_projections = [p['mu'] for p in player_data_list]
        threshold_T = calculate_slate_threshold(
            all_projections,
            percentile=threshold_percentile,
            offset=threshold_offset
        )

        # Second pass: build mixture models and calculate P(≥T)
        probabilities = []
        all_components = []

        for pdata in player_data_list:
            # Build mixture params
            mixture = build_mixture_params(
                player=pdata['player_dict'],
                sigma_typical=pdata['sigma_typical'],
                mu=pdata['mu']
            )

            # Calculate tail probability
            p_tail = mixture_tail_probability(mixture, threshold_T)

            probabilities.append(p_tail)

            if include_components:
                all_components.append({
                    'mu': round(mixture.mu, 1),
                    'sigma_typical': round(mixture.sigma_typical, 2),
                    'spike_weight': round(mixture.spike_weight, 3),
                    'spike_shift': round(mixture.spike_shift, 1),
                    'sigma_spike': round(mixture.sigma_spike, 2),
                    'threshold_T': round(threshold_T, 1),
                    'p_tail': round(p_tail, 4),
                    'r3': round(mixture.r3, 2),
                    'u': round(mixture.u, 2),
                    'vmin': round(mixture.vmin, 2),
                    'role_up': round(mixture.role_up, 2),
                })

        # Add to dataframe
        df['p_threshold'] = probabilities
        df['p_threshold_pct'] = [p * 100 for p in probabilities]
        df['threshold_T'] = threshold_T

        if include_components:
            components_df = pd.DataFrame(all_components)
            df = pd.concat([df, components_df], axis=1)

        # Sort by probability descending
        df = df.sort_values('p_threshold', ascending=False).reset_index(drop=True)
        df['mixture_rank'] = range(1, len(df) + 1)

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
    # Portfolio Win Probability
    # =========================================================================

    def simulate_portfolio_win_probability(
        self,
        game_date: str,
        lineups: List[List[int]],
        n_simulations: int = 10000
    ) -> Tuple[float, Dict]:
        """
        Calculate P(any of N lineups has highest SUM) via Monte Carlo.

        This is the CORRECT win probability for winner-take-all contests where
        the winner is determined by the lineup with the highest combined total.

        Mathematical approach:
        For each simulation:
        1. Sample correlated scores for all players on the slate
        2. Calculate each lineup's SUM
        3. Calculate best possible 3-player SUM (top 3 individual scores)
        4. Check if any portfolio lineup equals the best

        Args:
            game_date: Date to simulate
            lineups: List of lineups, each lineup is a list of player_ids
            n_simulations: Number of Monte Carlo trials

        Returns:
            Tuple of (win_probability, stats_dict) where stats_dict contains:
            - wins_by_lineup: How many wins came from each lineup
            - avg_shortfall: Average points below optimal when we don't win
            - best_lineup_idx: Which lineup won most often
        """
        df = self.get_predictions_for_date(game_date)

        if df.empty or len(lineups) == 0:
            return 0.0, {}

        # Get calibration params
        cal_intercept, cal_slope = self.get_calibration_params()

        # Get historical stddev calibration
        calibration_table = get_calibration_table(self.conn, days_back=60)

        # Build player data arrays
        player_ids = df['player_id'].tolist()
        id_to_idx = {pid: i for i, pid in enumerate(player_ids)}
        n_players = len(player_ids)

        means = np.zeros(n_players)
        stds = np.zeros(n_players)

        for idx, (_, row) in enumerate(df.iterrows()):
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

            # Calculate projected minutes for tier classification
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

            # Mean = calibrated projection
            mean = cal_intercept + cal_slope * _safe_get(row_dict, 'projected_ppg', 20)
            means[idx] = mean

            # Variance from calibration or ceiling-floor
            std, _ = estimate_scoring_stddev(row_dict, calibration_table)
            stds[idx] = std

        # Convert lineups to index arrays (filter out unknown players)
        lineup_indices = []
        for lineup in lineups:
            indices = [id_to_idx[pid] for pid in lineup if pid in id_to_idx]
            if len(indices) >= 3:  # Only include valid lineups
                lineup_indices.append(indices[:3])  # Take first 3 if more

        if not lineup_indices:
            return 0.0, {}

        n_lineups = len(lineup_indices)

        # Build game/team structure for correlation
        player_teams = df['team_name'].tolist()
        player_opponents = df['opponent_name'].tolist()

        team_to_indices = {}
        game_to_indices = {}

        for idx, (team, opp) in enumerate(zip(player_teams, player_opponents)):
            if team not in team_to_indices:
                team_to_indices[team] = []
            team_to_indices[team].append(idx)

            game_key = tuple(sorted([team, opp]))
            if game_key not in game_to_indices:
                game_to_indices[game_key] = []
            game_to_indices[game_key].append(idx)

        # Run Monte Carlo
        portfolio_wins = 0
        wins_by_lineup = np.zeros(n_lineups)
        shortfalls = []  # Track how far off when we lose

        np.random.seed(42)  # Reproducibility

        for _ in range(n_simulations):
            # Step 1: Game factors (same-game correlation)
            game_factors = {}
            for game_key in game_to_indices:
                game_factors[game_key] = np.random.normal(1.0, 0.08)

            # Step 2: Sample scores with game correlation
            sampled = np.zeros(n_players)
            for idx in range(n_players):
                team = player_teams[idx]
                opp = player_opponents[idx]
                game_key = tuple(sorted([team, opp]))
                gf = game_factors.get(game_key, 1.0)

                base_mean = means[idx] * gf
                sampled[idx] = max(0, np.random.normal(base_mean, stds[idx]))

            # Step 3: Teammate usage constraint (negative correlation)
            for team, indices in team_to_indices.items():
                if len(indices) < 2:
                    continue
                for i, idx in enumerate(indices):
                    if sampled[idx] > means[idx] * 1.2:
                        for j, teammate_idx in enumerate(indices):
                            if i != j:
                                sampled[teammate_idx] *= 0.95

            # Step 4: Calculate optimal 3-player sum (top 3 scorers)
            top3_indices = np.argsort(sampled)[-3:]
            optimal_sum = np.sum(sampled[top3_indices])

            # Step 5: Calculate each lineup's sum, find our best
            best_lineup_sum = -np.inf
            best_lineup_idx = -1

            for lineup_idx, indices in enumerate(lineup_indices):
                lineup_sum = np.sum(sampled[indices])
                if lineup_sum > best_lineup_sum:
                    best_lineup_sum = lineup_sum
                    best_lineup_idx = lineup_idx

            # Step 6: Check if we won
            if best_lineup_sum >= optimal_sum - 0.01:
                portfolio_wins += 1
                wins_by_lineup[best_lineup_idx] += 1
            else:
                shortfalls.append(optimal_sum - best_lineup_sum)

        # Calculate statistics
        win_prob = portfolio_wins / n_simulations

        stats = {
            'wins_by_lineup': wins_by_lineup.tolist(),
            'total_wins': portfolio_wins,
            'avg_shortfall': np.mean(shortfalls) if shortfalls else 0.0,
            'max_shortfall': np.max(shortfalls) if shortfalls else 0.0,
            'best_lineup_idx': int(np.argmax(wins_by_lineup)),
            'best_lineup_wins': int(np.max(wins_by_lineup)),
            'n_simulations': n_simulations,
            'n_lineups_evaluated': n_lineups
        }

        return win_prob, stats

    # =========================================================================
    # Combined Ranking
    # =========================================================================

    def rank_players(
        self,
        game_date: str,
        method: str = 'mixture',
        n_simulations: int = 10000,
        threshold_percentile: float = 92
    ) -> pd.DataFrame:
        """
        Rank players using specified method.

        Args:
            game_date: Date to rank for
            method: 'mixture' (recommended), 'top_scorer_score', 'simulation', or 'both'
            n_simulations: Number of simulations (if using simulation)
            threshold_percentile: For mixture method (92 for top-15, 97 for top-3)

        Returns:
            DataFrame with rankings

        Methods:
        - 'mixture': Two-component mixture model with P(≥T) ranking (RECOMMENDED)
        - 'top_scorer_score': Proj + k*σ heuristic (fast, good approximation)
        - 'simulation': Monte Carlo P(top-3) estimation
        - 'both': top_scorer_score + simulation merged
        """
        if method == 'mixture':
            return self.rank_by_mixture_probability(
                game_date,
                threshold_percentile=threshold_percentile,
                include_components=True
            )
        elif method == 'top_scorer_score':
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
# Training Data Analysis
# =========================================================================

def analyze_missed_top_performers(
    conn: sqlite3.Connection,
    days_back: int = 60,
    top_n: int = 15,
    rank_threshold: int = 20
) -> pd.DataFrame:
    """
    Build training dataset of "Missed Top-N" cases.

    These are players who:
    - Finished in top N scorers for the slate (actual_ppg)
    - But were ranked BELOW threshold by our model

    This dataset helps identify:
    - What features predict underranked high performers
    - Which player types we systematically underrate
    - Role-player/rookie "nuclear" outcomes

    Args:
        conn: Database connection
        days_back: How many days of history to analyze
        top_n: What counts as "top performer" (default: 15)
        rank_threshold: Rank cutoff for "missed" (default: 20)
                        If player ranked >= 20 but finished top 15 = miss

    Returns:
        DataFrame with columns:
            - game_date, player_name, team_name
            - actual_ppg, actual_rank (in slate by points)
            - our_rank (by top_scorer_score)
            - projected_ppg, proj_ceiling, sim_sigma
            - tier, role_trend, injury_adjusted
            - miss_magnitude (how much we underranked them)
    """
    query = f"""
        WITH slate_rankings AS (
            SELECT
                game_date,
                player_id,
                player_name,
                team_name,
                actual_ppg,
                projected_ppg,
                proj_ceiling,
                proj_floor,
                top_scorer_score,
                tier,
                injury_adjusted,
                season_avg_ppg,
                recent_avg_5,
                RANK() OVER (PARTITION BY game_date ORDER BY actual_ppg DESC) as actual_rank,
                RANK() OVER (PARTITION BY game_date ORDER BY top_scorer_score DESC) as our_rank
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND top_scorer_score IS NOT NULL
              AND game_date >= date('now', '-{days_back} days')
        )
        SELECT *
        FROM slate_rankings
        WHERE actual_rank <= {top_n}
          AND our_rank > {rank_threshold}
        ORDER BY game_date DESC, actual_rank ASC
    """

    try:
        df = pd.read_sql_query(query, conn)

        if df.empty:
            return df

        # Calculate miss magnitude (how badly we underranked them)
        df['miss_magnitude'] = df['our_rank'] - df['actual_rank']

        # Calculate hot ratio (recent vs season)
        df['hot_ratio'] = df.apply(
            lambda r: r['recent_avg_5'] / r['season_avg_ppg']
            if r['season_avg_ppg'] and r['season_avg_ppg'] > 0 else 1.0,
            axis=1
        )

        # Calculate implied σ from ceiling-floor
        df['implied_sigma'] = (df['proj_ceiling'] - df['proj_floor']) / 3.29

        # Categorize the miss type
        def categorize_miss(row):
            if row['injury_adjusted']:
                return 'INJURY_BENEFICIARY'
            elif row['hot_ratio'] >= 1.15:
                return 'HOT_STREAK'
            elif row['season_avg_ppg'] < 15:
                return 'ROLE_PLAYER_NUCLEAR'
            elif row['implied_sigma'] >= 8:
                return 'HIGH_VARIANCE'
            else:
                return 'OTHER'

        df['miss_category'] = df.apply(categorize_miss, axis=1)

        return df

    except Exception as e:
        print(f"Error analyzing missed top performers: {e}")
        return pd.DataFrame()


def get_miss_category_summary(conn: sqlite3.Connection, days_back: int = 60) -> Dict:
    """
    Get summary statistics for missed top performers by category.

    Returns dict with:
        - total_misses: Total missed top-15 performers
        - by_category: Count by miss category
        - avg_miss_magnitude: How badly we underrank on average
        - most_common_tier: Which tier we miss most often
    """
    df = analyze_missed_top_performers(conn, days_back)

    if df.empty:
        return {
            'total_misses': 0,
            'by_category': {},
            'avg_miss_magnitude': 0,
            'most_common_tier': None
        }

    return {
        'total_misses': len(df),
        'by_category': df['miss_category'].value_counts().to_dict(),
        'avg_miss_magnitude': df['miss_magnitude'].mean(),
        'most_common_tier': df['tier'].mode().iloc[0] if not df['tier'].mode().empty else None,
        'top_missed_players': df.head(10)[['game_date', 'player_name', 'actual_rank', 'our_rank', 'miss_category']].to_dict('records')
    }


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
