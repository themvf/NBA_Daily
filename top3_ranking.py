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
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


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

        Returns:
            Tuple of (score, component_breakdown)
        """
        components = {}

        # 1. Calibrated base projection
        calibrated_ppg = cal_intercept + cal_slope * row['projected_ppg']
        components['calibrated_base'] = calibrated_ppg

        # 2. Ceiling bonus: reward upside potential
        ceiling_upside = row['proj_ceiling'] - row['projected_ppg']
        ceiling_bonus = min(ceiling_upside * 0.5, 8.0)  # Cap at +8
        components['ceiling_bonus'] = ceiling_bonus

        # 3. Hot streak bonus: L5 vs season average + L3 acceleration
        hot_bonus = 0.0
        if row['season_avg_ppg'] and row['season_avg_ppg'] > 0 and row['recent_avg_5']:
            # Base: L5 vs Season
            hot_ratio = row['recent_avg_5'] / row['season_avg_ppg']
            hot_bonus = max(0, (hot_ratio - 1.0) * 30)  # +6 for 20% above season
            hot_bonus = min(hot_bonus, 6.0)  # Cap at +6

            # Acceleration bonus: L3 > L5 means getting hotter
            if row.get('recent_avg_3') and row['recent_avg_3'] > row['recent_avg_5']:
                acceleration = (row['recent_avg_3'] / row['recent_avg_5']) - 1.0
                accel_bonus = min(acceleration * 20, 3.0)  # Up to +3 for accelerating
                hot_bonus += accel_bonus
        components['hot_streak_bonus'] = hot_bonus

        # 4. Minutes/Usage trend bonus: recent opportunity increases
        minutes_bonus = 0.0
        confidence = row['proj_confidence'] if row['proj_confidence'] else 0.7

        # Base confidence bonus
        if row['projected_ppg'] > 20:
            minutes_bonus = confidence * 3  # Stars get up to +3
        else:
            minutes_bonus = confidence * 1.5  # Role players get up to +1.5
        components['minutes_confidence_bonus'] = minutes_bonus

        # 5. Injury beneficiary bonus
        injury_boost = 0.0
        if row['injury_adjusted'] and row['injury_adjustment_amount']:
            injury_boost = min(row['injury_adjustment_amount'] * 0.5, 4.0)
        if row['opponent_injury_detected'] and row['opponent_injury_boost_projection']:
            injury_boost += min(row['opponent_injury_boost_projection'] * 10, 3.0)
        components['injury_beneficiary_bonus'] = injury_boost

        # 6. Risk penalty (calculated separately via detect_risk_flags)
        risk_penalty = 0.0
        if row.get('injury_status') and row['injury_status'] in ('questionable', 'doubtful'):
            risk_penalty += 5.0
        if row.get('blowout_risk'):
            risk_penalty += 3.0
        if row.get('back_to_back'):
            risk_penalty += 2.0
        components['risk_penalty'] = -risk_penalty

        # Total score
        total = (
            calibrated_ppg +
            ceiling_bonus +
            hot_bonus +
            minutes_bonus +
            injury_boost -
            risk_penalty
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
        n_simulations: int = 10000,
        use_calibration: bool = True
    ) -> pd.DataFrame:
        """
        Estimate P(top 3) for each player via Monte Carlo simulation.

        Assumes each player's points follow a truncated normal distribution:
        - Mean = calibrated projection
        - Std = (ceiling - floor) / 3.29  (99% within floor-ceiling)

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

        # Prepare player data
        player_ids = df['player_id'].tolist()
        means = []
        stds = []

        for _, row in df.iterrows():
            mean = cal_intercept + cal_slope * row['projected_ppg']
            # Standard deviation: assume 99% of outcomes within floor-ceiling
            # 99% is ±2.576 std, but we use 3.29 for a bit more spread
            ceiling = row['proj_ceiling'] if row['proj_ceiling'] else mean * 1.3
            floor = row['proj_floor'] if row['proj_floor'] else mean * 0.7
            std = max((ceiling - floor) / 3.29, 2.0)  # Minimum std of 2 pts

            means.append(mean)
            stds.append(std)

        means = np.array(means)
        stds = np.array(stds)

        # Run simulation
        top3_counts = np.zeros(len(player_ids))

        for _ in range(n_simulations):
            # Sample points for all players (truncated at 0)
            sampled_points = np.maximum(0, np.random.normal(means, stds))

            # Find indices of top 3
            top3_indices = np.argsort(sampled_points)[-3:]

            # Increment counters
            for idx in top3_indices:
                top3_counts[idx] += 1

        # Convert to probabilities
        probabilities = top3_counts / n_simulations

        # Add to dataframe
        df['p_top3'] = probabilities
        df['p_top3_pct'] = (probabilities * 100).round(1)

        # Sort by probability
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
