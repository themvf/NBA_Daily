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

        Key principles:
        1. Calibrated projection is the foundation (fixes historical bias)
        2. Ceiling matters for tournaments (high variance = good)
        3. Only trust "hot streaks" if the role is SUSTAINABLE
        4. TODAY's matchup matters more than last week's opponents
        5. Injury beneficiary bonus only if teammate is STILL out

        Returns:
            Tuple of (score, component_breakdown)
        """
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
        # 3. SUSTAINABLE ROLE SCORE (Replaces naive hot streak)
        # =================================================================
        # We ONLY reward recent performance if it's sustainable
        # Key question: Is this player's recent elevated play their TRUE role?
        role_score = 0.0

        if row['season_avg_ppg'] and row['season_avg_ppg'] > 0:
            # Check if this player is an injury beneficiary whose boost is ENDING
            is_injury_beneficiary_today = (
                row.get('injury_adjusted') and
                row.get('injury_adjustment_amount') and
                row['injury_adjustment_amount'] > 0
            )

            # Get recent vs season ratio
            recent_avg = row.get('recent_avg_5') or row['season_avg_ppg']
            hot_ratio = recent_avg / row['season_avg_ppg']

            if is_injury_beneficiary_today:
                # Teammate is OUT today - their elevated role continues
                # Trust the hot streak IF it aligns with injury timeline
                if hot_ratio > 1.05:
                    # Recent stats are elevated AND teammate still out = trust it
                    role_score = min((hot_ratio - 1.0) * 25, 5.0)
            else:
                # No injury boost today - be skeptical of elevated recent stats
                # They might have been filling in for someone who's back
                if hot_ratio > 1.15:
                    # Very elevated without injury context = suspicious
                    # Could be: weak opponents, teammate was out but now back
                    # Give minimal credit, let projection handle it
                    role_score = min((hot_ratio - 1.0) * 10, 2.0)
                elif hot_ratio > 1.05:
                    # Modestly elevated = could be real improvement
                    role_score = min((hot_ratio - 1.0) * 15, 2.0)
                elif hot_ratio < 0.90:
                    # Cold streak for non-injury-beneficiary = real concern
                    role_score = max((hot_ratio - 1.0) * 15, -3.0)

        components['role_sustainability'] = role_score

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
