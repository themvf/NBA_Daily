#!/usr/bin/env python3
"""
Player Correlation Model for Tournament Strategy

Models correlations between NBA players' scoring for portfolio optimization.

Key concepts:
- Same-game correlation: Players in the same game are positively correlated
  (if one game goes high-scoring/OT, multiple players benefit)
- Teammate negative correlation: Two teammates both scoring 40+ is rare
  (usage competition - only so many shots/possessions)
- OT boost: Games likely to go OT have stronger correlations

Mathematical approach:
- Build NxN correlation matrix
- Use Cholesky decomposition for correlated sampling
- Run Monte Carlo to estimate p_top1, p_top3 with correlation effects
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CorrelationConfig:
    """Tunable correlation parameters."""
    # Base correlations
    same_game_corr: float = 0.25       # Players in same game (different teams)
    teammate_corr: float = -0.10       # Teammates (usage competition)

    # Modifiers
    ot_game_boost: float = 0.15        # Additional correlation if OT likely
    high_total_boost: float = 0.05     # Boost for high-total games
    star_teammate_penalty: float = -0.05  # Extra penalty when pairing stars

    # Simulation settings
    n_simulations: int = 10000
    random_seed: Optional[int] = 42    # For reproducibility (None = random)


DEFAULT_CONFIG = CorrelationConfig()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlayerSlateInfo:
    """Player information needed for correlation modeling."""
    player_id: int
    player_name: str
    team: str
    game_id: str
    mean_score: float        # Projected PPG
    sigma: float             # Standard deviation (ceiling - floor) / 2
    is_star: bool = False    # Season avg > 25 PPG
    p_top1_independent: float = 0.0  # Pre-computed without correlation


@dataclass
class CorrelatedSimResult:
    """Results from correlated Monte Carlo simulation."""
    player_id: int
    player_name: str
    p_top1: float           # P(this player is #1 scorer)
    p_top3: float           # P(this player finishes top 3)
    expected_rank: float    # E[rank]
    support_score: float    # P(top-10 | not #1) - useful for lineup support
    sigma_used: float       # Sigma used in simulation


# ============================================================================
# Correlation Matrix Builder
# ============================================================================

class PlayerCorrelationModel:
    """
    Build and use player correlation matrices for Monte Carlo simulation.

    The correlation matrix captures:
    1. Same-game effects (positive): If a game goes high-scoring, both teams benefit
    2. Teammate effects (negative): Usage competition limits both going off
    3. Game environment modifiers: OT-likely games have stronger correlations
    """

    def __init__(self, config: CorrelationConfig = None):
        self.config = config or DEFAULT_CONFIG
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def build_correlation_matrix(
        self,
        players: List[PlayerSlateInfo],
        game_environments: Dict[str, dict] = None
    ) -> np.ndarray:
        """
        Build NxN correlation matrix for all players on the slate.

        Args:
            players: List of PlayerSlateInfo objects
            game_environments: Dict of game_id -> {ot_probability, stack_score, ...}

        Returns:
            NxN numpy array where corr[i,j] is correlation between player i and j
        """
        n = len(players)
        corr_matrix = np.eye(n)  # Start with identity (diagonal = 1)

        game_environments = game_environments or {}

        for i in range(n):
            for j in range(i + 1, n):
                p1 = players[i]
                p2 = players[j]

                corr = self._compute_pairwise_correlation(p1, p2, game_environments)

                # Ensure symmetry
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # Ensure positive semi-definite (required for Cholesky)
        corr_matrix = self._ensure_positive_definite(corr_matrix)

        return corr_matrix

    def _compute_pairwise_correlation(
        self,
        p1: PlayerSlateInfo,
        p2: PlayerSlateInfo,
        game_envs: Dict[str, dict]
    ) -> float:
        """
        Compute correlation between two players.

        Logic:
        - Same team (teammates): negative correlation (usage competition)
        - Same game, different teams: positive correlation (game environment)
        - Different games: zero correlation (independent)
        """
        # Different games = independent
        if p1.game_id != p2.game_id:
            return 0.0

        # Same game
        game_env = game_envs.get(p1.game_id, {})
        ot_prob = game_env.get('ot_probability', 0.06)
        stack_score = game_env.get('stack_score', 0.5)

        if p1.team == p2.team:
            # Teammates: negative correlation (usage competition)
            base_corr = self.config.teammate_corr

            # Extra penalty if both are stars (fighting for touches)
            if p1.is_star and p2.is_star:
                base_corr += self.config.star_teammate_penalty

            # But OT boosts everyone's opportunity slightly
            if ot_prob > 0.08:
                base_corr += 0.03  # Reduce negativity in OT-likely games

            return max(-0.30, base_corr)  # Floor to prevent extreme negative

        else:
            # Different teams in same game: positive correlation
            base_corr = self.config.same_game_corr

            # OT boost
            if ot_prob > 0.08:
                base_corr += self.config.ot_game_boost * (ot_prob / 0.12)

            # High stack score boost
            if stack_score > 0.7:
                base_corr += self.config.high_total_boost

            return min(0.50, base_corr)  # Cap to prevent extreme positive

    def _ensure_positive_definite(self, matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Ensure correlation matrix is positive semi-definite.

        Required for Cholesky decomposition. Uses eigenvalue adjustment if needed.
        """
        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(matrix)

        if np.min(eigenvalues) < epsilon:
            # Matrix is not positive definite - fix it
            # Add small value to diagonal to make positive definite
            n = matrix.shape[0]
            adjustment = epsilon - np.min(eigenvalues) + 0.001
            matrix = matrix + adjustment * np.eye(n)

            # Re-normalize diagonal to 1
            d = np.sqrt(np.diag(matrix))
            matrix = matrix / np.outer(d, d)

            warnings.warn(
                f"Correlation matrix was not positive definite. "
                f"Applied adjustment of {adjustment:.6f}"
            )

        return matrix

    def sample_correlated_scores(
        self,
        means: np.ndarray,
        sigmas: np.ndarray,
        corr_matrix: np.ndarray,
        n_sims: int = None
    ) -> np.ndarray:
        """
        Generate correlated score samples using Cholesky decomposition.

        Mathematical approach:
        1. Generate independent standard normal samples Z
        2. Transform: X = L @ Z where L = Cholesky(corr_matrix)
        3. Scale and shift: scores = means + sigmas * X

        Args:
            means: Array of projected scores (length n_players)
            sigmas: Array of standard deviations (length n_players)
            corr_matrix: NxN correlation matrix
            n_sims: Number of simulations (default from config)

        Returns:
            (n_sims, n_players) array of simulated scores
        """
        n_sims = n_sims or self.config.n_simulations
        n_players = len(means)

        # Cholesky decomposition: corr_matrix = L @ L.T
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Fallback if still not positive definite
            warnings.warn("Cholesky failed, using independent samples")
            L = np.eye(n_players)

        # Generate independent standard normal samples
        z = np.random.standard_normal((n_sims, n_players))

        # Transform to correlated samples
        correlated_z = z @ L.T

        # Scale by sigma and shift by mean
        scores = means + sigmas * correlated_z

        # Floor at 0 (can't score negative points)
        scores = np.maximum(scores, 0)

        return scores

    def run_correlated_simulation(
        self,
        players: List[PlayerSlateInfo],
        game_environments: Dict[str, dict] = None,
        n_sims: int = None
    ) -> List[CorrelatedSimResult]:
        """
        Run full correlated Monte Carlo simulation.

        This is the main entry point for simulation.

        Args:
            players: List of PlayerSlateInfo objects
            game_environments: Dict of game_id -> GameEnvironment dict
            n_sims: Number of simulations (default from config)

        Returns:
            List of CorrelatedSimResult for each player
        """
        n_sims = n_sims or self.config.n_simulations
        n_players = len(players)

        if n_players == 0:
            return []

        # Build correlation matrix
        corr_matrix = self.build_correlation_matrix(players, game_environments)

        # Extract means and sigmas
        means = np.array([p.mean_score for p in players])
        sigmas = np.array([p.sigma for p in players])

        # Run simulation
        scores = self.sample_correlated_scores(means, sigmas, corr_matrix, n_sims)

        # Compute rankings for each simulation
        # ranks[sim, player] = rank of that player in that simulation (1 = best)
        ranks = np.zeros((n_sims, n_players), dtype=int)
        for sim in range(n_sims):
            # argsort gives indices that would sort ascending, we want descending
            sorted_indices = np.argsort(-scores[sim])
            for rank, player_idx in enumerate(sorted_indices, 1):
                ranks[sim, player_idx] = rank

        # Compute statistics for each player
        results = []
        for i, player in enumerate(players):
            player_ranks = ranks[:, i]

            # P(#1)
            p_top1 = np.mean(player_ranks == 1)

            # P(top 3)
            p_top3 = np.mean(player_ranks <= 3)

            # Expected rank
            expected_rank = np.mean(player_ranks)

            # Support score: P(top-10 | not #1)
            # Useful for evaluating "support" players in lineups
            not_top1_mask = player_ranks > 1
            if np.sum(not_top1_mask) > 0:
                support_score = np.mean(player_ranks[not_top1_mask] <= 10)
            else:
                support_score = 1.0  # Always #1, perfect support

            results.append(CorrelatedSimResult(
                player_id=player.player_id,
                player_name=player.player_name,
                p_top1=round(p_top1, 4),
                p_top3=round(p_top3, 4),
                expected_rank=round(expected_rank, 2),
                support_score=round(support_score, 4),
                sigma_used=player.sigma
            ))

        return results


# ============================================================================
# Lineup Win Probability
# ============================================================================

def lineup_win_probability(
    player_ids: List[int],
    sim_results: Dict[int, CorrelatedSimResult],
    support_weight: float = 0.5
) -> float:
    """
    Calculate approximate probability that a lineup wins the slate.

    Formula:
    P(win) ≈ P(A is #1 OR B is #1 OR C is #1) × support_factor

    Where:
    - P(any is #1) = 1 - ∏(1 - p_top1[i])
    - support_factor weights the quality of non-#1 picks

    Args:
        player_ids: List of player IDs in the lineup
        sim_results: Dict of player_id -> CorrelatedSimResult
        support_weight: How much to weight support quality (0-1)

    Returns:
        Estimated win probability (0-1)
    """
    if not player_ids:
        return 0.0

    # P(at least one is #1) = 1 - P(none are #1)
    p_none_top1 = 1.0
    support_scores = []

    for pid in player_ids:
        result = sim_results.get(pid)
        if result:
            p_none_top1 *= (1 - result.p_top1)
            support_scores.append(result.support_score)

    p_any_top1 = 1 - p_none_top1

    # Support factor: rewards lineups where non-#1 picks are still strong
    if support_scores:
        avg_support = np.mean(support_scores)
    else:
        avg_support = 0.5

    # Combine: P(any is #1) weighted by support quality
    # support_weight=0.5 means support can contribute up to 50% bonus
    win_prob = p_any_top1 * (1 - support_weight + support_weight * avg_support)

    return round(win_prob, 4)


def rank_lineups_by_win_prob(
    lineups: List[List[int]],
    sim_results: Dict[int, CorrelatedSimResult]
) -> List[Tuple[List[int], float]]:
    """
    Rank candidate lineups by their win probability.

    Args:
        lineups: List of lineups, each lineup is a list of player IDs
        sim_results: Dict of player_id -> CorrelatedSimResult

    Returns:
        List of (lineup, win_prob) tuples, sorted by win_prob descending
    """
    results = []
    for lineup in lineups:
        win_prob = lineup_win_probability(lineup, sim_results)
        results.append((lineup, win_prob))

    return sorted(results, key=lambda x: x[1], reverse=True)


# ============================================================================
# Utility Functions
# ============================================================================

def create_player_slate_info(
    predictions_df: pd.DataFrame,
    game_assignments: Dict[int, str] = None
) -> List[PlayerSlateInfo]:
    """
    Convert predictions DataFrame to PlayerSlateInfo objects.

    Expected columns:
    - player_id, player_name, team_abbreviation
    - projected_ppg, proj_ceiling, proj_floor (or sigma)
    - season_avg_ppg (for is_star determination)

    Args:
        predictions_df: DataFrame with player predictions
        game_assignments: Optional dict of player_id -> game_id

    Returns:
        List of PlayerSlateInfo objects
    """
    players = []

    for _, row in predictions_df.iterrows():
        player_id = row['player_id']

        # Calculate sigma from ceiling/floor if not provided
        if 'sigma' in row and pd.notna(row['sigma']):
            sigma = row['sigma']
        else:
            ceiling = row.get('proj_ceiling', 0) or 0
            floor = row.get('proj_floor', 0) or 0
            sigma = (ceiling - floor) / 4 if ceiling > floor else row.get('projected_ppg', 20) * 0.15

        # Determine game_id
        if game_assignments and player_id in game_assignments:
            game_id = game_assignments[player_id]
        elif 'game_id' in row and pd.notna(row['game_id']):
            game_id = row['game_id']
        else:
            # Fallback: use team as game identifier (won't enable stacking)
            game_id = f"game_{row.get('team_abbreviation', 'UNK')}"

        players.append(PlayerSlateInfo(
            player_id=player_id,
            player_name=row['player_name'],
            team=row.get('team_abbreviation', 'UNK'),
            game_id=game_id,
            mean_score=row.get('projected_ppg', 0) or 0,
            sigma=max(sigma, 1.0),  # Floor sigma at 1 point
            is_star=row.get('season_avg_ppg', 0) >= 25
        ))

    return players


def compare_correlated_vs_independent(
    players: List[PlayerSlateInfo],
    game_environments: Dict[str, dict] = None,
    n_sims: int = 10000
) -> pd.DataFrame:
    """
    Compare simulation results with and without correlation.

    Useful for validating that correlation effects are meaningful.

    Returns:
        DataFrame comparing p_top1, p_top3 for correlated vs independent
    """
    # Correlated simulation
    model = PlayerCorrelationModel()
    correlated_results = model.run_correlated_simulation(
        players, game_environments, n_sims
    )

    # Independent simulation (identity correlation matrix)
    independent_config = CorrelationConfig(
        same_game_corr=0.0,
        teammate_corr=0.0,
        ot_game_boost=0.0,
        high_total_boost=0.0
    )
    independent_model = PlayerCorrelationModel(independent_config)
    independent_results = independent_model.run_correlated_simulation(
        players, game_environments, n_sims
    )

    # Build comparison DataFrame
    corr_dict = {r.player_id: r for r in correlated_results}
    indep_dict = {r.player_id: r for r in independent_results}

    rows = []
    for player in players:
        pid = player.player_id
        corr = corr_dict.get(pid)
        indep = indep_dict.get(pid)

        rows.append({
            'player_id': pid,
            'player_name': player.player_name,
            'team': player.team,
            'game_id': player.game_id,
            'p_top1_correlated': corr.p_top1 if corr else 0,
            'p_top1_independent': indep.p_top1 if indep else 0,
            'p_top1_delta': (corr.p_top1 - indep.p_top1) if corr and indep else 0,
            'p_top3_correlated': corr.p_top3 if corr else 0,
            'p_top3_independent': indep.p_top3 if indep else 0,
            'p_top3_delta': (corr.p_top3 - indep.p_top3) if corr and indep else 0,
        })

    return pd.DataFrame(rows)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Demo with synthetic data
    print("Correlation Model Demo")
    print("=" * 60)

    # Create sample players (2 games, 4 players each)
    players = [
        # Game 1: LAL vs GSW
        PlayerSlateInfo(1, "LeBron James", "LAL", "game_1", 28.0, 6.0, is_star=True),
        PlayerSlateInfo(2, "Anthony Davis", "LAL", "game_1", 26.0, 5.5, is_star=True),
        PlayerSlateInfo(3, "Stephen Curry", "GSW", "game_1", 27.0, 7.0, is_star=True),
        PlayerSlateInfo(4, "Draymond Green", "GSW", "game_1", 10.0, 3.0, is_star=False),
        # Game 2: BOS vs MIA
        PlayerSlateInfo(5, "Jayson Tatum", "BOS", "game_2", 29.0, 6.5, is_star=True),
        PlayerSlateInfo(6, "Jaylen Brown", "BOS", "game_2", 24.0, 5.0, is_star=False),
        PlayerSlateInfo(7, "Jimmy Butler", "MIA", "game_2", 22.0, 6.0, is_star=False),
        PlayerSlateInfo(8, "Bam Adebayo", "MIA", "game_2", 18.0, 4.0, is_star=False),
    ]

    # Game environments
    game_envs = {
        "game_1": {"ot_probability": 0.10, "stack_score": 0.85},  # High OT, good stack
        "game_2": {"ot_probability": 0.06, "stack_score": 0.50},  # Normal game
    }

    # Run comparison
    print("\nComparing Correlated vs Independent Simulation:")
    comparison = compare_correlated_vs_independent(players, game_envs, n_sims=10000)

    print(f"\n{'Player':<20} {'Team':<5} {'p_top1 (corr)':<12} {'p_top1 (ind)':<12} {'Delta':<8}")
    print("-" * 65)
    for _, row in comparison.iterrows():
        delta_str = f"{row['p_top1_delta']:+.2%}" if row['p_top1_delta'] != 0 else "0.00%"
        print(f"{row['player_name']:<20} {row['team']:<5} "
              f"{row['p_top1_correlated']:.2%}       {row['p_top1_independent']:.2%}       {delta_str}")

    # Test lineup win probability
    print("\n" + "=" * 60)
    print("Lineup Win Probability Demo:")

    model = PlayerCorrelationModel()
    results = model.run_correlated_simulation(players, game_envs)
    results_dict = {r.player_id: r for r in results}

    # Compare lineups
    lineups = [
        [1, 3, 5],  # LeBron, Curry, Tatum (all stars, different games)
        [1, 2, 5],  # LeBron, AD, Tatum (2 teammates)
        [3, 4, 5],  # Curry, Draymond, Tatum (game stack: GSW)
    ]

    print(f"\n{'Lineup':<45} {'Win Prob':<10}")
    print("-" * 55)
    for lineup in lineups:
        names = [p.player_name for p in players if p.player_id in lineup]
        win_prob = lineup_win_probability(lineup, results_dict)
        print(f"{', '.join(names):<45} {win_prob:.2%}")
