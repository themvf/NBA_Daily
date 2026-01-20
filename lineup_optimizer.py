#!/usr/bin/env python3
"""
Tournament Lineup Optimizer for Winner-Take-All Contests

Builds 20 diversified lineups using a 4-bucket strategy:
1. Chalk Win (6): Highest p(#1) star + 2 strong supports
2. Game Stack (6): 2 players from high-env game + 1 elite #1
3. Leverage/Volatility (6): High-σ players with real ceiling
4. News Pivots (2): Questionable situations, injury beneficiaries

Key concepts:
- Maximize P(any of 20 lineups wins), not expected score
- Exposure caps prevent over-reliance on single players
- Correlation-aware stacking exploits game environments
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import warnings

from correlation_model import (
    PlayerCorrelationModel,
    CorrelatedSimResult,
    PlayerSlateInfo,
    lineup_win_probability,
    CorrelationConfig
)
from typing import FrozenSet


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OptimizerConfig:
    """Configuration for lineup optimizer."""
    # Number of lineups per bucket
    chalk_lineups: int = 6
    stack_lineups: int = 6
    leverage_lineups: int = 6
    news_lineups: int = 2

    # Exposure caps (fraction of total lineups)
    max_exposure_dominant: float = 0.60   # p_top1 > 15%: up to 12/20
    max_exposure_strong: float = 0.50     # p_top1 5-15%: up to 10/20
    max_exposure_leverage: float = 0.30   # p_top1 < 5%: up to 6/20

    # Thresholds
    star_p_top1_threshold: float = 0.05   # Minimum p_top1 to be considered "star"
    stack_score_threshold: float = 0.70   # Minimum stack_score for game stacking
    leverage_sigma_multiplier: float = 1.5 # σ > mean * this = high leverage

    # Lineup constraints
    lineup_size: int = 3                  # Players per lineup
    min_unique_players: int = 15          # Across all 20 lineups

    # Duplication penalty - prevents correlated fates
    max_core_overlap: int = 3             # Max lineups sharing same 2-player core
    min_portfolio_diversity: float = 0.60  # Minimum avg pairwise Jaccard distance


DEFAULT_CONFIG = OptimizerConfig()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlayerPool:
    """Player pool with simulation results and metadata."""
    player_id: int
    player_name: str
    team: str
    game_id: str
    projected_ppg: float
    sigma: float
    ceiling: float
    floor: float
    p_top1: float
    p_top3: float
    support_score: float
    expected_rank: float
    is_star: bool = False
    is_questionable: bool = False
    is_injury_beneficiary: bool = False
    ownership_pct: float = 0.0  # For leverage calculation


@dataclass
class Lineup:
    """A single 3-player lineup."""
    players: List[PlayerPool]
    bucket: str  # "chalk", "stack", "leverage", "news"
    win_probability: float = 0.0
    stack_game: Optional[str] = None  # Game ID if this is a stack lineup

    def player_ids(self) -> List[int]:
        return [p.player_id for p in self.players]

    def player_names(self) -> List[str]:
        return [p.player_name for p in self.players]

    def total_ceiling(self) -> float:
        return sum(p.ceiling for p in self.players)

    def avg_p_top1(self) -> float:
        return np.mean([p.p_top1 for p in self.players])


@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    lineups: List[Lineup]
    exposure_report: Dict[int, dict]  # player_id -> {count, pct, name}
    bucket_summary: Dict[str, int]    # bucket -> count
    total_win_probability: float      # P(any lineup wins)
    unique_players: int
    warnings: List[str] = field(default_factory=list)
    # Diversity metrics (duplication penalty)
    diversity_score: float = 1.0      # Avg pairwise Jaccard distance (higher = more diverse)
    core_saturation: Dict[str, Any] = field(default_factory=dict)  # Core usage report


# ============================================================================
# Exposure Manager
# ============================================================================

class ExposureManager:
    """
    Track and enforce player exposure across the portfolio.

    Exposure = fraction of lineups containing a player.
    Different player tiers have different max exposures.
    """

    def __init__(self, total_lineups: int, config: OptimizerConfig = None):
        self.total_lineups = total_lineups
        self.config = config or DEFAULT_CONFIG
        self.exposure_counts: Dict[int, int] = defaultdict(int)
        self.player_tiers: Dict[int, str] = {}  # player_id -> tier

    def set_player_tier(self, player_id: int, p_top1: float) -> str:
        """Assign exposure tier based on p_top1."""
        if p_top1 >= 0.15:
            tier = "dominant"
        elif p_top1 >= 0.05:
            tier = "strong"
        else:
            tier = "leverage"

        self.player_tiers[player_id] = tier
        return tier

    def get_max_exposure(self, player_id: int) -> int:
        """Get maximum lineup count for a player."""
        tier = self.player_tiers.get(player_id, "leverage")

        if tier == "dominant":
            max_pct = self.config.max_exposure_dominant
        elif tier == "strong":
            max_pct = self.config.max_exposure_strong
        else:
            max_pct = self.config.max_exposure_leverage

        return int(max_pct * self.total_lineups)

    def can_add_player(self, player_id: int) -> bool:
        """Check if player can be added to another lineup."""
        current = self.exposure_counts[player_id]
        max_allowed = self.get_max_exposure(player_id)
        return current < max_allowed

    def add_player(self, player_id: int) -> None:
        """Record player added to a lineup."""
        self.exposure_counts[player_id] += 1

    def remove_player(self, player_id: int) -> None:
        """Remove player from a lineup (for backtracking)."""
        if self.exposure_counts[player_id] > 0:
            self.exposure_counts[player_id] -= 1

    def get_exposure_pct(self, player_id: int) -> float:
        """Get current exposure percentage."""
        return self.exposure_counts[player_id] / self.total_lineups

    def get_exposure_report(self, player_pool: Dict[int, PlayerPool]) -> Dict[int, dict]:
        """Generate exposure report for all players used."""
        report = {}
        for pid, count in self.exposure_counts.items():
            if count > 0:
                player = player_pool.get(pid)
                report[pid] = {
                    'player_name': player.player_name if player else f"ID:{pid}",
                    'team': player.team if player else "?",
                    'count': count,
                    'pct': count / self.total_lineups,
                    'tier': self.player_tiers.get(pid, "unknown"),
                    'max_allowed': self.get_max_exposure(pid),
                    'at_cap': count >= self.get_max_exposure(pid)
                }
        return report


# ============================================================================
# Tournament Lineup Optimizer
# ============================================================================

class TournamentLineupOptimizer:
    """
    Build 20 diversified lineups for winner-take-all tournaments.

    Strategy:
    - Bucket 1 (Chalk): Anchor on highest p_top1 stars
    - Bucket 2 (Stack): Exploit game correlations
    - Bucket 3 (Leverage): High-σ contrarian plays
    - Bucket 4 (News): Pivot on uncertainty
    """

    def __init__(
        self,
        player_pool: List[PlayerPool],
        game_environments: Dict[str, dict],
        sim_results: Dict[int, CorrelatedSimResult] = None,
        config: OptimizerConfig = None
    ):
        self.pool = {p.player_id: p for p in player_pool}
        self.pool_list = player_pool
        self.game_envs = game_environments
        self.sim_results = sim_results or {}
        self.config = config or DEFAULT_CONFIG

        # Initialize exposure manager
        total_lineups = (
            self.config.chalk_lineups +
            self.config.stack_lineups +
            self.config.leverage_lineups +
            self.config.news_lineups
        )
        self.exposure_mgr = ExposureManager(total_lineups, self.config)

        # Set player tiers
        for p in player_pool:
            self.exposure_mgr.set_player_tier(p.player_id, p.p_top1)

        # Index players by game
        self.players_by_game: Dict[str, List[PlayerPool]] = defaultdict(list)
        for p in player_pool:
            self.players_by_game[p.game_id].append(p)

        # Index players by team
        self.players_by_team: Dict[str, List[PlayerPool]] = defaultdict(list)
        for p in player_pool:
            self.players_by_team[p.team].append(p)

        # Track unique lineups to prevent duplicates
        self.seen_lineups: Set[FrozenSet[int]] = set()

        # Track 2-player cores to prevent correlated fates (duplication penalty)
        # Key: frozenset of 2 player_ids, Value: count of lineups using this core
        self.core_counts: Dict[FrozenSet[int], int] = defaultdict(int)

    def _calculate_lineup_win_prob(self, player_ids: List[int]) -> float:
        """Calculate win probability for a lineup."""
        if self.sim_results:
            return lineup_win_probability(player_ids, self.sim_results)

        # Fallback: simple approximation from p_top1
        p_none_top1 = 1.0
        for pid in player_ids:
            p = self.pool.get(pid)
            if p:
                p_none_top1 *= (1 - p.p_top1)
        return 1 - p_none_top1

    def _is_unique_lineup(self, player_ids: List[int]) -> bool:
        """Check if lineup is unique (not seen before)."""
        lineup_key = frozenset(player_ids)
        return lineup_key not in self.seen_lineups

    def _register_lineup(self, player_ids: List[int]) -> bool:
        """
        Register a lineup as seen. Returns True if newly registered, False if duplicate.
        """
        lineup_key = frozenset(player_ids)
        if lineup_key in self.seen_lineups:
            return False
        self.seen_lineups.add(lineup_key)
        return True

    # =========================================================================
    # DUPLICATION PENALTY - 2-Player Core Tracking
    # =========================================================================

    def _get_all_cores(self, player_ids: List[int]) -> List[FrozenSet[int]]:
        """
        Get all 2-player cores from a lineup.

        For a 3-player lineup [A, B, C], returns cores:
        - {A, B}, {A, C}, {B, C}
        """
        from itertools import combinations
        return [frozenset(pair) for pair in combinations(player_ids, 2)]

    def _can_add_lineup_core(self, player_ids: List[int]) -> bool:
        """
        Check if adding this lineup would violate core overlap limits.

        Returns True if all 2-player cores have < max_core_overlap lineups.
        """
        max_overlap = self.config.max_core_overlap
        for core in self._get_all_cores(player_ids):
            if self.core_counts[core] >= max_overlap:
                return False
        return True

    def _register_lineup_cores(self, player_ids: List[int]) -> None:
        """Register all 2-player cores from a lineup."""
        for core in self._get_all_cores(player_ids):
            self.core_counts[core] += 1

    def _calculate_portfolio_diversity(self, lineups: List[Lineup]) -> float:
        """
        Calculate average pairwise Jaccard distance across portfolio.

        Jaccard distance = 1 - |A ∩ B| / |A ∪ B|

        Higher = more diverse (0 = identical, 1 = no overlap)
        """
        if len(lineups) < 2:
            return 1.0  # Single lineup = perfectly diverse (no overlap)

        distances = []
        for i in range(len(lineups)):
            for j in range(i + 1, len(lineups)):
                set_a = set(p.player_id for p in lineups[i].players)
                set_b = set(p.player_id for p in lineups[j].players)
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                if union > 0:
                    jaccard_distance = 1 - (intersection / union)
                    distances.append(jaccard_distance)

        return sum(distances) / len(distances) if distances else 1.0

    def _get_core_saturation_report(self) -> Dict[str, Any]:
        """
        Report on core usage across portfolio.

        Returns dict with:
        - max_core_usage: highest count for any 2-player core
        - saturated_cores: list of cores at max_core_overlap limit
        - core_distribution: histogram of core counts
        """
        if not self.core_counts:
            return {'max_core_usage': 0, 'saturated_cores': [], 'core_distribution': {}}

        max_usage = max(self.core_counts.values())
        saturated = [
            core for core, count in self.core_counts.items()
            if count >= self.config.max_core_overlap
        ]

        # Distribution: {count: num_cores_with_that_count}
        from collections import Counter
        dist = dict(Counter(self.core_counts.values()))

        return {
            'max_core_usage': max_usage,
            'saturated_cores': len(saturated),
            'core_distribution': dist
        }

    def _calculate_portfolio_win_prob_montecarlo(
        self,
        lineups: List[Lineup],
        n_sims: int = 10000
    ) -> float:
        """
        Calculate portfolio win probability via Monte Carlo simulation.

        CRITICAL: The contest winner is the lineup with the highest COMBINED SUM
        of 3 players' points - NOT the lineup containing the #1 individual scorer.

        This correctly handles:
        1. All lineups evaluated on the SAME simulated slate outcome each trial
        2. Portfolio wins if ANY of its lineups achieves the highest possible SUM

        For efficiency: With independent/positively-correlated players, the optimal
        3-player SUM is always the sum of the top 3 individual scorers in that trial.

        P(portfolio wins) = (# trials where any lineup has optimal sum) / n_sims
        """
        if not lineups:
            return 0.0

        # Get ALL players in the slate (not just those in our portfolio)
        # This is needed to compute the global optimal 3-player sum
        all_player_ids = sorted([p.player_id for p in self.pool_list])
        n_players = len(all_player_ids)

        if n_players < 3:
            return 0.0

        # Build id -> index mapping
        id_to_idx = {pid: i for i, pid in enumerate(all_player_ids)}

        # Get means and sigmas for all players
        means = np.zeros(n_players)
        sigmas = np.zeros(n_players)

        for pid in all_player_ids:
            idx = id_to_idx[pid]
            player = self.pool.get(pid)
            if player:
                means[idx] = player.projected_ppg
                sigmas[idx] = player.sigma
            else:
                means[idx] = 20.0  # fallback
                sigmas[idx] = 5.0

        # Build lineup player index lists for fast sum computation
        lineup_idx_lists = []
        for lineup in lineups:
            idx_list = [id_to_idx[pid] for pid in lineup.player_ids() if pid in id_to_idx]
            lineup_idx_lists.append(idx_list)

        # Run Monte Carlo
        portfolio_wins = 0
        np.random.seed(42)  # reproducibility

        for _ in range(n_sims):
            # Sample player scores (using normal distribution)
            simulated_scores = means + sigmas * np.random.standard_normal(n_players)

            # Compute the optimal 3-player SUM (top 3 individual scores summed)
            # With independent/positive-correlated samples, this is always optimal
            top3_indices = np.argsort(simulated_scores)[-3:]
            optimal_sum = np.sum(simulated_scores[top3_indices])

            # Compute our portfolio's best lineup SUM
            best_portfolio_sum = -np.inf
            for idx_list in lineup_idx_lists:
                lineup_sum = np.sum(simulated_scores[idx_list])
                if lineup_sum > best_portfolio_sum:
                    best_portfolio_sum = lineup_sum

            # Win if our best lineup equals the optimal (with small tolerance)
            if best_portfolio_sum >= optimal_sum - 0.01:
                portfolio_wins += 1

        return portfolio_wins / n_sims

    def _get_available_players(
        self,
        exclude_ids: Set[int] = None,
        min_p_top1: float = 0.0,
        min_sigma: float = 0.0,
        require_game: str = None,
        require_questionable: bool = False,
        require_injury_beneficiary: bool = False
    ) -> List[PlayerPool]:
        """Get players matching criteria and exposure constraints."""
        exclude_ids = exclude_ids or set()
        available = []

        for p in self.pool_list:
            if p.player_id in exclude_ids:
                continue
            if not self.exposure_mgr.can_add_player(p.player_id):
                continue
            if p.p_top1 < min_p_top1:
                continue
            if p.sigma < min_sigma:
                continue
            if require_game and p.game_id != require_game:
                continue
            if require_questionable and not p.is_questionable:
                continue
            if require_injury_beneficiary and not p.is_injury_beneficiary:
                continue

            available.append(p)

        return available

    def _select_supports(
        self,
        anchor: PlayerPool,
        n: int,
        exclude_ids: Set[int] = None,
        prefer_different_games: bool = True
    ) -> List[PlayerPool]:
        """
        Select support players for a lineup anchored by a star.

        Prioritizes:
        1. High support_score (P(top-10 | not #1))
        2. Different games (diversification)
        3. High p_top3
        """
        exclude_ids = exclude_ids or set()
        exclude_ids.add(anchor.player_id)

        candidates = self._get_available_players(exclude_ids)

        # Score candidates
        scored = []
        for p in candidates:
            score = p.support_score * 100  # Base: support quality

            # Prefer different games for diversification
            if prefer_different_games and p.game_id != anchor.game_id:
                score += 20

            # Bonus for high p_top3
            score += p.p_top3 * 30

            # Slight bonus for ceiling
            score += (p.ceiling / 50) * 10

            scored.append((p, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top n
        selected = []
        used_games = {anchor.game_id}

        for p, _ in scored:
            if len(selected) >= n:
                break

            # Skip if we already have someone from this game (diversification)
            if prefer_different_games and p.game_id in used_games and len(selected) < n - 1:
                # Allow same game for last slot if needed
                continue

            selected.append(p)
            used_games.add(p.game_id)

        return selected

    def build_chalk_lineups(self, n: int = None) -> List[Lineup]:
        """
        Build chalk lineups: anchor on highest p_top1 star + strong supports.

        Strategy:
        - Each lineup has 1 mega-star anchor
        - 2 support players with high support_score
        - Diversify supports across games
        - Exposure caps (not global exclusion) prevent over-concentration
        """
        n = n or self.config.chalk_lineups
        lineups = []

        # Get stars sorted by p_top1 (explicit sorting, not order-dependent)
        stars = sorted(
            [p for p in self.pool_list if p.p_top1 >= self.config.star_p_top1_threshold],
            key=lambda p: p.p_top1,
            reverse=True
        )

        if not stars:
            warnings.warn("No stars found for chalk lineups")
            return lineups

        max_attempts = n * 3  # Prevent infinite loops
        attempts = 0

        while len(lineups) < n and attempts < max_attempts:
            attempts += 1

            # Rotate through stars (don't always use #1)
            star_idx = (len(lineups)) % len(stars)
            anchor = stars[star_idx]

            if not self.exposure_mgr.can_add_player(anchor.player_id):
                # Star at exposure cap, try next available
                anchor = None
                for alt_star in stars:
                    if self.exposure_mgr.can_add_player(alt_star.player_id):
                        anchor = alt_star
                        break
                if anchor is None:
                    break  # No stars available at all

            # Select supports (no global used_in_chalk exclusion - let exposure caps handle it)
            supports = self._select_supports(
                anchor, 2, exclude_ids=set(), prefer_different_games=True
            )

            if len(supports) < 2:
                # Relax game diversification constraint
                supports = self._select_supports(
                    anchor, 2, exclude_ids=set(), prefer_different_games=False
                )

            if len(supports) < 2:
                continue  # Can't build this lineup

            players = [anchor] + supports
            player_ids = [p.player_id for p in players]

            # Check for duplicate lineup
            if not self._is_unique_lineup(player_ids):
                continue  # Skip duplicate

            # Check core overlap (duplication penalty)
            if not self._can_add_lineup_core(player_ids):
                continue  # Skip - would create too much correlated fate

            # Register lineup, cores, and record exposure
            self._register_lineup(player_ids)
            self._register_lineup_cores(player_ids)
            for p in players:
                self.exposure_mgr.add_player(p.player_id)

            lineup = Lineup(
                players=players,
                bucket="chalk",
                win_probability=self._calculate_lineup_win_prob(player_ids)
            )
            lineups.append(lineup)

        return lineups

    def build_stack_lineups(self, n: int = None) -> List[Lineup]:
        """
        Build game stack lineups: 2 players from same high-env game + 1 elite.

        Strategy:
        - Find games with high stack_score (tight spread + high total)
        - Pick 2 players from that game (ENFORCE different teams for positive correlation)
        - Add 1 elite #1 candidate from another game
        """
        n = n or self.config.stack_lineups
        lineups = []

        # Find stackable games (sorted by stack_score for explicit ordering)
        stackable_games = sorted(
            [
                game_id for game_id, env in self.game_envs.items()
                if env.get('stack_score', 0) >= self.config.stack_score_threshold
            ],
            key=lambda g: self.game_envs[g].get('stack_score', 0),
            reverse=True
        )

        if not stackable_games:
            warnings.warn("No stackable games found (stack_score >= threshold)")
            # Fallback 1: use highest stack_score games available from game_envs
            stackable_games = sorted(
                self.game_envs.keys(),
                key=lambda g: self.game_envs[g].get('stack_score', 0),
                reverse=True
            )[:3]

        if not stackable_games:
            # Fallback 2: use games from player pool, sorted by implied scoring potential
            # (sum of top-2 p_top3 in the game = proxy for high-scoring environment)
            def game_stack_proxy(game_id):
                players = self.players_by_game.get(game_id, [])
                if len(players) < 2:
                    return 0
                # Sort by p_top3 descending and sum top 2
                sorted_p = sorted([p.p_top3 for p in players], reverse=True)
                return sum(sorted_p[:2])

            stackable_games = sorted(
                self.players_by_game.keys(),
                key=game_stack_proxy,
                reverse=True
            )
            if stackable_games:
                warnings.warn(f"game_envs empty, using {len(stackable_games)} games from player pool "
                             f"(sorted by implied scoring potential)")

        if not stackable_games:
            # No games available at all - skip stack bucket
            warnings.warn("No games available for stacking. Skipping stack bucket.")
            return []

        max_attempts = n * 3
        attempts = 0

        while len(lineups) < n and attempts < max_attempts:
            attempts += 1

            # Rotate through stackable games (use attempts for rotation to ensure progress)
            game_id = stackable_games[(attempts - 1) % len(stackable_games)]

            # Get players from this game
            game_players = self.players_by_game.get(game_id, [])
            if len(game_players) < 2:
                continue

            # Get available players, grouped by team for opposite-team stacking
            available_game = [
                p for p in game_players
                if self.exposure_mgr.can_add_player(p.player_id)
            ]

            if len(available_game) < 2:
                continue

            # Find the best opposite-team pair
            # Group by team
            by_team: Dict[str, List[PlayerPool]] = defaultdict(list)
            for p in available_game:
                by_team[p.team].append(p)

            # Sort players within each team by p_top3 (explicit ordering)
            for team in by_team:
                by_team[team].sort(key=lambda p: p.p_top3, reverse=True)

            teams = list(by_team.keys())
            stack_players = None

            if len(teams) >= 2:
                # ENFORCE different teams: pick best from each of 2 teams
                best_pairs = []
                for i, team1 in enumerate(teams):
                    for team2 in teams[i+1:]:
                        if by_team[team1] and by_team[team2]:
                            p1, p2 = by_team[team1][0], by_team[team2][0]
                            pair_score = p1.p_top3 + p2.p_top3
                            best_pairs.append((p1, p2, pair_score))

                if best_pairs:
                    # Sort by combined p_top3 and take best pair
                    best_pairs.sort(key=lambda x: x[2], reverse=True)
                    stack_players = [best_pairs[0][0], best_pairs[0][1]]

            if not stack_players:
                # Fallback: same team if only one team available (rare)
                # Sort all available by p_top3
                available_game.sort(key=lambda p: p.p_top3, reverse=True)
                if len(available_game) >= 2:
                    stack_players = available_game[:2]
                else:
                    continue

            # Get elite from different game (sorted by p_top1 for explicit selection)
            elite_candidates = sorted(
                [
                    p for p in self.pool_list
                    if p.game_id != game_id
                    and p.p_top1 >= self.config.star_p_top1_threshold
                    and self.exposure_mgr.can_add_player(p.player_id)
                ],
                key=lambda p: p.p_top1,
                reverse=True
            )

            if not elite_candidates:
                elite_candidates = sorted(
                    [
                        p for p in self.pool_list
                        if p.game_id != game_id
                        and self.exposure_mgr.can_add_player(p.player_id)
                    ],
                    key=lambda p: p.p_top1,
                    reverse=True
                )

            if not elite_candidates:
                continue

            # Rotate through elites to get variety
            elite_idx = len(lineups) % min(3, len(elite_candidates))
            elite = elite_candidates[elite_idx]

            players = stack_players + [elite]
            player_ids = [p.player_id for p in players]

            # Check for duplicate lineup
            if not self._is_unique_lineup(player_ids):
                continue

            # Check core overlap (duplication penalty)
            if not self._can_add_lineup_core(player_ids):
                continue  # Skip - would create too much correlated fate

            # Register lineup, cores, and record exposure
            self._register_lineup(player_ids)
            self._register_lineup_cores(player_ids)
            for p in players:
                self.exposure_mgr.add_player(p.player_id)

            lineup = Lineup(
                players=players,
                bucket="stack",
                win_probability=self._calculate_lineup_win_prob(player_ids),
                stack_game=game_id
            )
            lineups.append(lineup)

        return lineups

    def build_leverage_lineups(self, n: int = None) -> List[Lineup]:
        """
        Build leverage/volatility lineups: high-σ players with real ceiling.

        Strategy:
        - Find players with high sigma relative to projection
        - Explicitly sort by sigma (not order-dependent)
        - Mix of some ceiling + some upside
        """
        n = n or self.config.leverage_lineups
        lineups = []

        # Find high-leverage players (explicitly sorted by sigma)
        avg_sigma = np.mean([p.sigma for p in self.pool_list])
        leverage_players = sorted(
            [
                p for p in self.pool_list
                if p.sigma >= avg_sigma * self.config.leverage_sigma_multiplier
                and p.ceiling >= 35  # Must have real ceiling
            ],
            key=lambda p: p.sigma,
            reverse=True
        )

        if len(leverage_players) < 3:
            warnings.warn("Few leverage players found, using top sigma players")
            leverage_players = sorted(
                self.pool_list,
                key=lambda p: p.sigma,
                reverse=True
            )[:10]

        max_attempts = n * 3
        attempts = 0

        while len(lineups) < n and attempts < max_attempts:
            attempts += 1

            # Select leverage players with rotation for variety
            available_leverage = sorted(
                [
                    p for p in leverage_players
                    if self.exposure_mgr.can_add_player(p.player_id)
                ],
                key=lambda p: p.sigma,
                reverse=True
            )

            if not available_leverage:
                break

            # Take 1-2 leverage players with rotation
            num_leverage = min(2, len(available_leverage))
            # Rotate starting point to get variety
            start_idx = (len(lineups) * num_leverage) % max(1, len(available_leverage))
            selected_leverage = []
            for i in range(num_leverage):
                idx = (start_idx + i) % len(available_leverage)
                selected_leverage.append(available_leverage[idx])

            # Fill remaining with solid supports (explicitly sorted by p_top3)
            remaining = self.config.lineup_size - num_leverage
            exclude = set(p.player_id for p in selected_leverage)

            supports = sorted(
                self._get_available_players(exclude_ids=exclude),
                key=lambda p: p.p_top3,
                reverse=True
            )
            selected_supports = supports[:remaining]

            if len(selected_supports) < remaining:
                continue

            players = selected_leverage + selected_supports
            player_ids = [p.player_id for p in players]

            # Check for duplicate lineup
            if not self._is_unique_lineup(player_ids):
                continue

            # Check core overlap (duplication penalty)
            if not self._can_add_lineup_core(player_ids):
                continue  # Skip - would create too much correlated fate

            # Register lineup, cores, and record exposure
            self._register_lineup(player_ids)
            self._register_lineup_cores(player_ids)
            for p in players:
                self.exposure_mgr.add_player(p.player_id)

            lineup = Lineup(
                players=players,
                bucket="leverage",
                win_probability=self._calculate_lineup_win_prob(player_ids)
            )
            lineups.append(lineup)

        return lineups

    def build_news_lineups(self, n: int = None) -> List[Lineup]:
        """
        Build news pivot lineups: questionable situations, injury beneficiaries.

        Strategy:
        - Target players marked as questionable (high uncertainty = high upside)
        - Include injury beneficiaries (teammate out = usage spike)
        - Explicitly sort by ceiling (deterministic selection)
        """
        n = n or self.config.news_lineups
        lineups = []

        # Find news-relevant players (sorted by ceiling for explicit ordering)
        questionable = [p for p in self.pool_list if p.is_questionable]
        injury_bens = [p for p in self.pool_list if p.is_injury_beneficiary]
        news_players_set = set(p.player_id for p in questionable + injury_bens)
        news_players = sorted(
            [p for p in self.pool_list if p.player_id in news_players_set],
            key=lambda p: p.ceiling,
            reverse=True
        )

        if not news_players:
            warnings.warn("No questionable/injury-beneficiary players marked")
            # Fallback: use highest ceiling players
            news_players = sorted(
                self.pool_list,
                key=lambda p: p.ceiling,
                reverse=True
            )[:5]

        max_attempts = n * 3
        attempts = 0

        while len(lineups) < n and attempts < max_attempts:
            attempts += 1

            # Get available news players (sorted by ceiling)
            available_news = sorted(
                [
                    p for p in news_players
                    if self.exposure_mgr.can_add_player(p.player_id)
                ],
                key=lambda p: p.ceiling,
                reverse=True
            )

            if not available_news:
                break

            # Pick news player with rotation for variety
            news_idx = len(lineups) % len(available_news)
            news_pick = available_news[news_idx]

            # Fill with supports (sorted by p_top3)
            exclude = {news_pick.player_id}
            supports = sorted(
                self._get_available_players(exclude_ids=exclude),
                key=lambda p: p.p_top3,
                reverse=True
            )
            selected_supports = supports[:2]

            if len(selected_supports) < 2:
                continue

            players = [news_pick] + selected_supports
            player_ids = [p.player_id for p in players]

            # Check for duplicate lineup
            if not self._is_unique_lineup(player_ids):
                continue

            # Check core overlap (duplication penalty)
            if not self._can_add_lineup_core(player_ids):
                continue  # Skip - would create too much correlated fate

            # Register lineup, cores, and record exposure
            self._register_lineup(player_ids)
            self._register_lineup_cores(player_ids)
            for p in players:
                self.exposure_mgr.add_player(p.player_id)

            lineup = Lineup(
                players=players,
                bucket="news",
                win_probability=self._calculate_lineup_win_prob(player_ids)
            )
            lineups.append(lineup)

        return lineups

    def optimize(self) -> PortfolioResult:
        """
        Build full 20-lineup portfolio.

        Returns:
            PortfolioResult with all lineups and analytics
        """
        all_lineups = []
        warnings_list = []

        # Build each bucket
        chalk = self.build_chalk_lineups()
        all_lineups.extend(chalk)
        if len(chalk) < self.config.chalk_lineups:
            warnings_list.append(
                f"Only built {len(chalk)}/{self.config.chalk_lineups} chalk lineups"
            )

        stack = self.build_stack_lineups()
        all_lineups.extend(stack)
        if len(stack) < self.config.stack_lineups:
            warnings_list.append(
                f"Only built {len(stack)}/{self.config.stack_lineups} stack lineups"
            )

        leverage = self.build_leverage_lineups()
        all_lineups.extend(leverage)
        if len(leverage) < self.config.leverage_lineups:
            warnings_list.append(
                f"Only built {len(leverage)}/{self.config.leverage_lineups} leverage lineups"
            )

        news = self.build_news_lineups()
        all_lineups.extend(news)
        if len(news) < self.config.news_lineups:
            warnings_list.append(
                f"Only built {len(news)}/{self.config.news_lineups} news lineups"
            )

        # Calculate portfolio win probability using Monte Carlo
        # This correctly accounts for correlation: all lineups evaluated on same slate
        total_win_prob = self._calculate_portfolio_win_prob_montecarlo(all_lineups)

        # Get exposure report
        # Update exposure manager's total_lineups to actual count for accurate percentages
        actual_total = len(all_lineups)
        if actual_total > 0 and actual_total != self.exposure_mgr.total_lineups:
            # Recalculate exposure percentages based on actual lineup count
            old_total = self.exposure_mgr.total_lineups
            self.exposure_mgr.total_lineups = actual_total
            warnings_list.append(
                f"Exposure calculated for {actual_total} lineups (target was {old_total})"
            )

        exposure_report = self.exposure_mgr.get_exposure_report(self.pool)

        # Count unique players
        unique_players = len([
            pid for pid, data in exposure_report.items()
            if data['count'] > 0
        ])

        if unique_players < self.config.min_unique_players:
            warnings_list.append(
                f"Only {unique_players} unique players (target: {self.config.min_unique_players})"
            )

        # Calculate diversity metrics (duplication penalty)
        diversity_score = self._calculate_portfolio_diversity(all_lineups)
        core_saturation = self._get_core_saturation_report()

        if diversity_score < self.config.min_portfolio_diversity:
            warnings_list.append(
                f"Low portfolio diversity: {diversity_score:.2f} (target: {self.config.min_portfolio_diversity:.2f})"
            )

        if core_saturation.get('saturated_cores', 0) > 5:
            warnings_list.append(
                f"High core saturation: {core_saturation['saturated_cores']} cores at max overlap"
            )

        return PortfolioResult(
            lineups=all_lineups,
            exposure_report=exposure_report,
            bucket_summary={
                'chalk': len(chalk),
                'stack': len(stack),
                'leverage': len(leverage),
                'news': len(news),
            },
            total_win_probability=total_win_prob,
            unique_players=unique_players,
            warnings=warnings_list,
            diversity_score=diversity_score,
            core_saturation=core_saturation
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_player_pool_from_predictions(
    predictions_df: pd.DataFrame,
    sim_results: Dict[int, CorrelatedSimResult] = None,
    games_df: pd.DataFrame = None
) -> List[PlayerPool]:
    """
    Convert predictions DataFrame to PlayerPool objects.

    Expected columns:
    - player_id, player_name, team_abbreviation
    - projected_ppg, proj_ceiling, proj_floor
    - season_avg_ppg (for is_star)
    - injury_status (optional), injury_adjusted (optional)

    Optional games_df columns:
    - game_id, home_team, away_team (for proper matchup-based game_id)
    """
    pool = []

    # Build team -> game_id mapping from games_df if available
    team_to_game = {}
    if games_df is not None and len(games_df) > 0:
        for _, game in games_df.iterrows():
            gid = game.get('game_id')
            home = game.get('home_team', game.get('home_team_abbreviation', ''))
            away = game.get('away_team', game.get('away_team_abbreviation', ''))
            if gid:
                if home:
                    team_to_game[home] = gid
                if away:
                    team_to_game[away] = gid

    # Build matchup-based fallback game_ids by grouping teams
    # Strategy: pair teams alphabetically if no explicit game_id
    teams_in_slate = predictions_df['team_abbreviation'].dropna().unique().tolist()
    teams_in_slate.sort()  # Alphabetical for consistent pairing

    for _, row in predictions_df.iterrows():
        player_id = row['player_id']

        # Get simulation results if available
        sim = sim_results.get(player_id) if sim_results else None

        # Calculate sigma (use ceiling-floor/4 consistently)
        ceiling = row.get('proj_ceiling', 0) or 0
        floor = row.get('proj_floor', 0) or 0
        sigma = (ceiling - floor) / 4 if ceiling > floor else row.get('projected_ppg', 20) * 0.15

        # Determine game_id - use proper matchup-based keys
        game_id = row.get('game_id')
        team = row.get('team_abbreviation', 'UNK')

        if not game_id or pd.isna(game_id):
            # Try to get from team_to_game mapping
            game_id = team_to_game.get(team)

        if not game_id:
            # Fallback: construct matchup-based key from team
            # Find this team's opponent by looking at other players
            # or use a placeholder that at least keeps same-team players together
            opponent = row.get('opponent', row.get('opponent_team', ''))
            if opponent:
                # Create consistent game_id: alphabetical ordering of teams
                teams = sorted([team, opponent])
                game_id = f"game_{teams[0]}@{teams[1]}"
            else:
                # Last resort: use team but warn
                game_id = f"game_{team}_unknown_opponent"

        pool.append(PlayerPool(
            player_id=player_id,
            player_name=row['player_name'],
            team=team,
            game_id=game_id,
            projected_ppg=row.get('projected_ppg', 0) or 0,
            sigma=max(sigma, 1.0),
            ceiling=ceiling,
            floor=floor,
            p_top1=sim.p_top1 if sim else row.get('p_top1', 0) or 0,
            p_top3=sim.p_top3 if sim else row.get('p_top3', 0) or 0,
            support_score=sim.support_score if sim else 0.5,
            expected_rank=sim.expected_rank if sim else 50,
            is_star=row.get('season_avg_ppg', 0) >= 25,
            is_questionable=row.get('injury_status', '') == 'questionable',
            is_injury_beneficiary=bool(row.get('injury_adjusted', False)),
        ))

    return pool


def format_portfolio_report(result: PortfolioResult) -> str:
    """Format portfolio result as readable string."""
    total_lineups = len(result.lineups)
    lines = []
    lines.append("=" * 70)
    lines.append(f"TOURNAMENT PORTFOLIO - {total_lineups} LINEUPS")
    lines.append("=" * 70)
    lines.append(f"\nPortfolio Win Probability (Monte Carlo): {result.total_win_probability:.1%}")
    lines.append(f"Unique Players: {result.unique_players}")
    lines.append(f"\nBucket Summary: {result.bucket_summary}")

    if result.warnings:
        lines.append("\nWarnings:")
        for w in result.warnings:
            lines.append(f"  [!] {w}")

    # Lineups by bucket
    for bucket in ['chalk', 'stack', 'leverage', 'news']:
        bucket_lineups = [l for l in result.lineups if l.bucket == bucket]
        if bucket_lineups:
            lines.append(f"\n{'-' * 50}")
            lines.append(f"BUCKET: {bucket.upper()} ({len(bucket_lineups)} lineups)")
            lines.append(f"{'-' * 50}")

            for i, lineup in enumerate(bucket_lineups, 1):
                names = ', '.join([p.player_name for p in lineup.players])
                teams = ', '.join([p.team for p in lineup.players])
                lines.append(f"  {i}. {names}")
                lines.append(f"     Teams: {teams} | "
                           f"Lineup Win Prob: {lineup.win_probability:.2%} | "
                           f"Ceiling: {lineup.total_ceiling():.0f}")
                if lineup.stack_game:
                    lines.append(f"     Stack Game: {lineup.stack_game}")

    # Exposure report
    lines.append(f"\n{'=' * 70}")
    lines.append(f"EXPOSURE REPORT (out of {total_lineups} lineups)")
    lines.append(f"{'=' * 70}")

    sorted_exposure = sorted(
        result.exposure_report.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    for pid, data in sorted_exposure[:15]:
        cap_indicator = "[AT CAP]" if data['at_cap'] else ""
        lines.append(
            f"  {data['player_name']:<25} {data['count']:>2}/{total_lineups} ({data['pct']:.0%}) "
            f"[{data['tier']}] {cap_indicator}"
        )

    # Diversity metrics
    lines.append(f"\n{'=' * 70}")
    lines.append(f"DIVERSITY METRICS (Duplication Penalty)")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Portfolio Diversity (Jaccard): {result.diversity_score:.2f}")
    if result.core_saturation:
        lines.append(f"  Max Core Usage: {result.core_saturation.get('max_core_usage', 0)} lineups")
        lines.append(f"  Saturated Cores: {result.core_saturation.get('saturated_cores', 0)}")
        core_dist = result.core_saturation.get('core_distribution', {})
        if core_dist:
            dist_str = ', '.join(f"{k}x:{v}" for k, v in sorted(core_dist.items()))
            lines.append(f"  Core Distribution: {dist_str}")

    return '\n'.join(lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Demo with synthetic data
    print("Tournament Lineup Optimizer Demo")
    print("=" * 60)

    # Create sample player pool with proper matchup-based game IDs
    # Game 1: LAL vs GSW (high stack potential)
    # Game 2: BOS vs MIA (moderate stack)
    # Game 3: DAL vs OKC (good stack)
    sample_pool = [
        # Game 1: LAL @ GSW
        PlayerPool(1, "LeBron James", "LAL", "game_GSW@LAL", 28.0, 6.0, 45.0, 18.0, 0.12, 0.35, 0.75, 8.5, True),
        PlayerPool(2, "Anthony Davis", "LAL", "game_GSW@LAL", 26.0, 5.5, 42.0, 16.0, 0.09, 0.30, 0.70, 10.2, True),
        PlayerPool(3, "Stephen Curry", "GSW", "game_GSW@LAL", 27.0, 7.0, 48.0, 14.0, 0.15, 0.38, 0.72, 7.8, True),
        PlayerPool(4, "Klay Thompson", "GSW", "game_GSW@LAL", 20.0, 5.0, 35.0, 10.0, 0.04, 0.18, 0.55, 18.5, False),
        # Game 2: BOS vs MIA
        PlayerPool(5, "Jayson Tatum", "BOS", "game_BOS@MIA", 29.0, 6.5, 46.0, 17.0, 0.14, 0.40, 0.78, 7.2, True),
        PlayerPool(6, "Jaylen Brown", "BOS", "game_BOS@MIA", 24.0, 5.0, 38.0, 14.0, 0.06, 0.22, 0.62, 14.8, False),
        PlayerPool(7, "Jimmy Butler", "MIA", "game_BOS@MIA", 22.0, 6.0, 40.0, 10.0, 0.05, 0.20, 0.58, 16.5, False, is_questionable=True),
        PlayerPool(8, "Bam Adebayo", "MIA", "game_BOS@MIA", 18.0, 4.0, 28.0, 12.0, 0.02, 0.12, 0.48, 22.0, False),
        # Game 3: DAL vs OKC
        PlayerPool(9, "Luka Doncic", "DAL", "game_DAL@OKC", 32.0, 7.5, 52.0, 18.0, 0.20, 0.48, 0.82, 5.2, True),
        PlayerPool(10, "Kyrie Irving", "DAL", "game_DAL@OKC", 25.0, 6.0, 40.0, 14.0, 0.08, 0.28, 0.65, 12.0, True),
        PlayerPool(11, "Shai Gilgeous", "OKC", "game_DAL@OKC", 31.0, 5.5, 45.0, 22.0, 0.18, 0.45, 0.80, 6.0, True),
        PlayerPool(12, "Chet Holmgren", "OKC", "game_DAL@OKC", 17.0, 5.0, 32.0, 8.0, 0.03, 0.15, 0.52, 20.0, False, is_injury_beneficiary=True),
    ]

    # Game environments (keyed by matchup-based game_id)
    game_envs = {
        "game_GSW@LAL": {"stack_score": 0.85, "ot_probability": 0.10, "pace_score": 1.05},
        "game_BOS@MIA": {"stack_score": 0.50, "ot_probability": 0.06, "pace_score": 0.98},
        "game_DAL@OKC": {"stack_score": 0.75, "ot_probability": 0.08, "pace_score": 1.02},
    }

    # Build simulation results dict
    sim_results = {
        p.player_id: CorrelatedSimResult(
            player_id=p.player_id,
            player_name=p.player_name,
            p_top1=p.p_top1,
            p_top3=p.p_top3,
            expected_rank=p.expected_rank,
            support_score=p.support_score,
            sigma_used=p.sigma
        )
        for p in sample_pool
    }

    # Run optimizer
    optimizer = TournamentLineupOptimizer(
        player_pool=sample_pool,
        game_environments=game_envs,
        sim_results=sim_results
    )

    result = optimizer.optimize()

    # Print report
    print(format_portfolio_report(result))

    # Verify uniqueness
    all_lineup_keys = [frozenset(l.player_ids()) for l in result.lineups]
    unique_keys = set(all_lineup_keys)
    print(f"\n[OK] Lineup uniqueness check: {len(all_lineup_keys)} total, {len(unique_keys)} unique")
