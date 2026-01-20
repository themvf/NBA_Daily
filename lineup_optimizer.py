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
        """
        n = n or self.config.chalk_lineups
        lineups = []

        # Get stars sorted by p_top1
        stars = sorted(
            [p for p in self.pool_list if p.p_top1 >= self.config.star_p_top1_threshold],
            key=lambda p: p.p_top1,
            reverse=True
        )

        if not stars:
            warnings.warn("No stars found for chalk lineups")
            return lineups

        # Build lineups anchored on top stars
        used_in_chalk: Set[int] = set()

        for i in range(n):
            # Rotate through stars (don't always use #1)
            star_idx = i % len(stars)
            anchor = stars[star_idx]

            if not self.exposure_mgr.can_add_player(anchor.player_id):
                # Star at exposure cap, try next
                for alt_star in stars:
                    if self.exposure_mgr.can_add_player(alt_star.player_id):
                        anchor = alt_star
                        break
                else:
                    continue  # No stars available

            # Select supports
            supports = self._select_supports(
                anchor, 2, exclude_ids=used_in_chalk, prefer_different_games=True
            )

            if len(supports) < 2:
                # Relax constraints
                supports = self._select_supports(
                    anchor, 2, exclude_ids=set(), prefer_different_games=False
                )

            if len(supports) < 2:
                continue  # Can't build this lineup

            players = [anchor] + supports
            player_ids = [p.player_id for p in players]

            # Record exposure
            for p in players:
                self.exposure_mgr.add_player(p.player_id)
                used_in_chalk.add(p.player_id)

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
        - Pick 2 players from that game (different teams preferred)
        - Add 1 elite #1 candidate from another game
        """
        n = n or self.config.stack_lineups
        lineups = []

        # Find stackable games
        stackable_games = [
            game_id for game_id, env in self.game_envs.items()
            if env.get('stack_score', 0) >= self.config.stack_score_threshold
        ]

        if not stackable_games:
            warnings.warn("No stackable games found (stack_score >= threshold)")
            # Fallback: use highest stack_score games available
            stackable_games = sorted(
                self.game_envs.keys(),
                key=lambda g: self.game_envs[g].get('stack_score', 0),
                reverse=True
            )[:3]

        used_in_stacks: Set[int] = set()

        for i in range(n):
            # Rotate through stackable games
            game_id = stackable_games[i % len(stackable_games)]

            # Get players from this game
            game_players = self.players_by_game.get(game_id, [])
            if len(game_players) < 2:
                continue

            # Select 2 players from the stack game
            # Prefer different teams for positive correlation
            available_game = [
                p for p in game_players
                if self.exposure_mgr.can_add_player(p.player_id)
                and p.player_id not in used_in_stacks
            ]

            if len(available_game) < 2:
                available_game = [
                    p for p in game_players
                    if self.exposure_mgr.can_add_player(p.player_id)
                ]

            if len(available_game) < 2:
                continue

            # Sort by p_top3 and take top 2
            available_game.sort(key=lambda p: p.p_top3, reverse=True)
            stack_players = available_game[:2]

            # Get elite from different game
            elite_candidates = [
                p for p in self.pool_list
                if p.game_id != game_id
                and p.p_top1 >= self.config.star_p_top1_threshold
                and self.exposure_mgr.can_add_player(p.player_id)
            ]

            if not elite_candidates:
                elite_candidates = [
                    p for p in self.pool_list
                    if p.game_id != game_id
                    and self.exposure_mgr.can_add_player(p.player_id)
                ]

            if not elite_candidates:
                continue

            elite_candidates.sort(key=lambda p: p.p_top1, reverse=True)
            elite = elite_candidates[0]

            players = stack_players + [elite]
            player_ids = [p.player_id for p in players]

            # Record exposure
            for p in players:
                self.exposure_mgr.add_player(p.player_id)
                used_in_stacks.add(p.player_id)

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
        - Avoid obvious duplicates from other buckets
        - Mix of some ceiling + some upside
        """
        n = n or self.config.leverage_lineups
        lineups = []

        # Find high-leverage players
        avg_sigma = np.mean([p.sigma for p in self.pool_list])
        leverage_players = [
            p for p in self.pool_list
            if p.sigma >= avg_sigma * self.config.leverage_sigma_multiplier
            and p.ceiling >= 35  # Must have real ceiling
        ]

        if len(leverage_players) < 3:
            warnings.warn("Few leverage players found, using top sigma players")
            leverage_players = sorted(
                self.pool_list,
                key=lambda p: p.sigma,
                reverse=True
            )[:10]

        used_in_leverage: Set[int] = set()

        for i in range(n):
            # Select 1-2 leverage players + support
            available_leverage = [
                p for p in leverage_players
                if self.exposure_mgr.can_add_player(p.player_id)
                and p.player_id not in used_in_leverage
            ]

            if len(available_leverage) < 2:
                available_leverage = [
                    p for p in leverage_players
                    if self.exposure_mgr.can_add_player(p.player_id)
                ]

            if not available_leverage:
                continue

            # Take 1-2 leverage players
            num_leverage = min(2, len(available_leverage))
            selected_leverage = available_leverage[:num_leverage]

            # Fill remaining with solid supports
            remaining = self.config.lineup_size - num_leverage
            exclude = set(p.player_id for p in selected_leverage) | used_in_leverage

            supports = self._get_available_players(exclude_ids=exclude)
            supports.sort(key=lambda p: p.p_top3, reverse=True)
            selected_supports = supports[:remaining]

            if len(selected_supports) < remaining:
                continue

            players = selected_leverage + selected_supports
            player_ids = [p.player_id for p in players]

            # Record exposure
            for p in players:
                self.exposure_mgr.add_player(p.player_id)
                used_in_leverage.add(p.player_id)

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
        """
        n = n or self.config.news_lineups
        lineups = []

        # Find news-relevant players
        questionable = [p for p in self.pool_list if p.is_questionable]
        injury_bens = [p for p in self.pool_list if p.is_injury_beneficiary]
        news_players = list(set(questionable + injury_bens))

        if not news_players:
            warnings.warn("No questionable/injury-beneficiary players marked")
            # Fallback: use highest ceiling players
            news_players = sorted(
                self.pool_list,
                key=lambda p: p.ceiling,
                reverse=True
            )[:5]

        used_in_news: Set[int] = set()

        for i in range(n):
            available_news = [
                p for p in news_players
                if self.exposure_mgr.can_add_player(p.player_id)
                and p.player_id not in used_in_news
            ]

            if not available_news:
                available_news = [
                    p for p in news_players
                    if self.exposure_mgr.can_add_player(p.player_id)
                ]

            if not available_news:
                continue

            # Take 1 news player
            news_pick = available_news[0]

            # Fill with supports
            exclude = {news_pick.player_id} | used_in_news
            supports = self._get_available_players(exclude_ids=exclude)
            supports.sort(key=lambda p: p.p_top3, reverse=True)
            selected_supports = supports[:2]

            if len(selected_supports) < 2:
                continue

            players = [news_pick] + selected_supports
            player_ids = [p.player_id for p in players]

            # Record exposure
            for p in players:
                self.exposure_mgr.add_player(p.player_id)
                used_in_news.add(p.player_id)

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

        # Calculate portfolio win probability
        # P(any wins) = 1 - P(none win)
        # Approximation: assume lineups are somewhat independent
        p_none_win = 1.0
        for lineup in all_lineups:
            p_none_win *= (1 - lineup.win_probability)
        total_win_prob = 1 - p_none_win

        # Get exposure report
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
            warnings=warnings_list
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_player_pool_from_predictions(
    predictions_df: pd.DataFrame,
    sim_results: Dict[int, CorrelatedSimResult] = None
) -> List[PlayerPool]:
    """
    Convert predictions DataFrame to PlayerPool objects.

    Expected columns:
    - player_id, player_name, team_abbreviation
    - projected_ppg, proj_ceiling, proj_floor
    - season_avg_ppg (for is_star)
    - injury_status (optional), injury_adjusted (optional)
    """
    pool = []

    for _, row in predictions_df.iterrows():
        player_id = row['player_id']

        # Get simulation results if available
        sim = sim_results.get(player_id) if sim_results else None

        # Calculate sigma
        ceiling = row.get('proj_ceiling', 0) or 0
        floor = row.get('proj_floor', 0) or 0
        sigma = (ceiling - floor) / 4 if ceiling > floor else row.get('projected_ppg', 20) * 0.15

        # Determine game_id
        game_id = row.get('game_id')
        if not game_id or pd.isna(game_id):
            game_id = f"game_{row.get('team_abbreviation', 'UNK')}"

        pool.append(PlayerPool(
            player_id=player_id,
            player_name=row['player_name'],
            team=row.get('team_abbreviation', 'UNK'),
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
    lines = []
    lines.append("=" * 70)
    lines.append("TOURNAMENT PORTFOLIO - 20 LINEUPS")
    lines.append("=" * 70)
    lines.append(f"\nTotal Win Probability: {result.total_win_probability:.1%}")
    lines.append(f"Unique Players: {result.unique_players}")
    lines.append(f"\nBucket Summary: {result.bucket_summary}")

    if result.warnings:
        lines.append("\nWarnings:")
        for w in result.warnings:
            lines.append(f"  ⚠️ {w}")

    # Lineups by bucket
    for bucket in ['chalk', 'stack', 'leverage', 'news']:
        bucket_lineups = [l for l in result.lineups if l.bucket == bucket]
        if bucket_lineups:
            lines.append(f"\n{'─' * 50}")
            lines.append(f"BUCKET: {bucket.upper()} ({len(bucket_lineups)} lineups)")
            lines.append(f"{'─' * 50}")

            for i, lineup in enumerate(bucket_lineups, 1):
                names = ', '.join([p.player_name for p in lineup.players])
                lines.append(f"  {i}. {names}")
                lines.append(f"     Win Prob: {lineup.win_probability:.2%} | "
                           f"Ceiling: {lineup.total_ceiling():.0f}")
                if lineup.stack_game:
                    lines.append(f"     Stack: {lineup.stack_game}")

    # Exposure report
    lines.append(f"\n{'=' * 70}")
    lines.append("EXPOSURE REPORT")
    lines.append(f"{'=' * 70}")

    sorted_exposure = sorted(
        result.exposure_report.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    for pid, data in sorted_exposure[:15]:
        cap_indicator = "🔴 AT CAP" if data['at_cap'] else ""
        lines.append(
            f"  {data['player_name']:<25} {data['count']:>2}/{20} ({data['pct']:.0%}) "
            f"[{data['tier']}] {cap_indicator}"
        )

    return '\n'.join(lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Demo with synthetic data
    print("Tournament Lineup Optimizer Demo")
    print("=" * 60)

    # Create sample player pool
    sample_pool = [
        PlayerPool(1, "LeBron James", "LAL", "game_1", 28.0, 6.0, 45.0, 18.0, 0.12, 0.35, 0.75, 8.5, True),
        PlayerPool(2, "Anthony Davis", "LAL", "game_1", 26.0, 5.5, 42.0, 16.0, 0.09, 0.30, 0.70, 10.2, True),
        PlayerPool(3, "Stephen Curry", "GSW", "game_1", 27.0, 7.0, 48.0, 14.0, 0.15, 0.38, 0.72, 7.8, True),
        PlayerPool(4, "Klay Thompson", "GSW", "game_1", 20.0, 5.0, 35.0, 10.0, 0.04, 0.18, 0.55, 18.5, False),
        PlayerPool(5, "Jayson Tatum", "BOS", "game_2", 29.0, 6.5, 46.0, 17.0, 0.14, 0.40, 0.78, 7.2, True),
        PlayerPool(6, "Jaylen Brown", "BOS", "game_2", 24.0, 5.0, 38.0, 14.0, 0.06, 0.22, 0.62, 14.8, False),
        PlayerPool(7, "Jimmy Butler", "MIA", "game_2", 22.0, 6.0, 40.0, 10.0, 0.05, 0.20, 0.58, 16.5, False, is_questionable=True),
        PlayerPool(8, "Bam Adebayo", "MIA", "game_2", 18.0, 4.0, 28.0, 12.0, 0.02, 0.12, 0.48, 22.0, False),
        PlayerPool(9, "Luka Doncic", "DAL", "game_3", 32.0, 7.5, 52.0, 18.0, 0.20, 0.48, 0.82, 5.2, True),
        PlayerPool(10, "Kyrie Irving", "DAL", "game_3", 25.0, 6.0, 40.0, 14.0, 0.08, 0.28, 0.65, 12.0, True),
        PlayerPool(11, "Shai Gilgeous", "OKC", "game_3", 31.0, 5.5, 45.0, 22.0, 0.18, 0.45, 0.80, 6.0, True),
        PlayerPool(12, "Chet Holmgren", "OKC", "game_3", 17.0, 5.0, 32.0, 8.0, 0.03, 0.15, 0.52, 20.0, False, is_injury_beneficiary=True),
    ]

    # Game environments
    game_envs = {
        "game_1": {"stack_score": 0.85, "ot_probability": 0.10},
        "game_2": {"stack_score": 0.50, "ot_probability": 0.06},
        "game_3": {"stack_score": 0.75, "ot_probability": 0.08},
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
