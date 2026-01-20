#!/usr/bin/env python3
"""
Slate Quality Gate for Tournament Portfolio Generation

Prevents "silently wrong" portfolios by validating data quality before generation.

Three possible outcomes:
- PASS: All quality checks pass, generate normally
- DEGRADE: Some issues detected, auto-adjust buckets and warn user
- FAIL: Critical issues, block generation entirely

Usage:
    from slate_quality_gate import validate_slate_inputs, SlateValidationResult

    result = validate_slate_inputs(predictions_df, game_envs, player_pool, config)
    if result.status == "FAIL":
        st.error(result.summary)
        return
    elif result.status == "DEGRADE":
        st.warning(result.summary)
        config = result.adjusted_config  # Use degraded config
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from lineup_optimizer import OptimizerConfig


@dataclass
class SlateValidationResult:
    """Result of slate quality validation."""
    status: str  # "PASS", "DEGRADE", "FAIL"
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    adjusted_config: Optional[OptimizerConfig] = None
    original_config: Optional[OptimizerConfig] = None

    @property
    def summary(self) -> str:
        """Human-readable summary of validation result."""
        if self.status == "PASS":
            return "✅ Slate quality checks passed"
        elif self.status == "DEGRADE":
            reasons_str = "\n".join(f"  • {r}" for r in self.reasons)
            return f"⚠️ Slate quality degraded - auto-adjusting:\n{reasons_str}"
        else:  # FAIL
            reasons_str = "\n".join(f"  • {r}" for r in self.reasons)
            return f"❌ Slate quality check failed:\n{reasons_str}"

    @property
    def mode_label(self) -> str:
        """Label for the portfolio mode."""
        if self.status == "PASS":
            return "Full Strategy"
        elif "stack" in str(self.adjusted_config) or any("stack" in r.lower() for r in self.reasons):
            return "NO-ODDS MODE"
        else:
            return "DEGRADED MODE"


# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

# Hard fail thresholds
MIN_PLAYERS_NORMAL_SLATE = 80   # Minimum predicted players for a normal slate
MIN_PLAYERS_SMALL_SLATE = 30   # Minimum for small slates (3-4 games)
SIM_COVERAGE_THRESHOLD = 0.95  # 95% of players must have p_top1/p_top3

# Degrade thresholds
ODDS_COVERAGE_FOR_STACKING = 0.70  # 70% game coverage needed for stacking
MINUTES_COVERAGE_THRESHOLD = 0.90  # 90% players should have projected minutes

# Auto-reallocation ratios when stacking disabled
LEVERAGE_REALLOC_RATIO = 0.70  # 70% of stack lineups → leverage
NEWS_REALLOC_RATIO = 0.30      # 30% of stack lineups → news


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_slate_inputs(
    predictions_df: pd.DataFrame,
    game_envs: Dict[str, dict],
    player_pool: List,  # List of PlayerPool objects
    config: OptimizerConfig,
    game_id_lookup: Dict[str, str] = None
) -> SlateValidationResult:
    """
    Validate slate inputs before portfolio generation.

    Args:
        predictions_df: DataFrame of predictions for the slate
        game_envs: Dict of game_id -> environment dict
        player_pool: List of PlayerPool objects
        config: Original optimizer configuration
        game_id_lookup: Optional mapping of normalized -> actual game_ids

    Returns:
        SlateValidationResult with status, reasons, and adjusted config if needed
    """
    reasons = []
    warnings = []
    metrics = {}
    status = "PASS"

    # =========================================================================
    # 1. Player Pool Size Check
    # =========================================================================
    n_players = len(player_pool)
    n_games = len(set(p.game_id for p in player_pool))
    metrics['n_players'] = n_players
    metrics['n_games'] = n_games

    # Determine if this is a small slate
    is_small_slate = n_games <= 4

    min_players = MIN_PLAYERS_SMALL_SLATE if is_small_slate else MIN_PLAYERS_NORMAL_SLATE

    if n_players < min_players:
        if n_players < 20:
            reasons.append(f"Only {n_players} players in pool (need at least 20 for any portfolio)")
            status = "FAIL"
        else:
            warnings.append(f"Small player pool: {n_players} players ({n_games} games)")

    # =========================================================================
    # 2. Simulation Coverage Check
    # =========================================================================
    players_with_sim = sum(1 for p in player_pool if p.p_top1 > 0 or p.p_top3 > 0)
    sim_coverage = players_with_sim / n_players if n_players > 0 else 0
    metrics['sim_coverage'] = sim_coverage
    metrics['players_with_sim'] = players_with_sim

    if sim_coverage < SIM_COVERAGE_THRESHOLD:
        if sim_coverage < 0.50:
            reasons.append(f"Simulation coverage critical: {sim_coverage:.0%} of players have p_top1/p_top3")
            status = "FAIL"
        else:
            warnings.append(f"Low simulation coverage: {sim_coverage:.0%} (target: {SIM_COVERAGE_THRESHOLD:.0%})")

    # =========================================================================
    # 3. Game Environment / Odds Coverage Check
    # =========================================================================
    pool_games = set(p.game_id for p in player_pool)
    env_games = set(game_envs.keys())
    matched_games = pool_games & env_games
    odds_coverage = len(matched_games) / len(pool_games) if pool_games else 0
    metrics['odds_coverage'] = odds_coverage
    metrics['pool_games'] = len(pool_games)
    metrics['env_games'] = len(env_games)
    metrics['matched_games'] = len(matched_games)

    stack_lineups_requested = config.stack_lineups

    if stack_lineups_requested > 0 and odds_coverage < ODDS_COVERAGE_FOR_STACKING:
        reasons.append(f"Odds coverage {odds_coverage:.0%} < {ODDS_COVERAGE_FOR_STACKING:.0%} threshold for stacking")
        if status != "FAIL":
            status = "DEGRADE"

    if odds_coverage == 0 and len(game_envs) == 0:
        warnings.append("No Vegas odds available - using player pool games for stacking fallback")

    # =========================================================================
    # 4. Minutes Coverage Check (if applicable)
    # =========================================================================
    # Check if player pool has meaningful projections (not all zeros/defaults)
    players_with_projection = sum(1 for p in player_pool if p.projected_ppg > 5)
    projection_coverage = players_with_projection / n_players if n_players > 0 else 0
    metrics['projection_coverage'] = projection_coverage

    if projection_coverage < 0.80:
        warnings.append(f"Low projection coverage: {projection_coverage:.0%} of players have meaningful projections")

    # =========================================================================
    # 5. Game Concentration Check
    # =========================================================================
    # Warn if too many players from single game (data quality issue)
    players_per_game = {}
    for p in player_pool:
        players_per_game[p.game_id] = players_per_game.get(p.game_id, 0) + 1

    max_players_per_game = max(players_per_game.values()) if players_per_game else 0
    avg_players_per_game = n_players / n_games if n_games > 0 else 0
    metrics['max_players_per_game'] = max_players_per_game
    metrics['avg_players_per_game'] = avg_players_per_game

    if max_players_per_game > 20:
        warnings.append(f"High player concentration: {max_players_per_game} players from single game")

    # =========================================================================
    # 6. Build Adjusted Config (if DEGRADE)
    # =========================================================================
    adjusted_config = None

    if status == "DEGRADE":
        adjusted_config = _build_degraded_config(config, reasons, metrics)

    # =========================================================================
    # Return Result
    # =========================================================================
    return SlateValidationResult(
        status=status,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        adjusted_config=adjusted_config,
        original_config=config
    )


def _build_degraded_config(
    original: OptimizerConfig,
    reasons: List[str],
    metrics: Dict
) -> OptimizerConfig:
    """
    Build an adjusted config when slate quality is degraded.

    Main adjustment: disable stacking and reallocate those lineups.
    """
    # Start with a copy of original values
    new_chalk = original.chalk_lineups
    new_stack = original.stack_lineups
    new_leverage = original.leverage_lineups
    new_news = original.news_lineups

    # If odds coverage issue, disable stacking
    odds_coverage = metrics.get('odds_coverage', 1.0)
    if odds_coverage < ODDS_COVERAGE_FOR_STACKING:
        # Reallocate stack lineups
        stack_to_reallocate = new_stack
        new_stack = 0

        # Distribute: 70% to leverage, 30% to news
        leverage_add = int(stack_to_reallocate * LEVERAGE_REALLOC_RATIO)
        news_add = stack_to_reallocate - leverage_add

        new_leverage += leverage_add
        new_news += news_add

    return OptimizerConfig(
        chalk_lineups=new_chalk,
        stack_lineups=new_stack,
        leverage_lineups=new_leverage,
        news_lineups=new_news,
        max_exposure_dominant=original.max_exposure_dominant,
        max_exposure_strong=original.max_exposure_strong,
        max_exposure_leverage=original.max_exposure_leverage,
        star_p_top1_threshold=original.star_p_top1_threshold,
        stack_score_threshold=original.stack_score_threshold,
        leverage_sigma_multiplier=original.leverage_sigma_multiplier,
        lineup_size=original.lineup_size,
        min_unique_players=original.min_unique_players
    )


def format_validation_metrics(result: SlateValidationResult) -> str:
    """Format validation metrics for display."""
    m = result.metrics
    lines = [
        f"Players: {m.get('n_players', 0)} across {m.get('n_games', 0)} games",
        f"Sim coverage: {m.get('sim_coverage', 0):.0%}",
        f"Odds coverage: {m.get('odds_coverage', 0):.0%} ({m.get('matched_games', 0)}/{m.get('pool_games', 0)} games)",
    ]

    if result.adjusted_config:
        orig = result.original_config
        adj = result.adjusted_config
        lines.append(f"Bucket adjustment: Stack {orig.stack_lineups}→{adj.stack_lineups}, "
                    f"Leverage {orig.leverage_lineups}→{adj.leverage_lineups}, "
                    f"News {orig.news_lineups}→{adj.news_lineups}")

    return "\n".join(lines)


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Slate Quality Gate Module")
    print("=" * 50)

    # Example validation result
    result = SlateValidationResult(
        status="DEGRADE",
        reasons=["Odds coverage 40% < 70% threshold for stacking"],
        warnings=["Low projection coverage: 85%"],
        metrics={
            'n_players': 100,
            'n_games': 6,
            'sim_coverage': 0.98,
            'odds_coverage': 0.40,
            'pool_games': 6,
            'matched_games': 2
        }
    )

    print(f"\nStatus: {result.status}")
    print(f"Summary:\n{result.summary}")
    print(f"Mode: {result.mode_label}")
