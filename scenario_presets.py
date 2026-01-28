#!/usr/bin/env python3
"""
Scenario Presets for Tournament Portfolio Generation

Presets encode strategic assumptions about game environments and adjust
portfolio construction accordingly.

Available Presets:
- BALANCED: Default mode, no adjustments
- CLOSE_GAMES: Target tight games (spread < 4), maximize stacking
- SHOOTOUT: High total + tight spread, both teams scoring
- BLOWOUT: Target garbage-time beneficiaries or avoid blowouts
- CHAOS: Maximum variance for low-probability moonshots

Usage:
    from scenario_presets import PRESETS, apply_scenario_preset

    preset = PRESETS["SHOOTOUT"]
    adjusted_config = apply_scenario_preset(config, preset, game_envs)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from lineup_optimizer import OptimizerConfig


@dataclass
class ScenarioPreset:
    """Configuration preset for a specific game scenario."""
    name: str
    description: str
    icon: str

    # Bucket adjustments (deltas from base config)
    chalk_delta: int = 0
    stack_delta: int = 0
    leverage_delta: int = 0
    news_delta: int = 0

    # Exposure adjustments
    exposure_multiplier_dominant: float = 1.0  # Multiply max_exposure_dominant
    exposure_multiplier_strong: float = 1.0
    exposure_multiplier_leverage: float = 1.0

    # Threshold adjustments
    stack_score_threshold_override: Optional[float] = None
    star_p_top1_threshold_override: Optional[float] = None
    leverage_sigma_multiplier_override: Optional[float] = None

    # Core overlap adjustment (duplication penalty)
    max_core_overlap_override: Optional[int] = None

    # Game environment filters
    min_game_total: Optional[float] = None  # Only stack games with total >= this
    max_game_spread: Optional[float] = None  # Only stack games with spread <= this
    min_game_spread: Optional[float] = None  # For blowout targeting

    # Strategy flags
    prefer_opposing_stacks: bool = False  # Prefer players from opposite teams in same game
    avoid_blowout_games: bool = False     # Skip games with spread > 10
    maximize_variance: bool = False       # Prefer high-sigma players

    @property
    def summary(self) -> str:
        """Human-readable summary of preset effects."""
        effects = []
        if self.chalk_delta != 0:
            effects.append(f"Chalk {'+' if self.chalk_delta > 0 else ''}{self.chalk_delta}")
        if self.stack_delta != 0:
            effects.append(f"Stack {'+' if self.stack_delta > 0 else ''}{self.stack_delta}")
        if self.leverage_delta != 0:
            effects.append(f"Leverage {'+' if self.leverage_delta > 0 else ''}{self.leverage_delta}")
        if self.news_delta != 0:
            effects.append(f"News {'+' if self.news_delta > 0 else ''}{self.news_delta}")
        if self.max_game_spread is not None:
            effects.append(f"Spread ≤ {self.max_game_spread}")
        if self.min_game_total is not None:
            effects.append(f"Total ≥ {self.min_game_total}")
        if self.maximize_variance:
            effects.append("Max variance")
        return " | ".join(effects) if effects else "No adjustments"


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

PRESETS: Dict[str, ScenarioPreset] = {
    "BALANCED": ScenarioPreset(
        name="Balanced",
        description="Default mode with no adjustments. Balanced exposure across all buckets.",
        icon="⚖️",
    ),

    "CLOSE_GAMES": ScenarioPreset(
        name="Close Games",
        description="Target tight games (spread < 4) where both teams' stars stay in late.",
        icon="🎯",
        stack_delta=2,       # More stacks (8 instead of 6)
        chalk_delta=-1,      # Fewer chalk (5 instead of 6)
        news_delta=-1,       # Fewer news (1 instead of 2)
        max_game_spread=4.0,
        stack_score_threshold_override=0.60,  # Lower threshold to catch more close games
        prefer_opposing_stacks=True,
    ),

    "SHOOTOUT": ScenarioPreset(
        name="Shootout",
        description="High-scoring close games. Both teams' players benefit from fast pace.",
        icon="🔥",
        stack_delta=3,       # Heavy stacking (9 instead of 6)
        leverage_delta=-2,   # Fewer contrarian (4 instead of 6)
        news_delta=-1,       # Fewer news (1 instead of 2)
        min_game_total=230.0,
        max_game_spread=5.0,
        stack_score_threshold_override=0.50,
        prefer_opposing_stacks=True,
        exposure_multiplier_dominant=1.2,  # Allow more star exposure
    ),

    "BLOWOUT": ScenarioPreset(
        name="Blowout Risk",
        description="Avoid games with large spreads. Target bench players who benefit from garbage time.",
        icon="💨",
        stack_delta=-3,      # Fewer stacks (3 instead of 6) - blowouts kill stacks
        leverage_delta=2,    # More contrarian (8 instead of 6)
        news_delta=1,        # More news (3 instead of 2)
        avoid_blowout_games=True,
        min_game_spread=8.0,  # Only consider blowout games for leverage plays
        exposure_multiplier_leverage=1.3,  # Allow more leverage exposure
    ),

    "CHAOS": ScenarioPreset(
        name="Chaos Mode",
        description="Maximum variance for low-probability moonshots. Go big or go home.",
        icon="🌪️",
        chalk_delta=-3,      # Minimal chalk (3 instead of 6)
        leverage_delta=4,    # Heavy leverage (10 instead of 6)
        news_delta=1,        # Extra news (3 instead of 2)
        stack_delta=-2,      # Fewer stacks (4 instead of 6)
        maximize_variance=True,
        leverage_sigma_multiplier_override=1.2,  # Lower bar for high-variance plays
        max_core_overlap_override=5,  # Allow more overlap for correlated moonshots
        exposure_multiplier_dominant=0.8,  # Reduce star exposure
        exposure_multiplier_leverage=1.5,  # Boost leverage exposure
    ),

    "STARS_ONLY": ScenarioPreset(
        name="Stars Align",
        description="Concentrate on elite talent. Fewer unique players, higher ceiling.",
        icon="⭐",
        chalk_delta=3,       # Heavy chalk (9 instead of 6)
        stack_delta=-2,      # Fewer stacks (4 instead of 6)
        leverage_delta=-2,   # Fewer contrarian (4 instead of 6)
        news_delta=1,        # Keep news for pivots
        exposure_multiplier_dominant=1.4,  # Allow much more star exposure
        exposure_multiplier_strong=1.2,
        star_p_top1_threshold_override=0.03,  # Lower bar for "star" status
        max_core_overlap_override=4,  # Allow more core overlap
    ),
}


# =============================================================================
# PRESET APPLICATION
# =============================================================================

def apply_scenario_preset(
    base_config: OptimizerConfig,
    preset: ScenarioPreset,
    game_envs: Dict[str, dict] = None
) -> OptimizerConfig:
    """
    Apply a scenario preset to an optimizer config.

    Args:
        base_config: Original optimizer configuration
        preset: Scenario preset to apply
        game_envs: Optional game environments for filtering

    Returns:
        New OptimizerConfig with preset adjustments
    """
    # Calculate new bucket counts (clamped to valid range)
    new_chalk = max(0, min(15, base_config.chalk_lineups + preset.chalk_delta))
    new_stack = max(0, min(15, base_config.stack_lineups + preset.stack_delta))
    new_leverage = max(0, min(15, base_config.leverage_lineups + preset.leverage_delta))
    new_news = max(0, min(10, base_config.news_lineups + preset.news_delta))

    # Ensure total doesn't exceed 20
    total = new_chalk + new_stack + new_leverage + new_news
    if total > 20:
        # Scale down proportionally
        scale = 20 / total
        new_chalk = int(new_chalk * scale)
        new_stack = int(new_stack * scale)
        new_leverage = int(new_leverage * scale)
        new_news = 20 - new_chalk - new_stack - new_leverage

    # Apply exposure multipliers
    new_exp_dominant = min(1.0, base_config.max_exposure_dominant * preset.exposure_multiplier_dominant)
    new_exp_strong = min(1.0, base_config.max_exposure_strong * preset.exposure_multiplier_strong)
    new_exp_leverage = min(1.0, base_config.max_exposure_leverage * preset.exposure_multiplier_leverage)

    # Apply threshold overrides
    new_stack_threshold = (
        preset.stack_score_threshold_override
        if preset.stack_score_threshold_override is not None
        else base_config.stack_score_threshold
    )
    new_star_threshold = (
        preset.star_p_top1_threshold_override
        if preset.star_p_top1_threshold_override is not None
        else base_config.star_p_top1_threshold
    )
    new_leverage_sigma = (
        preset.leverage_sigma_multiplier_override
        if preset.leverage_sigma_multiplier_override is not None
        else base_config.leverage_sigma_multiplier
    )
    new_core_overlap = (
        preset.max_core_overlap_override
        if preset.max_core_overlap_override is not None
        else base_config.max_core_overlap
    )

    return OptimizerConfig(
        chalk_lineups=new_chalk,
        stack_lineups=new_stack,
        leverage_lineups=new_leverage,
        news_lineups=new_news,
        max_exposure_dominant=new_exp_dominant,
        max_exposure_strong=new_exp_strong,
        max_exposure_leverage=new_exp_leverage,
        star_p_top1_threshold=new_star_threshold,
        stack_score_threshold=new_stack_threshold,
        leverage_sigma_multiplier=new_leverage_sigma,
        lineup_size=base_config.lineup_size,
        min_unique_players=base_config.min_unique_players,
        max_core_overlap=new_core_overlap,
        min_portfolio_diversity=base_config.min_portfolio_diversity,
    )


def filter_game_environments(
    game_envs: Dict[str, dict],
    preset: ScenarioPreset
) -> Dict[str, dict]:
    """
    Filter game environments based on preset criteria.

    Returns only games that match the preset's requirements.
    """
    filtered = {}

    for game_id, env in game_envs.items():
        total = env.get('implied_total', 220)
        spread = abs(env.get('spread', 0))

        # Apply filters
        if preset.min_game_total is not None and total < preset.min_game_total:
            continue
        if preset.max_game_spread is not None and spread > preset.max_game_spread:
            continue
        if preset.avoid_blowout_games and spread > 10:
            continue
        if preset.min_game_spread is not None and spread < preset.min_game_spread:
            continue

        filtered[game_id] = env

    return filtered


def get_preset_recommendation(game_envs: Dict[str, dict]) -> str:
    """
    Recommend a preset based on tonight's slate characteristics.

    Analyzes game environments to suggest the most appropriate preset.

    UPDATED: Backtest evidence shows CLOSE_GAMES outperforms overall and on
    large slates. High totals don't correlate well with top scorers.
    Default is now CLOSE_GAMES unless slate has specific characteristics.
    """
    if not game_envs:
        return "CLOSE_GAMES"  # Changed from BALANCED

    totals = [env.get('implied_total', 220) for env in game_envs.values()]
    spreads = [abs(env.get('spread', 0)) for env in game_envs.values()]

    avg_total = sum(totals) / len(totals)
    avg_spread = sum(spreads) / len(spreads)
    close_games = sum(1 for s in spreads if s <= 4)
    blowout_games = sum(1 for s in spreads if s >= 10)
    high_total_games = sum(1 for t in totals if t >= 230)

    # Decision logic - prioritize spread (close games) over totals
    # Evidence: only 32% of top-3 scorers came from high-total games

    # Small slate with mostly blowouts = BLOWOUT preset
    if blowout_games >= len(game_envs) * 0.5:
        return "BLOWOUT"

    # Small slate (3 or fewer games) = concentrate on elites
    if len(game_envs) <= 3:
        return "STARS_ONLY"

    # If we have several close games AND high totals = SHOOTOUT
    # (only use this combination, not totals alone)
    if high_total_games >= 2 and close_games >= 2 and avg_spread <= 5:
        return "SHOOTOUT"

    # Default: CLOSE_GAMES performs best across most slate types
    return "CLOSE_GAMES"


def format_preset_info(preset: ScenarioPreset) -> str:
    """Format preset information for display."""
    return f"""
**{preset.icon} {preset.name}**
{preset.description}

*Adjustments:* {preset.summary}
"""


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Scenario Presets Module")
    print("=" * 60)

    for preset_name, preset in PRESETS.items():
        print(f"\n{preset.icon} {preset_name}")
        print(f"   {preset.description}")
        print(f"   Effects: {preset.summary}")

    # Test preset application
    print("\n" + "=" * 60)
    print("Testing preset application:")

    base = OptimizerConfig()
    print(f"\nBase config: C={base.chalk_lineups}, S={base.stack_lineups}, L={base.leverage_lineups}, N={base.news_lineups}")

    for preset_name in ["CLOSE_GAMES", "SHOOTOUT", "CHAOS"]:
        preset = PRESETS[preset_name]
        adjusted = apply_scenario_preset(base, preset)
        print(f"{preset_name}: C={adjusted.chalk_lineups}, S={adjusted.stack_lineups}, L={adjusted.leverage_lineups}, N={adjusted.news_lineups}")
