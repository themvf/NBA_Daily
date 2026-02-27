"""
DraftKings NBA DFS Lineup Optimizer

Generates optimized DFS lineups for DraftKings NBA Classic contests.
Supports 100+ diversified GPP tournament lineups with exposure limits.

DraftKings Classic Rules:
- Salary Cap: $50,000
- Roster: 8 positions (PG, SG, SF, PF, C, G, F, UTIL)
- Constraint: Players from at least 2 different NBA games
- Scoring: PTS (+1), 3PM (+0.5), REB (+1.25), AST (+1.5),
           STL (+2), BLK (+2), TO (-0.5), DD (+1.5), TD (+3)

Version 2.0: Integrated sophisticated prediction system with rest days,
             uncertainty multipliers, and position-specific PPM adjustments.
"""

from __future__ import annotations

import random
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Import injury module for filtering out injured players
try:
    import injury_adjustment as ia
    INJURY_MODULE_AVAILABLE = True
except ImportError:
    INJURY_MODULE_AVAILABLE = False

# Import sophisticated prediction modules
try:
    import prediction_generator as pg
    from rest_days import calculate_rest_factor, get_rest_multiplier
    import player_correlation_analytics as pca
    import defense_type_analytics as dta
    import position_ppm_stats
    SOPHISTICATED_PREDICTIONS = True
except ImportError:
    SOPHISTICATED_PREDICTIONS = False

# Import correlation model for GPP optimization
try:
    from correlation_model import (
        PlayerCorrelationModel,
        PlayerSlateInfo,
        CorrelatedSimResult,
        create_player_slate_info,
        CorrelationConfig
    )
    CORRELATION_MODEL_AVAILABLE = True
except ImportError:
    CORRELATION_MODEL_AVAILABLE = False

# Import injury impact analytics for teammate injury stat boosts
try:
    from injury_impact_analytics import (
        get_injury_stat_boosts,
        InjuryStatBoost,
    )
    INJURY_IMPACT_AVAILABLE = True
except ImportError:
    INJURY_IMPACT_AVAILABLE = False


# =============================================================================
# Injury Filtering
# =============================================================================

def get_injured_player_ids(conn: sqlite3.Connection) -> Set[int]:
    """
    Get set of player IDs who are currently injured (OUT or DOUBTFUL).

    Uses the injury_list table to identify players who should be excluded
    from DFS lineups.

    Returns:
        Set of player_ids who are injured/out
    """
    if not INJURY_MODULE_AVAILABLE:
        return set()

    try:
        injuries = ia.get_active_injuries(conn, status_filter=['out', 'doubtful'])
        return {inj['player_id'] for inj in injuries if inj.get('player_id')}
    except Exception:
        # Fallback: try direct query if injury_adjustment fails
        try:
            query = """
                SELECT DISTINCT player_id
                FROM injury_list
                WHERE LOWER(status) IN ('out', 'doubtful', 'active')
            """
            df = pd.read_sql_query(query, conn)
            return set(df['player_id'].tolist())
        except Exception:
            return set()


def get_injured_player_names(conn: sqlite3.Connection) -> Dict[int, str]:
    """
    Get mapping of injured player IDs to their names and status.

    Returns:
        Dict mapping player_id -> "name (status)"
    """
    if not INJURY_MODULE_AVAILABLE:
        return {}

    try:
        injuries = ia.get_active_injuries(conn, status_filter=['out', 'doubtful', 'questionable'])
        return {
            inj['player_id']: f"{inj['player_name']} ({inj.get('status', 'OUT')})"
            for inj in injuries if inj.get('player_id')
        }
    except Exception:
        try:
            query = """
                SELECT player_id, player_name, status
                FROM injury_list
                WHERE LOWER(status) IN ('out', 'doubtful', 'questionable', 'active')
            """
            df = pd.read_sql_query(query, conn)
            return {
                row['player_id']: f"{row['player_name']} ({row['status']})"
                for _, row in df.iterrows()
            }
        except Exception:
            return {}


# =============================================================================
# Name Normalization & Matching
# =============================================================================

# Common nickname mappings (DraftKings name -> possible NBA API names)
NICKNAME_MAP = {
    # First name nicknames
    'nic': ['nicolas', 'nicholas', 'nick'],
    'nick': ['nicolas', 'nicholas', 'nic'],
    'herb': ['herbert'],
    'herbert': ['herb'],
    'cam': ['cameron'],
    'cameron': ['cam'],
    'mike': ['michael'],
    'michael': ['mike'],
    'chris': ['christopher'],
    'christopher': ['chris'],
    'matt': ['matthew'],
    'matthew': ['matt'],
    'rob': ['robert'],
    'robert': ['rob', 'bobby'],
    'bobby': ['robert', 'bob'],
    'will': ['william'],
    'william': ['will', 'willie', 'bill'],
    'alex': ['alexander', 'alexandre'],
    'alexander': ['alex'],
    'tony': ['anthony'],
    'anthony': ['tony'],
    'tim': ['timothy'],
    'timothy': ['tim'],
    'dan': ['daniel'],
    'daniel': ['dan', 'danny'],
    'danny': ['daniel', 'dan'],
    'ben': ['benjamin'],
    'benjamin': ['ben'],
    'jon': ['jonathan', 'john'],
    'jonathan': ['jon'],
    'josh': ['joshua'],
    'joshua': ['josh'],
    'nate': ['nathan', 'nathaniel'],
    'nathan': ['nate'],
    'nathaniel': ['nate'],
    'drew': ['andrew'],
    'andrew': ['drew', 'andy'],
    'andy': ['andrew'],
    'jake': ['jacob'],
    'jacob': ['jake'],
    'joe': ['joseph'],
    'joseph': ['joe', 'joey'],
    'joey': ['joseph', 'joe'],
    'max': ['maximilian', 'maxwell'],
    'ty': ['tyler', 'tyrese', 'tyrus'],
    'tj': ['t.j.'],
    'pj': ['p.j.'],
    'cj': ['c.j.'],
    'rj': ['r.j.'],
    'aj': ['a.j.'],
    'dj': ['d.j.'],
    'jt': ['j.t.'],
    'kt': ['k.t.'],
    # Special cases
    'gregory': ['gg', 'greg'],
    'gg': ['gregory', 'greg'],
    'greg': ['gregory', 'gg'],
    'scotty': ['scottie', 'scott'],
    'scottie': ['scotty', 'scott'],
    'scott': ['scottie', 'scotty'],
    'marjon': ['mar-jon'],  # MarJon Beauchamp
}

# Known problematic player name mappings (DK name -> NBA API name)
# Add specific overrides here as discovered
PLAYER_NAME_OVERRIDES = {
    'nicolas claxton': 'nic claxton',
    'nic claxton': 'nicolas claxton',
    'herbert jones': 'herb jones',
    'herb jones': 'herbert jones',
    'cameron johnson': 'cam johnson',
    'cam johnson': 'cameron johnson',
    'p.j. washington': 'pj washington',
    'pj washington': 'p.j. washington',
    'c.j. mccollum': 'cj mccollum',
    'cj mccollum': 'c.j. mccollum',
    'r.j. barrett': 'rj barrett',
    'rj barrett': 'r.j. barrett',
    'o.g. anunoby': 'og anunoby',
    'og anunoby': 'o.g. anunoby',
    't.j. mcconnell': 'tj mcconnell',
    'tj mcconnell': 't.j. mcconnell',
    'd.j. augustin': 'dj augustin',
    'dj augustin': 'd.j. augustin',
    'a.j. green': 'aj green',
    'aj green': 'a.j. green',
    'kenyon martin jr.': 'kenyon martin jr',
    'kenyon martin jr': 'kenyon martin jr.',
    'gary payton ii': 'gary payton',
    'larry nance jr.': 'larry nance jr',
    'larry nance jr': 'larry nance jr.',
    'tim hardaway jr.': 'tim hardaway jr',
    'tim hardaway jr': 'tim hardaway jr.',
    'kelly oubre jr.': 'kelly oubre jr',
    'kelly oubre jr': 'kelly oubre jr.',
    'marcus morris sr.': 'marcus morris sr',
    'marcus morris sr': 'marcus morris sr.',
    'wendell carter jr.': 'wendell carter jr',
    'wendell carter jr': 'wendell carter jr.',
    'jaren jackson jr.': 'jaren jackson jr',
    'jaren jackson jr': 'jaren jackson jr.',
    'marvin bagley iii': 'marvin bagley',
    'robert williams iii': 'robert williams',
    # GG Jackson / Gregory Jackson
    'gregory jackson': 'gg jackson',
    'gg jackson': 'gregory jackson',
    'gregory jackson ii': 'gg jackson',
    # Scotty/Scottie Pippen
    'scotty pippen jr': 'scottie pippen jr',
    'scottie pippen jr': 'scotty pippen jr',
    'scotty pippen': 'scottie pippen jr',
    # Fred VanVleet - space variations
    'fred vanvleet': 'fred van vleet',
    'fred van vleet': 'fred vanvleet',
    # MarJon Beauchamp
    'marjon beauchamp': 'mar-jon beauchamp',
    'mar-jon beauchamp': 'marjon beauchamp',
}


def normalize_name(name: str) -> str:
    """
    Normalize a player name for matching.

    - Converts to lowercase
    - Removes accents (Jokić -> jokic)
    - Removes periods from initials (P.J. -> pj)
    - Removes suffixes (Jr., III, II, Sr.)
    - Normalizes whitespace
    - Removes hyphens for comparison
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower().strip()

    # Remove accents (NFD decomposition, then remove combining marks)
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')

    # Remove periods (P.J. -> PJ)
    name = name.replace('.', '')

    # Remove common suffixes
    suffixes = [' jr', ' sr', ' iii', ' ii', ' iv', ' v']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Normalize whitespace
    name = ' '.join(name.split())

    return name


def generate_name_variants(name: str) -> Set[str]:
    """
    Generate possible name variants for fuzzy matching.

    Returns a set of normalized name variants to try matching against.
    """
    variants = set()
    normalized = normalize_name(name)
    variants.add(normalized)

    # Check direct overrides
    if normalized in PLAYER_NAME_OVERRIDES:
        variants.add(normalize_name(PLAYER_NAME_OVERRIDES[normalized]))

    # Split into parts
    parts = normalized.split()
    if len(parts) < 2:
        return variants

    first_name = parts[0]
    last_name = ' '.join(parts[1:])  # Handle compound last names

    # Add nickname variants for first name
    if first_name in NICKNAME_MAP:
        for alt_first in NICKNAME_MAP[first_name]:
            variants.add(f"{alt_first} {last_name}")

    # Also try without hyphen in last name
    if '-' in last_name:
        variants.add(f"{first_name} {last_name.replace('-', ' ')}")
        variants.add(f"{first_name} {last_name.replace('-', '')}")

    # Try with hyphen removed from entire name
    if '-' in normalized:
        variants.add(normalized.replace('-', ' '))
        variants.add(normalized.replace('-', ''))

    # Handle compound last names (VanVleet <-> Van Vleet)
    # Check for CamelCase patterns in original name
    original_parts = name.split()
    if len(original_parts) >= 2:
        original_last = ' '.join(original_parts[1:])
        # Check for internal capitals suggesting compound name (e.g., "VanVleet")
        if re.search(r'[a-z][A-Z]', original_last):
            # Split on internal capitals: "VanVleet" -> "Van Vleet"
            split_last = re.sub(r'([a-z])([A-Z])', r'\1 \2', original_last).lower()
            variants.add(f"{first_name} {split_last}")

    # Also try joining multi-word last names (Van Vleet -> vanvleet)
    if len(parts) > 2:
        joined_last = ''.join(parts[1:])
        variants.add(f"{first_name} {joined_last}")

    return variants


def find_best_player_match(
    dk_name: str,
    dk_team: str,
    db_lookup: Dict[str, Dict],
    team_lookup: Dict[str, List[str]]
) -> Optional[int]:
    """
    Find the best matching player_id for a DraftKings player name.

    Args:
        dk_name: Player name from DraftKings
        dk_team: Team abbreviation from DraftKings
        db_lookup: Dict mapping normalized name -> {player_id, team}
        team_lookup: Dict mapping team abbreviation -> list of normalized player names

    Returns:
        player_id if match found, None otherwise
    """
    # Generate all variants of the DK name
    variants = generate_name_variants(dk_name)

    # Try exact match first
    for variant in variants:
        if variant in db_lookup:
            return db_lookup[variant]['player_id']

    # Try team-constrained partial match
    dk_team_upper = dk_team.upper() if dk_team else None
    if dk_team_upper and dk_team_upper in team_lookup:
        team_players = team_lookup[dk_team_upper]

        # Get last name from DK name for partial matching
        dk_parts = normalize_name(dk_name).split()
        if len(dk_parts) >= 2:
            dk_last = dk_parts[-1]
            dk_first = dk_parts[0]

            for db_name in team_players:
                db_parts = db_name.split()
                if len(db_parts) >= 2:
                    db_last = db_parts[-1]
                    db_first = db_parts[0]

                    # Match if last names match and first names are similar
                    if dk_last == db_last:
                        # Check if first names match or are nickname variants
                        if dk_first == db_first:
                            return db_lookup[db_name]['player_id']

                        # Check nickname variants
                        dk_first_variants = {dk_first}
                        if dk_first in NICKNAME_MAP:
                            dk_first_variants.update(NICKNAME_MAP[dk_first])

                        if db_first in dk_first_variants:
                            return db_lookup[db_name]['player_id']

    # Fallback: try substring matching (for compound names)
    for variant in variants:
        for db_name, info in db_lookup.items():
            # Check if one contains the other (handles "Gilgeous-Alexander" variations)
            if variant in db_name or db_name in variant:
                # Verify team if available
                if not dk_team_upper or info.get('team', '').upper() == dk_team_upper:
                    return info['player_id']

    return None


# =============================================================================
# DraftKings Fantasy Point Calculator
# =============================================================================

def calculate_dk_fantasy_points(
    points: float,
    rebounds: float,
    assists: float,
    steals: float,
    blocks: float,
    turnovers: float,
    fg3m: float
) -> float:
    """
    Calculate DraftKings Classic fantasy points.

    Scoring:
    - Points: +1.0
    - 3-Pointers Made: +0.5
    - Rebounds: +1.25
    - Assists: +1.5
    - Steals: +2.0
    - Blocks: +2.0
    - Turnovers: -0.5
    - Double-Double: +1.5 (bonus)
    - Triple-Double: +3.0 (bonus, replaces DD bonus)

    Returns:
        float: Total DraftKings fantasy points
    """
    # Base scoring
    base = (
        points * 1.0 +
        fg3m * 0.5 +
        rebounds * 1.25 +
        assists * 1.5 +
        steals * 2.0 +
        blocks * 2.0 +
        turnovers * -0.5
    )

    # Count categories >= 10 for double/triple-double bonus
    # Eligible categories: points, rebounds, assists, steals, blocks
    categories = sum(
        1 for x in [points, rebounds, assists, steals, blocks]
        if x is not None and x >= 10
    )

    if categories >= 3:
        base += 3.0  # Triple-double bonus
    elif categories >= 2:
        base += 1.5  # Double-double bonus

    return base


def calculate_dk_fpts_from_row(row: pd.Series) -> float:
    """Calculate DK fantasy points from a DataFrame row."""
    return calculate_dk_fantasy_points(
        points=row.get('points', 0) or 0,
        rebounds=row.get('rebounds', 0) or 0,
        assists=row.get('assists', 0) or 0,
        steals=row.get('steals', 0) or 0,
        blocks=row.get('blocks', 0) or 0,
        turnovers=row.get('turnovers', 0) or 0,
        fg3m=row.get('fg3m', 0) or 0
    )


# =============================================================================
# DFS Player Data Class
# =============================================================================

@dataclass
class DFSPlayer:
    """Represents a player in the DFS player pool."""
    player_id: int
    name: str
    team: str
    opponent: str
    game_id: str
    positions: List[str]  # From DK CSV (e.g., ["PG", "SG"])
    salary: int           # From DK CSV

    # Projections (calculated from historical data)
    proj_points: float = 0.0
    proj_rebounds: float = 0.0
    proj_assists: float = 0.0
    proj_steals: float = 0.0
    proj_blocks: float = 0.0
    proj_turnovers: float = 0.0
    proj_fg3m: float = 0.0

    # Fantasy outputs
    proj_fpts: float = 0.0       # Expected DK fantasy points
    proj_ceiling: float = 0.0   # 90th percentile outcome
    proj_floor: float = 0.0     # 10th percentile outcome
    fpts_per_dollar: float = 0.0  # Value metric (proj_fpts / salary * 1000)

    # GPP metrics
    ownership_proj: float = 0.0  # Expected ownership %
    leverage_score: float = 0.0  # Ceiling / ownership ratio
    vegas_implied_fpts: float = 0.0  # Vegas-implied fantasy points from props feed
    vegas_edge_pct: float = 0.0      # (Vegas implied - model proj) / model proj
    vegas_signal: str = ""           # BOOST / WATCH / FADE guidance from props edge

    # Status flags
    is_locked: bool = False      # Force include in all lineups
    is_excluded: bool = False    # Never include in lineups
    is_injured: bool = False     # Player is OUT/DOUBTFUL - auto-excluded
    injury_status: str = ""      # Injury status (OUT, DOUBTFUL, etc.)
    is_fallback: bool = False    # Using DK avg instead of our projection (unmatched player)

    # DraftKings data
    dk_id: str = ""
    dk_avg_pts: float = 0.0  # AvgPointsPerGame from DK CSV

    # Rest day tracking (sophisticated predictions)
    days_rest: Optional[int] = None
    rest_multiplier: float = 1.0

    # Uncertainty tracking for GPP strategy
    uncertainty_multiplier: float = 1.0

    # Analytics indicators (e.g., "🎯🛡️" for correlation + defense)
    analytics_used: str = ""

    # Player position for PPM lookups
    position: str = ""

    # Correlation model outputs (GPP optimization)
    p_top1: float = 0.0           # Probability of being #1 scorer on slate
    p_top3: float = 0.0           # Probability of top 3 finish
    expected_rank: float = 0.0    # Expected rank on slate
    support_score: float = 0.0    # P(top-10 | not #1) for lineup support
    sigma: float = 0.0            # Scoring standard deviation for simulations

    # Stack scoring
    stack_score: float = 0.0      # Game stacking quality (0-1)
    is_star: bool = False         # Season avg >= 25 PPG

    # Minutes validation
    recent_minutes_avg: float = 0.0  # Last 5 games avg minutes
    minutes_validated: bool = True    # False if player fails minutes check
    minutes_variance: float = 0.0     # Last 10 games variance of non-zero minutes

    # Research / scouting fields
    role_tier: str = ""              # STAR, STARTER, ROTATION, BENCH
    avg_fpts_last7: float = 0.0     # Average DK FPTS over last 7 games
    fpts_variance: float = 0.0      # Std dev of recent DK FPTS

    # Segment-level calibration/risk controls
    primary_position: str = ""        # Canonical primary position bucket (PG/SG/SF/PF/C/UNK)
    salary_bucket: str = ""           # VALUE/LOW_MID/MID/UPPER/STUD
    segment_bias_correction: float = 0.0
    segment_mae: float = 0.0
    segment_sample_size: int = 0
    segment_risk_penalty: float = 0.0
    risk_adjusted_proj_fpts: float = 0.0
    segment_exposure_cap: float = 1.0
    is_overproj_redflag: bool = False

    def calculate_projections(self) -> None:
        """Calculate derived projection values."""
        self.proj_fpts = calculate_dk_fantasy_points(
            self.proj_points,
            self.proj_rebounds,
            self.proj_assists,
            self.proj_steals,
            self.proj_blocks,
            self.proj_turnovers,
            self.proj_fg3m
        )

        # Value metric: fantasy points per $1000 salary
        if self.salary > 0:
            self.fpts_per_dollar = (self.proj_fpts / self.salary) * 1000

        # Leverage score: projection + amplified upside, relative to ownership
        if self.ownership_proj > 0:
            upside = max(0, self.proj_ceiling - self.proj_fpts)
            self.leverage_score = (self.proj_fpts + upside * 1.5) / max(1, self.ownership_proj)

    def can_fill_slot(self, slot: str) -> bool:
        """Check if player can fill a roster slot."""
        slot_positions = POSITION_SLOTS.get(slot, [])
        return any(pos in slot_positions for pos in self.positions)


# =============================================================================
# Position Mapping
# =============================================================================

POSITION_SLOTS = {
    'PG': ['PG'],
    'SG': ['SG'],
    'SF': ['SF'],
    'PF': ['PF'],
    'C': ['C'],
    'G': ['PG', 'SG'],
    'F': ['SF', 'PF'],
    'UTIL': ['PG', 'SG', 'SF', 'PF', 'C']
}

ROSTER_SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
SALARY_CAP = 50000
MIN_GAMES = 2
MIN_PLAYER_SALARY = 3000
MIN_SALARY_PROJ_FPTS_GATE = 12.0
MAX_MIN_SALARY_PLAYERS_PER_LINEUP = 1
MINUTES_VARIANCE_FILTER_MIN_SAMPLE = 8
MINUTES_VARIANCE_FILTER_PERCENTILE = 75.0
MINUTES_VARIANCE_ELITE_PROJ_FPTS = 34.0
MINUTES_VARIANCE_ELITE_CEILING = 50.0
MAX_REDFLAG_SEGMENTS_PER_LINEUP = 1

# Segment calibration constants from DFS Player Review diagnostics.
# avg_error is defined as (actual_fpts - proj_fpts), so:
# - negative => historically over-projected (needs downward correction)
# - positive => historically under-projected (needs upward correction)
SEGMENT_CALIBRATION_METRICS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("C", "UPPER"): {"avg_error": 6.99, "mae": 10.09, "samples": 8},
    ("SF", "UPPER"): {"avg_error": -6.27, "mae": 11.08, "samples": 5},
    ("PG", "UPPER"): {"avg_error": -6.02, "mae": 12.47, "samples": 27},
    ("SG", "MID"): {"avg_error": -4.29, "mae": 7.77, "samples": 22},
    ("C", "MID"): {"avg_error": -3.90, "mae": 9.59, "samples": 29},
    ("PF", "MID"): {"avg_error": -3.00, "mae": 7.00, "samples": 11},
    ("C", "LOW_MID"): {"avg_error": -2.49, "mae": 11.10, "samples": 39},
    ("PG", "STUD"): {"avg_error": 2.23, "mae": 10.56, "samples": 7},
}

# Segment-aware caps for historically problematic buckets.
SEGMENT_EXPOSURE_CAPS: Dict[Tuple[str, str], float] = {
    ("PG", "UPPER"): 0.25,
    ("SG", "MID"): 0.35,
    ("C", "LOW_MID"): 0.30,
}

SEGMENT_SHRINKAGE_K = 30.0
SEGMENT_CORRECTION_CAP = 4.0
SEGMENT_MAE_BASELINE = 8.0
SEGMENT_MAE_PENALTY_WEIGHT = 0.18
SEGMENT_MAE_PENALTY_CAP = 2.0

# League average stats per game (2025-26 season, for opponent adjustment)
LEAGUE_AVG_TEAM_REB = 44.0
LEAGUE_AVG_TEAM_AST = 25.5
LEAGUE_AVG_TEAM_FG3M = 12.5

# Lineup model profiles used by the DFS builder UI.
# strategy_mix tuples are (strategy, randomization_factor, target_pct).
LINEUP_MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "standard_v1": {
        "label": "Standard v1",
        "description": "Balanced baseline blend of projection, ceiling, value, and leverage.",
        "aggressive_ceiling_stack": False,
        "overlap_cap": 6,
        "core_play_inject_prob": 0.20,
        "core_play_min_own_pct": 14.0,
        "core_play_max_own_pct": 60.0,
        "core_play_min_projection": 24.0,
        "low_own_inject_prob": 0.12,
        "low_own_max_own_pct": 10.0,
        "low_own_min_projection": 22.0,
        "standout_lock_prob": 0.10,
        "standout_min_ceiling_gap": 7.0,
        "strategy_mix": [
            ("projection", 0.25, 0.20),
            ("ceiling", 0.30, 0.25),
            ("value", 0.25, 0.15),
            ("leverage", 0.35, 0.15),
            ("projection", 0.60, 0.25),
        ],
    },
    "spike_v1_legacy": {
        "label": "Spike v1 (Legacy)",
        "description": "Legacy spike profile with stronger upside and leverage tilt.",
        "aggressive_ceiling_stack": True,
        "overlap_cap": 5,
        "core_play_inject_prob": 0.12,
        "core_play_min_own_pct": 14.0,
        "core_play_max_own_pct": 60.0,
        "core_play_min_projection": 23.0,
        "low_own_inject_prob": 0.30,
        "low_own_max_own_pct": 12.0,
        "low_own_min_projection": 20.0,
        "standout_lock_prob": 0.25,
        "standout_min_ceiling_gap": 8.0,
        "strategy_mix": [
            ("ceiling", 0.24, 0.45),
            ("leverage", 0.30, 0.25),
            ("projection", 0.35, 0.15),
            ("projection", 0.68, 0.15),
        ],
    },
    "spike_v2_tail": {
        "label": "Spike v2 (Tail)",
        "description": "Tail-seeking profile emphasizing ceiling outcomes and contrarian leverage.",
        "aggressive_ceiling_stack": True,
        "overlap_cap": 4,
        "core_play_inject_prob": 0.08,
        "core_play_min_own_pct": 15.0,
        "core_play_max_own_pct": 60.0,
        "core_play_min_projection": 24.0,
        "low_own_inject_prob": 0.42,
        "low_own_max_own_pct": 14.0,
        "low_own_min_projection": 18.0,
        "standout_lock_prob": 0.35,
        "standout_min_ceiling_gap": 9.5,
        "strategy_mix": [
            ("ceiling", 0.20, 0.55),
            ("leverage", 0.28, 0.25),
            ("projection", 0.48, 0.20),
        ],
    },
    "cluster_v1_experimental": {
        "label": "Cluster v1 (Experimental)",
        "description": "High-diversity exploratory profile with wider randomization.",
        "aggressive_ceiling_stack": False,
        "overlap_cap": 5,
        "core_play_inject_prob": 0.10,
        "core_play_min_own_pct": 14.0,
        "core_play_max_own_pct": 60.0,
        "core_play_min_projection": 23.0,
        "low_own_inject_prob": 0.25,
        "low_own_max_own_pct": 12.0,
        "low_own_min_projection": 20.0,
        "standout_lock_prob": 0.20,
        "standout_min_ceiling_gap": 7.5,
        "strategy_mix": [
            ("ceiling", 0.34, 0.30),
            ("leverage", 0.48, 0.20),
            ("value", 0.42, 0.15),
            ("projection", 0.70, 0.35),
        ],
    },
    "standout_v1_capture": {
        "label": "Standout v1 (Missed-Capture)",
        "description": "Ceiling-first profile for capturing overlooked breakout outcomes.",
        "aggressive_ceiling_stack": True,
        "overlap_cap": 4,
        "core_play_inject_prob": 0.06,
        "core_play_min_own_pct": 15.0,
        "core_play_max_own_pct": 60.0,
        "core_play_min_projection": 24.0,
        "low_own_inject_prob": 0.45,
        "low_own_max_own_pct": 15.0,
        "low_own_min_projection": 18.0,
        "standout_lock_prob": 0.55,
        "standout_min_ceiling_gap": 9.0,
        "strategy_mix": [
            ("ceiling", 0.18, 0.60),
            ("leverage", 0.26, 0.20),
            ("projection", 0.42, 0.10),
            ("projection", 0.78, 0.10),
        ],
    },
    "midrange_v1_minutes_vegas": {
        "label": "Midrange v1 (Minutes+Vegas Test)",
        "description": (
            "Targets $5k-$6.5k players when recent minutes are elevated or Vegas "
            "implied output beats the peer average."
        ),
        "aggressive_ceiling_stack": False,
        "overlap_cap": 5,
        "core_play_inject_prob": 0.10,
        "core_play_min_own_pct": 10.0,
        "core_play_max_own_pct": 45.0,
        "core_play_min_projection": 22.0,
        "low_own_inject_prob": 0.10,
        "low_own_max_own_pct": 12.0,
        "low_own_min_projection": 20.0,
        "standout_lock_prob": 0.08,
        "standout_min_ceiling_gap": 7.5,
        "midrange_focus_enabled": True,
        "midrange_focus_min_salary": 5000,
        "midrange_focus_max_salary": 6500,
        "midrange_signal_inject_prob": 0.55,
        "midrange_base_bonus": 0.12,
        "midrange_minutes_bonus": 0.16,
        "midrange_vegas_bonus": 0.16,
        "midrange_combo_bonus": 0.10,
        "midrange_minutes_floor": 26.0,
        "strategy_mix": [
            ("value", 0.22, 0.34),
            ("projection", 0.28, 0.28),
            ("ceiling", 0.24, 0.14),
            ("leverage", 0.30, 0.12),
            ("projection", 0.58, 0.12),
        ],
    },
}


def get_lineup_model_profiles() -> Dict[str, Dict[str, Any]]:
    """Return lineup model profile metadata for UI and engine wiring."""
    return LINEUP_MODEL_PROFILES


def _canonical_primary_position(positions: List[str], fallback: str = "") -> str:
    """Resolve a player's primary DFS position into canonical buckets."""
    valid = {"PG", "SG", "SF", "PF", "C"}

    # Generic DraftKings labels mapped to a deterministic primary bucket.
    alias_to_primary = {
        "G": "PG",
        "GUARD": "PG",
        "F": "PF",
        "FORWARD": "PF",
        "UTIL": "C",
        "U": "C",
    }

    def _expand_tokens(raw_value: object) -> List[str]:
        raw = str(raw_value or "").strip().upper().replace("-", "/")
        if not raw:
            return []
        parts = [p.strip() for p in raw.replace(",", "/").split("/") if p.strip()]
        return parts if parts else [raw]

    for pos in positions or []:
        for token in _expand_tokens(pos):
            if token in valid:
                return token
            mapped = alias_to_primary.get(token)
            if mapped:
                return mapped

    for token in _expand_tokens(fallback):
        if token in valid:
            return token
        mapped = alias_to_primary.get(token)
        if mapped:
            return mapped

    return "UNK"


def _salary_bucket_from_salary(salary: int) -> str:
    """Map salary to calibration bucket labels used by DFS diagnostics."""
    if salary >= 9500:
        return "STUD"
    if salary >= 8000:
        return "UPPER"
    if salary >= 6500:
        return "MID"
    if salary >= 5000:
        return "LOW_MID"
    return "VALUE"


def _is_overproj_redflag_segment(position: str, bucket: str) -> bool:
    """Flag robustly over-projected segments for lineup-level composition control."""
    metrics = SEGMENT_CALIBRATION_METRICS.get((position, bucket))
    if not metrics:
        return False
    return metrics.get("avg_error", 0.0) <= -4.0 and metrics.get("samples", 0) >= 20


def _apply_segment_calibration_and_risk(player: DFSPlayer) -> None:
    """Apply shrunk position/salary bias correction and MAE-driven risk controls."""
    player.primary_position = _canonical_primary_position(player.positions, player.position)
    player.salary_bucket = _salary_bucket_from_salary(int(player.salary or 0))

    metrics = SEGMENT_CALIBRATION_METRICS.get((player.primary_position, player.salary_bucket))
    if not metrics:
        player.segment_bias_correction = 0.0
        player.segment_mae = 0.0
        player.segment_sample_size = 0
        player.segment_risk_penalty = 0.0
        player.segment_exposure_cap = 1.0
        player.is_overproj_redflag = False
        player.risk_adjusted_proj_fpts = max(0.0, float(player.proj_fpts or 0.0))
        return

    avg_error = float(metrics.get("avg_error", 0.0) or 0.0)
    mae = float(metrics.get("mae", 0.0) or 0.0)
    samples = int(metrics.get("samples", 0) or 0)

    # Shrink toward zero when sample size is small.
    shrink = samples / (samples + SEGMENT_SHRINKAGE_K) if samples > 0 else 0.0
    correction = avg_error * shrink
    correction = max(-SEGMENT_CORRECTION_CAP, min(SEGMENT_CORRECTION_CAP, correction))
    player.proj_fpts = max(0.0, float(player.proj_fpts or 0.0) + correction)

    # MAE penalty controls lineup scoring/exposure for noisy segments.
    excess_mae = max(0.0, mae - SEGMENT_MAE_BASELINE)
    risk_penalty = min(SEGMENT_MAE_PENALTY_CAP, excess_mae * SEGMENT_MAE_PENALTY_WEIGHT)
    player.risk_adjusted_proj_fpts = max(0.0, float(player.proj_fpts or 0.0) - risk_penalty)

    player.segment_bias_correction = correction
    player.segment_mae = mae
    player.segment_sample_size = samples
    player.segment_risk_penalty = risk_penalty
    player.segment_exposure_cap = SEGMENT_EXPOSURE_CAPS.get(
        (player.primary_position, player.salary_bucket), 1.0
    )
    player.is_overproj_redflag = _is_overproj_redflag_segment(
        player.primary_position, player.salary_bucket
    )


def _build_midrange_minutes_vegas_context(
    player_pool: List[DFSPlayer], profile_cfg: Dict[str, Any]
) -> Dict[str, float]:
    """Compute peer averages for the midrange minutes/Vegas test profile."""
    if not bool(profile_cfg.get("midrange_focus_enabled", False)):
        return {}

    min_salary = int(profile_cfg.get("midrange_focus_min_salary", 5000) or 5000)
    max_salary = int(profile_cfg.get("midrange_focus_max_salary", 6500) or 6500)
    focus_players = [
        p
        for p in player_pool
        if min_salary <= int(getattr(p, "salary", 0) or 0) <= max_salary
        and not bool(getattr(p, "is_excluded", False))
        and not bool(getattr(p, "is_injured", False))
    ]

    recent_minutes = [
        float(getattr(p, "recent_minutes_avg", 0.0) or 0.0)
        for p in focus_players
        if float(getattr(p, "recent_minutes_avg", 0.0) or 0.0) > 0
    ]
    vegas_implied = [
        float(getattr(p, "vegas_implied_fpts", 0.0) or 0.0)
        for p in focus_players
        if float(getattr(p, "vegas_implied_fpts", 0.0) or 0.0) > 0
    ]
    vegas_edges = [
        float(getattr(p, "vegas_edge_pct", 0.0) or 0.0)
        for p in focus_players
        if str(getattr(p, "vegas_signal", "") or "").strip() or
        abs(float(getattr(p, "vegas_edge_pct", 0.0) or 0.0)) > 0
    ]

    recent_minutes_avg = float(np.mean(recent_minutes)) if recent_minutes else 0.0
    vegas_implied_avg = float(np.mean(vegas_implied)) if vegas_implied else 0.0
    vegas_edge_avg = float(np.mean(vegas_edges)) if vegas_edges else 0.0

    return {
        "focus_player_count": float(len(focus_players)),
        "focus_min_salary": float(min_salary),
        "focus_max_salary": float(max_salary),
        "recent_minutes_avg": recent_minutes_avg,
        "recent_minutes_threshold": max(
            float(profile_cfg.get("midrange_minutes_floor", 26.0) or 26.0),
            recent_minutes_avg,
        ),
        "vegas_implied_avg": vegas_implied_avg,
        "vegas_edge_avg": vegas_edge_avg,
    }


def _midrange_minutes_vegas_flags(
    player: DFSPlayer, profile_cfg: Dict[str, Any], model_context: Optional[Dict[str, float]]
) -> Dict[str, bool]:
    """Evaluate whether a player qualifies for the midrange minutes/Vegas bonuses."""
    profile_cfg = profile_cfg or {}
    model_context = model_context or {}

    min_salary = int(
        model_context.get(
            "focus_min_salary",
            float(profile_cfg.get("midrange_focus_min_salary", 5000) or 5000),
        )
    )
    max_salary = int(
        model_context.get(
            "focus_max_salary",
            float(profile_cfg.get("midrange_focus_max_salary", 6500) or 6500),
        )
    )
    salary = int(getattr(player, "salary", 0) or 0)
    in_range = min_salary <= salary <= max_salary

    recent_minutes = float(getattr(player, "recent_minutes_avg", 0.0) or 0.0)
    recent_minutes_threshold = float(
        model_context.get(
            "recent_minutes_threshold",
            profile_cfg.get("midrange_minutes_floor", 26.0) or 26.0,
        )
    )
    has_minutes_bonus = in_range and recent_minutes > 0 and recent_minutes >= recent_minutes_threshold

    vegas_implied = float(getattr(player, "vegas_implied_fpts", 0.0) or 0.0)
    vegas_edge = float(getattr(player, "vegas_edge_pct", 0.0) or 0.0)
    vegas_signal = str(getattr(player, "vegas_signal", "") or "").strip().upper()
    vegas_implied_avg = float(model_context.get("vegas_implied_avg", 0.0) or 0.0)
    vegas_edge_threshold = max(0.0, float(model_context.get("vegas_edge_avg", 0.0) or 0.0))
    has_vegas_bonus = in_range and (
        vegas_signal == "BOOST"
        or (vegas_implied > 0 and vegas_implied_avg > 0 and vegas_implied >= vegas_implied_avg)
        or (vegas_edge > 0 and vegas_edge >= vegas_edge_threshold)
    )

    return {
        "in_range": in_range,
        "has_minutes_bonus": has_minutes_bonus,
        "has_vegas_bonus": has_vegas_bonus,
    }


def _midrange_minutes_vegas_multiplier(
    player: DFSPlayer, profile_cfg: Dict[str, Any], model_context: Optional[Dict[str, float]]
) -> float:
    """Score multiplier for the $5k-$6.5k minutes/Vegas test profile."""
    if not bool((profile_cfg or {}).get("midrange_focus_enabled", False)):
        return 1.0

    flags = _midrange_minutes_vegas_flags(player, profile_cfg, model_context)
    if not flags["in_range"]:
        return 1.0

    bonus = float(profile_cfg.get("midrange_base_bonus", 0.0) or 0.0)
    if flags["has_minutes_bonus"]:
        bonus += float(profile_cfg.get("midrange_minutes_bonus", 0.0) or 0.0)
    if flags["has_vegas_bonus"]:
        bonus += float(profile_cfg.get("midrange_vegas_bonus", 0.0) or 0.0)
    if flags["has_minutes_bonus"] and flags["has_vegas_bonus"]:
        bonus += float(profile_cfg.get("midrange_combo_bonus", 0.0) or 0.0)

    return max(1.0, 1.0 + bonus)


def _midrange_minutes_vegas_signal_score(
    player: DFSPlayer, profile_cfg: Dict[str, Any], model_context: Optional[Dict[str, float]]
) -> float:
    """Weighted ranking score for selecting midrange signal candidates."""
    flags = _midrange_minutes_vegas_flags(player, profile_cfg, model_context)
    if not flags["in_range"]:
        return 0.0

    model_context = model_context or {}
    base_proj = float(
        getattr(player, "risk_adjusted_proj_fpts", 0.0)
        or getattr(player, "proj_fpts", 0.0)
        or 0.0
    )
    minutes_threshold = float(model_context.get("recent_minutes_threshold", 0.0) or 0.0)
    minutes_excess = max(0.0, float(getattr(player, "recent_minutes_avg", 0.0) or 0.0) - minutes_threshold)
    vegas_edge = max(0.0, float(getattr(player, "vegas_edge_pct", 0.0) or 0.0))
    multiplier = _midrange_minutes_vegas_multiplier(player, profile_cfg, model_context)

    return max(
        0.0,
        (base_proj * multiplier)
        + (minutes_excess * 0.60)
        + (vegas_edge * 0.12),
    )


# =============================================================================
# Lineup Data Class
# =============================================================================

@dataclass
class DFSLineup:
    """Represents a complete 8-player DFS lineup."""
    players: Dict[str, DFSPlayer] = field(default_factory=dict)  # slot -> player
    model_key: str = ""
    model_label: str = ""
    generation_strategy: str = ""

    @property
    def total_salary(self) -> int:
        return sum(p.salary for p in self.players.values())

    @property
    def remaining_salary(self) -> int:
        return SALARY_CAP - self.total_salary

    @property
    def total_proj_fpts(self) -> float:
        return sum(p.proj_fpts for p in self.players.values())

    @property
    def total_ceiling(self) -> float:
        return sum(p.proj_ceiling for p in self.players.values())

    @property
    def total_floor(self) -> float:
        return sum(p.proj_floor for p in self.players.values())

    @property
    def unique_games(self) -> Set[str]:
        return {p.game_id for p in self.players.values()}

    @property
    def is_valid(self) -> bool:
        """Check if lineup meets all DraftKings constraints."""
        # Check all slots filled
        if len(self.players) != 8:
            return False
        # Check salary cap
        if self.total_salary > SALARY_CAP:
            return False
        # Check minimum 2 games
        if len(self.unique_games) < MIN_GAMES:
            return False
        # Check no duplicate players
        player_ids = [p.player_id for p in self.players.values()]
        if len(player_ids) != len(set(player_ids)):
            return False
        return True

    @property
    def game_stacks(self) -> Dict[str, List[str]]:
        """Get players grouped by game."""
        stacks = {}
        for slot, player in self.players.items():
            if player.game_id not in stacks:
                stacks[player.game_id] = []
            stacks[player.game_id].append(player.name)
        return stacks

    def to_dk_export_row(self) -> Dict[str, str]:
        """Convert to DraftKings CSV export format."""
        row = {}
        for slot in ROSTER_SLOTS:
            player = self.players.get(slot)
            if player:
                row[slot] = player.dk_id if player.dk_id else str(player.player_id)
        return row

    def get_player_list(self) -> List[DFSPlayer]:
        """Return list of players in roster order."""
        return [self.players.get(slot) for slot in ROSTER_SLOTS if slot in self.players]


# =============================================================================
# Projection Generator
# =============================================================================

def load_player_historical_stats(
    conn: sqlite3.Connection,
    player_id: int,
    season: str = "2025-26",
    season_type: str = "Regular Season"
) -> pd.DataFrame:
    """Load historical game logs for a player.

    Handles databases with or without the extended DFS columns.
    Falls back to basic columns if extended stats aren't available.
    """
    # First, check which columns exist in the database
    cursor = conn.execute("PRAGMA table_info(player_game_logs)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Define required columns and optional DFS columns
    base_columns = ['game_date', 'matchup', 'points', 'fg3m', 'minutes']
    dfs_columns = ['rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fgm', 'fga', 'ftm', 'fta']

    # Build column list based on what exists
    select_columns = [c for c in base_columns if c in existing_columns]
    select_columns += [c for c in dfs_columns if c in existing_columns]

    if not select_columns:
        return pd.DataFrame()

    query = f"""
        SELECT {', '.join(select_columns)}
        FROM player_game_logs
        WHERE player_id = ?
          AND season = ?
          AND season_type = ?
        ORDER BY game_date DESC
    """

    df = pd.read_sql_query(query, conn, params=[player_id, season, season_type])

    # Add missing DFS columns with zeros for compatibility
    for col in dfs_columns:
        if col not in df.columns:
            df[col] = 0.0

    return df


def _calculate_fpts_ppg_ratio(logs: pd.DataFrame) -> float:
    """
    Calculate historical ratio of DK Fantasy Points to raw PPG.

    .. deprecated::
        No longer used by the sophisticated projection model, which now
        projects each stat independently via ``_project_stat_multi_factor()``
        and computes FPTS directly with ``calculate_dk_fantasy_points()``.
        Retained for the ``_generate_simple_projections()`` fallback path.

    This accounts for DFS scoring beyond just points:
    - Points (+1.0 per point)
    - 3PM bonus (+0.5)
    - Rebounds (+1.25)
    - Assists (+1.5)
    - Steals/Blocks (+2.0)
    - Turnovers (-0.5)
    - DD/TD bonuses

    Typical ratios by position:
    - Guards: ~1.8-2.0 FPTS per PPG (more assists)
    - Forwards: ~1.6-1.8 FPTS per PPG
    - Centers: ~1.5-1.7 FPTS per PPG (more rebounds)

    Args:
        logs: DataFrame with historical game data including dk_fpts and points

    Returns:
        Average FPTS/PPG ratio (default 1.7 if insufficient data)
    """
    if logs.empty:
        return 1.7  # Default ratio

    # Ensure dk_fpts is calculated
    if 'dk_fpts' not in logs.columns:
        logs = logs.copy()
        logs['dk_fpts'] = logs.apply(calculate_dk_fpts_from_row, axis=1)

    # Calculate ratio for games where points > 0
    valid = logs[logs['points'] > 0]
    if valid.empty or len(valid) < 3:
        return 1.7

    ratios = valid['dk_fpts'] / valid['points']
    return ratios.mean()


def _calc_minutes_deviation(logs: pd.DataFrame) -> float:
    """
    Calculate minutes deviation ratio for uncertainty estimation.

    Measures how much recent minutes deviate from season average.
    Higher values indicate inconsistent playing time (role uncertainty).

    Args:
        logs: DataFrame with 'minutes' column, ordered by date DESC

    Returns:
        |recent_min - season_min| / season_min (0.0 if unavailable)
    """
    if logs.empty or 'minutes' not in logs.columns:
        return 0.0

    valid_minutes = logs[logs['minutes'] > 0]['minutes']
    if len(valid_minutes) < 5:
        return 0.0

    season_avg = valid_minutes.mean()
    recent_avg = valid_minutes.head(5).mean()

    if season_avg <= 0:
        return 0.0

    return abs(recent_avg - season_avg) / season_avg


def _get_player_position(conn: sqlite3.Connection, player_id: int) -> str:
    """Get player position from database."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT position FROM players WHERE player_id = ?",
            (player_id,)
        )
        result = cursor.fetchone()
        if result and result[0]:
            return result[0]
    except Exception:
        pass
    return ""


def _get_matchup_history(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_team: str,
    season: str = "2025-26"
) -> Tuple[Dict[str, Optional[float]], int]:
    """
    Get player's historical performance vs opponent across all stats.

    Args:
        conn: Database connection
        player_id: Player ID
        opponent_team: Opponent team abbreviation
        season: Season string

    Returns:
        (dict of stat averages vs opponent, games_played)
        e.g. ({"points": 28.5, "rebounds": 7.2, ...}, 4)
    """
    ALL_STATS = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fg3m']
    empty_result = {stat: None for stat in ALL_STATS}

    try:
        query = f"""
            SELECT {', '.join(ALL_STATS)}, matchup
            FROM player_game_logs
            WHERE player_id = ?
              AND season = ?
              AND matchup LIKE ?
              AND points IS NOT NULL
        """
        # Match both home and away games vs opponent
        df = pd.read_sql_query(
            query, conn,
            params=[player_id, season, f"%{opponent_team}%"]
        )

        if df.empty:
            return empty_result, 0

        result = {stat: df[stat].mean() if stat in df.columns else None for stat in ALL_STATS}
        return result, len(df)
    except Exception:
        return empty_result, 0


# Mapping of stats that receive opponent-allowed adjustments.
# Steals, blocks, and turnovers are excluded because team-level
# opponent-allowed stats for those are too noisy to be useful.
STAT_OPP_ADJUSTMENT_MAP = {
    'rebounds': {'defense_key': 'avg_allowed_reb', 'league_avg': LEAGUE_AVG_TEAM_REB},
    'assists':  {'defense_key': 'avg_allowed_ast', 'league_avg': LEAGUE_AVG_TEAM_AST},
    'fg3m':     {'defense_key': 'avg_allowed_fg3m', 'league_avg': LEAGUE_AVG_TEAM_FG3M},
}


def _estimate_ownership(player, has_injury_boost: bool = False) -> float:
    """Estimate ownership % using nonlinear model calibrated to actual DFS data.

    Uses a nonlinear signal that combines value, DK card visibility, and
    raw projection strength to separate chalk (30%+) from contrarian (<3%).

    Key drivers:
    1. Value (FPTS/$) -- optimizers push toward high-value plays
    2. DK AvgPointsPerGame -- visible on DK player card, influences casual players
    3. Projection signal (projected FPTS) -- captures emerging chalk before DK averages catch up
    4. Salary tier -- determines base ownership and scaling ceiling
    5. Injury boost -- players absorbing injured teammates' usage draw higher ownership
    """
    if player.salary <= 0 or player.proj_fpts <= 0:
        return 0.5

    # --- Factor 1: Value score (0-10 range) ---
    value_score = min(10.0, player.fpts_per_dollar)

    # --- Factor 2: DK visibility (0-10 range) ---
    dk_avg = getattr(player, 'dk_avg_pts', 0.0) or 0.0
    dk_visibility = min(10.0, dk_avg / 7.0) if dk_avg > 0 else value_score * 0.7

    # --- Factor 3: Projection signal (0-10 range) ---
    projection_signal = min(10.0, float(getattr(player, 'proj_fpts', 0.0) or 0.0) / 5.0)

    # --- Factor 4: Salary-tier parameters ---
    # Stars: high base and visibility/projection driven
    # Mid/Value: balanced value + projection
    # Punt: low base but can jump on elite value + projection
    if player.salary >= 9000:
        tier_base = 8.0
        tier_ceiling = 55.0
        vis_weight = 0.50
        proj_weight = 0.30
    elif player.salary >= 7000:
        tier_base = 3.0
        tier_ceiling = 30.0
        vis_weight = 0.42
        proj_weight = 0.28
    elif player.salary >= 5000:
        tier_base = 1.5
        tier_ceiling = 32.0
        vis_weight = 0.28
        proj_weight = 0.32
    else:
        tier_base = 0.5
        tier_ceiling = 35.0
        vis_weight = 0.20
        proj_weight = 0.30

    # --- Combine factors with tier-specific weighting ---
    val_weight = max(0.0, 1.0 - vis_weight - proj_weight)
    raw_signal = (
        value_score * val_weight
        + dk_visibility * vis_weight
        + projection_signal * proj_weight
    )

    # --- Power-law sharpening (exponent 2.0) ---
    # High signals stay high, low signals collapse toward zero
    # This creates the chalk vs contrarian separation missing from the old linear model
    normalized = raw_signal / 10.0
    sharpened = normalized ** 2.0

    # --- Scale to ownership range ---
    ownership = tier_base + sharpened * (tier_ceiling - tier_base)

    # --- Injury boost: players absorbing injured teammate usage ---
    if has_injury_boost:
        ownership *= 2.25

    # --- Clamp to realistic range ---
    ownership = min(55.0, max(0.5, ownership))

    return round(ownership, 1)


def _project_player_ownership(
    conn: sqlite3.Connection,
    player: DFSPlayer,
    today_num_games: Optional[int] = None,
    has_injury_boost: bool = False,
) -> float:
    """Project ownership using blended historical + model estimate.

    Historical ownership is strong when available, but can be stale on
    rapidly changing slates. Blend it with the current-model estimate so
    projected ownership adapts to today's projection/value context.
    """
    model_own = _estimate_ownership(player, has_injury_boost=has_injury_boost)
    hist_own = _get_historical_ownership(conn, player.player_id, today_num_games=today_num_games)

    if hist_own is None:
        return model_own

    ceiling_gap = max(
        0.0,
        float(getattr(player, 'proj_ceiling', 0.0) or 0.0)
        - float(getattr(player, 'proj_fpts', 0.0) or 0.0),
    )
    value_score = float(getattr(player, 'fpts_per_dollar', 0.0) or 0.0)

    # Base: favor history but keep room for same-day model information.
    hist_weight = 0.65

    # Emerging chalk tends to show up first in value/projection signals.
    if value_score >= 6.0 or ceiling_gap >= 9.0:
        hist_weight = 0.55

    # If model and history diverge materially, reduce inertia.
    if abs(model_own - hist_own) >= 12.0:
        hist_weight = 0.50

    blended = hist_weight * hist_own + (1.0 - hist_weight) * model_own

    # Guardrail: avoid suppressing clear chalk too aggressively from stale history.
    if model_own >= 28.0:
        blended = max(blended, model_own * 0.70)

    return round(min(55.0, max(0.5, blended)), 1)


def _get_historical_ownership(conn: sqlite3.Connection, player_id: int,
                              today_num_games: int = None, lookback_days: int = 21) -> Optional[float]:
    """Get average ownership from past slates for a player, normalized by slate size.

    When today_num_games is provided, each historical ownership value is adjusted
    for the difference in slate size using: adjusted = own * (slate_games / today_games) ^ 0.4
    This accounts for ownership concentrating on smaller slates and spreading on larger ones.

    Prefers actual_ownership (from parsed contest results) when available.
    Falls back to ownership_proj from prior projections.
    Returns None if no historical data exists (will fall back to model estimate).

    Note: SQLite date('now') uses UTC, which may be ~1 day ahead of Eastern.
    This slightly widens the lookback window, which is acceptable.
    """
    try:
        # First try actual ownership from contest results (most accurate)
        rows = conn.execute("""
            WITH slate_sizes AS (
                SELECT slate_date, COUNT(DISTINCT team) / 2.0 as slate_games
                FROM dfs_slate_projections
                WHERE slate_date >= date('now', ? || ' days')
                GROUP BY slate_date
            )
            SELECT p.actual_ownership, s.slate_games
            FROM dfs_slate_projections p
            JOIN slate_sizes s ON s.slate_date = p.slate_date
            WHERE p.player_id = ?
              AND p.slate_date >= date('now', ? || ' days')
              AND p.actual_ownership IS NOT NULL
              AND p.actual_ownership > 0
        """, (f"-{lookback_days}", player_id, f"-{lookback_days}")).fetchall()

        if rows:
            if today_num_games and today_num_games > 0:
                adjusted = [
                    own * (sg / today_num_games) ** 0.4 if sg and sg > 0 else own
                    for own, sg in rows
                ]
                if adjusted:
                    return round(sum(adjusted) / len(adjusted), 1)
            else:
                return round(sum(own for own, _ in rows) / len(rows), 1)

        # Fall back to projected ownership from prior slates (less accurate but better than model)
        rows = conn.execute("""
            WITH slate_sizes AS (
                SELECT slate_date, COUNT(DISTINCT team) / 2.0 as slate_games
                FROM dfs_slate_projections
                WHERE slate_date >= date('now', ? || ' days')
                GROUP BY slate_date
            )
            SELECT p.ownership_proj, s.slate_games
            FROM dfs_slate_projections p
            JOIN slate_sizes s ON s.slate_date = p.slate_date
            WHERE p.player_id = ?
              AND p.slate_date >= date('now', ? || ' days')
              AND p.ownership_proj IS NOT NULL
              AND p.ownership_proj > 0
        """, (f"-{lookback_days}", player_id, f"-{lookback_days}")).fetchall()

        if rows:
            if today_num_games and today_num_games > 0:
                adjusted = [
                    own * (sg / today_num_games) ** 0.4 if sg and sg > 0 else own
                    for own, sg in rows
                ]
                if adjusted:
                    return round(sum(adjusted) / len(adjusted), 1)
            else:
                return round(sum(own for own, _ in rows) / len(rows), 1)

    except Exception as e:
        import logging
        logging.warning(f"Historical ownership query failed for player {player_id}: {e}")

    return None


def _project_stat_multi_factor(
    stat_name: str,
    logs: pd.DataFrame,
    vs_opp_stats: Dict[str, Optional[float]],
    vs_opp_games: int,
    opp_defense: Optional[Dict] = None
) -> float:
    """
    Project a single stat using multi-factor weighting (mirrors PPG model).

    Weighting:
    - 25% season average
    - 40% recent form (0.6 × L5 + 0.4 × L3)
    - Up to 35% matchup history (scaled by confidence = min(1, games/5))

    Then applies opponent-allowed adjustment for rebounds, assists, and fg3m:
    - Compares opponent's avg allowed stat to league average
    - Dampened 50% to avoid overfitting team-level data to individual players
    - Clamped to [0.85, 1.15] to prevent extreme swings

    Args:
        stat_name: Column name in logs (e.g. 'rebounds', 'assists')
        logs: Player game log DataFrame, ordered by date DESC
        vs_opp_stats: Dict of stat averages vs this opponent
        vs_opp_games: Number of games vs this opponent
        opp_defense: Opponent defense dict from defense_map (optional)

    Returns:
        Projected stat value (before rest/pace adjustments)
    """
    if logs.empty or stat_name not in logs.columns:
        return 0.0

    # --- Base components (same structure as PPG model) ---
    season_avg = logs[stat_name].mean()
    if pd.isna(season_avg):
        return 0.0

    recent_5 = logs.head(5)
    recent_3 = logs.head(3)
    recent_avg_5 = recent_5[stat_name].mean() if len(recent_5) > 0 else season_avg
    recent_avg_3 = recent_3[stat_name].mean() if len(recent_3) > 0 else recent_avg_5

    # Build weighted components
    components = {}
    weights = {}

    # Season baseline (25%)
    components["season"] = season_avg
    weights["season"] = 0.25

    # Recent form (40%) — blend of L5 and L3
    if pd.notna(recent_avg_5):
        components["recent"] = (recent_avg_5 * 0.6) + (recent_avg_3 * 0.4)
        weights["recent"] = 0.40
    else:
        components["recent"] = season_avg
        weights["recent"] = 0.20

    # Matchup history (up to 35%)
    vs_opp_val = vs_opp_stats.get(stat_name)
    if vs_opp_val is not None and pd.notna(vs_opp_val) and vs_opp_games >= 2:
        confidence_factor = min(1.0, vs_opp_games / 5.0)
        components["vs_team"] = vs_opp_val
        weights["vs_team"] = 0.35 * confidence_factor

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    # Compute weighted projection
    projection = sum(
        components.get(k, 0) * weights.get(k, 0)
        for k in components
        if components.get(k) is not None and pd.notna(components.get(k))
    )

    # --- Opponent-allowed adjustment (reb, ast, fg3m only) ---
    if opp_defense and stat_name in STAT_OPP_ADJUSTMENT_MAP:
        adj_info = STAT_OPP_ADJUSTMENT_MAP[stat_name]
        opp_allowed = opp_defense.get(adj_info['defense_key'])
        league_avg = adj_info['league_avg']

        if opp_allowed is not None and pd.notna(opp_allowed) and opp_allowed > 0 and league_avg > 0:
            raw_ratio = opp_allowed / league_avg
            # Dampen by 50%: move only halfway from 1.0 toward the raw ratio
            dampened_ratio = 1.0 + (raw_ratio - 1.0) * 0.5
            # Clamp to prevent extreme adjustments
            dampened_ratio = max(0.85, min(1.15, dampened_ratio))
            projection *= dampened_ratio

    return max(0.0, projection)


def _generate_simple_projections(
    conn: sqlite3.Connection,
    player: DFSPlayer,
    logs: pd.DataFrame,
    season: str = "2025-26",
    num_games: int = None
) -> DFSPlayer:
    """
    Simple projection model (fallback when sophisticated modules unavailable).

    Weighting:
    - 40% Season average
    - 40% Recent form (last 5 games)
    - 20% Matchup history (vs opponent, if available)
    """
    if logs.empty:
        return player

    # Calculate DK fantasy points for each game
    logs['dk_fpts'] = logs.apply(calculate_dk_fpts_from_row, axis=1)

    # Season averages
    season_avg = {
        'points': logs['points'].mean(),
        'rebounds': logs['rebounds'].mean(),
        'assists': logs['assists'].mean(),
        'steals': logs['steals'].mean(),
        'blocks': logs['blocks'].mean(),
        'turnovers': logs['turnovers'].mean(),
        'fg3m': logs['fg3m'].mean(),
        'dk_fpts': logs['dk_fpts'].mean()
    }

    # Recent form (last 5 games)
    recent = logs.head(5)
    recent_avg = {
        'points': recent['points'].mean(),
        'rebounds': recent['rebounds'].mean(),
        'assists': recent['assists'].mean(),
        'steals': recent['steals'].mean(),
        'blocks': recent['blocks'].mean(),
        'turnovers': recent['turnovers'].mean(),
        'fg3m': recent['fg3m'].mean(),
        'dk_fpts': recent['dk_fpts'].mean()
    }

    # Matchup history (vs opponent)
    opponent_abbrev = player.opponent
    matchup_logs = logs[logs['matchup'].str.contains(opponent_abbrev, na=False)]

    if len(matchup_logs) >= 2:
        matchup_avg = {
            'points': matchup_logs['points'].mean(),
            'rebounds': matchup_logs['rebounds'].mean(),
            'assists': matchup_logs['assists'].mean(),
            'steals': matchup_logs['steals'].mean(),
            'blocks': matchup_logs['blocks'].mean(),
            'turnovers': matchup_logs['turnovers'].mean(),
            'fg3m': matchup_logs['fg3m'].mean(),
            'dk_fpts': matchup_logs['dk_fpts'].mean()
        }
        weights = (0.4, 0.4, 0.2)
        sources = (season_avg, recent_avg, matchup_avg)
    else:
        weights = (0.5, 0.5)
        sources = (season_avg, recent_avg)

    # Calculate weighted projections
    for stat in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fg3m']:
        proj_value = sum(
            w * src.get(stat, 0)
            for w, src in zip(weights, sources)
            if pd.notna(src.get(stat))
        )
        setattr(player, f'proj_{stat}', proj_value if pd.notna(proj_value) else 0)

    # Calculate floor and ceiling from historical distribution
    if len(logs) >= 3:
        player.proj_floor = logs['dk_fpts'].quantile(0.10)
        player.proj_ceiling = logs['dk_fpts'].quantile(0.90)
    else:
        avg_fpts = season_avg.get('dk_fpts', 0)
        player.proj_floor = avg_fpts * 0.6
        player.proj_ceiling = avg_fpts * 1.5

    player.calculate_projections()
    _apply_segment_calibration_and_risk(player)
    if player.salary > 0:
        player.fpts_per_dollar = (player.proj_fpts / player.salary) * 1000

    if player.salary > 0:
        player.ownership_proj = _project_player_ownership(
            conn,
            player,
            today_num_games=num_games,
            has_injury_boost=False,
        )
        if player.ownership_proj > 0:
            upside = max(0, player.proj_ceiling - player.proj_fpts)
            player.leverage_score = (player.proj_fpts + upside * 1.5) / max(1, player.ownership_proj)

    player.position = _get_player_position(conn, player.player_id)

    return player


def generate_player_projections(
    conn: sqlite3.Connection,
    player: DFSPlayer,
    season: str = "2025-26",
    season_type: str = "Regular Season",
    game_date: str = None,
    # Sophisticated prediction dependencies (optional)
    defense_map: Dict = None,
    def_style_map: Dict = None,
    player_vs_team_history: Dict = None,
    def_ppm_df: pd.DataFrame = None,
    league_avg_ppm: float = 0.462,
    injury_status_map: Dict = None,
    teammate_injury_map: Dict = None,
    num_games: int = None,
) -> DFSPlayer:
    """
    Generate projections for a player using sophisticated prediction model.

    When sophisticated modules are available, this function integrates:
    - Multi-factor weighted PPG projection (season avg, recent form, matchup)
    - Rest day multipliers (B2B -8%, 3+ days +5%)
    - Uncertainty multipliers for GPP ceiling/floor
    - Position-specific defensive PPM adjustments

    Falls back to simple weighted average if modules unavailable.

    Args:
        conn: Database connection
        player: DFSPlayer to generate projections for
        season: NBA season string (e.g., "2025-26")
        season_type: "Regular Season" or "Playoffs"
        game_date: Game date string (YYYY-MM-DD) for rest day calculation
        defense_map: Dict[team_id -> defense stats] (optional)
        def_style_map: Dict[team_id -> defense style] (optional)
        player_vs_team_history: Dict[player_id -> Dict[team_id -> stats]] (optional)
        def_ppm_df: DataFrame with defensive PPM stats (optional)
        league_avg_ppm: League average PPM (default 0.462)
        injury_status_map: Dict[player_id -> injury status] (optional)
        teammate_injury_map: Dict[team_id -> list of injured player_ids] (optional)

    Returns:
        DFSPlayer with projections populated
    """
    # Handle fallback players (not in our database) - use DK average as projection
    if player.is_fallback:
        dk_avg = player.dk_avg_pts
        if dk_avg > 0:
            player.proj_fpts = dk_avg
            player.proj_floor = dk_avg * 0.5   # Wider range for uncertainty
            player.proj_ceiling = dk_avg * 1.8
            # Rough stat estimates (for display purposes only)
            player.proj_points = dk_avg * 0.45   # ~45% of FPTS from points
            player.proj_rebounds = dk_avg * 0.10
            player.proj_assists = dk_avg * 0.08
            _apply_segment_calibration_and_risk(player)
            player.fpts_per_dollar = (player.proj_fpts / player.salary * 1000) if player.salary > 0 else 0
            player.ownership_proj = _project_player_ownership(
                conn,
                player,
                today_num_games=num_games,
                has_injury_boost=False,
            )
            if player.ownership_proj > 0:
                upside = max(0, player.proj_ceiling - player.proj_fpts)
                player.leverage_score = (player.proj_fpts + upside * 1.5) / max(1, player.ownership_proj)
            player.position = _get_player_position(conn, player.player_id)
        return player

    # Load historical stats
    logs = load_player_historical_stats(conn, player.player_id, season, season_type)

    if logs.empty:
        player.minutes_validated = False
        return player

    # Calculate DK fantasy points for each game
    logs['dk_fpts'] = logs.apply(calculate_dk_fpts_from_row, axis=1)

    # --- Minutes validation ---
    # Check recent playing time to filter out DNP-risk players
    if 'minutes' in logs.columns:
        try:
            min_vals = pd.to_numeric(logs['minutes'].head(10), errors='coerce').fillna(0)
            recent_5_min = min_vals.head(5)
            player.recent_minutes_avg = round(float(recent_5_min.mean()), 1)
            valid_minutes = min_vals[min_vals > 0]
            if len(valid_minutes) > 1:
                player.minutes_variance = round(float(valid_minutes.var(ddof=0)), 4)
            else:
                player.minutes_variance = 0.0

            games_with_minutes = (min_vals > 0).sum()
            if games_with_minutes == 0:
                player.minutes_validated = False
            elif player.recent_minutes_avg < 5.0:
                player.minutes_validated = False
            # Check recency: if most recent game is older than 21 days
            if 'game_date' in logs.columns and len(logs) > 0:
                from datetime import datetime, timedelta
                try:
                    last_game = pd.to_datetime(logs['game_date'].iloc[0])
                    if (datetime.now() - last_game).days > 21:
                        player.minutes_validated = False
                except Exception:
                    pass
        except Exception:
            player.minutes_validated = False  # Mark invalid if check fails

    # --- Recent FPTS stats (for scouting table) ---
    try:
        recent_7 = logs['dk_fpts'].head(7)
        player.avg_fpts_last7 = round(float(recent_7.mean()), 1) if len(recent_7) > 0 else 0.0
        player.fpts_variance = round(float(recent_7.std()), 1) if len(recent_7) > 1 else 0.0
    except Exception:
        pass

    # --- Role tier classification ---
    try:
        avg_min = pd.to_numeric(logs['minutes'], errors='coerce').mean()
        avg_ppg = logs['points'].mean() if 'points' in logs.columns else 0
        from depth_chart import classify_role_tier
        player.role_tier = classify_role_tier(avg_min, avg_ppg, len(logs), avg_fpts=player.avg_fpts_last7)
    except Exception:
        player.role_tier = ""

    # If sophisticated prediction modules unavailable, use simple model
    if not SOPHISTICATED_PREDICTIONS or defense_map is None:
        return _generate_simple_projections(conn, player, logs, season, num_games=num_games)

    # =========================================================================
    # SOPHISTICATED PROJECTION MODEL
    # =========================================================================
    analytics_used = []

    # --- Step 1: Calculate base averages ---
    season_avg_pts = logs['points'].mean() if len(logs) > 0 else 0.0
    recent_5 = logs.head(5)
    recent_3 = logs.head(3)
    recent_avg_5 = recent_5['points'].mean() if len(recent_5) > 0 else season_avg_pts
    recent_avg_3 = recent_3['points'].mean() if len(recent_3) > 0 else recent_avg_5

    # --- Step 2: Get matchup history (all stats) ---
    vs_opp_stats, vs_opp_games = _get_matchup_history(
        conn, player.player_id, player.opponent, season
    )
    vs_opp_avg = vs_opp_stats.get('points')  # Points avg for PPG model

    # --- Step 3: Get opponent defense stats ---
    opponent_id = None
    opp_def_rating = None
    opp_pace = None
    opp_defense = None  # Full defense dict for stat-level opponent adjustments

    # Try to find opponent team_id from defense_map
    if defense_map:
        for team_id, stats in defense_map.items():
            team_abbrev = stats.get('abbreviation', '') or stats.get('team_abbreviation', '')
            if team_abbrev and team_abbrev.upper() == player.opponent.upper():
                opponent_id = team_id
                opp_def_rating = stats.get('def_rating')
                opp_pace = stats.get('avg_opp_possessions')
                opp_defense = stats  # Store full dict for per-stat adjustments
                break

    # --- Step 4: Calculate sophisticated PPG projection ---
    # Multi-factor weighted model similar to prediction_generator.py
    components = {}
    weights = {}

    # Season baseline (25%)
    components["season"] = season_avg_pts
    weights["season"] = 0.25

    # Recent form (40%)
    if recent_avg_5 is not None:
        components["recent"] = (recent_avg_5 * 0.6) + (recent_avg_3 * 0.4)
        weights["recent"] = 0.40
    else:
        components["recent"] = season_avg_pts
        weights["recent"] = 0.20

    # Team-specific matchup (up to 35%)
    if vs_opp_avg is not None and vs_opp_games >= 2:
        confidence_factor = min(1.0, vs_opp_games / 5.0)
        components["vs_team"] = vs_opp_avg
        weights["vs_team"] = 0.35 * confidence_factor
        analytics_used.append("🎯")  # Matchup history indicator

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate base PPG projection
    ppg_projection = sum(
        components.get(k, 0) * weights.get(k, 0)
        for k in components
        if components.get(k) is not None
    )

    # --- Step 5: Apply defense adjustment ---
    if opp_def_rating is not None:
        league_avg_def_rating = 112.0
        def_adjustment = 1.0 - ((opp_def_rating - league_avg_def_rating) / league_avg_def_rating) * 0.10
        ppg_projection *= def_adjustment
        if def_adjustment < 0.95 or def_adjustment > 1.05:
            analytics_used.append("🛡️")  # Defense adjustment indicator

    # --- Step 6: Apply pace adjustment ---
    pace_adjustment = 1.0  # Default; reused in Step 10 for all stats
    if opp_pace is not None:
        league_avg_pace = 99.0
        pace_adjustment = 1.0 + ((opp_pace - league_avg_pace) / league_avg_pace) * 0.05
        ppg_projection *= pace_adjustment

    # --- Step 7: Apply rest day multiplier ---
    rest_multiplier = 1.0
    days_rest = None

    if game_date:
        try:
            rest_data = calculate_rest_factor(conn, player.player_id, game_date, season)
            rest_multiplier = rest_data.get('multiplier', 1.0)
            days_rest = rest_data.get('days_rest')
            player.days_rest = days_rest
            player.rest_multiplier = rest_multiplier

            if rest_data.get('is_b2b'):
                analytics_used.append("😴")  # B2B indicator
            elif days_rest and days_rest >= 3:
                analytics_used.append("💪")  # Well-rested indicator
        except Exception:
            pass

    ppg_projection *= rest_multiplier

    # --- Step 8: Calculate uncertainty multiplier for GPP strategy ---
    # (Floor/ceiling now computed from FPTS distribution in Step 9)
    uncertainty_mult = 1.0

    try:
        minutes_deviation = _calc_minutes_deviation(logs)
        injury_status = (injury_status_map or {}).get(player.player_id)

        # Check for teammate injury (usage uncertainty)
        teammate_questionable = False
        if teammate_injury_map:
            # Get player's team_id
            cursor = conn.cursor()
            cursor.execute(
                "SELECT team_id FROM players WHERE player_id = ?",
                (player.player_id,)
            )
            result = cursor.fetchone()
            if result:
                team_injuries = teammate_injury_map.get(result[0], [])
                teammate_questionable = len(team_injuries) > 0

        uncertainty_mult = pg.calculate_uncertainty_multiplier(
            injury_status=injury_status,
            minutes_deviation_ratio=minutes_deviation,
            starter_changes_recent=0,
            traded_recently=False,
            teammate_questionable=teammate_questionable
        )

        player.uncertainty_multiplier = uncertainty_mult
    except Exception:
        pass

    # --- Step 9: Project all stats with multi-factor model ---
    player.proj_points = ppg_projection  # Already sophisticated (steps 4-7)

    # pace_adjustment reused from Step 6 (applies uniformly to all stats)
    for stat in ['rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'fg3m']:
        raw = _project_stat_multi_factor(
            stat, logs, vs_opp_stats, vs_opp_games, opp_defense
        )
        adjusted = raw * rest_multiplier * pace_adjustment
        setattr(player, f'proj_{stat}', max(0.0, adjusted))

    # --- Step 9b: Apply injury stat boosts from injured teammates ---
    if INJURY_IMPACT_AVAILABLE and teammate_injury_map:
        try:
            # Get player's team_id
            cursor = conn.cursor()
            cursor.execute(
                "SELECT team_id FROM players WHERE player_id = ?",
                (player.player_id,)
            )
            result = cursor.fetchone()
            if result:
                team_id = result[0]
                injured_ids = teammate_injury_map.get(team_id, [])
                if injured_ids:
                    # Get stat boosts from injured teammates
                    injury_boost = get_injury_stat_boosts(
                        conn,
                        player.player_id,
                        player.name,
                        injured_ids,
                        season=season,
                        season_type=season_type,
                        min_games_apart=3
                    )

                    # Apply boosts (scaled by confidence)
                    if injury_boost.confidence > 0:
                        scale = injury_boost.confidence
                        player.proj_points += injury_boost.pts_boost * scale
                        player.proj_rebounds += injury_boost.reb_boost * scale
                        player.proj_assists += injury_boost.ast_boost * scale
                        player.proj_steals += injury_boost.stl_boost * scale
                        player.proj_blocks += injury_boost.blk_boost * scale
                        player.proj_fg3m += injury_boost.fg3m_boost * scale

                        # Add indicator if significant boost
                        if injury_boost.to_fpts_boost() >= 2.0:
                            analytics_used.append("💉")  # Injury boost indicator
        except Exception:
            pass  # Don't fail projection if injury boost fails

    # Direct FPTS from individual stat projections (replaces ratio approach)
    player.proj_fpts = calculate_dk_fantasy_points(
        player.proj_points, player.proj_rebounds, player.proj_assists,
        player.proj_steals, player.proj_blocks, player.proj_turnovers,
        player.proj_fg3m
    )

    # --- Recency blend: nudge projection toward recent 7-game average ---
    # If a player's recent production diverges significantly from the model,
    # blend in the recent average at 20% weight (captures hot/cold streaks)
    if player.avg_fpts_last7 > 0 and abs(player.avg_fpts_last7 - player.proj_fpts) > 3.0:
        player.proj_fpts = round(player.proj_fpts * 0.80 + player.avg_fpts_last7 * 0.20, 1)

    # Apply position-salary segment calibration with shrinkage and MAE-based risk penalties.
    _apply_segment_calibration_and_risk(player)

    # Floor/ceiling from actual DK FPTS distribution (preserves stat correlations)
    if len(logs) >= 3:
        base_floor = logs['dk_fpts'].quantile(0.10) * rest_multiplier
        base_ceiling = logs['dk_fpts'].quantile(0.90) * rest_multiplier
    else:
        base_floor = player.proj_fpts * 0.6
        base_ceiling = player.proj_fpts * 1.5

    # --- Variance-adjusted ceiling/floor ---
    # Use actual recent variance to widen/narrow the range
    if player.fpts_variance > 0 and base_ceiling > base_floor:
        model_range = base_ceiling - base_floor
        # High variance (>12): widen ceiling by up to 10%
        # Low variance (<6): tighten ceiling by up to 5% (more predictable)
        if player.fpts_variance > 12:
            variance_boost = min(0.10, (player.fpts_variance - 12) / 40)
            base_ceiling *= (1.0 + variance_boost)
        elif player.fpts_variance < 6:
            variance_tighten = min(0.05, (6 - player.fpts_variance) / 30)
            base_ceiling *= (1.0 - variance_tighten)
            base_floor *= (1.0 + variance_tighten * 0.5)  # floor rises slightly

    # Apply uncertainty multiplier (calculated in Step 8)
    try:
        player.proj_floor, player.proj_ceiling = pg.apply_uncertainty_to_projection(
            base_floor, base_ceiling, uncertainty_mult
        )
    except Exception:
        player.proj_floor = base_floor
        player.proj_ceiling = base_ceiling

    # --- Step 10: Set analytics indicators ---
    player.analytics_used = ''.join(analytics_used)

    # --- Step 11: Calculate derived values ---
    if player.salary > 0:
        player.fpts_per_dollar = (player.proj_fpts / player.salary) * 1000

    # Ownership: blend historical ownership with slate-context model estimate.
    player.ownership_proj = _project_player_ownership(
        conn,
        player,
        today_num_games=num_games,
        has_injury_boost=("💉" in player.analytics_used),
    )

    # Leverage score: projection + amplified upside, relative to ownership
    if player.ownership_proj > 0:
        upside = max(0, player.proj_ceiling - player.proj_fpts)
        player.leverage_score = (player.proj_fpts + upside * 1.5) / max(1, player.ownership_proj)

    # Get player position for display
    player.position = _get_player_position(conn, player.player_id)

    return player


# =============================================================================
# DraftKings CSV Parser
# =============================================================================

def parse_dk_csv(
    csv_path: Path,
    conn: sqlite3.Connection
) -> Tuple[List[DFSPlayer], Dict[str, any]]:
    """
    Parse DraftKings contest CSV file.

    Expected columns: Name, Position, Salary, TeamAbbrev, Game Info, AvgPointsPerGame, ID

    Uses robust name matching to handle:
    - Nicknames (Nic vs Nicolas, Herb vs Herbert)
    - Initials with/without periods (P.J. vs PJ)
    - Suffixes (Jr., III, etc.)
    - Accented characters (Jokić vs Jokic)
    - Hyphenated names

    Returns:
        Tuple of (player list, metadata dict)
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Get injured players from database
    injured_player_ids = get_injured_player_ids(conn)
    injured_player_info = get_injured_player_names(conn)

    # Build lookup from database
    players_query = """
        SELECT DISTINCT player_id, player_name, team_abbreviation
        FROM player_game_logs
        WHERE season = '2025-26'
    """
    db_players = pd.read_sql_query(players_query, conn)

    # Create normalized name->id lookup
    name_to_id = {}
    team_to_players: Dict[str, List[str]] = {}  # team -> list of normalized names

    for _, row in db_players.iterrows():
        player_name = str(row['player_name'])
        normalized = normalize_name(player_name)
        team_abbrev = str(row['team_abbreviation']).upper() if row['team_abbreviation'] else ''

        name_to_id[normalized] = {
            'player_id': row['player_id'],
            'team': team_abbrev,
            'original_name': player_name
        }

        # Also add all variants of the DB name
        for variant in generate_name_variants(player_name):
            if variant not in name_to_id:
                name_to_id[variant] = {
                    'player_id': row['player_id'],
                    'team': team_abbrev,
                    'original_name': player_name
                }

        # Build team lookup
        if team_abbrev:
            if team_abbrev not in team_to_players:
                team_to_players[team_abbrev] = []
            team_to_players[team_abbrev].append(normalized)

    players = []
    unmatched = []
    matched_names = []  # For debugging

    for _, row in df.iterrows():
        name = str(row.get('name', '')).strip()
        if not name or name.lower() == 'nan':
            continue

        # Parse positions
        pos_str = str(row.get('position', row.get('roster_position', '')))
        positions = [p.strip() for p in pos_str.split('/') if p.strip()]

        # Parse game info to extract opponent
        game_info = str(row.get('game_info', ''))
        teams = game_info.split('@') if '@' in game_info else game_info.split(' ')
        team = str(row.get('teamabbrev', row.get('team', ''))).strip()
        opponent = ''
        for t in teams:
            t_clean = t.strip().split()[0] if t.strip() else ''
            if t_clean and t_clean != team:
                opponent = t_clean
                break

        # Create game_id from game_info (use teams as proxy)
        game_id = game_info.replace(' ', '_').replace('@', '_') if game_info else f"game_{team}"

        # Use robust name matching
        player_id = find_best_player_match(
            dk_name=name,
            dk_team=team,
            db_lookup=name_to_id,
            team_lookup=team_to_players
        )

        is_fallback = False
        if player_id is None:
            # Player not in our database - use fallback projection from DK average
            unmatched.append(name)
            # Generate a unique negative ID for unmatched players (hash of name)
            player_id = -abs(hash(name)) % 1000000
            is_fallback = True
        else:
            matched_names.append((name, name_to_id.get(normalize_name(name), {}).get('original_name', 'Unknown')))

        # Check if player is injured (from our API injury list)
        # Fallback players can't be checked against injury list (no player_id match)
        is_injured = player_id in injured_player_ids if not is_fallback else False
        injury_status = ""
        if is_injured and player_id in injured_player_info:
            # Extract status from "Name (STATUS)" format
            info = injured_player_info[player_id]
            if '(' in info and ')' in info:
                injury_status = info.split('(')[-1].rstrip(')')

        # Also detect DK-marked OUT: min salary ($3000) + 0.0 avg points
        # DraftKings zeros out AvgPointsPerGame and sets salary to $3000
        # for players they've confirmed as OUT, even if our API hasn't
        # caught the designation yet.
        dk_salary = int(row.get('salary', 0))
        dk_avg_pts = float(row.get('avgpointspergame', -1))
        if dk_salary <= 3000 and dk_avg_pts == 0.0 and not is_injured:
            is_injured = True
            injury_status = "OUT (DK)"

        player = DFSPlayer(
            player_id=player_id,
            name=name,
            team=team,
            opponent=opponent,
            game_id=game_id,
            positions=positions,
            salary=dk_salary,
            dk_id=str(row.get('id', '')),
            dk_avg_pts=dk_avg_pts if dk_avg_pts >= 0 else 0.0,
            is_injured=is_injured,
            injury_status=injury_status.upper() if injury_status else "",
            is_fallback=is_fallback
        )

        players.append(player)

    # Count injured players (from API + DK-detected)
    injured_players = [p for p in players if p.is_injured]
    dk_detected_out = [p for p in players if 'DK' in p.injury_status]
    fallback_players = [p for p in players if p.is_fallback]

    metadata = {
        'total_players': len(df),
        'matched_players': len(players) - len(fallback_players),
        'fallback_players': [(p.name, p.team, p.dk_avg_pts) for p in fallback_players],
        'fallback_count': len(fallback_players),
        'unmatched_players': unmatched,  # Keep for backwards compatibility
        'unique_games': len(set(p.game_id for p in players)),
        'salary_range': (
            min(p.salary for p in players) if players else 0,
            max(p.salary for p in players) if players else 0
        ),
        'match_rate': len(players) / len(df) * 100 if len(df) > 0 else 0,
        'injured_players': [(p.name, p.injury_status) for p in injured_players],
        'injured_count': len(injured_players),
        'dk_detected_out': [(p.name, p.team) for p in dk_detected_out],
        'dk_detected_out_count': len(dk_detected_out),
    }

    return players, metadata


# =============================================================================
# Correlation Model Integration for GPP Optimization
# =============================================================================

def enrich_players_with_correlation_model(
    players: List[DFSPlayer],
    conn: Optional[sqlite3.Connection] = None,
    game_date: str = "",
    n_simulations: int = 10000
) -> List[DFSPlayer]:
    """
    Run correlation model simulations and enrich player pool with p_top1, p_top3.

    The correlation model captures:
    - Same-game correlations (players in same game are positively correlated)
    - Teammate negative correlations (usage competition)
    - Game environment effects (OT likelihood, high totals)

    This helps identify GPP-optimal players who have:
    - High p_top1: Good chance of being the #1 scorer (tournament upside)
    - High p_top3: Good chance of top-3 finish (consistent high production)

    Args:
        players: List of DFSPlayer objects to enrich
        conn: Database connection for game environment data
        game_date: Game date for environment lookup
        n_simulations: Number of Monte Carlo simulations

    Returns:
        Enriched list of DFSPlayer objects with correlation metrics
    """
    if not CORRELATION_MODEL_AVAILABLE:
        return players

    if not players:
        return players

    # Convert DFSPlayers to PlayerSlateInfo for simulation
    slate_info = []
    for p in players:
        if p.is_injured or p.is_excluded:
            continue

        # Calculate sigma from ceiling/floor
        if p.proj_ceiling > p.proj_floor:
            sigma = (p.proj_ceiling - p.proj_floor) / 4  # Approximate 2 std devs
        else:
            sigma = p.proj_fpts * 0.20  # Fallback: 20% of projection

        slate_info.append(PlayerSlateInfo(
            player_id=p.player_id,
            player_name=p.name,
            team=p.team,
            game_id=p.game_id,
            mean_score=p.proj_fpts,
            sigma=max(sigma, 1.0),
            is_star=p.role_tier == 'STAR' or p.proj_points >= 25
        ))

    if len(slate_info) < 3:
        return players

    # Get game environment data for enhanced correlation
    game_environments = {}  # game_odds game_id -> env data
    team_env_lookup = {}    # frozenset({team1, team2}) -> env data (for DK game_id matching)
    if conn and game_date:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, home_team, away_team, ot_probability, stack_score, total
                FROM game_odds
                WHERE date(game_date) = date(?)
            """, (game_date,))

            for row in cursor.fetchall():
                gid = row[0]
                env = {
                    'ot_probability': row[3] or 0.06,
                    'stack_score': row[4] or 0.5,
                    'total': row[5] or 228
                }
                game_environments[gid] = env
                # Also key by team pair for matching DK game_ids
                if row[1] and row[2]:
                    team_env_lookup[frozenset([row[1], row[2]])] = env
        except Exception:
            pass

    # Remap game_environments from game_odds game_ids to DK game_ids
    # game_odds uses "2026-02-07_DAL_SAS", DK uses "DAL_SAS_0700PM"
    # Match via team_env_lookup (keyed by frozenset of team abbreviations)
    dk_game_environments = {}
    if team_env_lookup:
        # Build team set for each DK game_id from slate_info
        dk_game_teams: Dict[str, Set[str]] = {}
        for si in slate_info:
            dk_game_teams.setdefault(si.game_id, set()).add(si.team)
        for dk_gid, teams in dk_game_teams.items():
            key = frozenset(teams)
            if key in team_env_lookup:
                dk_game_environments[dk_gid] = team_env_lookup[key]

    # Run correlated simulation
    try:
        config = CorrelationConfig(n_simulations=n_simulations, random_seed=42)
        model = PlayerCorrelationModel(config)
        results = model.run_correlated_simulation(slate_info, dk_game_environments)

        # Build lookup by player_id
        results_dict = {r.player_id: r for r in results}

        # Enrich players
        for p in players:
            if p.player_id in results_dict:
                r = results_dict[p.player_id]
                p.p_top1 = r.p_top1
                p.p_top3 = r.p_top3
                p.expected_rank = r.expected_rank
                p.support_score = r.support_score
                p.sigma = r.sigma_used

                # Update stack score from game environment (match by team pair)
                team_key = frozenset([p.team, p.opponent]) if p.opponent else None
                if team_key and team_key in team_env_lookup:
                    p.stack_score = team_env_lookup[team_key].get('stack_score', 0.5)

                # Mark as star
                p.is_star = p.role_tier == 'STAR' or p.proj_points >= 25

    except Exception as e:
        # If simulation fails, continue with basic projections
        pass

    return players


def apply_correlation_ceiling_boost(
    players: List[DFSPlayer],
    stack_threshold: float = 0.7
) -> List[DFSPlayer]:
    """
    Apply ceiling boost for players in high-correlation games.

    Players in games with:
    - High totals (shootout potential)
    - Tight spreads (competitive throughout)
    - High OT probability

    Get ceiling boosts because these games have more upside variance.

    Args:
        players: List of DFSPlayer objects
        stack_threshold: Minimum stack_score for boost (default 0.7)

    Returns:
        Players with adjusted ceilings
    """
    for p in players:
        if p.stack_score >= stack_threshold:
            # Good stacking game - boost ceiling by 8%
            ceiling_boost = 1.08
            p.proj_ceiling = p.proj_ceiling * ceiling_boost

            # Stars in good stacking games get extra leverage
            if p.is_star:
                p.leverage_score = p.leverage_score * 1.10

        elif p.stack_score >= 0.5:
            # Moderate stacking game - slight boost
            p.proj_ceiling = p.proj_ceiling * 1.03

    return players


def rank_players_by_gpp_value(
    players: List[DFSPlayer],
    p_top3_weight: float = 0.4,
    ceiling_weight: float = 0.3,
    value_weight: float = 0.3
) -> List[DFSPlayer]:
    """
    Rank players by GPP-optimal composite score.

    Combines:
    - p_top3: Probability of top-3 finish (tournament upside)
    - proj_ceiling: Maximum outcome
    - fpts_per_dollar: Salary efficiency

    Args:
        players: List of DFSPlayer objects
        p_top3_weight: Weight for p_top3 (default 0.4)
        ceiling_weight: Weight for ceiling (default 0.3)
        value_weight: Weight for value (default 0.3)

    Returns:
        Players sorted by GPP composite score
    """
    # Normalize each metric to 0-1 scale
    if not players:
        return players

    # Get max values for normalization
    max_p_top3 = max(p.p_top3 for p in players) or 1.0
    max_ceiling = max(p.proj_ceiling for p in players) or 1.0
    max_value = max(p.fpts_per_dollar for p in players) or 1.0

    for p in players:
        # Normalize each component
        norm_p_top3 = p.p_top3 / max_p_top3 if max_p_top3 > 0 else 0
        norm_ceiling = p.proj_ceiling / max_ceiling if max_ceiling > 0 else 0
        norm_value = p.fpts_per_dollar / max_value if max_value > 0 else 0

        # Composite GPP score
        p.leverage_score = (
            norm_p_top3 * p_top3_weight +
            norm_ceiling * ceiling_weight +
            norm_value * value_weight
        ) * 100  # Scale to 0-100

    return sorted(players, key=lambda x: x.leverage_score, reverse=True)


# =============================================================================
# Lineup Optimizer
# =============================================================================

def _select_stack_players(
    available: List[DFSPlayer],
    stack_config: Dict[str, Dict],
    used_ids: Set[int],
    strategy: str
) -> List[DFSPlayer]:
    """
    Select players to force into a lineup based on game stacking configuration.

    Supports three stack types:
    - Primary (3-4 players): 2-3 from primary team + 1 bring-back from opponent
    - Mini (2 players): Correlated pair from same game
    - Auto: Chooses primary/mini based on stack_score

    Returns list of DFSPlayer objects to pre-place in the lineup.
    """
    selected = []

    def get_score(p: DFSPlayer) -> float:
        if strategy == "ceiling":
            return p.proj_ceiling
        elif strategy == "value":
            return p.fpts_per_dollar
        elif strategy == "leverage":
            return p.leverage_score
        return p.proj_fpts

    def weighted_pick(candidates: List[DFSPlayer], n: int = 1) -> List[DFSPlayer]:
        """Pick n players from candidates using weighted random selection."""
        if not candidates or n <= 0:
            return []
        n = min(n, len(candidates))
        scores = [max(0.1, get_score(p)) for p in candidates]
        picks = []
        remaining = list(zip(candidates, scores))
        for _ in range(n):
            if not remaining:
                break
            players_left, weights_left = zip(*remaining)
            total = sum(weights_left)
            weights_norm = [w / total for w in weights_left]
            chosen = random.choices(list(players_left), weights=weights_norm, k=1)[0]
            picks.append(chosen)
            remaining = [(p, w) for p, w in remaining if p.player_id != chosen.player_id]
        return picks

    for game_id, config in stack_config.items():
        stack_type = config.get('type', 'none')
        if stack_type == 'none':
            continue

        teams = config.get('teams', [])
        stack_score = config.get('stack_score', 0.5)

        # Resolve "auto" to primary or mini based on stack_score
        if stack_type == 'auto':
            if stack_score >= 0.75:
                stack_type = 'primary'
            elif stack_score >= 0.50:
                stack_type = 'mini'
            else:
                continue  # Low stack score, skip

        # Find available players in this game (not already used/selected)
        selected_ids = {p.player_id for p in selected}
        game_players = [
            p for p in available
            if p.game_id == game_id
            and p.player_id not in used_ids
            and p.player_id not in selected_ids
            and p.proj_fpts > 0
        ]

        if len(game_players) < 2:
            continue

        # Group by team
        by_team: Dict[str, List[DFSPlayer]] = {}
        for p in game_players:
            by_team.setdefault(p.team, []).append(p)

        team_names = list(by_team.keys())
        if not team_names:
            continue

        if stack_type == 'primary':
            # Primary stack: 2-3 from one team + 1 bring-back
            # Pick the team with more/better options as primary
            if len(team_names) >= 2:
                team_scores = {}
                for t in team_names:
                    team_scores[t] = sum(get_score(p) for p in by_team[t])
                # Weighted random: prefer higher-scoring team but allow variety
                t_names = list(team_scores.keys())
                t_weights = [max(1.0, s) for s in team_scores.values()]
                primary_team = random.choices(t_names, weights=t_weights, k=1)[0]
                opp_team = [t for t in team_names if t != primary_team][0]
            else:
                primary_team = team_names[0]
                opp_team = None

            # Pick 2-3 from primary team (60% chance of 3, 40% chance of 2)
            n_primary = 3 if random.random() < 0.6 else 2
            primary_picks = weighted_pick(by_team[primary_team], n_primary)
            selected.extend(primary_picks)

            # Pick 1 bring-back from opponent (prefer stars/high ceiling)
            if opp_team and by_team.get(opp_team):
                bringback = weighted_pick(by_team[opp_team], 1)
                selected.extend(bringback)

        elif stack_type == 'mini':
            # Mini stack: 2 correlated players
            if len(team_names) >= 2:
                # 70% same-team pair, 30% opposing stars
                if random.random() < 0.70:
                    # Same-team: pick the team with best options
                    best_team = max(team_names, key=lambda t: sum(get_score(p) for p in by_team[t]))
                    picks = weighted_pick(by_team[best_team], 2)
                    selected.extend(picks)
                else:
                    # Opposing stars: 1 from each team
                    for t in team_names[:2]:
                        if by_team.get(t):
                            picks = weighted_pick(by_team[t], 1)
                            selected.extend(picks)
            else:
                # Only one team available, pick 2 from them
                picks = weighted_pick(game_players, 2)
                selected.extend(picks)

    return selected


def optimize_lineup_randomized(
    player_pool: List[DFSPlayer],
    strategy: str = "projection",
    model_key: str = "",
    model_profile: Optional[Dict[str, Any]] = None,
    model_context: Optional[Dict[str, float]] = None,
    locked_players: Optional[List[DFSPlayer]] = None,
    excluded_ids: Optional[Set[int]] = None,
    max_exposure: Optional[Dict[int, float]] = None,
    current_exposures: Optional[Dict[int, int]] = None,
    num_lineups_total: int = 1,
    randomization_factor: float = 0.3,
    min_salary_floor: int = 0,
    stack_config: Optional[Dict[str, Dict]] = None,
    debug_stats: Optional[Dict[str, int]] = None,
) -> Optional[DFSLineup]:
    """
    Build an optimized lineup with controlled randomization for diversity.

    Uses weighted random selection instead of pure greedy to generate
    diverse lineups while still respecting projections/strategy.

    Args:
        player_pool: Available players
        strategy: Optimization target (projection, ceiling, value, leverage)
        locked_players: Players that must be in lineup
        excluded_ids: Player IDs to exclude
        max_exposure: Maximum exposure % per player
        current_exposures: Current lineup count per player
        num_lineups_total: Total lineups being generated
        randomization_factor: How much randomness (0=greedy, 1=random)
        min_salary_floor: Minimum total salary required (e.g., 49000 means max $1000 remaining)

    Returns:
        Optimized DFSLineup or None if no valid lineup found
    """
    excluded_ids = excluded_ids or set()
    locked_players = locked_players or []
    max_exposure = max_exposure or {}
    current_exposures = current_exposures or {}
    model_profile = model_profile or {}
    model_context = model_context or {}

    def _dbg(key: str, value: int = 1) -> None:
        if debug_stats is None:
            return
        debug_stats[key] = int(debug_stats.get(key, 0)) + int(value)

    _dbg("attempts_optimizer")

    def is_min_salary_player(player: DFSPlayer) -> bool:
        return player.salary <= MIN_PLAYER_SALARY

    def min_salary_player_eligible(player: DFSPlayer) -> bool:
        if not is_min_salary_player(player):
            return True
        proj_fpts = float(getattr(player, "proj_fpts", 0.0) or 0.0)
        minutes_ok = bool(getattr(player, "minutes_validated", False))
        # Allow $3K players only if they project well OR pass minutes validation.
        return proj_fpts >= MIN_SALARY_PROJ_FPTS_GATE or minutes_ok

    # Filter available players (exclude injured, excluded, and specified IDs)
    available = [
        p for p in player_pool
        if p.player_id not in excluded_ids
        and not p.is_excluded
        and not p.is_injured
    ]

    # Check exposure limits
    def can_use_player(player: DFSPlayer) -> bool:
        if player.player_id in excluded_ids:
            return False
        if player.is_injured:
            return False
        if player.is_excluded:
            return False
        if not min_salary_player_eligible(player):
            return False

        max_exp = max_exposure.get(player.player_id, 1.0)
        current = current_exposures.get(player.player_id, 0)
        max_lineups = int(num_lineups_total * max_exp)

        return current < max_lineups

    available = [p for p in available if can_use_player(p)]

    if not available:
        _dbg("reject_no_available_after_filters")
        return None

    # Get score function based on strategy.
    # Use risk-adjusted projection for historically noisy segments.
    def get_score(p: DFSPlayer) -> float:
        risk_penalty = float(getattr(p, "segment_risk_penalty", 0.0) or 0.0)
        proj_for_scoring = float(
            getattr(p, "risk_adjusted_proj_fpts", 0.0)
            or (float(getattr(p, "proj_fpts", 0.0) or 0.0) - risk_penalty)
        )
        proj_for_scoring = max(0.0, proj_for_scoring)
        if strategy == "ceiling":
            base_score = max(
                0.1, float(getattr(p, "proj_ceiling", 0.0) or 0.0) - (risk_penalty * 0.5)
            )
        elif strategy == "value":
            if p.salary <= 0:
                return 0.1
            base_score = max(0.1, (proj_for_scoring / p.salary) * 1000)
        elif strategy == "leverage":
            leverage = float(getattr(p, "leverage_score", 0.0) or 0.0)
            leverage_decay = max(0.70, 1.0 - (risk_penalty / 6.0))
            base_score = max(0.1, leverage * leverage_decay)
        else:  # projection
            base_score = max(0.1, proj_for_scoring)

        if model_key == "midrange_v1_minutes_vegas":
            base_score *= _midrange_minutes_vegas_multiplier(
                p, model_profile, model_context
            )

        return max(0.1, base_score)

    lineup = DFSLineup()
    used_player_ids = set()

    def lineup_min_salary_count() -> int:
        return sum(1 for p in lineup.players.values() if is_min_salary_player(p))

    def lineup_redflag_count() -> int:
        return sum(1 for p in lineup.players.values() if bool(getattr(p, "is_overproj_redflag", False)))

    # Add locked players first
    for player in locked_players:
        if not min_salary_player_eligible(player):
            _dbg("reject_locked_min_salary_gate")
            return None
        if (
            bool(getattr(player, "is_overproj_redflag", False))
            and lineup_redflag_count() >= MAX_REDFLAG_SEGMENTS_PER_LINEUP
        ):
            _dbg("reject_locked_redflag_cap")
            return None
        if (
            is_min_salary_player(player)
            and lineup_min_salary_count() >= MAX_MIN_SALARY_PLAYERS_PER_LINEUP
        ):
            _dbg("reject_locked_min_salary_cap")
            return None
        for slot in ROSTER_SLOTS:
            if slot not in lineup.players and player.can_fill_slot(slot):
                lineup.players[slot] = player
                used_player_ids.add(player.player_id)
                break

    # Add game stack players (after locked, before random fill)
    if stack_config:
        stack_players = _select_stack_players(available, stack_config, used_player_ids, strategy)
        for player in stack_players:
            if player.player_id in used_player_ids:
                continue
            if not min_salary_player_eligible(player):
                continue
            if (
                bool(getattr(player, "is_overproj_redflag", False))
                and lineup_redflag_count() >= MAX_REDFLAG_SEGMENTS_PER_LINEUP
            ):
                continue
            if (
                is_min_salary_player(player)
                and lineup_min_salary_count() >= MAX_MIN_SALARY_PLAYERS_PER_LINEUP
            ):
                continue
            for slot in ROSTER_SLOTS:
                if slot not in lineup.players and player.can_fill_slot(slot):
                    lineup.players[slot] = player
                    used_player_ids.add(player.player_id)
                    break

    # Fill remaining slots with weighted random selection
    for slot in ROSTER_SLOTS:
        if slot in lineup.players:
            continue

        # Find all valid candidates for this slot
        candidates = []
        remaining_slots = sum(1 for s in ROSTER_SLOTS if s not in lineup.players and s != slot)

        for player in available:
            if player.player_id in used_player_ids:
                continue
            if not player.can_fill_slot(slot):
                continue
            if (
                bool(getattr(player, "is_overproj_redflag", False))
                and lineup_redflag_count() >= MAX_REDFLAG_SEGMENTS_PER_LINEUP
            ):
                continue
            if is_min_salary_player(player):
                if not min_salary_player_eligible(player):
                    continue
                if lineup_min_salary_count() >= MAX_MIN_SALARY_PLAYERS_PER_LINEUP:
                    continue

            # Check salary cap constraint (don't exceed max)
            tentative_salary = lineup.total_salary + player.salary
            min_salary_needed = remaining_slots * 3500  # Conservative minimum for remaining

            if tentative_salary + min_salary_needed > SALARY_CAP:
                continue

            # Check salary floor constraint (must hit minimum)
            # If we have a floor, check if we can still reach it
            if min_salary_floor > 0:
                # Estimate max salary we could add in remaining slots
                # (assume we can find players around $8000 avg for remaining)
                max_remaining_salary = remaining_slots * 9500  # Optimistic max
                if tentative_salary + max_remaining_salary < min_salary_floor:
                    # Even with expensive players for remaining slots, we can't hit floor
                    continue

                # For the LAST slot, enforce the floor strictly
                if remaining_slots == 0 and tentative_salary < min_salary_floor:
                    continue

            # Check we can still get 2 games
            if remaining_slots == 0:
                test_games = lineup.unique_games | {player.game_id}
                if len(test_games) < MIN_GAMES:
                    continue

            candidates.append(player)

        if not candidates:
            _dbg("reject_no_candidates_for_slot")
            continue

        # Weighted random selection based on score
        scores = [max(0.1, get_score(p)) for p in candidates]
        max_score = max(scores)
        min_score = min(scores)

        # Calculate if we need to spend more to hit salary floor
        salary_boost_factor = 0.0
        if min_salary_floor > 0:
            current_salary = lineup.total_salary
            salary_deficit = min_salary_floor - current_salary
            # How much do we need per remaining slot (including this one)?
            slots_left = remaining_slots + 1
            target_per_slot = salary_deficit / slots_left if slots_left > 0 else 0

            # If we need high-salary players, boost their weights
            if target_per_slot > 5000:  # Need expensive players
                salary_boost_factor = 0.3  # 30% weight boost for salary

        # Normalize scores and apply randomization
        if max_score > min_score:
            weights = []
            for i, score in enumerate(scores):
                normalized = (score - min_score) / (max_score - min_score)
                # Mix between score-based and uniform
                weight = (1 - randomization_factor) * normalized + randomization_factor * 1.0

                # Boost weight for higher-salary players when we need to spend
                if salary_boost_factor > 0:
                    salary_normalized = candidates[i].salary / 10000  # Normalize salary (0-1 range roughly)
                    weight = weight * (1 + salary_boost_factor * salary_normalized)

                weights.append(max(0.01, weight))
        else:
            weights = [1.0] * len(candidates)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Select player
        selected = random.choices(candidates, weights=weights, k=1)[0]
        lineup.players[slot] = selected
        used_player_ids.add(selected.player_id)

    # Validate final lineup
    if lineup.is_valid:
        # Check minimum salary floor constraint
        if min_salary_floor > 0 and lineup.total_salary < min_salary_floor:
            # Lineup doesn't use enough salary - reject it
            _dbg("reject_final_salary_floor")
            return None
        if lineup_min_salary_count() > MAX_MIN_SALARY_PLAYERS_PER_LINEUP:
            _dbg("reject_final_min_salary_cap")
            return None
        _dbg("accepted_optimizer")
        return lineup

    _dbg("reject_invalid_lineup")
    return None


def generate_diversified_lineups(
    player_pool: List[DFSPlayer],
    num_lineups: int = 100,
    max_player_exposure: float = 0.50,  # 50% max exposure (was 40%)
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    excluded_games: Optional[Set[str]] = None,
    exposure_targets: Optional[Dict[int, float]] = None,
    min_salary_floor: int = 0,
    stack_config: Optional[Dict[str, Dict]] = None,
    strategy_mix: Optional[List[Tuple[str, float, float]]] = None,
    ceiling_focus_pct: int = 0,
    aggressive_ceiling_stack: bool = False,
    model_key: str = "",
    model_label: str = "",
    debug_stats: Optional[Dict[str, int]] = None,
) -> List[DFSLineup]:
    """
    Generate diversified GPP tournament lineups using randomized optimization.

    Strategy distribution:
    - Projection (25%): Weighted toward highest projected
    - Ceiling (30%): Weighted toward highest ceiling
    - Value (20%): Weighted toward best value plays
    - Balanced (25%): High randomization for diversity

    Args:
        player_pool: Available players with projections
        num_lineups: Target number of lineups
        max_player_exposure: Maximum exposure per player (0.0-1.0)
        progress_callback: Optional progress callback (current, total, message)
        excluded_games: Set of game_ids to exclude (for postponed games)
        exposure_targets: Dict mapping player_id -> target exposure fraction
                          (e.g. {12345: 0.30} means 30% of lineups)
        min_salary_floor: Minimum total salary required per lineup (e.g., 49000).
                          Lineups using less salary will be rejected.
                          Set to SALARY_CAP - max_remaining (e.g., 50000 - 1000 = 49000).
        strategy_mix: Optional list of (strategy, randomization_factor, target_pct)
                      tuples overriding the default generation mix.
        ceiling_focus_pct: 0-100 dial that shifts allocation toward ceiling/leverage.
        aggressive_ceiling_stack: If True, keep stacking active for ceiling/leverage.
        model_key: Optional model/profile key annotation on generated lineups.
        model_label: Optional display label annotation on generated lineups.

    Returns:
        List of valid DFSLineup objects
    """
    lineups: List[DFSLineup] = []
    exposures: Dict[int, int] = {}
    existing_lineup_keys: Set[Tuple[int, ...]] = set()

    def _dbg(key: str, value: int = 1) -> None:
        if debug_stats is None:
            return
        debug_stats[key] = int(debug_stats.get(key, 0)) + int(value)

    _dbg("pool_initial", len(player_pool))

    # Filter out excluded games
    excluded_games = excluded_games or set()
    filtered_pool = [p for p in player_pool if p.game_id not in excluded_games]
    _dbg("pool_after_game_filter", len(filtered_pool))

    # Filter out injured players (OUT/DOUBTFUL)
    injured_count = len([p for p in filtered_pool if p.is_injured])
    filtered_pool = [p for p in filtered_pool if not p.is_injured]
    _dbg("pool_after_injury_filter", len(filtered_pool))

    if not filtered_pool:
        _dbg("reject_empty_pool")
        return []

    # Get locked players (ensure they're not injured)
    locked = [p for p in filtered_pool if p.is_locked and not p.is_injured]
    locked_ids = {p.player_id for p in locked}

    # Minutes variance filter:
    # Exclude top-quartile volatility plays unless they are elite projection/ceiling
    # or explicitly locked by the user.
    variance_values = [
        float(getattr(p, "minutes_variance", 0.0) or 0.0)
        for p in filtered_pool
        if float(getattr(p, "minutes_variance", 0.0) or 0.0) > 0
    ]
    if len(variance_values) >= MINUTES_VARIANCE_FILTER_MIN_SAMPLE:
        variance_cutoff = float(np.percentile(variance_values, MINUTES_VARIANCE_FILTER_PERCENTILE))

        def _passes_minutes_variance_filter(player: DFSPlayer) -> bool:
            if player.player_id in locked_ids:
                return True
            minutes_var = float(getattr(player, "minutes_variance", 0.0) or 0.0)
            if minutes_var <= 0 or minutes_var <= variance_cutoff:
                return True

            proj_fpts = float(getattr(player, "proj_fpts", 0.0) or 0.0)
            proj_ceiling = float(getattr(player, "proj_ceiling", 0.0) or 0.0)
            return (
                proj_fpts >= MINUTES_VARIANCE_ELITE_PROJ_FPTS
                or proj_ceiling >= MINUTES_VARIANCE_ELITE_CEILING
            )

        filtered_pool = [p for p in filtered_pool if _passes_minutes_variance_filter(p)]
        _dbg("pool_after_minutes_variance_filter", len(filtered_pool))
        if not filtered_pool:
            _dbg("reject_empty_pool_after_minutes_filter")
            return []

    # Resolve exposure target players
    exposure_targets = exposure_targets or {}
    target_player_map: Dict[int, Dict] = {}
    for pid, target_pct in exposure_targets.items():
        if pid in locked_ids:
            continue  # Already locked at 100%
        player_obj = next(
            (p for p in filtered_pool if p.player_id == pid), None
        )
        if player_obj and not player_obj.is_excluded and not player_obj.is_injured:
            target_player_map[pid] = {
                'player': player_obj,
                'target': target_pct
            }

    # Build max exposure dict
    max_exp = {p.player_id: max_player_exposure for p in filtered_pool}

    # Targeted players: raise their max to at least their target
    for pid, info in target_player_map.items():
        max_exp[pid] = max(max_player_exposure, info['target'])

    # Segment-aware safety caps from calibration diagnostics.
    # These caps are intentionally stricter for historically problematic buckets.
    for p in filtered_pool:
        segment_cap = float(getattr(p, "segment_exposure_cap", 1.0) or 1.0)
        max_exp[p.player_id] = min(max_exp.get(p.player_id, max_player_exposure), segment_cap)

    profile_cfg = LINEUP_MODEL_PROFILES.get(model_key, {}) if model_key else {}
    overlap_cap = int(profile_cfg.get("overlap_cap", 8) or 8)
    core_play_inject_prob = float(profile_cfg.get("core_play_inject_prob", 0.0) or 0.0)
    core_play_min_own_pct = float(profile_cfg.get("core_play_min_own_pct", 14.0) or 14.0)
    core_play_max_own_pct = float(profile_cfg.get("core_play_max_own_pct", 60.0) or 60.0)
    core_play_min_projection = float(profile_cfg.get("core_play_min_projection", 24.0) or 24.0)
    low_own_inject_prob = float(profile_cfg.get("low_own_inject_prob", 0.0) or 0.0)
    low_own_max_own_pct = float(profile_cfg.get("low_own_max_own_pct", 12.0) or 12.0)
    low_own_min_projection = float(profile_cfg.get("low_own_min_projection", 20.0) or 20.0)
    standout_lock_prob = float(profile_cfg.get("standout_lock_prob", 0.0) or 0.0)
    standout_min_ceiling_gap = float(profile_cfg.get("standout_min_ceiling_gap", 8.0) or 8.0)
    midrange_signal_inject_prob = float(
        profile_cfg.get("midrange_signal_inject_prob", 0.0) or 0.0
    )
    model_context = _build_midrange_minutes_vegas_context(filtered_pool, profile_cfg)

    lineup_player_sets: List[Set[int]] = []

    def _can_use_for_lock(player: DFSPlayer) -> bool:
        max_pct = float(max_exp.get(player.player_id, max_player_exposure))
        max_lineups_for_player = int(num_lineups * max_pct)
        return exposures.get(player.player_id, 0) < max_lineups_for_player

    def _add_lock_candidate(attempt_locked: List[DFSPlayer], candidate: DFSPlayer) -> None:
        if candidate.player_id not in {p.player_id for p in attempt_locked}:
            attempt_locked.append(candidate)

    def _overlap_allowed(lineup_key: Tuple[int, ...]) -> bool:
        if overlap_cap >= 8:
            return True
        candidate_set = set(lineup_key)
        for existing in lineup_player_sets:
            if len(candidate_set & existing) > overlap_cap:
                return False
        return True

    def _risk_adjusted_fpts(player: DFSPlayer) -> float:
        return float(
            getattr(player, "risk_adjusted_proj_fpts", 0.0)
            or getattr(player, "proj_fpts", 0.0)
            or 0.0
        )

    core_play_candidates = [
        p for p in filtered_pool
        if (
            p.player_id not in locked_ids
            and not p.is_excluded
            and not p.is_injured
            and core_play_min_own_pct <= float(getattr(p, "ownership_proj", 0.0) or 0.0) <= core_play_max_own_pct
            and _risk_adjusted_fpts(p) >= core_play_min_projection
        )
    ]
    core_play_candidates = sorted(
        core_play_candidates,
        key=lambda p: (
            (
                _risk_adjusted_fpts(p) * 0.65
                + float(getattr(p, "proj_ceiling", 0.0) or 0.0) * 0.35
            )
            * (1.0 + min(30.0, float(getattr(p, "ownership_proj", 0.0) or 0.0)) / 100.0),
            float(getattr(p, "salary", 0.0) or 0.0),
        ),
        reverse=True,
    )

    low_own_candidates = [
        p for p in filtered_pool
        if (
            p.player_id not in locked_ids
            and not p.is_excluded
            and not p.is_injured
            and float(getattr(p, "ownership_proj", 0.0) or 0.0) <= low_own_max_own_pct
            and _risk_adjusted_fpts(p) >= low_own_min_projection
        )
    ]
    low_own_candidates = sorted(
        low_own_candidates,
        key=lambda p: (
            float(getattr(p, "leverage_score", 0.0) or 0.0),
            float(getattr(p, "proj_ceiling", 0.0) or 0.0),
        ),
        reverse=True,
    )

    standout_scored: List[Tuple[DFSPlayer, float]] = []
    for p in filtered_pool:
        if p.player_id in locked_ids or p.is_excluded or p.is_injured:
            continue
        ceiling_gap = float(getattr(p, "proj_ceiling", 0.0) or 0.0) - _risk_adjusted_fpts(p)
        if ceiling_gap < standout_min_ceiling_gap:
            continue
        ownership = max(1.0, float(getattr(p, "ownership_proj", 0.0) or 0.0))
        standout_score = (ceiling_gap ** 1.15) * (1.0 + max(0.0, 15.0 - ownership) / 18.0)
        standout_scored.append((p, standout_score))
    standout_scored.sort(key=lambda x: x[1], reverse=True)
    standout_scored = standout_scored[:40]

    midrange_signal_scored: List[Tuple[DFSPlayer, float]] = []
    if bool(profile_cfg.get("midrange_focus_enabled", False)):
        for p in filtered_pool:
            if p.player_id in locked_ids or p.is_excluded or p.is_injured:
                continue
            signal_score = _midrange_minutes_vegas_signal_score(
                p, profile_cfg, model_context
            )
            if signal_score <= 0:
                continue
            midrange_signal_scored.append((p, signal_score))
        midrange_signal_scored.sort(key=lambda x: x[1], reverse=True)
        midrange_signal_scored = midrange_signal_scored[:40]

    # Strategy configuration: (strategy, randomization_factor, target_count_pct)
    strategies = strategy_mix or [
        ("projection", 0.25, 0.20),  # core projection-optimal
        ("ceiling", 0.30, 0.25),     # upside-focused
        ("value", 0.25, 0.15),       # value-focused
        ("leverage", 0.35, 0.15),    # low-own/high-ceiling contrarian
        ("projection", 0.60, 0.25),  # balanced high-randomization
    ]

    # Ceiling focus dial shifts weight from projection/value into ceiling/leverage.
    # This keeps model profiles flexible while allowing a slate-level upside push.
    focus = max(0.0, min(1.0, float(ceiling_focus_pct) / 100.0))
    if focus > 0:
        adjusted = []
        for strategy, rand_factor, pct in strategies:
            if strategy == "ceiling":
                multiplier = 1.0 + (1.25 * focus)
            elif strategy == "leverage":
                multiplier = 1.0 + (0.65 * focus)
            elif strategy == "projection":
                multiplier = max(0.15, 1.0 - (0.50 * focus))
            elif strategy == "value":
                multiplier = max(0.10, 1.0 - (0.45 * focus))
            else:
                multiplier = 1.0
            adjusted.append((strategy, rand_factor, max(0.0, pct * multiplier)))
        strategies = adjusted

    total_pct = sum(max(0.0, pct) for _, _, pct in strategies)
    if total_pct <= 0:
        strategies = [("projection", 0.50, 1.0)]
    else:
        strategies = [(s, r, max(0.0, p) / total_pct) for s, r, p in strategies]

    total_generated = 0
    max_total_attempts = num_lineups * 50  # Allow many more attempts
    total_attempts = 0

    raw_targets = [num_lineups * pct for _, _, pct in strategies]
    target_counts = [int(t) for t in raw_targets]
    remainder = num_lineups - sum(target_counts)
    if remainder > 0:
        order = sorted(
            range(len(strategies)),
            key=lambda idx: raw_targets[idx] - target_counts[idx],
            reverse=True,
        )
        for idx in order[:remainder]:
            target_counts[idx] += 1

    for (strategy, rand_factor, pct), target_count in zip(strategies, target_counts):
        strategy_generated = 0
        strategy_attempts = 0
        max_strategy_attempts = target_count * 20  # More attempts per strategy

        while (strategy_generated < target_count and
               total_attempts < max_total_attempts and
               strategy_attempts < max_strategy_attempts and
               total_generated < num_lineups):

            total_attempts += 1
            strategy_attempts += 1

            if progress_callback and total_attempts % 10 == 0:
                progress_callback(
                    total_generated, num_lineups,
                    f"Building {strategy} lineups ({total_generated}/{num_lineups})..."
                )

            # Build per-attempt locked list: always-locked + exposure targets
            attempt_locked = list(locked)

            for pid, info in target_player_map.items():
                current_count = exposures.get(pid, 0)
                target_count_for_player = round(num_lineups * info['target'])
                remaining_lineups = num_lineups - total_generated
                needed = target_count_for_player - current_count

                if remaining_lineups > 0 and needed > 0:
                    # Catch-up probability: adjusts to converge on target
                    lock_prob = min(1.0, needed / remaining_lineups)
                    if random.random() < lock_prob:
                        _add_lock_candidate(attempt_locked, info['player'])

            # Profile: inject core-play anchors to reduce missed high-field outcomes.
            if core_play_candidates and random.random() < core_play_inject_prob:
                available_core = [
                    p for p in core_play_candidates
                    if _can_use_for_lock(p)
                    and p.player_id not in {lp.player_id for lp in attempt_locked}
                ]
                if available_core:
                    weighted_pool = available_core[:20]
                    weights = [
                        max(
                            0.05,
                            _risk_adjusted_fpts(p)
                            + (float(getattr(p, "proj_ceiling", 0.0) or 0.0) * 0.20)
                            + (float(getattr(p, "ownership_proj", 0.0) or 0.0) * 0.30),
                        )
                        for p in weighted_pool
                    ]
                    core_choice = random.choices(weighted_pool, weights=weights, k=1)[0]
                    _add_lock_candidate(attempt_locked, core_choice)

            # Profile: inject low-owned upside candidates at controlled frequency.
            if low_own_candidates and random.random() < low_own_inject_prob:
                available_low_own = [
                    p for p in low_own_candidates
                    if _can_use_for_lock(p)
                    and p.player_id not in {lp.player_id for lp in attempt_locked}
                ]
                if available_low_own:
                    weighted_pool = available_low_own[:20]
                    weights = [
                        max(
                            0.05,
                            float(getattr(p, "leverage_score", 0.0) or 0.0) +
                            (float(getattr(p, "proj_ceiling", 0.0) or 0.0) * 0.02),
                        )
                        for p in weighted_pool
                    ]
                    lock_choice = random.choices(weighted_pool, weights=weights, k=1)[0]
                    _add_lock_candidate(attempt_locked, lock_choice)

            # Profile: prioritize a $5k-$6.5k signal play with rising minutes or strong Vegas.
            if midrange_signal_scored and random.random() < midrange_signal_inject_prob:
                available_midrange = [
                    (p, score) for p, score in midrange_signal_scored
                    if _can_use_for_lock(p)
                    and p.player_id not in {lp.player_id for lp in attempt_locked}
                ]
                if available_midrange:
                    weighted_pool = available_midrange[:15]
                    lock_choice = random.choices(
                        [p for p, _ in weighted_pool],
                        weights=[score for _, score in weighted_pool],
                        k=1,
                    )[0]
                    _add_lock_candidate(attempt_locked, lock_choice)

            # Profile: force occasional standout missed-capture candidates.
            if standout_scored and random.random() < standout_lock_prob:
                available_standout = [
                    (p, score) for p, score in standout_scored
                    if _can_use_for_lock(p)
                    and p.player_id not in {lp.player_id for lp in attempt_locked}
                ]
                if available_standout:
                    weighted_pool = available_standout[:15]
                    lock_choice = random.choices(
                        [p for p, _ in weighted_pool],
                        weights=[score for _, score in weighted_pool],
                        k=1,
                    )[0]
                    _add_lock_candidate(attempt_locked, lock_choice)

            # Determine stack config for this strategy
            # Ceiling lineups: full stacking (maximize correlation upside)
            # Projection lineups: downgrade primary->mini for balance
            # Leverage lineups: 50% chance of no stacking (contrarian)
            # Value lineups: use as configured
            # Balanced (high-rand projection): randomize stack types
            attempt_stack = stack_config
            stack_locked_by_profile = False
            ranked_stacks = []
            if stack_config:
                active_stack_items = [
                    (gid, cfg)
                    for gid, cfg in stack_config.items()
                    if str((cfg or {}).get("type", "none")).lower() != "none"
                ]
                ranked_stacks = sorted(
                    active_stack_items,
                    key=lambda item: float((item[1] or {}).get("stack_score", 0.0) or 0.0),
                    reverse=True,
                )

            # Profile-specific stack scripts.
            if ranked_stacks and model_key == "spike_v2_tail" and strategy in {"ceiling", "leverage"}:
                attempt_stack = {}
                gid, cfg = ranked_stacks[0]
                attempt_stack[gid] = {**cfg, "type": "primary"}
                if len(ranked_stacks) > 1 and random.random() < 0.55:
                    gid2, cfg2 = ranked_stacks[1]
                    attempt_stack[gid2] = {**cfg2, "type": "mini"}
                stack_locked_by_profile = True
            elif ranked_stacks and model_key == "cluster_v1_experimental":
                cluster_roll = random.random()
                if cluster_roll < 0.34:
                    gid, cfg = ranked_stacks[0]
                    attempt_stack = {gid: {**cfg, "type": "primary"}}
                elif cluster_roll < 0.67:
                    attempt_stack = {
                        gid: {**cfg, "type": "mini"}
                        for gid, cfg in ranked_stacks[:2]
                    }
                else:
                    if random.random() < 0.40:
                        attempt_stack = None
                    else:
                        tail_games = ranked_stacks[-2:] if len(ranked_stacks) > 1 else ranked_stacks
                        attempt_stack = {gid: {**cfg, "type": "mini"} for gid, cfg in tail_games}
                stack_locked_by_profile = True
            elif ranked_stacks and model_key == "standout_v1_capture" and strategy in {"ceiling", "leverage"}:
                gid, cfg = ranked_stacks[0]
                attempt_stack = {gid: {**cfg, "type": "primary"}}
                stack_locked_by_profile = True

            if not stack_locked_by_profile:
                if stack_config and strategy == 'leverage':
                    # In aggressive ceiling-stack mode, keep leverage stacks active.
                    if not aggressive_ceiling_stack and random.random() < 0.5:
                        attempt_stack = None  # Contrarian: no forced stacking
                elif stack_config and strategy == 'projection' and rand_factor >= 0.5:
                    # Balanced bucket: randomize between full/mini/none
                    r = random.random()
                    if r < 0.33:
                        attempt_stack = None
                    elif r < 0.66:
                        # Downgrade primaries to minis
                        attempt_stack = {
                            gid: {**cfg, 'type': 'mini' if cfg.get('type') == 'primary' else cfg.get('type')}
                            for gid, cfg in stack_config.items()
                        }

            lineup = optimize_lineup_randomized(
                player_pool=filtered_pool,
                strategy=strategy,
                model_key=model_key,
                model_profile=profile_cfg,
                model_context=model_context,
                locked_players=attempt_locked,
                max_exposure=max_exp,
                current_exposures=exposures,
                num_lineups_total=num_lineups,
                randomization_factor=rand_factor,
                min_salary_floor=min_salary_floor,
                stack_config=attempt_stack,
                debug_stats=debug_stats,
            )

            if lineup and lineup.is_valid:
                # Check lineup uniqueness
                lineup_key = tuple(sorted(p.player_id for p in lineup.players.values()))

                if lineup_key not in existing_lineup_keys and _overlap_allowed(lineup_key):
                    lineup.model_key = model_key
                    lineup.model_label = model_label
                    lineup.generation_strategy = strategy
                    lineups.append(lineup)
                    existing_lineup_keys.add(lineup_key)
                    lineup_player_sets.append(set(lineup_key))
                    strategy_generated += 1
                    total_generated += 1
                    _dbg("accepted_lineups")

                    # Update exposures
                    for player in lineup.players.values():
                        exposures[player.player_id] = exposures.get(player.player_id, 0) + 1
                elif lineup_key in existing_lineup_keys:
                    _dbg("reject_duplicate_lineup")
                else:
                    _dbg("reject_overlap_cap")
            else:
                _dbg("reject_optimizer_none")

    if progress_callback:
        progress_callback(total_generated, num_lineups, "Complete!")

    _dbg("lineups_returned", len(lineups))

    return lineups


# =============================================================================
# Export Functions
# =============================================================================

def export_lineups_to_dk_csv(lineups: List[DFSLineup], output_path: Path) -> None:
    """
    Export lineups to DraftKings upload format.

    Format:
    PG,SG,SF,PF,C,G,F,UTIL
    player_id1,player_id2,...
    """
    rows = [lineup.to_dk_export_row() for lineup in lineups]
    df = pd.DataFrame(rows, columns=ROSTER_SLOTS)
    df.to_csv(output_path, index=False)


def generate_exposure_report(lineups: List[DFSLineup]) -> pd.DataFrame:
    """Generate player exposure report from lineups."""
    exposure_data = {}

    for lineup in lineups:
        for player in lineup.players.values():
            if player.player_id not in exposure_data:
                exposure_data[player.player_id] = {
                    'name': player.name,
                    'team': player.team,
                    'salary': player.salary,
                    'proj_fpts': player.proj_fpts,
                    'count': 0
                }
            exposure_data[player.player_id]['count'] += 1

    df = pd.DataFrame(exposure_data.values())
    df['exposure_pct'] = df['count'] / len(lineups) * 100
    df = df.sort_values('exposure_pct', ascending=False)

    return df


def generate_stack_report(lineups: List[DFSLineup]) -> pd.DataFrame:
    """Generate game stack distribution report."""
    stack_counts = {}

    for lineup in lineups:
        stacks = lineup.game_stacks
        for game_id, players in stacks.items():
            stack_size = len(players)
            if stack_size >= 2:  # Only count as stack if 2+ players
                key = (game_id, stack_size)
                if key not in stack_counts:
                    stack_counts[key] = {
                        'game': game_id,
                        'stack_size': stack_size,
                        'count': 0
                    }
                stack_counts[key]['count'] += 1

    df = pd.DataFrame(stack_counts.values())
    if not df.empty:
        df = df.sort_values(['stack_size', 'count'], ascending=[False, False])

    return df


# =============================================================================
# Contest Results Parser
# =============================================================================

def parse_contest_standings(csv_path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Parse DraftKings contest standings CSV to extract actual ownership and FPTS.

    The DK contest standings CSV has a dual-column layout:
    - Left (cols 0-5): Entry standings (Rank, EntryId, EntryName, TimeRemaining, Points, Lineup)
    - Col 6: Empty separator
    - Right (cols 7-10): Player data (Player, Roster Position, %Drafted, FPTS)

    The player data is a separate list sorted by %Drafted descending.

    Args:
        csv_path: Path to contest standings CSV file.

    Returns:
        Tuple of:
        - ownership_map: {player_name: ownership_pct} (e.g., {"Kyle Kuzma": 15.41})
        - actual_fpts_map: {player_name: fpts} (e.g., {"Kyle Kuzma": 57.0})
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    ownership_map = {}
    actual_fpts_map = {}

    # The player data is in columns: Player, Roster Position, %Drafted, FPTS
    # These are columns index 7, 8, 9, 10 (after the empty separator col 6)
    cols = df.columns.tolist()

    # Find the Player and %Drafted columns by name
    player_col = None
    drafted_col = None
    fpts_col = None

    for i, col in enumerate(cols):
        col_lower = str(col).strip().lower()
        if col_lower == 'player':
            player_col = col
        elif col_lower == '%drafted':
            drafted_col = col
        elif col_lower == 'fpts':
            fpts_col = col

    if player_col is None or drafted_col is None:
        raise ValueError(
            f"Could not find 'Player' and '%Drafted' columns. "
            f"Available columns: {cols}"
        )

    for _, row in df.iterrows():
        player_name = str(row.get(player_col, '')).strip()
        if not player_name or player_name == 'nan':
            continue

        # Parse %Drafted (e.g., "48.22%" -> 48.22)
        drafted_str = str(row.get(drafted_col, '0')).strip().replace('%', '')
        try:
            ownership = float(drafted_str)
            if not (0 <= ownership <= 100):
                continue  # Skip invalid ownership values
        except (ValueError, TypeError):
            continue

        # Parse FPTS
        fpts = 0.0
        if fpts_col:
            try:
                fpts = float(row.get(fpts_col, 0))
            except (ValueError, TypeError):
                fpts = 0.0

        # Only keep first occurrence (highest ownership if duplicates)
        if player_name not in ownership_map:
            ownership_map[player_name] = ownership
            actual_fpts_map[player_name] = fpts

    return ownership_map, actual_fpts_map


def parse_entry_name(entry_name: str) -> Tuple[str, int, int]:
    """Parse DK EntryName into (username, entry_number, max_entries).

    Examples:
        "DaddyBambi (9/20)"  -> ("DaddyBambi", 9, 20)
        "kwwagner1142"       -> ("kwwagner1142", 1, 1)

    Returns:
        (username, entry_number, max_entries)
    """
    if not entry_name or not isinstance(entry_name, str):
        return str(entry_name).strip(), 1, 1

    match = re.match(r'^(.+?)\s*\((\d+)/(\d+)\)$', entry_name.strip())
    if match:
        return match.group(1).strip(), int(match.group(2)), int(match.group(3))
    return entry_name.strip(), 1, 1


def parse_lineup_string(lineup_str: str) -> Dict[str, str]:
    """Parse DK lineup string into {position: player_name} dict.

    The lineup string looks like:
        "C Isaiah Jackson F Jalen Johnson G Ayo Dosunmu PF Kyle Filipowski
         PG Isaiah Collier SF Kyle Kuzma SG Quenton Jackson UTIL Ryan Rollins"

    Each position code is followed by a player name until the next position code.

    Returns:
        Dict like {'C': 'Isaiah Jackson', 'PG': 'Isaiah Collier', ...}
        Returns empty dict if lineup_str is empty/invalid.
    """
    if not lineup_str or not isinstance(lineup_str, str) or lineup_str.strip() == '':
        return {}

    text = lineup_str.strip()

    # Find position tokens and their locations
    # Regex: position token preceded by start-of-string or whitespace,
    # followed by a space and an uppercase letter (start of player name)
    pattern = r'(?:^|\s)(UTIL|PG|SG|SF|PF|C|G|F)\s+(?=[A-Z])'
    matches = list(re.finditer(pattern, text))

    if not matches:
        return {}

    result = {}
    for i, match in enumerate(matches):
        position = match.group(1)
        # Player name starts right after the position token + space
        name_start = match.end()
        # Player name ends at the start of the next position match (or end of string)
        if i + 1 < len(matches):
            name_end = matches[i + 1].start()
        else:
            name_end = len(text)

        player_name = text[name_start:name_end].strip()
        if player_name:
            result[position] = player_name

    return result


def parse_contest_entries(csv_path, contest_id: str) -> List[Dict]:
    """Parse the LEFT side of a DK contest standings CSV to extract entries.

    Reads Rank, EntryName, Points, Lineup columns, parses usernames and
    lineup strings, and deduplicates identical lineups per user.

    Args:
        csv_path: Path to the contest standings CSV.
        contest_id: Contest identifier (usually from filename).

    Returns:
        List of dicts with keys: username, max_entries, entry_count, rank,
        points, lineup_raw, pg, sg, sf, pf, c, g, f, util
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    cols = df.columns.tolist()

    # Find left-side columns by name (case-insensitive)
    rank_col = entry_col = points_col = lineup_col = None
    for col in cols:
        cl = str(col).strip().lower()
        if cl == 'rank':
            rank_col = col
        elif cl == 'entryname':
            entry_col = col
        elif cl == 'points':
            points_col = col
        elif cl == 'lineup':
            lineup_col = col

    if entry_col is None or lineup_col is None:
        raise ValueError(
            f"Could not find 'EntryName' and 'Lineup' columns. "
            f"Available columns: {cols}"
        )

    # Group by (username, lineup_raw) for deduplication
    # key -> {username, max_entries, entry_count, best_rank, best_points, lineup_raw, positions}
    grouped: Dict[Tuple[str, str], Dict] = {}

    for _, row in df.iterrows():
        entry_name = str(row.get(entry_col, '')).strip()
        lineup_raw = str(row.get(lineup_col, '')).strip()

        if not entry_name or entry_name == 'nan':
            continue
        if not lineup_raw or lineup_raw == 'nan':
            continue

        # Parse rank and points
        try:
            rank = int(row.get(rank_col, 0)) if rank_col else 0
        except (ValueError, TypeError):
            rank = 0
        try:
            points = float(row.get(points_col, 0)) if points_col else 0.0
        except (ValueError, TypeError):
            points = 0.0

        if points <= 0:
            continue  # Skip entries with no score (incomplete)

        username, entry_num, max_entries = parse_entry_name(entry_name)
        key = (username, lineup_raw)

        if key in grouped:
            # Same user, same lineup — increment count, keep best rank
            grouped[key]['entry_count'] += 1
            if rank > 0 and (grouped[key]['rank'] == 0 or rank < grouped[key]['rank']):
                grouped[key]['rank'] = rank
            if points > grouped[key]['points']:
                grouped[key]['points'] = points
            if max_entries > grouped[key]['max_entries']:
                grouped[key]['max_entries'] = max_entries
        else:
            positions = parse_lineup_string(lineup_raw)
            grouped[key] = {
                'username': username,
                'max_entries': max_entries,
                'entry_count': 1,
                'rank': rank,
                'points': points,
                'lineup_raw': lineup_raw,
                'pg': positions.get('PG', ''),
                'sg': positions.get('SG', ''),
                'sf': positions.get('SF', ''),
                'pf': positions.get('PF', ''),
                'c': positions.get('C', ''),
                'g': positions.get('G', ''),
                'f': positions.get('F', ''),
                'util': positions.get('UTIL', ''),
            }

    return list(grouped.values())
