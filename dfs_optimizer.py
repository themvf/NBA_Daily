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
from typing import Callable, Dict, List, Optional, Set, Tuple

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

    # Research / scouting fields
    role_tier: str = ""              # STAR, STARTER, ROTATION, BENCH
    avg_fpts_last7: float = 0.0     # Average DK FPTS over last 7 games
    fpts_variance: float = 0.0      # Std dev of recent DK FPTS

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

# League average stats per game (2025-26 season, for opponent adjustment)
LEAGUE_AVG_TEAM_REB = 44.0
LEAGUE_AVG_TEAM_AST = 25.5
LEAGUE_AVG_TEAM_FG3M = 12.5


# =============================================================================
# Lineup Data Class
# =============================================================================

@dataclass
class DFSLineup:
    """Represents a complete 8-player DFS lineup."""
    players: Dict[str, DFSPlayer] = field(default_factory=dict)  # slot -> player

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

    Calibrated against 312 players across 3 slates (2026-02-04 to 02-06).
    Uses power-law sharpening to separate chalk (30%+) from contrarian (<3%).

    Key drivers:
    1. Value (FPTS/$) -- optimizers push toward high-value plays
    2. DK AvgPointsPerGame -- visible on DK player card, influences casual players
    3. Salary tier -- determines base ownership and scaling ceiling
    4. Injury boost -- players absorbing injured teammates' usage draw higher ownership
    """
    if player.salary <= 0 or player.proj_fpts <= 0:
        return 0.5

    # --- Factor 1: Value score (0-10 range) ---
    value_score = min(10.0, player.fpts_per_dollar)

    # --- Factor 2: DK visibility (0-10 range) ---
    dk_avg = getattr(player, 'dk_avg_pts', 0.0) or 0.0
    dk_visibility = min(10.0, dk_avg / 7.0) if dk_avg > 0 else value_score * 0.7

    # --- Factor 3: Salary-tier parameters ---
    # Calibrated from actual contest ownership data:
    #   Stars:  high base (always rostered), visibility-driven
    #   Mid:    moderate base, balanced drivers
    #   Value:  low base, value-driven, can spike on obvious plays
    #   Punt:   very low base, almost entirely value-driven
    if player.salary >= 9000:
        tier_base = 8.0
        tier_ceiling = 55.0
        vis_weight = 0.65  # Stars: name recognition dominates
    elif player.salary >= 7000:
        tier_base = 3.0
        tier_ceiling = 30.0
        vis_weight = 0.55
    elif player.salary >= 5000:
        tier_base = 1.5
        tier_ceiling = 32.0
        vis_weight = 0.40  # Value: optimizer value matters more
    else:
        tier_base = 0.5
        tier_ceiling = 35.0
        vis_weight = 0.30  # Punt: almost entirely value-driven

    # --- Combine factors with tier-specific weighting ---
    val_weight = 1.0 - vis_weight
    raw_signal = value_score * val_weight + dk_visibility * vis_weight

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
    season: str = "2025-26"
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

    if player.salary > 0:
        value_score = player.fpts_per_dollar
        player.ownership_proj = min(30.0, max(1.0, value_score * 3))

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
            player.fpts_per_dollar = (dk_avg / player.salary * 1000) if player.salary > 0 else 0
            # Rough stat estimates (for display purposes only)
            player.proj_points = dk_avg * 0.45   # ~45% of FPTS from points
            player.proj_rebounds = dk_avg * 0.10
            player.proj_assists = dk_avg * 0.08
            player.ownership_proj = 5.0  # Assume low ownership for unknowns
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
        return _generate_simple_projections(conn, player, logs, season)

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

    # --- Calibration: Salary-tier bias correction ---
    # Based on 312-player analysis across 3 contest slates (2026-02-04 to 02-06):
    #   Stars ($8K+):    bias -1.60 (over-projecting) -> scale down 4%
    #   Mid ($6K-7.9K):  bias +0.64 (negligible)      -> no change
    #   Value ($4K-5.9K): bias +0.98 (under-proj)     -> scale up 5%
    #   Punt (<$4K):     bias +2.09 (under-proj)       -> scale up 20%
    if player.salary >= 8000:
        player.proj_fpts *= 0.96
    elif player.salary >= 6000:
        pass  # Mid-tier bias is negligible
    elif player.salary >= 4000:
        player.proj_fpts *= 1.05
    else:
        player.proj_fpts *= 1.20

    # --- Recency blend: nudge projection toward recent 7-game average ---
    # If a player's recent production diverges significantly from the model,
    # blend in the recent average at 20% weight (captures hot/cold streaks)
    if player.avg_fpts_last7 > 0 and abs(player.avg_fpts_last7 - player.proj_fpts) > 3.0:
        player.proj_fpts = round(player.proj_fpts * 0.80 + player.avg_fpts_last7 * 0.20, 1)

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

    # Estimate ownership using calibrated nonlinear model
    player.ownership_proj = _estimate_ownership(
        player, has_injury_boost=("💉" in player.analytics_used)
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
    locked_players: Optional[List[DFSPlayer]] = None,
    excluded_ids: Optional[Set[int]] = None,
    max_exposure: Optional[Dict[int, float]] = None,
    current_exposures: Optional[Dict[int, int]] = None,
    num_lineups_total: int = 1,
    randomization_factor: float = 0.3,
    min_salary_floor: int = 0,
    stack_config: Optional[Dict[str, Dict]] = None
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

        max_exp = max_exposure.get(player.player_id, 1.0)
        current = current_exposures.get(player.player_id, 0)
        max_lineups = int(num_lineups_total * max_exp)

        return current < max_lineups

    available = [p for p in available if can_use_player(p)]

    if not available:
        return None

    # Get score function based on strategy
    def get_score(p: DFSPlayer) -> float:
        if strategy == "ceiling":
            return p.proj_ceiling
        elif strategy == "value":
            return p.fpts_per_dollar
        elif strategy == "leverage":
            return p.leverage_score
        else:  # projection
            return p.proj_fpts

    lineup = DFSLineup()
    used_player_ids = set()

    # Add locked players first
    for player in locked_players:
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
            return None
        return lineup

    return None


def generate_diversified_lineups(
    player_pool: List[DFSPlayer],
    num_lineups: int = 100,
    max_player_exposure: float = 0.50,  # 50% max exposure (was 40%)
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    excluded_games: Optional[Set[str]] = None,
    exposure_targets: Optional[Dict[int, float]] = None,
    min_salary_floor: int = 0,
    stack_config: Optional[Dict[str, Dict]] = None
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

    Returns:
        List of valid DFSLineup objects
    """
    lineups: List[DFSLineup] = []
    exposures: Dict[int, int] = {}
    existing_lineup_keys: Set[Tuple[int, ...]] = set()

    # Filter out excluded games
    excluded_games = excluded_games or set()
    filtered_pool = [p for p in player_pool if p.game_id not in excluded_games]

    # Filter out injured players (OUT/DOUBTFUL)
    injured_count = len([p for p in filtered_pool if p.is_injured])
    filtered_pool = [p for p in filtered_pool if not p.is_injured]

    if not filtered_pool:
        return []

    # Get locked players (ensure they're not injured)
    locked = [p for p in filtered_pool if p.is_locked and not p.is_injured]
    locked_ids = {p.player_id for p in locked}

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

    # Strategy configuration: (strategy, randomization_factor, target_count_pct)
    strategies = [
        ('projection', 0.25, 0.20),  # 20% — core projection-optimal
        ('ceiling', 0.30, 0.25),     # 25% — upside-focused
        ('value', 0.25, 0.15),       # 15% — value-focused
        ('leverage', 0.35, 0.15),    # 15% — low-own/high-ceiling contrarian
        ('projection', 0.60, 0.25),  # 25% — balanced high-randomization
    ]

    total_generated = 0
    max_total_attempts = num_lineups * 50  # Allow many more attempts
    total_attempts = 0

    for strategy, rand_factor, pct in strategies:
        target_count = int(num_lineups * pct)
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
                        attempt_locked.append(info['player'])

            # Determine stack config for this strategy
            # Ceiling lineups: full stacking (maximize correlation upside)
            # Projection lineups: downgrade primary->mini for balance
            # Leverage lineups: 50% chance of no stacking (contrarian)
            # Value lineups: use as configured
            # Balanced (high-rand projection): randomize stack types
            attempt_stack = stack_config
            if stack_config and strategy == 'leverage':
                if random.random() < 0.5:
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
                locked_players=attempt_locked,
                max_exposure=max_exp,
                current_exposures=exposures,
                num_lineups_total=num_lineups,
                randomization_factor=rand_factor,
                min_salary_floor=min_salary_floor,
                stack_config=attempt_stack
            )

            if lineup and lineup.is_valid:
                # Check lineup uniqueness
                lineup_key = tuple(sorted(p.player_id for p in lineup.players.values()))

                if lineup_key not in existing_lineup_keys:
                    lineups.append(lineup)
                    existing_lineup_keys.add(lineup_key)
                    strategy_generated += 1
                    total_generated += 1

                    # Update exposures
                    for player in lineup.players.values():
                        exposures[player.player_id] = exposures.get(player.player_id, 0) + 1

    if progress_callback:
        progress_callback(total_generated, num_lineups, "Complete!")

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
