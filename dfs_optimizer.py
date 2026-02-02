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

    # DraftKings ID for export
    dk_id: str = ""

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

        # Leverage score: ceiling upside relative to expected ownership
        if self.ownership_proj > 0:
            self.leverage_score = self.proj_ceiling / self.ownership_proj

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


def generate_player_projections(
    conn: sqlite3.Connection,
    player: DFSPlayer,
    season: str = "2025-26",
    season_type: str = "Regular Season"
) -> DFSPlayer:
    """
    Generate projections for a player based on historical data.

    Weighting:
    - 40% Season average
    - 40% Recent form (last 5 games)
    - 20% Matchup history (vs opponent, if available)
    """
    logs = load_player_historical_stats(conn, player.player_id, season, season_type)

    if logs.empty:
        # No data - use conservative defaults
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
        # Use weighted average: 40% season, 40% recent, 20% matchup
        weights = (0.4, 0.4, 0.2)
        sources = (season_avg, recent_avg, matchup_avg)
    else:
        # Not enough matchup data - use 50% season, 50% recent
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
        # Not enough data - use simple estimates
        avg_fpts = season_avg.get('dk_fpts', 0)
        player.proj_floor = avg_fpts * 0.6
        player.proj_ceiling = avg_fpts * 1.5

    # Calculate derived values
    player.calculate_projections()

    # Estimate ownership based on salary and projection
    # Higher salary + higher projection = higher ownership
    if player.salary > 0:
        value_score = player.fpts_per_dollar
        # Simple ownership model: scale value score to 0-30% range
        player.ownership_proj = min(30.0, max(1.0, value_score * 3))

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

        if player_id is None:
            unmatched.append(name)
            continue

        matched_names.append((name, name_to_id.get(normalize_name(name), {}).get('original_name', 'Unknown')))

        player = DFSPlayer(
            player_id=player_id,
            name=name,
            team=team,
            opponent=opponent,
            game_id=game_id,
            positions=positions,
            salary=int(row.get('salary', 0)),
            dk_id=str(row.get('id', ''))
        )

        players.append(player)

    metadata = {
        'total_players': len(df),
        'matched_players': len(players),
        'unmatched_players': unmatched,
        'unique_games': len(set(p.game_id for p in players)),
        'salary_range': (
            min(p.salary for p in players) if players else 0,
            max(p.salary for p in players) if players else 0
        ),
        'match_rate': len(players) / len(df) * 100 if len(df) > 0 else 0,
    }

    return players, metadata


# =============================================================================
# Lineup Optimizer
# =============================================================================

def optimize_lineup(
    player_pool: List[DFSPlayer],
    strategy: str = "projection",  # projection, ceiling, value
    locked_players: Optional[List[DFSPlayer]] = None,
    excluded_ids: Optional[Set[int]] = None,
    max_exposure: Optional[Dict[int, float]] = None,
    current_exposures: Optional[Dict[int, int]] = None,
    num_lineups_total: int = 1
) -> Optional[DFSLineup]:
    """
    Build an optimized lineup using a greedy algorithm.

    Args:
        player_pool: Available players
        strategy: Optimization target (projection, ceiling, value)
        locked_players: Players that must be in lineup
        excluded_ids: Player IDs to exclude
        max_exposure: Maximum exposure % per player
        current_exposures: Current lineup count per player
        num_lineups_total: Total lineups being generated (for exposure calc)

    Returns:
        Optimized DFSLineup or None if no valid lineup found
    """
    excluded_ids = excluded_ids or set()
    locked_players = locked_players or []
    max_exposure = max_exposure or {}
    current_exposures = current_exposures or {}

    # Filter available players
    available = [
        p for p in player_pool
        if p.player_id not in excluded_ids
        and not p.is_excluded
    ]

    # Check exposure limits
    def can_use_player(player: DFSPlayer) -> bool:
        if player.player_id in excluded_ids:
            return False
        if player.is_excluded:
            return False

        max_exp = max_exposure.get(player.player_id, 1.0)
        current = current_exposures.get(player.player_id, 0)
        max_lineups = int(num_lineups_total * max_exp)

        return current < max_lineups

    available = [p for p in available if can_use_player(p)]

    # Sort by strategy metric
    if strategy == "ceiling":
        available.sort(key=lambda p: p.proj_ceiling, reverse=True)
    elif strategy == "value":
        available.sort(key=lambda p: p.fpts_per_dollar, reverse=True)
    elif strategy == "leverage":
        available.sort(key=lambda p: p.leverage_score, reverse=True)
    else:  # projection
        available.sort(key=lambda p: p.proj_fpts, reverse=True)

    lineup = DFSLineup()
    used_player_ids = set()

    # Add locked players first
    for player in locked_players:
        for slot in ROSTER_SLOTS:
            if slot not in lineup.players and player.can_fill_slot(slot):
                lineup.players[slot] = player
                used_player_ids.add(player.player_id)
                break

    # Fill remaining slots greedily
    for slot in ROSTER_SLOTS:
        if slot in lineup.players:
            continue

        # Find best available player for this slot
        for player in available:
            if player.player_id in used_player_ids:
                continue
            if not player.can_fill_slot(slot):
                continue

            # Check salary constraint
            tentative_salary = lineup.total_salary + player.salary
            remaining_slots = sum(1 for s in ROSTER_SLOTS if s not in lineup.players and s != slot)

            # Need to leave room for minimum salary players
            min_salary_needed = remaining_slots * 3000  # Assume min ~$3000 per player

            if tentative_salary + min_salary_needed > SALARY_CAP:
                continue

            # Check game diversity (need at least 2 games)
            current_games = lineup.unique_games
            slots_remaining_after = remaining_slots

            if slots_remaining_after == 0:
                # This is the last player - ensure we have 2+ games
                test_games = current_games | {player.game_id}
                if len(test_games) < MIN_GAMES:
                    continue

            # Add player
            lineup.players[slot] = player
            used_player_ids.add(player.player_id)
            break

    # Validate final lineup
    if lineup.is_valid:
        return lineup

    return None


def generate_diversified_lineups(
    player_pool: List[DFSPlayer],
    num_lineups: int = 100,
    max_player_exposure: float = 0.40,  # 40% max exposure
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[DFSLineup]:
    """
    Generate diversified GPP tournament lineups.

    Strategy buckets:
    - Max Projection (20%): Highest expected value
    - Ceiling Plays (30%): Maximize upside
    - Value Plays (25%): Best points per dollar
    - Contrarian (15%): Low ownership + high ceiling
    - Random Mix (10%): Diversification

    Args:
        player_pool: Available players with projections
        num_lineups: Target number of lineups
        max_player_exposure: Maximum exposure per player (0.0-1.0)
        progress_callback: Optional progress callback (current, total, message)

    Returns:
        List of valid DFSLineup objects
    """
    lineups = []
    exposures: Dict[int, int] = {}  # player_id -> lineup count

    # Calculate bucket sizes
    buckets = {
        'projection': int(num_lineups * 0.20),
        'ceiling': int(num_lineups * 0.30),
        'value': int(num_lineups * 0.25),
        'leverage': int(num_lineups * 0.15),
        'random': num_lineups - int(num_lineups * 0.90)  # Remainder
    }

    # Get locked players
    locked = [p for p in player_pool if p.is_locked]

    # Build max exposure dict
    max_exp = {p.player_id: max_player_exposure for p in player_pool}

    lineup_count = 0
    attempts = 0
    max_attempts = num_lineups * 10  # Limit total attempts

    for strategy, count in buckets.items():
        strategy_lineups = 0
        strategy_attempts = 0
        max_strategy_attempts = count * 5

        while strategy_lineups < count and attempts < max_attempts and strategy_attempts < max_strategy_attempts:
            attempts += 1
            strategy_attempts += 1

            if progress_callback:
                progress_callback(lineup_count, num_lineups, f"Generating {strategy} lineups...")

            # For random strategy, shuffle the pool
            if strategy == 'random':
                pool = player_pool.copy()
                random.shuffle(pool)
            else:
                pool = player_pool

            lineup = optimize_lineup(
                player_pool=pool,
                strategy=strategy if strategy != 'random' else 'projection',
                locked_players=locked,
                max_exposure=max_exp,
                current_exposures=exposures,
                num_lineups_total=num_lineups
            )

            if lineup and lineup.is_valid:
                # Check lineup uniqueness
                lineup_key = tuple(sorted(p.player_id for p in lineup.players.values()))
                existing_keys = {
                    tuple(sorted(p.player_id for p in l.players.values()))
                    for l in lineups
                }

                if lineup_key not in existing_keys:
                    lineups.append(lineup)
                    strategy_lineups += 1
                    lineup_count += 1

                    # Update exposures
                    for player in lineup.players.values():
                        exposures[player.player_id] = exposures.get(player.player_id, 0) + 1

    if progress_callback:
        progress_callback(len(lineups), num_lineups, "Complete!")

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
