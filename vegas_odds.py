#!/usr/bin/env python3
"""
Vegas Odds Integration for Tournament Strategy

Fetches NBA betting lines from TheOddsAPI and computes game environment scores
for portfolio optimization.

Key concepts:
- Game environment score: pace, volatility, OT likelihood, blowout risk
- High total + tight spread = good for game stacks (positive correlation)
- Large spread = blowout risk (star minutes at risk)
"""

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GameOdds:
    """Betting lines for a single NBA game."""
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    spread: float  # Home team spread (negative = favorite)
    total: float   # Over/under line
    home_ml: int   # Home moneyline
    away_ml: int   # Away moneyline
    commence_time: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GameEnvironment:
    """Derived game environment metrics for portfolio optimization."""
    game_id: str
    pace_score: float           # total / league_avg (>1 = fast pace)
    volatility_multiplier: float # Multiplier for player sigma
    ot_probability: float       # Estimated OT likelihood
    blowout_risk: float         # 0-1 scale, 1 = high blowout risk
    stack_score: float          # 0-1 scale, 1 = ideal for stacking
    home_team: str
    away_team: str
    spread: float
    total: float


# ============================================================================
# Constants
# ============================================================================

# League averages for 2024-25 season (update annually)
LEAGUE_AVG_TOTAL = 228.0  # Typical Vegas total
LEAGUE_AVG_PACE = 99.5    # Possessions per 48 min

# OT probability estimates based on historical data
BASE_OT_PROB = 0.06       # ~6% of NBA games go to OT
OT_TIGHT_SPREAD_BOOST = 0.03  # +3% if spread <= 3.5
OT_HIGH_TOTAL_BOOST = 0.02    # +2% if tight spread AND high total

# TheOddsAPI configuration
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"
SPORT_KEY = "basketball_nba"


# ============================================================================
# Team Name Mappings (TheOddsAPI to NBA API)
# ============================================================================

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


# ============================================================================
# VegasOddsClient
# ============================================================================

class VegasOddsClient:
    """
    Client for fetching NBA betting odds from TheOddsAPI.

    API docs: https://the-odds-api.com/liveapi/guides/v4/

    Required environment variable:
        ODDS_API_KEY: Your TheOddsAPI key
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.

        Args:
            api_key: TheOddsAPI key. If not provided, reads from ODDS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TheOddsAPI key required. Set ODDS_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.base_url = ODDS_API_BASE_URL
        self.remaining_requests = None  # Tracked from API response headers

    def fetch_nba_odds(self, date: Optional[str] = None) -> List[GameOdds]:
        """
        Fetch NBA odds for today's games (or specified date).

        Args:
            date: Optional date string (YYYY-MM-DD). Defaults to today.

        Returns:
            List of GameOdds objects for each game.

        Raises:
            requests.RequestException: On API failure
        """
        url = f"{self.base_url}/{SPORT_KEY}/odds"

        params = {
            "apiKey": self.api_key,
            "regions": "us",  # US bookmakers
            "markets": "spreads,totals,h2h",  # spread, over/under, moneyline
            "oddsFormat": "american",
        }

        response = requests.get(url, params=params, timeout=30)

        # Track remaining API requests
        self.remaining_requests = response.headers.get("x-requests-remaining")

        response.raise_for_status()
        data = response.json()

        # Filter to target date if specified
        target_date = date or datetime.now().strftime("%Y-%m-%d")

        games = []
        for game in data:
            # Parse commence time to check date
            commence_time = game.get("commence_time", "")
            if commence_time:
                game_date = commence_time[:10]  # Extract YYYY-MM-DD
                if date and game_date != target_date:
                    continue

            # Extract odds from bookmakers (use first available)
            spread, total, home_ml, away_ml = self._extract_consensus_odds(game)

            if spread is None or total is None:
                continue  # Skip games without complete odds

            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")

            # Convert to abbreviations
            home_abbr = TEAM_NAME_TO_ABBR.get(home_team, home_team[:3].upper())
            away_abbr = TEAM_NAME_TO_ABBR.get(away_team, away_team[:3].upper())

            game_id = f"{game_date}_{away_abbr}_{home_abbr}"

            games.append(GameOdds(
                game_id=game_id,
                game_date=game_date,
                home_team=home_abbr,
                away_team=away_abbr,
                spread=spread,
                total=total,
                home_ml=home_ml or 0,
                away_ml=away_ml or 0,
                commence_time=commence_time,
            ))

        return games

    def _extract_consensus_odds(self, game: dict) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
        """
        Extract consensus odds from bookmaker data.

        Uses FanDuel as primary, falls back to first available.

        Returns:
            (spread, total, home_ml, away_ml) tuple
        """
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            return None, None, None, None

        # Prefer FanDuel, then DraftKings, then first available
        preferred_order = ["fanduel", "draftkings", "betmgm", "caesars"]

        selected_book = None
        for pref in preferred_order:
            for book in bookmakers:
                if book.get("key") == pref:
                    selected_book = book
                    break
            if selected_book:
                break

        if not selected_book:
            selected_book = bookmakers[0]

        spread = None
        total = None
        home_ml = None
        away_ml = None
        home_team = game.get("home_team", "")

        for market in selected_book.get("markets", []):
            market_key = market.get("key")
            outcomes = market.get("outcomes", [])

            if market_key == "spreads":
                # Find home team spread
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        spread = outcome.get("point")
                        break

            elif market_key == "totals":
                # Get over/under line
                for outcome in outcomes:
                    if outcome.get("name") == "Over":
                        total = outcome.get("point")
                        break

            elif market_key == "h2h":
                # Moneylines
                for outcome in outcomes:
                    if outcome.get("name") == home_team:
                        home_ml = outcome.get("price")
                    else:
                        away_ml = outcome.get("price")

        return spread, total, home_ml, away_ml

    def get_remaining_requests(self) -> Optional[str]:
        """Return remaining API requests from last call."""
        return self.remaining_requests


# ============================================================================
# Game Environment Scoring
# ============================================================================

def compute_game_environment(odds: GameOdds) -> GameEnvironment:
    """
    Compute game environment metrics from betting lines.

    Key outputs:
    - pace_score: >1 means faster than average game
    - ot_probability: Higher for tight games
    - blowout_risk: Higher for large spreads
    - stack_score: Higher for stackable games (high total + tight spread)

    Args:
        odds: GameOdds object with spread/total

    Returns:
        GameEnvironment with derived metrics
    """
    # Pace score: normalized by league average total
    pace_score = odds.total / LEAGUE_AVG_TOTAL if odds.total > 0 else 1.0

    # Spread analysis
    spread_tight = abs(odds.spread) <= 3.5
    spread_moderate = abs(odds.spread) <= 6.0
    spread_large = abs(odds.spread) > 10.0

    # High total threshold
    high_total = odds.total >= 230
    very_high_total = odds.total >= 235

    # OT probability estimation
    ot_prob = BASE_OT_PROB
    if spread_tight:
        ot_prob += OT_TIGHT_SPREAD_BOOST
    if spread_tight and high_total:
        ot_prob += OT_HIGH_TOTAL_BOOST

    # Blowout risk (affects star minutes)
    if spread_large:
        blowout_risk = 0.8 + (abs(odds.spread) - 10) * 0.02  # Scale up
        blowout_risk = min(1.0, blowout_risk)
    elif abs(odds.spread) > 7:
        blowout_risk = 0.4
    else:
        blowout_risk = 0.1

    # Volatility multiplier (increase sigma for uncertain games)
    volatility_mult = 1.0
    if spread_tight:
        volatility_mult += 0.10  # More variance in close games
    if very_high_total:
        volatility_mult += 0.05  # More variance in shootouts

    # Stack score: ideal for same-game stacking
    # UPDATED: Backtest shows spread matters more than total for top scorers
    # Prioritize close games where stars stay in late
    # Total is now a secondary factor, not primary
    stack_score = 0.0
    if spread_tight:
        # Close games are the primary stacking signal
        stack_score = 0.85 if high_total else 0.75
    elif spread_moderate:
        # Moderate spreads still okay, total adds small boost
        stack_score = 0.60 if high_total else 0.50
    elif very_high_total and not (spread >= 10):
        # High total but not a blowout - moderate stack value
        stack_score = 0.45
    else:
        # Blowouts or low-total neutral games - avoid stacking
        stack_score = 0.25

    return GameEnvironment(
        game_id=odds.game_id,
        pace_score=round(pace_score, 3),
        volatility_multiplier=round(volatility_mult, 3),
        ot_probability=round(ot_prob, 3),
        blowout_risk=round(blowout_risk, 3),
        stack_score=round(stack_score, 3),
        home_team=odds.home_team,
        away_team=odds.away_team,
        spread=odds.spread,
        total=odds.total,
    )


def identify_stackable_games(game_envs: Dict[str, GameEnvironment],
                              min_stack_score: float = 0.7) -> List[str]:
    """
    Identify games worth stacking (2 players from same game).

    Args:
        game_envs: Dict of game_id -> GameEnvironment
        min_stack_score: Minimum stack_score threshold

    Returns:
        List of game_ids suitable for stacking
    """
    stackable = []
    for game_id, env in game_envs.items():
        if env.stack_score >= min_stack_score:
            stackable.append(game_id)
    return sorted(stackable, key=lambda g: game_envs[g].stack_score, reverse=True)


# ============================================================================
# Database Functions
# ============================================================================

def ensure_game_odds_table(conn: sqlite3.Connection) -> None:
    """Create game_odds table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_odds (
            game_id TEXT PRIMARY KEY,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            spread REAL,
            total REAL,
            home_ml INTEGER,
            away_ml INTEGER,
            commence_time TEXT,
            fetched_at TEXT,
            -- Derived environment metrics
            pace_score REAL,
            volatility_multiplier REAL,
            ot_probability REAL,
            blowout_risk REAL,
            stack_score REAL,
            UNIQUE(game_date, home_team, away_team)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_game_odds_date
        ON game_odds(game_date)
    """)
    conn.commit()


def store_game_odds(conn: sqlite3.Connection, odds: GameOdds, env: GameEnvironment) -> None:
    """Store game odds and environment metrics in database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO game_odds (
            game_id, game_date, home_team, away_team,
            spread, total, home_ml, away_ml, commence_time, fetched_at,
            pace_score, volatility_multiplier, ot_probability, blowout_risk, stack_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        odds.game_id, odds.game_date, odds.home_team, odds.away_team,
        odds.spread, odds.total, odds.home_ml, odds.away_ml,
        odds.commence_time, odds.fetched_at,
        env.pace_score, env.volatility_multiplier, env.ot_probability,
        env.blowout_risk, env.stack_score
    ))
    conn.commit()


def load_game_odds(conn: sqlite3.Connection, game_date: str) -> Dict[str, GameEnvironment]:
    """
    Load game odds and environment for a date.

    Returns:
        Dict of game_id -> GameEnvironment
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT game_id, home_team, away_team, spread, total,
               pace_score, volatility_multiplier, ot_probability,
               blowout_risk, stack_score
        FROM game_odds
        WHERE game_date = ?
    """, (game_date,))

    result = {}
    for row in cursor.fetchall():
        game_id, home, away, spread, total, pace, vol, ot, blow, stack = row
        result[game_id] = GameEnvironment(
            game_id=game_id,
            pace_score=pace or 1.0,
            volatility_multiplier=vol or 1.0,
            ot_probability=ot or 0.06,
            blowout_risk=blow or 0.1,
            stack_score=stack or 0.3,
            home_team=home,
            away_team=away,
            spread=spread or 0,
            total=total or 228,
        )
    return result


def fetch_and_store_odds(conn: sqlite3.Connection,
                         api_key: Optional[str] = None,
                         date: Optional[str] = None) -> List[GameEnvironment]:
    """
    Fetch odds from API, compute environments, and store in database.

    Args:
        conn: Database connection
        api_key: TheOddsAPI key (or use env var)
        date: Target date (YYYY-MM-DD) or None for today

    Returns:
        List of GameEnvironment objects for all games
    """
    ensure_game_odds_table(conn)

    client = VegasOddsClient(api_key)
    odds_list = client.fetch_nba_odds(date)

    environments = []
    for odds in odds_list:
        env = compute_game_environment(odds)
        store_game_odds(conn, odds, env)
        environments.append(env)

    print(f"Fetched and stored odds for {len(environments)} games")
    print(f"API requests remaining: {client.get_remaining_requests()}")

    return environments


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and display NBA Vegas odds")
    parser.add_argument("--date", help="Date to fetch (YYYY-MM-DD), default today")
    parser.add_argument("--db", default="nba_stats.db", help="Database path")
    parser.add_argument("--api-key", help="TheOddsAPI key (or set ODDS_API_KEY env)")
    parser.add_argument("--store", action="store_true", help="Store to database")
    args = parser.parse_args()

    try:
        client = VegasOddsClient(args.api_key)
        odds_list = client.fetch_nba_odds(args.date)

        print(f"\n{'='*70}")
        print(f"NBA VEGAS ODDS - {args.date or 'Today'}")
        print(f"{'='*70}\n")

        for odds in odds_list:
            env = compute_game_environment(odds)
            print(f"{odds.away_team} @ {odds.home_team}")
            print(f"  Spread: {odds.spread:+.1f} | Total: {odds.total:.1f}")
            print(f"  Pace: {env.pace_score:.2f} | Stack: {env.stack_score:.2f} | "
                  f"OT: {env.ot_probability:.1%} | Blowout: {env.blowout_risk:.1%}")
            print()

        print(f"API requests remaining: {client.get_remaining_requests()}")

        if args.store:
            conn = sqlite3.connect(args.db)
            ensure_game_odds_table(conn)
            for odds in odds_list:
                env = compute_game_environment(odds)
                store_game_odds(conn, odds, env)
            conn.close()
            print(f"\nStored {len(odds_list)} games to {args.db}")

    except Exception as e:
        print(f"Error: {e}")
        raise
