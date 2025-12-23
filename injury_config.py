#!/usr/bin/env python3
"""
Configuration for NBA Injury Automation System.

This module defines all settings for automated injury detection, including:
- Play probability model for injury statuses
- Exclusion and adjustment logic
- Team name mappings
- Feature flags and thresholds
"""

from typing import Dict

# ==============================================================================
# PLAY PROBABILITY MODEL - Single Source of Truth
# ==============================================================================
# Represents likelihood a player will actually play (0.0 = definitely out, 1.0 = definitely playing)
# This drives both prediction exclusion and teammate adjustment calculations
STATUS_PLAY_PROBABILITY: Dict[str, float] = {
    'out': 0.0,         # 0% - will not play
    'doubtful': 0.25,   # 25% - unlikely to play (historically plays ~25% of time)
    'questionable': 0.5, # 50% - 50/50 chance (coin flip)
    'probable': 0.8,    # 80% - likely to play (historically plays ~80% of time)
    'day-to-day': 0.6   # 60% - uncertain (better than questionable, worse than probable)
}

# ==============================================================================
# EXCLUSION LOGIC - Derived from Play Probability
# ==============================================================================
# Players with play probability BELOW this threshold are excluded from predictions entirely
# With default 0.3: excludes 'out' (0.0) and 'doubtful' (0.25)
# Players ABOVE threshold: still included in predictions but teammates get partial boost
PREDICTION_EXCLUSION_THRESHOLD: float = 0.3

# ==============================================================================
# ADJUSTMENT MULTIPLIERS - Derived from Play Probability
# ==============================================================================
def get_adjustment_multiplier(status: str) -> float:
    """
    Calculate teammate adjustment multiplier based on injury status.

    Logic:
    - If play_prob < PREDICTION_EXCLUSION_THRESHOLD:
      Player is excluded from predictions → teammates get FULL adjustment (1.0)
    - If play_prob >= PREDICTION_EXCLUSION_THRESHOLD:
      Player still in predictions → teammates get PARTIAL adjustment (1.0 - play_prob)

    Example:
    - OUT (0.0 play prob): Excluded → 100% adjustment
    - DOUBTFUL (0.25 play prob): Excluded → 100% adjustment
    - QUESTIONABLE (0.5 play prob): Included → 50% adjustment (teammate gets half the OUT boost)
    - PROBABLE (0.8 play prob): Included → 20% adjustment (teammate gets minimal boost)

    Rationale:
    If a player is 50% likely to be out, teammates should get 50% of the
    adjustment they'd get if the player was definitely out.

    Args:
        status: Injury status ('out', 'doubtful', 'questionable', 'probable', 'day-to-day')

    Returns:
        Adjustment multiplier (0.0 to 1.0)
    """
    play_prob = STATUS_PLAY_PROBABILITY.get(status, 1.0)

    if play_prob < PREDICTION_EXCLUSION_THRESHOLD:
        return 1.0  # Full adjustment (player excluded from predictions)

    return 1.0 - play_prob  # Partial adjustment (player included but risky)

# ==============================================================================
# TRIGGER ADJUSTMENTS - Which Statuses Affect Teammates
# ==============================================================================
# Only these statuses will trigger teammate boost calculations
# Note: 'probable' is intentionally EXCLUDED because:
#   - 20% adjustment is negligible (adds noise without meaningful signal)
#   - Historically, probable players play ~80% of time (minimal impact)
#   - Including it degrades model performance
TRIGGER_ADJUSTMENTS = ['out', 'doubtful', 'questionable', 'day-to-day']

# ==============================================================================
# CONFIDENCE THRESHOLDS
# ==============================================================================
# Minimum confidence score (0.0-1.0) required for automated actions
# Confidence comes from fuzzy matching quality when mapping player names to IDs
MIN_CONFIDENCE_FOR_EXCLUSION: float = 0.85  # Only exclude players if match confidence >= 85%

# Fuzzy matching threshold for player name matching (0-100 scale)
FUZZY_MATCH_THRESHOLD: int = 85  # Names must be 85% similar to be considered a match

# ==============================================================================
# TEAM NAME MAPPINGS
# ==============================================================================
# Maps full team names (from nbainjuries API) to 3-letter abbreviations (NBA API format)
TEAM_NAME_MAP: Dict[str, str] = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

# ==============================================================================
# FEATURE FLAGS & SETTINGS
# ==============================================================================
# Master kill switch - set to False to disable all injury automation
INJURY_AUTOMATION_ENABLED: bool = True

# Auto-fetch cooldown period (minutes) - prevents race conditions and excessive API calls
AUTO_FETCH_COOLDOWN_MINUTES: int = 15

# Cache raw API responses for replay/debugging
# When enabled, stores raw nbainjuries API responses to database for troubleshooting
CACHE_RAW_RESPONSES: bool = True

# Manual override window (hours) - how long manual entries take precedence over automated updates
# If user manually sets a player's status, automated updates won't override it for this many hours
MANUAL_OVERRIDE_WINDOW_HOURS: int = 24

# ==============================================================================
# API RETRY SETTINGS
# ==============================================================================
# Number of retry attempts for API calls
API_RETRY_COUNT: int = 3

# Delay between retries (seconds) - uses exponential backoff
# Retry 1: 2s, Retry 2: 4s, Retry 3: 8s
API_RETRY_BASE_DELAY: int = 2

# ==============================================================================
# STATUS MAPPINGS
# ==============================================================================
# Maps balldontlie.io API status values to our internal schema
# API may return slightly different status strings than our schema
BALLDONTLIE_STATUS_MAP: Dict[str, str] = {
    'Out': 'out',
    'Doubtful': 'doubtful',
    'Questionable': 'questionable',
    'Probable': 'probable',
    'Available': 'returned',
    'Day-To-Day': 'day-to-day',
    'GTD': 'questionable',  # Game Time Decision → treat as questionable
}

# ==============================================================================
# VALIDATION
# ==============================================================================
# Validate configuration on import
def _validate_config() -> None:
    """Validate configuration settings are consistent."""
    # Ensure all statuses in TRIGGER_ADJUSTMENTS have play probabilities
    for status in TRIGGER_ADJUSTMENTS:
        if status not in STATUS_PLAY_PROBABILITY:
            raise ValueError(f"Status '{status}' in TRIGGER_ADJUSTMENTS missing from STATUS_PLAY_PROBABILITY")

    # Ensure threshold is between 0 and 1
    if not (0 <= PREDICTION_EXCLUSION_THRESHOLD <= 1):
        raise ValueError(f"PREDICTION_EXCLUSION_THRESHOLD must be between 0 and 1, got {PREDICTION_EXCLUSION_THRESHOLD}")

    # Ensure fuzzy match threshold is between 0 and 100
    if not (0 <= FUZZY_MATCH_THRESHOLD <= 100):
        raise ValueError(f"FUZZY_MATCH_THRESHOLD must be between 0 and 100, got {FUZZY_MATCH_THRESHOLD}")

    # Ensure confidence threshold is between 0 and 1
    if not (0 <= MIN_CONFIDENCE_FOR_EXCLUSION <= 1):
        raise ValueError(f"MIN_CONFIDENCE_FOR_EXCLUSION must be between 0 and 1, got {MIN_CONFIDENCE_FOR_EXCLUSION}")

# Run validation on import
_validate_config()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def should_exclude_from_predictions(status: str, confidence: float = 1.0) -> bool:
    """
    Determine if a player should be excluded from predictions.

    Args:
        status: Injury status
        confidence: Match confidence (0.0-1.0)

    Returns:
        True if player should be excluded, False otherwise
    """
    if confidence < MIN_CONFIDENCE_FOR_EXCLUSION:
        # Low confidence match - don't auto-exclude (flag for manual review instead)
        return False

    play_prob = STATUS_PLAY_PROBABILITY.get(status, 1.0)
    return play_prob < PREDICTION_EXCLUSION_THRESHOLD

def get_status_display_emoji(status: str) -> str:
    """
    Get emoji for injury status display in UI.

    Args:
        status: Injury status

    Returns:
        Emoji string
    """
    emoji_map = {
        'out': '🔴',
        'doubtful': '🟠',
        'questionable': '🟡',
        'probable': '🟢',
        'day-to-day': '🔵',
        'returned': '✅'
    }
    return emoji_map.get(status, '⚪')
