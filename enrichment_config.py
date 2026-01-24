#!/usr/bin/env python3
"""
Enrichment Configuration Module

Centralizes all enrichment parameters with guardrails for safe tuning.
Parameters can be adjusted based on validation results while respecting
hard constraints that prevent the model from breaking.

Usage:
    from enrichment_config import ENRICHMENT_CONFIG, get_rest_multiplier_config

    # Get current B2B multiplier
    b2b_mult = ENRICHMENT_CONFIG['rest']['b2b_multiplier']

    # Apply guardrails to a new value
    safe_mult = apply_guardrail(0.88, 'rest', 'b2b_multiplier')
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
import json


# =============================================================================
# GUARDRAILS - Hard constraints that cannot be exceeded
# =============================================================================

GUARDRAILS = {
    'rest': {
        'b2b_multiplier': (0.85, 0.98),      # -15% to -2% (must be penalty)
        'rested_multiplier': (1.00, 1.10),   # 0% to +10% (never penalty)
        'rested_threshold_days': (2, 5),     # 2-5 days for "rested"
    },
    'game_script': {
        'blowout_spread_threshold': (8.0, 14.0),
        'close_spread_threshold': (1.0, 5.0),
        'star_blowout_minutes_adj': (-6.0, -2.0),
        'bench_blowout_minutes_adj': (2.0, 8.0),
        'star_close_minutes_adj': (0.5, 4.0),
        'minutes_adj_cap': (4.0, 8.0),
        'ppm_conversion': (0.45, 0.65),
    },
    'roles': {
        'star_min_minutes': (28.0, 34.0),
        'star_min_ppg': (16.0, 22.0),
        'starter_min_minutes': (22.0, 28.0),
        'starter_min_ppg': (8.0, 14.0),
        'rotation_min_minutes': (12.0, 18.0),
    },
    'position_defense': {
        'factor_min': (0.85, 0.95),
        'factor_max': (1.10, 1.20),
        'grade_a_factor': (0.90, 0.96),
        'grade_b_factor': (0.95, 0.99),
        'grade_c_factor': (0.98, 1.02),
        'grade_d_factor': (1.02, 1.08),
        'grade_f_factor': (1.07, 1.15),
    },
}


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

ENRICHMENT_CONFIG = {
    'rest': {
        # B2B penalty (0.92 = -8%)
        'b2b_multiplier': 0.92,
        # Well-rested boost (1.05 = +5%)
        'rested_multiplier': 1.05,
        # Normal rest multiplier (no adjustment)
        'normal_multiplier': 1.02,
        # Days threshold for "rested" status
        'rested_threshold_days': 3,
        # Enable/disable this enrichment
        'enabled': True,
    },

    'game_script': {
        # Spread thresholds for game classification
        'blowout_spread_threshold': 10.0,
        'close_spread_threshold': 3.0,
        'comfortable_spread_threshold': 5.0,

        # Minutes adjustments by role (blowout scenarios)
        'star_blowout_minutes_adj': -4.0,
        'starter_blowout_minutes_adj': -3.0,
        'rotation_blowout_minutes_adj': 4.0,
        'bench_blowout_minutes_adj': 5.0,

        # Minutes adjustments by role (close game scenarios)
        'star_close_minutes_adj': 2.0,
        'starter_close_minutes_adj': 1.5,
        'rotation_close_minutes_adj': -1.0,
        'bench_close_minutes_adj': -2.0,

        # Maximum minutes adjustment allowed (guardrail)
        'minutes_adj_cap': 6.0,

        # PPM conversion factor (points per minute)
        'ppm_conversion': 0.55,

        # Enable/disable this enrichment
        'enabled': True,
    },

    'roles': {
        # STAR thresholds
        'star_min_minutes': 30.0,
        'star_min_ppg': 18.0,
        'star_min_usage': 24.0,

        # STARTER thresholds
        'starter_min_minutes': 25.0,
        'starter_min_ppg': 10.0,

        # ROTATION thresholds
        'rotation_min_minutes': 15.0,
        'rotation_min_games_pct': 0.30,

        # Enable/disable this enrichment
        'enabled': True,
    },

    'position_defense': {
        # Grade-to-factor mapping
        'grade_factors': {
            'A': 0.93,   # Tough matchup: -7%
            'B': 0.97,   # Good defense: -3%
            'C': 1.00,   # Average: neutral
            'D': 1.05,   # Below average: +5%
            'F': 1.10,   # Poor defense: +10%
        },

        # Factor bounds (guardrails)
        'factor_min': 0.90,
        'factor_max': 1.15,

        # Minimum games for reliable position defense data
        'min_games_for_grade': 5,

        # Enable/disable this enrichment
        'enabled': True,
    },

    # Global settings
    'global': {
        # Maximum total adjustment from all enrichments
        'max_total_multiplier': 1.25,
        'min_total_multiplier': 0.75,

        # Logging verbosity
        'verbose': False,

        # Version for tracking config changes
        'version': '1.0.0',
        'last_updated': '2026-01-24',
    },
}


# =============================================================================
# PARAMETER ACCESS FUNCTIONS
# =============================================================================

def get_config(category: str, key: str) -> Any:
    """Get a configuration value."""
    return ENRICHMENT_CONFIG.get(category, {}).get(key)


def get_rest_config() -> Dict:
    """Get rest enrichment configuration."""
    return ENRICHMENT_CONFIG['rest']


def get_game_script_config() -> Dict:
    """Get game script enrichment configuration."""
    return ENRICHMENT_CONFIG['game_script']


def get_roles_config() -> Dict:
    """Get roles enrichment configuration."""
    return ENRICHMENT_CONFIG['roles']


def get_position_defense_config() -> Dict:
    """Get position defense enrichment configuration."""
    return ENRICHMENT_CONFIG['position_defense']


def is_enrichment_enabled(category: str) -> bool:
    """Check if an enrichment is enabled."""
    return ENRICHMENT_CONFIG.get(category, {}).get('enabled', True)


# =============================================================================
# GUARDRAIL ENFORCEMENT
# =============================================================================

def apply_guardrail(value: float, category: str, key: str) -> float:
    """
    Apply guardrails to ensure a value stays within safe bounds.

    Args:
        value: The proposed value
        category: Enrichment category (rest, game_script, etc.)
        key: Parameter key

    Returns:
        Value clamped to guardrail bounds
    """
    if category not in GUARDRAILS:
        return value

    if key not in GUARDRAILS[category]:
        return value

    min_val, max_val = GUARDRAILS[category][key]
    return max(min_val, min(max_val, value))


def validate_config() -> Dict[str, list]:
    """
    Validate current configuration against guardrails.

    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    issues = {'errors': [], 'warnings': []}

    for category, params in ENRICHMENT_CONFIG.items():
        if category == 'global':
            continue

        if category not in GUARDRAILS:
            continue

        for key, value in params.items():
            if key in ['enabled', 'grade_factors']:
                continue

            if key in GUARDRAILS[category]:
                min_val, max_val = GUARDRAILS[category][key]
                if value < min_val or value > max_val:
                    issues['errors'].append(
                        f"{category}.{key}={value} outside guardrails [{min_val}, {max_val}]"
                    )
                elif value == min_val or value == max_val:
                    issues['warnings'].append(
                        f"{category}.{key}={value} at guardrail boundary"
                    )

    return issues


# =============================================================================
# CONFIGURATION PERSISTENCE
# =============================================================================

def save_config(filepath: str = 'enrichment_config.json'):
    """Save current configuration to JSON file."""
    config_to_save = {
        'enrichment_config': ENRICHMENT_CONFIG,
        'guardrails': GUARDRAILS,
        'saved_at': datetime.now().isoformat(),
    }
    with open(filepath, 'w') as f:
        json.dump(config_to_save, f, indent=2)


def load_config(filepath: str = 'enrichment_config.json') -> bool:
    """
    Load configuration from JSON file.

    Returns:
        True if loaded successfully, False otherwise
    """
    global ENRICHMENT_CONFIG

    try:
        with open(filepath, 'r') as f:
            loaded = json.load(f)

        if 'enrichment_config' in loaded:
            # Validate before applying
            temp_config = loaded['enrichment_config']

            # Check version compatibility
            loaded_version = temp_config.get('global', {}).get('version', '0.0.0')
            current_version = ENRICHMENT_CONFIG['global']['version']

            if loaded_version != current_version:
                print(f"Warning: Config version mismatch ({loaded_version} vs {current_version})")

            ENRICHMENT_CONFIG.update(temp_config)
            return True

    except FileNotFoundError:
        return False
    except json.JSONDecodeError as e:
        print(f"Error loading config: {e}")
        return False

    return False


# =============================================================================
# PARAMETER TUNING HELPERS
# =============================================================================

@dataclass
class ParameterUpdate:
    """Represents a proposed parameter update."""
    category: str
    key: str
    current_value: float
    proposed_value: float
    reason: str
    validated: bool = False
    applied: bool = False


def propose_update(
    category: str,
    key: str,
    new_value: float,
    reason: str
) -> ParameterUpdate:
    """
    Propose a parameter update with guardrail validation.

    Args:
        category: Enrichment category
        key: Parameter key
        new_value: Proposed new value
        reason: Reason for the update

    Returns:
        ParameterUpdate object with validation status
    """
    current = get_config(category, key)
    safe_value = apply_guardrail(new_value, category, key)

    update = ParameterUpdate(
        category=category,
        key=key,
        current_value=current,
        proposed_value=safe_value,
        reason=reason,
        validated=(safe_value == new_value),  # True if no clamping needed
    )

    return update


def apply_update(update: ParameterUpdate) -> bool:
    """
    Apply a validated parameter update.

    Args:
        update: ParameterUpdate object

    Returns:
        True if applied successfully
    """
    if update.category not in ENRICHMENT_CONFIG:
        return False

    ENRICHMENT_CONFIG[update.category][update.key] = update.proposed_value
    update.applied = True

    # Log the change
    print(f"Applied update: {update.category}.{update.key} = {update.proposed_value} "
          f"(was {update.current_value})")

    return True


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enrichment configuration manager')
    parser.add_argument('--validate', action='store_true', help='Validate current config')
    parser.add_argument('--show', action='store_true', help='Show current config')
    parser.add_argument('--save', type=str, help='Save config to file')
    parser.add_argument('--load', type=str, help='Load config from file')
    args = parser.parse_args()

    if args.validate:
        issues = validate_config()
        if issues['errors']:
            print("ERRORS:")
            for e in issues['errors']:
                print(f"  - {e}")
        if issues['warnings']:
            print("WARNINGS:")
            for w in issues['warnings']:
                print(f"  - {w}")
        if not issues['errors'] and not issues['warnings']:
            print("Configuration is valid!")

    elif args.show:
        print(json.dumps(ENRICHMENT_CONFIG, indent=2))

    elif args.save:
        save_config(args.save)
        print(f"Configuration saved to {args.save}")

    elif args.load:
        if load_config(args.load):
            print(f"Configuration loaded from {args.load}")
        else:
            print(f"Failed to load configuration from {args.load}")

    else:
        # Default: show summary
        print("Enrichment Configuration Summary")
        print("=" * 50)
        for cat, params in ENRICHMENT_CONFIG.items():
            if cat == 'global':
                continue
            enabled = params.get('enabled', True)
            status = "ENABLED" if enabled else "DISABLED"
            print(f"\n{cat.upper()}: {status}")
            for k, v in params.items():
                if k != 'enabled':
                    print(f"  {k}: {v}")
