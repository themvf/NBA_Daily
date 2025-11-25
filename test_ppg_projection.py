#!/usr/bin/env python3
"""
Test script for smart PPG projection algorithm.
Tests various scenarios to ensure projections are reasonable and confidence scoring works.
"""

import sys


def calculate_smart_ppg_projection(
    season_avg: float,
    recent_avg_5: float | None,
    recent_avg_3: float | None,
    vs_opp_team_avg: float | None,
    vs_opp_team_games: int,
    vs_defense_style_avg: float | None,
    vs_defense_style_games: int,
    opp_def_rating: float | None,
    opp_pace: float | None,
    league_avg_def_rating: float = 112.0,
    league_avg_pace: float = 99.0,
) -> tuple[float, float, float, float, dict]:
    """
    Calculate smart PPG projection using multi-factor weighted model.
    (Copied from streamlit_app.py for testing)
    """
    components = {}
    weights = {}

    # 1. Season Average (25% weight) - Always available baseline
    components["season"] = season_avg
    weights["season"] = 0.25

    # 2. Recent Form (20% weight) - Last 3 games weighted more than last 5
    if recent_avg_3 is not None and recent_avg_5 is not None:
        components["recent"] = (recent_avg_3 * 0.6) + (recent_avg_5 * 0.4)
        weights["recent"] = 0.20
    elif recent_avg_5 is not None:
        components["recent"] = recent_avg_5
        weights["recent"] = 0.15
    elif recent_avg_3 is not None:
        components["recent"] = recent_avg_3
        weights["recent"] = 0.15

    # 3. Team-Specific Matchup (30% weight - HIGHEST when available)
    if vs_opp_team_avg is not None and vs_opp_team_games >= 2:
        confidence_factor = min(1.0, vs_opp_team_games / 5.0)
        components["vs_team"] = vs_opp_team_avg
        weights["vs_team"] = 0.30 * confidence_factor
    else:
        # 4. Defense Style Matchup (15% weight - fallback when no team history)
        if vs_defense_style_avg is not None and vs_defense_style_games >= 3:
            confidence_factor = min(1.0, vs_defense_style_games / 8.0)
            components["vs_style"] = vs_defense_style_avg
            weights["vs_style"] = 0.15 * confidence_factor

    # Normalize weights so they sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate base projection
    projection = sum(components[k] * weights[k] for k in components if components[k] is not None)

    # 5. Defense Quality Adjustment (10% adjustment)
    if opp_def_rating is not None and league_avg_def_rating > 0:
        def_adjustment = 1.0 - ((opp_def_rating - league_avg_def_rating) / league_avg_def_rating) * 0.10
        projection *= def_adjustment
    else:
        def_adjustment = 1.0

    # 6. Pace Adjustment (5% adjustment)
    if opp_pace is not None and league_avg_pace > 0:
        pace_adjustment = 1.0 + ((opp_pace - league_avg_pace) / league_avg_pace) * 0.05
        projection *= pace_adjustment
    else:
        pace_adjustment = 1.0

    # Calculate confidence score (0-95%)
    base_confidence = 0.30
    matchup_confidence = 0.0
    form_confidence = 0.0

    if vs_opp_team_avg is not None and vs_opp_team_games >= 2:
        matchup_confidence = min(0.40, vs_opp_team_games * 0.08)
    elif vs_defense_style_avg is not None and vs_defense_style_games >= 3:
        matchup_confidence = min(0.25, vs_defense_style_games * 0.03)

    if recent_avg_3 is not None and recent_avg_5 is not None:
        form_confidence = 0.20
    elif recent_avg_5 is not None or recent_avg_3 is not None:
        form_confidence = 0.10

    confidence_score = min(0.95, base_confidence + matchup_confidence + form_confidence)

    # Calculate floor and ceiling (confidence intervals)
    interval_width = season_avg * 0.30 * (1 - confidence_score)
    floor = max(0, projection - interval_width)
    ceiling = projection + interval_width

    # Build breakdown for transparency
    breakdown = {
        "components": components,
        "weights": weights,
        "def_adjustment": def_adjustment,
        "pace_adjustment": pace_adjustment,
        "confidence_factors": {
            "base": base_confidence,
            "matchup": matchup_confidence,
            "form": form_confidence,
        }
    }

    return projection, confidence_score, floor, ceiling, breakdown


def test_projection_scenarios():
    """Test various projection scenarios"""
    print("=" * 70)
    print("Smart PPG Projection - Test Suite")
    print("=" * 70)

    # Test Case 1: Star player with team history
    print("\nTest 1: Star Player with Strong Team History")
    print("-" * 70)
    projection, confidence, floor, ceiling, breakdown = calculate_smart_ppg_projection(
        season_avg=28.5,
        recent_avg_5=30.2,
        recent_avg_3=31.5,
        vs_opp_team_avg=32.0,  # Excels vs this team
        vs_opp_team_games=5,
        vs_defense_style_avg=None,
        vs_defense_style_games=0,
        opp_def_rating=118.0,  # Weak defense
        opp_pace=102.0,  # Fast pace
    )
    print(f"  Season Avg: 28.5 PPG")
    print(f"  Recent Form: L3=31.5, L5=30.2")
    print(f"  Vs This Team: 32.0 PPG (5 games)")
    print(f"  Opponent: 118.0 Def Rating (weak), 102.0 Pace (fast)")
    print(f"  Projection: {projection:.1f} PPG")
    print(f"  Range: {floor:.1f}-{ceiling:.1f} PPG")
    print(f"  Confidence: {confidence:.0%}")
    print(f"  Components: {breakdown['components']}")
    print(f"  Weights: {breakdown['weights']}")
    assert 30.0 < projection < 35.0, f"Expected projection 30-35, got {projection:.1f}"
    assert confidence > 0.75, f"Expected high confidence, got {confidence:.0%}"

    # Test Case 2: Player struggling vs opponent
    print("\nTest 2: Player Struggles vs This Team")
    print("-" * 70)
    projection, confidence, floor, ceiling, breakdown = calculate_smart_ppg_projection(
        season_avg=25.0,
        recent_avg_5=26.0,
        recent_avg_3=27.0,
        vs_opp_team_avg=18.0,  # Struggles vs this team
        vs_opp_team_games=4,
        vs_defense_style_avg=None,
        vs_defense_style_games=0,
        opp_def_rating=108.0,  # Elite defense
        opp_pace=96.0,  # Slow pace
    )
    print(f"  Season Avg: 25.0 PPG")
    print(f"  Recent Form: L3=27.0, L5=26.0")
    print(f"  Vs This Team: 18.0 PPG (4 games) - Struggles!")
    print(f"  Opponent: 108.0 Def Rating (elite), 96.0 Pace (slow)")
    print(f"  Projection: {projection:.1f} PPG")
    print(f"  Range: {floor:.1f}-{ceiling:.1f} PPG")
    print(f"  Confidence: {confidence:.0%}")
    print(f"  Note: Projection balances poor team history (18.0) with hot recent form (27.0)")
    # Expect projection between team history and recent form
    assert 20.0 < projection < 25.0, f"Expected projection 20-25, got {projection:.1f}"
    assert confidence > 0.70, f"Expected high confidence with team history, got {confidence:.0%}"

    # Test Case 3: No team history, use defense style
    print("\nTest 3: No Team History - Using Defense Style")
    print("-" * 70)
    projection, confidence, floor, ceiling, breakdown = calculate_smart_ppg_projection(
        season_avg=22.0,
        recent_avg_5=23.5,
        recent_avg_3=24.0,
        vs_opp_team_avg=None,  # No team history
        vs_opp_team_games=0,
        vs_defense_style_avg=20.5,  # Struggles vs this style
        vs_defense_style_games=8,
        opp_def_rating=110.0,
        opp_pace=99.0,  # League average pace
    )
    print(f"  Season Avg: 22.0 PPG")
    print(f"  Recent Form: L3=24.0, L5=23.5")
    print(f"  Vs This Style: 20.5 PPG (8 games)")
    print(f"  Opponent: 110.0 Def Rating, 99.0 Pace")
    print(f"  Projection: {projection:.1f} PPG")
    print(f"  Range: {floor:.1f}-{ceiling:.1f} PPG")
    print(f"  Confidence: {confidence:.0%}")
    assert 20.0 < projection < 24.0, f"Expected projection 20-24, got {projection:.1f}"
    assert 0.50 < confidence < 0.75, f"Expected medium confidence, got {confidence:.0%}"

    # Test Case 4: Minimal data (low confidence)
    print("\nTest 4: Minimal Data - Low Confidence")
    print("-" * 70)
    projection, confidence, floor, ceiling, breakdown = calculate_smart_ppg_projection(
        season_avg=18.0,
        recent_avg_5=None,
        recent_avg_3=19.0,
        vs_opp_team_avg=None,
        vs_opp_team_games=0,
        vs_defense_style_avg=None,
        vs_defense_style_games=0,
        opp_def_rating=None,
        opp_pace=None,
    )
    print(f"  Season Avg: 18.0 PPG")
    print(f"  Recent Form: L3=19.0 (limited data)")
    print(f"  No matchup history or opponent data")
    print(f"  Projection: {projection:.1f} PPG")
    print(f"  Range: {floor:.1f}-{ceiling:.1f} PPG")
    print(f"  Confidence: {confidence:.0%}")
    assert 17.0 < projection < 20.0, f"Expected projection 17-20, got {projection:.1f}"
    assert confidence < 0.50, f"Expected low confidence, got {confidence:.0%}"

    # Test Case 5: Favorable matchup with pace boost
    print("\nTest 5: Favorable Matchup with Pace Boost")
    print("-" * 70)
    projection, confidence, floor, ceiling, breakdown = calculate_smart_ppg_projection(
        season_avg=20.0,
        recent_avg_5=21.0,
        recent_avg_3=22.0,
        vs_opp_team_avg=24.0,  # Excels vs this team
        vs_opp_team_games=3,
        vs_defense_style_avg=None,
        vs_defense_style_games=0,
        opp_def_rating=120.0,  # Very weak defense
        opp_pace=105.0,  # Very fast pace
    )
    print(f"  Season Avg: 20.0 PPG")
    print(f"  Recent Form: L3=22.0, L5=21.0")
    print(f"  Vs This Team: 24.0 PPG (3 games)")
    print(f"  Opponent: 120.0 Def Rating (very weak), 105.0 Pace (very fast)")
    print(f"  Projection: {projection:.1f} PPG")
    print(f"  Range: {floor:.1f}-{ceiling:.1f} PPG")
    print(f"  Confidence: {confidence:.0%}")
    print(f"  Def Adjustment: {breakdown['def_adjustment']:.3f}")
    print(f"  Pace Adjustment: {breakdown['pace_adjustment']:.3f}")
    print(f"  Note: Limited team history (3 games) = moderate weight, season avg anchors projection")
    # With only 3 games vs team, expect projection between season avg and team matchup
    assert 21.0 < projection < 24.0, f"Expected projection 21-24, got {projection:.1f}"
    assert confidence > 0.65, f"Expected decent confidence, got {confidence:.0%}"

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_projection_scenarios()
        print("\nSmart PPG projection algorithm is working correctly!")
        print("\nKey Features Tested:")
        print("  - Multi-factor weighted model (season, recent, matchup, defense, pace)")
        print("  - Team-specific history prioritized over defense style")
        print("  - Confidence scoring based on data availability")
        print("  - Floor/ceiling intervals reflecting uncertainty")
        print("  - Defense quality and pace adjustments")
        sys.exit(0)
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
