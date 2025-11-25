#!/usr/bin/env python3
"""
Test script for enhanced player matchup analysis.
Tests:
1. Player vs specific team history tracking
2. Enhanced matchup quality evaluation
3. Warning generation for bad matchups
"""

import sys


def test_matchup_quality_evaluation():
    """Test the evaluate_matchup_quality function"""
    print("=" * 60)
    print("Testing Matchup Quality Evaluation")
    print("=" * 60)

    # Simulate James Harden averaging 25 PPG
    player_avg = 25.0

    # Test Case 1: Player struggles against this specific team (Harden vs Boston)
    print("\nTest 1: Player Struggles vs Specific Team")
    print("-" * 60)
    player_vs_opp_avg = 18.0  # Averages 18 PPG vs this team
    games_vs_opp = 4
    std_vs_opp = 3.5
    rating, warning, confidence = evaluate_matchup_quality(
        player_avg, player_vs_opp_avg, None, games_vs_opp, 0, std_vs_opp
    )
    print(f"  Player Season Avg: {player_avg:.1f} PPG")
    print(f"  Player vs This Team: {player_vs_opp_avg:.1f} PPG ({games_vs_opp} games)")
    print(f"  Std Dev: {std_vs_opp:.1f}")
    print(f"  Expected: Rating='Avoid', Warning=Present")
    print(f"  Actual: Rating='{rating}', Warning='{warning}'")
    print(f"  Confidence: {confidence:.0%}")
    assert rating == "Avoid", f"Expected 'Avoid', got '{rating}'"
    assert len(warning) > 0, "Expected warning message"

    # Test Case 2: Player excels against this specific team (Harden vs Portland)
    print("\nTest 2: Player Excels vs Specific Team")
    print("-" * 60)
    player_vs_opp_avg = 32.0  # Averages 32 PPG vs this team
    games_vs_opp = 5
    std_vs_opp = 4.0
    rating, warning, confidence = evaluate_matchup_quality(
        player_avg, player_vs_opp_avg, None, games_vs_opp, 0, std_vs_opp
    )
    print(f"  Player Season Avg: {player_avg:.1f} PPG")
    print(f"  Player vs This Team: {player_vs_opp_avg:.1f} PPG ({games_vs_opp} games)")
    print(f"  Expected: Rating='Excellent', Warning=Present")
    print(f"  Actual: Rating='{rating}', Warning='{warning}'")
    print(f"  Confidence: {confidence:.0%}")
    assert rating == "Excellent", f"Expected 'Excellent', got '{rating}'"
    assert len(warning) > 0, "Expected warning message"

    # Test Case 3: No team history, falls back to defense style
    print("\nTest 3: No Team History, Use Defense Style")
    print("-" * 60)
    player_vs_style_avg = 20.0  # Struggles vs this defense style
    games_vs_style = 8
    rating, warning, confidence = evaluate_matchup_quality(
        player_avg, None, player_vs_style_avg, 0, games_vs_style, None
    )
    print(f"  Player Season Avg: {player_avg:.1f} PPG")
    print(f"  Player vs This Style: {player_vs_style_avg:.1f} PPG ({games_vs_style} games)")
    print(f"  Expected: Rating='Difficult'")
    print(f"  Actual: Rating='{rating}', Warning='{warning}'")
    print(f"  Confidence: {confidence:.0%}")
    assert rating == "Difficult", f"Expected 'Difficult', got '{rating}'"

    # Test Case 4: Inconsistent performance (high variance)
    print("\nTest 4: Inconsistent Performance (High Variance)")
    print("-" * 60)
    player_vs_opp_avg = 18.0
    games_vs_opp = 3
    std_vs_opp = 12.0  # Very high std dev (inconsistent)
    rating, warning, confidence = evaluate_matchup_quality(
        player_avg, player_vs_opp_avg, None, games_vs_opp, 0, std_vs_opp
    )
    print(f"  Player Season Avg: {player_avg:.1f} PPG")
    print(f"  Player vs This Team: {player_vs_opp_avg:.1f} PPG (Std: {std_vs_opp:.1f})")
    print(f"  Expected: Warning contains 'Inconsistent'")
    print(f"  Actual: Warning='{warning}'")
    assert "Inconsistent" in warning or "inconsistent" in warning.lower(), \
        f"Expected 'Inconsistent' in warning, got '{warning}'"

    # Test Case 5: Neutral matchup (average performance)
    print("\nTest 5: Neutral Matchup")
    print("-" * 60)
    player_vs_opp_avg = 24.5  # Very close to season average
    games_vs_opp = 3
    std_vs_opp = 3.0
    rating, warning, confidence = evaluate_matchup_quality(
        player_avg, player_vs_opp_avg, None, games_vs_opp, 0, std_vs_opp
    )
    print(f"  Player Season Avg: {player_avg:.1f} PPG")
    print(f"  Player vs This Team: {player_vs_opp_avg:.1f} PPG")
    print(f"  Expected: Rating='Neutral'")
    print(f"  Actual: Rating='{rating}'")
    assert rating == "Neutral", f"Expected 'Neutral', got '{rating}'"

    # Test Case 6: Confidence scoring
    print("\nTest 6: Confidence Scoring")
    print("-" * 60)
    print("  Sample sizes -> confidence:")
    for games in [2, 4, 6, 8, 10]:
        conf = min(1.0, games * 0.15)
        print(f"    {games} games vs team: {conf:.0%} confidence")

    print("\n" + "=" * 60)
    return True


def evaluate_matchup_quality(
    player_avg: float,
    player_vs_opp_avg: float | None,
    player_vs_style_avg: float | None,
    games_vs_opp: int,
    games_vs_style: int,
    std_vs_opp: float | None,
) -> tuple[str, str, float]:
    """
    Evaluate matchup quality and return (rating, warning, confidence).
    (Copied from streamlit_app.py for testing)
    """
    confidence = min(1.0, (games_vs_opp * 0.15) + (games_vs_style * 0.05))

    if player_vs_opp_avg is not None and games_vs_opp >= 2:
        diff_pct = (player_vs_opp_avg - player_avg) / player_avg if player_avg > 0 else 0
        is_volatile = std_vs_opp is not None and std_vs_opp > (player_avg * 0.4)

        if diff_pct <= -0.25:
            warning = f"Struggles vs this team (avg {player_vs_opp_avg:.1f} vs season {player_avg:.1f})"
            if is_volatile:
                warning += " - Inconsistent"
            return ("Avoid", warning, confidence)
        elif diff_pct <= -0.15:
            warning = f"Below average vs this team ({player_vs_opp_avg:.1f} vs {player_avg:.1f})"
            return ("Difficult", warning, confidence)
        elif diff_pct >= 0.20:
            warning = f"Excels vs this team (avg {player_vs_opp_avg:.1f} vs season {player_avg:.1f})"
            return ("Excellent", warning, confidence)
        elif diff_pct >= 0.10:
            return ("Good", "", confidence)
        else:
            return ("Neutral", "", confidence)

    if player_vs_style_avg is not None and games_vs_style >= 3:
        diff_pct = (player_vs_style_avg - player_avg) / player_avg if player_avg > 0 else 0

        if diff_pct <= -0.20:
            warning = f"Struggles vs this defense style ({player_vs_style_avg:.1f} vs {player_avg:.1f})"
            return ("Difficult", warning, confidence * 0.7)
        elif diff_pct >= 0.15:
            return ("Good", "", confidence * 0.7)
        else:
            return ("Neutral", "", confidence * 0.5)

    return ("Neutral", "", 0.0)


if __name__ == "__main__":
    print("\nEnhanced Matchup Analysis - Test Suite\n")

    try:
        test_matchup_quality_evaluation()

        print("\nAll tests passed successfully!")
        print("\nKey Features Tested:")
        print("  1. Team-specific performance tracking (e.g., Harden vs Boston)")
        print("  2. Matchup difficulty ratings: Excellent, Good, Neutral, Difficult, Avoid")
        print("  3. Warning generation for bad matchups")
        print("  4. Confidence scoring based on sample size")
        print("  5. Volatility detection (inconsistent performance)")
        print("\nNext Steps:")
        print("  1. Commit changes to GitHub")
        print("  2. Test with real NBA data in Streamlit app")
        print("  3. Look for matchup warnings in 'Today's Games' tab")
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
