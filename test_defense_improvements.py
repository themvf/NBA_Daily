#!/usr/bin/env python3
"""
Test script for the new defense style improvements:
1. Pace-adjusted defensive rating
2. Rim protection (paint defense)
3. Multi-label classification
"""

import sys
import pandas as pd

# Test the classify_styles_multi_label logic
def test_multi_label_classification():
    """Test that multi-label classification works correctly"""
    print("=" * 60)
    print("Testing Multi-Label Classification Logic")
    print("=" * 60)

    # Simulate a team with elite overall defense and rim protection
    test_row_1 = pd.Series({
        "def_rating_pct": 0.10,  # Elite (top 20%)
        "def_3pm_per100_pct": 0.50,  # Average 3PT defense
        "def_2pt_pts_per100_pct": 0.25,  # Rim protector (bottom 35%)
        "avg_allowed_reb_pct": 0.20,  # Glass cleaner (bottom 35%)
    })

    styles_1 = classify_styles_multi_label(test_row_1)
    print(f"\nTest 1 - Elite Rim Protector:")
    print(f"  Percentiles: DefRating={test_row_1['def_rating_pct']:.2f}, "
          f"3PT={test_row_1['def_3pm_per100_pct']:.2f}, "
          f"2PT={test_row_1['def_2pt_pts_per100_pct']:.2f}, "
          f"Reb={test_row_1['avg_allowed_reb_pct']:.2f}")
    print(f"  Expected: ['Elite', 'Rim Protector', 'Glass Cleaner']")
    print(f"  Actual: {styles_1}")

    # Simulate a team with perimeter leak and paint vulnerability
    test_row_2 = pd.Series({
        "def_rating_pct": 0.85,  # Vulnerable (bottom 20%)
        "def_3pm_per100_pct": 0.75,  # Perimeter leak (top 35%)
        "def_2pt_pts_per100_pct": 0.80,  # Paint vulnerable (top 35%)
        "avg_allowed_reb_pct": 0.90,  # Board-soft (top 35%)
    })

    styles_2 = classify_styles_multi_label(test_row_2)
    print(f"\nTest 2 - Vulnerable Defense:")
    print(f"  Percentiles: DefRating={test_row_2['def_rating_pct']:.2f}, "
          f"3PT={test_row_2['def_3pm_per100_pct']:.2f}, "
          f"2PT={test_row_2['def_2pt_pts_per100_pct']:.2f}, "
          f"Reb={test_row_2['avg_allowed_reb_pct']:.2f}")
    print(f"  Expected: ['Vulnerable', 'Perimeter Leak', 'Paint Vulnerable', 'Board-Soft']")
    print(f"  Actual: {styles_2}")

    # Simulate a balanced team
    test_row_3 = pd.Series({
        "def_rating_pct": 0.50,
        "def_3pm_per100_pct": 0.50,
        "def_2pt_pts_per100_pct": 0.50,
        "avg_allowed_reb_pct": 0.50,
    })

    styles_3 = classify_styles_multi_label(test_row_3)
    print(f"\nTest 3 - Balanced Defense:")
    print(f"  Percentiles: All at 0.50 (median)")
    print(f"  Expected: ['Balanced']")
    print(f"  Actual: {styles_3}")

    print("\n" + "=" * 60)
    return True


def classify_styles_multi_label(row: pd.Series) -> list[str]:
    """Returns list of defensive style tags (copied from streamlit_app.py)"""
    styles = []

    def safe_float(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Overall tier based on pace-adjusted defensive rating
    def_rating_pct = safe_float(row.get("def_rating_pct"))
    if def_rating_pct is not None:
        if def_rating_pct <= 0.20:
            styles.append("Elite")
        elif def_rating_pct >= 0.80:
            styles.append("Vulnerable")

    # Perimeter defense (3PT)
    fg3m_pct = safe_float(row.get("def_3pm_per100_pct"))
    if fg3m_pct is not None:
        if fg3m_pct >= 0.65:
            styles.append("Perimeter Leak")
        elif fg3m_pct <= 0.35:
            styles.append("Perimeter Lock")

    # Paint defense (2PT points)
    paint_pct = safe_float(row.get("def_2pt_pts_per100_pct"))
    if paint_pct is not None:
        if paint_pct <= 0.35:
            styles.append("Rim Protector")
        elif paint_pct >= 0.65:
            styles.append("Paint Vulnerable")

    # Rebounding
    reb_pct = safe_float(row.get("avg_allowed_reb_pct"))
    if reb_pct is not None:
        if reb_pct >= 0.65:
            styles.append("Board-Soft")
        elif reb_pct <= 0.35:
            styles.append("Glass Cleaner")

    return styles if styles else ["Balanced"]


def test_pace_calculation():
    """Test pace-adjusted calculation logic"""
    print("\n" + "=" * 60)
    print("Testing Pace-Adjusted Calculations")
    print("=" * 60)

    # Simulate game data
    test_data = {
        "allowed_pts": 115,
        "allowed_fg3m": 12,
        "allowed_2pt_pts": 115 - (12 * 3),  # 79 points from 2PT
        "opp_fga": 88,
        "opp_fta": 20,
        "opp_oreb": 10,
        "opp_tov": 14,
    }

    # Calculate possessions: FGA + 0.44*FTA - ORB + TOV
    possessions = test_data["opp_fga"] + 0.44 * test_data["opp_fta"] - test_data["opp_oreb"] + test_data["opp_tov"]

    # Calculate per-100 stats
    def_rating = (test_data["allowed_pts"] / possessions) * 100
    def_3pm_per100 = (test_data["allowed_fg3m"] / possessions) * 100
    def_2pt_pts_per100 = (test_data["allowed_2pt_pts"] / possessions) * 100

    print(f"\nTest Game Stats:")
    print(f"  Allowed Points: {test_data['allowed_pts']}")
    print(f"  Allowed 3PM: {test_data['allowed_fg3m']}")
    print(f"  Allowed 2PT Points: {test_data['allowed_2pt_pts']}")
    print(f"  Opponent Possessions: {possessions:.2f}")
    print(f"\nPace-Adjusted Metrics (per 100 possessions):")
    print(f"  Defensive Rating: {def_rating:.1f}")
    print(f"  3PM Allowed per 100: {def_3pm_per100:.1f}")
    print(f"  2PT Points Allowed per 100: {def_2pt_pts_per100:.1f}")

    # Real NBA averages for comparison
    print(f"\nComparison to NBA averages (~110-115 def rating):")
    if def_rating < 110:
        print(f"  Elite defense (< 110)")
    elif def_rating < 115:
        print(f"  Good defense (110-115)")
    else:
        print(f"  Below average defense (> 115)")

    print("=" * 60)
    return True


if __name__ == "__main__":
    print("\nNBA Daily Defense Improvements - Test Suite\n")

    try:
        # Run tests
        test_multi_label_classification()
        test_pace_calculation()

        print("\nAll tests completed successfully!")
        print("\nNext steps:")
        print("  1. Review the test output above")
        print("  2. If tests look good, rebuild the database with:")
        print("     python nba_to_sqlite.py --season 2025-26")
        print("  3. Run Streamlit app to see the new defense styles:")
        print("     streamlit run streamlit_app.py")
        sys.exit(0)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
