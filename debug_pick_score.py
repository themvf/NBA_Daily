#!/usr/bin/env python3
"""Debug why all players show 'Avoid' grade"""

# Simulate the calculation with realistic values
def calculate_daily_pick_score(
    player_season_avg: float,
    player_projection: float,
    projection_confidence: float,
    matchup_rating: str,
    opp_def_rating: float | None,
    league_avg_def_rating: float = 112.0,
) -> tuple[float, str, str]:
    """Copy of the function from streamlit_app.py"""
    # 1. Base score from projected points (0-50 range)
    base_score = min(50, (player_projection / 45.0) * 50)

    # 2. Matchup quality bonus/penalty
    matchup_adjustments = {
        "Excellent": 20,
        "Good": 10,
        "Neutral": 0,
        "Difficult": -10,
        "Avoid": -20,
    }
    matchup_bonus = matchup_adjustments.get(matchup_rating, 0)

    # 3. Opponent defense adjustment
    if opp_def_rating is not None:
        def_diff = (league_avg_def_rating - opp_def_rating) / league_avg_def_rating
        defense_adjustment = def_diff * 10
    else:
        defense_adjustment = 0

    # 4. Confidence multiplier
    confidence_multiplier = 0.5 + (projection_confidence * 0.5)

    # Calculate final score
    raw_score = base_score + matchup_bonus + defense_adjustment
    final_score = max(0, min(100, raw_score * confidence_multiplier))

    # Grade
    if final_score >= 80:
        grade = "Elite"
    elif final_score >= 65:
        grade = "Excellent"
    elif final_score >= 50:
        grade = "Solid"
    elif final_score >= 35:
        grade = "Risky"
    else:
        grade = "Avoid"

    return (final_score, grade, f"Base:{base_score:.1f} Bonus:{matchup_bonus} Def:{defense_adjustment:.1f} Conf:{confidence_multiplier:.2f}")


print("=" * 80)
print("Daily Pick Score Debug - Testing Realistic Scenarios")
print("=" * 80)

# Test Case 1: Star player, neutral matchup
print("\nTest 1: Star Player (LeBron) - Neutral Matchup")
score, grade, breakdown = calculate_daily_pick_score(
    player_season_avg=27.5,
    player_projection=28.0,
    projection_confidence=0.55,  # 55% confidence (typical early season)
    matchup_rating="Neutral",
    opp_def_rating=112.0,  # League average defense
)
print(f"  Inputs: 28.0 proj, 55% conf, Neutral matchup, 112 def rating")
print(f"  Result: Score={score:.1f}, Grade={grade}")
print(f"  Breakdown: {breakdown}")
print(f"  Expected: Should be ~40-50 (Solid or Risky)")

# Test Case 2: High scorer with good matchup
print("\nTest 2: High Scorer (Curry) - Good Matchup")
score, grade, breakdown = calculate_daily_pick_score(
    player_season_avg=28.5,
    player_projection=30.2,
    projection_confidence=0.60,
    matchup_rating="Good",
    opp_def_rating=118.0,  # Weak defense
)
print(f"  Inputs: 30.2 proj, 60% conf, Good matchup, 118 def rating")
print(f"  Result: Score={score:.1f}, Grade={grade}")
print(f"  Breakdown: {breakdown}")
print(f"  Expected: Should be ~55-65 (Excellent)")

# Test Case 3: Role player
print("\nTest 3: Role Player - Neutral Matchup")
score, grade, breakdown = calculate_daily_pick_score(
    player_season_avg=12.0,
    player_projection=11.5,
    projection_confidence=0.45,
    matchup_rating="Neutral",
    opp_def_rating=112.0,
)
print(f"  Inputs: 11.5 proj, 45% conf, Neutral matchup, 112 def rating")
print(f"  Result: Score={score:.1f}, Grade={grade}")
print(f"  Breakdown: {breakdown}")
print(f"  Expected: Should be ~20-30 (Avoid - correctly)")

# Test Case 4: What confidence are we actually getting?
print("\n" + "=" * 80)
print("DIAGNOSIS: Check typical confidence values")
print("=" * 80)

print("\nTypical confidence calculation:")
print("  Base: 25%")
print("  + No team history, 5 games vs style: 12%")
print("  + Recent L3 & L5 consistent: 15%")
print("  + Def rating & pace available: 12%")
print("  = Total: 25% + 12% + 15% + 12% = 64% confidence")
print("\nBut if matchups are N/A:")
print("  Base: 25%")
print("  + No team history, no style data: 0%")
print("  + Recent L3 & L5: 15%")
print("  + Def rating: 8%")
print("  = Total: 48% confidence")

print("\n" + "=" * 80)
print("LIKELY ISSUE: All players show 'Avoid' means:")
print("=" * 80)
print("1. Confidence scores might be much LOWER than expected (~30-40%)")
print("2. This multiplier (0.5 + conf*0.5) = 0.65-0.70 is TOO punitive")
print("3. Even good projections get crushed by low confidence")
print("\nExample:")
print("  30 PPG projection → base 33.3")
print("  + Neutral (0) + Avg defense (0) = 33.3 raw")
print("  × 0.65 confidence multiplier = 21.6 final")
print("  = ❌ Avoid (< 35)")
print("\nSOLUTION: Adjust confidence multiplier to be less punitive")
