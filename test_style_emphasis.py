#!/usr/bin/env python3
"""Test revised scoring: boosted base + style-first matchups"""

def calculate_daily_pick_score_new(proj, conf, matchup_rating, def_rating, league_avg=112.0):
    """NEW: Boosted base formula"""
    base = max(0, (proj - 3) * 1.8)

    matchup_map = {"Excellent": 20, "Good": 10, "Neutral": 0, "Difficult": -10, "Avoid": -20}
    matchup_bonus = matchup_map.get(matchup_rating, 0)

    if def_rating:
        def_diff = (league_avg - def_rating) / league_avg
        def_adj = def_diff * 10
    else:
        def_adj = 0

    conf_bonus = conf * 15

    final = base + matchup_bonus + def_adj + conf_bonus
    final = max(0, min(100, final))

    if final >= 80: return final, "Elite"
    elif final >= 65: return final, "Strong"
    elif final >= 50: return final, "Solid"
    elif final >= 35: return final, "Risky"
    else: return final, "Avoid"

print("=" * 90)
print("REVISED SCORING: Boosted Base + Style-First Matchups")
print("=" * 90)

print("\nKEY CHANGES:")
print("  1. Base formula: (proj - 3) * 1.8  (was (proj - 5) * 1.5)")
print("  2. Matchup evaluation: Prioritizes defensive STYLE over head-to-head")
print("  3. Style confidence: 10% per game (was 5%), Head-to-head: 5% per game (was 15%)")

print("\n" + "=" * 90)
print("SCENARIO 1: Elite Scorers with NEUTRAL Matchup (avg def 112)")
print("=" * 90)

elite_scorers = [
    ("Giannis", 32.5, 0.85),
    ("Luka", 28.5, 0.80),
    ("SGA", 28.8, 0.82),
    ("AD", 27.8, 0.75),
    ("Tatum", 27.9, 0.78),
]

for name, ppg, conf in elite_scorers:
    score, grade = calculate_daily_pick_score_new(ppg, conf, "Neutral", 112.0)
    print(f"{name:15} {ppg:4.1f} PPG, {conf:.0%} conf -> {score:5.1f} ({grade})")

print("\n" + "=" * 90)
print("SCENARIO 2: Elite Scorers with GOOD Style Matchup (avg def 112)")
print("=" * 90)
print("Player performs 12%+ better vs this defensive style")

for name, ppg, conf in elite_scorers:
    score, grade = calculate_daily_pick_score_new(ppg, conf, "Good", 112.0)
    print(f"{name:15} {ppg:4.1f} PPG -> {score:5.1f} ({grade})")

print("\n" + "=" * 90)
print("SCENARIO 3: Elite Scorers with GOOD Style + Weak Defense (118)")
print("=" * 90)

for name, ppg, conf in elite_scorers:
    score, grade = calculate_daily_pick_score_new(ppg, conf, "Good", 118.0)
    print(f"{name:15} {ppg:4.1f} PPG -> {score:5.1f} ({grade})")

print("\n" + "=" * 90)
print("SCENARIO 4: Mid-Tier Scorers (20-25 PPG)")
print("=" * 90)

mid_tier = [
    ("DeRozan", 24.0, 0.72),
    ("Randle", 22.5, 0.68),
    ("Brunson", 25.5, 0.75),
]

for name, ppg, conf in mid_tier:
    neutral = calculate_daily_pick_score_new(ppg, conf, "Neutral", 112.0)
    good = calculate_daily_pick_score_new(ppg, conf, "Good", 112.0)
    print(f"{name:15} {ppg:4.1f} PPG -> Neutral: {neutral[0]:5.1f} ({neutral[1]:6}), Good: {good[0]:5.1f} ({good[1]:6})")

print("\n" + "=" * 90)
print("EXPECTED RESULTS:")
print("=" * 90)
print("Elite (80+):")
print("  - 30+ PPG with Good matchup + weak defense")
print("  - 32+ PPG with Excellent matchup")
print()
print("Strong (65+):")
print("  - 30+ PPG with Neutral matchup (NEW! Was Solid before)")
print("  - 28+ PPG with Good matchup")
print()
print("Solid (50+):")
print("  - 25+ PPG with Neutral matchup")
print("  - 22+ PPG with Good matchup")
print()
print("IMPACT: Elite scorers now get 'Strong' grades even without favorable matchups,")
print("        making the system more useful early in the season when style data")
print("        provides better predictions than limited head-to-head history.")
