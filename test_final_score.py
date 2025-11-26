#!/usr/bin/env python3
"""Test final revised scoring formula"""

def calc_score_final(proj, conf, matchup_bonus=0, def_adj=0):
    # NEW BASE: (proj - 5) * 1.5
    base = max(0, (proj - 5) * 1.5)
    # NEW CONFIDENCE: 0.7 + conf * 0.3
    conf_mult = 0.7 + (conf * 0.3)
    raw = base + matchup_bonus + def_adj
    final = raw * conf_mult

    if final >= 80:
        grade = "Elite"
    elif final >= 65:
        grade = "Excellent"
    elif final >= 50:
        grade = "Solid"
    elif final >= 35:
        grade = "Risky"
    else:
        grade = "Avoid"

    return final, grade, base, conf_mult

print("=" * 70)
print("FINAL Scoring Formula Test")
print("=" * 70)

tests = [
    ("LeBron (28 PPG, 55% conf, Neutral)", 28.0, 0.55, 0, 0, "40-50 (Solid)"),
    ("Curry (30 PPG, 60% conf, Good, weak def)", 30.2, 0.60, 10, 0.5, "50-65 (Excellent)"),
    ("Star + Excellent matchup (32 PPG, 75% conf)", 32.0, 0.75, 20, 5, "70-85 (Excellent/Elite)"),
    ("Role player (12 PPG, 45% conf)", 12.0, 0.45, 0, 0, "20-30 (Avoid)"),
    ("Low scorer (8 PPG, 45% conf)", 8.0, 0.45, 0, 0, "<20 (Avoid)"),
    ("Elite scorer (35 PPG, 80% conf, neutral)", 35.0, 0.80, 0, 0, "60-70 (Excellent)"),
]

for name, proj, conf, bonus, def_adj, expected in tests:
    score, grade, base, mult = calc_score_final(proj, conf, bonus, def_adj)
    print(f"\n{name}")
    print(f"  Base: {base:.1f}, Bonus: {bonus:+.1f}, Def: {def_adj:+.1f}")
    print(f"  Raw: {base+bonus+def_adj:.1f} x {mult:.2f} = {score:.1f}")
    print(f"  Grade: {grade} (expected: {expected})")

print("\n" + "=" * 70)
print("Base Score Examples:")
print("=" * 70)
for ppg in [8, 12, 16, 20, 24, 28, 32, 36, 40]:
    base = max(0, (ppg - 5) * 1.5)
    print(f"  {ppg} PPG -> Base {base:.1f}")

print("\n" + "=" * 70)
print("Full Score Examples (with typical 55% confidence = 0.86x):")
print("=" * 70)
for ppg in [12, 20, 25, 30, 35]:
    base = max(0, (ppg - 5) * 1.5)
    final_neutral = base * 0.86
    final_good = (base + 10) * 0.86
    print(f"  {ppg} PPG:")
    print(f"    Neutral matchup: {final_neutral:.1f}")
    print(f"    Good matchup: {final_good:.1f}")
