#!/usr/bin/env python3
"""Test revised confidence multiplier"""

def calc_score(proj, conf, matchup_bonus=0, def_adj=0):
    base = min(50, (proj / 45.0) * 50)
    # NEW: 0.7 + conf * 0.3 (range 0.7-1.0)
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

    return final, grade, conf_mult

print("=" * 70)
print("REVISED Confidence Multiplier Test")
print("=" * 70)

print("\nTest 1: LeBron (28 PPG, 55% conf, Neutral)")
score, grade, mult = calc_score(28.0, 0.55, 0, 0)
print(f"  Confidence: 55% -> Multiplier: {mult:.2f}")
print(f"  Score: {score:.1f} -> Grade: {grade}")
print(f"  Expected: ~35-50 (Risky or Solid)")

print("\nTest 2: Curry (30.2 PPG, 60% conf, Good matchup, weak def)")
score, grade, mult = calc_score(30.2, 0.60, 10, 0.5)
print(f"  Confidence: 60% -> Multiplier: {mult:.2f}")
print(f"  Score: {score:.1f} -> Grade: {grade}")
print(f"  Expected: ~50-65 (Solid or Excellent)")

print("\nTest 3: Star with Excellent matchup (32 PPG, 75% conf)")
score, grade, mult = calc_score(32.0, 0.75, 20, 5)
print(f"  Confidence: 75% -> Multiplier: {mult:.2f}")
print(f"  Score: {score:.1f} -> Grade: {grade}")
print(f"  Expected: ~70-85 (Excellent or Elite)")

print("\nTest 4: Role player (12 PPG, 45% conf)")
score, grade, mult = calc_score(12.0, 0.45, 0, 0)
print(f"  Confidence: 45% -> Multiplier: {mult:.2f}")
print(f"  Score: {score:.1f} -> Grade: {grade}")
print(f"  Expected: ~20-30 (Avoid - correct)")

print("\nTest 5: Low confidence star (28 PPG, 35% conf)")
score, grade, mult = calc_score(28.0, 0.35, 0, 0)
print(f"  Confidence: 35% -> Multiplier: {mult:.2f}")
print(f"  Score: {score:.1f} -> Grade: {grade}")
print(f"  Expected: ~30-40 (Risky)")

print("\n" + "=" * 70)
print("Comparison: Old vs New Multiplier")
print("=" * 70)
print("Confidence | Old (0.5+c*0.5) | New (0.7+c*0.3) | Difference")
print("-" * 70)
for conf in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
    old_mult = 0.5 + conf * 0.5
    new_mult = 0.7 + conf * 0.3
    diff = new_mult - old_mult
    print(f"  {conf:.0%}     |     {old_mult:.2f}        |     {new_mult:.2f}        | +{diff:.2f}")

print("\n" + "=" * 70)
print("RESULT: New multiplier is less punitive")
print("  - Old: 35% conf -> 0.68x (crushed to 68%)")
print("  - New: 35% conf -> 0.81x (modest 19% penalty)")
print("  - Old: 65% conf -> 0.83x (still harsh)")
print("  - New: 65% conf -> 0.90x (fair 10% penalty)")
print("=" * 70)
