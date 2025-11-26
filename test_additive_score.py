#!/usr/bin/env python3
"""Test additive scoring (no crushing multiplier)"""

def calc_score_additive(proj, conf, matchup_bonus=0, def_adj=0):
    # Base: (proj - 5) * 1.5
    base = max(0, (proj - 5) * 1.5)
    # Confidence: 0-15 bonus (not multiplier!)
    conf_bonus = conf * 15
    # All additive
    final = base + matchup_bonus + def_adj + conf_bonus
    final = max(0, min(100, final))

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

    return final, grade, base, conf_bonus

print("=" * 70)
print("ADDITIVE Scoring Test (No Crushing Multiplier)")
print("=" * 70)

tests = [
    ("LeBron (28 PPG, 55% conf, Neutral)", 28.0, 0.55, 0, 0),
    ("Curry (30 PPG, 60% conf, Good, weak def)", 30.2, 0.60, 10, 0.5),
    ("Star + Excellent matchup (32 PPG, 75% conf)", 32.0, 0.75, 20, 5),
    ("Role player (12 PPG, 45% conf)", 12.0, 0.45, 0, 0),
    ("Elite scorer neutral (35 PPG, 80% conf)", 35.0, 0.80, 0, 0),
    ("Mid-tier good matchup (22 PPG, 50% conf, Good)", 22.0, 0.50, 10, 0),
]

for name, proj, conf, bonus, def_adj in tests:
    score, grade, base, conf_bonus = calc_score_additive(proj, conf, bonus, def_adj)
    print(f"\n{name}")
    print(f"  Base: {base:.1f} + Matchup: {bonus:+.1f} + Def: {def_adj:+.1f} + Conf: {conf_bonus:+.1f}")
    print(f"  = {score:.1f} -> {grade}")

print("\n" + "=" * 70)
print("Quick Reference:")
print("=" * 70)
print("PPG | Base | +Conf(55%) | +Good | +Excellent")
print("-" * 70)
for ppg in [12, 20, 25, 30, 35]:
    base = max(0, (ppg - 5) * 1.5)
    with_conf = base + 8.25  # 55% conf
    with_good = with_conf + 10
    with_exc = with_conf + 20
    print(f" {ppg}  | {base:4.1f} |   {with_conf:4.1f}    | {with_good:4.1f}  |   {with_exc:4.1f}")

print("\n" + "=" * 70)
print("Grade Distribution Check:")
print("=" * 70)
print("Elite (80+): ~35+ PPG with Excellent matchup + high conf")
print("Excellent (65-79): ~28-32 PPG with Good/Excellent matchup")
print("Solid (50-64): ~22-28 PPG with decent matchup")
print("Risky (35-49): ~15-22 PPG or stars with bad matchups")
print("Avoid (<35): <15 PPG or very poor matchups")
