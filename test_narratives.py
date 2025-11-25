#!/usr/bin/env python3
"""Test narrative explanations for Pick Grades"""

def generate_explanation(proj, matchup, def_rating, conf):
    """Simulate the explanation generation"""
    factors = []

    # Projection factor
    if proj >= 30:
        factors.append("elite scorer")
    elif proj >= 25:
        factors.append("strong scorer")
    elif proj >= 18:
        factors.append("solid volume")
    else:
        factors.append("limited volume")

    # Matchup factor
    if matchup == "Excellent":
        factors.append("excels vs this matchup")
    elif matchup == "Good":
        factors.append("favorable matchup")
    elif matchup == "Difficult":
        factors.append("struggles vs this defense")
    elif matchup == "Avoid":
        factors.append("historically poor vs opponent")

    # Defense factor
    if def_rating >= 118:
        factors.append("weak defense")
    elif def_rating <= 106:
        factors.append("elite defense")

    # Confidence factor
    if conf >= 0.75:
        factors.append("high confidence")
    elif conf < 0.50:
        factors.append("limited data")

    return ", ".join(factors)

print("=" * 80)
print("Narrative Explanation Examples")
print("=" * 80)

scenarios = [
    ("Elite Pick", 32, "Excellent", 120, 0.80, "Elite pick"),
    ("Strong Scorer vs Weak D", 30, "Good", 118, 0.65, "Strong pick"),
    ("Solid Neutral", 25, "Neutral", 112, 0.60, "Reliable"),
    ("Star vs Elite Defense", 28, "Neutral", 105, 0.70, "Caution"),
    ("Star Struggling", 27, "Difficult", 110, 0.75, "Caution"),
    ("Role Player Bad Matchup", 14, "Avoid", 108, 0.45, "Avoid"),
    ("Mid-Tier Limited Data", 20, "Neutral", 112, 0.42, "Caution"),
    ("Elite Scorer Bad History", 33, "Avoid", 112, 0.80, "Caution"),
]

for name, proj, matchup, def_rating, conf, grade_prefix in scenarios:
    explanation = generate_explanation(proj, matchup, def_rating, conf)
    full_text = f"{grade_prefix}: {explanation}"
    print(f"\n{name}:")
    print(f"  Projection: {proj} PPG | Matchup: {matchup} | Def: {def_rating} | Conf: {conf:.0%}")
    print(f"  → {full_text}")

print("\n" + "=" * 80)
print("How They Appear in Table:")
print("=" * 80)
print("\n| Player | Pick Score | Grade |")
print("|--------|------------|-------|")
print("| Curry  | 82.5 | Elite pick: elite scorer, excels vs this matchup, weak defense, high confidence |")
print("| LeBron | 56.3 | Reliable: strong scorer, favorable matchup |")
print("| Harden | 41.2 | Caution: strong scorer, struggles vs this defense, elite defense, high confidence |")
print("| Bench  | 22.1 | Avoid: limited volume, historically poor vs opponent, limited data |")

print("\n" + "=" * 80)
print("Benefits:")
print("=" * 80)
print("✓ Users instantly understand WHY a player is graded this way")
print("✓ No need to cross-reference multiple columns")
print("✓ Contextual - explains the specific situation")
print("✓ Actionable - tells you what factors matter for THIS pick")
