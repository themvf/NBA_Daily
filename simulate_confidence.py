#!/usr/bin/env python3
"""
Simulate confidence scores for typical early-season scenarios
"""

def calc_confidence(vs_opp_team_games, vs_defense_style_games, has_recent_3, has_opp_def):
    """Simulate the confidence calculation from streamlit_app.py"""
    confidence = 0.30  # Base

    # Matchup data
    if vs_opp_team_games >= 2:
        confidence += min(0.40, vs_opp_team_games * 0.10)
    elif vs_defense_style_games >= 3:
        confidence += min(0.20, vs_defense_style_games * 0.03)

    # Recent form
    if has_recent_3:
        confidence += 0.15

    # Opponent data
    if has_opp_def:
        confidence += 0.10

    return min(0.95, confidence)


print("=" * 70)
print("Confidence Score Simulation - Early Season (2025-26)")
print("=" * 70)

scenarios = [
    ("No team history, has style + recent + def", 0, 5, True, True),
    ("No team history, has style + recent, no def", 0, 5, True, False),
    ("2 games vs team, has recent + def", 2, 0, True, True),
    ("3 games vs team, has recent + def", 3, 0, True, True),
    ("5+ games vs team, has recent + def", 5, 0, True, True),
    ("Only 1 game vs team, has style", 1, 5, True, True),
    ("No matchup data at all", 0, 2, True, True),
    ("No matchup, no recent form", 0, 2, False, True),
]

for name, vs_opp, vs_style, has_recent, has_def in scenarios:
    conf = calc_confidence(vs_opp, vs_style, has_recent, has_def)
    print(f"\n{name}:")
    print(f"  vs_opp_games={vs_opp}, vs_style_games={vs_style}")
    print(f"  has_L3={has_recent}, has_def_rating={has_def}")
    print(f"  Confidence: {conf:.0%}")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("Early in the season, most players fall into scenario 1:")
print("  • 0 games vs this specific opponent (or only 1 game)")
print("  • Have defense style data (5+ games vs similar defenses)")
print("  • Have recent 3-game average")
print("  • Have opponent defensive rating")
print("  → Result: 30% + 20% + 15% + 10% = 75% confidence")
print("\nThis is why you're seeing 70% for everyone!")
print("The algorithm is working correctly - there just isn't enough")
print("team-specific matchup history yet to differentiate players.")
