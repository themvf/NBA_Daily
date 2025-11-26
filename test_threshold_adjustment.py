#!/usr/bin/env python3
"""Test different threshold options"""

def calc_score_current(proj, conf, matchup_rating, def_rating, league_avg=112.0):
    """Current formula"""
    base = max(0, (proj - 5) * 1.5)
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

    # CURRENT THRESHOLDS
    if final >= 80: return final, "Elite"
    elif final >= 65: return final, "Strong"
    elif final >= 50: return final, "Solid"
    elif final >= 35: return final, "Risky"
    else: return final, "Avoid"

def calc_score_adjusted_thresholds(proj, conf, matchup_rating, def_rating, league_avg=112.0):
    """Same formula but LOWER thresholds"""
    base = max(0, (proj - 5) * 1.5)
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

    # ADJUSTED THRESHOLDS (lowered by ~10-15 points)
    if final >= 70: return final, "Elite"
    elif final >= 55: return final, "Strong"
    elif final >= 42: return final, "Solid"
    elif final >= 30: return final, "Risky"
    else: return final, "Avoid"

def calc_score_boosted_base(proj, conf, matchup_rating, def_rating, league_avg=112.0):
    """BOOSTED base formula (more generous scoring)"""
    # NEW: (proj - 3) * 1.8 instead of (proj - 5) * 1.5
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

    # Keep original thresholds
    if final >= 80: return final, "Elite"
    elif final >= 65: return final, "Strong"
    elif final >= 50: return final, "Solid"
    elif final >= 35: return final, "Risky"
    else: return final, "Avoid"

print("=" * 100)
print("COMPARISON: Current vs Adjusted Thresholds vs Boosted Base")
print("=" * 100)

players = [
    ("Giannis (32.5 PPG, 85% conf)", 32.5, 0.85),
    ("Luka (28.5 PPG, 80% conf)", 28.5, 0.80),
    ("AD (27.8 PPG, 75% conf)", 27.8, 0.75),
    ("LeBron (22.5 PPG, 70% conf)", 22.5, 0.70),
]

for scenario_name, matchup, def_rating in [
    ("NEUTRAL MATCHUP (Avg Def 112)", "Neutral", 112.0),
    ("GOOD MATCHUP (Avg Def 112)", "Good", 112.0),
    ("NEUTRAL MATCHUP (Weak Def 118)", "Neutral", 118.0),
]:
    print(f"\n{'=' * 100}")
    print(f"SCENARIO: {scenario_name}")
    print('=' * 100)
    print(f"{'Player':<32} | {'Current':<18} | {'Adj Thresholds':<18} | {'Boosted Base':<18}")
    print('-' * 100)

    for name, ppg, conf in players:
        s1, g1 = calc_score_current(ppg, conf, matchup, def_rating)
        s2, g2 = calc_score_adjusted_thresholds(ppg, conf, matchup, def_rating)
        s3, g3 = calc_score_boosted_base(ppg, conf, matchup, def_rating)

        print(f"{name:<32} | {s1:5.1f} ({g1:<8}) | {s2:5.1f} ({g2:<8}) | {s3:5.1f} ({g3:<8})")

print("\n" + "=" * 100)
print("RECOMMENDATION:")
print("=" * 100)
print("Option 1: ADJUSTED THRESHOLDS (Easiest fix)")
print("  - Elite: 70+ (was 80+)")
print("  - Strong: 55+ (was 65+)")
print("  - Solid: 42+ (was 50+)")
print("  - Risky: 30+ (was 35+)")
print("  Pros: Simple change, makes grades more attainable")
print("  Cons: Inflation - 'Elite' becomes less special")
print()
print("Option 2: BOOSTED BASE (More aggressive)")
print("  - Change base formula from (proj-5)*1.5 to (proj-3)*1.8")
print("  - Keep thresholds: Elite 80+, Strong 65+")
print("  Pros: Elite scorers reach Strong/Elite with neutral matchups")
print("  Cons: Bigger change to formula, need to retest edge cases")
print()
print("Option 3: HYBRID (Recommended)")
print("  - Slightly boosted base: (proj-4)*1.6")
print("  - Slightly lowered thresholds: Elite 75+, Strong 60+, Solid 45+")
print("  Pros: Balanced - elite scorers get Strong with neutral, Elite needs good matchup")
print("  Cons: Two changes instead of one")
