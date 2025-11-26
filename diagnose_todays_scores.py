#!/usr/bin/env python3
"""Diagnose why no Elite/Strong picks are showing today"""

import sqlite3
from datetime import datetime, timedelta

def calculate_daily_pick_score(proj, conf, matchup_rating, def_rating, league_avg=112.0):
    """Simplified scoring calculation"""
    # Base: (proj - 5) * 1.5
    base = max(0, (proj - 5) * 1.5)

    # Matchup bonus
    matchup_map = {"Excellent": 20, "Good": 10, "Neutral": 0, "Difficult": -10, "Avoid": -20}
    matchup_bonus = matchup_map.get(matchup_rating, 0)

    # Defense adjustment
    if def_rating:
        def_diff = (league_avg - def_rating) / league_avg
        def_adj = def_diff * 10
    else:
        def_adj = 0

    # Confidence bonus
    conf_bonus = conf * 15

    # Final
    final = base + matchup_bonus + def_adj + conf_bonus
    final = max(0, min(100, final))

    if final >= 80:
        grade = "Elite"
    elif final >= 65:
        grade = "Strong"
    elif final >= 50:
        grade = "Solid"
    elif final >= 35:
        grade = "Risky"
    else:
        grade = "Avoid"

    return final, grade, base, matchup_bonus, def_adj, conf_bonus

conn = sqlite3.connect('nba_2026.db')
cursor = conn.cursor()

# Get today and next few days
today = datetime.now().date()
dates_to_check = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)]

print("=" * 80)
print("Checking games for next 5 days...")
print("=" * 80)

for check_date in dates_to_check:
    cursor.execute("SELECT COUNT(*) FROM games WHERE game_date = ?", (check_date,))
    count = cursor.fetchone()[0]
    print(f"{check_date}: {count} games")

print("\n" + "=" * 80)
print("Analyzing Pick Scores for players with games TODAY or TOMORROW")
print("=" * 80)

# Get players playing in next 2 days with their stats
cursor.execute("""
    WITH player_stats AS (
        SELECT
            player_id,
            player_name,
            AVG(pts) as avg_pts,
            COUNT(*) as games,
            AVG(CASE WHEN pts > 0 THEN 1.0 ELSE 0.0 END) as confidence
        FROM player_game_log
        WHERE season_id = '2025-26'
        GROUP BY player_id, player_name
        HAVING games >= 3
    )
    SELECT
        ps.player_name,
        ps.avg_pts,
        ps.games,
        ps.confidence,
        g.game_date,
        g.home_team_abbreviation,
        g.visitor_team_abbreviation
    FROM player_stats ps
    JOIN games g ON 1=1
    WHERE g.game_date IN (?, ?)
    AND ps.avg_pts >= 15
    ORDER BY ps.avg_pts DESC
    LIMIT 30
""", (dates_to_check[0], dates_to_check[1]))

players = cursor.fetchall()

if not players:
    print("No players found with games in next 2 days and 15+ PPG average")
else:
    print(f"\nFound {len(players)} high-scoring players with upcoming games\n")

    # Simulate scoring with different scenarios
    scenarios = [
        ("Neutral matchup, avg defense (112)", "Neutral", 112.0),
        ("Good matchup, avg defense", "Good", 112.0),
        ("Excellent matchup, weak defense (118)", "Excellent", 118.0),
        ("Good matchup, weak defense (118)", "Good", 118.0),
    ]

    for scenario_name, matchup, def_rating in scenarios:
        print(f"\n--- SCENARIO: {scenario_name} ---")
        top_scores = []

        for player_name, avg_pts, games, confidence, game_date, home, away in players[:10]:
            score, grade, base, m_bonus, d_adj, c_bonus = calculate_daily_pick_score(
                avg_pts, confidence, matchup, def_rating
            )
            top_scores.append((player_name, avg_pts, confidence, score, grade, base, m_bonus, d_adj, c_bonus))

        # Show top 5
        top_scores.sort(key=lambda x: x[3], reverse=True)
        for i, (name, ppg, conf, score, grade, base, mb, da, cb) in enumerate(top_scores[:5], 1):
            print(f"{i}. {name}: {ppg:.1f} PPG, {conf:.0%} conf")
            print(f"   Score: {score:.1f} ({grade})")
            print(f"   Breakdown: Base={base:.1f} + Matchup={mb:+.1f} + Def={da:+.1f} + Conf={cb:+.1f}")

print("\n" + "=" * 80)
print("Key Insights:")
print("=" * 80)
print("To reach Elite (80+), a player needs:")
print("  - ~35+ PPG (45 base) + Excellent matchup (20) + weak def (5) + high conf (12) = 82")
print("  - OR ~30 PPG (37.5 base) + Excellent (20) + very weak def (10) + high conf (12) = 79.5 (close)")
print("\nTo reach Strong (65+), a player needs:")
print("  - ~28 PPG (34.5 base) + Good matchup (10) + weak def (5) + high conf (12) = 61.5 (close)")
print("  - ~30 PPG (37.5 base) + Good (10) + decent conf (10) = 57.5 (needs better matchup)")

conn.close()
