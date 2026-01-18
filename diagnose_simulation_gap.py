#!/usr/bin/env python3
"""
Diagnostic script to identify why stars are being buried in backtest.

Root cause: p_top3, p_top1, top_scorer_score columns don't exist in predictions table.
The simulation runs in Streamlit UI but results are NOT persisted to database.
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def diagnose_database(db_path: str = 'nba_stats.db'):
    """Run full diagnosis on the predictions table."""
    print("=" * 70)
    print("SIMULATION GAP DIAGNOSIS")
    print("=" * 70)

    conn = sqlite3.connect(db_path)

    # 1. Check predictions table schema
    print("\n1. SCHEMA CHECK")
    print("-" * 40)
    cursor = conn.execute("PRAGMA table_info(predictions)")
    cols = {row[1] for row in cursor.fetchall()}

    critical_cols = ['p_top3', 'p_top1', 'top_scorer_score']
    for col in critical_cols:
        status = "[OK] EXISTS" if col in cols else "[X] MISSING"
        print(f"   {col:20} {status}")

    missing = [c for c in critical_cols if c not in cols]
    if missing:
        print(f"\n   [!] CRITICAL: {len(missing)} simulation columns MISSING from schema!")
        print("   This means backtest falls back to projected_ppg for ranking.")

    # 2. Check data range
    print("\n2. DATA RANGE")
    print("-" * 40)
    cursor = conn.execute("SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM predictions")
    count, min_date, max_date = cursor.fetchone()
    print(f"   Total predictions: {count:,}")
    print(f"   Date range: {min_date} to {max_date}")

    # 3. Check by-date stats
    print("\n3. PER-DATE STATISTICS")
    print("-" * 40)

    query = """
        SELECT
            game_date,
            COUNT(*) as n_players,
            AVG(projected_ppg) as avg_proj,
            MAX(projected_ppg) as max_proj,
            SUM(CASE WHEN season_avg_ppg > 25 THEN 1 ELSE 0 END) as stars
        FROM predictions
        GROUP BY game_date
        ORDER BY game_date DESC
        LIMIT 14
    """
    df = pd.read_sql_query(query, conn)

    print(f"   {'Date':<12} {'Players':>8} {'Avg Proj':>10} {'Max Proj':>10} {'Stars(25+)':>10}")
    print("   " + "-" * 52)
    for _, row in df.iterrows():
        print(f"   {row['game_date']:<12} {int(row['n_players']):>8} {row['avg_proj']:>10.1f} {row['max_proj']:>10.1f} {int(row['stars']):>10}")

    # 4. Check which stars would have NULL p_top3
    print("\n4. STARS THAT WOULD HAVE NULL p_top3")
    print("-" * 40)
    print("   (These players get ranked by projected_ppg fallback, not simulation)")

    query = """
        SELECT
            game_date,
            player_name,
            projected_ppg,
            proj_ceiling,
            season_avg_ppg
        FROM predictions
        WHERE season_avg_ppg >= 25
        ORDER BY game_date DESC, projected_ppg DESC
        LIMIT 20
    """
    df = pd.read_sql_query(query, conn)

    print(f"   {'Date':<12} {'Player':<25} {'Proj':>8} {'Ceil':>8} {'SznAvg':>8}")
    print("   " + "-" * 64)
    for _, row in df.iterrows():
        print(f"   {row['game_date']:<12} {row['player_name'][:24]:<25} {row['projected_ppg']:>8.1f} {row['proj_ceiling']:>8.1f} {row['season_avg_ppg']:>8.1f}")

    # 5. Recommendations
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    if missing:
        print("""
ROOT CAUSE IDENTIFIED:
----------------------
The predictions table is MISSING these columns:
  - p_top3 (simulation probability of finishing top 3)
  - p_top1 (simulation probability of being #1)
  - top_scorer_score (heuristic score)

WHAT THIS MEANS:
----------------
1. The Monte Carlo simulation runs in Streamlit UI but results are NOT saved
2. When backtest runs with strategy='sim_p_top3', it can't find the column
3. My fix makes it fall back to projected_ppg, but that's not optimal
4. Stars like Murray/Shai get ranked by their projected_ppg, not P(top-3)

FIX REQUIRED:
-------------
Option A: Add columns to schema + persist simulation results
  1. ALTER TABLE predictions ADD COLUMN p_top3 REAL;
  2. ALTER TABLE predictions ADD COLUMN p_top1 REAL;
  3. ALTER TABLE predictions ADD COLUMN top_scorer_score REAL;
  4. Run simulation when generating predictions and save results

Option B: Run simulation on-the-fly during backtest
  - Slower but doesn't require schema change
  - Modify backtest to call Top3Ranker.simulate_top3_probability()

RECOMMENDED: Option A (one-time schema fix + persist results)
""")
    else:
        print("All simulation columns exist. Check for NULL values.")

    conn.close()


def add_simulation_columns(db_path: str = 'nba_stats.db'):
    """Add missing simulation columns to predictions table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns_to_add = [
        ('p_top3', 'REAL'),
        ('p_top1', 'REAL'),
        ('top_scorer_score', 'REAL'),
        ('sim_sigma', 'REAL'),
        ('sim_tier', 'TEXT'),
    ]

    for col_name, col_type in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            print(f"[OK] Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"[SKIP] Column already exists: {col_name}")
            else:
                print(f"[X] Error adding {col_name}: {e}")

    conn.commit()
    conn.close()
    print("\nSchema update complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose simulation gap in backtest')
    parser.add_argument('--fix', action='store_true', help='Add missing columns to schema')
    parser.add_argument('--db', default='nba_stats.db', help='Database path')
    args = parser.parse_args()

    if args.fix:
        add_simulation_columns(args.db)
    else:
        diagnose_database(args.db)
