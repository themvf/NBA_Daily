#!/usr/bin/env python3
"""
Top 3 Performance Tracking Module

Tracks and evaluates how well our top-3 predictions perform:
- Top-3 Recall: How many of actual top 3 did we pick?
- Top-3 Hit Rate: Did we get at least 1 of the top 3?
- Mean Rank of #1: Where did we rank the actual top scorer?

Usage:
    from top3_tracking import Top3Tracker

    tracker = Top3Tracker(conn)
    tracker.record_performance(game_date, predicted_ids, method='top_scorer_score')
    summary = tracker.get_performance_summary(days_back=30)
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class Top3Tracker:
    """
    Tracks and evaluates top-3 prediction performance.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Create tracking table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS top3_performance (
                game_date TEXT PRIMARY KEY,
                actual_top3_ids TEXT,
                actual_top3_names TEXT,
                actual_top3_points TEXT,
                predicted_top3_ids TEXT,
                predicted_top3_names TEXT,
                top3_recall INTEGER,
                top3_hit INTEGER,
                actual_1_our_rank INTEGER,
                top10_coverage INTEGER,
                method_used TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    # =========================================================================
    # Get Actual Results
    # =========================================================================

    def get_actual_top_scorers(
        self,
        game_date: str,
        n: int = 3
    ) -> pd.DataFrame:
        """
        Get the actual top N scorers for a date from recorded actuals.

        Returns DataFrame with player_id, player_name, actual_ppg, rank.
        """
        query = """
            SELECT
                player_id,
                player_name,
                team_name,
                actual_ppg
            FROM predictions
            WHERE game_date = ?
              AND actual_ppg IS NOT NULL
              AND actual_ppg > 0
            ORDER BY actual_ppg DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=[game_date, n])
        df['actual_rank'] = range(1, len(df) + 1)
        return df

    def get_all_actuals_ranked(self, game_date: str) -> pd.DataFrame:
        """Get all players with actuals, ranked by points."""
        query = """
            SELECT
                player_id,
                player_name,
                team_name,
                projected_ppg,
                actual_ppg
            FROM predictions
            WHERE game_date = ?
              AND actual_ppg IS NOT NULL
            ORDER BY actual_ppg DESC
        """
        df = pd.read_sql_query(query, self.conn, params=[game_date])
        df['actual_rank'] = range(1, len(df) + 1)
        return df

    # =========================================================================
    # Record Performance
    # =========================================================================

    def record_performance(
        self,
        game_date: str,
        predicted_top3_ids: List[int],
        method: str = 'top_scorer_score'
    ) -> Dict:
        """
        Record top-3 prediction performance for a date.

        Compares our predicted top 3 against actual top 3.

        Returns dict with performance metrics.
        """
        # Get actual results
        actual_top3_df = self.get_actual_top_scorers(game_date, n=3)
        actual_top10_df = self.get_actual_top_scorers(game_date, n=10)
        all_actuals_df = self.get_all_actuals_ranked(game_date)

        if actual_top3_df.empty:
            return {'error': f'No actual results found for {game_date}'}

        actual_top3_ids = set(actual_top3_df['player_id'].tolist())
        actual_top10_ids = set(actual_top10_df['player_id'].tolist())
        predicted_ids_set = set(predicted_top3_ids)

        # Calculate metrics
        # 1. Top-3 Recall: How many of actual top 3 did we pick?
        correct_picks = actual_top3_ids.intersection(predicted_ids_set)
        top3_recall = len(correct_picks)

        # 2. Top-3 Hit: Did we get at least 1?
        top3_hit = 1 if top3_recall > 0 else 0

        # 3. Mean rank of actual #1: Where did we rank them?
        actual_1_id = actual_top3_df.iloc[0]['player_id']

        # Get our ranking of this player
        from top3_ranking import Top3Ranker
        ranker = Top3Ranker(self.conn)
        our_rankings = ranker.rank_players(game_date, method=method)

        if not our_rankings.empty:
            our_rank_of_1 = our_rankings[
                our_rankings['player_id'] == actual_1_id
            ]['rank'].values
            actual_1_our_rank = int(our_rank_of_1[0]) if len(our_rank_of_1) > 0 else None
        else:
            actual_1_our_rank = None

        # 4. Top-10 coverage: How many of our top 3 finished in actual top 10?
        top10_coverage = len(predicted_ids_set.intersection(actual_top10_ids))

        # Prepare data for storage
        actual_names = actual_top3_df['player_name'].tolist()
        actual_points = actual_top3_df['actual_ppg'].tolist()

        # Get predicted player names
        pred_names = []
        for pid in predicted_top3_ids:
            name_row = all_actuals_df[all_actuals_df['player_id'] == pid]
            if not name_row.empty:
                pred_names.append(name_row.iloc[0]['player_name'])
            else:
                pred_names.append(f"Player {pid}")

        # Store in database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO top3_performance (
                game_date,
                actual_top3_ids,
                actual_top3_names,
                actual_top3_points,
                predicted_top3_ids,
                predicted_top3_names,
                top3_recall,
                top3_hit,
                actual_1_our_rank,
                top10_coverage,
                method_used,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            game_date,
            ','.join(map(str, actual_top3_df['player_id'].tolist())),
            ','.join(actual_names),
            ','.join(map(lambda x: f"{x:.1f}", actual_points)),
            ','.join(map(str, predicted_top3_ids)),
            ','.join(pred_names),
            top3_recall,
            top3_hit,
            actual_1_our_rank,
            top10_coverage,
            method,
            datetime.now().isoformat()
        ])
        self.conn.commit()

        return {
            'game_date': game_date,
            'top3_recall': top3_recall,
            'top3_hit': top3_hit,
            'actual_1_our_rank': actual_1_our_rank,
            'top10_coverage': top10_coverage,
            'actual_top3': actual_names,
            'predicted_top3': pred_names,
            'correct_picks': [n for n in actual_names if n in pred_names]
        }

    # =========================================================================
    # Performance Summary
    # =========================================================================

    def get_performance_summary(self, days_back: int = 30) -> Dict:
        """
        Get summary statistics for top-3 prediction performance.

        Returns dict with:
        - avg_recall: Average Top-3 Recall (0-3)
        - hit_rate: % of days with at least 1 correct pick
        - mean_rank_of_1: Average rank we assigned to actual #1
        - avg_top10_coverage: Average # of our picks in actual top 10
        - total_days: Number of days with data
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        query = """
            SELECT
                top3_recall,
                top3_hit,
                actual_1_our_rank,
                top10_coverage
            FROM top3_performance
            WHERE game_date >= ?
              AND game_date <= ?
        """
        df = pd.read_sql_query(
            query, self.conn,
            params=[str(start_date), str(end_date)]
        )

        if df.empty:
            return {
                'error': 'No performance data found',
                'total_days': 0
            }

        return {
            'total_days': len(df),
            'avg_recall': df['top3_recall'].mean(),
            'hit_rate': (df['top3_hit'].sum() / len(df)) * 100,
            'mean_rank_of_1': df['actual_1_our_rank'].mean(),
            'avg_top10_coverage': df['top10_coverage'].mean(),
            'perfect_days': (df['top3_recall'] == 3).sum(),
            'zero_days': (df['top3_recall'] == 0).sum(),
        }

    def get_daily_performance(self, days_back: int = 30) -> pd.DataFrame:
        """Get daily performance records for charting."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        query = """
            SELECT
                game_date,
                actual_top3_names,
                predicted_top3_names,
                top3_recall,
                top3_hit,
                actual_1_our_rank,
                top10_coverage,
                method_used
            FROM top3_performance
            WHERE game_date >= ?
              AND game_date <= ?
            ORDER BY game_date DESC
        """
        return pd.read_sql_query(
            query, self.conn,
            params=[str(start_date), str(end_date)]
        )

    # =========================================================================
    # Backfill Historical Performance
    # =========================================================================

    def backfill_performance(
        self,
        days_back: int = 30,
        method: str = 'top_scorer_score'
    ) -> Dict:
        """
        Backfill performance tracking for historical dates.

        Goes through past dates where we have actuals and records
        what our ranking would have predicted.
        """
        from top3_ranking import Top3Ranker
        ranker = Top3Ranker(self.conn)

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        # Get dates with actual results
        query = """
            SELECT DISTINCT game_date
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND game_date >= ?
              AND game_date <= ?
            ORDER BY game_date
        """
        dates_df = pd.read_sql_query(
            query, self.conn,
            params=[str(start_date), str(end_date)]
        )

        results = []
        for game_date in dates_df['game_date'].tolist():
            try:
                # Get our top 3 picks for this date
                rankings = ranker.rank_players(game_date, method=method)
                if rankings.empty:
                    continue

                top3_ids = rankings.head(3)['player_id'].tolist()

                # Record performance
                perf = self.record_performance(game_date, top3_ids, method)
                results.append(perf)

            except Exception as e:
                results.append({
                    'game_date': game_date,
                    'error': str(e)
                })

        # Get summary
        summary = self.get_performance_summary(days_back)
        summary['dates_processed'] = len(results)

        return summary

    # =========================================================================
    # Compare Methods
    # =========================================================================

    def compare_methods(self, days_back: int = 30) -> pd.DataFrame:
        """
        Compare performance of different ranking methods.

        Backfills for each method and returns comparison table.
        """
        methods = ['top_scorer_score', 'simulation']
        results = []

        for method in methods:
            summary = self.backfill_performance(days_back, method)
            summary['method'] = method
            results.append(summary)

        return pd.DataFrame(results)


# =========================================================================
# Convenience Functions
# =========================================================================

def quick_performance_check(conn: sqlite3.Connection, days_back: int = 30) -> None:
    """Print quick performance summary to console."""
    tracker = Top3Tracker(conn)
    summary = tracker.get_performance_summary(days_back)

    print(f"\n{'='*60}")
    print(f"TOP-3 PREDICTION PERFORMANCE (last {days_back} days)")
    print(f"{'='*60}")

    if summary.get('error'):
        print(f"Error: {summary['error']}")
        return

    print(f"Days analyzed: {summary['total_days']}")
    print(f"\nKey Metrics:")
    print(f"  Average Top-3 Recall: {summary['avg_recall']:.2f} / 3")
    print(f"  Hit Rate (≥1 correct): {summary['hit_rate']:.1f}%")
    print(f"  Mean Rank of #1 Scorer: {summary['mean_rank_of_1']:.1f}")
    print(f"  Avg Top-10 Coverage: {summary['avg_top10_coverage']:.1f} / 3")
    print(f"\nDistribution:")
    print(f"  Perfect days (3/3): {summary['perfect_days']}")
    print(f"  Zero days (0/3): {summary['zero_days']}")
