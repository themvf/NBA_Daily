#!/usr/bin/env python3
"""
Segment Analysis Module

Provides detailed breakdowns of prediction accuracy by various segments:
- Role tier (STAR, STARTER, ROTATION, BENCH)
- Game script (blowout, close_game, tossup)
- Rest status (B2B, rested, normal)
- Position matchup (favorable, neutral, unfavorable)
"""

import sqlite3
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class SegmentAnalyzer:
    """
    Analyzes prediction performance by segment.

    Usage:
        analyzer = SegmentAnalyzer(conn)
        results = analyzer.analyze_all_segments('2025-12-01', '2025-12-31')
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def load_data(self, date_from: str, date_to: str) -> pd.DataFrame:
        """Load predictions with actuals and enrichment factors."""
        query = """
            SELECT
                game_date, player_id, player_name,
                projected_ppg, actual_ppg, proj_floor, proj_ceiling,
                days_rest, rest_multiplier, is_b2b,
                game_script_tier, blowout_risk, minutes_adjustment,
                role_tier, position_matchup_factor
            FROM predictions
            WHERE date(game_date) BETWEEN date(?) AND date(?)
              AND actual_ppg IS NOT NULL
        """
        return pd.read_sql_query(query, self.conn, params=[date_from, date_to])

    def calculate_segment_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate accuracy metrics for a segment."""
        if df.empty:
            return {'count': 0, 'mae': None, 'rmse': None, 'bias': None}

        errors = df['actual_ppg'] - df['projected_ppg']
        ceiling_hits = ((df['actual_ppg'] >= df['proj_floor']) &
                        (df['actual_ppg'] <= df['proj_ceiling']))

        return {
            'count': len(df),
            'mae': errors.abs().mean(),
            'rmse': np.sqrt((errors ** 2).mean()),
            'bias': errors.mean(),
            'ceiling_hit_rate': ceiling_hits.mean(),
            'mean_actual': df['actual_ppg'].mean(),
            'mean_projected': df['projected_ppg'].mean(),
        }

    def analyze_by_role(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze performance by role tier."""
        results = {}
        for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
            role_df = df[df['role_tier'] == role]
            results[role] = self.calculate_segment_metrics(role_df)
        return results

    def analyze_by_game_script(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze performance by game script tier."""
        results = {}
        for script in ['blowout', 'comfortable', 'tossup', 'close_game', 'neutral']:
            script_df = df[df['game_script_tier'] == script]
            if len(script_df) > 0:
                results[script] = self.calculate_segment_metrics(script_df)
        return results

    def analyze_by_rest_status(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze performance by rest status."""
        results = {}

        # B2B
        b2b_df = df[df['is_b2b'] == 1]
        results['B2B'] = self.calculate_segment_metrics(b2b_df)

        # Rested (3+ days)
        rested_df = df[(df['days_rest'].notna()) & (df['days_rest'] >= 3)]
        results['RESTED'] = self.calculate_segment_metrics(rested_df)

        # Normal (2 days)
        normal_df = df[(df['days_rest'].notna()) & (df['days_rest'] == 2)]
        results['NORMAL'] = self.calculate_segment_metrics(normal_df)

        return results

    def analyze_by_position_factor(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze performance by position matchup factor."""
        results = {}

        # Favorable (factor > 1.0)
        favorable_df = df[df['position_matchup_factor'] > 1.0]
        results['FAVORABLE'] = self.calculate_segment_metrics(favorable_df)

        # Neutral (factor ~= 1.0)
        neutral_df = df[(df['position_matchup_factor'] >= 0.97) &
                        (df['position_matchup_factor'] <= 1.03)]
        results['NEUTRAL'] = self.calculate_segment_metrics(neutral_df)

        # Unfavorable (factor < 1.0)
        unfavorable_df = df[df['position_matchup_factor'] < 0.97]
        results['UNFAVORABLE'] = self.calculate_segment_metrics(unfavorable_df)

        return results

    def analyze_all_segments(
        self,
        date_from: str,
        date_to: str,
        verbose: bool = False
    ) -> Dict[str, Dict]:
        """
        Run all segment analyses.

        Returns:
            Dict with keys: 'role', 'game_script', 'rest_status', 'position_factor'
        """
        df = self.load_data(date_from, date_to)

        if df.empty:
            if verbose:
                print("No data found for date range")
            return {}

        if verbose:
            print(f"Analyzing {len(df)} predictions...")

        results = {
            'role': self.analyze_by_role(df),
            'game_script': self.analyze_by_game_script(df),
            'rest_status': self.analyze_by_rest_status(df),
            'position_factor': self.analyze_by_position_factor(df),
            'overall': self.calculate_segment_metrics(df),
        }

        return results

    def generate_segment_report(
        self,
        date_from: str,
        date_to: str
    ) -> str:
        """Generate markdown report of segment analysis."""
        results = self.analyze_all_segments(date_from, date_to)

        if not results:
            return "No data available for analysis."

        lines = [
            f"# Segment Analysis Report",
            f"",
            f"**Date Range:** {date_from} to {date_to}",
            f"**Total Predictions:** {results['overall']['count']}",
            f"**Overall MAE:** {results['overall']['mae']:.3f}",
            f"",
        ]

        # Role tier table
        lines.extend([
            "## By Role Tier",
            "",
            "| Role | Count | MAE | Bias | Ceiling Hit |",
            "|------|-------|-----|------|-------------|",
        ])
        for role, metrics in results['role'].items():
            if metrics['count'] > 0:
                lines.append(
                    f"| {role} | {metrics['count']} | {metrics['mae']:.3f} | "
                    f"{metrics['bias']:+.2f} | {metrics['ceiling_hit_rate']:.1%} |"
                )
        lines.append("")

        # Game script table
        lines.extend([
            "## By Game Script",
            "",
            "| Script | Count | MAE | Bias | Ceiling Hit |",
            "|--------|-------|-----|------|-------------|",
        ])
        for script, metrics in results['game_script'].items():
            if metrics['count'] > 0:
                lines.append(
                    f"| {script} | {metrics['count']} | {metrics['mae']:.3f} | "
                    f"{metrics['bias']:+.2f} | {metrics['ceiling_hit_rate']:.1%} |"
                )
        lines.append("")

        # Rest status table
        lines.extend([
            "## By Rest Status",
            "",
            "| Status | Count | MAE | Bias | Mean Actual |",
            "|--------|-------|-----|------|-------------|",
        ])
        for status, metrics in results['rest_status'].items():
            if metrics['count'] > 0:
                lines.append(
                    f"| {status} | {metrics['count']} | {metrics['mae']:.3f} | "
                    f"{metrics['bias']:+.2f} | {metrics['mean_actual']:.1f} |"
                )
        lines.append("")

        # Position factor table
        lines.extend([
            "## By Position Matchup",
            "",
            "| Matchup | Count | MAE | Bias | Mean Actual |",
            "|---------|-------|-----|------|-------------|",
        ])
        for matchup, metrics in results['position_factor'].items():
            if metrics['count'] > 0:
                lines.append(
                    f"| {matchup} | {metrics['count']} | {metrics['mae']:.3f} | "
                    f"{metrics['bias']:+.2f} | {metrics['mean_actual']:.1f} |"
                )

        return '\n'.join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Segment analysis')
    parser.add_argument('--from', dest='date_from', required=True)
    parser.add_argument('--to', dest='date_to', required=True)
    parser.add_argument('--db', default='nba_stats.db')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    analyzer = SegmentAnalyzer(conn)
    report = analyzer.generate_segment_report(args.date_from, args.date_to)
    print(report)
    conn.close()
