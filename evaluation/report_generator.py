#!/usr/bin/env python3
"""
Model Health Report Generator

Generates comprehensive markdown reports combining:
- Ablation study results
- Segment analysis
- Alert status
- Hypothesis validation
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

from .ablation_backtest import AblationBacktester, AblationResults
from .segment_analysis import SegmentAnalyzer


def generate_model_health_report(
    conn: sqlite3.Connection,
    date_from: str = None,
    date_to: str = None,
    output_path: str = None
) -> str:
    """
    Generate comprehensive model health report.

    Args:
        conn: Database connection
        date_from: Start date (defaults to 14 days ago)
        date_to: End date (defaults to yesterday)
        output_path: Optional file path to save report

    Returns:
        Markdown report string
    """
    if date_to is None:
        date_to = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    if date_from is None:
        date_from = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

    lines = [
        "# Model Health Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Date Range:** {date_from} to {date_to}",
        "",
        "---",
        "",
    ]

    # 1. Ablation Study Summary
    lines.extend([
        "## 1. Enrichment Impact (Ablation Study)",
        "",
    ])

    try:
        backtest = AblationBacktester(conn)
        ablation_results = backtest.run_full_ablation(date_from, date_to, verbose=False)

        lines.extend([
            "| Variant | MAE | vs Baseline | Ceiling Hit | Top 10 Capture |",
            "|---------|-----|-------------|-------------|----------------|",
        ])

        for name, result in ablation_results.variants.items():
            vs_baseline = f"{result.mae_improvement_pct:+.1f}%" if result.mae_improvement_pct else "-"
            lines.append(
                f"| {name} | {result.mae:.3f} | {vs_baseline} | "
                f"{result.ceiling_hit_rate:.1%} | {result.top_decile_capture:.1%} |"
            )

        lines.extend([
            "",
            f"**Best Variant:** {ablation_results.best_variant}",
            "",
        ])

        # Enrichment effectiveness summary
        full_result = ablation_results.variants.get('full')
        baseline_result = ablation_results.variants.get('baseline')

        if full_result and baseline_result:
            improvement = (baseline_result.mae - full_result.mae) / baseline_result.mae * 100
            if improvement > 0:
                lines.append(f"Enrichments reduce MAE by **{improvement:.1f}%**")
            else:
                lines.append(f"Enrichments increase MAE by **{abs(improvement):.1f}%** (investigate)")

    except Exception as e:
        lines.append(f"*Ablation study failed: {e}*")

    lines.append("")

    # 2. Segment Analysis
    lines.extend([
        "---",
        "",
        "## 2. Segment Performance",
        "",
    ])

    try:
        analyzer = SegmentAnalyzer(conn)
        segment_results = analyzer.analyze_all_segments(date_from, date_to)

        # Role tier summary
        lines.extend([
            "### By Role Tier",
            "",
            "| Role | Count | MAE | Bias |",
            "|------|-------|-----|------|",
        ])

        for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
            if role in segment_results.get('role', {}):
                r = segment_results['role'][role]
                if r['count'] > 0:
                    lines.append(f"| {role} | {r['count']} | {r['mae']:.3f} | {r['bias']:+.2f} |")

        lines.append("")

        # Rest status summary
        lines.extend([
            "### By Rest Status",
            "",
            "| Status | Count | MAE | Mean Actual |",
            "|--------|-------|-----|-------------|",
        ])

        for status in ['B2B', 'NORMAL', 'RESTED']:
            if status in segment_results.get('rest_status', {}):
                r = segment_results['rest_status'][status]
                if r['count'] > 0:
                    lines.append(f"| {status} | {r['count']} | {r['mae']:.3f} | {r['mean_actual']:.1f} |")

    except Exception as e:
        lines.append(f"*Segment analysis failed: {e}*")

    lines.append("")

    # 3. Alert Status
    lines.extend([
        "---",
        "",
        "## 3. Active Alerts",
        "",
    ])

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT alert_type, severity, message, created_at
            FROM enrichment_alerts
            WHERE is_active = 1
            ORDER BY severity DESC, created_at DESC
        """)
        alerts = cursor.fetchall()

        if alerts:
            lines.extend([
                "| Severity | Alert | Message | Created |",
                "|----------|-------|---------|---------|",
            ])
            for alert in alerts:
                lines.append(f"| {alert[1]} | {alert[0]} | {alert[2]} | {alert[3][:10]} |")
        else:
            lines.append("No active alerts.")

    except Exception as e:
        lines.append(f"*Alert check failed: {e}*")

    lines.append("")

    # 4. Hypothesis Validation Status
    lines.extend([
        "---",
        "",
        "## 4. Hypothesis Validation Status",
        "",
        "| ID | Hypothesis | Status | Evidence |",
        "|-----|-----------|--------|----------|",
    ])

    # Check key hypotheses based on segment results
    try:
        # R1: B2B effect
        if 'rest_status' in segment_results:
            b2b = segment_results['rest_status'].get('B2B', {})
            normal = segment_results['rest_status'].get('NORMAL', {})
            if b2b.get('count', 0) > 10 and normal.get('count', 0) > 10:
                effect = (b2b['mean_actual'] - normal['mean_actual']) / normal['mean_actual']
                status = "CONFIRMED" if -0.10 < effect < -0.03 else "NEEDS_REVIEW"
                lines.append(f"| R1 | B2B decreases PPG ~8% | {status} | Effect: {effect:.1%} |")

        # D1: Role correlation with minutes
        if 'role' in segment_results:
            star = segment_results['role'].get('STAR', {})
            bench = segment_results['role'].get('BENCH', {})
            if star.get('mae') and bench.get('mae'):
                status = "CONFIRMED" if star['mae'] < bench['mae'] else "NEEDS_REVIEW"
                lines.append(f"| D2 | STARs more stable | {status} | MAE: {star['mae']:.2f} vs {bench['mae']:.2f} |")

    except Exception as e:
        lines.append(f"| - | Validation failed | ERROR | {e} |")

    lines.extend([
        "",
        "---",
        "",
        f"*Report generated by evaluation/report_generator.py*",
    ])

    report = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate model health report')
    parser.add_argument('--from', dest='date_from', help='Start date')
    parser.add_argument('--to', dest='date_to', help='End date')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--db', default='nba_stats.db')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    report = generate_model_health_report(
        conn,
        date_from=args.date_from,
        date_to=args.date_to,
        output_path=args.output
    )
    print(report)
    conn.close()
