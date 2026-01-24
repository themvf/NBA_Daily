"""
Evaluation Module for NBA Prediction Enrichments

This module provides tools for:
- Ablation studies comparing enrichment variants
- Segment analysis (by role, game script, rest status)
- Calibration bucket analysis
- Report generation

Usage:
    from evaluation import AblationBacktester, generate_model_health_report

    backtest = AblationBacktester(conn)
    results = backtest.run_full_ablation('2025-12-01', '2025-12-31')
    generate_model_health_report(results, 'model_health.md')
"""

from .ablation_backtest import AblationBacktester, run_ablation_study
from .segment_analysis import SegmentAnalyzer
from .report_generator import generate_model_health_report

__all__ = [
    'AblationBacktester',
    'run_ablation_study',
    'SegmentAnalyzer',
    'generate_model_health_report',
]
