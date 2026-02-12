#!/usr/bin/env python3
"""
NBA Daily - Local REST API Server

Read-only FastAPI server exposing model accuracy data from nba_stats.db.
Provides programmatic access to prediction accuracy, DFS tracking,
and backtest results that are otherwise locked behind the Streamlit UI.

Usage:
    python api_server.py                    # Start on default port 8000
    python api_server.py --port 9000        # Custom port
    python api_server.py --db other.db      # Custom DB path

Swagger docs: http://127.0.0.1:8000/docs
"""

import argparse
import math
import os
import sqlite3
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Reuse existing analysis functions (light dependency chain)
# ---------------------------------------------------------------------------
from prediction_tracking import (
    AccuracyMetrics,
    calculate_accuracy_metrics,
    get_best_worst_predictions,
    get_fanduel_comparison_summary,
    get_overall_model_performance,
    get_predictions_vs_actuals,
)
from prediction_evaluation_metrics import (
    calculate_enhanced_metrics,
    get_metrics_by_player_tier,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nba_stats.db")
MAX_LIMIT = 1000
DEFAULT_LIMIT = 100

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NBA Daily - Model Accuracy API",
    description=(
        "Read-only REST API for querying NBA prediction accuracy, "
        "DFS slate tracking, and backtest results from the NBA_Daily model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database dependency
# ---------------------------------------------------------------------------
_db_path: str = DEFAULT_DB


def get_db():
    """Yield a read-only SQLite connection per request."""
    conn = sqlite3.connect(f"file:{_db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))


def _rows_to_dicts(rows: list) -> List[Dict[str, Any]]:
    """Convert sqlite3.Row objects to plain dicts."""
    return [dict(row) for row in rows]


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types to Python natives and replace NaN/Infinity with None."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def _df_to_records(df) -> List[Dict[str, Any]]:
    """Convert a pandas DataFrame to a JSON-safe list of dicts."""
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return _sanitize_for_json(records)


# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 1: Health & Meta
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health & Meta"])
def health(conn=Depends(get_db)):
    """Database status, table list, and record counts."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    counts = {}
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        counts[table] = cursor.fetchone()[0]

    return {
        "status": "ok",
        "database": os.path.basename(_db_path),
        "tables": tables,
        "record_counts": counts,
    }


@app.get("/dates/predictions", tags=["Health & Meta"])
def prediction_dates(conn=Depends(get_db)):
    """All distinct game dates that have predictions."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT game_date FROM predictions ORDER BY game_date DESC"
    )
    dates = [row[0] for row in cursor.fetchall()]
    return {"count": len(dates), "dates": dates}


@app.get("/dates/dfs-slates", tags=["Health & Meta"])
def dfs_slate_dates(conn=Depends(get_db)):
    """All DFS slate dates."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT slate_date FROM dfs_slate_projections ORDER BY slate_date DESC"
    )
    dates = [row[0] for row in cursor.fetchall()]
    return {"count": len(dates), "dates": dates}


@app.get("/dates/backtests", tags=["Health & Meta"])
def backtest_dates(conn=Depends(get_db)):
    """All backtest slate dates."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT slate_date FROM backtest_daily_results ORDER BY slate_date DESC"
    )
    dates = [row[0] for row in cursor.fetchall()]
    return {"count": len(dates), "dates": dates}


# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 2: Prediction Accuracy
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/predictions", tags=["Prediction Accuracy"])
def list_predictions(
    date: Optional[str] = Query(None, description="Single game_date (YYYY-MM-DD)"),
    start_date: Optional[str] = Query(None, description="Start date inclusive"),
    end_date: Optional[str] = Query(None, description="End date inclusive"),
    player_name: Optional[str] = Query(None, description="Filter by player name (LIKE match)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    conn=Depends(get_db),
):
    """Raw predictions with flexible filtering and pagination."""
    query = """
        SELECT prediction_id, game_date, player_name, team_name, opponent_name,
               projected_ppg, actual_ppg, error, abs_error, proj_confidence,
               proj_floor, proj_ceiling, hit_floor_ceiling,
               dfs_score, dfs_grade, analytics_used, did_play
        FROM predictions WHERE 1=1
    """
    params: list = []

    if date:
        query += " AND game_date = ?"
        params.append(date)
    if start_date:
        query += " AND game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND game_date <= ?"
        params.append(end_date)
    if player_name:
        query += " AND player_name LIKE ?"
        params.append(f"%{player_name}%")
    if team:
        query += " AND team_name = ?"
        params.append(team)

    query += " ORDER BY game_date DESC, dfs_score DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"count": len(rows), "limit": limit, "offset": offset, "data": rows})


@app.get("/predictions/{game_date}", tags=["Prediction Accuracy"])
def predictions_vs_actuals(game_date: str, conn=Depends(get_db)):
    """Predictions vs actuals for a specific game date (reuses prediction_tracking)."""
    df = get_predictions_vs_actuals(conn, game_date)
    records = _df_to_records(df)
    return {"game_date": game_date, "count": len(records), "data": records}


@app.get("/accuracy/metrics", tags=["Prediction Accuracy"])
def accuracy_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_confidence: Optional[float] = None,
    conn=Depends(get_db),
):
    """Aggregate MAE, RMSE, hit rate, and confidence-split metrics."""
    metrics: AccuracyMetrics = calculate_accuracy_metrics(
        conn, start_date, end_date, min_confidence
    )
    return _sanitize_for_json(asdict(metrics))


@app.get("/accuracy/model-performance", tags=["Prediction Accuracy"])
def model_performance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=Depends(get_db),
):
    """Overall model performance summary."""
    result = get_overall_model_performance(conn, start_date, end_date)
    return _sanitize_for_json(result)


@app.get("/accuracy/enhanced", tags=["Prediction Accuracy"])
def enhanced_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=Depends(get_db),
):
    """Spearman/Pearson correlation, calibration, outlier analysis."""
    result = calculate_enhanced_metrics(conn, start_date=start_date, end_date=end_date)
    return _sanitize_for_json(result)


@app.get("/accuracy/by-tier", tags=["Prediction Accuracy"])
def metrics_by_tier(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=Depends(get_db),
):
    """Accuracy broken down by player tier (Role/Bench/Starter/Star/Superstar)."""
    df = get_metrics_by_player_tier(conn, start_date, end_date)
    return {"tiers": _df_to_records(df)}


@app.get("/accuracy/best-worst", tags=["Prediction Accuracy"])
def best_worst(
    n: int = Query(10, ge=1, le=100),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=Depends(get_db),
):
    """Best and worst prediction misses (by absolute error)."""
    result = get_best_worst_predictions(conn, n, start_date, end_date)
    return {
        "best": _df_to_records(result.get("best")),
        "worst": _df_to_records(result.get("worst")),
    }


@app.get("/accuracy/fanduel-comparison", tags=["Prediction Accuracy"])
def fanduel_comparison(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=Depends(get_db),
):
    """Model vs FanDuel accuracy comparison."""
    result = get_fanduel_comparison_summary(conn, start_date, end_date)
    return _sanitize_for_json(result)


@app.get("/accuracy/daily-summary", tags=["Prediction Accuracy"])
def daily_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=Depends(get_db),
):
    """Per-date accuracy summary (MAE, hit rate, prediction count)."""
    query = """
        SELECT
            game_date,
            COUNT(*) as total_predictions,
            COUNT(actual_ppg) as with_actuals,
            ROUND(AVG(abs_error), 2) as mae,
            ROUND(AVG(CASE WHEN hit_floor_ceiling = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as hit_rate_pct,
            SUM(CASE WHEN error < 0 THEN 1 ELSE 0 END) as over_projected,
            SUM(CASE WHEN error > 0 THEN 1 ELSE 0 END) as under_projected
        FROM predictions
        WHERE actual_ppg IS NOT NULL
    """
    params: list = []
    if start_date:
        query += " AND game_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND game_date <= ?"
        params.append(end_date)

    query += " GROUP BY game_date ORDER BY game_date DESC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"count": len(rows), "data": rows})


# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 3: DFS Tracking (raw SQL — avoids heavy dfs_optimizer import)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/dfs/slate-results", tags=["DFS Tracking"])
def dfs_slate_results(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    conn=Depends(get_db),
):
    """Aggregate DFS slate metrics (MAE, correlation, lineup efficiency)."""
    query = "SELECT * FROM dfs_slate_results WHERE 1=1"
    params: list = []
    if start_date:
        query += " AND slate_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND slate_date <= ?"
        params.append(end_date)
    query += " ORDER BY slate_date DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"count": len(rows), "limit": limit, "offset": offset, "data": rows})


@app.get("/dfs/slate-results/{slate_date}", tags=["DFS Tracking"])
def dfs_slate_result_single(slate_date: str, conn=Depends(get_db)):
    """Single slate MAE, correlation, and lineup efficiency."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dfs_slate_results WHERE slate_date = ?", (slate_date,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No DFS results for {slate_date}")
    return _sanitize_for_json(dict(row))


@app.get("/dfs/projections/{slate_date}", tags=["DFS Tracking"])
def dfs_projections(
    slate_date: str,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    conn=Depends(get_db),
):
    """Player projections vs actuals for a DFS slate."""
    query = """
        SELECT slate_date, player_name, team, opponent, salary, positions,
               proj_fpts, proj_points, proj_rebounds, proj_assists,
               proj_steals, proj_blocks, proj_turnovers, proj_fg3m,
               proj_floor, proj_ceiling, fpts_per_dollar,
               actual_fpts, actual_points, actual_rebounds, actual_assists,
               actual_steals, actual_blocks, actual_turnovers, actual_fg3m,
               actual_minutes, did_play, ownership_proj
        FROM dfs_slate_projections
        WHERE slate_date = ? AND did_play = 1
        ORDER BY actual_fpts DESC
        LIMIT ? OFFSET ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (slate_date, limit, offset))
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"slate_date": slate_date, "count": len(rows), "data": rows})


@app.get("/dfs/lineups/{slate_date}", tags=["DFS Tracking"])
def dfs_lineups(slate_date: str, conn=Depends(get_db)):
    """Lineup performance for a DFS slate."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM dfs_slate_lineups WHERE slate_date = ? ORDER BY lineup_num",
        (slate_date,),
    )
    rows = _rows_to_dicts(cursor.fetchall())
    if not rows:
        raise HTTPException(status_code=404, detail=f"No lineups for {slate_date}")
    return _sanitize_for_json({"slate_date": slate_date, "count": len(rows), "data": rows})


@app.get("/dfs/pending-slates", tags=["DFS Tracking"])
def dfs_pending_slates(conn=Depends(get_db)):
    """Slates that have projections but no results yet (awaiting actuals)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT p.slate_date, COUNT(*) as player_count
        FROM dfs_slate_projections p
        LEFT JOIN dfs_slate_results r ON p.slate_date = r.slate_date
        WHERE r.slate_date IS NULL
        GROUP BY p.slate_date
        ORDER BY p.slate_date DESC
    """)
    rows = _rows_to_dicts(cursor.fetchall())
    return {"count": len(rows), "data": rows}


@app.get("/dfs/bias-by-team", tags=["DFS Tracking"])
def dfs_bias_by_team(
    days: int = Query(30, ge=1, le=365, description="Lookback window in days"),
    conn=Depends(get_db),
):
    """Projection bias by opponent team (avg error, sample size)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            opponent,
            COUNT(*) as sample_size,
            ROUND(AVG(proj_fpts - actual_fpts), 2) as avg_bias,
            ROUND(AVG(ABS(proj_fpts - actual_fpts)), 2) as mae,
            ROUND(AVG(actual_fpts), 1) as avg_actual_fpts
        FROM dfs_slate_projections
        WHERE did_play = 1
          AND actual_fpts IS NOT NULL
          AND slate_date >= date('now', ? || ' days')
        GROUP BY opponent
        HAVING COUNT(*) >= 5
        ORDER BY avg_bias DESC
    """, (f"-{days}",))
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"days": days, "count": len(rows), "data": rows})


# ═══════════════════════════════════════════════════════════════════════════
#  GROUP 4: Backtests (raw SQL)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/backtest/top3", tags=["Backtests"])
def backtest_top3(
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    conn=Depends(get_db),
):
    """Top-3 scorer daily backtest results."""
    query = "SELECT * FROM backtest_daily_results WHERE 1=1"
    params: list = []
    if strategy:
        query += " AND strategy_name = ?"
        params.append(strategy)
    if start_date:
        query += " AND slate_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND slate_date <= ?"
        params.append(end_date)
    query += " ORDER BY slate_date DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"count": len(rows), "limit": limit, "offset": offset, "data": rows})


@app.get("/backtest/top3/summary", tags=["Backtests"])
def backtest_top3_summary(
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn=Depends(get_db),
):
    """Hit rates and overlap stats for top-3 backtest."""
    query = """
        SELECT
            strategy_name,
            COUNT(*) as total_slates,
            ROUND(AVG(overlap) * 1.0, 2) as avg_overlap,
            SUM(hit_any) as hit_any_count,
            ROUND(AVG(hit_any) * 100, 1) as hit_any_pct,
            SUM(hit_2plus) as hit_2plus_count,
            ROUND(AVG(hit_2plus) * 100, 1) as hit_2plus_pct,
            SUM(hit_exact) as hit_exact_count,
            ROUND(AVG(hit_exact) * 100, 1) as hit_exact_pct,
            ROUND(AVG(tie_friendly_overlap), 2) as avg_tie_friendly_overlap,
            ROUND(AVG(avg_rank_actual_top3), 1) as avg_rank_of_actual_top3,
            ROUND(AVG(closest_miss), 1) as avg_closest_miss
        FROM backtest_daily_results
        WHERE 1=1
    """
    params: list = []
    if strategy:
        query += " AND strategy_name = ?"
        params.append(strategy)
    if start_date:
        query += " AND slate_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND slate_date <= ?"
        params.append(end_date)
    query += " GROUP BY strategy_name ORDER BY hit_any_pct DESC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"data": rows})


@app.get("/backtest/top3/strategies", tags=["Backtests"])
def backtest_strategies(conn=Depends(get_db)):
    """List all available backtest strategies and their slate counts."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT strategy_name, COUNT(*) as slate_count,
               MIN(slate_date) as first_date, MAX(slate_date) as last_date
        FROM backtest_daily_results
        GROUP BY strategy_name
        ORDER BY strategy_name
    """)
    rows = _rows_to_dicts(cursor.fetchall())
    return {"strategies": rows}


@app.get("/backtest/portfolio", tags=["Backtests"])
def backtest_portfolio(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    conn=Depends(get_db),
):
    """Portfolio backtest results (lineup performance per slate)."""
    query = "SELECT * FROM backtest_portfolio_results WHERE 1=1"
    params: list = []
    if start_date:
        query += " AND slate_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND slate_date <= ?"
        params.append(end_date)
    query += " ORDER BY slate_date DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = _rows_to_dicts(cursor.fetchall())
    return _sanitize_for_json({"count": len(rows), "limit": limit, "offset": offset, "data": rows})


@app.get("/backtest/portfolio/summary", tags=["Backtests"])
def backtest_portfolio_summary(conn=Depends(get_db)):
    """Portfolio backtest aggregate stats: win rate, shortfall, rank percentile."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*) as total_slates,
            SUM(won_slate) as wins,
            ROUND(AVG(won_slate) * 100, 1) as win_rate_pct,
            ROUND(AVG(sum_shortfall), 1) as avg_shortfall,
            ROUND(AVG(rank_percentile), 1) as avg_rank_percentile,
            ROUND(AVG(best_lineup_sum), 1) as avg_best_lineup_pts,
            ROUND(AVG(optimal_lineup_sum), 1) as avg_optimal_lineup_pts,
            ROUND(AVG(best_lineup_sum) / NULLIF(AVG(optimal_lineup_sum), 0) * 100, 1) as lineup_efficiency_pct,
            SUM(hit_2_of_top3_any) as hit_2_of_top3_count,
            ROUND(AVG(hit_2_of_top3_any) * 100, 1) as hit_2_of_top3_pct,
            SUM(hit_all_top3_any) as hit_all_top3_count,
            ROUND(AVG(hit_all_top3_any) * 100, 1) as hit_all_top3_pct,
            MIN(slate_date) as first_date,
            MAX(slate_date) as last_date
        FROM backtest_portfolio_results
    """)
    row = cursor.fetchone()
    if not row or row[0] == 0:
        return {"data": None, "message": "No portfolio backtest data found"}
    return _sanitize_for_json(dict(row))


# ═══════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Daily API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite database")
    args = parser.parse_args()

    _db_path = args.db
    print(f"Starting NBA Daily API Server")
    print(f"  Database: {_db_path}")
    print(f"  Swagger:  http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port)
