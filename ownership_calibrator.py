#!/usr/bin/env python3
"""Build and fit a DFS ownership calibration dataset.

The goal is to learn toward actual ownership while treating third-party
ownership sources as explanatory features, not as the prediction target.

This module exposes two main entry points:
    - build_ownership_training_dataset(...)
    - fit_ownership_calibrator(...)

It can also be run directly as a CLI to export the training frame and save
calibration summaries back into SQLite.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import token_hex
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - fallback only used if dependency is missing
    XGBRegressor = None
    HAS_XGBOOST = False


MODEL_VERSION = (
    "ownership_calibrator_v2_xgboost"
    if HAS_XGBOOST
    else "ownership_calibrator_v2_ridge_fallback"
)
SOURCE_SHRINK_K = 250.0

BASE_NUMERIC_FEATURES: Tuple[str, ...] = (
    "salary",
    "salary_k",
    "proj_fpts",
    "proj_floor",
    "proj_ceiling",
    "ceiling_gap",
    "floor_gap",
    "fpts_per_dollar",
    "our_ownership_proj",
    "projected_ppg",
    "proj_confidence",
    "season_avg_ppg",
    "recent_avg_3",
    "recent_avg_5",
    "dfs_score",
    "proj_minutes",
    "l5_minutes_avg",
    "minutes_confidence",
    "role_change",
    "p_top1",
    "p_top3",
    "sim_sigma",
    "vegas_implied_fpts",
    "vegas_vs_proj_diff",
    "vegas_edge_fpts",
    "game_total",
    "game_spread_abs",
    "game_pace_score",
    "game_blowout_risk",
    "game_stack_score",
    "slate_game_count",
    "slate_player_count",
    "proj_fpts_rank_pct",
    "value_rank_pct",
    "salary_rank_pct",
    "our_ownership_rank_pct",
    "game_total_rank_pct",
    "injury_adjusted",
    "opponent_injury_detected",
    "opponent_def_rating",
    "opponent_pace",
    "flag_pace",
    "flag_defense",
    "flag_injury",
    "flag_rest",
)
BASE_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "primary_position",
    "salary_tier",
    "ownership_tier",
    "role_tier",
)
SOURCE_NUMERIC_FEATURES: Tuple[str, ...] = (
    "salary",
    "proj_fpts",
    "our_ownership_proj",
    "proj_confidence",
    "game_total",
    "rw_present",
    "ls_present",
    "supplement_source_count",
    "rw_own_pct",
    "ls_own_pct",
    "supplement_own_consensus",
    "rw_our_own_gap",
    "ls_our_own_gap",
    "rw_ls_own_gap",
    "supplement_consensus_own_gap",
    "rw_proj_fpts",
    "ls_proj_fpts",
    "supplement_proj_consensus",
    "rw_our_proj_gap",
    "ls_our_proj_gap",
    "rw_ls_proj_gap",
    "supplement_consensus_proj_gap",
    "both_sources_zero_own",
    "both_sources_zero_proj",
    "rw_match_score",
    "ls_match_score",
    "source_match_score_avg",
)
SOURCE_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "primary_position",
    "role_tier",
)
BENCHMARK_MODEL_SPECS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("raw_our", "Raw Our", "our_ownership_proj", False),
    ("source_consensus", "Source Consensus", "supplement_own_consensus", True),
    ("base_model", "Base Model", "base_ownership_pred", False),
    ("final_model", "Final Model", "final_ownership_pred", False),
)


@dataclass
class OwnershipCalibrationResult:
    """Container for the fitted calibration summary."""

    run_key: str
    model_version: str
    fitted_at: str
    total_rows: int
    train_rows: int
    test_rows: int
    source_rows_train: int
    source_rows_test: int
    train_start_date: Optional[str]
    train_end_date: Optional[str]
    holdout_start_date: Optional[str]
    holdout_end_date: Optional[str]
    feature_count_base: int
    feature_count_source: int
    base_metrics: Dict[str, Any]
    final_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    artifact: Dict[str, Any]


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return bool(row)


def _normalize_source_bucket(source_name: object) -> str:
    raw = str(source_name or "").strip().lower()
    if "rotowire" in raw:
        return "rotowire"
    if "lineupstarter" in raw or "linestarter" in raw or "line starter" in raw:
        return "lineupstarter"
    return ""


def _primary_position(raw_positions: object) -> str:
    raw = str(raw_positions or "").upper().replace("-", "/").replace(",", "/")
    if not raw:
        return "UNK"
    valid = {"PG", "SG", "SF", "PF", "C"}
    alias_map = {
        "G": "PG",
        "GUARD": "PG",
        "F": "PF",
        "FORWARD": "PF",
        "UTIL": "C",
        "U": "C",
    }
    for token in [piece.strip() for piece in raw.split("/") if piece.strip()]:
        if token in valid:
            return token
        if token in alias_map:
            return alias_map[token]
    return "UNK"


def _ensure_calibration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dfs_ownership_calibration_runs (
            run_key TEXT PRIMARY KEY,
            model_version TEXT NOT NULL,
            fitted_at TEXT NOT NULL,
            train_start_date TEXT,
            train_end_date TEXT,
            holdout_start_date TEXT,
            holdout_end_date TEXT,
            total_rows INTEGER,
            train_rows INTEGER,
            test_rows INTEGER,
            source_rows_train INTEGER,
            source_rows_test INTEGER,
            feature_count_base INTEGER,
            feature_count_source INTEGER,
            base_metrics_json TEXT,
            final_metrics_json TEXT,
            baseline_metrics_json TEXT,
            artifact_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_docr_fitted_at
        ON dfs_ownership_calibration_runs(fitted_at)
        """
    )
    conn.commit()


def _read_base_ownership_rows(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if not _table_exists(conn, "dfs_slate_projections"):
        return pd.DataFrame()

    filters = ["p.salary > 0"]
    params: List[Any] = []
    if start_date:
        filters.append("date(p.slate_date) >= date(?)")
        params.append(start_date)
    if end_date:
        filters.append("date(p.slate_date) <= date(?)")
        params.append(end_date)
    where_clause = " AND ".join(filters)

    contest_cte = (
        "contest_slates AS (SELECT DISTINCT slate_date FROM dfs_contest_entries)"
        if _table_exists(conn, "dfs_contest_entries")
        else "contest_slates AS (SELECT NULL AS slate_date WHERE 1 = 0)"
    )

    query = f"""
        WITH {contest_cte}
        SELECT
            date(p.slate_date) AS slate_date,
            p.player_id,
            p.player_name,
            p.team,
            p.opponent,
            p.salary,
            p.positions,
            p.proj_fpts,
            p.proj_floor,
            p.proj_ceiling,
            p.fpts_per_dollar,
            p.ownership_proj AS our_ownership_proj,
            p.actual_ownership AS recorded_actual_ownership,
            CASE
                WHEN cs.slate_date IS NOT NULL THEN COALESCE(p.actual_ownership, 0.0)
                ELSE p.actual_ownership
            END AS target_actual_ownership,
            CASE
                WHEN cs.slate_date IS NOT NULL AND p.actual_ownership IS NULL THEN 1
                ELSE 0
            END AS actual_ownership_imputed_zero,
            CASE WHEN cs.slate_date IS NOT NULL THEN 1 ELSE 0 END AS contest_imported,
            p.actual_fpts,
            p.actual_minutes,
            COALESCE(p.did_play, 0) AS did_play
        FROM dfs_slate_projections p
        LEFT JOIN contest_slates cs
            ON cs.slate_date = p.slate_date
        WHERE {where_clause}
          AND (
              p.actual_ownership IS NOT NULL
              OR cs.slate_date IS NOT NULL
          )
    """
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["target_actual_ownership"] = pd.to_numeric(
        df["target_actual_ownership"], errors="coerce"
    )
    df = df.dropna(subset=["target_actual_ownership"]).copy()
    df["target_actual_ownership"] = df["target_actual_ownership"].clip(0.0, 100.0)
    return df


def _read_predictions_frame(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if not _table_exists(conn, "predictions"):
        return pd.DataFrame()

    filters = []
    params: List[Any] = []
    if start_date:
        filters.append("date(game_date) >= date(?)")
        params.append(start_date)
    if end_date:
        filters.append("date(game_date) <= date(?)")
        params.append(end_date)
    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    query = f"""
        WITH ranked_predictions AS (
            SELECT
                date(game_date) AS slate_date,
                player_id,
                projected_ppg,
                proj_confidence,
                season_avg_ppg,
                recent_avg_3,
                recent_avg_5,
                dfs_score,
                proj_minutes,
                l5_minutes_avg,
                minutes_confidence,
                role_change,
                p_top1,
                p_top3,
                sim_sigma,
                role_tier,
                vegas_implied_fpts,
                vegas_vs_proj_diff,
                injury_adjusted,
                opponent_injury_detected,
                opponent_def_rating,
                opponent_pace,
                analytics_used,
                ROW_NUMBER() OVER (
                    PARTITION BY date(game_date), player_id
                    ORDER BY COALESCE(last_refreshed_at, created_at, prediction_date) DESC,
                             prediction_id DESC
                ) AS rn
            FROM predictions
            {where_clause}
        )
        SELECT
            slate_date,
            player_id,
            projected_ppg,
            proj_confidence,
            season_avg_ppg,
            recent_avg_3,
            recent_avg_5,
            dfs_score,
            proj_minutes,
            l5_minutes_avg,
            minutes_confidence,
            role_change,
            p_top1,
            p_top3,
            sim_sigma,
            role_tier,
            vegas_implied_fpts,
            vegas_vs_proj_diff,
            injury_adjusted,
            opponent_injury_detected,
            opponent_def_rating,
            opponent_pace,
            analytics_used
        FROM ranked_predictions
        WHERE rn = 1
    """
    return pd.read_sql_query(query, conn, params=params)


def _read_game_environment_frame(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if not _table_exists(conn, "game_odds"):
        return pd.DataFrame()

    filters = []
    params: List[Any] = []
    if start_date:
        filters.append("date(game_date) >= date(?)")
        params.append(start_date)
    if end_date:
        filters.append("date(game_date) <= date(?)")
        params.append(end_date)
    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    odds_df = pd.read_sql_query(
        f"""
        SELECT
            date(game_date) AS slate_date,
            home_team,
            away_team,
            total,
            spread,
            pace_score,
            blowout_risk,
            stack_score
        FROM game_odds
        {where_clause}
        """,
        conn,
        params=params,
    )
    if odds_df.empty:
        return odds_df

    rows: List[Dict[str, Any]] = []
    for row in odds_df.to_dict("records"):
        shared = {
            "slate_date": row.get("slate_date"),
            "game_total": pd.to_numeric(row.get("total"), errors="coerce"),
            "game_spread_abs": abs(float(row.get("spread") or 0.0)),
            "game_pace_score": pd.to_numeric(row.get("pace_score"), errors="coerce"),
            "game_blowout_risk": pd.to_numeric(row.get("blowout_risk"), errors="coerce"),
            "game_stack_score": pd.to_numeric(row.get("stack_score"), errors="coerce"),
        }
        rows.append(
            {
                **shared,
                "team": row.get("home_team"),
                "opponent": row.get("away_team"),
            }
        )
        rows.append(
            {
                **shared,
                "team": row.get("away_team"),
                "opponent": row.get("home_team"),
            }
        )
    return pd.DataFrame(rows)


def _read_latest_supplement_frame(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if not _table_exists(conn, "dfs_supplement_runs") or not _table_exists(
        conn, "dfs_supplement_player_deltas"
    ):
        return pd.DataFrame()

    filters = []
    params: List[Any] = []
    if start_date:
        filters.append("date(slate_date) >= date(?)")
        params.append(start_date)
    if end_date:
        filters.append("date(slate_date) <= date(?)")
        params.append(end_date)
    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    runs_df = pd.read_sql_query(
        f"""
        SELECT run_key, date(slate_date) AS slate_date, source_name, created_at
        FROM dfs_supplement_runs
        {where_clause}
        """,
        conn,
        params=params,
    )
    if runs_df.empty:
        return runs_df

    runs_df["source_bucket"] = runs_df["source_name"].map(_normalize_source_bucket)
    runs_df = runs_df[runs_df["source_bucket"] != ""].copy()
    if runs_df.empty:
        return runs_df

    runs_df["created_sort"] = pd.to_datetime(runs_df["created_at"], errors="coerce")
    runs_df = runs_df.sort_values(
        ["slate_date", "source_bucket", "created_sort", "run_key"],
        ascending=[True, True, False, False],
    )
    latest_runs = (
        runs_df.groupby(["slate_date", "source_bucket"], as_index=False)
        .head(1)
        .copy()
    )
    run_keys = latest_runs["run_key"].dropna().astype(str).tolist()
    if not run_keys:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(run_keys))
    deltas_df = pd.read_sql_query(
        f"""
        SELECT
            run_key,
            date(slate_date) AS slate_date,
            player_id,
            supplement_proj_fpts,
            supplement_own_pct,
            proj_delta,
            own_delta_pp,
            match_score
        FROM dfs_supplement_player_deltas
        WHERE run_key IN ({placeholders})
        """,
        conn,
        params=run_keys,
    )
    if deltas_df.empty:
        return deltas_df

    deltas_df = deltas_df.merge(
        latest_runs[["run_key", "source_bucket"]],
        on="run_key",
        how="left",
    )
    output_frames: List[pd.DataFrame] = []
    for source_bucket, prefix in (("rotowire", "rw"), ("lineupstarter", "ls")):
        source_df = deltas_df[deltas_df["source_bucket"] == source_bucket].copy()
        if source_df.empty:
            continue
        source_df = source_df[
            [
                "slate_date",
                "player_id",
                "supplement_proj_fpts",
                "supplement_own_pct",
                "proj_delta",
                "own_delta_pp",
                "match_score",
            ]
        ].rename(
            columns={
                "supplement_proj_fpts": f"{prefix}_proj_fpts",
                "supplement_own_pct": f"{prefix}_own_pct",
                "proj_delta": f"{prefix}_proj_delta",
                "own_delta_pp": f"{prefix}_own_delta_pp",
                "match_score": f"{prefix}_match_score",
            }
        )
        output_frames.append(source_df)

    if not output_frames:
        return pd.DataFrame()

    merged = output_frames[0]
    for frame in output_frames[1:]:
        merged = merged.merge(frame, on=["slate_date", "player_id"], how="outer")
    return merged


def _add_rank_features(df: pd.DataFrame, column: str, feature_key: str) -> None:
    if column not in df.columns:
        return
    numeric = pd.to_numeric(df[column], errors="coerce")
    df[column] = numeric
    df[f"{feature_key}_rank"] = df.groupby("slate_date")[column].rank(
        ascending=False,
        method="average",
    )
    df[f"{feature_key}_rank_pct"] = df.groupby("slate_date")[column].rank(
        ascending=False,
        method="average",
        pct=True,
    )


def _series_or_default(
    df: pd.DataFrame,
    column: str,
    default_value: Any,
) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default_value, index=df.index)


def _assemble_ownership_feature_frame(
    base_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame] = None,
    game_env_df: Optional[pd.DataFrame] = None,
    supplement_df: Optional[pd.DataFrame] = None,
    *,
    sort_with_target: bool = True,
) -> pd.DataFrame:
    """Build the shared ownership feature frame for training or live scoring."""
    if base_df is None or base_df.empty:
        return pd.DataFrame()

    df = base_df.copy()
    predictions_df = predictions_df if predictions_df is not None else pd.DataFrame()
    game_env_df = game_env_df if game_env_df is not None else pd.DataFrame()
    supplement_df = supplement_df if supplement_df is not None else pd.DataFrame()

    if not predictions_df.empty:
        df = df.merge(
            predictions_df,
            on=["slate_date", "player_id"],
            how="left",
        )
    if not game_env_df.empty:
        df = df.merge(
            game_env_df,
            on=["slate_date", "team", "opponent"],
            how="left",
        )
    if not supplement_df.empty:
        df = df.merge(
            supplement_df,
            on=["slate_date", "player_id"],
            how="left",
        )

    numeric_defaults = [
        "proj_fpts",
        "proj_floor",
        "proj_ceiling",
        "fpts_per_dollar",
        "our_ownership_proj",
        "recorded_actual_ownership",
        "actual_fpts",
        "actual_minutes",
        "projected_ppg",
        "proj_confidence",
        "season_avg_ppg",
        "recent_avg_3",
        "recent_avg_5",
        "dfs_score",
        "proj_minutes",
        "l5_minutes_avg",
        "minutes_confidence",
        "role_change",
        "p_top1",
        "p_top3",
        "sim_sigma",
        "vegas_implied_fpts",
        "vegas_vs_proj_diff",
        "injury_adjusted",
        "opponent_injury_detected",
        "opponent_def_rating",
        "opponent_pace",
        "game_total",
        "game_spread_abs",
        "game_pace_score",
        "game_blowout_risk",
        "game_stack_score",
        "rw_proj_fpts",
        "rw_own_pct",
        "rw_proj_delta",
        "rw_own_delta_pp",
        "rw_match_score",
        "ls_proj_fpts",
        "ls_own_pct",
        "ls_proj_delta",
        "ls_own_delta_pp",
        "ls_match_score",
    ]
    for column in numeric_defaults:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in [
        "rw_proj_fpts",
        "rw_own_pct",
        "rw_proj_delta",
        "rw_own_delta_pp",
        "rw_match_score",
        "ls_proj_fpts",
        "ls_own_pct",
        "ls_proj_delta",
        "ls_own_delta_pp",
        "ls_match_score",
        "vegas_implied_fpts",
        "game_total",
        "game_spread_abs",
        "game_pace_score",
        "game_blowout_risk",
        "game_stack_score",
        "projected_ppg",
        "proj_confidence",
        "season_avg_ppg",
        "recent_avg_3",
        "recent_avg_5",
        "dfs_score",
        "proj_minutes",
        "l5_minutes_avg",
        "minutes_confidence",
        "role_change",
        "p_top1",
        "p_top3",
        "sim_sigma",
        "injury_adjusted",
        "opponent_injury_detected",
        "opponent_def_rating",
        "opponent_pace",
    ]:
        if column not in df.columns:
            df[column] = np.nan
    for column, default_value in (("role_tier", "UNK"), ("analytics_used", "")):
        if column not in df.columns:
            df[column] = default_value

    df["primary_position"] = df["positions"].map(_primary_position)
    df["role_tier"] = df["role_tier"].fillna("UNK").astype(str).str.upper()
    df["salary_k"] = pd.to_numeric(df["salary"], errors="coerce") / 1000.0
    df["ceiling_gap"] = (
        pd.to_numeric(df["proj_ceiling"], errors="coerce")
        - pd.to_numeric(df["proj_fpts"], errors="coerce")
    )
    df["floor_gap"] = (
        pd.to_numeric(df["proj_fpts"], errors="coerce")
        - pd.to_numeric(df["proj_floor"], errors="coerce")
    )
    df["vegas_edge_fpts"] = (
        pd.to_numeric(_series_or_default(df, "vegas_implied_fpts", np.nan), errors="coerce")
        - pd.to_numeric(df["proj_fpts"], errors="coerce")
    )

    analytics_text = df["analytics_used"].fillna("").astype(str)
    df["flag_pace"] = analytics_text.str.contains("PACE", case=False, na=False).astype(int)
    df["flag_defense"] = analytics_text.str.contains("DEF", case=False, na=False).astype(int)
    df["flag_injury"] = analytics_text.str.contains("INJ", case=False, na=False).astype(int)
    df["flag_rest"] = analytics_text.str.contains("REST|B2B", case=False, na=False).astype(int)

    df["rw_present"] = df["rw_own_pct"].notna().astype(int)
    df["ls_present"] = df["ls_own_pct"].notna().astype(int)
    df["supplement_source_count"] = df["rw_present"] + df["ls_present"]

    df["supplement_own_consensus"] = df[["rw_own_pct", "ls_own_pct"]].mean(
        axis=1, skipna=True
    )
    df["supplement_proj_consensus"] = df[["rw_proj_fpts", "ls_proj_fpts"]].mean(
        axis=1, skipna=True
    )
    df["source_match_score_avg"] = df[["rw_match_score", "ls_match_score"]].mean(
        axis=1, skipna=True
    )

    df["rw_our_own_gap"] = df["rw_own_pct"] - df["our_ownership_proj"]
    df["ls_our_own_gap"] = df["ls_own_pct"] - df["our_ownership_proj"]
    df["rw_ls_own_gap"] = df["rw_own_pct"] - df["ls_own_pct"]
    df["supplement_consensus_own_gap"] = (
        df["supplement_own_consensus"] - df["our_ownership_proj"]
    )

    df["rw_our_proj_gap"] = df["rw_proj_fpts"] - df["proj_fpts"]
    df["ls_our_proj_gap"] = df["ls_proj_fpts"] - df["proj_fpts"]
    df["rw_ls_proj_gap"] = df["rw_proj_fpts"] - df["ls_proj_fpts"]
    df["supplement_consensus_proj_gap"] = (
        df["supplement_proj_consensus"] - df["proj_fpts"]
    )

    df["both_sources_zero_own"] = (
        (df["rw_present"] == 1)
        & (df["ls_present"] == 1)
        & (df["rw_own_pct"].fillna(0.0) <= 0.0)
        & (df["ls_own_pct"].fillna(0.0) <= 0.0)
    ).astype(int)
    df["both_sources_zero_proj"] = (
        df["rw_proj_fpts"].notna()
        & df["ls_proj_fpts"].notna()
        & (df["rw_proj_fpts"].fillna(0.0) <= 0.0)
        & (df["ls_proj_fpts"].fillna(0.0) <= 0.0)
    ).astype(int)

    slate_summary = (
        df.groupby("slate_date", observed=True)
        .agg(
            slate_player_count=("player_id", "count"),
            slate_team_count=("team", "nunique"),
        )
        .reset_index()
    )
    slate_summary["slate_game_count"] = (
        pd.to_numeric(slate_summary["slate_team_count"], errors="coerce") / 2.0
    )
    df = df.merge(slate_summary, on="slate_date", how="left")

    for source_column in (
        "rw_own_pct",
        "ls_own_pct",
        "supplement_own_consensus",
        "rw_our_own_gap",
        "ls_our_own_gap",
        "rw_ls_own_gap",
        "supplement_consensus_own_gap",
        "rw_proj_fpts",
        "ls_proj_fpts",
        "supplement_proj_consensus",
        "rw_our_proj_gap",
        "ls_our_proj_gap",
        "rw_ls_proj_gap",
        "supplement_consensus_proj_gap",
        "rw_match_score",
        "ls_match_score",
        "source_match_score_avg",
    ):
        if source_column in df.columns:
            df[source_column] = pd.to_numeric(df[source_column], errors="coerce").fillna(0.0)

    df["salary_tier"] = pd.cut(
        pd.to_numeric(df["salary"], errors="coerce"),
        bins=[0, 4000, 6000, 8000, np.inf],
        labels=["PUNT", "VALUE", "MID", "STUD"],
        include_lowest=True,
        right=False,
    ).astype(str)
    df["ownership_tier"] = pd.cut(
        pd.to_numeric(df["our_ownership_proj"], errors="coerce"),
        bins=[-0.1, 5.0, 15.0, 30.0, np.inf],
        labels=["LOW", "MID", "CHALK", "MEGA"],
        include_lowest=True,
        right=False,
    ).astype(str)

    _add_rank_features(df, "proj_fpts", "proj_fpts")
    _add_rank_features(df, "fpts_per_dollar", "value")
    _add_rank_features(df, "salary", "salary")
    _add_rank_features(df, "our_ownership_proj", "our_ownership")
    _add_rank_features(df, "game_total", "game_total")

    sort_cols = ["slate_date", "player_name"]
    ascending = [True, True]
    if sort_with_target and "target_actual_ownership" in df.columns:
        sort_cols = ["slate_date", "target_actual_ownership", "player_name"]
        ascending = [True, False, True]
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def build_ownership_training_dataset(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build a per-player, per-slate ownership training frame."""

    base_df = _read_base_ownership_rows(conn, start_date=start_date, end_date=end_date)
    if base_df.empty:
        return base_df

    predictions_df = _read_predictions_frame(conn, start_date=start_date, end_date=end_date)
    game_env_df = _read_game_environment_frame(conn, start_date=start_date, end_date=end_date)
    supplement_df = _read_latest_supplement_frame(
        conn, start_date=start_date, end_date=end_date
    )
    return _assemble_ownership_feature_frame(
        base_df,
        predictions_df=predictions_df,
        game_env_df=game_env_df,
        supplement_df=supplement_df,
        sort_with_target=True,
    )


def _build_ridge_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    alpha: float,
) -> Pipeline:
    transformers: List[Tuple[str, Any, Sequence[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_features),
            )
        )
    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _build_xgboost_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    stage: str,
) -> Pipeline:
    transformers: List[Tuple[str, Any, Sequence[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_features),
            )
        )
    if stage == "source":
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=6,
            subsample=0.82,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=2.5,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=180,
            learning_rate=0.045,
            max_depth=4,
            min_child_weight=8,
            subsample=0.86,
            colsample_bytree=0.84,
            reg_alpha=0.1,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("model", model),
        ]
    )


def _build_ownership_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    stage: str,
) -> Tuple[Pipeline, str]:
    if HAS_XGBOOST:
        return (
            _build_xgboost_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                stage=stage,
            ),
            "xgboost",
        )
    ridge_alpha = 18.0 if stage == "source" else 10.0
    return (
        _build_ridge_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            alpha=ridge_alpha,
        ),
        "ridge",
    )


def _metrics_from_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Any]:
    actual = pd.to_numeric(pd.Series(list(y_true)), errors="coerce").reset_index(drop=True)
    pred = pd.to_numeric(pd.Series(list(y_pred)), errors="coerce").reset_index(drop=True)
    mask = actual.notna() & pred.notna()
    if mask.sum() == 0:
        return {
            "rows": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "correlation": None,
            "rank_correlation": None,
            "top10_capture_pct": None,
            "top20_capture_pct": None,
            "top10_overlap_pct": None,
            "top20_overlap_pct": None,
        }
    actual_arr = actual[mask].to_numpy(dtype=float)
    pred_arr = pred[mask].to_numpy(dtype=float)
    err = actual_arr - pred_arr
    corr = None
    if len(actual_arr) >= 2 and np.unique(actual_arr).size > 1 and np.unique(pred_arr).size > 1:
        corr = float(np.corrcoef(actual_arr, pred_arr)[0, 1])
    rank_corr = None
    if len(actual_arr) >= 2 and np.unique(actual_arr).size > 1 and np.unique(pred_arr).size > 1:
        rank_corr = float(
            pd.Series(actual_arr).corr(pd.Series(pred_arr), method="spearman")
        )
    actual_series = pd.Series(actual_arr)
    pred_series = pd.Series(pred_arr)
    return {
        "rows": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(np.mean(err)),
        "correlation": corr,
        "rank_correlation": rank_corr,
        "top10_capture_pct": _top_k_capture_pct(actual_series, pred_series, 10),
        "top20_capture_pct": _top_k_capture_pct(actual_series, pred_series, 20),
        "top10_overlap_pct": _top_k_overlap_pct(actual_series, pred_series, 10),
        "top20_overlap_pct": _top_k_overlap_pct(actual_series, pred_series, 20),
    }


def _top_k_capture_pct(
    actual: pd.Series,
    pred: pd.Series,
    top_k: int,
) -> Optional[float]:
    mask = actual.notna() & pred.notna()
    if int(mask.sum()) == 0:
        return None
    actual = pd.to_numeric(actual.loc[mask], errors="coerce")
    pred = pd.to_numeric(pred.loc[mask], errors="coerce")
    row_count = int(len(actual))
    if row_count == 0:
        return None
    k = min(max(int(top_k), 1), row_count)
    actual_top = actual.nlargest(k)
    pred_top = pred.nlargest(k)
    denom = float(actual.loc[actual_top.index].sum())
    if denom <= 0.0:
        return None
    numer = float(actual.loc[pred_top.index].sum())
    return 100.0 * numer / denom


def _top_k_overlap_pct(
    actual: pd.Series,
    pred: pd.Series,
    top_k: int,
) -> Optional[float]:
    mask = actual.notna() & pred.notna()
    if int(mask.sum()) == 0:
        return None
    actual = pd.to_numeric(actual.loc[mask], errors="coerce")
    pred = pd.to_numeric(pred.loc[mask], errors="coerce")
    row_count = int(len(actual))
    if row_count == 0:
        return None
    k = min(max(int(top_k), 1), row_count)
    actual_idx = set(actual.nlargest(k).index.tolist())
    pred_idx = set(pred.nlargest(k).index.tolist())
    if not actual_idx:
        return None
    overlap = len(actual_idx & pred_idx)
    return 100.0 * float(overlap) / float(k)


def _coefficient_summary(model: Optional[Pipeline], top_n: int = 20) -> List[Dict[str, Any]]:
    if model is None:
        return []
    try:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        estimator = model.named_steps["model"]
    except Exception:
        return []

    metric_name = "coef"
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is None:
        coefficients = getattr(estimator, "feature_importances_", None)
        metric_name = "feature_importance"
    if coefficients is None:
        return []

    pairs = [
        {
            "feature": str(name),
            "coef": float(value),
            "metric": metric_name,
        }
        for name, value in zip(feature_names, coefficients)
    ]
    pairs.sort(key=lambda row: abs(row["coef"]), reverse=True)
    return pairs[:top_n]


def _select_holdout_dates(df: pd.DataFrame, holdout_slates: int) -> List[str]:
    unique_dates = sorted(df["slate_date"].dropna().astype(str).unique().tolist())
    if holdout_slates <= 0 or len(unique_dates) <= 1:
        return []

    source_dates = sorted(
        df.loc[df["supplement_source_count"] > 0, "slate_date"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    effective_holdout = min(holdout_slates, len(unique_dates) - 1)
    if source_dates and len(source_dates) <= effective_holdout:
        effective_holdout = max(1, len(source_dates) - 1)
    if effective_holdout <= 0:
        return []
    return unique_dates[-effective_holdout:]


def _filter_live_features(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    numeric_live = [
        col
        for col in numeric_features
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    categorical_live = [
        col
        for col in categorical_features
        if col in df.columns and df[col].fillna("").astype(str).ne("").any()
    ]
    return numeric_live, categorical_live


def _predict_with_model_bundle(
    feature_df: pd.DataFrame,
    model_bundle: Dict[str, Any],
) -> pd.DataFrame:
    """Score a feature frame with a previously trained ownership model bundle."""
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()

    scored_df = feature_df.copy()
    base_columns = list(model_bundle.get("base_columns") or [])
    source_columns = list(model_bundle.get("source_columns") or [])
    for col in base_columns + source_columns:
        if col in scored_df.columns:
            continue
        if col in BASE_CATEGORICAL_FEATURES or col in SOURCE_CATEGORICAL_FEATURES:
            scored_df[col] = ""
        else:
            scored_df[col] = np.nan

    base_model = model_bundle["base_model"]
    base_pred = np.clip(base_model.predict(scored_df[base_columns]), 0.0, 100.0)

    source_model = model_bundle.get("source_model")
    source_adjustment = np.zeros(len(scored_df), dtype=float)
    if source_model is not None and source_columns:
        source_active = scored_df["supplement_source_count"].gt(0).astype(float).to_numpy()
        raw_adjustment = source_model.predict(scored_df[source_columns])
        source_adjustment = (
            raw_adjustment
            * float(model_bundle.get("source_adjustment_weight") or 0.0)
            * source_active
        )

    final_pred = np.clip(base_pred + source_adjustment, 0.0, 100.0)
    prediction_df = scored_df.copy()
    prediction_df["base_ownership_pred"] = base_pred
    prediction_df["source_adjustment"] = source_adjustment
    prediction_df["final_ownership_pred"] = final_pred
    prediction_df["calibration_delta"] = (
        prediction_df["final_ownership_pred"] - prediction_df["our_ownership_proj"]
    )
    return prediction_df


def _slate_size_bucket(game_count: object) -> str:
    game_count_value = pd.to_numeric(pd.Series([game_count]), errors="coerce").iloc[0]
    if pd.isna(game_count_value):
        return "Unknown"
    if float(game_count_value) <= 4.0:
        return "Small"
    if float(game_count_value) <= 8.0:
        return "Medium"
    return "Large"


def _build_benchmark_metrics_rows(
    prediction_df: pd.DataFrame,
    *,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if prediction_df is None or prediction_df.empty:
        return rows

    actual_col = "target_actual_ownership"
    for model_key, model_label, pred_col, require_sources in BENCHMARK_MODEL_SPECS:
        scoped_df = prediction_df.copy()
        if require_sources:
            scoped_df = scoped_df[scoped_df["supplement_source_count"].gt(0)].copy()
        metrics = _metrics_from_predictions(scoped_df[actual_col], scoped_df[pred_col])
        row = {
            "model_key": model_key,
            "model_label": model_label,
            "prediction_column": pred_col,
            "source_required": bool(require_sources),
            "slates": int(scoped_df["slate_date"].nunique()) if "slate_date" in scoped_df.columns else 0,
        }
        row.update(metrics)
        if extra_fields:
            row.update(extra_fields)
        rows.append(row)
    return rows


def _rows_to_sorted_frame(rows: List[Dict[str, Any]], sort_col: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or sort_col not in df.columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_col).reset_index(drop=True)


def run_walkforward_ownership_benchmark(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    min_train_rows: int = 100,
    min_test_rows: int = 10,
    max_slates: Optional[int] = None,
) -> Dict[str, Any]:
    """Benchmark the ownership calibrator by walking forward across historical slates."""

    full_df = build_ownership_training_dataset(conn, start_date=start_date, end_date=end_date)
    if full_df.empty:
        return {"error": "No historical ownership rows were available for benchmarking."}

    full_df = full_df.dropna(subset=["target_actual_ownership"]).copy()
    if full_df.empty:
        return {"error": "No historical ownership rows with actuals were available."}

    unique_dates = sorted(full_df["slate_date"].dropna().astype(str).unique().tolist())
    if len(unique_dates) < 2:
        return {"error": "Walk-forward benchmarking requires at least two distinct slate dates."}
    evaluation_dates = list(unique_dates)
    if max_slates is not None and int(max_slates) > 0:
        evaluation_dates = evaluation_dates[-int(max_slates):]

    per_slate_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    skipped_rows: List[Dict[str, Any]] = []

    for slate_date in evaluation_dates:
        train_df = full_df[full_df["slate_date"].astype(str) < str(slate_date)].copy()
        test_df = full_df[full_df["slate_date"].astype(str) == str(slate_date)].copy()
        if len(test_df) < int(min_test_rows):
            skipped_rows.append(
                {
                    "slate_date": str(slate_date),
                    "reason": "insufficient_test_rows",
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue
        if len(train_df) < int(min_train_rows):
            skipped_rows.append(
                {
                    "slate_date": str(slate_date),
                    "reason": "insufficient_train_rows",
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue

        model_bundle = _train_ownership_calibration_models(train_df, holdout_slates=0)
        if model_bundle.get("error"):
            skipped_rows.append(
                {
                    "slate_date": str(slate_date),
                    "reason": str(model_bundle.get("error")),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue

        scored_df = _predict_with_model_bundle(test_df, model_bundle)
        scored_df["slate_size_bucket"] = scored_df["slate_game_count"].map(_slate_size_bucket)
        prediction_frames.append(scored_df)

        slate_row: Dict[str, Any] = {
            "slate_date": str(slate_date),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(scored_df)),
            "source_rows_test": int(scored_df["supplement_source_count"].gt(0).sum()),
            "source_coverage_pct": 100.0
            * float(scored_df["supplement_source_count"].gt(0).mean() or 0.0),
            "slate_game_count": float(
                pd.to_numeric(scored_df["slate_game_count"], errors="coerce")
                .dropna()
                .median()
                if "slate_game_count" in scored_df.columns
                else np.nan
            ),
            "slate_size_bucket": _slate_size_bucket(
                pd.to_numeric(scored_df["slate_game_count"], errors="coerce")
                .dropna()
                .median()
                if "slate_game_count" in scored_df.columns
                else np.nan
            ),
            "base_model_backend": str(model_bundle.get("base_backend") or ""),
            "source_model_backend": str(model_bundle.get("source_backend") or ""),
        }
        for metric_row in _build_benchmark_metrics_rows(scored_df):
            prefix = str(metric_row["model_key"])
            for metric_key in (
                "rows",
                "mae",
                "rmse",
                "bias",
                "correlation",
                "rank_correlation",
                "top10_capture_pct",
                "top20_capture_pct",
                "top10_overlap_pct",
                "top20_overlap_pct",
            ):
                slate_row[f"{prefix}_{metric_key}"] = metric_row.get(metric_key)
        raw_mae = slate_row.get("raw_our_mae")
        final_mae = slate_row.get("final_model_mae")
        source_mae = slate_row.get("source_consensus_mae")
        slate_row["final_vs_raw_mae_improvement"] = (
            float(raw_mae) - float(final_mae)
            if raw_mae is not None and final_mae is not None
            else None
        )
        slate_row["final_vs_source_mae_improvement"] = (
            float(source_mae) - float(final_mae)
            if source_mae is not None and final_mae is not None
            else None
        )
        per_slate_rows.append(slate_row)

    if not prediction_frames:
        skipped_df = _rows_to_sorted_frame(skipped_rows, "slate_date")
        return {
            "error": "No benchmarkable slates met the walk-forward minimums.",
            "skipped_slates_df": skipped_df,
            "summary": {
                "total_candidate_slates": int(len(evaluation_dates)),
                "benchmarked_slates": 0,
                "skipped_slates": int(len(skipped_df)),
                "min_train_rows": int(min_train_rows),
                "min_test_rows": int(min_test_rows),
            },
        }

    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    per_slate_df = _rows_to_sorted_frame(per_slate_rows, "slate_date")
    skipped_df = _rows_to_sorted_frame(skipped_rows, "slate_date")
    model_order = {
        model_key: idx for idx, (model_key, _, _, _) in enumerate(BENCHMARK_MODEL_SPECS)
    }

    summary_df = pd.DataFrame(_build_benchmark_metrics_rows(prediction_df))
    if not summary_df.empty:
        summary_df["model_order"] = summary_df["model_key"].map(model_order).fillna(999)
        summary_df["benchmarked_slates"] = int(per_slate_df["slate_date"].nunique())
        summary_df = (
            summary_df.sort_values(["model_order", "model_key"])
            .drop(columns=["model_order"])
            .reset_index(drop=True)
        )

    regime_rows: List[Dict[str, Any]] = []
    for bucket, bucket_df in prediction_df.groupby("slate_size_bucket", dropna=False):
        regime_rows.extend(
            _build_benchmark_metrics_rows(
                bucket_df,
                extra_fields={
                    "slate_size_bucket": str(bucket),
                },
            )
        )
    regime_summary_df = pd.DataFrame(regime_rows)
    if not regime_summary_df.empty:
        regime_summary_df["model_order"] = (
            regime_summary_df["model_key"].map(model_order).fillna(999)
        )
        regime_summary_df = (
            regime_summary_df.sort_values(
                ["slate_size_bucket", "model_order", "model_key"]
            )
            .drop(columns=["model_order"])
            .reset_index(drop=True)
        )

    return {
        "summary": {
            "total_candidate_slates": int(len(evaluation_dates)),
            "benchmarked_slates": int(per_slate_df["slate_date"].nunique()),
            "skipped_slates": int(len(skipped_df)),
            "min_train_rows": int(min_train_rows),
            "min_test_rows": int(min_test_rows),
            "max_slates": int(max_slates) if max_slates is not None else None,
            "prediction_rows": int(len(prediction_df)),
        },
        "summary_df": summary_df,
        "per_slate_df": per_slate_df,
        "regime_summary_df": regime_summary_df,
        "prediction_df": prediction_df,
        "skipped_slates_df": skipped_df,
    }


def _train_ownership_calibration_models(
    df: pd.DataFrame,
    holdout_slates: int = 2,
) -> Dict[str, Any]:
    """Fit the base and supplement residual ownership models."""
    target = pd.to_numeric(df["target_actual_ownership"], errors="coerce")
    valid_mask = target.notna()
    df = df.loc[valid_mask].copy()
    target = target.loc[valid_mask].astype(float)
    if df.empty:
        return {"error": "No ownership rows with actuals were available for calibration."}

    holdout_dates = _select_holdout_dates(df, holdout_slates=holdout_slates)
    if holdout_dates:
        test_mask = df["slate_date"].astype(str).isin(holdout_dates)
    else:
        test_mask = pd.Series(False, index=df.index)
    train_mask = ~test_mask

    if train_mask.sum() < 25:
        train_mask[:] = True
        test_mask[:] = False
        holdout_dates = []

    base_numeric_features = [col for col in BASE_NUMERIC_FEATURES if col in df.columns]
    base_categorical_features = [col for col in BASE_CATEGORICAL_FEATURES if col in df.columns]
    source_numeric_features = [col for col in SOURCE_NUMERIC_FEATURES if col in df.columns]
    source_categorical_features = [
        col for col in SOURCE_CATEGORICAL_FEATURES if col in df.columns
    ]
    train_df = df.loc[train_mask].copy()
    base_numeric_features, base_categorical_features = _filter_live_features(
        train_df, base_numeric_features, base_categorical_features
    )
    source_numeric_features, source_categorical_features = _filter_live_features(
        train_df, source_numeric_features, source_categorical_features
    )

    base_model, base_backend = _build_ownership_pipeline(
        numeric_features=base_numeric_features,
        categorical_features=base_categorical_features,
        stage="base",
    )
    base_columns = base_numeric_features + base_categorical_features
    base_model.fit(df.loc[train_mask, base_columns], target.loc[train_mask])

    base_pred_all = np.clip(base_model.predict(df[base_columns]), 0.0, 100.0)

    source_train_mask = train_mask & df["supplement_source_count"].gt(0)
    source_rows_train = int(source_train_mask.sum())
    source_rows_test = int((test_mask & df["supplement_source_count"].gt(0)).sum())
    source_adjustment_weight = (
        source_rows_train / (source_rows_train + SOURCE_SHRINK_K)
        if source_rows_train > 0
        else 0.0
    )

    source_model: Optional[Pipeline] = None
    source_columns: List[str] = []
    source_backend = ""
    source_adjustment_all = np.zeros(len(df), dtype=float)
    if source_rows_train >= 25 and source_numeric_features + source_categorical_features:
        source_model, source_backend = _build_ownership_pipeline(
            numeric_features=source_numeric_features,
            categorical_features=source_categorical_features,
            stage="source",
        )
        residual_target = target.loc[train_mask] - base_pred_all[train_mask.to_numpy()]
        source_columns = source_numeric_features + source_categorical_features
        source_model.fit(df.loc[train_mask, source_columns], residual_target)
        raw_adjustment = source_model.predict(df[source_columns])
        source_active = df["supplement_source_count"].gt(0).astype(float).to_numpy()
        source_adjustment_all = raw_adjustment * source_adjustment_weight * source_active

    final_pred_all = np.clip(base_pred_all + source_adjustment_all, 0.0, 100.0)
    return {
        "df": df,
        "target": target,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "holdout_dates": holdout_dates,
        "base_model": base_model,
        "base_backend": base_backend,
        "base_columns": base_columns,
        "base_pred_all": base_pred_all,
        "source_model": source_model,
        "source_backend": source_backend,
        "source_columns": source_columns,
        "source_adjustment_all": source_adjustment_all,
        "final_pred_all": final_pred_all,
        "source_rows_train": source_rows_train,
        "source_rows_test": source_rows_test,
        "source_adjustment_weight": source_adjustment_weight,
        "base_numeric_features": base_numeric_features,
        "base_categorical_features": base_categorical_features,
        "source_numeric_features": source_numeric_features,
        "source_categorical_features": source_categorical_features,
    }


def fit_ownership_calibrator(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    holdout_slates: int = 2,
    save_run: bool = True,
) -> Dict[str, Any]:
    """Fit a two-stage ownership calibrator and return training artifacts."""

    df = build_ownership_training_dataset(conn, start_date=start_date, end_date=end_date)
    if df.empty:
        return {"error": "No ownership rows with actuals were available for calibration."}
    model_bundle = _train_ownership_calibration_models(df, holdout_slates=holdout_slates)
    if model_bundle.get("error"):
        return {"error": str(model_bundle.get("error"))}
    df = model_bundle["df"]
    target = model_bundle["target"]
    train_mask = model_bundle["train_mask"]
    test_mask = model_bundle["test_mask"]
    holdout_dates = model_bundle["holdout_dates"]
    base_model = model_bundle["base_model"]
    source_model = model_bundle["source_model"]
    base_backend = str(model_bundle.get("base_backend") or "ridge")
    source_backend = str(model_bundle.get("source_backend") or "")
    base_pred_all = model_bundle["base_pred_all"]
    source_adjustment_all = model_bundle["source_adjustment_all"]
    final_pred_all = model_bundle["final_pred_all"]
    source_rows_train = int(model_bundle["source_rows_train"])
    source_rows_test = int(model_bundle["source_rows_test"])
    source_adjustment_weight = float(model_bundle["source_adjustment_weight"])
    base_numeric_features = list(model_bundle["base_numeric_features"])
    base_categorical_features = list(model_bundle["base_categorical_features"])
    source_numeric_features = list(model_bundle["source_numeric_features"])
    source_categorical_features = list(model_bundle["source_categorical_features"])

    baseline_metrics = {
        "our_model_all": _metrics_from_predictions(target, df["our_ownership_proj"]),
        "our_model_test": _metrics_from_predictions(
            target.loc[test_mask], df.loc[test_mask, "our_ownership_proj"]
        ),
        "source_consensus_all": _metrics_from_predictions(
            target.loc[df["supplement_source_count"].gt(0)],
            df.loc[df["supplement_source_count"].gt(0), "supplement_own_consensus"],
        ),
        "source_consensus_test": _metrics_from_predictions(
            target.loc[test_mask & df["supplement_source_count"].gt(0)],
            df.loc[test_mask & df["supplement_source_count"].gt(0), "supplement_own_consensus"],
        ),
    }
    base_metrics = {
        "base_model_all": _metrics_from_predictions(target, base_pred_all),
        "base_model_test": _metrics_from_predictions(
            target.loc[test_mask], base_pred_all[test_mask.to_numpy()]
        ),
    }
    final_metrics = {
        "final_model_all": _metrics_from_predictions(target, final_pred_all),
        "final_model_test": _metrics_from_predictions(
            target.loc[test_mask],
            final_pred_all[test_mask.to_numpy()],
        ),
        "final_model_source_test": _metrics_from_predictions(
            target.loc[test_mask & df["supplement_source_count"].gt(0)],
            final_pred_all[(test_mask & df["supplement_source_count"].gt(0)).to_numpy()],
        ),
    }

    run_key = token_hex(10)
    fitted_at = datetime.now().isoformat(timespec="seconds")
    train_dates = sorted(df.loc[train_mask, "slate_date"].astype(str).unique().tolist())
    test_dates = sorted(df.loc[test_mask, "slate_date"].astype(str).unique().tolist())

    prediction_frame = df[
        [
            "slate_date",
            "player_id",
            "player_name",
            "team",
            "salary",
            "proj_fpts",
            "our_ownership_proj",
            "supplement_own_consensus",
            "target_actual_ownership",
            "supplement_source_count",
        ]
    ].copy()
    prediction_frame["base_ownership_pred"] = base_pred_all
    prediction_frame["source_adjustment"] = source_adjustment_all
    prediction_frame["final_ownership_pred"] = final_pred_all
    prediction_frame["split"] = np.where(test_mask.to_numpy(), "test", "train")

    artifact = {
        "xgboost_available": bool(HAS_XGBOOST),
        "base_model_backend": base_backend,
        "source_model_backend": source_backend,
        "source_adjustment_weight": round(float(source_adjustment_weight), 6),
        "holdout_dates": test_dates,
        "base_top_coefficients": _coefficient_summary(base_model),
        "source_top_coefficients": _coefficient_summary(source_model),
    }
    result = OwnershipCalibrationResult(
        run_key=run_key,
        model_version=MODEL_VERSION,
        fitted_at=fitted_at,
        total_rows=int(len(df)),
        train_rows=int(train_mask.sum()),
        test_rows=int(test_mask.sum()),
        source_rows_train=source_rows_train,
        source_rows_test=source_rows_test,
        train_start_date=train_dates[0] if train_dates else None,
        train_end_date=train_dates[-1] if train_dates else None,
        holdout_start_date=test_dates[0] if test_dates else None,
        holdout_end_date=test_dates[-1] if test_dates else None,
        feature_count_base=len(base_numeric_features) + len(base_categorical_features),
        feature_count_source=len(source_numeric_features) + len(source_categorical_features),
        base_metrics=base_metrics,
        final_metrics=final_metrics,
        baseline_metrics=baseline_metrics,
        artifact=artifact,
    )

    if save_run:
        _ensure_calibration_table(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO dfs_ownership_calibration_runs (
                run_key,
                model_version,
                fitted_at,
                train_start_date,
                train_end_date,
                holdout_start_date,
                holdout_end_date,
                total_rows,
                train_rows,
                test_rows,
                source_rows_train,
                source_rows_test,
                feature_count_base,
                feature_count_source,
                base_metrics_json,
                final_metrics_json,
                baseline_metrics_json,
                artifact_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.run_key,
                result.model_version,
                result.fitted_at,
                result.train_start_date,
                result.train_end_date,
                result.holdout_start_date,
                result.holdout_end_date,
                result.total_rows,
                result.train_rows,
                result.test_rows,
                result.source_rows_train,
                result.source_rows_test,
                result.feature_count_base,
                result.feature_count_source,
                json.dumps(result.base_metrics, sort_keys=True),
                json.dumps(result.final_metrics, sort_keys=True),
                json.dumps(result.baseline_metrics, sort_keys=True),
                json.dumps(result.artifact, sort_keys=True),
            ),
        )
        conn.commit()

    return {
        "summary": result,
        "training_df": df,
        "prediction_df": prediction_frame,
    }


def _build_live_base_ownership_rows(
    players: Sequence[Any],
    slate_date: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for player in players or []:
        try:
            player_id = int(getattr(player, "player_id", 0) or 0)
        except (TypeError, ValueError):
            player_id = 0
        if player_id <= 0:
            continue

        current_own = max(0.0, float(getattr(player, "ownership_proj", 0.0) or 0.0))
        stored_base = max(
            0.0,
            float(getattr(player, "_ownership_calibration_base_proj", 0.0) or 0.0),
        )
        stored_delta = float(getattr(player, "_ownership_calibration_delta", 0.0) or 0.0)
        expected_adjusted = stored_base + stored_delta if stored_base > 0 else None
        if expected_adjusted is not None and abs(current_own - expected_adjusted) <= 0.05:
            base_own = stored_base
        else:
            base_own = current_own

        positions = getattr(player, "positions", []) or []
        rows.append(
            {
                "slate_date": str(slate_date),
                "player_id": player_id,
                "player_name": str(getattr(player, "name", "") or ""),
                "team": str(getattr(player, "team", "") or ""),
                "opponent": str(getattr(player, "opponent", "") or ""),
                "salary": int(getattr(player, "salary", 0) or 0),
                "positions": "/".join([str(pos).strip() for pos in positions if str(pos).strip()]),
                "proj_fpts": float(getattr(player, "proj_fpts", 0.0) or 0.0),
                "proj_floor": float(getattr(player, "proj_floor", 0.0) or 0.0),
                "proj_ceiling": float(getattr(player, "proj_ceiling", 0.0) or 0.0),
                "fpts_per_dollar": float(getattr(player, "fpts_per_dollar", 0.0) or 0.0),
                "our_ownership_proj": base_own,
                "recorded_actual_ownership": np.nan,
                "target_actual_ownership": np.nan,
                "actual_ownership_imputed_zero": 0,
                "contest_imported": 0,
                "actual_fpts": np.nan,
                "actual_minutes": pd.to_numeric(
                    getattr(player, "recent_minutes_avg", np.nan),
                    errors="coerce",
                ),
                "did_play": np.nan,
                "role_tier": str(getattr(player, "role_tier", "UNK") or "UNK"),
                "analytics_used": str(getattr(player, "analytics_used", "") or ""),
            }
        )
    return pd.DataFrame(rows)


def _supplement_feature_prefix(source_name: object) -> str:
    bucket = _normalize_source_bucket(source_name)
    if bucket == "lineupstarter":
        return "ls"
    if bucket == "rotowire":
        return "rw"
    lowered = str(source_name or "").strip().lower()
    if "lineupstarter" in lowered or "lineup starter" in lowered or "linestarter" in lowered:
        return "ls"
    return "rw"


def _build_live_supplement_frame(
    slate_date: str,
    supplement_state: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    state = dict(supplement_state or {})
    source_player_maps = state.get("source_player_maps") or []
    if not source_player_maps:
        player_map = state.get("player_map") or {}
        source_name = (
            state.get("ownership_source_name")
            or state.get("source_name")
            or ""
        )
        if player_map and source_name:
            source_player_maps = [
                {
                    "source_name": source_name,
                    "player_map": player_map,
                }
            ]
    if not source_player_maps:
        return pd.DataFrame()

    merged_df = pd.DataFrame()
    for entry in source_player_maps:
        source_name = str(entry.get("source_name") or "")
        player_map = entry.get("player_map") or {}
        prefix = _supplement_feature_prefix(source_name)
        rows: List[Dict[str, Any]] = []
        for raw_player_id, info in player_map.items():
            try:
                player_id = int(raw_player_id or 0)
            except (TypeError, ValueError):
                player_id = 0
            if player_id <= 0:
                continue
            row = {
                "slate_date": str(slate_date),
                "player_id": player_id,
                f"{prefix}_proj_fpts": pd.to_numeric(
                    info.get("supplement_proj_fpts"),
                    errors="coerce",
                ),
                f"{prefix}_own_pct": pd.to_numeric(
                    info.get("supplement_ownership"),
                    errors="coerce",
                ),
                f"{prefix}_proj_delta": pd.to_numeric(
                    info.get("proj_delta"),
                    errors="coerce",
                ),
                f"{prefix}_own_delta_pp": pd.to_numeric(
                    info.get("own_delta_pp"),
                    errors="coerce",
                ),
                f"{prefix}_match_score": pd.to_numeric(
                    info.get("match_score"),
                    errors="coerce",
                ),
            }
            rows.append(row)
        source_df = pd.DataFrame(rows)
        if source_df.empty:
            continue
        if merged_df.empty:
            merged_df = source_df
        else:
            merged_df = merged_df.merge(
                source_df,
                on=["slate_date", "player_id"],
                how="outer",
            )
    return merged_df


def apply_live_ownership_calibration(
    conn: sqlite3.Connection,
    players: Sequence[Any],
    slate_date: str,
    supplement_state: Optional[Dict[str, Any]] = None,
    min_train_rows: int = 100,
) -> Dict[str, Any]:
    """Fit the ownership calibrator on historical slates and apply it live."""
    stats = {
        "active": False,
        "mode": "calibrator",
        "model_version": MODEL_VERSION,
        "calibrated_players": 0,
        "train_rows": 0,
        "source_rows_train": 0,
        "source_adjustment_weight": 0.0,
        "train_end_date": "",
        "avg_delta_pp": 0.0,
        "base_model_backend": "",
        "source_model_backend": "",
    }

    base_df = _build_live_base_ownership_rows(players, slate_date)
    if base_df.empty:
        stats["reason"] = "no_live_players"
        return stats

    slate_ts = pd.to_datetime(slate_date, errors="coerce")
    if pd.isna(slate_ts):
        train_end_date = None
    else:
        train_end_date = (slate_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        stats["train_end_date"] = train_end_date

    training_df = build_ownership_training_dataset(
        conn,
        start_date=None,
        end_date=train_end_date,
    )
    if training_df.empty or len(training_df) < int(min_train_rows):
        stats["reason"] = "insufficient_training_rows"
        stats["train_rows"] = int(len(training_df))
        return stats

    model_bundle = _train_ownership_calibration_models(training_df, holdout_slates=0)
    if model_bundle.get("error"):
        stats["reason"] = str(model_bundle.get("error"))
        return stats
    stats["base_model_backend"] = str(model_bundle.get("base_backend") or "ridge")
    stats["source_model_backend"] = str(model_bundle.get("source_backend") or "")

    live_df = _assemble_ownership_feature_frame(
        base_df,
        predictions_df=_read_predictions_frame(conn, start_date=slate_date, end_date=slate_date),
        game_env_df=_read_game_environment_frame(conn, start_date=slate_date, end_date=slate_date),
        supplement_df=_build_live_supplement_frame(slate_date, supplement_state),
        sort_with_target=False,
    )
    if live_df.empty:
        stats["reason"] = "empty_live_frame"
        return stats

    scored_live_df = _predict_with_model_bundle(live_df, model_bundle)
    prediction_df = scored_live_df[
        [
            "player_id",
            "our_ownership_proj",
            "supplement_source_count",
            "base_ownership_pred",
            "source_adjustment",
            "final_ownership_pred",
            "calibration_delta",
        ]
    ].copy()
    prediction_df = prediction_df.rename(
        columns={
            "base_ownership_pred": "base_pred",
            "final_ownership_pred": "calibrated_ownership",
        }
    )
    prediction_map = {
        int(row["player_id"]): row for row in prediction_df.to_dict("records")
    }

    adjusted_players = 0
    delta_values: List[float] = []
    for player in players or []:
        try:
            player_id = int(getattr(player, "player_id", 0) or 0)
        except (TypeError, ValueError):
            player_id = 0
        row = prediction_map.get(player_id)
        if not row:
            continue
        calibrated_own = float(row["calibrated_ownership"] or 0.0)
        base_own = float(row["our_ownership_proj"] or 0.0)
        delta_own = float(row["calibration_delta"] or 0.0)
        player.ownership_proj = round(calibrated_own, 3)
        player._ownership_calibration_base_proj = round(base_own, 4)
        player._ownership_calibration_delta = round(delta_own, 4)
        player.ownership_calibration_active = True
        player.ownership_calibration_base_pred = round(float(row["base_pred"] or 0.0), 4)
        player.ownership_calibration_source_adjustment = round(
            float(row["source_adjustment"] or 0.0),
            4,
        )
        if abs(delta_own) >= 0.1:
            adjusted_players += 1
            delta_values.append(delta_own)

    stats.update(
        {
            "active": True,
            "calibrated_players": int(len(prediction_map)),
            "adjusted_players": int(adjusted_players),
            "train_rows": int(len(model_bundle["df"])),
            "source_rows_train": int(model_bundle["source_rows_train"]),
            "source_adjustment_weight": float(model_bundle["source_adjustment_weight"]),
            "avg_delta_pp": (
                float(np.mean(np.abs(delta_values))) if delta_values else 0.0
            ),
        }
    )
    return stats


def get_latest_ownership_calibration_run(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Load the most recent saved ownership calibration summary."""
    if not _table_exists(conn, "dfs_ownership_calibration_runs"):
        return None
    row = conn.execute(
        """
        SELECT
            run_key,
            model_version,
            fitted_at,
            train_start_date,
            train_end_date,
            holdout_start_date,
            holdout_end_date,
            total_rows,
            train_rows,
            test_rows,
            source_rows_train,
            source_rows_test,
            feature_count_base,
            feature_count_source,
            base_metrics_json,
            final_metrics_json,
            baseline_metrics_json,
            artifact_json
        FROM dfs_ownership_calibration_runs
        ORDER BY datetime(fitted_at) DESC, run_key DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None

    columns = [
        "run_key",
        "model_version",
        "fitted_at",
        "train_start_date",
        "train_end_date",
        "holdout_start_date",
        "holdout_end_date",
        "total_rows",
        "train_rows",
        "test_rows",
        "source_rows_train",
        "source_rows_test",
        "feature_count_base",
        "feature_count_source",
        "base_metrics_json",
        "final_metrics_json",
        "baseline_metrics_json",
        "artifact_json",
    ]
    payload = dict(zip(columns, row))
    for json_key in (
        "base_metrics_json",
        "final_metrics_json",
        "baseline_metrics_json",
        "artifact_json",
    ):
        if payload.get(json_key):
            payload[json_key[:-5]] = json.loads(payload[json_key])
        payload.pop(json_key, None)
    return payload


def _print_summary(result: OwnershipCalibrationResult) -> None:
    final_test = result.final_metrics.get("final_model_test", {})
    base_test = result.base_metrics.get("base_model_test", {})
    our_test = result.baseline_metrics.get("our_model_test", {})
    source_test = result.baseline_metrics.get("source_consensus_test", {})

    print(f"Run key: {result.run_key}")
    print(f"Model version: {result.model_version}")
    if result.artifact:
        print(
            "Backends:"
            f" base={result.artifact.get('base_model_backend')}"
            f" source={result.artifact.get('source_model_backend') or 'n/a'}"
        )
    print(
        f"Rows: total={result.total_rows} train={result.train_rows} "
        f"test={result.test_rows}"
    )
    print(
        f"Source rows: train={result.source_rows_train} "
        f"test={result.source_rows_test}"
    )
    if result.holdout_start_date and result.holdout_end_date:
        print(
            f"Holdout slates: {result.holdout_start_date} -> "
            f"{result.holdout_end_date}"
        )
    print(
        "Test MAE:"
        f" our={our_test.get('mae')}"
        f" base={base_test.get('mae')}"
        f" final={final_test.get('mae')}"
        f" source_consensus={source_test.get('mae')}"
    )
    print(
        "Test corr:"
        f" our={our_test.get('correlation')}"
        f" base={base_test.get('correlation')}"
        f" final={final_test.get('correlation')}"
        f" source_consensus={source_test.get('correlation')}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit DFS ownership calibrator")
    parser.add_argument("--db-path", default="nba_stats.db", help="SQLite database path")
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD lower bound")
    parser.add_argument("--end-date", default=None, help="Optional YYYY-MM-DD upper bound")
    parser.add_argument(
        "--holdout-slates",
        type=int,
        default=2,
        help="Number of latest slates to reserve as a holdout set",
    )
    parser.add_argument(
        "--export-training-csv",
        default=None,
        help="Optional path to export the training dataset as CSV",
    )
    parser.add_argument(
        "--export-predictions-csv",
        default=None,
        help="Optional path to export fitted train/test predictions as CSV",
    )
    parser.add_argument(
        "--no-save-run",
        action="store_true",
        help="Do not save the calibration summary back into SQLite",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        result = fit_ownership_calibrator(
            conn,
            start_date=args.start_date,
            end_date=args.end_date,
            holdout_slates=args.holdout_slates,
            save_run=not args.no_save_run,
        )
        if "error" in result:
            raise SystemExit(result["error"])

        summary: OwnershipCalibrationResult = result["summary"]
        training_df: pd.DataFrame = result["training_df"]
        prediction_df: pd.DataFrame = result["prediction_df"]

        if args.export_training_csv:
            output_path = Path(args.export_training_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            training_df.to_csv(output_path, index=False)
        if args.export_predictions_csv:
            output_path = Path(args.export_predictions_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_df.to_csv(output_path, index=False)

        _print_summary(summary)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
