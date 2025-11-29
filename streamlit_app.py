"""
Streamlit dashboard for exploring locally built NBA stats.
Version: 2025-11-26 (CSV Export Feature)
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional

import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for Windows builds without tzdata
    ZoneInfo = None

from nba_api.stats.endpoints import scoreboardv2

from nba_to_sqlite import build_database
import injury_impact_analytics as iia
import injury_adjustment as ia
import player_correlation_analytics as pca
import defense_type_analytics as dta
import prediction_tracking as pt
import s3_storage

DEFAULT_DB_PATH = Path(__file__).with_name("nba_stats.db")
DEFAULT_PREDICTIONS_PATH = Path(__file__).with_name("predictions.csv")
DEFAULT_SEASON = "2025-26"
DEFAULT_SEASON_TYPE = "Regular Season"
DEFAULT_DEFENSE_MIX_SEASON = "2025-26"
MATCHUP_INTERNAL_COLUMNS = ["away_team_id", "home_team_id", "game_id"]
WEIGHT_METRIC_MAP = {
    "avg": "avg_points",
    "median": "median_points",
    "max": "max_points",
    "last3": "avg_pts_last3",
    "last5": "avg_pts_last5",
    "minutes5": "avg_minutes_last5",
    "usage5": "avg_usg_last5",
}
DEFAULT_WEIGHTS = {
    "avg": 0.2,
    "median": 0.15,
    "max": 0.1,
    "last3": 0.2,
    "last5": 0.15,
    "minutes5": 0.1,
    "usage5": 0.1,
}
DEFAULT_MIN_GAMES = 10
TOP_LEADERS_COUNT = 5
DAILY_LEADERS_MAX = 10


def _resolve_eastern_zone() -> timezone:
    if ZoneInfo is not None:
        try:
            return ZoneInfo("America/New_York")
        except Exception:
            return timezone.utc
    return timezone.utc


EASTERN_TZ = _resolve_eastern_zone()

st.set_page_config(page_title="NBA Daily Insights", layout="wide", initial_sidebar_state="expanded")
st.title("NBA Daily Insights")
st.caption(
    "Explore league standings, leaderboards, and prediction outputs produced "
    "by the local NBA SQLite builder."
)


# S3 Database Sync on Startup
@st.cache_resource
def sync_database_from_s3() -> tuple[bool, str]:
    """
    Download database from S3 on app startup.
    Only runs once per session due to cache_resource.
    """
    storage = s3_storage.S3PredictionStorage()

    if not storage.is_connected():
        return False, "S3 not configured (running in local mode)"

    db_path = DEFAULT_DB_PATH

    # If database already exists locally, don't overwrite it
    # (this preserves local development workflows)
    if db_path.exists():
        # But check if S3 has a newer version
        s3_info = storage.get_backup_info()
        if s3_info.get('exists'):
            local_mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc)
            s3_mtime = s3_info['last_modified']

            if s3_mtime > local_mtime:
                # S3 version is newer, download it
                success, message = storage.download_database(db_path)
                if success:
                    return True, f"✅ Restored database from S3 ({message})"
                else:
                    return False, f"⚠️ Failed to download newer S3 backup: {message}"
            else:
                return True, "Using local database (newer than S3 backup)"
        else:
            return True, "Using existing local database (no S3 backup found)"

    # No local database, try to download from S3
    success, message = storage.download_database(db_path)

    if success:
        return True, f"✅ Restored database from S3 ({message})"
    else:
        # No backup exists (normal for first run)
        if "No backup found" in message or "404" in message:
            return True, "No S3 backup yet (this is normal for first run)"
        return False, f"⚠️ S3 restore failed: {message}"


# Run S3 sync on startup
s3_sync_status, s3_sync_message = sync_database_from_s3()

# Show S3 sync status in sidebar
with st.sidebar:
    st.markdown("### ☁️ Cloud Backup Status")
    if "S3 not configured" in s3_sync_message:
        st.info("💻 Running in local mode")
    elif s3_sync_status:
        st.success(s3_sync_message)
    else:
        st.warning(s3_sync_message)
    st.divider()


@st.cache_resource
def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Auto-upgrade database schema for injury adjustments (one-time migration)
    try:
        pt.upgrade_predictions_table_for_injuries(conn)
    except Exception:
        pass  # Schema already upgraded or predictions table doesn't exist yet

    return conn


@st.cache_data(ttl=300)
def run_query(db_path: str, query: str, params: Iterable[object] | None = None) -> pd.DataFrame:
    conn = get_connection(db_path)
    return pd.read_sql_query(query, conn, params=params or [])


@st.cache_data(ttl=300)
def fetch_distinct_values(db_path: str, table: str, column: str) -> list[str]:
    query = f"SELECT DISTINCT {column} FROM {table} ORDER BY {column} DESC"
    df = run_query(db_path, query)
    return df[column].dropna().astype(str).tolist()


def render_dataframe(df: pd.DataFrame, use_container_width: bool = True) -> None:
    if df.empty:
        st.info("No rows returned for the current selection.")
    else:
        st.dataframe(df, use_container_width=use_container_width)


def persist_uploaded_file(file, suffix: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "nba_daily_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload{suffix}"
    with open(temp_path, "wb") as handle:
        handle.write(file.getbuffer())
    return temp_path


def normalize_weight_map(weight_map: Dict[str, float]) -> Dict[str, float]:
    sanitized = {k: max(0.0, float(v)) for k, v in weight_map.items()}
    total = sum(sanitized.values())
    if total <= 0:
        uniform = 1.0 / len(sanitized)
        return {k: uniform for k in sanitized}
    return {k: value / total for k, value in sanitized.items()}


@st.cache_data(ttl=600)
def aggregate_player_scoring(
    db_path: str,
    season: str,
    season_type: str,
) -> pd.DataFrame:
    query = """
        SELECT player_id,
               player_name,
               team_id,
               points,
               fg3m,
               game_date,
               minutes,
               usg_pct
        FROM player_game_logs
        WHERE season = ?
          AND season_type = ?
    """
    try:
        logs_df = run_query(db_path, query, params=(season, season_type))
    except Exception as exc:  # noqa: BLE001
        if "no such column: usg_pct" in str(exc).lower():
            fallback_query = """
                SELECT player_id,
                       player_name,
                       team_id,
                       points,
                       fg3m,
                       game_date,
                       minutes
                FROM player_game_logs
                WHERE season = ?
                  AND season_type = ?
            """
            logs_df = run_query(db_path, fallback_query, params=(season, season_type))
            logs_df["usg_pct"] = None
        else:
            raise
    if logs_df.empty:
        return pd.DataFrame()
    numeric_cols = ["player_id", "team_id", "points", "fg3m", "usg_pct"]
    for col in numeric_cols:
        logs_df[col] = pd.to_numeric(logs_df[col], errors="coerce")
    logs_df["minutes_float"] = logs_df["minutes"].apply(minutes_str_to_float)
    logs_df["game_date"] = pd.to_datetime(logs_df["game_date"], errors="coerce")
    logs_df = logs_df.dropna(subset=["player_id", "team_id", "points", "game_date"])
    grouped = (
        logs_df.groupby(["player_id", "player_name"])
        .agg(
            games_played=("points", "count"),
            avg_points=("points", "mean"),
            median_points=("points", "median"),
            max_points=("points", "max"),
            avg_fg3m=("fg3m", "mean"),
            median_fg3m=("fg3m", "median"),
            avg_minutes=("minutes_float", "mean"),
            median_minutes=("minutes_float", "median"),
        )
        .reset_index()
    )

    latest_team = (
        logs_df.sort_values("game_date")
        .groupby("player_id")
        .tail(1)[["player_id", "team_id"]]
    )

    recent_records = []
    for player_id, player_df in logs_df.groupby("player_id"):
        player_df = player_df.sort_values("game_date", ascending=False)

        def calc_recent(window: int, column: str) -> float | None:
            subset = player_df.head(window)[column].dropna()
            if subset.empty:
                return None
            return subset.mean()

        recent_records.append(
            {
                "player_id": player_id,
                "avg_pts_last3": calc_recent(3, "points"),
                "avg_pts_last5": calc_recent(5, "points"),
                "avg_fg3m_last3": calc_recent(3, "fg3m"),
                "avg_fg3m_last5": calc_recent(5, "fg3m"),
                "avg_minutes_last5": calc_recent(5, "minutes_float"),
                "avg_usg_last5": calc_recent(5, "usg_pct"),
            }
        )

    recent_df = pd.DataFrame(recent_records)
    if recent_df.empty:
        recent_df = pd.DataFrame(
            columns=[
                "player_id",
                "avg_pts_last3",
                "avg_pts_last5",
                "avg_fg3m_last3",
                "avg_fg3m_last5",
                "avg_minutes_last5",
                "avg_usg_last5",
            ]
        )
    grouped = grouped.merge(latest_team, on="player_id", how="left")
    grouped = grouped.merge(recent_df, on="player_id", how="left")
    team_names = run_query(
        db_path,
        "SELECT team_id, full_name FROM teams",
    )
    team_names["team_id"] = pd.to_numeric(team_names["team_id"], errors="coerce")
    grouped = grouped.merge(team_names, on="team_id", how="left")
    try:
        usage_df = run_query(
            db_path,
            """
            SELECT player_id, usg_pct
            FROM player_season_totals
            WHERE season = ?
              AND season_type = ?
            """,
            params=(season, season_type),
        )
    except Exception:
        usage_df = pd.DataFrame()
    if not usage_df.empty and "usg_pct" in usage_df.columns:
        usage_df["player_id"] = pd.to_numeric(usage_df["player_id"], errors="coerce")
        grouped = grouped.merge(usage_df, on="player_id", how="left")
    return grouped


def prepare_weighted_scores(
    stats_df: pd.DataFrame,
    min_games: int,
    weights: Dict[str, float],
) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()
    filtered = stats_df[stats_df["games_played"] >= max(1, min_games)].copy()
    if filtered.empty:
        return filtered
    for weight_key, column in WEIGHT_METRIC_MAP.items():
        z_col = f"z_{weight_key}"
        if column not in filtered.columns:
            filtered[column] = 0.0
        col_values = pd.to_numeric(filtered[column], errors="coerce")
        mean = col_values.mean()
        std = col_values.std(ddof=0)
        if std == 0 or pd.isna(std):
            filtered[z_col] = 0.0
        else:
            filtered[z_col] = (col_values.fillna(mean) - mean) / std
    score_series = pd.Series(0.0, index=filtered.index, dtype=float)
    for weight_key, weight_value in weights.items():
        z_col = f"z_{weight_key}"
        if z_col in filtered:
            score_series += weight_value * filtered[z_col]
    filtered["composite_score"] = score_series
    filtered["weighted_score"] = score_series
    filtered["team_rank"] = filtered.groupby("team_id")["weighted_score"].rank(
        method="first", ascending=False
    )
    return filtered.sort_values("composite_score", ascending=False)


@st.cache_data(ttl=300)
def load_team_scoring_stats(
    db_path: str,
    season: str,
    season_type: str,
) -> pd.DataFrame:
    query = """
        SELECT team_id, pts, fg3m, game_date
        FROM team_game_logs
        WHERE season = ?
          AND season_type = ?
    """
    df = run_query(db_path, query, params=(season, season_type))
    if df.empty:
        return df
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["pts"] = pd.to_numeric(df["pts"], errors="coerce")
    df["fg3m"] = pd.to_numeric(df["fg3m"], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["team_id", "pts"])
    aggregates = (
        df.groupby("team_id")
        .agg(
            games_played=("pts", "count"),
            total_pts=("pts", "sum"),
            avg_pts=("pts", "mean"),
            median_pts=("pts", "median"),
            median_fg3m=("fg3m", "median"),
        )
        .reset_index()
    )
    def compute_recent(team_df: pd.DataFrame, window: int, column: str, agg_fn: str) -> float | None:
        subset = (
            team_df.sort_values("game_date", ascending=False)
            .head(window)
        )
        if subset.empty:
            return None
        values = subset[column].dropna()
        if values.empty:
            return None
        if agg_fn == "mean":
            return values.mean()
        if agg_fn == "median":
            return values.median()
        return None

    recent_stats = []
    for team_id, team_df in df.groupby("team_id"):
        recent_stats.append(
            {
                "team_id": team_id,
                "median_pts_last5": compute_recent(team_df, 5, "pts", "median"),
                "avg_pts_last5": compute_recent(team_df, 5, "pts", "mean"),
                "avg_pts_last3": compute_recent(team_df, 3, "pts", "mean"),
                "avg_fg3m_last3": compute_recent(team_df, 3, "fg3m", "mean"),
                "avg_fg3m_last5": compute_recent(team_df, 5, "fg3m", "mean"),
            }
        )
    recent_df = pd.DataFrame(recent_stats)
    aggregates = aggregates.merge(recent_df, on="team_id", how="left")
    return aggregates


@st.cache_data(ttl=300)
def load_team_defense_stats(
    db_path: str,
    season: str,
    season_type: str,
    rolling_window: int | None = None,
) -> pd.DataFrame:
    # Try query with new columns first (pace-adjusted stats)
    query_new = """
        SELECT t.team_id,
               t.game_date,
               opp.pts AS allowed_pts,
               opp.fg3m AS allowed_fg3m,
               opp.fg3a AS allowed_fg3a,
               opp.reb AS allowed_reb,
               opp.ast AS allowed_ast,
               opp.fga AS opp_fga,
               opp.fta AS opp_fta,
               opp.oreb AS opp_oreb,
               opp.tov AS opp_tov
        FROM team_game_logs AS t
        JOIN team_game_logs AS opp
          ON opp.game_id = t.game_id
         AND opp.team_id <> t.team_id
        WHERE t.season = ?
          AND t.season_type = ?
    """

    # Fallback query for old database schema (without pace stats)
    query_old = """
        SELECT t.team_id,
               t.game_date,
               opp.pts AS allowed_pts,
               opp.fg3m AS allowed_fg3m,
               opp.fg3a AS allowed_fg3a,
               opp.reb AS allowed_reb,
               opp.ast AS allowed_ast
        FROM team_game_logs AS t
        JOIN team_game_logs AS opp
          ON opp.game_id = t.game_id
         AND opp.team_id <> t.team_id
        WHERE t.season = ?
          AND t.season_type = ?
    """

    try:
        df = run_query(db_path, query_new, params=(season, season_type))
    except Exception:
        # Fall back to old schema if new columns don't exist
        df = run_query(db_path, query_old, params=(season, season_type))
        # Add placeholder columns for missing data
        df["opp_fga"] = None
        df["opp_fta"] = None
        df["opp_oreb"] = None
        df["opp_tov"] = None

    if df.empty:
        return df
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["allowed_pts"] = pd.to_numeric(df["allowed_pts"], errors="coerce")
    for col in ["allowed_fg3m", "allowed_fg3a", "allowed_reb", "allowed_ast",
                "opp_fga", "opp_fta", "opp_oreb", "opp_tov"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate 2PT points allowed (total points - 3PT points)
    df["allowed_2pt_pts"] = df["allowed_pts"] - (df["allowed_fg3m"] * 3)

    # Check if pace data is available
    has_pace_data = df["opp_fga"].notna().any()

    if has_pace_data:
        # Estimate opponent possessions: FGA + 0.44*FTA - ORB + TOV
        df["opp_possessions"] = (
            df["opp_fga"] +
            0.44 * df["opp_fta"].fillna(0) -
            df["opp_oreb"].fillna(0) +
            df["opp_tov"].fillna(0)
        )
    else:
        # No pace data available - use placeholder
        df["opp_possessions"] = None

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["team_id", "allowed_pts"])
    if rolling_window and rolling_window > 0:
        df = (
            df.sort_values("game_date")
            .groupby("team_id")
            .tail(rolling_window)
        )
    aggregates = (
        df.groupby("team_id")
        .agg(
            games_played=("allowed_pts", "count"),
            total_allowed_pts=("allowed_pts", "sum"),
            avg_allowed_pts=("allowed_pts", "mean"),
            median_allowed_pts=("allowed_pts", "median"),
            avg_allowed_fg3m=("allowed_fg3m", "mean"),
            avg_allowed_fg3a=("allowed_fg3a", "mean"),
            avg_allowed_reb=("allowed_reb", "mean"),
            avg_allowed_ast=("allowed_ast", "mean"),
            avg_allowed_2pt_pts=("allowed_2pt_pts", "mean"),
            avg_opp_possessions=("opp_possessions", "mean"),
        )
        .reset_index()
    )

    # Calculate pace-adjusted defensive rating (points allowed per 100 possessions)
    # Only if pace data is available
    if has_pace_data and aggregates["avg_opp_possessions"].notna().any():
        aggregates["def_rating"] = (
            aggregates["avg_allowed_pts"] / aggregates["avg_opp_possessions"].replace(0, 1)
        ) * 100
        aggregates["def_3pm_per100"] = (
            aggregates["avg_allowed_fg3m"] / aggregates["avg_opp_possessions"].replace(0, 1)
        ) * 100
        aggregates["def_2pt_pts_per100"] = (
            aggregates["avg_allowed_2pt_pts"] / aggregates["avg_opp_possessions"].replace(0, 1)
        ) * 100
    else:
        # Fallback: use raw averages (not pace-adjusted)
        # Scale to roughly match pace-adjusted range for compatibility
        aggregates["def_rating"] = aggregates["avg_allowed_pts"]
        aggregates["def_3pm_per100"] = aggregates["avg_allowed_fg3m"]
        aggregates["def_2pt_pts_per100"] = aggregates["avg_allowed_2pt_pts"]

    def compute_recent(team_df: pd.DataFrame, window: int) -> float | None:
        subset = (
            team_df.sort_values("game_date", ascending=False)
            .head(window)["allowed_pts"]
            .dropna()
        )
        if subset.empty:
            return None
        return subset.mean()

    recent_records = []
    for team_id, team_df in df.groupby("team_id"):
        recent_records.append(
            {
                "team_id": team_id,
                "avg_allowed_pts_last3": compute_recent(team_df, 3),
                "avg_allowed_pts_last5": compute_recent(team_df, 5),
            }
        )
    recent_df = pd.DataFrame(recent_records)
    if recent_df.empty:
        recent_df = pd.DataFrame(
            columns=["team_id", "avg_allowed_pts_last3", "avg_allowed_pts_last5"]
        )
    aggregates = aggregates.merge(recent_df, on="team_id", how="left")
    std_df = (
        df.groupby("team_id")["allowed_pts"]
        .std(ddof=0)
        .reset_index(name="std_allowed_pts")
    )
    aggregates = aggregates.merge(std_df, on="team_id", how="left")
    avg_allowed = aggregates["avg_allowed_pts"].fillna(0.0)
    recent5 = aggregates["avg_allowed_pts_last5"].fillna(avg_allowed)
    recent3 = aggregates["avg_allowed_pts_last3"].fillna(avg_allowed)
    std_allowed = aggregates["std_allowed_pts"].fillna(avg_allowed).replace(0, 1.0)
    aggregates["def_composite_score"] = (
        0.4 * avg_allowed
        + 0.3 * recent5
        + 0.2 * recent3
        + 0.1 * (avg_allowed / (std_allowed + 1.0))
    )
    # Percentiles for multi-label style classification
    for col in ["def_rating", "avg_allowed_pts", "avg_allowed_fg3m", "avg_allowed_reb",
                "def_2pt_pts_per100", "def_3pm_per100"]:
        if col not in aggregates.columns:
            aggregates[col] = None
        filled = aggregates[col].fillna(aggregates[col].median())
        aggregates[f"{col}_pct"] = filled.rank(pct=True)

    def classify_styles_multi_label(row: pd.Series) -> list[str]:
        """Returns list of defensive style tags"""
        styles = []

        # Overall tier based on pace-adjusted defensive rating
        def_rating_pct = safe_float(row.get("def_rating_pct"))
        if def_rating_pct is not None:
            if def_rating_pct <= 0.20:
                styles.append("🔒 Elite")
            elif def_rating_pct >= 0.80:
                styles.append("⚠️ Vulnerable")

        # Perimeter defense (3PT)
        fg3m_pct = safe_float(row.get("def_3pm_per100_pct"))
        if fg3m_pct is not None:
            if fg3m_pct >= 0.65:
                styles.append("🎯 Perimeter Leak")
            elif fg3m_pct <= 0.35:
                styles.append("🛡️ Perimeter Lock")

        # Paint defense (2PT points)
        paint_pct = safe_float(row.get("def_2pt_pts_per100_pct"))
        if paint_pct is not None:
            if paint_pct <= 0.35:
                styles.append("🏰 Rim Protector")
            elif paint_pct >= 0.65:
                styles.append("🚪 Paint Vulnerable")

        # Rebounding
        reb_pct = safe_float(row.get("avg_allowed_reb_pct"))
        if reb_pct is not None:
            if reb_pct >= 0.65:
                styles.append("🏀 Board-Soft")
            elif reb_pct <= 0.35:
                styles.append("💪 Glass Cleaner")

        return styles if styles else ["⚖️ Balanced"]

    aggregates["defense_styles"] = aggregates.apply(classify_styles_multi_label, axis=1)
    aggregates["style_tags"] = aggregates["defense_styles"].apply(lambda x: " | ".join(x))

    # Keep single legacy style for backwards compatibility
    def classify_style_legacy(row: pd.Series) -> str:
        styles = row.get("defense_styles", [])
        if "🔒 Elite" in str(styles):
            return "Elite"
        if "🎯 Perimeter Leak" in str(styles):
            return "Perimeter Leak"
        if "🏀 Board-Soft" in str(styles):
            return "Board-Soft"
        if "🏰 Rim Protector" in str(styles):
            return "Rim Protector"
        return "Balanced"

    aggregates["defense_style"] = aggregates.apply(classify_style_legacy, axis=1)
    return aggregates


def build_player_style_splits(
    db_path: str,
    season: str,
    season_type: str,
    style_map: Dict[int, str],
) -> Dict[int, Dict[str, float]]:
    """
    Compute average points per player against each defense style.
    """
    if not style_map:
        return {}
    query = """
        SELECT p.player_id,
               p.points,
               t.opp_team_id
        FROM player_game_logs AS p
        JOIN team_game_logs AS t
          ON p.game_id = t.game_id
         AND p.team_id = t.team_id
        WHERE p.season = ?
          AND p.season_type = ?
          AND p.points IS NOT NULL
    """
    df = run_query(db_path, query, params=(season, season_type))

    # Fallback to most recent season if requested season has no data
    if df.empty:
        try:
            debug_query = "SELECT DISTINCT season FROM player_game_logs ORDER BY season DESC LIMIT 1"
            available_seasons = run_query(db_path, debug_query)
            if not available_seasons.empty:
                fallback_season = available_seasons.iloc[0]["season"]
                print(f"[INFO] No player style data for season '{season}'. Using fallback: {fallback_season}")
                df = run_query(db_path, query, params=(fallback_season, season_type))
                if df.empty:
                    return {}
            else:
                return {}
        except Exception:
            return {}

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["opp_team_id"] = pd.to_numeric(df["opp_team_id"], errors="coerce")
    df = df.dropna(subset=["player_id", "points", "opp_team_id"])
    df["defense_style"] = df["opp_team_id"].map(style_map).fillna("Neutral")
    df = df[df["defense_style"] != "Neutral"]
    grouped = (
        df.groupby(["player_id", "defense_style"])["points"]
        .mean()
        .reset_index()
    )
    splits: Dict[int, Dict[str, float]] = {}
    for _, row in grouped.iterrows():
        pid = int(row["player_id"])
        style = str(row["defense_style"])
        splits.setdefault(pid, {})[style] = float(row["points"])
    return splits


def build_player_vs_team_history(
    db_path: str,
    season: str,
    season_type: str,
) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    Build comprehensive player vs specific team history.
    Returns: {player_id: {opponent_team_id: {stats}}}
    """
    query = """
        SELECT p.player_id,
               p.player_name,
               p.points,
               p.fg3m,
               p.minutes,
               t.opp_team_id,
               p.game_date
        FROM player_game_logs AS p
        JOIN team_game_logs AS t
          ON p.game_id = t.game_id
         AND p.team_id = t.team_id
        WHERE p.season = ?
          AND p.season_type = ?
          AND p.points IS NOT NULL
        ORDER BY p.player_id, t.opp_team_id, p.game_date
    """
    df = run_query(db_path, query, params=(season, season_type))

    # Debug: Check if we got any data
    if df.empty:
        # Try to find what seasons are actually available and use most recent
        try:
            debug_query = "SELECT DISTINCT season FROM player_game_logs ORDER BY season DESC LIMIT 1"
            available_seasons = run_query(db_path, debug_query)
            if not available_seasons.empty:
                fallback_season = available_seasons.iloc[0]["season"]
                print(f"[INFO] No data for season '{season}'. Using fallback season: {fallback_season}")

                # Try query again with fallback season
                df = run_query(db_path, query, params=(fallback_season, season_type))
                if df.empty:
                    return {}
            else:
                return {}
        except Exception as e:
            print(f"[ERROR] Failed to build player vs team history: {e}")
            return {}

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["opp_team_id"] = pd.to_numeric(df["opp_team_id"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["fg3m"] = pd.to_numeric(df["fg3m"], errors="coerce")
    df = df.dropna(subset=["player_id", "opp_team_id", "points"])

    # Group by player and opponent team
    grouped = df.groupby(["player_id", "opp_team_id"])["points"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    grouped.columns = ["player_id", "opp_team_id", "games", "avg_pts",
                       "median_pts", "std_pts", "min_pts", "max_pts"]

    # Build nested dictionary
    history: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for _, row in grouped.iterrows():
        pid = int(row["player_id"])
        opp_id = int(row["opp_team_id"])

        if pid not in history:
            history[pid] = {}

        history[pid][opp_id] = {
            "games": int(row["games"]),
            "avg_pts": float(row["avg_pts"]),
            "median_pts": float(row["median_pts"]),
            "std_pts": float(row["std_pts"]) if pd.notna(row["std_pts"]) else 0.0,
            "min_pts": float(row["min_pts"]),
            "max_pts": float(row["max_pts"]),
        }

    return history


def evaluate_matchup_quality(
    player_avg: float,
    player_vs_opp_avg: float | None,
    player_vs_style_avg: float | None,
    games_vs_opp: int,
    games_vs_style: int,
    std_vs_opp: float | None,
) -> tuple[str, str, float]:
    """
    Evaluate matchup quality and return (rating, warning, confidence).

    PRIORITY ORDER (revised to emphasize defensive style):
    1. Defense style matchups (larger sample, more predictive)
    2. Head-to-head history (only if significant sample and strong signal)
    3. Neutral default

    Returns:
        rating: "Excellent", "Good", "Neutral", "Difficult", "Avoid"
        warning: Human-readable warning message or empty string
        confidence: 0.0 to 1.0 confidence score
    """
    # Calculate confidence: style data weighted MORE than head-to-head
    # Style: 10% per "game", Head-to-head: 5% per game
    confidence = min(1.0, (games_vs_style * 0.10) + (games_vs_opp * 0.05))

    # PRIMARY: Check defense style matchup (if available, use this first)
    if player_vs_style_avg is not None and games_vs_style >= 3:
        diff_pct = (player_vs_style_avg - player_avg) / player_avg if player_avg > 0 else 0

        if diff_pct <= -0.18:  # 18%+ worse vs this defense style
            warning = f"Struggles vs this defense style ({player_vs_style_avg:.1f} vs {player_avg:.1f})"
            # Only override if head-to-head strongly contradicts
            if player_vs_opp_avg is not None and games_vs_opp >= 4:
                opp_diff = (player_vs_opp_avg - player_avg) / player_avg
                if opp_diff >= 0.25:  # Strong positive head-to-head overrides
                    return ("Good", f"Excels vs this team despite tough style", confidence)
            return ("Difficult", warning, confidence)

        elif diff_pct >= 0.12:  # 12%+ better vs this defense style
            # Check if head-to-head contradicts strongly
            if player_vs_opp_avg is not None and games_vs_opp >= 4:
                opp_diff = (player_vs_opp_avg - player_avg) / player_avg
                if opp_diff <= -0.25:  # Strong negative head-to-head overrides
                    warning = f"Historically struggles vs this team ({player_vs_opp_avg:.1f})"
                    return ("Difficult", warning, confidence)
            return ("Good", f"Performs well vs this defense style", confidence)

        else:
            # Neutral style matchup - check head-to-head for tie-breaker
            if player_vs_opp_avg is not None and games_vs_opp >= 3:
                opp_diff = (player_vs_opp_avg - player_avg) / player_avg
                if opp_diff >= 0.20:
                    return ("Good", f"Solid history vs this team", confidence)
                elif opp_diff <= -0.20:
                    warning = f"Below average vs this team ({player_vs_opp_avg:.1f})"
                    return ("Difficult", warning, confidence)
            return ("Neutral", "", max(0.3, confidence))

    # SECONDARY: Head-to-head only (no style data available)
    if player_vs_opp_avg is not None and games_vs_opp >= 3:
        diff_pct = (player_vs_opp_avg - player_avg) / player_avg if player_avg > 0 else 0

        # Check for high variance (inconsistent matchup)
        is_volatile = std_vs_opp is not None and std_vs_opp > (player_avg * 0.4)

        if diff_pct <= -0.25:  # 25%+ worse
            warning = f"Struggles vs this team (avg {player_vs_opp_avg:.1f} vs season {player_avg:.1f})"
            if is_volatile:
                warning += " - Inconsistent"
            return ("Avoid", warning, confidence)
        elif diff_pct <= -0.15:  # 15-25% worse
            warning = f"Below average vs this team ({player_vs_opp_avg:.1f} vs {player_avg:.1f})"
            return ("Difficult", warning, confidence)
        elif diff_pct >= 0.20:  # 20%+ better
            warning = f"Excels vs this team (avg {player_vs_opp_avg:.1f} vs season {player_avg:.1f})"
            return ("Excellent", warning, confidence)
        elif diff_pct >= 0.10:  # 10-20% better
            return ("Good", "", confidence)
        else:
            return ("Neutral", "", confidence * 0.6)

    # No meaningful history
    return ("Neutral", "", 0.0)


def calculate_daily_pick_score(
    player_season_avg: float,
    player_projection: float,
    projection_confidence: float,
    matchup_rating: str,
    opp_def_rating: float | None,
    league_avg_def_rating: float = 112.0,
) -> tuple[float, str, str]:
    """
    Calculate unified DFS Score (0-100) combining all analytics.

    This score answers: "How good is this player as a pick TODAY?"
    Combines: projection, matchup history, opponent defense, confidence.

    Args:
        player_season_avg: Player's season average PPG
        player_projection: Smart projection for today
        projection_confidence: Confidence in projection (0-1)
        matchup_rating: Player-specific matchup quality
        opp_def_rating: Opponent's defensive rating
        league_avg_def_rating: League average for comparison

    Returns:
        (score 0-100, grade, explanation)
    """
    # 1. Base score from projected points (0-70 range)
    # Scale typical 5-40 PPG range to reasonable scores
    # Formula: (proj - 3) * 1.8 gives 5PPG=3.6, 30PPG=48.6, 40PPG=66.6
    # More generous to reward elite scorers even with neutral matchups
    base_score = max(0, (player_projection - 3) * 1.8)

    # 2. Matchup quality bonus/penalty (-20 to +20)
    matchup_adjustments = {
        "Excellent": 20,
        "Good": 10,
        "Neutral": 0,
        "Difficult": -10,
        "Avoid": -20,
    }
    matchup_bonus = matchup_adjustments.get(matchup_rating, 0)

    # 3. Opponent defense adjustment (-10 to +10)
    # Weaker defense = bonus, stronger defense = penalty
    if opp_def_rating is not None:
        def_diff = (league_avg_def_rating - opp_def_rating) / league_avg_def_rating
        defense_adjustment = def_diff * 10  # Scale to ±10
    else:
        defense_adjustment = 0

    # 4. Confidence bonus (0 to +15)
    # High confidence = bonus points, low confidence = no bonus (not penalty)
    # Formula: confidence * 15 gives 0-15 point bonus
    confidence_bonus = projection_confidence * 15

    # Calculate final score (all additive, no crushing multiplier)
    final_score = base_score + matchup_bonus + defense_adjustment + confidence_bonus
    final_score = max(0, min(100, final_score))

    # Build contextual explanation based on what's driving the score
    factors = []

    # Projection factor
    if player_projection >= 30:
        factors.append("elite scorer")
    elif player_projection >= 25:
        factors.append("strong scorer")
    elif player_projection >= 18:
        factors.append("solid volume")
    else:
        factors.append("limited volume")

    # Matchup factor
    if matchup_rating == "Excellent":
        factors.append("excels vs this matchup")
    elif matchup_rating == "Good":
        factors.append("favorable matchup")
    elif matchup_rating == "Difficult":
        factors.append("struggles vs this defense")
    elif matchup_rating == "Avoid":
        factors.append("historically poor vs opponent")

    # Defense factor (only mention if significant)
    if opp_def_rating is not None:
        if opp_def_rating >= 118:
            factors.append("weak defense")
        elif opp_def_rating <= 106:
            factors.append("elite defense")

    # Confidence factor
    if projection_confidence >= 0.75:
        factors.append("high confidence")
    elif projection_confidence < 0.50:
        factors.append("limited data")

    # Build explanation sentence
    explanation = ", ".join(factors)

    # Grade the score with clear tiers
    if final_score >= 80:
        grade = "🔥 Elite"
        explanation = f"Elite pick: {explanation}"
    elif final_score >= 65:
        grade = "⭐ Excellent"
        explanation = f"Strong pick: {explanation}"
    elif final_score >= 50:
        grade = "✓ Solid"
        explanation = f"Reliable: {explanation}"
    elif final_score >= 35:
        grade = "⚠️ Risky"
        explanation = f"Caution: {explanation}"
    else:
        grade = "❌ Avoid"
        explanation = f"Avoid: {explanation}"

    return (final_score, grade, explanation)


def calculate_smart_ppg_projection(
    season_avg: float,
    recent_avg_5: float | None,
    recent_avg_3: float | None,
    vs_opp_team_avg: float | None,
    vs_opp_team_games: int,
    vs_defense_style_avg: float | None,
    vs_defense_style_games: int,
    opp_def_rating: float | None,
    opp_pace: float | None,
    league_avg_def_rating: float = 112.0,
    league_avg_pace: float = 99.0,
    # NEW: Enhanced analytics parameters
    opponent_correlation: Optional[pca.OpponentCorrelation] = None,
    pace_split: Optional[dta.DefenseTypePerformance] = None,
    defense_quality_split: Optional[dta.DefenseTypePerformance] = None,
) -> tuple[float, float, float, float, dict, str]:
    """
    Calculate smart PPG projection using multi-factor weighted model.

    Args:
        season_avg: Player's season average PPG
        recent_avg_5: Last 5 games average
        recent_avg_3: Last 3 games average
        vs_opp_team_avg: Historical average vs this specific opponent
        vs_opp_team_games: Games played vs this opponent
        vs_defense_style_avg: Average vs this defense style
        vs_defense_style_games: Games vs this defense style
        opp_def_rating: Opponent's defensive rating (per 100 possessions)
        opp_pace: Opponent's pace (possessions per game)
        league_avg_def_rating: League average defensive rating
        league_avg_pace: League average pace

    Returns:
        (projection, confidence, floor, ceiling, breakdown_dict)
    """
    # Initialize components
    components = {}
    weights = {}

    # 1. Season Average (Baseline - 25% weight)
    components["season"] = season_avg
    weights["season"] = 0.25

    # 2. Recent Form (20% weight, higher if trending)
    if recent_avg_3 is not None and recent_avg_5 is not None:
        # Weight more recent games higher
        components["recent"] = (recent_avg_3 * 0.6) + (recent_avg_5 * 0.4)
        weights["recent"] = 0.20
    elif recent_avg_5 is not None:
        components["recent"] = recent_avg_5
        weights["recent"] = 0.15
    else:
        components["recent"] = season_avg
        weights["recent"] = 0.10

    # 3. Team-Specific Matchup (25-35% weight - HIGHEST PRIORITY)
    # ENHANCED: Use opponent correlation if available for better accuracy
    analytics_used = []

    if opponent_correlation is not None and opponent_correlation.games_vs_opponent >= 2:
        # Use correlation data (more sophisticated than simple average)
        # Correlation score (0-100) indicates matchup quality
        correlation_confidence = opponent_correlation.matchup_score / 100.0

        # Use PPG delta from correlation (accounts for sample size + consistency)
        components["vs_team"] = season_avg + opponent_correlation.pts_delta

        # Weight increases with better correlation scores (55-70 = good data)
        base_weight = 0.25
        if opponent_correlation.matchup_score >= 60:
            base_weight = 0.35  # Elite matchup = higher confidence
        elif opponent_correlation.matchup_score >= 55:
            base_weight = 0.30  # Good matchup
        elif opponent_correlation.matchup_score <= 40:
            base_weight = 0.20  # Poor matchup but still use data

        weights["vs_team"] = base_weight * min(1.0, opponent_correlation.games_vs_opponent / 5.0)
        analytics_used.append("🎯")  # Correlation indicator

    elif vs_opp_team_avg is not None and vs_opp_team_games >= 2:
        # Fallback to simple historical average
        confidence_factor = min(1.0, vs_opp_team_games / 5.0)
        components["vs_team"] = vs_opp_team_avg
        weights["vs_team"] = 0.30 * confidence_factor
    else:
        components["vs_team"] = None
        weights["vs_team"] = 0.0

    # 4. Defense Style Matchup (10-15% weight - only if team matchup weak)
    if weights["vs_team"] < 0.15 and vs_defense_style_avg is not None and vs_defense_style_games >= 3:
        components["vs_style"] = vs_defense_style_avg
        weights["vs_style"] = 0.15
    else:
        components["vs_style"] = None
        weights["vs_style"] = 0.0

    # 5. Opponent Defense Quality (10-15% adjustment)
    # ENHANCED: Use player-specific defense quality split if available
    if defense_quality_split is not None and defense_quality_split.games_played >= 2:
        # Player-specific performance vs this defense quality level
        components["def_quality"] = season_avg + defense_quality_split.pts_vs_average
        weights["def_quality"] = 0.15  # Higher weight with real data
        analytics_used.append("🛡️")  # Defense split indicator

    elif opp_def_rating is not None:
        # Fallback to generic adjustment
        def_adjustment = (league_avg_def_rating - opp_def_rating) / league_avg_def_rating
        def_adjustment = max(-0.15, min(0.15, def_adjustment))
        components["def_quality"] = season_avg * (1 + def_adjustment)
        weights["def_quality"] = 0.10
    else:
        components["def_quality"] = season_avg
        weights["def_quality"] = 0.0

    # 6. Pace Adjustment (5-15% adjustment)
    # ENHANCED: Use player-specific pace split if available
    if pace_split is not None and pace_split.games_played >= 2:
        # Player-specific performance vs this pace type
        components["pace"] = season_avg + pace_split.pts_vs_average
        weights["pace"] = 0.15  # Significantly higher weight with real data
        analytics_used.append("⚡")  # Pace split indicator

    elif opp_pace is not None:
        # Fallback to generic adjustment
        pace_adjustment = (opp_pace - league_avg_pace) / league_avg_pace
        pace_adjustment = max(-0.10, min(0.10, pace_adjustment))
        components["pace"] = season_avg * (1 + pace_adjustment)
        weights["pace"] = 0.05
    else:
        components["pace"] = season_avg
        weights["pace"] = 0.0

    # Track which analytics were used
    analytics_indicators = "".join(analytics_used) if analytics_used else "📊"

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate weighted projection
    projection = sum(
        components[k] * weights[k]
        for k in components
        if components[k] is not None and weights[k] > 0
    )

    # Calculate confidence (0-100%) with more granular scoring
    confidence_score = 0.0

    # 1. Base confidence from season data (20-30% based on sample size)
    # Assume typical player has played 15-25 games in early season
    base_conf = 0.25  # Middle ground
    confidence_score += base_conf

    # 2. Matchup-specific confidence (most important for differentiation)
    if vs_opp_team_games >= 5:
        # Strong team history - very confident
        confidence_score += 0.40
    elif vs_opp_team_games >= 3:
        # Good team history - confident
        confidence_score += 0.30
    elif vs_opp_team_games >= 2:
        # Some team history - moderately confident
        confidence_score += 0.20
    elif vs_defense_style_games >= 8:
        # Extensive style data - decent fallback
        confidence_score += 0.18
    elif vs_defense_style_games >= 5:
        # Good style data - moderate fallback
        confidence_score += 0.12
    elif vs_defense_style_games >= 3:
        # Some style data - weak fallback
        confidence_score += 0.08

    # 3. Recent form confidence (varies by recency and volatility)
    if recent_avg_3 is not None and recent_avg_5 is not None:
        # Have both L3 and L5 - check consistency
        if abs(recent_avg_3 - recent_avg_5) / season_avg < 0.15:
            # Consistent recent form - more confident
            confidence_score += 0.15
        else:
            # Volatile recent form - less confident
            confidence_score += 0.10
    elif recent_avg_3 is not None:
        # Only L3 - moderate confidence
        confidence_score += 0.10
    elif recent_avg_5 is not None:
        # Only L5 - lower confidence
        confidence_score += 0.08

    # 4. Opponent data confidence (granular based on quality)
    if opp_def_rating is not None and opp_pace is not None:
        # Have both def rating and pace - good opponent intel
        confidence_score += 0.12
    elif opp_def_rating is not None:
        # Only def rating - moderate intel
        confidence_score += 0.08

    # 5. Enhanced analytics confidence boosts
    if opponent_correlation is not None and opponent_correlation.games_vs_opponent >= 2:
        # High-quality player-specific matchup data
        confidence_score += 0.15
    if pace_split is not None and pace_split.games_played >= 2:
        # Player-specific pace performance data
        confidence_score += 0.08
    if defense_quality_split is not None and defense_quality_split.games_played >= 2:
        # Player-specific defense quality performance data
        confidence_score += 0.08

    # Cap at 95% (never 100% certain)
    confidence_score = min(0.95, confidence_score)

    # Calculate floor and ceiling (confidence intervals)
    # Wider intervals for lower confidence
    interval_width = season_avg * 0.30 * (1 - confidence_score)
    floor = max(0, projection - interval_width)
    ceiling = projection + interval_width

    # Build breakdown for transparency
    breakdown = {
        "projection": projection,
        "confidence": confidence_score,
        "floor": floor,
        "ceiling": ceiling,
        "components": {
            k: {"value": components[k], "weight": weights[k] * 100}
            for k in components
            if components[k] is not None and weights[k] > 0
        },
        "analytics_used": analytics_indicators
    }

    return projection, confidence_score, floor, ceiling, breakdown, analytics_indicators


def build_player_style_leaders(
    db_path: str,
    season: str,
    season_type: str,
    style_map: Dict[int, str],
    min_games: int = 1,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return top players by average points vs each defense style.
    """
    if not style_map:
        return pd.DataFrame()
    query = """
        SELECT p.player_id,
               p.player_name,
               p.team_id,
               p.points,
               t.opp_team_id
        FROM player_game_logs AS p
        JOIN team_game_logs AS t
          ON p.game_id = t.game_id
         AND p.team_id = t.team_id
        WHERE p.season = ?
          AND p.season_type = ?
          AND p.points IS NOT NULL
    """
    df = run_query(db_path, query, params=(season, season_type))
    if df.empty:
        return pd.DataFrame()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["opp_team_id"] = pd.to_numeric(df["opp_team_id"], errors="coerce")
    df = df.dropna(subset=["player_id", "team_id", "points", "opp_team_id"])
    df["defense_style"] = df["opp_team_id"].map(style_map).fillna("Neutral")
    df = df[df["defense_style"] != "Neutral"]
    grouped = (
        df.groupby(["defense_style", "player_id", "player_name", "team_id"])["points"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "games_vs_style", "mean": "avg_pts_vs_style"})
    )
    grouped = grouped[grouped["games_vs_style"] >= max(1, min_games)]
    if grouped.empty:
        return grouped
    grouped = (
        grouped.sort_values(["defense_style", "avg_pts_vs_style"], ascending=[True, False])
        .groupby("defense_style")
        .head(top_n)
    )
    team_lookup = run_query(
        db_path,
        "SELECT team_id, full_name FROM teams",
    )
    team_lookup["team_id"] = pd.to_numeric(team_lookup["team_id"], errors="coerce")
    grouped = grouped.merge(team_lookup, on="team_id", how="left")
    return grouped


def default_game_date() -> date:
    return datetime.now(EASTERN_TZ).date()


def format_game_date_str(target: date) -> str:
    return target.strftime("%m/%d/%Y")


def parse_game_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    value_str = str(value)
    if not value_str:
        return None
    if value_str.endswith("Z"):
        value_str = value_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(value_str)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=EASTERN_TZ)
    else:
        dt = dt.astimezone(EASTERN_TZ)
    return dt


def format_tipoff_display(row: Mapping[str, Any]) -> Tuple[str, float]:
    dt = parse_game_datetime(row.get("GAME_DATE_EST"))
    default_display = "TBD"
    timestamp = 0.0
    if dt:
        timestamp = dt.timestamp()
        default_display = dt.strftime("%I:%M %p ET").lstrip("0")
    status = str(row.get("GAME_STATUS_TEXT") or "").strip()
    if status and status.lower() not in {"scheduled"}:
        return status, timestamp
    return default_display, timestamp


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def format_team_label(city: Any, name: Any) -> str:
    parts = [safe_text(city), safe_text(name)]
    label = " ".join(part for part in parts if part).strip()
    return label or "TBD"


def format_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def format_rank(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        rank_int = int(value)
    except (TypeError, ValueError):
        return ""
    return f"#{rank_int}"


def format_number(value: Any, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if decimals <= 0:
        return f"{number:.0f}"
    return f"{number:.{decimals}f}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def derive_home_away(team_name: str, matchup: str) -> tuple[str, str]:
    if not matchup:
        return team_name, ""
    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
        home_team = team_name
        away_team = parts[1].strip()
    elif " @ " in matchup:
        parts = matchup.split(" @ ")
        away_team = team_name
        home_team = parts[1].strip()
    else:
        home_team = team_name
        away_team = ""
    return home_team, away_team


def minutes_str_to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value)
    if ":" in value_str:
        parts = value_str.split(":")
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + seconds / 60.0
        except ValueError:
            return None
    try:
        return float(value_str)
    except ValueError:
        return None


@st.cache_data(ttl=300)
def fetch_scoreboard_frames(game_date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date_str, league_id="00")
    return (
        scoreboard.game_header.get_data_frame(),
        scoreboard.line_score.get_data_frame(),
    )


def build_games_table(
    db_path: str,
    game_date: date,
    context_season: str,
    context_season_type: str,
) -> tuple[pd.DataFrame, Dict[int, Mapping[str, Any]]]:
    header_df, line_df = fetch_scoreboard_frames(format_game_date_str(game_date))
    if header_df.empty:
        return pd.DataFrame(), {}

    line_lookup: Dict[int, Mapping[str, Any]] = {}
    for _, line in line_df.iterrows():
        line_lookup[int(line["TEAM_ID"])] = line

    context_query = """
        SELECT team_id, wins, losses, win_pct, conference_rank, division_rank
        FROM standings
        WHERE season = ?
          AND season_type = ?
    """
    context_df = run_query(db_path, context_query, params=(context_season, context_season_type))
    if not context_df.empty and "conference" in context_df.columns:
        context_df["computed_conf_rank"] = (
            context_df.groupby("conference")["win_pct"]
            .rank(method="min", ascending=False)
        )
    context_map: Dict[int, Mapping[str, Any]] = {
        int(row["team_id"]): row for _, row in context_df.iterrows()
    }

    team_lookup_query = """
        SELECT team_id, full_name, nickname, city
        FROM teams
    """
    team_lookup_df = run_query(db_path, team_lookup_query)
    team_label_map: Dict[int, str] = {}
    for _, row in team_lookup_df.iterrows():
        label = row.get("full_name") or format_team_label(row.get("city"), row.get("nickname"))
        if label:
            team_label_map[int(row["team_id"])] = label

    scoring_map: Dict[int, Mapping[str, Any]] = {}
    scoring_df = load_team_scoring_stats(db_path, context_season, context_season_type)
    if not scoring_df.empty:
        scoring_map = {
            int(row["team_id"]): row.to_dict() for _, row in scoring_df.iterrows()
        }

    def get_context(team_id: int) -> Mapping[str, Any]:
        return context_map.get(team_id, {})

    def get_line_stat(team_id: int, column: str) -> str:
        row = line_lookup.get(team_id)
        if row is None:
            return ""
        value = row.get(column)
        return "" if value is None else str(value)

    def resolve_team_label(team_id: int, city: Any, name: Any) -> str:
        label = format_team_label(city, name)
        if label == "TBD":
            return team_label_map.get(team_id, label)
        return label

    def get_record(team_id: int) -> str:
        record = get_line_stat(team_id, "TEAM_WINS_LOSSES")
        if record:
            return record
        ctx = get_context(team_id)
        wins = ctx.get("wins")
        losses = ctx.get("losses")
        if wins is None or losses is None:
            return ""
        try:
            return f"{int(wins)}-{int(losses)}"
        except (TypeError, ValueError):
            return ""

    def get_conf_rank(team_id: int) -> Any:
        ctx = get_context(team_id)
        rank_value = ctx.get("conference_rank")
        if rank_value in (None, "", "None"):
            rank_value = ctx.get("computed_conf_rank")
        return rank_value

    def get_scoring_stat(team_id: int, column: str, decimals: int) -> str:
        stats = scoring_map.get(team_id)
        if not stats:
            return ""
        return format_number(stats.get(column), decimals)

    rows: list[Dict[str, Any]] = []
    for _, game in header_df.iterrows():
        home_id = int(game["HOME_TEAM_ID"])
        away_id = int(game["VISITOR_TEAM_ID"])
        tipoff_display, tipoff_ts = format_tipoff_display(game)
        arena = safe_text(game.get("ARENA_NAME"))
        city = safe_text(game.get("HOME_TEAM_CITY"))
        arena_display = f"{arena} ({city})" if arena and city else arena or city
        national_tv = str(game.get("NATL_TV_BROADCASTER_ABBREVIATION") or "")
        away_ctx = get_context(away_id)
        home_ctx = get_context(home_id)
        rows.append(
            {
                "game_id": game["GAME_ID"],
                "away_team_id": away_id,
                "home_team_id": home_id,
                "sort_ts": tipoff_ts,
                "Start (ET)": tipoff_display,
                "Status": str(game.get("GAME_STATUS_TEXT") or ""),
                "Away": resolve_team_label(
                    away_id, game.get("VISITOR_TEAM_CITY"), game.get("VISITOR_TEAM_NAME")
                ),
                "Away Record": get_record(away_id),
                "Away Win% (Standings)": format_pct(away_ctx.get("win_pct")),
                "Away Conf Rank": format_rank(get_conf_rank(away_id)),
                "Away Total Pts": get_scoring_stat(away_id, "total_pts", 0),
                "Away Avg Pts": get_scoring_stat(away_id, "avg_pts", 1),
                "Away Median Pts": get_scoring_stat(away_id, "median_pts", 1),
                "Away Median 3PM": get_scoring_stat(away_id, "median_fg3m", 1),
                "Away Last5 Median Pts": get_scoring_stat(away_id, "median_pts_last5", 1),
                "Away Last3 Avg Pts": get_scoring_stat(away_id, "avg_pts_last3", 1),
                "Away Last3 Avg 3PM": get_scoring_stat(away_id, "avg_fg3m_last3", 1),
                "Away Last5 Avg Pts": get_scoring_stat(away_id, "avg_pts_last5", 1),
                "Away Last5 Avg 3PM": get_scoring_stat(away_id, "avg_fg3m_last5", 1),
                "Home": resolve_team_label(
                    home_id, game.get("HOME_TEAM_CITY"), game.get("HOME_TEAM_NAME")
                ),
                "Home Record": get_record(home_id),
                "Home Win% (Standings)": format_pct(home_ctx.get("win_pct")),
                "Home Conf Rank": format_rank(get_conf_rank(home_id)),
                "Home Total Pts": get_scoring_stat(home_id, "total_pts", 0),
                "Home Avg Pts": get_scoring_stat(home_id, "avg_pts", 1),
                "Home Median Pts": get_scoring_stat(home_id, "median_pts", 1),
                "Home Median 3PM": get_scoring_stat(home_id, "median_fg3m", 1),
                "Home Last5 Median Pts": get_scoring_stat(home_id, "median_pts_last5", 1),
                "Home Last3 Avg Pts": get_scoring_stat(home_id, "avg_pts_last3", 1),
                "Home Last3 Avg 3PM": get_scoring_stat(home_id, "avg_fg3m_last3", 1),
                "Home Last5 Avg Pts": get_scoring_stat(home_id, "avg_pts_last5", 1),
                "Home Last5 Avg 3PM": get_scoring_stat(home_id, "avg_fg3m_last5", 1),
                "Arena": arena_display,
                "National TV": national_tv,
                "Series": str(game.get("SERIES_TEXT") or ""),
            }
        )

    games_df = pd.DataFrame(rows)
    if "sort_ts" in games_df.columns:
        games_df = games_df.sort_values("sort_ts").drop(columns=["sort_ts"])
    return games_df, scoring_map


def normalize_optional(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip()
    return cleaned or None


def rebuild_database(config: Dict[str, Any], reason: str) -> bool:
    try:
        with st.spinner(
            f"{reason} (season {config['season']} - {config['season_type']}). "
            "This can take several minutes while the NBA API is queried."
        ):
            build_database(**config)
        st.cache_data.clear()
        st.success("Database refreshed from the NBA API.")
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Database rebuild failed: {exc}")
        return False


if "db_path_input" not in st.session_state:
    st.session_state["db_path_input"] = str(DEFAULT_DB_PATH)
if "predictions_path_input" not in st.session_state:
    st.session_state["predictions_path_input"] = str(DEFAULT_PREDICTIONS_PATH)
if "auto_build_attempted" not in st.session_state:
    st.session_state["auto_build_attempted"] = False

with st.sidebar:
    st.header("Data Inputs")

    st.text_input(
        "SQLite database path",
        value=st.session_state["db_path_input"],
        key="db_path_input",
        help="Point to an existing nba_stats.db or let the app build one automatically.",
    )

    uploaded_db = st.file_uploader(
        "Upload nba_stats.db",
        type=["db", "sqlite"],
        help="Used mainly on Streamlit Cloud; uploads are stored temporarily per session.",
    )
    if uploaded_db is not None:
        saved_db_path = persist_uploaded_file(uploaded_db, suffix=".db")
        st.session_state["db_path_input"] = str(saved_db_path)
        st.success(f"Uploaded database stored at {saved_db_path}")

    st.text_input(
        "Predictions CSV (optional)",
        value=st.session_state["predictions_path_input"],
        key="predictions_path_input",
    )
    uploaded_predictions = st.file_uploader(
        "Upload predictions.csv",
        type=["csv"],
        help="Optional file produced by predict_top_3pm.py for the Predictions tab.",
        key="predictions_uploader",
    )
    if uploaded_predictions is not None:
        saved_predictions_path = persist_uploaded_file(uploaded_predictions, suffix=".csv")
        st.session_state["predictions_path_input"] = str(saved_predictions_path)
        st.success(f"Uploaded predictions stored at {saved_predictions_path}")

    if st.button("Clear cached data"):
        st.cache_data.clear()
        st.success("Caches cleared. Rerun queries to refresh.")

    with st.expander("NBA API Builder", expanded=False):
        builder_season = st.text_input(
            "Season label (e.g., 2024-25)",
            value=DEFAULT_SEASON,
            key="builder_season",
        )
        builder_season_type = st.selectbox(
            "Season type",
            options=[DEFAULT_SEASON_TYPE, "Playoffs", "Pre Season"],
            index=0,
            key="builder_season_type",
        )
        builder_include_rosters = st.checkbox(
            "Include team rosters",
            value=True,
            key="builder_include_rosters",
        )
        builder_throttle = st.slider(
            "Throttle between API calls (seconds)",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.1,
            key="builder_throttle",
        )
        shooting_season_override = st.text_input(
            "Shooting season override (optional)",
            placeholder="Leave blank to mirror season above",
            key="builder_shooting_season",
        )
        shooting_type_override = st.text_input(
            "Shooting season type override (optional)",
            placeholder="Leave blank to mirror season type above",
            key="builder_shooting_type",
        )
        link_views = st.checkbox(
            "Use the same season for leader/defense views",
            value=True,
            key="builder_link_views",
        )

        def get_view_input(label: str, key: str, default: str) -> str:
            return st.text_input(
                label,
                value=default,
                key=key,
            )

        three_pt_view_season = builder_season
        points_view_season = builder_season
        defense_view_season = builder_season
        defense_pts_view_season = builder_season
        defense_mix_view_season = DEFAULT_DEFENSE_MIX_SEASON
        if not link_views:
            three_pt_view_season = get_view_input(
                "3PT leaderboard season", "builder_three_pt_view", builder_season
            )
            points_view_season = get_view_input(
                "Points leaderboard season", "builder_points_view", builder_season
            )
            defense_view_season = get_view_input(
                "3PT defense season", "builder_defense_view", builder_season
            )
            defense_pts_view_season = get_view_input(
                "Points allowed season", "builder_defense_pts_view", builder_season
            )
            defense_mix_view_season = get_view_input(
                "Defense mix season", "builder_defense_mix_view", DEFAULT_DEFENSE_MIX_SEASON
            )

        builder_config: Dict[str, Any] = {
            "db_path": st.session_state["db_path_input"],
            "season": builder_season.strip() or DEFAULT_SEASON,
            "season_type": builder_season_type,
            "include_rosters": builder_include_rosters,
            "throttle_seconds": float(builder_throttle),
            "shooting_season": normalize_optional(shooting_season_override),
            "shooting_season_type": normalize_optional(shooting_type_override),
            "top_3pt_view_season": three_pt_view_season.strip() or builder_season,
            "top_3pt_view_season_type": None,
            "top_pts_view_season": points_view_season.strip() or builder_season,
            "top_pts_view_season_type": None,
            "defense_view_season": defense_view_season.strip() or builder_season,
            "defense_view_season_type": None,
            "defense_pts_view_season": defense_pts_view_season.strip() or builder_season,
            "defense_pts_view_season_type": None,
            "defense_mix_view_season": defense_mix_view_season.strip() or builder_season,
            "defense_mix_view_season_type": None,
        }

        if st.button("Rebuild database from NBA API", use_container_width=True):
            st.session_state["auto_build_attempted"] = True
            rebuild_database(builder_config, "Manual rebuild requested")

db_path = Path(st.session_state["db_path_input"]).expanduser()
if not db_path.exists():
    if not st.session_state["auto_build_attempted"]:
        st.session_state["auto_build_attempted"] = True
        st.info("No local database found. Building from the NBA API now...")
        build_ok = rebuild_database(builder_config, "Automatic rebuild")
        if not build_ok:
            st.stop()
        db_path = Path(st.session_state["db_path_input"]).expanduser()
    else:
        st.error(
            f"SQLite database not found at `{db_path}` even after rebuild attempts."
        )
        st.stop()

# Key metrics ---------------------------------------------------------------
summary_queries = {
    "Teams": "SELECT COUNT(*) AS total FROM teams",
    "Players": "SELECT COUNT(*) AS total FROM players",
    "Player Games": "SELECT COUNT(*) AS total FROM player_game_logs",
    "Team Games": "SELECT COUNT(*) AS total FROM team_game_logs",
}

metric_cols = st.columns(len(summary_queries))
for metric_col, (label, query) in zip(metric_cols, summary_queries.items()):
    try:
        total_df = run_query(str(db_path), query)
        metric_col.metric(label, f"{int(total_df.iloc[0, 0]):,}")
    except Exception as exc:  # noqa: BLE001 - surface errors to the UI
        metric_col.error(f"{label}: {exc}")

tab_titles = [
    "Today's Games",
    "Matchup Spotlight",
    "Daily Leaders",
    "Player Impact",
    "Standings",
    "3PT Leaders",
    "Scoring Leaders",
    "3PT Defense",
    "Points Allowed",
    "Defense Mix",
    "Prediction Log",
    "Defense Styles",
    "Injury Admin",
    "Admin Panel",
]
tabs = st.tabs(tab_titles)
(
    games_tab,
    matchup_spotlight_tab,
    daily_leaders_tab,
    injury_impact_tab,
    standings_tab,
    three_pt_tab,
    scoring_tab,
    defense_3pt_tab,
    defense_pts_tab,
    defense_mix_tab,
    predictions_tab,
    defense_styles_tab,
    injury_admin_tab,
    admin_tab,
) = tabs
defense_style_tab = st.tabs(["Defense Styles"])[0]

matchup_spotlight_rows: list[Dict[str, Any]] = []
daily_power_rows_points: list[Dict[str, Any]] = []
daily_power_rows_3pm: list[Dict[str, Any]] = []
daily_top_scorers_rows: list[Dict[str, Any]] = []
player_season_stats_map: Dict[int, Mapping[str, Any]] = {}

# Today's games tab --------------------------------------------------------
with games_tab:
    # Get database connection for logging predictions
    games_conn = get_connection(str(db_path))
    # Ensure predictions table exists
    pt.create_predictions_table(games_conn)

    # Track prediction logging for debugging
    predictions_logged = 0
    predictions_failed = 0

    # Collect predictions for CSV export
    predictions_for_export = []

    st.subheader("Today's Games (Scoreboard)")
    selected_date = st.date_input(
        "Game date",
        value=default_game_date(),
        key="scoreboard_date",
    )
    context_season = st.text_input(
        "Standings season for context",
        value=builder_config["season"],
        key="scoreboard_context_season",
    ).strip() or builder_config["season"]
    context_season_type = st.selectbox(
        "Season type for context",
        options=[DEFAULT_SEASON_TYPE, "Playoffs", "Pre Season"],
        index=0,
        key="scoreboard_context_season_type",
    )
    with st.expander("Matchup leader settings", expanded=False):
        min_games_input = st.number_input(
            "Minimum games played",
            min_value=1,
            max_value=82,
            value=DEFAULT_MIN_GAMES,
            step=1,
            key="leader_min_games",
        )
        weight_cols_top = st.columns(3)
        weight_cols_bottom = st.columns(4)
        weight_inputs = {
            "avg": weight_cols_top[0].number_input(
                "Weight: Avg PPG",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULT_WEIGHTS["avg"],
                step=0.1,
                key="weight_avg_ppg",
            ),
            "median": weight_cols_top[1].number_input(
                "Weight: Median PPG",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULT_WEIGHTS["median"],
                step=0.1,
                key="weight_median_ppg",
            ),
            "max": weight_cols_top[2].number_input(
                "Weight: Max PPG",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULT_WEIGHTS["max"],
                step=0.1,
                key="weight_max_ppg",
            ),
            "last3": weight_cols_bottom[0].number_input(
                "Weight: Last 3 Avg Pts",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULT_WEIGHTS["last3"],
                step=0.1,
                key="weight_last3_pts",
            ),
            "last5": weight_cols_bottom[1].number_input(
                "Weight: Last 5 Avg Pts",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULT_WEIGHTS["last5"],
                step=0.1,
                key="weight_last5_pts",
            ),
            "minutes5": weight_cols_bottom[2].number_input(
                "Weight: Last 5 Avg Minutes",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULT_WEIGHTS["minutes5"],
                step=0.1,
                key="weight_last5_minutes",
            ),
            "usage5": weight_cols_bottom[3].number_input(
                "Weight: Last 5 Usage %",
                min_value=0.0,
                max_value=5.0,
                value=DEFAULT_WEIGHTS["usage5"],
                step=0.1,
                key="weight_last5_usage",
            ),
        }
        normalized_weights = normalize_weight_map(weight_inputs)
        weights_display = ", ".join(
            [
                f"Avg: {normalized_weights.get('avg', 0.0):.2f}",
                f"Median: {normalized_weights.get('median', 0.0):.2f}",
                f"Max: {normalized_weights.get('max', 0.0):.2f}",
                f"Last3: {normalized_weights.get('last3', 0.0):.2f}",
                f"Last5: {normalized_weights.get('last5', 0.0):.2f}",
                f"Min(L5): {normalized_weights.get('minutes5', 0.0):.2f}",
                f"Usage(L5): {normalized_weights.get('usage5', 0.0):.2f}",
            ]
        )
        st.caption(f"Normalized weights -> {weights_display}")
    try:
        matchup_spotlight_rows.clear()
        daily_power_rows_points.clear()
        daily_power_rows_3pm.clear()
        games_df, scoring_map = build_games_table(
            str(db_path),
            selected_date,
            context_season,
            context_season_type,
        )
        if games_df.empty:
            st.info("No NBA games scheduled for this date per the official scoreboard.")
        else:
            display_df = games_df.drop(columns=MATCHUP_INTERNAL_COLUMNS, errors="ignore")
            render_dataframe(display_df)
            st.caption(
                "Data source: nba_api ScoreboardV2 + standings context from the local SQLite database."
            )
            base_stats = aggregate_player_scoring(
                str(db_path),
                context_season,
                context_season_type,
            )
            leaders_df = prepare_weighted_scores(
                base_stats,
                int(min_games_input),
                normalized_weights,
            )
            player_season_stats_map.clear()
            for _, row in leaders_df.iterrows():
                player_id = safe_int(row.get("player_id"))
                if player_id is None:
                    continue
                player_season_stats_map[player_id] = row.to_dict()
            defense_stats = load_team_defense_stats(
                str(db_path),
                context_season,
                context_season_type,
                rolling_window=None,
            )
            defense_map: Dict[int, Mapping[str, Any]] = {
                int(row["team_id"]): row.to_dict() for _, row in defense_stats.iterrows()
            }
            def_style_map: Dict[int, str] = {
                int(row["team_id"]): (str(row.get("defense_style") or "Neutral"))
                for _, row in defense_stats.iterrows()
            }
            player_style_splits = build_player_style_splits(
                str(db_path),
                context_season,
                context_season_type,
                def_style_map,
            )
            player_vs_team_history = build_player_vs_team_history(
                str(db_path),
                context_season,
                context_season_type,
            )
            def_score_series = defense_stats["def_composite_score"].dropna() if not defense_stats.empty else pd.Series(dtype=float)
            if not def_score_series.empty:
                low_thresh = def_score_series.quantile(0.33)
                mid_thresh = def_score_series.quantile(0.66)
            else:
                low_thresh = mid_thresh = None

            def classify_difficulty(score: float | None) -> str:
                if score is None or pd.isna(score):
                    return "Unknown"
                if low_thresh is None or mid_thresh is None:
                    return "Neutral"
                if score <= low_thresh:
                    return "Hard"
                if score <= mid_thresh:
                    return "Neutral"
                return "Favorable"

            st.subheader("Top scorers per matchup")
            if leaders_df.empty:
                st.info(
                    "No qualifying player scoring data yet for the selected season/type. "
                    "Rebuild the database or adjust the minimum games threshold."
                )
            else:
                score_column = (
                    "composite_score" if "composite_score" in leaders_df.columns else "weighted_score"
                )

                # Get list of injured player IDs for today to exclude from predictions
                active_injuries = ia.get_active_injuries(games_conn, check_return_dates=True)
                injured_player_ids = {inj['player_id'] for inj in active_injuries}

                for _, matchup in games_df.iterrows():
                    away_id = matchup.get("away_team_id")
                    home_id = matchup.get("home_team_id")
                    if pd.isna(away_id) or pd.isna(home_id):
                        continue
                    away_id = int(away_id)
                    home_id = int(home_id)
                    matchup_rows: list[Dict[str, Any]] = []
                    for team_label, team_id, team_name in [
                        ("Away", away_id, matchup["Away"]),
                        ("Home", home_id, matchup["Home"]),
                    ]:
                        team_leaders = leaders_df[leaders_df["team_id"] == team_id].nlargest(
                            TOP_LEADERS_COUNT, score_column
                        )

                        # Filter out injured players from predictions
                        if injured_player_ids:
                            team_leaders = team_leaders[~team_leaders['player_id'].isin(injured_player_ids)]

                        if team_leaders.empty:
                            continue
                        opponent_id = home_id if team_label == "Away" else away_id
                        opponent_name = matchup["Home"] if team_label == "Away" else matchup["Away"]
                        opponent_stats = (
                            defense_map.get(int(opponent_id))
                            if opponent_id is not None
                            else None
                        )
                        opp_avg_allowed = safe_float(opponent_stats.get("avg_allowed_pts")) if opponent_stats else None
                        opp_recent_allowed = safe_float(
                            opponent_stats.get("avg_allowed_pts_last5")
                        ) if opponent_stats else None
                        team_stats = scoring_map.get(team_id)
                        team_avg_pts = safe_float(team_stats.get("avg_pts")) if team_stats else None
                        opp_composite = safe_float(
                            opponent_stats.get("def_composite_score")
                        ) if opponent_stats else None
                        opp_style = (
                            def_style_map.get(int(opponent_id), "Neutral")
                            if opponent_id is not None
                            else "Neutral"
                        )
                        opp_difficulty = classify_difficulty(opp_composite)
                        for _, player in team_leaders.iterrows():
                            avg_pts_last5 = safe_float(player.get("avg_pts_last5")) or safe_float(
                                player.get("avg_points")
                            )
                            avg_pts_last3 = safe_float(player.get("avg_pts_last3")) or avg_pts_last5
                            avg_fg3_last3 = safe_float(player.get("avg_fg3m_last3"))
                            avg_fg3_last5 = safe_float(player.get("avg_fg3m_last5"))
                            avg_minutes_last5 = safe_float(player.get("avg_minutes_last5"))
                            avg_usg_last5 = safe_float(player.get("avg_usg_last5"))
                            composite_score = safe_float(player.get(score_column)) or 0.0
                            season_avg_pts = safe_float(player.get("avg_points")) or 0.0
                            season_avg_minutes = safe_float(player.get("avg_minutes"))
                            usage_pct = (
                                avg_usg_last5
                                if avg_usg_last5 is not None
                                else safe_float(player.get("usg_pct"))
                            )
                            usage_pct_display = (
                                usage_pct * 100.0 if usage_pct is not None and usage_pct <= 1.0 else usage_pct
                            )
                            defense_factor = (
                                opp_recent_allowed
                                if opp_recent_allowed is not None
                                else opp_composite
                                if opp_composite is not None
                                else opp_avg_allowed
                                if opp_avg_allowed is not None
                                else 0.0
                            )
                            if team_avg_pts and team_avg_pts > 0 and defense_factor is not None:
                                opportunity_index = defense_factor * (season_avg_pts / team_avg_pts)
                            else:
                                opportunity_index = defense_factor
                            matchup_score = (
                                season_avg_pts * 0.4
                                + (avg_pts_last5 or season_avg_pts) * 0.3
                                + composite_score * 10.0 * 0.2
                                + defense_factor * 0.1
                            )
                            pts_delta = (
                                (avg_pts_last5 - season_avg_pts)
                                if avg_pts_last5 is not None and season_avg_pts is not None
                                else None
                            )
                            min_delta = (
                                (avg_minutes_last5 - season_avg_minutes)
                                if avg_minutes_last5 is not None and season_avg_minutes is not None
                                else None
                            )
                            usage_delta = (
                                (avg_usg_last5 - usage_pct)
                                if avg_usg_last5 is not None and usage_pct is not None
                                else None
                            )
                            # Get player history vs this opponent
                            player_id_val = safe_int(player.get("player_id"))
                            vs_opp_history = None
                            avg_vs_opp = None
                            games_vs_opp = 0
                            std_vs_opp = None

                            if player_id_val is not None and opponent_id is not None:
                                vs_opp_history = player_vs_team_history.get(player_id_val, {}).get(int(opponent_id))
                                if vs_opp_history:
                                    avg_vs_opp = vs_opp_history["avg_pts"]
                                    games_vs_opp = vs_opp_history["games"]
                                    std_vs_opp = vs_opp_history["std_pts"]

                            # Get player history vs this defense style
                            avg_vs_style = None
                            games_vs_style = 0
                            if player_id_val is not None:
                                style_data = player_style_splits.get(player_id_val, {})
                                avg_vs_style = style_data.get(opp_style)
                                # Estimate games from all teams with this style
                                if avg_vs_style is not None:
                                    games_vs_style = 5  # Conservative estimate

                            # Evaluate matchup quality
                            matchup_rating, matchup_warning, matchup_confidence = evaluate_matchup_quality(
                                season_avg_pts,
                                avg_vs_opp,
                                avg_vs_style,
                                games_vs_opp,
                                games_vs_style,
                                std_vs_opp,
                            )

                            # Get opponent correlation/matchup advantage indicator
                            # NEW: Also store the correlation object for enhanced projection
                            matchup_indicator = ""
                            matchup_advantage_text = ""
                            opponent_correlation_obj = None
                            if player_id_val is not None:
                                try:
                                    # Calculate opponent correlations for this player
                                    opponent_corrs = pca.calculate_opponent_correlations(
                                        sqlite3.connect(str(db_path)),
                                        player_id_val,
                                        context_season,
                                        context_season_type,
                                        min_games_vs=1
                                    )

                                    # Find this specific opponent in the correlations
                                    if opponent_corrs and opponent_id is not None:
                                        for opp_corr in opponent_corrs:
                                            if opp_corr.opponent_team_id == int(opponent_id):
                                                # Found the matchup!
                                                opponent_correlation_obj = opp_corr  # Store for projection
                                                score = opp_corr.matchup_score
                                                delta = opp_corr.pts_delta

                                                if score >= 60:
                                                    matchup_indicator = "✅✅"  # Excellent matchup
                                                    matchup_advantage_text = f"Elite matchup ({delta:+.1f} PPG)"
                                                elif score >= 55:
                                                    matchup_indicator = "✅"  # Good matchup
                                                    matchup_advantage_text = f"Favorable ({delta:+.1f} PPG)"
                                                elif score <= 40:
                                                    matchup_indicator = "❌❌"  # Very poor matchup
                                                    matchup_advantage_text = f"Struggles ({delta:+.1f} PPG)"
                                                elif score <= 45:
                                                    matchup_indicator = "❌"  # Poor matchup
                                                    matchup_advantage_text = f"Tough ({delta:+.1f} PPG)"
                                                else:
                                                    matchup_indicator = "➖"  # Neutral
                                                    matchup_advantage_text = f"Neutral ({delta:+.1f} PPG)"
                                                break
                                except Exception:
                                    # Silently fail - don't break the display if correlation calc fails
                                    pass

                            # NEW: Calculate pace and defense quality splits for enhanced projection
                            pace_split_obj = None
                            defense_quality_split_obj = None
                            if player_id_val is not None and opponent_id is not None:
                                try:
                                    conn_for_splits = sqlite3.connect(str(db_path))

                                    # Get all defense type splits for this player
                                    defense_splits = dta.calculate_defense_type_splits(
                                        conn_for_splits,
                                        player_id_val,
                                        context_season,
                                        context_season_type,
                                        min_games=2
                                    )

                                    # Also need to categorize the opponent team to know which split to use
                                    team_categories = dta.categorize_teams_by_defense(
                                        conn_for_splits,
                                        context_season,
                                        context_season_type
                                    )

                                    if int(opponent_id) in team_categories:
                                        opp_pace_cat = team_categories[int(opponent_id)]['pace_category']
                                        opp_def_cat = team_categories[int(opponent_id)]['defense_category']

                                        # Find matching splits
                                        for split in defense_splits:
                                            if split.defense_type == f"{opp_pace_cat} Pace":
                                                pace_split_obj = split
                                            elif split.defense_type == f"{opp_def_cat} Defense":
                                                defense_quality_split_obj = split

                                    conn_for_splits.close()
                                except Exception:
                                    # Silently fail - don't break the display
                                    pass

                            # Display string for vs opponent
                            if avg_vs_opp is not None and games_vs_opp >= 2:
                                vs_opp_display = f"{avg_vs_opp:.1f} ({games_vs_opp}G)"
                            else:
                                vs_opp_display = "N/A"

                            # Display string for vs style
                            avg_vs_style_display = (
                                format_number(avg_vs_style, 1) if avg_vs_style is not None else "N/A"
                            )

                            # Calculate smart PPG projection with enhanced analytics
                            opp_def_rating = safe_float(opponent_stats.get("def_rating")) if opponent_stats else None
                            opp_pace = safe_float(opponent_stats.get("avg_opp_possessions")) if opponent_stats else None

                            projection, proj_confidence, proj_floor, proj_ceiling, breakdown, analytics_indicators = calculate_smart_ppg_projection(
                                season_avg=season_avg_pts,
                                recent_avg_5=avg_pts_last5,
                                recent_avg_3=avg_pts_last3,
                                vs_opp_team_avg=avg_vs_opp,
                                vs_opp_team_games=games_vs_opp,
                                vs_defense_style_avg=avg_vs_style,
                                vs_defense_style_games=games_vs_style,
                                opp_def_rating=opp_def_rating,
                                opp_pace=opp_pace,
                                # NEW: Enhanced analytics parameters
                                opponent_correlation=opponent_correlation_obj,
                                pace_split=pace_split_obj,
                                defense_quality_split=defense_quality_split_obj,
                            )

                            # Calculate unified DFS Score
                            daily_pick_score, pick_grade, pick_explanation = calculate_daily_pick_score(
                                player_season_avg=season_avg_pts,
                                player_projection=projection,
                                projection_confidence=proj_confidence,
                                matchup_rating=matchup_rating,
                                opp_def_rating=opp_def_rating,
                            )

                            matchup_rows.append(
                                {
                                    "Side": team_label,
                                    "Team": team_name,
                                    "Player": player["player_name"],
                                    "Matchup": matchup_indicator,  # NEW: Matchup advantage indicator
                                    "Analytics": analytics_indicators,  # NEW: Analytics quality indicators
                                    "DFS Score": f"{daily_pick_score:.1f}",
                                    "Grade": f"{pick_grade}\n{pick_explanation}",
                                    "Games": int(player["games_played"]),
                                    "Projected PPG": f"{projection:.1f}",
                                    "Proj Conf": f"{proj_confidence:.0%}",
                                    "Avg PPG": f"{player['avg_points']:.1f}",
                                    "Median PPG": f"{player['median_points']:.1f}",
                                    "Max PPG": f"{player['max_points']:.1f}",
                                    "Proj Range": f"{proj_floor:.1f}-{proj_ceiling:.1f}",
                                    "Avg 3PM": f"{player['avg_fg3m']:.1f}",
                                    "Median 3PM": f"{player['median_fg3m']:.1f}",
                                    "Last3 Avg Pts": format_number(player.get("avg_pts_last3"), 1),
                                    "Last3 Avg 3PM": format_number(player.get("avg_fg3m_last3"), 1),
                                    "Last5 Avg Pts": format_number(player.get("avg_pts_last5"), 1),
                                    "Last5 Avg 3PM": format_number(player.get("avg_fg3m_last5"), 1),
                                    "Last5 Avg Minutes": format_number(avg_minutes_last5, 1),
                                    "Last5 Usage %": format_number(usage_pct_display, 1),
                                    "Usage %": format_number(usage_pct_display, 1),
                                    "Composite Score": format_number(composite_score, 2),
                                    "Comp Z Avg": format_number(player.get("z_avg"), 2),
                                    "Comp Z Last3": format_number(player.get("z_last3"), 2),
                                    "Comp Z Last5": format_number(player.get("z_last5"), 2),
                                    "Comp Z Min5": format_number(player.get("z_minutes5"), 2),
                                    "Comp Z Usg5": format_number(player.get("z_usage5"), 2),
                                    "Last5 vs Season Pts": format_number(pts_delta, 1),
                                    "Last5 vs Season Min": format_number(min_delta, 1),
                                    "Last5 vs Season Usage": format_number(usage_delta, 1),
                                    "Opp Defense Style": opp_style,
                                    "Opp Difficulty": opp_difficulty,
                                    "Vs This Team": vs_opp_display,
                                    "Vs This Style": avg_vs_style_display,
                                    "Matchup Rating": matchup_rating,
                                    "Matchup Adv": matchup_advantage_text,  # NEW: Detailed matchup text
                                    "Warning": matchup_warning,
                                    "Confidence": f"{matchup_confidence:.0%}" if matchup_confidence > 0 else "Low",
                                }
                            )

                            # NEW: Log prediction for tracking
                            try:
                                prediction = pt.Prediction(
                                    prediction_id=None,
                                    prediction_date=str(date.today()),
                                    game_date=str(selected_date),
                                    player_id=player_id_val,
                                    player_name=player["player_name"],
                                    team_id=team_id,
                                    team_name=team_name,
                                    opponent_id=opponent_id if opponent_id else 0,
                                    opponent_name=opponent_name,
                                    projected_ppg=projection,
                                    proj_confidence=proj_confidence,
                                    proj_floor=proj_floor,
                                    proj_ceiling=proj_ceiling,
                                    season_avg_ppg=season_avg_pts,
                                    recent_avg_3=avg_pts_last3,
                                    recent_avg_5=avg_pts_last5,
                                    vs_opponent_avg=avg_vs_opp,
                                    vs_opponent_games=games_vs_opp,
                                    analytics_used=analytics_indicators,
                                    opponent_def_rating=opp_def_rating,
                                    opponent_pace=opp_pace,
                                    dfs_score=daily_pick_score,
                                    dfs_grade=pick_grade
                                )

                                # Add to export list
                                predictions_for_export.append({
                                    'game_date': str(selected_date),
                                    'player_name': player["player_name"],
                                    'team_name': team_name,
                                    'opponent_name': opponent_name,
                                    'projected_ppg': projection,
                                    'proj_confidence': proj_confidence,
                                    'proj_floor': proj_floor,
                                    'proj_ceiling': proj_ceiling,
                                    'season_avg_ppg': season_avg_pts,
                                    'recent_avg_3': avg_pts_last3,
                                    'recent_avg_5': avg_pts_last5,
                                    'vs_opponent_avg': avg_vs_opp,
                                    'vs_opponent_games': games_vs_opp,
                                    'analytics_used': analytics_indicators,
                                    'opponent_def_rating': opp_def_rating,
                                    'opponent_pace': opp_pace,
                                    'dfs_score': daily_pick_score,
                                    'dfs_grade': pick_grade
                                })

                                # Try to log to database (optional, won't break if it fails)
                                pt.log_prediction(games_conn, prediction)
                                predictions_logged += 1
                            except Exception as e:
                                # Show error in sidebar to help debug without breaking main display
                                predictions_failed += 1
                                if predictions_failed <= 3:  # Only show first 3 errors to avoid spam
                                    st.sidebar.warning(f"⚠️ Prediction logging error: {str(e)[:100]}")

                            matchup_spotlight_rows.append(
                                {
                                    "Matchup": f"{matchup['Away']} at {matchup['Home']}",
                                    "Side": team_label,
                                    "Team": team_name,
                                    "Player": player["player_name"],
                                    "Matchup Ind": matchup_indicator,  # NEW: Matchup indicator
                                    "DFS Score": daily_pick_score,
                                    "Pick Grade": f"{pick_grade}: {pick_explanation}",
                                    "Season Avg PPG": season_avg_pts,
                                    "Projected PPG": projection,
                                    "Proj Floor": proj_floor,
                                    "Proj Ceiling": proj_ceiling,
                                    "Proj Confidence": proj_confidence,
                                    "Last5 Avg PPG": avg_pts_last5,
                                    "Last3 Avg PPG": avg_pts_last3,
                                    "Last5 Avg 3PM": avg_fg3_last5,
                                    "Last3 Avg 3PM": avg_fg3_last3,
                                    "Opponent": opponent_name,
                                    "Opp Avg Allowed PPG": opp_avg_allowed,
                                    "Opp Last5 Avg Allowed": opp_recent_allowed,
                                    "Opp Def Composite": opp_composite,
                                    "Opp Defense Style": opp_style,
                                    "Last5 Avg Minutes": avg_minutes_last5,
                                    "Last5 Usage %": usage_pct_display,
                                    "Usage %": usage_pct_display,
                                    "Composite Score": composite_score,
                                    "Comp Z Avg": player.get("z_avg"),
                                    "Comp Z Last3": player.get("z_last3"),
                                    "Comp Z Last5": player.get("z_last5"),
                                    "Comp Z Min5": player.get("z_minutes5"),
                                    "Comp Z Usg5": player.get("z_usage5"),
                                    "Last5 vs Season Pts": pts_delta,
                                    "Last5 vs Season Min": min_delta,
                                    "Last5 vs Season Usage": usage_delta,
                                    "Opp Difficulty": opp_difficulty,
                                    "Vs This Team": vs_opp_display,
                                    "Vs This Style": avg_vs_style_display,
                                    "Matchup Rating": matchup_rating,
                                    "Matchup Adv": matchup_advantage_text,  # NEW: Matchup advantage text
                                    "Warning": matchup_warning,
                                    "Matchup Score": matchup_score,
                                }
                            )
                            daily_power_rows_points.append(
                                {
                                    "Matchup": f"{matchup['Away']} at {matchup['Home']}",
                                    "Player": player["player_name"],
                                    "Team": team_name,
                                    "Ind": matchup_indicator,  # NEW: Matchup indicator (shorter column name)
                                    "DFS Score": daily_pick_score,
                                    "Grade": f"{pick_grade}: {pick_explanation}",
                                    "Projected PPG": projection,
                                    "Season Avg PPG": season_avg_pts,
                                    "Proj Conf": f"{proj_confidence:.0%}",
                                    "Last5 Avg PPG": avg_pts_last5,
                                    "Opp Avg Allowed PPG": opp_avg_allowed,
                                    "Opp Last5 Avg Allowed": opp_recent_allowed,
                                    "Opp Def Composite": opp_composite,
                                    "Opportunity Index": opportunity_index,
                                    "Usage %": usage_pct_display,
                                    "Opp Defense Style": opp_style,
                                    "Opp Difficulty": opp_difficulty,
                                    "Matchup Score": matchup_score,
                                }
                            )
                            daily_power_rows_3pm.append(
                                {
                                    "Matchup": f"{matchup['Away']} at {matchup['Home']}",
                                    "Player": player["player_name"],
                                    "Team": team_name,
                                    "Season Avg 3PM": safe_float(player.get("avg_fg3m")),
                                    "Last5 Avg 3PM": avg_fg3_last5,
                                    "Opp Avg Allowed PPG": opp_avg_allowed,
                                    "Opp Last5 Avg Allowed": opp_recent_allowed,
                                    "Opp Def Composite": opp_composite,
                                    "Opportunity Index": opportunity_index,
                                    "Usage %": usage_pct_display,
                                    "Opp Defense Style": opp_style,
                                    "Opp Difficulty": opp_difficulty,
                                    "Matchup Score": (
                                        (avg_fg3_last5 or safe_float(player.get("avg_fg3m")) or 0.0) * 0.6
                                        + (avg_fg3_last3 or safe_float(player.get("avg_fg3m")) or 0.0) * 0.3
                                        + (opp_recent_allowed or opp_avg_allowed or 0.0) * 0.1
                                    ),
                                }
                            )
                    st.markdown(f"**{matchup['Away']} at {matchup['Home']}**")
                    if matchup_rows:
                        matchup_df = pd.DataFrame(matchup_rows)

                        # Apply styling based on matchup rating
                        def style_matchup_rating(val):
                            if val == "Excellent":
                                return "background-color: #d4edda; color: #155724"  # Green
                            elif val == "Good":
                                return "background-color: #d1ecf1; color: #0c5460"  # Blue
                            elif val == "Difficult":
                                return "background-color: #fff3cd; color: #856404"  # Yellow
                            elif val == "Avoid":
                                return "background-color: #f8d7da; color: #721c24"  # Red
                            return ""

                        def style_warning(val):
                            if val and isinstance(val, str) and len(val) > 0:
                                return "background-color: #fff3cd; font-weight: bold"
                            return ""

                        def style_projection_confidence(val):
                            """Style projection confidence - green for high, yellow for medium, gray for low"""
                            if not val or val == "N/A":
                                return ""
                            try:
                                # Remove % sign and convert to float
                                conf_str = val.replace("%", "").strip()
                                conf_val = float(conf_str) / 100.0
                                if conf_val >= 0.75:
                                    return "background-color: #d4edda; color: #155724; font-weight: bold"  # High confidence - Green
                                elif conf_val >= 0.50:
                                    return "background-color: #fff3cd; color: #856404"  # Medium confidence - Yellow
                                else:
                                    return "background-color: #e2e3e5; color: #383d41"  # Low confidence - Gray
                            except (ValueError, AttributeError):
                                return ""

                        styled_df = matchup_df.style.applymap(
                            style_matchup_rating, subset=["Matchup Rating"]
                        ).applymap(
                            style_warning, subset=["Warning"]
                        ).applymap(
                            style_projection_confidence, subset=["Proj Conf"]
                        )

                        st.dataframe(styled_df, use_container_width=True)

                        # Show warnings prominently
                        warnings = matchup_df[matchup_df["Warning"].notna() & (matchup_df["Warning"] != "")]
                        if not warnings.empty:
                            with st.expander("⚠️ Matchup Warnings", expanded=True):
                                for _, row in warnings.iterrows():
                                    if row["Matchup Rating"] == "Avoid":
                                        st.error(f"**{row['Player']} ({row['Team']})**: {row['Warning']}")
                                    elif row["Matchup Rating"] == "Difficult":
                                        st.warning(f"**{row['Player']} ({row['Team']})**: {row['Warning']}")
                                    elif row["Matchup Rating"] == "Excellent":
                                        st.success(f"**{row['Player']} ({row['Team']})**: {row['Warning']}")
                    else:
                        st.caption("No qualified players for this matchup yet.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load today's games: {exc}")

    # Show prediction logging status
    if predictions_logged > 0 or predictions_failed > 0:
        st.sidebar.success(f"✅ Logged {predictions_logged} predictions")
        if predictions_failed > 0:
            st.sidebar.error(f"❌ Failed to log {predictions_failed} predictions")

        # Auto-apply injury adjustments if there are active injuries
        if predictions_logged > 0 and injured_player_ids:
            try:
                adjusted, skipped, records = ia.apply_injury_adjustments(
                    list(injured_player_ids),
                    str(selected_date),
                    games_conn,
                    min_historical_games=3
                )
                if adjusted > 0:
                    st.sidebar.info(f"🚑 Auto-adjusted {adjusted} predictions for {len(injured_player_ids)} injuries")
                    if skipped > 0:
                        st.sidebar.caption(f"({skipped} skipped - insufficient data)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Injury adjustment failed: {e}")

        # Auto-backup to S3 after logging predictions
        if predictions_logged > 0:
            storage = s3_storage.S3PredictionStorage()
            if storage.is_connected():
                success, message = storage.upload_database(db_path)
                if success:
                    st.sidebar.info(f"☁️ {message}")
                else:
                    st.sidebar.warning(f"⚠️ S3 backup failed: {message}")

    # CSV Export Button
    if predictions_for_export:
        st.divider()
        st.subheader("📥 Export Today's Predictions")

        # Convert to DataFrame
        export_df = pd.DataFrame(predictions_for_export)

        # Sort by DFS score descending
        export_df = export_df.sort_values('dfs_score', ascending=False).reset_index(drop=True)

        # Show summary
        st.caption(f"Total predictions: {len(export_df)} | Top DFS Score: {export_df['dfs_score'].max():.1f} | Date: {export_df['game_date'].iloc[0]}")

        # Convert to CSV
        csv = export_df.to_csv(index=False)

        # Download button
        col1, col2 = st.columns([2, 1])

        with col1:
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"nba_predictions_{export_df['game_date'].iloc[0]}.csv",
                mime="text/csv",
                help="Download all predictions from this page as CSV"
            )

        with col2:
            # Manual S3 backup button
            storage = s3_storage.S3PredictionStorage()
            if storage.is_connected():
                if st.button("☁️ Backup to S3", help="Manually backup database to S3"):
                    with st.spinner("Uploading to S3..."):
                        success, message = storage.upload_database(db_path)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")

                # Show S3 backup status
                backup_info = storage.get_backup_info()
                if backup_info.get('exists'):
                    last_backup = backup_info['last_modified'].strftime("%Y-%m-%d %H:%M:%S UTC")
                    st.caption(f"Last S3 backup: {last_backup}")
            else:
                st.caption("S3 backup not configured")

# Matchup spotlight tab ----------------------------------------------------
with matchup_spotlight_tab:
    st.subheader("Player Matchup Spotlight")
    if not matchup_spotlight_rows:
        st.info("Run the Today's Games tab to populate matchup insights.")
    else:
        spotlight_df = pd.DataFrame(matchup_spotlight_rows)
        sort_column = st.selectbox(
            "Sort by",
            options=[
                "Matchup Score",
                "Season Avg PPG",
                "Last5 Avg PPG",
                "Opp Avg Allowed PPG",
                "Composite Score",
            ],
            index=0,
        )
        ascending = st.checkbox("Sort ascending (default descending)", value=False)
        top_n = st.slider("Rows to display", min_value=10, max_value=200, value=50, step=10)
        display_df = (
            spotlight_df.sort_values(sort_column, ascending=ascending)
            .head(top_n)
            .reset_index(drop=True)
        )
        if "Composite Score" in display_df.columns:
            display_df["Composite Score"] = display_df["Composite Score"].map(lambda v: format_number(v, 2))
        if "Opp Difficulty" in display_df.columns:
            display_df["Opp Difficulty"] = display_df["Opp Difficulty"].fillna("Unknown")
        if "Usage %" in display_df.columns:
            display_df["Usage %"] = display_df["Usage %"].map(lambda v: format_number(v, 1))
        if "Last5 Usage %" in display_df.columns:
            display_df["Last5 Usage %"] = display_df["Last5 Usage %"].map(lambda v: format_number(v, 1))
        if "Last5 Avg Minutes" in display_df.columns:
            display_df["Last5 Avg Minutes"] = display_df["Last5 Avg Minutes"].map(lambda v: format_number(v, 1))
        if "Opp Def Composite" in display_df.columns:
            display_df["Opp Def Composite"] = display_df["Opp Def Composite"].map(
                lambda v: format_number(v, 1)
            )
        st.dataframe(display_df, use_container_width=True)
        st.subheader("Daily Power Rankings")
        col_points, col_3pm = st.columns(2)
        if daily_power_rows_points:
            points_df = pd.DataFrame(daily_power_rows_points)
            points_top = points_df.sort_values("DFS Score", ascending=False).head(10)
            points_top = points_top.assign(
                **{
                    "DFS Score": points_top["DFS Score"].map(lambda v: format_number(v, 1)),
                    "Projected PPG": points_top["Projected PPG"].map(lambda v: format_number(v, 1)),
                    "Opp Avg Allowed PPG": points_top["Opp Avg Allowed PPG"].map(lambda v: format_number(v, 1)),
                    "Opp Last5 Avg Allowed": points_top["Opp Last5 Avg Allowed"].map(lambda v: format_number(v, 1)),
                    "Opportunity Index": points_top["Opportunity Index"].map(lambda v: format_number(v, 2)),
                    "Opp Def Composite": points_top["Opp Def Composite"].map(lambda v: format_number(v, 1)),
                    "Usage %": points_top["Usage %"].map(lambda v: format_number(v, 1)),
                }
            )

            # Apply confidence styling
            def style_proj_conf(val):
                if not val or val == "N/A":
                    return ""
                try:
                    conf_val = float(val.replace("%", "").strip()) / 100.0
                    if conf_val >= 0.70:
                        return "background-color: #d4edda; color: #155724; font-weight: bold"
                    elif conf_val >= 0.50:
                        return "background-color: #fff3cd; color: #856404"
                    else:
                        return "background-color: #e2e3e5; color: #383d41"
                except (ValueError, AttributeError):
                    return ""

            styled_points = points_top.style.applymap(style_proj_conf, subset=["Proj Conf"])
            col_points.markdown("**🎯 Top 10 Daily Picks (by DFS Score)**")
            col_points.dataframe(styled_points, use_container_width=True)
        else:
            col_points.info("No scoring data yet. Refresh Today's Games.")
        if daily_power_rows_3pm:
            threes_df = pd.DataFrame(daily_power_rows_3pm)
            threes_top = threes_df.sort_values("Matchup Score", ascending=False).head(10)
            threes_top = threes_top.assign(
                **{
                    "Opp Avg Allowed PPG": threes_top["Opp Avg Allowed PPG"].map(lambda v: format_number(v, 1)),
                    "Opp Last5 Avg Allowed": threes_top["Opp Last5 Avg Allowed"].map(lambda v: format_number(v, 1)),
                    "Opportunity Index": threes_top["Opportunity Index"].map(lambda v: format_number(v, 2)),
                    "Opp Def Composite": threes_top["Opp Def Composite"].map(lambda v: format_number(v, 1)),
                    "Usage %": threes_top["Usage %"].map(lambda v: format_number(v, 1)),
                }
            )
            col_3pm.markdown("**Top 10 Players by Matchup Score (3PM)**")
            col_3pm.dataframe(threes_top, use_container_width=True)
        else:
            col_3pm.info("No three-point data yet. Refresh Today's Games.")

        top_scorers_query = """
            WITH ranked AS (
                SELECT
                    player_id,
                    player_name,
                    team_name,
                    game_date,
                    matchup,
                    points,
                    minutes,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_date
                        ORDER BY points DESC
                    ) AS rn
                FROM player_game_logs
                WHERE season = ?
                  AND season_type = ?
                  AND points IS NOT NULL
            )
            SELECT
                r.game_date,
                r.player_id,
                r.player_name,
                r.team_name,
                r.matchup,
                r.points,
                r.minutes,
                pst.usg_pct
            FROM ranked AS r
            LEFT JOIN player_season_totals AS pst
              ON pst.player_id = r.player_id
             AND pst.season = ?
             AND pst.season_type = ?
            WHERE r.rn <= ?
            ORDER BY r.game_date DESC, r.points DESC
        """
        try:
            top_df = run_query(
                str(db_path),
                top_scorers_query,
                params=(
                    context_season,
                    context_season_type,
                    context_season,
                    context_season_type,
                    DAILY_LEADERS_MAX,
                ),
            )
            daily_top_scorers_rows.clear()
            for _, row in top_df.iterrows():
                game_date = pd.to_datetime(row["game_date"]).date()
                home_team, away_team = derive_home_away(row["team_name"], row["matchup"])
                minutes_float = minutes_str_to_float(row["minutes"])
                usage_pct = safe_float(row["usg_pct"])
                player_id = safe_int(row.get("player_id"))
                season_stats = player_season_stats_map.get(player_id) if player_id is not None else {}
                season_avg_pts = safe_float((season_stats or {}).get("avg_points"))
                season_median_pts = safe_float((season_stats or {}).get("median_points"))
                daily_top_scorers_rows.append(
                    {
                        "Date": game_date.isoformat(),
                        "Player": row["player_name"],
                        "Home Team": home_team,
                        "Away Team": away_team,
                        "Total Points": row["points"],
                        "Minutes": minutes_float,
                        "Usage %": (usage_pct * 100.0) if usage_pct is not None else None,
                        "Season Avg Pts": season_avg_pts,
                        "Season Median Pts": season_median_pts,
                    }
                )
        except Exception as exc:
            st.warning(f"Unable to load daily top scorers: {exc}")

# Daily leaders tab --------------------------------------------------------
with daily_leaders_tab:
    st.subheader("Daily Top Scorers (Top 3 per day)")
    if not daily_top_scorers_rows:
        st.info("Run the Today's Games tab to populate the leaders table.")
    else:
        leaders_df = pd.DataFrame(daily_top_scorers_rows)
        leaders_df["Date"] = pd.to_datetime(leaders_df["Date"])
        available_dates = sorted(leaders_df["Date"].dt.date.unique(), reverse=True)
        selected_date = st.selectbox(
            "Select date",
            options=available_dates,
            index=0,
        )
        max_players = st.slider(
            "Players shown per day",
            min_value=1,
            max_value=DAILY_LEADERS_MAX,
            value=5,
            step=1,
        )
        filtered_df = leaders_df[leaders_df["Date"].dt.date == selected_date].copy()
        filtered_df = filtered_df.sort_values("Total Points", ascending=False).head(max_players)
        filtered_df["Minutes"] = filtered_df["Minutes"].map(lambda v: format_number(v, 1))
        filtered_df["Usage %"] = filtered_df["Usage %"].map(lambda v: format_number(v, 1))
        filtered_df["Season Avg Pts"] = filtered_df["Season Avg Pts"].map(lambda v: format_number(v, 1))
        filtered_df["Season Median Pts"] = filtered_df["Season Median Pts"].map(lambda v: format_number(v, 1))
        display_df = filtered_df[
            [
                "Date",
                "Player",
                "Home Team",
                "Away Team",
                "Total Points",
                "Minutes",
                "Usage %",
                "Season Avg Pts",
                "Season Median Pts",
            ]
        ].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# Standings tab -------------------------------------------------------------
with standings_tab:
    st.subheader("League Standings")
    seasons = fetch_distinct_values(str(db_path), "standings", "season")
    season = st.selectbox("Season", options=seasons, index=0 if seasons else None)
    season_types = fetch_distinct_values(str(db_path), "standings", "season_type")
    season_type = st.selectbox("Season Type", options=season_types, index=0 if season_types else None)
    if season and season_type:
        standings_query = """
            SELECT t.full_name AS team,
                   s.wins,
                   s.losses,
                   s.win_pct,
                   s.conference_rank,
                   s.division_rank,
                   s.streak
            FROM standings AS s
            JOIN teams AS t ON t.team_id = s.team_id
            WHERE s.season = ?
              AND s.season_type = ?
            ORDER BY s.win_pct DESC
        """
        standings = run_query(str(db_path), standings_query, params=(season, season_type))
        standings["win_pct"] = standings["win_pct"].map(lambda x: f"{x:.3f}")
        render_dataframe(standings)
    else:
        st.info("Standings data not available in the database.")

# 3PT Leaders tab ----------------------------------------------------------
with three_pt_tab:
    st.subheader("Player 3PT Leaders View")
    row_limit = st.slider("Rows to display", min_value=10, max_value=100, value=25, step=5)
    try:
        leaders_query = f"""
            SELECT player_name,
                   team_abbreviation AS team,
                   total_fg3m,
                   avg_fg3m_per_game,
                   avg_fg3a_per_game,
                   median_fg3m_per_game,
                   max_fg3m_per_game,
                   rank_fg3m_per_game
            FROM players_2025_top_3pt
            ORDER BY rank_fg3m_per_game ASC
            LIMIT {row_limit}
        """
        leaders_df = run_query(str(db_path), leaders_query)
        render_dataframe(leaders_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Three-point leaderboard view not available: {exc}")

# Scoring leaders tab ------------------------------------------------------
with scoring_tab:
    st.subheader("Player Scoring Leaders View")
    row_limit_pts = st.slider("Rows to display ", min_value=10, max_value=100, value=25, step=5, key="pts_slider")
    try:
        pts_query = f"""
            SELECT player_name,
                   team_abbreviation AS team,
                   total_points,
                   avg_points_per_game,
                   median_points_per_game,
                   max_points_per_game,
                   rank_points_per_game
            FROM players_2025_top_pts
            ORDER BY rank_points_per_game ASC
            LIMIT {row_limit_pts}
        """
        pts_df = run_query(str(db_path), pts_query)
        render_dataframe(pts_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Scoring leaderboard view not available: {exc}")

# 3PT Defense tab ----------------------------------------------------------
with defense_3pt_tab:
    st.subheader("Team 3PT Defense View")
    try:
        defense_query = """
            SELECT team_name,
                   avg_allowed_fg3m,
                   avg_allowed_fg3a,
                   median_allowed_fg3m,
                   max_allowed_fg3m,
                   rank_avg_allowed_fg3m
            FROM teams_2025_defense_3pt
            ORDER BY rank_avg_allowed_fg3m ASC
        """
        defense_df = run_query(str(db_path), defense_query)
        render_dataframe(defense_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"3PT defensive view not available: {exc}")

# Points allowed tab -------------------------------------------------------
with defense_pts_tab:
    st.subheader("Team Points Allowed View")
    try:
        pts_def_query = """
            SELECT team_name,
                   avg_allowed_pts,
                   median_allowed_pts,
                   max_allowed_pts,
                   rank_avg_allowed_pts
            FROM teams_2025_defense_pts
            ORDER BY rank_avg_allowed_pts ASC
        """
        pts_def_df = run_query(str(db_path), pts_def_query)
        render_dataframe(pts_def_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Points-allowed view not available: {exc}")

# Defense mix tab ----------------------------------------------------------
with defense_mix_tab:
    st.subheader("Team Defense Mix (Points vs 3PM)")
    try:
        mix_query = """
            SELECT team_name,
                   total_allowed_pts,
                   avg_allowed_pts,
                   total_allowed_fg3m,
                   median_allowed_pts,
                   median_allowed_fg3m,
                   total_allowed_ast,
                   avg_allowed_ast,
                   total_allowed_reb,
                   avg_allowed_reb,
                   pct_points_from_3_total,
                   pct_points_from_3_median
            FROM teams_2026_defense_mix
            ORDER BY pct_points_from_3_total DESC
        """
        try:
            mix_df = run_query(str(db_path), mix_query)
        except Exception as inner_exc:
            if "no such column" in str(inner_exc).lower():
                fallback_query = """
                    SELECT team_name,
                           total_allowed_pts,
                           median_allowed_pts,
                           total_allowed_fg3m,
                           median_allowed_fg3m,
                           pct_points_from_3_total,
                           pct_points_from_3_median
                    FROM teams_2026_defense_mix
                    ORDER BY pct_points_from_3_total DESC
                """
                mix_df = run_query(str(db_path), fallback_query)
                st.caption(
                    "Defense mix view missing new assist/rebound columns. "
                    "Rebuild the database to populate them."
                )
            else:
                raise
        render_dataframe(mix_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Defense mix view not available: {exc}")

# Defense styles tab -------------------------------------------------------
with defense_styles_tab:
    st.subheader("Team Defense Styles")
    try:
        styles_df = load_team_defense_stats(
            str(db_path),
            context_season,
            context_season_type,
            rolling_window=None,
        )
        if styles_df.empty:
            st.info("No defense data available. Rebuild the database for the selected season/type.")
        else:
            team_lookup_df = run_query(
                str(db_path),
                "SELECT team_id, full_name FROM teams",
            )
            team_lookup_df["team_id"] = pd.to_numeric(team_lookup_df["team_id"], errors="coerce")
            styles_df = styles_df.merge(team_lookup_df, on="team_id", how="left")
            display_cols = [
                "full_name",
                "style_tags",
                "def_rating",
                "avg_allowed_pts",
                "def_3pm_per100",
                "def_2pt_pts_per100",
                "avg_allowed_reb",
                "def_composite_score",
            ]
            rename_map = {
                "full_name": "Team",
                "style_tags": "Defense Styles",
                "def_rating": "Def Rating (per 100)",
                "avg_allowed_pts": "Avg Pts Allowed",
                "def_3pm_per100": "3PM Allowed (per 100)",
                "def_2pt_pts_per100": "2PT Pts Allowed (per 100)",
                "avg_allowed_reb": "Avg Reb Allowed",
                "def_composite_score": "Def Composite",
            }
            display_df = styles_df[display_cols].rename(columns=rename_map)
            # Format numeric columns
            for col in ["Def Rating (per 100)", "Avg Pts Allowed", "3PM Allowed (per 100)",
                       "2PT Pts Allowed (per 100)", "Avg Reb Allowed", "Def Composite"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda v: format_number(v, 1))
            st.dataframe(
                display_df.sort_values("Defense Styles"),
                use_container_width=True,
            )
            grouped = (
                styles_df.groupby("defense_style")["avg_allowed_pts"]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "defense_style": "Defense Style",
                        "avg_allowed_pts": "Avg Points Allowed",
                    }
                )
            )
            st.markdown("**Average points allowed by style**")
            st.dataframe(grouped, use_container_width=True, hide_index=True)
            st.markdown("**Top players vs each defense style (by avg points)**")
            style_leaders = build_player_style_leaders(
                str(db_path),
                context_season,
                context_season_type,
                def_style_map,
                min_games=1,
                top_n=15,
            )
            if style_leaders.empty:
                st.info(
                    "No player vs style splits yet for the current styles. "
                    "Try another season/type or wait for more non-neutral style samples."
                )
            else:
                style_leaders = style_leaders.rename(
                    columns={
                        "defense_style": "Defense Style",
                        "player_name": "Player",
                        "full_name": "Team",
                        "games_vs_style": "Games vs Style",
                        "avg_pts_vs_style": "Avg Pts vs Style",
                    }
                )
                style_leaders["Avg Pts vs Style"] = style_leaders["Avg Pts vs Style"].map(
                    lambda v: format_number(v, 1)
                )
                st.dataframe(style_leaders, use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load defense styles: {exc}")

# Player Impact tab --------------------------------------------------------
with injury_impact_tab:
    st.subheader("Player Impact Analysis")
    st.markdown(
        "Comprehensive player performance analysis across multiple dimensions: "
        "Home/Away splits, defense matchups, teammate correlations, and absence impact."
    )

    # PLAYER SELECTION AT TOP (moved from below)
    # Get significant players for selection
    try:
        conn_impact = get_connection(str(db_path))

        # Season selection in one row
        impact_col1, impact_col2 = st.columns(2)
        with impact_col1:
            impact_season = st.text_input(
                "Season for analysis",
                value=builder_config["season"],
                key="impact_season",
            ).strip() or builder_config["season"]
        with impact_col2:
            impact_season_type = st.selectbox(
                "Season type",
                options=[DEFAULT_SEASON_TYPE, "Playoffs", "Pre Season"],
                index=0,
                key="impact_season_type",
            )

        # Filters in expander to keep UI clean
        with st.expander("🔍 Player Filters", expanded=False):
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                min_games_impact = st.slider(
                    "Minimum games played",
                    min_value=5,
                    max_value=30,
                    value=10,
                    key="min_games_impact",
                )
            with filter_col2:
                min_ppg_impact = st.slider(
                    "Minimum points per game",
                    min_value=5.0,
                    max_value=20.0,
                    value=10.0,
                    step=1.0,
                    key="min_ppg_impact",
                )

        significant_players = iia.get_significant_players(
            conn_impact,
            season=impact_season,
            season_type=impact_season_type,
            min_games=min_games_impact,
            min_avg_points=min_ppg_impact,
        )

        if not significant_players.empty:
            player_options = {
                f"{row['player_name']} ({row['team_name']}) - {row['avg_points']} PPG": row['player_id']
                for _, row in significant_players.iterrows()
            }

            # PLAYER SELECTOR - Prominent at top
            selected_player_display = st.selectbox(
                "👤 Select Player to Analyze",
                options=list(player_options.keys()),
                key="impact_player_select",
                help="Choose a player to see their performance across different scenarios"
            )

            selected_player_id = player_options[selected_player_display]
            selected_player_name = selected_player_display.split(" (")[0]

    except Exception as exc:
        st.error(f"Error loading player list: {exc}")
        selected_player_id = None
        selected_player_name = None

    st.divider()

    # Create sub-tabs for different impact analyses (moved below player selection)
    impact_subtabs = st.tabs([
        "🏠 Home/Away Splits",
        "🛡️ vs Defense Types",
        "👥 Player Correlations",
        "🚑 Absence Impact"
    ])

    home_away_tab, defense_types_tab, correlations_tab, absence_tab = impact_subtabs

    # HOME/AWAY SPLITS TAB
    with home_away_tab:
        st.markdown("### Home vs Away Performance")
        if selected_player_id:
            st.info("Home/Away split analysis coming soon! This will show how the player performs at home vs. on the road.")
        else:
            st.warning("Select a player above to view Home/Away splits.")

    # VS DEFENSE TYPES TAB
    with defense_types_tab:
        st.markdown("### Performance vs Different Defense Types")
        st.markdown(
            "Analyze how this player performs against different defensive styles: "
            "fast vs slow pace teams, and elite vs weak defenses."
        )

        if not selected_player_id:
            st.warning("Select a player above to view defense type analysis.")
        else:
            try:
                with st.spinner(f"Analyzing {selected_player_name}'s performance vs defense types..."):
                    defense_splits = dta.calculate_defense_type_splits(
                        conn_impact,
                        selected_player_id,
                        impact_season,
                        impact_season_type,
                        min_games=2
                    )

                if defense_splits:
                    # Create two columns for pace vs defense quality
                    def_col1, def_col2 = st.columns(2)

                    # PACE SPLITS
                    with def_col1:
                        st.markdown("#### ⚡ Pace-Based Splits")
                        st.caption("Performance against fast vs slow tempo teams")

                        pace_splits = [s for s in defense_splits if 'Pace' in s.defense_type]

                        if pace_splits:
                            pace_data = []
                            for split in pace_splits:
                                # Indicator based on performance delta
                                if split.pts_vs_average >= 3:
                                    indicator = "🟢🟢"  # Excels
                                elif split.pts_vs_average >= 1:
                                    indicator = "🟢"  # Good
                                elif split.pts_vs_average <= -3:
                                    indicator = "🔴🔴"  # Struggles
                                elif split.pts_vs_average <= -1:
                                    indicator = "🔴"  # Below average
                                else:
                                    indicator = "🟡"  # Neutral

                                pace_data.append({
                                    "": indicator,
                                    "Pace Type": split.defense_type,
                                    "Games": split.games_played,
                                    "PPG": f"{split.avg_pts:.1f}",
                                    "vs Avg": f"{split.pts_vs_average:+.1f}",
                                    "3PM": f"{split.avg_fg3m:.1f}",
                                    "3PM Δ": f"{split.fg3m_vs_average:+.1f}",
                                    "Min": f"{split.avg_minutes:.1f}",
                                    "Sample Teams": ", ".join(split.sample_teams[:2])
                                })

                            pace_df = pd.DataFrame(pace_data)
                            st.dataframe(pace_df, use_container_width=True, hide_index=True)

                            st.caption(
                                "🟢🟢 Excels (+3 PPG) | 🟢 Above avg (+1 PPG) | 🟡 Neutral | "
                                "🔴 Below avg (-1 PPG) | 🔴🔴 Struggles (-3 PPG)"
                            )

                            # Highlight best pace matchup
                            best_pace = max(pace_splits, key=lambda x: x.pts_vs_average)
                            if best_pace.pts_vs_average >= 2:
                                st.success(
                                    f"💡 **Pace Advantage**: {selected_player_name} performs best against "
                                    f"**{best_pace.defense_type}** teams, averaging **{best_pace.pts_vs_average:+.1f} more PPG** "
                                    f"than usual ({best_pace.avg_pts:.1f} vs {best_pace.avg_pts - best_pace.pts_vs_average:.1f} PPG)."
                                )
                        else:
                            st.info("Not enough games against different pace types yet.")

                    # DEFENSE QUALITY SPLITS
                    with def_col2:
                        st.markdown("#### 🛡️ Defense Quality Splits")
                        st.caption("Performance against elite vs weak defenses")

                        def_quality_splits = [s for s in defense_splits if 'Defense' in s.defense_type]

                        if def_quality_splits:
                            def_data = []
                            for split in def_quality_splits:
                                # Indicator based on performance delta
                                if split.pts_vs_average >= 3:
                                    indicator = "🟢🟢"
                                elif split.pts_vs_average >= 1:
                                    indicator = "🟢"
                                elif split.pts_vs_average <= -3:
                                    indicator = "🔴🔴"
                                elif split.pts_vs_average <= -1:
                                    indicator = "🔴"
                                else:
                                    indicator = "🟡"

                                def_data.append({
                                    "": indicator,
                                    "Defense Type": split.defense_type,
                                    "Games": split.games_played,
                                    "PPG": f"{split.avg_pts:.1f}",
                                    "vs Avg": f"{split.pts_vs_average:+.1f}",
                                    "3PM": f"{split.avg_fg3m:.1f}",
                                    "3PM Δ": f"{split.fg3m_vs_average:+.1f}",
                                    "Min": f"{split.avg_minutes:.1f}",
                                    "Sample Teams": ", ".join(split.sample_teams[:2])
                                })

                            def_df = pd.DataFrame(def_data)
                            st.dataframe(def_df, use_container_width=True, hide_index=True)

                            st.caption(
                                "🟢🟢 Excels (+3 PPG) | 🟢 Above avg (+1 PPG) | 🟡 Neutral | "
                                "🔴 Below avg (-1 PPG) | 🔴🔴 Struggles (-3 PPG)"
                            )

                            # Highlight against weak defenses
                            weak_def = next((s for s in def_quality_splits if s.defense_type == 'Weak Defense'), None)
                            elite_def = next((s for s in def_quality_splits if s.defense_type == 'Elite Defense'), None)

                            if weak_def and weak_def.pts_vs_average >= 2:
                                st.success(
                                    f"💡 **Exploits Weak Defenses**: {selected_player_name} averages "
                                    f"**{weak_def.pts_vs_average:+.1f} more PPG** against weak defenses "
                                    f"({weak_def.avg_pts:.1f} PPG in {weak_def.games_played} games)."
                                )
                            elif elite_def and elite_def.pts_vs_average >= 1:
                                st.success(
                                    f"💡 **Elite Defense Buster**: {selected_player_name} still scores well against "
                                    f"elite defenses, averaging {elite_def.avg_pts:.1f} PPG "
                                    f"({elite_def.pts_vs_average:+.1f} vs season average)."
                                )
                        else:
                            st.info("Not enough games against different defense quality levels yet.")

                else:
                    st.info(
                        f"📊 **Not enough data for defense type analysis**\n\n"
                        f"To analyze {selected_player_name}'s performance vs different defense types, we need:\n"
                        f"- At least 2 games against fast pace teams\n"
                        f"- At least 2 games against slow pace teams\n"
                        f"- At least 2 games against elite defenses\n"
                        f"- At least 2 games against weak defenses\n\n"
                        f"This analysis becomes more reliable as the season progresses."
                    )

            except Exception as exc:
                st.error(f"Error analyzing defense type performance: {exc}")
                import traceback
                st.code(traceback.format_exc())

    # PLAYER CORRELATIONS TAB
    with correlations_tab:
        st.markdown("### Player Correlations & Lineup Combinations")
        st.markdown(
            "Discover which teammates boost this player's performance (ideal for DFS stacks) "
            "and which opponent teams they excel against (matchup advantages)."
        )

        if not selected_player_id:
            st.warning("Select a player above to view player correlations.")
        else:
            # Create sub-sections for teammates vs opponents
            corr_col1, corr_col2 = st.columns(2)

            # TEAMMATE CORRELATIONS
            with corr_col1:
                st.markdown("#### 🤝 Teammate Synergy")
                st.caption("How do teammates perform when this player is in/out of the lineup?")

                try:
                    with st.spinner("Analyzing teammate correlations..."):
                        teammate_correlations = pca.calculate_teammate_correlations(
                            conn_impact,
                            selected_player_id,
                            impact_season,
                            impact_season_type,
                            min_games_together=2
                        )

                    if teammate_correlations:
                        # Show top 10 correlations
                        st.markdown(f"**Top Synergies for {selected_player_name}**")

                        teammate_data = []
                        for tc in teammate_correlations[:10]:
                            # Color code based on correlation strength
                            if tc.correlation_score >= 60:
                                indicator = "🟢"  # Strong positive
                            elif tc.correlation_score >= 50:
                                indicator = "🟡"  # Neutral/slight positive
                            else:
                                indicator = "🔴"  # Negative

                            teammate_data.append({
                                "": indicator,
                                "Teammate": tc.teammate_name,
                                "PPG Together": f"{tc.avg_pts_together:.1f}",
                                "PPG Apart": f"{tc.avg_pts_apart:.1f}",
                                "Δ PPG": f"{tc.pts_delta:+.1f}",
                                "3PM Δ": f"{tc.fg3m_delta:+.1f}",
                                "Games": f"{tc.games_together}/{tc.games_apart}",
                                "Score": f"{tc.correlation_score:.0f}"
                            })

                        teammate_df = pd.DataFrame(teammate_data)
                        st.dataframe(teammate_df, use_container_width=True, hide_index=True)

                        # Explanation
                        st.caption(
                            "🟢 Strong synergy (60+) | 🟡 Moderate synergy (50-60) | 🔴 Negative synergy (<50)\n\n"
                            "**Score** combines PPG boost, 3PM boost, and sample size. Higher = better for DFS stacks."
                        )

                        # Highlight best stack
                        best_teammate = teammate_correlations[0]
                        if best_teammate.correlation_score >= 55:
                            st.success(
                                f"💡 **DFS Stack Opportunity:** {selected_player_name} + {best_teammate.teammate_name}\n\n"
                                f"{best_teammate.teammate_name} averages **{best_teammate.pts_delta:+.1f} more points** "
                                f"when playing WITH {selected_player_name} "
                                f"({best_teammate.avg_pts_together:.1f} vs {best_teammate.avg_pts_apart:.1f} PPG without)."
                            )
                    else:
                        st.info(
                            f"📊 **Not enough data for teammate correlation analysis**\n\n"
                            f"To analyze how teammates perform with/without {selected_player_name}, we need:\n"
                            f"- At least 2 games where {selected_player_name} played\n"
                            f"- At least 1 game where {selected_player_name} was absent (so teammates have comparison data)\n\n"
                            f"This analysis becomes available when {selected_player_name} misses games due to rest/injury."
                        )

                except Exception as exc:
                    st.error(f"Error analyzing teammate correlations: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

            # OPPONENT CORRELATIONS
            with corr_col2:
                st.markdown("#### 🎯 Matchup Advantages")
                st.caption("Which opposing teams does this player excel against?")

                try:
                    with st.spinner("Analyzing opponent matchups..."):
                        opponent_correlations = pca.calculate_opponent_correlations(
                            conn_impact,
                            selected_player_id,
                            impact_season,
                            impact_season_type,
                            min_games_vs=1
                        )

                    if opponent_correlations:
                        # Show top/bottom matchups
                        st.markdown(f"**Best/Worst Matchups for {selected_player_name}**")

                        # Get top 5 and bottom 5
                        top_matchups = opponent_correlations[:5]
                        bottom_matchups = opponent_correlations[-5:][::-1]  # Reverse to show worst first

                        opponent_data = []

                        # Add top matchups
                        for oc in top_matchups:
                            if oc.matchup_score >= 55:
                                indicator = "✅"  # Favorable matchup
                            elif oc.matchup_score >= 50:
                                indicator = "➖"  # Neutral
                            else:
                                indicator = "❌"  # Unfavorable

                            opponent_data.append({
                                "": indicator,
                                "Opponent": oc.opponent_team_name,
                                "PPG vs": f"{oc.avg_pts_vs:.1f}",
                                "PPG vs Others": f"{oc.avg_pts_vs_others:.1f}",
                                "Δ PPG": f"{oc.pts_delta:+.1f}",
                                "3PM Δ": f"{oc.fg3m_delta:+.1f}",
                                "Games": oc.games_vs_opponent,
                                "Score": f"{oc.matchup_score:.0f}"
                            })

                        opponent_df = pd.DataFrame(opponent_data)
                        st.dataframe(opponent_df, use_container_width=True, hide_index=True)

                        # Explanation
                        st.caption(
                            "✅ Favorable matchup (55+) | ➖ Neutral (45-55) | ❌ Unfavorable (<45)\n\n"
                            "**Score** shows matchup advantage. Higher = player performs better vs this opponent."
                        )

                        # Highlight best matchup
                        best_matchup = opponent_correlations[0]
                        if best_matchup.matchup_score >= 55 and best_matchup.pts_delta > 2:
                            st.success(
                                f"💡 **Matchup Advantage:** {selected_player_name} vs {best_matchup.opponent_team_name}\n\n"
                                f"{selected_player_name} averages **{best_matchup.pts_delta:+.1f} more points** "
                                f"against {best_matchup.opponent_team_name} "
                                f"({best_matchup.avg_pts_vs:.1f} vs {best_matchup.avg_pts_vs_others:.1f} PPG vs others)."
                            )
                    else:
                        st.info(
                            f"Not enough data to analyze opponent matchups for {selected_player_name}. "
                            "Player needs at least 1 game against multiple different opponents."
                        )

                except Exception as exc:
                    st.error(f"Error analyzing opponent matchups: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

    # ABSENCE IMPACT TAB (existing injury analysis)
    with absence_tab:
        st.markdown("### Player Absence Impact")
        st.markdown(
            "Analyze how the team, teammates, and opponents perform when this player is absent "
            "(injury, rest, suspension, etc.). Absences are inferred from missing game logs."
        )

        if not selected_player_id:
            st.warning("Select a player above to view absence impact analysis.")
        else:
            # Calculate impact metrics using the selected player
            try:
                with st.spinner(f"Analyzing impact of {selected_player_name}'s absences..."):
                    team_impact = iia.calculate_team_impact(
                        conn_impact,
                        selected_player_id,
                        selected_player_name,
                        impact_season,
                        impact_season_type,
                    )

                    teammate_impacts = iia.calculate_teammate_redistribution(
                        conn_impact,
                        selected_player_id,
                        impact_season,
                        impact_season_type,
                        min_games=1,
                    )

                    opponent_impact = iia.calculate_opponent_impact(
                        conn_impact,
                        selected_player_id,
                        impact_season,
                        impact_season_type,
                    )

                if team_impact is None:
                    st.warning(f"No data found for {selected_player_name} in {impact_season}.")
                else:
                    # Display key metrics
                    st.markdown(f"### Team Performance: {team_impact.team_name}")

                    # Warning if data looks incomplete
                    if team_impact.games_absent > 0 and team_impact.team_avg_pts_without == 0.0:
                        st.warning(
                            f"⚠️ The game(s) {selected_player_name} missed may not have final scores yet "
                            "(scheduled future game or incomplete data). Impact metrics may show as 0.0."
                        )

                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric(
                            "Games with Player",
                            team_impact.games_played,
                            f"{team_impact.total_team_games} total team games",
                        )
                    with metric_cols[1]:
                        st.metric(
                            "Games without Player",
                            team_impact.games_absent,
                        )
                    with metric_cols[2]:
                        win_pct_delta_display = f"{team_impact.win_pct_delta:+.1%}"
                        st.metric(
                            "Win % Impact",
                            f"{team_impact.team_win_pct_with:.1%}",
                            win_pct_delta_display,
                            delta_color="normal" if team_impact.win_pct_delta >= 0 else "inverse",
                        )
                    with metric_cols[3]:
                        pts_delta_display = f"{team_impact.offensive_rating_delta:+.1f}"
                        st.metric(
                            "Offensive Impact (PPG)",
                            f"{team_impact.team_avg_pts_with:.1f}",
                            pts_delta_display,
                            delta_color="normal" if team_impact.offensive_rating_delta >= 0 else "inverse",
                        )

                    # Detailed comparison table
                    st.markdown("#### Detailed Team Performance Comparison")
                    comparison_data = {
                        "Metric": [
                            "Record",
                            "Win %",
                            "Avg Points Scored",
                            "Avg Points Allowed",
                            "Point Differential",
                        ],
                        "With Player": [
                            f"{team_impact.team_wins_with}-{team_impact.team_losses_with}",
                            f"{team_impact.team_win_pct_with:.1%}",
                            f"{team_impact.team_avg_pts_with:.1f}",
                            f"{team_impact.team_avg_pts_allowed_with:.1f}",
                            f"{team_impact.team_avg_pts_with - team_impact.team_avg_pts_allowed_with:+.1f}",
                        ],
                        "Without Player": [
                            f"{team_impact.team_wins_without}-{team_impact.team_losses_without}",
                            f"{team_impact.team_win_pct_without:.1%}",
                            f"{team_impact.team_avg_pts_without:.1f}",
                            f"{team_impact.team_avg_pts_allowed_without:.1f}",
                            f"{team_impact.team_avg_pts_without - team_impact.team_avg_pts_allowed_without:+.1f}",
                        ],
                        "Delta": [
                            "",
                            f"{team_impact.win_pct_delta:+.1%}",
                            f"{team_impact.offensive_rating_delta:+.1f}",
                            f"{team_impact.defensive_rating_delta:+.1f}",
                            f"{(team_impact.offensive_rating_delta - team_impact.defensive_rating_delta):+.1f}",
                        ],
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                    # Teammate redistribution analysis
                    if teammate_impacts:
                        st.markdown("#### Teammate Usage & Scoring Redistribution")
                        st.markdown(
                            f"How do teammates perform when **{selected_player_name}** is absent? "
                            "Positive deltas indicate increased production."
                        )

                        teammate_data = []
                        for tm in teammate_impacts[:10]:  # Top 10 most affected
                            teammate_data.append({
                                "Teammate": tm.teammate_name,
                                "PPG With": f"{tm.avg_pts_with:.1f}",
                                "PPG Without": f"{tm.avg_pts_without:.1f}",
                                "Pts Δ": f"{tm.pts_delta:+.1f}",
                                "USG% With": f"{tm.avg_usg_with:.1f}",
                                "USG% Without": f"{tm.avg_usg_without:.1f}",
                                "USG Δ": f"{tm.usg_delta:+.1f}",
                                "Min Δ": f"{tm.minutes_delta:+.1f}",
                                "Games Together": tm.games_together,
                                "Games Apart": tm.games_apart,
                            })

                        teammate_df = pd.DataFrame(teammate_data)
                        st.dataframe(teammate_df, use_container_width=True, hide_index=True)

                        # Visualization: Top beneficiaries
                        st.markdown("##### Top Beneficiaries (Biggest PPG Increase)")
                        top_beneficiaries = [tm for tm in teammate_impacts if tm.pts_delta > 0][:5]
                        if top_beneficiaries:
                            import plotly.graph_objects as go

                            fig = go.Figure()
                            teammate_names = [tm.teammate_name for tm in top_beneficiaries]
                            pts_with = [tm.avg_pts_with for tm in top_beneficiaries]
                            pts_without = [tm.avg_pts_without for tm in top_beneficiaries]

                            fig.add_trace(go.Bar(
                                name=f'With {selected_player_name}',
                                x=teammate_names,
                                y=pts_with,
                                text=[f"{v:.1f}" for v in pts_with],
                                textposition='auto',
                            ))
                            fig.add_trace(go.Bar(
                                name=f'Without {selected_player_name}',
                                x=teammate_names,
                                y=pts_without,
                                text=[f"{v:.1f}" for v in pts_without],
                                textposition='auto',
                            ))

                            fig.update_layout(
                                title="Points Per Game: With vs Without Key Player",
                                xaxis_title="Teammate",
                                yaxis_title="Points Per Game",
                                barmode='group',
                                height=400,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No teammates showed increased scoring when player was absent.")
                    else:
                        if team_impact.games_absent > 0:
                            st.info(
                                f"Unable to analyze teammate redistribution. This usually means:\n"
                                f"- The game(s) {selected_player_name} missed may not have complete data yet (scheduled/in-progress)\n"
                                f"- Teammates who played when {selected_player_name} was out didn't have enough overlapping games\n"
                                f"- Try selecting a different season or player with more complete absence data"
                            )
                        else:
                            st.info(f"{selected_player_name} has not missed any games this season (perfect attendance).")

                    # Opponent impact analysis
                    if opponent_impact:
                        st.markdown("#### Opponent Performance Impact")
                        st.markdown(
                            f"How do opposing teams perform when **{selected_player_name}** is absent? "
                            "Positive deltas indicate opponents perform better without the player."
                        )

                        opp_metric_cols = st.columns(3)
                        with opp_metric_cols[0]:
                            st.metric(
                                "Opp PPG (with player)",
                                f"{opponent_impact.avg_opp_pts_with:.1f}",
                                f"{opponent_impact.opp_pts_delta:+.1f} without",
                                delta_color="inverse" if opponent_impact.opp_pts_delta > 0 else "normal",
                            )
                        with opp_metric_cols[1]:
                            st.metric(
                                "Opp 3PM (with player)",
                                f"{opponent_impact.avg_opp_fg3m_with:.1f}",
                                f"{opponent_impact.opp_fg3m_delta:+.1f} without",
                                delta_color="inverse" if opponent_impact.opp_fg3m_delta > 0 else "normal",
                            )
                        with opp_metric_cols[2]:
                            if opponent_impact.avg_opp_fg_pct_with > 0:
                                st.metric(
                                    "Opp Efficiency (with)",
                                    f"{opponent_impact.avg_opp_fg_pct_with:.1f}%",
                                    f"{opponent_impact.opp_fg_pct_delta:+.1f}% without",
                                    delta_color="inverse" if opponent_impact.opp_fg_pct_delta > 0 else "normal",
                                )
                            else:
                                st.metric("Opp Efficiency", "N/A")

                        opp_comparison_data = {
                            "Metric": [
                                "Opponent Avg Points",
                                "Opponent Avg 3PM",
                            ],
                            "With Player": [
                                f"{opponent_impact.avg_opp_pts_with:.1f}",
                                f"{opponent_impact.avg_opp_fg3m_with:.1f}",
                            ],
                            "Without Player": [
                                f"{opponent_impact.avg_opp_pts_without:.1f}",
                                f"{opponent_impact.avg_opp_fg3m_without:.1f}",
                            ],
                            "Delta": [
                                f"{opponent_impact.opp_pts_delta:+.1f}",
                                f"{opponent_impact.opp_fg3m_delta:+.1f}",
                            ],
                        }
                        opp_comparison_df = pd.DataFrame(opp_comparison_data)
                        st.dataframe(opp_comparison_df, use_container_width=True, hide_index=True)

            except Exception as exc:  # noqa: BLE001
                st.error(f"Error loading injury impact data: {exc}")
                import traceback
                st.code(traceback.format_exc())

# Prediction tab -----------------------------------------------------------
with predictions_tab:
    st.subheader("📊 Prediction Accuracy Tracker")
    st.caption("Track projection accuracy vs actual performance to improve the model")

    # Get database connection
    pred_conn = get_connection(str(db_path))

    # Date selector for viewing predictions
    col1, col2 = st.columns([1, 1])

    with col1:
        # Get available dates with predictions
        try:
            # Ensure predictions table exists
            pt.create_predictions_table(pred_conn)

            cursor = pred_conn.cursor()
            cursor.execute("""
                SELECT DISTINCT game_date
                FROM predictions
                ORDER BY game_date DESC
                LIMIT 30
            """)
            available_dates = [row[0] for row in cursor.fetchall()]

            if available_dates:
                selected_date = st.selectbox(
                    "Game Date",
                    options=available_dates,
                    index=0
                )
            else:
                st.info("No predictions logged yet. They will appear after viewing Daily Games.")
                selected_date = None
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            import traceback
            st.code(traceback.format_exc())
            selected_date = None

    with col2:
        if selected_date:
            # Button to fetch latest game data and update actuals
            if st.button("🔄 Fetch & Score Latest Games", help="Download latest NBA game data and update prediction actuals"):
                try:
                    # Step 1: Fetch latest game data
                    with st.spinner("Step 1/3: Fetching latest game data from NBA API..."):
                        # Import and run the database builder directly
                        from nba_to_sqlite import build_database

                        build_database(
                            db_path=db_path,
                            season="2025-26",
                            season_type="Regular Season",
                            include_rosters=False,
                            throttle_seconds=0.6
                        )

                        st.success("✅ Game data fetched successfully!")

                    # Step 2: Score predictions
                    with st.spinner("Step 2/3: Scoring predictions..."):
                        # Create a new connection for querying unscored dates
                        import sqlite3
                        temp_conn = sqlite3.connect(str(db_path))
                        temp_cursor = temp_conn.cursor()

                        # Get unscored dates
                        temp_cursor.execute('''
                            SELECT DISTINCT game_date
                            FROM predictions
                            WHERE did_play IS NULL OR actual_ppg IS NULL
                            ORDER BY game_date DESC
                        ''')
                        unscored_dates = [row[0] for row in temp_cursor.fetchall()]
                        temp_conn.close()

                        if unscored_dates:
                            st.info(f"Found {len(unscored_dates)} date(s) with unscored predictions")

                            # Score predictions (score_predictions creates its own connection)
                            import score_predictions as sp
                            for date_to_score in unscored_dates:
                                sp.score_predictions_for_date(date_to_score)

                            st.success(f"✅ Predictions scored for {len(unscored_dates)} date(s)!")
                        else:
                            st.info("✅ All predictions are already scored!")

                    # Step 3: Upload to S3
                    with st.spinner("Step 3/3: Syncing to cloud..."):
                        storage = s3_storage.S3PredictionStorage()
                        if storage.is_connected():
                            success, message = storage.upload_database(db_path)
                            if success:
                                st.success(f"✅ {message}")
                                st.info("💡 Restart your Streamlit Cloud app to see the updates!")
                            else:
                                st.warning(f"⚠️ S3 upload: {message}")
                        else:
                            st.info("ℹ️ S3 not configured - updates saved locally only")

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Alternative: Just update actuals from existing game logs (faster)
            if st.button("⚡ Quick Update from Existing Logs", help="Update actuals using game logs already in database (faster)"):
                with st.spinner(f"Updating actuals for {selected_date}..."):
                    updated = pt.bulk_update_actuals_from_game_logs(pred_conn, selected_date)
                    st.success(f"Updated {updated} predictions with actual performance!")
                    st.rerun()

    if selected_date:
        st.divider()

        # Summary metrics for the selected date
        try:
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(actual_ppg) as with_actuals,
                    AVG(CASE WHEN actual_ppg IS NOT NULL THEN abs_error ELSE NULL END) as avg_error,
                    AVG(CASE WHEN hit_floor_ceiling = 1 THEN 1.0 ELSE 0.0 END) as hit_rate,
                    SUM(CASE WHEN error < 0 THEN 1 ELSE 0 END) as over_proj,
                    SUM(CASE WHEN error > 0 THEN 1 ELSE 0 END) as under_proj
                FROM predictions
                WHERE game_date = ?
            """, (selected_date,))

            result = cursor.fetchone()
            total, with_actuals, avg_error, hit_rate, over_proj, under_proj = result

            # Display metrics in columns
            metric_cols = st.columns(5)

            with metric_cols[0]:
                st.metric("Total Predictions", total)

            with metric_cols[1]:
                st.metric("With Actuals", with_actuals or 0)

            with metric_cols[2]:
                if avg_error:
                    st.metric("Avg Error (MAE)", f"{avg_error:.2f} PPG")
                else:
                    st.metric("Avg Error (MAE)", "N/A")

            with metric_cols[3]:
                if with_actuals and with_actuals > 0:
                    st.metric("Hit Rate", f"{hit_rate:.1%}")
                else:
                    st.metric("Hit Rate", "N/A")

            with metric_cols[4]:
                if with_actuals:
                    bias = f"{over_proj or 0} over, {under_proj or 0} under"
                    st.metric("Projection Bias", bias)
                else:
                    st.metric("Projection Bias", "N/A")

        except Exception as e:
            st.error(f"Error calculating metrics: {e}")

        st.divider()

        # Predictions vs Actuals table
        st.subheader(f"Predictions for {selected_date}")

        try:
            df = pt.get_predictions_vs_actuals(pred_conn, selected_date)

            if not df.empty:
                # Add helpful columns
                df['Status'] = df['actual_ppg'].apply(
                    lambda x: '✅ Complete' if pd.notna(x) else '⏳ Pending'
                )
                df['Accuracy'] = df.apply(
                    lambda row: '🎯 Hit Range' if row['hit_floor_ceiling'] == 1
                    else '📉 Outside Range' if pd.notna(row['actual_ppg'])
                    else '',
                    axis=1
                )

                # Format numeric columns
                display_df = df[['player_name', 'team_name', 'opponent_name', 'Status',
                                 'projected_ppg', 'actual_ppg', 'error', 'abs_error',
                                 'proj_confidence', 'proj_floor', 'proj_ceiling', 'Accuracy',
                                 'dfs_score', 'dfs_grade', 'analytics_used']].copy()

                display_df.columns = ['Player', 'Team', 'Opponent', 'Status',
                                      'Proj PPG', 'Actual PPG', 'Error', 'Abs Error',
                                      'Confidence', 'Floor', 'Ceiling', 'Accuracy',
                                      'DFS Score', 'Grade', 'Analytics']

                # Sort by DFS Score descending
                display_df = display_df.sort_values('DFS Score', ascending=False)

                # Filter options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    show_only_actuals = st.checkbox("Show only predictions with actuals", value=False)
                with filter_col2:
                    min_dfs = st.slider("Min DFS Score", 0.0, 100.0, 0.0, 5.0)

                if show_only_actuals:
                    display_df = display_df[display_df['Status'] == '✅ Complete']

                if min_dfs > 0:
                    display_df = display_df[display_df['DFS Score'] >= min_dfs]

                # Display
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=600
                )

                # Action buttons
                col1, col2 = st.columns(2)

                with col1:
                    # Download button
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"predictions_{selected_date}.csv",
                        mime="text/csv"
                    )

                with col2:
                    # S3 Backup button
                    storage = s3_storage.S3PredictionStorage()
                    if storage.is_connected():
                        if st.button("☁️ Sync to Cloud", help="Upload scored predictions to S3 for Streamlit Cloud"):
                            with st.spinner("Uploading to S3..."):
                                success, message = storage.upload_database(db_path)
                                if success:
                                    st.success(f"✅ {message}")
                                    st.info("💡 Restart your Streamlit Cloud app to see the updated predictions!")
                                else:
                                    st.error(f"❌ {message}")

            else:
                st.info(f"No predictions found for {selected_date}")

        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            import traceback
            st.code(traceback.format_exc())

        st.divider()

        # Best and worst predictions
        if with_actuals and with_actuals > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Best Predictions")
                try:
                    cursor.execute("""
                        SELECT player_name, projected_ppg, actual_ppg, error, abs_error, analytics_used
                        FROM predictions
                        WHERE game_date = ? AND actual_ppg IS NOT NULL
                        ORDER BY abs_error ASC
                        LIMIT 5
                    """, (selected_date,))

                    best = cursor.fetchall()
                    if best:
                        for player_name, proj, actual, error, abs_err, analytics in best:
                            st.write(f"**{player_name}** {analytics}")
                            st.write(f"Proj: {proj:.1f}, Actual: {actual:.1f} ({error:+.1f}, MAE: {abs_err:.1f})")
                            st.write("---")
                except Exception as e:
                    st.error(f"Error: {e}")

            with col2:
                st.subheader("❌ Worst Predictions")
                try:
                    cursor.execute("""
                        SELECT player_name, projected_ppg, actual_ppg, error, abs_error, analytics_used
                        FROM predictions
                        WHERE game_date = ? AND actual_ppg IS NOT NULL
                        ORDER BY abs_error DESC
                        LIMIT 5
                    """, (selected_date,))

                    worst = cursor.fetchall()
                    if worst:
                        for player_name, proj, actual, error, abs_err, analytics in worst:
                            st.write(f"**{player_name}** {analytics}")
                            st.write(f"Proj: {proj:.1f}, Actual: {actual:.1f} ({error:+.1f}, MAE: {abs_err:.1f})")
                            st.write("---")
                except Exception as e:
                    st.error(f"Error: {e}")

# Injury Admin Tab --------------------------------------------------------
with injury_admin_tab:
    st.header("🚑 Injury Administration")
    st.write("Manage injured players and adjust predictions for teammate impacts")

    injury_conn = get_connection(str(db_path))

    # Ensure injury_list table exists
    ia.create_injury_list_table(injury_conn)

    st.divider()

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Active Injury List")

        # Display current injury list
        injury_list = ia.get_active_injuries(injury_conn, check_return_dates=True)

        if injury_list:
            # Convert to DataFrame for display
            injury_df = pd.DataFrame(injury_list)
            display_df = injury_df[['player_name', 'team_name', 'injury_date', 'expected_return_date', 'notes']]
            display_df.columns = ['Player', 'Team', 'Injury Date', 'Expected Return', 'Notes']
            st.dataframe(display_df, use_container_width=True)

            # Mark as Returned functionality
            st.markdown("##### Mark Player as Returned")
            returned_player = st.selectbox(
                "Select player to mark as returned",
                options=[(inj['player_id'], inj['player_name']) for inj in injury_list],
                format_func=lambda x: x[1],
                key="mark_returned_selector"
            )

            if st.button("✅ Mark as Returned", key="mark_returned_btn"):
                player_id = returned_player[0]
                success = ia.remove_from_injury_list(injury_conn, player_id)
                if success:
                    st.success(f"✅ {returned_player[1]} marked as returned")
                    st.rerun()
                else:
                    st.error("Failed to update injury status")
        else:
            st.info("No active injuries. Add players using the form on the right.")

    with col2:
        st.subheader("➕ Add Injured Player")

        # Build player options from database
        # Query all players from player_game_logs (most recent games for current season)
        player_query = """
            SELECT DISTINCT
                p.player_id,
                p.player_name,
                p.team_name
            FROM player_game_logs p
            WHERE p.season = ?
            ORDER BY p.team_name, p.player_name
        """

        try:
            all_players_df = pd.read_sql_query(
                player_query,
                injury_conn,
                params=[DEFAULT_SEASON]
            )

            if not all_players_df.empty:
                player_options = [
                    (row['player_id'], row['player_name'], row['team_name'])
                    for _, row in all_players_df.iterrows()
                ]

                selected_player = st.selectbox(
                    "Select injured player",
                    options=player_options,
                    format_func=lambda x: f"{x[2]} - {x[1]}",  # "LAL - LeBron James"
                    key="add_injured_player_selector"
                )

                return_date = st.date_input(
                    "Expected return date (optional)",
                    value=None,
                    key="return_date_input"
                )

                notes = st.text_area(
                    "Notes (e.g., injury type, severity)",
                    placeholder="Sprained ankle, day-to-day",
                    key="injury_notes_input"
                )

                if st.button("➕ Add to Injury List", type="primary", key="add_injury_btn"):
                    player_id, player_name, team_name = selected_player

                    injury_id = ia.add_to_injury_list(
                        injury_conn,
                        player_id=player_id,
                        player_name=player_name,
                        team_name=team_name,
                        expected_return_date=str(return_date) if return_date else None,
                        notes=notes if notes else None
                    )

                    if injury_id:
                        st.success(f"✅ Added {player_name} to injury list")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ {player_name} is already on the active injury list")
            else:
                st.error("No players found in database")

        except Exception as e:
            st.error(f"Error loading players: {e}")

    st.divider()

    # Apply Adjustments Section
    st.subheader("⚙️ Apply Injury Adjustments to Predictions")
    st.caption("Use active injury list to automatically adjust predictions for a specific game date")

    adjustment_date = st.date_input(
        "Game date to adjust",
        value=default_game_date(),
        key="adjustment_game_date"
    )

    # Get active injuries for preview
    active_injuries = ia.get_active_injuries(injury_conn, check_return_dates=True)

    if active_injuries:
        st.markdown(f"**Active injuries:** {len(active_injuries)} players")
        injured_names = [inj['player_name'] for inj in active_injuries]
        st.write(", ".join(injured_names))

        # Preview adjustments
        injured_ids = [inj['player_id'] for inj in active_injuries]

        if st.button("🔍 Preview Adjustments", key="preview_adj_btn"):
            with st.spinner("Calculating adjustments..."):
                preview_df = ia.preview_adjustments(
                    injured_ids,
                    str(adjustment_date),
                    injury_conn,
                    min_historical_games=3
                )

            if not preview_df.empty:
                st.dataframe(preview_df, use_container_width=True)

                # Apply/Reset buttons
                adj_col1, adj_col2 = st.columns(2)

                with adj_col1:
                    if st.button("✅ Apply Adjustments", type="primary", key="apply_adj_btn"):
                        with st.spinner("Applying..."):
                            adjusted, skipped, records = ia.apply_injury_adjustments(
                                injured_ids,
                                str(adjustment_date),
                                injury_conn
                            )
                        st.success(f"✅ Adjusted {adjusted} predictions ({skipped} skipped)")
                        st.rerun()

                with adj_col2:
                    if st.button("🔄 Reset Adjustments", key="reset_adj_btn"):
                        with st.spinner("Resetting..."):
                            reset_count = ia.reset_adjustments(str(adjustment_date), injury_conn)
                        st.success(f"🔄 Reset {reset_count} predictions")
                        st.rerun()
            else:
                st.info("No teammates found with historical data for these injuries (need 3+ games)")

        # Show before/after if adjustments exist
        adjusted_summary = ia.get_adjusted_predictions_summary(str(adjustment_date), injury_conn)
        if not adjusted_summary.empty:
            st.divider()
            st.markdown("### Before/After Comparison")

            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.markdown("#### Original")
                st.dataframe(
                    adjusted_summary[['Player', 'Team', 'Original PPG', 'Original Range']],
                    use_container_width=True
                )

            with comp_col2:
                st.markdown("#### Adjusted")
                st.dataframe(
                    adjusted_summary[['Player', 'Team', 'Adjusted PPG', 'New Range', 'Boost']],
                    use_container_width=True
                )
    else:
        st.info("No active injuries. Add players using the form above.")

# Admin Panel tab --------------------------------------------------------
with admin_tab:
    st.header("🔧 Admin Panel")
    st.write("One-click data updates and prediction scoring")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Database Status")

        try:
            admin_conn = get_connection(str(db_path))
            cursor = admin_conn.cursor()

            # Get yesterday's date
            from datetime import datetime, timedelta
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            # Check predictions
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE game_date = ?', (yesterday,))
            pred_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM predictions WHERE game_date = ? AND actual_ppg IS NOT NULL', (yesterday,))
            scored_count = cursor.fetchone()[0]

            # Check game logs
            cursor.execute(f"SELECT COUNT(DISTINCT game_id) FROM player_game_logs WHERE game_date LIKE '{yesterday}%'")
            games_count = cursor.fetchone()[0]

            st.metric("Yesterday's Predictions", pred_count)
            st.metric("Scored Predictions", f"{scored_count}/{pred_count}")
            st.metric("Games Available", games_count)

        except Exception as e:
            st.error(f"Error checking status: {e}")

    with col2:
        st.subheader("🚀 One-Click Update")
        st.write("This button will:")
        st.write("1. Fetch latest NBA game data")
        st.write("2. Score all unscored predictions")
        st.write("3. Upload to S3 (if configured)")

        if st.button("▶️ Run Daily Update", type="primary", use_container_width=True):
            import sys
            import io
            from contextlib import redirect_stdout, redirect_stderr

            # Create string buffers to capture output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            with st.spinner("Running daily update..."):
                try:
                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        # Step 1: Fetch latest NBA game data
                        st.info("Step 1/3: Fetching latest NBA game data...")
                        print("="*70)
                        print("STEP 1: Fetching Latest NBA Game Data")
                        print("="*70)

                        # Import and run nba_to_sqlite
                        sys.argv = ["nba_to_sqlite.py", "--season", "2025-26", "--season-type", "Regular Season", "--no-include-rosters"]
                        import nba_to_sqlite
                        nba_to_sqlite.main()

                        # Step 2: Score predictions
                        st.info("Step 2/3: Scoring all unscored predictions...")
                        print("\n" + "="*70)
                        print("STEP 2: Scoring All Unscored Predictions")
                        print("="*70)

                        # Import and run score_predictions
                        import score_predictions
                        score_predictions.score_all_unscored()

                        # Step 3: Upload to S3
                        st.info("Step 3/3: Uploading to S3...")
                        print("\n" + "="*70)
                        print("STEP 3: Uploading to S3")
                        print("="*70)

                        # Import and run S3 upload
                        import s3_storage
                        from pathlib import Path

                        storage = s3_storage.S3PredictionStorage()
                        if storage.is_connected():
                            success, message = storage.upload_database(Path("nba_stats.db"))
                            if success:
                                print(f"\n[SUCCESS] {message}")
                                s3_success = True
                            else:
                                print(f"\n[ERROR] S3 upload failed: {message}")
                                s3_success = False
                        else:
                            print("\nWARNING: S3 not configured. Skipping upload.")
                            s3_success = False

                    # Get captured output
                    output = stdout_buffer.getvalue()
                    errors = stderr_buffer.getvalue()

                    # Show success message
                    st.success("✅ Daily update completed successfully!")

                    # Show output in expandable section
                    if output:
                        with st.expander("📋 Update Log", expanded=True):
                            st.code(output, language="text")

                    if errors:
                        with st.expander("⚠️ Warnings/Errors", expanded=False):
                            st.code(errors, language="text")

                    # Show S3 upload status
                    if s3_success:
                        st.info("💡 Database uploaded to S3! The app will now refresh to show updates.")
                    else:
                        st.warning("⚠️ S3 upload was skipped or failed. Local database updated only.")

                    # Auto-refresh the app to show new data
                    st.info("🔄 Refreshing app in 2 seconds...")
                    import time
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    # Get any captured output before the error
                    output = stdout_buffer.getvalue()
                    errors = stderr_buffer.getvalue()

                    st.error(f"❌ Update failed: {str(e)}")

                    if output:
                        with st.expander("📋 Output Before Error", expanded=True):
                            st.code(output, language="text")

                    if errors:
                        with st.expander("❌ Error Output", expanded=True):
                            st.code(errors, language="text")

                    import traceback
                    with st.expander("🔍 Full Error Traceback", expanded=True):
                        st.code(traceback.format_exc(), language="python")

st.divider()
st.caption(
    "Need more context? Re-run the builder (`python nba_to_sqlite.py ...`) to refresh "
    "underlying tables, then use this app to validate outputs."
)
