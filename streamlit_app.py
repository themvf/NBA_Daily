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
import prediction_refresh as pr
import prediction_generator as pg
import s3_storage
import ppm_stats
import position_ppm_stats

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
DEFAULT_MIN_GAMES = 1
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

# Sidebar navigation
st.sidebar.title("📊 Navigation")

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
    "Ceiling Analytics",
    "Injury Admin",
    "Admin Panel",
    "Tournament Strategy",
]

# Initialize selected page in session state
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Tournament Strategy"

# Sidebar selectbox for navigation
selected_page = st.sidebar.selectbox(
    "Select View:",
    tab_titles,
    index=tab_titles.index(st.session_state.selected_page) if st.session_state.selected_page in tab_titles else 0,
    key='page_selector'
)

# Update session state
st.session_state.selected_page = selected_page

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

    # PREDICTION GENERATION - Quick Action Button
    st.markdown("### 🎯 Quick Actions")
    st.markdown("#### Generate Predictions")

    # Date picker for predictions
    pred_date = st.date_input(
        "Date:",
        value=default_game_date(),
        key="quick_gen_date",
        label_visibility="collapsed"
    )

    # Check if predictions already exist for this date
    try:
        conn = get_connection(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE game_date = ?",
            (str(pred_date),)
        )
        existing_count = cursor.fetchone()[0]
        conn.close()

        if existing_count > 0:
            st.caption(f"✓ {existing_count} predictions exist")
    except Exception:
        existing_count = 0

    # Button label changes based on whether predictions exist
    gen_button_label = "🔄 Regenerate" if existing_count > 0 else "🎯 Generate Predictions"

    if st.button(gen_button_label, type="primary", use_container_width=True, key="sidebar_gen_predictions"):
        generate_predictions_ui(pred_date, db_path, builder_config)

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


@st.cache_data(ttl=3600)
def load_ppm_stats(db_path: str, season: str = '2025-26') -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load PPM stats for all teams (offensive and defensive).

    Cached for 1 hour to avoid recalculating for every prediction.

    Returns:
        tuple: (off_ppm_df, def_ppm_df, league_avg_ppm)
    """
    conn = get_connection(db_path)
    off_ppm_df = ppm_stats.get_team_offensive_ppm_stats(conn, season)
    def_ppm_df = ppm_stats.get_team_defensive_ppm_stats(conn, season)
    league_avg_ppm = ppm_stats.get_league_avg_ppm(conn, season)
    return off_ppm_df, def_ppm_df, league_avg_ppm


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


def generate_predictions_ui(pred_date: date, db_path: Path, builder_config: Dict):
    """Handle prediction generation with progress UI in sidebar."""
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    def update_progress(current: int, total: int, message: str):
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"{message} ({current}/{total})")

    try:
        result = pg.generate_predictions_for_date(
            game_date=pred_date,
            db_path=db_path,
            season=builder_config["season"],
            season_type=DEFAULT_SEASON_TYPE,
            progress_callback=update_progress
        )

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Show results
        if result['predictions_logged'] > 0:
            st.sidebar.success(
                f"✅ Generated {result['predictions_logged']} predictions\n\n"
                f"Avg confidence: {result['summary']['avg_confidence']:.0%}\n\n"
                f"Avg DFS score: {result['summary']['avg_dfs_score']:.1f}"
            )

            # Trigger S3 backup if configured
            try:
                storage = s3_storage.S3PredictionStorage()
                if storage.is_connected():
                    backup_success, backup_message = storage.upload_database(db_path)
                    if backup_success:
                        st.sidebar.info("☁️ Backed up to S3")
            except Exception:
                pass  # Silent fail for S3

        if result['predictions_failed'] > 0:
            st.sidebar.warning(f"⚠️ {result['predictions_failed']} predictions failed")

        if result['errors']:
            with st.sidebar.expander("❌ View Errors", expanded=False):
                for error in result['errors'][:5]:  # Show first 5
                    st.error(error)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.sidebar.error(f"❌ Generation failed: {str(e)}")


def load_predictions_for_date(conn: sqlite3.Connection, game_date: str) -> pd.DataFrame:
    """Load all predictions for a specific date from database."""
    query = """
        SELECT
            player_name, team_name, opponent_name,
            projected_ppg, proj_confidence, proj_floor, proj_ceiling,
            season_avg_ppg, recent_avg_5, recent_avg_3,
            dfs_score, dfs_grade, analytics_used,
            opponent_def_rating, vs_opponent_avg, vs_opponent_games,
            prediction_date, player_id, team_id, opponent_id
        FROM predictions
        WHERE game_date = ?
        ORDER BY dfs_score DESC
    """
    return pd.read_sql_query(query, conn, params=[game_date])


def display_team_predictions(team_preds: pd.DataFrame):
    """Display predictions for one team in a formatted table."""
    if team_preds.empty:
        st.caption("No predictions available")
        return

    # Format for display
    display_df = team_preds[[
        'player_name', 'projected_ppg', 'proj_confidence',
        'proj_floor', 'proj_ceiling', 'dfs_score', 'dfs_grade'
    ]].copy()

    display_df.columns = ['Player', 'Proj PPG', 'Confidence', 'Floor', 'Ceiling', 'DFS Score', 'Grade']

    # Format numeric columns
    display_df['Proj PPG'] = display_df['Proj PPG'].map(lambda x: f"{x:.1f}")
    display_df['Confidence'] = display_df['Confidence'].map(lambda x: f"{x:.0%}")
    display_df['Floor'] = display_df['Floor'].map(lambda x: f"{x:.1f}")
    display_df['Ceiling'] = display_df['Ceiling'].map(lambda x: f"{x:.1f}")
    display_df['DFS Score'] = display_df['DFS Score'].map(lambda x: f"{x:.1f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


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


def _calculate_tournament_dfs_score(
    player_season_avg: float,
    player_projection: float,
    proj_ceiling: float,
    recent_avg_5: float | None,
    matchup_rating: str,
    opp_def_rating: float | None,
    opp_team_id: int | None,
    league_avg_def_rating: float = 112.0,
    injury_adjusted: bool = False,
    projection_boost: float = 0.0,
    def_ppm_df: Optional[pd.DataFrame] = None,  # PPM Integration
    player_position: str = "",  # Position-specific PPM
    game_date: str = "",  # For season progress calculation
    conn: Optional[sqlite3.Connection] = None,  # Database connection for position PPM
    league_avg_ppm: float = 0.462,  # League average PPM
) -> tuple[float, str, str]:
    """
    Calculate Tournament DFS Score optimized for winner-take-all contests.

    Tournament strategy emphasizes:
    1. CEILING (explosive potential) over projection
    2. Recent hot streaks (momentum = higher boom probability)
    3. Opponent defense variance (allows big games?)
    4. Upside/variance (boom/bust is GOOD, not bad)

    Returns higher scores for:
    - High ceilings (40-50+) even if inconsistent
    - Players on hot streaks (last 5 > season avg)
    - Matchups vs high-variance defenses
    - Boom/bust profiles over consistent mid-range players

    Args:
        player_season_avg: Season average PPG
        player_projection: Today's projection
        proj_ceiling: 90th percentile ceiling
        recent_avg_5: Last 5 games average (for hot streak detection)
        matchup_rating: Player-specific matchup quality
        opp_def_rating: Opponent defensive rating
        opp_team_id: Opponent team ID (for variance lookup)
        league_avg_def_rating: League average

    Returns:
        (score 0-100, grade, explanation)
    """
    factors = []

    # 1. CEILING BASE SCORE (0-50 range) - Most important factor
    # Tournament winners need 40-50+ ceilings, not 28 PPG projections
    if proj_ceiling >= 50:
        ceiling_base = 50
        factors.append("elite 50+ ceiling")
    elif proj_ceiling >= 45:
        ceiling_base = 46
        factors.append("monster 45+ ceiling")
    elif proj_ceiling >= 40:
        ceiling_base = 42
        factors.append("huge 40+ ceiling")
    elif proj_ceiling >= 35:
        ceiling_base = 35
        factors.append("strong 35+ ceiling")
    elif proj_ceiling >= 30:
        ceiling_base = 25
        factors.append("decent 30+ ceiling")
    else:
        ceiling_base = max(0, (proj_ceiling - 15) * 1.5)
        factors.append("limited ceiling")

    # 2. HOT STREAK BONUS (0-20 points) - Recent form heavily weighted
    # Players on hot streaks are more likely to have explosive games
    hot_streak_bonus = 0
    if recent_avg_5 is not None and player_season_avg > 0:
        hot_streak_ratio = recent_avg_5 / player_season_avg

        if hot_streak_ratio >= 1.25:
            # On FIRE (25%+ above season avg)
            hot_streak_bonus = 20
            factors.append("ON FIRE (L5 +25%)")
        elif hot_streak_ratio >= 1.15:
            # Very hot (15-25% above avg)
            hot_streak_bonus = 15
            factors.append("very hot L5")
        elif hot_streak_ratio >= 1.05:
            # Warm (5-15% above avg)
            hot_streak_bonus = 10
            factors.append("trending up")
        elif hot_streak_ratio <= 0.85:
            # Cold (15%+ below avg)
            hot_streak_bonus = -5
            factors.append("cold streak")
        elif hot_streak_ratio <= 0.95:
            # Slightly cool
            hot_streak_bonus = 0
            factors.append("slightly cool")
        else:
            # Steady
            hot_streak_bonus = 5
            factors.append("steady")

    # 3. OPPONENT DEFENSE VARIANCE BONUS (0-15 points)
    # High-variance defenses allow explosive games
    # ENHANCED: Position-specific PPM ceiling factor for more accurate targeting
    variance_bonus = 0
    position_exploit_bonus = 0  # NEW: Component 7 (calculated here for context)

    if opp_team_id is not None:
        # Priority 1: Use position-specific PPM ceiling factor (most accurate)
        if def_ppm_df is not None and player_position and player_position in ['Guard', 'Forward', 'Center'] and conn is not None:
            try:
                # Get team overall PPM for blending
                opp_def_ppm_row = def_ppm_df[def_ppm_df['team_id'] == opp_team_id]
                if not opp_def_ppm_row.empty:
                    team_overall_def_ppm = opp_def_ppm_row['avg_def_ppm'].iloc[0]

                    # Get position-specific blended matchup data
                    blended_matchup = position_ppm_stats.get_blended_def_ppm_for_matchup(
                        conn=conn,
                        def_team_id=opp_team_id,
                        off_player_position=player_position,
                        team_overall_def_ppm=team_overall_def_ppm,
                        season='2025-26',
                        current_date=game_date if game_date else None
                    )

                    # Get position-specific data for ceiling factor calculation
                    position_data = position_ppm_stats.get_position_defensive_ppm(
                        conn, opp_team_id, player_position, season='2025-26'
                    )

                    if position_data and position_data['games_count'] >= 5:
                        # Calculate position-specific ceiling factor
                        position_ceiling_factor = position_data['p90_ppm'] / position_data['avg_def_ppm_vs_position']

                        # Variance bonus based on position-specific ceiling factor
                        if position_ceiling_factor >= 1.15:
                            variance_bonus = 15
                            factors.append(f"elite {player_position} variance")
                        elif position_ceiling_factor >= 1.12:
                            variance_bonus = 10
                            factors.append(f"high {player_position} variance")
                        elif position_ceiling_factor >= 1.10:
                            variance_bonus = 5
                            factors.append("above avg position variance")

                        # Component 7: Position Exploit Bonus (0-10 points)
                        # Severe position weaknesses are GPP gold
                        if blended_matchup['exploit_detected']:
                            if blended_matchup['exploit_severity'] == 'severe':
                                position_exploit_bonus = 10
                                factors.append("🎯🔥 SEVERE position exploit")
                            elif blended_matchup['exploit_severity'] == 'moderate':
                                position_exploit_bonus = 7
                                factors.append("🎯 moderate position exploit")
                            elif blended_matchup['exploit_severity'] == 'minor':
                                position_exploit_bonus = 4
                                factors.append("🎯 minor position exploit")
                    else:
                        # Not enough position data, fall back to team ceiling factor
                        team_ceiling_factor = opp_def_ppm_row['ceiling_factor'].iloc[0]
                        if team_ceiling_factor >= 1.15:
                            variance_bonus = 15
                            factors.append("elite team PPM variance")
                        elif team_ceiling_factor >= 1.12:
                            variance_bonus = 10
                            factors.append("high team PPM variance")
                        elif team_ceiling_factor >= 1.10:
                            variance_bonus = 5
                            factors.append("above avg team variance")
            except Exception:
                # Fall through to team-level or old method
                pass

        # Priority 2: Use team-level PPM ceiling factor
        if variance_bonus == 0 and def_ppm_df is not None:
            try:
                opp_def_ppm_row = def_ppm_df[def_ppm_df['team_id'] == opp_team_id]
                if not opp_def_ppm_row.empty:
                    ppm_ceiling_factor = opp_def_ppm_row['ceiling_factor'].iloc[0]

                    if ppm_ceiling_factor >= 1.15:
                        variance_bonus = 15
                        factors.append("elite PPM variance")
                    elif ppm_ceiling_factor >= 1.12:
                        variance_bonus = 10
                        factors.append("high PPM variance")
                    elif ppm_ceiling_factor >= 1.10:
                        variance_bonus = 5
                        factors.append("above avg variance")
            except:
                pass

        # Priority 3: Fallback to old points-per-game ceiling factor
        if variance_bonus == 0:
            try:
                ceiling_factor = get_opponent_defense_ceiling_factor(opp_team_id)
                if ceiling_factor >= 1.15:
                    variance_bonus = 15
                    factors.append("elite variance matchup")
                elif ceiling_factor >= 1.12:
                    variance_bonus = 10
                    factors.append("good variance matchup")
                elif ceiling_factor >= 1.10:
                    variance_bonus = 5
                    factors.append("above avg variance")
            except:
                pass

    # 4. MATCHUP HISTORY BONUS (0-10 points)
    # Less important than cash games but still relevant
    matchup_adjustments = {
        "Excellent": 10,
        "Good": 7,
        "Neutral": 3,
        "Difficult": 0,
        "Avoid": -5,
    }
    matchup_bonus = matchup_adjustments.get(matchup_rating, 3)

    if matchup_rating == "Excellent":
        factors.append("excels vs this team")
    elif matchup_rating == "Good":
        factors.append("favorable history")

    # 5. DEFENSE QUALITY ADJUSTMENT (±5 points)
    # Weaker defense helps ceiling, but variance is more important
    defense_adjustment = 0
    if opp_def_rating is not None:
        if opp_def_rating >= 118:
            defense_adjustment = 5
            factors.append("weak defense")
        elif opp_def_rating >= 114:
            defense_adjustment = 3
        elif opp_def_rating <= 106:
            defense_adjustment = -3
            factors.append("elite defense")

    # 6. INJURY BENEFICIARY BONUS (0-12 points)
    # Players who get boosted usage/minutes due to teammate injuries
    # This is CRITICAL for tournaments - low ownership + spike potential
    injury_bonus = 0
    if injury_adjusted and projection_boost > 0:
        # Projection boost indicates teammate(s) are OUT
        # Scale bonus based on size of boost
        if projection_boost >= 5.0:
            # Massive boost (5+ PPG) - Star player out
            injury_bonus = 12
            factors.append("🔥 STAR OUT (major usage)")
        elif projection_boost >= 3.0:
            # Significant boost (3-5 PPG) - Key rotation player out
            injury_bonus = 8
            factors.append("key teammate out")
        elif projection_boost >= 1.5:
            # Moderate boost (1.5-3 PPG) - Rotation player out
            injury_bonus = 5
            factors.append("teammate out")

        # Injury beneficiaries are often LOW OWNED (field can't react)
        # This is PURE GOLD for tournaments - differentiation + upside

    # Calculate final score
    final_score = (
        ceiling_base +
        hot_streak_bonus +
        variance_bonus +
        matchup_bonus +
        defense_adjustment +
        injury_bonus +
        position_exploit_bonus  # Component 7: Position Exploit Bonus (0-10 points)
    )
    final_score = max(0, min(110, final_score))  # Raised cap to 110 to allow for position exploit bonus

    # Build explanation
    explanation = ", ".join(factors[:4])  # Top 4 factors

    # Grade with tournament-specific thresholds
    if final_score >= 85:
        grade = "🔥🔥 GPP Lock"
        explanation = f"GPP Lock: {explanation}"
    elif final_score >= 75:
        grade = "🔥 Core Play"
        explanation = f"Core play: {explanation}"
    elif final_score >= 65:
        grade = "⭐ Strong"
        explanation = f"Strong: {explanation}"
    elif final_score >= 55:
        grade = "✓ Playable"
        explanation = f"Playable: {explanation}"
    elif final_score >= 45:
        grade = "⚠️ Punt/Fade"
        explanation = f"Punt: {explanation}"
    else:
        grade = "❌ Avoid"
        explanation = f"Avoid: {explanation}"

    return (final_score, grade, explanation)


def calculate_daily_pick_score(
    player_season_avg: float,
    player_projection: float,
    projection_confidence: float,
    matchup_rating: str,
    opp_def_rating: float | None,
    league_avg_def_rating: float = 112.0,
    # NEW: Tournament-specific parameters
    proj_ceiling: float | None = None,
    recent_avg_5: float | None = None,
    opp_team_id: int | None = None,
    tournament_mode: bool = False,
    injury_adjusted: bool = False,
    projection_boost: float = 0.0,
    def_ppm_df: Optional[pd.DataFrame] = None,  # PPM Integration
    player_position: str = "",  # Position-specific PPM
    game_date: str = "",  # Season progress calculation
    conn: Optional[sqlite3.Connection] = None,  # Database connection
    league_avg_ppm: float = 0.462,  # League average PPM
) -> tuple[float, str, str]:
    """
    Calculate unified DFS Score (0-100) combining all analytics.

    This score answers: "How good is this player as a pick TODAY?"

    Two modes:
    - Cash game mode (default): Optimizes for projection accuracy and consistency
    - Tournament mode: Optimizes for ceiling, recent hot streaks, and opponent variance

    Args:
        player_season_avg: Player's season average PPG
        player_projection: Smart projection for today
        projection_confidence: Confidence in projection (0-1)
        matchup_rating: Player-specific matchup quality
        opp_def_rating: Opponent's defensive rating
        league_avg_def_rating: League average for comparison
        proj_ceiling: Player's 90th percentile ceiling (for tournament mode)
        recent_avg_5: Last 5 games average (for tournament mode hot streak detection)
        opp_team_id: Opponent team ID (for tournament mode variance bonus)
        tournament_mode: If True, optimize for ceiling/variance instead of consistency

    Returns:
        (score 0-100, grade, explanation)
    """
    # TOURNAMENT MODE: Completely different scoring logic
    if tournament_mode and proj_ceiling is not None:
        return _calculate_tournament_dfs_score(
            player_season_avg=player_season_avg,
            player_projection=player_projection,
            proj_ceiling=proj_ceiling,
            recent_avg_5=recent_avg_5,
            matchup_rating=matchup_rating,
            opp_def_rating=opp_def_rating,
            opp_team_id=opp_team_id,
            league_avg_def_rating=league_avg_def_rating,
            injury_adjusted=injury_adjusted,
            projection_boost=projection_boost,
            def_ppm_df=def_ppm_df,  # PPM Integration
            player_position=player_position,  # Position-specific PPM
            game_date=game_date,  # Season progress
            conn=conn,  # Database connection
            league_avg_ppm=league_avg_ppm,  # League average PPM
        )

    # CASH GAME MODE (default - original logic)
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


# Cache for opponent defense variance (updated once per season)
_opponent_defense_variance_cache = {}

def get_opponent_defense_ceiling_factor(opp_team_id: int, season: str = '2024-25') -> float:
    """
    Calculate opponent's defense variance ceiling factor for tournament play.

    Ceiling factor = (90th percentile points allowed) / (avg points allowed)

    Teams with high ceiling factors (1.15+) allow explosive scoring games more often,
    making them ideal matchups for tournament strategy where you need 40-50+ point ceilings.

    Args:
        opp_team_id: Opponent team ID
        season: NBA season (default: 2024-25)

    Returns:
        Ceiling factor (1.0-1.20), where:
        - 1.15+ = Elite ceiling spot (allow 15%+ more in big games)
        - 1.12-1.15 = Good ceiling matchup
        - 1.10-1.12 = Above average variance
        - <1.10 = Average/tight defense
    """
    cache_key = f"{season}_{opp_team_id}"

    # Check cache first
    if cache_key in _opponent_defense_variance_cache:
        return _opponent_defense_variance_cache[cache_key]

    try:
        # Calculate defense variance from game logs
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get points scored against this team
        cursor.execute("""
            SELECT pts
            FROM team_game_logs
            WHERE opp_team_id = ?
              AND season = ?
        """, (opp_team_id, season))

        pts_against = [row[0] for row in cursor.fetchall() if row[0] is not None]
        conn.close()

        if len(pts_against) < 15:
            # Not enough games, return neutral factor
            return 1.0

        # Calculate ceiling factor
        import numpy as np
        avg_allowed = np.mean(pts_against)
        p90_allowed = np.percentile(pts_against, 90)

        ceiling_factor = p90_allowed / avg_allowed if avg_allowed > 0 else 1.0

        # Cache result
        _opponent_defense_variance_cache[cache_key] = ceiling_factor

        return ceiling_factor

    except Exception:
        # If calculation fails, return neutral factor
        return 1.0


def get_opponent_defense_grade(opp_team_id: int, opp_def_rating: float | None, season: str = '2024-25') -> str:
    """
    Generate opponent defense grade for tournament strategy.

    Combines defense quality (def rating) with variance (ceiling factor) to categorize matchups.

    Args:
        opp_team_id: Opponent team ID
        opp_def_rating: Opponent's defensive rating (points allowed per 100 possessions)
        season: NBA season (default: 2024-25)

    Returns:
        Grade string indicating matchup quality:
        - "A+ Elite Spot" = Bad defense + high variance (dream matchup)
        - "A Smash Spot" = Bad defense + good variance
        - "A- Variance Play" = Good defense but high variance (boom/bust)
        - "B+ Good Matchup" = Average defense + good variance
        - "B Solid" = Good matchup overall
        - "C+ Playable" = Slightly favorable
        - "C Neutral" = Average matchup
        - "D Tough" = Good defense, low variance
        - "F Avoid" = Elite defense, low variance
    """
    # Get ceiling factor (variance)
    ceiling_factor = get_opponent_defense_ceiling_factor(opp_team_id, season)

    # Default league average if no def rating
    if opp_def_rating is None:
        opp_def_rating = 112.0

    # Categorize defense quality (lower is better defense)
    # League average ~112, Elite defense ~105, Poor defense ~118+
    if opp_def_rating >= 118:
        def_quality = "bad"  # Poor defense
    elif opp_def_rating >= 114:
        def_quality = "below_avg"  # Below average
    elif opp_def_rating >= 110:
        def_quality = "avg"  # Average
    elif opp_def_rating >= 107:
        def_quality = "good"  # Good defense
    else:
        def_quality = "elite"  # Elite defense

    # Categorize variance (ceiling factor)
    if ceiling_factor >= 1.15:
        variance = "elite"  # Elite ceiling spot
    elif ceiling_factor >= 1.12:
        variance = "good"  # Good variance
    elif ceiling_factor >= 1.10:
        variance = "above_avg"  # Above average
    else:
        variance = "low"  # Tight/consistent defense

    # Combine into tournament grade
    grade_matrix = {
        # Bad defense matchups
        ("bad", "elite"): "A+ Elite Spot",
        ("bad", "good"): "A Smash Spot",
        ("bad", "above_avg"): "A- Great Spot",
        ("bad", "low"): "B+ Good Matchup",

        # Below average defense
        ("below_avg", "elite"): "A Elite Variance",
        ("below_avg", "good"): "A- Smash Spot",
        ("below_avg", "above_avg"): "B+ Good Matchup",
        ("below_avg", "low"): "B Solid",

        # Average defense
        ("avg", "elite"): "A- Variance Play",
        ("avg", "good"): "B+ Boom/Bust",
        ("avg", "above_avg"): "B Playable",
        ("avg", "low"): "C+ Neutral+",

        # Good defense (harder matchup but variance helps)
        ("good", "elite"): "B Variance Play",
        ("good", "good"): "B- Boom/Bust",
        ("good", "above_avg"): "C+ Playable",
        ("good", "low"): "C Tough",

        # Elite defense (avoid unless high variance)
        ("elite", "elite"): "B- High Variance",
        ("elite", "good"): "C Boom/Bust",
        ("elite", "above_avg"): "C- Risky",
        ("elite", "low"): "D Avoid",
    }

    return grade_matrix.get((def_quality, variance), "C Neutral")


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
    opp_team_id: Optional[int] = None,  # NEW: For defense variance ceiling boost
    # PPM Integration Parameters
    player_name: str = "",  # For PPM consistency lookup
    opponent_id: Optional[int] = None,  # For PPM defense lookup
    def_ppm_df: Optional[pd.DataFrame] = None,  # Defensive PPM dataframe
    league_avg_ppm: float = 0.462,  # League average PPM
    conn: Optional[sqlite3.Connection] = None,  # For player PPM lookup
    player_position: str = "",  # Player position (Guard/Forward/Center) for position-specific PPM
    game_date: str = "",  # Game date for season progress calculation
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
        # Updated weighting based on predictive value analysis (275 predictions)
        # Last 5 games has superior predictive value:
        # - MAE: 7.57 (Last 5) vs 7.93 (Last 3) - 4.8% more accurate
        # - Correlation: 0.451 (Last 5) vs 0.400 (Last 3) - 12.9% stronger
        # - Win rate: 57.8% (Last 5) vs 41.1% (Last 3)
        components["recent"] = (recent_avg_5 * 0.6) + (recent_avg_3 * 0.4)
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

    elif def_ppm_df is not None and opponent_id is not None:
        # ENHANCED: Position-Specific PPM with Bayesian Blending
        # Uses blended team + position-specific defensive PPM for more accurate matchup analysis
        opp_def_ppm_row = def_ppm_df[def_ppm_df['team_id'] == opponent_id]
        if not opp_def_ppm_row.empty:
            team_def_ppm = opp_def_ppm_row['avg_def_ppm'].iloc[0]

            # Try to get blended position-specific PPM if player position is known
            if player_position and player_position in ['Guard', 'Forward', 'Center'] and conn is not None:
                try:
                    blended_matchup = position_ppm_stats.get_blended_def_ppm_for_matchup(
                        conn=conn,
                        def_team_id=opponent_id,
                        off_player_position=player_position,
                        team_overall_def_ppm=team_def_ppm,
                        season='2025-26',
                        current_date=game_date if game_date else None
                    )

                    # Use blended PPM (combines team and position-specific data)
                    opp_def_ppm = blended_matchup['blended_def_ppm']

                    # Add position exploit indicator if detected
                    if blended_matchup['exploit_detected']:
                        if blended_matchup['exploit_severity'] == 'severe':
                            analytics_used.append("🎯🔥")  # Severe position exploit
                        else:
                            analytics_used.append("🎯")  # Moderate position exploit
                except Exception:
                    # Fallback to team overall PPM if position-specific fails
                    opp_def_ppm = team_def_ppm
            else:
                # Use team overall PPM if position unknown
                opp_def_ppm = team_def_ppm

            # Calculate PPM boost factor: higher def PPM allowed = easier matchup
            ppm_boost_factor = opp_def_ppm / league_avg_ppm

            # Cap at ±20% (slightly higher than old ±15% to account for PPM variance)
            ppm_boost_factor = max(0.80, min(1.20, ppm_boost_factor))

            components["def_ppm_quality"] = season_avg * ppm_boost_factor
            weights["def_ppm_quality"] = 0.15  # Increase weight from 10% to 15%
        else:
            # Fallback to old def_rating if PPM data missing
            if opp_def_rating is not None:
                def_adjustment = (league_avg_def_rating - opp_def_rating) / league_avg_def_rating
                def_adjustment = max(-0.15, min(0.15, def_adjustment))
                components["def_quality"] = season_avg * (1 + def_adjustment)
                weights["def_quality"] = 0.10
            else:
                components["def_quality"] = season_avg
                weights["def_quality"] = 0.0

    elif opp_def_rating is not None:
        # Fallback to defensive rating if PPM not available
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

    # Add momentum indicator if breakout detected (will be appended after momentum check)

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

    # MOMENTUM MODIFIER: Catch breakout performances (Murray 52, Pritchard 42, Vassell 35)
    # Only triggers when ALL conditions are met (conservative guardrail)
    momentum_bonus = 0.0
    momentum_applied = False
    if recent_avg_3 is not None and season_avg > 0 and season_avg_minutes is not None:
        # Condition 1: Recent surge (last 3 games ≥ +15% vs season)
        recent_surge = (recent_avg_3 - season_avg) / season_avg

        # Condition 2: Minutes stable or rising (≥90% of season avg)
        minutes_stable = (avg_minutes_last5 is not None and
                         avg_minutes_last5 >= season_avg_minutes * 0.90)

        # Condition 3: Usage stable (within ±2% if available)
        usage_stable = True  # Default to True if no usage data
        if avg_usg_last5 is not None and usage_pct is not None and usage_pct > 0:
            usage_change = abs(avg_usg_last5 - usage_pct) / usage_pct
            usage_stable = usage_change <= 0.02

        # Apply bonus if all conditions met
        if recent_surge >= 0.15 and minutes_stable and usage_stable:
            # Scale bonus: +15% surge → 5% bonus, +30% surge → 10% bonus, +50% surge → 12% bonus
            # Hard capped at 12%
            if recent_surge >= 0.50:
                momentum_bonus = 0.12
            elif recent_surge >= 0.30:
                momentum_bonus = 0.10
            elif recent_surge >= 0.20:
                momentum_bonus = 0.07
            else:
                momentum_bonus = 0.05

            projection = projection * (1 + momentum_bonus)
            momentum_applied = True
            analytics_indicators += "🔥"  # Momentum boost indicator

    # OPPONENT INJURY IMPACT: Easier matchups when opponent's defenders are OUT
    # When opponent's primary defender or rim protector is absent, our player faces
    # easier matchup → increase ceiling & projection
    opponent_injury_boost = 0.0
    opponent_injury_ceiling_boost = 0.0
    opponent_injury_detected = False

    if opp_team_id is not None and conn is not None and game_date:
        try:
            import opponent_injury_impact as oii

            injury_impact = oii.calculate_opponent_injury_impact(
                conn,
                player_position=player_position if player_position else None,
                opponent_team_id=opp_team_id,
                game_date=game_date,
                season='2025-26'
            )

            if injury_impact['has_significant_injuries']:
                # Apply boosts from opponent injuries
                opponent_injury_boost = injury_impact['projection_boost_pct']
                opponent_injury_ceiling_boost = injury_impact['ceiling_boost_pct']
                opponent_injury_detected = True

                # Apply projection boost
                if opponent_injury_boost > 0:
                    projection = projection * (1 + opponent_injury_boost)

                # Add injury indicator
                analytics_indicators += "🚑"  # Opponent injury boost indicator

        except Exception:
            # If opponent injury calculation fails, continue without boost
            pass

    # FIX 2: Add regression to mean for high scorers (22+ PPG)
    # High scorers are over-projected by ~3.5 PPG on average
    if projection >= 22.0:
        regression_factor = 0.93  # Reduce by 7%
        projection = projection * regression_factor

    # Calculate VARIANCE (uncertainty) metric - INVERTED from old "confidence"
    # Higher variance = less certain, wider ranges
    # Lower variance = more certain, tighter ranges
    variance_score = 0.0

    # 1. Base variance from limited season data
    # Start with moderate uncertainty
    base_variance = 0.25
    variance_score += base_variance

    # 2. Matchup-specific data REDUCES variance (more data = less uncertainty)
    if vs_opp_team_games >= 5:
        # Strong team history - VERY CERTAIN (low variance)
        variance_score -= 0.15
    elif vs_opp_team_games >= 3:
        # Good team history - CERTAIN (low variance)
        variance_score -= 0.10
    elif vs_opp_team_games >= 2:
        # Some team history - moderately certain
        variance_score -= 0.05
    elif vs_defense_style_games >= 8:
        # Extensive style data - decent fallback
        variance_score -= 0.04
    elif vs_defense_style_games >= 5:
        # Good style data - moderate fallback
        variance_score -= 0.02
    elif vs_defense_style_games >= 3:
        # Some style data - weak fallback
        variance_score -= 0.01

    # 3. Recent form VOLATILITY increases variance
    if recent_avg_3 is not None and recent_avg_5 is not None:
        # Check volatility of recent form
        volatility = abs(recent_avg_3 - recent_avg_5) / max(season_avg, 1)
        if volatility > 0.25:
            # High volatility - LESS CERTAIN (high variance)
            variance_score += 0.15
        elif volatility > 0.15:
            # Moderate volatility - uncertain
            variance_score += 0.08
        else:
            # Consistent recent form - MORE CERTAIN (low variance)
            variance_score -= 0.05

    # 4. Opponent data availability REDUCES variance
    if opp_def_rating is not None and opp_pace is not None:
        # Have both def rating and pace - more certain
        variance_score -= 0.05
    elif opp_def_rating is not None:
        # Only def rating - slightly more certain
        variance_score -= 0.02

    # 5. Enhanced analytics REDUCE variance (more data = more certain)
    if opponent_correlation is not None and opponent_correlation.games_vs_opponent >= 2:
        # High-quality player-specific matchup data
        variance_score -= 0.08
    if pace_split is not None and pace_split.games_played >= 2:
        # Player-specific pace performance data
        variance_score -= 0.04
    if defense_quality_split is not None and defense_quality_split.games_played >= 2:
        # Player-specific defense quality performance data
        variance_score -= 0.04

    # Keep variance in reasonable bounds [0.05, 0.50]
    variance_score = max(0.05, min(0.50, variance_score))

    # FIX 3: Convert variance to "confidence" for display (inverted for user-friendliness)
    # High variance (0.5) = Low confidence (50%)
    # Low variance (0.05) = High confidence (95%)
    confidence_score = 1.0 - variance_score

    # FIX 1: Calculate floor and ceiling with WIDER ranges
    # Use a BASE range that reflects inherent game variance
    # Then adjust slightly based on data quality (±20% based on uncertainty)
    # This prevents ranges from collapsing when confidence is high
    uncertainty = 1.0 - confidence_score  # High confidence → low uncertainty (0.05 to 0.95)

    # FLOOR: Conservative estimate (bad game scenario)
    # Base floor: 14% below projection (inherent game variance + off-night risk)
    # Uncertainty adjustment: scale between 80% and 120% of base
    # UPDATED: Widened from 9% to 14% based on actual prediction log analysis
    # showing only 13% hit rate with narrower ranges
    floor_base_interval = season_avg * 0.14
    floor_uncertainty_factor = 0.8 + (0.4 * uncertainty)
    floor_interval = floor_base_interval * floor_uncertainty_factor
    floor = max(0, projection - floor_interval)

    # CEILING: 90th percentile outcome (explosive game scenario)
    # For tournament play, ceiling must represent what happens when everything goes right:
    # - Hot shooting night (3-4 extra made shots = +6-12 points)
    # - Blowout/Overtime (extra minutes = +15-25% production)
    # - Weak defense matchup (already partially captured in projection)
    # - High usage game (injury to teammate, foul trouble for others)
    #
    # Statistical analysis of NBA player game logs shows:
    # - Top 10% of games for star players (25+ season PPG): +60-80% above average
    # - Top 10% of games for role players (15-20 PPG): +50-70% above average
    # - Top 10% of games for bench players (<15 PPG): +80-120% above average
    #
    # Formula: Base ceiling multiplier adjusted by player tier and uncertainty
    # UPDATED: Increased multipliers by 15-20% based on prediction log analysis
    # showing under-projection of breakout performances (only 13% hit rate)
    if season_avg >= 25:
        # Star players: 70-95% upside (e.g., 28 PPG → 48-55 ceiling)
        ceiling_base_multiplier = 0.70
        ceiling_uncertainty_bonus = 0.25 * uncertainty  # More uncertainty = higher ceiling
    elif season_avg >= 20:
        # High-volume scorers: 65-90% upside (e.g., 22 PPG → 36-42 ceiling)
        ceiling_base_multiplier = 0.65
        ceiling_uncertainty_bonus = 0.25 * uncertainty
    elif season_avg >= 15:
        # Mid-tier players: 60-85% upside (e.g., 17 PPG → 27-31 ceiling)
        ceiling_base_multiplier = 0.60
        ceiling_uncertainty_bonus = 0.25 * uncertainty
    else:
        # Role/bench players: 85-120% upside (e.g., 10 PPG → 18-22 ceiling)
        ceiling_base_multiplier = 0.85
        ceiling_uncertainty_bonus = 0.35 * uncertainty

    ceiling_multiplier = ceiling_base_multiplier + ceiling_uncertainty_bonus

    # PPM CONSISTENCY ADJUSTMENT: Adjust floor/ceiling by player PPM consistency
    # High consistency players = tighter ranges (smaller intervals)
    # Low consistency players = wider ranges (larger intervals)
    floor_interval_multiplier = 1.0  # Default: no adjustment
    ceiling_multiplier_adjustment = 1.0  # Default: no adjustment

    if conn is not None and player_name:
        try:
            player_ppm_data = ppm_stats.get_player_ppm_stats(conn, player_name, season='2025-26')
            if player_ppm_data:
                consistency_score = player_ppm_data['consistency_score']

                # High consistency = tighter range (A/A+ grade: 75+)
                # Reduce interval size by 20%, reduce ceiling multiplier by 10%
                if consistency_score >= 75:
                    floor_interval_multiplier = 0.80  # Tighter floor (smaller interval)
                    ceiling_multiplier_adjustment = 0.90  # Lower ceiling multiplier
                # Low consistency = wider range (D grade: <55)
                # Increase interval size by 25%, increase ceiling multiplier by 15%
                elif consistency_score < 55:
                    floor_interval_multiplier = 1.25  # Wider floor (larger interval)
                    ceiling_multiplier_adjustment = 1.15  # Higher ceiling multiplier
                # Average consistency (B/C grade: 55-74) - no adjustment needed
        except Exception:
            # If PPM lookup fails, use defaults (no adjustment)
            pass

    # Apply PPM consistency adjustment to floor (recalculate with adjusted interval)
    floor = max(0, projection - (floor_interval * floor_interval_multiplier))

    # Apply PPM consistency adjustment to ceiling multiplier
    ceiling_multiplier = ceiling_multiplier * ceiling_multiplier_adjustment

    # TOURNAMENT ENHANCEMENT: Opponent defense variance boost
    # Teams with high defensive variance allow explosive scoring games more often
    # This is critical for tournament strategy - you want matchups that can produce 40-50+ ceilings
    defense_variance_boost = 0.0
    if opp_team_id is not None:
        try:
            ceiling_factor = get_opponent_defense_ceiling_factor(opp_team_id)

            # Apply boost based on ceiling factor thresholds
            if ceiling_factor >= 1.15:
                # Elite ceiling spot (OKC, POR, DAL) - allow 15%+ more in big games
                defense_variance_boost = 0.10  # +10% ceiling boost
            elif ceiling_factor >= 1.12:
                # Good ceiling matchup - allow 12-15% more in big games
                defense_variance_boost = 0.06  # +6% ceiling boost
            elif ceiling_factor >= 1.10:
                # Above average variance - allow 10-12% more in big games
                defense_variance_boost = 0.03  # +3% ceiling boost
            # else: tight defense, no boost

        except Exception:
            # If calculation fails, no boost
            pass

    # Apply combined multiplier with defense variance boost + opponent injury boost
    final_ceiling_multiplier = ceiling_multiplier + defense_variance_boost + opponent_injury_ceiling_boost
    ceiling = projection * (1 + final_ceiling_multiplier)

    # CEILING CONFIDENCE SCORE (DFS Overlay - doesn't touch mean projection)
    # Ranks ceiling quality for tournament play (0-100 scale)
    ceiling_confidence = 50.0  # Base neutral score

    # Factor 1: Pace environment (+/- 15 points)
    if opp_pace is not None and league_avg_pace > 0:
        pace_factor = ((opp_pace - league_avg_pace) / league_avg_pace) * 100
        ceiling_confidence += min(15, max(-15, pace_factor))

    # Factor 2: Correlation strength (+/- 20 points)
    if opponent_correlation is not None and opponent_correlation.games_vs_opponent >= 2:
        # Strong positive matchup = high ceiling confidence
        if opponent_correlation.matchup_score >= 60:
            ceiling_confidence += 20
        elif opponent_correlation.matchup_score >= 55:
            ceiling_confidence += 12
        elif opponent_correlation.matchup_score <= 40:
            ceiling_confidence -= 20
        elif opponent_correlation.matchup_score <= 45:
            ceiling_confidence -= 12

    # Factor 3: Opponent ceiling allowance (+/- 15 points)
    # ENHANCED: Use detailed ceiling analytics (35+/40+/45+ game rates)
    if opp_team_id is not None:
        try:
            import opponent_ceiling_analytics as oca
            ceiling_profile = oca.get_team_ceiling_profile(
                conn,
                opp_team_id,
                season='2025-26',
                min_games=5
            )

            if ceiling_profile:
                volatility = ceiling_profile['ceiling_volatility']
                # Elite ceiling spots (70+): +15 pts
                # High ceiling (55-70): +10 pts
                # Average (40-55): 0 pts
                # Low ceiling (25-40): -8 pts
                # Suppressors (<25): -15 pts
                if volatility >= 70:
                    ceiling_confidence += 15
                elif volatility >= 55:
                    ceiling_confidence += 10
                elif volatility >= 40:
                    pass  # Neutral
                elif volatility >= 25:
                    ceiling_confidence -= 8
                else:
                    ceiling_confidence -= 15
            else:
                # Fallback to old method if no ceiling profile data
                ceiling_factor = get_opponent_defense_ceiling_factor(opp_team_id)
                if ceiling_factor >= 1.15:
                    ceiling_confidence += 10
                elif ceiling_factor >= 1.12:
                    ceiling_confidence += 6
                elif ceiling_factor >= 1.10:
                    ceiling_confidence += 3
                elif ceiling_factor <= 0.90:
                    ceiling_confidence -= 10
        except Exception:
            pass

    # Factor 4: Recent efficiency spike (+/- 10 points)
    if recent_avg_3 is not None and season_avg > 0:
        efficiency_trend = (recent_avg_3 - season_avg) / season_avg
        if efficiency_trend >= 0.20:
            ceiling_confidence += 10  # Hot streak
        elif efficiency_trend >= 0.10:
            ceiling_confidence += 5
        elif efficiency_trend <= -0.20:
            ceiling_confidence -= 10  # Cold streak

    # Factor 5: Player tier adjustment (+/- 5 points)
    # High-usage stars have more reliable ceilings
    if season_avg >= 25:
        ceiling_confidence += 5
    elif season_avg <= 10:
        ceiling_confidence -= 5  # Boom-bust risk

    # Factor 6: Opponent injury impact (+/- 12 points)
    # Significant opponent injuries = easier defensive matchup = higher ceiling potential
    if opponent_injury_detected and opponent_injury_ceiling_boost > 0:
        # Scale based on magnitude of boost
        if opponent_injury_ceiling_boost >= 0.10:  # 10%+ ceiling boost
            ceiling_confidence += 12  # Major defensive gap
        elif opponent_injury_ceiling_boost >= 0.06:  # 6-10% boost
            ceiling_confidence += 8
        elif opponent_injury_ceiling_boost >= 0.03:  # 3-6% boost
            ceiling_confidence += 4

    # Clamp to 0-100 scale
    ceiling_confidence = max(0, min(100, ceiling_confidence))

    # Build breakdown for transparency
    breakdown = {
        "projection": projection,
        "confidence": confidence_score,
        "floor": floor,
        "ceiling": ceiling,
        "ceiling_confidence": ceiling_confidence,  # NEW: DFS overlay score
        "momentum_bonus": momentum_bonus if momentum_applied else 0.0,  # Track momentum
        "opponent_injury_detected": opponent_injury_detected,  # Track opponent injuries
        "opponent_injury_boost_projection": opponent_injury_boost,  # Projection boost %
        "opponent_injury_boost_ceiling": opponent_injury_ceiling_boost,  # Ceiling boost %
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
    """
    Fetch scoreboard data from NBA API with extended timeout and retry logic.

    Args:
        game_date_str: Game date in format 'MM/DD/YYYY'

    Returns:
        Tuple of (game_header DataFrame, line_score DataFrame)

    Raises:
        Exception: If all retry attempts fail
    """
    import time
    from requests.exceptions import ReadTimeout, ConnectTimeout

    # Retry configuration
    max_retries = 3
    timeout = 60  # Increase from default 30s to 60s
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=game_date_str,
                league_id="00",
                timeout=timeout
            )
            return (
                scoreboard.game_header.get_data_frame(),
                scoreboard.line_score.get_data_frame(),
            )
        except (ReadTimeout, ConnectTimeout) as e:
            if attempt < max_retries - 1:
                # Not the last attempt, retry
                time.sleep(retry_delay)
                continue
            else:
                # Last attempt failed, re-raise
                raise Exception(
                    f"NBA API timeout after {max_retries} attempts. "
                    f"The stats.nba.com API may be slow or unavailable. "
                    f"Try again in a few minutes or check https://www.nba.com/stats"
                ) from e
        except Exception as e:
            # Other errors (not timeout), raise immediately
            raise


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

# Initialize injury tracking tables -----------------------------------------
# Ensure injury_list and predictions tables exist with proper schema
try:
    games_conn = get_connection(str(db_path))

    # Create injury_list table if it doesn't exist
    ia.create_injury_list_table(games_conn)

    # Ensure predictions table has injury tracking columns
    pt.create_predictions_table(games_conn)
    pt.upgrade_predictions_table_for_injuries(games_conn)
    pt.upgrade_predictions_table_for_refresh(games_conn)
    pt.upgrade_predictions_table_for_opponent_injury(games_conn)  # NEW: Opponent injury impact tracking

except Exception as init_exc:
    st.warning(f"Could not initialize injury tracking tables: {init_exc}")

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

matchup_spotlight_rows: list[Dict[str, Any]] = []
daily_power_rows_points: list[Dict[str, Any]] = []
daily_power_rows_3pm: list[Dict[str, Any]] = []
daily_top_scorers_rows: list[Dict[str, Any]] = []
player_season_stats_map: Dict[int, Mapping[str, Any]] = {}
game_totals_by_game: Dict[str, Dict[str, Any]] = {}  # Track team totals by game_id

# Today's games tab --------------------------------------------------------
if selected_page == "Today's Games":
    st.subheader("Today's Games - Predictions View")

    # Get database connection
    games_conn = get_connection(str(db_path))
    # Ensure predictions table exists
    pt.create_predictions_table(games_conn)

    # Date selector
    selected_date = st.date_input(
        "Game date",
        value=default_game_date(),
        key="todays_games_date",
    )

    # DNP WARNING SYSTEM: Alert if OUT players have predictions
    try:
        import injury_adjustment as ia
        selected_date_str = selected_date.strftime('%Y-%m-%d')

        # Check for OUT players with predictions
        out_players_query = """
            SELECT COUNT(*) as count
            FROM predictions p
            INNER JOIN injury_list i ON p.player_id = i.player_id
            WHERE p.game_date = ?
              AND i.status IN ('out', 'doubtful')
        """
        cursor = games_conn.cursor()
        cursor.execute(out_players_query, (selected_date_str,))
        out_count = cursor.fetchone()[0]

        if out_count > 0:
            st.warning(f"""
⚠️ **{out_count} OUT/DOUBTFUL player(s) have predictions for {selected_date}!**

These predictions should be removed to avoid DNP errors.

👉 Go to **Injury Admin** tab → Click **"Refresh Predictions"** to:
- Delete OUT player predictions
- Redistribute expected points to healthy teammates
- Update your projections with latest injury news
            """, icon="⚠️")
    except Exception:
        # Silently fail if injury check fails
        pass

    # Load predictions from database
    predictions_df = load_predictions_for_date(games_conn, str(selected_date))

    if predictions_df.empty:
        st.info(
            f"📊 No predictions found for {selected_date}.\n\n"
            f"Use the **Generate Predictions** button in the sidebar to create predictions for this date."
        )

        # Show games schedule without predictions
        try:
            games_df, _ = build_games_table(
                str(db_path),
                selected_date,
                builder_config["season"],
                DEFAULT_SEASON_TYPE,
            )
            if not games_df.empty:
                st.markdown("### Games Schedule")
                display_df = games_df.drop(columns=MATCHUP_INTERNAL_COLUMNS, errors="ignore")
                render_dataframe(display_df)
                st.caption(
                    "Data source: nba_api ScoreboardV2 + standings context from the local SQLite database."
                )
            else:
                st.info("No NBA games scheduled for this date per the official scoreboard.")
        except Exception as e:
            st.error(f"Could not load games schedule: {e}")

    else:
        # Display predictions grouped by game
        st.success(f"✅ Loaded {len(predictions_df)} predictions for {selected_date}")

        # Action buttons row
        col1, col2, col3 = st.columns([2, 1, 1])

        with col2:
            if st.button("🔄 Regenerate All", help="Regenerate all predictions for this date", use_container_width=True):
                generate_predictions_ui(selected_date, db_path, builder_config)
                st.rerun()

        with col3:
            # CSV export
            if not predictions_df.empty:
                export_df = predictions_df.copy()
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 CSV",
                    data=csv,
                    file_name=f"nba_predictions_{selected_date}.csv",
                    mime="text/csv",
                    help="Download predictions as CSV",
                    use_container_width=True
                )

        # Group predictions by game (matchup)
        # Create game key: "Away @ Home"
        predictions_df['game_key'] = predictions_df['opponent_name'] + ' @ ' + predictions_df['team_name']

        # Get unique games
        games = predictions_df['game_key'].unique()

        st.markdown("### Game-by-Game Predictions")
        st.caption(f"Showing {len(games)} game(s) with predictions")

        for game_key in games:
            game_preds = predictions_df[predictions_df['game_key'] == game_key]

            # Parse away/home teams
            if ' @ ' in game_key:
                away_team, home_team = game_key.split(' @ ')
            else:
                continue

            st.markdown(f"#### {away_team} at {home_team}")

            # Split into away and home columns
            col_away, col_home = st.columns(2)

            with col_away:
                st.markdown(f"**{away_team}** (Away)")
                # Get away team predictions (where team_name != home_team or opponent_name == home_team)
                # Actually, in the predictions table, team_name is the team the player is ON
                # opponent_name is who they're playing against
                # So if opponent_name == home_team, then this player is on the away team
                away_preds = game_preds[game_preds['opponent_name'] == home_team]
                if not away_preds.empty:
                    display_team_predictions(away_preds)
                else:
                    st.caption("No predictions for this team")

            with col_home:
                st.markdown(f"**{home_team}** (Home)")
                # If opponent_name == away_team, then this player is on the home team
                home_preds = game_preds[game_preds['opponent_name'] == away_team]
                if not home_preds.empty:
                    display_team_predictions(home_preds)
                else:
                    st.caption("No predictions for this team")

            st.markdown("---")

        # S3 backup option at bottom
        st.markdown("### Cloud Backup")
        col_backup1, col_backup2 = st.columns(2)

        with col_backup1:
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
if selected_page == "Matchup Spotlight":
    st.subheader("Player Matchup Spotlight")

    # Load predictions directly from database (self-contained)
    try:
        # Get selected date
        spotlight_date = st.date_input(
            "Select Date",
            value=default_game_date(),
            key="spotlight_date_input"
        )
        spotlight_date_str = spotlight_date.strftime('%Y-%m-%d')

        # Query predictions for this date
        spotlight_query = """
            SELECT
                player_name,
                team_name,
                opponent_name,
                projected_ppg,
                proj_floor,
                proj_ceiling,
                proj_confidence,
                season_avg_ppg,
                recent_avg_5 as last5_avg_ppg,
                recent_avg_3 as last3_avg_ppg,
                vs_opponent_avg,
                vs_opponent_games,
                dfs_score,
                dfs_grade,
                opponent_def_rating,
                opponent_pace,
                game_date
            FROM predictions
            WHERE game_date = ?
            ORDER BY dfs_score DESC
        """

        spotlight_conn = get_connection(str(db_path))
        spotlight_df_raw = run_query(str(db_path), spotlight_query, params=(spotlight_date_str,))

        if spotlight_df_raw.empty:
            st.info(f"📊 No predictions found for {spotlight_date_str}. Go to Today's Games tab and generate predictions first.")
        else:
            # Rebuild matchup_spotlight_rows from database
            matchup_spotlight_rows = []
            for _, row in spotlight_df_raw.iterrows():
                matchup_spotlight_rows.append({
                    "Matchup": f"{row['opponent_name']} at {row['team_name']}",  # Simplified
                    "Player": row['player_name'],
                    "Team": row['team_name'],
                    "Opponent": row['opponent_name'],
                    "DFS Score": row['dfs_score'],
                    "Pick Grade": row['dfs_grade'],
                    "Season Avg PPG": row['season_avg_ppg'],
                    "Projected PPG": row['projected_ppg'],
                    "Proj Floor": row['proj_floor'],
                    "Proj Ceiling": row['proj_ceiling'],
                    "Proj Confidence": row['proj_confidence'],
                    "Last5 Avg PPG": row['last5_avg_ppg'],
                    "Last3 Avg PPG": row['last3_avg_ppg'],
                    "Vs This Team": f"{row['vs_opponent_avg']:.1f} ({row['vs_opponent_games']}g)" if row['vs_opponent_avg'] else "N/A",
                    "Opp Def Rating": row['opponent_def_rating'],
                    "Matchup Score": row['dfs_score'],  # Use DFS score as matchup score
                    "Composite Score": row['dfs_score'],  # Use DFS score as composite
                })

            # Calculate game totals from predictions
            game_totals_by_game = {}
            for _, row in spotlight_df_raw.iterrows():
                matchup_key = f"{row['opponent_name']} @ {row['team_name']}"

                if matchup_key not in game_totals_by_game:
                    game_totals_by_game[matchup_key] = {
                        'away_team': row['opponent_name'] if '@' in matchup_key else row['team_name'],
                        'home_team': row['team_name'] if '@' in matchup_key else row['opponent_name'],
                        'away_projected_total': 0.0,
                        'home_projected_total': 0.0,
                        'away_top_5_total': 0.0,
                        'home_top_5_total': 0.0,
                        'away_count': 0,
                        'home_count': 0,
                    }

                # Aggregate by team
                if row['team_name'] == game_totals_by_game[matchup_key]['home_team']:
                    game_totals_by_game[matchup_key]['home_projected_total'] += row['projected_ppg'] or 0
                    game_totals_by_game[matchup_key]['home_count'] += 1
                else:
                    game_totals_by_game[matchup_key]['away_projected_total'] += row['projected_ppg'] or 0
                    game_totals_by_game[matchup_key]['away_count'] += 1

            # Now continue with existing display logic
    except Exception as exc:
        st.error(f"❌ **Failed to load matchup spotlight:** {exc}")
        import traceback
        st.code(traceback.format_exc())
        matchup_spotlight_rows = []
        game_totals_by_game = {}

    if matchup_spotlight_rows:
        # NEW: Game Totals Summary Section
        if game_totals_by_game:
            st.markdown("### 🏀 Projected Game Totals")
            st.caption("Sum of all player projections for each team")

            totals_data = []
            for game_id, totals in game_totals_by_game.items():
                totals_data.append({
                    "Matchup": f"{totals['away_team']} @ {totals['home_team']}",
                    "Away Team": totals['away_team'],
                    "Away Proj": f"{totals['away_projected_total']:.1f}",
                    "Away Top 5": f"{totals['away_top_5_total']:.1f}",
                    "Home Team": totals['home_team'],
                    "Home Proj": f"{totals['home_projected_total']:.1f}",
                    "Home Top 5": f"{totals['home_top_5_total']:.1f}",
                    "Game Total": f"{totals['away_projected_total'] + totals['home_projected_total']:.1f}",
                    "Top 5 Total": f"{totals['away_top_5_total'] + totals['home_top_5_total']:.1f}",
                })

            totals_df = pd.DataFrame(totals_data)

            # Sort by game total descending
            totals_df_sorted = totals_df.copy()
            totals_df_sorted['_game_total_numeric'] = totals_df['Game Total'].astype(float)
            totals_df_sorted = totals_df_sorted.sort_values('_game_total_numeric', ascending=False)
            totals_df_sorted = totals_df_sorted.drop(columns=['_game_total_numeric'])

            st.dataframe(totals_df_sorted, use_container_width=True, hide_index=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_game_total = sum(t['away_projected_total'] + t['home_projected_total'] for t in game_totals_by_game.values()) / len(game_totals_by_game)
                st.metric("Avg Game Total", f"{avg_game_total:.1f}")
            with col2:
                max_game = max(game_totals_by_game.values(), key=lambda x: x['away_projected_total'] + x['home_projected_total'])
                max_total = max_game['away_projected_total'] + max_game['home_projected_total']
                st.metric("Highest Projected", f"{max_total:.1f}", f"{max_game['away_team']} @ {max_game['home_team']}")
            with col3:
                min_game = min(game_totals_by_game.values(), key=lambda x: x['away_projected_total'] + x['home_projected_total'])
                min_total = min_game['away_projected_total'] + min_game['home_projected_total']
                st.metric("Lowest Projected", f"{min_total:.1f}", f"{min_game['away_team']} @ {min_game['home_team']}")

            st.divider()
        # END NEW SECTION
        spotlight_df = pd.DataFrame(matchup_spotlight_rows)
        sort_column = st.selectbox(
            "Sort by",
            options=[
                "Matchup Score",
                "DFS Score",
                "Projected PPG",
                "Season Avg PPG",
                "Last5 Avg PPG",
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

        # Build daily power rankings from predictions
        daily_power_rows_points = []
        for _, row in spotlight_df_raw.iterrows():
            daily_power_rows_points.append({
                "Matchup": f"{row['opponent_name']} @ {row['team_name']}",
                "Player": row['player_name'],
                "DFS Score": row['dfs_score'],
                "Projected PPG": row['projected_ppg'],
                "Opp Avg Allowed PPG": 0,  # Not available
                "Opp Last5 Avg Allowed": 0,  # Not available
                "Opportunity Index": row['dfs_score'] / 10,  # Approximate
                "Opp Def Composite": row['opponent_def_rating'],
                "Usage %": 0,  # Not available
                "Proj Conf": f"{row['proj_confidence']:.0%}" if row['proj_confidence'] else "N/A",
            })

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

        # 3PM rankings not available from predictions table
        daily_power_rows_3pm = []

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
            col_3pm.info("💡 3PM rankings available in Today's Games tab during prediction generation.")

# Daily leaders tab --------------------------------------------------------
if selected_page == "Daily Leaders":
    st.subheader("Daily Top Scorers (Top 3 per day)")

    # Season filter
    col1, col2 = st.columns([2, 1])
    with col1:
        leaders_season = st.text_input(
            "Season",
            value=DEFAULT_SEASON,
            key="leaders_season_input",
        ).strip() or DEFAULT_SEASON
    with col2:
        leaders_season_type = st.selectbox(
            "Season type",
            options=[DEFAULT_SEASON_TYPE, "Playoffs", "Pre Season"],
            index=0,
            key="leaders_season_type_input",
        )

    # Load data directly from database
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
                leaders_season,
                leaders_season_type,
                leaders_season,
                leaders_season_type,
                DAILY_LEADERS_MAX,
            ),
        )

        if top_df.empty:
            st.warning(f"""
            ⚠️ **No data found for {leaders_season} {leaders_season_type}**

            This could mean:
            - No games have been played yet for this season
            - The database needs to be updated
            - The season name is incorrect

            Try changing the season filter above or updating your database.
            """)
        else:
            # Build leaders list
            leaders_rows = []
            for _, row in top_df.iterrows():
                game_date = pd.to_datetime(row["game_date"]).date()
                home_team, away_team = derive_home_away(row["team_name"], row["matchup"])
                minutes_float = minutes_str_to_float(row["minutes"])
                usage_pct = safe_float(row["usg_pct"])
                leaders_rows.append({
                    "Date": game_date.isoformat(),
                    "Player": row["player_name"],
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Total Points": row["points"],
                    "Minutes": minutes_float,
                    "Usage %": (usage_pct * 100.0) if usage_pct is not None else None,
                })

            leaders_df = pd.DataFrame(leaders_rows)
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
            filtered_df["Usage %"] = filtered_df["Usage %"].map(lambda v: format_number(v, 1) if v is not None else "N/A")

            display_df = filtered_df[
                [
                    "Date",
                    "Player",
                    "Home Team",
                    "Away Team",
                    "Total Points",
                    "Minutes",
                    "Usage %",
                ]
            ].reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"❌ **Failed to load daily leaders:** {exc}")
        import traceback
        st.code(traceback.format_exc(), language="python")

# Standings tab -------------------------------------------------------------
if selected_page == "Standings":
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
if selected_page == "3PT Leaders":
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
if selected_page == "Scoring Leaders":
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
if selected_page == "3PT Defense":
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
if selected_page == "Points Allowed":
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
if selected_page == "Defense Mix":
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

        # Add PPM stats to Defense Mix
        try:
            off_ppm_df, def_ppm_df, league_avg_ppm = load_ppm_stats(str(db_path), '2025-26')

            # Merge defensive PPM stats by team_name
            mix_df = mix_df.merge(
                def_ppm_df[['team_name', 'avg_def_ppm', 'def_ppm_grade', 'ceiling_factor', 'std_def_ppm']],
                on='team_name',
                how='left'
            )

            # Rename columns for clarity
            mix_df = mix_df.rename(columns={
                'avg_def_ppm': 'Def PPM',
                'def_ppm_grade': 'PPM Grade',
                'ceiling_factor': 'Ceiling Factor',
                'std_def_ppm': 'PPM StdDev'
            })

            st.caption(f"✅ PPM stats loaded (League Avg: {league_avg_ppm:.3f} PPM)")
        except Exception as ppm_exc:
            st.caption(f"⚠️ PPM stats unavailable: {ppm_exc}")

        render_dataframe(mix_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Defense mix view not available: {exc}")

# Defense styles tab -------------------------------------------------------
if selected_page == "Defense Styles":
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
if selected_page == "Player Impact":
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
if selected_page == "Prediction Log":
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

                    # Clear cached database connections and rerun
                    st.cache_resource.clear()
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
                # Add minutes data
                minutes_query = """
                    SELECT
                        player_name,
                        minutes as actual_minutes,
                        (SELECT AVG(CAST(minutes AS REAL))
                         FROM player_game_logs
                         WHERE player_name = pgl.player_name
                           AND season = '2025-26'
                           AND CAST(minutes AS REAL) > 0
                        ) as avg_minutes
                    FROM player_game_logs pgl
                    WHERE DATE(game_date) = ?
                """
                try:
                    minutes_df = pd.read_sql_query(minutes_query, pred_conn, params=[selected_date])
                    if not minutes_df.empty:
                        # Convert to numeric
                        minutes_df['actual_minutes'] = pd.to_numeric(minutes_df['actual_minutes'], errors='coerce')
                        minutes_df['avg_minutes'] = pd.to_numeric(minutes_df['avg_minutes'], errors='coerce')
                        # Merge with main df
                        df = df.merge(minutes_df, on='player_name', how='left')
                    else:
                        df['actual_minutes'] = None
                        df['avg_minutes'] = None
                except:
                    df['actual_minutes'] = None
                    df['avg_minutes'] = None

                # Add PPM consistency data
                ppm_query = """
                    SELECT
                        player_name,
                        AVG(points / CAST(minutes AS REAL)) as avg_ppm,
                        (SELECT COUNT(*)
                         FROM player_game_logs
                         WHERE player_name = pgl.player_name
                           AND season = '2025-26'
                           AND CAST(minutes AS REAL) >= 10
                        ) as games_counted
                    FROM player_game_logs pgl
                    WHERE season = '2025-26'
                      AND CAST(minutes AS REAL) >= 10
                    GROUP BY player_name
                """
                try:
                    ppm_df = pd.read_sql_query(ppm_query, pred_conn)

                    # Calculate PPM standard deviation for each player
                    ppm_std_query = """
                        SELECT
                            player_name,
                            points / CAST(minutes AS REAL) as ppm
                        FROM player_game_logs
                        WHERE season = '2025-26'
                          AND CAST(minutes AS REAL) >= 10
                    """
                    ppm_raw = pd.read_sql_query(ppm_std_query, pred_conn)
                    ppm_std_df = ppm_raw.groupby('player_name')['ppm'].std().reset_index()
                    ppm_std_df.columns = ['player_name', 'std_ppm']

                    # Merge avg and std
                    ppm_df = ppm_df.merge(ppm_std_df, on='player_name', how='left')

                    # Calculate consistency score
                    ppm_df['cv_ppm'] = ppm_df['std_ppm'] / ppm_df['avg_ppm']
                    ppm_df['consistency_score'] = 100 * (1 - ppm_df['cv_ppm'].clip(upper=1.0))

                    # Assign grades
                    def get_ppm_grade(score):
                        if pd.isna(score):
                            return 'N/A'
                        elif score >= 85:
                            return 'A+'
                        elif score >= 75:
                            return 'A'
                        elif score >= 65:
                            return 'B'
                        elif score >= 55:
                            return 'C'
                        else:
                            return 'D'

                    ppm_df['ppm_grade'] = ppm_df['consistency_score'].apply(get_ppm_grade)

                    # Merge with main df
                    if not ppm_df.empty:
                        df = df.merge(ppm_df[['player_name', 'avg_ppm', 'consistency_score', 'ppm_grade']],
                                     on='player_name', how='left')
                    else:
                        df['avg_ppm'] = None
                        df['consistency_score'] = None
                        df['ppm_grade'] = 'N/A'
                except Exception as e:
                    df['avg_ppm'] = None
                    df['consistency_score'] = None
                    df['ppm_grade'] = 'N/A'

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
                                 'actual_minutes', 'avg_minutes', 'avg_ppm', 'consistency_score', 'ppm_grade',
                                 'proj_confidence', 'proj_floor', 'proj_ceiling', 'Accuracy',
                                 'dfs_score', 'dfs_grade', 'analytics_used']].copy()

                display_df.columns = ['Player', 'Team', 'Opponent', 'Status',
                                      'Proj PPG', 'Actual PPG', 'Error', 'Abs Error',
                                      'Actual Mins', 'Avg Mins', 'PPM', 'PPM Consistency', 'PPM Grade',
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

                # Minutes Analysis Section
                with st.expander("⏱️ Minutes Analysis - Post-Game Insights", expanded=False):
                    st.markdown("**Analyze prediction misses caused by reduced playing time**")

                    # Query to get actual minutes played vs season average
                    minutes_query = """
                        SELECT
                            p.player_name,
                            p.team_name,
                            p.opponent_name,
                            p.projected_ppg,
                            p.actual_ppg,
                            p.error,
                            p.abs_error,
                            pgl.minutes as actual_minutes,
                            (SELECT AVG(CAST(minutes AS REAL))
                             FROM player_game_logs
                             WHERE player_name = p.player_name
                               AND season = '2025-26'
                               AND CAST(minutes AS REAL) > 0
                            ) as avg_minutes
                        FROM predictions p
                        LEFT JOIN player_game_logs pgl
                            ON p.player_name = pgl.player_name
                            AND DATE(pgl.game_date) = p.game_date
                        WHERE p.game_date = ?
                          AND p.actual_ppg IS NOT NULL
                          AND pgl.minutes IS NOT NULL
                        ORDER BY p.abs_error DESC
                    """

                    try:
                        minutes_df = pd.read_sql_query(minutes_query, pred_conn, params=[selected_date])

                        if not minutes_df.empty:
                            # Convert minutes to float
                            minutes_df['actual_minutes'] = minutes_df['actual_minutes'].astype(float)
                            minutes_df['avg_minutes'] = minutes_df['avg_minutes'].fillna(0).astype(float)

                            # Calculate minutes % of average
                            minutes_df['min_pct'] = (minutes_df['actual_minutes'] / minutes_df['avg_minutes'] * 100).fillna(0)

                            # Flag reduced minutes (< 70% of average)
                            minutes_df['reduced_mins'] = minutes_df['min_pct'] < 70

                            # Add analysis column
                            def minutes_analysis(row):
                                if row['reduced_mins']:
                                    return f"⚠️ Low Minutes ({row['min_pct']:.0f}% of avg)"
                                elif row['min_pct'] > 110:
                                    return f"✅ High Minutes ({row['min_pct']:.0f}% of avg)"
                                else:
                                    return f"Normal ({row['min_pct']:.0f}% of avg)"

                            minutes_df['Minutes Flag'] = minutes_df.apply(minutes_analysis, axis=1)

                            # Display summary metrics
                            min_cols = st.columns(3)
                            with min_cols[0]:
                                reduced_count = minutes_df['reduced_mins'].sum()
                                st.metric("Players with Reduced Minutes", f"{reduced_count}")
                            with min_cols[1]:
                                if reduced_count > 0:
                                    reduced_avg_error = minutes_df[minutes_df['reduced_mins']]['abs_error'].mean()
                                    st.metric("Avg Error (Reduced Mins)", f"{reduced_avg_error:.1f} pts")
                            with min_cols[2]:
                                normal_mins = minutes_df[~minutes_df['reduced_mins']]
                                if len(normal_mins) > 0:
                                    normal_avg_error = normal_mins['abs_error'].mean()
                                    st.metric("Avg Error (Normal Mins)", f"{normal_avg_error:.1f} pts")

                            st.divider()

                            # Display detailed table
                            st.markdown("**Players Ordered by Prediction Error:**")
                            display_mins = minutes_df[['player_name', 'team_name', 'opponent_name',
                                                       'projected_ppg', 'actual_ppg', 'error', 'abs_error',
                                                       'actual_minutes', 'avg_minutes', 'min_pct', 'Minutes Flag']].copy()

                            display_mins.columns = ['Player', 'Team', 'Opponent', 'Proj PPG', 'Actual PPG',
                                                   'Error', 'Abs Error', 'Actual Mins', 'Avg Mins', 'Min %', 'Analysis']

                            # Format columns
                            display_mins['Proj PPG'] = display_mins['Proj PPG'].round(1)
                            display_mins['Actual PPG'] = display_mins['Actual PPG'].round(1)
                            display_mins['Error'] = display_mins['Error'].round(1)
                            display_mins['Abs Error'] = display_mins['Abs Error'].round(1)
                            display_mins['Actual Mins'] = display_mins['Actual Mins'].round(1)
                            display_mins['Avg Mins'] = display_mins['Avg Mins'].round(1)
                            display_mins['Min %'] = display_mins['Min %'].round(0)

                            st.dataframe(display_mins, use_container_width=True, height=400)

                            # Key insights
                            st.markdown("**💡 Key Insights:**")
                            reduced = minutes_df[minutes_df['reduced_mins']]
                            if len(reduced) > 0:
                                st.markdown(f"- **{len(reduced)} player(s) played <70% of normal minutes**")
                                worst_mins = reduced.nlargest(3, 'abs_error')
                                for _, row in worst_mins.iterrows():
                                    st.markdown(f"  - **{row['player_name']}**: {row['actual_minutes']:.0f} mins (avg {row['avg_minutes']:.0f}) → {row['abs_error']:.1f} pt error")
                            else:
                                st.success("✅ All players played normal minutes - errors not due to playing time")
                        else:
                            st.info("No minutes data available for this date yet. Run 'Fetch & Score Latest Games' first.")

                    except Exception as e:
                        st.error(f"Error loading minutes analysis: {str(e)}")

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

        # Enhanced Metrics Section
        st.divider()

        with st.expander("📊 Enhanced Evaluation Metrics", expanded=False):
            st.markdown("**Deep-dive prediction quality analysis beyond simple MAE**")

            # Date range selector for enhanced metrics
            metric_date_option = st.radio(
                "Analysis Period",
                options=["Single Date", "Last 7 Days", "Last 30 Days", "Season to Date"],
                horizontal=True,
                key="enhanced_metrics_period"
            )

            try:
                import prediction_evaluation_metrics as pem
                from datetime import datetime, timedelta

                # Calculate date range based on selection
                if metric_date_option == "Single Date":
                    metrics = pem.calculate_enhanced_metrics(
                        pred_conn,
                        game_date=selected_date,
                        min_actual_ppg=1.0  # Filter DNPs
                    )
                    analysis_label = selected_date
                elif metric_date_option == "Last 7 Days":
                    end_date = datetime.strptime(selected_date, '%Y-%m-%d')
                    start_date = end_date - timedelta(days=7)
                    metrics = pem.calculate_enhanced_metrics(
                        pred_conn,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=selected_date,
                        min_actual_ppg=1.0
                    )
                    analysis_label = f"Last 7 Days (through {selected_date})"
                elif metric_date_option == "Last 30 Days":
                    end_date = datetime.strptime(selected_date, '%Y-%m-%d')
                    start_date = end_date - timedelta(days=30)
                    metrics = pem.calculate_enhanced_metrics(
                        pred_conn,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=selected_date,
                        min_actual_ppg=1.0
                    )
                    analysis_label = f"Last 30 Days (through {selected_date})"
                else:  # Season to Date
                    metrics = pem.calculate_enhanced_metrics(
                        pred_conn,
                        start_date='2025-10-01',
                        end_date=selected_date,
                        min_actual_ppg=1.0
                    )
                    analysis_label = f"Season to Date (through {selected_date})"

                if 'error' in metrics:
                    st.warning(metrics['error'])
                elif metrics['total_predictions'] == 0:
                    st.info("No predictions found for selected period")
                else:
                    st.markdown(f"### Analysis: {analysis_label}")
                    st.caption(f"Total Predictions: {metrics['total_predictions']} (filtered DNPs)")

                    # Core metrics
                    st.markdown("#### 📈 Core Accuracy Metrics")
                    core_cols = st.columns(4)
                    core_cols[0].metric("MAE", f"{metrics['mae']:.2f} PPG")
                    core_cols[1].metric("RMSE", f"{metrics['rmse']:.2f} PPG")
                    core_cols[2].metric("Bias", f"{metrics['bias']:+.2f} PPG",
                                       help="Positive = over-projecting, Negative = under-projecting")
                    core_cols[3].metric("Median Error", f"{metrics['median_error']:+.2f} PPG")

                    st.divider()

                    # Hit rates
                    st.markdown("#### 🎯 Hit Rates")
                    hit_cols = st.columns(5)
                    hit_cols[0].metric("Within ±5 PPG", f"{metrics['hit_rate_within_5']:.1f}%")
                    hit_cols[1].metric("Within ±10 PPG", f"{metrics['hit_rate_within_10']:.1f}%")
                    hit_cols[2].metric("Floor-Ceiling Hit", f"{metrics['floor_ceiling_hit_rate']:.1f}%",
                                      help="% of actuals within projected range")
                    hit_cols[3].metric("Above Floor", f"{metrics['floor_coverage']:.1f}%")
                    hit_cols[4].metric("Below Ceiling", f"{metrics['ceiling_coverage']:.1f}%")

                    st.divider()

                    # Rank-ordering quality
                    st.markdown("#### 🏆 Rank-Ordering Quality")
                    st.caption("How well do we order players relative to each other? (Critical for DFS)")

                    rank_cols = st.columns(3)
                    rank_cols[0].metric(
                        "Spearman Correlation",
                        f"{metrics['spearman_correlation']:.3f}",
                        help="Perfect rank-ordering = 1.0, Random = 0.0"
                    )
                    rank_cols[1].metric(
                        "Pearson Correlation",
                        f"{metrics['pearson_correlation']:.3f}",
                        help="Linear correlation between projected and actual"
                    )

                    # Interpretation
                    if metrics['spearman_correlation'] >= 0.75:
                        rank_quality = "🟢 Excellent - Strong rank-ordering"
                    elif metrics['spearman_correlation'] >= 0.65:
                        rank_quality = "🟡 Good - Reliable rank-ordering"
                    elif metrics['spearman_correlation'] >= 0.50:
                        rank_quality = "🟠 Fair - Some rank-ordering skill"
                    else:
                        rank_quality = "🔴 Poor - Weak rank-ordering"

                    rank_cols[2].info(rank_quality)

                    st.divider()

                    # Outlier analysis
                    st.markdown("#### 🔥 Outlier Analysis")
                    outlier_cols = st.columns(3)

                    if metrics['top_10_pct_miss_rate']:
                        outlier_cols[0].metric(
                            "Top 10% Worst Misses",
                            f"{metrics['top_10_pct_miss_rate']:.2f} PPG MAE",
                            help="Average error for worst 10% of predictions"
                        )
                        outlier_cols[1].metric(
                            "Top 10% Max Miss",
                            f"{metrics['top_10_pct_max_miss']:.2f} PPG",
                            help="Single worst prediction error"
                        )

                    if metrics['bottom_10_pct_mae']:
                        outlier_cols[2].metric(
                            "Bottom 10% Best",
                            f"{metrics['bottom_10_pct_mae']:.2f} PPG MAE",
                            help="Average error for best 10% of predictions"
                        )

                    st.divider()

                    # Over/under balance
                    st.markdown("#### ⚖️ Over/Under Balance")
                    balance_cols = st.columns(4)
                    balance_cols[0].metric("Over-Projections", metrics['over_projections'])
                    balance_cols[1].metric("Under-Projections", metrics['under_projections'])
                    balance_cols[2].metric("Over %", f"{metrics['over_pct']:.1f}%")
                    balance_cols[3].metric("Under %", f"{metrics['under_pct']:.1f}%")

                    # Balance interpretation
                    balance_diff = abs(metrics['over_pct'] - 50.0)
                    if balance_diff <= 5:
                        st.success("✅ Well-balanced projection bias")
                    elif balance_diff <= 10:
                        st.info("ℹ️ Slight projection bias")
                    else:
                        if metrics['over_pct'] > 50:
                            st.warning(f"⚠️ Tendency to over-project ({metrics['over_pct']:.1f}%)")
                        else:
                            st.warning(f"⚠️ Tendency to under-project ({metrics['under_pct']:.1f}%)")

                    st.divider()

                    # Metrics by player tier
                    st.markdown("#### 📊 Performance by Player Tier")

                    if metric_date_option == "Single Date":
                        tier_start = selected_date
                        tier_end = selected_date
                    elif metric_date_option == "Last 7 Days":
                        tier_end = selected_date
                        tier_start = (datetime.strptime(selected_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
                    elif metric_date_option == "Last 30 Days":
                        tier_end = selected_date
                        tier_start = (datetime.strptime(selected_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
                    else:
                        tier_start = '2025-10-01'
                        tier_end = selected_date

                    tier_metrics = pem.get_metrics_by_player_tier(
                        pred_conn,
                        start_date=tier_start,
                        end_date=tier_end
                    )

                    if not tier_metrics.empty:
                        st.dataframe(
                            tier_metrics,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Key insights
                        st.caption("**Key Insights:**")
                        st.caption("- MAE = Mean Absolute Error (lower is better)")
                        st.caption("- Floor_Ceiling_Rate = % of predictions within projected range")
                        st.caption("- Compare tiers to identify if model performs differently for stars vs role players")

            except Exception as e:
                st.error(f"Error calculating enhanced metrics: {e}")
                import traceback
                st.code(traceback.format_exc())

# Ceiling Analytics Tab ---------------------------------------------------
if selected_page == "Ceiling Analytics":
    st.header("🎯 Opponent Ceiling Analytics")
    st.write("Which teams allow explosive scoring performances? Track ceiling volatility for DFS tournament strategy.")

    ceiling_conn = get_connection(str(db_path))

    # Season selector
    ceiling_season = st.selectbox(
        "Select season",
        options=['2025-26', '2024-25', '2023-24'],
        index=0,
        key="ceiling_season"
    )

    try:
        import opponent_ceiling_analytics as oca

        st.divider()

        # Get rankings
        with st.spinner("Calculating ceiling analytics..."):
            rankings = oca.get_all_teams_ceiling_rankings(
                ceiling_conn,
                season=ceiling_season,
                min_games=5
            )

        if not rankings.empty:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)

            elite_count = len(rankings[rankings['ceiling_volatility'] >= 70])
            high_count = len(rankings[rankings['ceiling_volatility'] >= 55])
            suppressor_count = len(rankings[rankings['ceiling_volatility'] < 25])
            avg_volatility = rankings['ceiling_volatility'].mean()

            col1.metric("Elite Ceiling Spots", elite_count, help="Volatility ≥ 70")
            col2.metric("High Ceiling Spots", high_count, help="Volatility ≥ 55")
            col3.metric("Ceiling Suppressors", suppressor_count, help="Volatility < 25")
            col4.metric("Avg Volatility", f"{avg_volatility:.1f}")

            st.divider()

            # Top ceiling spots
            st.subheader("🔥 Top 10 Ceiling Spots (Best for DFS Tournaments)")
            st.caption("These teams allow the most 35+, 40+, and 45+ point performances")

            top10 = rankings.head(10).copy()
            top10['ceiling_volatility'] = top10['ceiling_volatility'].apply(lambda x: f"{x:.1f}")
            top10['pct_35plus'] = top10['pct_35plus'].apply(lambda x: f"{x:.1f}%")
            top10['pct_40plus'] = top10['pct_40plus'].apply(lambda x: f"{x:.1f}%")
            top10['pct_45plus'] = top10['pct_45plus'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(
                top10[['rank', 'team_name', 'ceiling_volatility', 'tier',
                       'pct_35plus', 'pct_40plus', 'pct_45plus', 'total_games']],
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # Bottom ceiling spots
            st.subheader("🛡️ Bottom 10 Ceiling Suppressors (Avoid for Tournaments)")
            st.caption("These teams rarely allow explosive scoring performances")

            bottom10 = rankings.tail(10).copy()
            bottom10['ceiling_volatility'] = bottom10['ceiling_volatility'].apply(lambda x: f"{x:.1f}")
            bottom10['pct_35plus'] = bottom10['pct_35plus'].apply(lambda x: f"{x:.1f}%")
            bottom10['pct_40plus'] = bottom10['pct_40plus'].apply(lambda x: f"{x:.1f}%")
            bottom10['pct_45plus'] = bottom10['pct_45plus'].apply(lambda x: f"{x:.1f}%")

            st.dataframe(
                bottom10[['rank', 'team_name', 'ceiling_volatility', 'tier',
                         'pct_35plus', 'pct_40plus', 'pct_45plus', 'total_games']],
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # Full rankings
            with st.expander("📊 View Full Rankings", expanded=False):
                full_rankings = rankings.copy()
                full_rankings['ceiling_volatility'] = full_rankings['ceiling_volatility'].apply(lambda x: f"{x:.1f}")
                full_rankings['pct_35plus'] = full_rankings['pct_35plus'].apply(lambda x: f"{x:.1f}%")
                full_rankings['pct_40plus'] = full_rankings['pct_40plus'].apply(lambda x: f"{x:.1f}%")
                full_rankings['pct_45plus'] = full_rankings['pct_45plus'].apply(lambda x: f"{x:.1f}%")
                full_rankings['avg_allowed'] = full_rankings['avg_allowed'].apply(lambda x: f"{x:.1f}")

                st.dataframe(
                    full_rankings,
                    use_container_width=True,
                    hide_index=True
                )

            # Explanation
            with st.expander("ℹ️ How Ceiling Volatility Works"):
                st.markdown("""
**Ceiling Volatility Score (0-100):**

Measures how often a team allows high-scoring individual performances. This is different
from raw defensive rating - some teams suppress averages but allow spikes (volatile),
while others are consistently tight (suppressor).

**Scoring Breakdown:**
- **35+ point games:** Base weight (30%)
- **40+ point games:** 2x weight (60%) - these are DFS tournament targets
- **45+ point games:** 3x weight (90%) - nuclear outcomes
- **Variance:** Contributes 0-20 points based on scoring distribution

**Tiers:**
- **Elite Ceiling Spot (70+):** Frequently allows explosive games - target for tournaments
- **High Ceiling Spot (55-70):** Above average ceiling potential
- **Average Ceiling (40-55):** Moderate ceiling potential
- **Low Ceiling Spot (25-40):** Below average ceiling potential
- **Ceiling Suppressor (<25):** Rarely allows big games - avoid for tournaments

**DFS Strategy:**
- Tournament play: Target players vs Elite/High Ceiling Spots
- Cash games: Avoid extreme ceiling spots (higher variance)
- This score is already integrated into your ceiling confidence predictions!
                """)

        else:
            st.warning(f"No data available for {ceiling_season} season yet. Data requires at least 5 games per team.")

    except Exception as e:
        st.error(f"Error loading ceiling analytics: {e}")
        import traceback
        st.code(traceback.format_exc())

# Injury Admin Tab --------------------------------------------------------
if selected_page == "Injury Admin":
    st.header("🚑 Injury Administration")
    st.write("Manage injured players and adjust predictions for teammate impacts")

    injury_conn = get_connection(str(db_path))

    # Ensure injury_list table exists
    ia.create_injury_list_table(injury_conn)

    # Ensure injury_fetch_lock table exists (for auto-fetch cooldown)
    ia.create_injury_fetch_lock_table(injury_conn)

    # Ensure player_aliases table exists (for name matching)
    ia.create_player_aliases_table(injury_conn)

    st.divider()

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Active Injury List")

        # Status filter
        status_filter_options = st.multiselect(
            "Filter by status",
            options=['out', 'doubtful', 'questionable', 'probable', 'day-to-day', 'returned'],
            default=['out', 'doubtful', 'questionable'],
            key="injury_status_filter"
        )

        source_filter = st.selectbox(
            "Filter by source",
            options=['all', 'manual', 'automated'],
            key="injury_source_filter"
        )

        # Option to disable return date filtering
        check_dates = st.checkbox(
            "Hide players past expected return date",
            value=True,
            key="check_return_dates",
            help="Uncheck to show ALL injuries regardless of return date (useful if dates need updating)"
        )

        # Display current injury list with filters
        if status_filter_options:
            injury_list = ia.get_active_injuries(
                injury_conn,
                check_return_dates=check_dates,
                status_filter=status_filter_options
            )
        else:
            injury_list = []

        # Apply source filter
        if source_filter != 'all' and injury_list:
            injury_list = [inj for inj in injury_list if inj.get('source') == source_filter]

        if injury_list:
            # Helper function for status emoji
            def get_status_emoji(status):
                emoji_map = {
                    'out': '🔴',
                    'doubtful': '🟠',
                    'questionable': '🟡',
                    'probable': '🟢',
                    'day-to-day': '🔵',
                    'returned': '✅'
                }
                return emoji_map.get(status, '⚪')

            # Convert to DataFrame for display
            injury_df = pd.DataFrame(injury_list)

            # Format status with emoji
            injury_df['status_display'] = injury_df['status'].apply(
                lambda s: f"{get_status_emoji(s)} {s.upper()}"
            )

            # Select and rename columns
            display_df = injury_df[[
                'player_name', 'team_name', 'status_display',
                'injury_type', 'injury_date', 'expected_return_date',
                'source', 'confidence', 'notes'
            ]]
            display_df.columns = [
                'Player', 'Team', 'Status', 'Injury Type',
                'Injury Date', 'Expected Return',
                'Source', 'Conf', 'Notes'
            ]

            # Format confidence as percentage
            display_df['Conf'] = display_df['Conf'].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "N/A")
            display_df['Injury Type'] = display_df['Injury Type'].fillna('N/A')
            display_df['Notes'] = display_df['Notes'].fillna('')

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

                    # Backup to S3 after injury status change
                    try:
                        storage = s3_storage.S3PredictionStorage()
                        if storage.is_connected():
                            backup_success, backup_message = storage.upload_database(db_path)
                            if backup_success:
                                st.sidebar.info("☁️ Database backed up to S3")
                            else:
                                st.sidebar.warning(f"⚠️ S3 backup failed: {backup_message}")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ S3 backup error: {str(e)}")

                    # Clear cached database connection to force fresh query
                    st.cache_resource.clear()

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

                # Helper function for status display
                def format_status_option(status):
                    emoji_map = {
                        'out': '🔴',
                        'doubtful': '🟠',
                        'questionable': '🟡',
                        'probable': '🟢',
                        'day-to-day': '🔵'
                    }
                    return f"{emoji_map.get(status, '⚪')} {status.upper()}"

                injury_status = st.selectbox(
                    "Injury Status",
                    options=['out', 'doubtful', 'questionable', 'probable', 'day-to-day'],
                    format_func=format_status_option,
                    index=0,  # Default to 'out'
                    key="injury_status_selector"
                )

                injury_type_input = st.text_input(
                    "Injury Type",
                    placeholder="e.g., Ankle Sprain, Rest, Illness, Back Soreness",
                    key="injury_type_input"
                )

                return_date = st.date_input(
                    "Expected return date (optional)",
                    value=None,
                    key="return_date_input"
                )

                notes = st.text_area(
                    "Additional Notes",
                    placeholder="Optional: Additional details about the injury",
                    key="injury_notes_input"
                )

                if st.button("➕ Add to Injury List", type="primary", key="add_injury_btn"):
                    player_id, player_name, team_name = selected_player

                    injury_id = ia.add_to_injury_list(
                        injury_conn,
                        player_id=player_id,
                        player_name=player_name,
                        team_name=team_name,
                        status=injury_status,
                        injury_type=injury_type_input if injury_type_input else None,
                        expected_return_date=str(return_date) if return_date else None,
                        notes=notes if notes else None,
                        source='manual'
                    )

                    if injury_id:
                        st.success(f"✅ Added {player_name} to injury list")

                        # Backup to S3 after injury status change
                        try:
                            storage = s3_storage.S3PredictionStorage()
                            if storage.is_connected():
                                backup_success, backup_message = storage.upload_database(db_path)
                                if backup_success:
                                    st.sidebar.info("☁️ Database backed up to S3")
                                else:
                                    st.sidebar.warning(f"⚠️ S3 backup failed: {backup_message}")
                        except Exception as e:
                            st.sidebar.warning(f"⚠️ S3 backup error: {str(e)}")

                        # Clear cached database connection to force fresh query
                        st.cache_resource.clear()

                        st.rerun()
                    else:
                        st.warning(f"⚠️ {player_name} is already on the active injury list")
            else:
                st.error("No players found in database")

        except Exception as e:
            st.error(f"Error loading players: {e}")

    st.divider()

    # DIAGNOSTIC: Search for specific player
    with st.expander("🔍 Debug: Search Specific Player", expanded=False):
        st.caption("Search for a player by name to see their exact database record (ignores all filters)")
        search_name = st.text_input("Player name", placeholder="e.g., Giannis", key="debug_search_player")

        if search_name and len(search_name) >= 3:
            try:
                cursor = injury_conn.cursor()
                cursor.execute("""
                    SELECT player_id, player_name, team_name, status,
                           injury_type, expected_return_date, source,
                           confidence, injury_date, updated_at
                    FROM injury_list
                    WHERE player_name LIKE ?
                """, (f'%{search_name}%',))

                results = cursor.fetchall()
                if results:
                    st.success(f"Found {len(results)} record(s):")
                    for row in results:
                        st.write("---")
                        st.write(f"**Player:** {row[1]}")
                        st.write(f"**Team:** {row[2]}")
                        st.write(f"**Status:** `{row[3]}` ← Check if this matches your status filter!")
                        st.write(f"**Injury Type:** {row[4] or 'N/A'}")
                        st.write(f"**Expected Return:** `{row[5]}` ← Check format!")
                        st.write(f"**Source:** {row[6]}")
                        st.write(f"**Confidence:** {row[7]}")
                        st.write(f"**Injury Date:** {row[8]}")
                        st.write(f"**Updated:** {row[9]}")

                        # Check why it might be filtered
                        today = date.today().strftime('%Y-%m-%d')
                        if row[5]:  # has return date
                            if row[5] < today:
                                st.error(f"⚠️ FILTERED OUT: Return date ({row[5]}) < today ({today})")
                            else:
                                st.info(f"✓ Return date ({row[5]}) >= today ({today})")
                else:
                    st.warning(f"No records found for '{search_name}'")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    # Auto-Fetch Section
    st.subheader("📡 Auto-Fetch Current Injury Reports")
    st.caption("Fetch latest injury data from balldontlie.io API and sync to database")

    # Display lock status
    try:
        cursor = injury_conn.cursor()
        cursor.execute("SELECT locked, locked_at, locked_by FROM injury_fetch_lock WHERE lock_id = 1")
        result = cursor.fetchone()
        if result:
            lock_status = "locked" if result[0] else "unlocked"
            st.info(f"🔒 Lock Status: {lock_status.upper()} | Last: {result[1] or 'Never'}")
    except Exception:
        # Lock table will be created on first run
        pass

    if st.button("🔄 Fetch Now", type="primary", use_container_width=True, key="manual_fetch_btn"):
        with st.spinner("Fetching injury reports from API..."):
            try:
                import fetch_injury_data
                updated, new, skipped, removed, errors = fetch_injury_data.fetch_current_injuries(injury_conn)

                # Show results
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Updated", updated, delta=None)
                col_b.metric("New", new, delta=None)
                col_c.metric("Returned", removed, delta=None, delta_color="normal")
                col_d.metric("Skipped", skipped, delta=None)

                if updated > 0 or new > 0 or removed > 0:
                    parts = []
                    if updated > 0:
                        parts.append(f"{updated} updated")
                    if new > 0:
                        parts.append(f"{new} new")
                    if removed > 0:
                        parts.append(f"{removed} returned")
                    st.success(f"✅ Injury data refreshed! {', '.join(parts)}")

                    # Backup to S3 after successful fetch
                    try:
                        storage = s3_storage.S3PredictionStorage()
                        if storage.is_connected():
                            backup_success, backup_message = storage.upload_database(db_path)
                            if backup_success:
                                st.sidebar.info("☁️ Database backed up to S3")
                    except:
                        pass  # Silent fail for S3 backup

                    # Clear cache and refresh
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.info("ℹ️ No updates needed. All injury data is current.")

                if errors:
                    with st.expander(f"⚠️ {len(errors)} Warnings", expanded=False):
                        for error in errors:
                            st.warning(error)

            except Exception as e:
                st.error(f"❌ Auto-fetch failed: {e}")
                st.caption("You can still add injuries manually using the form above.")

    # Clear duplicate records button
    if st.button("🧹 Clean Up Duplicate Records", help="Remove old duplicate injury statuses for same player", key="cleanup_dupes_btn"):
        try:
            cursor = injury_conn.cursor()

            # Find duplicates
            cursor.execute("""
                SELECT player_id, player_name, COUNT(*) as cnt
                FROM injury_list
                GROUP BY player_id
                HAVING COUNT(*) > 1
            """)
            duplicates = cursor.fetchall()

            if not duplicates:
                st.info("✓ No duplicate records found. Database is clean!")
            else:
                deleted_total = 0
                for player_id, player_name, count in duplicates:
                    # Keep most recent, delete older ones
                    cursor.execute("""
                        SELECT injury_id
                        FROM injury_list
                        WHERE player_id = ?
                        ORDER BY updated_at DESC
                    """, (player_id,))

                    all_ids = [row[0] for row in cursor.fetchall()]
                    keep_id = all_ids[0]
                    delete_ids = all_ids[1:]

                    if delete_ids:
                        placeholders = ','.join('?' * len(delete_ids))
                        cursor.execute(f"DELETE FROM injury_list WHERE injury_id IN ({placeholders})", delete_ids)
                        deleted_total += cursor.rowcount

                injury_conn.commit()
                st.success(f"✅ Cleaned up {len(duplicates)} players with duplicate records ({deleted_total} old records deleted)")

                # Backup to S3
                try:
                    storage = s3_storage.S3PredictionStorage()
                    if storage.is_connected():
                        storage.upload_database(db_path)
                        st.sidebar.info("☁️ Database backed up to S3")
                except:
                    pass

                st.rerun()

        except Exception as e:
            st.error(f"Error cleaning duplicates: {e}")

    # Clear lock button (for stuck locks)
    if st.button("🔓 Clear Fetch Lock", help="Clear the fetch cooldown if stuck", key="clear_lock_btn"):
        try:
            cursor = injury_conn.cursor()
            cursor.execute("UPDATE injury_fetch_lock SET locked = 0, locked_at = NULL WHERE lock_id = 1")
            injury_conn.commit()
            st.success("✅ Fetch lock cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing lock: {e}")

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

    # Refresh Predictions Section
    st.divider()
    st.subheader("🔄 Refresh Predictions")
    st.caption("Remove OUT players from existing predictions and recalculate teammate projections")

    # Get today's date
    today = date.today().strftime('%Y-%m-%d')

    try:
        # Get refresh status
        status = pr.get_refresh_status(today, injury_conn)

        if status['predictions_count'] > 0:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**{status['predictions_count']} predictions** logged for {today}")

                if status['last_refreshed']:
                    st.caption(f"Last refreshed: {status['last_refreshed']} ({status['refresh_count']} time(s))")

            with col2:
                if status['needs_refresh']:
                    st.metric("OUT players with predictions", len(status['out_players_with_predictions']))
                else:
                    st.success("✅ All clean")

            # Show OUT players who still have predictions
            if status['needs_refresh']:
                st.warning(f"⚠️ {len(status['out_players_with_predictions'])} OUT player(s) still have predictions")

                out_players_df = pd.DataFrame(status['out_players_with_predictions'])
                st.dataframe(
                    out_players_df[['player_name', 'team_name', 'projected_ppg']],
                    column_config={
                        'player_name': 'Player',
                        'team_name': 'Team',
                        'projected_ppg': st.column_config.NumberColumn('Projected PPG', format="%.1f")
                    },
                    hide_index=True,
                    use_container_width=True
                )

                # Refresh button
                if st.button("🔄 Refresh Predictions for Today", type="primary", use_container_width=True):
                    with st.spinner("Regenerating predictions..."):
                        try:
                            result = pr.refresh_predictions_for_date(today, injury_conn)

                            if result['error']:
                                st.error(f"❌ Refresh failed: {result['error']}")
                            else:
                                st.success(f"""
✅ Predictions refreshed successfully!
- **{result['removed']}** predictions removed (OUT players)
- **{result['adjusted']}** predictions adjusted (teammates)
- **{result['skipped']}** skipped (insufficient data)
                                """)

                                if result['affected_players']:
                                    with st.expander("Affected Players"):
                                        for player in result['affected_players']:
                                            st.write(f"- {player}")

                                # Upload to S3 after successful refresh
                                storage = s3_storage.S3PredictionStorage()
                                if storage.is_connected():
                                    success, message = storage.upload_database(db_path)
                                    if success:
                                        st.info(f"☁️ {message}")

                                st.rerun()  # Refresh UI to show updated status

                        except Exception as e:
                            st.error(f"❌ Refresh failed: {e}")

                # Show explanation
                with st.expander("ℹ️ How Refresh Works"):
                    st.markdown("""
**What happens when you refresh:**

1. **Delete OUT Player Predictions** - Removes projections for players marked as OUT
2. **Apply Injury Adjustments** - Redistributes expected points to teammates based on historical data
3. **Update Audit Trail** - Tracks when and why predictions were refreshed

**When to use refresh:**
- You generate predictions in the morning with all players healthy
- Later in the day, injury news comes out (e.g., player ruled OUT at 4 PM)
- Mark the player as OUT in the injury management section above
- Click "Refresh Predictions" to update today's projections

**Note:** Teammate boosts are calculated using historical performance when the injured player was absent (min 3 games), capped at 25% increase.
                    """)
            else:
                st.info("✅ No OUT players have predictions. Your projections are current!")

                # Show active injuries for context
                active_injuries = ia.get_active_injuries(injury_conn, check_return_dates=True)
                if active_injuries:
                    st.caption(f"Active injuries: {len(active_injuries)} player(s) (already filtered from predictions)")
        else:
            st.info(f"No predictions logged for {today} yet. Generate predictions in the 'Today's Games' tab first.")

    except Exception as e:
        st.error(f"Could not load refresh status: {e}")

# Admin Panel tab --------------------------------------------------------
if selected_page == "Admin Panel":
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

    # Position Data Management Section
    st.subheader("🏀 Player Position Data")
    st.write("Populate Guard/Forward/Center positions from NBA API for position-specific PPM analysis")

    try:
        admin_conn = get_connection(str(db_path))
        cursor = admin_conn.cursor()

        # MIGRATION: Add position column if it doesn't exist
        cursor.execute("PRAGMA table_info(players)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'position' not in columns:
            st.info("🔧 Adding position column to players table...")
            cursor.execute("ALTER TABLE players ADD COLUMN position TEXT")
            admin_conn.commit()
            st.success("✅ Position column added successfully!")

        # Check current position data status
        cursor.execute('SELECT COUNT(*) FROM players')
        total_players = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM players WHERE position IS NOT NULL AND position != ""')
        players_with_positions = cursor.fetchone()[0]

        # Get position distribution
        cursor.execute('''
            SELECT position, COUNT(*) as count
            FROM players
            WHERE position IS NOT NULL AND position != ""
            GROUP BY position
            ORDER BY count DESC
        ''')
        position_dist = cursor.fetchall()

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Total Players", total_players)
            st.metric("Players with Position Data", f"{players_with_positions}/{total_players}")

            if position_dist:
                st.write("**Current Distribution:**")
                for pos, count in position_dist:
                    st.write(f"- {pos}: {count}")
            else:
                st.warning("⚠️ No position data populated yet")

        with col4:
            st.write("**Populate Position Data:**")
            st.write("This will:")
            st.write("- Fetch positions from NBA API for all players")
            st.write("- Normalize to Guard/Forward/Center")
            st.write("- Enable position-specific PPM analysis")
            st.write("- Take ~5-10 minutes for ~600 players")

            if st.button("🎯 Populate Player Positions", type="secondary", use_container_width=True):
                import time
                from nba_api.stats.endpoints import commonplayerinfo
                import pandas as pd

                def normalize_position(position: str) -> str:
                    """Normalize NBA API positions to Guard/Forward/Center."""
                    if not position:
                        return ""
                    pos_lower = position.lower()
                    if 'guard' in pos_lower:
                        return "Guard"
                    if 'center' in pos_lower:
                        return "Center"
                    if 'forward' in pos_lower:
                        return "Forward"
                    return ""

                with st.spinner("Fetching position data from NBA API..."):
                    try:
                        # Get all unique players from player_game_logs who don't have positions
                        query = """
                            SELECT DISTINCT pgl.player_id, pgl.player_name
                            FROM player_game_logs pgl
                            LEFT JOIN players p ON pgl.player_id = p.player_id
                            WHERE p.position IS NULL OR p.position = ''
                            ORDER BY pgl.player_name
                        """
                        players_df = pd.read_sql_query(query, admin_conn)

                        if players_df.empty:
                            st.success("✅ All players already have position data!")
                        else:
                            progress_text = st.empty()
                            progress_bar = st.progress(0)
                            success_count = 0
                            error_count = 0
                            total = len(players_df)

                            progress_text.text(f"Processing {total} players...")

                            for idx, row in players_df.iterrows():
                                player_id = row['player_id']
                                player_name = row['player_name']

                                try:
                                    # Fetch player info from NBA API
                                    player_info = commonplayerinfo.CommonPlayerInfo(
                                        player_id=player_id,
                                        timeout=30
                                    )

                                    df = player_info.get_data_frames()[0]

                                    if not df.empty and 'POSITION' in df.columns:
                                        raw_position = df['POSITION'].iloc[0]
                                        normalized_position = normalize_position(raw_position)

                                        if normalized_position:
                                            # Update database
                                            cursor.execute(
                                                "UPDATE players SET position = ? WHERE player_id = ?",
                                                (normalized_position, player_id)
                                            )
                                            admin_conn.commit()
                                            success_count += 1

                                    # Rate limiting
                                    time.sleep(0.6)

                                    # Update progress
                                    progress = (idx + 1) / total
                                    progress_bar.progress(progress)
                                    progress_text.text(f"Processing {idx + 1}/{total}: {player_name} ({success_count} successful)")

                                except Exception as e:
                                    error_count += 1
                                    time.sleep(1.2)  # Double delay after error

                            progress_bar.empty()
                            progress_text.empty()

                            # Show summary
                            st.success(f"✅ Position data population complete!")
                            st.info(f"📊 Successfully updated: **{success_count}** players")
                            if error_count > 0:
                                st.warning(f"⚠️ Errors: {error_count} players")

                            # Auto-refresh to show updated stats
                            st.info("🔄 Refreshing in 2 seconds to show updated distribution...")
                            time.sleep(2)
                            st.rerun()

                    except Exception as e:
                        st.error(f"❌ Position population failed: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details", expanded=True):
                            st.code(traceback.format_exc(), language="python")

    except Exception as e:
        st.error(f"Error checking position data: {e}")

# ============================================================================
# TOURNAMENT STRATEGY TAB - PHASE 1: BASIC CEILING ANALYSIS
# ============================================================================
if selected_page == "Tournament Strategy":
    st.header("🏆 Tournament Strategy - Winner-Take-All")
    st.caption("Ceiling-focused player selection for 3-player tournaments vs 2,500 opponents")

    # Position & GPP Factors Legend
    with st.expander("ℹ️ GPP Score Breakdown & Position Guide", expanded=False):
        st.markdown("### 🎯 How GPP Score is Calculated (Max: 110 points)")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Component 1: Ceiling Base (0-50 pts, 45%)**")
            st.markdown("""
            - 50+ PPG ceiling: **50 pts** (elite)
            - 45-50 ceiling: **46 pts** (monster)
            - 40-45 ceiling: **42 pts** (huge)
            - 35-40 ceiling: **35 pts** (strong)
            - 30-35 ceiling: **25 pts** (decent)
            - <30 ceiling: *scaled down*

            *Ceiling is the #1 driver - GPP winners need 40-50+ upside*
            """)

            st.markdown("**Component 2: Hot Streak (-5 to +20 pts, 18%)**")
            st.markdown("""
            - L5 +25% above avg: **+20 pts** (ON FIRE)
            - L5 +15-25%: **+15 pts** (very hot)
            - L5 +5-15%: **+10 pts** (trending up)
            - L5 within ±5%: **+5 pts** (steady)
            - L5 -5 to -15%: **0 pts** (slightly cool)
            - L5 -15%+: **-5 pts** (cold streak)
            """)

        with col2:
            st.markdown("**Component 3: Variance Bonus (0-15 pts, 14%)**")
            st.markdown("""
            *Position-specific PPM ceiling factor (P90/Avg):*
            - 1.15+ ceiling factor: **15 pts** (elite variance)
            - 1.12-1.15: **10 pts** (high variance)
            - 1.10-1.12: **5 pts** (above avg)
            - <1.10: **0 pts**

            *High variance = defense allows occasional blowup games*
            """)

            st.markdown("**Component 4: Matchup History (0-10 pts, 9%)**")
            st.markdown("""
            - Excellent history: **10 pts**
            - Good history: **7 pts**
            - Neutral: **3 pts**
            - Difficult: **0 pts**
            - Avoid: **-5 pts**
            """)

            st.markdown("**Component 5: Defense Quality (±5 pts, 5%)**")
            st.markdown("""
            - Def Rating 118+: **+5 pts** (weak defense)
            - Def Rating 114-118: **+3 pts**
            - Def Rating 106-: **-3 pts** (elite defense)
            """)

        with col3:
            st.markdown("**Component 6: Injury Beneficiary (0-12 pts, 11%)**")
            st.markdown("""
            - Star teammate out (+5 PPG boost): **12 pts** 🔥
            - Key player out (+3-5 PPG): **8 pts**
            - Rotation player out (+1.5-3 PPG): **5 pts**

            *Low ownership + upside spike = GPP gold*
            """)

            st.markdown("**Component 7: Position Exploit (0-10 pts, 9%)**")
            st.markdown("""
            - 🎯🔥 **SEVERE** (15%+ worse): **10 pts**
            - 🎯 **Moderate** (12-15% worse): **7 pts**
            - 🎯 **Minor** (8-12% worse): **4 pts**

            *Defense is weak specifically vs this position*
            """)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Position Categories:**")
            st.markdown("""
            - **Guard**: Point Guards, Shooting Guards, Guard-Forwards
            - **Forward**: Small Forwards, Power Forwards
            - **Center**: Centers, Forward-Centers
            """)

        with col_b:
            st.markdown("**Example GPP Score Breakdown:**")
            st.markdown("""
            **Zubac @ CLE: 92 points**
            - Ceiling Base (33 PPG): 35 pts
            - Hot Streak (+18%): 15 pts
            - Position Variance (1.16): 15 pts
            - Matchup History: 7 pts
            - Defense Quality: 0 pts
            - Injury Beneficiary: 0 pts
            - 🎯🔥 SEVERE Center exploit: 10 pts
            - **Total: 82 pts → ⭐⭐ Elite**
            """)

    # Database connection
    tourn_conn = get_connection(str(db_path))

    # Check if predictions table exists and has data
    try:
        cursor = tourn_conn.cursor()
        cursor.execute("""
            SELECT DISTINCT game_date
            FROM predictions
            ORDER BY game_date DESC
            LIMIT 30
        """)
        available_dates = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        st.warning("⚠️ No predictions table found. Please run the 'Today's Games' tab first to generate predictions.")
        st.info("""
        **How to get started:**
        1. Navigate to the "Today's Games" tab
        2. Click "Generate Predictions" to analyze today's matchups
        3. Come back to this tab to see ceiling-ranked players
        """)
        st.stop()

    if not available_dates:
        st.warning("📊 No predictions available yet. Run 'Today's Games' tab first.")
        st.info("""
        **How to get started:**
        1. Navigate to the "Today's Games" tab
        2. Click "Generate Predictions" to analyze today's matchups
        3. Come back to this tab to see ceiling-ranked players
        """)
        st.stop()

    # Date selector
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_date = st.selectbox(
            "Game Date",
            options=available_dates,
            index=0,
            key="tourn_date_select"
        )

    with col2:
        min_ceiling = st.slider(
            "Minimum Ceiling (PPG)",
            min_value=25,
            max_value=45,
            value=35,
            step=1,
            help="Filter for explosive scoring potential"
        )

    st.divider()

    # Query ceiling candidates (include injury data for tournament score bonus)
    # Exclude players marked as injured in injury_list
    query = """
        SELECT
            p.player_name,
            p.player_id,
            p.team_name,
            p.opponent_name,
            p.opponent_id,
            p.projected_ppg,
            p.proj_ceiling,
            p.proj_floor,
            p.dfs_score,
            p.dfs_grade,
            p.opponent_def_rating,
            p.season_avg_ppg,
            p.recent_avg_5,
            p.injury_adjusted,
            p.injury_adjustment_amount,
            (p.proj_ceiling - p.proj_floor) as upside_range,
            ROUND((p.proj_ceiling - p.proj_floor) / p.projected_ppg, 2) as variance_ratio
        FROM predictions p
        LEFT JOIN injury_list il ON p.player_id = il.player_id
            AND il.status IN ('out', 'doubtful')
            AND (il.expected_return_date IS NULL OR il.expected_return_date >= DATE('now'))
        WHERE p.game_date = ?
          AND p.proj_ceiling >= ?
          AND il.player_id IS NULL  -- Exclude injured players (OUT/DOUBTFUL)
        ORDER BY p.proj_ceiling DESC
    """

    try:
        df = pd.read_sql_query(query, tourn_conn, params=[selected_date, min_ceiling])
    except Exception as e:
        st.error(f"❌ Error querying predictions: {str(e)}")
        st.info("The predictions table may be missing required columns. Please regenerate predictions in the 'Today's Games' tab.")
        st.stop()

    # Load PPM stats for enhanced tournament analysis
    try:
        off_ppm_df, def_ppm_df, league_avg_ppm = load_ppm_stats(str(db_path), '2025-26')
        ppm_loaded = True

        # Merge defensive PPM stats by opponent_id
        df = df.merge(
            def_ppm_df[['team_id', 'avg_def_ppm', 'def_ppm_grade', 'ceiling_factor', 'std_def_ppm']],
            left_on='opponent_id',
            right_on='team_id',
            how='left'
        )
        # Drop redundant team_id column from merge
        df = df.drop(columns=['team_id'], errors='ignore')

        # Add player positions for position-specific PPM
        try:
            position_query = """
                SELECT player_id, position
                FROM players
                WHERE position IS NOT NULL AND position != ''
            """
            positions_df = pd.read_sql_query(position_query, tourn_conn)
            df = df.merge(
                positions_df,
                on='player_id',
                how='left'
            )
            df['position'] = df['position'].fillna('')  # Fill missing positions with empty string
        except Exception as pos_exc:
            df['position'] = ''  # Add empty position column if lookup fails

        st.caption(f"✅ PPM stats loaded (League Avg: {league_avg_ppm:.3f} PPM)")
    except Exception as ppm_exc:
        st.caption(f"⚠️ PPM stats unavailable: {ppm_exc}")
        def_ppm_df = None
        ppm_loaded = False
        df['position'] = ''  # Add empty position column

    # Debug: Show injured players being filtered out
    with st.expander("🔍 Debug: Players Filtered by Injury Status", expanded=False):
        debug_query = """
            SELECT
                p.player_name,
                p.team_name,
                p.proj_ceiling,
                il.status as injury_status,
                il.injury_type,
                il.injury_date,
                il.source
            FROM predictions p
            INNER JOIN injury_list il ON p.player_id = il.player_id
                AND il.status IN ('out', 'doubtful')
                AND (il.expected_return_date IS NULL OR il.expected_return_date >= DATE('now'))
            WHERE p.game_date = ?
                AND p.proj_ceiling >= ?
            ORDER BY p.proj_ceiling DESC
        """
        injured_df = pd.read_sql_query(debug_query, tourn_conn, params=[selected_date, min_ceiling])
        if injured_df.empty:
            st.success("✅ No players filtered out due to injury status")
        else:
            st.warning(f"⚠️ {len(injured_df)} player(s) excluded due to active injury:")
            st.dataframe(injured_df, use_container_width=True)

    if df.empty:
        st.info(f"No players with ceiling >= {min_ceiling} PPG found for {selected_date}")
    else:
        # Display key metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Ceiling Candidates", len(df))
        with metric_cols[1]:
            st.metric("Avg Ceiling", f"{df['proj_ceiling'].mean():.1f}")
        with metric_cols[2]:
            st.metric("Max Ceiling", f"{df['proj_ceiling'].max():.1f}")
        with metric_cols[3]:
            st.metric("Avg Variance", f"{df['variance_ratio'].mean():.2f}")

        st.divider()

        # Display ceiling candidates table
        st.subheader(f"🎯 Ceiling Candidates ({selected_date})")
        st.caption("Players ranked by explosive scoring potential (proj_ceiling)")

        # Calculate Opponent Defense Grade for each player
        def calculate_opp_def_grade_row(row):
            try:
                return get_opponent_defense_grade(
                    opp_team_id=int(row['opponent_id']),
                    opp_def_rating=row['opponent_def_rating']
                )
            except:
                return "C Neutral"

        df['opp_def_grade'] = df.apply(calculate_opp_def_grade_row, axis=1)

        # Calculate Tournament DFS Score for each player (different from cash game score)
        def calculate_tournament_score_row(row):
            try:
                # Handle injury data (may be None/NaN for older predictions)
                is_injury_adjusted = bool(row.get('injury_adjusted', False))
                injury_boost = float(row.get('injury_adjustment_amount', 0.0)) if pd.notna(row.get('injury_adjustment_amount')) else 0.0

                # Get player position (from merged position column)
                player_position = row.get('position', '')

                score, grade, explanation = calculate_daily_pick_score(
                    player_season_avg=row['season_avg_ppg'],
                    player_projection=row['projected_ppg'],
                    projection_confidence=0.75,  # Not used in tournament mode
                    matchup_rating="Neutral",  # Simplified for now
                    opp_def_rating=row['opponent_def_rating'],
                    # Tournament-specific parameters
                    proj_ceiling=row['proj_ceiling'],
                    recent_avg_5=row['recent_avg_5'],
                    opp_team_id=int(row['opponent_id']),
                    tournament_mode=True,
                    # Injury beneficiary bonus
                    injury_adjusted=is_injury_adjusted,
                    projection_boost=injury_boost,
                    # PPM Integration
                    def_ppm_df=def_ppm_df if ppm_loaded else None,
                    # Position-specific PPM
                    player_position=player_position,
                    game_date=str(selected_date),
                    conn=tourn_conn,
                    league_avg_ppm=league_avg_ppm if ppm_loaded else 0.462,
                )
                return pd.Series({'tourn_score': score, 'tourn_grade': grade, 'tourn_explanation': explanation})
            except Exception as e:
                return pd.Series({'tourn_score': 50.0, 'tourn_grade': '✓ Playable', 'tourn_explanation': ''})

        df[['tourn_score', 'tourn_grade', 'tourn_explanation']] = df.apply(calculate_tournament_score_row, axis=1)

        # ADD PROJECTION ANCHOR (15% weight)
        # Prevents low-projection players from ranking high on ceiling narrative alone
        # Formula: (Player Proj - Min Proj) / (Max Proj - Min Proj) × 15
        # This adds "plausibility gravity" - ceiling still dominates, but volume/role matters
        min_proj = df['projected_ppg'].min()
        max_proj = df['projected_ppg'].max()

        if max_proj > min_proj:  # Avoid division by zero
            # Normalize projections to 0-15 scale (15% weight)
            df['proj_anchor_score'] = ((df['projected_ppg'] - min_proj) / (max_proj - min_proj)) * 15

            # Add projection anchor to tournament score
            df['tourn_score'] = df['tourn_score'] + df['proj_anchor_score']

            # Update explanation to include projection anchor
            df['tourn_explanation'] = df.apply(
                lambda row: f"{row['tourn_explanation']}, proj anchor (+{row['proj_anchor_score']:.0f})"
                if pd.notna(row['proj_anchor_score']) else row['tourn_explanation'],
                axis=1
            )
        else:
            df['proj_anchor_score'] = 0  # Fallback if all projections are identical

        # Sort by tournament score (not ceiling) for better tournament prioritization
        df = df.sort_values('tourn_score', ascending=False)

        # Format display dataframe
        display_df = df.copy()
        display_df = display_df.rename(columns={
            'player_name': 'Player',
            'team_name': 'Team',
            'opponent_name': 'Opponent',
            'position': 'Pos',
            'projected_ppg': 'Proj PPG',
            'proj_ceiling': 'Ceiling',
            'proj_floor': 'Floor',
            'opponent_def_rating': 'Opp Def',
            'opp_def_grade': 'Opp Def Grade',
            'tourn_score': 'GPP Score',
            'tourn_grade': 'GPP Grade',
            'tourn_explanation': 'GPP Factors',
            'recent_avg_5': 'L5 Avg',
            'dfs_score': 'Cash Score',
            'dfs_grade': 'Cash Grade',
            'upside_range': 'Range',
            'variance_ratio': 'Variance',
            # PPM columns
            'avg_def_ppm': 'Opp Def PPM',
            'def_ppm_grade': 'Opp PPM Grade',
            'ceiling_factor': 'Ceiling Factor',
            'std_def_ppm': 'PPM StdDev'
        })

        # Round numeric columns
        display_df['Proj PPG'] = display_df['Proj PPG'].round(1)
        display_df['Ceiling'] = display_df['Ceiling'].round(1)
        display_df['Floor'] = display_df['Floor'].round(1)
        display_df['Opp Def'] = display_df['Opp Def'].round(1)
        display_df['L5 Avg'] = display_df['L5 Avg'].round(1)
        display_df['GPP Score'] = display_df['GPP Score'].round(0)
        display_df['Cash Score'] = display_df['Cash Score'].round(1)
        display_df['Range'] = display_df['Range'].round(1)
        # Round PPM columns (if available)
        if 'Opp Def PPM' in display_df.columns:
            display_df['Opp Def PPM'] = display_df['Opp Def PPM'].round(3)
        if 'Ceiling Factor' in display_df.columns:
            display_df['Ceiling Factor'] = display_df['Ceiling Factor'].round(3)
        if 'PPM StdDev' in display_df.columns:
            display_df['PPM StdDev'] = display_df['PPM StdDev'].round(3)

        # Select and reorder columns for display (include PPM if available)
        base_columns = [
            'Player', 'Pos', 'Team', 'Opponent', 'Ceiling', 'L5 Avg', 'Proj PPG',
            'GPP Score', 'GPP Grade', 'GPP Factors', 'Opp Def Grade'
        ]

        # Add PPM columns if they exist
        ppm_columns = []
        if 'Opp Def PPM' in display_df.columns:
            ppm_columns.extend(['Opp Def PPM', 'Opp PPM Grade', 'Ceiling Factor'])

        # Add remaining columns
        remaining_columns = ['Opp Def', 'Range', 'Variance']

        display_columns = base_columns + ppm_columns + remaining_columns

        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Insight box with updated guide
        st.info("""
        **💡 Tournament Strategy Guide:**

        **Key Columns (sorted by GPP Score):**
        - **GPP Score (0-100)**: Tournament-optimized score emphasizing ceiling, hot streaks, and variance
        - **Ceiling**: 90th percentile outcome (explosive game potential)
        - **L5 Avg**: Last 5 games average (hot streak indicator)
        - **Opp Def Grade**: Matchup quality (defense + variance)

        **GPP Score Factors:**
        1. **Ceiling (40%)**: 40-50+ ceilings get elite scores
        2. **Hot Streak (20%)**: L5 > season avg = bonus (25%+ = huge bonus)
        3. **Projection Anchor (15%)**: Volume/opportunity requirement (prevents narrative-only rankings)
        4. **Defense Variance (15%)**: High-variance defenses = bonus
        5. **Matchup History (10%)**: Historical performance vs opponent

        **GPP Grade Legend:**
        - 🔥🔥 **GPP Lock (85+)**: Must-play (high ceiling + hot streak + variance)
        - 🔥 **Core Play (75-84)**: Build lineups around these players
        - ⭐ **Strong (65-74)**: Solid tournament picks
        - ✓ **Playable (55-64)**: Pivot/contrarian options
        - ⚠️ **Punt/Fade (<55)**: Low ceiling potential

        **Opp Def Grade:** A+/A/A- = elite variance, B = good, C = average, D = avoid

        **Target**: Combined ceiling 105+, at least 2 players with GPP Score 70+
        """)

        # Add grade distribution summary
        grade_counts = display_df['Opp Def Grade'].value_counts()
        if not grade_counts.empty:
            st.caption("**Opp Def Grade Distribution:**")
            grade_summary = " | ".join([f"{grade}: {count}" for grade, count in grade_counts.head(5).items()])
            st.caption(grade_summary)

        st.divider()

        # ========== MULTI-LINEUP CONSTRUCTOR ==========
        st.subheader("🏗️ Multi-Lineup Portfolio Constructor")
        st.caption("Build 3 differentiated tournament lineups following optimal portfolio strategy")

        # Create player options list (sorted by GPP Score)
        player_options = display_df[['Player', 'Team', 'Opponent', 'GPP Score', 'Ceiling', 'Proj PPG']].copy()
        player_options['Display'] = player_options.apply(
            lambda x: f"{x['Player']} ({x['Team']}) - GPP {x['GPP Score']:.0f} | Ceiling {x['Ceiling']:.1f}",
            axis=1
        )
        player_list = ['[Select Player]'] + player_options['Display'].tolist()

        # Helper function to extract player name from display string
        def get_player_name(display_str):
            if display_str == '[Select Player]':
                return None
            return display_str.split(' (')[0]

        # Auto-generate optimal lineups function
        def auto_generate_lineups(df):
            """Generate optimal 3-lineup portfolio based on GPP Score rankings."""
            if len(df) < 8:
                return None, "Need at least 8 players to generate lineups"

            # Sort by GPP Score descending
            sorted_players = df.sort_values('GPP Score', ascending=False).copy()

            # Helper to find game stacks (2 players facing each other)
            def find_game_stack_partner(player_idx, df, used_players):
                """Find a player from the same game as player_idx."""
                player = df.iloc[player_idx]
                player_opp = player['Opponent']

                # Look for players on the opponent team
                for idx, candidate in df.iterrows():
                    if candidate['Player'] in used_players:
                        continue
                    if candidate['Team'] == player_opp:
                        return candidate['Display']
                return None

            # LINEUP A: Balanced Core
            # Strategy: Top 3 GPP scores, try to get 2 from same game
            lineup_a = []
            used_players_a = set()

            # Player 1: Highest GPP
            p1_display = sorted_players.iloc[0]['Display']
            lineup_a.append(p1_display)
            used_players_a.add(sorted_players.iloc[0]['Player'])

            # Player 2: Try to find game stack partner, else 2nd highest GPP
            stack_partner = find_game_stack_partner(0, sorted_players, used_players_a)
            if stack_partner:
                lineup_a.append(stack_partner)
                used_players_a.add(get_player_name(stack_partner))
            else:
                # No stack available, take 2nd highest GPP
                p2_display = sorted_players.iloc[1]['Display']
                lineup_a.append(p2_display)
                used_players_a.add(sorted_players.iloc[1]['Player'])

            # Player 3: Next highest GPP not in lineup
            for idx, row in sorted_players.iterrows():
                if row['Player'] not in used_players_a:
                    lineup_a.append(row['Display'])
                    used_players_a.add(row['Player'])
                    break

            # LINEUP B: Contrarian Leverage
            # Strategy: 0 overlap with A, players ranked 4-10 in GPP Score
            lineup_b = []
            used_players_b = set()

            # Skip players in Lineup A, take next 3 highest GPP
            for idx, row in sorted_players.iterrows():
                if row['Player'] in used_players_a:
                    continue
                if len(lineup_b) < 3:
                    lineup_b.append(row['Display'])
                    used_players_b.add(row['Player'])

            # LINEUP B CEILING CHECK: Ensure combined ceiling ≥ 100
            # Without ownership data, use ceiling threshold as scenario validator
            lineup_b_players = [get_player_name(p) for p in lineup_b]
            lineup_b_ceiling = sum(
                sorted_players[sorted_players['Player'] == p]['Ceiling'].iloc[0]
                for p in lineup_b_players if p
            )

            # If combined ceiling < 100, swap lowest-ceiling player for higher-ceiling volatility
            if lineup_b_ceiling < 100:
                # Find the "mid-ceiling drag" player (lowest ceiling in lineup)
                lineup_b_ceilings = [
                    (p, sorted_players[sorted_players['Player'] == p]['Ceiling'].iloc[0])
                    for p in lineup_b_players if p
                ]
                lineup_b_ceilings.sort(key=lambda x: x[1])  # Sort by ceiling ascending
                drag_player = lineup_b_ceilings[0][0]  # Lowest ceiling player

                # Search for high-ceiling replacement
                # Criteria: Ceiling ≥ 40, not in Lineup A, GPP score reasonable
                for idx, row in sorted_players.iterrows():
                    if row['Player'] in used_players_a or row['Player'] in used_players_b:
                        continue
                    if row['Ceiling'] >= 40 and row['Ceiling'] > lineup_b_ceilings[0][1] + 5:
                        # Swap: remove drag player, add high-ceiling player
                        lineup_b = [p for p in lineup_b if get_player_name(p) != drag_player]
                        lineup_b.append(row['Display'])
                        used_players_b.remove(drag_player)
                        used_players_b.add(row['Player'])
                        break  # Only swap once

            # LINEUP C: Volatility Play
            # Strategy: 1 overlap with A (highest GPP anchor), 2 different from A/B
            lineup_c = []
            used_players_c = set()

            # Player 1: Same as Lineup A's anchor (highest GPP)
            lineup_c.append(lineup_a[0])
            used_players_c.add(sorted_players.iloc[0]['Player'])

            # Players 2-3: Next available not in A or B
            all_used = used_players_a | used_players_b
            for idx, row in sorted_players.iterrows():
                if row['Player'] in all_used:
                    continue
                if len(lineup_c) < 3:
                    lineup_c.append(row['Display'])
                    used_players_c.add(row['Player'])

            # If we couldn't fill Lineup C (not enough players), take from B
            if len(lineup_c) < 3:
                for player_display in lineup_b:
                    player_name = get_player_name(player_display)
                    if player_name not in used_players_c and len(lineup_c) < 3:
                        lineup_c.append(player_display)
                        used_players_c.add(player_name)

            # LINEUP D (OPTIONAL): Game Stack / Correlation Bet
            # Only activate when there's a clear high-total game outlier
            lineup_d = []
            lineup_d_game = None
            lineup_d_debug_info = {}  # Store diagnostic info for debug display

            # Step 1: Group players by game and calculate projected game totals
            game_totals = {}
            for idx, row in sorted_players.iterrows():
                # Create game identifier (team_a vs team_b, sorted alphabetically)
                teams = tuple(sorted([row['Team'], row['Opponent']]))
                if teams not in game_totals:
                    game_totals[teams] = {
                        'projected_total': 0,
                        'ceiling_total': 0,
                        'players': []
                    }
                game_totals[teams]['projected_total'] += row['Proj PPG']
                game_totals[teams]['ceiling_total'] += row['Ceiling']
                game_totals[teams]['players'].append(row)

            # Step 2: Find if there's a clear high-total outlier
            if len(game_totals) >= 2:  # Need at least 2 games to compare
                # Calculate median projected total
                all_totals = [g['projected_total'] for g in game_totals.values()]
                median_total = sorted(all_totals)[len(all_totals) // 2]

                # Find highest-total game
                highest_game = max(game_totals.items(), key=lambda x: x[1]['projected_total'])
                highest_total = highest_game[1]['projected_total']

                # Calculate threshold percentage
                threshold_pct = ((highest_total / median_total) - 1) * 100 if median_total > 0 else 0

                # Store debug info
                lineup_d_debug_info = {
                    'game_totals': {f"{t[0]} vs {t[1]}": round(g['projected_total'], 1)
                                   for t, g in sorted(game_totals.items(),
                                                      key=lambda x: x[1]['projected_total'],
                                                      reverse=True)},
                    'median_total': round(median_total, 1),
                    'highest_total': round(highest_total, 1),
                    'highest_game': f"{highest_game[0][0]} vs {highest_game[0][1]}",
                    'threshold_pct': round(threshold_pct, 1),
                    'required_pct': 15.0,
                    'passed_threshold': threshold_pct >= 15.0
                }

                # Check if it's a clear outlier (15%+ above median)
                if highest_total >= median_total * 1.15:
                    # This is a qualifying high-total game
                    lineup_d_game = highest_game[0]  # (Team_A, Team_B)
                    game_players = highest_game[1]['players']

                    # Sort game players by ceiling (descending)
                    game_players_sorted = sorted(game_players, key=lambda x: x['Ceiling'], reverse=True)

                    # Build Lineup D with 2-3 top players from this game
                    for player_row in game_players_sorted[:3]:
                        if len(lineup_d) < 3:
                            lineup_d.append(player_row['Display'])

                    # Validate ceiling threshold (must be 110+)
                    lineup_d_ceiling = sum(
                        p['Ceiling'] for p in game_players_sorted[:len(lineup_d)]
                    )
                    lineup_d_debug_info['combined_ceiling'] = round(lineup_d_ceiling, 1)
                    lineup_d_debug_info['ceiling_threshold_met'] = lineup_d_ceiling >= 110

                    if lineup_d_ceiling < 110:
                        # Doesn't meet ceiling threshold, clear lineup
                        lineup_d = []
                        lineup_d_game = None

            return (lineup_a, lineup_b, lineup_c, lineup_d, lineup_d_game, lineup_d_debug_info), None

        # Auto-generate button
        button_col1, button_col2 = st.columns([1, 3])

        # Initialize widget refresh counter if needed
        if 'lineup_refresh_counter' not in st.session_state:
            st.session_state.lineup_refresh_counter = 0

        with button_col1:
            if st.button("🤖 Auto-Generate Optimal Lineups", type="primary", use_container_width=True, key='auto_gen_btn'):
                lineups, error = auto_generate_lineups(player_options)
                if error:
                    st.error(error)
                else:
                    st.session_state.lineup_a = lineups[0]
                    st.session_state.lineup_b = lineups[1]
                    st.session_state.lineup_c = lineups[2]
                    st.session_state.lineup_d = lineups[3]  # Optional game stack
                    st.session_state.lineup_d_game = lineups[4]  # Game matchup info
                    st.session_state.lineup_d_debug_info = lineups[5]  # Debug info
                    st.session_state.lineup_refresh_counter += 1
                    st.rerun()

        with button_col2:
            if st.button("🔄 Clear All Lineups", use_container_width=True, key='clear_btn'):
                st.session_state.lineup_a = []
                st.session_state.lineup_b = []
                st.session_state.lineup_c = []
                st.session_state.lineup_d = []
                st.session_state.lineup_d_game = None
                st.session_state.lineup_d_debug_info = {}
                st.session_state.lineup_refresh_counter += 1
                st.rerun()

        st.divider()

        # Initialize session state for lineups if needed
        if 'lineup_a' not in st.session_state:
            st.session_state.lineup_a = []
        if 'lineup_b' not in st.session_state:
            st.session_state.lineup_b = []
        if 'lineup_c' not in st.session_state:
            st.session_state.lineup_c = []
        if 'lineup_d' not in st.session_state:
            st.session_state.lineup_d = []
        if 'lineup_d_game' not in st.session_state:
            st.session_state.lineup_d_game = None
        if 'lineup_d_debug_info' not in st.session_state:
            st.session_state.lineup_d_debug_info = {}

        # Display generated lineups (3 or 4 columns depending on Lineup D activation)
        if st.session_state.lineup_a and st.session_state.lineup_b and st.session_state.lineup_c:
            # Check if Lineup D is activated
            has_lineup_d = st.session_state.lineup_d and len(st.session_state.lineup_d) > 0

            if has_lineup_d:
                lineup_cols = st.columns(4)
            else:
                lineup_cols = st.columns(3)

            # Lineup A - Balanced Core
            with lineup_cols[0]:
                st.markdown("### 🎯 Lineup A: Balanced Core")
                st.caption("Top GPP scores, 2 from same game, max 1 chalk")
                for i, player in enumerate(st.session_state.lineup_a, 1):
                    st.markdown(f"**{i}.** {player}")

            # Lineup B - Contrarian Leverage
            with lineup_cols[1]:
                st.markdown("### 🎲 Lineup B: Contrarian")
                st.caption("0 overlap with A, injury beneficiaries, low owned")
                for i, player in enumerate(st.session_state.lineup_b, 1):
                    st.markdown(f"**{i}.** {player}")

            # Lineup C - Volatility Play
            with lineup_cols[2]:
                st.markdown("### ⚡ Lineup C: Volatility")
                st.caption("1 overlap with A (anchor), high variance picks")
                for i, player in enumerate(st.session_state.lineup_c, 1):
                    st.markdown(f"**{i}.** {player}")

            # Lineup D - Game Stack (OPTIONAL - only shows when activated)
            if has_lineup_d:
                with lineup_cols[3]:
                    game_label = ""
                    if st.session_state.lineup_d_game:
                        team_a, team_b = st.session_state.lineup_d_game
                        game_label = f" ({team_a} vs {team_b})"
                    st.markdown(f"### 🔥 Lineup D: Game Stack{game_label}")
                    st.caption("High-total game outlier, correlation bet")
                    for i, player in enumerate(st.session_state.lineup_d, 1):
                        st.markdown(f"**{i}.** {player}")
        else:
            st.info("👆 Click **Auto-Generate Optimal Lineups** to build your tournament portfolio")

        # Debug: Show Lineup D detection status
        with st.expander("🔍 Debug: Lineup D Status", expanded=False):
            st.write("**Session State Values:**")
            st.write(f"- lineup_d exists: {bool(st.session_state.lineup_d)}")
            st.write(f"- lineup_d length: {len(st.session_state.lineup_d) if st.session_state.lineup_d else 0}")
            st.write(f"- lineup_d_game: {st.session_state.lineup_d_game}")
            st.write(f"- lineup_d content: {st.session_state.lineup_d}")

            # Show detailed game total analysis
            if st.session_state.lineup_d_debug_info:
                st.divider()
                st.write("**Game Total Analysis:**")
                debug_info = st.session_state.lineup_d_debug_info

                # Show all game totals (sorted highest to lowest)
                if 'game_totals' in debug_info:
                    st.write("Game Totals (Projected PPG):")
                    for game, total in debug_info['game_totals'].items():
                        st.write(f"  • {game}: **{total}** PPG")

                st.write(f"- Median Total: **{debug_info.get('median_total', 'N/A')}** PPG")
                st.write(f"- Highest Total: **{debug_info.get('highest_total', 'N/A')}** PPG ({debug_info.get('highest_game', 'N/A')})")
                st.write(f"- Threshold: **{debug_info.get('threshold_pct', 0)}%** above median (need **{debug_info.get('required_pct', 15)}%**)")

                if debug_info.get('passed_threshold'):
                    st.success(f"✅ Passed threshold! ({debug_info.get('threshold_pct', 0)}% ≥ 15%)")
                    if 'combined_ceiling' in debug_info:
                        ceiling_met = debug_info.get('ceiling_threshold_met', False)
                        ceiling_val = debug_info.get('combined_ceiling', 0)
                        if ceiling_met:
                            st.success(f"✅ Ceiling threshold met! ({ceiling_val} ≥ 110 PPG)")
                        else:
                            st.error(f"❌ Ceiling threshold NOT met ({ceiling_val} < 110 PPG)")
                else:
                    st.warning(f"❌ Did NOT pass threshold ({debug_info.get('threshold_pct', 0)}% < 15%)")
                    gap = 15.0 - debug_info.get('threshold_pct', 0)
                    st.caption(f"Need {gap:.1f}% more separation for Lineup D to activate")

            if st.session_state.lineup_d and len(st.session_state.lineup_d) > 0:
                st.divider()
                st.success("✅ Lineup D is ACTIVE (should be showing in 4th column)")
            else:
                st.divider()
                st.warning("❌ Lineup D is INACTIVE (showing 3 columns only)")
                st.caption("Reasons Lineup D might not activate:")
                st.caption("• No game has 15%+ higher total than median")
                st.caption("• Combined ceiling of top 3 players < 110 PPG")
                st.caption("• Fewer than 2 games on slate")

        st.divider()

        # ========== LINEUP VALIDATION & METRICS ==========
        st.subheader("📊 Portfolio Analysis")

        # Extract player names from lineup displays
        lineup_a_players = [get_player_name(p) for p in st.session_state.lineup_a]
        lineup_b_players = [get_player_name(p) for p in st.session_state.lineup_b]
        lineup_c_players = [get_player_name(p) for p in st.session_state.lineup_c]

        # Calculate metrics for each lineup
        def calculate_lineup_metrics(player_names, display_df):
            if None in player_names or len(set(player_names)) < 3:
                return None

            metrics = {
                'players': [],
                'total_ceiling': 0,
                'total_proj': 0,
                'avg_gpp_score': 0,
                'teams': [],
                'opponents': [],
            }

            for pname in player_names:
                player_row = display_df[display_df['Player'] == pname]
                if player_row.empty:
                    return None

                p = player_row.iloc[0]
                metrics['players'].append(pname)
                metrics['total_ceiling'] += p['Ceiling']
                metrics['total_proj'] += p['Proj PPG']
                metrics['avg_gpp_score'] += p['GPP Score']
                metrics['teams'].append(p['Team'])
                metrics['opponents'].append(p['Opponent'])

            metrics['avg_gpp_score'] /= 3
            return metrics

        lineup_a_metrics = calculate_lineup_metrics(lineup_a_players, display_df)
        lineup_b_metrics = calculate_lineup_metrics(lineup_b_players, display_df)
        lineup_c_metrics = calculate_lineup_metrics(lineup_c_players, display_df)

        # Calculate slate average projection for relative projection signal
        slate_avg_proj = display_df['Proj PPG'].mean() if not display_df.empty else 0

        # Display lineup summaries
        summary_cols = st.columns(3)

        with summary_cols[0]:
            st.markdown("#### Lineup A Summary")
            if lineup_a_metrics:
                st.metric("Combined Ceiling", f"{lineup_a_metrics['total_ceiling']:.1f} PPG")
                st.metric("Combined Proj", f"{lineup_a_metrics['total_proj']:.1f} PPG")
                st.metric("Avg GPP Score", f"{lineup_a_metrics['avg_gpp_score']:.0f}")

                # EDIT #1: Relative Projection Signal
                if slate_avg_proj > 0:
                    lineup_a_avg_proj = lineup_a_metrics['total_proj'] / 3
                    proj_vs_slate = ((lineup_a_avg_proj - slate_avg_proj) / slate_avg_proj) * 100
                    proj_indicator = "🟢" if proj_vs_slate >= 5 else "🟡" if proj_vs_slate >= -5 else "🔴"
                    st.caption(f"{proj_indicator} **Proj vs Slate:** {proj_vs_slate:+.1f}%")

                # Check for game stack
                opponent_counts = {}
                for opp in lineup_a_metrics['opponents']:
                    opponent_counts[opp] = opponent_counts.get(opp, 0) + 1
                game_stack = any(count >= 2 for count in opponent_counts.values())
                st.caption(f"{'✅' if game_stack else '⚠️'} Game Stack: {'YES' if game_stack else 'NO'}")

                # EDIT #2: Win Condition Label
                st.info("🎯 **Wins if:** Slate plays normal, stars hit ceiling")
            else:
                st.caption("⚠️ Select 3 different players")

        with summary_cols[1]:
            st.markdown("#### Lineup B Summary")
            if lineup_b_metrics:
                st.metric("Combined Ceiling", f"{lineup_b_metrics['total_ceiling']:.1f} PPG")
                st.metric("Combined Proj", f"{lineup_b_metrics['total_proj']:.1f} PPG")
                st.metric("Avg GPP Score", f"{lineup_b_metrics['avg_gpp_score']:.0f}")

                # EDIT #1: Relative Projection Signal
                if slate_avg_proj > 0:
                    lineup_b_avg_proj = lineup_b_metrics['total_proj'] / 3
                    proj_vs_slate = ((lineup_b_avg_proj - slate_avg_proj) / slate_avg_proj) * 100
                    proj_indicator = "🟢" if proj_vs_slate >= 5 else "🟡" if proj_vs_slate >= -5 else "🔴"
                    st.caption(f"{proj_indicator} **Proj vs Slate:** {proj_vs_slate:+.1f}%")

                # Lineup B Sanity Checklist (ownership-free ceiling threshold)
                ceiling_ok = lineup_b_metrics['total_ceiling'] >= 100
                ceiling_indicator = "✅" if ceiling_ok else "⚠️"
                st.caption(f"{ceiling_indicator} **Ceiling Threshold:** {lineup_b_metrics['total_ceiling']:.1f} (target: ≥100)")

                # Check for overlap with A
                overlap_with_a = len(set(lineup_a_players) & set(lineup_b_players)) if lineup_a_metrics else 0
                st.caption(f"{'✅' if overlap_with_a == 0 else '❌'} Overlap with A: {overlap_with_a} players")

                # EDIT #2: Win Condition Label
                st.info("🎲 **Wins if:** Chalk fails, leverage hits")
            else:
                st.caption("⚠️ Select 3 different players")

        with summary_cols[2]:
            st.markdown("#### Lineup C Summary")
            if lineup_c_metrics:
                st.metric("Combined Ceiling", f"{lineup_c_metrics['total_ceiling']:.1f} PPG")
                st.metric("Combined Proj", f"{lineup_c_metrics['total_proj']:.1f} PPG")
                st.metric("Avg GPP Score", f"{lineup_c_metrics['avg_gpp_score']:.0f}")

                # EDIT #1: Relative Projection Signal
                if slate_avg_proj > 0:
                    lineup_c_avg_proj = lineup_c_metrics['total_proj'] / 3
                    proj_vs_slate = ((lineup_c_avg_proj - slate_avg_proj) / slate_avg_proj) * 100
                    proj_indicator = "🟢" if proj_vs_slate >= 5 else "🟡" if proj_vs_slate >= -5 else "🔴"
                    st.caption(f"{proj_indicator} **Proj vs Slate:** {proj_vs_slate:+.1f}%")

                # Check for overlap with A (should be exactly 1)
                overlap_with_a = len(set(lineup_a_players) & set(lineup_c_players)) if lineup_a_metrics else 0
                st.caption(f"{'✅' if overlap_with_a == 1 else '⚠️'} Overlap with A: {overlap_with_a} players")

                # EDIT #2: Win Condition Label
                st.info("⚡ **Wins if:** Game environment spikes, variance plays hit")
            else:
                st.caption("⚠️ Select 3 different players")

        st.divider()

        # Portfolio-level validation
        if lineup_a_metrics and lineup_b_metrics and lineup_c_metrics:
            st.markdown("### ✅ Portfolio Validation")

            validation_cols = st.columns(2)

            with validation_cols[0]:
                st.markdown("**Lineup Requirements:**")

                # Lineup A ceiling check
                a_ceiling_ok = lineup_a_metrics['total_ceiling'] >= 110
                st.markdown(f"{'✅' if a_ceiling_ok else '⚠️'} Lineup A ceiling ≥110: {lineup_a_metrics['total_ceiling']:.1f}")

                # Lineup B ceiling check
                b_ceiling_ok = lineup_b_metrics['total_ceiling'] >= 105
                st.markdown(f"{'✅' if b_ceiling_ok else '⚠️'} Lineup B ceiling ≥105: {lineup_b_metrics['total_ceiling']:.1f}")

                # Lineup C ceiling check
                c_ceiling_ok = lineup_c_metrics['total_ceiling'] >= 105
                st.markdown(f"{'✅' if c_ceiling_ok else '⚠️'} Lineup C ceiling ≥105: {lineup_c_metrics['total_ceiling']:.1f}")

                # B has 0 overlap with A
                b_overlap = len(set(lineup_a_players) & set(lineup_b_players))
                b_overlap_ok = b_overlap == 0
                st.markdown(f"{'✅' if b_overlap_ok else '❌'} Lineup B: 0 overlap with A ({b_overlap} found)")

            with validation_cols[1]:
                st.markdown("**Portfolio Diversity:**")

                # C has exactly 1 overlap with A
                c_overlap = len(set(lineup_a_players) & set(lineup_c_players))
                c_overlap_ok = c_overlap == 1
                st.markdown(f"{'✅' if c_overlap_ok else '⚠️'} Lineup C: 1 overlap with A ({c_overlap} found)")

                # Total unique players
                all_players = set(lineup_a_players + lineup_b_players + lineup_c_players)
                unique_count = len(all_players)
                unique_ok = 7 <= unique_count <= 8
                st.markdown(f"{'✅' if unique_ok else '⚠️'} Unique players: {unique_count} (target: 7-8)")

                # Combined ceiling across all lineups
                total_ceiling = lineup_a_metrics['total_ceiling'] + lineup_b_metrics['total_ceiling'] + lineup_c_metrics['total_ceiling']
                st.markdown(f"📊 **Total Portfolio Ceiling:** {total_ceiling:.1f} PPG")

            # Overall portfolio grade
            all_checks = [a_ceiling_ok, b_ceiling_ok, c_ceiling_ok, b_overlap_ok, c_overlap_ok, unique_ok]
            passed_checks = sum(all_checks)

            st.divider()
            if passed_checks == 6:
                st.success("🎉 **OPTIMAL PORTFOLIO** - All validation checks passed! Ready for tournament entry.")
            elif passed_checks >= 4:
                st.warning(f"⚠️ **GOOD PORTFOLIO** - {passed_checks}/6 checks passed. Review warnings above.")
            else:
                st.error(f"❌ **NEEDS IMPROVEMENT** - Only {passed_checks}/6 checks passed. Adjust lineups.")

            # EDIT #3: Player Exposure Summary
            st.divider()
            st.markdown("### 📊 Player Exposure Summary")
            st.caption("Shows how many lineups each player appears in (out of 3 total)")

            # Calculate exposure counts
            all_players_list = lineup_a_players + lineup_b_players + lineup_c_players
            exposure_counts = {}
            for player in all_players_list:
                if player:  # Skip None values
                    exposure_counts[player] = exposure_counts.get(player, 0) + 1

            # Sort by exposure count (descending), then alphabetically
            sorted_exposure = sorted(exposure_counts.items(), key=lambda x: (-x[1], x[0]))

            # Display in columns
            if sorted_exposure:
                exposure_text = []
                for player, count in sorted_exposure:
                    percentage = (count / 3) * 100
                    if count == 3:
                        emoji = "🔥"  # Full exposure (anchor player)
                    elif count == 2:
                        emoji = "⚡"  # High exposure
                    else:
                        emoji = "💫"  # Single exposure
                    exposure_text.append(f"{emoji} **{player}:** {count}/3 ({percentage:.0f}%)")

                # Display in 3 columns for better layout
                exp_cols = st.columns(3)
                for idx, text in enumerate(exposure_text):
                    with exp_cols[idx % 3]:
                        st.markdown(text)
            else:
                st.caption("No players selected yet")

st.divider()
st.caption(
    "Need more context? Re-run the builder (`python nba_to_sqlite.py ...`) to refresh "
    "underlying tables, then use this app to validate outputs."
)
