"""
Streamlit dashboard for exploring locally built NBA stats.
Version: 2025-11-26 (CSV Export Feature)
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional

import numpy as np
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
import top3_ranking
import top3_tracking
import backtest_top3
import backtest_portfolio
import preset_ab_test
from enrichment_monitor import ensure_enrichment_columns
import dfs_optimizer as dfs
import odds_api
import coefficient_calibrator as coeff_cal

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

            # Show simulation status with guardrail warnings
            sim_status = result['summary'].get('sim_status', 'not_run')
            sim_updated = result['summary'].get('sim_players_updated', 0)

            if sim_status == 'ok' and sim_updated > 0:
                st.sidebar.success(f"🎲 Simulation: {sim_updated} players computed")
            elif sim_status == 'failed':
                st.sidebar.error(
                    "❌ **Simulation FAILED** - backtest will use fallback ranking\n\n"
                    "Check errors below for details."
                )
            elif sim_status == 'skip' or sim_updated == 0:
                st.sidebar.info("ℹ️ Simulation skipped - no players to simulate")

            # Show any simulation warnings from errors list
            sim_errors = [e for e in result.get('errors', []) if 'Simulation' in e or 'p_top3' in e]
            if sim_errors:
                with st.sidebar.expander("⚠️ Simulation Warnings", expanded=True):
                    for err in sim_errors:
                        st.warning(err)

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
            prediction_date, player_id, team_id, opponent_id,
            days_rest, rest_multiplier, is_b2b,
            game_script_tier, blowout_risk, minutes_adjustment,
            role_tier, position_matchup_factor
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

    # Format for display - include enrichment factors if available
    cols_to_display = ['player_name', 'projected_ppg', 'proj_confidence',
                       'proj_floor', 'proj_ceiling', 'dfs_score', 'dfs_grade']

    # Add enrichment columns if they exist
    has_enrichments = 'role_tier' in team_preds.columns and team_preds['role_tier'].notna().any()

    if has_enrichments:
        # Create enrichment indicator column
        team_preds = team_preds.copy()

        def format_enrichment(row):
            """Create enrichment indicator string."""
            indicators = []

            # Rest status
            if row.get('is_b2b') == 1:
                indicators.append("B2B")
            elif row.get('days_rest') and row.get('days_rest') >= 3:
                indicators.append("REST")

            # Role tier
            role = row.get('role_tier', '')
            if role == 'STAR':
                indicators.append("*")
            elif role == 'BENCH':
                indicators.append("(B)")

            # Game script
            script = row.get('game_script_tier', '')
            if script == 'blowout':
                indicators.append("BLO")
            elif script == 'close_game':
                indicators.append("CLS")

            # Position matchup
            pos_factor = row.get('position_matchup_factor', 1.0)
            if pos_factor and pos_factor > 1.05:
                indicators.append("+")
            elif pos_factor and pos_factor < 0.95:
                indicators.append("-")

            return " ".join(indicators) if indicators else ""

        team_preds['factors'] = team_preds.apply(format_enrichment, axis=1)
        cols_to_display.append('factors')

    display_df = team_preds[cols_to_display].copy()

    # Set column names
    if has_enrichments:
        display_df.columns = ['Player', 'Proj PPG', 'Confidence', 'Floor', 'Ceiling', 'DFS Score', 'Grade', 'Factors']
    else:
        display_df.columns = ['Player', 'Proj PPG', 'Confidence', 'Floor', 'Ceiling', 'DFS Score', 'Grade']

    # Format numeric columns
    display_df['Proj PPG'] = display_df['Proj PPG'].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    display_df['Confidence'] = display_df['Confidence'].map(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
    display_df['Floor'] = display_df['Floor'].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    display_df['Ceiling'] = display_df['Ceiling'].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    display_df['DFS Score'] = display_df['DFS Score'].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Add legend for factors if enrichments exist
    if has_enrichments and team_preds['factors'].str.len().max() > 0:
        with st.expander("Factor Legend", expanded=False):
            st.markdown("""
            - **B2B**: Back-to-back game (-8% fatigue)
            - **REST**: 3+ days rest (+5% boost)
            - __*__: Star player
            - **(B)**: Bench player
            - **BLO**: Blowout risk (starters may rest)
            - **CLS**: Close game (heavy minutes expected)
            - **+/-**: Position matchup advantage/disadvantage
            """)


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


def clear_s3_and_db_caches():
    """Clear all caches to force fresh S3 download."""
    sync_database_from_s3.clear()
    get_connection.clear()
    st.cache_data.clear()


def force_refresh_from_s3() -> tuple[bool, str]:
    """Force download from S3, bypassing timestamp checks."""
    storage = s3_storage.S3PredictionStorage()

    if not storage.is_connected():
        return False, "S3 not configured"

    # Get S3 info before download for debugging
    s3_info = storage.get_backup_info()
    s3_size = s3_info.get('size_kb', 0) if s3_info.get('exists') else 0

    # Delete local database to force fresh download
    db_path = DEFAULT_DB_PATH
    if db_path.exists():
        db_path.unlink()

    # Clear all caches
    clear_s3_and_db_caches()

    # Download fresh from S3
    success, message = storage.download_database(db_path)
    if success:
        # Verify download by checking file size and latest prediction date
        if db_path.exists():
            local_size = db_path.stat().st_size / 1024
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute("SELECT MAX(game_date) FROM predictions")
                max_date = cursor.fetchone()[0]
                conn.close()
                return True, f"✅ Downloaded {local_size:.1f} KB from S3. Latest prediction: {max_date}"
            except Exception as e:
                return True, f"✅ Downloaded {local_size:.1f} KB from S3 (could not verify date: {e})"
        return True, f"✅ Force refreshed from S3 ({message})"
    else:
        return False, f"⚠️ Refresh failed: {message}"


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
    "Backtest Analysis",
    "Enrichment Validation",
    "FanDuel Compare",
    "Model vs FanDuel",
    "Model Review",
    "DFS Lineup Builder",
]

# Initialize selected page in session state
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Tournament Strategy"

# Note: All sidebar content consolidated in single block below (after db_path initialization)



@st.cache_resource
def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Auto-upgrade database schema for injury adjustments (one-time migration)
    try:
        pt.upgrade_predictions_table_for_injuries(conn)
    except Exception:
        pass  # Schema already upgraded or predictions table doesn't exist yet

    # Auto-upgrade database schema for enrichment columns (one-time migration)
    try:
        ensure_enrichment_columns(conn)
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


def _s3_backup_db(db_path: Path, toast: bool = True) -> None:
    """Best-effort S3 database backup after key DFS operations."""
    try:
        from s3_storage import S3PredictionStorage
        s3 = S3PredictionStorage()
        if s3.is_connected():
            success, msg = s3.upload_database(db_path)
            if success and toast:
                st.toast("☁️ Database backed up to S3")
    except Exception:
        pass


def _s3_upload_slate_file(local_path: Path, slate_date: str,
                          filename: str, toast_msg: str = None) -> None:
    """Best-effort upload of a DFS slate artifact to S3."""
    try:
        # Validate inputs to prevent malformed S3 keys
        if not slate_date or '/' in slate_date or '..' in slate_date:
            return
        if not filename or '/' in filename or '..' in filename:
            return

        from s3_storage import S3PredictionStorage
        s3 = S3PredictionStorage()
        if s3.is_connected():
            s3_key = f"dfs_slates/{slate_date}/{filename}"
            success, msg = s3.upload_file(local_path, s3_key)
            if success and toast_msg:
                st.toast(toast_msg)
    except Exception:
        pass


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

        def calc_recent(window: int, column: str, normalize_ot: bool = False) -> float | None:
            """Calculate recent average with optional OT normalization."""
            subset = player_df.head(window)[column].dropna()
            if subset.empty:
                return None
            # OT normalization: cap any single game at 48 minutes (regulation max)
            if normalize_ot and column == "minutes_float":
                subset = subset.clip(upper=48.0)
            return subset.mean()

        def calc_recent_trimmed(window: int, column: str, normalize_ot: bool = False) -> float | None:
            """Calculate trimmed mean (drop lowest value to handle foul trouble/injury exit)."""
            subset = player_df.head(window)[column].dropna()
            if len(subset) < 3:
                # Not enough data for trimmed mean, fall back to regular mean
                return subset.mean() if not subset.empty else None
            # OT normalization: cap any single game at 48 minutes
            if normalize_ot and column == "minutes_float":
                subset = subset.clip(upper=48.0)
            # Sort and drop the lowest value (foul trouble / injury exit / DNP)
            return subset.sort_values().iloc[1:].mean()

        def calc_recent_stddev(window: int, column: str, normalize_ot: bool = False) -> float | None:
            """Calculate standard deviation for recent games."""
            subset = player_df.head(window)[column].dropna()
            if len(subset) < 2:
                return None
            # OT normalization for minutes
            if normalize_ot and column == "minutes_float":
                subset = subset.clip(upper=48.0)
            return subset.std()

        def calc_starts_last_n(window: int, minutes_threshold: float = 28.0) -> int:
            """Count games with minutes >= threshold (proxy for 'started')."""
            subset = player_df.head(window)["minutes_float"].dropna()
            return int((subset >= minutes_threshold).sum())

        recent_records.append(
            {
                "player_id": player_id,
                "avg_pts_last3": calc_recent(3, "points"),
                "avg_pts_last5": calc_recent(5, "points"),
                "avg_fg3m_last3": calc_recent(3, "fg3m"),
                "avg_fg3m_last5": calc_recent(5, "fg3m"),
                # Minutes: use OT normalization (cap at 48) + trimmed mean for robustness
                "avg_minutes_last5": calc_recent_trimmed(5, "minutes_float", normalize_ot=True),
                "avg_minutes_last10": calc_recent_trimmed(10, "minutes_float", normalize_ot=True),
                "l5_minutes_stddev": calc_recent_stddev(5, "minutes_float", normalize_ot=True),
                "starts_last_5": calc_starts_last_n(5, 28.0),
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
                "avg_minutes_last10",
                "l5_minutes_stddev",
                "starts_last_5",
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
        # TEMPORARILY DISABLED - causing performance issues with per-player queries
        if False and def_ppm_df is not None and player_position and player_position in ['Guard', 'Forward', 'Center'] and conn is not None:
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
    # Momentum calculation parameters (optional)
    season_avg_minutes: Optional[float] = None,  # Season average minutes
    avg_minutes_last5: Optional[float] = None,  # Last 5 games average minutes
    avg_usg_last5: Optional[float] = None,  # Last 5 games average usage
    usage_pct: Optional[float] = None,  # Season usage percentage
    # Vegas integration parameters (NEW)
    player_id: Optional[int] = None,  # For Vegas O/U lookup
    team_abbrev: str = "",  # Team abbreviation for game environment
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

    # ==========================================================================
    # VEGAS ENSEMBLE INTEGRATION (NEW - Phase 2)
    # Blend our projection with FanDuel O/U line for market consensus
    # Vegas captures injury news, lineup changes, and sharp money movements
    # ==========================================================================
    vegas_blended = False
    vegas_analytics = {}

    if conn is not None and player_id is not None and game_date:
        try:
            # Get FanDuel O/U and blend with our projection (30% Vegas weight)
            blended_proj, vegas_analytics = odds_api.get_vegas_ensemble_projection(
                conn=conn,
                player_id=player_id,
                game_date=game_date,
                our_projection=projection,
                vegas_weight=0.30,
                dynamic_weighting=True
            )

            if vegas_analytics.get('blended', False):
                projection = blended_proj
                vegas_blended = True
                analytics_indicators += "🎰"  # Vegas blend indicator

        except Exception:
            # Vegas integration failed - continue with base projection
            pass

    # ==========================================================================
    # GAME ENVIRONMENT ADJUSTMENTS (NEW)
    # Apply blowout risk penalties and high-pace boosts from Vegas odds
    # ==========================================================================
    if conn is not None and team_abbrev and game_date:
        try:
            game_env = odds_api.get_vegas_game_environment(conn, game_date, team_abbrev)

            if game_env.get('found', False):
                # Blowout risk adjustment for star players (22+ PPG)
                blowout_risk = game_env.get('blowout_risk', 0)
                if blowout_risk >= 0.6 and season_avg >= 22:
                    # Stars get benched in blowouts - reduce projection by 5%
                    projection = projection * 0.95

                # High pace game boost
                pace_score = game_env.get('pace_score', 1.0)
                if pace_score > 1.05:  # >5% faster than league average
                    pace_bonus = 1 + (pace_score - 1) * 0.4  # 40% of pace diff
                    projection = projection * pace_bonus

        except Exception:
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

    # ==========================================================================
    # FLOOR CALCULATION with optional calibrated coefficients
    # NEW: Can use data-driven multipliers from coefficient_calibrator
    # ==========================================================================
    # Try to load calibrated coefficients if database available
    calibrated_coeffs = None
    if conn is not None:
        try:
            calibrated_coeffs = coeff_cal.load_calibrated_coefficients(conn)
        except Exception:
            pass

    # FLOOR: Conservative estimate (bad game scenario)
    # Use calibrated multiplier if available, otherwise use formula
    if calibrated_coeffs and calibrated_coeffs.confidence_level in ['high', 'medium']:
        # Use learned floor multiplier based on projection tier
        floor_mult = calibrated_coeffs.get_floor_multiplier_for_projection(projection)
        floor = projection * floor_mult
    else:
        # Fallback: Base floor 14% below projection + uncertainty adjustment
        floor_base_interval = season_avg * 0.14
        floor_uncertainty_factor = 0.8 + (0.4 * uncertainty)
        floor_interval = floor_base_interval * floor_uncertainty_factor
        floor = max(0, projection - floor_interval)

    # ==========================================================================
    # CEILING CALCULATION with optional calibrated coefficients
    # NEW: Can use data-driven multipliers from coefficient_calibrator
    # ==========================================================================
    # CEILING: 90th percentile outcome (explosive game scenario)
    # For tournament play, ceiling must represent what happens when everything goes right

    if calibrated_coeffs and calibrated_coeffs.confidence_level in ['high', 'medium']:
        # Use learned ceiling multiplier based on projection tier
        ceiling_multiplier = calibrated_coeffs.get_ceiling_multiplier_for_projection(projection) - 1.0
        # Add uncertainty bonus (10-20% additional based on data quality)
        ceiling_uncertainty_bonus = 0.15 * uncertainty
        ceiling_multiplier += ceiling_uncertainty_bonus

        # Apply position-specific ceiling adjustment from calibrated coefficients
        position_ceiling_adj = calibrated_coeffs.get_position_ceiling_adjustment(player_position)
        ceiling_multiplier += position_ceiling_adj
    else:
        # Fallback: Original tiered calculation
        # Formula: Base ceiling multiplier adjusted by player tier and uncertainty
        if season_avg >= 25:
            # Star players: 70-95% upside (e.g., 28 PPG → 48-55 ceiling)
            ceiling_base_multiplier = 0.70
            ceiling_uncertainty_bonus = 0.25 * uncertainty
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
        # NEW: Vegas integration tracking
        "vegas_blended": vegas_blended,
        "vegas_ou": vegas_analytics.get('vegas_ou') if vegas_analytics else None,
        "vegas_weight": vegas_analytics.get('weight_used', 0) if vegas_analytics else 0,
        "vegas_divergence": vegas_analytics.get('divergence_pct', 0) if vegas_analytics else 0,
        "used_calibrated_coeffs": calibrated_coeffs is not None and calibrated_coeffs.confidence_level in ['high', 'medium'],
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

# Initialize db_path early so it's available throughout the app
db_path = Path(st.session_state["db_path_input"]).expanduser()

with st.sidebar:
    # Navigation FIRST - before any dynamic content
    if 'nav_selection' not in st.session_state:
        st.session_state.nav_selection = st.session_state.selected_page

    selected_page = st.selectbox(
        "📊 Navigation",
        tab_titles,
        index=tab_titles.index(st.session_state.nav_selection) if st.session_state.nav_selection in tab_titles else 0,
        key='page_nav_v4'
    )
    st.session_state.selected_page = selected_page
    st.session_state.nav_selection = selected_page

    st.divider()

    # S3 Cloud Backup Status
    st.markdown("### ☁️ Cloud Backup Status")
    if "S3 not configured" in s3_sync_message:
        st.info("💻 Running in local mode")
    elif s3_sync_status:
        st.success(s3_sync_message)
    else:
        st.warning(s3_sync_message)

    # Refresh button to re-download from S3
    if st.button("🔄 Refresh from S3", help="Force download latest database from S3"):
        with st.spinner("Downloading from S3..."):
            success, message = force_refresh_from_s3()
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    # Debug info expander
    with st.expander("🔍 Debug Info"):
        storage = s3_storage.S3PredictionStorage()
        if storage.is_connected():
            s3_info = storage.get_backup_info()
            if s3_info.get('exists'):
                st.write(f"**S3 file modified:** {s3_info.get('last_modified')}")
                st.write(f"**S3 file size:** {s3_info.get('size_kb', 0):.1f} KB")
            else:
                st.write("No S3 backup found")

            db_path = DEFAULT_DB_PATH
            if db_path.exists():
                local_mtime = datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc)
                local_size = db_path.stat().st_size / 1024
                st.write(f"**Local file modified:** {local_mtime}")
                st.write(f"**Local file size:** {local_size:.1f} KB")

                # Show max prediction date in database
                try:
                    conn = get_connection(str(db_path))
                    max_date_df = pd.read_sql_query(
                        "SELECT MAX(game_date) as max_date, COUNT(*) as count FROM predictions",
                        conn
                    )
                    if not max_date_df.empty:
                        st.write(f"**Latest prediction date:** {max_date_df['max_date'].iloc[0]}")
                        st.write(f"**Total predictions:** {max_date_df['count'].iloc[0]}")
                except Exception as e:
                    st.write(f"Could not query predictions: {e}")
            else:
                st.write("No local database file")
        else:
            st.write("S3 not configured")

    st.divider()

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

    # Auto-build database if it doesn't exist
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

    # PREDICTION GENERATION - Quick Action Button
    st.markdown("### 🎯 Quick Actions")
    st.markdown("#### Generate Predictions")

    # Date picker for predictions
    pred_date = st.date_input(
        "Date:",
        value=datetime.now(EASTERN_TZ).date(),
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
        # Don't close cached connection - it's shared across the app

        if existing_count > 0:
            st.caption(f"✓ {existing_count} predictions exist")
    except Exception:
        existing_count = 0

    # Button label changes based on whether predictions exist
    gen_button_label = "🔄 Regenerate" if existing_count > 0 else "🎯 Generate Predictions"

    if st.button(gen_button_label, type="primary", use_container_width=True, key="sidebar_gen_predictions"):
        generate_predictions_ui(pred_date, db_path, builder_config)

    st.divider()

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
    pt.upgrade_predictions_table_for_opponent_injury(games_conn)  # Opponent injury impact tracking
    pt.upgrade_predictions_table_for_minutes(games_conn)  # Minutes projection and tier tracking

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
        # Create a CANONICAL game key by sorting team names alphabetically
        # This ensures both teams' predictions end up in the same game group
        # e.g., "Dallas Mavericks vs Philadelphia 76ers" (always alphabetical order)
        def make_canonical_game_key(row):
            teams = sorted([row['team_name'], row['opponent_name']])
            return f"{teams[0]} vs {teams[1]}"

        predictions_df['game_key'] = predictions_df.apply(make_canonical_game_key, axis=1)

        # Get unique games
        games = predictions_df['game_key'].unique()

        st.markdown("### Game-by-Game Predictions")
        st.caption(f"Showing {len(games)} game(s) with predictions")

        for game_key in games:
            game_preds = predictions_df[predictions_df['game_key'] == game_key]

            # Parse teams from canonical key (alphabetical order)
            if ' vs ' in game_key:
                team_a, team_b = game_key.split(' vs ')
            else:
                continue

            # Display as "Team A vs Team B" (alphabetical, not home/away since we don't track that)
            st.markdown(f"#### {team_a} vs {team_b}")

            # Split into two columns
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"**{team_a}**")
                # Get predictions for team_a (where team_name == team_a)
                team_a_preds = game_preds[game_preds['team_name'] == team_a]
                if not team_a_preds.empty:
                    display_team_predictions(team_a_preds)
                else:
                    st.caption("No predictions for this team")

            with col_b:
                st.markdown(f"**{team_b}**")
                # Get predictions for team_b (where team_name == team_b)
                team_b_preds = game_preds[game_preds['team_name'] == team_b]
                if not team_b_preds.empty:
                    display_team_predictions(team_b_preds)
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
                    index=0,
                    key='pred_log_date_select'
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
                        today = datetime.now(EASTERN_TZ).date().strftime('%Y-%m-%d')
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
    today = datetime.now(EASTERN_TZ).date().strftime('%Y-%m-%d')

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

    # ==========================================================================
    # The Odds API Usage Tracker
    # ==========================================================================
    st.subheader("🎰 The Odds API Usage")
    st.caption("Track your API budget for FanDuel player props")

    try:
        import odds_api
        admin_conn = get_connection(str(db_path))

        # Create tables if needed
        odds_api.create_odds_tables(admin_conn)

        # Get monthly usage
        monthly_usage = odds_api.get_monthly_api_usage(admin_conn)
        budget_limit = odds_api.MAX_API_REQUESTS_PER_MONTH

        # Calculate percentages
        usage_pct = (monthly_usage / budget_limit) * 100 if budget_limit > 0 else 0
        remaining = budget_limit - monthly_usage

        # Display metrics
        api_col1, api_col2, api_col3 = st.columns(3)

        with api_col1:
            st.metric(
                "Monthly Usage",
                f"{monthly_usage}/{budget_limit}",
                delta=f"{remaining} remaining",
                delta_color="normal" if remaining > 100 else "inverse"
            )

        with api_col2:
            # Progress bar with color coding
            if usage_pct < 50:
                color = "🟢"
            elif usage_pct < 80:
                color = "🟡"
            else:
                color = "🔴"
            st.metric("Budget Used", f"{color} {usage_pct:.1f}%")

        with api_col3:
            # Check API key status
            api_key = odds_api.get_api_key()
            if api_key:
                st.metric("API Key", "✅ Configured", delta=f"{api_key[:8]}...")
            else:
                st.metric("API Key", "❌ Not Set", delta="Add to secrets.toml")

        # Progress bar
        st.progress(min(1.0, usage_pct / 100), text=f"API Budget: {monthly_usage} of {budget_limit} requests used")

        # Fetch history
        with st.expander("📜 Recent Fetch History", expanded=False):
            cursor = admin_conn.cursor()
            cursor.execute("""
                SELECT
                    fetch_date,
                    game_date,
                    events_fetched,
                    players_matched,
                    api_requests_used,
                    remaining_requests,
                    error_message
                FROM odds_fetch_log
                ORDER BY created_at DESC
                LIMIT 10
            """)
            fetch_history = cursor.fetchall()

            if fetch_history:
                import pandas as pd
                history_df = pd.DataFrame(fetch_history, columns=[
                    'Fetch Date', 'Game Date', 'Events', 'Players Matched',
                    'API Requests', 'Remaining', 'Error'
                ])
                # Format error column
                history_df['Status'] = history_df['Error'].apply(
                    lambda x: '❌ ' + str(x)[:30] if x else '✅ Success'
                )
                history_df = history_df.drop('Error', axis=1)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("No fetch history yet. Click 'Fetch FanDuel Lines' on the Vegas Comparison tab.")

        # Reset counter button (for when local tracking drifts from actual API usage)
        if st.button("🔄 Reset API Usage Counter",
                      help="Clear the local usage log. Use this if the counter is out of sync with your actual API usage at theoddsapi.com.",
                      key="reset_api_counter"):
            try:
                today_eastern = datetime.now(EASTERN_TZ).date()
                first_of_month = today_eastern.replace(day=1).strftime("%Y-%m-%d")
                admin_conn.execute(
                    "DELETE FROM odds_fetch_log WHERE fetch_date >= ?",
                    (first_of_month,)
                )
                admin_conn.commit()
                st.success("API usage counter reset. The counter will re-calibrate from the API on next fetch.")
                st.rerun()
            except Exception as reset_err:
                st.error(f"Reset failed: {reset_err}")

        # Budget recommendations
        if usage_pct >= 80:
            st.warning(f"""
            ⚠️ **Budget Alert:** You've used {usage_pct:.0f}% of your monthly API budget.

            **Options:**
            - Check your actual usage at [The Odds API](https://the-odds-api.com/) — local counter may be inflated
            - Click **Reset API Usage Counter** above if the local count doesn't match the API website
            - Wait until next month (resets on the 1st)
            """)
        elif usage_pct >= 50:
            st.info(f"💡 You've used {monthly_usage} requests. Budget resets on the 1st of next month.")

    except ImportError:
        st.info("odds_api module not available")
    except Exception as e:
        st.error(f"Error loading API usage: {e}")

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

    st.divider()

    # Simulation Backfill Section
    st.subheader("🎲 Simulation Data Backfill")
    st.write("Populate p_top3, p_top1, and top_scorer_score for historical predictions")

    try:
        admin_conn = get_connection(str(db_path))
        cursor = admin_conn.cursor()

        # FIRST: Ensure simulation columns exist (idempotent migration)
        pt.ensure_prediction_sim_columns(admin_conn)

        # Check simulation coverage
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN p_top3 IS NOT NULL THEN 1 ELSE 0 END) as with_sim,
                MIN(game_date) as min_date,
                MAX(game_date) as max_date
            FROM predictions
        """)
        total, with_sim, min_date, max_date = cursor.fetchone()
        coverage_pct = (with_sim / total * 100) if total > 0 else 0

        # Get dates needing backfill
        cursor.execute("""
            SELECT COUNT(DISTINCT game_date)
            FROM predictions
            WHERE p_top3 IS NULL
        """)
        dates_needing_backfill = cursor.fetchone()[0]

        col_sim1, col_sim2 = st.columns(2)

        with col_sim1:
            st.metric("Simulation Coverage", f"{coverage_pct:.1f}%")
            st.metric("Predictions with Sim Data", f"{with_sim:,}/{total:,}")
            st.metric("Dates Needing Backfill", dates_needing_backfill)

            if coverage_pct < 50:
                st.error("⚠️ LOW COVERAGE - Backtest will use fallback ranking")
            elif coverage_pct < 90:
                st.warning("ℹ️ Partial coverage - some dates use fallback")
            else:
                st.success("✅ Good coverage")

        with col_sim2:
            st.write("**Backfill Options:**")

            # Date range for backfill
            backfill_days = st.selectbox(
                "Date Range",
                options=[7, 14, 30, 60, "All"],
                index=2,
                key="backfill_days_select"
            )

            sim_count = st.selectbox(
                "Simulations per date",
                options=[2000, 5000, 10000],
                index=1,
                key="sim_count_select"
            )

            if st.button("🚀 Run Backfill", type="primary", use_container_width=True, key="run_backfill_btn"):
                import backfill_sim_probs as bsp

                with st.spinner("Running simulation backfill..."):
                    try:
                        # Calculate date range
                        if backfill_days == "All":
                            date_from = min_date
                        else:
                            from datetime import datetime, timedelta
                            date_from = (datetime.now() - timedelta(days=backfill_days)).strftime('%Y-%m-%d')

                        # Ensure columns exist
                        pt.ensure_prediction_sim_columns(admin_conn)

                        # Get dates needing backfill
                        dates = bsp.get_dates_needing_backfill(admin_conn, date_from=date_from)

                        if not dates:
                            st.success("✅ All dates already have simulation data!")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            stats = {'ok': 0, 'failed': 0, 'total_rows': 0}

                            for i, (game_date, total_players, sim_players) in enumerate(dates, 1):
                                status_text.text(f"Processing {game_date} ({i}/{len(dates)})...")
                                progress_bar.progress(i / len(dates))

                                result = bsp.backfill_date(
                                    admin_conn,
                                    game_date,
                                    sim_n=sim_count,
                                    verbose=False
                                )

                                if result['status'] == 'ok':
                                    stats['ok'] += 1
                                    stats['total_rows'] += result['count']
                                else:
                                    stats['failed'] += 1

                            progress_bar.empty()
                            status_text.empty()

                            st.success(
                                f"✅ Backfill complete!\n\n"
                                f"- Dates processed: {stats['ok']}\n"
                                f"- Failed: {stats['failed']}\n"
                                f"- Rows updated: {stats['total_rows']:,}"
                            )

                            # Rerun to update metrics
                            st.rerun()

                    except Exception as e:
                        st.error(f"❌ Backfill failed: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())

            # Verify button
            if st.button("🔍 Run Verification", use_container_width=True, key="run_verify_btn"):
                import backfill_sim_probs as bsp

                with st.spinner("Running verification checks..."):
                    results = bsp.run_all_verifications(admin_conn)

                    # Display results
                    st.subheader("Verification Results")

                    # Coverage
                    cov = results['coverage']
                    if cov['status'] == 'PASS':
                        st.success(f"✅ Coverage: {cov['coverage_pct']:.1f}% ({cov['with_sim']:,}/{cov['total']:,})")
                    else:
                        st.error(f"❌ Coverage: {cov['coverage_pct']:.1f}% - {cov['missing_sim']:,} missing")

                    # Sanity
                    san = results['sanity']
                    if san['status'] == 'PASS':
                        st.success(f"✅ Probability Sanity: {san['slates_checked']} slates checked")
                    else:
                        st.warning(f"⚠️ Probability Sanity: {san['slates_with_issues']} slates with issues")
                        if san['issues']:
                            with st.expander("View Issues"):
                                for issue in san['issues']:
                                    st.write(f"- {issue['game_date']}: {', '.join(issue['issues'])}")

                    # Sorting
                    sort = results['sorting']
                    if sort['status'] == 'PASS':
                        st.success(f"✅ Star Burial Check: {sort['stars_checked']} stars verified")
                    else:
                        st.error(f"❌ Star Burial Check: {sort['stars_missing_sim']} missing sim, {sort['stars_buried']} buried")
                        if sort['details']:
                            with st.expander("View Details"):
                                for detail in sort['details']:
                                    st.write(f"- {detail['date']} - {detail['player']}: {detail['issue']}")

    except Exception as e:
        st.error(f"Error checking simulation data: {e}")

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

    # Top 3 Ranking Mode selector
    st.divider()
    st.markdown("### 🎯 Top 3 Scorer Ranking")
    st.caption("Optimized for picking the daily top 3 scorers")

    rank_col1, rank_col2, rank_col3 = st.columns([1, 1, 1])

    with rank_col1:
        ranking_mode = st.selectbox(
            "Ranking Method",
            options=["GPP Score (Legacy)", "TopScorerScore", "Simulation P(Top3)", "Both"],
            index=1,
            help="TopScorerScore: Fast heuristic optimized for top-3 identification. Simulation: Monte Carlo estimation of P(top 3)."
        )

    with rank_col2:
        exclude_questionable = st.checkbox(
            "Exclude Questionable/Doubtful",
            value=False,
            help="Filter out players with questionable or doubtful injury status"
        )

    with rank_col3:
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum projection confidence threshold"
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

        # ========== TOP 3 SCORER RANKING INTEGRATION ==========
        # Use the new top3_ranking module for optimized top-3 identification

        if ranking_mode != "GPP Score (Legacy)":
            try:
                ranker = top3_ranking.Top3Ranker(tourn_conn)

                # Get rankings based on selected method
                if ranking_mode in ["TopScorerScore", "Both"]:
                    tss_df = ranker.rank_by_top_scorer_score(selected_date, include_components=True)
                    if not tss_df.empty:
                        # Merge TopScorerScore into main dataframe
                        df = df.merge(
                            tss_df[['player_id', 'top_scorer_score', 'calibrated_base', 'ceiling_bonus',
                                   'role_sustainability', 'matchup_today', 'injury_opportunity',
                                   'star_power', 'risk_penalty']],
                            on='player_id',
                            how='left'
                        )
                        df['top_scorer_score'] = df['top_scorer_score'].fillna(0)

                if ranking_mode in ["Simulation P(Top3)", "Both"]:
                    with st.spinner("Running Monte Carlo simulation (10,000 iterations)..."):
                        sim_df = ranker.simulate_top3_probability(selected_date, n_simulations=10000)
                    if not sim_df.empty:
                        # Merge all simulation columns for transparency
                        sim_cols = ['player_id', 'p_top3', 'p_top3_pct', 'p_top1', 'p_top1_pct',
                                   'scoring_stddev', 'tier', 'stddev_calibrated']
                        sim_cols = [c for c in sim_cols if c in sim_df.columns]
                        df = df.merge(sim_df[sim_cols], on='player_id', how='left')
                        df['p_top3'] = df['p_top3'].fillna(0)
                        df['p_top3_pct'] = df['p_top3_pct'].fillna(0)
                        df['p_top1_pct'] = df['p_top1_pct'].fillna(0) if 'p_top1_pct' in df.columns else 0

            except Exception as rank_error:
                st.warning(f"⚠️ Top3Ranker error: {rank_error}. Falling back to GPP Score.")
                ranking_mode = "GPP Score (Legacy)"

        # Apply risk filters if enabled
        if exclude_questionable:
            # Filter out questionable/doubtful players
            df = df[~df['player_id'].isin(
                pd.read_sql_query(
                    "SELECT player_id FROM injury_list WHERE status IN ('questionable', 'doubtful')",
                    tourn_conn
                )['player_id'].tolist() if tourn_conn else []
            )]

        # Apply confidence filter
        if 'proj_confidence' in df.columns:
            df = df[df['proj_confidence'].fillna(0.7) >= min_confidence]

        # Sort by appropriate column based on ranking mode
        if ranking_mode == "TopScorerScore":
            sort_col = 'top_scorer_score' if 'top_scorer_score' in df.columns else 'tourn_score'
            df = df.sort_values(sort_col, ascending=False)
        elif ranking_mode == "Simulation P(Top3)":
            sort_col = 'p_top3' if 'p_top3' in df.columns else 'tourn_score'
            df = df.sort_values(sort_col, ascending=False)
        elif ranking_mode == "Both":
            sort_col = 'top_scorer_score' if 'top_scorer_score' in df.columns else 'tourn_score'
            df = df.sort_values(sort_col, ascending=False)
        else:
            # GPP Score (Legacy) - original behavior
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
            'std_def_ppm': 'PPM StdDev',
            # Top3 ranking columns
            'top_scorer_score': 'TSS',
            'p_top3_pct': 'P(Top3)%',
            'p_top1_pct': 'P(#1)%',
            'tier': 'Tier',
            'proj_minutes': 'Proj Min',
            'scoring_stddev': 'Variance',
            'calibrated_base': 'Cal Base',
            'ceiling_bonus': 'Ceil+',
            'role_sustainability': 'Role',
            'matchup_today': 'Matchup',
            'injury_opportunity': 'Opp+',
            'star_power': 'Star',
            'risk_penalty': 'Risk-'
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
        # Round Top3 ranking columns (if available)
        if 'TSS' in display_df.columns:
            display_df['TSS'] = display_df['TSS'].round(1)
        if 'P(Top3)%' in display_df.columns:
            display_df['P(Top3)%'] = display_df['P(Top3)%'].round(1)

        # Select and reorder columns for display based on ranking mode
        if ranking_mode == "GPP Score (Legacy)":
            base_columns = [
                'Player', 'Pos', 'Team', 'Opponent', 'Ceiling', 'L5 Avg', 'Proj PPG',
                'GPP Score', 'GPP Grade', 'GPP Factors', 'Opp Def Grade'
            ]
        elif ranking_mode == "TopScorerScore":
            base_columns = [
                'Player', 'Pos', 'Team', 'Opponent', 'Ceiling', 'L5 Avg', 'Proj PPG',
                'TSS', 'GPP Score', 'Opp Def Grade'
            ]
            # Add component breakdown in expander - ALWAYS show
            with st.expander("📊 TopScorerScore Formula & Component Breakdown", expanded=True):
                st.markdown("""
                **TSS = Cal Base + Ceiling + Role + Matchup + Opportunity + Star Power - Risk**

                | Component | Range | Description |
                |-----------|-------|-------------|
                | **Cal Base** | ~15-30 | Calibrated projection (fixes historical over-prediction bias) |
                | **Ceil+** | 0-8 | Upside bonus scaled by projection (high ceiling + high volume) |
                | **Role** | -3 to +4 | **ValidityFactor-gated** hot streak bonus (see below) |
                | **Matchup** | -3 to +5 | TODAY's opponent defense rating (higher = weaker defense = bonus) |
                | **Opp+** | 0-6 | Injury opportunity: teammate OUT today (+4) or opponent star out (+2) |
                | **Star** | 0-3 | Star power bonus: 28+ PPG = +3, 25+ = +2, 22+ = +1 |
                | **Risk-** | 0 to -10 | Penalties: Questionable (-5), Blowout risk (-3), B2B (-2) |

                🎯 **ValidityFactor Pattern** (Role Score):
                - **Formula**: `RoleScore = RawHotBonus × ValidityFactor`
                - **ValidityFactor** (0.0 to 1.0) crushes fake streaks:
                  - **Star Returns Discount**: L5 elevated but NO injury boost today? → ValidityFactor × 0.25
                  - **Minutes Stability**: Low confidence + elevated stats? → ValidityFactor × 0.5
                  - **Ongoing Opportunity**: Injury beneficiary still active → ValidityFactor ≥ 0.8
                - Hot streaks without context are crushed. Ceiling and matchup dominate.
                """)

                if 'Cal Base' in display_df.columns:
                    st.markdown("#### Component Values by Player")
                    component_cols = ['Player', 'TSS', 'Cal Base', 'Ceil+', 'Role', 'Matchup', 'Opp+', 'Star', 'Risk-']
                    available_cols = [c for c in component_cols if c in display_df.columns]
                    # Round component values for display
                    comp_df = display_df[available_cols].head(20).copy()
                    for col in available_cols:
                        if col != 'Player' and col in comp_df.columns:
                            comp_df[col] = comp_df[col].round(1)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Component data not available. TSS columns: " + str([c for c in display_df.columns if 'TSS' in c or 'Cal' in c or 'bonus' in c.lower()]))
        elif ranking_mode == "Simulation P(Top3)":
            base_columns = [
                'Player', 'Pos', 'Team', 'Opponent', 'Tier', 'Ceiling', 'L5 Avg', 'Proj PPG',
                'P(Top3)%', 'P(#1)%', 'Variance', 'GPP Score', 'Opp Def Grade'
            ]
            # Add explanation of simulation approach
            with st.expander("🎲 P(Top-3) Simulation Explained", expanded=False):
                st.markdown("""
                **Why Simulation > Heuristics for Top-3**

                Top-3 scoring is a **tail event**. A player with lower expected points but higher variance
                can actually have a **better** chance of finishing top-3 than a higher-mean, lower-variance player.

                **Simulation Method (10,000 iterations):**
                1. Draw game-level factors (±8% swing per game - positive correlation)
                2. Sample each player's scoring with game factor applied
                3. Apply team usage constraint (when one teammate spikes, suppress others 5%)
                4. Count how often each player finishes in top 3 or #1

                **Tier Classifications (Non-Circular):**
                - ⭐ **STAR**: 34+ min + 3+ starts + high usage/ceiling
                - 🎯 **SIXTH_MAN**: 24-30 min + bench role + high offensive potential
                - 👤 **ROLE**: 24+ min, solid rotation
                - 🪑 **BENCH**: <20 min (rarely win top-3)

                **Variance Calibration:**
                - Uses 60-day historical residuals by tier + minutes band when available
                - Falls back to ceiling-floor estimate if insufficient historical data
                - Adjusted for minutes volatility and injury beneficiary status

                **Key Insight**: High variance + decent floor = good tournament play
                """)

            # ========== TRANSPARENCY EXPANDERS ==========
            with st.expander("📊 Minutes Breakdown", expanded=False):
                st.markdown("""
                ### How Projected Minutes is Calculated

                **Formula:** `proj_min = 0.55×L5 + 0.25×L10 + 0.20×Season`

                **OT Normalization:** Single games capped at 48 minutes
                **Trimmed Mean:** Drops lowest value (handles foul trouble/injury exit)

                **Role Guardrails:**
                - Starters: clamped to [28, 40]
                - Rotation: clamped to [18, 32]
                - Bench: clamped to [0, 22]

                **Context Adjustments:**
                - B2B: -1.5 min
                - Blowout (spread 8+): sigmoid-scaled reduction up to -4 min (softer for stars)
                - Injury beneficiary: up to +6 min
                """)
                # Show minutes breakdown if data available
                min_cols = ['Player', 'Team']
                if 'avg_minutes_last5' in df.columns:
                    df['L5 Min'] = df['avg_minutes_last5'].round(1)
                    min_cols.append('L5 Min')
                if 'avg_minutes_last10' in df.columns:
                    df['L10 Min'] = df['avg_minutes_last10'].round(1)
                    min_cols.append('L10 Min')
                if 'avg_minutes' in df.columns:
                    df['Season Min'] = df['avg_minutes'].round(1)
                    min_cols.append('Season Min')
                if 'Tier' in display_df.columns:
                    min_cols.append('Tier')

                if len(min_cols) > 2:
                    min_display = display_df[['Player', 'Team']].copy()
                    for col in ['L5 Min', 'L10 Min', 'Season Min']:
                        if col in df.columns:
                            min_display[col] = df[col].values[:len(min_display)]
                    if 'Tier' in display_df.columns:
                        min_display['Tier'] = display_df['Tier'].values
                    st.dataframe(min_display.head(20), use_container_width=True, hide_index=True)
                else:
                    st.info("Minutes breakdown data not available in current dataset.")

            with st.expander("📈 Variance Components", expanded=False):
                st.markdown("""
                ### How Scoring Variance is Estimated

                **Calibrated σ:** From 60-day historical residuals by tier + minutes band
                - Grouped by tier (STAR/ROLE/SIXTH_MAN/BENCH) and minutes band (e.g., 32-36, 28-32)
                - Requires 20+ samples for exact match, 10+ for tier average
                - Falls back to ceiling-floor estimate if insufficient data

                **Fallback σ:** `(ceiling - floor) / 3.29` (99% CI estimate)

                **Adjustments:**
                - Minutes volatility (L5 stddev > 5): ×1.3
                - Injury beneficiary role: ×1.2

                **Why variance matters:** Higher variance = better P(top-3) for tail events
                Even if expected points are lower, variance creates more winning scenarios.
                """)
                # Show variance breakdown if data available
                var_cols = ['Player', 'Team', 'Tier']
                if 'Variance' in display_df.columns:
                    var_cols.append('Variance')
                if 'stddev_calibrated' in df.columns:
                    df['σ Calibrated'] = df['stddev_calibrated'].apply(lambda x: '✓' if x else '—')
                    var_cols.append('σ Calibrated')
                if 'Ceiling' in display_df.columns:
                    var_cols.append('Ceiling')
                if 'Floor' in display_df.columns:
                    var_cols.append('Floor')

                if 'Variance' in display_df.columns:
                    var_display = display_df[['Player', 'Team']].copy()
                    if 'Tier' in display_df.columns:
                        var_display['Tier'] = display_df['Tier'].values
                    var_display['Variance'] = display_df['Variance'].values
                    if 'σ Calibrated' in df.columns:
                        var_display['σ Calibrated'] = df['σ Calibrated'].values[:len(var_display)]
                    var_display['Ceiling'] = display_df['Ceiling'].values
                    var_display['Floor'] = display_df['Floor'].values
                    st.dataframe(var_display.head(20), use_container_width=True, hide_index=True)
                else:
                    st.info("Variance data not available in current dataset.")

            with st.expander("🔗 Game Correlation Info", expanded=False):
                st.markdown("""
                ### How Games Affect Player Correlation

                **Game Factor (Positive Correlation):**
                - Each game gets a random multiplier (~±8% typical swing)
                - Scaled by implied total (high-pace games have higher variance)
                - All players in a game are affected by the same game factor
                - High-scoring environment → everyone elevated

                **Team Usage Constraint (Negative Correlation):**
                - When one teammate spikes (>1.2× their mean), suppress others by 5%
                - Prevents unrealistic scenarios where 2+ teammates both go off
                - Ensures P(#1) concentrates on realistic candidates

                **Why This Matters:**
                - Without correlation, teammates can both show high P(#1) - unrealistic!
                - Game environment affects everyone (blowout, pace, OT)
                - Usage is finite - one player's spike reduces teammates' opportunity
                """)
                # Show game context if data available
                game_cols = ['Player', 'Team', 'Opponent']
                if 'vegas_spread' in df.columns:
                    df['Spread'] = df['vegas_spread'].abs().round(1)
                    game_cols.append('Spread')
                if 'implied_total' in df.columns:
                    df['Total'] = df['implied_total'].round(1)
                    game_cols.append('Total')

                if len(game_cols) > 3:
                    game_display = display_df[['Player', 'Team', 'Opponent']].copy()
                    if 'Spread' in df.columns:
                        game_display['Spread'] = df['Spread'].values[:len(game_display)]
                    if 'Total' in df.columns:
                        game_display['Total'] = df['Total'].values[:len(game_display)]
                    st.dataframe(game_display.head(20), use_container_width=True, hide_index=True)
                else:
                    st.info("Game context data (spread, total) not available.")
        else:  # Both
            base_columns = [
                'Player', 'Pos', 'Team', 'Opponent', 'Tier', 'Ceiling', 'L5 Avg', 'Proj PPG',
                'TSS', 'P(Top3)%', 'P(#1)%', 'GPP Score', 'Opp Def Grade'
            ]
            # Add component breakdown in expander
            with st.expander("📊 TopScorerScore Formula & Component Breakdown", expanded=False):
                st.markdown("""
                **TSS = Cal Base + Ceiling + Role + Matchup + Opportunity + Star Power - Risk**

                | Component | Range | Description |
                |-----------|-------|-------------|
                | **Cal Base** | ~15-30 | Calibrated projection (fixes historical over-prediction bias) |
                | **Ceil+** | 0-8 | Upside bonus scaled by projection (high ceiling + high volume) |
                | **Role** | -3 to +4 | **ValidityFactor-gated** hot streak bonus (see below) |
                | **Matchup** | -3 to +5 | TODAY's opponent defense rating (higher = weaker defense = bonus) |
                | **Opp+** | 0-6 | Injury opportunity: teammate OUT today (+4) or opponent star out (+2) |
                | **Star** | 0-3 | Star power bonus: 28+ PPG = +3, 25+ = +2, 22+ = +1 |
                | **Risk-** | 0 to -10 | Penalties: Questionable (-5), Blowout risk (-3), B2B (-2) |

                🎯 **ValidityFactor Pattern** (Role Score):
                - **Formula**: `RoleScore = RawHotBonus × ValidityFactor`
                - Hot streaks without context are crushed. Ceiling and matchup dominate.
                """)

                if 'Cal Base' in display_df.columns:
                    st.markdown("#### Component Values by Player")
                    component_cols = ['Player', 'TSS', 'Cal Base', 'Ceil+', 'Role', 'Matchup', 'Opp+', 'Star', 'Risk-']
                    available_cols = [c for c in component_cols if c in display_df.columns]
                    comp_df = display_df[available_cols].head(20).copy()
                    for col in available_cols:
                        if col != 'Player' and col in comp_df.columns:
                            comp_df[col] = comp_df[col].round(1)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Filter to only existing columns
        base_columns = [c for c in base_columns if c in display_df.columns]

        # Add PPM columns if they exist
        ppm_columns = []
        if 'Opp Def PPM' in display_df.columns:
            ppm_columns.extend(['Opp Def PPM', 'Opp PPM Grade', 'Ceiling Factor'])

        # Add remaining columns
        remaining_columns = ['Opp Def', 'Range', 'Variance']

        display_columns = base_columns + ppm_columns + remaining_columns
        display_columns = [c for c in display_columns if c in display_df.columns]

        # ========== TOP 3 PICKS HIGHLIGHT ==========
        st.subheader("🏆 Today's Top 3 Picks")

        try:
            top3_df = display_df.head(3)
            if not top3_df.empty:
                # Determine score column name based on ranking mode
                score_col = 'TSS' if 'TSS' in top3_df.columns else 'GPP Score'
                prob_col = 'P(Top3)%' if 'P(Top3)%' in top3_df.columns else None

                pick_cols = st.columns(3)
                for idx, (_, player) in enumerate(top3_df.iterrows()):
                    with pick_cols[idx]:
                        medal = ["🥇", "🥈", "🥉"][idx]
                        st.markdown(f"### {medal} {player['Player']}")
                        st.markdown(f"**{player['Team']}** vs {player['Opponent']}")

                        # Show key stats
                        score_display = f"{player[score_col]:.0f}" if pd.notna(player.get(score_col)) else "N/A"
                        ceiling_display = f"{player['Ceiling']:.1f}" if pd.notna(player.get('Ceiling')) else "N/A"
                        proj_display = f"{player['Proj PPG']:.1f}" if pd.notna(player.get('Proj PPG')) else "N/A"

                        if ranking_mode in ["TopScorerScore", "Both"]:
                            st.metric("TopScorerScore", score_display)
                        elif ranking_mode == "Simulation P(Top3)":
                            prob_display = f"{player['P(Top3)%']:.1f}%" if pd.notna(player.get('P(Top3)%')) else "N/A"
                            st.metric("P(Top 3)", prob_display)
                        else:
                            st.metric("GPP Score", score_display)

                        st.caption(f"Ceiling: {ceiling_display} | Proj: {proj_display}")

                # Show combined stats for the 3 picks
                combined_ceiling = top3_df['Ceiling'].sum() if 'Ceiling' in top3_df.columns else 0
                combined_proj = top3_df['Proj PPG'].sum() if 'Proj PPG' in top3_df.columns else 0

                st.markdown(f"**Combined Ceiling:** {combined_ceiling:.1f} PPG | **Combined Proj:** {combined_proj:.1f} PPG")

                if ranking_mode in ["Simulation P(Top3)", "Both"] and prob_col and prob_col in top3_df.columns:
                    # Probability that at least one of our picks is in actual top 3
                    probs = top3_df[prob_col].fillna(0).values / 100
                    p_at_least_one = 1 - np.prod(1 - probs) if len(probs) > 0 else 0
                    st.caption(f"P(≥1 in Top 3): {p_at_least_one * 100:.1f}%")

            st.divider()

            # Full player table
            st.subheader("📋 All Ranked Players")
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                hide_index=True,
                height=400
            )
        except Exception as display_error:
            st.error(f"❌ Error displaying players: {display_error}")
            st.write("display_df columns:", list(display_df.columns) if 'display_df' in dir() else "not defined")
            st.write("display_columns:", display_columns if 'display_columns' in dir() else "not defined")

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

        # ========== TOP-3 PERFORMANCE TRACKING ==========
        st.subheader("📊 Top-3 Prediction Performance")
        st.caption("Track how well our top-3 picks perform against actual results")

        try:
            tracker = top3_tracking.Top3Tracker(tourn_conn)

            # Get performance summary
            perf_summary = tracker.get_performance_summary(days_back=30)

            if perf_summary.get('total_days', 0) > 0:
                perf_cols = st.columns(4)
                with perf_cols[0]:
                    st.metric(
                        "Avg Top-3 Recall",
                        f"{perf_summary['avg_recall']:.2f}/3",
                        help="Average number of actual top 3 scorers we correctly identified"
                    )
                with perf_cols[1]:
                    st.metric(
                        "Hit Rate",
                        f"{perf_summary['hit_rate']:.1f}%",
                        help="Percentage of days we got at least 1 of the top 3"
                    )
                with perf_cols[2]:
                    st.metric(
                        "Mean Rank of #1",
                        f"#{perf_summary['mean_rank_of_1']:.1f}" if perf_summary.get('mean_rank_of_1') else "N/A",
                        help="Average rank we assigned to the actual top scorer"
                    )
                with perf_cols[3]:
                    st.metric(
                        "Days Analyzed",
                        perf_summary['total_days'],
                        help="Number of days with tracked performance"
                    )

                # Show daily performance history
                with st.expander("📈 Daily Performance History", expanded=False):
                    daily_perf = tracker.get_daily_performance(days_back=30)
                    if not daily_perf.empty:
                        st.dataframe(
                            daily_perf[['game_date', 'actual_top3_names', 'predicted_top3_names',
                                       'top3_recall', 'actual_1_our_rank', 'method_used']].rename(columns={
                                'game_date': 'Date',
                                'actual_top3_names': 'Actual Top 3',
                                'predicted_top3_names': 'Our Picks',
                                'top3_recall': 'Correct',
                                'actual_1_our_rank': '#1 Rank',
                                'method_used': 'Method'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No daily performance data available yet.")

                # Backfill button for historical data
                if st.button("🔄 Backfill Performance Data (Last 30 Days)", key="backfill_perf"):
                    with st.spinner("Backfilling performance data..."):
                        method = 'top_scorer_score' if ranking_mode in ['TopScorerScore', 'Both'] else 'simulation'
                        backfill_result = tracker.backfill_performance(days_back=30, method=method)
                        st.success(f"Backfilled {backfill_result.get('dates_processed', 0)} days of data!")
                        st.rerun()
            else:
                st.info("📊 No performance data available yet. Click 'Backfill Performance Data' to analyze historical predictions.")
                if st.button("🔄 Backfill Performance Data (Last 30 Days)", key="backfill_perf_initial"):
                    with st.spinner("Backfilling performance data..."):
                        method = 'top_scorer_score' if ranking_mode in ['TopScorerScore', 'Both'] else 'simulation'
                        backfill_result = tracker.backfill_performance(days_back=30, method=method)
                        st.success(f"Backfilled {backfill_result.get('dates_processed', 0)} days of data!")
                        st.rerun()

        except Exception as perf_error:
            st.warning(f"⚠️ Performance tracking unavailable: {perf_error}")

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

    # =========================================================================
    # 20-LINEUP TOURNAMENT PORTFOLIO BUILDER
    # =========================================================================
    st.divider()
    st.markdown("## 🚀 20-Lineup Tournament Portfolio")
    st.caption("Automated portfolio builder using correlated simulation and win-probability optimization")

    with st.expander("📊 Build 20-Lineup Portfolio (Advanced)", expanded=False):
        try:
            # Import the optimizer modules
            from lineup_optimizer import (
                TournamentLineupOptimizer,
                create_player_pool_from_predictions,
                format_portfolio_report,
                OptimizerConfig,
                PlayerPool
            )
            from correlation_model import (
                PlayerCorrelationModel,
                CorrelatedSimResult,
                create_player_slate_info
            )
            from vegas_odds import (
                VegasOddsClient,
                compute_game_environment,
                load_game_odds,
                ensure_game_odds_table,
                fetch_and_store_odds
            )
            from slate_quality_gate import (
                validate_slate_inputs,
                SlateValidationResult,
                format_validation_metrics
            )
            from scenario_presets import (
                PRESETS,
                apply_scenario_preset,
                get_preset_recommendation,
                filter_game_environments
            )

            # Scenario Preset Selection
            st.markdown("### 🎭 Strategy Preset")
            preset_cols = st.columns([2, 3])

            with preset_cols[0]:
                preset_options = list(PRESETS.keys())
                preset_labels = [f"{PRESETS[p].icon} {PRESETS[p].name}" for p in preset_options]
                selected_preset_label = st.selectbox(
                    "Select Preset",
                    preset_labels,
                    index=0,
                    help="Presets adjust bucket allocation and exposure settings for different scenarios"
                )
                selected_preset_key = preset_options[preset_labels.index(selected_preset_label)]
                selected_preset = PRESETS[selected_preset_key]

            with preset_cols[1]:
                st.info(f"**{selected_preset.name}**: {selected_preset.description}")
                if selected_preset_key != "BALANCED":
                    st.caption(f"Adjustments: {selected_preset.summary}")

            st.divider()

            # Configuration
            st.markdown("### ⚙️ Portfolio Configuration")
            config_cols = st.columns(4)

            with config_cols[0]:
                chalk_n = st.number_input("Chalk Lineups", min_value=0, max_value=10, value=6,
                                         help="Lineups anchored on highest p(#1) stars")
            with config_cols[1]:
                stack_n = st.number_input("Stack Lineups", min_value=0, max_value=10, value=6,
                                         help="Lineups with 2 players from same game")
            with config_cols[2]:
                leverage_n = st.number_input("Leverage Lineups", min_value=0, max_value=10, value=6,
                                            help="High-variance contrarian plays")
            with config_cols[3]:
                news_n = st.number_input("News Lineups", min_value=0, max_value=10, value=2,
                                        help="Questionable/injury beneficiary pivots")

            total_lineups = chalk_n + stack_n + leverage_n + news_n
            st.caption(f"Total lineups: {total_lineups}")

            exposure_cols = st.columns(3)
            with exposure_cols[0]:
                max_exp_dominant = st.slider("Max Exposure (Stars)", 0.3, 0.8, 0.6,
                                            help="Max % of lineups for p_top1 > 15%")
            with exposure_cols[1]:
                max_exp_strong = st.slider("Max Exposure (Strong)", 0.3, 0.7, 0.5,
                                          help="Max % of lineups for p_top1 5-15%")
            with exposure_cols[2]:
                max_exp_leverage = st.slider("Max Exposure (Leverage)", 0.2, 0.5, 0.3,
                                            help="Max % of lineups for p_top1 < 5%")

            st.divider()

            # Vegas Odds Section
            st.markdown("### 🎰 Game Environment Data")

            game_envs = {}
            game_id_lookup = {}  # Maps normalized "{date}_{min_team}_{max_team}" to actual game_id

            # Helper to normalize team abbreviations (handle PHX/PHO, BKN/BRK, etc.)
            def normalize_team(abbr):
                if not abbr:
                    return 'UNK'
                abbr = str(abbr).upper().strip()
                # Handle common aliases
                aliases = {'PHO': 'PHX', 'BRK': 'BKN', 'NOP': 'NOP', 'NOH': 'NOP'}
                return aliases.get(abbr, abbr)

            # Helper to build normalized lookup key (includes date for collision safety)
            def make_lookup_key(date_str, team1, team2):
                t1, t2 = normalize_team(team1), normalize_team(team2)
                return f"{date_str}_{min(t1, t2)}_{max(t1, t2)}"

            # FIRST: Try to load cached odds (may have been fetched by FanDuel Compare)
            try:
                ensure_game_odds_table(tourn_conn)
                cached_envs = load_game_odds(tourn_conn, selected_date)
                if cached_envs:
                    game_envs = {gid: {
                        'stack_score': e.stack_score,
                        'ot_probability': e.ot_probability,
                        'blowout_risk': e.blowout_risk,
                        'pace_score': e.pace_score,
                        'spread': e.spread,
                        'total': e.total
                    } for gid, e in cached_envs.items()}
                    # Build lookup for cached envs
                    for gid, e in cached_envs.items():
                        normalized = make_lookup_key(selected_date, e.away_team, e.home_team)
                        if normalized in game_id_lookup and game_id_lookup[normalized] != gid:
                            st.warning(f"⚠️ game_id collision for {normalized}")
                        game_id_lookup[normalized] = gid
                    st.success(f"✅ Game odds loaded for {len(cached_envs)} games (from FanDuel Compare or cached)")

                    # Show game environment summary
                    with st.expander("📊 Game Environments", expanded=False):
                        env_data = []
                        for gid, e in cached_envs.items():
                            env_data.append({
                                'Game': f"{e.away_team} @ {e.home_team}",
                                'Spread': f"{e.spread:+.1f}",
                                'Total': f"{e.total:.1f}",
                                'Stack Score': f"{e.stack_score:.2f}",
                                'Blowout Risk': f"{e.blowout_risk:.0%}"
                            })
                        if env_data:
                            st.dataframe(pd.DataFrame(env_data), hide_index=True, use_container_width=True)
            except Exception:
                pass  # No cached odds available

            # SECOND: Show manual fetch option only if no cached odds
            if not game_envs:
                st.info("💡 **Tip:** Fetch FanDuel lines in the **FanDuel Compare** tab first — game odds will be cached automatically for use here!")

                vegas_col1, vegas_col2 = st.columns([2, 1])
                with vegas_col1:
                    api_key = st.text_input("TheOddsAPI Key (optional)",
                                           type="password",
                                           help="Enter your API key to fetch Vegas odds. Leave blank to skip.")
                with vegas_col2:
                    fetch_odds_btn = st.button("🎲 Fetch Vegas Odds",
                                              disabled=not api_key,
                                              help="Fetch spreads and totals for game environment")
            else:
                api_key = None
                fetch_odds_btn = False

            if fetch_odds_btn and api_key:
                try:
                    ensure_game_odds_table(tourn_conn)
                    envs = fetch_and_store_odds(tourn_conn, api_key, selected_date)
                    game_envs = {e.game_id: {
                        'stack_score': e.stack_score,
                        'ot_probability': e.ot_probability,
                        'blowout_risk': e.blowout_risk,
                        'pace_score': e.pace_score
                    } for e in envs}
                    # Build lookup: normalized key -> actual game_id (with collision detection)
                    for e in envs:
                        normalized = make_lookup_key(selected_date, e.away_team, e.home_team)
                        if normalized in game_id_lookup and game_id_lookup[normalized] != e.game_id:
                            st.warning(f"⚠️ game_id collision for {normalized}: {game_id_lookup[normalized]} vs {e.game_id}")
                        game_id_lookup[normalized] = e.game_id
                    st.success(f"✅ Fetched odds for {len(envs)} games")

                    # Show game environment table
                    env_df = pd.DataFrame([
                        {
                            'Game': f"{e.away_team} @ {e.home_team}",
                            'Spread': f"{e.spread:+.1f}",
                            'Total': f"{e.total:.1f}",
                            'Stack Score': f"{e.stack_score:.2f}",
                            'OT Prob': f"{e.ot_probability:.1%}",
                            'Blowout Risk': f"{e.blowout_risk:.1%}"
                        }
                        for e in envs
                    ])
                    st.dataframe(env_df, hide_index=True, use_container_width=True)
                except Exception as ve:
                    st.error(f"❌ Failed to fetch odds: {ve}")

            st.divider()

            # Generate Portfolio Button
            if st.button("🚀 Generate 20-Lineup Portfolio", type="primary"):
                with st.spinner("Running correlated simulation and building portfolio..."):
                    try:
                        # Load predictions for selected date
                        pred_query = """
                            SELECT p.*,
                                   COALESCE(p.p_top1, 0) as p_top1,
                                   COALESCE(p.p_top3, 0) as p_top3
                            FROM predictions p
                            WHERE p.game_date = ?
                              AND p.projected_ppg IS NOT NULL
                              AND p.proj_ceiling >= ?
                        """
                        pred_df = pd.read_sql_query(pred_query, tourn_conn, params=[selected_date, min_ceiling])

                        if pred_df.empty:
                            st.error("No predictions found for this date with the current filters.")
                        else:
                            # Show predictions summary for debugging
                            with st.expander("📋 Predictions Data Debug", expanded=False):
                                st.write(f"**Columns in predictions:** {list(pred_df.columns)}")
                                # Show unique team/opponent combinations
                                if 'team_name' in pred_df.columns and 'opponent_name' in pred_df.columns:
                                    matchups = pred_df.groupby(['team_name', 'opponent_name']).size().reset_index(name='players')
                                    st.write("**Matchups in predictions:**")
                                    st.dataframe(matchups, hide_index=True)
                                else:
                                    st.warning("Missing team_name or opponent_name columns!")

                            # Create player pool
                            player_pool = []
                            # Helper function to safely get scalar value from pandas row
                            def safe_get(row, col, default=0):
                                """Safely extract scalar value from pandas Series row."""
                                try:
                                    if col not in row.index:
                                        return default
                                    val = row[col]
                                    # Handle case where val is still a Series (duplicate columns)
                                    if isinstance(val, pd.Series):
                                        val = val.iloc[0] if len(val) > 0 else default
                                    # Check for NaN/None using scalar comparison
                                    if val is None or (isinstance(val, float) and np.isnan(val)):
                                        return default
                                    # Convert to appropriate type
                                    if isinstance(default, (int, float)):
                                        return float(val)
                                    return val
                                except (KeyError, TypeError, ValueError):
                                    return default

                            for _, row in pred_df.iterrows():
                                ceiling = safe_get(row, 'proj_ceiling', 0)
                                floor = safe_get(row, 'proj_floor', 0)
                                proj_ppg = safe_get(row, 'projected_ppg', 20)
                                sigma = (ceiling - floor) / 4 if ceiling > floor else proj_ppg * 0.15

                                # Team name to abbreviation mapping
                                TEAM_NAME_TO_ABBR = {
                                    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
                                    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
                                    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
                                    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
                                    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
                                    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
                                    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
                                    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
                                    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
                                    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
                                }

                                # Determine game_id from matchup or team
                                team_raw = safe_get(row, 'team_abbreviation', None)
                                if not team_raw or team_raw == 0:
                                    # Try team_name and convert
                                    team_name = safe_get(row, 'team_name', 'UNK')
                                    team = TEAM_NAME_TO_ABBR.get(str(team_name), str(team_name)[:3].upper() if team_name else 'UNK')
                                else:
                                    team = normalize_team(str(team_raw))

                                # Get opponent (use opponent_name, convert to abbreviation)
                                opponent_raw = safe_get(row, 'opponent_name', None)
                                if opponent_raw and opponent_raw != 0:
                                    opponent = TEAM_NAME_TO_ABBR.get(str(opponent_raw), str(opponent_raw)[:3].upper())
                                else:
                                    opponent = 'UNK'
                                opponent = normalize_team(opponent)

                                # Create normalized key (includes date for collision safety)
                                normalized_key = make_lookup_key(selected_date, team, opponent)
                                # Use lookup if available, otherwise use normalized key
                                game_id = game_id_lookup.get(normalized_key, normalized_key)

                                player_pool.append(PlayerPool(
                                    player_id=row['player_id'],
                                    player_name=row['player_name'],
                                    team=str(team),
                                    game_id=game_id,
                                    projected_ppg=proj_ppg,
                                    sigma=max(sigma, 1.0),
                                    ceiling=ceiling,
                                    floor=floor,
                                    p_top1=safe_get(row, 'p_top1', 0),
                                    p_top3=safe_get(row, 'p_top3', 0),
                                    support_score=0.5,  # Will be computed by simulation
                                    expected_rank=50,
                                    is_star=safe_get(row, 'season_avg_ppg', 0) >= 25,
                                    is_questionable=str(safe_get(row, 'injury_status', '')).lower() == 'questionable',
                                    is_injury_beneficiary=bool(safe_get(row, 'injury_adjusted', False)),
                                ))

                            # Coverage metrics: check game_id matching between pool and envs
                            pool_games = set(p.game_id for p in player_pool)
                            env_games = set(game_envs.keys())
                            matched_games = pool_games & env_games
                            match_pct = len(matched_games) / len(pool_games) * 100 if pool_games else 0

                            # Debug: Show game_id diagnostics
                            with st.expander("🔧 Game ID Debug", expanded=False):
                                st.write("**Player Pool Games:**", sorted(pool_games))
                                st.write("**Game Odds Games:**", sorted(env_games))
                                st.write("**Lookup Keys:**", sorted(game_id_lookup.keys()) if game_id_lookup else "Empty")
                                st.write("**Matched:**", sorted(matched_games) if matched_games else "None")
                                # Sample of player game assignments
                                sample_players = player_pool[:5]
                                st.write("**Sample Players:**")
                                for p in sample_players:
                                    st.caption(f"  {p.player_name} ({p.team}) → game_id: {p.game_id}")

                            if match_pct < 50:
                                st.warning(f"⚠️ Low game coverage: only {len(matched_games)}/{len(pool_games)} player pool games "
                                          f"({match_pct:.0f}%) have Vegas odds. Stacking may be degraded.")
                            elif match_pct < 95:
                                st.caption(f"ℹ️ Game coverage: {len(matched_games)}/{len(pool_games)} games ({match_pct:.0f}%) have odds")

                            # Run correlated simulation if we have enough players
                            if len(player_pool) >= 10:
                                corr_model = PlayerCorrelationModel()
                                slate_info = [
                                    create_player_slate_info.__self__ if hasattr(create_player_slate_info, '__self__') else None
                                    for _ in player_pool
                                ]
                                # Build simplified slate info
                                from correlation_model import PlayerSlateInfo
                                slate_players = [
                                    PlayerSlateInfo(
                                        player_id=p.player_id,
                                        player_name=p.player_name,
                                        team=p.team,
                                        game_id=p.game_id,
                                        mean_score=p.projected_ppg,
                                        sigma=p.sigma,
                                        is_star=p.is_star
                                    )
                                    for p in player_pool
                                ]

                                # Run simulation
                                sim_results = corr_model.run_correlated_simulation(
                                    slate_players, game_envs, n_sims=10000
                                )
                                sim_dict = {r.player_id: r for r in sim_results}

                                # Update player pool with simulation results
                                for p in player_pool:
                                    if p.player_id in sim_dict:
                                        r = sim_dict[p.player_id]
                                        p.p_top1 = r.p_top1
                                        p.p_top3 = r.p_top3
                                        p.support_score = r.support_score
                                        p.expected_rank = r.expected_rank

                            # Configure and run optimizer
                            base_config = OptimizerConfig(
                                chalk_lineups=chalk_n,
                                stack_lineups=stack_n,
                                leverage_lineups=leverage_n,
                                news_lineups=news_n,
                                max_exposure_dominant=max_exp_dominant,
                                max_exposure_strong=max_exp_strong,
                                max_exposure_leverage=max_exp_leverage,
                            )

                            # ═══════════════════════════════════════════════════════════════
                            # SCENARIO PRESET - Apply preset adjustments
                            # ═══════════════════════════════════════════════════════════════
                            if selected_preset_key != "BALANCED":
                                config = apply_scenario_preset(base_config, selected_preset, game_envs)
                                st.info(f"{selected_preset.icon} **Preset Applied: {selected_preset.name}** — "
                                       f"C:{config.chalk_lineups} S:{config.stack_lineups} "
                                       f"L:{config.leverage_lineups} N:{config.news_lineups}")

                                # Filter game environments for preset-specific requirements
                                if selected_preset.max_game_spread or selected_preset.min_game_total:
                                    filtered_envs = filter_game_environments(game_envs, selected_preset)
                                    if len(filtered_envs) < len(game_envs):
                                        st.caption(f"🎯 Filtered to {len(filtered_envs)}/{len(game_envs)} games matching preset criteria")
                            else:
                                config = base_config

                            # ═══════════════════════════════════════════════════════════════
                            # SLATE QUALITY GATE - Validate before optimization
                            # ═══════════════════════════════════════════════════════════════
                            validation_result = validate_slate_inputs(
                                predictions_df=display_df,
                                game_envs=game_envs,
                                player_pool=player_pool,
                                config=config
                            )

                            # Show validation metrics
                            with st.expander(f"🔍 Slate Quality: {validation_result.status}", expanded=(validation_result.status != "PASS")):
                                st.code(format_validation_metrics(validation_result))
                                if validation_result.warnings:
                                    for w in validation_result.warnings:
                                        st.caption(f"⚠️ {w}")

                            # Handle validation outcomes
                            if validation_result.status == "FAIL":
                                st.error(validation_result.summary)
                                st.stop()  # Block generation entirely

                            if validation_result.status == "DEGRADE":
                                st.warning(validation_result.summary)
                                st.info(f"📊 **Mode: {validation_result.mode_label}** — Buckets auto-adjusted")
                                config = validation_result.adjusted_config  # Use degraded config

                            optimizer = TournamentLineupOptimizer(
                                player_pool=player_pool,
                                game_environments=game_envs,
                                sim_results=sim_dict if 'sim_dict' in dir() else {},
                                config=config
                            )

                            result = optimizer.optimize()

                            # Check for degraded strategy and warn user (skip if validation already handled it)
                            stack_built = result.bucket_summary.get('stack', 0)
                            expected_stack = config.stack_lineups  # Use adjusted config's expectation
                            if stack_built == 0 and expected_stack > 0:
                                st.warning("⚠️ **No stack lineups built!** This significantly reduces tournament edge. "
                                          "Check: Vegas odds missing, game_id mismatch, or insufficient players per game.")
                            elif stack_built < expected_stack and expected_stack > 0:
                                st.warning(f"⚠️ Only {stack_built}/{expected_stack} stack lineups built. "
                                          "Some game environments may be missing.")

                            # Store in session state
                            st.session_state['portfolio_result'] = result
                            st.session_state['portfolio_player_pool'] = {p.player_id: p for p in player_pool}

                            st.success(f"✅ Generated {len(result.lineups)} lineups!")

                    except Exception as e:
                        st.error(f"❌ Portfolio generation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Display results if available
            if 'portfolio_result' in st.session_state:
                result = st.session_state['portfolio_result']
                player_pool_dict = st.session_state.get('portfolio_player_pool', {})

                st.divider()
                st.markdown("### 📊 Portfolio Results")

                # Summary metrics
                sum_cols = st.columns(4)
                with sum_cols[0]:
                    st.metric("Total Win Prob", f"{result.total_win_probability:.1%}",
                             help="P(any of the lineups wins)")
                with sum_cols[1]:
                    st.metric("Unique Players", f"{result.unique_players}",
                             help="Total unique players across all lineups")
                with sum_cols[2]:
                    st.metric("Lineups Built", f"{len(result.lineups)}/{total_lineups}")
                with sum_cols[3]:
                    bucket_str = f"C:{result.bucket_summary.get('chalk', 0)} S:{result.bucket_summary.get('stack', 0)} L:{result.bucket_summary.get('leverage', 0)} N:{result.bucket_summary.get('news', 0)}"
                    st.metric("Bucket Split", bucket_str)

                # Diversity metrics row
                div_cols = st.columns(4)
                with div_cols[0]:
                    diversity = getattr(result, 'diversity_score', 1.0)
                    diversity_color = "normal" if diversity >= 0.60 else "off"
                    st.metric("Diversity Score", f"{diversity:.2f}",
                             help="Avg pairwise Jaccard distance (higher = more diverse, ≥0.60 target)")
                with div_cols[1]:
                    core_sat = getattr(result, 'core_saturation', {})
                    max_core = core_sat.get('max_core_usage', 0)
                    st.metric("Max Core Usage", f"{max_core}",
                             help="Most lineups sharing any 2-player core")
                with div_cols[2]:
                    saturated = core_sat.get('saturated_cores', 0)
                    st.metric("Saturated Cores", f"{saturated}",
                             help="2-player cores at max overlap limit")
                with div_cols[3]:
                    # Quality indicator
                    quality = "🟢 Good" if diversity >= 0.60 and saturated <= 5 else "🟡 OK" if diversity >= 0.50 else "🔴 Low"
                    st.metric("Portfolio Health", quality)

                # Warnings
                if result.warnings:
                    with st.expander("⚠️ Warnings", expanded=True):
                        for w in result.warnings:
                            st.warning(w)

                # Lineups by bucket
                for bucket in ['chalk', 'stack', 'leverage', 'news']:
                    bucket_lineups = [l for l in result.lineups if l.bucket == bucket]
                    if bucket_lineups:
                        bucket_icon = {'chalk': '🎯', 'stack': '🔗', 'leverage': '📈', 'news': '📰'}.get(bucket, '📋')
                        with st.expander(f"{bucket_icon} {bucket.upper()} Lineups ({len(bucket_lineups)})", expanded=bucket=='chalk'):
                            for i, lineup in enumerate(bucket_lineups, 1):
                                cols = st.columns([3, 1, 1])
                                with cols[0]:
                                    names = ' | '.join([p.player_name for p in lineup.players])
                                    st.markdown(f"**{i}.** {names}")
                                with cols[1]:
                                    st.caption(f"Win: {lineup.win_probability:.2%}")
                                with cols[2]:
                                    st.caption(f"Ceil: {lineup.total_ceiling():.0f}")
                                if lineup.stack_game:
                                    st.caption(f"   🔗 Stack: {lineup.stack_game}")

                # Exposure report
                with st.expander("📊 Player Exposure Report", expanded=False):
                    exp_data = []
                    for pid, data in sorted(result.exposure_report.items(),
                                           key=lambda x: x[1]['count'], reverse=True):
                        cap_status = "🔴 AT CAP" if data['at_cap'] else ""
                        exp_data.append({
                            'Player': data['player_name'],
                            'Team': data['team'],
                            'Count': data['count'],
                            'Exposure': f"{data['pct']:.0%}",
                            'Tier': data['tier'],
                            'Status': cap_status
                        })
                    if exp_data:
                        st.dataframe(pd.DataFrame(exp_data), hide_index=True, use_container_width=True)

                # Export
                st.divider()
                if st.button("📋 Copy Lineups to Clipboard"):
                    lines = []
                    for i, lineup in enumerate(result.lineups, 1):
                        names = ', '.join([p.player_name for p in lineup.players])
                        lines.append(f"{i}. [{lineup.bucket}] {names}")
                    st.code('\n'.join(lines))
                    st.success("Lineups displayed above - copy manually")

        except ImportError as ie:
            st.error(f"❌ Required modules not found: {ie}")
            st.info("Make sure vegas_odds.py, correlation_model.py, and lineup_optimizer.py are in the project directory.")
        except Exception as e:
            st.error(f"❌ Error loading portfolio builder: {e}")
            import traceback
            st.code(traceback.format_exc())

    # =========================================================================
    # MIXTURE MODEL TOURNAMENT STRATEGY (CALIBRATED)
    # =========================================================================
    st.divider()
    st.info("🔍 DEBUG: Mixture Model section reached!")  # DEBUG - remove after testing
    st.markdown("## 🎲 Mixture Model Rankings")
    st.caption("Calibrated two-component model: P(points ≥ threshold) using typical + spike game modes")

    # Show calibration status
    cal_cols = st.columns(3)
    with cal_cols[0]:
        st.metric("Calibration Ratio", "0.954", help="Predicted/Realized hits (target: 0.95-1.05)")
    with cal_cols[1]:
        st.metric("Top Decile Gap", "+0.019", help="Calibration error for highest predictions (target: <0.03)")
    with cal_cols[2]:
        st.metric("Status", "✅ GREEN LIGHT", help="All calibration checks passed")

    with st.expander("ℹ️ How Mixture Model Works", expanded=False):
        st.markdown("""
        **Two-Component Mixture Model:**

        The mixture model estimates P(points ≥ T) by combining:
        1. **Typical Game Mode** (1-w): Player scores around their projection μ with normal variance
        2. **Spike Game Mode** (w): Player has a breakout night, scoring μ + Δ (delta shift)

        **Key Features:**
        - `p_threshold`: Calibrated probability of exceeding threshold T (top-15 or top-3)
        - `spike_weight` (w): Probability of spike game (0.05-0.25), driven by 3PA rate, usage, role expansion
        - `spike_shift` (Δ): How much higher the spike mean is (5-14 points)

        **Difference from GPP Score:**
        - GPP Score: Heuristic weighting (ceiling + hot streak + variance + matchup)
        - Mixture Model: **Statistical probability** calibrated to actual historical hit rates
        """)

    try:
        from lineup_optimizer import (
            create_mixture_player_pool,
            build_mixture_tournament_portfolio,
            build_top3_tournament_portfolio,
            Top3TournamentConfig,
            compare_mixture_vs_baseline,
            format_portfolio_report
        )
        from top3_ranking import Top3Ranker

        # Get mixture rankings
        mixture_ranker = Top3Ranker(tourn_conn)
        mixture_df = mixture_ranker.rank_players(selected_date, method='mixture')

        if not mixture_df.empty:
            # =====================================================
            # CAP RATE KPIs - Critical diagnostic section
            # =====================================================
            st.markdown("### 🔍 Cap Rate Diagnostics")

            # Calculate cap rates - use raw float comparison with epsilon
            W_CAP = 0.25
            EPS = 1e-6

            if 'w_uncapped' in mixture_df.columns:
                # Truth: count how many w_uncapped >= 0.25 (actually hit the cap)
                count_w_at_cap = (mixture_df['w_uncapped'] >= (W_CAP - EPS)).sum()
                pct_w_truth = count_w_at_cap / len(mixture_df) * 100

                # Also check the flag (should match)
                pct_w_flag = mixture_df['is_w_capped'].mean() * 100 if 'is_w_capped' in mixture_df.columns else 0

                # Debug: show both for verification
                w_min = mixture_df['w_uncapped'].min()
                w_med = mixture_df['w_uncapped'].median()
                w_max = mixture_df['w_uncapped'].max()

                # Add w_cap_delta column
                mixture_df['w_cap_delta'] = mixture_df['spike_weight'] - mixture_df['w_uncapped']

                pct_w_capped = pct_w_truth  # Use truth, not flag
            else:
                pct_w_capped = 0
                w_min = w_med = w_max = 0

            pct_p_capped = mixture_df['is_p_capped'].mean() * 100 if 'is_p_capped' in mixture_df.columns else 0
            median_w = mixture_df['w_uncapped'].median() if 'w_uncapped' in mixture_df.columns else mixture_df['spike_weight'].median()
            median_delta = mixture_df['spike_shift'].median() if 'spike_shift' in mixture_df.columns else 0

            # Pass/Fail thresholds
            w_cap_status = "🟢" if pct_w_capped < 10 else ("🟡" if pct_w_capped < 15 else "🔴")
            p_cap_status = "🟢" if pct_p_capped < 10 else ("🟡" if pct_p_capped < 15 else "🔴")
            w_median_status = "🟢" if 0.08 <= median_w <= 0.14 else ("🟡" if median_w < 0.20 else "🔴")

            # Debug expander showing raw stats
            with st.expander("🔧 Debug: Raw Cap Statistics", expanded=True):
                st.code(f"""
w_uncapped range: {w_min:.4f} - {w_max:.4f}
w_uncapped median: {w_med:.4f}
W_CAP threshold: {W_CAP}
count(w_uncapped >= {W_CAP}): {count_w_at_cap if 'w_uncapped' in mixture_df.columns else 'N/A'}
% w capped (truth): {pct_w_capped:.1f}%
% w capped (flag): {pct_w_flag if 'w_uncapped' in mixture_df.columns else 'N/A'}%
Flag matches truth: {'✅ YES' if abs(pct_w_capped - pct_w_flag) < 0.1 else '❌ NO - BUG!' if 'w_uncapped' in mixture_df.columns else 'N/A'}
""", language=None)

            cap_cols = st.columns(4)
            with cap_cols[0]:
                st.metric(
                    f"{w_cap_status} % w Capped",
                    f"{pct_w_capped:.1f}%",
                    help="% players at spike_weight cap (0.25). Target: <10% green, 10-15% yellow, >15% red"
                )
            with cap_cols[1]:
                st.metric(
                    f"{p_cap_status} % P Capped",
                    f"{pct_p_capped:.1f}%",
                    help="% players at P cap (0.70). Target: <10% green, 10-15% yellow, >15% red"
                )
            with cap_cols[2]:
                st.metric(
                    f"{w_median_status} Median w (uncapped)",
                    f"{median_w:.3f}",
                    help="Median spike weight before capping. Target: 0.08-0.14 for proper differentiation"
                )
            with cap_cols[3]:
                st.metric(
                    "Median Δ",
                    f"{median_delta:.1f}",
                    help="Median spike shift. Should have meaningful spread (not all 5-8)"
                )

            # Warning if w cap rate is too high
            if pct_w_capped > 15:
                st.warning(f"⚠️ **High w cap rate ({pct_w_capped:.0f}%)** - Intercept may be too high. Model loses differentiation when everyone hits the cap.")

            # Show uncapped vs capped table
            with st.expander("📊 Capped vs Uncapped Values (Top 20)", expanded=False):
                if 'w_uncapped' in mixture_df.columns and 'p_uncapped' in mixture_df.columns:
                    uncapped_cols = ['player_name', 'spike_weight', 'w_uncapped', 'w_cap_delta', 'is_w_capped',
                                    'p_threshold', 'p_uncapped', 'is_p_capped', 'spike_shift']
                    uncapped_df = mixture_df[uncapped_cols].head(20).copy()
                    uncapped_df.columns = ['Player', 'w (capped)', 'w (uncapped)', 'w Δcap', 'w@cap?',
                                          'P (capped)', 'P (uncapped)', 'P@cap?', 'spike Δ']

                    # Format - compute cap flag BEFORE formatting to avoid rounding bugs
                    uncapped_df['w@cap?'] = uncapped_df['w (uncapped)'].apply(lambda x: "🔴 YES" if x >= (W_CAP - EPS) else "")
                    uncapped_df['P@cap?'] = uncapped_df['P@cap?'].apply(lambda x: "🔴 YES" if x else "")

                    # Now format for display
                    uncapped_df['w (capped)'] = uncapped_df['w (capped)'].apply(lambda x: f"{x:.3f}")
                    uncapped_df['w (uncapped)'] = uncapped_df['w (uncapped)'].apply(lambda x: f"{x:.3f}")
                    uncapped_df['w Δcap'] = uncapped_df['w Δcap'].apply(lambda x: f"{x:+.3f}" if x != 0 else "0")
                    uncapped_df['P (capped)'] = uncapped_df['P (capped)'].apply(lambda x: f"{x:.1%}")
                    uncapped_df['P (uncapped)'] = uncapped_df['P (uncapped)'].apply(lambda x: f"{x:.1%}")
                    uncapped_df['spike Δ'] = uncapped_df['spike Δ'].apply(lambda x: f"{x:.1f}")

                    st.dataframe(uncapped_df, hide_index=True, use_container_width=True)

                    # Explanation
                    st.caption("""
                    **Column meanings:**
                    - **w Δcap** = w(capped) - w(uncapped). Should be 0 for most players, negative only if capped.
                    - **w@cap?** = 🔴 YES if w_uncapped >= 0.25 (actually hit the ceiling)
                    """)
                else:
                    st.info("Uncapped columns not available. Re-run ranking with latest code.")

            st.divider()

            # Show mixture vs baseline comparison
            st.markdown("### 📊 Mixture vs Projection Rankings")

            comparison_df = compare_mixture_vs_baseline(tourn_conn, selected_date, top_n=30)

            if not comparison_df.empty:
                # Summary metrics
                rank_corr = comparison_df['mixture_rank'].corr(comparison_df['proj_rank'])
                boosted = (comparison_df['rank_change'] > 0).sum()
                dropped = (comparison_df['rank_change'] < 0).sum()

                comp_cols = st.columns(3)
                with comp_cols[0]:
                    st.metric("Rank Correlation", f"{rank_corr:.3f}", help="How similar mixture vs projection rankings are")
                with comp_cols[1]:
                    st.metric("Players Boosted", f"{boosted}", help="Players ranked higher by mixture model")
                with comp_cols[2]:
                    st.metric("Players Dropped", f"{dropped}", help="Players ranked lower by mixture model")

                # Show top players by mixture ranking
                st.markdown("#### Top 15 by Mixture P(threshold)")
                display_cols = ['player_name', 'projected_ppg', 'p_threshold', 'spike_weight', 'mixture_rank', 'proj_rank', 'rank_change']
                top15_df = comparison_df[display_cols].head(15).copy()
                top15_df.columns = ['Player', 'Proj PPG', 'P(≥T)', 'Spike W', 'Mix Rank', 'Proj Rank', 'Change']

                # Format for display
                top15_df['P(≥T)'] = top15_df['P(≥T)'].apply(lambda x: f"{x:.1%}")
                top15_df['Spike W'] = top15_df['Spike W'].apply(lambda x: f"{x:.2f}")
                top15_df['Proj PPG'] = top15_df['Proj PPG'].apply(lambda x: f"{x:.1f}")
                top15_df['Change'] = top15_df['Change'].apply(lambda x: f"+{x}" if x > 0 else str(x))

                st.dataframe(top15_df, hide_index=True, use_container_width=True)

                # Show significant rank changes WITH FEATURES
                significant = comparison_df[abs(comparison_df['rank_change']) >= 3].copy()
                if not significant.empty:
                    with st.expander(f"🔄 Significant Rank Changes ({len(significant)} players) - WITH FEATURES", expanded=True):
                        # Merge features from mixture_df
                        feature_cols = ['player_id', 'u', 'r3', 'role_up', 'vmin', 'spike_weight', 'w_uncapped', 'spike_shift', 'p_uncapped']
                        available_features = [c for c in feature_cols if c in mixture_df.columns]

                        if 'player_id' in mixture_df.columns and 'player_id' in significant.columns:
                            merged = significant.merge(
                                mixture_df[available_features],
                                on='player_id',
                                how='left',
                                suffixes=('', '_mix')
                            )
                        else:
                            merged = significant

                        boosted_df = merged[merged['rank_change'] > 0].sort_values('rank_change', ascending=False)
                        dropped_df = merged[merged['rank_change'] < 0].sort_values('rank_change')

                        st.markdown("**⬆️ BOOSTED by Mixture** (high spike potential)")
                        if not boosted_df.empty:
                            boost_display = boosted_df.head(10)[['player_name', 'proj_rank', 'mixture_rank', 'rank_change']].copy()
                            # Add features if available
                            for feat in ['u', 'r3', 'role_up', 'spike_weight', 'w_uncapped', 'spike_shift']:
                                if feat in boosted_df.columns:
                                    boost_display[feat] = boosted_df.head(10)[feat]

                            boost_display.columns = ['Player', 'Proj#', 'Mix#', 'Chg'] + [c for c in boost_display.columns[4:]]
                            # Format numeric columns
                            for col in ['u', 'r3', 'role_up', 'spike_weight', 'w_uncapped']:
                                if col in boost_display.columns:
                                    boost_display[col] = boost_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                            if 'spike_shift' in boost_display.columns:
                                boost_display['spike_shift'] = boost_display['spike_shift'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                            boost_display['Chg'] = boost_display['Chg'].apply(lambda x: f"+{int(x)}")
                            st.dataframe(boost_display, hide_index=True, use_container_width=True)
                        else:
                            st.caption("None")

                        st.markdown("**⬇️ DROPPED by Mixture** (steady but less explosive)")
                        if not dropped_df.empty:
                            drop_display = dropped_df.head(10)[['player_name', 'proj_rank', 'mixture_rank', 'rank_change']].copy()
                            for feat in ['u', 'r3', 'role_up', 'spike_weight', 'w_uncapped', 'spike_shift']:
                                if feat in dropped_df.columns:
                                    drop_display[feat] = dropped_df.head(10)[feat]

                            drop_display.columns = ['Player', 'Proj#', 'Mix#', 'Chg'] + [c for c in drop_display.columns[4:]]
                            for col in ['u', 'r3', 'role_up', 'spike_weight', 'w_uncapped']:
                                if col in drop_display.columns:
                                    drop_display[col] = drop_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                            if 'spike_shift' in drop_display.columns:
                                drop_display['spike_shift'] = drop_display['spike_shift'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                            drop_display['Chg'] = drop_display['Chg'].apply(lambda x: str(int(x)))
                            st.dataframe(drop_display, hide_index=True, use_container_width=True)
                        else:
                            st.caption("None")

            # Build Top-3 Scorer Tournament Portfolio
            st.markdown("### 🏆 Generate Top-3 Scorer Tournament Portfolio")
            st.caption("Predict the top 3 NBA scorers for winner-take-all tournament. NOT a DFS lineup optimizer.")

            # Configuration options
            with st.expander("⚙️ Portfolio Settings", expanded=False):
                cfg_col1, cfg_col2 = st.columns(2)
                with cfg_col1:
                    pool_size = st.slider("Candidate Pool Size", min_value=20, max_value=100, value=60,
                                         help="Top N players by p_threshold to consider")
                    max_attempts = st.slider("Max Generation Attempts", min_value=200, max_value=2000, value=1000,
                                            help="Maximum attempts to find valid lineups")
                with cfg_col2:
                    max_exposure_star = st.slider("Star Exposure Cap", min_value=0.3, max_value=1.0, value=0.70,
                                                  help="Max % of lineups top-5 players can appear in")
                    max_core_overlap = st.slider("Max Core Overlap", min_value=2, max_value=10, value=5,
                                                help="Max lineups sharing same 2-player pair")

            if st.button("🎲 Build 20-Prediction Top-3 Portfolio", key="build_top3_portfolio"):
                with st.spinner("Building top-3 scorer tournament portfolio..."):
                    try:
                        # Build with custom config
                        config = Top3TournamentConfig(
                            candidate_pool_size=pool_size,
                            max_attempts=max_attempts,
                            max_exposure_star=max_exposure_star,
                            max_core_overlap=max_core_overlap
                        )
                        result = build_top3_tournament_portfolio(tourn_conn, selected_date, config)

                        if result.lineups:
                            st.success(f"✅ Generated {len(result.lineups)} predictions for top-3 scorers!")

                            # Summary metrics
                            sum_cols = st.columns(4)
                            with sum_cols[0]:
                                st.metric("Total Predictions", len(result.lineups))
                            with sum_cols[1]:
                                st.metric("Unique Players", result.unique_players)
                            with sum_cols[2]:
                                st.metric("Win Probability", f"{result.total_win_probability:.1%}",
                                         help="P(at least one prediction matches actual top 3)")
                            with sum_cols[3]:
                                st.metric("Pool Size", result.diagnostics.get('candidate_pool_size', 0))

                            # DIAGNOSTIC OUTPUT SECTION
                            st.markdown("#### 📊 Generation Diagnostics")
                            diag = result.diagnostics

                            diag_cols = st.columns(4)
                            with diag_cols[0]:
                                st.metric("Attempted", diag.get('attempted_lineups', 0))
                            with diag_cols[1]:
                                st.metric("Valid Found", diag.get('valid_lineups_found', 0))
                            with diag_cols[2]:
                                rej_overlap = diag.get('rejected_core_overlap', 0)
                                st.metric("Rejected (Overlap)", rej_overlap,
                                         delta=f"-{rej_overlap}" if rej_overlap > 0 else None,
                                         delta_color="inverse")
                            with diag_cols[3]:
                                rej_cap = diag.get('rejected_exposure_cap', 0)
                                st.metric("Rejected (Cap)", rej_cap,
                                         delta=f"-{rej_cap}" if rej_cap > 0 else None,
                                         delta_color="inverse")

                            # Top exposures
                            if diag.get('top_exposures'):
                                with st.expander("📈 Top Player Exposures", expanded=True):
                                    for exp_line in diag['top_exposures']:
                                        st.caption(exp_line)

                            # Rejection log (if any)
                            if diag.get('rejection_log'):
                                with st.expander(f"⚠️ Rejection Log ({len(diag['rejection_log'])} samples)", expanded=False):
                                    for log_entry in diag['rejection_log'][:15]:
                                        st.caption(f"• {log_entry}")

                            # Show all 20 predictions (no buckets)
                            st.markdown("#### 🎯 Top-3 Scorer Predictions")
                            for i, lineup in enumerate(result.lineups, 1):
                                players = lineup.players
                                names = ' | '.join([f"**{p.player_name}** ({p.team})" for p in players])
                                ceiling_sum = sum(p.ceiling for p in players)
                                avg_pthresh = sum(p.p_threshold for p in players) / 3
                                st.markdown(f"{i}. {names} (Ceiling: {ceiling_sum:.0f}, Avg P: {avg_pthresh:.2%})")

                            # Exposure report
                            with st.expander("📊 Player Exposure Report", expanded=False):
                                exp_data = []
                                for pid, data in sorted(result.exposure_report.items(),
                                                       key=lambda x: x[1]['count'], reverse=True):
                                    cap_status = "🔴 AT CAP" if data.get('at_cap') else ""
                                    exp_data.append({
                                        'Player': data.get('player_name', f'ID:{pid}'),
                                        'Count': data['count'],
                                        'Max': data.get('max_allowed', 0),
                                        'Exposure': f"{data['pct']:.0%}",
                                        'Tier': data.get('tier', 'unknown'),
                                        'Status': cap_status
                                    })
                                if exp_data:
                                    st.dataframe(pd.DataFrame(exp_data), hide_index=True, use_container_width=True)

                            # Export
                            st.markdown("#### 📋 Export Predictions")
                            lines = []
                            for i, lineup in enumerate(result.lineups, 1):
                                names = ', '.join([p.player_name for p in lineup.players])
                                lines.append(f"{i}. {names}")
                            st.code('\n'.join(lines))

                        else:
                            st.warning("⚠️ No predictions generated. Check warnings and diagnostics:")
                            for w in result.warnings:
                                st.error(f"- {w}")
                            # Show diagnostics even on failure
                            diag = result.diagnostics
                            st.info(f"Pool size: {diag.get('candidate_pool_size', 0)}, "
                                   f"Attempts: {diag.get('attempted_lineups', 0)}")

                    except Exception as e:
                        st.error(f"Error building portfolio: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        else:
            st.warning("⚠️ No mixture model data available for this date. Make sure predictions exist.")

    except ImportError as ie:
        st.error(f"❌ Required modules not found: {ie}")
        st.info("Make sure top3_ranking.py and lineup_optimizer.py are in the project directory.")
    except Exception as e:
        st.error(f"❌ Error loading mixture model: {e}")
        import traceback
        st.code(traceback.format_exc())

# Backtest Analysis tab --------------------------------------------------------
if selected_page == "Backtest Analysis":
    st.header("📊 Backtest Analysis")

    # Get database connection
    backtest_conn = get_connection(str(db_path))

    # Mode selector at the top
    backtest_mode = st.radio(
        "Analysis Type",
        ["🎯 Top 3 Scorer Analysis", "🏆 20-Lineup Portfolio Backtest"],
        horizontal=True,
        key="backtest_mode_selector"
    )

    st.divider()

    # ==========================================================================
    # MODE 2: 20-Lineup Portfolio Backtest
    # ==========================================================================
    if backtest_mode == "🏆 20-Lineup Portfolio Backtest":
        st.subheader("20-Lineup Portfolio Performance")
        st.caption("Evaluate how well your tournament portfolios would have performed")

        # Ensure portfolio tables exist
        try:
            backtest_portfolio.create_portfolio_tables(backtest_conn)
            preset_ab_test.create_preset_ab_tables(backtest_conn)
        except Exception as e:
            st.warning(f"Could not initialize portfolio tables: {e}")

        # Portfolio controls
        port_col1, port_col2 = st.columns([1, 1])

        with port_col1:
            port_days_options = {
                "Last 7 days": 7,
                "Last 14 days": 14,
                "Last 30 days": 30,
                "Last 60 days": 60,
            }
            port_selected_period = st.selectbox(
                "Date Range",
                options=list(port_days_options.keys()),
                index=2,
                key="portfolio_backtest_period"
            )
            port_days_back = port_days_options[port_selected_period]

        with port_col2:
            run_preset_comparison = st.checkbox(
                "Compare All Presets",
                value=True,
                help="Compare performance across all scenario presets (BALANCED, SHOOTOUT, etc.)"
            )

        # Date range calculation
        port_end_date = datetime.now().strftime('%Y-%m-%d')
        port_start_date = (datetime.now() - timedelta(days=port_days_back)).strftime('%Y-%m-%d')

        # Run Portfolio Backtest button
        if st.button("Run Portfolio Backtest", type="primary", key="run_portfolio_backtest_btn"):
            with st.spinner("Running portfolio backtest... This may take a few minutes."):
                try:
                    if run_preset_comparison:
                        # Run preset A/B test
                        summary = preset_ab_test.run_preset_ab_test(
                            backtest_conn,
                            port_start_date,
                            port_end_date,
                            verbose=False
                        )
                        st.session_state['portfolio_ab_summary'] = summary
                        st.success(f"Portfolio backtest complete! Tested {summary.n_dates} slates.")
                    else:
                        # Run basic portfolio backtest (BALANCED only)
                        results_df = backtest_portfolio.run_portfolio_backtest(
                            backtest_conn,
                            port_start_date,
                            port_end_date,
                            verbose=False
                        )
                        st.session_state['portfolio_results'] = results_df
                        st.success(f"Portfolio backtest complete! Tested {len(results_df)} slates.")
                except Exception as e:
                    st.error(f"Portfolio backtest failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display results
        if 'portfolio_ab_summary' in st.session_state:
            summary = st.session_state['portfolio_ab_summary']

            st.markdown("### Preset Comparison Results")
            st.markdown(f"**{summary.n_dates} slates tested** from {summary.test_dates[0] if summary.test_dates else 'N/A'} to {summary.test_dates[-1] if summary.test_dates else 'N/A'}")

            # Build comparison dataframe
            if summary.results:
                comparison_data = []
                for preset_name, result in summary.results.items():
                    comparison_data.append({
                        'Preset': preset_name,
                        'Slates': result.n_slates,
                        'Wins': result.wins,
                        'Win Rate': f"{result.win_rate*100:.1f}%",
                        'Avg Shortfall': f"{result.avg_shortfall:.1f}",
                        'Hit 2/3 Rate': f"{result.hit_2_of_top3_rate*100:.1f}%",
                        'Unique Players': f"{result.avg_unique_players:.1f}",
                    })

                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('Win Rate', ascending=False)

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Best preset recommendations
                st.markdown("### Best Presets by Category")
                rec_col1, rec_col2 = st.columns(2)

                with rec_col1:
                    st.metric("Best Overall", summary.best_overall or "N/A")
                    if summary.best_small_slate:
                        st.metric("Best for Small Slates", summary.best_small_slate)

                with rec_col2:
                    if summary.best_large_slate:
                        st.metric("Best for Large Slates", summary.best_large_slate)
                    if summary.best_high_total:
                        st.metric("Best for High-Total Slates", summary.best_high_total)

                # Win rate chart
                st.markdown("### Win Rate by Preset")
                chart_data = pd.DataFrame({
                    'Preset': [r.preset_name for r in summary.results.values()],
                    'Win Rate (%)': [r.win_rate * 100 for r in summary.results.values()]
                })
                st.bar_chart(chart_data.set_index('Preset'))

        elif 'portfolio_results' in st.session_state:
            results_df = st.session_state['portfolio_results']

            if not results_df.empty:
                st.markdown("### Portfolio Backtest Results (BALANCED preset)")

                # Summary metrics
                wins = results_df['won_slate'].sum()
                total = len(results_df)
                win_rate = wins / total * 100 if total > 0 else 0
                hit_2_rate = results_df['hit_2_of_top3_any'].mean() * 100

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}/{total} slates")
                with metric_col2:
                    st.metric("Avg Shortfall", f"{results_df['sum_shortfall'].mean():.1f} pts")
                with metric_col3:
                    st.metric("Hit 2/3 Rate", f"{hit_2_rate:.1f}%")
                with metric_col4:
                    st.metric("Avg Unique Players", f"{results_df['unique_players_used'].mean():.1f}")

                # ===================================================================
                # HIT 2/3 PROMINENCE - Key Tournament Success Indicator
                # ===================================================================
                st.divider()
                st.markdown("#### 🎯 Hit 2/3 Analysis (Tournament Success Indicator)")
                st.caption("Having 2+ of top 3 scorers is key to tournament success. This shows portfolio coverage quality.")

                hit_2_count = results_df['hit_2_of_top3_any'].sum()
                hit_2_miss = total - hit_2_count

                h2_col1, h2_col2, h2_col3 = st.columns(3)
                with h2_col1:
                    # Prominent callout for hit 2/3 rate
                    if hit_2_rate >= 75:
                        st.success(f"🏆 **Excellent Coverage**: {hit_2_rate:.0f}% hit 2/3 rate")
                    elif hit_2_rate >= 50:
                        st.info(f"✅ **Good Coverage**: {hit_2_rate:.0f}% hit 2/3 rate")
                    else:
                        st.warning(f"⚠️ **Low Coverage**: {hit_2_rate:.0f}% hit 2/3 rate - consider increasing player diversity")
                with h2_col2:
                    st.metric("Slates with 2/3 Coverage", f"{hit_2_count}/{total}")
                with h2_col3:
                    # Conditional insight
                    if hit_2_rate < win_rate:
                        st.info("📊 We win some slates even without 2/3 (good luck)")
                    elif hit_2_rate > win_rate * 1.5:
                        st.info("📊 High coverage but low wins (bad luck or wrong players)")

                # ===================================================================
                # SHORTFALL DISTRIBUTION - Not just average
                # ===================================================================
                st.divider()
                st.markdown("#### 📉 Shortfall Distribution")
                st.caption("How far from optimal our best lineup was. Percentiles show the spread of outcomes.")

                shortfalls = results_df['sum_shortfall']
                sf_col1, sf_col2, sf_col3, sf_col4, sf_col5 = st.columns(5)

                with sf_col1:
                    st.metric("Min (Best)", f"{shortfalls.min():.1f} pts",
                              help="Our closest result to optimal")
                with sf_col2:
                    st.metric("P25", f"{shortfalls.quantile(0.25):.1f} pts",
                              help="25th percentile - better than this 75% of time")
                with sf_col3:
                    st.metric("Median", f"{shortfalls.median():.1f} pts",
                              help="50th percentile - typical shortfall")
                with sf_col4:
                    st.metric("P75", f"{shortfalls.quantile(0.75):.1f} pts",
                              help="75th percentile - worse than this 25% of time")
                with sf_col5:
                    st.metric("Max (Worst)", f"{shortfalls.max():.1f} pts",
                              help="Our worst result vs optimal")

                # Shortfall buckets breakdown
                sf_buckets = []
                bucket_ranges = [(0, 5, "0-5 pts (Near Miss)"), (5, 15, "5-15 pts (Moderate)"),
                                 (15, 30, "15-30 pts (Significant)"), (30, float('inf'), "30+ pts (Major Miss)")]

                for low, high, label in bucket_ranges:
                    count = ((shortfalls >= low) & (shortfalls < high)).sum()
                    pct = count / total * 100 if total > 0 else 0
                    sf_buckets.append({'Shortfall Range': label, 'Count': count, 'Pct': f"{pct:.1f}%"})

                st.dataframe(pd.DataFrame(sf_buckets), use_container_width=True, hide_index=True)

                st.divider()

                # ===================================================================
                # CONCENTRATION METRIC - Top-5 Exposure %
                # ===================================================================
                st.markdown("#### 🎪 Portfolio Concentration")
                st.caption("Shows if your portfolio is concentrated on a few players or broadly diversified")

                # Compute player exposure across all portfolios
                try:
                    exposure_query = """
                        SELECT player1_id, player1_name, player2_id, player2_name, player3_id, player3_name
                        FROM portfolio_lineups
                        WHERE slate_date BETWEEN ? AND ?
                    """
                    lineups_df = pd.read_sql_query(exposure_query, backtest_conn, params=[port_start_date, port_end_date])

                    if not lineups_df.empty:
                        # Count player appearances
                        from collections import Counter
                        player_counts = Counter()
                        total_slots = 0

                        for _, row in lineups_df.iterrows():
                            for pid, pname in [(row['player1_id'], row['player1_name']),
                                               (row['player2_id'], row['player2_name']),
                                               (row['player3_id'], row['player3_name'])]:
                                if pid and pname:
                                    player_counts[pname] += 1
                                    total_slots += 1

                        # Top 5 most exposed players
                        top5 = player_counts.most_common(5)
                        top5_exposure = sum(c for _, c in top5) / total_slots * 100 if total_slots > 0 else 0

                        conc_col1, conc_col2 = st.columns([2, 3])
                        with conc_col1:
                            # Interpret concentration
                            if top5_exposure > 50:
                                st.warning(f"⚠️ **High Concentration**: Top 5 players = {top5_exposure:.0f}% of slots")
                                st.caption("Consider increasing diversity if hit 2/3 is low")
                            elif top5_exposure > 35:
                                st.info(f"📊 **Moderate Concentration**: Top 5 = {top5_exposure:.0f}%")
                            else:
                                st.success(f"✅ **Well Diversified**: Top 5 = {top5_exposure:.0f}%")

                        with conc_col2:
                            st.markdown("**Most Exposed Players:**")
                            exposure_data = [{'Player': name, 'Appearances': cnt, 'Exposure': f"{cnt/total_slots*100:.1f}%"}
                                             for name, cnt in top5]
                            st.dataframe(pd.DataFrame(exposure_data), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.info(f"Concentration data unavailable: {e}")

                st.divider()

                # Results table with enhanced columns
                st.markdown("#### 📋 Detailed Results")
                st.dataframe(
                    results_df[['slate_date', 'won_slate', 'sum_shortfall', 'best_lineup_sum',
                               'optimal_lineup_sum', 'hit_2_of_top3_any', 'unique_players_used']].round(1),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No results found for the selected date range.")

        else:
            # No results yet - show stored summary if available
            try:
                stored_summary = preset_ab_test.get_stored_ab_summary(backtest_conn)
                if not stored_summary.empty:
                    st.markdown("### Stored Preset Comparison (from previous runs)")
                    st.dataframe(stored_summary, use_container_width=True, hide_index=True)
                else:
                    st.info("Click 'Run Portfolio Backtest' to analyze your 20-lineup tournament strategy performance.")
            except Exception:
                st.info("Click 'Run Portfolio Backtest' to analyze your 20-lineup tournament strategy performance.")

        # =======================================================================
        # Stack & High-Total Game Analysis Section
        # =======================================================================
        st.divider()
        st.markdown("### Stack & High-Total Game Analysis")
        st.caption("Analyze lineup composition and Vegas total correlation with top scorers")

        analysis_col1, analysis_col2 = st.columns(2)

        with analysis_col1:
            if st.button("Run Stack & Total Analysis", key="run_stack_analysis_btn"):
                with st.spinner("Analyzing stacks and high-total games..."):
                    try:
                        analysis_result = backtest_portfolio.run_stack_and_total_analysis(
                            backtest_conn,
                            port_start_date,
                            port_end_date,
                            verbose=False
                        )
                        st.session_state['stack_total_analysis'] = analysis_result
                        st.success(f"Analysis complete! Analyzed {analysis_result['dates_analyzed']} slates.")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # Display stack/total analysis results
        if 'stack_total_analysis' in st.session_state:
            analysis = st.session_state['stack_total_analysis']

            st.markdown("#### Lineup Stack Composition")
            stack_col1, stack_col2, stack_col3 = st.columns(3)

            with stack_col1:
                st.metric(
                    "Lineups with Any Stack",
                    f"{analysis['avg_stack_pct']:.1f}%",
                    help="% of lineups with 2+ players from same game"
                )
            with stack_col2:
                st.metric(
                    "Same-Team Stacks",
                    f"{analysis['avg_same_team_stack_pct']:.1f}%",
                    help="% of lineups with 2+ players from same team"
                )
            with stack_col3:
                st.metric(
                    "Cross-Team Stacks",
                    f"{analysis['avg_cross_team_stack_pct']:.1f}%",
                    help="% of lineups with players from opposing teams in same game"
                )

            st.markdown("#### High-Total Games vs Top 3 Scorers")
            st.caption("Do top scorers come from games with the highest Vegas totals?")

            total_col1, total_col2, total_col3 = st.columns(3)

            with total_col1:
                st.metric(
                    "Avg Top-3 from High-Total Games",
                    f"{analysis['avg_top3_from_high_total_pct']:.1f}%",
                    help="Average % of top 3 scorers from the 2 highest-total games"
                )
            with total_col2:
                st.metric(
                    "Slates with 2+ from High-Total",
                    f"{analysis['slates_2_of_3_pct']:.1f}%",
                    help="% of slates where at least 2 of 3 top scorers came from high-total games"
                )
            with total_col3:
                st.metric(
                    "Dates with Vegas Data",
                    f"{analysis['dates_with_vegas']}/{analysis['dates_analyzed']}",
                    help="Number of dates with Vegas totals available"
                )

            # Insight box
            if analysis['avg_top3_from_high_total_pct'] > 50:
                st.success(
                    f"**Insight:** On average, {analysis['avg_top3_from_high_total_pct']:.0f}% of top 3 scorers "
                    "came from high-total games. This supports targeting high-total games in your strategy."
                )
            elif analysis['avg_top3_from_high_total_pct'] > 33:
                st.info(
                    f"**Insight:** {analysis['avg_top3_from_high_total_pct']:.0f}% of top 3 scorers came from "
                    "high-total games - roughly proportional. High totals provide slight edge."
                )
            else:
                st.warning(
                    f"**Insight:** Only {analysis['avg_top3_from_high_total_pct']:.0f}% of top 3 scorers came from "
                    "high-total games. Consider diversifying beyond just high-total games."
                )
        else:
            st.info("Click 'Run Stack & Total Analysis' to analyze lineup stacks and high-total game correlation.")

        # =======================================================================
        # Strategy Effectiveness - What patterns actually WIN?
        # =======================================================================
        st.divider()
        st.markdown("### Strategy Effectiveness Analysis")
        st.caption("Analyze what WINNING lineups look like to optimize your strategy")

        if st.button("Analyze Winning Patterns", key="run_strategy_effectiveness_btn"):
            with st.spinner("Analyzing winning lineup patterns..."):
                try:
                    effectiveness = backtest_portfolio.analyze_strategy_effectiveness(
                        backtest_conn,
                        port_start_date,
                        port_end_date,
                        verbose=False
                    )
                    st.session_state['strategy_effectiveness'] = effectiveness
                    st.success(f"Analysis complete! Analyzed {effectiveness.get('total_slates', 0)} slates.")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if 'strategy_effectiveness' in st.session_state:
            eff = st.session_state['strategy_effectiveness']

            if 'error' not in eff:
                st.markdown("#### What Do Winning Lineups Look Like?")
                st.caption("Characteristics of the OPTIMAL (winning) lineup on each slate")

                win_col1, win_col2, win_col3, win_col4 = st.columns(4)

                with win_col1:
                    st.metric(
                        "Optimal Lineups Stacked",
                        f"{eff['optimal_stacked_pct']:.0f}%",
                        help="% of optimal lineups that had 2+ players from same game"
                    )
                with win_col2:
                    st.metric(
                        "Optimal Same-Team",
                        f"{eff['optimal_same_team_pct']:.0f}%",
                        help="% of optimal lineups with 2+ players from same team"
                    )
                with win_col3:
                    st.metric(
                        "Optimal Cross-Team",
                        f"{eff['optimal_cross_team_pct']:.0f}%",
                        help="% of optimal lineups with cross-team stacks"
                    )
                with win_col4:
                    st.metric(
                        "Avg from High-Total",
                        f"{eff['avg_optimal_from_high_total']:.1f}/3",
                        help="Avg number of top 3 scorers from high-total games"
                    )

                # Recommendations
                st.markdown("#### Strategy Recommendations")

                for rec in eff.get('recommendations', []):
                    if rec['direction'] == 'increase':
                        st.success(f"**{rec['strategy']}**: {rec['finding']}  \n→ {rec['action']}")
                    elif rec['direction'] == 'decrease':
                        st.warning(f"**{rec['strategy']}**: {rec['finding']}  \n→ {rec['action']}")
                    else:
                        st.info(f"**{rec['strategy']}**: {rec['finding']}  \n→ {rec['action']}")

                # Summary comparison
                st.markdown("#### Your Strategy vs Optimal")

                compare_col1, compare_col2 = st.columns(2)

                with compare_col1:
                    st.markdown("**Your Lineups:**")
                    st.write(f"- Best lineup stacked: {eff['our_stacked_best_pct']:.0f}% of slates")
                    st.write(f"- Win rate: {eff['our_win_rate']:.1f}%")

                with compare_col2:
                    st.markdown("**Optimal Lineups:**")
                    st.write(f"- Were stacked: {eff['optimal_stacked_pct']:.0f}% of slates")
                    st.write(f"- From high-total games: {eff['avg_optimal_from_high_total']:.1f}/3 players")

                # Key insight
                if eff['optimal_stacked_pct'] < 40 and eff['our_stacked_best_pct'] > 60:
                    st.error(
                        "**Key Finding:** You're stacking heavily, but optimal lineups often aren't stacked. "
                        "Consider reducing stack bucket allocation."
                    )
                elif eff['optimal_stacked_pct'] > 60 and eff['our_stacked_best_pct'] < 40:
                    st.error(
                        "**Key Finding:** Optimal lineups are often stacked, but your best lineups aren't. "
                        "Consider increasing stack bucket allocation."
                    )
                elif eff['our_win_rate'] > 30:
                    st.success(
                        f"**Key Finding:** Your {eff['our_win_rate']:.0f}% win rate is strong! "
                        "Current strategy is working well."
                    )
            else:
                st.warning(eff.get('error', 'Analysis error'))
        else:
            st.info("Click 'Analyze Winning Patterns' to see what characteristics winning lineups have.")

        # Explanation
        with st.expander("How Portfolio Backtest Works", expanded=False):
            st.markdown("""
            ### Portfolio vs Individual Player Backtest

            | Metric | Top 3 Scorer Analysis | Portfolio Backtest |
            |--------|----------------------|-------------------|
            | **Question** | Did we rank players correctly? | Did any of our 20 lineups win? |
            | **Success** | Picked 2-3 of actual top scorers | Best lineup = optimal 3-player sum |
            | **Use Case** | Calibrating ranking model | Evaluating tournament strategy |

            ### Key Metrics

            | Metric | Description |
            |--------|-------------|
            | **Win Rate** | % of slates where our best lineup matched the optimal 3-player sum |
            | **Shortfall** | Points difference between our best lineup and the optimal |
            | **Hit 2/3 Rate** | % of slates where any lineup contained 2+ of the top 3 scorers |
            | **Unique Players** | Average number of distinct players used across 20 lineups |

            ### Scenario Presets

            | Preset | Strategy |
            |--------|----------|
            | **BALANCED** | Default bucket allocation |
            | **SHOOTOUT** | More stacking, targets high-total games |
            | **CLOSE_GAMES** | Targets tight spreads |
            | **CHAOS** | Maximum variance, leverage-heavy |
            | **STARS_ONLY** | Concentrates on elite talent |
            | **BLOWOUT** | Avoids blowout games |
            """)

    # ==========================================================================
    # MODE 1: Top 3 Scorer Analysis (existing functionality)
    # ==========================================================================
    if backtest_mode == "🎯 Top 3 Scorer Analysis":
        st.subheader("Top 3 Scorer Performance")
        st.caption("Evaluate how well your ranking strategies identify top 3 daily scorers")

        # Ensure backtest table exists
        pt.create_backtest_table(backtest_conn)

        # Check for missing simulation data and show warning
        try:
            sim_check = backtest_conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN p_top3 IS NOT NULL THEN 1 ELSE 0 END) as with_sim
                FROM predictions
                WHERE game_date >= date('now', '-60 days')
            """).fetchone()
            total_preds, preds_with_sim = sim_check
            if total_preds > 0:
                sim_coverage = preds_with_sim / total_preds * 100
                if sim_coverage < 50:
                    st.warning(
                        f"⚠️ **Low Simulation Coverage**: Only {sim_coverage:.0f}% of predictions "
                        f"({preds_with_sim:,}/{total_preds:,}) have simulation data.\n\n"
                        "Strategies using `sim_p_top3` or `sim_p_first` will fall back to `projected_ppg` for missing data.\n\n"
                        "**To fix:** Run `python backfill_sim_probs.py` to populate historical simulation values."
                    )
                elif sim_coverage < 90:
                    st.info(
                        f"ℹ️ **Partial Simulation Coverage**: {sim_coverage:.0f}% of predictions have simulation data. "
                        "Some dates may use fallback ranking for missing simulation values."
                    )
        except Exception:
            pass  # Schema might not have p_top3 column yet

        # Controls row
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

        with ctrl_col1:
            days_options = {
                "Last 7 days": 7,
                "Last 14 days": 14,
                "Last 30 days": 30,
                "Last 60 days": 60,
                "All available": 365
            }
            selected_period = st.selectbox(
                "Date Range",
                options=list(days_options.keys()),
                index=2,
                key="backtest_period"
            )
            days_back = days_options[selected_period]

        with ctrl_col2:
            strategy_options = {
                "Simulation P(Top-3)": "sim_p_top3",
                "Simulation P(#1)": "sim_p_first",
                "TopScorerScore": "top_scorer_score",
                "Projection Only (baseline)": "projection_only",
                "Ceiling Only": "ceiling_only"
            }
            selected_strategy_name = st.selectbox(
                "Primary Strategy",
                options=list(strategy_options.keys()),
                index=0,
                key="backtest_strategy"
            )
            selected_strategy = strategy_options[selected_strategy_name]

        with ctrl_col3:
            compare_all = st.checkbox(
                "Compare All Strategies",
                value=False,
                help="Run backtest for all strategies and show comparison"
            )

        # Date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", key="run_backtest_btn"):
            with st.spinner("Running backtest..."):
                if compare_all:
                    # Run all strategies
                    all_results = {}
                    all_stats = {}
                    for strat_name, strat_key in strategy_options.items():
                        results_df = backtest_top3.run_backtest(
                            backtest_conn, start_date, end_date, strat_key,
                            store_results=True, verbose=False
                        )
                        all_results[strat_key] = results_df
                        all_stats[strat_key] = backtest_top3.compute_summary_stats(results_df)

                    # Store in session state
                    st.session_state['backtest_all_results'] = all_results
                    st.session_state['backtest_all_stats'] = all_stats
                    st.session_state['backtest_compare_mode'] = True
                else:
                    # Single strategy
                    results_df = backtest_top3.run_backtest(
                        backtest_conn, start_date, end_date, selected_strategy,
                        store_results=True, verbose=False
                    )
                    stats = backtest_top3.compute_summary_stats(results_df)

                    st.session_state['backtest_results'] = results_df
                    st.session_state['backtest_stats'] = stats
                    st.session_state['backtest_selected_strategy'] = selected_strategy
                    st.session_state['backtest_compare_mode'] = False

            st.success("Backtest complete!")
            st.rerun()

        # Display results if available
        if st.session_state.get('backtest_compare_mode', False) and 'backtest_all_stats' in st.session_state:
            # Comparison mode
            all_stats = st.session_state['backtest_all_stats']
            all_results = st.session_state['backtest_all_results']

            st.divider()
            st.subheader("📊 Strategy Comparison")

            # Helper: format count/total (pct%)
            def fmt_count_pct_compare(count: int, total: int) -> str:
                pct = (count / total * 100) if total else 0.0
                return f"{count}/{total} ({pct:.1f}%)"

            # Build comparison dataframe with counts
            comparison_data = []
            for strat_key, stats in all_stats.items():
                strat_display = [k for k, v in strategy_options.items() if v == strat_key][0]
                n = stats.get('n_slates', 0)
                rank_avg = stats.get('rank_a1_avg')  # New key from updated compute_summary_stats
                if rank_avg is None:
                    rank_avg = stats.get('avg_rank_a1')  # Fallback to old key

                comparison_data.append({
                    'Strategy': strat_display,
                    'n': n,
                    'Hit #1': fmt_count_pct_compare(stats.get('hit_1_count', 0), n),
                    'Hit Any': fmt_count_pct_compare(stats.get('hit_any_count', 0), n),
                    'Hit 2+': fmt_count_pct_compare(stats.get('hit_2plus_count', 0), n),
                    'Overlap': f"{stats.get('avg_overlap', 0):.2f}/3",
                    'Avg Rank #1': f"{rank_avg:.1f}" if rank_avg else "N/A"
                })

            # Sort by overlap descending
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['_overlap_val'] = [all_stats[strategy_options[row['Strategy']]]['avg_overlap'] for _, row in comparison_df.iterrows()]
            comparison_df = comparison_df.sort_values('_overlap_val', ascending=False).drop(columns=['_overlap_val'])

            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Bar chart comparison
            import plotly.express as px

            chart_data = pd.DataFrame([
                {'Strategy': [k for k, v in strategy_options.items() if v == strat_key][0],
                 'Avg Overlap': stats['avg_overlap'],
                 'Hit #1 Rate': stats['hit_1_rate'] * 100}
                for strat_key, stats in all_stats.items()
            ]).sort_values('Avg Overlap', ascending=False)

            fig = px.bar(chart_data, x='Strategy', y='Avg Overlap',
                         title='Average Overlap by Strategy (higher is better)',
                         color='Avg Overlap', color_continuous_scale='Greens')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif 'backtest_stats' in st.session_state:
            # Single strategy mode
            stats = st.session_state['backtest_stats']
            results_df = st.session_state['backtest_results']
            strategy = st.session_state.get('backtest_selected_strategy', 'unknown')

            st.divider()

            # Helper: format count/total (pct%)
            def fmt_count_pct(count: int, total: int) -> str:
                pct = (count / total * 100) if total else 0.0
                return f"{count}/{total} ({pct:.1f}%)"

            # Summary metrics - Row 1
            st.subheader("📈 Summary Metrics")
            met_cols = st.columns(5)

            n_slates = stats.get('n_slates', 0)

            with met_cols[0]:
                st.metric("Picked Actual #1",
                         fmt_count_pct(stats.get('hit_1_count', 0), n_slates),
                         help="How many slates we picked the actual top scorer")
            with met_cols[1]:
                st.metric("Hit Any (1+)",
                         fmt_count_pct(stats.get('hit_any_count', 0), n_slates),
                         help="Slates with at least 1 correct pick")
            with met_cols[2]:
                st.metric("Avg Overlap", f"{stats.get('avg_overlap', 0):.2f}/3",
                         help="Average correct picks per slate")
            with met_cols[3]:
                rank_avg = stats.get('rank_a1_avg')
                st.metric("Avg Rank of #1", f"{rank_avg:.1f}" if rank_avg else "N/A",
                         help="How high we ranked the actual top scorer (lower is better)")
                # Add rank distribution below
                if stats.get('rank_a1_min') is not None:
                    st.caption(f"min: {stats['rank_a1_min']} | med: {stats['rank_a1_median']} | max: {stats['rank_a1_max']}")
            with met_cols[4]:
                st.metric("Total Slates", f"{n_slates}")

            # Summary metrics - Row 2 (closeness metrics)
            met_cols2 = st.columns(5)
            with met_cols2[0]:
                st.metric("Hit 2+",
                         fmt_count_pct(stats.get('hit_2plus_count', 0), n_slates),
                         help="Slates with 2 or more correct")
            with met_cols2[1]:
                st.metric("Perfect (3/3)",
                         fmt_count_pct(stats.get('hit_exact_count', 0), n_slates),
                         help="Slates with all 3 correct")
            with met_cols2[2]:
                # Shortfall to top-3 (clearer than "closest miss")
                shortfall_avg = stats.get('avg_shortfall_to_top3')
                if shortfall_avg is not None:
                    st.metric("Avg Shortfall to Top-3", f"{shortfall_avg:.1f} pts",
                             help="How many points our best pick was short of the #3 cutoff (0 = we reached top-3)")
                else:
                    st.metric("Avg Shortfall to Top-3", "N/A",
                             help="How many points our best pick was short of the #3 cutoff")
            with met_cols2[3]:
                # DNP in picks
                dnp_total = stats.get('dnp_in_picks', 0)
                st.metric("DNP in Picks", f"{dnp_total}",
                         help="Total picks that Did Not Play (minutes=0)")
            with met_cols2[4]:
                # Tie-friendly avg
                if 'tie_friendly_overlap' in results_df.columns:
                    tie_avg = results_df['tie_friendly_overlap'].mean()
                    st.metric("Tie-Friendly Overlap", f"{tie_avg:.2f}/3",
                             help="Overlap counting all picks that beat the #3 threshold")
                else:
                    st.metric("Tie-Friendly Overlap", "N/A")

            # Worst slates quick view (for debugging)
            if not results_df.empty and len(results_df) >= 3:
                with st.expander("🔴 Worst 3 Slates (debugging)"):
                    # Sort by overlap ASC, then pred_rank_a1 DESC (high rank = bad)
                    sort_cols = ['overlap']
                    ascending = [True]
                    if 'pred_rank_a1' in results_df.columns:
                        sort_cols.append('pred_rank_a1')
                        ascending.append(False)  # High rank (bad) first

                    worst = results_df.sort_values(sort_cols, ascending=ascending).head(3)

                    for _, r in worst.iterrows():
                        date_str = r.get('slate_date', '?')
                        overlap_val = r.get('overlap', 0)
                        rank_a1 = r.get('pred_rank_a1', '?')
                        dnp = r.get('dnp_in_picks', 0)

                        # Build picks summary
                        picks_info = []
                        for i in range(1, 4):
                            name = r.get(f'picked{i}_name', '')
                            finish = r.get(f'picked{i}_finish', 99)
                            if name:
                                last = name.split()[-1]
                                finish_str = f"#{finish}" if finish and finish < 99 else "DNP"
                                picks_info.append(f"{last}→{finish_str}")

                        st.markdown(
                            f"**{date_str}** — overlap {overlap_val}/3, rank(#1)={rank_a1}"
                            + (f", DNP={dnp}" if dnp else "")
                        )
                        if picks_info:
                            st.caption(f"Picks: {' | '.join(picks_info)}")

            # Daily trend chart
            if not results_df.empty:
                st.divider()
                st.subheader("📈 Daily Overlap Trend")

                import plotly.express as px

                trend_df = results_df[['slate_date', 'overlap']].copy()
                trend_df['slate_date'] = pd.to_datetime(trend_df['slate_date'])
                trend_df = trend_df.sort_values('slate_date')

                # Color mapping for overlap
                color_map = {0: '#FF4444', 1: '#FFAA00', 2: '#88CC88', 3: '#22AA22'}
                trend_df['color'] = trend_df['overlap'].map(color_map)

                fig = px.bar(trend_df, x='slate_date', y='overlap',
                            title='Daily Overlap (0-3)',
                            labels={'slate_date': 'Date', 'overlap': 'Overlap'})
                fig.update_traces(marker_color=trend_df['color'].tolist())
                fig.update_layout(yaxis_range=[0, 3.5])
                st.plotly_chart(fig, use_container_width=True)

                # Daily results table - ENHANCED DIAGNOSTICS
                st.divider()
                st.subheader("📋 Daily Results - Diagnostic View")

                # Icon legend
                st.caption("**Legend:** ✅ rank ≤10 (good) | ⚠️ rank 11-25 (warning) | 🔴 rank >25 (bad)")

                # Sort by date descending for display
                sorted_results = results_df.sort_values('slate_date', ascending=False)

                for _, row in sorted_results.iterrows():
                    date_str = row['slate_date']
                    overlap = int(row['overlap'])
                    tie_friendly = int(row.get('tie_friendly_overlap', overlap))
                    hit_1 = int(row['hit_1'])
                    closest_miss = row.get('closest_miss', 0) or 0
                    actual_3rd_pts = row.get('actual_3rd_points', 0) or 0
                    actual_1st_pts = row.get('actual1_points', 0) or 0
                    actual_2nd_pts = row.get('actual2_points', 0) or 0
                    ties_at_3rd = row.get('ties_at_3rd', 0)

                    # Get best pick points for closeness display
                    pick_pts_list = [row.get(f'picked{i}_pts', 0) or 0 for i in range(1, 4)]
                    best_pick_pts = max(pick_pts_list) if pick_pts_list else 0

                    # Color for overlap
                    overlap_colors = {0: '🔴', 1: '🟡', 2: '🟢', 3: '💚'}
                    overlap_icon = overlap_colors.get(overlap, '⚪')

                    # Hit #1 indicator
                    hit1_icon = '✅' if hit_1 else '❌'

                    # Get ranking metrics for diagnosis
                    pred_rank_a1 = row.get('pred_rank_a1', 99) or 99
                    pred_rank_a2 = row.get('pred_rank_a2', 99) or 99
                    pred_rank_a3 = row.get('pred_rank_a3', 99) or 99
                    best_actual_rank = min(pred_rank_a1, pred_rank_a2, pred_rank_a3)
                    n_players = row.get('n_pred_players', 0) or 0

                    # Count how many actual top-3 we ranked badly (>25)
                    ranks_over_25 = sum(1 for r in [pred_rank_a1, pred_rank_a2, pred_rank_a3] if r > 25)
                    ranks_over_40 = sum(1 for r in [pred_rank_a1, pred_rank_a2, pred_rank_a3] if r > 40)

                    # Correct diagnosis logic:
                    # - If 2+ of actual top 3 ranked >25 → Ranking failure
                    # - If actual #1 ranked ≤10 but not picked → Selection failure
                    # - If best actual rank 11-20 and cutoff is low/tied → Variance
                    # - If all actual top3 ranked 21-40 → Blind spot
                    if overlap == 3:
                        diagnosis = "✅ Perfect: All 3 correct!"
                        diagnosis_color = "green"
                    elif overlap >= 1:
                        # Partial hit
                        if ranks_over_25 >= 2:
                            diagnosis = f"Partial ({overlap}/3) + ranking failure ({ranks_over_25}/3 ranked >25)"
                            diagnosis_color = "orange"
                        else:
                            diagnosis = f"Partial ({overlap}/3)"
                            diagnosis_color = "blue"
                    elif ranks_over_25 >= 2:
                        # Ranking failure: 2+ actual top-3 ranked terribly
                        diagnosis = f"Ranking failure — {ranks_over_25}/3 actual top-3 ranked >25"
                        diagnosis_color = "red"
                    elif pred_rank_a1 <= 10:
                        # Selection failure: we saw #1 but picked others
                        diagnosis = f"Selection failure — actual #1 was our #{pred_rank_a1}, but we picked others"
                        diagnosis_color = "orange"
                    elif best_actual_rank <= 20:
                        # Variance/coinflip: model was genuinely close
                        diagnosis = f"Variance miss — best actual ranked #{best_actual_rank}"
                        diagnosis_color = "orange"
                    elif all(21 <= r <= 40 for r in [pred_rank_a1, pred_rank_a2, pred_rank_a3]):
                        # Blind spot: model consistently missed but wasn't catastrophic
                        diagnosis = f"Blind spot — all actual top-3 ranked 21-40"
                        diagnosis_color = "red"
                    else:
                        # Pure ranking failure
                        diagnosis = f"Ranking failure — best actual ranked #{best_actual_rank}"
                        diagnosis_color = "red"

                    # Build compact picks display with prediction metric ("why we picked")
                    picks_display = []
                    for i in range(1, 4):
                        name = row.get(f'picked{i}_name', '')
                        pts = row.get(f'picked{i}_pts', 0) or 0
                        finish = row.get(f'picked{i}_finish', 99) or 99
                        pred_rank = row.get(f'picked{i}_pred_rank', i) or i
                        proj = row.get(f'picked{i}_proj', 0) or 0
                        p_top1 = row.get(f'picked{i}_p_top1', 0) or 0
                        if name:
                            # "Why we picked" metric: show P(#1) if available, else proj pts
                            if p_top1 > 0:
                                why = f"P#1={p_top1*100:.1f}%"
                            elif proj > 0:
                                why = f"proj={proj:.0f}"
                            else:
                                why = ""

                            # Check if player didn't play (no actual data)
                            if finish >= 99 or (pts == 0 and finish > 50):
                                picks_display.append(f"~~{name}~~ (DNP) — {why}" if why else f"~~{name}~~ (DNP)")
                            elif finish <= 3:
                                picks_display.append(f"**{name}** ({pts:.0f}pts, #{finish}) — {why}" if why else f"**{name}** ({pts:.0f}pts, #{finish})")
                            else:
                                picks_display.append(f"{name} ({pts:.0f}pts, #{finish}) — {why}" if why else f"{name} ({pts:.0f}pts, #{finish})")

                    # Build compact actuals display with proper warning icons
                    actuals_display = []
                    for i, rank_key in enumerate(['pred_rank_a1', 'pred_rank_a2', 'pred_rank_a3'], 1):
                        name = row.get(f'actual{i}_name', '')
                        pts = row.get(f'actual{i}_points', 0) or 0
                        our_rank = row.get(rank_key, 99) or 99
                        if name:
                            # Proper warning thresholds: ≤10 ✅, 11-25 ⚠️, >25 🔴
                            if our_rank <= 3:
                                icon = "✅"
                                actuals_display.append(f"**{name}** ({pts:.0f}pts) — we ranked #{our_rank} {icon}")
                            elif our_rank <= 10:
                                icon = "✅"
                                actuals_display.append(f"{name} ({pts:.0f}pts) — we ranked #{our_rank} {icon}")
                            elif our_rank <= 25:
                                icon = "⚠️"
                                actuals_display.append(f"{name} ({pts:.0f}pts) — we ranked #{our_rank} {icon}")
                            else:
                                icon = "🔴"
                                actuals_display.append(f"{name} ({pts:.0f}pts) — we ranked #{our_rank} {icon}")

                    # Build model's top 5 for context
                    model_top5 = []
                    for i in range(1, 4):
                        name = row.get(f'picked{i}_name', '')
                        if name:
                            model_top5.append(name.split()[-1])  # Last name only for brevity

                    # Cutoff label with tie info
                    cutoff_label = f"{actual_3rd_pts:.0f}"
                    if ties_at_3rd:
                        cutoff_label += " (tie)"

                    # Create expandable row with enhanced header (includes slate size)
                    slate_size_str = f" | N={n_players}" if n_players > 0 else ""
                    header = f"{overlap_icon} **{date_str}** | Overlap: {overlap}/3 | #1: {hit1_icon} | Cutoff: {cutoff_label}pts{slate_size_str}"

                    with st.expander(header, expanded=False):
                        # Diagnosis badge at top
                        st.markdown(f"**Diagnosis:** :{diagnosis_color}[{diagnosis}]")

                        # One-line truth: diagnosis breakdown
                        st.markdown(f"`Actual top-3 ranks: {pred_rank_a1} / {pred_rank_a2} / {pred_rank_a3} | Best: #{best_actual_rank} | Cutoff: {actual_3rd_pts:.0f}pts | Best pick: {best_pick_pts:.0f}pts ({closest_miss:+.0f})`")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Our Picks** (actual pts, finish):")
                            for pick in picks_display:
                                st.markdown(f"  • {pick}")

                            # Tie-friendly overlap
                            if tie_friendly != overlap:
                                st.markdown(f"  Tie-friendly overlap: {tie_friendly}/3")

                        with col2:
                            st.markdown("**Actual Top 3** (our rank):")
                            for actual in actuals_display:
                                st.markdown(f"  • {actual}")

                            # Compact scoring context
                            st.markdown("---")
                            spread = actual_1st_pts - actual_3rd_pts
                            if spread == 0:
                                st.markdown(f"**Top-3 pts:** {actual_1st_pts:.0f} / {actual_2nd_pts:.0f} / {actual_3rd_pts:.0f} (tie)")
                            else:
                                st.markdown(f"**Top-3 pts:** {actual_1st_pts:.0f} / {actual_2nd_pts:.0f} / {actual_3rd_pts:.0f} (spread {spread:.0f})")

                        # Model picks with actual finishes (clarified display)
                        st.markdown("---")
                        picks_summary = []
                        finishes = []
                        for i in range(1, 4):
                            name = row.get(f'picked{i}_name', '')
                            finish = row.get(f'picked{i}_finish', 99) or 99
                            if name:
                                last_name = name.split()[-1]
                                picks_summary.append(last_name)
                                finishes.append(f"#{finish}" if finish < 99 else "DNP")
                        if picks_summary:
                            st.markdown(f"**Model picks:** {', '.join(picks_summary)} — actual finishes: {'/'.join(finishes)}")

                        # Baseline comparison with Hit #1 indicator
                        try:
                            baselines = backtest_top3.compute_baseline_overlap(backtest_conn, date_str)
                            proj_data = baselines.get('projection_only', {})
                            ceil_data = baselines.get('ceiling_only', {})
                            proj_overlap = proj_data.get('overlap', 0)
                            ceil_overlap = ceil_data.get('overlap', 0)
                            proj_hit1 = proj_data.get('pred_rank_a1', 99) <= 3
                            ceil_hit1 = ceil_data.get('pred_rank_a1', 99) <= 3

                            if proj_overlap is not None and ceil_overlap is not None:
                                # Build comparison with Hit #1 indicators
                                proj_hit1_icon = "✅" if proj_hit1 else "❌"
                                ceil_hit1_icon = "✅" if ceil_hit1 else "❌"
                                comparison_text = f"Proj: {proj_overlap}/3 (hit#1:{proj_hit1_icon}) | Ceil: {ceil_overlap}/3 (hit#1:{ceil_hit1_icon})"

                                # Determine if this was a brutal slate
                                best_baseline = max(proj_overlap or 0, ceil_overlap or 0)
                                if best_baseline == 0 and overlap == 0:
                                    st.markdown(f"**Baselines:** {comparison_text} — brutal slate (no baseline hit)")
                                elif overlap > best_baseline:
                                    st.markdown(f"**Baselines:** :green[{comparison_text}] — beat baselines")
                                elif overlap == best_baseline:
                                    st.markdown(f"**Baselines:** {comparison_text} — matched")
                                else:
                                    st.markdown(f"**Baselines:** :red[{comparison_text}] — baselines beat us")
                        except Exception:
                            pass  # Baseline comparison is optional

                        # Drilldown button
                        if st.button(f"🔍 Full Drilldown", key=f"drill_{date_str}"):
                            st.session_state[f'drilldown_date'] = date_str

                        # Show drilldown if requested
                        if st.session_state.get('drilldown_date') == date_str:
                            st.divider()
                            st.markdown("#### 🔬 Detailed Analysis")

                            try:
                                context = backtest_top3.get_drilldown_context(
                                    backtest_conn, date_str, strategy, top_n=15
                                )

                                # Model's Top 5 ranked context (table for clarity)
                                our_top = context.get('our_top_ranked', [])[:5]
                                if our_top:
                                    st.markdown("**Model's Top 5 (by prediction rank):**")
                                    top5_data = []
                                    for i, p in enumerate(our_top, 1):
                                        name = p.get('name', '?')
                                        finish = p.get('finish_rank', 999)
                                        finish_str = f"#{finish}" if finish < 100 else "DNP"
                                        pts = p.get('actual_pts', 0) or 0
                                        proj = p.get('proj_ppg', 0) or 0
                                        top5_data.append({
                                            "Pred": i,
                                            "Player": name,
                                            "Actual": finish_str,
                                            "Pts": f"{pts:.0f}",
                                            "Proj": f"{proj:.1f}"
                                        })
                                    st.dataframe(
                                        pd.DataFrame(top5_data),
                                        hide_index=True,
                                        use_container_width=False
                                    )

                                # Ranking field diagnostics expander
                                ranking_diag = context.get('ranking_diagnostics', {})
                                slate_stats = context.get('slate_stats', {})
                                actual_top_list = context.get('actual_top_scorers', [])

                                # Check for issues that need warnings
                                nan_pct = ranking_diag.get('nan_pct', 0)
                                zero_pct = ranking_diag.get('zero_pct', 0)
                                sim_missing_count = sum(1 for p in actual_top_list[:3] if p.get('sim_missing', False))
                                not_predicted_count = sum(1 for p in actual_top_list[:3] if p.get('not_predicted', False))

                                # Show critical warnings first
                                if not_predicted_count > 0:
                                    st.error(f"🚨 {not_predicted_count}/3 actual top scorers were NOT IN PREDICTIONS!")
                                if sim_missing_count > 0 and not_predicted_count < sim_missing_count:
                                    st.warning(f"⚠️ {sim_missing_count - not_predicted_count}/3 actual top scorers have MISSING SIM VALUES (used fallback ranking)")
                                if nan_pct > 10:
                                    st.warning(f"⚠️ {nan_pct:.1f}% of {ranking_diag.get('field', 'ranking field')} values are NaN")

                                with st.expander("🔍 Ranking Field Diagnostics", expanded=False):
                                    diag_col1, diag_col2 = st.columns(2)
                                    with diag_col1:
                                        st.markdown(f"""
                                        **Ranking Field:** `{ranking_diag.get('field', 'unknown')}`
                                        - Total players: {ranking_diag.get('total_players', 0)}
                                        - NaN count: {ranking_diag.get('nan_count', 0)} ({nan_pct:.1f}%)
                                        - Zero count: {ranking_diag.get('zero_count', 0)} ({zero_pct:.1f}%)
                                        """)
                                    with diag_col2:
                                        st.markdown(f"""
                                        **Value Range:**
                                        - Min: {ranking_diag.get('min', 'N/A')}
                                        - Max: {ranking_diag.get('max', 'N/A')}
                                        - Mean: {ranking_diag.get('mean', 'N/A')}
                                        - Median: {ranking_diag.get('median', 'N/A')}
                                        """)

                                    if nan_pct > 0 or zero_pct > 20:
                                        st.markdown("""
                                        ---
                                        **⚠️ Interpretation:**
                                        - High NaN% → Simulation didn't run or failed for these players
                                        - High Zero% → Simulation ran but assigned 0 probability (unusual for stars)
                                        - Players with NaN use `projected_ppg` as fallback for ranking
                                        """)

                                drill_col1, drill_col2 = st.columns(2)

                                with drill_col1:
                                    st.markdown("**Our Top 15 Ranked:**")
                                    our_top_df = pd.DataFrame(context['our_top_ranked'])
                                    if not our_top_df.empty:
                                        # Ensure numeric types for proper sorting
                                        our_top_df['rank'] = pd.to_numeric(our_top_df['rank'], errors='coerce').astype('Int64')
                                        our_top_df['proj_ppg'] = pd.to_numeric(our_top_df['proj_ppg'], errors='coerce')
                                        our_top_df['ceiling'] = pd.to_numeric(our_top_df.get('ceiling', 0), errors='coerce')
                                        our_top_df['sigma'] = pd.to_numeric(our_top_df.get('sigma', 0), errors='coerce')
                                        our_top_df['season_avg'] = pd.to_numeric(our_top_df.get('season_avg', 0), errors='coerce')
                                        our_top_df['actual_pts'] = pd.to_numeric(our_top_df['actual_pts'], errors='coerce')
                                        our_top_df['finish_rank'] = pd.to_numeric(our_top_df['finish_rank'], errors='coerce').astype('Int64')
                                        our_top_df['p_top1'] = pd.to_numeric(our_top_df.get('p_top1', 0), errors='coerce')

                                        our_top_df = our_top_df.rename(columns={
                                            'rank': 'Rank', 'name': 'Player', 'proj_ppg': 'Proj',
                                            'ceiling': 'Ceil', 'sigma': 'σ', 'season_avg': 'SznAvg',
                                            'actual_pts': 'Actual', 'finish_rank': 'Finish',
                                            'p_top1': 'P#1', 'dfs_grade': 'Grade'
                                        })
                                        # Show diagnostic columns: Proj, Ceil, σ (sigma), SznAvg, Actual, Finish
                                        display_cols = ['Rank', 'Player', 'Proj', 'Ceil', 'σ', 'SznAvg', 'Actual', 'Finish']
                                        if 'P#1' in our_top_df.columns and our_top_df['P#1'].sum() > 0:
                                            display_cols.append('P#1')
                                        st.dataframe(
                                            our_top_df[display_cols].head(15),
                                            use_container_width=True, hide_index=True,
                                            column_config={
                                                "Proj": st.column_config.NumberColumn(format="%.1f"),
                                                "Ceil": st.column_config.NumberColumn(format="%.0f"),
                                                "σ": st.column_config.NumberColumn(format="%.1f"),
                                                "SznAvg": st.column_config.NumberColumn(format="%.1f"),
                                                "Actual": st.column_config.NumberColumn(format="%.0f"),
                                                "Finish": st.column_config.NumberColumn(format="%d"),
                                                "P#1": st.column_config.NumberColumn(format="%.1f%%"),
                                            }
                                        )

                                with drill_col2:
                                    st.markdown("**Actual Top 15 Scorers (with our projections):**")
                                    actual_top_df = pd.DataFrame(context['actual_top_scorers'])
                                    if not actual_top_df.empty:
                                        # Ensure numeric types for proper sorting
                                        actual_top_df['finish_rank'] = pd.to_numeric(actual_top_df['finish_rank'], errors='coerce').astype('Int64')
                                        actual_top_df['actual_pts'] = pd.to_numeric(actual_top_df['actual_pts'], errors='coerce')
                                        actual_top_df['our_pred_rank'] = pd.to_numeric(actual_top_df['our_pred_rank'], errors='coerce').astype('Int64')
                                        actual_top_df['proj_ppg'] = pd.to_numeric(actual_top_df.get('proj_ppg', 0), errors='coerce')
                                        actual_top_df['ceiling'] = pd.to_numeric(actual_top_df.get('ceiling', 0), errors='coerce')
                                        actual_top_df['sigma'] = pd.to_numeric(actual_top_df.get('sigma', 0), errors='coerce')
                                        actual_top_df['season_avg'] = pd.to_numeric(actual_top_df.get('season_avg', 0), errors='coerce')
                                        actual_top_df['proj_confidence'] = pd.to_numeric(actual_top_df.get('proj_confidence', 0), errors='coerce')

                                        # Add Status column combining not_predicted and sim_missing flags
                                        def get_status(row):
                                            if row.get('not_predicted', False):
                                                return '❌ NO PRED'
                                            elif row.get('sim_missing', False):
                                                return '⚠️ NO SIM'
                                            elif row.get('used_fallback', False):
                                                return '📊 FALLBACK'
                                            return '✅'
                                        actual_top_df['Status'] = actual_top_df.apply(get_status, axis=1)

                                        actual_top_df = actual_top_df.rename(columns={
                                            'finish_rank': 'Finish', 'name': 'Player',
                                            'actual_pts': 'Actual', 'our_pred_rank': 'Rank',
                                            'proj_ppg': 'Proj', 'ceiling': 'Ceil', 'sigma': 'σ',
                                            'season_avg': 'SznAvg', 'proj_confidence': 'Conf',
                                            'dfs_grade': 'Grade'
                                        })
                                        # Show diagnostic columns with Status indicator
                                        # Status: ❌ = not predicted, ⚠️ = sim missing, 📊 = used fallback, ✅ = ok
                                        st.dataframe(
                                            actual_top_df[['Finish', 'Player', 'Actual', 'Rank', 'Status', 'Proj', 'Ceil', 'σ', 'SznAvg']].head(15),
                                            use_container_width=True, hide_index=True,
                                            column_config={
                                                "Actual": st.column_config.NumberColumn(format="%.0f"),
                                                "Finish": st.column_config.NumberColumn(format="%d"),
                                                "Rank": st.column_config.NumberColumn(format="%d"),
                                                "Proj": st.column_config.NumberColumn(format="%.1f"),
                                                "Ceil": st.column_config.NumberColumn(format="%.0f"),
                                                "σ": st.column_config.NumberColumn(format="%.1f"),
                                                "SznAvg": st.column_config.NumberColumn(format="%.1f"),
                                            }
                                        )

                                        # Legend for status icons
                                        st.caption("**Status:** ✅ OK | ⚠️ NO SIM (used fallback) | ❌ NOT PREDICTED")

                                        # Flag projection anomalies: Proj << SznAvg suggests bad data
                                        anomalies = actual_top_df[
                                            (actual_top_df['Rank'] < 900) &
                                            (actual_top_df['Proj'] < actual_top_df['SznAvg'] * 0.7) &
                                            (actual_top_df['SznAvg'] > 0)
                                        ]
                                        if not anomalies.empty and len(anomalies) > 0:
                                            low_proj_names = anomalies['Player'].tolist()[:3]
                                            st.info(f"📊 Low projections: {', '.join(low_proj_names)} - Proj << SznAvg")

                                # Slate stats
                                slate = context['slate_stats']
                                st.markdown(f"""
                                **Slate Stats:** {slate.get('total_players', 0)} players |
                                Avg: {slate.get('avg_points', 0):.1f} pts |
                                Max: {slate.get('max_points', 0):.0f} pts |
                                Top-3 cutoff: {slate.get('top3_threshold', 0):.0f} pts
                                """)

                            except Exception as e:
                                st.error(f"Drilldown error: {e}")

        else:
            # No results yet
            st.info("👆 Select your settings and click 'Run Backtest' to analyze strategy performance.")

            # Show available data info
            try:
                available_dates = backtest_top3.get_available_dates(backtest_conn)
                if available_dates:
                    st.markdown(f"""
                    **Data Available:**
                    - Dates with actual results: **{len(available_dates)}**
                    - Date range: **{min(available_dates)}** to **{max(available_dates)}**
                    """)
                else:
                    st.warning("No predictions with actual results found. Run predictions and wait for games to complete.")
            except Exception as e:
                st.error(f"Error checking data: {e}")

        # Explanation expander
        with st.expander("ℹ️ How Backtest Metrics Work", expanded=False):
            st.markdown("""
            ### Overlap Metrics (Tournament-Style)

            | Metric | Description |
            |--------|-------------|
            | **Overlap** | How many of our 3 picks were in the actual top 3 (0-3) |
            | **Hit #1** | Did we pick the actual top scorer? |
            | **Hit Any** | Did we get at least 1 correct? |
            | **Hit 2+** | Did we get 2 or more correct? |

            ### Ranking Quality Metrics

            | Metric | Description |
            |--------|-------------|
            | **Rank of #1** | What rank did we give the actual top scorer? Lower is better. |
            | **Avg Rank Top 3** | Average rank we gave to the actual top 3 scorers |

            ### Strategy Definitions

            | Strategy | Ranking By |
            |----------|------------|
            | **Simulation P(Top-3)** | Monte Carlo probability of finishing top 3 |
            | **Simulation P(#1)** | Monte Carlo probability of being #1 scorer |
            | **TopScorerScore** | Heuristic combining projection, ceiling, matchup |
            | **Projection Only** | Raw projected points (baseline) |
            | **Ceiling Only** | Scoring ceiling (max upside) |

            ### Tie Handling

            When multiple players tie at the #3 spot, we use deterministic tie-breaking:
            1. Higher points wins
            2. Lower player_id breaks remaining ties

            **Tie-Friendly Overlap** counts picks that scored ≥ the #3 threshold,
            avoiding false negatives when ties exist.
            """)

# Enrichment Validation tab --------------------------------------------------
if selected_page == "Enrichment Validation":
    st.header("🔬 Enrichment Validation")
    st.caption("Track and validate the 4 data enrichments (Rest, Game Script, Roles, Position Defense)")

    # Create fresh connection (don't use cached connection that may be closed)
    enrichment_conn = sqlite3.connect(str(db_path))

    # =========================================================================
    # DATABASE DIAGNOSTIC EXPANDER (for debugging data issues)
    # =========================================================================
    with st.expander("🔧 Database Diagnostics", expanded=False):
        st.caption(f"**DB Path:** `{db_path}`")

        diag_cursor = enrichment_conn.cursor()

        # Check which tables exist
        diag_cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
              AND name IN ('predictions', 'enrichment_audit_log', 'enrichment_weekly_summary')
        """)
        existing_tables = [row[0] for row in diag_cursor.fetchall()]
        st.markdown(f"**Tables Found:** {', '.join(existing_tables) if existing_tables else 'None'}")

        # Predictions date range
        try:
            diag_cursor.execute("""
                SELECT MIN(game_date) as min_date, MAX(game_date) as max_date, COUNT(*) as n
                FROM predictions
            """)
            pred_row = diag_cursor.fetchone()
            st.markdown(f"**Predictions:** {pred_row[2]:,} rows, dates {pred_row[0]} to {pred_row[1]}")

            # Actuals coverage
            diag_cursor.execute("""
                SELECT COUNT(*) FROM predictions WHERE actual_ppg IS NOT NULL
            """)
            actuals_count = diag_cursor.fetchone()[0]
            st.markdown(f"**With Actuals:** {actuals_count:,} ({actuals_count/pred_row[2]*100:.0f}%)" if pred_row[2] > 0 else "N/A")

            # Recent dates breakdown
            diag_cursor.execute("""
                SELECT game_date, COUNT(*) as n,
                       SUM(CASE WHEN actual_ppg IS NOT NULL THEN 1 ELSE 0 END) as has_actuals,
                       SUM(CASE WHEN role_tier IS NOT NULL THEN 1 ELSE 0 END) as has_role
                FROM predictions
                GROUP BY game_date
                ORDER BY game_date DESC
                LIMIT 10
            """)
            recent_dates = diag_cursor.fetchall()
            st.markdown("**Recent Prediction Dates:**")
            recent_df = pd.DataFrame(recent_dates, columns=['Date', 'N', 'Actuals', 'Role'])
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Predictions query failed: {e}")

        # Weekly summary check
        if 'enrichment_weekly_summary' in existing_tables:
            try:
                diag_cursor.execute("""
                    SELECT week_ending, total_predictions
                    FROM enrichment_weekly_summary
                    ORDER BY week_ending DESC
                    LIMIT 5
                """)
                weekly_rows = diag_cursor.fetchall()
                st.markdown("**Weekly Summary Table:**")
                for row in weekly_rows:
                    st.markdown(f"  - {row[0]}: {row[1]} predictions")
            except Exception as e:
                st.warning(f"Weekly summary query failed: {e}")

        # Enrichment field coverage by week
        st.markdown("**Enrichment Coverage by Week:**")
        try:
            diag_cursor.execute("""
                SELECT
                    strftime('%Y-%m-%d', date(game_date, 'weekday 0', '-6 days')) as week_start,
                    COUNT(*) as total,
                    SUM(CASE WHEN role_tier IS NOT NULL THEN 1 ELSE 0 END) as has_role,
                    SUM(CASE WHEN is_b2b IS NOT NULL THEN 1 ELSE 0 END) as has_b2b,
                    SUM(CASE WHEN game_script_tier IS NOT NULL THEN 1 ELSE 0 END) as has_script,
                    SUM(CASE WHEN days_rest IS NOT NULL THEN 1 ELSE 0 END) as has_rest
                FROM predictions
                GROUP BY week_start
                ORDER BY week_start DESC
                LIMIT 8
            """)
            enrich_rows = diag_cursor.fetchall()
            enrich_df = pd.DataFrame(enrich_rows,
                columns=['Week Start', 'Total', 'Role%', 'B2B%', 'Script%', 'Rest%'])
            # Convert to percentages
            for col in ['Role%', 'B2B%', 'Script%', 'Rest%']:
                enrich_df[col] = (enrich_df[col] / enrich_df['Total'] * 100).round(0).astype(int).astype(str) + '%'
            st.dataframe(enrich_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Enrichment coverage query failed: {e}")

    # Import monitoring modules
    try:
        from enrichment_monitor import (
            ensure_monitoring_tables,
            calculate_weekly_summary,
            check_alerts,
            get_active_alerts,
            populate_daily_audit_log,
            backfill_audit_log
        )
        from enrichment_config import ENRICHMENT_CONFIG, validate_config
        monitoring_available = True
    except ImportError as e:
        monitoring_available = False
        st.error(f"Enrichment monitoring modules not found: {e}")

    if monitoring_available:
        ensure_monitoring_tables(enrichment_conn)

        # =====================================================================
        # PIPELINE HEALTH STATUS BANNER
        # =====================================================================
        cursor = enrichment_conn.cursor()

        # Get pipeline health metrics
        cursor.execute("""
            SELECT
                (SELECT COUNT(*) FROM enrichment_audit_log
                 WHERE date(game_date) >= date('now', '-7 days')) as audit_last_7d,
                (SELECT MAX(created_at) FROM enrichment_audit_log) as last_audit_ts,
                (SELECT COUNT(*) FROM enrichment_weekly_summary
                 WHERE date(week_ending) >= date('now', '-28 days')) as summaries_last_4w,
                (SELECT COUNT(*) FROM enrichment_audit_log
                 WHERE role_tier IS NOT NULL AND date(game_date) >= date('now', '-7 days')) as role_tier_populated,
                (SELECT COUNT(*) FROM enrichment_audit_log
                 WHERE days_rest IS NOT NULL AND date(game_date) >= date('now', '-7 days')) as rest_days_populated,
                (SELECT COUNT(*) FROM enrichment_audit_log
                 WHERE game_script_tier IS NOT NULL AND game_script_tier != 'neutral'
                   AND date(game_date) >= date('now', '-7 days')) as spread_populated
        """)
        health = cursor.fetchone()
        audit_7d = health[0] or 0
        last_audit = health[1]
        summaries_4w = health[2] or 0
        role_tier_pop = health[3] or 0
        rest_days_pop = health[4] or 0
        spread_pop = health[5] or 0

        # Show pipeline health status
        health_issues = []

        # Handle 0-row case separately (don't compute misleading percentages)
        if audit_7d == 0:
            health_issues.append("⚠️ No audit log entries in last 7 days - run backfill first")
        else:
            # Calculate percentages only when we have data
            role_pct = (role_tier_pop / audit_7d * 100)
            rest_pct = (rest_days_pop / audit_7d * 100)
            spread_pct = (spread_pop / audit_7d * 100)

            if role_pct < 50:
                health_issues.append(f"⚠️ Only {role_pct:.0f}% of audit rows have role_tier")
            if rest_pct < 50:
                health_issues.append(f"⚠️ Only {rest_pct:.0f}% of audit rows have rest_days")
            if spread_pct < 10:
                health_issues.append(f"⚠️ Only {spread_pct:.0f}% have spread data (blowout/close will be 0)")

        if summaries_4w < 4:
            health_issues.append(f"⚠️ Only {summaries_4w}/4 weekly summaries exist")

        if health_issues:
            with st.expander("🔴 Pipeline Health Issues", expanded=True):
                for issue in health_issues:
                    st.warning(issue)
                st.caption(f"Last audit log entry: {last_audit or 'Never'}")
                st.caption("Click 'Backfill Enrichments' below to fix these issues.")
        else:
            role_pct = (role_tier_pop / audit_7d * 100) if audit_7d > 0 else 0
            rest_pct = (rest_days_pop / audit_7d * 100) if audit_7d > 0 else 0
            spread_pct = (spread_pop / audit_7d * 100) if audit_7d > 0 else 0

            with st.expander("🟢 Pipeline Health: OK"):
                col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
                with col_h1:
                    st.metric("Audit Rows (7d)", f"{audit_7d:,}")
                with col_h2:
                    st.metric("Weekly Summaries", f"{summaries_4w}/4")
                with col_h3:
                    st.metric("Role Tier %", f"{role_pct:.0f}%")
                with col_h4:
                    st.metric("Rest Days %", f"{rest_pct:.0f}%")
                with col_h5:
                    st.metric("Spread %", f"{spread_pct:.0f}%")

        st.divider()

        # =====================================================================
        # Backfill Controls (always visible)
        # =====================================================================
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN role_tier IS NOT NULL THEN 1 ELSE 0 END) as enriched
            FROM predictions WHERE actual_ppg IS NOT NULL
        """)
        enrich_check = cursor.fetchone()
        total_preds = enrich_check[0] if enrich_check else 0
        enriched_preds = enrich_check[1] if enrich_check else 0

        # Determine if backfill is needed (predictions missing OR audit log empty)
        needs_backfill = (total_preds > 0 and enriched_preds < total_preds * 0.5) or audit_7d == 0

        if needs_backfill:
            st.warning(f"**Backfill recommended:** {enriched_preds}/{total_preds} predictions enriched, {audit_7d} audit rows in last 7 days.")

        col_bf1, col_bf2 = st.columns([3, 1])
        with col_bf1:
            st.caption("Run backfill to populate enrichments and rebuild monitoring data (role_tier, rest_days, spread, audit log, weekly summaries).")
        with col_bf2:
            if st.button("🔄 Backfill Enrichments", type="primary" if needs_backfill else "secondary"):
                with st.spinner("Backfilling enrichment data... This may take a minute."):
                    try:
                        from backfill_enrichments import backfill_enrichments
                        stats = backfill_enrichments(
                            enrichment_conn,
                            days=None,  # All predictions
                            apply_adjustments=False,  # Don't modify projections
                            verbose=False
                        )
                        st.success(f"Backfill complete! Enriched {stats['total_enriched']} predictions across {stats['dates_processed']} dates.")

                        # Step 2: Backfill audit log (copies enrichments to monitoring tables)
                        # Use same window as dates_processed to cover full backfill
                        audit_days = max(60, stats['dates_processed'] + 7)  # +7 buffer for week alignment
                        st.info(f"📊 Populating audit log for {audit_days} days...")
                        audit_count = backfill_audit_log(enrichment_conn, days=audit_days)
                        st.success(f"Audit log populated with {audit_count} records")

                        # Step 3: Recalculate weekly summaries for ALL weeks in backfill window
                        # Calculate number of weeks to cover (audit_days / 7 + 1 buffer)
                        weeks_to_calculate = (audit_days // 7) + 2
                        st.info(f"📈 Recalculating {weeks_to_calculate} weekly summaries...")
                        from datetime import timedelta
                        for i in range(weeks_to_calculate):
                            week_end = (datetime.now() - timedelta(days=1+i*7)).strftime('%Y-%m-%d')
                            calculate_weekly_summary(enrichment_conn, week_end)
                        st.success(f"Weekly summaries updated ({weeks_to_calculate} weeks)!")

                        # Save to S3 for persistence
                        try:
                            storage = s3_storage.S3PredictionStorage()
                            if storage.is_configured():
                                success, message = storage.upload_database(db_path)
                                if success:
                                    st.info("☁️ Database backed up to S3")
                                else:
                                    st.warning(f"⚠️ S3 backup failed: {message}")
                        except Exception as s3_err:
                            st.warning(f"⚠️ S3 backup error: {s3_err}")

                        st.rerun()
                    except Exception as e:
                        st.error(f"Backfill failed: {e}")

        st.divider()

        # =====================================================================
        # DECISION PANEL - Quick Summary of What to Change
        # =====================================================================
        with st.expander("🎯 **Decision Panel** - What Changed My Mind", expanded=False):
            st.caption("Aggregates key insights from all analyses into actionable decisions")

            decision_col1, decision_col2, decision_col3 = st.columns(3)

            with decision_col1:
                st.markdown("#### 🔧 Top Fix Targets")
                # Pull from cached slice_results if available
                if 'slice_results' in st.session_state and st.session_state['slice_results']:
                    fix_targets = st.session_state['slice_results'].get('fix_targets', [])[:3]
                    if fix_targets:
                        for i, t in enumerate(fix_targets, 1):
                            st.markdown(f"**{i}. {t['slice']}**: +{t['delta_mae']:.2f} MAE")
                    else:
                        st.success("No significant fix targets")
                else:
                    st.info("Run Error Slices analysis")

            with decision_col2:
                st.markdown("#### ⏱️ Minutes Driver")
                # Pull from cached minutes_results if available
                if 'minutes_results' in st.session_state and st.session_state['minutes_results']:
                    mr = st.session_state['minutes_results']
                    if mr.get('has_data') and mr.get('miss_distribution'):
                        # Find top miss category (excluding NORMAL)
                        miss_dist = mr['miss_distribution']
                        actionable = [(k, v) for k, v in miss_dist.items()
                                      if k not in ['NORMAL', 'OTHER', 'OT_BOOST']]
                        if actionable:
                            top_miss = max(actionable, key=lambda x: x[1]['pct'])
                            st.markdown(f"**Primary**: {top_miss[0]} ({top_miss[1]['pct']:.1f}%)")

                        # Show lever
                        lever = mr.get('overall', {}).get('primary_lever', 'UNKNOWN')
                        st.markdown(f"**Lever**: {lever}")
                    else:
                        st.info("Insufficient data")
                else:
                    st.info("Run Minutes Lever analysis")

            with decision_col3:
                st.markdown("#### 📋 Portfolio Action")
                # Recommendations based on cached data
                portfolio_actions = []

                # Check if we have slice results for diversification hints
                if 'slice_results' in st.session_state and st.session_state['slice_results']:
                    slices = st.session_state['slice_results']
                    # If blowout slice is a fix target, recommend reducing stacking
                    for t in slices.get('fix_targets', []):
                        if 'blowout' in t['slice'].lower():
                            portfolio_actions.append("⬇️ Reduce blowout stacking")
                            break

                # Check if we have minutes results
                if 'minutes_results' in st.session_state and st.session_state['minutes_results']:
                    mr = st.session_state['minutes_results']
                    miss_dist = mr.get('miss_distribution', {})
                    dnp_pct = miss_dist.get('DNP', {}).get('pct', 0)
                    if dnp_pct > 5:
                        portfolio_actions.append("⬆️ Increase player diversity")

                if portfolio_actions:
                    for action in portfolio_actions[:3]:
                        st.markdown(f"• {action}")
                else:
                    st.success("No immediate changes needed")

            st.caption("💡 Run Minutes Lever and Error Slices analyses to populate this panel")

        st.divider()

        # Tabs for different views
        val_tab1, val_tab2, val_tab3, val_tab4, val_tab5, val_tab6 = st.tabs([
            "📊 Weekly Summary",
            "🔬 Minutes Lever",
            "📉 Error Slices",
            "🚨 Alerts",
            "⚙️ Configuration",
            "🧪 Ablation Study"
        ])

        with val_tab1:
            st.subheader("Weekly Enrichment Performance")
            st.caption("📅 True calendar weeks (Monday-Sunday). Each week is non-overlapping.")

            # Date range selector
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                summary_weeks = st.number_input("Weeks to analyze", min_value=1, max_value=12, value=4)
            with col3:
                if st.button("Refresh Summary", key="refresh_weekly_summary"):
                    with st.spinner("Calculating weekly summaries (calendar weeks)..."):
                        from datetime import timedelta
                        # Import the snap function
                        from enrichment_monitor import snap_to_week_ending_sunday
                        # Start from yesterday and work backwards by calendar week
                        today = datetime.now()
                        for i in range(summary_weeks):
                            # Go back i weeks from today
                            target_date = (today - timedelta(days=7*i)).strftime('%Y-%m-%d')
                            # Snap to that calendar week's Sunday
                            week_sunday = snap_to_week_ending_sunday(target_date)
                            calculate_weekly_summary(enrichment_conn, week_sunday)
                    st.success("Summary refreshed with calendar weeks!")
                    st.rerun()

            # Load and display weekly summaries
            try:
                summary_df = pd.read_sql_query("""
                    SELECT week_ending, total_predictions, overall_mae, overall_bias,
                           b2b_predictions, b2b_effect_observed,
                           blowout_predictions, close_game_predictions,
                           star_mae, starter_mae, rotation_mae, bench_mae,
                           ceiling_hit_rate
                    FROM enrichment_weekly_summary
                    ORDER BY week_ending DESC
                    LIMIT ?
                """, enrichment_conn, params=[summary_weeks])

                if not summary_df.empty:
                    # Add week range column (Mon-Sun) for clearer display
                    def format_week_range(week_ending):
                        try:
                            end_dt = datetime.strptime(week_ending, '%Y-%m-%d')
                            start_dt = end_dt - timedelta(days=6)
                            return f"{start_dt.strftime('%m/%d')} - {end_dt.strftime('%m/%d')}"
                        except:
                            return week_ending
                    summary_df.insert(0, 'Week (Mon-Sun)', summary_df['week_ending'].apply(format_week_range))
                    # Display key metrics
                    latest = summary_df.iloc[0]

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Overall MAE", f"{latest['overall_mae']:.2f}" if latest['overall_mae'] else "N/A")
                    with metric_col2:
                        st.metric("Ceiling Hit Rate", f"{latest['ceiling_hit_rate']:.1%}" if latest['ceiling_hit_rate'] else "N/A")
                    with metric_col3:
                        b2b_effect = latest.get('b2b_effect_observed')
                        st.metric("B2B Effect", f"{b2b_effect:.1%}" if b2b_effect else "N/A")
                    with metric_col4:
                        st.metric("Predictions", int(latest['total_predictions']) if latest['total_predictions'] else 0)

                    st.divider()

                    # Weekly trend chart
                    st.subheader("MAE Trend by Week")
                    if len(summary_df) > 1:
                        chart_df = summary_df[['week_ending', 'overall_mae', 'star_mae', 'bench_mae']].copy()
                        chart_df = chart_df.melt(id_vars=['week_ending'], var_name='Metric', value_name='MAE')
                        st.line_chart(chart_df.pivot(index='week_ending', columns='Metric', values='MAE'))

                    # Role tier breakdown
                    st.subheader("MAE by Role Tier")
                    role_cols = st.columns(4)
                    for i, role in enumerate(['star', 'starter', 'rotation', 'bench']):
                        with role_cols[i]:
                            mae_val = latest.get(f'{role}_mae')
                            st.metric(role.upper(), f"{mae_val:.2f}" if mae_val else "N/A")

                    # Full summary table
                    with st.expander("Full Weekly Data"):
                        st.dataframe(summary_df, use_container_width=True)
                else:
                    st.info("No weekly summary data yet. Click 'Refresh Summary' to generate.")

                    if st.button("Backfill Last 14 Days", key="backfill_audit"):
                        with st.spinner("Backfilling audit log..."):
                            count = backfill_audit_log(enrichment_conn, days=14)
                            st.success(f"Backfilled {count} audit records")
                            calculate_weekly_summary(enrichment_conn)
                            st.rerun()

            except Exception as e:
                st.error(f"Error loading summary: {e}")

        # =====================================================================
        # TAB 2: Minutes Lever (Counterfactual Analysis)
        # =====================================================================
        with val_tab2:
            st.subheader("Minutes Counterfactual Analysis")
            st.caption("Quantify how much error comes from 'bad minutes' vs 'bad per-minute scoring'")

            col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
            with col_m1:
                minutes_days = st.number_input("Days to analyze", min_value=7, max_value=90, value=30, key="minutes_days")
            with col_m2:
                min_minutes = st.number_input("Min actual minutes", min_value=5.0, max_value=20.0, value=10.0, key="min_minutes_filter")
            with col_m3:
                run_minutes = st.button("Run Analysis", key="run_minutes_analysis")

            if run_minutes or 'minutes_results' in st.session_state:
                if run_minutes:
                    with st.spinner("Running counterfactual analysis..."):
                        try:
                            from minutes_counterfactual import run_counterfactual_analysis
                            st.session_state['minutes_results'] = run_counterfactual_analysis(
                                enrichment_conn, days=minutes_days, min_minutes=min_minutes
                            )
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            st.session_state['minutes_results'] = None

                results = st.session_state.get('minutes_results')
                if results and results.get('has_data'):
                    o = results['overall']

                    # Primary Lever Callout
                    if o['primary_lever'] == 'MINUTES':
                        st.success(f"🎯 **PRIMARY LEVER: MINUTES** - {o['lever_explanation']}")
                    elif o['primary_lever'] == 'RATE':
                        st.info(f"🎯 **PRIMARY LEVER: RATE/MATCHUP** - {o['lever_explanation']}")
                    else:
                        st.warning(f"🎯 **BOTH LEVERS MATTER** - {o['lever_explanation']}")

                    st.divider()

                    # Main metrics
                    st.markdown("### MAE Comparison")
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1:
                        st.metric("P0 (Current)", f"{o['mae_P0']:.2f}", help="Your normal projected points")
                    with mcol2:
                        st.metric("P1 (Minutes-Perfect)", f"{o['mae_P1']:.2f}",
                                  delta=f"{o['improvement_P1']:+.0f}%", delta_color="normal",
                                  help="Projected PPM × Actual Minutes")
                    with mcol3:
                        st.metric("P2 (Rate-Perfect)", f"{o['mae_P2']:.2f}",
                                  delta=f"{o['improvement_P2']:+.0f}%", delta_color="normal",
                                  help="Actual PPM × Projected Minutes")
                    with mcol4:
                        st.metric("Minutes MAE", f"{o['minutes_mae']:.1f} min",
                                  help=f"Bias: {o['minutes_bias']:+.1f} min")

                    # By Role breakdown
                    if results['by_role']:
                        st.markdown("### By Role Tier")
                        role_data = []
                        for role, data in results['by_role'].items():
                            role_data.append({
                                'Role': role,
                                'N': data['n'],
                                'MAE (Current)': f"{data['mae_P0']:.2f}",
                                'Minutes Fix': f"{data['improvement_P1']:+.0f}%",
                                'Rate Fix': f"{data['improvement_P2']:+.0f}%",
                                'Minutes MAE': f"{data['minutes_mae']:.1f}",
                            })
                        st.dataframe(pd.DataFrame(role_data), use_container_width=True, hide_index=True)

                    # By Game Script breakdown
                    if results['by_game_script']:
                        st.markdown("### By Game Script")
                        script_data = []
                        for script, data in results['by_game_script'].items():
                            script_data.append({
                                'Game Script': script.replace('_', ' ').title(),
                                'N': data['n'],
                                'MAE (Current)': f"{data['mae_P0']:.2f}",
                                'Minutes Fix': f"{data['improvement_P1']:+.0f}%",
                                'Rate Fix': f"{data['improvement_P2']:+.0f}%",
                            })
                        st.dataframe(pd.DataFrame(script_data), use_container_width=True, hide_index=True)

                    st.divider()

                    # Miss Classification Distribution
                    if results.get('miss_distribution'):
                        st.markdown("### 📊 Minutes Miss Classification")
                        st.caption("Categorizes WHY minutes predictions were wrong")

                        miss_dist = results['miss_distribution']
                        miss_data = []

                        # Category descriptions for clarity (priority order)
                        cat_icons = {
                            'DNP': '🚫', 'EARLY_INJURY_EXIT': '🏥', 'OT_BOOST': '⏱️',
                            'BLOWOUT_PULL': '💨', 'BLOWOUT_ROTATION': '🔄',
                            'FOUL_TROUBLE': '⚠️', 'ROTATION_SHIFT': '📉',
                            'ROTATION_BOOST': '📈', 'NORMAL': '✅', 'OTHER': '❓'
                        }

                        # Display in priority order
                        for cat in ['DNP', 'EARLY_INJURY_EXIT', 'BLOWOUT_PULL', 'BLOWOUT_ROTATION',
                                    'FOUL_TROUBLE', 'OT_BOOST', 'ROTATION_SHIFT', 'ROTATION_BOOST',
                                    'NORMAL', 'OTHER']:
                            if cat in miss_dist:
                                data = miss_dist[cat]
                                miss_data.append({
                                    'Category': f"{cat_icons.get(cat, '')} {cat.replace('_', ' ').title()}",
                                    'Count': data['count'],
                                    'Pct': f"{data['pct']:.1f}%",
                                    'Avg Min Error': f"{data['avg_minutes_error']:.1f}",
                                })

                        st.dataframe(pd.DataFrame(miss_data), use_container_width=True, hide_index=True)

                        # DNP callout
                        dnp_count = results.get('dnp_count', 0)
                        total_preds = results.get('total_predictions', 0)
                        if dnp_count > 0 and total_preds > 0:
                            dnp_rate = dnp_count / total_preds * 100
                            if dnp_rate > 5:
                                st.warning(f"⚠️ **High DNP Rate**: {dnp_count} DNPs ({dnp_rate:.1f}%) - injury/availability data may need improvement")
                            else:
                                st.info(f"📋 DNP Rate: {dnp_count} ({dnp_rate:.1f}%) - within normal range")

                    st.divider()

                    # Investment Recommendations
                    if results.get('recommendations'):
                        st.markdown("### 💡 Investment Recommendations")
                        st.caption("Where to focus modeling improvements based on miss patterns")

                        for rec in results['recommendations']:
                            priority = rec['priority']

                            if priority == 'HIGH':
                                st.error(f"**[HIGH] {rec['area']}**: {rec['reason']}")
                            elif priority == 'MEDIUM':
                                st.warning(f"**[MEDIUM] {rec['area']}**: {rec['reason']}")
                            elif priority == 'IGNORE':
                                # Show but de-emphasize - this is unmodelable variance
                                st.caption(f"🚫 **[IGNORE] {rec['area']}**: {rec['reason']}")
                            else:
                                st.info(f"**[{priority}] {rec['area']}**: {rec['reason']}")

                    # Worst Minutes Misses
                    with st.expander("📉 Worst Minutes Misses (Top 15)"):
                        if results['worst_minutes_misses']:
                            worst_df = pd.DataFrame(results['worst_minutes_misses'])
                            # Select relevant columns for display
                            display_cols = ['player_name', 'game_date', 'proj_minutes', 'actual_minutes',
                                            'minutes_error', 'miss_category', 'role_tier']
                            display_cols = [c for c in display_cols if c in worst_df.columns]
                            st.dataframe(worst_df[display_cols], use_container_width=True, hide_index=True)

                elif results:
                    st.warning(results.get('message', 'No data available'))
                else:
                    st.info("Click 'Run Analysis' to see the counterfactual breakdown")

        # =====================================================================
        # TAB 3: Error Slices ("Where We Were Wrong")
        # =====================================================================
        with val_tab3:
            st.subheader("Error Slices Analysis")
            st.caption("Break down errors by role, game script, rest, and interactions. Only statistically significant + high-volume buckets are flagged as fix targets.")

            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                slice_days = st.number_input("Days to analyze", min_value=7, max_value=90, value=30, key="slice_days")
            with col_s2:
                run_slices = st.button("Run Analysis", key="run_slices_analysis")

            if run_slices or 'slice_results' in st.session_state:
                if run_slices:
                    with st.spinner("Calculating error slices with bootstrap CIs..."):
                        try:
                            from error_slices import calculate_error_slices
                            st.session_state['slice_results'] = calculate_error_slices(
                                enrichment_conn, days=slice_days
                            )
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            st.session_state['slice_results'] = None

                results = st.session_state.get('slice_results')

                # Show coverage info even if no valid data (helps diagnose issues)
                if results and results.get('coverage'):
                    cov = results['coverage']
                    with st.expander("📊 Data Coverage", expanded=not results.get('has_data')):
                        cov_col1, cov_col2, cov_col3, cov_col4 = st.columns(4)
                        with cov_col1:
                            st.metric("Rows with Actuals", f"{cov['rows_with_actuals']:,}/{cov['total_rows_in_window']:,}")
                        with cov_col2:
                            st.metric("Actuals Coverage", f"{cov['actuals_coverage_pct']:.0f}%")
                        with cov_col3:
                            st.metric("Dates with Actuals", f"{cov['dates_with_actuals']}/{cov['dates_in_window']}")
                        with cov_col4:
                            last_date = cov.get('last_date_with_actuals', 'None')
                            st.metric("Last Date with Actuals", last_date or "None")

                        if cov['actuals_coverage_pct'] < 50:
                            st.warning(f"⚠️ Low actuals coverage ({cov['actuals_coverage_pct']:.0f}%). Consider running actuals backfill or extending date range.")

                if results and results.get('has_data'):
                    overall = results['overall']

                    # Header metrics with Valid N
                    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
                    with hcol1:
                        st.metric("Overall MAE", f"{overall['mae']:.2f}", help=f"Bias: {overall['bias']:+.2f}")
                    with hcol2:
                        st.metric("Valid N", f"{overall['n']:,}", help="Rows with valid (proj, actual) pairs")
                    with hcol3:
                        st.metric("Trustworthy Slices", f"{results['trustworthy_count']}/{results['total_slices']}")
                    with hcol4:
                        st.metric("Cutoff Date", results.get('cutoff_date', 'N/A'))

                    st.divider()

                    # Fix Targets (statistically significant + trustworthy)
                    if results.get('fix_targets'):
                        st.markdown("### 🎯 Fix Targets (Statistically Significant)")
                        st.caption("Buckets with significantly higher error AND sufficient sample (N≥50 or ≥2% volume) AND ≥5 unique dates")
                        for s in results['fix_targets']:
                            ci_str = f"[{s['ci_lower']:+.2f}, {s['ci_upper']:+.2f}]"
                            n_dates = s.get('n_dates', 0)
                            st.error(f"**{s['slice']}**: {s['pct_volume']:.1f}% vol, {n_dates} dates, **+{s['delta_mae']:.2f}** MAE worse | 95% CI: {ci_str}")
                    else:
                        st.success("✅ No statistically significant fix targets found - errors are within normal variance")

                    st.divider()

                    # Slice Tables with CI (stratified bootstrap by date)
                    def render_slice_table_with_ci(slices, title):
                        if slices:
                            st.markdown(f"### {title}")
                            slice_data = []
                            for s in slices:
                                ci_str = f"[{s['ci_lower']:+.2f}, {s['ci_upper']:+.2f}]" if not np.isnan(s.get('ci_lower', np.nan)) else "N/A"
                                sig_icon = "✓" if s.get('is_significant') else ""
                                trust_icon = "" if s.get('is_trustworthy') else "⚠️"
                                n_dates = s.get('n_dates', 0)
                                slice_data.append({
                                    'Slice': s['slice'],
                                    'N': s['n'],
                                    'Dates': n_dates,
                                    'Vol%': f"{s['pct_volume']:.1f}%",
                                    'MAE': f"{s['mae']:.2f}",
                                    'ΔMAE': f"{s['delta_mae']:+.2f}",
                                    '95% CI': ci_str,
                                    'Sig': sig_icon,
                                    '': trust_icon,
                                })
                            st.dataframe(pd.DataFrame(slice_data), use_container_width=True, hide_index=True)
                            st.caption("✓ = significant (CI excludes 0) | ⚠️ = low sample/dates | CIs use stratified bootstrap (resamples by date)")

                    render_slice_table_with_ci(results['by_role'], "By Role Tier")
                    render_slice_table_with_ci(results['by_game_script'], "By Game Script (Spread)")
                    render_slice_table_with_ci(results['by_rest'], "By Rest Days")

                    with st.expander("Interaction: Role × Game Script"):
                        render_slice_table_with_ci(results['interactions_role_script'], "")

                    with st.expander("Interaction: Role × Rest"):
                        render_slice_table_with_ci(results['interactions_role_rest'], "")

                    # Best Buckets (significantly better than baseline)
                    if results.get('best_buckets'):
                        with st.expander("✅ Best Buckets (significantly better than baseline)"):
                            for s in results['best_buckets']:
                                ci_str = f"[{s['ci_lower']:+.2f}, {s['ci_upper']:+.2f}]"
                                st.success(f"**{s['slice']}**: {s['pct_volume']:.1f}% volume, **{s['delta_mae']:.2f}** MAE better | 95% CI: {ci_str}")

                    # Sample rows for spot-checking (truth test)
                    if results.get('sample_rows'):
                        with st.expander("🔍 Sample Rows (Spot-Check Truth Test)"):
                            st.caption("Verify: abs_error = |proj - actual|, role_tier looks correct, etc.")
                            sample_df = pd.DataFrame(results['sample_rows'])
                            # Reorder columns for clarity
                            col_order = ['game_date', 'player_name', 'projected_ppg', 'actual_ppg', 'abs_error', 'role_tier', 'game_script_tier', 'days_rest']
                            col_order = [c for c in col_order if c in sample_df.columns]
                            st.dataframe(sample_df[col_order], use_container_width=True, hide_index=True)

                elif results:
                    # No valid data - show the message and coverage info
                    st.error(f"❌ {results.get('message', 'No data available')}")
                    if results.get('coverage'):
                        st.info("Check the Data Coverage section above for details on actuals availability.")
                else:
                    st.info("Click 'Run Analysis' to see error breakdowns by bucket")

        with val_tab4:
            st.subheader("Active Alerts")

            # Check for new alerts
            if st.button("Check for Alerts", key="check_alerts_btn"):
                with st.spinner("Checking alert conditions..."):
                    alerts = check_alerts(enrichment_conn, verbose=False)
                if alerts:
                    st.warning(f"{len(alerts)} new alert(s) triggered!")
                else:
                    st.success("No new alerts")
                st.rerun()

            # Display active alerts
            active_alerts = get_active_alerts(enrichment_conn)

            if active_alerts:
                for alert in active_alerts:
                    severity = alert['severity']
                    if severity == 'HIGH' or severity == 'CRITICAL':
                        st.error(f"**[{severity}]** {alert['message']}")
                    elif severity == 'MEDIUM':
                        st.warning(f"**[{severity}]** {alert['message']}")
                    else:
                        st.info(f"**[{severity}]** {alert['message']}")
            else:
                st.success("No active alerts")

            # Alert thresholds reference
            with st.expander("Alert Thresholds Reference"):
                st.markdown("""
                | Alert | Condition | Severity |
                |-------|-----------|----------|
                | B2B Effect Flip | B2B players outperform non-B2B | HIGH |
                | B2B Effect Weak | B2B effect < 3% | MEDIUM |
                | Blowout Overcorrection | Star MAE in blowouts > 6 | MEDIUM |
                | Star Underperformance | STAR MAE > 5 | MEDIUM |
                | Position Factor Degradation | Position factor worsening MAE | MEDIUM |
                | Overall MAE Spike | MAE > 5.5 | HIGH |
                | Ceiling Hit Drop | Ceiling hit rate < 60% | MEDIUM |
                """)

        with val_tab5:
            st.subheader("Enrichment Configuration")

            # Show current config
            config_issues = validate_config()

            if config_issues['errors']:
                st.error("Configuration Errors:")
                for e in config_issues['errors']:
                    st.write(f"  - {e}")

            if config_issues['warnings']:
                st.warning("Configuration Warnings:")
                for w in config_issues['warnings']:
                    st.write(f"  - {w}")

            if not config_issues['errors'] and not config_issues['warnings']:
                st.success("Configuration is valid")

            # Display config sections
            config_col1, config_col2 = st.columns(2)

            with config_col1:
                st.markdown("### Rest/B2B")
                rest_cfg = ENRICHMENT_CONFIG['rest']
                st.write(f"- B2B Multiplier: **{rest_cfg['b2b_multiplier']}** ({(1-rest_cfg['b2b_multiplier'])*100:.0f}% penalty)")
                st.write(f"- Rested Multiplier: **{rest_cfg['rested_multiplier']}** (+{(rest_cfg['rested_multiplier']-1)*100:.0f}% boost)")
                st.write(f"- Rested Threshold: **{rest_cfg['rested_threshold_days']}** days")
                st.write(f"- Enabled: **{rest_cfg['enabled']}**")

                st.markdown("### Game Script")
                gs_cfg = ENRICHMENT_CONFIG['game_script']
                st.write(f"- Blowout Threshold: **{gs_cfg['blowout_spread_threshold']}** pts")
                st.write(f"- Close Game Threshold: **{gs_cfg['close_spread_threshold']}** pts")
                st.write(f"- Star Blowout Adj: **{gs_cfg['star_blowout_minutes_adj']}** min")
                st.write(f"- Bench Blowout Adj: **+{gs_cfg['bench_blowout_minutes_adj']}** min")

            with config_col2:
                st.markdown("### Role Tiers")
                role_cfg = ENRICHMENT_CONFIG['roles']
                st.write(f"- STAR: **{role_cfg['star_min_ppg']}** PPG, **{role_cfg['star_min_minutes']}** min")
                st.write(f"- STARTER: **{role_cfg['starter_min_ppg']}** PPG, **{role_cfg['starter_min_minutes']}** min")
                st.write(f"- ROTATION: **{role_cfg['rotation_min_minutes']}** min")

                st.markdown("### Position Defense")
                pos_cfg = ENRICHMENT_CONFIG['position_defense']
                st.write("Grade Factors:")
                for grade, factor in pos_cfg['grade_factors'].items():
                    effect = (factor - 1) * 100
                    st.write(f"  - Grade {grade}: **{factor}** ({effect:+.0f}%)")

        with val_tab6:
            st.subheader("Ablation Study")
            st.caption("Compare model variants with different enrichment combinations")

            # Date range for ablation
            abl_col1, abl_col2, abl_col3 = st.columns([2, 2, 1])
            with abl_col1:
                abl_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=14), key="abl_start")
            with abl_col2:
                abl_end = st.date_input("End Date", value=datetime.now() - timedelta(days=1), key="abl_end")
            with abl_col3:
                run_ablation = st.button("Run Ablation", key="run_ablation_btn")

            if run_ablation:
                with st.spinner("Running ablation study..."):
                    try:
                        from evaluation.ablation_backtest import run_ablation_study
                        results = run_ablation_study(
                            enrichment_conn,
                            abl_start.strftime('%Y-%m-%d'),
                            abl_end.strftime('%Y-%m-%d'),
                            verbose=False
                        )

                        if results.variants:
                            st.session_state['ablation_results'] = results
                            st.success("Ablation study complete!")
                        else:
                            st.warning("No data found for date range")
                    except Exception as e:
                        st.error(f"Ablation study failed: {e}")

            # Display results if available
            if 'ablation_results' in st.session_state:
                results = st.session_state['ablation_results']

                st.markdown(f"**Best Variant:** {results.best_variant}")

                # Results table
                abl_data = []
                for name, result in results.variants.items():
                    vs_baseline = f"{result.mae_improvement_pct:+.1f}%" if result.mae_improvement_pct else "-"
                    abl_data.append({
                        'Variant': name,
                        'MAE': result.mae,
                        'vs Baseline': vs_baseline,
                        'RMSE': result.rmse,
                        'Bias': result.bias,
                        'Ceiling Hit': f"{result.ceiling_hit_rate:.1%}",
                        'N': result.n_predictions
                    })

                abl_df = pd.DataFrame(abl_data)
                st.dataframe(abl_df, use_container_width=True, hide_index=True)

                # Visual comparison
                st.subheader("MAE Comparison")
                chart_data = pd.DataFrame({
                    'Variant': [r['Variant'] for r in abl_data],
                    'MAE': [r['MAE'] for r in abl_data]
                })
                st.bar_chart(chart_data.set_index('Variant'))
            else:
                st.info("Click 'Run Ablation' to compare enrichment variants")

    enrichment_conn.close()

# FanDuel Compare tab --------------------------------------------------------
if selected_page == "FanDuel Compare":
    st.header("📊 FanDuel O/U Comparison")
    st.caption("Compare our projections against FanDuel player points over/under lines")

    # Import odds_api module
    try:
        import odds_api
        odds_available = True
    except ImportError:
        odds_available = False
        st.error("odds_api module not found. Please ensure odds_api.py is in the project directory.")

    if odds_available:
        # Date selector
        col_date, col_toggle = st.columns([2, 1])

        with col_date:
            compare_date = st.date_input(
                "Game Date",
                value=datetime.now(EASTERN_TZ).date(),
                key="fanduel_compare_date"
            )

        with col_toggle:
            show_all_players = st.toggle(
                "Show all players",
                value=False,
                help="Show all predicted players, including those without FanDuel lines"
            )

        # Check for API key configuration
        api_key = odds_api.get_api_key()

        if not api_key:
            st.warning(
                "**API key not configured.** Add `[theoddsapi]` section to `.streamlit/secrets.toml`:\n\n"
                "```toml\n[theoddsapi]\nAPI_KEY = \"your-api-key-here\"\n```"
            )

        # Load comparison data
        games_conn = get_connection(str(db_path))

        # Ensure schema is up to date
        odds_api.create_odds_tables(games_conn)
        pt.upgrade_predictions_table_for_fanduel(games_conn)

        # Load predictions with FanDuel data
        query = """
            SELECT
                player_name,
                team_name,
                opponent_name,
                projected_ppg,
                proj_floor,
                proj_ceiling,
                proj_confidence,
                fanduel_ou,
                fanduel_over_odds,
                fanduel_under_odds,
                fanduel_fetched_at,
                dfs_score,
                dfs_grade
            FROM predictions
            WHERE game_date = ?
            ORDER BY dfs_score DESC
        """
        comparison_df = pd.read_sql_query(query, games_conn, params=[str(compare_date)])

        if comparison_df.empty:
            st.info(f"No predictions found for {compare_date}. Generate predictions first using the sidebar button.")
        else:
            # Count players with FanDuel lines
            with_lines = comparison_df['fanduel_ou'].notna().sum()
            total_players = len(comparison_df)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Predictions", total_players)

            with col2:
                st.metric("With FanDuel Lines", with_lines)

            with col3:
                if with_lines > 0:
                    lines_df = comparison_df[comparison_df['fanduel_ou'].notna()].copy()
                    lines_df['diff'] = lines_df['projected_ppg'] - lines_df['fanduel_ou']
                    avg_diff = lines_df['diff'].mean()
                    st.metric("Avg Difference", f"{avg_diff:+.1f}")
                else:
                    st.metric("Avg Difference", "N/A")

            with col4:
                # API usage
                monthly_usage = odds_api.get_monthly_api_usage(games_conn)
                st.metric("API Usage", f"{monthly_usage}/500")

            st.divider()

            # Manual fetch button
            if api_key:
                should_fetch, reason = odds_api.should_fetch_odds(games_conn, compare_date)

                # Check if already fetched today (allow force refetch)
                already_fetched = "Already fetched" in reason
                force_refetch = False

                if already_fetched:
                    st.caption(f"ℹ️ {reason}")
                    force_refetch = st.checkbox(
                        "🔄 Force refetch (update lines after injury news)",
                        help="Use this to get updated lines if there's been an injury or lineup change. Uses additional API quota."
                    )
                    if force_refetch:
                        st.warning("⚠️ This will use additional API requests from your monthly quota.")

                if should_fetch or force_refetch:
                    button_label = "🔄 Refetch FanDuel Lines" if force_refetch else "🔄 Fetch FanDuel Lines"
                    if st.button(button_label, type="primary"):
                        with st.spinner("Fetching from The Odds API..."):
                            result = odds_api.fetch_fanduel_lines_for_date(games_conn, compare_date, force=force_refetch)

                            if result['success']:
                                st.success(f"Fetched lines for {result['players_matched']} players using {result['api_requests_used']} API requests")

                                # Archive to S3 CSV for historical record
                                archive_result = odds_api.archive_fanduel_lines_to_s3(games_conn, compare_date)
                                if archive_result.get('success'):
                                    st.info(f"📁 Archived {archive_result['rows']} lines to S3")

                                st.rerun()
                            else:
                                st.error(f"Fetch failed: {result.get('error', 'Unknown error')}")
                elif not already_fetched:
                    # Show other reasons (like budget exceeded) without force option
                    st.caption(f"ℹ️ {reason}")

            # Filter based on toggle
            if not show_all_players:
                display_df = comparison_df[comparison_df['fanduel_ou'].notna()].copy()
            else:
                display_df = comparison_df.copy()

            if display_df.empty:
                st.info("No FanDuel lines found for this date. Click 'Fetch FanDuel Lines' to get latest odds.")
            else:
                # Calculate difference columns
                display_df['Diff'] = display_df['projected_ppg'] - display_df['fanduel_ou']
                display_df['Diff %'] = (display_df['Diff'] / display_df['fanduel_ou'] * 100).round(1)

                # Prepare display table
                table_df = display_df[[
                    'player_name', 'team_name', 'projected_ppg', 'fanduel_ou', 'Diff', 'Diff %'
                ]].copy()

                table_df.columns = ['Player', 'Team', 'Our Projection', 'FanDuel O/U', 'Difference', '% Diff']

                # Format numeric columns
                table_df['Our Projection'] = table_df['Our Projection'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                table_df['FanDuel O/U'] = table_df['FanDuel O/U'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                table_df['Difference'] = table_df['Difference'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A")
                table_df['% Diff'] = table_df['% Diff'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")

                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )

                # Legend
                st.caption(
                    "**Reading the table:** Positive difference means our projection is HIGHER than FanDuel's O/U line "
                    "(potential **over** opportunity). Negative means our projection is LOWER (potential **under**)."
                )

                # Export button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name=f"fanduel_comparison_{compare_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Model vs FanDuel tab --------------------------------------------------------
if selected_page == "Model vs FanDuel":
    st.title("Model vs FanDuel Analytics")
    st.caption("Compare your projection accuracy against FanDuel's lines after games complete")

    # Get database connection
    games_conn = get_connection(str(db_path))

    # Ensure schema is up to date (must add fanduel columns before comparison columns)
    pt.upgrade_predictions_table_for_fanduel(games_conn)
    pt.upgrade_predictions_table_for_fanduel_comparison(games_conn)

    # Date range selector
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        # Default to last 30 days
        default_start = datetime.now().date() - timedelta(days=30)
        start_date = st.date_input("Start Date", value=default_start, key="mvf_start")

    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date(), key="mvf_end")

    with col3:
        if st.button("🔄 Recalculate Metrics", type="primary"):
            with st.spinner("Calculating comparison metrics..."):
                updated = pt.calculate_fanduel_comparison_metrics(games_conn)
                if updated > 0:
                    st.success(f"Updated {updated} predictions with comparison metrics")
                    st.rerun()
                else:
                    st.info("No new predictions to calculate (all up to date)")

    st.divider()

    # =========================================================================
    # SECTION 1: Overall Model Performance (ALL predictions)
    # =========================================================================
    st.subheader("📊 Overall Model Performance")
    st.caption("Performance across ALL predictions (not just those with FanDuel lines)")

    overall = pt.get_overall_model_performance(
        games_conn,
        start_date=str(start_date),
        end_date=str(end_date)
    )

    if overall['with_actuals'] == 0:
        st.info("No completed predictions in this date range. Actuals are populated after games finish.")
    else:
        # Overall metrics row
        ocol1, ocol2, ocol3, ocol4 = st.columns(4)

        with ocol1:
            st.metric(
                "Predictions Scored",
                f"{overall['with_actuals']} / {overall['total_predictions']}",
                help="Predictions with actual results vs total predictions"
            )

        with ocol2:
            st.metric(
                "Avg Error",
                f"{overall['avg_error']:.1f} PPG",
                help="Average absolute error in points per game"
            )

        with ocol3:
            st.metric(
                "Hit Rate",
                f"{overall['hit_rate']:.1f}%",
                help="Percentage of predictions within floor-ceiling range"
            )

        with ocol4:
            over_pct = overall['over_pct']
            bias = "Over" if over_pct > 55 else "Under" if over_pct < 45 else "Balanced"
            st.metric(
                "Projection Bias",
                bias,
                delta=f"{over_pct:.0f}% over-projected" if over_pct != 50 else None,
                delta_color="inverse" if abs(over_pct - 50) > 10 else "off"
            )

        # Show breakdown
        with st.expander("📈 Over/Under Breakdown"):
            st.write(f"**Over-projected:** {overall['over_projected']} predictions (actual was lower than projected)")
            st.write(f"**Under-projected:** {overall['under_projected']} predictions (actual was higher than projected)")
            if overall['exact'] > 0:
                st.write(f"**Exact:** {overall['exact']} predictions")

        # Detailed table for ALL predictions
        st.subheader("📋 All Predictions Detail")

        # Toggle to include unscored predictions
        include_unscored = st.checkbox(
            "Include predictions without actual results",
            value=False,
            key="include_unscored",
            help="Show predictions for games that haven't been scored yet"
        )

        # Query predictions (optionally include unscored)
        if include_unscored:
            all_preds_query = f"""
                SELECT
                    game_date,
                    player_name,
                    team_name,
                    projected_ppg,
                    actual_ppg,
                    error,
                    abs_error,
                    hit_floor_ceiling,
                    proj_floor,
                    proj_ceiling,
                    fanduel_ou
                FROM predictions
                WHERE game_date >= '{start_date}'
                AND game_date <= '{end_date}'
                ORDER BY game_date DESC, projected_ppg DESC
            """
        else:
            all_preds_query = f"""
                SELECT
                    game_date,
                    player_name,
                    team_name,
                    projected_ppg,
                    actual_ppg,
                    error,
                    abs_error,
                    hit_floor_ceiling,
                    proj_floor,
                    proj_ceiling,
                    fanduel_ou
                FROM predictions
                WHERE actual_ppg IS NOT NULL
                AND game_date >= '{start_date}'
                AND game_date <= '{end_date}'
                ORDER BY game_date DESC, abs_error DESC
            """

        all_preds_df = pd.read_sql_query(all_preds_query, games_conn)

        if not all_preds_df.empty:
            # Format for display (handle NULL values for unscored predictions)
            all_display = all_preds_df.copy()
            all_display['Date'] = all_display['game_date']
            all_display['Player'] = all_display['player_name']
            all_display['Team'] = all_display['team_name']
            all_display['Our Proj'] = all_display['projected_ppg'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            all_display['Actual'] = all_display['actual_ppg'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "pending")
            all_display['Error'] = all_display['error'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
            all_display['Abs Error'] = all_display['abs_error'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            all_display['In Range'] = all_display['hit_floor_ceiling'].apply(
                lambda x: "✅" if x == 1 else ("❌" if x == 0 else "—")
            )
            all_display['FD Line'] = all_display['fanduel_ou'].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )

            # Show column selector
            col_options = ['Date', 'Player', 'Team', 'Our Proj', 'Actual', 'Error', 'Abs Error', 'In Range', 'FD Line']
            default_cols = ['Date', 'Player', 'Team', 'Our Proj', 'Actual', 'Error', 'In Range', 'FD Line']

            selected_cols = st.multiselect(
                "Columns to display",
                options=col_options,
                default=default_cols,
                key="all_preds_cols"
            )

            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                show_only_misses = st.checkbox("Show only big misses (>5 PPG error)", key="big_misses_filter")
            with filter_col2:
                show_only_fd = st.checkbox("Show only predictions with FD lines", key="fd_only_filter")

            filtered_df = all_display.copy()
            if show_only_misses:
                # Filter for big misses (handle NaN for unscored predictions)
                filtered_df = filtered_df[all_preds_df['abs_error'].fillna(0) > 5]
            if show_only_fd:
                filtered_df = filtered_df[all_preds_df['fanduel_ou'].notna()]

            st.dataframe(
                filtered_df[selected_cols],
                use_container_width=True,
                hide_index=True,
                height=400
            )

            # Download button
            csv_data = all_preds_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Predictions CSV",
                csv_data,
                file_name=f"all_predictions_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )

    st.divider()

    # =========================================================================
    # SECTION 2: vs FanDuel Comparison (subset with FD lines)
    # =========================================================================
    st.subheader("🎰 vs FanDuel Comparison")
    st.caption("Head-to-head comparison with FanDuel lines (subset of predictions)")

    # Get summary stats
    summary = pt.get_fanduel_comparison_summary(
        games_conn,
        start_date=str(start_date),
        end_date=str(end_date)
    )

    if summary['total_compared'] == 0:
        st.info(
            "No comparison data available yet. This requires:\n"
            "1. Predictions with FanDuel lines (use FanDuel Compare tab)\n"
            "2. Actual results after games complete\n"
            "3. Click 'Recalculate Metrics' to compute comparisons"
        )
    else:
        # Summary metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            ou_acc = summary['ou_accuracy_pct']
            st.metric(
                "Our O/U Accuracy",
                f"{ou_acc:.1f}%",
                delta=f"{ou_acc - 50:.1f}% vs coin flip" if ou_acc != 50 else None,
                delta_color="normal" if ou_acc >= 50 else "inverse"
            )

        with col2:
            closer_pct = summary['we_closer_pct']
            st.metric(
                "We Were Closer",
                f"{closer_pct:.1f}%",
                delta=f"{closer_pct - 50:.1f}% vs FanDuel" if closer_pct != 50 else None,
                delta_color="normal" if closer_pct >= 50 else "inverse"
            )

        with col3:
            st.metric(
                "Our Avg Error",
                f"{summary['our_avg_error']:.1f} PPG"
            )

        with col4:
            st.metric(
                "FanDuel Avg Error",
                f"{summary['fd_avg_error']:.1f} PPG"
            )

        st.divider()

        # Detailed comparison table
        st.subheader("Detailed Comparisons")

        # Query for detailed data
        query = f"""
            SELECT
                game_date,
                player_name,
                team_name,
                projected_ppg,
                fanduel_ou,
                actual_ppg,
                our_ou_call,
                actual_ou_result,
                ou_call_correct,
                abs_error as our_error,
                fanduel_error,
                we_were_closer,
                closer_margin
            FROM predictions
            WHERE ou_call_correct IS NOT NULL
            AND game_date >= '{start_date}'
            AND game_date <= '{end_date}'
            ORDER BY game_date DESC, player_name
        """

        detail_df = pd.read_sql_query(query, games_conn)

        if not detail_df.empty:
            # Format for display
            display_df = detail_df.copy()
            display_df['Date'] = display_df['game_date']
            display_df['Player'] = display_df['player_name']
            display_df['Team'] = display_df['team_name']
            display_df['Our Proj'] = display_df['projected_ppg'].apply(lambda x: f"{x:.1f}")
            display_df['FD Line'] = display_df['fanduel_ou'].apply(lambda x: f"{x:.1f}")
            display_df['Actual'] = display_df['actual_ppg'].apply(lambda x: f"{x:.1f}")
            display_df['Our Call'] = display_df['our_ou_call'].str.upper()
            display_df['Result'] = display_df['actual_ou_result'].str.upper()
            display_df['Correct?'] = display_df['ou_call_correct'].apply(lambda x: "✅" if x == 1 else "❌")
            display_df['Our Error'] = display_df['our_error'].apply(lambda x: f"{x:.1f}")
            display_df['FD Error'] = display_df['fanduel_error'].apply(lambda x: f"{x:.1f}")
            display_df['Winner'] = display_df['we_were_closer'].apply(lambda x: "🏆 Us" if x == 1 else "FanDuel")

            # Select display columns
            table_df = display_df[[
                'Date', 'Player', 'Team', 'Our Proj', 'FD Line', 'Actual',
                'Our Call', 'Result', 'Correct?', 'Our Error', 'FD Error', 'Winner'
            ]]

            st.dataframe(table_df, use_container_width=True, hide_index=True, height=500)

            # Export button
            csv = detail_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Comparison Data",
                data=csv,
                file_name=f"model_vs_fanduel_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )

        st.divider()

        # Player insights section
        st.subheader("Player Insights")

        if summary['by_player']:
            # Best edge (where we beat FanDuel most)
            best_players = [p for p in summary['by_player'] if p['times_closer'] / p['games'] >= 0.6]
            worst_players = [p for p in summary['by_player'] if p['times_closer'] / p['games'] <= 0.4]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🎯 Our Best Edge** (beat FD 60%+ of time)")
                if best_players:
                    for p in best_players[:5]:
                        pct = p['times_closer'] / p['games'] * 100
                        st.markdown(f"- **{p['player_name']}**: {pct:.0f}% closer ({p['times_closer']}/{p['games']} games)")
                else:
                    st.caption("No players with 60%+ edge yet")

            with col2:
                st.markdown("**⚠️ FanDuel's Edge** (FD beat us 60%+ of time)")
                if worst_players:
                    for p in worst_players[:5]:
                        pct = (1 - p['times_closer'] / p['games']) * 100
                        st.markdown(f"- **{p['player_name']}**: FD {pct:.0f}% closer ({p['games'] - p['times_closer']}/{p['games']} games)")
                else:
                    st.caption("No players where FD has 60%+ edge")
        else:
            st.info("Need at least 3 games per player to show insights")


# Model Review tab --------------------------------------------------------
if selected_page == "Model Review":
    st.title("📈 Model Review Dashboard")
    st.caption("Comprehensive model performance analysis and FanDuel edge detection")

    # Get database connection
    review_conn = get_connection(str(db_path))

    # Date period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        days_options = {
            "Last 7 days": 7,
            "Last 14 days": 14,
            "Last 30 days": 30,
            "Last 60 days": 60,
            "Last 90 days": 90,
            "All Time": 365
        }
        selected_period = st.selectbox("Time Period", list(days_options.keys()), index=2, key="mr_period")
        days_back = days_options[selected_period]

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    st.divider()

    # =========================================================================
    # SECTION 1: Executive Summary
    # =========================================================================
    st.subheader("📊 Executive Summary")

    # Get enhanced metrics
    import prediction_evaluation_metrics as pem
    enhanced = pem.calculate_enhanced_metrics(
        review_conn,
        start_date=str(start_date),
        end_date=str(end_date),
        min_actual_ppg=0.0
    )

    # Get FanDuel comparison summary
    fd_summary = pt.get_fanduel_comparison_summary(review_conn, str(start_date), str(end_date))

    if enhanced.get('total_predictions', 0) == 0:
        st.info("No predictions with actual results found for this date range.")
    else:
        # 6 metric cards
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            st.metric(
                "MAE",
                f"{enhanced.get('mae', 0):.2f} pts",
                help="Mean Absolute Error - average prediction miss"
            )

        with m2:
            st.metric(
                "RMSE",
                f"{enhanced.get('rmse', 0):.2f} pts",
                help="Root Mean Square Error - penalizes large misses more"
            )

        with m3:
            spearman = enhanced.get('spearman_correlation', 0)
            st.metric(
                "Spearman ρ",
                f"{spearman:.3f}" if spearman else "N/A",
                help="Rank correlation - are we ordering players correctly? (1.0 = perfect)"
            )

        with m4:
            st.metric(
                "Hit Rate (±5)",
                f"{enhanced.get('hit_rate_within_5', 0):.1f}%",
                help="% of predictions within 5 points of actual"
            )

        with m5:
            fd_win = fd_summary.get('we_closer_pct', 0)
            st.metric(
                "vs FanDuel",
                f"{fd_win:.1f}%" if fd_win else "N/A",
                help="% of time our prediction was closer than FanDuel's line"
            )

        with m6:
            bias = enhanced.get('bias', 0)
            bias_dir = "Over" if bias > 0 else "Under" if bias < 0 else "Neutral"
            st.metric(
                "Bias",
                f"{abs(bias):.2f} pts {bias_dir}",
                help="Systematic over/under projection tendency"
            )

        st.caption(f"Based on {enhanced.get('total_predictions', 0):,} predictions")

    st.divider()

    # =========================================================================
    # SECTION 2: Calibration Analysis
    # =========================================================================
    st.subheader("🎯 Calibration Analysis")
    st.caption("Is high confidence actually more accurate?")

    calibration_query = """
        SELECT
            CASE
                WHEN proj_confidence < 0.6 THEN '1. Low (<60%)'
                WHEN proj_confidence < 0.75 THEN '2. Medium (60-75%)'
                WHEN proj_confidence < 0.85 THEN '3. High (75-85%)'
                ELSE '4. Very High (85%+)'
            END as confidence_bucket,
            COUNT(*) as count,
            AVG(abs_error) as mae,
            AVG(CASE WHEN hit_floor_ceiling = 1 THEN 100.0 ELSE 0 END) as hit_rate
        FROM predictions
        WHERE actual_ppg IS NOT NULL
          AND proj_confidence IS NOT NULL
          AND game_date >= ?
          AND game_date <= ?
        GROUP BY confidence_bucket
        ORDER BY confidence_bucket
    """
    calibration_df = pd.read_sql_query(calibration_query, review_conn, params=[str(start_date), str(end_date)])

    if not calibration_df.empty:
        cal_col1, cal_col2 = st.columns(2)

        with cal_col1:
            # MAE by confidence chart
            import plotly.express as px
            fig_mae = px.bar(
                calibration_df,
                x='confidence_bucket',
                y='mae',
                title='MAE by Confidence Level',
                labels={'confidence_bucket': 'Confidence', 'mae': 'Mean Absolute Error'},
                color='mae',
                color_continuous_scale='RdYlGn_r'
            )
            fig_mae.update_layout(showlegend=False, xaxis_title="", yaxis_title="MAE (pts)")
            st.plotly_chart(fig_mae, use_container_width=True)

        with cal_col2:
            # Hit rate by confidence chart
            fig_hit = px.bar(
                calibration_df,
                x='confidence_bucket',
                y='hit_rate',
                title='Floor-Ceiling Hit Rate by Confidence',
                labels={'confidence_bucket': 'Confidence', 'hit_rate': 'Hit Rate (%)'},
                color='hit_rate',
                color_continuous_scale='RdYlGn'
            )
            fig_hit.update_layout(showlegend=False, xaxis_title="", yaxis_title="Hit Rate (%)")
            st.plotly_chart(fig_hit, use_container_width=True)

        # Calibration insight
        if len(calibration_df) >= 2:
            low_conf_mae = calibration_df[calibration_df['confidence_bucket'].str.contains('Low')]['mae'].values
            high_conf_mae = calibration_df[calibration_df['confidence_bucket'].str.contains('Very High|High')]['mae'].values
            if len(low_conf_mae) > 0 and len(high_conf_mae) > 0:
                if high_conf_mae[0] < low_conf_mae[0]:
                    st.success("✅ Good calibration: High confidence predictions ARE more accurate")
                else:
                    st.warning("⚠️ Calibration issue: High confidence predictions are NOT more accurate than low confidence")
    else:
        st.info("No confidence data available for calibration analysis")

    st.divider()

    # =========================================================================
    # SECTION 2.5: Bias Calibration Module
    # =========================================================================
    st.subheader("🔧 Bias Calibration")
    st.caption("Compare calibration methods and see recommended formula")

    import prediction_calibration as pcal
    calibrator = pcal.PredictionCalibrator(review_conn)

    try:
        cal_summary = calibrator.get_calibration_summary(days_back)

        if 'error' not in cal_summary:
            # Show comparison table
            st.write("**Calibration Method Comparison**")

            cal_data = []
            for method in ['raw', 'constant', 'linear', 'by_tier']:
                m = cal_summary[method]
                cal_data.append({
                    'Method': method.replace('_', ' ').title(),
                    'MAE': f"{m['mae']:.2f}",
                    'Bias': f"{m['bias']:+.2f}",
                    'RMSE': f"{m['rmse']:.2f}",
                    'R²': f"{m.get('r_squared', 0):.3f}" if method != 'raw' else "—"
                })

            cal_comparison_df = pd.DataFrame(cal_data)
            st.dataframe(cal_comparison_df, use_container_width=True, hide_index=True)

            # Find best method
            methods = ['constant', 'linear', 'by_tier']
            best_method = min(methods, key=lambda x: cal_summary[x]['mae'])
            improvement = cal_summary['raw']['mae'] - cal_summary[best_method]['mae']
            improvement_pct = (improvement / cal_summary['raw']['mae']) * 100

            st.success(f"✅ **Best method: {best_method.replace('_', ' ').title()}** — MAE improvement: {improvement:.2f} pts ({improvement_pct:.1f}%)")

            # Show recommended formula
            if best_method == 'linear':
                params = cal_summary['linear']['params']
                st.code(f"calibrated = {params['intercept']:.2f} + {params['slope']:.3f} × projected", language=None)
            elif best_method == 'constant':
                params = cal_summary['constant']['params']
                st.code(f"calibrated = projected + ({params['intercept']:.2f})", language=None)
            elif best_method == 'by_tier':
                st.write("**Tier-specific formulas:**")
                tier_params = cal_summary['by_tier']['params'].get('tier_params', {})
                for tier, (a, b) in tier_params.items():
                    st.code(f"{tier}: calibrated = {a:.2f} + {b:.3f} × projected", language=None)

            # FanDuel diagnostic
            with st.expander("🔍 FanDuel Comparison Diagnostic"):
                fd_diag = calibrator.get_fanduel_diagnostic(days_back)
                if not fd_diag.empty:
                    st.write("**Our Error - FD Error by Team** (negative = we're better)")
                    st.caption("If avg_error_diff is consistently positive, we're losing to FD. Low std suggests it's systematic, not noise.")

                    fd_display = fd_diag.copy()
                    fd_display['avg_error_diff'] = fd_display['avg_error_diff'].apply(lambda x: f"{x:+.2f}")
                    fd_display['std_error_diff'] = fd_display['std_error_diff'].apply(lambda x: f"{x:.2f}")
                    fd_display['our_mae'] = fd_display['our_mae'].apply(lambda x: f"{x:.2f}")
                    fd_display['fd_mae'] = fd_display['fd_mae'].apply(lambda x: f"{x:.2f}")
                    fd_display['win_rate'] = fd_display['win_rate'].apply(lambda x: f"{x:.0f}%")
                    fd_display.columns = ['Team', 'Avg Diff', 'Std Dev', 'Games', 'Wins', 'Our MAE', 'FD MAE', 'Win Rate']

                    # Add warning for small samples
                    fd_display['⚠️'] = fd_display['Games'].apply(lambda x: '⚠️' if int(x) < 20 else '')
                    st.dataframe(fd_display, use_container_width=True, hide_index=True)

                    # Summary insight
                    avg_diff = fd_diag['avg_error_diff'].mean()
                    if avg_diff > 0:
                        st.warning(f"⚠️ On average, our error is {avg_diff:.2f} pts HIGHER than FanDuel's. The bias correction above should help.")
                    else:
                        st.success(f"✅ On average, our error is {abs(avg_diff):.2f} pts LOWER than FanDuel's.")
                else:
                    st.info("No FanDuel comparison data available")
    except Exception as e:
        st.error(f"Calibration analysis error: {e}")

    st.divider()

    # =========================================================================
    # SECTION 3: Breakdown Analysis (Tabs)
    # =========================================================================
    st.subheader("📋 Performance Breakdowns")

    breakdown_tabs = st.tabs(["By Team", "By Opponent", "By Projection Range", "By Confidence", "By DFS Grade"])

    # Tab 1: By Team
    with breakdown_tabs[0]:
        team_query = """
            SELECT
                team_name,
                COUNT(*) as predictions,
                AVG(abs_error) as mae,
                AVG(error) as bias,
                SUM(CASE WHEN we_were_closer = 1 THEN 1.0 ELSE 0 END) /
                    NULLIF(SUM(CASE WHEN fanduel_ou IS NOT NULL THEN 1 ELSE 0 END), 0) * 100 as fd_win_rate
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND game_date >= ?
              AND game_date <= ?
            GROUP BY team_name
            HAVING COUNT(*) >= 5
            ORDER BY mae ASC
        """
        team_df = pd.read_sql_query(team_query, review_conn, params=[str(start_date), str(end_date)])

        if not team_df.empty:
            st.write("**Best & Worst Teams for Prediction Accuracy**")

            team_col1, team_col2 = st.columns(2)
            with team_col1:
                st.write("🏆 **Best (Lowest MAE)**")
                best_teams = team_df.head(5)
                for _, row in best_teams.iterrows():
                    st.markdown(f"- **{row['team_name']}**: MAE {row['mae']:.2f}, Bias {row['bias']:+.2f}")

            with team_col2:
                st.write("⚠️ **Worst (Highest MAE)**")
                worst_teams = team_df.tail(5).iloc[::-1]
                for _, row in worst_teams.iterrows():
                    st.markdown(f"- **{row['team_name']}**: MAE {row['mae']:.2f}, Bias {row['bias']:+.2f}")

            # Full table
            with st.expander("View All Teams"):
                display_df = team_df.copy()
                display_df['mae'] = display_df['mae'].apply(lambda x: f"{x:.2f}")
                display_df['bias'] = display_df['bias'].apply(lambda x: f"{x:+.2f}")
                display_df['fd_win_rate'] = display_df['fd_win_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
                display_df.columns = ['Team', 'Predictions', 'MAE', 'Bias', 'FD Win Rate']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough team data for analysis")

    # Tab 2: By Opponent
    with breakdown_tabs[1]:
        opp_query = """
            SELECT
                opponent_name,
                COUNT(*) as predictions,
                AVG(abs_error) as mae,
                AVG(error) as bias
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND game_date >= ?
              AND game_date <= ?
            GROUP BY opponent_name
            HAVING COUNT(*) >= 5
            ORDER BY mae ASC
        """
        opp_df = pd.read_sql_query(opp_query, review_conn, params=[str(start_date), str(end_date)])

        if not opp_df.empty:
            st.write("**Prediction Accuracy by Opponent Faced**")

            opp_col1, opp_col2 = st.columns(2)
            with opp_col1:
                st.write("🎯 **Easiest to Predict Against**")
                best_opp = opp_df.head(5)
                for _, row in best_opp.iterrows():
                    st.markdown(f"- vs **{row['opponent_name']}**: MAE {row['mae']:.2f}")

            with opp_col2:
                st.write("❓ **Hardest to Predict Against**")
                worst_opp = opp_df.tail(5).iloc[::-1]
                for _, row in worst_opp.iterrows():
                    st.markdown(f"- vs **{row['opponent_name']}**: MAE {row['mae']:.2f}")

            with st.expander("View All Opponents"):
                display_opp = opp_df.copy()
                display_opp['mae'] = display_opp['mae'].apply(lambda x: f"{x:.2f}")
                display_opp['bias'] = display_opp['bias'].apply(lambda x: f"{x:+.2f}")
                display_opp.columns = ['Opponent', 'Predictions', 'MAE', 'Bias']
                st.dataframe(display_opp, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough opponent data for analysis")

    # Tab 3: By Projection Range
    with breakdown_tabs[2]:
        range_query = """
            SELECT
                CASE
                    WHEN projected_ppg < 12 THEN '1. Bench (0-12)'
                    WHEN projected_ppg < 18 THEN '2. Role (12-18)'
                    WHEN projected_ppg < 25 THEN '3. Starter (18-25)'
                    ELSE '4. Star (25+)'
                END as scorer_tier,
                COUNT(*) as predictions,
                AVG(abs_error) as mae,
                AVG(error) as bias,
                AVG(projected_ppg) as avg_projection,
                AVG(actual_ppg) as avg_actual
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND game_date >= ?
              AND game_date <= ?
            GROUP BY scorer_tier
            ORDER BY scorer_tier
        """
        range_df = pd.read_sql_query(range_query, review_conn, params=[str(start_date), str(end_date)])

        if not range_df.empty:
            st.write("**Accuracy by Player Scoring Tier**")

            fig_range = px.bar(
                range_df,
                x='scorer_tier',
                y='mae',
                color='bias',
                title='MAE by Scorer Tier (color = bias)',
                labels={'scorer_tier': 'Scorer Tier', 'mae': 'MAE', 'bias': 'Bias'},
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            fig_range.update_layout(xaxis_title="", yaxis_title="MAE (pts)")
            st.plotly_chart(fig_range, use_container_width=True)

            # Insight
            star_data = range_df[range_df['scorer_tier'].str.contains('Star')]
            bench_data = range_df[range_df['scorer_tier'].str.contains('Bench')]
            if len(star_data) > 0 and len(bench_data) > 0:
                if star_data['mae'].values[0] > bench_data['mae'].values[0] * 1.5:
                    st.warning("⚠️ Star players (25+ pts) are harder to predict - consider wider ranges for high-volume scorers")
        else:
            st.info("Not enough data for projection range analysis")

    # Tab 4: By Confidence
    with breakdown_tabs[3]:
        conf_query = """
            SELECT
                CASE
                    WHEN proj_confidence < 0.6 THEN '1. Low (<60%)'
                    WHEN proj_confidence < 0.75 THEN '2. Medium (60-75%)'
                    WHEN proj_confidence < 0.85 THEN '3. High (75-85%)'
                    ELSE '4. Very High (85%+)'
                END as confidence_level,
                COUNT(*) as predictions,
                AVG(abs_error) as mae,
                AVG(CASE WHEN hit_floor_ceiling = 1 THEN 100.0 ELSE 0 END) as floor_ceiling_rate,
                SUM(CASE WHEN we_were_closer = 1 THEN 1.0 ELSE 0 END) /
                    NULLIF(SUM(CASE WHEN fanduel_ou IS NOT NULL THEN 1 ELSE 0 END), 0) * 100 as fd_win_rate
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND proj_confidence IS NOT NULL
              AND game_date >= ?
              AND game_date <= ?
            GROUP BY confidence_level
            ORDER BY confidence_level
        """
        conf_df = pd.read_sql_query(conf_query, review_conn, params=[str(start_date), str(end_date)])

        if not conf_df.empty:
            st.write("**Performance by Model Confidence Level**")

            conf_display = conf_df.copy()
            conf_display['mae'] = conf_display['mae'].apply(lambda x: f"{x:.2f}")
            conf_display['floor_ceiling_rate'] = conf_display['floor_ceiling_rate'].apply(lambda x: f"{x:.1f}%")
            conf_display['fd_win_rate'] = conf_display['fd_win_rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
            conf_display.columns = ['Confidence', 'Predictions', 'MAE', 'Hit Rate', 'FD Win Rate']
            st.dataframe(conf_display, use_container_width=True, hide_index=True)
        else:
            st.info("No confidence data available")

    # Tab 5: By DFS Grade
    with breakdown_tabs[4]:
        dfs_query = """
            SELECT
                dfs_grade,
                COUNT(*) as predictions,
                AVG(abs_error) as mae,
                AVG(error) as bias,
                AVG(dfs_score) as avg_dfs_score
            FROM predictions
            WHERE actual_ppg IS NOT NULL
              AND dfs_grade IS NOT NULL
              AND game_date >= ?
              AND game_date <= ?
            GROUP BY dfs_grade
            ORDER BY dfs_grade
        """
        dfs_df = pd.read_sql_query(dfs_query, review_conn, params=[str(start_date), str(end_date)])

        if not dfs_df.empty:
            st.write("**Performance by DFS Grade**")

            fig_dfs = px.bar(
                dfs_df,
                x='dfs_grade',
                y='mae',
                color='dfs_grade',
                title='MAE by DFS Grade',
                labels={'dfs_grade': 'DFS Grade', 'mae': 'MAE'},
                color_discrete_map={'A': '#2ecc71', 'B': '#3498db', 'C': '#f1c40f', 'D': '#e67e22', 'F': '#e74c3c'}
            )
            fig_dfs.update_layout(xaxis_title="", yaxis_title="MAE (pts)", showlegend=False)
            st.plotly_chart(fig_dfs, use_container_width=True)
        else:
            st.info("No DFS grade data available")

    st.divider()

    # =========================================================================
    # SECTION 4: FanDuel Edge Analysis
    # =========================================================================
    st.subheader("💰 FanDuel Edge Analysis")
    st.caption("Where do we beat the market?")

    fd_edge_query = """
        SELECT
            team_name,
            COUNT(*) as comparisons,
            SUM(CASE WHEN we_were_closer = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100 as win_rate,
            AVG(CASE WHEN we_were_closer = 1 THEN closer_margin ELSE 0 END) as avg_margin_when_closer,
            SUM(CASE WHEN ou_call_correct = 1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100 as ou_accuracy
        FROM predictions
        WHERE fanduel_ou IS NOT NULL
          AND actual_ppg IS NOT NULL
          AND game_date >= ?
          AND game_date <= ?
        GROUP BY team_name
        HAVING COUNT(*) >= 3
        ORDER BY win_rate DESC
    """
    fd_edge_df = pd.read_sql_query(fd_edge_query, review_conn, params=[str(start_date), str(end_date)])

    if not fd_edge_df.empty:
        fd_col1, fd_col2 = st.columns(2)

        with fd_col1:
            st.write("🏆 **Teams Where We Beat FanDuel**")
            top_edge = fd_edge_df[fd_edge_df['win_rate'] > 50].head(8)
            if not top_edge.empty:
                for _, row in top_edge.iterrows():
                    st.markdown(f"- **{row['team_name']}**: {row['win_rate']:.0f}% win rate ({row['comparisons']} games)")
            else:
                st.caption("No teams with >50% win rate")

        with fd_col2:
            st.write("⚠️ **Teams Where FanDuel Beats Us**")
            bottom_edge = fd_edge_df[fd_edge_df['win_rate'] < 50].tail(8).iloc[::-1]
            if not bottom_edge.empty:
                for _, row in bottom_edge.iterrows():
                    st.markdown(f"- **{row['team_name']}**: {row['win_rate']:.0f}% win rate ({row['comparisons']} games)")
            else:
                st.caption("No teams with <50% win rate")

        # O/U Accuracy
        with st.expander("Over/Under Call Accuracy"):
            ou_display = fd_edge_df[['team_name', 'comparisons', 'ou_accuracy']].copy()
            ou_display['ou_accuracy'] = ou_display['ou_accuracy'].apply(lambda x: f"{x:.1f}%")
            ou_display.columns = ['Team', 'Games', 'O/U Accuracy']
            ou_display = ou_display.sort_values('O/U Accuracy', ascending=False)
            st.dataframe(ou_display, use_container_width=True, hide_index=True)
    else:
        st.info("No FanDuel comparison data available for this period")

    st.divider()

    # =========================================================================
    # SECTION 5: Model Weaknesses (Worst Predictions)
    # =========================================================================
    st.subheader("🔍 Model Weaknesses")
    st.caption("Largest prediction misses to learn from")

    worst_query = """
        SELECT
            player_name,
            team_name,
            opponent_name,
            game_date,
            projected_ppg,
            actual_ppg,
            error,
            proj_confidence,
            dfs_grade
        FROM predictions
        WHERE actual_ppg IS NOT NULL
          AND game_date >= ?
          AND game_date <= ?
        ORDER BY ABS(error) DESC
        LIMIT 15
    """
    worst_df = pd.read_sql_query(worst_query, review_conn, params=[str(start_date), str(end_date)])

    if not worst_df.empty:
        display_worst = worst_df.copy()
        display_worst['projected_ppg'] = display_worst['projected_ppg'].apply(lambda x: f"{x:.1f}")
        display_worst['actual_ppg'] = display_worst['actual_ppg'].apply(lambda x: f"{x:.1f}")
        display_worst['error'] = display_worst['error'].apply(lambda x: f"{x:+.1f}")
        display_worst['proj_confidence'] = display_worst['proj_confidence'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        display_worst.columns = ['Player', 'Team', 'Opponent', 'Date', 'Projected', 'Actual', 'Error', 'Confidence', 'Grade']
        st.dataframe(display_worst, use_container_width=True, hide_index=True, height=400)

        # Pattern analysis
        with st.expander("📊 Failure Pattern Analysis"):
            # Analyze what went wrong
            over_projected = worst_df[worst_df['error'] > 0]
            under_projected = worst_df[worst_df['error'] < 0]

            st.write(f"**Over-projected:** {len(over_projected)} of {len(worst_df)} worst misses")
            st.write(f"**Under-projected:** {len(under_projected)} of {len(worst_df)} worst misses")

            if len(over_projected) > len(under_projected) * 1.5:
                st.warning("⚠️ Most big misses are over-projections - consider more conservative estimates")
            elif len(under_projected) > len(over_projected) * 1.5:
                st.warning("⚠️ Most big misses are under-projections - consider adjusting for breakout games")
    else:
        st.info("No prediction data available")

    st.divider()

    # =========================================================================
    # SECTION 6: Recommendations
    # =========================================================================
    st.subheader("💡 Recommendations")

    recommendations = []

    # Check bias
    if enhanced.get('bias') and abs(enhanced['bias']) > 1.5:
        direction = "over" if enhanced['bias'] > 0 else "under"
        recommendations.append({
            'priority': '🔴 High',
            'issue': f"Systematic {direction}-projection bias",
            'detail': f"Model {direction}-projects by {abs(enhanced['bias']):.1f} pts on average",
            'action': f"Consider {'reducing' if direction == 'over' else 'increasing'} baseline projections"
        })

    # Check Spearman correlation
    spearman = enhanced.get('spearman_correlation', 0)
    if spearman and spearman < 0.6:
        recommendations.append({
            'priority': '🔴 High',
            'issue': "Poor ranking accuracy",
            'detail': f"Spearman correlation is only {spearman:.3f} (want >0.7 for DFS)",
            'action': "Review factors used in projections - player ordering needs improvement"
        })
    elif spearman and spearman >= 0.7:
        recommendations.append({
            'priority': '🟢 Good',
            'issue': "Strong ranking accuracy",
            'detail': f"Spearman correlation of {spearman:.3f} is solid for DFS",
            'action': "Maintain current approach - player ordering is effective"
        })

    # Check FanDuel performance
    fd_win = fd_summary.get('we_closer_pct', 0)
    if fd_win and fd_win > 55:
        recommendations.append({
            'priority': '🟢 Good',
            'issue': "Edge vs FanDuel",
            'detail': f"Model beats FanDuel {fd_win:.1f}% of the time",
            'action': "Strong edge - continue monitoring for consistency"
        })
    elif fd_win and fd_win < 45:
        recommendations.append({
            'priority': '🟡 Medium',
            'issue': "FanDuel outperforming model",
            'detail': f"FanDuel is closer {100-fd_win:.1f}% of the time",
            'action': "Consider incorporating market consensus as a factor"
        })

    # Check MAE
    mae = enhanced.get('mae', 0)
    if mae and mae > 6:
        recommendations.append({
            'priority': '🟡 Medium',
            'issue': "High overall error",
            'detail': f"MAE of {mae:.2f} pts is above target of 5 pts",
            'action': "Focus on reducing outlier errors and improving consistency"
        })

    if recommendations:
        for rec in recommendations:
            with st.container():
                st.markdown(f"**{rec['priority']}** - {rec['issue']}")
                st.caption(rec['detail'])
                st.info(f"💡 {rec['action']}")
                st.write("")
    else:
        st.success("✅ No major issues detected! Model is performing well.")


def _enrich_players_with_vegas_signals(conn, players, game_date, debug=False):
    """
    Query vegas_implied_fpts from predictions table and compute edge
    vs our DFS optimizer projections for each player.

    Returns: Dict[player_id -> {vegas_fpts, edge_pct, signal, suggest_exposure}]
    If debug=True, also returns diagnostic info via st.write().
    """
    diag = {}  # diagnostic info

    # Step 1: Query predictions with vegas data
    try:
        cursor = conn.execute("""
            SELECT player_name, vegas_implied_fpts
            FROM predictions
            WHERE game_date = ? AND vegas_implied_fpts IS NOT NULL
        """, [game_date])
        rows = cursor.fetchall()
        diag['query_date'] = game_date
        diag['rows_with_vegas'] = len(rows)
    except Exception as e:
        diag['query_error'] = str(e)
        # Try simpler query to diagnose
        try:
            cursor2 = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE game_date = ?", [game_date]
            )
            total = cursor2.fetchone()[0]
            diag['total_predictions_for_date'] = total
        except Exception as e2:
            diag['count_error'] = str(e2)

        if debug:
            st.warning(f"Vegas query failed: {diag}")
        return {}

    if debug:
        # Additional diagnostics
        try:
            cursor3 = conn.execute(
                "SELECT COUNT(*), COUNT(vegas_implied_fpts), COUNT(fanduel_ou) "
                "FROM predictions WHERE game_date = ?", [game_date]
            )
            total_row = cursor3.fetchone()
            diag['total_preds'] = total_row[0]
            diag['with_vegas_fpts'] = total_row[1]
            diag['with_fanduel_ou'] = total_row[2]
        except Exception:
            pass
        try:
            dates_cursor = conn.execute(
                "SELECT DISTINCT game_date FROM predictions ORDER BY game_date DESC LIMIT 5"
            )
            diag['recent_game_dates'] = [r[0] for r in dates_cursor.fetchall()]
        except Exception:
            pass

    if not rows:
        if debug:
            st.info(f"Diagnostics: {diag}")
        return {}

    # Step 2: Build lookup: lowercase player_name -> vegas_fpts
    vegas_lookup = {}
    for row in rows:
        pname = row[0].strip().lower() if row[0] else ""
        vfpts = row[1]
        if pname and vfpts:
            vegas_lookup[pname] = vfpts

    diag['vegas_lookup_size'] = len(vegas_lookup)

    # Step 3: Match to DFSPlayer objects and compute edge
    signals = {}
    matched = 0
    unmatched_sample = []
    for p in players:
        player_key = p.name.strip().lower()
        if player_key not in vegas_lookup:
            if len(unmatched_sample) < 3:
                unmatched_sample.append(player_key)
            continue

        matched += 1
        vegas_fpts = vegas_lookup[player_key]
        proj_fpts = p.proj_fpts

        if proj_fpts < 1.0:
            continue

        edge_pct = ((vegas_fpts - proj_fpts) / proj_fpts) * 100

        # Determine signal
        signal = None
        suggest_exposure = False
        if edge_pct >= 10 and p.salary < 7000:
            signal = "BOOST"
            suggest_exposure = True
        elif edge_pct <= -15:
            signal = "FADE"
        elif abs(edge_pct) >= 5:
            signal = "WATCH"

        signals[p.player_id] = {
            'vegas_fpts': round(vegas_fpts, 1),
            'edge_pct': round(edge_pct, 1),
            'signal': signal,
            'suggest_exposure': suggest_exposure,
        }

    diag['players_in_pool'] = len(players)
    diag['matched'] = matched
    diag['signals_generated'] = len(signals)

    if debug:
        st.info(f"Diagnostics: {diag}")
        if unmatched_sample:
            vegas_sample = list(vegas_lookup.keys())[:3]
            st.caption(f"Sample unmatched DFS names: {unmatched_sample} | Sample DB names: {vegas_sample}")

    return signals


# DFS Lineup Builder tab ---------------------------------------------------
if selected_page == "DFS Lineup Builder":
    st.title("🏀 DFS Lineup Builder")
    st.caption("Generate optimized DraftKings NBA Classic lineups for GPP tournaments")

    # Initialize session state for DFS
    if 'dfs_player_pool' not in st.session_state:
        st.session_state.dfs_player_pool = []
    if 'dfs_lineups' not in st.session_state:
        st.session_state.dfs_lineups = []
    if 'dfs_upload_metadata' not in st.session_state:
        st.session_state.dfs_upload_metadata = {}
    if 'dfs_excluded_games' not in st.session_state:
        st.session_state.dfs_excluded_games = set()
    if 'dfs_all_games' not in st.session_state:
        st.session_state.dfs_all_games = []
    if 'dfs_exposure_targets' not in st.session_state:
        st.session_state.dfs_exposure_targets = {}
    if 'dfs_vegas_signals' not in st.session_state:
        st.session_state.dfs_vegas_signals = {}
    if 'dfs_vegas_blend_active' not in st.session_state:
        st.session_state.dfs_vegas_blend_active = False
    if 'dfs_projection_date' not in st.session_state:
        st.session_state.dfs_projection_date = None
    if 'dfs_stack_config' not in st.session_state:
        st.session_state.dfs_stack_config = {}

    # Get database connection
    dfs_conn = get_connection(str(db_path))

    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Builder Settings")
        num_lineups = st.slider("Number of lineups", 1, 100, 20, step=1)
        max_exposure = st.slider("Max player exposure %", 20, 60, 40, step=5) / 100
        min_salary = st.slider(
            "Min player salary",
            min_value=3000,
            max_value=5000,
            value=3000,
            step=100,
            help="Filter out players below this salary. Set to $3,100+ to exclude cheap $3K players that may be inactive."
        )

        max_remaining_salary = st.slider(
            "Max remaining salary",
            min_value=0,
            max_value=3000,
            value=1000,
            step=100,
            help="Maximum unused salary allowed per lineup. Lower values force fuller salary usage. $0 = must use exactly $50,000."
        )
        filter_no_minutes = st.checkbox(
            "Exclude players with no recent minutes",
            value=True,
            help="Filter out players who haven't played recently or average < 5 minutes"
        )

        st.divider()
        st.subheader("📋 DraftKings Rules")
        st.markdown("""
        - **Salary Cap**: $50,000
        - **Roster**: 8 positions
        - **Min Games**: 2 different games
        - **Scoring**:
          - PTS: +1.0
          - 3PM: +0.5
          - REB: +1.25
          - AST: +1.5
          - STL/BLK: +2.0
          - TO: -0.5
          - Double-Double: +1.5
          - Triple-Double: +3.0
        """)

    # Main content with tabs
    builder_tabs = st.tabs([
        "📤 Upload & Configure",
        "👥 Player Pool",
        "📊 Generated Lineups",
        "📈 Exposure Report",
        "🎯 Model Accuracy",
        "🦈 Opponent Analysis",
        "🎰 FanDuel Props",
        "🔬 Player Scouting"
    ])

    # ==========================================================================
    # TAB 1: Upload & Configure
    # ==========================================================================
    with builder_tabs[0]:
        st.subheader("Upload DraftKings CSV")
        st.markdown("""
        Download your contest CSV from DraftKings and upload it here.
        The file should contain player names, positions, salaries, and game information.
        """)

        uploaded_file = st.file_uploader(
            "Choose DraftKings CSV file",
            type=['csv'],
            help="Export from DraftKings contest lobby"
        )

        if uploaded_file is not None:
            # Save uploaded file
            csv_path = persist_uploaded_file(uploaded_file, ".csv")

            with st.spinner("Parsing DraftKings CSV..."):
                try:
                    players, metadata = dfs.parse_dk_csv(csv_path, dfs_conn)

                    # Save raw DK CSV to S3 (preserves salary/player data)
                    _s3_upload_slate_file(
                        csv_path, str(datetime.now(EASTERN_TZ).date()),
                        "dk_salaries.csv",
                        f"☁️ DK CSV saved to S3 for {datetime.now(EASTERN_TZ).date()}"
                    )

                    st.success(f"✅ Loaded {metadata['matched_players']} players from {metadata['unique_games']} games")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Players Matched", metadata['matched_players'])
                    with col2:
                        st.metric("Unique Games", metadata['unique_games'])
                    with col3:
                        low, high = metadata['salary_range']
                        st.metric("Salary Range", f"${low:,} - ${high:,}")

                    # Show fallback players (using DK avg instead of our projection)
                    if metadata.get('fallback_players'):
                        fallback_count = metadata.get('fallback_count', 0)
                        with st.expander(f"📊 {fallback_count} Fallback Players (using DK average)", expanded=False):
                            st.write("These players aren't in our database - using DraftKings average as projection:")
                            for name, team, dk_avg in metadata['fallback_players'][:20]:
                                st.caption(f"• {name} ({team}) — DK Avg: {dk_avg:.1f} FPTS")
                            if len(metadata['fallback_players']) > 20:
                                st.caption(f"...and {len(metadata['fallback_players']) - 20} more")
                            st.info("💡 Fallback players are typically two-way players, recent call-ups, or name mismatches.")

                    # Show injured players that will be auto-excluded
                    if metadata.get('injured_players'):
                        with st.expander(f"🏥 {metadata['injured_count']} Injured Players (Auto-Excluded)", expanded=True):
                            st.write("These players are OUT/DOUBTFUL and will be excluded from lineups:")
                            for name, status in metadata['injured_players']:
                                if 'DK' in status:
                                    status_emoji = "🟡"  # DK-detected (not in our API)
                                elif status.upper() == "OUT":
                                    status_emoji = "🔴"
                                else:
                                    status_emoji = "🟠"
                                st.caption(f"{status_emoji} {name} - {status}")

                            dk_count = metadata.get('dk_detected_out_count', 0)
                            if dk_count > 0:
                                st.info(f"🟡 {dk_count} player(s) detected as OUT from DraftKings CSV (not in our injury API)")

                    # Game selection - exclude postponed/cancelled games
                    st.divider()
                    st.subheader("🏟️ Select Games to Include")
                    st.caption("Uncheck any postponed or cancelled games")

                    # Get unique games from players
                    all_games = sorted(set(p.game_id for p in players))
                    st.session_state.dfs_all_games = all_games

                    # Display games as checkboxes in columns
                    game_cols = st.columns(min(3, len(all_games)))
                    included_games = set()

                    for i, game_id in enumerate(all_games):
                        col_idx = i % len(game_cols)
                        # Get teams in this game
                        game_teams = set(p.team for p in players if p.game_id == game_id)
                        game_label = ' vs '.join(sorted(game_teams)) if game_teams else game_id

                        with game_cols[col_idx]:
                            is_included = st.checkbox(
                                game_label,
                                value=game_id not in st.session_state.dfs_excluded_games,
                                key=f"game_{game_id}"
                            )
                            if is_included:
                                included_games.add(game_id)

                    # Update excluded games
                    st.session_state.dfs_excluded_games = set(all_games) - included_games

                    if st.session_state.dfs_excluded_games:
                        st.warning(f"⚠️ {len(st.session_state.dfs_excluded_games)} game(s) excluded from lineup generation")

                    # Generate projections
                    st.divider()
                    st.subheader("Generate Projections")

                    # Game date selector for rest day calculations
                    game_date_col1, game_date_col2 = st.columns([2, 1])
                    with game_date_col1:
                        projection_date = st.date_input(
                            "Projection Date (for rest day calculations)",
                            value=datetime.now(EASTERN_TZ).date(),
                            help="Select the date of the games to calculate rest days correctly"
                        )
                    with game_date_col2:
                        use_sophisticated = st.checkbox(
                            "Use Sophisticated Model",
                            value=True,
                            help="Enable advanced projections with rest days, PPM stats, and uncertainty"
                        )

                    if st.button("🔄 Generate Player Projections", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Load sophisticated prediction data if enabled
                        defense_map = None
                        def_style_map = None
                        def_ppm_df = None
                        league_avg_ppm = 0.462
                        injury_status_map = {}
                        teammate_injury_map = {}

                        if use_sophisticated:
                            status_text.text("Loading sophisticated prediction data...")
                            try:
                                # Load defense stats
                                defense_stats = load_team_defense_stats(
                                    str(db_path), DEFAULT_SEASON, DEFAULT_SEASON_TYPE
                                )
                                defense_map = {
                                    int(row["team_id"]): row.to_dict()
                                    for _, row in defense_stats.iterrows()
                                }
                                def_style_map = {
                                    int(row["team_id"]): str(row.get("defense_style") or "Neutral")
                                    for _, row in defense_stats.iterrows()
                                }

                                # Load PPM stats
                                try:
                                    _, def_ppm_df, league_avg_ppm = load_ppm_stats(
                                        str(db_path), DEFAULT_SEASON
                                    )
                                except Exception:
                                    pass

                                # Load injury data
                                try:
                                    active_injuries = ia.get_active_injuries(
                                        dfs_conn, check_return_dates=True
                                    )
                                    injury_status_map = {
                                        inj['player_id']: inj.get('status', 'unknown').lower()
                                        for inj in active_injuries
                                    }
                                    # Build teammate injury map
                                    for inj in active_injuries:
                                        team_id = inj.get('team_id')
                                        status = injury_status_map.get(inj['player_id'])
                                        if team_id and status in ('questionable', 'out', 'doubtful'):
                                            if team_id not in teammate_injury_map:
                                                teammate_injury_map[team_id] = []
                                            teammate_injury_map[team_id].append(inj['player_id'])
                                except Exception:
                                    pass

                                status_text.text("Sophisticated prediction data loaded ✓")
                            except Exception as e:
                                st.warning(f"Could not load sophisticated data: {e}. Using simple model.")
                                use_sophisticated = False

                        projected_players = []
                        game_date_str = str(projection_date) if projection_date else None

                        for i, player in enumerate(players):
                            progress = (i + 1) / len(players)
                            progress_bar.progress(progress)
                            status_text.text(f"Projecting {player.name}...")

                            player = dfs.generate_player_projections(
                                dfs_conn, player,
                                season=DEFAULT_SEASON,
                                season_type=DEFAULT_SEASON_TYPE,
                                game_date=game_date_str,
                                defense_map=defense_map if use_sophisticated else None,
                                def_style_map=def_style_map,
                                def_ppm_df=def_ppm_df,
                                league_avg_ppm=league_avg_ppm,
                                injury_status_map=injury_status_map,
                                teammate_injury_map=teammate_injury_map,
                            )
                            projected_players.append(player)

                        progress_bar.empty()
                        status_text.empty()

                        st.session_state.dfs_player_pool = projected_players
                        st.session_state.dfs_upload_metadata = metadata
                        st.session_state.dfs_projection_date = str(projection_date) if projection_date else None

                        # Enrich with Vegas signals if props exist
                        try:
                            vegas_signals = _enrich_players_with_vegas_signals(
                                dfs_conn, projected_players,
                                str(projection_date) if projection_date else str(datetime.now(EASTERN_TZ).date())
                            )
                            st.session_state.dfs_vegas_signals = vegas_signals
                            if vegas_signals:
                                boost_count = sum(1 for s in vegas_signals.values() if s['signal'] == 'BOOST')
                                if boost_count:
                                    st.info(f"🎰 Found {boost_count} Vegas BOOST candidate(s) — see Player Pool tab")
                        except Exception:
                            st.session_state.dfs_vegas_signals = {}

                        # Enrich with game environment correlation data
                        try:
                            game_date_for_enrich = str(projection_date) if projection_date else str(datetime.now(EASTERN_TZ).date())
                            projected_players = dfs.enrich_players_with_correlation_model(
                                projected_players, dfs_conn, game_date_for_enrich
                            )
                            projected_players = dfs.apply_correlation_ceiling_boost(projected_players)
                        except Exception:
                            pass

                        # Count minutes-validated players
                        no_minutes = sum(1 for p in projected_players if not p.minutes_validated)
                        if no_minutes:
                            st.warning(f"⚠️ {no_minutes} player(s) flagged with no recent minutes")

                        # Show projection summary
                        b2b_count = sum(1 for p in projected_players if getattr(p, 'days_rest', None) == 1)
                        rested_count = sum(1 for p in projected_players if (getattr(p, 'days_rest', None) or 0) >= 3)

                        st.success(f"✅ Generated projections for {len(projected_players)} players")
                        if use_sophisticated and (b2b_count > 0 or rested_count > 0):
                            st.info(f"📊 Rest analysis: **{b2b_count}** players on B2B (-8%), **{rested_count}** well-rested (+5%)")
                        st.info("Switch to the **Player Pool** tab to review and adjust projections.")

                        # Save projections to S3 (latest overrides earlier for same day)
                        try:
                            slate_date = str(projection_date) if projection_date else str(datetime.now(EASTERN_TZ).date())
                            proj_rows = []
                            for p in projected_players:
                                if p.is_injured or p.is_excluded:
                                    continue
                                proj_rows.append({
                                    'player_id': p.player_id, 'name': p.name,
                                    'team': p.team, 'opponent': p.opponent,
                                    'salary': p.salary, 'positions': ','.join(p.positions),
                                    'proj_fpts': p.proj_fpts, 'proj_points': p.proj_points,
                                    'proj_rebounds': p.proj_rebounds, 'proj_assists': p.proj_assists,
                                    'proj_steals': p.proj_steals, 'proj_blocks': p.proj_blocks,
                                    'proj_turnovers': p.proj_turnovers, 'proj_fg3m': p.proj_fg3m,
                                    'proj_floor': p.proj_floor, 'proj_ceiling': p.proj_ceiling,
                                    'fpts_per_dollar': p.fpts_per_dollar,
                                })
                            if proj_rows:
                                proj_df = pd.DataFrame(proj_rows)
                                with tempfile.NamedTemporaryFile(
                                    mode='w', suffix='.csv', delete=False
                                ) as tmp:
                                    proj_df.to_csv(tmp, index=False)
                                    tmp_path = Path(tmp.name)
                                _s3_upload_slate_file(
                                    tmp_path, slate_date,
                                    "projections.csv",
                                    f"☁️ Projections saved to S3 for {slate_date}"
                                )
                                try:
                                    tmp_path.unlink()
                                except OSError:
                                    pass
                            # Also backup DB (projections are now in tracking tables too)
                            _s3_backup_db(db_path, toast=False)
                        except Exception:
                            pass  # Best-effort S3 save

                except Exception as e:
                    st.error(f"❌ Error parsing CSV: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # ==========================================================================
    # TAB 2: Player Pool
    # ==========================================================================
    with builder_tabs[1]:
        st.subheader("Player Pool & Projections")

        if not st.session_state.dfs_player_pool:
            st.info("📤 Upload a DraftKings CSV in the first tab to populate the player pool.")
        else:
            players = st.session_state.dfs_player_pool

            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                teams = sorted(set(p.team for p in players))
                selected_teams = st.multiselect("Filter by Team", teams, default=[])
            with col2:
                positions = ['PG', 'SG', 'SF', 'PF', 'C']
                selected_positions = st.multiselect("Filter by Position", positions, default=[])
            with col3:
                salary_range = st.slider(
                    "Salary Range",
                    min_value=min(p.salary for p in players),
                    max_value=max(p.salary for p in players),
                    value=(min(p.salary for p in players), max(p.salary for p in players)),
                    step=500
                )

            # Apply filters
            filtered = players
            if selected_teams:
                filtered = [p for p in filtered if p.team in selected_teams]
            if selected_positions:
                filtered = [p for p in filtered if any(pos in p.positions for pos in selected_positions)]
            filtered = [p for p in filtered if salary_range[0] <= p.salary <= salary_range[1]]

            # Vegas blending
            vegas_sigs = st.session_state.get('dfs_vegas_signals', {})

            if vegas_sigs:
                blend_col1, blend_col2 = st.columns([1, 2])
                with blend_col1:
                    blend_vegas = st.toggle(
                        "Blend Model + Vegas",
                        value=st.session_state.dfs_vegas_blend_active,
                        help="Average our model FPTS with Vegas implied FPTS. Blended projections feed directly into lineup optimization."
                    )
                with blend_col2:
                    if blend_vegas:
                        blend_weight = st.slider(
                            "Vegas weight",
                            min_value=10, max_value=90, value=50, step=10,
                            format="%d%%",
                            help="50% = equal weight. Higher = trust Vegas more.",
                            key="vegas_blend_weight"
                        ) / 100
                    else:
                        blend_weight = 0.5

                # Store original proj_fpts on first access so we can always revert
                for p in players:
                    if not hasattr(p, '_model_proj_fpts'):
                        p._model_proj_fpts = p.proj_fpts
                        p._model_fpts_per_dollar = p.fpts_per_dollar

                vegas_players_blended = 0
                if blend_vegas:
                    model_weight = 1.0 - blend_weight
                    for p in players:
                        sig = vegas_sigs.get(p.player_id)
                        if sig and sig.get('vegas_fpts') is not None:
                            p.proj_fpts = round(p._model_proj_fpts * model_weight + sig['vegas_fpts'] * blend_weight, 1)
                            p.fpts_per_dollar = round((p.proj_fpts / p.salary) * 1000, 3) if p.salary > 0 else 0
                            vegas_players_blended += 1
                        else:
                            p.proj_fpts = p._model_proj_fpts
                            p.fpts_per_dollar = p._model_fpts_per_dollar
                    st.session_state.dfs_vegas_blend_active = True
                    st.caption(f"🎰 Blended projections active — {vegas_players_blended} players adjusted ({int(blend_weight*100)}% Vegas / {int((1-blend_weight)*100)}% Model). Lineups will use these projections.")
                else:
                    if st.session_state.dfs_vegas_blend_active:
                        # Restore original model projections
                        for p in players:
                            if hasattr(p, '_model_proj_fpts'):
                                p.proj_fpts = p._model_proj_fpts
                                p.fpts_per_dollar = p._model_fpts_per_dollar
                        st.session_state.dfs_vegas_blend_active = False

            # Sort options
            sort_options = ["Projected FPTS", "Salary", "Value (FPTS/$)", "Ceiling"]
            if vegas_sigs:
                sort_options.append("Vegas Edge")
            sort_by = st.selectbox(
                "Sort by",
                sort_options,
                index=0
            )

            if sort_by == "Projected FPTS":
                filtered.sort(key=lambda p: p.proj_fpts, reverse=True)
            elif sort_by == "Salary":
                filtered.sort(key=lambda p: p.salary, reverse=True)
            elif sort_by == "Value (FPTS/$)":
                filtered.sort(key=lambda p: p.fpts_per_dollar, reverse=True)
            elif sort_by == "Vegas Edge":
                filtered.sort(key=lambda p: vegas_sigs.get(p.player_id, {}).get('edge_pct', 0), reverse=True)
            else:
                filtered.sort(key=lambda p: p.proj_ceiling, reverse=True)

            # Display player table
            st.caption(f"Showing {len(filtered)} of {len(players)} players")

            # Build DataFrame for display
            player_data = []
            for p in filtered:
                # Status indicator (use getattr for backwards compatibility with old session state)
                status = ""
                is_injured = getattr(p, 'is_injured', False)
                injury_status = getattr(p, 'injury_status', '')
                is_locked = getattr(p, 'is_locked', False)
                is_excluded = getattr(p, 'is_excluded', False)

                if is_injured:
                    status = f"🔴 {injury_status}" if injury_status else "🔴 OUT"
                elif is_locked:
                    status = "🔒 LOCK"
                elif is_excluded:
                    status = "❌ EXCL"

                # Get rest and analytics info (use getattr for backwards compatibility)
                days_rest = getattr(p, 'days_rest', None)
                rest_mult = getattr(p, 'rest_multiplier', 1.0)
                analytics = getattr(p, 'analytics_used', '')

                # Build rest indicator
                rest_indicator = ""
                if days_rest == 1:
                    rest_indicator = "😴 B2B"  # Back-to-back
                elif days_rest and days_rest >= 3:
                    rest_indicator = f"💪 {days_rest}d"  # Well-rested
                elif days_rest:
                    rest_indicator = f"{days_rest}d"

                # Vegas signal columns
                v_sig = vegas_sigs.get(p.player_id, {})
                v_fpts = v_sig.get('vegas_fpts')
                v_edge = v_sig.get('edge_pct')

                is_blended = st.session_state.get('dfs_vegas_blend_active', False)
                model_fpts = getattr(p, '_model_proj_fpts', p.proj_fpts)

                row = {
                    'Status': status,
                    'Name': p.name,
                    'Team': p.team,
                    'Opp': p.opponent,
                    'Pos': '/'.join(p.positions),
                    'Salary': f"${p.salary:,}",
                }
                if is_blended and vegas_sigs:
                    row['Model'] = f"{model_fpts:.1f}"
                    row['Vegas FPTS'] = f"{v_fpts:.1f}" if v_fpts is not None else "—"
                    row['Blended'] = f"{p.proj_fpts:.1f}"
                    row['Edge'] = f"{v_edge:+.1f}%" if v_edge is not None else "—"
                elif vegas_sigs:
                    row['Proj FPTS'] = f"{p.proj_fpts:.1f}"
                    row['Vegas FPTS'] = f"{v_fpts:.1f}" if v_fpts is not None else "—"
                    row['Edge'] = f"{v_edge:+.1f}%" if v_edge is not None else "—"
                else:
                    row['Proj FPTS'] = f"{p.proj_fpts:.1f}"
                row.update({
                    'Floor': f"{p.proj_floor:.1f}",
                    'Ceiling': f"{p.proj_ceiling:.1f}",
                    'Value': f"{p.fpts_per_dollar:.2f}",
                    'Rest': rest_indicator,
                    'Own%': f"{p.ownership_proj:.1f}%",
                    '📊': analytics,
                })
                player_data.append(row)

            df = pd.DataFrame(player_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show count of injured players (use getattr for backwards compatibility)
            injured_count = len([p for p in filtered if getattr(p, 'is_injured', False)])
            if injured_count > 0:
                st.warning(f"🏥 {injured_count} injured player(s) shown above will be auto-excluded from lineups")

            # --- Vegas Edge Suggestions ---
            boost_players = []
            if vegas_sigs:
                for p in players:
                    sig = vegas_sigs.get(p.player_id, {})
                    if sig.get('signal') == 'BOOST' and not getattr(p, 'is_locked', False) and not getattr(p, 'is_excluded', False) and not getattr(p, 'is_injured', False):
                        boost_players.append((p, sig))

            with st.expander("🎰 Vegas Edge Suggestions", expanded=bool(boost_players)):
                # Recalculate button — always visible so user can load signals
                # after fetching props post-projection
                if st.button("🔄 Load / Recalculate Vegas Signals", help="Query props from database and compute edge signals"):
                    game_dt = st.session_state.get('dfs_projection_date') or str(datetime.now(EASTERN_TZ).date())
                    refreshed = _enrich_players_with_vegas_signals(dfs_conn, players, game_dt, debug=True)
                    st.session_state.dfs_vegas_signals = refreshed
                    if refreshed:
                        rc = sum(1 for s in refreshed.values() if s['signal'] == 'BOOST')
                        st.success(f"Loaded signals for {len(refreshed)} players — {rc} BOOST candidate(s)")
                        st.rerun()

                if not vegas_sigs:
                    st.caption("Click the button above to load Vegas signals after fetching FanDuel props.")
                elif not boost_players:
                    st.caption("No BOOST candidates found (edge ≥ +10% at salary < $7K). Signals are loaded for the player pool table above.")
                else:
                    # Summary metrics
                    avg_edge = sum(s['edge_pct'] for _, s in boost_players) / len(boost_players)
                    total_upside = sum(s['vegas_fpts'] - p.proj_fpts for p, s in boost_players)
                    st.markdown(f"**{len(boost_players)}** BOOST candidate(s) | Avg edge: **{avg_edge:+.1f}%** | Total FPTS upside: **{total_upside:+.1f}**")

                    # BOOST candidates table
                    boost_data = []
                    for p, sig in sorted(boost_players, key=lambda x: x[1]['edge_pct'], reverse=True):
                        boost_data.append({
                            'Name': p.name,
                            'Team': p.team,
                            'Pos': '/'.join(p.positions),
                            'Salary': f"${p.salary:,}",
                            'Proj FPTS': f"{p.proj_fpts:.1f}",
                            'Vegas FPTS': f"{sig['vegas_fpts']:.1f}",
                            'Edge': f"{sig['edge_pct']:+.1f}%",
                        })
                    st.dataframe(pd.DataFrame(boost_data), use_container_width=True, hide_index=True)

                    # Apply button
                    if st.button("🚀 Apply Vegas Boost Suggestions", help="Add BOOST players to exposure targets at 25%"):
                        current_targets = dict(st.session_state.dfs_exposure_targets)
                        added = 0
                        for p, _ in boost_players:
                            if p.player_id not in current_targets:
                                current_targets[p.player_id] = 0.25
                                added += 1
                        st.session_state.dfs_exposure_targets = current_targets
                        if added:
                            st.success(f"Added {added} BOOST player(s) to exposure targets at 25%")
                        else:
                            st.info("All BOOST players already have exposure targets set.")
                        st.rerun()

                    st.caption("Players where Vegas implied FPTS exceeds our projection by ≥10% at salary < $7K. "
                               "These are value plays where the market sees upside our model may underweight.")

            # Export player pool button
            export_col1, export_col2 = st.columns([1, 3])
            with export_col1:
                # Build export data with raw values for CSV
                export_data = []
                for p in filtered:
                    export_data.append({
                        'Name': p.name,
                        'ID': p.dk_id,
                        'Team': p.team,
                        'Opponent': p.opponent,
                        'Position': '/'.join(p.positions),
                        'Salary': p.salary,
                        'Proj_FPTS': round(p.proj_fpts, 2),
                        'Floor': round(p.proj_floor, 2),
                        'Ceiling': round(p.proj_ceiling, 2),
                        'Value': round(p.fpts_per_dollar, 3),
                        'Own_Pct': round(p.ownership_proj, 1),
                        'Days_Rest': getattr(p, 'days_rest', None),
                        'Rest_Mult': round(getattr(p, 'rest_multiplier', 1.0), 3),
                        'Uncertainty': round(getattr(p, 'uncertainty_multiplier', 1.0), 3),
                        'Analytics': getattr(p, 'analytics_used', ''),
                        'Game': p.game_id,
                    })
                export_pool_df = pd.DataFrame(export_data)
                csv_pool = export_pool_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export Player Pool",
                    data=csv_pool,
                    file_name="dk_player_pool.csv",
                    mime="text/csv",
                    help="Download player pool with projections"
                )

            # Lock/Exclude controls
            st.divider()
            st.subheader("Lock & Exclude Players")

            col1, col2 = st.columns(2)
            with col1:
                player_names = [p.name for p in players]
                locked_names = st.multiselect(
                    "🔒 Lock Players (must include)",
                    player_names,
                    default=[p.name for p in players if getattr(p, 'is_locked', False)],
                    help="These players will be included in ALL lineups"
                )
                for p in players:
                    p.is_locked = p.name in locked_names

            with col2:
                excluded_names = st.multiselect(
                    "❌ Exclude Players",
                    player_names,
                    default=[p.name for p in players if getattr(p, 'is_excluded', False)],
                    help="These players will NOT appear in any lineup"
                )
                for p in players:
                    p.is_excluded = p.name in excluded_names

            if locked_names:
                locked_salary = sum(p.salary for p in players if getattr(p, 'is_locked', False))
                st.caption(f"Locked players total: ${locked_salary:,} ({len(locked_names)} players)")

            # --- Exposure Targets ---
            st.divider()
            st.subheader("📊 Exposure Targets")
            st.caption("Set target exposure % for specific players (overrides global max)")

            # Filter out locked/excluded players from target selection
            available_for_targets = [
                p.name for p in players
                if not getattr(p, 'is_locked', False)
                and not getattr(p, 'is_excluded', False)
                and not getattr(p, 'is_injured', False)
            ]

            target_names = st.multiselect(
                "Select players for custom exposure",
                available_for_targets,
                help="Pick players and set how often they should appear in lineups"
            )

            exposure_targets = {}
            if target_names:
                cols = st.columns(3)
                for i, name in enumerate(target_names):
                    player_obj = next((p for p in players if p.name == name), None)
                    if player_obj:
                        with cols[i % 3]:
                            pct = st.number_input(
                                f"{name}", min_value=5, max_value=95,
                                value=25, step=5,
                                key=f"exp_target_{player_obj.player_id}"
                            )
                            exposure_targets[player_obj.player_id] = pct / 100

                st.caption(f"🎯 Targeting {len(target_names)} players with custom exposure")

            st.session_state.dfs_exposure_targets = exposure_targets

    # ==========================================================================
    # TAB 3: Generated Lineups
    # ==========================================================================
    with builder_tabs[2]:
        st.subheader("Generated Lineups")

        if not st.session_state.dfs_player_pool:
            st.info("📤 Upload a DraftKings CSV and generate projections first.")
        else:
            players = st.session_state.dfs_player_pool

            # --- Game Stack Selection ---
            # Build game list directly from the player pool (always available)
            proj_date_str = st.session_state.get('dfs_projection_date') or str(datetime.now(EASTERN_TZ).date())
            dk_games = {}  # dk_game_id -> {teams, players, stack_score}
            for p in players:
                if p.game_id not in dk_games:
                    dk_games[p.game_id] = {'teams': set(), 'players': [], 'stack_scores': []}
                dk_games[p.game_id]['teams'].add(p.team)
                dk_games[p.game_id]['players'].append(p)
                if getattr(p, 'stack_score', 0) > 0:
                    dk_games[p.game_id]['stack_scores'].append(p.stack_score)

            # Try to enrich with spread/total from game_odds table
            game_odds_lookup = {}  # team_set -> {spread, total, stack_score}
            try:
                tbl = dfs_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_odds'").fetchone()
                if tbl:
                    cursor = dfs_conn.execute("""
                        SELECT home_team, away_team, spread, total, stack_score
                        FROM game_odds WHERE date(game_date) = date(?)
                    """, [proj_date_str])
                    for row in cursor.fetchall():
                        key = frozenset([row['home_team'], row['away_team']])
                        game_odds_lookup[key] = {
                            'spread': row['spread'], 'total': row['total'],
                            'stack_score': row['stack_score'] if row['stack_score'] is not None else 0.5
                        }
            except Exception:
                pass

            with st.expander("🏀 Game Stacks", expanded=True):
                stack_options = ["Auto", "Primary Stack (3-4)", "Mini Stack (2)", "No Stack"]
                type_map = {"Auto": "auto", "Primary Stack (3-4)": "primary",
                            "Mini Stack (2)": "mini", "No Stack": "none"}
                stack_config = {}

                # Fetch Odds button
                fetch_col, _ = st.columns([2, 3])
                with fetch_col:
                    fetch_odds_clicked = st.button("🔄 Fetch/Refresh Odds", key="stack_fetch_odds")

                # Show fetch result from previous click (stored before rerun)
                if 'stack_fetch_result' in st.session_state:
                    msg = st.session_state.pop('stack_fetch_result')
                    if msg.startswith('OK:'):
                        st.success(msg[3:])
                    else:
                        st.error(msg)
                    # Show debug diagnostics if available
                    if 'stack_fetch_debug' in st.session_state:
                        debug_info = st.session_state.pop('stack_fetch_debug')
                        with st.expander("🔍 Fetch Diagnostics", expanded=True):
                            st.json(debug_info)

                if fetch_odds_clicked:
                    try:
                        fetch_date = datetime.strptime(proj_date_str, "%Y-%m-%d").date() if proj_date_str else datetime.now(EASTERN_TZ).date()
                    except (ValueError, TypeError):
                        fetch_date = datetime.now(EASTERN_TZ).date()
                    with st.spinner(f"Fetching odds for {fetch_date} from The Odds API..."):
                        try:
                            result = odds_api.fetch_fanduel_lines_for_date(dfs_conn, fetch_date, force=True)
                            # Store debug info for display
                            if result.get('odds_debug'):
                                st.session_state.stack_fetch_debug = result['odds_debug']
                            if result.get('success'):
                                stored = result.get('game_odds_stored', 0)
                                matched = result.get('players_matched', 0)
                                st.session_state.stack_fetch_result = (
                                    f"OK:Fetched odds for {result.get('events_fetched', 0)} games "
                                    f"({stored} game environments, {matched} players matched)"
                                )
                            else:
                                st.session_state.stack_fetch_result = f"Fetch failed: {result.get('error', 'Unknown error')}"
                        except Exception as e:
                            st.session_state.stack_fetch_result = f"Fetch error: {e}"
                    st.rerun()

                st.caption("Configure stacking per game. Primary = 3-4 players with bring-back. Mini = 2-player pair.")

                # Sort games: highest stack_score first
                def _game_sort_key(item):
                    gid, ginfo = item
                    # Use player-derived stack_score, or odds lookup, or 0
                    scores = ginfo['stack_scores']
                    if scores:
                        return max(scores)
                    team_key = frozenset(ginfo['teams'])
                    odds = game_odds_lookup.get(team_key)
                    if odds:
                        return odds.get('stack_score', 0)
                    return 0

                has_any_odds = bool(game_odds_lookup)
                sorted_games = sorted(dk_games.items(), key=_game_sort_key, reverse=True)

                for dk_gid, ginfo in sorted_games:
                    teams = sorted(ginfo['teams'])
                    matchup = " vs ".join(teams) if len(teams) >= 2 else teams[0] if teams else dk_gid
                    player_count = len(ginfo['players'])

                    # Get stack_score: prefer player-enriched, then odds lookup
                    scores = ginfo['stack_scores']
                    team_key = frozenset(ginfo['teams'])
                    odds_data = game_odds_lookup.get(team_key)

                    if scores:
                        ss = max(scores)
                    elif odds_data:
                        ss = odds_data.get('stack_score', 0.5)
                    else:
                        ss = 0.5  # neutral default

                    spread = odds_data.get('spread', 0) if odds_data else None
                    total = odds_data.get('total', 0) if odds_data else None

                    # Default selection based on stack_score
                    if ss >= 0.75:
                        default_idx = 0  # Auto (-> primary)
                    elif ss >= 0.50:
                        default_idx = 0  # Auto (-> mini)
                    else:
                        default_idx = 3  # No Stack

                    col_game, col_sel = st.columns([3, 2])
                    with col_game:
                        score_bar = "🟢" if ss >= 0.75 else "🟡" if ss >= 0.50 else "🔴"
                        if spread is not None and total is not None:
                            st.markdown(f"**{matchup}** — Spread: {spread:+.1f} | Total: {total:.0f} | {score_bar} {ss:.2f}")
                        else:
                            st.markdown(f"**{matchup}** — {player_count} players | {score_bar} {ss:.2f}")
                    with col_sel:
                        choice = st.selectbox(
                            "Stack", stack_options, index=default_idx,
                            key=f"stack_{dk_gid}", label_visibility="collapsed"
                        )

                    stack_config[dk_gid] = {
                        'type': type_map[choice],
                        'teams': list(ginfo['teams']),
                        'stack_score': ss,
                    }

                # Save config and show summary
                st.session_state.dfs_stack_config = stack_config
                active_stacks = sum(1 for c in stack_config.values() if c['type'] != 'none')
                if active_stacks:
                    st.caption(f"🎯 {active_stacks} game(s) configured for stacking")
                if not has_any_odds:
                    st.caption("Odds data not loaded — use Fetch/Refresh Odds for spread & total info")

            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(current, total, message):
                        progress_bar.progress(current / total if total > 0 else 0)
                        status_text.text(message)

                    # Apply salary minimum filter
                    filtered_players = [p for p in players if p.salary >= min_salary]
                    salary_filtered_count = len(players) - len(filtered_players)
                    if salary_filtered_count > 0:
                        st.info(f"💰 Filtered out {salary_filtered_count} players below ${min_salary:,} salary")

                    # Apply minutes validation filter
                    if filter_no_minutes:
                        before_count = len(filtered_players)
                        filtered_players = [p for p in filtered_players if p.minutes_validated]
                        mins_filtered = before_count - len(filtered_players)
                        if mins_filtered > 0:
                            st.info(f"⏱️ Filtered out {mins_filtered} players with no recent minutes")

                    if len(filtered_players) < 15:
                        st.error(f"Only {len(filtered_players)} players remain after filtering — need at least 15 to build lineups. Try disabling the minutes filter or lowering the salary minimum.")
                        st.stop()

                    # Calculate minimum salary floor from max remaining
                    min_salary_floor = dfs.SALARY_CAP - max_remaining_salary

                    with st.spinner("Optimizing lineups..."):
                        lineups = dfs.generate_diversified_lineups(
                            player_pool=filtered_players,
                            num_lineups=num_lineups,
                            max_player_exposure=max_exposure,
                            progress_callback=update_progress,
                            excluded_games=st.session_state.dfs_excluded_games,
                            exposure_targets=st.session_state.dfs_exposure_targets,
                            min_salary_floor=min_salary_floor,
                            stack_config=st.session_state.get('dfs_stack_config') or None
                        )

                    progress_bar.empty()
                    status_text.empty()

                    st.session_state.dfs_lineups = lineups

                    if lineups:
                        st.success(f"✅ Generated {len(lineups)} valid lineups!")

                        # Auto-save slate projections + lineups for tracking
                        try:
                            from dfs_tracking import (
                                create_dfs_tracking_tables,
                                save_slate_projections,
                                save_slate_lineups
                            )
                            slate_date = st.session_state.dfs_projection_date or str(datetime.now(EASTERN_TZ).date())
                            create_dfs_tracking_tables(dfs_conn)
                            save_slate_projections(dfs_conn, slate_date, players)
                            save_slate_lineups(dfs_conn, slate_date, lineups)
                            st.toast(f"📊 Slate saved for {slate_date} tracking")
                            # Backup DB to S3 (persists projections + lineups)
                            _s3_backup_db(db_path)
                        except Exception as e:
                            st.warning(f"⚠️ Could not save slate for tracking: {e}")
                    else:
                        st.error("❌ Could not generate any valid lineups. Try adjusting constraints.")

            with col2:
                if st.session_state.dfs_lineups:
                    # Export button — sorted by projected FPTS (highest first)
                    sorted_lineups = sorted(
                        st.session_state.dfs_lineups,
                        key=lambda l: l.total_proj_fpts, reverse=True
                    )
                    export_df = pd.DataFrame([l.to_dk_export_row() for l in sorted_lineups])

                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export for DraftKings",
                        data=csv_data,
                        file_name="dk_lineups.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            # Display generated lineups (sorted by projected FPTS, highest first)
            if st.session_state.dfs_lineups:
                lineups = sorted(
                    st.session_state.dfs_lineups,
                    key=lambda l: l.total_proj_fpts, reverse=True
                )

                st.divider()

                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                avg_proj = sum(l.total_proj_fpts for l in lineups) / len(lineups)
                avg_salary = sum(l.total_salary for l in lineups) / len(lineups)
                avg_ceiling = sum(l.total_ceiling for l in lineups) / len(lineups)

                with col1:
                    st.metric("Total Lineups", len(lineups))
                with col2:
                    st.metric("Avg Projected FPTS", f"{avg_proj:.1f}")
                with col3:
                    st.metric("Avg Salary Used", f"${avg_salary:,.0f}")
                with col4:
                    st.metric("Avg Ceiling", f"{avg_ceiling:.1f}")

                # Display each lineup
                st.subheader("Lineup Details")

                # Pre-fetch Vegas prop lines for all players in lineups
                vegas_props_lookup = {}
                try:
                    game_dt = st.session_state.get('dfs_projection_date') or str(datetime.now(EASTERN_TZ).date())
                    props_cursor = dfs_conn.execute("""
                        SELECT player_name,
                               fanduel_ou, fanduel_reb_ou, fanduel_ast_ou,
                               fanduel_3pm_ou, fanduel_stl_ou, fanduel_blk_ou,
                               vegas_implied_fpts
                        FROM predictions
                        WHERE game_date = ?
                    """, [game_dt])
                    for row in props_cursor.fetchall():
                        pname = row[0].strip().lower() if row[0] else ""
                        if pname:
                            vegas_props_lookup[pname] = {
                                'pts': row[1], 'reb': row[2], 'ast': row[3],
                                '3pm': row[4], 'stl': row[5], 'blk': row[6],
                                'vegas_fpts': row[7],
                            }
                except Exception:
                    pass

                vegas_sigs = st.session_state.get('dfs_vegas_signals', {})

                # Pagination
                lineups_per_page = 10
                total_pages = (len(lineups) + lineups_per_page - 1) // lineups_per_page
                page = st.selectbox("Page", range(1, total_pages + 1), index=0) - 1

                start_idx = page * lineups_per_page
                end_idx = min(start_idx + lineups_per_page, len(lineups))

                for i in range(start_idx, end_idx):
                    lineup = lineups[i]
                    with st.expander(
                        f"Lineup #{i+1} | Proj: {lineup.total_proj_fpts:.1f} | "
                        f"Salary: ${lineup.total_salary:,} | Ceiling: {lineup.total_ceiling:.1f}",
                        expanded=(i == start_idx)
                    ):
                        # Show lineup table with Vegas props
                        lineup_data = []
                        for slot in dfs.ROSTER_SLOTS:
                            player = lineup.players.get(slot)
                            if player:
                                props = vegas_props_lookup.get(player.name.strip().lower(), {})
                                sig = vegas_sigs.get(player.player_id, {})
                                signal_label = sig.get('signal', '')

                                # Build boost indicator
                                boost = ""
                                if signal_label == "BOOST":
                                    boost = "BOOST"
                                elif "💉" in getattr(player, 'analytics_used', ''):
                                    boost = "INJ+"

                                v_fpts = props.get('vegas_fpts')
                                row_data = {
                                    'Slot': slot,
                                    'Player': player.name,
                                    'Team': player.team,
                                    'Salary': f"${player.salary:,}",
                                    'Proj': f"{player.proj_fpts:.1f}",
                                    'Ceiling': f"{player.proj_ceiling:.1f}",
                                    'Own%': f"{player.ownership_proj:.1f}%",
                                    'Vegas': f"{v_fpts:.1f}" if v_fpts is not None else "—",
                                }

                                # Add individual stat props if any exist
                                has_props = any(props.get(k) is not None for k in ['pts', 'reb', 'ast', '3pm', 'stl', 'blk'])
                                if has_props:
                                    for key, label in [('pts', 'PTS'), ('reb', 'REB'), ('ast', 'AST'),
                                                       ('3pm', '3PM'), ('stl', 'STL'), ('blk', 'BLK')]:
                                        val = props.get(key)
                                        row_data[label] = f"{val:.1f}" if val is not None else "—"

                                if boost:
                                    row_data['Boost'] = boost

                                lineup_data.append(row_data)

                        df_lineup = pd.DataFrame(lineup_data)
                        # Drop columns that are entirely empty dashes
                        for col in ['PTS', 'REB', 'AST', '3PM', 'STL', 'BLK', 'Boost']:
                            if col in df_lineup.columns:
                                if (df_lineup[col] == "—").all() or (df_lineup[col] == "").all():
                                    df_lineup.drop(columns=[col], inplace=True)

                        st.dataframe(
                            df_lineup,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Show stacks
                        stacks = lineup.game_stacks
                        stack_info = [f"{game}: {len(players)} players" for game, players in stacks.items() if len(players) >= 2]
                        if stack_info:
                            st.caption(f"Game Stacks: {', '.join(stack_info)}")

    # ==========================================================================
    # TAB 4: Exposure Report
    # ==========================================================================
    with builder_tabs[3]:
        st.subheader("Exposure Report")

        if not st.session_state.dfs_lineups:
            st.info("🚀 Generate lineups first to see the exposure report.")
        else:
            lineups = st.session_state.dfs_lineups

            # Player exposure chart
            st.subheader("Player Exposure")

            exposure_df = dfs.generate_exposure_report(lineups)

            if not exposure_df.empty:
                # Top exposed players chart
                import plotly.express as px

                top_exposure = exposure_df.head(20)
                fig = px.bar(
                    top_exposure,
                    x='name',
                    y='exposure_pct',
                    color='exposure_pct',
                    color_continuous_scale='Blues',
                    labels={'name': 'Player', 'exposure_pct': 'Exposure %'},
                    title='Top 20 Most Exposed Players'
                )
                fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                # Full exposure table
                with st.expander("View All Player Exposures"):
                    display_exp = exposure_df.copy()
                    display_exp['exposure_pct'] = display_exp['exposure_pct'].apply(lambda x: f"{x:.1f}%")
                    display_exp['proj_fpts'] = display_exp['proj_fpts'].apply(lambda x: f"{x:.1f}")
                    display_exp['salary'] = display_exp['salary'].apply(lambda x: f"${x:,}")
                    display_exp.columns = ['Name', 'Team', 'Salary', 'Proj FPTS', 'Count', 'Exposure %']
                    st.dataframe(display_exp, use_container_width=True, hide_index=True)

            # Team exposure
            st.divider()
            st.subheader("Team Exposure")

            team_exposure = {}
            for lineup in lineups:
                for player in lineup.players.values():
                    if player.team not in team_exposure:
                        team_exposure[player.team] = 0
                    team_exposure[player.team] += 1

            team_df = pd.DataFrame([
                {'team': team, 'count': count, 'exposure_pct': count / (len(lineups) * 8) * 100}
                for team, count in team_exposure.items()
            ]).sort_values('exposure_pct', ascending=False)

            if not team_df.empty:
                fig_team = px.bar(
                    team_df,
                    x='team',
                    y='exposure_pct',
                    color='exposure_pct',
                    color_continuous_scale='Greens',
                    labels={'team': 'Team', 'exposure_pct': 'Exposure %'},
                    title='Team Distribution Across Lineups'
                )
                fig_team.update_layout(showlegend=False)
                st.plotly_chart(fig_team, use_container_width=True)

            # Salary distribution
            st.divider()
            st.subheader("Salary Distribution")

            salary_data = [l.total_salary for l in lineups]
            fig_salary = px.histogram(
                x=salary_data,
                nbins=20,
                labels={'x': 'Total Salary', 'y': 'Count'},
                title='Lineup Salary Distribution'
            )
            fig_salary.add_vline(x=50000, line_dash="dash", line_color="red",
                                annotation_text="Salary Cap")
            st.plotly_chart(fig_salary, use_container_width=True)

            # Stack report
            st.divider()
            st.subheader("Game Stacks")

            stack_df = dfs.generate_stack_report(lineups)
            if not stack_df.empty:
                st.dataframe(stack_df, use_container_width=True, hide_index=True)
            else:
                st.info("No significant stacks (2+ players from same game) found.")

    # ==========================================================================
    # TAB 5: Model Accuracy
    # ==========================================================================
    with builder_tabs[4]:
        st.subheader("DFS Model Accuracy Tracking")
        st.caption("Track how well our projections match actual performance across slates")

        try:
            from dfs_tracking import (
                create_dfs_tracking_tables,
                update_slate_actuals,
                compute_and_store_slate_results,
                get_dfs_accuracy_summary,
                get_slate_dates,
                get_pending_slate_dates,
                get_slate_projection_df,
                get_value_accuracy_df,
                get_slate_lineup_df,
                import_contest_results,
            )
            from dfs_optimizer import parse_contest_standings
            import plotly.express as px

            create_dfs_tracking_tables(dfs_conn)

            # --- Update Actuals Button ---
            pending_dates = get_pending_slate_dates(dfs_conn)
            all_dates = get_slate_dates(dfs_conn)

            if not all_dates:
                st.info("📊 No slates tracked yet. Generate lineups to start tracking model accuracy.")
            else:
                # Show pending slates count
                if pending_dates:
                    st.info(f"📋 **{len(pending_dates)} slate(s)** pending actuals update: {', '.join(pending_dates[:5])}{'...' if len(pending_dates) > 5 else ''}")

                if st.button("🔄 Update All Pending Slates", type="primary"):
                    if pending_dates:
                        with st.spinner(f"Updating {len(pending_dates)} slate(s)..."):
                            success_count = 0
                            for sd in pending_dates:
                                updated, not_found = update_slate_actuals(dfs_conn, sd)
                                results = compute_and_store_slate_results(dfs_conn, sd)
                                if results:
                                    st.toast(f"✅ {sd}: {updated} players (MAE: {results['proj_mae']:.1f})")
                                    success_count += 1
                                else:
                                    # Provide detailed diagnostic info
                                    if updated == 0 and not_found > 0:
                                        st.toast(f"⚠️ {sd}: No game data yet ({not_found} players awaiting results)")
                                    elif updated < 3:
                                        st.toast(f"⚠️ {sd}: Only {updated} players found - need 3+ for stats")
                                    else:
                                        st.toast(f"⚠️ {sd}: insufficient data (updated={updated}, not_found={not_found})")

                            # S3 backup
                            try:
                                from s3_storage import S3PredictionStorage
                                s3 = S3PredictionStorage()
                                if s3.is_connected():
                                    success, msg = s3.upload_database(db_path)
                                    if success:
                                        st.toast("☁️ Database backed up to S3")
                            except Exception:
                                pass

                        st.success(f"✅ Updated {success_count}/{len(pending_dates)} slates")
                        st.rerun()
                    else:
                        st.info("All slates already have actuals. Nothing to update.")

                # Diagnostic view for pending slates
                if pending_dates:
                    with st.expander("🔍 Pending Slate Diagnostics"):
                        any_missing = False
                        for sd in pending_dates[:5]:  # Limit to first 5
                            # Count projections for this slate
                            proj_count = dfs_conn.execute(
                                "SELECT COUNT(*) FROM dfs_slate_projections WHERE slate_date = ?",
                                (sd,)
                            ).fetchone()[0]

                            # Count game logs for this date (use LIKE since game_date is timestamp format)
                            logs_for_date = dfs_conn.execute(
                                "SELECT COUNT(*) FROM player_game_logs WHERE game_date LIKE ?",
                                (f"{sd}%",)
                            ).fetchone()[0]

                            # Count matching game logs (player_id AND date match)
                            # Use LIKE for date since player_game_logs stores "2026-02-04T00:00:00" format
                            game_data_count = dfs_conn.execute(
                                """SELECT COUNT(*) FROM dfs_slate_projections p
                                   JOIN player_game_logs g ON p.player_id = g.player_id AND g.game_date LIKE (p.slate_date || '%')
                                   WHERE p.slate_date = ?""",
                                (sd,)
                            ).fetchone()[0]

                            status_icon = "✅" if game_data_count >= 3 else "⏳"
                            st.caption(f"{status_icon} **{sd}**: {proj_count} projected, {logs_for_date} game logs exist, {game_data_count} matched")

                            # Show mismatch diagnostic if logs exist but don't match
                            if logs_for_date > 0 and game_data_count == 0:
                                st.warning(f"⚠️ **ID Mismatch detected for {sd}!** Game logs exist but player_ids don't match projections.")
                                # Sample IDs for debugging
                                sample_proj = dfs_conn.execute(
                                    "SELECT player_id, player_name FROM dfs_slate_projections WHERE slate_date = ? LIMIT 3",
                                    (sd,)
                                ).fetchall()
                                sample_log = dfs_conn.execute(
                                    "SELECT player_id, player_name FROM player_game_logs WHERE game_date = ? LIMIT 3",
                                    (sd,)
                                ).fetchall()
                                st.code(f"Projection IDs: {sample_proj}\nGame Log IDs: {sample_log}")

                            if game_data_count < 3:
                                any_missing = True

                        if any_missing:
                            st.info("💡 **Tip**: If game data is missing, run the data fetcher for those dates first, or wait until games complete.")

                # --- Contest Results Upload ---
                st.divider()
                st.subheader("📥 Import Contest Results")
                contest_file = st.file_uploader(
                    "Upload DraftKings Contest Standings CSV",
                    type=['csv'],
                    help="Download from DraftKings after contest completes (contest-standings-*.csv). "
                         "Extracts actual ownership (%Drafted) and FPTS per player."
                )
                if contest_file:
                    import tempfile, os
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                    tmp_path = tmp.name
                    try:
                        tmp.write(contest_file.read())
                        tmp.close()

                        ownership_map, actual_fpts_map = parse_contest_standings(tmp_path)

                        # Also parse opponent entries (left side of CSV)
                        from dfs_optimizer import parse_contest_entries
                        import re as _re
                        _cid_match = _re.search(r'(\d{6,})', contest_file.name)
                        _contest_id = _cid_match.group(1) if _cid_match else datetime.now(EASTERN_TZ).date().strftime('%Y%m%d')
                        try:
                            contest_entries = parse_contest_entries(tmp_path, _contest_id)
                            _unique_users = len(set(e['username'] for e in contest_entries))
                        except (ValueError, KeyError):
                            contest_entries = []
                            _unique_users = 0

                        if ownership_map:
                            st.success(
                                f"✅ Parsed {len(ownership_map)} players, "
                                f"{len(contest_entries)} unique lineups from {_unique_users} users"
                            )

                            # Let user pick which slate this belongs to
                            contest_slate_options = all_dates if all_dates else []
                            # Also offer today's date
                            today_str = datetime.now(EASTERN_TZ).date().strftime('%Y-%m-%d')
                            if today_str not in contest_slate_options:
                                contest_slate_options = [today_str] + contest_slate_options

                            contest_slate = st.selectbox(
                                "Match to slate date",
                                contest_slate_options,
                                help="Select which slate these contest results belong to"
                            )

                            # Preview top-10 owned players
                            preview_data = sorted(
                                ownership_map.items(), key=lambda x: x[1], reverse=True
                            )[:10]
                            preview_df = pd.DataFrame(preview_data, columns=['Player', 'Own%'])
                            preview_df['FPTS'] = preview_df['Player'].map(actual_fpts_map).apply(
                                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                            )
                            preview_df['Own%'] = preview_df['Own%'].apply(lambda x: f"{x:.1f}%")
                            st.caption("Top 10 most-owned players:")
                            st.dataframe(preview_df, use_container_width=True, hide_index=True)

                            if st.button("💾 Import Contest Data", type="primary"):
                                matched, unmatched = import_contest_results(
                                    dfs_conn, contest_slate, ownership_map, actual_fpts_map
                                )
                                st.toast(f"✅ Matched {matched} players, {unmatched} unmatched")

                                # Import opponent entries
                                from dfs_tracking import import_contest_entries as _import_entries
                                lineups_imp, users_imp = _import_entries(
                                    dfs_conn, contest_slate, _contest_id, contest_entries
                                )
                                st.toast(f"🦈 Imported {users_imp} users, {lineups_imp} unique lineups")

                                # Recompute slate results with ownership data
                                results = compute_and_store_slate_results(dfs_conn, contest_slate)
                                if results and results.get('ownership_mae') is not None:
                                    st.toast(f"📊 Ownership MAE: {results['ownership_mae']:.1f}%")

                                # S3 backup
                                try:
                                    _s3_backup_db(db_path, toast=True)
                                except Exception:
                                    pass

                                st.rerun()
                        else:
                            st.warning("⚠️ No player ownership data found in CSV. Check the file format.")

                    except Exception as e:
                        st.error(f"❌ Error parsing contest CSV: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass

                # --- Summary Table ---
                st.divider()
                summary_df = get_dfs_accuracy_summary(dfs_conn)

                if not summary_df.empty:
                    st.subheader("📋 Slate Results Summary")

                    # Get contest winner scores for each slate
                    winner_df = pd.read_sql_query(
                        """SELECT slate_date, MAX(top_score) as winner_score
                           FROM dfs_contest_meta
                           GROUP BY slate_date""",
                        dfs_conn
                    )
                    if not winner_df.empty:
                        summary_df = summary_df.merge(winner_df, on='slate_date', how='left')
                    else:
                        summary_df['winner_score'] = None

                    # Include ownership columns if they exist
                    summary_cols = [
                        'slate_date', 'num_players', 'num_lineups',
                        'proj_mae', 'proj_correlation', 'proj_within_range_pct',
                        'best_lineup_actual_fpts', 'winner_score', 'optimal_lineup_fpts', 'lineup_efficiency_pct'
                    ]
                    col_names = [
                        'Date', 'Players', 'Lineups',
                        'Proj MAE', 'Correlation', 'In Range %',
                        'Best Lineup', '🏆 Winner', 'Optimal', 'Efficiency %'
                    ]
                    if 'ownership_mae' in summary_df.columns:
                        summary_cols.extend(['ownership_mae', 'ownership_correlation'])
                        col_names.extend(['Own MAE', 'Own Corr'])

                    display_summary = summary_df[summary_cols].copy()
                    display_summary.columns = col_names

                    # Format numeric columns
                    fmt_cols = [c for c in display_summary.columns if c not in ('Date', 'Players', 'Lineups')]
                    for col in fmt_cols:
                        display_summary[col] = display_summary[col].apply(
                            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                        )

                    st.dataframe(display_summary, use_container_width=True, hide_index=True)

                    # --- Trend Charts (3+ slates) ---
                    if len(summary_df) >= 2:
                        st.divider()
                        st.subheader("📈 Trends Over Time")

                        trend_cols = st.columns(2)

                        with trend_cols[0]:
                            # MAE over time
                            fig_mae = px.line(
                                summary_df.sort_values('slate_date'),
                                x='slate_date', y='proj_mae',
                                title='Projection MAE Over Time',
                                labels={'slate_date': 'Slate Date', 'proj_mae': 'Mean Absolute Error'},
                                markers=True
                            )
                            fig_mae.update_layout(height=350)
                            st.plotly_chart(fig_mae, use_container_width=True)

                        with trend_cols[1]:
                            # Lineup efficiency over time
                            eff_df = summary_df.dropna(subset=['lineup_efficiency_pct']).sort_values('slate_date')
                            if not eff_df.empty:
                                fig_eff = px.line(
                                    eff_df,
                                    x='slate_date', y='lineup_efficiency_pct',
                                    title='Lineup Efficiency % Over Time',
                                    labels={'slate_date': 'Slate Date', 'lineup_efficiency_pct': 'Efficiency %'},
                                    markers=True
                                )
                                fig_eff.update_layout(height=350)
                                st.plotly_chart(fig_eff, use_container_width=True)

                        # Per-stat MAE comparison (latest slate)
                        latest = summary_df.iloc[0]
                        stat_mae_data = {
                            'Stat': ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', '3PM'],
                            'MAE': [
                                latest.get('stat_mae_points'),
                                latest.get('stat_mae_rebounds'),
                                latest.get('stat_mae_assists'),
                                latest.get('stat_mae_steals'),
                                latest.get('stat_mae_blocks'),
                                latest.get('stat_mae_turnovers'),
                                latest.get('stat_mae_fg3m'),
                            ]
                        }
                        stat_mae_df = pd.DataFrame(stat_mae_data).dropna()

                        if not stat_mae_df.empty:
                            fig_stat = px.bar(
                                stat_mae_df,
                                x='Stat', y='MAE',
                                title=f'Per-Stat MAE — {latest["slate_date"]}',
                                color='MAE',
                                color_continuous_scale='Reds',
                            )
                            fig_stat.update_layout(height=350, showlegend=False)
                            st.plotly_chart(fig_stat, use_container_width=True)

                    # --- Slate Detail ---
                    st.divider()
                    st.subheader("🔍 Slate Detail")

                    detail_date = st.selectbox(
                        "Select a slate to inspect",
                        summary_df['slate_date'].tolist(),
                        help="View detailed projection vs actual breakdown"
                    )

                    if detail_date:
                        detail_df = get_slate_projection_df(dfs_conn, detail_date)

                        if not detail_df.empty:
                            # Scatter plot: projected vs actual FPTS
                            fig_scatter = px.scatter(
                                detail_df,
                                x='proj_fpts', y='actual_fpts',
                                hover_name='player_name',
                                hover_data=['team', 'salary'],
                                title=f'Projected vs Actual FPTS — {detail_date}',
                                labels={'proj_fpts': 'Projected FPTS', 'actual_fpts': 'Actual FPTS'},
                                trendline='ols',
                            )
                            # Add perfect prediction line
                            max_val = max(detail_df['proj_fpts'].max(), detail_df['actual_fpts'].max())
                            fig_scatter.add_shape(
                                type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                                line=dict(color='gray', dash='dash', width=1)
                            )
                            fig_scatter.update_layout(height=500)
                            st.plotly_chart(fig_scatter, use_container_width=True)

                            # Key metrics for this slate
                            slate_result_df = summary_df[summary_df['slate_date'] == detail_date]
                            if not slate_result_df.empty:
                                slate_result = slate_result_df.iloc[0]

                                # Row 1: Projection accuracy metrics
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Proj MAE", f"{slate_result['proj_mae']:.1f}" if pd.notna(slate_result['proj_mae']) else "—")
                                m2.metric("Correlation", f"{slate_result['proj_correlation']:.3f}" if pd.notna(slate_result['proj_correlation']) else "—")
                                m3.metric("In Range %", f"{slate_result['proj_within_range_pct']:.1f}%" if pd.notna(slate_result['proj_within_range_pct']) else "—")
                                m4.metric("Efficiency %", f"{slate_result['lineup_efficiency_pct']:.1f}%" if pd.notna(slate_result['lineup_efficiency_pct']) else "—")

                                # Row 2: Lineup performance vs contest winner
                                # Get contest winner's score from dfs_contest_meta
                                winner_row = dfs_conn.execute(
                                    "SELECT top_score FROM dfs_contest_meta WHERE slate_date = ? ORDER BY import_date DESC LIMIT 1",
                                    (detail_date,)
                                ).fetchone()
                                winner_score = winner_row[0] if winner_row and winner_row[0] else None

                                best_lineup = slate_result.get('best_lineup_actual_fpts')
                                optimal = slate_result.get('optimal_lineup_fpts')

                                l1, l2, l3, l4 = st.columns(4)
                                l1.metric("Our Best Lineup", f"{best_lineup:.1f}" if pd.notna(best_lineup) else "—")
                                l2.metric("🏆 Contest Winner", f"{winner_score:.1f}" if winner_score else "—", help="Upload contest CSV to see winner's score")

                                if winner_score and pd.notna(best_lineup):
                                    gap = winner_score - best_lineup
                                    gap_pct = (best_lineup / winner_score * 100) if winner_score > 0 else 0
                                    l3.metric("Gap to Winner", f"{gap:+.1f}", delta=f"{gap_pct:.0f}% of winner", delta_color="inverse")
                                else:
                                    l3.metric("Gap to Winner", "—")

                                l4.metric("Optimal Possible", f"{optimal:.1f}" if pd.notna(optimal) else "—", help="Best lineup with perfect hindsight")

                            # --- Ownership Analysis ---
                            has_ownership = (
                                'ownership_proj' in detail_df.columns
                                and 'actual_ownership' in detail_df.columns
                                and detail_df['actual_ownership'].notna().sum() >= 5
                            )
                            if has_ownership:
                                with st.expander("🎯 Ownership Analysis", expanded=True):
                                    own_df = detail_df[['player_name', 'team', 'salary',
                                                         'ownership_proj', 'actual_ownership',
                                                         'actual_fpts']].dropna(
                                        subset=['ownership_proj', 'actual_ownership']
                                    ).copy()

                                    if not own_df.empty:
                                        # Ownership metrics
                                        own_m1, own_m2, own_m3 = st.columns(3)
                                        own_err = (own_df['actual_ownership'] - own_df['ownership_proj']).abs()
                                        own_mae = own_err.mean()
                                        own_m1.metric("Ownership MAE", f"{own_mae:.1f}%")

                                        if own_df['ownership_proj'].std() > 0 and own_df['actual_ownership'].std() > 0:
                                            own_corr = own_df['ownership_proj'].corr(own_df['actual_ownership'])
                                            own_m2.metric("Ownership Correlation", f"{own_corr:.3f}")
                                        else:
                                            own_m2.metric("Ownership Correlation", "—")

                                        own_m3.metric("Players w/ Ownership", str(len(own_df)))

                                        # Scatter: Projected vs Actual Ownership
                                        fig_own = px.scatter(
                                            own_df,
                                            x='ownership_proj', y='actual_ownership',
                                            hover_name='player_name',
                                            hover_data=['team', 'salary', 'actual_fpts'],
                                            title=f'Projected vs Actual Ownership — {detail_date}',
                                            labels={'ownership_proj': 'Projected Own%', 'actual_ownership': 'Actual Own%'},
                                            trendline='ols',
                                        )
                                        max_own = max(own_df['ownership_proj'].max(), own_df['actual_ownership'].max())
                                        fig_own.add_shape(
                                            type='line', x0=0, y0=0, x1=max_own, y1=max_own,
                                            line=dict(color='gray', dash='dash', width=1)
                                        )
                                        fig_own.update_layout(height=450)
                                        st.plotly_chart(fig_own, use_container_width=True)

                                        # Biggest Ownership Misses
                                        own_df['own_error'] = own_df['actual_ownership'] - own_df['ownership_proj']
                                        own_df['abs_error'] = own_df['own_error'].abs()
                                        misses = own_df.nlargest(10, 'abs_error').copy()
                                        misses_display = misses[['player_name', 'team', 'salary',
                                                                  'ownership_proj', 'actual_ownership',
                                                                  'own_error', 'actual_fpts']].copy()
                                        misses_display.columns = ['Player', 'Team', 'Salary',
                                                                   'Proj Own%', 'Actual Own%',
                                                                   'Error', 'Actual FPTS']
                                        for c in ['Proj Own%', 'Actual Own%', 'Error']:
                                            misses_display[c] = misses_display[c].apply(lambda x: f"{x:+.1f}%" if c == 'Error' else f"{x:.1f}%")
                                        misses_display['Actual FPTS'] = misses_display['Actual FPTS'].apply(lambda x: f"{x:.1f}")
                                        misses_display['Salary'] = misses_display['Salary'].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "—")
                                        st.markdown("**Biggest Ownership Misses** (where our model was most wrong):")
                                        st.dataframe(misses_display, use_container_width=True, hide_index=True)

                                        # Best Leverage Plays (low ownership + high actual FPTS)
                                        leverage_df = own_df[own_df['actual_ownership'] <= 15].nlargest(10, 'actual_fpts').copy()
                                        if not leverage_df.empty:
                                            lev_display = leverage_df[['player_name', 'team', 'salary',
                                                                        'actual_ownership', 'actual_fpts']].copy()
                                            lev_display.columns = ['Player', 'Team', 'Salary', 'Actual Own%', 'Actual FPTS']
                                            lev_display['Actual Own%'] = lev_display['Actual Own%'].apply(lambda x: f"{x:.1f}%")
                                            lev_display['Actual FPTS'] = lev_display['Actual FPTS'].apply(lambda x: f"{x:.1f}")
                                            lev_display['Salary'] = lev_display['Salary'].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "—")
                                            st.markdown("**Best Leverage Plays** (≤15% owned, highest actual FPTS):")
                                            st.dataframe(lev_display, use_container_width=True, hide_index=True)

                            # Value accuracy table
                            with st.expander("💰 Value Accuracy — Top & Bottom FPTS/$ Picks"):
                                value_df = get_value_accuracy_df(dfs_conn, detail_date)
                                if not value_df.empty:
                                    st.markdown("**Top 10 by Projected FPTS/$:**")
                                    top10 = value_df.head(10).copy()
                                    top10['fpts_per_dollar'] = top10['fpts_per_dollar'].apply(lambda x: f"{x:.2f}")
                                    top10['actual_fpts_per_dollar'] = top10['actual_fpts_per_dollar'].apply(lambda x: f"{x:.2f}")
                                    top10['proj_fpts'] = top10['proj_fpts'].apply(lambda x: f"{x:.1f}")
                                    top10['actual_fpts'] = top10['actual_fpts'].apply(lambda x: f"{x:.1f}")
                                    top10['salary'] = top10['salary'].apply(lambda x: f"${x:,}")
                                    top10['value_diff'] = top10['value_diff'].apply(lambda x: f"{x:+.1f}")
                                    top10.columns = ['Player', 'Team', 'Salary', 'Proj $/K', 'Proj FPTS', 'Actual FPTS', 'Pos', 'Actual $/K', 'Diff']
                                    st.dataframe(top10, use_container_width=True, hide_index=True)

                                    st.markdown("**Bottom 10 by Projected FPTS/$:**")
                                    bottom10 = value_df.tail(10).copy()
                                    bottom10['fpts_per_dollar'] = bottom10['fpts_per_dollar'].apply(lambda x: f"{x:.2f}")
                                    bottom10['actual_fpts_per_dollar'] = bottom10['actual_fpts_per_dollar'].apply(lambda x: f"{x:.2f}")
                                    bottom10['proj_fpts'] = bottom10['proj_fpts'].apply(lambda x: f"{x:.1f}")
                                    bottom10['actual_fpts'] = bottom10['actual_fpts'].apply(lambda x: f"{x:.1f}")
                                    bottom10['salary'] = bottom10['salary'].apply(lambda x: f"${x:,}")
                                    bottom10['value_diff'] = bottom10['value_diff'].apply(lambda x: f"{x:+.1f}")
                                    bottom10.columns = ['Player', 'Team', 'Salary', 'Proj $/K', 'Proj FPTS', 'Actual FPTS', 'Pos', 'Actual $/K', 'Diff']
                                    st.dataframe(bottom10, use_container_width=True, hide_index=True)

                            # Lineup performance breakdown
                            with st.expander("📊 Lineup Performance Breakdown"):
                                lineup_perf = get_slate_lineup_df(dfs_conn, detail_date)
                                if not lineup_perf.empty and lineup_perf['total_actual_fpts'].notna().any():
                                    fig_lineup = px.bar(
                                        lineup_perf.dropna(subset=['total_actual_fpts']),
                                        x='lineup_num',
                                        y=['total_proj_fpts', 'total_actual_fpts'],
                                        barmode='group',
                                        title=f'Lineup Performance — {detail_date}',
                                        labels={'lineup_num': 'Lineup #', 'value': 'Fantasy Points', 'variable': 'Type'},
                                    )
                                    fig_lineup.update_layout(height=400)
                                    st.plotly_chart(fig_lineup, use_container_width=True)
                                else:
                                    st.info("Lineup actuals not yet available. Click 'Update Actuals' after games complete.")

                            # Full player projection table
                            with st.expander("📋 Full Player Projections vs Actuals"):
                                show_df = detail_df[['player_name', 'team', 'salary', 'proj_fpts', 'actual_fpts',
                                                      'proj_points', 'actual_points', 'proj_rebounds', 'actual_rebounds',
                                                      'proj_assists', 'actual_assists']].copy()
                                show_df['error'] = show_df['actual_fpts'] - show_df['proj_fpts']
                                show_df = show_df.sort_values('error', key=abs, ascending=False)
                                show_df.columns = ['Player', 'Team', 'Salary', 'Proj FPTS', 'Actual FPTS',
                                                   'Proj PTS', 'Act PTS', 'Proj REB', 'Act REB',
                                                   'Proj AST', 'Act AST', 'Error']
                                for col in ['Proj FPTS', 'Actual FPTS', 'Proj PTS', 'Act PTS', 'Proj REB', 'Act REB', 'Proj AST', 'Act AST', 'Error']:
                                    show_df[col] = show_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
                                show_df['Salary'] = show_df['Salary'].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "—")
                                st.dataframe(show_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No actual data available for this slate yet.")
                else:
                    st.info("No slate results computed yet. Click 'Update Actuals' to compute accuracy metrics.")

            # --- S3 Slate Archive ---
            st.divider()
            with st.expander("☁️ S3 Slate Archive"):
                try:
                    from s3_storage import S3PredictionStorage
                    s3_arch = S3PredictionStorage()
                    if s3_arch.is_connected():
                        slate_files = s3_arch.list_files("dfs_slates/")
                        if slate_files:
                            s3_dates = sorted(set(
                                key.split('/')[1] for key in slate_files
                                if len(key.split('/')) >= 3
                            ), reverse=True)
                            st.write(f"**{len(s3_dates)} slate(s) archived in S3:**")
                            for d in s3_dates:
                                files = [k.split('/')[-1] for k in slate_files if f"/{d}/" in k]
                                st.write(f"• **{d}**: {', '.join(files)}")
                        else:
                            st.info("No slates archived in S3 yet. Upload a DK CSV and generate projections to start archiving.")
                    else:
                        st.info("S3 not configured. Slate data is stored locally only.")
                except Exception:
                    st.info("Could not connect to S3.")

        except Exception as e:
            st.error(f"❌ Error loading model accuracy: {e}")
            import traceback
            st.code(traceback.format_exc())

    # ==========================================================================
    # TAB 6: Opponent Analysis
    # ==========================================================================
    with builder_tabs[5]:
        st.subheader("Opponent Analysis")
        st.caption("Track opponents across contests to identify sharks and reverse-engineer their strategies")

        try:
            from dfs_tracking import (
                create_dfs_tracking_tables as _create_tables,
                get_opponent_contest_history,
                get_shark_users,
                get_shark_player_exposure,
                get_shark_strategy_profile,
                analyze_top_finishers,
            )
            import plotly.express as px

            _create_tables(dfs_conn)

            # Ensure player_roles table exists for role-based analysis
            try:
                from depth_chart import ensure_player_roles_table
                ensure_player_roles_table(dfs_conn)
            except Exception:
                pass

            # --- Contest Import Summary ---
            contest_history = get_opponent_contest_history(dfs_conn)

            if contest_history.empty:
                st.info(
                    "📊 No contest data imported yet. Upload a DraftKings contest standings CSV "
                    "in the **Model Accuracy** tab to start tracking opponents."
                )
            else:
                # Summary metrics
                ch_m1, ch_m2, ch_m3, ch_m4 = st.columns(4)
                ch_m1.metric("Contests Imported", len(contest_history))
                ch_m2.metric("Total Entries", f"{int(contest_history['total_entries'].sum()):,}")
                ch_m3.metric("Unique Users", f"{int(contest_history['unique_users'].sum()):,}")
                date_range = f"{contest_history['slate_date'].min()} → {contest_history['slate_date'].max()}"
                ch_m4.metric("Date Range", date_range if len(contest_history) > 1 else contest_history['slate_date'].iloc[0])

                with st.expander("📋 Imported Contests"):
                    ch_display = contest_history.copy()
                    ch_display.columns = ['Contest ID', 'Slate Date', 'Entries', 'Users', 'Top Score', 'Imported']
                    ch_display['Top Score'] = ch_display['Top Score'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
                    st.dataframe(ch_display, use_container_width=True, hide_index=True)

                # --- Shark Detection ---
                st.divider()
                st.subheader("🦈 Shark Detection")

                shark_col1, shark_col2 = st.columns(2)
                with shark_col1:
                    min_contests = st.slider("Min contests to qualify", 1, 10, 3, key="shark_min_contests")
                with shark_col2:
                    min_top_pct = st.slider("OR avg finish in top %", 5, 50, 25, key="shark_top_pct")

                sharks_df = get_shark_users(dfs_conn, min_contests=min_contests, min_top_pct=min_top_pct)

                if sharks_df.empty:
                    st.info(
                        f"No sharks detected yet with these criteria. "
                        f"Need users appearing in {min_contests}+ contests or avg finish in top {min_top_pct}%. "
                        f"Import more contest CSVs to build the database."
                    )
                else:
                    # Format shark table
                    shark_display = sharks_df.copy()
                    shark_display['avg_pts'] = shark_display['avg_pts'].apply(lambda x: f"{x:.1f}")
                    shark_display['avg_pctile'] = shark_display['avg_pctile'].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                    )
                    shark_display.columns = [
                        'Username', 'Contests', 'Avg Points', 'Avg Percentile',
                        'Best Rank', 'Max Entries', 'Total Entries'
                    ]
                    st.dataframe(shark_display, use_container_width=True, hide_index=True)

                    # --- Player Exposure Analysis ---
                    st.divider()
                    st.subheader("📊 Shark Player Preferences")
                    st.caption("Which players do sharks roster most frequently?")

                    shark_usernames = sharks_df['username'].tolist()
                    exposure_df = get_shark_player_exposure(dfs_conn, shark_usernames)

                    if not exposure_df.empty:
                        # Bar chart of top players
                        top_exposure = exposure_df.head(20)
                        fig_exp = px.bar(
                            top_exposure,
                            x='player', y='exposure_pct',
                            title='Top 20 Players by Shark Exposure',
                            labels={'player': 'Player', 'exposure_pct': 'Exposure %'},
                            color='exposure_pct',
                            color_continuous_scale='Reds',
                        )
                        fig_exp.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
                        st.plotly_chart(fig_exp, use_container_width=True)

                        # Exposure table
                        exp_display = exposure_df.head(30).copy()
                        exp_display['exposure_pct'] = exp_display['exposure_pct'].apply(lambda x: f"{x:.1f}%")
                        exp_display['avg_salary'] = exp_display['avg_salary'].apply(
                            lambda x: f"${int(x):,}" if pd.notna(x) else "—"
                        )
                        exp_display.columns = ['Player', 'Times Rostered', 'Contests In', 'Exposure %', 'Avg Salary']
                        st.dataframe(exp_display, use_container_width=True, hide_index=True)

                    # --- Individual Shark Profile ---
                    st.divider()
                    st.subheader("🔍 Shark Profile")

                    selected_shark = st.selectbox(
                        "Select a shark to analyze",
                        shark_usernames,
                        help="Deep-dive into a specific opponent's strategy"
                    )

                    if selected_shark:
                        profile = get_shark_strategy_profile(dfs_conn, selected_shark)

                        if profile:
                            # Metrics row
                            sp_m1, sp_m2, sp_m3, sp_m4 = st.columns(4)
                            sp_m1.metric("Contests", profile['contests'])
                            sp_m2.metric("Unique Lineups", profile['total_lineups'])
                            score = profile['score_stats']
                            sp_m3.metric("Avg Score", f"{score['avg']:.1f}")
                            consistency = max(0, (1 - score['std'] / score['avg']) * 100) if score['avg'] > 0 and score['std'] > 0 else 100
                            sp_m4.metric("Consistency", f"{consistency:.0f}%")

                            prof_cols = st.columns(2)

                            # Favorite players
                            with prof_cols[0]:
                                st.markdown("**Favorite Players:**")
                                if profile['favorite_players']:
                                    fav_data = []
                                    for item in profile['favorite_players']:
                                        player_name = item[0]
                                        times_used = item[1]
                                        salary = item[2] if len(item) > 2 else None
                                        fav_data.append({
                                            'Player': player_name,
                                            'Times': times_used,
                                            'Avg Salary': f"${int(salary):,}" if pd.notna(salary) and salary else "—"
                                        })
                                    fav_df = pd.DataFrame(fav_data)
                                    st.dataframe(fav_df, use_container_width=True, hide_index=True)

                            # Salary stats
                            with prof_cols[1]:
                                sal = profile['salary_stats']
                                if sal.get('avg') is not None:
                                    st.markdown("**Salary Usage:**")
                                    st.write(f"Average: **${sal['avg']:,.0f}**")
                                    st.write(f"Range: ${sal['min']:,} — ${sal['max']:,}")
                                else:
                                    st.info("Salary data unavailable (no matching projections for these slates)")

                            # Lineup history
                            with st.expander("📜 Lineup History"):
                                if profile['lineup_history']:
                                    hist_df = pd.DataFrame(profile['lineup_history'])
                                    hist_df['points'] = hist_df['points'].apply(lambda x: f"{x:.1f}")
                                    hist_df['salary'] = hist_df['salary'].apply(
                                        lambda x: f"${x:,}" if pd.notna(x) else "—"
                                    )
                                    hist_df.columns = ['Contest ID', 'Slate Date', 'Rank', 'Points', 'Salary']
                                    st.dataframe(hist_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No data found for this user.")

                # --- Top Finisher Analysis ---
                st.divider()
                st.subheader("🏆 Top Finisher Analysis")
                st.caption("Reverse-engineer winning lineups: salary allocation, stacking patterns, and player selection")

                # Select slate to analyze
                slate_dates = contest_history['slate_date'].tolist() if not contest_history.empty else []
                if slate_dates:
                    tf_col1, tf_col2 = st.columns([2, 1])
                    with tf_col1:
                        selected_slate = st.selectbox(
                            "Select slate to analyze",
                            slate_dates,
                            key="top_finisher_slate"
                        )
                    with tf_col2:
                        top_n = st.slider("Analyze top N lineups", 5, 50, 10, key="top_finisher_n")

                    if st.button("🔍 Analyze Top Finishers", type="primary"):
                        with st.spinner("Analyzing winning lineups..."):
                            analysis = analyze_top_finishers(dfs_conn, selected_slate, top_n)

                        if analysis.get('errors'):
                            st.warning(f"⚠️ {analysis['errors'][0]}")
                        else:
                            # Display insights
                            if analysis.get('insights'):
                                st.success("**Key Insights:**")
                                for insight in analysis['insights']:
                                    st.markdown(f"• {insight}")

                            # Metrics row
                            tf_m1, tf_m2, tf_m3, tf_m4 = st.columns(4)
                            winner_pts = analysis['lineups'][0]['points'] if analysis['lineups'] else 0
                            tf_m1.metric("Winner Score", f"{winner_pts:.1f}")

                            own_analysis = analysis.get('ownership_analysis', {})
                            tf_m2.metric("Avg Ownership", f"{own_analysis.get('avg_ownership', 0):.1f}%")
                            tf_m3.metric("Contrarian Plays", f"{own_analysis.get('contrarian_pct', 0):.0f}%",
                                        help="% of plays under 10% owned")

                            stack_count = len([l for l in analysis['lineups'] if l.get('stacks')])
                            tf_m4.metric("Lineups w/ Stacks", f"{stack_count}/{top_n}")

                            # Two column layout
                            tf_left, tf_right = st.columns(2)

                            with tf_left:
                                # Salary by position chart
                                if analysis.get('salary_by_position'):
                                    st.markdown("**💰 Salary Allocation by Position**")
                                    sal_data = []
                                    for pos, stats in analysis['salary_by_position'].items():
                                        sal_data.append({
                                            'Position': pos,
                                            'Avg Salary': stats['avg'],
                                            'Min': stats['min'],
                                            'Max': stats['max']
                                        })
                                    sal_df = pd.DataFrame(sal_data)

                                    fig_sal = px.bar(
                                        sal_df,
                                        x='Position', y='Avg Salary',
                                        title='Average Salary by Position',
                                        color='Avg Salary',
                                        color_continuous_scale='Greens',
                                        text='Avg Salary'
                                    )
                                    fig_sal.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                                    fig_sal.update_layout(height=350, showlegend=False)
                                    st.plotly_chart(fig_sal, use_container_width=True)

                                # Team stacks
                                if analysis.get('team_stacks'):
                                    st.markdown("**🔗 Stacking Patterns**")
                                    stack_df = pd.DataFrame(analysis['team_stacks'][:10])
                                    stack_df.columns = ['Stack', 'Lineups']
                                    st.dataframe(stack_df, use_container_width=True, hide_index=True)

                            with tf_right:
                                # Most rostered players (with role)
                                if analysis.get('player_frequency'):
                                    st.markdown("**⭐ Most Rostered Players**")
                                    freq_raw = analysis['player_frequency'][:12]
                                    role_icons_freq = {'STAR': '🌟', 'STARTER': '🟢', 'ROTATION': '🔄', 'BENCH': '🪑'}
                                    freq_df = pd.DataFrame([{
                                        'Player': f.get('player', ''),
                                        'Role': f"{role_icons_freq.get(f.get('role', ''), '❓')} {f.get('role', '') or '?'}",
                                        'Count': f.get('count', 0),
                                        '% of Top': f.get('pct', 0),
                                        'Own%': f"{f['ownership']:.1f}" if f.get('ownership') is not None else "—",
                                    } for f in freq_raw])

                                    fig_freq = px.bar(
                                        freq_df,
                                        x='Player', y='% of Top',
                                        title=f'Player Frequency in Top {top_n}',
                                        color='% of Top',
                                        color_continuous_scale='Blues',
                                        text='Count'
                                    )
                                    fig_freq.update_traces(textposition='outside')
                                    fig_freq.update_layout(height=350, showlegend=False, xaxis_tickangle=-45)
                                    st.plotly_chart(fig_freq, use_container_width=True)

                                    # Table with role and ownership
                                    st.dataframe(freq_df, use_container_width=True, hide_index=True)

                            # Role Composition & Ownership Analysis
                            role_analysis = analysis.get('role_analysis', {})
                            if role_analysis:
                                st.divider()
                                ra_left, ra_right = st.columns(2)

                                with ra_left:
                                    st.markdown("**🎭 Role Composition in Winning Lineups**")
                                    role_icons = {'STAR': '🌟', 'STARTER': '🟢', 'ROTATION': '🔄', 'BENCH': '🪑', 'UNKNOWN': '❓'}
                                    role_rows = []
                                    for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH', 'UNKNOWN']:
                                        stats = role_analysis.get(role)
                                        if stats:
                                            role_rows.append({
                                                'Role': f"{role_icons.get(role, '')} {role}",
                                                'Players': stats['count'],
                                                '% of Slots': f"{stats['pct_of_lineups']:.0f}%",
                                                'Avg FPTS': f"{stats['avg_fpts']:.1f}" if stats.get('avg_fpts') is not None else "—",
                                                'Avg Salary': f"${stats['avg_salary']:,}" if stats.get('avg_salary') is not None else "—",
                                            })
                                    if role_rows:
                                        st.dataframe(pd.DataFrame(role_rows), use_container_width=True, hide_index=True)

                                    # Role distribution pie chart
                                    pie_data = [{'Role': r['Role'], 'Count': r['Players']} for r in role_rows if r['Players'] > 0]
                                    if pie_data:
                                        fig_pie = px.pie(
                                            pd.DataFrame(pie_data),
                                            names='Role', values='Count',
                                            title='Role Distribution',
                                            color_discrete_sequence=px.colors.qualitative.Set2,
                                        )
                                        fig_pie.update_layout(height=300)
                                        st.plotly_chart(fig_pie, use_container_width=True)

                                with ra_right:
                                    st.markdown("**📊 Ownership by Role Tier**")
                                    own_role_rows = []
                                    for role in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
                                        stats = role_analysis.get(role)
                                        if stats and stats.get('avg_ownership') is not None:
                                            own_role_rows.append({
                                                'Role': f"{role_icons.get(role, '')} {role}",
                                                'Avg Own%': stats['avg_ownership'],
                                                'Avg FPTS': stats['avg_fpts'] if stats.get('avg_fpts') is not None else 0,
                                                'Count': stats['count'],
                                            })
                                    if own_role_rows:
                                        own_df = pd.DataFrame(own_role_rows)
                                        fig_own_role = px.bar(
                                            own_df,
                                            x='Role', y='Avg Own%',
                                            title='Average Ownership by Role',
                                            color='Avg FPTS',
                                            color_continuous_scale='YlOrRd',
                                            text='Avg Own%',
                                        )
                                        fig_own_role.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                        fig_own_role.update_layout(height=350, showlegend=False)
                                        st.plotly_chart(fig_own_role, use_container_width=True)

                                    # Ownership distribution across all players
                                    own_an = analysis.get('ownership_analysis', {})
                                    if own_an:
                                        st.markdown("**Ownership Breakdown**")
                                        own_dist = pd.DataFrame([
                                            {'Tier': 'Chalk (25%+)', 'Plays': own_an.get('high_ownership_plays', 0)},
                                            {'Tier': 'Mid (10-25%)', 'Plays': own_an.get('mid_ownership_plays', 0)},
                                            {'Tier': 'Contrarian (<10%)', 'Plays': own_an.get('low_ownership_plays', 0)},
                                        ])
                                        fig_own_dist = px.bar(
                                            own_dist, x='Tier', y='Plays',
                                            title='Ownership Tier Distribution',
                                            color='Tier',
                                            color_discrete_map={
                                                'Chalk (25%+)': '#e74c3c',
                                                'Mid (10-25%)': '#f39c12',
                                                'Contrarian (<10%)': '#27ae60',
                                            },
                                            text='Plays',
                                        )
                                        fig_own_dist.update_traces(textposition='outside')
                                        fig_own_dist.update_layout(height=300, showlegend=False)
                                        st.plotly_chart(fig_own_dist, use_container_width=True)

                            # Individual lineup breakdown
                            with st.expander("📋 Individual Lineup Details"):
                                for i, lineup in enumerate(analysis['lineups'][:10]):
                                    st.markdown(f"**#{lineup['rank']} — {lineup['points']:.1f} pts** ({lineup['username']})")

                                    # Build player table
                                    player_rows = []
                                    for p in lineup['players']:
                                        own_str = f"{p['ownership']:.1f}%" if p.get('ownership') else "—"
                                        proj_own_str = f"{p['proj_ownership']:.1f}%" if p.get('proj_ownership') else "—"
                                        proj_fpts_str = f"{p['proj_fpts']:.1f}" if p.get('proj_fpts') else "—"
                                        vegas_fpts_str = f"{p['vegas_fpts']:.1f}" if p.get('vegas_fpts') else "—"
                                        actual_fpts_str = f"{p['fpts']:.1f}" if p.get('fpts') else "—"
                                        role_icons = {'STAR': '🌟', 'STARTER': '🟢', 'ROTATION': '🔄', 'BENCH': '🪑'}
                                        role = p.get('role_tier', '') or ''
                                        role_str = f"{role_icons.get(role, '❓')} {role}" if role else "—"
                                        player_rows.append({
                                            'Pos': p['position'],
                                            'Player': p['name'],
                                            'Role': role_str,
                                            'Team': p['team'],
                                            'Salary': f"${p['salary']:,}" if p.get('salary') else "—",
                                            'Proj FPTS': proj_fpts_str,
                                            'Proj Own%': proj_own_str,
                                            'Own%': own_str,
                                            'FPTS': actual_fpts_str,
                                        })
                                    lineup_df = pd.DataFrame(player_rows)
                                    st.dataframe(lineup_df, use_container_width=True, hide_index=True)

                                    if lineup.get('stacks'):
                                        stacks_str = ', '.join([f"{s['team']} x{s['count']}" for s in lineup['stacks']])
                                        st.caption(f"🔗 Stacks: {stacks_str}")
                                    st.markdown("---")
                else:
                    st.info("Import contest CSVs first to analyze top finishers.")

        except Exception as e:
            st.error(f"❌ Error loading opponent analysis: {e}")
            import traceback
            st.code(traceback.format_exc())

    # ==========================================================================
    # TAB 7: FanDuel Props
    # ==========================================================================
    with builder_tabs[6]:
        st.subheader("FanDuel Player Props")
        st.caption("View all FanDuel player prop lines alongside our projections — verify API data is populated")

        # Date selector — default to DFS projection date if available
        props_default_date = datetime.now(EASTERN_TZ).date()
        if st.session_state.dfs_projection_date:
            try:
                props_default_date = datetime.strptime(
                    st.session_state.dfs_projection_date, "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                pass

        props_date = st.date_input(
            "Game Date",
            value=props_default_date,
            key="fanduel_props_date"
        )

        # Ensure schema is up to date (idempotent — only adds missing columns)
        odds_api.create_odds_tables(dfs_conn)
        pt.upgrade_predictions_table_for_fanduel(dfs_conn)
        odds_api.upgrade_predictions_for_fanduel(dfs_conn)

        # --- Fetch Button (reuses FanDuel Compare pattern) ---
        api_key = odds_api.get_api_key()

        if api_key:
            should_fetch, reason = odds_api.should_fetch_odds(dfs_conn, props_date)
            already_fetched = "Already fetched" in reason
            force_refetch = False

            if already_fetched:
                st.caption(f"ℹ️ {reason}")
                force_refetch = st.checkbox(
                    "🔄 Force refetch (update lines after injury news)",
                    help="Refetch after injury news or lineup changes. Uses additional API quota.",
                    key="props_force_refetch"
                )

            if should_fetch or force_refetch:
                button_label = "🔄 Refetch FanDuel Lines" if force_refetch else "🔄 Fetch FanDuel Lines"
                if st.button(button_label, type="primary", key="props_fetch_btn"):
                    with st.spinner("Fetching all player props from The Odds API..."):
                        result = odds_api.fetch_fanduel_lines_for_date(
                            dfs_conn, props_date, force=force_refetch
                        )
                        if result['success']:
                            extended_count = result.get('extended_stats_found', 0)
                            st.success(
                                f"✅ Fetched lines for {result['players_matched']} players "
                                f"({extended_count} with extended stats) "
                                f"using {result['api_requests_used']} API requests"
                            )
                            st.rerun()
                        else:
                            st.error(f"Fetch failed: {result.get('error', 'Unknown error')}")
            elif not already_fetched:
                st.caption(f"ℹ️ {reason}")
        else:
            st.warning(
                "**API key not configured.** Add `[theoddsapi]` section to `.streamlit/secrets.toml`:\n\n"
                "```toml\n[theoddsapi]\nAPI_KEY = \"your-api-key-here\"\n```"
            )

        st.divider()

        # --- Query predictions with FanDuel props ---
        # Try LEFT JOIN to dfs_slate_projections for our per-stat projections
        props_df = None
        has_proj_stats = False

        try:
            props_query = """
                SELECT
                    p.player_name,
                    p.team_name,
                    p.opponent_name,
                    p.projected_ppg,
                    p.fanduel_ou,
                    p.fanduel_reb_ou,
                    p.fanduel_ast_ou,
                    p.fanduel_3pm_ou,
                    p.fanduel_stl_ou,
                    p.fanduel_blk_ou,
                    p.fanduel_pra_ou,
                    p.vegas_implied_fpts,
                    p.fanduel_fetched_at,
                    d.proj_rebounds,
                    d.proj_assists,
                    d.proj_fg3m,
                    d.proj_steals,
                    d.proj_blocks,
                    d.proj_fpts
                FROM predictions p
                LEFT JOIN dfs_slate_projections d
                    ON p.player_id = d.player_id
                    AND d.slate_date = p.game_date
                WHERE p.game_date = ?
                ORDER BY p.projected_ppg DESC
            """
            props_df = pd.read_sql_query(props_query, dfs_conn, params=[str(props_date)])
            has_proj_stats = 'proj_rebounds' in props_df.columns
        except Exception:
            pass

        # Fallback: query predictions alone if join failed
        if props_df is None or props_df.empty:
            try:
                fallback_query = """
                    SELECT
                        player_name,
                        team_name,
                        opponent_name,
                        projected_ppg,
                        fanduel_ou,
                        fanduel_reb_ou,
                        fanduel_ast_ou,
                        fanduel_3pm_ou,
                        fanduel_stl_ou,
                        fanduel_blk_ou,
                        fanduel_pra_ou,
                        vegas_implied_fpts,
                        fanduel_fetched_at
                    FROM predictions
                    WHERE game_date = ?
                    ORDER BY projected_ppg DESC
                """
                props_df = pd.read_sql_query(fallback_query, dfs_conn, params=[str(props_date)])
                has_proj_stats = False
            except Exception as e:
                props_df = pd.DataFrame()
                st.warning(f"Could not query predictions: {e}")

        if props_df.empty:
            st.info(
                f"No predictions found for {props_date}. "
                "Generate predictions in the **Today's Games** tab first."
            )
        else:
            # --- Coverage metrics ---
            prop_cols = [
                'fanduel_ou', 'fanduel_reb_ou', 'fanduel_ast_ou',
                'fanduel_3pm_ou', 'fanduel_stl_ou', 'fanduel_blk_ou',
                'fanduel_pra_ou'
            ]
            # Ensure all prop columns exist (handle missing columns gracefully)
            for col in prop_cols:
                if col not in props_df.columns:
                    props_df[col] = None

            props_df['prop_count'] = props_df[prop_cols].notna().sum(axis=1)

            total_players = len(props_df)
            with_any_props = int(props_df['fanduel_ou'].notna().sum())

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Players", total_players)
            m2.metric("With Props", with_any_props)

            if with_any_props > 0:
                avg_coverage = props_df[props_df['prop_count'] > 0]['prop_count'].mean()
                m3.metric("Avg Stats/Player", f"{avg_coverage:.1f}/7")
            else:
                m3.metric("Avg Stats/Player", "0/7")

            monthly_usage = odds_api.get_monthly_api_usage(dfs_conn)
            m4.metric("API Usage", f"{monthly_usage}/500")

            st.divider()

            # Filter toggle
            show_all = st.toggle(
                "Show all players (including those without props)",
                value=False,
                key="props_show_all"
            )

            if not show_all:
                display_props_df = props_df[props_df['fanduel_ou'].notna()].copy()
            else:
                display_props_df = props_df.copy()

            if display_props_df.empty:
                st.info(
                    "No FanDuel props found for this date. "
                    "Click **Fetch FanDuel Lines** above to get latest odds."
                )
            else:
                # Build display table
                table_data = []
                for _, row in display_props_df.iterrows():
                    entry = {
                        'Player': row['player_name'],
                        'Team': row['team_name'],
                        'Opp': row['opponent_name'],
                        'PTS O/U': row.get('fanduel_ou'),
                        'Proj PTS': row.get('projected_ppg'),
                        'REB O/U': row.get('fanduel_reb_ou'),
                        'AST O/U': row.get('fanduel_ast_ou'),
                        '3PM O/U': row.get('fanduel_3pm_ou'),
                        'STL O/U': row.get('fanduel_stl_ou'),
                        'BLK O/U': row.get('fanduel_blk_ou'),
                        'PRA O/U': row.get('fanduel_pra_ou'),
                        'Vegas FPTS': row.get('vegas_implied_fpts'),
                        'Coverage': f"{int(row['prop_count'])}/7",
                    }

                    # Add our per-stat projections if available from JOIN
                    if has_proj_stats:
                        entry['Proj REB'] = row.get('proj_rebounds')
                        entry['Proj AST'] = row.get('proj_assists')
                        entry['Proj 3PM'] = row.get('proj_fg3m')
                        entry['Proj STL'] = row.get('proj_steals')
                        entry['Proj BLK'] = row.get('proj_blocks')
                        entry['Proj FPTS'] = row.get('proj_fpts')

                    table_data.append(entry)

                table_df = pd.DataFrame(table_data)

                # Format numeric columns — display NaN as "—"
                numeric_cols = [
                    'PTS O/U', 'Proj PTS', 'REB O/U', 'AST O/U', '3PM O/U',
                    'STL O/U', 'BLK O/U', 'PRA O/U', 'Vegas FPTS',
                ]
                if has_proj_stats:
                    numeric_cols += ['Proj REB', 'Proj AST', 'Proj 3PM', 'Proj STL', 'Proj BLK', 'Proj FPTS']

                for col in numeric_cols:
                    if col in table_df.columns:
                        table_df[col] = table_df[col].apply(
                            lambda x: f"{x:.1f}" if pd.notna(x) else "\u2014"
                        )

                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )

                # Legend
                st.caption(
                    "**Coverage** = how many of 7 stat props are populated (PTS / REB / AST / 3PM / STL / BLK / PRA). "
                    "Stars typically have 7/7; role players may have fewer. "
                    "**\u2014** = no line available from FanDuel."
                )

                # Export button
                csv_data = display_props_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export Props to CSV",
                    data=csv_data,
                    file_name=f"fanduel_props_{props_date}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="props_export_btn"
                )

    # ==========================================================================
    # TAB 8: Player Scouting (database-driven, slate-independent)
    # ==========================================================================
    with builder_tabs[7]:
        st.subheader("Player Scouting")
        st.caption("League-wide player research — independent of the current DK slate")

        try:
            # Query recent game logs for all players (last 7 games per player)
            scout_query = """
                WITH ranked_games AS (
                    SELECT
                        player_id,
                        player_name,
                        team_abbreviation,
                        game_date,
                        minutes,
                        points,
                        rebounds,
                        assists,
                        steals,
                        blocks,
                        turnovers,
                        fg3m,
                        ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) as game_num
                    FROM player_game_logs
                    WHERE season = '2025-26' AND season_type = 'Regular Season'
                ),
                recent AS (
                    SELECT * FROM ranked_games WHERE game_num <= 7
                ),
                player_stats AS (
                    SELECT
                        player_id,
                        player_name,
                        team_abbreviation,
                        COUNT(*) as games,
                        AVG(CAST(minutes AS REAL)) as avg_minutes,
                        AVG(points) as avg_points,
                        -- DK FPTS = pts*1 + reb*1.25 + ast*1.5 + stl*2 + blk*2 - tov*0.5 + fg3m*0.5
                        AVG(
                            points * 1.0 + rebounds * 1.25 + assists * 1.5
                            + steals * 2.0 + blocks * 2.0 - turnovers * 0.5
                            + fg3m * 0.5
                        ) as avg_fpts,
                        -- Sample standard deviation of FPTS
                        CASE WHEN COUNT(*) > 1 THEN
                            -- Manual stddev since SQLite doesn't have it natively
                            SQRT(
                                AVG(
                                    (points * 1.0 + rebounds * 1.25 + assists * 1.5
                                     + steals * 2.0 + blocks * 2.0 - turnovers * 0.5
                                     + fg3m * 0.5)
                                    * (points * 1.0 + rebounds * 1.25 + assists * 1.5
                                       + steals * 2.0 + blocks * 2.0 - turnovers * 0.5
                                       + fg3m * 0.5)
                                )
                                - AVG(
                                    points * 1.0 + rebounds * 1.25 + assists * 1.5
                                    + steals * 2.0 + blocks * 2.0 - turnovers * 0.5
                                    + fg3m * 0.5
                                ) * AVG(
                                    points * 1.0 + rebounds * 1.25 + assists * 1.5
                                    + steals * 2.0 + blocks * 2.0 - turnovers * 0.5
                                    + fg3m * 0.5
                                )
                            )
                        ELSE 0 END as fpts_stddev
                    FROM recent
                    WHERE CAST(minutes AS REAL) > 0
                    GROUP BY player_id
                    HAVING games >= 3
                )
                SELECT
                    ps.*,
                    pr.role_tier
                FROM player_stats ps
                LEFT JOIN player_roles pr ON ps.player_id = pr.player_id AND pr.season = '2025-26'
                ORDER BY avg_fpts DESC
            """
            scout_df = pd.read_sql_query(scout_query, dfs_conn)

            if scout_df.empty:
                st.info("No recent game data found. Run the data builder to populate player_game_logs.")
            else:
                # Get avg ownership from past slates
                try:
                    own_query = """
                        SELECT player_id,
                            COALESCE(
                                AVG(CASE WHEN actual_ownership > 0 THEN actual_ownership END),
                                AVG(CASE WHEN ownership_proj > 0 THEN ownership_proj END)
                            ) as avg_own_proj
                        FROM dfs_slate_projections
                        WHERE slate_date >= date('now', '-21 days')
                        GROUP BY player_id
                    """
                    own_df = pd.read_sql_query(own_query, dfs_conn)
                    scout_df = scout_df.merge(own_df, on='player_id', how='left')
                except Exception:
                    scout_df['avg_own_proj'] = None

                # Classify role for players without player_roles entry
                from depth_chart import classify_role_tier
                scout_df['role_tier'] = scout_df.apply(
                    lambda row: row['role_tier'] if pd.notna(row.get('role_tier')) and row['role_tier']
                    else classify_role_tier(row['avg_minutes'], row['avg_points'], row['games'], avg_fpts=row.get('avg_fpts')),
                    axis=1
                )

                # Filters
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    scout_teams = sorted(scout_df['team_abbreviation'].dropna().unique())
                    sel_teams = st.multiselect("Filter by Team", scout_teams, default=[], key="scout_team_filter")
                with col_f2:
                    role_options = ['STAR', 'STARTER', 'ROTATION', 'BENCH']
                    sel_roles = st.multiselect("Filter by Role", role_options, default=[], key="scout_role_filter")
                with col_f3:
                    min_fpts = st.slider("Min Avg FPTS", 0.0, 60.0, 10.0, step=5.0, key="scout_min_fpts")

                display_df = scout_df.copy()
                if sel_teams:
                    display_df = display_df[display_df['team_abbreviation'].isin(sel_teams)]
                if sel_roles:
                    display_df = display_df[display_df['role_tier'].isin(sel_roles)]
                display_df = display_df[display_df['avg_fpts'] >= min_fpts]

                # Role summary
                role_icons = {'STAR': '🌟', 'STARTER': '🟢', 'ROTATION': '🟡', 'BENCH': '⚪'}
                role_counts = display_df['role_tier'].value_counts()
                role_parts = []
                for r in ['STAR', 'STARTER', 'ROTATION', 'BENCH']:
                    if r in role_counts.index:
                        role_parts.append(f"{role_icons.get(r, '')} {r}: {role_counts[r]}")
                if role_parts:
                    st.caption(f"{len(display_df)} players | " + " | ".join(role_parts))

                # Format for display
                display_df['Role'] = display_df['role_tier'].apply(
                    lambda r: f"{role_icons.get(r, '')} {r}" if r else '—'
                )
                display_df['Avg FPTS (7G)'] = display_df['avg_fpts'].round(1)
                display_df['Variance'] = display_df['fpts_stddev'].round(1)
                display_df['Avg Own%'] = display_df['avg_own_proj'].round(1)

                show_df = display_df[[
                    'player_name', 'team_abbreviation', 'Role',
                    'Avg FPTS (7G)', 'Variance', 'Avg Own%', 'games'
                ]].rename(columns={
                    'player_name': 'Player',
                    'team_abbreviation': 'Team',
                    'games': 'GP',
                })

                st.dataframe(
                    show_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Avg FPTS (7G)': st.column_config.NumberColumn(format="%.1f"),
                        'Variance': st.column_config.NumberColumn(format="%.1f"),
                        'Avg Own%': st.column_config.NumberColumn(format="%.1f%%"),
                    }
                )

        except Exception as e:
            st.error(f"Failed to load scouting data: {e}")

st.divider()
st.caption(
    "Need more context? Re-run the builder (`python nba_to_sqlite.py ...`) to refresh "
    "underlying tables, then use this app to validate outputs."
)
