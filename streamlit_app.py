"""
Streamlit dashboard for exploring locally built NBA stats.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for Windows builds without tzdata
    ZoneInfo = None

from nba_api.stats.endpoints import scoreboardv2

from nba_to_sqlite import build_database

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


@st.cache_resource
def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
) -> pd.DataFrame:
    query = """
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
    df = run_query(db_path, query, params=(season, season_type))
    if df.empty:
        return df
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
    df["allowed_pts"] = pd.to_numeric(df["allowed_pts"], errors="coerce")
    for col in ["allowed_fg3m", "allowed_fg3a", "allowed_reb", "allowed_ast"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["team_id", "allowed_pts"])
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
        )
        .reset_index()
    )

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
    # Percentiles for style classification
    for col in ["avg_allowed_pts", "avg_allowed_fg3m", "avg_allowed_reb"]:
        if col not in aggregates.columns:
            aggregates[col] = None
        aggregates[f"{col}_pct"] = aggregates[col].fillna(0.0).rank(pct=True)

    def classify_style(row: pd.Series) -> str:
        fg3m_pct = safe_float(row.get("avg_allowed_fg3m_pct"))
        reb_pct = safe_float(row.get("avg_allowed_reb_pct"))
        pts_pct = safe_float(row.get("avg_allowed_pts_pct"))
        if fg3m_pct is not None and fg3m_pct >= 0.75:
            return "Perimeter Leak"
        if reb_pct is not None and reb_pct >= 0.75:
            return "Board-Soft"
        if (
            fg3m_pct is not None
            and pts_pct is not None
            and fg3m_pct <= 0.30
            and pts_pct <= 0.30
        ):
            return "Clamp"
        return "Neutral"

    aggregates["defense_style"] = aggregates.apply(classify_style, axis=1)
    aggregates["defense_style"] = aggregates["defense_style"].replace("", "Neutral")
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
    if df.empty:
        return {}
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["opp_team_id"] = pd.to_numeric(df["opp_team_id"], errors="coerce")
    df = df.dropna(subset=["player_id", "points", "opp_team_id"])
    df["defense_style"] = df["opp_team_id"].map(style_map).fillna("Neutral")
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
        return pd.DataFrame()

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
    "Standings",
    "3PT Leaders",
    "Scoring Leaders",
    "3PT Defense",
    "Points Allowed",
    "Defense Mix",
    "Prediction Log",
    "Defense Styles",
]
tabs = st.tabs(tab_titles)
(
    games_tab,
    matchup_spotlight_tab,
    daily_leaders_tab,
    standings_tab,
    three_pt_tab,
    scoring_tab,
    defense_3pt_tab,
    defense_pts_tab,
    defense_mix_tab,
    predictions_tab,
    defense_styles_tab,
) = tabs
defense_style_tab = st.tabs(["Defense Styles"])[0]

matchup_spotlight_rows: list[Dict[str, Any]] = []
daily_power_rows_points: list[Dict[str, Any]] = []
daily_power_rows_3pm: list[Dict[str, Any]] = []
daily_top_scorers_rows: list[Dict[str, Any]] = []
player_season_stats_map: Dict[int, Mapping[str, Any]] = {}

# Today's games tab --------------------------------------------------------
with games_tab:
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
                            avg_vs_style = None
                            player_id_val = safe_int(player.get("player_id"))
                            if player_id_val is not None:
                                avg_vs_style = player_style_splits.get(player_id_val, {}).get(
                                    opp_style
                                )
                            avg_vs_style_filled = (
                                avg_vs_style if avg_vs_style is not None else season_avg_pts
                            )
                            avg_vs_style_display = (
                                format_number(avg_vs_style_filled, 1)
                                if avg_vs_style_filled is not None
                                else "N/A"
                            )
                            matchup_rows.append(
                                {
                                    "Side": team_label,
                                    "Team": team_name,
                                    "Player": player["player_name"],
                                    "Games": int(player["games_played"]),
                                    "Avg PPG": f"{player['avg_points']:.1f}",
                                    "Median PPG": f"{player['median_points']:.1f}",
                                    "Max PPG": f"{player['max_points']:.1f}",
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
                                    "Avg Pts vs Opp Style": avg_vs_style_display,
                                }
                            )
                            matchup_spotlight_rows.append(
                                {
                                    "Matchup": f"{matchup['Away']} at {matchup['Home']}",
                                    "Side": team_label,
                                    "Team": team_name,
                                    "Player": player["player_name"],
                                    "Season Avg PPG": season_avg_pts,
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
                                    "Avg Pts vs Opp Style": avg_vs_style_display,
                                    "Matchup Score": matchup_score,
                                }
                            )
                            daily_power_rows_points.append(
                                {
                                    "Matchup": f"{matchup['Away']} at {matchup['Home']}",
                                    "Player": player["player_name"],
                                    "Team": team_name,
                                    "Season Avg PPG": season_avg_pts,
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
                        st.dataframe(matchup_df, use_container_width=True)
                    else:
                        st.caption("No qualified players for this matchup yet.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to load today's games: {exc}")

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
            points_top = points_df.sort_values("Matchup Score", ascending=False).head(10)
            points_top = points_top.assign(
                **{
                    "Opp Avg Allowed PPG": points_top["Opp Avg Allowed PPG"].map(lambda v: format_number(v, 1)),
                    "Opp Last5 Avg Allowed": points_top["Opp Last5 Avg Allowed"].map(lambda v: format_number(v, 1)),
                    "Opportunity Index": points_top["Opportunity Index"].map(lambda v: format_number(v, 2)),
                    "Opp Def Composite": points_top["Opp Def Composite"].map(lambda v: format_number(v, 1)),
                    "Usage %": points_top["Usage %"].map(lambda v: format_number(v, 1)),
                }
            )
            col_points.markdown("**Top 10 Players by Matchup Score (Points)**")
            col_points.dataframe(points_top, use_container_width=True)
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
                "defense_style",
                "avg_allowed_pts",
                "avg_allowed_fg3m",
                "avg_allowed_reb",
                "def_composite_score",
            ]
            rename_map = {
                "full_name": "Team",
                "defense_style": "Defense Style",
                "avg_allowed_pts": "Avg Pts Allowed",
                "avg_allowed_fg3m": "Avg 3PM Allowed",
                "avg_allowed_reb": "Avg Reb Allowed",
                "def_composite_score": "Def Composite",
            }
            display_df = styles_df[display_cols].rename(columns=rename_map)
            st.dataframe(
                display_df.sort_values("Defense Style"),
                use_container_width=True,
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to load defense styles: {exc}")

# Prediction tab -----------------------------------------------------------
with predictions_tab:
    st.subheader("3PT Leader Predictions")
    predictions_path = Path(st.session_state["predictions_path_input"]).expanduser()
    if not predictions_path.exists():
        st.info("No predictions CSV detected. Run `predict_top_3pm.py` with `--output-predictions`.")
    else:
        try:
            predictions_df = pd.read_csv(predictions_path)
            sort_col = st.selectbox(
                "Sort by",
                options=["pred_prob", "correct", "season", "team_id"],
                index=0,
            )
            ascending = st.checkbox("Sort ascending", value=False)
            display_df = predictions_df.sort_values(sort_col, ascending=ascending)
            render_dataframe(display_df)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Unable to read predictions CSV: {exc}")

st.divider()
st.caption(
    "Need more context? Re-run the builder (`python nba_to_sqlite.py ...`) to refresh "
    "underlying tables, then use this app to validate outputs."
)
