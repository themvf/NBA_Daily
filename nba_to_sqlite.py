#!/usr/bin/env python3
"""
Builds a SQLite database with up-to-date NBA reference data fetched through
https://github.com/swar/nba-api.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashplayerstats,
    leaguestandings,
    playergamelogs,
    teamgamelogs,
)
from nba_api.stats.static import players as players_api
from nba_api.stats.static import teams as teams_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an NBA SQLite database.")
    parser.add_argument(
        "--db-path",
        default="nba_stats.db",
        help="Destination SQLite file (default: nba_stats.db).",
    )
    parser.add_argument(
        "--season",
        default="2023-24",
        help="Season label to use for standings/rosters calls (e.g. 2023-24).",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="NBA stats season type (Regular Season, Playoffs, etc.).",
    )
    parser.add_argument(
        "--include-rosters",
        dest="include_rosters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle fetching team rosters (default: on).",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.6,
        help="Delay between roster API calls to avoid rate limiting.",
    )
    parser.add_argument(
        "--shooting-season",
        default=None,
        help="Season label used for player shooting totals (default: --season).",
    )
    parser.add_argument(
        "--shooting-season-type",
        default=None,
        help="Season type used for player shooting totals (default: --season-type).",
    )
    parser.add_argument(
        "--top-3pt-view-season",
        default="2024-25",
        help=(
            "Season label captured in the players_2025_top_3pt view "
            "(default: 2024-25)."
        ),
    )
    parser.add_argument(
        "--top-3pt-view-season-type",
        default=None,
        help=(
            "Season type captured in the players_2025_top_3pt view "
            "(default: shooting season type)."
        ),
    )
    parser.add_argument(
        "--defense-view-season",
        default="2024-25",
        help=(
            "Season label captured in the teams_2025_defense_3pt view "
            "(default: 2024-25)."
        ),
    )
    parser.add_argument(
        "--defense-view-season-type",
        default=None,
        help=(
            "Season type captured in the teams_2025_defense_3pt view "
            "(default: shooting season type)."
        ),
    )
    parser.add_argument(
        "--top-pts-view-season",
        default="2024-25",
        help=(
            "Season label captured in the players_2025_top_pts view "
            "(default: 2024-25)."
        ),
    )
    parser.add_argument(
        "--top-pts-view-season-type",
        default=None,
        help=(
            "Season type captured in the players_2025_top_pts view "
            "(default: shooting season type)."
        ),
    )
    parser.add_argument(
        "--defense-pts-view-season",
        default="2024-25",
        help=(
            "Season label captured in the teams_2025_defense_pts view "
            "(default: 2024-25)."
        ),
    )
    parser.add_argument(
        "--defense-pts-view-season-type",
        default=None,
        help=(
            "Season type captured in the teams_2025_defense_pts view "
            "(default: shooting season type)."
        ),
    )
    parser.add_argument(
        "--defense-mix-view-season",
        default="2025-26",
        help=(
            "Season label captured in the teams_2026_defense_mix view "
            "(default: 2025-26)."
        ),
    )
    parser.add_argument(
        "--defense-mix-view-season-type",
        default=None,
        help=(
            "Season type captured in the teams_2026_defense_mix view "
            "(default: mix view season type)."
        ),
    )
    return parser.parse_args()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            abbreviation TEXT,
            nickname TEXT,
            city TEXT,
            state TEXT,
            year_founded INTEGER,
            is_nba_team INTEGER NOT NULL DEFAULT 1,
            is_nba_franchise INTEGER NOT NULL DEFAULT 1,
            is_all_star INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            is_active INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS standings (
            season TEXT NOT NULL,
            season_id TEXT,
            season_type TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            conference TEXT,
            wins INTEGER,
            losses INTEGER,
            win_pct REAL,
            home_record TEXT,
            road_record TEXT,
            last_ten TEXT,
            streak TEXT,
            clinch_code TEXT,
            conference_rank INTEGER,
            division_rank INTEGER,
            PRIMARY KEY (season, season_type, team_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS team_rosters (
            season TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            player TEXT NOT NULL,
            jersey TEXT,
            position TEXT,
            height TEXT,
            weight TEXT,
            birth_date TEXT,
            age REAL,
            experience TEXT,
            school TEXT,
            PRIMARY KEY (season, team_id, player_id),
            FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE,
            FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS player_season_totals (
            season TEXT NOT NULL,
            season_type TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_id INTEGER,
            team_abbreviation TEXT,
            team_name TEXT,
            games_played INTEGER,
            minutes REAL,
            fgm REAL,
            fga REAL,
            fg_pct REAL,
            fg3m REAL,
            fg3a REAL,
            fg3_pct REAL,
            ftm REAL,
            fta REAL,
            ft_pct REAL,
            rebounds REAL,
            assists REAL,
            points REAL,
            usg_pct REAL,
            PRIMARY KEY (season, season_type, player_id),
            FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE,
            FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS player_game_logs (
            season TEXT NOT NULL,
            season_type TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_id INTEGER,
            team_abbreviation TEXT,
            team_name TEXT,
            game_id TEXT NOT NULL,
            game_date TEXT,
            matchup TEXT,
            fg3m REAL,
            fg3a REAL,
            minutes TEXT,
            usg_pct REAL,
            points REAL,
            PRIMARY KEY (season, season_type, player_id, game_id)
        );

        CREATE TABLE IF NOT EXISTS team_game_logs (
            season TEXT NOT NULL,
            season_type TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            team_name TEXT NOT NULL,
            team_abbreviation TEXT,
            game_id TEXT NOT NULL,
            game_date TEXT,
            matchup TEXT,
            opp_team_id INTEGER,
            opp_team_name TEXT,
            opp_team_abbreviation TEXT,
            fg3m REAL,
            fg3a REAL,
            opp_fg3m REAL,
            opp_fg3a REAL,
            pts REAL,
            opp_pts REAL,
            ast REAL,
            reb REAL,
            opp_ast REAL,
            opp_reb REAL,
            PRIMARY KEY (season, season_type, team_id, game_id)
        );
        """
    )
    alter_columns = [
        ("team_game_logs", "ast", "REAL"),
        ("team_game_logs", "reb", "REAL"),
        ("team_game_logs", "opp_ast", "REAL"),
        ("team_game_logs", "opp_reb", "REAL"),
        ("player_game_logs", "usg_pct", "REAL"),
    ]
    for table, column, col_type in alter_columns:
        try:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                continue
            raise
    try:
        conn.execute(
            "ALTER TABLE player_season_totals ADD COLUMN usg_pct REAL"
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise
    conn.commit()


def lower_keys(record: Mapping) -> Dict[str, object]:
    return {str(key).lower(): value for key, value in record.items()}


def safe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def split_full_name(full_name: str | None) -> tuple[str | None, str | None]:
    if not full_name:
        return None, None
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def bool_to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"y", "yes", "true", "1"}:
            return 1
        if value in {"n", "no", "false", "0"}:
            return 0
    return 0


def upsert_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: Sequence[Mapping[str, object]],
    conflict_cols: Sequence[str],
) -> int:
    if not rows:
        return 0
    column_names = list(rows[0].keys())
    columns_sql = ", ".join(column_names)
    placeholders = ", ".join(f":{col}" for col in column_names)
    sql = f"INSERT INTO {table} ({columns_sql}) VALUES ({placeholders})"
    if conflict_cols:
        updates = ", ".join(
            f"{col}=excluded.{col}"
            for col in column_names
            if col not in conflict_cols
        )
        if updates:
            sql += f" ON CONFLICT({', '.join(conflict_cols)}) DO UPDATE SET {updates}"
        else:
            sql += f" ON CONFLICT({', '.join(conflict_cols)}) DO NOTHING"
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def load_teams(conn: sqlite3.Connection) -> int:
    raw_teams = teams_api.get_teams()
    rows: List[Dict[str, object]] = []
    for record in raw_teams:
        data = lower_keys(record)
        rows.append(
            {
                "team_id": data.get("id"),
                "full_name": data.get("full_name"),
                "abbreviation": data.get("abbreviation"),
                "nickname": data.get("nickname"),
                "city": data.get("city"),
                "state": data.get("state"),
                "year_founded": data.get("year_founded"),
                "is_nba_team": bool_to_int(data.get("is_nba_team", 1)),
                "is_nba_franchise": bool_to_int(data.get("is_nba_franchise", 1)),
                "is_all_star": bool_to_int(data.get("is_all_star", 0)),
            }
        )
    return upsert_rows(conn, "teams", rows, ["team_id"])


def load_players(conn: sqlite3.Connection) -> int:
    raw_players = players_api.get_players()
    rows: List[Dict[str, object]] = []
    for record in raw_players:
        data = lower_keys(record)
        rows.append(
            {
                "player_id": data.get("id"),
                "full_name": data.get("full_name"),
                "first_name": data.get("first_name"),
                "last_name": data.get("last_name"),
                "is_active": bool_to_int(data.get("is_active")),
            }
        )
    return upsert_rows(conn, "players", rows, ["player_id"])


def fetch_league_standings(season: str, season_type: str) -> List[Dict[str, object]]:
    endpoint = leaguestandings.LeagueStandings(
        season=season,
        season_type=season_type,
    )
    df: pd.DataFrame = endpoint.get_data_frames()[0]
    records = df.to_dict(orient="records")
    normalized: List[Dict[str, object]] = []
    for record in records:
        data = lower_keys(record)
        team_city = data.get("teamcity") or ""
        team_name = data.get("teamname") or data.get("team") or ""
        normalized.append(
            {
                "season": season,
                "season_id": data.get("seasonid"),
                "season_type": season_type,
                "team_id": data.get("teamid"),
                "team_name": f"{team_city} {team_name}".strip(),
                "conference": data.get("conference"),
                "wins": data.get("w") or data.get("wins"),
                "losses": data.get("l") or data.get("losses"),
                "win_pct": data.get("winpct"),
                "home_record": data.get("homerecord"),
                "road_record": data.get("roadrecord"),
                "last_ten": data.get("lastten"),
                "streak": data.get("currentstreak") or data.get("streak"),
                "clinch_code": data.get("clinchcode"),
                "conference_rank": data.get("conferencerank"),
                "division_rank": data.get("divisionrank"),
            }
        )
    return normalized


def load_standings(
    conn: sqlite3.Connection, season: str, season_type: str
) -> int:
    rows = fetch_league_standings(season, season_type)
    return upsert_rows(
        conn,
        "standings",
        rows,
        ["season", "season_type", "team_id"],
    )


def load_player_shooting_totals(
    conn: sqlite3.Connection,
    season: str,
    season_type: str,
) -> int:
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed="Totals",
    )
    df: pd.DataFrame = endpoint.get_data_frames()[0]
    records = df.to_dict(orient="records")
    rows: List[Dict[str, object]] = []
    for record in records:
        data = lower_keys(record)
        rows.append(
            {
                "season": season,
                "season_type": season_type,
                "player_id": data.get("player_id"),
                "player_name": data.get("player_name"),
                "team_id": data.get("team_id"),
                "team_abbreviation": data.get("team_abbreviation"),
                "team_name": data.get("team_name"),
                "games_played": data.get("gp"),
                "minutes": data.get("min"),
                "fgm": data.get("fgm"),
                "fga": data.get("fga"),
                "fg_pct": data.get("fg_pct"),
                "fg3m": data.get("fg3m"),
                "fg3a": data.get("fg3a"),
                "fg3_pct": data.get("fg3_pct"),
                "ftm": data.get("ftm"),
                "fta": data.get("fta"),
                "ft_pct": data.get("ft_pct"),
                "rebounds": data.get("reb"),
                "assists": data.get("ast"),
                "points": data.get("pts"),
            }
        )
    return upsert_rows(
        conn,
        "player_season_totals",
        rows,
        ["season", "season_type", "player_id"],
    )


def update_player_usage(
    conn: sqlite3.Connection,
    season: str,
    season_type: str,
) -> int:
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
    )
    df: pd.DataFrame = endpoint.get_data_frames()[0]
    records = df.to_dict(orient="records")
    updates: List[tuple[float | None, int, str, str]] = []
    for record in records:
        data = lower_keys(record)
        player_id = safe_int(data.get("player_id"))
        if player_id is None:
            continue
        usg_pct = normalize_float(data.get("usg_pct"))
        updates.append((usg_pct, player_id, season, season_type))
    if not updates:
        return 0
    conn.executemany(
        """
        UPDATE player_season_totals
        SET usg_pct = ?
        WHERE player_id = ?
          AND season = ?
          AND season_type = ?
        """,
        updates,
    )
    conn.commit()
    return len(updates)


def load_player_game_logs(
    conn: sqlite3.Connection,
    season: str,
    season_type: str,
) -> int:
    endpoint = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        player_id_nullable=None,
    )
    df: pd.DataFrame = endpoint.get_data_frames()[0]
    records = df.to_dict(orient="records")
    rows: List[Dict[str, object]] = []
    for record in records:
        data = lower_keys(record)
        player_id = safe_int(data.get("player_id") or data.get("playerid"))
        game_id = data.get("game_id")
        if player_id is None or not game_id:
            continue
        rows.append(
            {
                "season": season,
                "season_type": season_type,
                "player_id": player_id,
                "player_name": data.get("player_name") or "Unknown Player",
                "team_id": data.get("team_id"),
                "team_abbreviation": data.get("team_abbreviation"),
                "team_name": data.get("team_name"),
                "game_id": game_id,
                "game_date": data.get("game_date"),
                "matchup": data.get("matchup"),
                "fg3m": data.get("fg3m"),
                "fg3a": data.get("fg3a"),
                "minutes": data.get("min"),
                "usg_pct": normalize_float(data.get("usg_pct")),
                "points": data.get("pts"),
            }
        )
    return upsert_rows(
        conn,
        "player_game_logs",
        rows,
        ["season", "season_type", "player_id", "game_id"],
    )


def load_team_game_logs(
    conn: sqlite3.Connection,
    season: str,
    season_type: str,
) -> int:
    endpoint = teamgamelogs.TeamGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",
    )
    df: pd.DataFrame = endpoint.get_data_frames()[0]
    records = df.to_dict(orient="records")
    rows: List[Dict[str, object]] = []
    for record in records:
        data = lower_keys(record)
        team_id = safe_int(data.get("team_id") or data.get("teamid"))
        game_id = data.get("game_id")
        if team_id is None or not game_id:
            continue
        rows.append(
            {
                "season": season,
                "season_type": season_type,
                "team_id": team_id,
                "team_name": data.get("team_name") or "Unknown Team",
                "team_abbreviation": data.get("team_abbreviation"),
                "game_id": game_id,
                "game_date": data.get("game_date"),
                "matchup": data.get("matchup"),
                "opp_team_id": safe_int(
                    data.get("opp_team_id") or data.get("opp_teamid")
                ),
                "opp_team_name": data.get("opp_team_name"),
                "opp_team_abbreviation": data.get("opp_team_abbreviation"),
                "fg3m": data.get("fg3m"),
                "fg3a": data.get("fg3a"),
                "opp_fg3m": data.get("opp_fg3m"),
                "opp_fg3a": data.get("opp_fg3a"),
                "pts": data.get("pts"),
                "opp_pts": data.get("opp_pts"),
                "ast": data.get("ast"),
                "reb": data.get("reb"),
                "opp_ast": data.get("opp_ast"),
                "opp_reb": data.get("opp_reb"),
            }
        )
    return upsert_rows(
        conn,
        "team_game_logs",
        rows,
        ["season", "season_type", "team_id", "game_id"],
    )


def normalize_number(value: object) -> str | None:
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


def normalize_float(value: object) -> float | None:
    if value in (None, "", "-", " "):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_team_roster(team_id: int, season: str) -> List[Dict[str, object]]:
    endpoint = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    df: pd.DataFrame = endpoint.get_data_frames()[0]
    records = df.to_dict(orient="records")
    formatted: List[Dict[str, object]] = []
    for record in records:
        data = lower_keys(record)
        formatted.append(
            {
                "season": season,
                "team_id": data.get("teamid") or team_id,
                "player_id": data.get("player_id") or data.get("playerid"),
                "player": data.get("player"),
                "jersey": normalize_number(data.get("num")),
                "position": data.get("position"),
                "height": data.get("height"),
                "weight": data.get("weight"),
                "birth_date": data.get("birth_date"),
                "age": normalize_float(data.get("age")),
                "experience": data.get("exp"),
                "school": data.get("school"),
            }
        )
    return formatted


def ensure_roster_players_exist(
    conn: sqlite3.Connection, roster_rows: Sequence[Mapping[str, object]]
) -> int:
    player_ids = {
        row.get("player_id")
        for row in roster_rows
        if row.get("player_id") is not None
    }
    player_ids.discard(None)
    if not player_ids:
        return 0
    placeholders = ",".join("?" for _ in player_ids)
    existing = set()
    cursor = conn.execute(
        f"SELECT player_id FROM players WHERE player_id IN ({placeholders})",
        tuple(player_ids),
    )
    existing.update(row[0] for row in cursor.fetchall())
    missing_ids = [pid for pid in player_ids if pid not in existing]
    if not missing_ids:
        return 0
    rows: List[Dict[str, object]] = []
    id_set = set(missing_ids)
    for row in roster_rows:
        pid = row.get("player_id")
        if pid not in id_set:
            continue
        first_name, last_name = split_full_name(row.get("player"))
        rows.append(
            {
                "player_id": pid,
                "full_name": row.get("player"),
                "first_name": first_name,
                "last_name": last_name,
                "is_active": 1,
            }
        )
    if not rows:
        return 0
    return upsert_rows(conn, "players", rows, ["player_id"])


def load_team_rosters(
    conn: sqlite3.Connection,
    season: str,
    throttle: float,
) -> int:
    cursor = conn.execute(
        "SELECT team_id FROM teams WHERE is_nba_team = 1 ORDER BY team_id"
    )
    team_ids = [row[0] for row in cursor.fetchall()]
    inserted = 0
    for idx, team_id in enumerate(team_ids, 1):
        try:
            roster_rows = fetch_team_roster(team_id, season)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to fetch roster for team {team_id}: {exc}")
            continue
        if roster_rows:
            ensure_roster_players_exist(conn, roster_rows)
            inserted += upsert_rows(
                conn,
                "team_rosters",
                roster_rows,
                ["season", "team_id", "player_id"],
            )
        if throttle > 0 and idx < len(team_ids):
            time.sleep(throttle)
    return inserted


def quote_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def create_three_point_view(
    conn: sqlite3.Connection, season: str, season_type: str
) -> None:
    conn.execute("DROP VIEW IF EXISTS players_2025_top_3pt")
    season_literal = quote_sql_literal(season)
    season_type_literal = quote_sql_literal(season_type)
    conn.execute(
        f"""
        CREATE VIEW players_2025_top_3pt AS
        WITH totals AS (
            SELECT
                player_id,
                player_name,
                team_id,
                team_abbreviation,
                team_name,
                games_played,
                fg3m AS total_fg3m,
                fg3a AS total_fg3a,
                fg3m / NULLIF(games_played, 0) AS avg_fg3m_per_game,
                fg3a / NULLIF(games_played, 0) AS avg_fg3a_per_game
            FROM player_season_totals
            WHERE season = '{season_literal}'
              AND season_type = '{season_type_literal}'
              AND games_played > 0
        ),
        median AS (
            SELECT
                season,
                season_type,
                player_id,
                AVG(fg3m) AS median_fg3m_per_game
            FROM (
                SELECT
                    season,
                    season_type,
                    player_id,
                    fg3m,
                    ROW_NUMBER() OVER (
                        PARTITION BY season, season_type, player_id
                        ORDER BY fg3m
                    ) AS rn,
                    COUNT(*) OVER (
                        PARTITION BY season, season_type, player_id
                    ) AS cnt
                FROM player_game_logs
                WHERE season = '{season_literal}'
                  AND season_type = '{season_type_literal}'
            )
            WHERE rn IN (
                (cnt + 1) / 2,
                (cnt + 2) / 2
            )
            GROUP BY season, season_type, player_id
        ),
        max_per_game AS (
            SELECT
                season,
                season_type,
                player_id,
                MAX(fg3m) AS max_fg3m_per_game
            FROM player_game_logs
            WHERE season = '{season_literal}'
              AND season_type = '{season_type_literal}'
            GROUP BY season, season_type, player_id
        )
        SELECT
            t.player_id,
            t.player_name,
            t.team_id,
            t.team_abbreviation,
            t.team_name,
            t.total_fg3m,
            t.avg_fg3a_per_game,
            COALESCE(m.median_fg3m_per_game, t.avg_fg3m_per_game) AS median_fg3m_per_game,
            COALESCE(mx.max_fg3m_per_game, 0) AS max_fg3m_per_game,
            t.avg_fg3m_per_game,
            RANK() OVER (
                ORDER BY t.avg_fg3m_per_game DESC, t.player_name ASC
            ) AS rank_fg3m_per_game
        FROM totals AS t
        LEFT JOIN median AS m
            ON m.player_id = t.player_id
           AND m.season = '{season_literal}'
           AND m.season_type = '{season_type_literal}'
        LEFT JOIN max_per_game AS mx
            ON mx.player_id = t.player_id
           AND mx.season = '{season_literal}'
           AND mx.season_type = '{season_type_literal}'
        ORDER BY t.avg_fg3m_per_game DESC,
                 t.player_name ASC
        """
    )
    conn.commit()


def create_player_points_view(
    conn: sqlite3.Connection, season: str, season_type: str
) -> None:
    conn.execute("DROP VIEW IF EXISTS players_2025_top_pts")
    season_literal = quote_sql_literal(season)
    season_type_literal = quote_sql_literal(season_type)
    conn.execute(
        f"""
        CREATE VIEW players_2025_top_pts AS
        WITH totals AS (
            SELECT
                player_id,
                player_name,
                team_id,
                team_abbreviation,
                team_name,
                games_played,
                points AS total_points,
                points / NULLIF(games_played, 0) AS avg_points_per_game
            FROM player_season_totals
            WHERE season = '{season_literal}'
              AND season_type = '{season_type_literal}'
              AND games_played > 0
        ),
        median AS (
            SELECT
                season,
                season_type,
                player_id,
                AVG(points) AS median_points_per_game
            FROM (
                SELECT
                    season,
                    season_type,
                    player_id,
                    points,
                    ROW_NUMBER() OVER (
                        PARTITION BY season, season_type, player_id
                        ORDER BY points
                    ) AS rn,
                    COUNT(*) OVER (
                        PARTITION BY season, season_type, player_id
                    ) AS cnt
                FROM player_game_logs
                WHERE season = '{season_literal}'
                  AND season_type = '{season_type_literal}'
            )
            WHERE rn IN (
                (cnt + 1) / 2,
                (cnt + 2) / 2
            )
            GROUP BY season, season_type, player_id
        ),
        max_per_game AS (
            SELECT
                season,
                season_type,
                player_id,
                MAX(points) AS max_points_per_game
            FROM player_game_logs
            WHERE season = '{season_literal}'
              AND season_type = '{season_type_literal}'
            GROUP BY season, season_type, player_id
        )
        SELECT
            t.player_id,
            t.player_name,
            t.team_id,
            t.team_abbreviation,
            t.team_name,
            t.total_points,
            t.avg_points_per_game,
            COALESCE(m.median_points_per_game, t.avg_points_per_game) AS median_points_per_game,
            COALESCE(mx.max_points_per_game, 0) AS max_points_per_game,
            RANK() OVER (
                ORDER BY t.avg_points_per_game DESC, t.player_name ASC
            ) AS rank_points_per_game
        FROM totals AS t
        LEFT JOIN median AS m
            ON m.player_id = t.player_id
           AND m.season = '{season_literal}'
           AND m.season_type = '{season_type_literal}'
        LEFT JOIN max_per_game AS mx
            ON mx.player_id = t.player_id
           AND mx.season = '{season_literal}'
           AND mx.season_type = '{season_type_literal}'
        ORDER BY t.avg_points_per_game DESC,
                 t.player_name ASC
        """
    )
    conn.commit()


def create_team_defense_three_point_view(
    conn: sqlite3.Connection, season: str, season_type: str
) -> None:
    conn.execute("DROP VIEW IF EXISTS teams_2025_defense_3pt")
    season_literal = quote_sql_literal(season)
    season_type_literal = quote_sql_literal(season_type)
    conn.execute(
        f"""
        CREATE VIEW teams_2025_defense_3pt AS
        WITH paired AS (
            SELECT
                t.team_id,
                t.team_name,
                t.team_abbreviation,
                t.game_id,
                opp.fg3m AS allowed_fg3m,
                opp.fg3a AS allowed_fg3a
            FROM team_game_logs AS t
            JOIN team_game_logs AS opp
                ON opp.game_id = t.game_id
               AND opp.team_id <> t.team_id
            WHERE t.season = '{season_literal}'
              AND t.season_type = '{season_type_literal}'
        ),
        logs AS (
            SELECT
                team_id,
                MAX(team_name) AS team_name,
                MAX(team_abbreviation) AS team_abbreviation,
                COUNT(*) AS games_played,
                AVG(allowed_fg3m) AS avg_allowed_fg3m,
                AVG(allowed_fg3a) AS avg_allowed_fg3a
            FROM paired
            GROUP BY team_id
            HAVING COUNT(*) > 0
        ),
        median AS (
            SELECT
                team_id,
                AVG(allowed_fg3m) AS median_allowed_fg3m
            FROM (
                SELECT
                    team_id,
                    allowed_fg3m,
                    ROW_NUMBER() OVER (
                        PARTITION BY team_id
                        ORDER BY allowed_fg3m
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY team_id) AS cnt
                FROM paired
            )
            WHERE rn IN (
                (cnt + 1) / 2,
                (cnt + 2) / 2
            )
            GROUP BY team_id
        ),
        max_vals AS (
            SELECT
                team_id,
                MAX(allowed_fg3m) AS max_allowed_fg3m
            FROM paired
            GROUP BY team_id
        )
        SELECT
            l.team_id,
            l.team_name,
            l.team_abbreviation,
            l.games_played,
            l.avg_allowed_fg3a,
            COALESCE(m.median_allowed_fg3m, l.avg_allowed_fg3m) AS median_allowed_fg3m,
            COALESCE(mx.max_allowed_fg3m, 0) AS max_allowed_fg3m,
            l.avg_allowed_fg3m,
            RANK() OVER (
                ORDER BY l.avg_allowed_fg3m DESC, l.team_name ASC
            ) AS rank_avg_allowed_fg3m
        FROM logs AS l
        LEFT JOIN median AS m
            ON m.team_id = l.team_id
        LEFT JOIN max_vals AS mx
            ON mx.team_id = l.team_id
        ORDER BY l.avg_allowed_fg3m DESC,
                 l.team_name ASC
        """
    )
    conn.commit()


def create_team_defense_points_view(
    conn: sqlite3.Connection, season: str, season_type: str
) -> None:
    conn.execute("DROP VIEW IF EXISTS teams_2025_defense_pts")
    season_literal = quote_sql_literal(season)
    season_type_literal = quote_sql_literal(season_type)
    conn.execute(
        f"""
        CREATE VIEW teams_2025_defense_pts AS
        WITH paired AS (
            SELECT
                t.team_id,
                t.team_name,
                t.team_abbreviation,
                t.game_id,
                opp.pts AS allowed_pts
            FROM team_game_logs AS t
            JOIN team_game_logs AS opp
                ON opp.game_id = t.game_id
               AND opp.team_id <> t.team_id
            WHERE t.season = '{season_literal}'
              AND t.season_type = '{season_type_literal}'
        ),
        logs AS (
            SELECT
                team_id,
                MAX(team_name) AS team_name,
                MAX(team_abbreviation) AS team_abbreviation,
                COUNT(*) AS games_played,
                AVG(allowed_pts) AS avg_allowed_pts
            FROM paired
            GROUP BY team_id
            HAVING COUNT(*) > 0
        ),
        median AS (
            SELECT
                team_id,
                AVG(allowed_pts) AS median_allowed_pts
            FROM (
                SELECT
                    team_id,
                    allowed_pts,
                    ROW_NUMBER() OVER (
                        PARTITION BY team_id
                        ORDER BY allowed_pts
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY team_id) AS cnt
                FROM paired
            )
            WHERE rn IN (
                (cnt + 1) / 2,
                (cnt + 2) / 2
            )
            GROUP BY team_id
        ),
        max_vals AS (
            SELECT
                team_id,
                MAX(allowed_pts) AS max_allowed_pts
            FROM paired
            GROUP BY team_id
        )
        SELECT
            l.team_id,
            l.team_name,
            l.team_abbreviation,
            l.games_played,
            l.avg_allowed_pts,
            COALESCE(m.median_allowed_pts, l.avg_allowed_pts) AS median_allowed_pts,
            COALESCE(mx.max_allowed_pts, 0) AS max_allowed_pts,
            RANK() OVER (
                ORDER BY l.avg_allowed_pts DESC, l.team_name ASC
            ) AS rank_avg_allowed_pts
        FROM logs AS l
        LEFT JOIN median AS m
            ON m.team_id = l.team_id
        LEFT JOIN max_vals AS mx
            ON mx.team_id = l.team_id
        ORDER BY l.avg_allowed_pts DESC,
                 l.team_name ASC
        """
    )
    conn.commit()


def create_team_defense_mix_view(
    conn: sqlite3.Connection, season: str, season_type: str
) -> None:
    conn.execute("DROP VIEW IF EXISTS teams_2026_defense_mix")
    season_literal = quote_sql_literal(season)
    season_type_literal = quote_sql_literal(season_type)
    conn.execute(
        f"""
        CREATE VIEW teams_2026_defense_mix AS
        WITH paired AS (
            SELECT
                t.team_id,
                t.team_name,
                t.team_abbreviation,
                t.game_id,
                opp.pts AS allowed_pts,
                opp.fg3m AS allowed_fg3m,
                opp.ast AS allowed_ast,
                opp.reb AS allowed_reb
            FROM team_game_logs AS t
            JOIN team_game_logs AS opp
                ON opp.game_id = t.game_id
               AND opp.team_id <> t.team_id
            WHERE t.season = '{season_literal}'
              AND t.season_type = '{season_type_literal}'
        ),
        aggregates AS (
            SELECT
                team_id,
                MAX(team_name) AS team_name,
                MAX(team_abbreviation) AS team_abbreviation,
                COUNT(*) AS games_played,
                SUM(allowed_pts) AS total_allowed_pts,
                AVG(allowed_pts) AS avg_allowed_pts,
                SUM(allowed_fg3m) AS total_allowed_fg3m,
                AVG(allowed_fg3m) AS avg_allowed_fg3m,
                SUM(allowed_ast) AS total_allowed_ast,
                AVG(allowed_ast) AS avg_allowed_ast,
                SUM(allowed_reb) AS total_allowed_reb,
                AVG(allowed_reb) AS avg_allowed_reb
            FROM paired
            GROUP BY team_id
            HAVING COUNT(*) > 0
        ),
        median_pts AS (
            SELECT
                team_id,
                AVG(allowed_pts) AS median_allowed_pts
            FROM (
                SELECT
                    team_id,
                    allowed_pts,
                    ROW_NUMBER() OVER (
                        PARTITION BY team_id
                        ORDER BY allowed_pts
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY team_id) AS cnt
                FROM paired
            )
            WHERE rn IN (
                (cnt + 1) / 2,
                (cnt + 2) / 2
            )
            GROUP BY team_id
        ),
        median_fg3m AS (
            SELECT
                team_id,
                AVG(allowed_fg3m) AS median_allowed_fg3m
            FROM (
                SELECT
                    team_id,
                    allowed_fg3m,
                    ROW_NUMBER() OVER (
                        PARTITION BY team_id
                        ORDER BY allowed_fg3m
                    ) AS rn,
                    COUNT(*) OVER (PARTITION BY team_id) AS cnt
                FROM paired
            )
            WHERE rn IN (
                (cnt + 1) / 2,
                (cnt + 2) / 2
            )
            GROUP BY team_id
        )
        SELECT
            a.team_id,
            a.team_name,
            a.team_abbreviation,
            a.games_played,
            a.total_allowed_pts,
            COALESCE(mp.median_allowed_pts, a.avg_allowed_pts) AS median_allowed_pts,
            a.total_allowed_fg3m,
            COALESCE(mf.median_allowed_fg3m, a.avg_allowed_fg3m) AS median_allowed_fg3m,
            a.total_allowed_ast,
            a.avg_allowed_ast,
            a.total_allowed_reb,
            a.avg_allowed_reb,
            CASE
                WHEN a.total_allowed_pts > 0
                    THEN (a.total_allowed_fg3m * 3.0) / a.total_allowed_pts
                ELSE NULL
            END AS pct_points_from_3_total,
            CASE
                WHEN COALESCE(mp.median_allowed_pts, a.avg_allowed_pts) > 0
                    THEN (
                        COALESCE(mf.median_allowed_fg3m, a.avg_allowed_fg3m) * 3.0
                    ) / COALESCE(mp.median_allowed_pts, a.avg_allowed_pts)
                ELSE NULL
            END AS pct_points_from_3_median
        FROM aggregates AS a
        LEFT JOIN median_pts AS mp
            ON mp.team_id = a.team_id
        LEFT JOIN median_fg3m AS mf
            ON mf.team_id = a.team_id
        ORDER BY pct_points_from_3_total DESC,
                 a.team_name ASC
        """
    )
    conn.commit()


def build_database(
    *,
    db_path: str | Path = "nba_stats.db",
    season: str = "2023-24",
    season_type: str = "Regular Season",
    include_rosters: bool = True,
    throttle_seconds: float = 0.6,
    shooting_season: str | None = None,
    shooting_season_type: str | None = None,
    top_3pt_view_season: str = "2024-25",
    top_3pt_view_season_type: str | None = None,
    defense_view_season: str = "2024-25",
    defense_view_season_type: str | None = None,
    top_pts_view_season: str = "2024-25",
    top_pts_view_season_type: str | None = None,
    defense_pts_view_season: str = "2024-25",
    defense_pts_view_season_type: str | None = None,
    defense_mix_view_season: str = "2025-26",
    defense_mix_view_season_type: str | None = None,
) -> Path:
    """Build the NBA SQLite database and return the resulting path."""
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        print("Loading team directory ...")
        team_count = load_teams(conn)
        print(f"Upserted {team_count} teams.")

        print("Loading player directory ...")
        player_count = load_players(conn)
        print(f"Upserted {player_count} players.")

        print(f"Loading standings for season {season} ({season_type}) ...")
        standings_count = load_standings(conn, season, season_type)
        print(f"Upserted {standings_count} standings rows.")

        if include_rosters:
            print(f"Loading team rosters for {season} ...")
            roster_count = load_team_rosters(
                conn,
                season,
                max(throttle_seconds, 0.0),
            )
            print(f"Upserted {roster_count} roster rows.")

        shooting_season = shooting_season or season
        shooting_season_type = shooting_season_type or season_type
        three_pt_view_season = top_3pt_view_season or shooting_season
        three_pt_view_season_type = (
            top_3pt_view_season_type or shooting_season_type
        )
        points_view_season = top_pts_view_season or shooting_season
        points_view_season_type = (
            top_pts_view_season_type or shooting_season_type
        )

        def add_unique(targets, season, season_type):
            pair = (season, season_type)
            if pair not in targets:
                targets.append(pair)

        player_total_targets: List[tuple[str, str]] = []
        add_unique(player_total_targets, shooting_season, shooting_season_type)
        add_unique(player_total_targets, three_pt_view_season, three_pt_view_season_type)
        add_unique(player_total_targets, points_view_season, points_view_season_type)
        for season, season_type in player_total_targets:
            print(
                "Loading player shooting totals for "
                f"{season} ({season_type}) ..."
            )
            shooting_count = load_player_shooting_totals(
                conn,
                season,
                season_type,
            )
            print(
                f"Upserted {shooting_count} player shooting rows "
                f"for {season} ({season_type})."
            )
            usage_count = update_player_usage(
                conn,
                season,
                season_type,
            )
            print(
                f"Updated usage% for {usage_count} players "
                f"in {season} ({season_type})."
            )

        player_log_targets: List[tuple[str, str]] = []
        add_unique(player_log_targets, shooting_season, shooting_season_type)
        add_unique(player_log_targets, three_pt_view_season, three_pt_view_season_type)
        add_unique(player_log_targets, points_view_season, points_view_season_type)
        for season, season_type in player_log_targets:
            print(
                "Loading player game logs for "
                f"{season} ({season_type}) ..."
            )
            game_log_count = load_player_game_logs(
                conn,
                season,
                season_type,
            )
            print(
                f"Upserted {game_log_count} player game log rows "
                f"for {season} ({season_type})."
            )

        create_three_point_view(conn, three_pt_view_season, three_pt_view_season_type)
        print(
            "Created/updated players_2025_top_3pt view for "
            f"{three_pt_view_season} ({three_pt_view_season_type})."
        )
        create_player_points_view(conn, points_view_season, points_view_season_type)
        print(
            "Created/updated players_2025_top_pts view for "
            f"{points_view_season} ({points_view_season_type})."
        )

        defense_view_season = defense_view_season or shooting_season
        defense_view_season_type = (
            defense_view_season_type or shooting_season_type
        )
        defense_pts_view_season = (
            defense_pts_view_season or defense_view_season
        )
        defense_pts_view_season_type = (
            defense_pts_view_season_type or defense_view_season_type
        )
        defense_mix_view_season = (
            defense_mix_view_season or defense_pts_view_season
        )
        defense_mix_view_season_type = (
            defense_mix_view_season_type or defense_pts_view_season_type
        )

        team_log_targets: List[tuple[str, str]] = []
        add_unique(team_log_targets, defense_view_season, defense_view_season_type)
        add_unique(
            team_log_targets,
            defense_pts_view_season,
            defense_pts_view_season_type,
        )
        add_unique(
            team_log_targets,
            defense_mix_view_season,
            defense_mix_view_season_type,
        )
        for season, season_type in team_log_targets:
            print(
                "Loading team game logs for "
                f"{season} ({season_type}) ..."
            )
            team_log_count = load_team_game_logs(
                conn,
                season,
                season_type,
            )
            print(
                f"Upserted {team_log_count} team game log rows "
                f"for {season} ({season_type})."
            )

        create_team_defense_three_point_view(
            conn,
            defense_view_season,
            defense_view_season_type,
        )
        print(
            "Created/updated teams_2025_defense_3pt view for "
            f"{defense_view_season} ({defense_view_season_type})."
        )
        create_team_defense_points_view(
            conn,
            defense_pts_view_season,
            defense_pts_view_season_type,
        )
        print(
            "Created/updated teams_2025_defense_pts view for "
            f"{defense_pts_view_season} ({defense_pts_view_season_type})."
        )
        create_team_defense_mix_view(
            conn,
            defense_mix_view_season,
            defense_mix_view_season_type,
        )
        print(
            "Created/updated teams_2026_defense_mix view for "
            f"{defense_mix_view_season} ({defense_mix_view_season_type})."
        )
    finally:
        conn.close()
    print(f"Database ready at {db_path}")
    return db_path


def main() -> None:
    args = parse_args()
    build_database(**vars(args))


if __name__ == "__main__":
    main()
