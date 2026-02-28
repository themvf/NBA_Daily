#!/usr/bin/env python3
"""DFS Slate Tracking System for NBA_Daily.

Tracks projection accuracy, stat-level accuracy, lineup performance,
and value accuracy across DFS slates. Stores results in SQLite for
trend analysis and model improvement.

Tables:
    dfs_slate_projections — Per-player projections + actuals for each slate
    dfs_slate_lineups     — Generated lineups per slate with actual FPTS
    dfs_slate_results     — Aggregate accuracy metrics per slate
    dfs_supplement_runs   — Saved third-party supplement snapshot summaries
    dfs_supplement_player_deltas — Per-player supplement comparison rows
"""

import sqlite3
import unicodedata
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, List, Optional, Dict, Tuple
from pathlib import Path

# Import DFS types and scoring from the optimizer
from dfs_optimizer import DFSPlayer, DFSLineup, calculate_dk_fantasy_points


# =============================================================================
# Table Creation
# =============================================================================

def create_dfs_tracking_tables(conn: sqlite3.Connection) -> None:
    """Create the 3 DFS tracking tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dfs_slate_projections (
            slate_date TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            salary INTEGER,
            positions TEXT,
            proj_fpts REAL,
            proj_points REAL,
            proj_rebounds REAL,
            proj_assists REAL,
            proj_steals REAL,
            proj_blocks REAL,
            proj_turnovers REAL,
            proj_fg3m REAL,
            proj_floor REAL,
            proj_ceiling REAL,
            fpts_per_dollar REAL,
            actual_fpts REAL,
            actual_points REAL,
            actual_rebounds REAL,
            actual_assists REAL,
            actual_steals REAL,
            actual_blocks REAL,
            actual_turnovers REAL,
            actual_fg3m REAL,
            actual_minutes REAL,
            did_play INTEGER,
            PRIMARY KEY (slate_date, player_id)
        );

        CREATE TABLE IF NOT EXISTS dfs_slate_lineups (
            slate_date TEXT NOT NULL,
            lineup_num INTEGER NOT NULL,
            model_key TEXT,
            model_label TEXT,
            generation_strategy TEXT,
            total_proj_fpts REAL,
            total_salary INTEGER,
            pg_id INTEGER,
            sg_id INTEGER,
            sf_id INTEGER,
            pf_id INTEGER,
            c_id INTEGER,
            g_id INTEGER,
            f_id INTEGER,
            util_id INTEGER,
            total_actual_fpts REAL,
            created_at TEXT,
            PRIMARY KEY (slate_date, lineup_num)
        );

        CREATE TABLE IF NOT EXISTS dfs_slate_results (
            slate_date TEXT PRIMARY KEY,
            num_players INTEGER,
            num_lineups INTEGER,
            proj_mae REAL,
            proj_rmse REAL,
            proj_correlation REAL,
            proj_within_range_pct REAL,
            stat_mae_points REAL,
            stat_mae_rebounds REAL,
            stat_mae_assists REAL,
            stat_mae_steals REAL,
            stat_mae_blocks REAL,
            stat_mae_turnovers REAL,
            stat_mae_fg3m REAL,
            best_lineup_actual_fpts REAL,
            optimal_lineup_fpts REAL,
            lineup_efficiency_pct REAL,
            avg_lineup_actual_fpts REAL,
            top10_value_avg_actual REAL,
            bottom10_value_avg_actual REAL,
            value_correlation REAL,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS dfs_supplement_runs (
            run_key TEXT PRIMARY KEY,
            slate_date TEXT NOT NULL,
            source_name TEXT,
            source_filename TEXT,
            projection_col TEXT,
            ownership_col TEXT,
            rows_total INTEGER,
            rows_matched INTEGER,
            rows_unmatched INTEGER,
            match_rate REAL,
            projection_mae REAL,
            ownership_mae REAL,
            avg_proj_delta REAL,
            avg_own_delta REAL,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS dfs_supplement_player_deltas (
            run_key TEXT NOT NULL,
            slate_date TEXT NOT NULL,
            player_id INTEGER,
            supplement_player TEXT,
            supplement_team TEXT,
            our_player TEXT,
            our_team TEXT,
            pos TEXT,
            salary INTEGER,
            match_method TEXT,
            match_score REAL,
            our_proj_fpts REAL,
            supplement_proj_fpts REAL,
            proj_delta REAL,
            our_own_pct REAL,
            supplement_own_pct REAL,
            own_delta_pp REAL,
            created_at TEXT,
            PRIMARY KEY (run_key, player_id)
        );
    """)

    # Migration-safe column additions (check first, then add if missing)
    migrations = [
        ("dfs_slate_projections", "ownership_proj", "REAL"),
        ("dfs_slate_projections", "actual_ownership", "REAL"),
        ("dfs_slate_results", "ownership_mae", "REAL"),
        ("dfs_slate_results", "ownership_correlation", "REAL"),
        ("dfs_slate_lineups", "model_key", "TEXT"),
        ("dfs_slate_lineups", "model_label", "TEXT"),
        ("dfs_slate_lineups", "generation_strategy", "TEXT"),
        ("dfs_slate_lineups", "created_at", "TEXT"),
    ]
    for table, col, col_type in migrations:
        # Check if column exists first to avoid ALTER TABLE errors
        existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column might have been added concurrently

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_dsl_slate_model
            ON dfs_slate_lineups(slate_date, model_key);
        CREATE INDEX IF NOT EXISTS idx_dsl_model
            ON dfs_slate_lineups(model_key);
        CREATE INDEX IF NOT EXISTS idx_dsr_slate
            ON dfs_supplement_runs(slate_date);
        CREATE INDEX IF NOT EXISTS idx_dspd_slate
            ON dfs_supplement_player_deltas(slate_date);
        CREATE INDEX IF NOT EXISTS idx_dspd_player
            ON dfs_supplement_player_deltas(player_id);
    """)

    # --- Opponent tracking tables ---
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dfs_contest_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contest_id TEXT NOT NULL,
            slate_date TEXT NOT NULL,
            username TEXT NOT NULL,
            max_entries INTEGER DEFAULT 1,
            entry_count INTEGER DEFAULT 1,
            rank INTEGER,
            points REAL,
            lineup_raw TEXT,
            pg TEXT, sg TEXT, sf TEXT, pf TEXT,
            c TEXT, g TEXT, f TEXT, util TEXT,
            total_salary INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(contest_id, username, lineup_raw)
        );

        CREATE INDEX IF NOT EXISTS idx_dce_username
            ON dfs_contest_entries(username);
        CREATE INDEX IF NOT EXISTS idx_dce_slate
            ON dfs_contest_entries(slate_date);

        CREATE TABLE IF NOT EXISTS dfs_contest_meta (
            contest_id TEXT PRIMARY KEY,
            slate_date TEXT NOT NULL,
            total_entries INTEGER,
            unique_users INTEGER,
            top_score REAL,
            import_date TEXT DEFAULT (datetime('now'))
        );
    """)

    conn.commit()


# =============================================================================
# Save Functions (pre-game)
# =============================================================================

def save_slate_projections(
    conn: sqlite3.Connection,
    slate_date: str,
    players: List[DFSPlayer]
) -> int:
    """Save player projections for a slate date.

    Returns:
        Number of players saved.
    """
    cursor = conn.cursor()
    saved = 0

    for p in players:
        if p.is_injured or p.is_excluded:
            continue  # Don't track excluded/injured players

        cursor.execute("""
            INSERT OR REPLACE INTO dfs_slate_projections (
                slate_date, player_id, player_name, team, opponent,
                salary, positions, proj_fpts, proj_points, proj_rebounds,
                proj_assists, proj_steals, proj_blocks, proj_turnovers,
                proj_fg3m, proj_floor, proj_ceiling, fpts_per_dollar,
                ownership_proj
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            slate_date, p.player_id, p.name, p.team, p.opponent,
            p.salary, ','.join(p.positions), p.proj_fpts, p.proj_points,
            p.proj_rebounds, p.proj_assists, p.proj_steals, p.proj_blocks,
            p.proj_turnovers, p.proj_fg3m, p.proj_floor, p.proj_ceiling,
            p.fpts_per_dollar, p.ownership_proj
        ))
        saved += 1

    conn.commit()
    return saved


def save_slate_lineups(
    conn: sqlite3.Connection,
    slate_date: str,
    lineups: List[DFSLineup]
) -> int:
    """Save generated lineups for a slate date, sorted by projected FPTS.

    Returns:
        Number of lineups saved.
    """
    sorted_lineups = sorted(lineups, key=lambda l: l.total_proj_fpts, reverse=True)
    cursor = conn.cursor()

    # Clear existing lineups for this slate (in case of re-generation)
    cursor.execute("DELETE FROM dfs_slate_lineups WHERE slate_date = ?", (slate_date,))

    for i, lineup in enumerate(sorted_lineups, start=1):
        players = lineup.players
        model_key = getattr(lineup, "model_key", "") or "standard_v1"
        model_label = getattr(lineup, "model_label", "") or model_key
        generation_strategy = getattr(lineup, "generation_strategy", "") or ""
        cursor.execute("""
            INSERT OR REPLACE INTO dfs_slate_lineups (
                slate_date, lineup_num, model_key, model_label, generation_strategy,
                total_proj_fpts, total_salary,
                pg_id, sg_id, sf_id, pf_id, c_id, g_id, f_id, util_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            slate_date, i, model_key, model_label, generation_strategy,
            lineup.total_proj_fpts, lineup.total_salary,
            players['PG'].player_id if players.get('PG') else None,
            players['SG'].player_id if players.get('SG') else None,
            players['SF'].player_id if players.get('SF') else None,
            players['PF'].player_id if players.get('PF') else None,
            players['C'].player_id if players.get('C') else None,
            players['G'].player_id if players.get('G') else None,
            players['F'].player_id if players.get('F') else None,
            players['UTIL'].player_id if players.get('UTIL') else None,
            datetime.now().isoformat(),
        ))

    conn.commit()
    return len(sorted_lineups)


def save_supplement_snapshot(
    conn: sqlite3.Connection,
    slate_date: str,
    run_key: str,
    source_name: str,
    source_filename: str,
    projection_col: Optional[str],
    ownership_col: Optional[str],
    comparison_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    rows_total: int,
) -> str:
    """Persist a supplement comparison snapshot for later daily review."""
    comparison_df = comparison_df.copy()
    unmatched_df = unmatched_df.copy()

    proj_comp_df = comparison_df.dropna(
        subset=["Supplement Proj FPTS", "Our Proj FPTS"]
    ).copy()
    own_comp_df = comparison_df.dropna(
        subset=["Supplement Own %", "Our Own %"]
    ).copy()

    rows_matched = int(len(comparison_df))
    rows_unmatched = int(len(unmatched_df))
    match_rate = float((rows_matched / rows_total) * 100.0) if rows_total > 0 else 0.0
    projection_mae = (
        float(pd.to_numeric(proj_comp_df["Proj Delta"], errors="coerce").abs().mean())
        if not proj_comp_df.empty else None
    )
    ownership_mae = (
        float(pd.to_numeric(own_comp_df["Own Delta (pp)"], errors="coerce").abs().mean())
        if not own_comp_df.empty else None
    )
    avg_proj_delta = (
        float(pd.to_numeric(proj_comp_df["Proj Delta"], errors="coerce").mean())
        if not proj_comp_df.empty else None
    )
    avg_own_delta = (
        float(pd.to_numeric(own_comp_df["Own Delta (pp)"], errors="coerce").mean())
        if not own_comp_df.empty else None
    )

    created_at = datetime.now().isoformat(timespec="seconds")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dfs_supplement_player_deltas WHERE run_key = ?", (run_key,))
    cursor.execute(
        """
        INSERT OR REPLACE INTO dfs_supplement_runs (
            run_key, slate_date, source_name, source_filename,
            projection_col, ownership_col, rows_total, rows_matched,
            rows_unmatched, match_rate, projection_mae, ownership_mae,
            avg_proj_delta, avg_own_delta, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_key,
            slate_date,
            source_name,
            source_filename,
            projection_col or "",
            ownership_col or "",
            int(rows_total),
            rows_matched,
            rows_unmatched,
            match_rate,
            projection_mae,
            ownership_mae,
            avg_proj_delta,
            avg_own_delta,
            created_at,
        ),
    )

    for row in comparison_df.to_dict("records"):
        try:
            player_id = int(row.get("Our Player ID") or 0)
        except (TypeError, ValueError):
            player_id = 0
        if player_id <= 0:
            continue
        cursor.execute(
            """
            INSERT OR REPLACE INTO dfs_supplement_player_deltas (
                run_key, slate_date, player_id, supplement_player, supplement_team,
                our_player, our_team, pos, salary, match_method, match_score,
                our_proj_fpts, supplement_proj_fpts, proj_delta, our_own_pct,
                supplement_own_pct, own_delta_pp, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_key,
                slate_date,
                player_id,
                row.get("Supplement Player"),
                row.get("Supplement Team"),
                row.get("Our Player"),
                row.get("Our Team"),
                row.get("Pos"),
                int(row.get("Salary") or 0),
                row.get("Match Method"),
                float(row.get("Match Score") or 0.0),
                float(row.get("Our Proj FPTS")) if pd.notna(row.get("Our Proj FPTS")) else None,
                float(row.get("Supplement Proj FPTS")) if pd.notna(row.get("Supplement Proj FPTS")) else None,
                float(row.get("Proj Delta")) if pd.notna(row.get("Proj Delta")) else None,
                float(row.get("Our Own %")) if pd.notna(row.get("Our Own %")) else None,
                float(row.get("Supplement Own %")) if pd.notna(row.get("Supplement Own %")) else None,
                float(row.get("Own Delta (pp)")) if pd.notna(row.get("Own Delta (pp)")) else None,
                created_at,
            ),
        )

    conn.commit()
    return run_key


def get_recent_supplement_runs(
    conn: sqlite3.Connection,
    limit: int = 20,
) -> pd.DataFrame:
    """Return recent saved supplement snapshots for review."""
    return pd.read_sql_query(
        """
        SELECT
            run_key,
            slate_date,
            source_name,
            source_filename,
            projection_col,
            ownership_col,
            rows_total,
            rows_matched,
            rows_unmatched,
            match_rate,
            projection_mae,
            ownership_mae,
            avg_proj_delta,
            avg_own_delta,
            created_at
        FROM dfs_supplement_runs
        ORDER BY slate_date DESC, created_at DESC
        LIMIT ?
        """,
        conn,
        params=[int(limit)],
    )


def get_supplement_run_player_deltas(
    conn: sqlite3.Connection,
    run_key: str,
) -> pd.DataFrame:
    """Return per-player deltas for a saved supplement snapshot."""
    return pd.read_sql_query(
        """
        SELECT
            slate_date,
            supplement_player,
            supplement_team,
            our_player,
            our_team,
            pos,
            salary,
            match_method,
            match_score,
            our_proj_fpts,
            supplement_proj_fpts,
            proj_delta,
            our_own_pct,
            supplement_own_pct,
            own_delta_pp
        FROM dfs_supplement_player_deltas
        WHERE run_key = ?
        ORDER BY ABS(COALESCE(proj_delta, 0.0)) DESC,
                 ABS(COALESCE(own_delta_pp, 0.0)) DESC,
                 supplement_player ASC
        """,
        conn,
        params=[run_key],
    )


# =============================================================================
# Update Actuals (post-game)
# =============================================================================

def update_slate_actuals(conn: sqlite3.Connection, slate_date: str) -> Tuple[int, int]:
    """Update actual stats from player_game_logs for a slate date.

    Joins dfs_slate_projections with player_game_logs to fill in actual
    performance data after games have been played.

    Returns:
        Tuple of (players_updated, players_not_found).
    """
    cursor = conn.cursor()

    # Check which columns exist in player_game_logs
    col_cursor = conn.execute("PRAGMA table_info(player_game_logs)")
    existing_columns = {row[1] for row in col_cursor.fetchall()}

    # Get all projected players for this slate
    proj_rows = cursor.execute(
        "SELECT player_id FROM dfs_slate_projections WHERE slate_date = ?",
        (slate_date,)
    ).fetchall()

    if not proj_rows:
        return 0, 0

    updated = 0
    not_found = 0

    # Build safe column list once (whitelist only — never user input)
    ALLOWED_STAT_COLS = ['points', 'rebounds', 'assists', 'steals',
                         'blocks', 'turnovers', 'fg3m', 'minutes']
    safe_cols = [c for c in ALLOWED_STAT_COLS if c in existing_columns]

    if not safe_cols:
        return 0, len(proj_rows)

    select_cols = ', '.join(safe_cols)

    for (player_id,) in proj_rows:
        # Use LIKE for date matching since player_game_logs stores timestamps
        # like "2026-02-04T00:00:00" but slate_date is just "2026-02-04"
        row = cursor.execute(f"""
            SELECT {select_cols}
            FROM player_game_logs
            WHERE player_id = ? AND game_date LIKE ?
        """, (player_id, f"{slate_date}%")).fetchone()

        if row is None:
            # Player didn't play or no data yet
            cursor.execute("""
                UPDATE dfs_slate_projections
                SET did_play = 0
                WHERE slate_date = ? AND player_id = ?
            """, (slate_date, player_id))
            not_found += 1
            continue

        # Map results back to stat names
        stats = {safe_cols[i]: (row[i] or 0.0) for i in range(len(row))}

        # Calculate actual fantasy points
        actual_fpts = calculate_dk_fantasy_points(
            points=stats.get('points', 0),
            rebounds=stats.get('rebounds', 0),
            assists=stats.get('assists', 0),
            steals=stats.get('steals', 0),
            blocks=stats.get('blocks', 0),
            turnovers=stats.get('turnovers', 0),
            fg3m=stats.get('fg3m', 0)
        )

        cursor.execute("""
            UPDATE dfs_slate_projections SET
                actual_fpts = ?,
                actual_points = ?,
                actual_rebounds = ?,
                actual_assists = ?,
                actual_steals = ?,
                actual_blocks = ?,
                actual_turnovers = ?,
                actual_fg3m = ?,
                actual_minutes = ?,
                did_play = 1
            WHERE slate_date = ? AND player_id = ?
        """, (
            actual_fpts,
            stats.get('points', 0),
            stats.get('rebounds', 0),
            stats.get('assists', 0),
            stats.get('steals', 0),
            stats.get('blocks', 0),
            stats.get('turnovers', 0),
            stats.get('fg3m', 0),
            stats.get('minutes', 0),
            slate_date, player_id
        ))
        updated += 1

    # Update lineup actual totals
    _update_lineup_actuals(conn, slate_date)

    conn.commit()
    return updated, not_found


def _update_lineup_actuals(conn: sqlite3.Connection, slate_date: str) -> None:
    """Compute total_actual_fpts for each lineup from player actuals."""
    cursor = conn.cursor()

    lineups = cursor.execute(
        "SELECT lineup_num, pg_id, sg_id, sf_id, pf_id, c_id, g_id, f_id, util_id "
        "FROM dfs_slate_lineups WHERE slate_date = ?",
        (slate_date,)
    ).fetchall()

    # Build player_id -> actual_fpts lookup
    actuals = cursor.execute(
        "SELECT player_id, actual_fpts FROM dfs_slate_projections "
        "WHERE slate_date = ? AND did_play = 1",
        (slate_date,)
    ).fetchall()
    fpts_lookup = {pid: fpts for pid, fpts in actuals}

    for row in lineups:
        lineup_num = row[0]
        player_ids = row[1:]  # pg_id through util_id
        total = sum(fpts_lookup.get(pid, 0) for pid in player_ids if pid is not None)

        cursor.execute(
            "UPDATE dfs_slate_lineups SET total_actual_fpts = ? "
            "WHERE slate_date = ? AND lineup_num = ?",
            (total, slate_date, lineup_num)
        )


# =============================================================================
# Import Contest Results (post-contest ownership data)
# =============================================================================

def import_contest_results(
    conn: sqlite3.Connection,
    slate_date: str,
    ownership_map: Dict[str, float],
    actual_fpts_map: Dict[str, float]
) -> Tuple[int, int]:
    """Import actual ownership from contest standings CSV.

    Matches player names from the contest CSV to dfs_slate_projections rows
    and updates actual_ownership. Uses normalized name matching to handle
    minor discrepancies (accents, suffixes, etc).

    Args:
        conn: SQLite connection
        slate_date: Date string (YYYY-MM-DD)
        ownership_map: {player_name: ownership_pct} from parse_contest_standings()
        actual_fpts_map: {player_name: fpts} from parse_contest_standings()

    Returns:
        Tuple of (matched_count, unmatched_count)
    """
    cursor = conn.cursor()

    # Get all projected players for this slate
    rows = cursor.execute(
        "SELECT player_id, player_name FROM dfs_slate_projections WHERE slate_date = ?",
        (slate_date,)
    ).fetchall()

    if not rows:
        return 0, len(ownership_map)

    # Build normalized name lookup from contest data
    # Normalize: lowercase, strip accents, remove suffixes like Jr./Sr./III
    def _normalize(name: str) -> str:
        n = name.strip().lower()
        # Remove common suffixes
        for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
            if n.endswith(suffix):
                n = n[:-len(suffix)].strip()
        # Strip accents (e.g., Nikola Jokić -> nikola jokic)
        n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode('ascii')
        return n

    contest_by_norm = {}
    for name, own_pct in ownership_map.items():
        norm = _normalize(name)
        contest_by_norm[norm] = (name, own_pct, actual_fpts_map.get(name, 0.0))

    matched = 0
    unmatched_players = []

    for player_id, player_name in rows:
        norm_name = _normalize(player_name)

        if norm_name in contest_by_norm:
            _, own_pct, contest_fpts = contest_by_norm[norm_name]
            # Update ownership; also fill actual_fpts/did_play from contest
            # CSV when not already populated by update_slate_actuals()
            cursor.execute(
                """UPDATE dfs_slate_projections
                   SET actual_ownership = ?,
                       actual_fpts = COALESCE(actual_fpts, ?),
                       did_play = CASE WHEN did_play = 1 THEN 1
                                       WHEN ? > 0 THEN 1 ELSE did_play END
                   WHERE slate_date = ? AND player_id = ?""",
                (own_pct, contest_fpts, contest_fpts, slate_date, player_id)
            )
            matched += 1
        else:
            unmatched_players.append(player_name)

    # Recompute lineup actual totals now that actual_fpts may be populated
    _update_lineup_actuals(conn, slate_date)

    conn.commit()

    unmatched_count = len(unmatched_players)
    return matched, unmatched_count


# =============================================================================
# Optimal Lineup Computation
# =============================================================================

# Position eligibility for each DK slot
SLOT_ELIGIBILITY = {
    'PG': ['PG'],
    'SG': ['SG'],
    'SF': ['SF'],
    'PF': ['PF'],
    'C':  ['C'],
    'G':  ['PG', 'SG'],
    'F':  ['SF', 'PF'],
    'UTIL': ['PG', 'SG', 'SF', 'PF', 'C'],
}

# Slot assignment priority: most restrictive first
SLOT_PRIORITY = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']


def compute_optimal_lineup(conn: sqlite3.Connection, slate_date: str) -> float:
    """Compute the theoretical optimal lineup (position-constrained, salary-unconstrained).

    Uses a greedy approach: sort players by actual FPTS descending,
    assign each to the most restrictive eligible slot first.

    Returns:
        Total actual FPTS of the optimal lineup, or 0.0 if insufficient data.
    """
    df = pd.read_sql_query(
        "SELECT player_id, positions, actual_fpts "
        "FROM dfs_slate_projections "
        "WHERE slate_date = ? AND did_play = 1 AND actual_fpts IS NOT NULL",
        conn, params=[slate_date]
    )

    if df.empty:
        return 0.0

    # Sort by actual FPTS descending
    df = df.sort_values('actual_fpts', ascending=False).reset_index(drop=True)

    # Track which slots are filled
    filled_slots = {}  # slot_name -> player_id
    used_players = set()

    for _, row in df.iterrows():
        if len(filled_slots) == 8:
            break  # All slots filled

        pid = row['player_id']
        if pid in used_players:
            continue

        player_positions = [p.strip() for p in str(row['positions']).split(',') if p.strip()]

        # Try to assign to the most restrictive eligible slot first
        assigned = False
        for slot in SLOT_PRIORITY:
            if slot in filled_slots:
                continue  # Slot already taken

            eligible_positions = SLOT_ELIGIBILITY[slot]
            if any(pos in eligible_positions for pos in player_positions):
                filled_slots[slot] = pid
                used_players.add(pid)
                assigned = True
                break

    if not filled_slots:
        return 0.0

    # Sum actual FPTS for all assigned players
    assigned_ids = list(filled_slots.values())
    total = df[df['player_id'].isin(assigned_ids)]['actual_fpts'].sum()
    return float(total)


# =============================================================================
# Compute & Store Aggregate Results
# =============================================================================

def compute_and_store_slate_results(conn: sqlite3.Connection, slate_date: str) -> Optional[Dict]:
    """Compute all accuracy metrics for a slate and store in dfs_slate_results.

    Returns:
        Dict of computed metrics, or None if insufficient data.
    """
    # Load projections with actuals
    df = pd.read_sql_query(
        "SELECT * FROM dfs_slate_projections "
        "WHERE slate_date = ? AND did_play = 1 AND actual_fpts IS NOT NULL",
        conn, params=[slate_date]
    )

    if len(df) < 3:
        return None  # Need minimum players for meaningful stats

    # --- Projection Accuracy ---
    errors = df['actual_fpts'] - df['proj_fpts']
    abs_errors = errors.abs()
    proj_mae = float(abs_errors.mean())
    proj_rmse = float(np.sqrt((errors ** 2).mean()))

    # Pearson correlation (None if std=0, meaning undefined — not zero)
    if df['proj_fpts'].std() > 0 and df['actual_fpts'].std() > 0:
        proj_correlation = float(df['proj_fpts'].corr(df['actual_fpts']))
    else:
        proj_correlation = None

    # Within floor-ceiling range
    within_range = (
        (df['actual_fpts'] >= df['proj_floor']) &
        (df['actual_fpts'] <= df['proj_ceiling'])
    )
    proj_within_range_pct = float(within_range.mean() * 100)

    # --- Stat-Level MAE ---
    stat_pairs = [
        ('proj_points', 'actual_points', 'stat_mae_points'),
        ('proj_rebounds', 'actual_rebounds', 'stat_mae_rebounds'),
        ('proj_assists', 'actual_assists', 'stat_mae_assists'),
        ('proj_steals', 'actual_steals', 'stat_mae_steals'),
        ('proj_blocks', 'actual_blocks', 'stat_mae_blocks'),
        ('proj_turnovers', 'actual_turnovers', 'stat_mae_turnovers'),
        ('proj_fg3m', 'actual_fg3m', 'stat_mae_fg3m'),
    ]

    stat_maes = {}
    for proj_col, actual_col, result_key in stat_pairs:
        if proj_col in df.columns and actual_col in df.columns:
            valid = df[[proj_col, actual_col]].dropna()
            if len(valid) > 0:
                stat_maes[result_key] = float((valid[proj_col] - valid[actual_col]).abs().mean())
            else:
                stat_maes[result_key] = None
        else:
            stat_maes[result_key] = None

    # --- Lineup Performance ---
    lineup_df = pd.read_sql_query(
        "SELECT total_actual_fpts FROM dfs_slate_lineups "
        "WHERE slate_date = ? AND total_actual_fpts IS NOT NULL",
        conn, params=[slate_date]
    )

    best_lineup_actual = float(lineup_df['total_actual_fpts'].max()) if not lineup_df.empty else None
    avg_lineup_actual = float(lineup_df['total_actual_fpts'].mean()) if not lineup_df.empty else None
    num_lineups = len(lineup_df) if not lineup_df.empty else 0

    optimal_fpts = compute_optimal_lineup(conn, slate_date)

    if optimal_fpts > 0 and best_lineup_actual is not None:
        lineup_efficiency = (best_lineup_actual / optimal_fpts) * 100
    else:
        lineup_efficiency = None

    # --- Ownership Accuracy ---
    ownership_mae = None
    ownership_corr = None

    if 'ownership_proj' in df.columns and 'actual_ownership' in df.columns:
        own_df = df[['ownership_proj', 'actual_ownership']].dropna()
        if len(own_df) >= 5:
            own_errors = (own_df['actual_ownership'] - own_df['ownership_proj']).abs()
            ownership_mae = float(own_errors.mean())
            if own_df['ownership_proj'].std() > 0 and own_df['actual_ownership'].std() > 0:
                ownership_corr = float(
                    own_df['ownership_proj'].corr(own_df['actual_ownership'])
                )

    # --- Value Accuracy ---
    value_df = df[['fpts_per_dollar', 'actual_fpts', 'salary']].dropna().copy()
    value_df = value_df[value_df['salary'] > 0]  # Guard against division by zero
    top10_val_avg = None
    bottom10_val_avg = None
    value_corr = None

    if len(value_df) >= 20:
        value_df = value_df.sort_values('fpts_per_dollar', ascending=False)
        top10 = value_df.head(10)
        bottom10 = value_df.tail(10)
        top10_val_avg = float(top10['actual_fpts'].mean())
        bottom10_val_avg = float(bottom10['actual_fpts'].mean())

        # Compute actual FPTS per dollar for correlation
        value_df['actual_fpts_per_dollar'] = value_df['actual_fpts'] / value_df['salary'] * 1000
        if value_df['fpts_per_dollar'].std() > 0 and value_df['actual_fpts_per_dollar'].std() > 0:
            value_corr = float(value_df['fpts_per_dollar'].corr(value_df['actual_fpts_per_dollar']))

    # --- Store Results ---
    results = {
        'slate_date': slate_date,
        'num_players': len(df),
        'num_lineups': num_lineups,
        'proj_mae': proj_mae,
        'proj_rmse': proj_rmse,
        'proj_correlation': proj_correlation,
        'proj_within_range_pct': proj_within_range_pct,
        'stat_mae_points': stat_maes.get('stat_mae_points'),
        'stat_mae_rebounds': stat_maes.get('stat_mae_rebounds'),
        'stat_mae_assists': stat_maes.get('stat_mae_assists'),
        'stat_mae_steals': stat_maes.get('stat_mae_steals'),
        'stat_mae_blocks': stat_maes.get('stat_mae_blocks'),
        'stat_mae_turnovers': stat_maes.get('stat_mae_turnovers'),
        'stat_mae_fg3m': stat_maes.get('stat_mae_fg3m'),
        'best_lineup_actual_fpts': best_lineup_actual,
        'optimal_lineup_fpts': optimal_fpts if optimal_fpts > 0 else None,
        'lineup_efficiency_pct': lineup_efficiency,
        'avg_lineup_actual_fpts': avg_lineup_actual,
        'top10_value_avg_actual': top10_val_avg,
        'bottom10_value_avg_actual': bottom10_val_avg,
        'value_correlation': value_corr,
        'ownership_mae': ownership_mae,
        'ownership_correlation': ownership_corr,
        'created_at': datetime.now().isoformat(),
    }

    conn.execute("""
        INSERT OR REPLACE INTO dfs_slate_results (
            slate_date, num_players, num_lineups,
            proj_mae, proj_rmse, proj_correlation, proj_within_range_pct,
            stat_mae_points, stat_mae_rebounds, stat_mae_assists,
            stat_mae_steals, stat_mae_blocks, stat_mae_turnovers, stat_mae_fg3m,
            best_lineup_actual_fpts, optimal_lineup_fpts, lineup_efficiency_pct,
            avg_lineup_actual_fpts,
            top10_value_avg_actual, bottom10_value_avg_actual, value_correlation,
            ownership_mae, ownership_correlation,
            created_at
        ) VALUES (
            :slate_date, :num_players, :num_lineups,
            :proj_mae, :proj_rmse, :proj_correlation, :proj_within_range_pct,
            :stat_mae_points, :stat_mae_rebounds, :stat_mae_assists,
            :stat_mae_steals, :stat_mae_blocks, :stat_mae_turnovers, :stat_mae_fg3m,
            :best_lineup_actual_fpts, :optimal_lineup_fpts, :lineup_efficiency_pct,
            :avg_lineup_actual_fpts,
            :top10_value_avg_actual, :bottom10_value_avg_actual, :value_correlation,
            :ownership_mae, :ownership_correlation,
            :created_at
        )
    """, results)
    conn.commit()

    return results


# =============================================================================
# Query Functions (for Streamlit UI)
# =============================================================================

def get_dfs_accuracy_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get all slate results as a DataFrame for the summary table."""
    return pd.read_sql_query(
        "SELECT * FROM dfs_slate_results ORDER BY slate_date DESC",
        conn
    )


def get_slate_dates(conn: sqlite3.Connection) -> List[str]:
    """Get all tracked slate dates (both with and without results)."""
    rows = conn.execute(
        "SELECT DISTINCT slate_date FROM dfs_slate_projections ORDER BY slate_date DESC"
    ).fetchall()
    return [r[0] for r in rows]


def get_pending_slate_dates(conn: sqlite3.Connection) -> List[str]:
    """Get slate dates that have projections but no results yet."""
    rows = conn.execute("""
        SELECT DISTINCT p.slate_date
        FROM dfs_slate_projections p
        LEFT JOIN dfs_slate_results r ON p.slate_date = r.slate_date
        WHERE r.slate_date IS NULL
        ORDER BY p.slate_date DESC
    """).fetchall()
    return [r[0] for r in rows]


def get_slate_projection_df(conn: sqlite3.Connection, slate_date: str) -> pd.DataFrame:
    """Get projections + actuals for a specific slate (players who played)."""
    return pd.read_sql_query(
        "SELECT * FROM dfs_slate_projections "
        "WHERE slate_date = ? AND did_play = 1 AND actual_fpts IS NOT NULL "
        "ORDER BY actual_fpts DESC",
        conn, params=[slate_date]
    )


def get_value_accuracy_df(conn: sqlite3.Connection, slate_date: str) -> pd.DataFrame:
    """Get value accuracy data for a specific slate."""
    df = pd.read_sql_query(
        "SELECT player_name, team, salary, fpts_per_dollar, proj_fpts, "
        "actual_fpts, positions "
        "FROM dfs_slate_projections "
        "WHERE slate_date = ? AND did_play = 1 AND actual_fpts IS NOT NULL "
        "AND salary > 0 "
        "ORDER BY fpts_per_dollar DESC",
        conn, params=[slate_date]
    )

    if not df.empty:
        df['actual_fpts_per_dollar'] = df['actual_fpts'] / df['salary'] * 1000
        df['value_diff'] = df['actual_fpts'] - df['proj_fpts']

    return df


def get_slate_lineup_df(conn: sqlite3.Connection, slate_date: str) -> pd.DataFrame:
    """Get lineup performance data for a specific slate."""
    return pd.read_sql_query(
        "SELECT lineup_num, "
        "COALESCE(NULLIF(model_key, ''), 'standard_v1') AS model_key, "
        "COALESCE(NULLIF(model_label, ''), COALESCE(NULLIF(model_key, ''), 'standard_v1')) AS model_label, "
        "generation_strategy, total_proj_fpts, total_salary, total_actual_fpts "
        "FROM dfs_slate_lineups "
        "WHERE slate_date = ? "
        "ORDER BY lineup_num",
        conn, params=[slate_date]
    )


def get_slate_model_backtest_df(conn: sqlite3.Connection, slate_date: str) -> pd.DataFrame:
    """Get per-model lineup performance for a specific slate."""
    return pd.read_sql_query(
        """
        WITH normalized AS (
            SELECT
                slate_date,
                COALESCE(NULLIF(model_key, ''), 'standard_v1') AS model_key,
                COALESCE(NULLIF(model_label, ''), COALESCE(NULLIF(model_key, ''), 'standard_v1')) AS model_label,
                total_proj_fpts,
                total_actual_fpts,
                total_salary
            FROM dfs_slate_lineups
            WHERE slate_date = ?
        )
        SELECT
            model_key,
            MAX(model_label) AS model_label,
            COUNT(*) AS lineups,
            AVG(total_proj_fpts) AS avg_proj_fpts,
            AVG(total_actual_fpts) AS avg_actual_fpts,
            AVG(total_actual_fpts - total_proj_fpts) AS avg_vs_proj_fpts,
            MAX(total_actual_fpts) AS best_actual_fpts,
            AVG(total_salary) AS avg_salary
        FROM normalized
        GROUP BY model_key
        ORDER BY avg_actual_fpts DESC, lineups DESC
        """,
        conn,
        params=[slate_date],
    )


def get_model_backtest_summary(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_lineups: int = 1,
) -> pd.DataFrame:
    """Get cross-slate model performance for backtesting."""
    return pd.read_sql_query(
        """
        WITH normalized AS (
            SELECT
                slate_date,
                COALESCE(NULLIF(model_key, ''), 'standard_v1') AS model_key,
                COALESCE(NULLIF(model_label, ''), COALESCE(NULLIF(model_key, ''), 'standard_v1')) AS model_label,
                total_proj_fpts,
                total_actual_fpts,
                total_salary
            FROM dfs_slate_lineups
            WHERE total_actual_fpts IS NOT NULL
              AND (? IS NULL OR slate_date >= ?)
              AND (? IS NULL OR slate_date <= ?)
        )
        SELECT
            model_key,
            MAX(model_label) AS model_label,
            COUNT(*) AS lineups,
            COUNT(DISTINCT slate_date) AS slates,
            AVG(total_proj_fpts) AS avg_proj_fpts,
            AVG(total_actual_fpts) AS avg_actual_fpts,
            AVG(total_actual_fpts - total_proj_fpts) AS avg_vs_proj_fpts,
            MAX(total_actual_fpts) AS best_actual_fpts,
            AVG(total_salary) AS avg_salary,
            AVG(
                CASE
                    WHEN total_proj_fpts > 0
                    THEN (100.0 * total_actual_fpts / total_proj_fpts)
                    ELSE NULL
                END
            ) AS proj_capture_pct
        FROM normalized
        GROUP BY model_key
        HAVING COUNT(*) >= ?
        ORDER BY avg_actual_fpts DESC, lineups DESC
        """,
        conn,
        params=[start_date, start_date, end_date, end_date, int(min_lineups)],
    )


# =============================================================================
# Opponent Tracking — Import & Analysis
# =============================================================================

def import_contest_entries(
    conn: sqlite3.Connection,
    slate_date: str,
    contest_id: str,
    entries: List[Dict]
) -> Tuple[int, int]:
    """Import parsed contest entries into opponent tracking tables.

    Args:
        conn: SQLite connection
        slate_date: Date string (YYYY-MM-DD)
        contest_id: Contest identifier (e.g., "187849487")
        entries: List of dicts from parse_contest_entries()

    Returns:
        Tuple of (unique_lineups_imported, unique_users_imported)
    """
    if not entries:
        return 0, 0

    cursor = conn.cursor()

    # Build salary lookup from projections (best-effort enrichment)
    salary_lookup = {}
    proj_rows = cursor.execute(
        "SELECT player_name, salary FROM dfs_slate_projections WHERE slate_date = ?",
        (slate_date,)
    ).fetchall()
    for pname, sal in proj_rows:
        salary_lookup[_normalize_name(pname)] = sal

    users = set()
    imported = 0

    for entry in entries:
        username = entry['username']
        users.add(username)

        # Best-effort salary enrichment
        total_salary = None
        pos_names = [entry.get(p, '') for p in ['pg', 'sg', 'sf', 'pf', 'c', 'g', 'f', 'util']]
        salaries = []
        for pn in pos_names:
            if pn:
                norm = _normalize_name(pn)
                if norm in salary_lookup:
                    salaries.append(salary_lookup[norm])
        if len(salaries) == 8:
            total_salary = sum(salaries)

        cursor.execute("""
            INSERT OR REPLACE INTO dfs_contest_entries (
                contest_id, slate_date, username, max_entries, entry_count,
                rank, points, lineup_raw,
                pg, sg, sf, pf, c, g, f, util,
                total_salary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contest_id, slate_date, username, entry['max_entries'],
            entry['entry_count'], entry['rank'], entry['points'],
            entry['lineup_raw'],
            entry.get('pg', ''), entry.get('sg', ''), entry.get('sf', ''),
            entry.get('pf', ''), entry.get('c', ''), entry.get('g', ''),
            entry.get('f', ''), entry.get('util', ''),
            total_salary,
        ))
        imported += 1

    # Upsert contest metadata
    total_entry_count = sum(e['entry_count'] for e in entries)
    top_score = max((e['points'] for e in entries), default=0)
    cursor.execute("""
        INSERT OR REPLACE INTO dfs_contest_meta (
            contest_id, slate_date, total_entries, unique_users, top_score
        ) VALUES (?, ?, ?, ?, ?)
    """, (contest_id, slate_date, total_entry_count, len(users), top_score))

    conn.commit()
    return imported, len(users)


def _normalize_name(name: str) -> str:
    """Normalize a player name for matching (lowercase, strip accents/suffixes)."""
    n = name.strip().lower()
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        if n.endswith(suffix):
            n = n[:-len(suffix)].strip()
    n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode('ascii')
    return n


def _stack_shape_from_teams(teams: List[str]) -> Tuple[int, int, str]:
    """Summarize lineup team concentration as distinct teams, max stack, and shape."""
    clean_teams = [str(team).strip() for team in teams if str(team).strip()]
    if not clean_teams:
        return 0, 0, ""

    counts = Counter(clean_teams)
    ordered = sorted(counts.values(), reverse=True)
    return len(counts), int(ordered[0]), "-".join(str(v) for v in ordered)


def _pairwise_overlap_summary(lineup_sets: List[set]) -> Dict[str, Optional[float]]:
    """Return average/min/max shared-player overlap across a lineup portfolio."""
    clean_sets = [set(s) for s in lineup_sets if s]
    if len(clean_sets) < 2:
        return {'avg': None, 'min': None, 'max': None}

    overlaps: List[int] = []
    for idx, left in enumerate(clean_sets[:-1]):
        for right in clean_sets[idx + 1:]:
            overlaps.append(len(left & right))

    if not overlaps:
        return {'avg': None, 'min': None, 'max': None}

    return {
        'avg': float(np.mean(overlaps)),
        'min': float(np.min(overlaps)),
        'max': float(np.max(overlaps)),
    }


def _normalized_entropy(counts: List[float]) -> Optional[float]:
    """Return normalized Shannon entropy for a non-negative count vector."""
    clean = np.asarray(
        [float(c) for c in counts if pd.notna(c) and float(c) > 0],
        dtype=float,
    )
    if clean.size <= 1:
        return None

    probs = clean / clean.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(clean.size))
    if max_entropy <= 0:
        return None
    return entropy / max_entropy


def get_opponent_contest_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get list of imported contests with metadata."""
    return pd.read_sql_query(
        "SELECT contest_id, slate_date, total_entries, unique_users, top_score, "
        "import_date FROM dfs_contest_meta ORDER BY slate_date DESC",
        conn
    )


def get_shark_users(
    conn: sqlite3.Connection,
    min_contests: int = 3,
    min_top_pct: float = 25.0
) -> pd.DataFrame:
    """Identify shark users across imported contests.

    Sharks are users who appear in min_contests+ contests OR whose average
    percentile finish is in the top min_top_pct%.

    Returns:
        DataFrame with columns: username, contests, avg_pts, avg_pctile,
        best_rank, typical_max_entries, total_entries
    """
    return pd.read_sql_query("""
        WITH user_stats AS (
            SELECT
                e.username,
                COUNT(DISTINCT e.contest_id) AS contests,
                AVG(e.points) AS avg_pts,
                AVG(100.0 * e.rank / NULLIF(m.total_entries, 0)) AS avg_pctile,
                MIN(e.rank) AS best_rank,
                MAX(e.max_entries) AS typical_max_entries,
                SUM(e.entry_count) AS total_entries
            FROM dfs_contest_entries e
            JOIN dfs_contest_meta m ON e.contest_id = m.contest_id
            WHERE e.points > 0
            GROUP BY e.username
        )
        SELECT * FROM user_stats
        WHERE contests >= ? OR avg_pctile <= ?
        ORDER BY avg_pctile ASC
    """, conn, params=[min_contests, min_top_pct])


def get_shark_player_exposure(
    conn: sqlite3.Connection,
    usernames: List[str]
) -> pd.DataFrame:
    """Get player exposure for a list of shark users.

    Unpivots the 8 lineup position columns and counts how frequently
    each player appears in shark lineups. Joins with projections to get
    average salary.

    Returns:
        DataFrame with columns: player, times_rostered, contests_in,
        exposure_pct, avg_salary
    """
    if not usernames:
        return pd.DataFrame()

    placeholders = ','.join(['?'] * len(usernames))

    return pd.read_sql_query(f"""
        WITH rostered AS (
            SELECT e.username, e.contest_id, e.slate_date, e.pg AS player FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.pg != ''
            UNION ALL SELECT e.username, e.contest_id, e.slate_date, e.sg FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.sg != ''
            UNION ALL SELECT e.username, e.contest_id, e.slate_date, e.sf FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.sf != ''
            UNION ALL SELECT e.username, e.contest_id, e.slate_date, e.pf FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.pf != ''
            UNION ALL SELECT e.username, e.contest_id, e.slate_date, e.c FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.c != ''
            UNION ALL SELECT e.username, e.contest_id, e.slate_date, e.g FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.g != ''
            UNION ALL SELECT e.username, e.contest_id, e.slate_date, e.f FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.f != ''
            UNION ALL SELECT e.username, e.contest_id, e.slate_date, e.util FROM dfs_contest_entries e
                WHERE e.username IN ({placeholders}) AND e.util != ''
        ),
        total_lineups AS (
            SELECT COUNT(*) AS cnt FROM dfs_contest_entries
            WHERE username IN ({placeholders})
        )
        SELECT
            r.player,
            COUNT(*) AS times_rostered,
            COUNT(DISTINCT r.contest_id) AS contests_in,
            ROUND(100.0 * COUNT(*) / NULLIF(t.cnt, 0), 1) AS exposure_pct,
            ROUND(AVG(p.salary), 0) AS avg_salary
        FROM rostered r
        CROSS JOIN total_lineups t
        LEFT JOIN dfs_slate_projections p
            ON r.slate_date = p.slate_date
            AND LOWER(r.player) = LOWER(p.player_name)
        GROUP BY r.player
        ORDER BY times_rostered DESC
        LIMIT 30
    """, conn, params=usernames * 9)  # 8 UNION ALL + 1 total_lineups = 9 sets of placeholders


# =============================================================================
# Prediction Accuracy Analysis (for Model Improvement)
# =============================================================================

def analyze_prediction_accuracy(
    conn: sqlite3.Connection,
    days: int = 30
) -> Dict:
    """
    Analyze prediction accuracy to identify systematic biases by tier, position, and opponent.

    This diagnostic function helps identify where the model is over/under-predicting
    to guide coefficient calibration and model improvements.

    Args:
        conn: SQLite database connection
        days: Number of days to analyze (default 30)

    Returns:
        Dict with comprehensive accuracy analysis:
        - tier_accuracy: MAE and bias by projected PPG tier
        - confidence_accuracy: MAE by confidence bucket
        - floor_ceiling_hit_rates: How often actual falls within projected range
        - position_accuracy: MAE by position (if available)
        - recent_trends: 7-day rolling MAE trend
    """
    cursor = conn.cursor()
    results = {
        'tier_accuracy': [],
        'confidence_accuracy': [],
        'floor_ceiling_hit_rates': [],
        'position_accuracy': [],
        'recent_trends': [],
        'overall_stats': {},
        'errors': []
    }

    try:
        # Query 1: MAE by projected PPG tier
        tier_query = """
            SELECT
                CASE
                    WHEN proj_fpts >= 40 THEN 'Star (40+ FPTS)'
                    WHEN proj_fpts >= 30 THEN 'Starter (30-40 FPTS)'
                    WHEN proj_fpts >= 20 THEN 'Role (20-30 FPTS)'
                    ELSE 'Bench (<20 FPTS)'
                END as tier,
                COUNT(*) as n,
                ROUND(AVG(actual_fpts - proj_fpts), 2) as bias,
                ROUND(AVG(ABS(actual_fpts - proj_fpts)), 2) as mae,
                ROUND(AVG(proj_fpts), 1) as avg_projected,
                ROUND(AVG(actual_fpts), 1) as avg_actual
            FROM dfs_slate_projections
            WHERE actual_fpts IS NOT NULL
            AND did_play = 1
            AND slate_date >= date('now', '-' || ? || ' days')
            GROUP BY tier
            ORDER BY avg_projected DESC
        """

        for row in cursor.execute(tier_query, (days,)).fetchall():
            results['tier_accuracy'].append({
                'tier': row[0],
                'n': row[1],
                'bias': row[2],
                'mae': row[3],
                'avg_projected': row[4],
                'avg_actual': row[5]
            })

        # Query 2: Floor/Ceiling hit rates by tier
        floor_ceiling_query = """
            SELECT
                CASE
                    WHEN proj_fpts >= 40 THEN 'Star (40+ FPTS)'
                    WHEN proj_fpts >= 30 THEN 'Starter (30-40 FPTS)'
                    WHEN proj_fpts >= 20 THEN 'Role (20-30 FPTS)'
                    ELSE 'Bench (<20 FPTS)'
                END as tier,
                COUNT(*) as n,
                SUM(CASE WHEN actual_fpts >= proj_floor AND actual_fpts <= proj_ceiling THEN 1 ELSE 0 END) as within_range,
                SUM(CASE WHEN actual_fpts < proj_floor THEN 1 ELSE 0 END) as below_floor,
                SUM(CASE WHEN actual_fpts > proj_ceiling THEN 1 ELSE 0 END) as above_ceiling,
                ROUND(AVG(proj_floor), 1) as avg_floor,
                ROUND(AVG(proj_ceiling), 1) as avg_ceiling,
                ROUND(AVG(actual_fpts), 1) as avg_actual
            FROM dfs_slate_projections
            WHERE actual_fpts IS NOT NULL
            AND did_play = 1
            AND proj_floor IS NOT NULL
            AND proj_ceiling IS NOT NULL
            AND slate_date >= date('now', '-' || ? || ' days')
            GROUP BY tier
            ORDER BY avg_actual DESC
        """

        for row in cursor.execute(floor_ceiling_query, (days,)).fetchall():
            n = row[1]
            within = row[2]
            below = row[3]
            above = row[4]

            results['floor_ceiling_hit_rates'].append({
                'tier': row[0],
                'n': n,
                'within_range_pct': round(100 * within / n, 1) if n > 0 else 0,
                'below_floor_pct': round(100 * below / n, 1) if n > 0 else 0,
                'above_ceiling_pct': round(100 * above / n, 1) if n > 0 else 0,
                'avg_floor': row[5],
                'avg_ceiling': row[6],
                'avg_actual': row[7]
            })

        # Query 3: Overall statistics
        overall_query = """
            SELECT
                COUNT(*) as total_predictions,
                ROUND(AVG(actual_fpts - proj_fpts), 2) as overall_bias,
                ROUND(AVG(ABS(actual_fpts - proj_fpts)), 2) as overall_mae,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN actual_fpts >= proj_floor AND actual_fpts <= proj_ceiling
                        THEN 1 ELSE 0
                    END) / COUNT(*),
                1) as overall_hit_rate
            FROM dfs_slate_projections
            WHERE actual_fpts IS NOT NULL
            AND did_play = 1
            AND slate_date >= date('now', '-' || ? || ' days')
        """

        overall = cursor.execute(overall_query, (days,)).fetchone()
        if overall:
            results['overall_stats'] = {
                'total_predictions': overall[0],
                'overall_bias': overall[1],
                'overall_mae': overall[2],
                'overall_hit_rate': overall[3],
                'days_analyzed': days
            }

        # Query 4: 7-day rolling MAE trend (for monitoring improvement)
        trend_query = """
            SELECT
                slate_date,
                COUNT(*) as n,
                ROUND(AVG(actual_fpts - proj_fpts), 2) as bias,
                ROUND(AVG(ABS(actual_fpts - proj_fpts)), 2) as mae,
                ROUND(
                    100.0 * SUM(CASE
                        WHEN actual_fpts >= proj_floor AND actual_fpts <= proj_ceiling
                        THEN 1 ELSE 0
                    END) / COUNT(*),
                1) as hit_rate
            FROM dfs_slate_projections
            WHERE actual_fpts IS NOT NULL
            AND did_play = 1
            AND slate_date >= date('now', '-' || ? || ' days')
            GROUP BY slate_date
            ORDER BY slate_date DESC
            LIMIT 14
        """

        for row in cursor.execute(trend_query, (days,)).fetchall():
            results['recent_trends'].append({
                'date': row[0],
                'n': row[1],
                'bias': row[2],
                'mae': row[3],
                'hit_rate': row[4]
            })

    except Exception as e:
        results['errors'].append(f"Analysis error: {str(e)}")

    return results


def learn_floor_ceiling_from_data(
    conn: sqlite3.Connection,
    days: int = 60
) -> Dict[str, Tuple[float, float]]:
    """
    Learn actual 10th/90th percentile ratios by tier from historical data.

    This provides data-driven floor/ceiling multipliers instead of hardcoded values.
    The multipliers represent what percentage of the projection the actual
    10th percentile (floor) and 90th percentile (ceiling) outcomes were.

    Args:
        conn: SQLite database connection
        days: Number of days of history to analyze (default 60)

    Returns:
        Dict mapping tier -> (floor_multiplier, ceiling_multiplier)
        Example: {'Star (40+ FPTS)': (0.72, 1.45), ...}

        floor_multiplier: actual_10th_pct / avg_projection
        ceiling_multiplier: actual_90th_pct / avg_projection
    """
    cursor = conn.cursor()
    results = {}

    # Define tiers with projection ranges
    tiers = [
        ('Star (40+ FPTS)', 40, 999),
        ('Starter (30-40 FPTS)', 30, 40),
        ('Role (20-30 FPTS)', 20, 30),
        ('Bench (<20 FPTS)', 0, 20),
    ]

    for tier_name, min_proj, max_proj in tiers:
        try:
            # Get all actual/projected pairs for this tier
            query = """
                SELECT proj_fpts, actual_fpts
                FROM dfs_slate_projections
                WHERE actual_fpts IS NOT NULL
                AND did_play = 1
                AND proj_fpts >= ? AND proj_fpts < ?
                AND slate_date >= date('now', '-' || ? || ' days')
            """

            rows = cursor.execute(query, (min_proj, max_proj, days)).fetchall()

            if len(rows) >= 20:  # Need minimum sample size
                projections = [r[0] for r in rows]
                actuals = [r[1] for r in rows]

                avg_projection = np.mean(projections)

                # Calculate actual 10th and 90th percentiles
                actual_10th = np.percentile(actuals, 10)
                actual_90th = np.percentile(actuals, 90)

                # Calculate multipliers relative to average projection
                floor_mult = actual_10th / avg_projection if avg_projection > 0 else 0.8
                ceiling_mult = actual_90th / avg_projection if avg_projection > 0 else 1.3

                results[tier_name] = {
                    'floor_multiplier': round(floor_mult, 3),
                    'ceiling_multiplier': round(ceiling_mult, 3),
                    'sample_size': len(rows),
                    'avg_projection': round(avg_projection, 1),
                    'actual_10th_pct': round(actual_10th, 1),
                    'actual_90th_pct': round(actual_90th, 1),
                    'actual_median': round(np.median(actuals), 1)
                }
            else:
                # Insufficient data - use defaults
                results[tier_name] = {
                    'floor_multiplier': 0.80,
                    'ceiling_multiplier': 1.35,
                    'sample_size': len(rows),
                    'note': 'Insufficient data - using defaults'
                }

        except Exception as e:
            results[tier_name] = {
                'floor_multiplier': 0.80,
                'ceiling_multiplier': 1.35,
                'error': str(e)
            }

    return results


def get_prediction_bias_by_team(
    conn: sqlite3.Connection,
    days: int = 30
) -> pd.DataFrame:
    """
    Analyze prediction bias by opponent team to identify defensive mismatches.

    Helps identify teams against which the model consistently over/under-predicts.

    Returns:
        DataFrame with columns: opponent, n, bias, mae
    """
    query = """
        SELECT
            opponent as team,
            COUNT(*) as n,
            ROUND(AVG(actual_fpts - proj_fpts), 2) as bias,
            ROUND(AVG(ABS(actual_fpts - proj_fpts)), 2) as mae
        FROM dfs_slate_projections
        WHERE actual_fpts IS NOT NULL
        AND did_play = 1
        AND slate_date >= date('now', '-' || ? || ' days')
        GROUP BY opponent
        HAVING COUNT(*) >= 5
        ORDER BY bias DESC
    """

    return pd.read_sql_query(query, conn, params=[days])


def validate_model_changes(conn: sqlite3.Connection, days: int = 7) -> Dict:
    """
    Quick validation check for model changes.

    Run this after implementing model improvements to verify:
    1. MAE is improving (or at least not getting worse)
    2. Floor-ceiling hit rate is improving
    3. Bias is closer to zero

    Returns:
        Dict with validation metrics and pass/fail indicators
    """
    analysis = analyze_prediction_accuracy(conn, days)

    overall = analysis.get('overall_stats', {})

    validation = {
        'mae': overall.get('overall_mae', 0),
        'hit_rate': overall.get('overall_hit_rate', 0),
        'bias': overall.get('overall_bias', 0),
        'n': overall.get('total_predictions', 0),
        'checks': {}
    }

    # Check thresholds
    validation['checks']['mae_acceptable'] = validation['mae'] < 12.0  # Target: <12 FPTS MAE
    validation['checks']['hit_rate_acceptable'] = validation['hit_rate'] > 45.0  # Target: >45%
    validation['checks']['bias_acceptable'] = abs(validation['bias']) < 2.0  # Target: within ±2

    validation['overall_pass'] = all(validation['checks'].values())

    return validation


def get_shark_strategy_profile(
    conn: sqlite3.Connection,
    username: str
) -> Dict:
    """Get detailed strategy profile for a single shark user.

    Returns dict with:
        - score_stats: avg, std, min, max of points
        - salary_stats: avg, min, max total salary
        - contests: number of contests
        - total_lineups: number of unique lineups
        - favorite_players: list of (player, count) tuples
        - lineup_history: list of dicts (contest_id, slate_date, rank, points, lineup players)
    """
    # Score & salary stats
    stats_df = pd.read_sql_query(
        "SELECT points, total_salary, rank, contest_id, slate_date "
        "FROM dfs_contest_entries WHERE username = ? AND points > 0 "
        "ORDER BY slate_date DESC, rank ASC",
        conn, params=[username]
    )

    if stats_df.empty:
        return {}

    score_stats = {
        'avg': float(stats_df['points'].mean()),
        'std': float(stats_df['points'].std()) if len(stats_df) > 1 else 0.0,
        'min': float(stats_df['points'].min()),
        'max': float(stats_df['points'].max()),
    }

    sal_valid = stats_df['total_salary'].dropna()
    salary_stats = {
        'avg': float(sal_valid.mean()) if not sal_valid.empty else None,
        'min': int(sal_valid.min()) if not sal_valid.empty else None,
        'max': int(sal_valid.max()) if not sal_valid.empty else None,
    }

    # Favorite players (unpivot with salary join)
    fav_df = pd.read_sql_query("""
        WITH rostered AS (
            SELECT e.slate_date, e.pg AS player FROM dfs_contest_entries e WHERE e.username = ? AND e.pg != ''
            UNION ALL SELECT e.slate_date, e.sg FROM dfs_contest_entries e WHERE e.username = ? AND e.sg != ''
            UNION ALL SELECT e.slate_date, e.sf FROM dfs_contest_entries e WHERE e.username = ? AND e.sf != ''
            UNION ALL SELECT e.slate_date, e.pf FROM dfs_contest_entries e WHERE e.username = ? AND e.pf != ''
            UNION ALL SELECT e.slate_date, e.c FROM dfs_contest_entries e WHERE e.username = ? AND e.c != ''
            UNION ALL SELECT e.slate_date, e.g FROM dfs_contest_entries e WHERE e.username = ? AND e.g != ''
            UNION ALL SELECT e.slate_date, e.f FROM dfs_contest_entries e WHERE e.username = ? AND e.f != ''
            UNION ALL SELECT e.slate_date, e.util FROM dfs_contest_entries e WHERE e.username = ? AND e.util != ''
        )
        SELECT r.player, COUNT(*) AS times_used, ROUND(AVG(p.salary), 0) AS avg_salary
        FROM rostered r
        LEFT JOIN dfs_slate_projections p
            ON r.slate_date = p.slate_date
            AND LOWER(r.player) = LOWER(p.player_name)
        GROUP BY r.player
        ORDER BY times_used DESC
        LIMIT 15
    """, conn, params=[username] * 8)

    # Lineup history
    history = []
    for _, row in stats_df.iterrows():
        history.append({
            'contest_id': row['contest_id'],
            'slate_date': row['slate_date'],
            'rank': int(row['rank']) if pd.notna(row['rank']) else None,
            'points': float(row['points']),
            'salary': int(row['total_salary']) if pd.notna(row['total_salary']) else None,
        })

    return {
        'username': username,
        'contests': int(stats_df['contest_id'].nunique()),
        'total_lineups': len(stats_df),
        'score_stats': score_stats,
        'salary_stats': salary_stats,
        'favorite_players': list(fav_df.itertuples(index=False, name=None)),
        'lineup_history': history,
    }


# =============================================================================
# Tournament Postmortem Review
# =============================================================================

def build_tournament_postmortem(
    conn: sqlite3.Connection,
    slate_date: str,
    top_n: int = 50,
    core_field_exposure_pct: float = 35.0,
    underexposure_ratio: float = 0.75,
    standout_min_error: float = 12.0,
) -> Dict[str, Any]:
    """Build a postmortem review packet for a DFS tournament slate.

    Compares top field lineups vs. our generated lineups and identifies:
    - winner gap and top-end capture
    - field-vs-our player exposure differences
    - missed core plays and standout outcomes
    - raw lineup structure and portfolio concentration diagnostics
    - concise "what went right / wrong" notes and next-slate actions
    """
    top_n = max(1, int(top_n))
    core_field_exposure_pct = float(max(0.0, min(100.0, core_field_exposure_pct)))
    underexposure_ratio = float(max(0.0, min(1.0, underexposure_ratio)))
    standout_min_error = float(standout_min_error)

    result: Dict[str, Any] = {
        'slate_date': slate_date,
        'top_n': top_n,
        'metrics': {},
        'top_field_players_df': pd.DataFrame(),
        'exposure_comparison_df': pd.DataFrame(),
        'missed_core_df': pd.DataFrame(),
        'missed_standouts_df': pd.DataFrame(),
        'ownership_polarity_df': pd.DataFrame(),
        'overexposed_duds_df': pd.DataFrame(),
        'value_tier_misses_df': pd.DataFrame(),
        'combo_capture_misses_df': pd.DataFrame(),
        'team_concentration_mismatch_df': pd.DataFrame(),
        'field_lineup_structures_df': pd.DataFrame(),
        'our_lineup_structures_df': pd.DataFrame(),
        'portfolio_diagnostics_df': pd.DataFrame(),
        'model_strategy_breakdown_df': pd.DataFrame(),
        'model_signal_coverage_df': pd.DataFrame(),
        'improvements_df': pd.DataFrame(),
        'right_notes': [],
        'wrong_notes': [],
        'errors': [],
    }

    try:
        # 1) Top-N field lineups from contest standings
        top_entries_df = pd.read_sql_query(
            """
            SELECT rank, points, username, total_salary, pg, sg, sf, pf, c, g, f, util
            FROM dfs_contest_entries
            WHERE slate_date = ? AND rank IS NOT NULL
            ORDER BY rank ASC
            LIMIT ?
            """,
            conn,
            params=[slate_date, top_n],
        )
        if top_entries_df.empty:
            result['errors'].append("No contest standings found for this slate.")
            return result

        field_lineups = int(len(top_entries_df))
        top_entries_df = top_entries_df.reset_index(drop=True).copy()
        top_entries_df['field_lineup_num'] = np.arange(1, field_lineups + 1)
        pos_cols = ['pg', 'sg', 'sf', 'pf', 'c', 'g', 'f', 'util']
        field_long = top_entries_df[['field_lineup_num'] + pos_cols].melt(
            id_vars=['field_lineup_num'],
            var_name='slot',
            value_name='player',
        )
        field_long['player'] = field_long['player'].astype(str).str.strip()
        field_long = field_long[field_long['player'].ne('') & field_long['player'].ne('nan')]
        if field_long.empty:
            result['errors'].append("Contest standings have no parsed lineup player names.")
            return result
        field_long['name_key'] = field_long['player'].map(_normalize_name)
        field_lineup_name_sets = (
            field_long.groupby('field_lineup_num')['name_key']
            .apply(lambda s: {x for x in s.tolist() if x})
            .tolist()
        )

        field_exposure_df = (
            field_long.groupby('name_key', as_index=False)
            .agg(
                field_slots=('player', 'size'),
                field_player=('player', lambda s: s.value_counts().index[0]),
            )
        )
        field_exposure_df['field_exposure_pct'] = (
            100.0 * field_exposure_df['field_slots'] / float(max(1, field_lineups))
        )

        # 2) Our generated lineup exposures on this slate
        our_lineups_df = pd.read_sql_query(
            """
            SELECT lineup_num,
                   COALESCE(NULLIF(model_key, ''), 'standard_v1') AS model_key,
                   COALESCE(NULLIF(model_label, ''), COALESCE(NULLIF(model_key, ''), 'standard_v1')) AS model_label,
                   COALESCE(generation_strategy, '') AS generation_strategy,
                   total_proj_fpts,
                   total_salary,
                   total_actual_fpts,
                   pg_id, sg_id, sf_id, pf_id, c_id, g_id, f_id, util_id
            FROM dfs_slate_lineups
            WHERE slate_date = ?
            ORDER BY lineup_num
            """,
            conn,
            params=[slate_date],
        )
        our_lineups = int(len(our_lineups_df))

        id_map_df = pd.read_sql_query(
            """
            SELECT DISTINCT player_id, player_name
            FROM dfs_slate_projections
            WHERE slate_date = ? AND player_id IS NOT NULL
            """,
            conn,
            params=[slate_date],
        )
        id_to_name = {}
        for _, row in id_map_df.iterrows():
            pid = row.get('player_id')
            if pd.notna(pid):
                try:
                    id_to_name[int(pid)] = row.get('player_name')
                except (TypeError, ValueError):
                    continue

        id_cols = ['pg_id', 'sg_id', 'sf_id', 'pf_id', 'c_id', 'g_id', 'f_id', 'util_id']
        our_name_cols: List[str] = []
        if not our_lineups_df.empty:
            def _map_player_name(pid: Any) -> Optional[str]:
                if pd.isna(pid):
                    return None
                text = str(pid).strip()
                if not text:
                    return None
                try:
                    return id_to_name.get(int(float(text)))
                except (TypeError, ValueError):
                    return None

            for col in id_cols:
                name_col = f"{col}_name"
                our_name_cols.append(name_col)
                our_lineups_df[name_col] = our_lineups_df[col].apply(_map_player_name)

        our_exposure_df = pd.DataFrame(columns=['name_key', 'our_slots', 'our_player', 'our_exposure_pct'])
        our_long = pd.DataFrame(columns=['lineup_num', 'slot', 'player', 'name_key'])
        lineup_name_sets: List[set] = []
        if not our_lineups_df.empty:
            for col in ['model_key', 'model_label', 'generation_strategy']:
                our_lineups_df[col] = our_lineups_df[col].fillna('').astype(str).str.strip()
            our_lineups_df['model_key'] = our_lineups_df['model_key'].replace('', 'standard_v1')
            our_lineups_df['model_label'] = our_lineups_df['model_label'].where(
                our_lineups_df['model_label'].ne(''),
                our_lineups_df['model_key'],
            )
        if our_lineups > 0 and our_name_cols:
            our_long = our_lineups_df[['lineup_num'] + our_name_cols].melt(
                id_vars=['lineup_num'],
                var_name='slot',
                value_name='player',
            )
            our_long['player'] = our_long['player'].astype(str).str.strip()
            our_long = our_long[our_long['player'].ne('') & our_long['player'].ne('nan')]
            if not our_long.empty:
                our_long['name_key'] = our_long['player'].map(_normalize_name)
                our_exposure_df = (
                    our_long.groupby('name_key', as_index=False)
                    .agg(
                        our_slots=('player', 'size'),
                        our_player=('player', lambda s: s.value_counts().index[0]),
                    )
                )
                our_exposure_df['our_exposure_pct'] = (
                    100.0 * our_exposure_df['our_slots'] / float(max(1, our_lineups))
                )
                lineup_name_sets = (
                    our_long.groupby('lineup_num')['name_key']
                    .apply(lambda s: {x for x in s.tolist() if x})
                    .tolist()
                )

        # 3) Player context (projection, actuals, ownership) for enrichment + metrics
        player_context_df = pd.read_sql_query(
            """
            SELECT player_name, team, salary, proj_fpts, actual_fpts,
                   ownership_proj, actual_ownership, did_play
            FROM dfs_slate_projections
            WHERE slate_date = ?
            """,
            conn,
            params=[slate_date],
        )
        player_context_agg = pd.DataFrame(
            columns=[
                'name_key', 'player_name', 'team', 'salary',
                'proj_fpts', 'actual_fpts', 'actual_minus_proj',
                'ownership_proj', 'actual_ownership',
            ]
        )
        perf_df = pd.DataFrame()
        perf_agg = pd.DataFrame(
            columns=[
                'name_key', 'player_name', 'team', 'salary',
                'proj_fpts', 'actual_fpts', 'actual_minus_proj',
                'ownership_proj', 'actual_ownership',
            ]
        )
        if not player_context_df.empty:
            player_context_df['name_key'] = player_context_df['player_name'].map(_normalize_name)
            player_context_df['actual_minus_proj'] = (
                pd.to_numeric(player_context_df['actual_fpts'], errors='coerce')
                - pd.to_numeric(player_context_df['proj_fpts'], errors='coerce')
            )
            player_context_agg = (
                player_context_df.groupby('name_key', as_index=False)
                .agg(
                    player_name=('player_name', 'first'),
                    team=('team', 'first'),
                    salary=('salary', 'first'),
                    proj_fpts=('proj_fpts', 'first'),
                    actual_fpts=('actual_fpts', 'first'),
                    actual_minus_proj=('actual_minus_proj', 'first'),
                    ownership_proj=('ownership_proj', 'first'),
                    actual_ownership=('actual_ownership', 'first'),
                )
            )
            perf_df = player_context_df[
                (pd.to_numeric(player_context_df['did_play'], errors='coerce') == 1)
                & pd.to_numeric(player_context_df['actual_fpts'], errors='coerce').notna()
            ].copy()
            if not perf_df.empty:
                perf_agg = (
                    perf_df.groupby('name_key', as_index=False)
                    .agg(
                        player_name=('player_name', 'first'),
                        team=('team', 'first'),
                        salary=('salary', 'first'),
                        proj_fpts=('proj_fpts', 'first'),
                        actual_fpts=('actual_fpts', 'first'),
                        actual_minus_proj=('actual_minus_proj', 'first'),
                        ownership_proj=('ownership_proj', 'first'),
                        actual_ownership=('actual_ownership', 'first'),
                    )
                )

        context_cols = [
            'name_key',
            'team',
            'salary',
            'proj_fpts',
            'actual_fpts',
            'actual_minus_proj',
            'ownership_proj',
            'actual_ownership',
        ]
        if not field_long.empty and not player_context_agg.empty:
            field_long = field_long.merge(
                player_context_agg[context_cols],
                on='name_key',
                how='left',
            )
        if not our_long.empty and not player_context_agg.empty:
            our_long = our_long.merge(
                player_context_agg[context_cols],
                on='name_key',
                how='left',
            )
        if not our_long.empty:
            our_long = our_long.merge(
                our_lineups_df[['lineup_num', 'model_key', 'model_label', 'generation_strategy']],
                on='lineup_num',
                how='left',
            )

        # 4) Exposure comparison frame
        exposure_df = field_exposure_df.merge(
            our_exposure_df[['name_key', 'our_slots', 'our_player', 'our_exposure_pct']],
            on='name_key',
            how='outer',
        ).merge(
            perf_agg,
            on='name_key',
            how='left',
        )

        for num_col in ['field_slots', 'field_exposure_pct', 'our_slots', 'our_exposure_pct']:
            exposure_df[num_col] = pd.to_numeric(exposure_df.get(num_col), errors='coerce').fillna(0.0)

        exposure_df['display_name'] = (
            exposure_df.get('field_player')
            .fillna(exposure_df.get('our_player'))
            .fillna(exposure_df.get('player_name'))
            .fillna(exposure_df.get('name_key'))
        )
        exposure_df['exposure_gap_pct'] = exposure_df['field_exposure_pct'] - exposure_df['our_exposure_pct']
        exposure_df = exposure_df.sort_values(
            ['field_exposure_pct', 'actual_minus_proj'],
            ascending=[False, False],
        )

        top_field_players_df = exposure_df[exposure_df['field_slots'] > 0].copy()

        missed_core_df = top_field_players_df[
            (top_field_players_df['field_exposure_pct'] >= core_field_exposure_pct)
            & (
                top_field_players_df['our_exposure_pct']
                < (top_field_players_df['field_exposure_pct'] * underexposure_ratio)
            )
        ].copy().sort_values('exposure_gap_pct', ascending=False)

        missed_standouts_df = top_field_players_df[
            (pd.to_numeric(top_field_players_df['actual_minus_proj'], errors='coerce') >= standout_min_error)
            & (top_field_players_df['field_exposure_pct'] >= 10.0)
            & (top_field_players_df['our_exposure_pct'] < top_field_players_df['field_exposure_pct'])
        ].copy().sort_values(
            ['actual_minus_proj', 'field_exposure_pct'],
            ascending=[False, False],
        )

        field_lineup_structures_df = pd.DataFrame()
        if not top_entries_df.empty:
            field_rows: List[Dict[str, Any]] = []
            field_groups = {
                int(num): grp.copy()
                for num, grp in field_long.groupby('field_lineup_num')
            }
            for _, lineup_row in top_entries_df.iterrows():
                lineup_num = int(lineup_row.get('field_lineup_num') or 0)
                lineup_players_df = field_groups.get(lineup_num, pd.DataFrame())
                ordered_players = [
                    str(lineup_row.get(col) or '').strip()
                    for col in pos_cols
                    if str(lineup_row.get(col) or '').strip()
                ]
                team_count, top_stack_size, stack_shape = _stack_shape_from_teams(
                    lineup_players_df.get('team', pd.Series(dtype=object)).dropna().astype(str).tolist()
                )
                proj_vals = pd.to_numeric(lineup_players_df.get('proj_fpts'), errors='coerce').dropna()
                own_proj_vals = pd.to_numeric(lineup_players_df.get('ownership_proj'), errors='coerce').dropna()
                own_actual_vals = pd.to_numeric(lineup_players_df.get('actual_ownership'), errors='coerce').dropna()
                field_rows.append({
                    'field_lineup_num': lineup_num,
                    'rank': int(lineup_row['rank']) if pd.notna(lineup_row.get('rank')) else None,
                    'username': str(lineup_row.get('username') or '').strip(),
                    'contest_points': float(lineup_row['points']) if pd.notna(lineup_row.get('points')) else None,
                    'total_salary': int(lineup_row['total_salary']) if pd.notna(lineup_row.get('total_salary')) else None,
                    'total_proj_fpts': float(proj_vals.sum()) if not proj_vals.empty else None,
                    'avg_proj_ownership': float(own_proj_vals.mean()) if not own_proj_vals.empty else None,
                    'avg_actual_ownership': float(own_actual_vals.mean()) if not own_actual_vals.empty else None,
                    'team_count': int(team_count),
                    'top_stack_size': int(top_stack_size),
                    'stack_shape': stack_shape,
                    'players': ' | '.join(ordered_players),
                })
            field_lineup_structures_df = pd.DataFrame(field_rows)

        our_lineup_structures_df = pd.DataFrame()
        if not our_lineups_df.empty:
            our_rows: List[Dict[str, Any]] = []
            our_groups = {
                int(num): grp.copy()
                for num, grp in our_long.groupby('lineup_num')
            } if not our_long.empty else {}
            for _, lineup_row in our_lineups_df.iterrows():
                lineup_num = int(lineup_row.get('lineup_num') or 0)
                lineup_players_df = our_groups.get(lineup_num, pd.DataFrame())
                ordered_players = [
                    str(lineup_row.get(col) or '').strip()
                    for col in our_name_cols
                    if str(lineup_row.get(col) or '').strip()
                ]
                team_count, top_stack_size, stack_shape = _stack_shape_from_teams(
                    lineup_players_df.get('team', pd.Series(dtype=object)).dropna().astype(str).tolist()
                )
                own_proj_vals = pd.to_numeric(lineup_players_df.get('ownership_proj'), errors='coerce').dropna()
                own_actual_vals = pd.to_numeric(lineup_players_df.get('actual_ownership'), errors='coerce').dropna()
                our_rows.append({
                    'lineup_num': lineup_num,
                    'model_key': str(lineup_row.get('model_key') or 'standard_v1'),
                    'model_label': str(lineup_row.get('model_label') or lineup_row.get('model_key') or 'standard_v1'),
                    'generation_strategy': str(lineup_row.get('generation_strategy') or ''),
                    'total_proj_fpts': float(lineup_row['total_proj_fpts']) if pd.notna(lineup_row.get('total_proj_fpts')) else None,
                    'total_actual_fpts': float(lineup_row['total_actual_fpts']) if pd.notna(lineup_row.get('total_actual_fpts')) else None,
                    'total_salary': int(lineup_row['total_salary']) if pd.notna(lineup_row.get('total_salary')) else None,
                    'avg_proj_ownership': float(own_proj_vals.mean()) if not own_proj_vals.empty else None,
                    'avg_actual_ownership': float(own_actual_vals.mean()) if not own_actual_vals.empty else None,
                    'team_count': int(team_count),
                    'top_stack_size': int(top_stack_size),
                    'stack_shape': stack_shape,
                    'players': ' | '.join(ordered_players),
                })
            our_lineup_structures_df = pd.DataFrame(our_rows)

        # 4b) Additional postmortem diagnostics
        for num_col in ['salary', 'proj_fpts', 'actual_fpts', 'actual_minus_proj', 'ownership_proj', 'actual_ownership']:
            exposure_df[num_col] = pd.to_numeric(exposure_df.get(num_col), errors='coerce')
            top_field_players_df[num_col] = pd.to_numeric(top_field_players_df.get(num_col), errors='coerce')

        ownership_polarity_df = top_field_players_df.dropna(subset=['actual_ownership', 'ownership_proj']).copy()
        if not ownership_polarity_df.empty:
            ownership_polarity_df['ownership_error_pp'] = (
                ownership_polarity_df['actual_ownership'] - ownership_polarity_df['ownership_proj']
            )
            ownership_polarity_df['ownership_error_abs'] = ownership_polarity_df['ownership_error_pp'].abs()
            ownership_polarity_df['polarity'] = np.where(
                ownership_polarity_df['ownership_error_pp'] >= 8.0,
                'Underprojected Chalk',
                np.where(
                    ownership_polarity_df['ownership_error_pp'] <= -8.0,
                    'Overprojected Chalk',
                    'Neutral',
                ),
            )
            ownership_polarity_df = ownership_polarity_df[
                ownership_polarity_df['ownership_error_abs'] >= 8.0
            ].copy().sort_values(
                ['ownership_error_abs', 'field_exposure_pct'],
                ascending=[False, False],
            )

        overexposed_duds_df = exposure_df.copy()
        if not overexposed_duds_df.empty:
            overexposed_duds_df['our_minus_field_pct'] = (
                overexposed_duds_df['our_exposure_pct'] - overexposed_duds_df['field_exposure_pct']
            )
            overexposed_duds_df = overexposed_duds_df[
                (overexposed_duds_df['our_exposure_pct'] >= 10.0)
                & (overexposed_duds_df['our_minus_field_pct'] >= 8.0)
                & (pd.to_numeric(overexposed_duds_df['actual_minus_proj'], errors='coerce') <= -6.0)
            ].copy().sort_values(
                ['our_minus_field_pct', 'actual_minus_proj'],
                ascending=[False, True],
            )

        value_tier_misses_df = top_field_players_df[
            (pd.to_numeric(top_field_players_df['salary'], errors='coerce') >= 4000)
            & (pd.to_numeric(top_field_players_df['salary'], errors='coerce') <= 5000)
            & (top_field_players_df['field_exposure_pct'] >= 10.0)
        ].copy()
        if not value_tier_misses_df.empty:
            def _value_tier_tag(row: pd.Series) -> str:
                tags: List[str] = []
                if pd.to_numeric(row.get('our_exposure_pct'), errors='coerce') < pd.to_numeric(row.get('field_exposure_pct'), errors='coerce'):
                    tags.append('underexposed')
                if pd.to_numeric(row.get('actual_minus_proj'), errors='coerce') >= standout_min_error:
                    tags.append('standout')
                if pd.to_numeric(row.get('field_exposure_pct'), errors='coerce') >= core_field_exposure_pct:
                    tags.append('core-field')
                return ', '.join(tags)

            value_tier_misses_df['miss_tags'] = value_tier_misses_df.apply(_value_tier_tag, axis=1)
            value_tier_misses_df = value_tier_misses_df[
                value_tier_misses_df['miss_tags'].astype(str).str.len() > 0
            ].copy().sort_values(
                ['field_exposure_pct', 'actual_minus_proj'],
                ascending=[False, False],
            )

        combo_rows: List[Dict[str, Any]] = []
        if field_lineup_name_sets:
            eligible_combo_keys = set(
                top_field_players_df[
                    pd.to_numeric(top_field_players_df['field_exposure_pct'], errors='coerce') >= 20.0
                ]['name_key'].dropna().astype(str).tolist()
            )
            combo_counts = {2: Counter(), 3: Counter()}
            for lineup_keys in field_lineup_name_sets:
                clean_keys = sorted([k for k in lineup_keys if k and k in eligible_combo_keys])
                for combo_size in (2, 3):
                    if len(clean_keys) >= combo_size:
                        for combo in combinations(clean_keys, combo_size):
                            combo_counts[combo_size][combo] += 1

            name_lookup: Dict[str, str] = {}
            actual_lookup: Dict[str, Any] = {}
            for _, row in exposure_df[['name_key', 'display_name', 'actual_fpts']].iterrows():
                key = str(row.get('name_key') or '').strip()
                if not key:
                    continue
                if key not in name_lookup:
                    name_lookup[key] = str(row.get('display_name') or key)
                if key not in actual_lookup:
                    actual_lookup[key] = row.get('actual_fpts')

            min_pair_count = max(2, int(np.ceil(field_lineups * 0.30)))
            min_triple_count = max(2, int(np.ceil(field_lineups * 0.20)))
            min_combo_gap_pct = 10.0

            for combo_size, counter in combo_counts.items():
                min_count = min_pair_count if combo_size == 2 else min_triple_count
                for combo_keys, field_count in counter.items():
                    if field_count < min_count:
                        continue

                    field_combo_pct = 100.0 * float(field_count) / float(max(1, field_lineups))
                    if our_lineups > 0 and lineup_name_sets:
                        our_count = int(sum(1 for s in lineup_name_sets if set(combo_keys).issubset(s)))
                        our_combo_pct = 100.0 * float(our_count) / float(max(1, our_lineups))
                    else:
                        our_count = 0
                        our_combo_pct = 0.0
                    combo_gap_pct = field_combo_pct - our_combo_pct
                    if combo_gap_pct < min_combo_gap_pct:
                        continue

                    combo_players = [name_lookup.get(k, k) for k in combo_keys]
                    actual_vals = [pd.to_numeric(actual_lookup.get(k), errors='coerce') for k in combo_keys]
                    actual_vals = [v for v in actual_vals if pd.notna(v)]
                    combo_actual_sum = float(np.sum(actual_vals)) if actual_vals else None

                    combo_rows.append({
                        'combo_size': int(combo_size),
                        'combo_players': ' + '.join(combo_players),
                        'field_combo_count': int(field_count),
                        'field_combo_pct': float(field_combo_pct),
                        'our_combo_count': int(our_count),
                        'our_combo_pct': float(our_combo_pct),
                        'combo_gap_pct': float(combo_gap_pct),
                        'combo_actual_fpts_sum': combo_actual_sum,
                    })

        combo_capture_misses_df = pd.DataFrame(combo_rows)
        combo_capture_miss_raw_count = 0
        if not combo_capture_misses_df.empty:
            combo_capture_miss_raw_count = int(len(combo_capture_misses_df))
            combo_capture_misses_df = combo_capture_misses_df.sort_values(
                ['combo_gap_pct', 'field_combo_pct', 'combo_size', 'combo_actual_fpts_sum'],
                ascending=[False, False, False, False],
            ).head(60)

        team_lookup_df = player_context_agg[['name_key', 'team']].dropna(subset=['name_key']).copy()
        if not team_lookup_df.empty:
            team_lookup_df['team'] = team_lookup_df['team'].astype(str).str.strip()
            team_lookup_df = team_lookup_df[
                team_lookup_df['team'].ne('')
                & team_lookup_df['team'].ne('nan')
                & team_lookup_df['team'].ne('?')
            ]

        team_concentration_mismatch_df = pd.DataFrame()
        if not team_lookup_df.empty and not field_long.empty:
            field_team_slots = field_long[['field_lineup_num', 'name_key']].merge(
                team_lookup_df,
                on='name_key',
                how='left',
            ).dropna(subset=['team'])
            field_team_agg = pd.DataFrame(columns=['team', 'field_team_slots', 'field_team_lineups'])
            if not field_team_slots.empty:
                field_team_agg = field_team_slots.groupby('team', as_index=False).agg(
                    field_team_slots=('name_key', 'size'),
                    field_team_lineups=('field_lineup_num', 'nunique'),
                )
                field_team_agg['field_team_slot_pct'] = (
                    100.0 * field_team_agg['field_team_slots'] / float(max(1, field_lineups * 8))
                )
                field_team_agg['field_team_lineup_pct'] = (
                    100.0 * field_team_agg['field_team_lineups'] / float(max(1, field_lineups))
                )

            our_team_agg = pd.DataFrame(columns=['team', 'our_team_slots', 'our_team_lineups'])
            if our_lineups > 0 and not our_long.empty:
                our_team_slots = our_long[['lineup_num', 'name_key']].merge(
                    team_lookup_df,
                    on='name_key',
                    how='left',
                ).dropna(subset=['team'])
                if not our_team_slots.empty:
                    our_team_agg = our_team_slots.groupby('team', as_index=False).agg(
                        our_team_slots=('name_key', 'size'),
                        our_team_lineups=('lineup_num', 'nunique'),
                    )
                    our_team_agg['our_team_slot_pct'] = (
                        100.0 * our_team_agg['our_team_slots'] / float(max(1, our_lineups * 8))
                    )
                    our_team_agg['our_team_lineup_pct'] = (
                        100.0 * our_team_agg['our_team_lineups'] / float(max(1, our_lineups))
                    )

            team_conc_df = field_team_agg.merge(
                our_team_agg,
                on='team',
                how='outer',
            )
            for c in [
                'field_team_slots', 'field_team_lineups', 'field_team_slot_pct', 'field_team_lineup_pct',
                'our_team_slots', 'our_team_lineups', 'our_team_slot_pct', 'our_team_lineup_pct',
            ]:
                team_conc_df[c] = pd.to_numeric(team_conc_df.get(c), errors='coerce').fillna(0.0)

            team_conc_df['slot_gap_pct'] = team_conc_df['field_team_slot_pct'] - team_conc_df['our_team_slot_pct']
            team_conc_df['lineup_gap_pct'] = (
                team_conc_df['field_team_lineup_pct'] - team_conc_df['our_team_lineup_pct']
            )
            team_conc_df['abs_slot_gap_pct'] = team_conc_df['slot_gap_pct'].abs()
            team_concentration_mismatch_df = team_conc_df[
                (team_conc_df['abs_slot_gap_pct'] >= 5.0)
                | (team_conc_df['lineup_gap_pct'].abs() >= 20.0)
            ].copy().sort_values(
                ['abs_slot_gap_pct', 'field_team_slot_pct'],
                ascending=[False, False],
            )

        ownership_polarity_count = int(len(ownership_polarity_df))
        underprojected_chalk_count = int(
            ((ownership_polarity_df.get('ownership_error_pp') >= 8.0).sum())
            if not ownership_polarity_df.empty
            else 0
        )
        overprojected_chalk_count = int(
            ((ownership_polarity_df.get('ownership_error_pp') <= -8.0).sum())
            if not ownership_polarity_df.empty
            else 0
        )
        overexposed_dud_count = int(len(overexposed_duds_df))
        value_tier_miss_count = int(len(value_tier_misses_df))
        combo_capture_miss_count = int(combo_capture_miss_raw_count)
        combo_capture_rows_exported = int(len(combo_capture_misses_df))
        team_concentration_mismatch_count = int(len(team_concentration_mismatch_df))
        largest_team_slot_gap = None
        largest_team_slot_gap_team = ""
        if not team_concentration_mismatch_df.empty:
            largest_team_slot_gap = float(team_concentration_mismatch_df.iloc[0].get('slot_gap_pct'))
            largest_team_slot_gap_team = str(team_concentration_mismatch_df.iloc[0].get('team') or '')

        # 5) Score and accuracy metrics
        winner_row = conn.execute(
            """
            SELECT top_score
            FROM dfs_contest_meta
            WHERE slate_date = ?
            ORDER BY import_date DESC
            LIMIT 1
            """,
            (slate_date,),
        ).fetchone()
        winner_score = float(winner_row[0]) if winner_row and pd.notna(winner_row[0]) else None

        our_scores = pd.to_numeric(our_lineups_df.get('total_actual_fpts'), errors='coerce').dropna()
        our_best_score = float(our_scores.max()) if not our_scores.empty else None
        our_avg_score = float(our_scores.mean()) if not our_scores.empty else None
        winner_gap = (
            float(winner_score - our_best_score)
            if winner_score is not None and our_best_score is not None
            else None
        )

        topn_avg_points = float(pd.to_numeric(top_entries_df['points'], errors='coerce').mean())
        topn_best_points = float(pd.to_numeric(top_entries_df['points'], errors='coerce').max())

        proj_mae = None
        proj_rank_corr = None
        if not perf_df.empty:
            proj_errors = (
                pd.to_numeric(perf_df['actual_fpts'], errors='coerce')
                - pd.to_numeric(perf_df['proj_fpts'], errors='coerce')
            )
            proj_mae = float(proj_errors.abs().mean())
            if (
                pd.to_numeric(perf_df['proj_fpts'], errors='coerce').nunique() > 1
                and pd.to_numeric(perf_df['actual_fpts'], errors='coerce').nunique() > 1
            ):
                proj_rank_corr = float(
                    pd.to_numeric(perf_df['proj_fpts'], errors='coerce').corr(
                        pd.to_numeric(perf_df['actual_fpts'], errors='coerce'),
                        method='spearman',
                    )
                )

        ownership_mae = None
        ownership_rank_corr = None
        if not perf_df.empty:
            own_df = perf_df[['ownership_proj', 'actual_ownership']].dropna()
            if not own_df.empty:
                ownership_mae = float(
                    (
                        pd.to_numeric(own_df['actual_ownership'], errors='coerce')
                        - pd.to_numeric(own_df['ownership_proj'], errors='coerce')
                    ).abs().mean()
                )
                if (
                    pd.to_numeric(own_df['ownership_proj'], errors='coerce').nunique() > 1
                    and pd.to_numeric(own_df['actual_ownership'], errors='coerce').nunique() > 1
                ):
                    ownership_rank_corr = float(
                        pd.to_numeric(own_df['ownership_proj'], errors='coerce').corr(
                            pd.to_numeric(own_df['actual_ownership'], errors='coerce'),
                            method='spearman',
                        )
                    )

        our_name_keys = set()
        if not our_exposure_df.empty:
            our_name_keys = set(our_exposure_df['name_key'].tolist())

        top_actual_df = top_field_players_df.dropna(subset=['actual_fpts']).copy()
        top_actual_df = top_actual_df.sort_values('actual_fpts', ascending=False)
        top_scorer_name = ""
        top_scorer_actual = None
        top_scorer_in_our = None
        top3_target_count = 0
        top3_covered_count = 0
        top3_all_in_single = None
        top3_keys: List[str] = []
        if not top_actual_df.empty:
            top_scorer_name = str(top_actual_df.iloc[0].get('display_name') or "")
            top_scorer_actual = pd.to_numeric(top_actual_df.iloc[0].get('actual_fpts'), errors='coerce')
            top_scorer_key = str(top_actual_df.iloc[0].get('name_key') or "")
            top_scorer_in_our = bool(top_scorer_key and top_scorer_key in our_name_keys)

            top3_keys = [k for k in top_actual_df['name_key'].head(3).tolist() if k]
            top3_target_count = len(top3_keys)
            top3_covered_count = int(sum(1 for k in top3_keys if k in our_name_keys))
            if top3_target_count > 0:
                if lineup_name_sets:
                    top3_all_in_single = any(set(top3_keys).issubset(s) for s in lineup_name_sets)
                else:
                    top3_all_in_single = False

        core_signal_keys = set(
            top_field_players_df[
                pd.to_numeric(top_field_players_df['field_exposure_pct'], errors='coerce') >= core_field_exposure_pct
            ]['name_key'].dropna().astype(str).tolist()
        )
        standout_signal_keys = set(
            missed_standouts_df['name_key'].dropna().astype(str).tolist()
        )
        missed_core_keys = set(missed_core_df['name_key'].dropna().astype(str).tolist())
        top3_signal_keys = set(top3_keys)
        high_signal_keys = core_signal_keys | standout_signal_keys | top3_signal_keys

        exposure_df['is_field_core'] = exposure_df['name_key'].astype(str).isin(core_signal_keys)
        exposure_df['is_missed_core'] = exposure_df['name_key'].astype(str).isin(missed_core_keys)
        exposure_df['is_standout'] = (
            pd.to_numeric(exposure_df['actual_minus_proj'], errors='coerce') >= standout_min_error
        ) & (pd.to_numeric(exposure_df['field_exposure_pct'], errors='coerce') >= 10.0)
        exposure_df['is_top3_actual'] = exposure_df['name_key'].astype(str).isin(top3_signal_keys)
        exposure_df['is_value_tier'] = (
            pd.to_numeric(exposure_df['salary'], errors='coerce').between(4000, 5000, inclusive='both')
        )

        def _build_signal_tags(row: pd.Series) -> str:
            tags: List[str] = []
            if bool(row.get('is_field_core')):
                tags.append('core')
            if bool(row.get('is_missed_core')):
                tags.append('missed-core')
            if bool(row.get('is_standout')):
                tags.append('standout')
            if bool(row.get('is_top3_actual')):
                tags.append('top3-actual')
            if bool(row.get('is_value_tier')):
                tags.append('value-tier')
            return ', '.join(tags)

        exposure_df['signal_tags'] = exposure_df.apply(_build_signal_tags, axis=1)
        top_field_players_df = exposure_df[exposure_df['field_slots'] > 0].copy()

        field_overlap = _pairwise_overlap_summary(field_lineup_name_sets)
        our_overlap = _pairwise_overlap_summary(lineup_name_sets)
        field_entropy = _normalized_entropy(
            pd.to_numeric(field_exposure_df.get('field_slots'), errors='coerce').fillna(0.0).tolist()
        )
        our_entropy = _normalized_entropy(
            pd.to_numeric(our_exposure_df.get('our_slots'), errors='coerce').fillna(0.0).tolist()
        )
        field_top5_slot_share = None
        if not field_exposure_df.empty:
            field_total_slots = float(max(1, field_lineups * 8))
            field_top5_slot_share = 100.0 * float(
                pd.to_numeric(field_exposure_df['field_slots'], errors='coerce').fillna(0.0).nlargest(5).sum()
            ) / field_total_slots
        our_top5_slot_share = None
        if not our_exposure_df.empty:
            our_total_slots = float(max(1, our_lineups * 8))
            our_top5_slot_share = 100.0 * float(
                pd.to_numeric(our_exposure_df['our_slots'], errors='coerce').fillna(0.0).nlargest(5).sum()
            ) / our_total_slots

        portfolio_diagnostics_df = pd.DataFrame([
            {
                'metric': 'Avg Pairwise Overlap',
                'field_value': field_overlap.get('avg'),
                'our_value': our_overlap.get('avg'),
                'delta_our_minus_field': (
                    None
                    if field_overlap.get('avg') is None or our_overlap.get('avg') is None
                    else float(our_overlap['avg'] - field_overlap['avg'])
                ),
            },
            {
                'metric': 'Max Pairwise Overlap',
                'field_value': field_overlap.get('max'),
                'our_value': our_overlap.get('max'),
                'delta_our_minus_field': (
                    None
                    if field_overlap.get('max') is None or our_overlap.get('max') is None
                    else float(our_overlap['max'] - field_overlap['max'])
                ),
            },
            {
                'metric': 'Exposure Entropy (Normalized)',
                'field_value': field_entropy,
                'our_value': our_entropy,
                'delta_our_minus_field': (
                    None
                    if field_entropy is None or our_entropy is None
                    else float(our_entropy - field_entropy)
                ),
            },
            {
                'metric': 'Top Player Exposure %',
                'field_value': (
                    float(pd.to_numeric(field_exposure_df['field_exposure_pct'], errors='coerce').max())
                    if not field_exposure_df.empty
                    else None
                ),
                'our_value': (
                    float(pd.to_numeric(our_exposure_df['our_exposure_pct'], errors='coerce').max())
                    if not our_exposure_df.empty
                    else None
                ),
                'delta_our_minus_field': (
                    None
                    if field_exposure_df.empty or our_exposure_df.empty
                    else float(
                        pd.to_numeric(our_exposure_df['our_exposure_pct'], errors='coerce').max()
                        - pd.to_numeric(field_exposure_df['field_exposure_pct'], errors='coerce').max()
                    )
                ),
            },
            {
                'metric': 'Top-5 Slot Share %',
                'field_value': field_top5_slot_share,
                'our_value': our_top5_slot_share,
                'delta_our_minus_field': (
                    None
                    if field_top5_slot_share is None or our_top5_slot_share is None
                    else float(our_top5_slot_share - field_top5_slot_share)
                ),
            },
        ])
        overlap_gap_avg = (
            None
            if field_overlap.get('avg') is None or our_overlap.get('avg') is None
            else float(our_overlap['avg'] - field_overlap['avg'])
        )

        field_exposure_lookup = {
            str(k): float(v)
            for k, v in exposure_df.set_index('name_key')['field_exposure_pct'].dropna().items()
        }
        actual_fpts_lookup = {
            str(k): float(v)
            for k, v in exposure_df.set_index('name_key')['actual_fpts'].dropna().items()
        }
        actual_error_lookup = {
            str(k): float(v)
            for k, v in exposure_df.set_index('name_key')['actual_minus_proj'].dropna().items()
        }
        display_lookup = {
            str(k): str(v)
            for k, v in exposure_df.set_index('name_key')['display_name'].fillna('').items()
            if str(k).strip()
        }

        model_strategy_breakdown_df = pd.DataFrame()
        model_signal_coverage_df = pd.DataFrame()
        if not our_lineups_df.empty:
            model_rows: List[Dict[str, Any]] = []
            signal_rows: List[Dict[str, Any]] = []
            signal_key_order = sorted(
                high_signal_keys,
                key=lambda key: (
                    -field_exposure_lookup.get(key, 0.0),
                    -actual_error_lookup.get(key, -9999.0),
                    display_lookup.get(key, key),
                ),
            )
            model_group_cols = ['model_key', 'model_label', 'generation_strategy']
            for group_vals, group_lineups_df in our_lineups_df.groupby(model_group_cols, dropna=False):
                model_key, model_label, generation_strategy = group_vals
                group_lineup_nums = group_lineups_df['lineup_num'].tolist()
                group_size = int(len(group_lineups_df))
                group_long_df = our_long[our_long['lineup_num'].isin(group_lineup_nums)].copy()
                group_lineup_sets = (
                    group_long_df.groupby('lineup_num')['name_key']
                    .apply(lambda s: {x for x in s.tolist() if x})
                    .tolist()
                    if not group_long_df.empty
                    else []
                )
                group_overlap = _pairwise_overlap_summary(group_lineup_sets)
                group_structures_df = our_lineup_structures_df[
                    our_lineup_structures_df['lineup_num'].isin(group_lineup_nums)
                ].copy()
                model_exp_counts = (
                    group_long_df.groupby('name_key').size()
                    if not group_long_df.empty
                    else pd.Series(dtype=float)
                )
                model_exp_pct = (
                    100.0 * model_exp_counts / float(max(1, group_size))
                    if not model_exp_counts.empty
                    else pd.Series(dtype=float)
                )

                core_gap_values: List[float] = []
                core_hit_count = 0
                standout_hit_count = 0
                high_signal_hit_count = 0
                for name_key in signal_key_order:
                    model_pct = float(model_exp_pct.get(name_key, 0.0))
                    field_pct = float(field_exposure_lookup.get(name_key, 0.0))
                    signal_type_parts: List[str] = []
                    if name_key in core_signal_keys:
                        signal_type_parts.append('Core')
                        core_gap_values.append(field_pct - model_pct)
                        if model_pct > 0:
                            core_hit_count += 1
                    if name_key in standout_signal_keys:
                        signal_type_parts.append('Standout')
                        if model_pct > 0:
                            standout_hit_count += 1
                    if name_key in top3_signal_keys:
                        signal_type_parts.append('Top-3 Actual')
                    if model_pct > 0:
                        high_signal_hit_count += 1

                    signal_rows.append({
                        'model_key': str(model_key or 'standard_v1'),
                        'model_label': str(model_label or model_key or 'standard_v1'),
                        'generation_strategy': str(generation_strategy or ''),
                        'player': display_lookup.get(name_key, name_key),
                        'signal_type': ', '.join(signal_type_parts) or 'Signal',
                        'field_exposure_pct': field_pct,
                        'model_exposure_pct': model_pct,
                        'gap_vs_field_pct': field_pct - model_pct,
                        'actual_fpts': actual_fpts_lookup.get(name_key),
                        'actual_minus_proj': actual_error_lookup.get(name_key),
                    })

                avg_top_stack_size = pd.to_numeric(
                    group_structures_df.get('top_stack_size'),
                    errors='coerce',
                ).dropna()
                model_rows.append({
                    'model_key': str(model_key or 'standard_v1'),
                    'model_label': str(model_label or model_key or 'standard_v1'),
                    'generation_strategy': str(generation_strategy or ''),
                    'lineups': group_size,
                    'avg_proj_fpts': float(pd.to_numeric(group_lineups_df['total_proj_fpts'], errors='coerce').mean())
                    if group_size > 0 else None,
                    'avg_actual_fpts': float(pd.to_numeric(group_lineups_df['total_actual_fpts'], errors='coerce').mean())
                    if group_size > 0 else None,
                    'best_actual_fpts': float(pd.to_numeric(group_lineups_df['total_actual_fpts'], errors='coerce').max())
                    if group_size > 0 else None,
                    'avg_salary': float(pd.to_numeric(group_lineups_df['total_salary'], errors='coerce').mean())
                    if group_size > 0 else None,
                    'avg_top_stack_size': float(avg_top_stack_size.mean()) if not avg_top_stack_size.empty else None,
                    'avg_pairwise_overlap': group_overlap.get('avg'),
                    'core_players_hit_pct': (
                        100.0 * float(core_hit_count) / float(max(1, len(core_signal_keys)))
                        if core_signal_keys else None
                    ),
                    'standout_players_hit_pct': (
                        100.0 * float(standout_hit_count) / float(max(1, len(standout_signal_keys)))
                        if standout_signal_keys else None
                    ),
                    'high_signal_players_hit_pct': (
                        100.0 * float(high_signal_hit_count) / float(max(1, len(high_signal_keys)))
                        if high_signal_keys else None
                    ),
                    'avg_core_gap_pct': (
                        float(np.mean(core_gap_values))
                        if core_gap_values else None
                    ),
                    'captured_top3_in_single_lineup': (
                        any(set(top3_keys).issubset(s) for s in group_lineup_sets)
                        if top3_keys and group_lineup_sets
                        else False
                    ),
                })

            model_strategy_breakdown_df = (
                pd.DataFrame(model_rows).sort_values(
                    ['best_actual_fpts', 'avg_actual_fpts', 'lineups'],
                    ascending=[False, False, False],
                )
                if model_rows else pd.DataFrame()
            )
            model_signal_coverage_df = (
                pd.DataFrame(signal_rows).sort_values(
                    ['field_exposure_pct', 'actual_minus_proj', 'gap_vs_field_pct'],
                    ascending=[False, False, False],
                )
                if signal_rows else pd.DataFrame()
            )

        result['metrics'] = {
            'field_lineups_analyzed': field_lineups,
            'our_lineups': our_lineups,
            'winner_score': winner_score,
            'our_best_score': our_best_score,
            'winner_gap': winner_gap,
            'our_avg_score': our_avg_score,
            'topn_avg_score': topn_avg_points,
            'topn_best_score': topn_best_points,
            'projection_mae': proj_mae,
            'projection_rank_corr': proj_rank_corr,
            'ownership_mae': ownership_mae,
            'ownership_rank_corr': ownership_rank_corr,
            'missed_core_count': int(len(missed_core_df)),
            'missed_standout_count': int(len(missed_standouts_df)),
            'ownership_polarity_count': ownership_polarity_count,
            'underprojected_chalk_count': underprojected_chalk_count,
            'overprojected_chalk_count': overprojected_chalk_count,
            'overexposed_dud_count': overexposed_dud_count,
            'value_tier_miss_count': value_tier_miss_count,
            'combo_capture_miss_count': combo_capture_miss_count,
            'combo_capture_rows_exported': combo_capture_rows_exported,
            'team_concentration_mismatch_count': team_concentration_mismatch_count,
            'largest_team_slot_gap': largest_team_slot_gap,
            'largest_team_slot_gap_team': largest_team_slot_gap_team,
            'top_scorer_name': top_scorer_name,
            'top_scorer_actual': float(top_scorer_actual) if pd.notna(top_scorer_actual) else None,
            'top_scorer_in_our_lineups': top_scorer_in_our,
            'top3_target_count': int(top3_target_count),
            'top3_covered_count': int(top3_covered_count),
            'top3_all_in_single_lineup': top3_all_in_single,
            'field_avg_pairwise_overlap': field_overlap.get('avg'),
            'our_avg_pairwise_overlap': our_overlap.get('avg'),
            'field_max_pairwise_overlap': field_overlap.get('max'),
            'our_max_pairwise_overlap': our_overlap.get('max'),
            'field_exposure_entropy': field_entropy,
            'our_exposure_entropy': our_entropy,
            'field_top5_slot_share_pct': field_top5_slot_share,
            'our_top5_slot_share_pct': our_top5_slot_share,
            'model_group_count': int(model_strategy_breakdown_df['model_key'].nunique())
            if isinstance(model_strategy_breakdown_df, pd.DataFrame) and not model_strategy_breakdown_df.empty
            else 0,
            'high_signal_player_count': int(len(high_signal_keys)),
        }

        # 6) What went right / wrong notes
        right_notes: List[str] = []
        wrong_notes: List[str] = []

        if winner_gap is not None:
            if winner_gap <= 15:
                right_notes.append(f"Winner gap was tight at {winner_gap:.1f} DK points.")
            else:
                wrong_notes.append(f"Winner gap was {winner_gap:.1f} DK points.")
        else:
            wrong_notes.append("Winner gap is unavailable (missing winner score or lineup actuals).")

        if proj_mae is not None:
            if proj_mae <= 10:
                right_notes.append(f"Projection MAE was solid at {proj_mae:.2f}.")
            else:
                wrong_notes.append(f"Projection MAE was high at {proj_mae:.2f}.")
        else:
            wrong_notes.append("Projection MAE is unavailable for this slate.")

        if ownership_mae is not None:
            if ownership_mae <= 10:
                right_notes.append(f"Ownership MAE was controlled at {ownership_mae:.2f}%.")
            else:
                wrong_notes.append(f"Ownership MAE was elevated at {ownership_mae:.2f}%.")

        if len(missed_core_df) == 0:
            right_notes.append("No major underexposure on core field plays.")
        else:
            wrong_notes.append(
                f"Missed {len(missed_core_df)} core field plays (field >= {core_field_exposure_pct:.0f}% with underexposure)."
            )

        if len(missed_standouts_df) > 0:
            wrong_notes.append(
                f"Missed {len(missed_standouts_df)} standout performers (actual-projection >= {standout_min_error:.1f})."
            )
        else:
            right_notes.append("No major missed standout outcomes under current thresholds.")

        if top3_target_count > 0:
            if top3_covered_count == top3_target_count:
                right_notes.append("Covered all top-3 actual scorers in at least one lineup.")
            else:
                wrong_notes.append(
                    f"Covered {top3_covered_count}/{top3_target_count} top-3 actual scorers."
                )
            if top3_all_in_single is True:
                right_notes.append("At least one lineup contained all top-3 actual scorers.")
            elif top3_all_in_single is False:
                wrong_notes.append("No lineup captured all top-3 actual scorers together.")

        if ownership_polarity_count > 0:
            wrong_notes.append(
                f"Ownership polarity misses: {ownership_polarity_count} high-signal players were materially misestimated."
            )

        if overexposed_dud_count > 0:
            wrong_notes.append(
                f"Overexposed dud watchlist triggered on {overexposed_dud_count} players (high our exposure + underperformance)."
            )

        if value_tier_miss_count > 0:
            wrong_notes.append(
                f"Value-tier miss tracker flagged {value_tier_miss_count} players in the $4k-$5k range."
            )

        if combo_capture_miss_count > 0:
            top_combo = str(combo_capture_misses_df.iloc[0].get('combo_players') or '')
            top_combo_gap = pd.to_numeric(combo_capture_misses_df.iloc[0].get('combo_gap_pct'), errors='coerce')
            if top_combo and pd.notna(top_combo_gap):
                wrong_notes.append(
                    f"Combo capture miss: `{top_combo}` field-minus-our gap was {float(top_combo_gap):.1f}pp."
                )

        if team_concentration_mismatch_count > 0 and largest_team_slot_gap_team:
            if largest_team_slot_gap is not None and pd.notna(largest_team_slot_gap):
                wrong_notes.append(
                    f"Team concentration mismatch led by {largest_team_slot_gap_team} ({float(largest_team_slot_gap):+.1f}pp slot-share gap)."
                )
            else:
                wrong_notes.append("Team concentration mismatch detected between field and our builds.")

        if overlap_gap_avg is not None:
            if overlap_gap_avg <= -1.0:
                wrong_notes.append(
                    f"Portfolio overlap ran {abs(overlap_gap_avg):.2f} players below the field average, suggesting over-diversification."
                )
            elif overlap_gap_avg >= 1.0:
                wrong_notes.append(
                    f"Portfolio overlap ran {overlap_gap_avg:.2f} players above the field average, suggesting over-concentration."
                )
            else:
                right_notes.append("Portfolio overlap tracked close to field average.")

        result['right_notes'] = right_notes
        result['wrong_notes'] = wrong_notes

        # 7) Next-slate improvement actions
        improvement_rows: List[Dict[str, Any]] = []
        priority = 1

        if proj_mae is not None:
            improvement_rows.append({
                'priority': priority,
                'area': 'Projection Calibration',
                'why': f"MAE={proj_mae:.2f}, rank_corr={proj_rank_corr:.2f}" if proj_rank_corr is not None else f"MAE={proj_mae:.2f}",
                'next_slate_change': 'Recalibrate projection weights by role/salary tier for this slate profile.',
                'success_metric': 'Lower MAE while holding or improving rank correlation.',
            })
            priority += 1

        if ownership_mae is not None:
            improvement_rows.append({
                'priority': priority,
                'area': 'Ownership Calibration',
                'why': (
                    f"ownership_mae={ownership_mae:.2f}, rank_corr={ownership_rank_corr:.2f}"
                    if ownership_rank_corr is not None
                    else f"ownership_mae={ownership_mae:.2f}"
                ),
                'next_slate_change': 'Adjust ownership curve by projection tier and game environment.',
                'success_metric': 'Reduce ownership MAE and improve ownership rank correlation.',
            })
            priority += 1

        if winner_gap is not None and winner_gap > 0:
            improvement_rows.append({
                'priority': priority,
                'area': 'Top-End Ceiling Capture',
                'why': f"winner_gap={winner_gap:.1f}",
                'next_slate_change': 'Increase exposure to high-ceiling constructions present in top field entries.',
                'success_metric': 'Shrink winner gap and raise best-lineup actual FPTS.',
            })
            priority += 1

        if not missed_core_df.empty:
            top_core = str(missed_core_df.iloc[0].get('display_name') or '')
            improvement_rows.append({
                'priority': priority,
                'area': 'Core Field Coverage',
                'why': f"missed_core={len(missed_core_df)} (largest gap: {top_core})",
                'next_slate_change': 'Add guardrail minimum exposure for core field plays in high-leverage builds.',
                'success_metric': 'Reduce count of core underexposure misses.',
            })
            priority += 1

        if not missed_standouts_df.empty:
            top_miss = str(missed_standouts_df.iloc[0].get('display_name') or '')
            top_err = pd.to_numeric(missed_standouts_df.iloc[0].get('actual_minus_proj'), errors='coerce')
            improvement_rows.append({
                'priority': priority,
                'area': 'Missed Standout Detection',
                'why': (
                    f"missed_standouts={len(missed_standouts_df)} (top: {top_miss}, +{top_err:.1f})"
                    if pd.notna(top_err)
                    else f"missed_standouts={len(missed_standouts_df)}"
                ),
                'next_slate_change': 'Strengthen standout rules for volatile upside profiles and stack contexts.',
                'success_metric': 'Improve coverage of high actual-minus-projection players.',
            })
            priority += 1

        if ownership_polarity_count > 0:
            top_own = str(ownership_polarity_df.iloc[0].get('display_name') or '')
            own_err = pd.to_numeric(ownership_polarity_df.iloc[0].get('ownership_error_pp'), errors='coerce')
            improvement_rows.append({
                'priority': priority,
                'area': 'Ownership Polarity Calibration',
                'why': (
                    f"polarity_misses={ownership_polarity_count} (top: {top_own}, error={own_err:+.1f}pp)"
                    if pd.notna(own_err)
                    else f"polarity_misses={ownership_polarity_count}"
                ),
                'next_slate_change': 'Add bidirectional ownership guardrails for underprojected/overprojected chalk.',
                'success_metric': 'Lower ownership polarity misses and improve ownership rank alignment.',
            })
            priority += 1

        if overexposed_dud_count > 0:
            top_dud = str(overexposed_duds_df.iloc[0].get('display_name') or '')
            over_gap = pd.to_numeric(overexposed_duds_df.iloc[0].get('our_minus_field_pct'), errors='coerce')
            improvement_rows.append({
                'priority': priority,
                'area': 'Overexposed Dud Guardrails',
                'why': (
                    f"overexposed_duds={overexposed_dud_count} (top: {top_dud}, overexposure={over_gap:+.1f}pp)"
                    if pd.notna(over_gap)
                    else f"overexposed_duds={overexposed_dud_count}"
                ),
                'next_slate_change': 'Cap fragile high-exposure plays when field is underweight and downside risk is high.',
                'success_metric': 'Reduce negative-leverage overexposure without sacrificing ceiling.',
            })
            priority += 1

        if value_tier_miss_count > 0:
            top_value = str(value_tier_misses_df.iloc[0].get('display_name') or '')
            improvement_rows.append({
                'priority': priority,
                'area': 'Value-Tier (4k-5k) Coverage',
                'why': f"value_tier_misses={value_tier_miss_count} (top: {top_value})",
                'next_slate_change': 'Reweight sub-$5k value detection and minutes/role volatility filters.',
                'success_metric': 'Improve hit rate on 4k-5k core and standout outcomes.',
            })
            priority += 1

        if combo_capture_miss_count > 0:
            top_combo = str(combo_capture_misses_df.iloc[0].get('combo_players') or '')
            top_combo_gap = pd.to_numeric(combo_capture_misses_df.iloc[0].get('combo_gap_pct'), errors='coerce')
            improvement_rows.append({
                'priority': priority,
                'area': 'Pair/Triple Combo Capture',
                'why': (
                    f"combo_misses={combo_capture_miss_count} (top: {top_combo}, gap={top_combo_gap:.1f}pp)"
                    if pd.notna(top_combo_gap)
                    else f"combo_misses={combo_capture_miss_count}"
                ),
                'next_slate_change': 'Add co-occurrence boosts for high-field player pairs/triples in lineup generation.',
                'success_metric': 'Increase capture rate of top-field combo structures.',
            })
            priority += 1

        if team_concentration_mismatch_count > 0:
            improvement_rows.append({
                'priority': priority,
                'area': 'Team Concentration Alignment',
                'why': (
                    f"team_mismatches={team_concentration_mismatch_count} "
                    f"(largest: {largest_team_slot_gap_team} {largest_team_slot_gap:+.1f}pp)"
                    if largest_team_slot_gap is not None and largest_team_slot_gap_team
                    else f"team_mismatches={team_concentration_mismatch_count}"
                ),
                'next_slate_change': 'Constrain team slot-share drift vs top-field concentration patterns.',
                'success_metric': 'Shrink team concentration gap on heavily targeted games/teams.',
            })
            priority += 1

        if overlap_gap_avg is not None and abs(overlap_gap_avg) >= 1.0:
            improvement_rows.append({
                'priority': priority,
                'area': 'Portfolio Concentration',
                'why': f"avg_overlap_gap={overlap_gap_avg:+.2f} players vs field",
                'next_slate_change': 'Tune lineup uniqueness / overlap targets so portfolio concentration stays closer to field-winning structure.',
                'success_metric': 'Reduce pairwise overlap gap while preserving best-lineup ceiling.',
            })

        result['improvements_df'] = (
            pd.DataFrame(improvement_rows).sort_values('priority')
            if improvement_rows
            else pd.DataFrame(columns=['priority', 'area', 'why', 'next_slate_change', 'success_metric'])
        )

        result['top_field_players_df'] = top_field_players_df.reset_index(drop=True)
        result['exposure_comparison_df'] = exposure_df.reset_index(drop=True)
        result['missed_core_df'] = missed_core_df.reset_index(drop=True)
        result['missed_standouts_df'] = missed_standouts_df.reset_index(drop=True)
        result['ownership_polarity_df'] = ownership_polarity_df.reset_index(drop=True)
        result['overexposed_duds_df'] = overexposed_duds_df.reset_index(drop=True)
        result['value_tier_misses_df'] = value_tier_misses_df.reset_index(drop=True)
        result['combo_capture_misses_df'] = combo_capture_misses_df.reset_index(drop=True)
        result['team_concentration_mismatch_df'] = team_concentration_mismatch_df.reset_index(drop=True)
        result['field_lineup_structures_df'] = field_lineup_structures_df.reset_index(drop=True)
        result['our_lineup_structures_df'] = our_lineup_structures_df.reset_index(drop=True)
        result['portfolio_diagnostics_df'] = portfolio_diagnostics_df.reset_index(drop=True)
        result['model_strategy_breakdown_df'] = model_strategy_breakdown_df.reset_index(drop=True)
        result['model_signal_coverage_df'] = model_signal_coverage_df.reset_index(drop=True)

    except Exception as e:
        result['errors'].append(f"Postmortem build error: {e}")

    return result


# =============================================================================
# Top Finisher Analysis (Reverse Engineering Winning Lineups)
# =============================================================================

def analyze_top_finishers(
    conn: sqlite3.Connection,
    slate_date: str,
    top_n: int = 10
) -> Dict:
    """
    Analyze the top N finishing lineups to understand winning strategies.

    Returns insights on:
    - Salary distribution by position
    - Stacking patterns (team/game stacks)
    - Ownership levels of selected players
    - Contrarian vs chalk plays
    - Game environment preferences

    Args:
        conn: SQLite connection
        slate_date: Date string (YYYY-MM-DD)
        top_n: Number of top finishers to analyze (default 10)

    Returns:
        Dict with comprehensive analysis
    """
    results = {
        'slate_date': slate_date,
        'top_n': top_n,
        'lineups': [],
        'salary_by_position': {},
        'player_frequency': {},
        'team_stacks': [],
        'ownership_analysis': {},
        'role_analysis': {},
        'game_totals': {},
        'insights': [],
        'errors': []
    }

    # Get top N lineups
    top_lineups = pd.read_sql_query("""
        SELECT rank, points, username, total_salary,
               pg, sg, sf, pf, c, g, f, util
        FROM dfs_contest_entries
        WHERE slate_date = ? AND rank IS NOT NULL
        ORDER BY rank ASC
        LIMIT ?
    """, conn, params=[slate_date, top_n])

    if top_lineups.empty:
        results['errors'].append("No contest data found for this slate")
        return results

    # Get player info from projections (salary, team, ownership, vegas, role)
    player_info = pd.read_sql_query("""
        SELECT s.player_name, s.team, s.salary, s.actual_ownership, s.actual_fpts,
               s.opponent, s.proj_fpts, s.ownership_proj,
               p.vegas_implied_fpts,
               pr.role_tier
        FROM dfs_slate_projections s
        LEFT JOIN predictions p
            ON s.player_id = p.player_id AND p.game_date = s.slate_date
        LEFT JOIN player_roles pr
            ON s.player_id = pr.player_id AND pr.season = '2025-26'
        WHERE s.slate_date = ?
    """, conn, params=[slate_date])

    # Create multiple lookups for flexible name matching
    player_lookup = {}          # normalized full name -> info
    last_name_lookup = {}       # last_name -> list of infos (for fallback)

    for _, row in player_info.iterrows():
        info = {
            'name': row['player_name'],
            'team': row['team'],
            'salary': row['salary'],
            'ownership': row['actual_ownership'],
            'fpts': row['actual_fpts'],
            'opponent': row['opponent'],
            'proj_fpts': row.get('proj_fpts'),
            'proj_ownership': row.get('ownership_proj'),
            'vegas_fpts': row.get('vegas_implied_fpts'),
            'role_tier': row.get('role_tier', ''),
        }

        norm_name = _normalize_name(row['player_name'])
        player_lookup[norm_name] = info

        # Also index by last name for fallback matching
        parts = norm_name.split()
        if parts:
            last_name = parts[-1]
            if last_name not in last_name_lookup:
                last_name_lookup[last_name] = []
            last_name_lookup[last_name].append(info)

    def find_player_info(player_name: str) -> dict:
        """Find player info with fallback matching."""
        norm = _normalize_name(player_name)

        # Try exact normalized match first
        if norm in player_lookup:
            return player_lookup[norm]

        # Fallback: try last name match
        parts = norm.split()
        if parts:
            last_name = parts[-1]
            first_initial = parts[0][0] if parts else ''

            candidates = last_name_lookup.get(last_name, [])

            # If only one player with that last name, use it
            if len(candidates) == 1:
                return candidates[0]

            # If multiple, try to match first initial
            if len(candidates) > 1 and first_initial:
                for c in candidates:
                    c_norm = _normalize_name(c['name'])
                    if c_norm.startswith(first_initial):
                        return c

        return {}

    # Position slots
    positions = ['pg', 'sg', 'sf', 'pf', 'c', 'g', 'f', 'util']
    position_labels = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    # Initialize tracking
    salary_by_pos = {pos: [] for pos in position_labels}
    player_counts = {}
    all_teams_in_lineups = []
    ownership_levels = []
    role_data = []  # List of {'role': str, 'ownership': float, 'fpts': float, 'salary': int}

    # Analyze each lineup
    for _, lineup in top_lineups.iterrows():
        lineup_data = {
            'rank': int(lineup['rank']),
            'points': float(lineup['points']),
            'username': lineup['username'],
            'salary': lineup['total_salary'],
            'players': [],
            'teams': [],
            'stacks': []
        }

        team_counts = {}

        for pos, label in zip(positions, position_labels):
            player_name = lineup[pos]
            if not player_name or pd.isna(player_name):
                continue

            # Use flexible matching with fallback
            info = find_player_info(player_name)

            player_data = {
                'position': label,
                'name': player_name,
                'team': info.get('team', '?'),
                'salary': info.get('salary', 0),
                'ownership': info.get('ownership'),
                'fpts': info.get('fpts'),
                'opponent': info.get('opponent', '?'),
                'proj_fpts': info.get('proj_fpts'),
                'proj_ownership': info.get('proj_ownership'),
                'vegas_fpts': info.get('vegas_fpts'),
                'role_tier': info.get('role_tier', ''),
            }
            lineup_data['players'].append(player_data)

            # Track salary by position
            if info.get('salary'):
                salary_by_pos[label].append(info['salary'])

            # Track player frequency (with role)
            if player_name not in player_counts:
                player_counts[player_name] = {'count': 0, 'role': info.get('role_tier', ''), 'ownership': info.get('ownership')}
            player_counts[player_name]['count'] += 1

            # Track team stacks
            team = info.get('team', '?')
            if team != '?':
                team_counts[team] = team_counts.get(team, 0) + 1
                lineup_data['teams'].append(team)

            # Track ownership
            if info.get('ownership') is not None:
                ownership_levels.append(info['ownership'])

            # Track role tier data for analysis
            role = info.get('role_tier', '') or 'UNKNOWN'
            role_entry = {'role': role, 'salary': info.get('salary', 0)}
            if info.get('ownership') is not None:
                role_entry['ownership'] = info['ownership']
            if info.get('fpts') is not None:
                role_entry['fpts'] = info['fpts']
            role_data.append(role_entry)

        # Identify stacks (2+ players from same team)
        for team, count in team_counts.items():
            if count >= 2:
                lineup_data['stacks'].append({'team': team, 'count': count})

        all_teams_in_lineups.extend(lineup_data['teams'])
        results['lineups'].append(lineup_data)

    # Aggregate salary by position
    for pos, salaries in salary_by_pos.items():
        if salaries:
            results['salary_by_position'][pos] = {
                'avg': round(np.mean(salaries)),
                'min': min(salaries),
                'max': max(salaries),
                'count': len(salaries)
            }

    # Player frequency (most rostered in top lineups) — includes role tier
    results['player_frequency'] = sorted(
        [{'player': k, 'count': v['count'], 'pct': round(100 * v['count'] / top_n, 1),
          'role': v.get('role', ''), 'ownership': v.get('ownership')}
         for k, v in player_counts.items()],
        key=lambda x: x['count'],
        reverse=True
    )[:15]  # Top 15 most used

    # Stacking analysis
    stack_counts = {}
    for lineup in results['lineups']:
        for stack in lineup.get('stacks', []):
            key = f"{stack['team']} ({stack['count']})"
            stack_counts[key] = stack_counts.get(key, 0) + 1

    results['team_stacks'] = sorted(
        [{'stack': k, 'lineups': v} for k, v in stack_counts.items()],
        key=lambda x: x['lineups'],
        reverse=True
    )

    # Ownership analysis
    if ownership_levels:
        low_own = [o for o in ownership_levels if o < 10]
        mid_own = [o for o in ownership_levels if 10 <= o < 25]
        high_own = [o for o in ownership_levels if o >= 25]

        results['ownership_analysis'] = {
            'avg_ownership': round(np.mean(ownership_levels), 1),
            'low_ownership_plays': len(low_own),
            'mid_ownership_plays': len(mid_own),
            'high_ownership_plays': len(high_own),
            'contrarian_pct': round(100 * len(low_own) / len(ownership_levels), 1) if ownership_levels else 0
        }

    # Role tier analysis
    if role_data:
        role_order = ['STAR', 'STARTER', 'ROTATION', 'BENCH', 'UNKNOWN']
        role_stats = {}
        for role in role_order:
            entries = [r for r in role_data if r['role'] == role]
            if entries:
                own_vals = [r['ownership'] for r in entries if 'ownership' in r]
                fpts_vals = [r['fpts'] for r in entries if 'fpts' in r]
                sal_vals = [r['salary'] for r in entries if r.get('salary')]
                role_stats[role] = {
                    'count': len(entries),
                    'pct_of_lineups': round(100 * len(entries) / (top_n * 8), 1),
                    'avg_ownership': round(np.mean(own_vals), 1) if own_vals else None,
                    'avg_fpts': round(np.mean(fpts_vals), 1) if fpts_vals else None,
                    'avg_salary': round(np.mean(sal_vals)) if sal_vals else None,
                }
        results['role_analysis'] = role_stats

    # Generate insights
    insights = []

    # Salary insight
    if results['salary_by_position']:
        high_spend = max(results['salary_by_position'].items(), key=lambda x: x[1]['avg'])
        low_spend = min(results['salary_by_position'].items(), key=lambda x: x[1]['avg'])
        insights.append(f"💰 Highest spend at {high_spend[0]} (avg ${high_spend[1]['avg']:,}), lowest at {low_spend[0]} (avg ${low_spend[1]['avg']:,})")

    # Stacking insight
    if results['team_stacks']:
        top_stack = results['team_stacks'][0]
        insights.append(f"🔗 Most common stack: {top_stack['stack']} in {top_stack['lineups']}/{top_n} lineups")

    # Ownership insight
    if results['ownership_analysis']:
        own = results['ownership_analysis']
        if own['contrarian_pct'] >= 30:
            insights.append(f"🎯 Contrarian builds: {own['contrarian_pct']:.0f}% of plays under 10% owned")
        else:
            insights.append(f"📊 Chalk-heavy builds: only {own['contrarian_pct']:.0f}% contrarian plays")

    # Must-have player insight
    if results['player_frequency']:
        must_have = [p for p in results['player_frequency'] if p['pct'] >= 70]
        if must_have:
            names = ', '.join([p['player'].split()[-1] for p in must_have[:3]])
            insights.append(f"⭐ Must-have players: {names} (70%+ of top lineups)")

    # Role composition insight
    if results.get('role_analysis'):
        ra = results['role_analysis']
        star_pct = ra.get('STAR', {}).get('pct_of_lineups', 0)
        rotation_pct = ra.get('ROTATION', {}).get('pct_of_lineups', 0)
        bench_pct = ra.get('BENCH', {}).get('pct_of_lineups', 0)
        if star_pct > 0:
            star_avg = ra['STAR'].get('avg_fpts')
            star_own = ra['STAR'].get('avg_ownership')
            parts = [f"STARs = {star_pct:.0f}% of roster spots"]
            if star_avg:
                parts.append(f"avg {star_avg:.1f} FPTS")
            if star_own:
                parts.append(f"{star_own:.0f}% owned")
            insights.append(f"🌟 {', '.join(parts)}")
        if rotation_pct + bench_pct > 20:
            insights.append(
                f"🔄 Value plays: ROTATION+BENCH = {rotation_pct + bench_pct:.0f}% of roster spots"
            )

    results['insights'] = insights

    return results


def get_game_totals_for_slate(conn: sqlite3.Connection, slate_date: str) -> Dict[str, float]:
    """Get Vegas game totals for a slate date (for game environment analysis)."""
    try:
        totals = pd.read_sql_query("""
            SELECT home_team, away_team, total
            FROM game_odds
            WHERE game_date = ? AND total IS NOT NULL
        """, conn, params=[slate_date])

        result = {}
        for _, row in totals.iterrows():
            game_key = f"{row['away_team']}@{row['home_team']}"
            result[row['home_team']] = row['total']
            result[row['away_team']] = row['total']
        return result
    except Exception:
        return {}


# DraftKings → NBA API abbreviation normalization
DK_TO_NBA_ABBR = {
    'PHO': 'PHX', 'GS': 'GSW', 'SA': 'SAS', 'NY': 'NYK', 'NO': 'NOP',
}


def _normalize_team_abbr(abbr: str) -> str:
    """Convert DraftKings team abbreviation to NBA API format."""
    return DK_TO_NBA_ABBR.get(abbr.upper(), abbr.upper()) if abbr else ''


def analyze_game_environment(
    conn: sqlite3.Connection,
    slate_date: str,
    top_lineups_analysis: Dict,
    top_n: int = 10,
) -> Dict:
    """Analyze how game environment (Vegas totals, spreads) correlates with top lineup construction.

    Compares Vegas projections to actual scores and measures whether top-finishing
    lineups disproportionately target high-total games — a key GPP strategy signal
    that feeds back into the stack_score algorithm.

    Args:
        conn: SQLite connection (must have game_odds and team_game_logs tables).
        slate_date: Date string 'YYYY-MM-DD'.
        top_lineups_analysis: Output from analyze_top_finishers().
        top_n: Number of top lineups to analyze.

    Returns:
        Dict with keys: games, correlation_stats, insights, errors.
    """
    result = {
        'games': [],
        'correlation_stats': {},
        'insights': [],
        'errors': [],
    }

    try:
        # --- 1. Get Vegas odds for all games on this slate ---
        odds_df = pd.read_sql_query("""
            SELECT home_team, away_team, total, spread, stack_score, pace_score, blowout_risk
            FROM game_odds
            WHERE game_date = ? AND total IS NOT NULL
        """, conn, params=[slate_date])

        if odds_df.empty:
            result['errors'].append(f"No game odds found for {slate_date}")
            return result

        # --- 2. Get actual scores from team_game_logs ---
        actual_scores = {}
        try:
            scores_df = pd.read_sql_query("""
                SELECT team_abbreviation, pts, opp_pts, matchup
                FROM team_game_logs
                WHERE date(game_date) = ?
            """, conn, params=[slate_date])

            for _, row in scores_df.iterrows():
                team = row['team_abbreviation']
                actual_scores[team] = {
                    'pts': row['pts'],
                    'opp_pts': row['opp_pts'],
                    'matchup': row['matchup'],
                }
        except Exception as e:
            result['errors'].append(f"Could not fetch actual scores: {e}")

        # --- 3. Count player slots per game from top lineups ---
        game_player_counts = {}  # "AWAY@HOME" -> count
        lineups = top_lineups_analysis.get('lineups', [])[:top_n]
        total_slots = 0

        for lineup in lineups:
            for p in lineup.get('players', []):
                team_raw = p.get('team', '')
                team = _normalize_team_abbr(team_raw)
                if team:
                    # Find which game this team belongs to
                    for _, odds_row in odds_df.iterrows():
                        if team in (odds_row['home_team'], odds_row['away_team']):
                            game_key = f"{odds_row['away_team']}@{odds_row['home_team']}"
                            game_player_counts[game_key] = game_player_counts.get(game_key, 0) + 1
                            break
                total_slots += 1

        # --- 4. Build per-game detail ---
        all_games = []
        for _, odds_row in odds_df.iterrows():
            home = odds_row['home_team']
            away = odds_row['away_team']
            game_key = f"{away}@{home}"
            game_label = f"{away} @ {home}"

            # Actual totals from team_game_logs (home team perspective)
            actual_total = None
            actual_margin = None
            home_scores = actual_scores.get(home)
            if home_scores:
                actual_total = home_scores['pts'] + home_scores['opp_pts']
                actual_margin = home_scores['opp_pts'] - home_scores['pts']  # away - home (matches spread sign)

            total_diff = None
            if actual_total is not None and odds_row['total'] is not None:
                total_diff = actual_total - odds_row['total']

            players_in_top = game_player_counts.get(game_key, 0)
            pct_of_slots = (players_in_top / total_slots * 100) if total_slots > 0 else 0.0

            all_games.append({
                'game_label': game_label,
                'vegas_total': odds_row['total'],
                'actual_total': actual_total,
                'total_diff': total_diff,
                'spread': odds_row['spread'],
                'actual_margin': actual_margin,
                'stack_score': odds_row.get('stack_score'),
                'pace_score': odds_row.get('pace_score'),
                'blowout_risk': odds_row.get('blowout_risk'),
                'top_lineup_players': players_in_top,
                'pct_of_slots': round(pct_of_slots, 1),
            })

        result['games'] = sorted(all_games, key=lambda g: g['vegas_total'] or 0, reverse=True)

        # --- 5. Compute correlation stats ---
        games_with_players = [g for g in all_games if g['top_lineup_players'] > 0]
        all_totals = [g['vegas_total'] for g in all_games if g['vegas_total'] is not None]
        avg_total_all = np.mean(all_totals) if all_totals else 0.0

        if games_with_players:
            weights = [g['top_lineup_players'] for g in games_with_players]
            totals = [g['vegas_total'] for g in games_with_players]
            avg_total_used = np.average(totals, weights=weights)
        else:
            avg_total_used = 0.0

        total_bias = avg_total_used - avg_total_all

        # Over/under hit rate
        games_with_results = [g for g in all_games if g['total_diff'] is not None]
        overs = [g for g in games_with_results if g['total_diff'] > 0]
        over_hit_rate = (len(overs) / len(games_with_results) * 100) if games_with_results else 0.0

        most_targeted = max(all_games, key=lambda g: g['top_lineup_players'])['game_label'] if all_games else 'N/A'

        # Correlation: Vegas total vs player usage (filter out games with missing totals)
        correlation = None
        valid_games = [g for g in all_games if g['vegas_total'] is not None]
        if len(valid_games) >= 3:
            totals_arr = np.array([g['vegas_total'] for g in valid_games])
            usage_arr = np.array([g['top_lineup_players'] for g in valid_games])
            if np.std(totals_arr) > 0 and np.std(usage_arr) > 0:
                correlation = float(np.corrcoef(totals_arr, usage_arr)[0, 1])

        result['correlation_stats'] = {
            'avg_total_used_games': round(avg_total_used, 1),
            'avg_total_all_games': round(avg_total_all, 1),
            'total_bias': round(total_bias, 1),
            'most_targeted_game': most_targeted,
            'over_hit_rate': round(over_hit_rate, 1),
            'correlation': correlation,
            'n_games': len(all_games),
        }

        # --- 6. Generate insights ---
        insights = []
        if total_bias > 0:
            insights.append(f"Top lineups targeted games ~{total_bias:.1f} pts above slate average total")
        elif total_bias < 0:
            insights.append(f"Top lineups targeted games ~{abs(total_bias):.1f} pts below slate average total")

        if over_hit_rate >= 60:
            insights.append(f"{over_hit_rate:.0f}% of games went OVER the Vegas total")
        elif over_hit_rate <= 40 and games_with_results:
            insights.append(f"Only {over_hit_rate:.0f}% of games went OVER — a low-scoring slate")

        if correlation is not None:
            if correlation > 0.5:
                insights.append(f"Strong positive correlation ({correlation:.2f}) between Vegas total and player usage")
            elif correlation > 0.2:
                insights.append(f"Moderate positive correlation ({correlation:.2f}) between Vegas total and player usage")
            elif correlation < -0.2:
                insights.append(f"Negative correlation ({correlation:.2f}) — top lineups avoided high-total games")

        if most_targeted != 'N/A':
            top_game = max(all_games, key=lambda g: g['top_lineup_players'])
            insights.append(f"Most targeted game: {most_targeted} ({top_game['top_lineup_players']} player slots)")

        result['insights'] = insights

    except Exception as e:
        result['errors'].append(f"Game environment analysis error: {e}")

    return result
