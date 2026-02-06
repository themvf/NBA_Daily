#!/usr/bin/env python3
"""DFS Slate Tracking System for NBA_Daily.

Tracks projection accuracy, stat-level accuracy, lineup performance,
and value accuracy across DFS slates. Stores results in SQLite for
trend analysis and model improvement.

Tables:
    dfs_slate_projections — Per-player projections + actuals for each slate
    dfs_slate_lineups     — Generated lineups per slate with actual FPTS
    dfs_slate_results     — Aggregate accuracy metrics per slate
"""

import sqlite3
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Tuple
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
    """)

    # Migration-safe column additions (check first, then add if missing)
    migrations = [
        ("dfs_slate_projections", "ownership_proj", "REAL"),
        ("dfs_slate_projections", "actual_ownership", "REAL"),
        ("dfs_slate_results", "ownership_mae", "REAL"),
        ("dfs_slate_results", "ownership_correlation", "REAL"),
    ]
    for table, col, col_type in migrations:
        # Check if column exists first to avoid ALTER TABLE errors
        existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column might have been added concurrently

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
        cursor.execute("""
            INSERT OR REPLACE INTO dfs_slate_lineups (
                slate_date, lineup_num, total_proj_fpts, total_salary,
                pg_id, sg_id, sf_id, pf_id, c_id, g_id, f_id, util_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            slate_date, i, lineup.total_proj_fpts, lineup.total_salary,
            players['PG'].player_id if players.get('PG') else None,
            players['SG'].player_id if players.get('SG') else None,
            players['SF'].player_id if players.get('SF') else None,
            players['PF'].player_id if players.get('PF') else None,
            players['C'].player_id if players.get('C') else None,
            players['G'].player_id if players.get('G') else None,
            players['F'].player_id if players.get('F') else None,
            players['UTIL'].player_id if players.get('UTIL') else None,
        ))

    conn.commit()
    return len(sorted_lineups)


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
            _, own_pct, _ = contest_by_norm[norm_name]
            cursor.execute(
                "UPDATE dfs_slate_projections SET actual_ownership = ? "
                "WHERE slate_date = ? AND player_id = ?",
                (own_pct, slate_date, player_id)
            )
            matched += 1
        else:
            unmatched_players.append(player_name)

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
        "SELECT lineup_num, total_proj_fpts, total_salary, total_actual_fpts "
        "FROM dfs_slate_lineups "
        "WHERE slate_date = ? "
        "ORDER BY lineup_num",
        conn, params=[slate_date]
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

    # Get player info from projections (salary, team, ownership)
    player_info = pd.read_sql_query("""
        SELECT player_name, team, salary, actual_ownership, actual_fpts, opponent
        FROM dfs_slate_projections
        WHERE slate_date = ?
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
            'opponent': row['opponent']
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

    # Initialize salary tracking
    salary_by_pos = {pos: [] for pos in position_labels}
    player_counts = {}
    all_teams_in_lineups = []
    ownership_levels = []

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
                'opponent': info.get('opponent', '?')
            }
            lineup_data['players'].append(player_data)

            # Track salary by position
            if info.get('salary'):
                salary_by_pos[label].append(info['salary'])

            # Track player frequency
            player_counts[player_name] = player_counts.get(player_name, 0) + 1

            # Track team stacks
            team = info.get('team', '?')
            if team != '?':
                team_counts[team] = team_counts.get(team, 0) + 1
                lineup_data['teams'].append(team)

            # Track ownership
            if info.get('ownership') is not None:
                ownership_levels.append(info['ownership'])

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

    # Player frequency (most rostered in top lineups)
    results['player_frequency'] = sorted(
        [{'player': k, 'count': v, 'pct': round(100 * v / top_n, 1)}
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

    results['insights'] = insights

    return results


def get_game_totals_for_slate(conn: sqlite3.Connection, slate_date: str) -> Dict[str, float]:
    """Get Vegas game totals for a slate date (for game environment analysis)."""
    try:
        totals = pd.read_sql_query("""
            SELECT home_team, away_team, game_total
            FROM game_odds
            WHERE game_date = ? AND game_total IS NOT NULL
        """, conn, params=[slate_date])

        result = {}
        for _, row in totals.iterrows():
            game_key = f"{row['away_team']}@{row['home_team']}"
            result[row['home_team']] = row['game_total']
            result[row['away_team']] = row['game_total']
        return result
    except Exception:
        return {}
