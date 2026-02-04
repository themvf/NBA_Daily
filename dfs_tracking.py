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

    # Migration-safe column additions (won't fail if columns already exist)
    migrations = [
        ("dfs_slate_projections", "ownership_proj", "REAL"),
        ("dfs_slate_projections", "actual_ownership", "REAL"),
        ("dfs_slate_results", "ownership_mae", "REAL"),
        ("dfs_slate_results", "ownership_correlation", "REAL"),
    ]
    for table, col, col_type in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

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
        row = cursor.execute(f"""
            SELECT {select_cols}
            FROM player_game_logs
            WHERE player_id = ? AND game_date = ?
        """, (player_id, slate_date)).fetchone()

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
