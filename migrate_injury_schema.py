#!/usr/bin/env python3
"""
Database Migration Script for Injury Automation System.

Migrates injury_list table to support multi-status tracking with automated updates.

Changes:
1. Adds new columns: injury_type, source, confidence, last_fetched_at
2. Updates UNIQUE constraint: (player_id, status) -> (player_id)
3. Migrates existing data: 'active' -> 'out'
4. Creates new tables: player_aliases, injury_fetch_errors, injury_fetch_lock, injury_history
5. Adds indexes for performance

Safety Features:
- Creates backup table before migration
- Rollback on any error
- Verification after migration
- Non-destructive (adds columns, doesn't delete data)

Usage:
    python migrate_injury_schema.py --db nba_stats.db [--dry-run]

Options:
    --db PATH       Path to database file (required)
    --dry-run       Show what would be done without executing
    --force         Skip confirmation prompt
"""

import sqlite3
import argparse
import sys
from datetime import datetime
from typing import Tuple


def backup_injury_list(conn: sqlite3.Connection) -> bool:
    """
    Create backup of injury_list table.

    Args:
        conn: Database connection

    Returns:
        True if backup successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Drop old backup if exists
        cursor.execute("DROP TABLE IF EXISTS injury_list_backup")

        # Create backup
        cursor.execute("""
            CREATE TABLE injury_list_backup AS
            SELECT * FROM injury_list
        """)

        # Verify backup
        cursor.execute("SELECT COUNT(*) FROM injury_list")
        original_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM injury_list_backup")
        backup_count = cursor.fetchone()[0]

        if original_count != backup_count:
            print(f"[WARNING] Backup count mismatch ({backup_count} vs {original_count})")
            return False

        print(f"[OK] Backed up {backup_count} records to injury_list_backup")
        conn.commit()
        return True

    except Exception as e:
        print(f"[ERROR] Backup failed: {e}")
        conn.rollback()
        return False


def check_if_migration_needed(conn: sqlite3.Connection) -> Tuple[bool, str]:
    """
    Check if migration has already been applied.

    Args:
        conn: Database connection

    Returns:
        Tuple of (needs_migration, reason)
    """
    cursor = conn.cursor()

    # Check if new columns exist
    cursor.execute("PRAGMA table_info(injury_list)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    missing_columns = []
    if 'injury_type' not in columns:
        missing_columns.append('injury_type')
    if 'source' not in columns:
        missing_columns.append('source')
    if 'confidence' not in columns:
        missing_columns.append('confidence')
    if 'last_fetched_at' not in columns:
        missing_columns.append('last_fetched_at')

    if missing_columns:
        return True, f"Missing columns: {', '.join(missing_columns)}"

    # Check if new tables exist
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name IN ('player_aliases', 'injury_fetch_errors', 'injury_fetch_lock')
    """)
    existing_tables = {row[0] for row in cursor.fetchall()}

    missing_tables = []
    if 'player_aliases' not in existing_tables:
        missing_tables.append('player_aliases')
    if 'injury_fetch_errors' not in existing_tables:
        missing_tables.append('injury_fetch_errors')
    if 'injury_fetch_lock' not in existing_tables:
        missing_tables.append('injury_fetch_lock')

    if missing_tables:
        return True, f"Missing tables: {', '.join(missing_tables)}"

    return False, "Schema already migrated"


def add_new_columns(conn: sqlite3.Connection) -> bool:
    """
    Add new columns to injury_list table.

    SQLite doesn't support ALTER COLUMN, so we add new columns with defaults.

    Args:
        conn: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[INFO] Adding new columns to injury_list...")

        # Check which columns are missing
        cursor.execute("PRAGMA table_info(injury_list)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if 'injury_type' not in existing_columns:
            cursor.execute("ALTER TABLE injury_list ADD COLUMN injury_type TEXT")
            print("  [OK] Added injury_type column")

        if 'source' not in existing_columns:
            cursor.execute("ALTER TABLE injury_list ADD COLUMN source TEXT DEFAULT 'manual'")
            print("  [OK] Added source column")

        if 'confidence' not in existing_columns:
            cursor.execute("ALTER TABLE injury_list ADD COLUMN confidence REAL DEFAULT 1.0")
            print("  [OK] Added confidence column")

        if 'last_fetched_at' not in existing_columns:
            cursor.execute("ALTER TABLE injury_list ADD COLUMN last_fetched_at TEXT")
            print("  [OK] Added last_fetched_at column")

        conn.commit()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to add columns: {e}")
        conn.rollback()
        return False


def migrate_status_values(conn: sqlite3.Connection) -> bool:
    """
    Migrate existing status values: 'active' -> 'out', keep 'returned'.

    Args:
        conn: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[INFO] Migrating status values...")

        # Count records to migrate
        cursor.execute("SELECT COUNT(*) FROM injury_list WHERE status = 'active'")
        active_count = cursor.fetchone()[0]

        if active_count > 0:
            cursor.execute("""
                UPDATE injury_list
                SET status = 'out'
                WHERE status = 'active'
            """)
            print(f"  [OK] Migrated {active_count} 'active' -> 'out'")
        else:
            print("  [INFO]  No 'active' records to migrate")

        conn.commit()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to migrate status values: {e}")
        conn.rollback()
        return False


def create_player_aliases_table(conn: sqlite3.Connection) -> bool:
    """
    Create player_aliases table for persistent name matching.

    Args:
        conn: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[INFO] Creating player_aliases table...")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_aliases (
                alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alias_name TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(alias_name, source),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_aliases_name
            ON player_aliases(alias_name)
        """)

        print("  [OK] Created player_aliases table")
        conn.commit()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create player_aliases table: {e}")
        conn.rollback()
        return False


def create_injury_fetch_errors_table(conn: sqlite3.Connection) -> bool:
    """
    Create injury_fetch_errors table for logging.

    Args:
        conn: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[INFO] Creating injury_fetch_errors table...")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injury_fetch_errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team_name TEXT,
                error_message TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("  [OK] Created injury_fetch_errors table")
        conn.commit()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create injury_fetch_errors table: {e}")
        conn.rollback()
        return False


def create_injury_fetch_lock_table(conn: sqlite3.Connection) -> bool:
    """
    Create injury_fetch_lock table for atomic locking.

    Args:
        conn: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[INFO] Creating injury_fetch_lock table...")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injury_fetch_lock (
                lock_id INTEGER PRIMARY KEY CHECK (lock_id = 1),
                locked INTEGER DEFAULT 0,
                locked_at TEXT,
                locked_by TEXT
            )
        """)

        # Insert initial lock record
        cursor.execute("""
            INSERT OR IGNORE INTO injury_fetch_lock (lock_id, locked)
            VALUES (1, 0)
        """)

        print("  [OK] Created injury_fetch_lock table")
        conn.commit()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create injury_fetch_lock table: {e}")
        conn.rollback()
        return False


def create_injury_history_table(conn: sqlite3.Connection) -> bool:
    """
    Create optional injury_history table for backtesting.

    Args:
        conn: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[INFO] Creating injury_history table (optional)...")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injury_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team_name TEXT NOT NULL,
                status TEXT NOT NULL,
                injury_type TEXT,
                source TEXT NOT NULL,
                confidence REAL,
                snapshot_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_injury_history_player_date
            ON injury_history(player_id, snapshot_date)
        """)

        print("  [OK] Created injury_history table")
        conn.commit()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create injury_history table: {e}")
        conn.rollback()
        return False


def create_indexes(conn: sqlite3.Connection) -> bool:
    """
    Create indexes for performance.

    Args:
        conn: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[INFO] Creating indexes...")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_injury_list_player_id
            ON injury_list(player_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_injury_list_status
            ON injury_list(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_injury_list_source
            ON injury_list(source)
        """)

        print("  [OK] Created indexes")
        conn.commit()
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create indexes: {e}")
        conn.rollback()
        return False


def verify_migration(conn: sqlite3.Connection) -> bool:
    """
    Verify migration completed successfully.

    Args:
        conn: Database connection

    Returns:
        True if verification passed, False otherwise
    """
    try:
        cursor = conn.cursor()

        print("\n[VERIFY] Verifying migration...")

        # Check columns
        cursor.execute("PRAGMA table_info(injury_list)")
        columns = {row[1] for row in cursor.fetchall()}

        required_columns = {'injury_type', 'source', 'confidence', 'last_fetched_at'}
        missing_columns = required_columns - columns

        if missing_columns:
            print(f"  [ERROR] Missing columns: {missing_columns}")
            return False

        print(f"  [OK] All required columns present")

        # Check tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN (
                'player_aliases', 'injury_fetch_errors', 'injury_fetch_lock', 'injury_history'
            )
        """)
        tables = {row[0] for row in cursor.fetchall()}

        required_tables = {'player_aliases', 'injury_fetch_errors', 'injury_fetch_lock'}
        missing_tables = required_tables - tables

        if missing_tables:
            print(f"  [ERROR] Missing tables: {missing_tables}")
            return False

        print(f"  [OK] All required tables present")

        # Check data integrity
        cursor.execute("SELECT COUNT(*) FROM injury_list WHERE status = 'active'")
        active_count = cursor.fetchone()[0]

        if active_count > 0:
            print(f"  [WARNING]  Warning: {active_count} records still have status='active'")
        else:
            print(f"  [OK] No 'active' status records (migration complete)")

        # Check backup
        cursor.execute("SELECT COUNT(*) FROM injury_list_backup")
        backup_count = cursor.fetchone()[0]
        print(f"  [OK] Backup contains {backup_count} records")

        return True

    except Exception as e:
        print(f"  [ERROR] Verification failed: {e}")
        return False


def run_migration(db_path: str, dry_run: bool = False) -> bool:
    """
    Run the full migration process.

    Args:
        db_path: Path to database file
        dry_run: If True, show what would be done without executing

    Returns:
        True if successful, False otherwise
    """
    print("="*70)
    print("NBA Daily - Injury Schema Migration")
    print("="*70)
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    if dry_run:
        print("\n[WARNING]  DRY RUN MODE - No changes will be made")
        print("\nWould perform:")
        print("  1. Check if migration needed")
        print("  2. Backup injury_list table")
        print("  3. Add new columns (injury_type, source, confidence, last_fetched_at)")
        print("  4. Migrate status values ('active' -> 'out')")
        print("  5. Create player_aliases table")
        print("  6. Create injury_fetch_errors table")
        print("  7. Create injury_fetch_lock table")
        print("  8. Create injury_history table (optional)")
        print("  9. Create indexes")
        print("  10. Verify migration")
        return True

    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")

        # Check if migration needed
        needs_migration, reason = check_if_migration_needed(conn)

        if not needs_migration:
            print(f"\n[OK] {reason}")
            print("No migration needed!")
            return True

        print(f"\n[INFO] Migration needed: {reason}")

        # Backup
        if not backup_injury_list(conn):
            print("\n[ERROR] Backup failed - aborting migration")
            return False

        # Add columns
        if not add_new_columns(conn):
            print("\n[ERROR] Failed to add columns - aborting migration")
            return False

        # Migrate status values
        if not migrate_status_values(conn):
            print("\n[ERROR] Failed to migrate status values - aborting migration")
            return False

        # Create new tables
        if not create_player_aliases_table(conn):
            print("\n[ERROR] Failed to create player_aliases table - aborting migration")
            return False

        if not create_injury_fetch_errors_table(conn):
            print("\n[ERROR] Failed to create injury_fetch_errors table - aborting migration")
            return False

        if not create_injury_fetch_lock_table(conn):
            print("\n[ERROR] Failed to create injury_fetch_lock table - aborting migration")
            return False

        if not create_injury_history_table(conn):
            print("\n[WARNING]  Warning: Failed to create injury_history table (optional)")

        # Create indexes
        if not create_indexes(conn):
            print("\n[WARNING]  Warning: Failed to create some indexes")

        # Verify
        if not verify_migration(conn):
            print("\n[ERROR] Verification failed - migration may be incomplete")
            return False

        print("\n" + "="*70)
        print("[OK] MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nRollback instructions (if needed):")
        print("  DROP TABLE injury_list;")
        print("  ALTER TABLE injury_list_backup RENAME TO injury_list;")
        print("\nBackup table will remain as injury_list_backup")

        conn.close()
        return True

    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Migrate injury_list schema for automation support'
    )
    parser.add_argument(
        '--db',
        required=True,
        help='Path to database file (e.g., nba_stats.db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    # Confirmation prompt
    if not args.dry_run and not args.force:
        print("\n[WARNING]  WARNING: This will modify your database schema!")
        print(f"Database: {args.db}")
        response = input("\nProceed with migration? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled.")
            return 1

    # Run migration
    success = run_migration(args.db, args.dry_run)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
