import sqlite3

import injury_adjustment as ia
import injury_config as config


def test_normalize_injury_status_variants():
    assert config.normalize_injury_status("Out For Season") == "out"
    assert config.normalize_injury_status("out indefinitely") == "out"
    assert config.normalize_injury_status("Questionable") == "questionable"
    assert config.normalize_injury_status("Day To Day") == "day-to-day"
    assert config.normalize_injury_status("Available") == "returned"


def test_create_injury_list_table_normalizes_legacy_status_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE injury_list (
            injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_name TEXT NOT NULL,
            injury_date TEXT NOT NULL,
            expected_return_date TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO injury_list (
            player_id, player_name, team_name, injury_date, status
        ) VALUES (?, ?, ?, DATE('now'), ?)
        """,
        (203078, "Bradley Beal", "LAC", "out for season"),
    )
    conn.commit()

    ia.create_injury_list_table(conn)

    status = conn.execute(
        "SELECT status FROM injury_list WHERE player_id = ?",
        (203078,),
    ).fetchone()[0]
    assert status == "out"


def test_get_active_injuries_treats_out_variants_as_excluded():
    conn = sqlite3.connect(":memory:")
    ia.create_injury_list_table(conn)
    ia.add_to_injury_list(
        conn,
        player_id=203078,
        player_name="Bradley Beal",
        team_name="LAC",
        status="out for season",
        source="automated",
    )
    ia.add_to_injury_list(
        conn,
        player_id=203999,
        player_name="Test Questionable",
        team_name="LAC",
        status="questionable",
        source="automated",
    )

    excluded = ia.get_active_injuries(
        conn,
        check_return_dates=False,
        status_filter=["out", "doubtful"],
    )

    assert len(excluded) == 1
    assert excluded[0]["player_name"] == "Bradley Beal"
    assert excluded[0]["status"] == "out"
