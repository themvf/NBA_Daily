import sqlite3
from types import SimpleNamespace

import pandas as pd

import ownership_calibrator as oc


def _create_minimal_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE dfs_slate_projections (
            slate_date TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            salary INTEGER,
            positions TEXT,
            proj_fpts REAL,
            proj_floor REAL,
            proj_ceiling REAL,
            fpts_per_dollar REAL,
            ownership_proj REAL,
            actual_ownership REAL,
            actual_fpts REAL,
            actual_minutes REAL,
            did_play INTEGER
        );

        CREATE TABLE dfs_contest_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contest_id TEXT NOT NULL,
            slate_date TEXT NOT NULL,
            username TEXT NOT NULL
        );

        CREATE TABLE predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            game_date TEXT,
            player_id INTEGER,
            projected_ppg REAL,
            proj_confidence REAL,
            season_avg_ppg REAL,
            recent_avg_3 REAL,
            recent_avg_5 REAL,
            dfs_score REAL,
            proj_minutes REAL,
            l5_minutes_avg REAL,
            minutes_confidence REAL,
            role_change REAL,
            p_top1 REAL,
            p_top3 REAL,
            sim_sigma REAL,
            role_tier TEXT,
            vegas_implied_fpts REAL,
            vegas_vs_proj_diff REAL,
            injury_adjusted REAL,
            opponent_injury_detected REAL,
            opponent_def_rating REAL,
            opponent_pace REAL,
            analytics_used TEXT,
            last_refreshed_at TEXT,
            created_at TEXT
        );

        CREATE TABLE dfs_supplement_runs (
            run_key TEXT PRIMARY KEY,
            slate_date TEXT NOT NULL,
            source_name TEXT,
            created_at TEXT
        );

        CREATE TABLE dfs_supplement_player_deltas (
            run_key TEXT NOT NULL,
            slate_date TEXT NOT NULL,
            player_id INTEGER,
            supplement_proj_fpts REAL,
            supplement_own_pct REAL,
            proj_delta REAL,
            own_delta_pp REAL,
            match_score REAL
        );
        """
    )
    conn.commit()


def test_build_ownership_training_dataset_imputes_zero_for_imported_slates():
    conn = sqlite3.connect(":memory:")
    _create_minimal_tables(conn)

    conn.execute(
        """
        INSERT INTO dfs_slate_projections (
            slate_date, player_id, player_name, team, opponent, salary, positions,
            proj_fpts, proj_floor, proj_ceiling, fpts_per_dollar, ownership_proj,
            actual_ownership, actual_fpts, actual_minutes, did_play
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-01",
            1,
            "Alpha Guard",
            "AAA",
            "BBB",
            6200,
            "PG/SG",
            34.0,
            27.0,
            42.0,
            5.5,
            18.0,
            22.0,
            39.0,
            34.0,
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO dfs_slate_projections (
            slate_date, player_id, player_name, team, opponent, salary, positions,
            proj_fpts, proj_floor, proj_ceiling, fpts_per_dollar, ownership_proj,
            actual_ownership, actual_fpts, actual_minutes, did_play
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-01",
            2,
            "Beta Wing",
            "AAA",
            "BBB",
            3600,
            "SF/PF",
            19.0,
            12.0,
            27.0,
            5.3,
            5.0,
            None,
            0.0,
            0.0,
            0,
        ),
    )
    conn.execute(
        """
        INSERT INTO dfs_contest_entries (contest_id, slate_date, username)
        VALUES (?, ?, ?)
        """,
        ("contest-1", "2026-03-01", "tester"),
    )
    conn.commit()

    df = oc.build_ownership_training_dataset(conn)

    assert len(df) == 2
    beta_row = df[df["player_id"] == 2].iloc[0]
    assert beta_row["target_actual_ownership"] == 0.0
    assert beta_row["actual_ownership_imputed_zero"] == 1


def test_fit_ownership_calibrator_builds_and_saves_run():
    conn = sqlite3.connect(":memory:")
    _create_minimal_tables(conn)

    slates = ["2026-02-25", "2026-02-28", "2026-03-04"]
    positions = ["PG", "SG", "SF", "PF", "C"]
    teams = [("AAA", "BBB"), ("CCC", "DDD"), ("EEE", "FFF")]

    for slate in slates:
        conn.execute(
            """
            INSERT INTO dfs_contest_entries (contest_id, slate_date, username)
            VALUES (?, ?, ?)
            """,
            (f"contest-{slate}", slate, "tester"),
        )

    player_id = 100
    for slate_idx, slate in enumerate(slates):
        for idx in range(15):
            team, opponent = teams[idx % len(teams)]
            pos = positions[idx % len(positions)]
            salary = 3200 + (idx * 350)
            proj_fpts = 16.0 + idx + (slate_idx * 0.8)
            proj_floor = proj_fpts - 6.0
            proj_ceiling = proj_fpts + 10.0
            fpts_per_dollar = (proj_fpts / salary) * 1000.0
            our_own = min(55.0, 1.5 + proj_fpts * 0.42 + max(0.0, 6200 - salary) / 900.0)
            rw_own = max(0.0, our_own + (2.5 if idx % 3 == 0 else -1.0))
            ls_own = max(0.0, our_own + (1.5 if idx % 4 == 0 else -1.5))
            actual_own = max(
                0.0,
                min(
                    60.0,
                    (our_own * 0.72)
                    + (rw_own * 0.18)
                    + (ls_own * 0.10)
                    + ((idx % 5) - 2) * 0.6,
                ),
            )

            conn.execute(
                """
                INSERT INTO dfs_slate_projections (
                    slate_date, player_id, player_name, team, opponent, salary, positions,
                    proj_fpts, proj_floor, proj_ceiling, fpts_per_dollar, ownership_proj,
                    actual_ownership, actual_fpts, actual_minutes, did_play
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slate,
                    player_id,
                    f"Player {player_id}",
                    team,
                    opponent,
                    salary,
                    pos,
                    proj_fpts,
                    proj_floor,
                    proj_ceiling,
                    fpts_per_dollar,
                    our_own,
                    actual_own,
                    proj_fpts + 3.0,
                    24.0 + (idx % 8),
                    1,
                ),
            )

            conn.execute(
                """
                INSERT INTO predictions (
                    prediction_date, game_date, player_id, projected_ppg, proj_confidence,
                    season_avg_ppg, recent_avg_3, recent_avg_5, dfs_score, proj_minutes,
                    l5_minutes_avg, minutes_confidence, role_change, p_top1, p_top3,
                    sim_sigma, role_tier, vegas_implied_fpts, vegas_vs_proj_diff,
                    injury_adjusted, opponent_injury_detected, opponent_def_rating,
                    opponent_pace, analytics_used, last_refreshed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slate,
                    slate,
                    player_id,
                    proj_fpts * 0.6,
                    0.55 + ((idx % 5) * 0.05),
                    proj_fpts * 0.5,
                    proj_fpts * 0.56,
                    proj_fpts * 0.59,
                    proj_fpts * 1.2,
                    25.0 + (idx % 7),
                    24.0 + (idx % 6),
                    0.6 + ((idx % 4) * 0.05),
                    float(idx % 2),
                    0.02 * (idx % 5),
                    0.05 * ((idx % 6) + 1),
                    6.0 + (idx % 4),
                    "STARTER" if idx < 10 else "ROTATION",
                    proj_fpts + 1.8,
                    1.8,
                    float(idx % 3 == 0),
                    float(idx % 4 == 0),
                    112.0 + (idx % 3),
                    99.0 + (idx % 4),
                    "PACE|INJ" if idx % 3 == 0 else "DEF",
                    f"{slate}T12:00:00",
                    f"{slate}T12:00:00",
                ),
            )

            player_id += 1

    supplement_runs = [
        ("run-rw-2026-02-25", "2026-02-25", "RotoWire NBA Optimizer"),
        ("run-ls-2026-02-25", "2026-02-25", "LineupStarter CSV"),
        ("run-rw-2026-02-28", "2026-02-28", "RotoWire NBA Optimizer"),
        ("run-ls-2026-02-28", "2026-02-28", "LineupStarter CSV"),
    ]
    for run_key, slate, source_name in supplement_runs:
        conn.execute(
            """
            INSERT INTO dfs_supplement_runs (run_key, slate_date, source_name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_key, slate, source_name, f"{slate}T10:00:00"),
        )

    df = pd.read_sql_query(
        """
        SELECT slate_date, player_id, ownership_proj, proj_fpts
        FROM dfs_slate_projections
        WHERE slate_date IN ('2026-02-25', '2026-02-28')
        """,
        conn,
    )
    for row in df.to_dict("records"):
        slate = row["slate_date"]
        player = int(row["player_id"])
        own = float(row["ownership_proj"])
        proj = float(row["proj_fpts"])
        conn.execute(
            """
            INSERT INTO dfs_supplement_player_deltas (
                run_key, slate_date, player_id, supplement_proj_fpts, supplement_own_pct,
                proj_delta, own_delta_pp, match_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"run-rw-{slate}",
                slate,
                player,
                proj + 0.8,
                max(0.0, own + 1.5),
                0.8,
                1.5,
                99.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO dfs_supplement_player_deltas (
                run_key, slate_date, player_id, supplement_proj_fpts, supplement_own_pct,
                proj_delta, own_delta_pp, match_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"run-ls-{slate}",
                slate,
                player,
                proj + 0.4,
                max(0.0, own - 0.8),
                0.4,
                -0.8,
                97.0,
            ),
        )
    conn.commit()

    result = oc.fit_ownership_calibrator(conn, holdout_slates=1, save_run=True)

    assert "error" not in result
    summary = result["summary"]
    prediction_df = result["prediction_df"]

    assert summary.total_rows == 45
    assert summary.train_rows > 0
    assert summary.test_rows > 0
    assert summary.source_rows_train > 0
    assert "final_ownership_pred" in prediction_df.columns
    assert conn.execute(
        "SELECT COUNT(*) FROM dfs_ownership_calibration_runs"
    ).fetchone()[0] == 1

    latest = oc.get_latest_ownership_calibration_run(conn)
    assert latest is not None
    assert latest["run_key"] == summary.run_key


def test_apply_live_ownership_calibration_updates_runtime_players() -> None:
    conn = sqlite3.connect(":memory:")
    _create_minimal_tables(conn)

    training_slates = ["2026-03-01", "2026-03-03", "2026-03-05"]
    player_id = 200
    for slate in training_slates:
        conn.execute(
            """
            INSERT INTO dfs_contest_entries (contest_id, slate_date, username)
            VALUES (?, ?, ?)
            """,
            (f"contest-{slate}", slate, "tester"),
        )

    for slate_idx, slate in enumerate(training_slates):
        conn.execute(
            """
            INSERT INTO dfs_supplement_runs (run_key, slate_date, source_name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (f"rw-{slate}", slate, "RotoWire NBA Optimizer", f"{slate}T19:00:00"),
        )
        for idx in range(12):
            salary = 3600 + (idx * 420)
            proj_fpts = 18.0 + idx + (slate_idx * 0.7)
            proj_floor = proj_fpts - 5.5
            proj_ceiling = proj_fpts + 9.0
            fpts_per_dollar = (proj_fpts / salary) * 1000.0
            our_own = max(1.0, min(40.0, 3.0 + proj_fpts * 0.35 - max(0, salary - 6800) / 1200.0))
            rw_own = max(0.0, our_own + (4.0 if salary <= 5600 else -2.0))
            actual_own = max(
                0.0,
                min(
                    60.0,
                    (our_own * 0.55)
                    + (rw_own * 0.45)
                    + (1.2 if idx % 4 == 0 else -0.6),
                ),
            )

            conn.execute(
                """
                INSERT INTO dfs_slate_projections (
                    slate_date, player_id, player_name, team, opponent, salary, positions,
                    proj_fpts, proj_floor, proj_ceiling, fpts_per_dollar, ownership_proj,
                    actual_ownership, actual_fpts, actual_minutes, did_play
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slate,
                    player_id,
                    f"Player {player_id}",
                    f"T{idx % 6}",
                    f"O{idx % 6}",
                    salary,
                    "PG/SG" if idx % 2 == 0 else "SF/PF",
                    proj_fpts,
                    proj_floor,
                    proj_ceiling,
                    fpts_per_dollar,
                    our_own,
                    actual_own,
                    proj_fpts + 2.0,
                    26.0 + (idx % 6),
                    1,
                ),
            )
            conn.execute(
                """
                INSERT INTO predictions (
                    prediction_date, game_date, player_id, projected_ppg, proj_confidence,
                    season_avg_ppg, recent_avg_3, recent_avg_5, dfs_score, proj_minutes,
                    l5_minutes_avg, minutes_confidence, role_change, p_top1, p_top3,
                    sim_sigma, role_tier, vegas_implied_fpts, vegas_vs_proj_diff,
                    injury_adjusted, opponent_injury_detected, opponent_def_rating,
                    opponent_pace, analytics_used, last_refreshed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slate,
                    slate,
                    player_id,
                    proj_fpts * 0.58,
                    0.6,
                    proj_fpts * 0.5,
                    proj_fpts * 0.56,
                    proj_fpts * 0.59,
                    proj_fpts * 1.15,
                    27.0,
                    26.0,
                    0.68,
                    0.0,
                    0.01 * idx,
                    0.04 * ((idx % 5) + 1),
                    6.5,
                    "STARTER",
                    proj_fpts + 1.0,
                    1.0,
                    0.0,
                    0.0,
                    112.0,
                    99.0,
                    "PACE",
                    f"{slate}T12:00:00",
                    f"{slate}T12:00:00",
                ),
            )
            conn.execute(
                """
                INSERT INTO dfs_supplement_player_deltas (
                    run_key, slate_date, player_id, supplement_proj_fpts, supplement_own_pct,
                    proj_delta, own_delta_pp, match_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"rw-{slate}",
                    slate,
                    player_id,
                    proj_fpts + 0.5,
                    rw_own,
                    0.5,
                    rw_own - our_own,
                    99.0,
                ),
            )
            player_id += 1

    live_players = [
        SimpleNamespace(
            player_id=901,
            name="Live Value",
            team="AAA",
            opponent="BBB",
            positions=["PG"],
            salary=4800,
            proj_fpts=27.0,
            proj_floor=20.0,
            proj_ceiling=36.0,
            fpts_per_dollar=(27.0 / 4800) * 1000.0,
            ownership_proj=10.0,
            recent_minutes_avg=31.0,
            role_tier="STARTER",
            analytics_used="PACE",
        ),
        SimpleNamespace(
            player_id=902,
            name="Live Chalk",
            team="CCC",
            opponent="DDD",
            positions=["SF"],
            salary=9200,
            proj_fpts=45.0,
            proj_floor=35.0,
            proj_ceiling=58.0,
            fpts_per_dollar=(45.0 / 9200) * 1000.0,
            ownership_proj=28.0,
            recent_minutes_avg=35.0,
            role_tier="STARTER",
            analytics_used="",
        ),
    ]

    supplement_state = {
        "source_name": "RotoWire NBA Optimizer",
        "player_map": {
            901: {
                "supplement_proj_fpts": 27.5,
                "supplement_ownership": 16.5,
                "proj_delta": 0.5,
                "own_delta_pp": 6.5,
                "match_score": 99.0,
            },
            902: {
                "supplement_proj_fpts": 44.0,
                "supplement_ownership": 21.0,
                "proj_delta": -1.0,
                "own_delta_pp": -7.0,
                "match_score": 99.0,
            },
        },
    }

    result = oc.apply_live_ownership_calibration(
        conn,
        live_players,
        "2026-03-08",
        supplement_state=supplement_state,
        min_train_rows=20,
    )

    assert result["active"] is True
    assert result["train_rows"] >= 20
    assert result["calibrated_players"] == 2
    assert live_players[0].ownership_proj != 10.0
    assert live_players[1].ownership_proj != 28.0
    assert getattr(live_players[0], "_ownership_calibration_base_proj") == 10.0


def test_build_live_supplement_frame_preserves_multiple_raw_sources() -> None:
    state = {
        "source_name": "RotoWire NBA Optimizer + LineupStarter CSV (Signal Matrix)",
        "source_player_maps": [
            {
                "source_name": "RotoWire NBA Optimizer",
                "player_map": {
                    1: {
                        "supplement_proj_fpts": 30.0,
                        "supplement_ownership": 18.0,
                        "proj_delta": 1.0,
                        "own_delta_pp": 5.0,
                        "match_score": 99.0,
                    }
                },
            },
            {
                "source_name": "LineupStarter CSV",
                "player_map": {
                    1: {
                        "supplement_proj_fpts": 31.0,
                        "supplement_ownership": 16.0,
                        "proj_delta": 2.0,
                        "own_delta_pp": 3.0,
                        "match_score": 97.0,
                    }
                },
            },
        ],
    }

    df = oc._build_live_supplement_frame("2026-03-08", state)

    assert list(df["player_id"]) == [1]
    row = df.iloc[0]
    assert row["rw_own_pct"] == 18.0
    assert row["ls_own_pct"] == 16.0
    assert row["rw_proj_fpts"] == 30.0
    assert row["ls_proj_fpts"] == 31.0
