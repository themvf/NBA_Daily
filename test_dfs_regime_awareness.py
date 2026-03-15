import sqlite3

from dfs_optimizer import (
    DFSLineup,
    DFSPlayer,
    get_lineup_model_profiles,
    resolve_lineup_generation_regime,
)
from dfs_tracking import (
    build_tournament_postmortem,
    create_dfs_tracking_tables,
    save_slate_lineups,
    save_slate_projections,
)


def _make_player(
    player_id: int,
    name: str,
    team: str,
    opponent: str,
    game_id: str,
    positions: list[str],
    salary: int,
    proj_fpts: float,
    ownership_proj: float,
) -> DFSPlayer:
    player = DFSPlayer(
        player_id=player_id,
        name=name,
        team=team,
        opponent=opponent,
        game_id=game_id,
        positions=positions,
        salary=salary,
    )
    player.proj_fpts = proj_fpts
    player.proj_floor = max(0.0, proj_fpts - 6.0)
    player.proj_ceiling = proj_fpts + 10.0
    player.fpts_per_dollar = (proj_fpts / salary) * 1000.0
    player.ownership_proj = ownership_proj
    return player


def _insert_contest_entry(
    conn: sqlite3.Connection,
    *,
    contest_id: str,
    slate_date: str,
    username: str,
    rank: int,
    points: float,
    players: list[str],
    total_salary: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO dfs_contest_entries (
            contest_id, slate_date, username, max_entries, entry_count, rank, points,
            lineup_raw, pg, sg, sf, pf, c, g, f, util, total_salary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            contest_id,
            slate_date,
            username,
            1,
            1,
            rank,
            points,
            " | ".join(players),
            *players,
            total_salary,
        ),
    )


def test_resolve_lineup_generation_regime_auto_small_slate_small_field() -> None:
    regime = resolve_lineup_generation_regime(
        player_pool=[],
        num_lineups=20,
        regime_hint="auto",
        slate_game_count=3,
        contest_field_size=250,
    )

    assert regime["slate_bucket"] == "small_slate"
    assert regime["field_bucket"] == "small_field"
    assert regime["contest_field_size"] == 250
    assert regime["overlap_cap_delta"] == 2
    assert regime["ceiling_focus_delta"] == -20


def test_resolve_lineup_generation_regime_auto_large_slate_large_field() -> None:
    regime = resolve_lineup_generation_regime(
        player_pool=[],
        num_lineups=3,
        regime_hint="auto",
        slate_game_count=9,
        contest_field_size=20000,
    )

    assert regime["slate_bucket"] == "large_slate"
    assert regime["field_bucket"] == "large_field"
    assert regime["contest_field_size"] == 20000
    assert regime["overlap_cap_delta"] == -2
    assert regime["force_aggressive_ceiling_stack"] is True


def test_legacy_rotowire_model_is_hidden_and_not_source_gated() -> None:
    profiles = get_lineup_model_profiles()

    legacy_profile = profiles["rotowire_both_v1_blend"]

    assert legacy_profile["ui_hidden"] is True
    assert not legacy_profile.get("supplement_source_required")


def test_postmortem_exposes_saved_regime_breakdown() -> None:
    conn = sqlite3.connect(":memory:")
    create_dfs_tracking_tables(conn)

    slate_date = "2026-03-08"
    players = [
        _make_player(1, "Alpha Guard", "AAA", "BBB", "AAA_BBB", ["PG"], 7000, 38.0, 18.0),
        _make_player(2, "Bravo Wing", "AAA", "BBB", "AAA_BBB", ["SG"], 6800, 36.0, 16.0),
        _make_player(3, "Charlie Forward", "CCC", "DDD", "CCC_DDD", ["SF"], 6400, 34.0, 14.0),
        _make_player(4, "Delta Big", "CCC", "DDD", "CCC_DDD", ["PF"], 6200, 32.0, 12.0),
        _make_player(5, "Echo Center", "EEE", "FFF", "EEE_FFF", ["C"], 7600, 40.0, 20.0),
        _make_player(6, "Foxtrot Guard", "EEE", "FFF", "EEE_FFF", ["PG", "SG"], 5600, 29.0, 9.0),
        _make_player(7, "Golf Forward", "GGG", "HHH", "GGG_HHH", ["SF", "PF"], 5400, 28.0, 8.0),
        _make_player(8, "Hotel Utility", "GGG", "HHH", "GGG_HHH", ["SG", "SF"], 5000, 27.0, 7.0),
    ]
    save_slate_projections(conn, slate_date, players)

    for player in players:
        conn.execute(
            """
            UPDATE dfs_slate_projections
            SET actual_fpts = ?, actual_ownership = ?, did_play = 1
            WHERE slate_date = ? AND player_id = ?
            """,
            (
                player.proj_fpts + 2.0,
                player.ownership_proj + 1.5,
                slate_date,
                player.player_id,
            ),
        )

    lineup = DFSLineup(
        players={
            "PG": players[0],
            "SG": players[1],
            "SF": players[2],
            "PF": players[3],
            "C": players[4],
            "G": players[5],
            "F": players[6],
            "UTIL": players[7],
        },
        model_key="standout_v1_capture",
        model_label="Standout v1 (Missed-Capture)",
        generation_strategy="ceiling",
        regime_key="small_slate__small_field",
        regime_label="Small Slate + Small Field / SE / 3-max",
        regime_notes="4 games | overlap +2 | field 250 | core x1.50 | low-own x0.52 | standout x0.72 | ceiling -20",
        regime_hint="auto",
        contest_field_size=250,
    )
    save_slate_lineups(conn, slate_date, [lineup])
    conn.execute(
        """
        UPDATE dfs_slate_lineups
        SET total_actual_fpts = ?
        WHERE slate_date = ? AND lineup_num = 1
        """,
        (290.0, slate_date),
    )

    conn.execute(
        """
        INSERT INTO dfs_contest_entries (
            contest_id, slate_date, username, max_entries, entry_count, rank, points,
            lineup_raw, pg, sg, sf, pf, c, g, f, util, total_salary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "contest-1",
            slate_date,
            "shark_user",
            1,
            1,
            1,
            295.0,
            "alpha-bravo-charlie-delta-echo-foxtrot-golf-hotel",
            players[0].name,
            players[1].name,
            players[2].name,
            players[3].name,
            players[4].name,
            players[5].name,
            players[6].name,
            players[7].name,
            lineup.total_salary,
        ),
    )
    conn.execute(
        """
        INSERT INTO dfs_contest_meta (contest_id, slate_date, total_entries, unique_users, top_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("contest-1", slate_date, 1, 1, 295.0),
    )
    conn.commit()

    payload = build_tournament_postmortem(conn, slate_date, top_n=1)

    assert not payload["errors"]
    assert payload["metrics"]["active_regime_label"] == "Small Slate + Small Field / SE / 3-max"
    assert payload["metrics"]["active_regime_hint"] == "auto"
    assert payload["metrics"]["active_contest_field_size"] == 250
    assert not payload["regime_breakdown_df"].empty
    assert "regime_label" in payload["our_lineup_structures_df"].columns
    assert "contest_field_size" in payload["our_lineup_structures_df"].columns
    assert payload["regime_breakdown_df"].iloc[0]["regime_label"] == "Small Slate + Small Field / SE / 3-max"


def test_postmortem_selects_best_matching_contest_and_uses_consistent_winner_score() -> None:
    conn = sqlite3.connect(":memory:")
    create_dfs_tracking_tables(conn)

    slate_date = "2026-03-09"
    players = [
        _make_player(1, "Alpha Guard", "AAA", "BBB", "AAA_BBB", ["PG"], 7000, 38.0, 18.0),
        _make_player(2, "Bravo Wing", "AAA", "BBB", "AAA_BBB", ["SG"], 6800, 36.0, 16.0),
        _make_player(3, "Charlie Forward", "CCC", "DDD", "CCC_DDD", ["SF"], 6400, 34.0, 14.0),
        _make_player(4, "Delta Big", "CCC", "DDD", "CCC_DDD", ["PF"], 6200, 32.0, 12.0),
        _make_player(5, "Echo Center", "EEE", "FFF", "EEE_FFF", ["C"], 7600, 40.0, 20.0),
        _make_player(6, "Foxtrot Guard", "EEE", "FFF", "EEE_FFF", ["PG", "SG"], 5600, 29.0, 9.0),
        _make_player(7, "Golf Forward", "GGG", "HHH", "GGG_HHH", ["SF", "PF"], 5400, 28.0, 8.0),
        _make_player(8, "Hotel Utility", "GGG", "HHH", "GGG_HHH", ["SG", "SF"], 5000, 27.0, 7.0),
    ]
    save_slate_projections(conn, slate_date, players)

    for player in players:
        conn.execute(
            """
            UPDATE dfs_slate_projections
            SET actual_fpts = ?, actual_ownership = ?, did_play = 1
            WHERE slate_date = ? AND player_id = ?
            """,
            (
                player.proj_fpts + 3.0,
                player.ownership_proj + 1.0,
                slate_date,
                player.player_id,
            ),
        )

    lineup = DFSLineup(
        players={
            "PG": players[0],
            "SG": players[1],
            "SF": players[2],
            "PF": players[3],
            "C": players[4],
            "G": players[5],
            "F": players[6],
            "UTIL": players[7],
        },
        model_key="rotowire_both_v1_blend",
        model_label="RotoWire v1 (Proj+Own Blend)",
        generation_strategy="value",
        regime_key="small_slate__medium_field",
        regime_label="Small Slate + Mid-Field / 20-max",
        regime_notes="3 games | overlap +1 | field 1800 | core x1.25 | low-own x0.70 | standout x0.85 | ceiling -10",
        regime_hint="auto",
        contest_field_size=1800,
    )
    save_slate_lineups(conn, slate_date, [lineup])
    conn.execute(
        """
        UPDATE dfs_slate_lineups
        SET total_actual_fpts = ?
        WHERE slate_date = ? AND lineup_num = 1
        """,
        (275.0, slate_date),
    )

    valid_players = [player.name for player in players]
    _insert_contest_entry(
        conn,
        contest_id="contest-good",
        slate_date=slate_date,
        username="valid_user",
        rank=1,
        points=295.0,
        players=valid_players,
        total_salary=lineup.total_salary,
    )
    conn.execute(
        """
        INSERT INTO dfs_contest_meta (contest_id, slate_date, total_entries, unique_users, top_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("contest-good", slate_date, 1800, 1, 280.0),
    )

    _insert_contest_entry(
        conn,
        contest_id="contest-bad",
        slate_date=slate_date,
        username="wrong_user",
        rank=1,
        points=350.75,
        players=[
            "Wrong One",
            "Wrong Two",
            "Wrong Three",
            "Wrong Four",
            "Wrong Five",
            "Wrong Six",
            "Wrong Seven",
            "Wrong Eight",
        ],
        total_salary=None,
    )
    conn.execute(
        """
        INSERT INTO dfs_contest_meta (contest_id, slate_date, total_entries, unique_users, top_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("contest-bad", slate_date, 5040, 1, 350.75),
    )
    conn.commit()

    payload = build_tournament_postmortem(conn, slate_date, top_n=1)

    assert not payload["errors"]
    assert payload["metrics"]["selected_contest_id"] == "contest-good"
    assert payload["metrics"]["selected_contest_count"] == 2
    assert payload["metrics"]["selected_contest_slot_match_pct"] == 100.0
    assert payload["metrics"]["winner_score"] == 295.0
    assert payload["metrics"]["winner_score_source"] == "contest_entries"
    assert payload["metrics"]["contest_meta_top_score"] == 280.0
    assert payload["metrics"]["topn_best_score"] == 295.0
    assert payload["metrics"]["field_lineups_available"] == 1
    assert payload["field_lineup_structures_df"].iloc[0]["contest_points"] == 295.0
    assert set(payload["top_field_players_df"]["display_name"]) == set(valid_players)
    assert payload["top_field_players_df"]["player_id"].notna().all()


def test_postmortem_excludes_unresolved_topn_lineups_from_exposure_analysis() -> None:
    conn = sqlite3.connect(":memory:")
    create_dfs_tracking_tables(conn)

    slate_date = "2026-03-10"
    players = [
        _make_player(1, "Alpha Guard", "AAA", "BBB", "AAA_BBB", ["PG"], 7000, 38.0, 18.0),
        _make_player(2, "Bravo Wing", "AAA", "BBB", "AAA_BBB", ["SG"], 6800, 36.0, 16.0),
        _make_player(3, "Charlie Forward", "CCC", "DDD", "CCC_DDD", ["SF"], 6400, 34.0, 14.0),
        _make_player(4, "Delta Big", "CCC", "DDD", "CCC_DDD", ["PF"], 6200, 32.0, 12.0),
        _make_player(5, "Echo Center", "EEE", "FFF", "EEE_FFF", ["C"], 7600, 40.0, 20.0),
        _make_player(6, "Foxtrot Guard", "EEE", "FFF", "EEE_FFF", ["PG", "SG"], 5600, 29.0, 9.0),
        _make_player(7, "Golf Forward", "GGG", "HHH", "GGG_HHH", ["SF", "PF"], 5400, 28.0, 8.0),
        _make_player(8, "Hotel Utility", "GGG", "HHH", "GGG_HHH", ["SG", "SF"], 5000, 27.0, 7.0),
    ]
    save_slate_projections(conn, slate_date, players)

    for player in players:
        conn.execute(
            """
            UPDATE dfs_slate_projections
            SET actual_fpts = ?, actual_ownership = ?, did_play = 1
            WHERE slate_date = ? AND player_id = ?
            """,
            (
                player.proj_fpts + 2.5,
                player.ownership_proj + 1.0,
                slate_date,
                player.player_id,
            ),
        )

    lineup = DFSLineup(
        players={
            "PG": players[0],
            "SG": players[1],
            "SF": players[2],
            "PF": players[3],
            "C": players[4],
            "G": players[5],
            "F": players[6],
            "UTIL": players[7],
        },
        model_key="standard_v1",
        model_label="Standard v1",
        generation_strategy="projection",
        regime_key="small_slate__medium_field",
        regime_label="Small Slate + Mid-Field / 20-max",
        regime_notes="test",
        regime_hint="auto",
        contest_field_size=1800,
    )
    save_slate_lineups(conn, slate_date, [lineup])
    conn.execute(
        """
        UPDATE dfs_slate_lineups
        SET total_actual_fpts = ?
        WHERE slate_date = ? AND lineup_num = 1
        """,
        (282.0, slate_date),
    )

    valid_players = [player.name for player in players]
    _insert_contest_entry(
        conn,
        contest_id="contest-cleanup",
        slate_date=slate_date,
        username="resolved_user",
        rank=1,
        points=300.0,
        players=valid_players,
        total_salary=lineup.total_salary,
    )
    _insert_contest_entry(
        conn,
        contest_id="contest-cleanup",
        slate_date=slate_date,
        username="bad_alias_user",
        rank=2,
        points=298.0,
        players=[
            players[0].name,
            players[1].name,
            players[2].name,
            players[3].name,
            "Luka Doncic",
            players[5].name,
            players[6].name,
            players[7].name,
        ],
        total_salary=lineup.total_salary,
    )
    conn.execute(
        """
        INSERT INTO dfs_contest_meta (contest_id, slate_date, total_entries, unique_users, top_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("contest-cleanup", slate_date, 1800, 2, 300.0),
    )
    conn.commit()

    payload = build_tournament_postmortem(conn, slate_date, top_n=2)

    assert not payload["errors"]
    assert payload["metrics"]["field_lineups_raw_sampled"] == 2
    assert payload["metrics"]["field_lineups_analyzed"] == 1
    assert payload["metrics"]["field_unresolved_topn_count"] == 1
    assert payload["metrics"]["field_resolved_topn_pct"] == 50.0
    assert "Luka Doncic" not in set(payload["top_field_players_df"]["display_name"])
    assert payload["field_lineup_structures_df"].shape[0] == 1
    assert any(
        "Excluded 1 unresolved top-2 field lineup" in note
        for note in payload["wrong_notes"]
    )


def test_postmortem_exports_source_vs_actual_and_exclusion_attribution() -> None:
    conn = sqlite3.connect(":memory:")
    create_dfs_tracking_tables(conn)

    slate_date = "2026-03-11"
    players = [
        _make_player(1, "Alpha Guard", "AAA", "BBB", "AAA_BBB", ["PG"], 7000, 38.0, 18.0),
        _make_player(2, "Bravo Wing", "AAA", "BBB", "AAA_BBB", ["SG"], 6800, 36.0, 16.0),
        _make_player(3, "Charlie Forward", "CCC", "DDD", "CCC_DDD", ["SF"], 6400, 34.0, 14.0),
        _make_player(4, "Delta Big", "CCC", "DDD", "CCC_DDD", ["PF"], 6200, 32.0, 12.0),
        _make_player(5, "Echo Center", "EEE", "FFF", "EEE_FFF", ["C"], 7600, 40.0, 20.0),
        _make_player(6, "Foxtrot Guard", "EEE", "FFF", "EEE_FFF", ["PG", "SG"], 5600, 29.0, 9.0),
        _make_player(7, "Golf Forward", "GGG", "HHH", "GGG_HHH", ["SF", "PF"], 5400, 28.0, 8.0),
        _make_player(8, "Hotel Core", "GGG", "HHH", "GGG_HHH", ["SG", "SF"], 4800, 25.0, 8.0),
        _make_player(9, "Indigo Pivot", "III", "JJJ", "III_JJJ", ["SG", "SF"], 4900, 24.0, 18.0),
    ]
    save_slate_projections(conn, slate_date, players)

    actual_ownership = {
        1: 19.0,
        2: 17.0,
        3: 15.0,
        4: 13.0,
        5: 21.0,
        6: 10.0,
        7: 9.0,
        8: 35.0,
        9: 5.0,
    }
    actual_fpts = {
        1: 39.0,
        2: 36.5,
        3: 34.5,
        4: 33.0,
        5: 41.0,
        6: 29.5,
        7: 28.5,
        8: 31.0,
        9: 18.0,
    }
    for player in players:
        conn.execute(
            """
            UPDATE dfs_slate_projections
            SET actual_fpts = ?, actual_ownership = ?, did_play = 1
            WHERE slate_date = ? AND player_id = ?
            """,
            (
                actual_fpts[player.player_id],
                actual_ownership[player.player_id],
                slate_date,
                player.player_id,
            ),
        )

    lineup = DFSLineup(
        players={
            "PG": players[0],
            "SG": players[1],
            "SF": players[2],
            "PF": players[3],
            "C": players[4],
            "G": players[5],
            "F": players[6],
            "UTIL": players[8],
        },
        model_key="standard_v1",
        model_label="Standard v1",
        generation_strategy="projection",
        regime_key="small_slate__medium_field",
        regime_label="Small Slate + Mid-Field / 20-max",
        regime_notes="test",
        regime_hint="auto",
        contest_field_size=1800,
    )
    save_slate_lineups(conn, slate_date, [lineup])
    conn.execute(
        """
        UPDATE dfs_slate_lineups
        SET total_actual_fpts = ?
        WHERE slate_date = ? AND lineup_num = 1
        """,
        (268.0, slate_date),
    )

    _insert_contest_entry(
        conn,
        contest_id="contest-attribution",
        slate_date=slate_date,
        username="top_user",
        rank=1,
        points=300.0,
        players=[
            players[0].name,
            players[1].name,
            players[2].name,
            players[3].name,
            players[4].name,
            players[5].name,
            players[6].name,
            players[7].name,
        ],
        total_salary=55400,
    )
    conn.execute(
        """
        INSERT INTO dfs_contest_meta (contest_id, slate_date, total_entries, unique_users, top_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("contest-attribution", slate_date, 1800, 1, 300.0),
    )
    conn.execute(
        """
        INSERT INTO dfs_supplement_runs (
            run_key, slate_date, source_name, source_filename, ownership_col,
            rows_total, rows_matched, rows_unmatched, match_rate, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "supp-run-1",
            slate_date,
            "RotoWire",
            "rotowire.csv",
            "ownership",
            1,
            1,
            0,
            100.0,
            "2026-03-11T23:30:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO dfs_supplement_player_deltas (
            run_key, slate_date, player_id, supplement_player, supplement_team,
            our_player, our_team, pos, salary, match_method, match_score,
            our_proj_fpts, supplement_proj_fpts, proj_delta, our_own_pct,
            supplement_own_pct, own_delta_pp, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "supp-run-1",
            slate_date,
            8,
            "Hotel Core",
            "GGG",
            "Hotel Core",
            "GGG",
            "SG/SF",
            4800,
            "player_id",
            1.0,
            25.0,
            27.0,
            2.0,
            8.0,
            12.0,
            4.0,
            "2026-03-11T23:30:00",
        ),
    )
    conn.commit()

    payload = build_tournament_postmortem(conn, slate_date, top_n=1)

    assert not payload["errors"]
    source_df = payload["source_vs_actual_ownership_df"]
    exclusion_df = payload["player_exclusion_attribution_df"]

    assert not source_df.empty
    hotel_source = source_df.loc[source_df["display_name"] == "Hotel Core"].iloc[0]
    assert hotel_source["supplement_own_pct"] == 12.0
    assert hotel_source["ownership_source_winner"] == "supplement"
    assert hotel_source["ownership_miss_context"] == "ours-undercalled, source-undercalled"

    assert not exclusion_df.empty
    hotel_exclusion = exclusion_df.loc[exclusion_df["Name"] == "Hotel Core"].iloc[0]
    assert bool(hotel_exclusion["is_missed_core"]) is True
    assert hotel_exclusion["primary_blocker"] == "core own low"
    assert hotel_exclusion["ownership_teacher_status"] == "both-undercall"
    assert hotel_exclusion["attribution_label"] == "ownership gate + teacher miss"
    assert payload["metrics"]["source_ownership_beats_our_count"] == 1
    assert payload["metrics"]["missed_core_ownership_blocker_count"] == 1
    assert payload["metrics"]["missed_core_teacher_signal_miss_count"] == 1
