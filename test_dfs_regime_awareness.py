import sqlite3

from dfs_optimizer import DFSLineup, DFSPlayer, resolve_lineup_generation_regime
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
