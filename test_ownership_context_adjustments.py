import pytest

import dfs_optimizer as dfs


def _make_player(
    *,
    player_id: int,
    name: str,
    team: str,
    salary: int,
    proj_fpts: float,
    proj_ceiling: float,
    ownership_proj: float,
) -> dfs.DFSPlayer:
    player = dfs.DFSPlayer(
        player_id=player_id,
        name=name,
        team=team,
        opponent="OPP",
        game_id=f"{team}_OPP",
        positions=["PG"],
        salary=salary,
        proj_fpts=proj_fpts,
        proj_ceiling=proj_ceiling,
        ownership_proj=ownership_proj,
    )
    player.fpts_per_dollar = (proj_fpts / salary) * 1000.0 if salary > 0 else 0.0
    return player


def _build_player_pool() -> list[dfs.DFSPlayer]:
    return [
        _make_player(
            player_id=1,
            name="Cluster Guard",
            team="ORL",
            salary=6600,
            proj_fpts=32.0,
            proj_ceiling=45.0,
            ownership_proj=12.0,
        ),
        _make_player(
            player_id=2,
            name="Cluster Big",
            team="ORL",
            salary=5200,
            proj_fpts=26.0,
            proj_ceiling=38.0,
            ownership_proj=11.5,
        ),
        _make_player(
            player_id=3,
            name="Cluster Wing",
            team="ORL",
            salary=4800,
            proj_fpts=25.0,
            proj_ceiling=34.0,
            ownership_proj=8.0,
        ),
        _make_player(
            player_id=4,
            name="Stud Anchor",
            team="DEN",
            salary=12600,
            proj_fpts=69.0,
            proj_ceiling=91.0,
            ownership_proj=35.0,
        ),
        _make_player(
            player_id=5,
            name="Fake Chalk Star",
            team="LAC",
            salary=9800,
            proj_fpts=47.0,
            proj_ceiling=61.0,
            ownership_proj=31.0,
        ),
        _make_player(
            player_id=6,
            name="Fake Chalk Guard",
            team="MIA",
            salary=7900,
            proj_fpts=38.0,
            proj_ceiling=47.0,
            ownership_proj=27.0,
        ),
        _make_player(
            player_id=7,
            name="Midrange Stable",
            team="SAC",
            salary=6300,
            proj_fpts=31.0,
            proj_ceiling=44.0,
            ownership_proj=18.0,
        ),
        _make_player(
            player_id=8,
            name="Value Center",
            team="LAC",
            salary=3600,
            proj_fpts=22.5,
            proj_ceiling=31.0,
            ownership_proj=6.5,
        ),
    ]


def test_slate_context_ownership_adjustments_boost_value_clusters_and_trim_fake_chalk() -> None:
    players = _build_player_pool()
    base_total = sum(float(p.ownership_proj or 0.0) for p in players)

    stats = dfs.apply_slate_context_ownership_adjustments(players, today_num_games=6)
    own_map = {p.name: float(p.ownership_proj or 0.0) for p in players}

    assert stats["adjusted_players"] >= 4
    assert stats["boosted_players"] >= 2
    assert stats["trimmed_players"] >= 2
    assert own_map["Cluster Guard"] > 12.0
    assert own_map["Cluster Big"] > 11.5
    assert own_map["Cluster Wing"] > 8.0
    assert own_map["Fake Chalk Star"] < 31.0
    assert own_map["Fake Chalk Guard"] < 27.0
    assert sum(own_map.values()) == pytest.approx(base_total, abs=0.25)


def test_slate_context_ownership_adjustments_are_idempotent_on_repeat_calls() -> None:
    players = _build_player_pool()

    dfs.apply_slate_context_ownership_adjustments(players, today_num_games=6)
    first_pass = {p.name: float(p.ownership_proj or 0.0) for p in players}

    dfs.apply_slate_context_ownership_adjustments(players, today_num_games=6)
    second_pass = {p.name: float(p.ownership_proj or 0.0) for p in players}

    assert second_pass == pytest.approx(first_pass, abs=0.01)
