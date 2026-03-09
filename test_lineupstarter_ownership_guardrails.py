import dfs_optimizer as dfs


def _make_player(
    *,
    name: str,
    proj_fpts: float,
    proj_ceiling: float,
    salary: int,
    recent_minutes_avg: float,
    fpts_per_dollar: float,
) -> dfs.DFSPlayer:
    return dfs.DFSPlayer(
        player_id=1,
        name=name,
        team="AAA",
        opponent="BBB",
        game_id="AAA_BBB",
        positions=["PG"],
        salary=salary,
        proj_fpts=proj_fpts,
        proj_ceiling=proj_ceiling,
        recent_minutes_avg=recent_minutes_avg,
        fpts_per_dollar=fpts_per_dollar,
    )


def test_lineupstarter_guardrails_promote_likely_chalk_core():
    player = _make_player(
        name="RJ Barrett",
        proj_fpts=33.0,
        proj_ceiling=47.0,
        salary=7600,
        recent_minutes_avg=35.0,
        fpts_per_dollar=4.34,
    )

    result = dfs.get_guardrailed_supplement_ownership_projection(
        player=player,
        base_ownership=11.0,
        supplement_ownership=50.4,
        requested_weight=0.35,
        source_name="LineupStarter CSV",
        delta_cap=0.0,
    )

    assert result["lineupstarter_guardrails"] is True
    assert result["guardrail_tag"] == "ls_core_chalk"
    assert result["weight_applied"] >= 0.60
    assert result["projected_ownership"] > 25.0


def test_lineupstarter_guardrails_block_fake_chalk_jump():
    player = _make_player(
        name="Bench Punt",
        proj_fpts=18.0,
        proj_ceiling=24.0,
        salary=3700,
        recent_minutes_avg=21.0,
        fpts_per_dollar=4.86,
    )

    result = dfs.get_guardrailed_supplement_ownership_projection(
        player=player,
        base_ownership=3.0,
        supplement_ownership=29.0,
        requested_weight=0.45,
        source_name="LineupStarter CSV",
        delta_cap=0.0,
    )

    assert result["lineupstarter_guardrails"] is True
    assert "anti_fake_chalk" in str(result["guardrail_tag"])
    assert result["weight_applied"] <= 0.35
    assert result["projected_ownership"] < 6.0


def test_non_lineupstarter_source_keeps_requested_weight():
    player = _make_player(
        name="RotoWire Only",
        proj_fpts=28.0,
        proj_ceiling=40.0,
        salary=6900,
        recent_minutes_avg=33.0,
        fpts_per_dollar=4.06,
    )

    result = dfs.get_guardrailed_supplement_ownership_projection(
        player=player,
        base_ownership=15.0,
        supplement_ownership=28.0,
        requested_weight=1.0,
        source_name="RotoWire NBA Optimizer",
        delta_cap=0.0,
    )

    assert result["lineupstarter_guardrails"] is False
    assert result["guardrail_tag"] == "standard"
    assert result["weight_applied"] == 1.0
    assert result["projected_ownership"] == 28.0
