import sqlite3

import pandas as pd

import dfs_tracking as dt


def _comparison_df(player_id: int, player_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Our Player ID": player_id,
                "Supplement Player": player_name,
                "Supplement Team": "AAA",
                "Our Player": player_name,
                "Our Team": "AAA",
                "Pos": "PG",
                "Salary": 5000,
                "Match Method": "exact_name",
                "Match Score": 1.0,
                "Our Proj FPTS": 25.0,
                "Supplement Proj FPTS": 27.0,
                "Proj Delta": 2.0,
                "Our Own %": 10.0,
                "Supplement Own %": 12.0,
                "Own Delta (pp)": 2.0,
            }
        ]
    )


def test_get_recent_supplement_runs_filters_by_slate_date_and_source():
    conn = sqlite3.connect(":memory:")
    dt.create_dfs_tracking_tables(conn)

    unmatched_df = pd.DataFrame()
    dt.save_supplement_snapshot(
        conn,
        slate_date="2026-03-08",
        run_key="rw_today",
        source_name="RotoWire NBA Optimizer",
        source_filename="rw_today.csv",
        projection_col="proj",
        ownership_col="own",
        comparison_df=_comparison_df(1, "Today RW"),
        unmatched_df=unmatched_df,
        rows_total=1,
    )
    dt.save_supplement_snapshot(
        conn,
        slate_date="2026-03-08",
        run_key="ls_today",
        source_name="LineupStarter CSV",
        source_filename="ls_today.csv",
        projection_col="proj",
        ownership_col="own",
        comparison_df=_comparison_df(2, "Today LS"),
        unmatched_df=unmatched_df,
        rows_total=1,
    )
    dt.save_supplement_snapshot(
        conn,
        slate_date="2026-03-07",
        run_key="rw_yesterday",
        source_name="RotoWire NBA Optimizer",
        source_filename="rw_yesterday.csv",
        projection_col="proj",
        ownership_col="own",
        comparison_df=_comparison_df(3, "Yesterday RW"),
        unmatched_df=unmatched_df,
        rows_total=1,
    )

    current_slate_df = dt.get_recent_supplement_runs(
        conn,
        limit=10,
        slate_date="2026-03-08",
    )
    assert set(current_slate_df["run_key"]) == {"rw_today", "ls_today"}
    assert set(current_slate_df["slate_date"]) == {"2026-03-08"}

    rotowire_only_df = dt.get_recent_supplement_runs(
        conn,
        limit=10,
        slate_date="2026-03-08",
        source_name_filter="rotowire",
    )
    assert list(rotowire_only_df["run_key"]) == ["rw_today"]

    full_archive_df = dt.get_recent_supplement_runs(conn, limit=10)
    assert set(full_archive_df["run_key"]) == {"rw_today", "ls_today", "rw_yesterday"}
