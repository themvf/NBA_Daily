#!/usr/bin/env python3
"""
Train + evaluate a model that predicts which player on a team will make the most
three-pointers in a given game.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict per-team three-point leaders for each game."
    )
    parser.add_argument(
        "--db-path",
        default="nba_stats.db",
        help="Path to the SQLite database built via nba_to_sqlite.py.",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help="Season type to use (default: Regular Season).",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        default=["2024-25"],
        help="Seasons used for training (space separated).",
    )
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        default=["2025-26"],
        help="Seasons used for evaluation.",
    )
    parser.add_argument(
        "--min-history-games",
        type=int,
        default=3,
        help="Minimum prior games required before we trust a player sample.",
    )
    parser.add_argument(
        "--output-predictions",
        default=None,
        help="Optional CSV file to store per-game predictions.",
    )
    return parser.parse_args()


def minutes_to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value)
    if ":" in value_str:
        parts = value_str.split(":")
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + seconds / 60.0
        except ValueError:
            return None
    try:
        return float(value_str)
    except ValueError:
        return None


def load_dataframe(
    conn: sqlite3.Connection,
    query: str,
    params: Sequence[object],
) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def fetch_player_logs(
    conn: sqlite3.Connection,
    seasons: Sequence[str],
    season_type: str,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
        SELECT
            season,
            season_type,
            game_id,
            team_id,
            player_id,
            player_name,
            fg3m,
            fg3a,
            minutes,
            points,
            game_date
        FROM player_game_logs
        WHERE season_type = ?
          AND season IN ({placeholders})
    """
    params = [season_type, *seasons]
    df = load_dataframe(conn, query, params)
    if df.empty:
        raise ValueError("player_game_logs query returned no rows.")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["minutes_float"] = df["minutes"].apply(minutes_to_float)
    numeric_cols = [
        "team_id",
        "player_id",
        "fg3m",
        "fg3a",
        "points",
        "minutes_float",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team_id"] = df["team_id"].astype("Int64")
    df["player_id"] = df["player_id"].astype("Int64")
    return df


def fetch_team_logs(
    conn: sqlite3.Connection,
    seasons: Sequence[str],
    season_type: str,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
        SELECT
            season,
            season_type,
            game_id,
            team_id,
            opp_team_id,
            fg3m,
            fg3a,
            opp_fg3m,
            opp_fg3a,
            pts,
            opp_pts,
            game_date
        FROM team_game_logs
        WHERE season_type = ?
          AND season IN ({placeholders})
    """
    params = [season_type, *seasons]
    df = load_dataframe(conn, query, params)
    if df.empty:
        raise ValueError("team_game_logs query returned no rows.")
    df["game_date"] = pd.to_datetime(df["game_date"])
    numeric_cols = [
        "team_id",
        "opp_team_id",
        "fg3m",
        "fg3a",
        "opp_fg3m",
        "opp_fg3a",
        "pts",
        "opp_pts",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team_id"] = df["team_id"].astype("Int64")
    df["opp_team_id"] = df["opp_team_id"].astype("Int64")
    return df


def fetch_player_totals(
    conn: sqlite3.Connection,
    seasons: Sequence[str],
    season_type: str,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
        SELECT
            season,
            season_type,
            player_id,
            games_played,
            fg3m,
            fg3a,
            minutes,
            points
        FROM player_season_totals
        WHERE season_type = ?
          AND season IN ({placeholders})
    """
    params = [season_type, *seasons]
    df = load_dataframe(conn, query, params)
    if df.empty:
        raise ValueError("player_season_totals query returned no rows.")
    numeric_cols = ["games_played", "fg3m", "fg3a", "minutes", "points"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["season_fg3m_per_game"] = df["fg3m"] / df["games_played"].replace(0, np.nan)
    df["season_fg3a_per_game"] = df["fg3a"] / df["games_played"].replace(0, np.nan)
    df["season_minutes_per_game"] = df["minutes"] / df["games_played"].replace(
        0, np.nan
    )
    df["season_points_per_game"] = df["points"] / df["games_played"].replace(
        0, np.nan
    )
    return df


def add_player_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "game_date"])
    group = df.groupby("player_id", sort=False)
    df["games_played_prior"] = group.cumcount()
    for col in ["fg3m", "fg3a", "minutes_float", "points"]:
        df[f"{col}_lag1"] = group[col].shift(1)
        df[f"{col}_roll3"] = (
            group[col]
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
        )
    return df


def add_team_history_features(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.sort_values(["team_id", "game_date"])
    group = team_df.groupby("team_id", sort=False)
    for col in ["fg3m", "fg3a", "pts"]:
        team_df[f"team_{col}_roll3"] = (
            group[col].shift(1).rolling(3, min_periods=1).mean()
        )
    for col, suffix in [("opp_fg3m", "fg3m"), ("opp_pts", "pts")]:
        team_df[f"team_allowed_{suffix}_roll3"] = (
            group[col].shift(1).rolling(3, min_periods=1).mean()
        )
    return team_df


def prepare_dataset(
    conn: sqlite3.Connection,
    seasons: Sequence[str],
    season_type: str,
    min_history_games: int,
) -> pd.DataFrame:
    player_logs = fetch_player_logs(conn, seasons, season_type)
    team_logs = fetch_team_logs(conn, seasons, season_type)
    season_totals = fetch_player_totals(conn, seasons, season_type)

    player_logs = add_player_history_features(player_logs)
    team_logs = add_team_history_features(team_logs)

    merged = player_logs.merge(
        season_totals[
            [
                "season",
                "player_id",
                "season_fg3m_per_game",
                "season_fg3a_per_game",
                "season_minutes_per_game",
                "season_points_per_game",
            ]
        ],
        on=["season", "player_id"],
        how="left",
    )

    team_features = team_logs[
        [
            "game_id",
            "team_id",
            "opp_team_id",
            "team_fg3m_roll3",
            "team_fg3a_roll3",
            "team_pts_roll3",
            "team_allowed_fg3m_roll3",
            "team_allowed_pts_roll3",
        ]
    ]
    merged = merged.merge(
        team_features,
        on=["game_id", "team_id"],
        how="left",
    )

    opponent_def = team_features[
        [
            "game_id",
            "team_id",
            "team_allowed_fg3m_roll3",
            "team_allowed_pts_roll3",
        ]
    ].rename(
        columns={
            "team_id": "opp_team_id",
            "team_allowed_fg3m_roll3": "opp_allowed_fg3m_roll3",
            "team_allowed_pts_roll3": "opp_allowed_pts_roll3",
        }
    )
    merged = merged.merge(
        opponent_def,
        on=["game_id", "opp_team_id"],
        how="left",
    )

    merged["label_top_3pm"] = (
        merged.groupby(["game_id", "team_id"])["fg3m"].transform("max")
        == merged["fg3m"]
    ).astype(int)

    merged = merged[merged["games_played_prior"] >= min_history_games].copy()
    return merged


def train_and_evaluate(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    output_predictions: Path | None = None,
) -> None:
    feature_cols = [
        "fg3m_lag1",
        "fg3m_roll3",
        "fg3a_lag1",
        "fg3a_roll3",
        "minutes_float_lag1",
        "minutes_float_roll3",
        "points_lag1",
        "points_roll3",
        "season_fg3m_per_game",
        "season_fg3a_per_game",
        "season_minutes_per_game",
        "season_points_per_game",
        "team_fg3m_roll3",
        "team_fg3a_roll3",
        "team_pts_roll3",
        "team_allowed_fg3m_roll3",
        "team_allowed_pts_roll3",
        "opp_allowed_fg3m_roll3",
        "opp_allowed_pts_roll3",
    ]

    feature_df = df[feature_cols]
    feature_df = feature_df.fillna(feature_df.mean(numeric_only=True))

    X_train = feature_df[train_mask]
    y_train = df.loc[train_mask, "label_top_3pm"]

    X_test = feature_df[test_mask]
    y_test = df.loc[test_mask, "label_top_3pm"]

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing splits are empty; adjust seasons.")

    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_probs)
    print(f"Test ROC-AUC: {roc_auc:.3f}")

    test_df = df[test_mask].copy()
    test_df["pred_prob"] = test_probs

    grouped = (
        test_df.groupby(["game_id", "team_id"], sort=False)
        .apply(lambda g: g.loc[g["pred_prob"].idxmax()])
        .reset_index(drop=True)
    )
    grouped["correct"] = grouped["label_top_3pm"]
    accuracy = grouped["correct"].mean()
    print(
        f"Per-team game accuracy (picked correct leader): "
        f"{accuracy:.3%} over {len(grouped)} games."
    )

    display_cols = [
        "game_id",
        "team_id",
        "player_id",
        "player_name",
        "season",
        "fg3m",
        "pred_prob",
        "correct",
    ]
    print("Sample predictions:")
    print(grouped[display_cols].head(10))

    if output_predictions:
        grouped.to_csv(output_predictions, index=False)
        print(f"Wrote predictions to {output_predictions}")


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    conn = sqlite3.connect(db_path)
    try:
        seasons = sorted(set(args.train_seasons + args.test_seasons))
        dataset = prepare_dataset(
            conn,
            seasons,
            args.season_type,
            args.min_history_games,
        )
    finally:
        conn.close()

    train_mask = dataset["season"].isin(args.train_seasons)
    test_mask = dataset["season"].isin(args.test_seasons)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(
            "Requested season split missing; applying chronological 80/20 split."
        )
        cutoff = dataset["game_date"].quantile(0.8)
        train_mask = dataset["game_date"] < cutoff
        test_mask = dataset["game_date"] >= cutoff
    output_path = Path(args.output_predictions) if args.output_predictions else None
    train_and_evaluate(dataset, train_mask, test_mask, output_path)


if __name__ == "__main__":
    main()
