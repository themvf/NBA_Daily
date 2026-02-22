"""
OpenAI-powered DFS projection review agent.

This module reviews a DFS player pool using:
- current slate projections/ownership
- recent player game-log form from SQLite
- historical ownership tracking from SQLite
- optional Vegas edge signals

The agent returns structured recommendations that can be applied to
projection deltas and exposure targets in the Streamlit app.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


DEFAULT_MODEL = "gpt-4.1-mini"
MAX_ABS_DELTA_FPTS = 6.0
MAX_TARGET_EXPOSURE_PCT = 95.0
VALID_ACTIONS = {"core", "boost", "watch", "fade"}


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        [table_name],
    ).fetchone()
    return row is not None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if pd.isna(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _calculate_dk_fpts(df: pd.DataFrame) -> pd.Series:
    points = pd.to_numeric(df.get("points", 0), errors="coerce").fillna(0.0)
    rebounds = pd.to_numeric(df.get("rebounds", 0), errors="coerce").fillna(0.0)
    assists = pd.to_numeric(df.get("assists", 0), errors="coerce").fillna(0.0)
    steals = pd.to_numeric(df.get("steals", 0), errors="coerce").fillna(0.0)
    blocks = pd.to_numeric(df.get("blocks", 0), errors="coerce").fillna(0.0)
    turnovers = pd.to_numeric(df.get("turnovers", 0), errors="coerce").fillna(0.0)
    fg3m = pd.to_numeric(df.get("fg3m", 0), errors="coerce").fillna(0.0)

    return (
        points
        + rebounds * 1.25
        + assists * 1.5
        + steals * 2.0
        + blocks * 2.0
        - turnovers * 0.5
        + fg3m * 0.5
    )


def _select_candidate_players(players: Iterable[Any], max_players: int = 80) -> List[Any]:
    candidates = [
        p
        for p in players
        if not getattr(p, "is_injured", False) and not getattr(p, "is_excluded", False)
    ]
    if len(candidates) <= max_players:
        return candidates

    by_proj = sorted(candidates, key=lambda p: _to_float(getattr(p, "proj_fpts", 0.0)), reverse=True)
    by_value = sorted(
        candidates,
        key=lambda p: _to_float(getattr(p, "fpts_per_dollar", 0.0)),
        reverse=True,
    )
    by_contrarian_ceiling = sorted(
        candidates,
        key=lambda p: (
            (_to_float(getattr(p, "proj_ceiling", 0.0)) - _to_float(getattr(p, "proj_fpts", 0.0)))
            * max(1.0, 25.0 - _to_float(getattr(p, "ownership_proj", 0.0)))
        ),
        reverse=True,
    )

    selected: Dict[int, Any] = {}
    for group in (by_proj[:50], by_value[:35], by_contrarian_ceiling[:35]):
        for player in group:
            pid = int(getattr(player, "player_id", 0) or 0)
            if pid and pid not in selected:
                selected[pid] = player
            if len(selected) >= max_players:
                break
        if len(selected) >= max_players:
            break

    return list(selected.values())


def _load_recent_form(
    conn: sqlite3.Connection,
    player_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    if not player_ids or not _table_exists(conn, "player_game_logs"):
        return {}

    placeholders = ",".join("?" for _ in player_ids)
    query = f"""
        SELECT
            player_id, game_date, points, rebounds, assists,
            steals, blocks, turnovers, fg3m, minutes
        FROM player_game_logs
        WHERE player_id IN ({placeholders})
    """
    logs = pd.read_sql_query(query, conn, params=player_ids)
    if logs.empty:
        return {}

    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    logs = logs.sort_values(["player_id", "game_date"], ascending=[True, False])
    logs["dk_fpts"] = _calculate_dk_fpts(logs)
    logs["minutes"] = pd.to_numeric(logs.get("minutes", 0), errors="coerce").fillna(0.0)
    logs["rn"] = logs.groupby("player_id").cumcount() + 1

    recent10 = logs[logs["rn"] <= 10].copy()
    recent3 = logs[logs["rn"] <= 3].copy()
    if recent10.empty:
        return {}

    agg10 = (
        recent10.groupby("player_id")
        .agg(
            avg_fpts_10=("dk_fpts", "mean"),
            std_fpts_10=("dk_fpts", "std"),
            avg_minutes_10=("minutes", "mean"),
            games_10=("dk_fpts", "count"),
        )
        .fillna(0.0)
    )
    agg3 = (
        recent3.groupby("player_id")
        .agg(
            avg_fpts_3=("dk_fpts", "mean"),
            avg_minutes_3=("minutes", "mean"),
        )
        .fillna(0.0)
    )

    merged = agg10.join(agg3, how="left").fillna(0.0)
    merged["fpts_trend_3v10"] = merged["avg_fpts_3"] - merged["avg_fpts_10"]
    merged["minutes_trend_3v10"] = merged["avg_minutes_3"] - merged["avg_minutes_10"]

    out: Dict[int, Dict[str, float]] = {}
    for pid, row in merged.iterrows():
        out[int(pid)] = {
            "avg_fpts_10": float(row["avg_fpts_10"]),
            "std_fpts_10": float(row["std_fpts_10"]),
            "avg_minutes_10": float(row["avg_minutes_10"]),
            "games_10": float(row["games_10"]),
            "avg_fpts_3": float(row["avg_fpts_3"]),
            "avg_minutes_3": float(row["avg_minutes_3"]),
            "fpts_trend_3v10": float(row["fpts_trend_3v10"]),
            "minutes_trend_3v10": float(row["minutes_trend_3v10"]),
        }
    return out


def _load_ownership_history(
    conn: sqlite3.Connection,
    player_ids: List[int],
    lookback_days: int = 120,
) -> Dict[int, Dict[str, float]]:
    if not player_ids or not _table_exists(conn, "dfs_slate_projections"):
        return {}

    placeholders = ",".join("?" for _ in player_ids)
    query = f"""
        SELECT
            player_id,
            AVG(CASE WHEN ownership_proj > 0 THEN ownership_proj END) AS avg_proj_own,
            AVG(CASE WHEN actual_ownership > 0 THEN actual_ownership END) AS avg_actual_own,
            COUNT(*) AS sample_size
        FROM dfs_slate_projections
        WHERE player_id IN ({placeholders})
          AND slate_date >= date('now', ?)
        GROUP BY player_id
    """
    params: List[Any] = list(player_ids) + [f"-{max(7, int(lookback_days))} days"]
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return {}

    out: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        pid = int(row["player_id"])
        avg_proj = _to_float(row["avg_proj_own"], 0.0)
        avg_actual = _to_float(row["avg_actual_own"], 0.0)
        out[pid] = {
            "avg_proj_own": avg_proj,
            "avg_actual_own": avg_actual,
            "ownership_bias_pts": avg_actual - avg_proj if avg_proj > 0 and avg_actual > 0 else 0.0,
            "sample_size": _to_float(row["sample_size"], 0.0),
        }
    return out


def build_player_context(
    conn: sqlite3.Connection,
    players: Iterable[Any],
    vegas_signals: Optional[Mapping[int, Mapping[str, Any]]] = None,
    lookback_days: int = 120,
    max_players: int = 80,
) -> List[Dict[str, Any]]:
    candidate_players = _select_candidate_players(players, max_players=max_players)
    player_ids = [int(getattr(p, "player_id", 0) or 0) for p in candidate_players]
    player_ids = [pid for pid in player_ids if pid > 0]

    recent_form = _load_recent_form(conn, player_ids)
    ownership_hist = _load_ownership_history(conn, player_ids, lookback_days=lookback_days)
    vegas_signals = vegas_signals or {}

    context_rows: List[Dict[str, Any]] = []
    for p in candidate_players:
        pid = int(getattr(p, "player_id", 0) or 0)
        if pid <= 0:
            continue

        form = recent_form.get(pid, {})
        hist = ownership_hist.get(pid, {})
        vegas = vegas_signals.get(pid, {})
        positions = getattr(p, "positions", []) or []

        context_rows.append(
            {
                "player_id": pid,
                "name": str(getattr(p, "name", "")),
                "team": str(getattr(p, "team", "")),
                "opponent": str(getattr(p, "opponent", "")),
                "positions": "/".join(str(x) for x in positions) if positions else "",
                "salary": int(_to_float(getattr(p, "salary", 0), 0)),
                "proj_fpts": round(_to_float(getattr(p, "proj_fpts", 0.0)), 2),
                "proj_floor": round(_to_float(getattr(p, "proj_floor", 0.0)), 2),
                "proj_ceiling": round(_to_float(getattr(p, "proj_ceiling", 0.0)), 2),
                "fpts_per_dollar": round(_to_float(getattr(p, "fpts_per_dollar", 0.0)), 3),
                "ownership_proj": round(_to_float(getattr(p, "ownership_proj", 0.0)), 2),
                "p_top3": round(_to_float(getattr(p, "p_top3", 0.0)), 4),
                "stack_score": round(_to_float(getattr(p, "stack_score", 0.0)), 3),
                "days_rest": getattr(p, "days_rest", None),
                "minutes_validated": bool(getattr(p, "minutes_validated", True)),
                "role_tier": str(getattr(p, "role_tier", "")),
                "recent_avg_fpts_10": round(_to_float(form.get("avg_fpts_10", 0.0)), 2),
                "recent_std_fpts_10": round(_to_float(form.get("std_fpts_10", 0.0)), 2),
                "recent_avg_minutes_10": round(_to_float(form.get("avg_minutes_10", 0.0)), 2),
                "fpts_trend_3v10": round(_to_float(form.get("fpts_trend_3v10", 0.0)), 2),
                "minutes_trend_3v10": round(_to_float(form.get("minutes_trend_3v10", 0.0)), 2),
                "ownership_avg_proj_lookback": round(_to_float(hist.get("avg_proj_own", 0.0)), 2),
                "ownership_avg_actual_lookback": round(_to_float(hist.get("avg_actual_own", 0.0)), 2),
                "ownership_bias_pts": round(_to_float(hist.get("ownership_bias_pts", 0.0)), 2),
                "ownership_samples": int(_to_float(hist.get("sample_size", 0.0), 0)),
                "vegas_edge_pct": round(_to_float(vegas.get("edge_pct", 0.0)), 2)
                if vegas
                else None,
                "vegas_signal": vegas.get("signal") if vegas else None,
            }
        )

    context_rows.sort(key=lambda r: r["proj_fpts"], reverse=True)
    return context_rows


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _safe_json_loads(raw_text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(raw_text)
    return json.loads(cleaned)


def _coerce_recommendations(
    recommendations: List[Dict[str, Any]],
    player_lookup: Mapping[int, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    name_lookup = {
        str(getattr(player, "name", "")).strip().lower(): pid
        for pid, player in player_lookup.items()
    }

    for raw in recommendations:
        if not isinstance(raw, dict):
            continue

        pid = int(_to_float(raw.get("player_id", 0), 0))
        if pid not in player_lookup:
            fallback_name = str(raw.get("name", "")).strip().lower()
            if fallback_name in name_lookup:
                pid = name_lookup[fallback_name]
            else:
                continue

        action = str(raw.get("action", "watch")).strip().lower()
        if action not in VALID_ACTIONS:
            action = "watch"

        delta = _to_float(raw.get("projection_delta_fpts", 0.0), 0.0)
        delta = max(-MAX_ABS_DELTA_FPTS, min(MAX_ABS_DELTA_FPTS, delta))

        if action in {"core", "boost"} and delta < 0:
            delta = abs(delta)
        if action == "fade" and delta > 0:
            delta = -delta

        confidence = _to_float(raw.get("confidence", 0.6), 0.6)
        confidence = max(0.0, min(1.0, confidence))

        target_exposure = raw.get("target_exposure_pct")
        if target_exposure is None or target_exposure == "":
            target_exposure_val: Optional[float] = None
        else:
            target_exposure_val = _to_float(target_exposure, 0.0)
            target_exposure_val = max(0.0, min(MAX_TARGET_EXPOSURE_PCT, target_exposure_val))
            if action == "fade":
                target_exposure_val = min(target_exposure_val, 15.0)

        rationale = str(raw.get("rationale", "")).strip()
        if not rationale:
            rationale = "No rationale provided by model."
        rationale = rationale[:220]

        player = player_lookup[pid]
        out.append(
            {
                "player_id": pid,
                "name": str(getattr(player, "name", "")),
                "team": str(getattr(player, "team", "")),
                "action": action,
                "confidence": round(confidence, 2),
                "projection_delta_fpts": round(delta, 2),
                "target_exposure_pct": round(target_exposure_val, 1)
                if target_exposure_val is not None
                else None,
                "rationale": rationale,
            }
        )

    deduped: Dict[int, Dict[str, Any]] = {}
    for rec in sorted(
        out,
        key=lambda x: (abs(x["projection_delta_fpts"]) * x["confidence"]),
        reverse=True,
    ):
        pid = rec["player_id"]
        if pid not in deduped:
            deduped[pid] = rec

    return list(deduped.values())


def run_projection_review(
    conn: sqlite3.Connection,
    players: Iterable[Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    vegas_signals: Optional[Mapping[int, Mapping[str, Any]]] = None,
    projection_date: Optional[str] = None,
    lookback_days: int = 120,
    max_players: int = 80,
) -> Dict[str, Any]:
    """
    Run an OpenAI review for DFS projection opportunities and risks.

    Returns:
        Dict with keys:
        - status: "ok" | "error"
        - summary: short model summary
        - lineup_strategy: list[str]
        - recommendations: list[dict]
        - model: model name used
        - candidate_count: number of players sent to model
        - timestamp_utc: ISO timestamp
    """
    if OpenAI is None:
        return {
            "status": "error",
            "error": "OpenAI package not installed. Add `openai` to requirements.",
            "model": model,
        }
    if not api_key:
        return {
            "status": "error",
            "error": "OPENAI_API_KEY is missing.",
            "model": model,
        }

    player_list = list(players)
    player_lookup = {
        int(getattr(p, "player_id", 0) or 0): p
        for p in player_list
        if int(getattr(p, "player_id", 0) or 0) > 0
    }
    context_rows = build_player_context(
        conn=conn,
        players=player_list,
        vegas_signals=vegas_signals,
        lookback_days=lookback_days,
        max_players=max_players,
    )
    if len(context_rows) < 10:
        return {
            "status": "error",
            "error": "Not enough valid players to review. Generate projections first.",
            "model": model,
        }

    system_prompt = (
        "You are an NBA DFS analyst focused on GPP lineup leverage. "
        "Review provided player context and identify players that are potentially "
        "overlooked or overvalued relative to projection, recent form, and ownership. "
        "Use only the provided data. Return strict JSON only."
    )

    user_payload = {
        "slate_date": projection_date,
        "task": (
            "Return JSON with keys: summary, lineup_strategy, recommendations. "
            "recommendations must be an array with 6-16 entries. "
            "Each entry: player_id (int), name (string), action (core|boost|watch|fade), "
            "confidence (0-1), projection_delta_fpts (-6 to 6), "
            "target_exposure_pct (0-95 or null), rationale (<=180 chars). "
            "For fades use non-positive deltas. For core/boost use positive deltas. "
            "Prioritize low-owned upside where supported by data."
        ),
        "players": context_rows,
    }

    client = OpenAI(api_key=api_key)
    try:
        request_kwargs = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
            ],
        }
        # gpt-5 currently supports only default temperature behavior.
        if not str(model).lower().startswith("gpt-5"):
            request_kwargs["temperature"] = 0.15
        try:
            # gpt-5 family expects max_completion_tokens.
            completion = client.chat.completions.create(
                **request_kwargs,
                max_completion_tokens=1600,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if (
                "max_completion_tokens" in msg
                and ("unsupported" in msg or "unknown parameter" in msg)
            ):
                # Backward-compatibility fallback for older model behavior.
                completion = client.chat.completions.create(
                    **request_kwargs,
                    max_tokens=1600,
                )
            else:
                raise

        content = completion.choices[0].message.content or "{}"
        parsed = _safe_json_loads(content)
    except Exception as exc:
        return {
            "status": "error",
            "error": f"OpenAI request failed: {exc}",
            "model": model,
            "candidate_count": len(context_rows),
        }

    raw_recs = parsed.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    recommendations = _coerce_recommendations(raw_recs, player_lookup)
    if recommendations:
        recommendations = sorted(
            recommendations,
            key=lambda x: (abs(x["projection_delta_fpts"]), x["confidence"]),
            reverse=True,
        )

    lineup_strategy = parsed.get("lineup_strategy", [])
    if not isinstance(lineup_strategy, list):
        lineup_strategy = []
    lineup_strategy = [str(x)[:200] for x in lineup_strategy[:6]]

    summary = str(parsed.get("summary", "")).strip()[:500]
    if not summary and recommendations:
        summary = "AI review complete."

    return {
        "status": "ok",
        "summary": summary,
        "lineup_strategy": lineup_strategy,
        "recommendations": recommendations,
        "model": model,
        "candidate_count": len(context_rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
