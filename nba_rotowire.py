from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import pandas as pd
import requests


class RotoWireClientError(RuntimeError):
    """Raised for RotoWire optimizer request failures."""


SITE_ID_BY_NAME = {
    "draftkings": 1,
    "dk": 1,
}


@dataclass
class RotoWireClient:
    base_url: str = "https://www.rotowire.com"
    timeout_seconds: int = 20
    max_retries: int = 3
    retry_backoff_seconds: float = 0.75
    cookie_header: str | None = None
    user_agent: str = "Mozilla/5.0 (compatible; NBA_Daily/1.0)"

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "User-Agent": self.user_agent,
                "Referer": f"{self.base_url.rstrip('/')}/daily/nba/optimizer.php",
            }
        )
        if self.cookie_header:
            self.session.headers["Cookie"] = self.cookie_header

    def close(self) -> None:
        self.session.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    self._url(path),
                    params=dict(params or {}),
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 400:
                    detail = response.text[:500]
                    raise RotoWireClientError(
                        f"GET {path} failed ({response.status_code}): {detail}"
                    )
                return response.json()
            except (requests.RequestException, ValueError, RotoWireClientError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        raise RotoWireClientError(f"GET {path} failed after retries: {last_error}")

    def fetch_slate_catalog(self, site_id: int = 1) -> dict[str, Any]:
        payload = self.get_json("/daily/nba/api/slate-list.php", params={"siteID": site_id})
        if not isinstance(payload, Mapping):
            raise RotoWireClientError(
                f"Unexpected slate catalog payload type: {type(payload).__name__}"
            )
        slates = payload.get("slates")
        if not isinstance(slates, list):
            raise RotoWireClientError("Slate catalog payload missing list field `slates`.")
        return {"slates": slates, "games": payload.get("games")}

    def fetch_players(self, slate_id: int) -> list[dict[str, Any]]:
        payload = self.get_json("/daily/nba/api/players.php", params={"slateID": slate_id})
        if not isinstance(payload, list):
            raise RotoWireClientError(
                f"Unexpected players payload type: {type(payload).__name__}"
            )
        return [row for row in payload if isinstance(row, Mapping)]


def parse_site_id(site: str | None, site_id: int | None) -> int:
    if site_id is not None:
        return int(site_id)
    if not site:
        return 1
    normalized = site.strip().lower()
    if normalized in SITE_ID_BY_NAME:
        return SITE_ID_BY_NAME[normalized]
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported site value: {site}") from exc


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        if not cleaned:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    return None


def flatten_slates(catalog: Mapping[str, Any], site_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slate in catalog.get("slates", []):
        if not isinstance(slate, Mapping):
            continue
        rows.append(
            {
                "site_id": site_id,
                "slate_id": _to_int(slate.get("slateID")),
                "contest_type": slate.get("contestType"),
                "slate_name": slate.get("slateName"),
                "salary_cap": _to_int(slate.get("salaryCap")),
                "start_datetime": slate.get("startDate"),
                "end_datetime": slate.get("endDate"),
                "slate_date": slate.get("startDateOnly"),
                "start_time": slate.get("timeOnly"),
                "default_slate": bool(slate.get("defaultSlate", False)),
                "game_count": len(slate.get("games", []))
                if isinstance(slate.get("games"), list)
                else 0,
                "game_ids": ",".join(
                    str(x) for x in slate.get("games", []) if x is not None
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "slate_id",
                "contest_type",
                "slate_name",
                "salary_cap",
                "start_datetime",
                "end_datetime",
                "slate_date",
                "start_time",
                "default_slate",
                "game_count",
                "game_ids",
            ]
        )
    return out.sort_values(
        ["slate_date", "start_datetime", "contest_type", "slate_name"],
        kind="stable",
    ).reset_index(drop=True)


def normalize_players(
    raw_players: list[Mapping[str, Any]],
    slate_row: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    slate_meta = slate_row or {}
    rows: list[dict[str, Any]] = []
    for player in raw_players:
        team = player.get("team") if isinstance(player.get("team"), Mapping) else {}
        opponent = (
            player.get("opponent") if isinstance(player.get("opponent"), Mapping) else {}
        )
        odds = player.get("odds") if isinstance(player.get("odds"), Mapping) else {}
        stats = player.get("stats") if isinstance(player.get("stats"), Mapping) else {}
        avg_fpts = (
            stats.get("avgFpts") if isinstance(stats.get("avgFpts"), Mapping) else {}
        )
        advanced = (
            stats.get("advanced") if isinstance(stats.get("advanced"), Mapping) else {}
        )
        game = player.get("game") if isinstance(player.get("game"), Mapping) else {}
        lineup = player.get("lineup") if isinstance(player.get("lineup"), Mapping) else {}

        first_name = str(player.get("firstName") or "").strip()
        last_name = str(player.get("lastName") or "").strip()
        full_name = " ".join(part for part in [first_name, last_name] if part).strip()
        salary = _to_int(player.get("salary"))
        proj_fpts = _to_float(player.get("pts"))

        rows.append(
            {
                "site_id": slate_meta.get("site_id"),
                "slate_id": _to_int(player.get("slateID")) or slate_meta.get("slate_id"),
                "slate_date": slate_meta.get("slate_date"),
                "contest_type": slate_meta.get("contest_type"),
                "slate_name": slate_meta.get("slate_name"),
                "rw_id": _to_int(player.get("rwID")),
                "player_name": full_name or None,
                "first_name": first_name or None,
                "last_name": last_name or None,
                "roto_position": player.get("rotoPos"),
                "site_positions": "/".join(
                    str(pos) for pos in player.get("pos", []) if pos
                )
                if isinstance(player.get("pos"), list)
                else None,
                "injury_status": player.get("injuryStatus"),
                "is_home": bool(player.get("isHome", False)),
                "team_abbr": team.get("abbr"),
                "team_name": team.get("city"),
                "team_nickname": team.get("nickname"),
                "opp_abbr": opponent.get("team"),
                "game_datetime": game.get("dateTime"),
                "salary": salary,
                "proj_fantasy_points": proj_fpts,
                "proj_minutes": _to_float(player.get("minutes")),
                "proj_ownership": _to_float(player.get("rostership")),
                "lineup_slot": lineup.get("slot"),
                "lineup_confirmed": bool(lineup.get("isConfirmed", False)),
                "proj_value_per_1k": (
                    (proj_fpts / salary) * 1000.0
                    if proj_fpts is not None and salary not in (None, 0)
                    else None
                ),
                "moneyline": _to_float(odds.get("moneyline")),
                "over_under": _to_float(odds.get("overUnder")),
                "spread": _to_float(odds.get("spread")),
                "implied_points": _to_float(odds.get("impliedPts")),
                "implied_win_prob": _to_float(odds.get("impliedWinProb")),
                "season_games": _to_int(
                    (stats.get("season") or {}).get("games")
                    if isinstance(stats.get("season"), Mapping)
                    else None
                ),
                "season_minutes": _to_float(
                    (stats.get("season") or {}).get("minutes")
                    if isinstance(stats.get("season"), Mapping)
                    else None
                ),
                "avg_fpts_last3": _to_float(avg_fpts.get("last3")),
                "avg_fpts_last5": _to_float(avg_fpts.get("last5")),
                "avg_fpts_last7": _to_float(avg_fpts.get("last7")),
                "avg_fpts_last14": _to_float(avg_fpts.get("last14")),
                "avg_fpts_season": _to_float(avg_fpts.get("season")),
                "usage_rate": _to_float(advanced.get("usage")),
                "player_link": player.get("link"),
                "team_link": player.get("teamLink"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "slate_id",
                "slate_date",
                "contest_type",
                "slate_name",
                "rw_id",
                "player_name",
                "first_name",
                "last_name",
                "roto_position",
                "site_positions",
                "injury_status",
                "is_home",
                "team_abbr",
                "team_name",
                "team_nickname",
                "opp_abbr",
                "game_datetime",
                "salary",
                "proj_fantasy_points",
                "proj_minutes",
                "proj_ownership",
                "lineup_slot",
                "lineup_confirmed",
                "proj_value_per_1k",
                "moneyline",
                "over_under",
                "spread",
                "implied_points",
                "implied_win_prob",
                "season_games",
                "season_minutes",
                "avg_fpts_last3",
                "avg_fpts_last5",
                "avg_fpts_last7",
                "avg_fpts_last14",
                "avg_fpts_season",
                "usage_rate",
                "player_link",
                "team_link",
            ]
        )
    return out.sort_values(
        ["proj_fantasy_points", "salary", "player_name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
