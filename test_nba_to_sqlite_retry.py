import pandas as pd
import pytest
from requests.exceptions import ReadTimeout

import nba_to_sqlite as nts


class _DummyEndpoint:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_data_frames(self):
        return [self._frame]


def test_fetch_nba_stats_frame_retries_timeout_then_succeeds(monkeypatch):
    attempts = {"count": 0}
    monkeypatch.setattr(nts.time, "sleep", lambda _: None)

    def factory(timeout: int):
        attempts["count"] += 1
        assert timeout == 45
        if attempts["count"] < 3:
            raise ReadTimeout("Read timed out.")
        return _DummyEndpoint(pd.DataFrame([{"value": 7}]))

    frame = nts.fetch_nba_stats_frame(
        factory,
        label="Test endpoint",
        timeout=45,
        max_retries=3,
        retry_delay=0.01,
    )

    assert attempts["count"] == 3
    assert frame.to_dict(orient="records") == [{"value": 7}]


def test_fetch_nba_stats_frame_raises_runtime_error_after_retries(monkeypatch):
    attempts = {"count": 0}
    monkeypatch.setattr(nts.time, "sleep", lambda _: None)

    def factory(timeout: int):
        attempts["count"] += 1
        raise ReadTimeout("Read timed out.")

    with pytest.raises(RuntimeError, match="failed after 2 timeout attempts"):
        nts.fetch_nba_stats_frame(
            factory,
            label="Test endpoint",
            timeout=30,
            max_retries=2,
            retry_delay=0.01,
        )

    assert attempts["count"] == 2


def test_fetch_nba_stats_frame_does_not_retry_non_timeout_error():
    attempts = {"count": 0}

    def factory(timeout: int):
        attempts["count"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        nts.fetch_nba_stats_frame(
            factory,
            label="Test endpoint",
            timeout=30,
            max_retries=3,
            retry_delay=0.01,
        )

    assert attempts["count"] == 1
