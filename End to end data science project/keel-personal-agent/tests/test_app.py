from __future__ import annotations

from pathlib import Path

import pytest

from keel.clock import SystemClock
from keel.db import connect
from keel.demo import seed

testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


def test_app_runs_without_a_key_and_approves(tmp_path, monkeypatch):
    db = tmp_path / "keel.db"
    seed(connect(db), SystemClock())
    monkeypatch.setenv("KEEL_DB", str(db))
    monkeypatch.setenv("KEEL_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    app = testing.AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert "Set ANTHROPIC_API_KEY" in app.info[0].value
    assert "Team standup" in app.code[0].value

    app.button(key="approve-1").click().run()
    assert not app.exception
    assert (tmp_path / "outbox" / "draft-1.eml").exists()
