from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app/app.py"


def test_app_renders_default_plan():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert labels == ["Following the rule", "Choosing at random"]


def test_budget_change_updates_the_plan():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    before = app.metric[0].value
    app.slider[0].set_value(30).run()
    assert not app.exception
    assert app.metric[0].value != before
    assert any("larger than the both-category group" in info.value for info in app.info)
