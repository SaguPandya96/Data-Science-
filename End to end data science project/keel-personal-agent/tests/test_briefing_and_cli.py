from __future__ import annotations

from datetime import timedelta

from keel.briefing import build_brief
from keel.cli import main
from keel.demo import seed, seed_if_empty


def test_empty_brief(toolbox):
    brief = build_brief(toolbox.context)
    assert brief.empty
    assert "A clear day" in brief.render()


def test_demo_brief_covers_every_section(conn, clock, toolbox):
    seed(conn, clock)
    brief = build_brief(toolbox.context)
    assert [e["title"] for e in brief.events] == ["Team standup", "1:1 with manager"]
    assert [t["title"] for t in brief.overdue] == ["Buy running shoes"]
    assert [t["title"] for t in brief.due_today] == ["Book physio check"]
    assert [m["title"] for m in brief.milestones_soon] == ["Run 5K without stopping"]
    assert [g["title"] for g in brief.stale_goals] == ["Run a half marathon"]
    assert len(brief.pending_approvals) == 1
    assert not seed_if_empty(conn, clock)


def test_brief_flags_clashes(toolbox, clock):
    day = clock.now().replace(hour=0)
    for title, start in (("A", 9), ("B", 9)):
        toolbox.run(
            "t",
            "calendar_add",
            {
                "title": title,
                "start": (day + timedelta(hours=start)).isoformat(),
                "end": (day + timedelta(hours=start + 1)).isoformat(),
                "allow_conflict": True,
            },
        )
    assert build_brief(toolbox.context).conflicts == [("A", "B")]


def test_cli_demo_flow(tmp_path, capsys):
    db = str(tmp_path / "k.db")
    assert main(["--db", db, "demo"]) == 0
    assert main(["--db", db, "brief"]) == 0
    assert main(["--db", db, "memories", "--all"]) == 0
    out = capsys.readouterr().out
    assert "Lives in Denver. {home_city}  [replaced]" in out
    outbox = ["--outbox", str(tmp_path / "out")]
    assert main(["--db", db, "approve", "1", *outbox, "--yes"]) == 0
    assert (tmp_path / "out" / "draft-1.eml").exists()
    assert main(["--db", db, "approve", "1", "--yes"]) == 2  # already executed
    assert main(["--db", db, "forget", "4"]) == 0
    assert main(["--db", db, "forget", "999"]) == 2


def test_cli_eval_live_needs_confirmation(capsys):
    assert main(["eval-live", "--personas", "1"]) == 1
    assert "--yes" in capsys.readouterr().out


def test_cli_approve_shows_the_email_and_asks_first(tmp_path, capsys, monkeypatch):
    db = str(tmp_path / "k.db")
    outbox = tmp_path / "out"
    assert main(["--db", db, "demo"]) == 0
    capsys.readouterr()

    answers = iter(["n", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    for _ in range(2):  # an explicit no, then just pressing Enter
        assert main(["--db", db, "approve", "1", "--outbox", str(outbox)]) == 1
    out = capsys.readouterr().out
    assert "To: " in out and "Subject: " in out
    assert "Not sent. #1 is still waiting" in out
    assert not outbox.exists()

    def no_terminal(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_terminal)
    assert main(["--db", db, "approve", "1", "--outbox", str(outbox)]) == 1
    assert not outbox.exists()

    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert main(["--db", db, "approve", "1", "--outbox", str(outbox)]) == 0
    assert (outbox / "draft-1.eml").exists()
