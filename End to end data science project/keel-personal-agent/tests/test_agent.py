from __future__ import annotations

from types import SimpleNamespace

from keel.agent import SYSTEM_PROMPT, Agent
from keel.model import ClaudeModel, ModelReply, ScriptedModel, text, tool_calls


def make_agent(conn, clock, *replies, **kwargs):
    model = ScriptedModel(replies)
    return Agent(conn, model, clock, session_id="s1", **kwargs), model


def test_plain_answer_includes_memory_context(conn, clock, store):
    store.add("Lives in Austin.", kind="fact", key="home_city")
    agent, model = make_agent(conn, clock, text("You live in Austin."))
    turn = agent.send("Where do I live?")
    assert turn.text == "You live in Austin."
    sent = model.requests[0]["messages"][0]["content"]
    assert "Lives in Austin." in sent[0]["text"]
    assert "Memory keys in use: home_city" in sent[0]["text"]
    assert sent[1]["text"] == "Where do I live?"
    assert model.requests[0]["system"] == SYSTEM_PROMPT


def test_tool_call_round_trip(conn, clock, store):
    agent, model = make_agent(
        conn,
        clock,
        tool_calls(
            ("tu_1", "remember", {"text": "Vegetarian.", "kind": "preference", "key": "diet"})
        ),
        text("Noted: vegetarian."),
    )
    turn = agent.send("I'm vegetarian now.")
    assert turn.text == "Noted: vegetarian."
    assert [m.text for m in store.all()] == ["Vegetarian."]
    result_message = model.requests[1]["messages"][-1]
    assert result_message["role"] == "user"
    assert result_message["content"][0]["tool_use_id"] == "tu_1"
    assert "is_error" not in result_message["content"][0]


def test_parallel_calls_return_in_one_message_with_errors_flagged(conn, clock):
    agent, model = make_agent(
        conn,
        clock,
        tool_calls(
            ("a", "task_add", {"title": "Buy shoes"}),
            ("b", "task_complete", {"task_id": 42}),
        ),
        text("Done."),
    )
    turn = agent.send("Add a task and finish task 42.")
    blocks = model.requests[1]["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["a", "b"]
    assert blocks[1]["is_error"] is True
    assert [r.is_error for r in turn.tool_results] == [False, True]


def test_outward_action_surfaces_approval(conn, clock):
    agent, _ = make_agent(
        conn,
        clock,
        tool_calls(("d", "email_draft", {"to": "a@b.co", "subject": "Hi", "body": "Hello"})),
        tool_calls(("s", "email_send", {"draft_id": 1})),
        text("Drafted; waiting for your approval."),
    )
    turn = agent.send("Email a@b.co hello")
    assert turn.approvals == [1]
    assert conn.execute("SELECT status FROM drafts").fetchone()[0] == "draft"


def test_step_limit_stops_runaway_loops(conn, clock):
    replies = [tool_calls((f"x{i}", "task_list", {})) for i in range(3)]
    agent, _ = make_agent(conn, clock, *replies, max_steps=3)
    turn = agent.send("loop forever")
    assert turn.stop_reason == "max_steps"
    assert turn.steps == 3


def test_refusal_and_max_tokens(conn, clock):
    refusal = ModelReply(content=[], stop_reason="refusal", refusal_category="cyber")
    agent, _ = make_agent(conn, clock, refusal)
    assert "declined: cyber" in agent.send("something").text

    cut = ModelReply(content=[{"type": "text", "text": "Partial"}], stop_reason="max_tokens")
    agent, _ = make_agent(conn, clock, cut)
    assert agent.send("long").text.endswith("[Reply cut off at the length limit.]")


def test_pause_turn_continues_without_a_new_user_message(conn, clock):
    paused = ModelReply(content=[{"type": "server_tool_use", "id": "w"}], stop_reason="pause_turn")
    agent, model = make_agent(conn, clock, paused, text("Found it."))
    assert agent.send("search the web").text == "Found it."
    assert model.requests[1]["messages"][-1]["role"] == "assistant"


def test_history_is_append_only_across_turns(conn, clock):
    agent, model = make_agent(conn, clock, text("One."), text("Two."))
    agent.send("first")
    first_history = [m["content"] for m in model.requests[0]["messages"]]
    agent.send("second")
    second_history = [m["content"] for m in model.requests[1]["messages"]]
    assert second_history[: len(first_history)] == first_history


def test_reflect_saves_and_closes_session(conn, clock, store):
    agent, _ = make_agent(
        conn,
        clock,
        text("Congrats on the new job!"),
        tool_calls(
            ("r", "remember", {"text": "Works at Brightline.", "kind": "fact", "key": "employer"})
        ),
        text("Saved."),
    )
    agent.send("I started at Brightline today")
    agent.reflect()
    assert [m.text for m in store.all()] == ["Works at Brightline."]
    assert conn.execute("SELECT reflected FROM sessions").fetchone()[0] == 1


class _FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=7)
        block = SimpleNamespace(to_dict=lambda: {"type": "text", "text": "hi"})
        return SimpleNamespace(content=[block], stop_reason="end_turn", usage=usage)


def test_claude_model_request_shape():
    messages = _FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    reply = ClaudeModel(client=client).reply("sys", [{"role": "user", "content": "x"}], [])
    sent = messages.kwargs
    assert sent["model"] == "claude-opus-5-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["fallbacks"] == "default"
    assert sent["tools"][-1]["type"] == "web_search_20260209"
    assert reply.content == [{"type": "text", "text": "hi"}]
    assert reply.usage["cache_read_input_tokens"] == 7
