"""keel: chat with your agent, read your brief, review what it wants to do.

keel chat                 talk to Keel (needs an Anthropic API key)
keel brief                today's agenda, due tasks, stale goals, pending approvals
keel memories [--all]     what Keel remembers (--all includes retired values)
keel forget ID            delete a memory
keel approvals            actions waiting for you
keel approve ID | reject ID
keel demo                 load a made-up user to explore
keel eval                 run the offline memory benchmark
keel eval-live --yes      end-to-end benchmark with Claude (costs money)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from keel import approvals
from keel.clock import SystemClock
from keel.db import connect
from keel.memory.store import MemoryStore
from keel.tools.registry import ToolContext

DEFAULT_DB = Path(os.environ.get("KEEL_DB", "~/.keel/keel.db"))
ROOT = Path(__file__).resolve().parents[2]


def _ctx(conn: sqlite3.Connection) -> ToolContext:
    clock = SystemClock()
    return ToolContext(conn, clock, MemoryStore(conn, clock))


def cmd_chat(args: argparse.Namespace) -> int:
    import anthropic

    from keel.agent import Agent
    from keel.model import ClaudeModel, api_key_available

    if not api_key_available():
        print("Set ANTHROPIC_API_KEY (or run `ant auth login`) to chat.", file=sys.stderr)
    conn = connect(args.db)
    agent = Agent(conn, ClaudeModel(model=args.model, effort=args.effort))
    print("Keel is listening. /brief, /memories, /approvals, /quit\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            message = "/quit"
            print()
        if not message:
            continue
        if message == "/quit":
            turn = agent.reflect()
            saved = [r for r in turn.tool_results if r.name == "remember" and not r.is_error]
            if saved:
                print(f"(saved {len(saved)} memories from this session)")
            return 0
        if message == "/brief":
            cmd_brief(args)
            continue
        if message == "/memories":
            cmd_memories(argparse.Namespace(db=args.db, all=False))
            continue
        if message == "/approvals":
            cmd_approvals(args)
            continue
        start = len(agent.messages)
        try:
            turn = agent.send(message)
        except anthropic.AuthenticationError:
            print("The API rejected the credentials. Check ANTHROPIC_API_KEY.", file=sys.stderr)
            return 1
        except anthropic.APIError as error:
            print(f"(API error on that message: {error})\n", file=sys.stderr)
            del agent.messages[start:]  # roll back the unfinished turn; history stays valid
            continue
        for result in turn.tool_results:
            flag = " (error)" if result.is_error else ""
            print(f"  · {result.name}{flag}")
        print(f"\nkeel> {turn.text}\n")
        if turn.approvals:
            ids = ", ".join(f"#{i}" for i in turn.approvals)
            print(f"  Waiting for your approval: {ids}. Run `keel approve <id>`.\n")


def cmd_brief(args: argparse.Namespace) -> int:
    from keel.briefing import build_brief

    print(build_brief(_ctx(connect(args.db))).render())
    return 0


def cmd_memories(args: argparse.Namespace) -> int:
    store = MemoryStore(connect(args.db))
    for memory in store.all(include_inactive=args.all):
        status = "" if memory.active else ("  [forgotten]" if memory.deleted else "  [replaced]")
        key = f" {{{memory.key}}}" if memory.key else ""
        print(f"#{memory.id:<4} {memory.render()}{key}{status}")
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    store = MemoryStore(connect(args.db))
    memory = store.forget(args.id)
    print(f"Forgot #{memory.id}: {memory.text}")
    return 0


def cmd_approvals(args: argparse.Namespace) -> int:
    items = approvals.pending(connect(args.db))
    if not items:
        print("Nothing waiting for approval.")
    for item in items:
        print(f"#{item.id} {item.summary}  ({item.created_at[:16]})")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    item = approvals.get(conn, args.id)
    if item.action == "email_send":
        draft = conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (item.payload["draft_id"],)
        ).fetchone()
        print(f"To: {draft['to_addr']}\nSubject: {draft['subject']}\n\n{draft['body']}\n")
    done = approvals.approve(conn, args.id, SystemClock(), Path(args.outbox))
    print(f"Approved #{done.id}. Written to {done.result}")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    done = approvals.reject(connect(args.db), args.id, SystemClock())
    print(f"Rejected #{done.id}: {done.summary}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from keel.demo import seed_if_empty

    if seed_if_empty(connect(args.db), SystemClock()):
        print(f"Demo user loaded into {Path(args.db).expanduser()}.")
    else:
        print("That database already has memories; use a fresh --db for the demo.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from keel.evaluation.benchmark import run
    from keel.evaluation.report import to_markdown

    results = run(args.config, args.split)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    print(to_markdown(results))
    return 0 if results["pass_rule"]["passed"] else 1


def cmd_eval_live(args: argparse.Namespace) -> int:
    from keel.evaluation.live import run_live
    from keel.model import ClaudeModel

    arms = args.arms.split(",")
    calls = args.personas * 16 * len(arms)
    print(f"This makes {calls} API calls to {args.model}.")
    if not args.yes:
        print("Re-run with --yes to spend that.")
        return 1
    model = ClaudeModel(model=args.model, effort="low", web_search=False, max_tokens=2000)
    results = run_live(args.config, model, args.personas, arms)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    for arm, stat in results["accuracy"].items():
        print(
            f"{arm:8s} {100 * stat['mean']:.1f}% "
            f"({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="database file (env KEEL_DB)")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="talk to Keel")
    chat.add_argument("--model", default="claude-opus-5-5")
    chat.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    chat.set_defaults(func=cmd_chat)

    sub.add_parser("brief", help="today's brief").set_defaults(func=cmd_brief)

    memories = sub.add_parser("memories", help="list memories")
    memories.add_argument("--all", action="store_true", help="include replaced and forgotten")
    memories.set_defaults(func=cmd_memories)

    forget = sub.add_parser("forget", help="delete a memory")
    forget.add_argument("id", type=int)
    forget.set_defaults(func=cmd_forget)

    sub.add_parser("approvals", help="pending approvals").set_defaults(func=cmd_approvals)
    approve = sub.add_parser("approve", help="approve and carry out an action")
    approve.add_argument("id", type=int)
    approve.add_argument("--outbox", default=os.environ.get("KEEL_OUTBOX", "~/.keel/outbox"))
    approve.set_defaults(func=cmd_approve)
    reject = sub.add_parser("reject", help="reject an action")
    reject.add_argument("id", type=int)
    reject.set_defaults(func=cmd_reject)

    sub.add_parser("demo", help="load a made-up user").set_defaults(func=cmd_demo)

    config = str(ROOT / "configs" / "eval.toml")
    evaluate = sub.add_parser("eval", help="offline memory benchmark")
    evaluate.add_argument("--split", default="test", choices=["dev", "test"])
    evaluate.add_argument("--config", default=config)
    evaluate.add_argument("--json", help="also write results here")
    evaluate.set_defaults(func=cmd_eval)

    live = sub.add_parser("eval-live", help="end-to-end benchmark with Claude")
    live.add_argument("--personas", type=int, default=10)
    live.add_argument("--arms", default="recent,lexical,keel")
    live.add_argument("--model", default="claude-opus-5-5")
    live.add_argument("--config", default=config)
    live.add_argument("--out", default=str(ROOT / "reports" / "metrics" / "live.json"))
    live.add_argument("--yes", action="store_true", help="confirm the API spend")
    live.set_defaults(func=cmd_eval_live)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
