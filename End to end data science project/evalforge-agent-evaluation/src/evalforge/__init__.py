"""EvalForge: automated evaluation and adversarial stress testing for multi-turn AI agents.

EvalForge treats a *complete agent session* as the unit of evaluation. A session is a
multi-turn conversation in which the agent accumulates facts, honours persistent
constraints, calls tools, recovers from injected failures and produces artifacts. The
package generates adversarial sessions, executes them against an agent under test,
records a full trace, and scores the trace with deterministic evaluators plus optional
semantic and LLM-based judges.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
