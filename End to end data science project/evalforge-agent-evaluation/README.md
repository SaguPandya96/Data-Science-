# EvalForge

**Automated Evaluation and Adversarial Stress Testing for Multi-Turn AI Agents**

EvalForge evaluates *complete agent sessions* — not isolated model responses. It generates
adversarial multi-turn conversations, runs a productivity agent through them, records a full
execution trace, and scores the session on context retention, instruction adherence, tool
reliability, failure recovery and prompt-injection resistance.

> Full documentation is being assembled as the implementation lands. See `docs/` for the
> architecture, evaluation methodology and failure taxonomy.

## Status

Work in progress on branch `feature/evalforge-agent-evaluation`.

## Quick start

```bash
pip install -e ".[dev,dashboard]"
evalforge demo
```

Everything runs offline with a deterministic mock provider. No API keys required.
