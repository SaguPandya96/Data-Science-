# Keel Agent: Design

| | |
| --- | --- |
| **Status** | Working prototype. Offline benchmark complete; end-to-end evaluation not yet run. |
| **Owner** | Sagar Pandya |
| **Code** | `src/keel/` |
| **Related** | [README](../README.md) (results), [Analysis plan](ANALYSIS_PLAN.md) (evaluation design) |

## 1. Purpose

Keel is a single-user personal agent. It keeps a long-term memory of the person it works
for, plans toward their goals, manages their calendar, tasks and notes, and prepares
actions such as emails for them to approve.

The design is built around three commitments:

1. **Current over complete.** The model should see what is true about the user *now*.
   An outdated fact in the prompt is treated as an error, not as harmless extra context.
2. **The user decides anything that leaves the machine.** The agent can read and organize
   the user's own data freely. It can never send, book or pay on its own.
3. **Everything is inspectable.** Every memory, every tool call and every pending action
   can be listed, and memories can be deleted.

### Non-goals

- Multi-user or hosted deployment. Keel runs locally against one SQLite file.
- Real integrations with mail, calendar or payment providers. The approval gate is
  implemented; the delivery end (an `.eml` file in an outbox folder) is a stub.
- Autonomous background work. The daily brief can run from cron, but the agent loop only
  runs when the user sends a message.

## 2. Architecture

```text
┌──────────┐   message   ┌─────────────────────────────── Agent ────────────────────────────────┐
│ CLI /    │ ──────────► │ 1. Context builder: time, retrieved memories, memory keys, agenda   │
│ Streamlit│             │ 2. Model call (system prompt + tools are static and cached)          │
│          │ ◄────────── │ 3. Tool loop: execute, append results, repeat until final answer     │
└──────────┘   answer    └───────────────┬──────────────────────────────┬───────────────────────┘
                                         │                              │
                               ┌─────────▼─────────┐          ┌─────────▼─────────┐
                               │ Memory            │          │ Toolbox           │
                               │ store + retriever │          │ validate → risk → │
                               └─────────┬─────────┘          │ run / queue → log │
                                         │                    └─────────┬─────────┘
                                         └──────────┬───────────────────┘
                                              ┌─────▼──────┐
                                              │  SQLite    │  memories, events, tasks, goals,
                                              │  keel.db   │  notes, drafts, approvals, audit
                                              └────────────┘
```

| Component | Module | Responsibility |
| --- | --- | --- |
| Agent | `agent.py` | Builds per-turn context, runs the model and tool loop, ends sessions. |
| Model | `model.py` | Claude client; a scripted model with the same interface for tests. |
| Memory store | `memory/store.py` | Writes memories, retires superseded values, deletes on request. |
| Retriever | `memory/retrieval.py` | Chooses which memories enter the prompt. |
| Embeddings | `memory/embeddings.py` | Sentence vectors: mpnet (ONNX), WordLlama fallback; model download and checksum. |
| Reranker | `memory/rerank.py` | Cross-encoder second stage over the retriever's short list. |
| Toolbox | `tools/registry.py` | Tool schemas, input validation, risk handling, audit log. |
| Tools | `tools/builtin.py` | Memory, calendar, tasks, goals, notes, email, brief. |
| Approvals | `approvals.py` | Queue for outward actions; approve or reject. |
| Brief | `briefing.py` | Daily summary computed from the database, no model call. |

## 3. Request lifecycle

For each user message:

1. **Retrieve.** The retriever selects up to 8 memories for the message (section 4).
2. **Assemble context.** A `<context>` block is placed at the start of the user message,
   holding the current time, the selected memories, the memory keys already in use,
   today's calendar and the number of open approvals.
3. **Call the model.** The system prompt and tool list are identical on every request, so
   the provider caches them. Everything that varies goes in the user message.
4. **Run tools.** Every `tool_use` block in the reply goes through the Toolbox. All results
   from one reply go back in a single message, with failures marked as errors rather than
   raised.
5. **Repeat** until the model answers without calling a tool, declines, or reaches the
   step limit (12). A paused server-side tool (web search) is resumed by sending the
   history back unchanged.

The conversation history is append-only: model replies are stored exactly as received and
never edited. When a turn fails on an API error, the CLI removes that whole turn so the
history stays valid.

At the end of a session, `reflect()` gives the model one final turn to save anything it
learned but did not store.

## 4. Memory

### Model

| Field | Meaning |
| --- | --- |
| `kind` | `fact`, `preference`, `constraint`, `goal` or `episode` |
| `key` | Optional name of the personal detail, e.g. `home_city`. Normalized so `Home City` and `user_home_city` match. |
| `importance` | 1 to 5. Defaults by kind: constraint 5, goal 4, fact and preference 3, episode 1. |
| `superseded_by` | Set when a newer memory with the same key is written. |
| `deleted` | Set by `forget`; the text is blanked at the same time. |

### Supersession

Writing a memory under a key that already has an active memory retires the old one. It
stays in the database, so `keel memories --all` can show the history, but it is never
retrieved again. This is what keeps "lives in Denver" out of the prompt once the user has
moved to Austin.

Supersession only works if the model reuses the key. To make that likely, the context
block lists every key in use, the `remember` tool's description asks for reuse, and the
tool's result reports what was replaced. The benchmark's key-noise test measures the cost
when this fails.

### Retrieval

1. Only active memories are candidates.
2. Constraints (allergies, hard limits) are always included, highest importance first,
   taking at most half of the budget.
3. The remaining slots go to the highest hybrid score:

   ```text
   score = 1.0  × BM25(question, key + text) / max BM25
         + w    × cosine(embed(question), embed(key + text))
         + 0.25 × 0.5^(age in days / 30)
         + 0.15 × (importance − 1) / 4
   ```

   BM25 uses Snowball stemming and a fixed list of synonym groups for personal topics,
   with synonyms weighted at 0.6. The embedding term uses the best encoder available, with
   its weight `w` chosen by the benchmark (rounds 2 and 6):

   | Encoder | `w` | Used when |
   | --- | --- | --- |
   | `all-mpnet-base-v2` (ONNX Runtime, CPU) | 12.0 | the `transformer` extra is installed and the model is cached or downloadable |
   | WordLlama `l2_supercat` | 0.5 | the `embed` extra is installed |
   | none | — | neither; BM25 hybrid only |

   The mpnet archive (about 400 MB) is downloaded once to `~/.cache/keel/models`, checked
   against a pinned SHA-256 and discarded on mismatch. Inference is single-threaded so a
   given text always produces the same vector.
4. **Reranking.** When the encoder is mpnet and the reranker is available, the 20
   highest-scoring candidates are rescored by `ms-marco-MiniLM-L-6-v2`, a cross-encoder
   that reads the question and the memory's key and text together:

   ```text
   final = score + 2.0 × cross-encoder logit
   ```

   Memories outside those 20 are never shown. The model (about 90 MB) is downloaded and
   verified like the encoder; without it, step 3's score is used as is. The list size and
   weight were chosen by round 7 of the benchmark.
5. A candidate whose words match one already chosen is skipped.

The `recall` tool uses the same retriever without the always-on constraints, so the model
can search again with its own wording when the context block is not enough.

### Privacy

- `remember` rejects text that looks like a card number or a US social security number,
  or that mentions a password, passcode, PIN code, API key or secret key, and tells the
  model it was not saved.
- `forget` blanks the memory text as well as hiding it.
- The database lives on the user's machine and is excluded from version control.

## 5. Tools

Every tool declares a risk level, which the Toolbox enforces:

| Risk | Meaning | Handling |
| --- | --- | --- |
| `read` | Looks something up | Runs immediately |
| `write` | Changes the user's own local data | Runs immediately; logged |
| `outward` | Reaches another person or service | Never runs; becomes a pending approval |

| Tool | Risk | Purpose |
| --- | --- | --- |
| `remember` | write | Save a durable fact, preference, constraint, goal or event |
| `recall` | read | Search memory with the model's own wording |
| `forget` | write | Delete a memory at the user's request |
| `calendar_list` | read | Events overlapping a date range |
| `calendar_add` | write | Add an event; refuses on conflict unless told otherwise |
| `calendar_find_free` | read | Free slots of a given length on a date |
| `task_add` / `task_list` / `task_complete` | write / read / write | To-dos, optionally linked to a goal |
| `goal_create` / `goal_plan` / `goal_checkin` / `goal_list` | write / write / write / read | Goals, dated milestones (never after the target date), progress |
| `note_write` / `note_search` | write / read | Longer notes, searched with BM25 |
| `email_draft` | write | Save a draft; validates the address |
| `email_send` | outward | Request that a draft be sent |
| `daily_brief` | read | The same brief as `keel brief` |
| `web_search` | server-side | Provided by the model API, up to 5 searches per request |

Input is checked against each tool's schema before it runs: missing or unexpected fields,
malformed dates and out-of-range values come back to the model as error results it can
correct. Unknown tools are rejected the same way.

## 6. Safety model

| Risk | Control |
| --- | --- |
| Agent acts on the user's behalf without consent | Outward tools only create approvals. Only the CLI (`keel approve`) or the app can execute them. |
| Agent claims an action happened when it didn't | The queued result tells the model nothing was sent; the system prompt forbids claiming otherwise; the CLI and app show pending approvals after each turn. |
| Instructions hidden in web pages or documents | The system prompt treats tool and web content as data. The approval gate limits the damage if that fails. |
| Secrets written to long-term memory | Pattern checks in `remember`, independent of the model. |
| Outdated facts drive decisions | Supersession and active-only retrieval. |
| Runaway tool loops | Step limit of 12 per turn. |
| Model declines a request | Server-side fallbacks retry the request on a fallback model; a final refusal is shown to the user with its category. |
| Silent failures | Every tool call, including errors, is written to the `audit` table. |

## 7. Configuration

| Setting | Default | Where |
| --- | --- | --- |
| Database | `~/.keel/keel.db` | `--db` or `KEEL_DB` |
| Outbox for approved email | `~/.keel/outbox` | `--outbox` or `KEEL_OUTBOX` |
| Model | `claude-opus-5-5` | `keel chat --model` |
| Effort | `medium` | `keel chat --effort` |
| Memories per turn | 8 | `Agent(memory_k=...)` |
| Step limit | 12 | `Agent(max_steps=...)` |
| Embedding search | Best available encoder | `pip install -e ".[embed,transformer]"` |
| Reranking | On when the encoder and reranker are available | the `transformer` extra |
| Model cache | `~/.cache/keel/models` | `KEEL_MODEL_DIR` |
| Forbid model downloads | off | `KEEL_OFFLINE=1` |
| Credentials | `ANTHROPIC_API_KEY` or an `ant auth login` profile | Environment |

## 8. Operations

- **Daily brief:** `keel brief` needs no API key and makes no model call, so it can run
  from cron, e.g. `45 7 * * 1-5 keel brief | mail -s "Today" me@example.com`.
- **Review:** `keel approvals`, `keel approve ID`, `keel reject ID`.
- **Inspect memory:** `keel memories --all` (history included), `keel forget ID`.
- **Audit:** the app's Activity tab, or `SELECT * FROM audit` on the database.

## 9. Testing

- **Unit and integration tests** (`tests/`) run the full agent loop against a scripted
  model: tool round trips, parallel calls, error results, step limits, refusals, paused
  turns, append-only history, approvals, the brief, the CLI and the Streamlit app. They
  also cover model download, checksum rejection and offline fallback. They need no key and
  never download a model.
- **Retrieval benchmark** (`scripts/run_benchmark.py`, `run_round2.py`, `run_round4.py`
  to `run_round7.py`): 200 synthetic users with pre-registered pass rules. CI reruns every
  round. Rounds 1 and 2 must match exactly; rounds 4 to 7 run transformer models, whose
  last bits vary with the CPU, so their decisions must match exactly and their numbers to
  within 0.1 points (`scripts/check_metrics.py`).
- **End-to-end check** (`keel eval-live`): the model answers the held-out benchmark
  questions from each retriever's memories, and the answers are graded automatically. It
  costs money, so it is run manually. Calls run in parallel, and every answer is saved to
  `reports/live/answers.jsonl` as it arrives, so an interrupted run resumes without paying
  twice.

## 10. Known limitations

- Questions that only imply their topic are the weak spot: 76.2% retrieved correctly on
  the latest held-out set (benchmark round 7), against 99.9% for direct questions.
- The first run of the agent downloads the encoder and reranker (about 400 MB and 90 MB).
  Offline, it falls back to WordLlama without reranking, which retrieves less well.
- Reranking adds about 130 ms to every turn on one CPU thread. Unlike embedding, this
  cost is paid per question, not once per memory.
- The encoder embeds about 50 texts a second on one CPU thread. Fine for one person's
  memory, since each text is embedded once, but slow for bulk imports.
- Supersession depends on consistent keys. With every update written under a new key,
  clean retrieval of changed details falls from 67.8% to 26.1%.
- The benchmark assumes perfect memory writing; how well the model chooses what to
  remember has not been measured.
- Single user, single process, local SQLite. No authentication, because there is no
  network surface.
- Times are local and timezone-naive.

## 11. Future work

1. Run `keel eval-live` to test whether better retrieval produces better answers.
2. Try embedding weights above 12, where round 6 stopped, and a quantized mpnet and
   reranker to cut the download, embedding and reranking time.
3. Suggest existing keys to the model when a new key looks like a near-duplicate of one
   already in use.
4. Real calendar and mail integrations behind the existing approval gate.
