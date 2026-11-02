# Expense Helper: Project Context

Read this first. It gives you the full picture of the project so you can answer quickly and accurately.

## What this project is
A **simple single-agent learning project** in Python. One AI agent chats with the user and uses three tools to record and report expenses. It is a stepping stone before a simple multi-agent project.

- **Goal:** understand the single-agent tool-use loop (ReAct style) end to end.
- **Not a goal:** production quality, a UI, or multi-agent design (that comes later).

## Tech stack
- Python 3, plain scripts (no agent framework on purpose, so every step is visible)
- `openai` Python SDK, used against NVIDIA's OpenAI-compatible API
- `python-dotenv` to load settings from `.env`
- `pytest` for tests
- Model: `nvidia/nemotron-3.5-super-vl-120b-a12b` (early-access reasoning model, supports tool calling, accepts text/image/video, outputs text)
- Base URL: `https://integrate.api.nvidia.com/v1`

## Files
| File | Purpose |
|---|---|
| `expense_agent.py` | The whole agent: connection, tools, tool schemas, agent loop, chat prompt |
| `.env` | Real secrets and settings. **Never read aloud, print, commit, or share.** |
| `.env.example` | Safe template with fake values (optional; may not exist in the project folder) |
| `test_expense_agent.py` | pytest tests for the project |
| `expenses.db` | SQLite database storing expenses (created on first run). Excluded from git. |
| `.pytest_cache/`, `__pycache__/` | Auto-generated; ignore, never edit |
| `.gitignore` | Excludes `.env`, `__pycache__/`, `expenses.db` |
| `AGENTS.md` | This file |

## Configuration (`.env`)
```
NVIDIA_API_KEY=nvapi-...        # secret
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
EXPENSES_DB_PATH=expenses.db    # optional; defaults to expenses.db
```
- No quotes or spaces around values.
- The script calls `load_dotenv()`, then reads `os.environ[...]`.
- A `KeyError` usually means the `.env` file is missing, misnamed (e.g. `.env.txt`), or the script is run from a different folder.

## How the code is organized (`expense_agent.py`)
1. **Connect:** `load_dotenv()`, create `OpenAI(base_url=..., api_key=...)`, set `MODEL`.
2. **Data:** Expenses are stored in a SQLite table (`expenses` table: `id`, `amount`, `category`, `note`) in the file given by `EXPENSES_DB_PATH` (default `expenses.db`). Data persists across restarts.
3. **Tools** (plain Python functions):
   - `add_expense(amount, category, note="")`
   - `list_expenses()`
   - `total_by_category(category)`
4. **Tool registry and schemas:** `TOOL_FUNCTIONS` maps tool name to function. `TOOLS` is the JSON "menu" the model sees (name, description, parameters).
5. **System prompt:** `SYSTEM_PROMPT` tells the model to use tools and never invent numbers.
6. **Agent loop:** `run_agent(user_message, history)`.
7. **Chat loop:** `if __name__ == "__main__":` reads input until `quit` or `exit`.

## The agent loop (core idea)
```
append user message to history
repeat up to MAX_STEPS (5):
    call the model with history + TOOLS
    append the model's reply to history
    if the reply has no tool_calls -> return its text (done)
    for each tool call:
        parse arguments (JSON) and run the matching function
        on any error, use "Error: ..." as the result (never crash)
        print "[tool] name -> result" (simple logging)
        append a {"role": "tool", "tool_call_id": ..., "content": ...} message
return "Stopped: too many steps."
```
Key rules:
- Every tool result must be sent back with the matching `tool_call_id`.
- `history` keeps the whole conversation, so the agent has short-term memory within one run.
- `MAX_STEPS` prevents infinite loops.

## How to add a new tool (3 edits)
1. Write the Python function.
2. Add it to `TOOL_FUNCTIONS`.
3. Add its schema to `TOOLS` with a **clear description** (the model relies on this to decide when to use it).

## Testing
- Test file: `test_expense_agent.py` (pytest). Project folder: `C:\expense_AI_agent`.
- Run from the project folder: `pytest` (or `pytest -v` for detail, `pytest test_expense_agent.py::test_name` for one test).
- Install if missing: `pip install pytest`
- **Tests must not call the real model or use the real API key.** Test the tool functions directly and, if testing the agent loop, mock the model client. This keeps tests free, fast, and repeatable.
- Each test gets a fresh temporary database (via pytest `tmp_path`) so tests don't affect each other. The database path is set through `EXPENSES_DB_PATH`.
- What to cover:
  - Each tool: `add_expense`, `list_expenses`, `total_by_category`
  - Edge cases: empty list, unknown category, mixed-case category (categories are stored lowercase), amount given as a string or number
  - Agent loop (mocked model): returns text when there are no tool calls, runs a requested tool, returns an error string instead of crashing on a bad tool call, and stops at `MAX_STEPS`
- When adding a new tool, add a test for it in the same change.
- Before saying a change is done, run `pytest` and report the result.

## Known limits and notes
- Data persists in SQLite (`expenses.db` by default); no JSON file yet.
- No input validation beyond what the model sends (for example, negative amounts are not rejected).
- No human-approval step before actions. Add one before any delete or other irreversible tool.
- The model is early access and may change or behave differently from stable models. Tool-calling behavior has not been fully tested.
- A reasoning model may pause to think before replying.
- The model accepts images, but this project uses text only.

## Planned learning steps (in order)
1. Run it and read the `[tool]` log lines. (done)
2. Add `delete_last_expense`.
3. Handle an unsupported request (for example, weather) gracefully.
4. Save expenses to persistent storage (done — SQLite). (update the tests too)
5. Add confirmation before deleting (human-in-the-loop).
6. Later: split into a simple multi-agent project (for example, a recorder agent and a summary agent).

## How to help me
- Explain in **plain language with simple analogies**, then show the code.
- Keep answers short and practical; show only the lines that change.
- Never ask me to paste my API key. Refer to it as `NVIDIA_API_KEY`.
- When something fails, ask for the error message and the line it points to.
- When a test fails, show the failing test name and the assertion message, and say whether the bug is in the code or the test.
- Prefer small changes that keep the single-file structure.
