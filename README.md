# Expense Helper

A simple single-agent learning project in Python. One AI agent chats with you and uses four tools (add, list, total-by-category, delete) to record, report, and remove expenses. It is a stepping stone before a simple multi-agent project.

Data persists in a local SQLite database, so your expenses survive across restarts.

## Features

- Add an expense (amount, category, optional note, date/time)
- List all recorded expenses
- Get the total spent in one category
- Delete an expense by ID, note, amount, and/or timestamp
- Future-dated expenses are previewed first and require confirmation before being added
- Deletions are previewed first and require confirmation before rows are removed
- Optional Streamlit web UI with two tabs: Chat and Expenses
- Plain Python scripts — no agent framework, so every step of the tool-use loop is visible

## Technology stack

- Python 3
- `openai` Python SDK, used against NVIDIA's OpenAI-compatible API
- `python-dotenv` to load settings from `.env`
- SQLite (built into Python) for persistent storage
- `pytest` for tests
- `streamlit` for the optional web UI (`app.py`)
- Model: `nvidia/nemotron-3.5-super-vl-120b-a12b` (early-access reasoning model, tool-calling support)

## Files

| File | Purpose |
|---|---|
| `expense_agent.py` | The whole agent: connection, tools, tool schemas, agent loop, chat prompt |
| `app.py` | Streamlit web UI (Chat + Expenses tabs). Run with `streamlit run app.py` |
| `test_expense_agent.py` | pytest tests |
| `AGENTS.md` | Developer context and project notes |
| `.gitignore` | Excludes `.env`, `__pycache__/`, `.pytest_cache/`, `expenses.db` |

## Installation and configuration

### 1. Clone or open the project folder

Open a terminal in the project folder (`C:\expense_AI_agent`).

### 2. Install dependencies

```powershell
pip install openai python-dotenv
```

For the optional web UI:

```powershell
pip install streamlit pytest
```

### 3. Create a `.env` file

Create a file named `.env` in the project folder (next to `expense_agent.py`) with your settings. Do not commit or share this file — it is excluded from git by `.gitignore`.

```
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
EXPENSES_DB_PATH=expenses.db
```

- `NVIDIA_API_KEY` — your NVIDIA API key (required; never hard-code it).
- `NVIDIA_BASE_URL` — the OpenAI-compatible base URL (default shown above).
- `NVIDIA_MODEL` — optional; defaults to `nvidia/nemotron-3.5-super-vl-120b-a12b`.
- `EXPENSES_DB_PATH` — optional path to the SQLite database; defaults to `expenses.db`.

Use no quotes or spaces around values. The script calls `load_dotenv()`, then reads `os.environ[...]`. A `KeyError` usually means `.env` is missing, misnamed, or the script is run from a different folder.

## How to run the application

### Command-line chat

```powershell
python expense_agent.py
```

Type your message and press Enter. Type `quit` or `exit` to leave.

Example conversation:

```
You: I spent 12.50 on food for lunch today
Agent: Added 12.5 to food.

You: what is my total for food?
Agent: Total for Food: 12.5

You: list my expenses
Agent: 12.5 - food - lunch - 2026-11-05 02:30 PM
```

### Streamlit web UI

```powershell
streamlit run app.py
```

The UI has two tabs:

- **Chat:** type a message with `st.chat_input`; assistant replies appear with `st.chat_message`. Under each assistant reply, an expander called "Tool calls" lists the tools used and their results.
- **Expenses:** a table of all expenses (newest first, times in 12-hour AM/PM) plus a total per category. Includes a Refresh button.

Errors (model, network, database) are shown with `st.error`, never a crash.

## How the AI agent works

### Connection

`expense_agent.py` loads settings from `.env`, creates an `OpenAI` client pointed at NVIDIA's base URL, and selects the model from `NVIDIA_MODEL` (or the default).

### Tools (the four things the agent can do)

| Tool | What it does |
|---|---|
| `add_expense` | Records a new expense. Amount and category are required; note and spent_at are optional. |
| `list_expenses` | Shows every recorded expense with its 12-hour display time. |
| `total_by_category` | Adds up all expenses in one category. |
| `delete_expense` | Deletes expenses matching all provided criteria (id, note, amount, spent_at). Requires at least one criterion. |

Amounts can be numbers or strings. Categories are stored lowercase.

### Confirmation guardrails

- **Future-dated add:** if `spent_at` is in the future, the tool previews the expense and asks `Are you sure you want to add this record? (y/n)` without inserting anything. Call again with `confirm="y"` to add, or `confirm="n"` to cancel. Non-future timestamps insert immediately.
- **Delete:** calling `delete_expense` without `confirm` previews matching rows and asks `Are you sure you want to delete this record? (y/n)` without deleting anything. Call again with `confirm="y"` to delete, or `confirm="n"` to cancel. Calling with no criteria returns an error.

### Agent loop

`run_agent(user_message, history)`:

1. Appends the current local date/time (12-hour AM/PM plus weekday) and the user message to the conversation history.
2. Calls the model with the history and the tool menu (`TOOLS`), up to `MAX_STEPS` (5) times.
3. If the model replies with no tool call, returns its text — done.
4. If the model requests tool calls, runs each matching Python function, sends the result back with the matching `tool_call_id`, and continues the loop.
5. On any error, returns `Error: ...` instead of crashing.
6. If it hits `MAX_STEPS`, returns `Stopped: too many steps.`

The model is instructed by `SYSTEM_PROMPT` to use tools, never invent numbers, convert relative times (e.g. "yesterday", "at 3pm", "noon", "midnight") into exact date/times, and use the delete-then-add confirmation flow when correcting mistakes.

## SQLite database details

- Database file: `expenses.db` by default (path controlled by `EXPENSES_DB_PATH`).
- Created automatically on first run (`init_db()` runs at import).
- Table `expenses`:

| Column | Type | Notes |
|---|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Row identifier |
| `amount` | `REAL NOT NULL` | Amount spent |
| `category` | `TEXT NOT NULL` | Stored lowercase |
| `note` | `TEXT DEFAULT ''` | Optional description |
| `spent_at` | `TEXT` | Stored as `YYYY-MM-DD HH:MM` (24-hour local time) |

- A safe migration adds the `spent_at` column to existing databases that lack it and backfills old rows with today's date at `00:00`.
- Data persists across restarts. Excluded from git by `.gitignore`.

### Helper

`format_time_12h(stored_time)` converts `YYYY-MM-DD HH:MM` to `YYYY-MM-DD hh:MM AM/PM` for display.

## Example usage

```powershell
# Command-line chat
python expense_agent.py

# Streamlit UI
streamlit run app.py

# Run tests
pytest
```

Tests use a fresh temporary database per test and mock the model client, so they never call the real model or use the real API key. Run from the project folder:

```powershell
pytest
pytest -v
pytest test_expense_agent.py::test_add_expense_records_amount_category_note
```

## Known limits

- No input validation beyond what the model sends (for example, negative amounts are not rejected).
- The model is early access and may change or behave differently from stable models. Tool-calling behavior has not been fully tested.
- A reasoning model may pause to think before replying.
- The model accepts images, but this project uses text only.
