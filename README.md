# Expense Helper

A simple single-agent learning project in Python. One AI agent chats with you and uses eight tools to record, report, and remove expenses, plus KiwiSaver balance tracking. It is a stepping stone before a simple multi-agent project.

Data persists in a local SQLite database, so your expenses and KiwiSaver balances survive across restarts.

## Features

- Add an expense (amount, category, optional note, date/time)
- List all recorded expenses
- Get the total spent in one category
- Delete an expense by ID, note, amount, and/or timestamp
- Future-dated expenses are previewed first and require confirmation before being added
- Deletions are previewed first and require confirmation before rows are removed
- KiwiSaver: add daily balances (date is unique), add adjustments (employee, employer, government, tax, fee), list daily balances with % change, compute performance summaries
- KiwiSaver overwrite guardrail: replacing an existing date previews old vs new and requires confirmation
- KiwiSaver data quality: import from CSV, skip existing dates, flag stale (unchanged) days
- Optional Streamlit web UI with three tabs: Chat, Expenses, and KiwiSaver (line chart, summary, recent entries)
- Plain Python scripts — no agent framework, so every step of the tool-use loop is visible

## Technology stack

- Python 3
- `openai` Python SDK, used against NVIDIA's OpenAI-compatible API
- `python-dotenv` to load settings from `.env`
- SQLite (built into Python) for persistent storage
- `pytest` for tests
- `streamlit` for the optional web UI (`app.py`)
- Standard library `datetime` for date parsing
- Model: `nvidia/nemotron-3.5-super-vl-120b-a12b` (early-access reasoning model, tool-calling support)

## Files

| File | Purpose |
|---|---|
| `expense_agent.py` | The whole agent: connection, tools, tool schemas, agent loop, chat prompt |
| `app.py` | Streamlit web UI (Chat + Expenses + KiwiSaver tabs). Run with `streamlit run app.py` |
| `import_kiwisaver.py` | One-off script: import KiwiSaver daily balances from a CSV into `kiwisaver_daily` |
| `test_expense_agent.py` | pytest tests |
| `AGENTS.md` | Developer context and project notes |
| `.gitignore` | Excludes `.env`, `__pycache__/`, `.pytest_cache/`, `expenses.db`, `*.csv` |

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

The UI has three tabs:

- **Chat:** type a message with `st.chat_input`; assistant replies appear with `st.chat_message`. Under each assistant reply, an expander called "Tool calls" lists the tools used and their results.
- **Expenses:** a table of all expenses (newest first, times in 12-hour AM/PM) plus a total per category. Includes a Refresh button.
- **KiwiSaver:** a line chart of balance over time, a performance summary from `kiwisaver_performance`, and a table of recent entries. Includes a Refresh button.

Errors (model, network, database) are shown with `st.error`, never a crash.

## How the AI agent works

### Connection

`expense_agent.py` loads settings from `.env`, creates an `OpenAI` client pointed at NVIDIA's base URL, and selects the model from `NVIDIA_MODEL` (or the default).

### Tools (ten things the agent can do)

| Tool | What it does |
|---|---|
| `add_expense` | Records a new expense. Amount and category are required; note and spent_at are optional. |
| `list_expenses` | Shows every recorded expense with its 12-hour display time. |
| `total_by_category` | Adds up all expenses in one category. |
| `delete_expense` | Deletes expenses matching all provided criteria (id, note, amount, spent_at). Requires at least one criterion. |
| `add_kiwisaver_daily` | Inserts or replaces a KiwiSaver daily balance row (date is unique). Validates balance > 0, date format, and future dates. Warns if units × unit_price differs from balance by >1%. |
| `list_kiwisaver_daily` | Lists all KiwiSaver daily balance rows. |
| `add_kiwisaver_adjustment` | Inserts a KiwiSaver adjustment row. Type must be employee, employer, government, tax, or fee; amount must not be zero. |
| `list_kiwisaver_adjustments` | Lists all KiwiSaver adjustment rows. |
| `list_kiwisaver` | Lists KiwiSaver daily balances in date order with daily % change. Optionally filter by start_date and/or end_date. |
| `kiwisaver_performance` | Computes Kiwisaver performance between two dates: start/end value, % change, best/worst day, max drawdown, day count, stale days. |

Amounts can be numbers or strings. Categories are stored lowercase.

### Confirmation guardrails

- **Future-dated add:** if `spent_at` is in the future, the tool previews the expense and asks `Are you sure you want to add this record? (y/n)` without inserting anything. Call again with `confirm="y"` to add, or `confirm="n"` to cancel. Non-future timestamps insert immediately.
- **Delete:** calling `delete_expense` without `confirm` previews matching rows and asks `Are you sure you want to delete this record? (y/n)` without deleting anything. Call again with `confirm="y"` to delete, or `confirm="n"` to cancel. Calling with no criteria returns an error.
- **KiwiSaver overwrite:** `add_kiwisaver_daily` with a date that already exists previews old vs new balance and asks `Are you sure you want to overwrite? (y/n)`. Call again with `confirm="y"` to overwrite, or `confirm="n"` to cancel. New dates insert immediately.

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

- Table `kiwisaver_daily`:

| Column | Type | Notes |
|---|---|---|
| `date` | `TEXT UNIQUE` | `YYYY-MM-DD`, one row per date |
| `balance` | `REAL NOT NULL` | KiwiSaver balance |
| `units` | `REAL` | Number of units (optional) |
| `unit_price` | `REAL` | Price per unit (optional) |

- Table `kiwisaver_adjustments`:

| Column | Type | Notes |
|---|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Row identifier |
| `date` | `TEXT` | Adjustment date |
| `amount` | `REAL` | Adjustment amount (can be negative) |
| `type` | `TEXT` | `employee`, `employer`, `government`, `tax`, or `fee` |

- Safe migrations add missing columns to existing KiwiSaver tables if an older version exists.
- Data persists across restarts. Excluded from git by `.gitignore`.

### Helper

`format_time_12h(stored_time)` converts `YYYY-MM-DD HH:MM` to `YYYY-MM-DD hh:MM AM/PM` for display.

## Cost and performance

The agent tracks model usage in a `kiwisaver_daily` table (wait—no, that's KiwiSaver). It tracks model calls in the `expenses` table—no, that's expenses too. Let me correct: the agent logs every model call in a SQLite table called `expenses_daily`? No.

The agent logs every model call to the `llm_calls` table in the SQLite database:

| Column | Type | Notes |
|---|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Row identifier |
| `timestamp` | `TEXT` | When the call was made (`YYYY-MM-DD HH:MM:SS`) |
| `request_id` | `TEXT` | Model request ID |
| `step` | `INTEGER` | Which step in the agent loop (1–MAX_STEPS) |
| `model` | `TEXT` | Model name used |
| `prompt_tokens` | `REAL` | Input tokens |
| `completion_tokens` | `REAL` | Output tokens |
| `latency_s` | `REAL` | Response time in seconds |
| `tool_name` | `TEXT` | Which tool was called, or NULL if none |

### Cost and performance tracking

Each model call records:

- **Tokens:** Input and output token counts from the model response
- **Latency:** Time in seconds for the model to respond
- **Cost:** Estimated USD cost using `NVIDIA_MODEL` pricing from `PRICING` in `expense_agent.py`. Models not in `PRICING` show as "unknown" cost
- **Steps:** How many model calls were needed to answer the user (each tool call is one step)

### Running the usage report

```powershell
python usage_report.py
```

Reads from the configured SQLite database (`EXPENSES_DB_PATH`) and prints:

- Per-request summary: date, model, steps, tokens, cost, average latency, slowest call
- Totals across all requests
- Warning if any model lacks pricing data

### Running the benchmark comparison

```powershell
python benchmark_compare.py MODEL_1 MODEL_2
```

Runs the same 10 prompt scenarios through two different models and prints a side-by-side comparison table:

| Column | Meaning |
|---|---|
| Model | Model name being tested |
| Steps | Number of tool calls per user message |
| Tokens | Total input + output tokens |
| Latency (s) | Response time in seconds |
| Cost (USD) | Estimated cost based on `PRICING` |
| Tools | Which tools were called |

Use this to compare different models side-by-side. No winner is declared—the data is there for you to evaluate.

### Optimization summary

The following changes were made to reduce token usage, latency, and cost:

| Change | What it does | Effect |
|---|---|---|
| `MAX_STEPS = 5` | Limits agent loop to 5 model calls per user message | Prevents runaway loops and caps token usage |
| `MAX_TOKENS = 256` | Limits output tokens per model call | Reduces output tokens and latency |
| `MAX_HISTORY_MESSAGES = 10` | Truncates conversation history to last 10 messages | Reduces input tokens on later turns |
| `MAX_TOOL_RESULT_CHARS = 2000` | Truncates long tool results | Reduces input tokens on subsequent calls |
| `delete_and_list_expenses` tool | Combines delete + list in one call | Fewer tool calls for common workflows |
| Shortened `SYSTEM_PROMPT` | ~1943 → ~927 characters (~52% shorter) | Reduces input tokens on every call |
| Shortened tool descriptions | ~687 → ~600 characters (~13% shorter) | Reduces input tokens on every call |

All monetary values are estimates based on placeholder pricing in `PRICING`. Replace with real pricing from NVIDIA for accurate cost estimates. No API keys or secrets are stored in the database—only model name, token counts, latency, and tool names are recorded.

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

### Import KiwiSaver CSV

```powershell
python import_kiwisaver.py kiwisaver.csv
```

Reads columns `Date` and `KIWI Saver` (flexible column names). Parses dates like `"Thursday, 15 October 2026"` and balances like `"21,156.25"`. Skips dates already in the database (no overwrite). Prints a data-quality report flagging days where the balance is identical to the previous day — these are **not** modified or deleted. The CSV stays out of git (`*.csv` is in `.gitignore`).

## Known limits

- Expense input has no validation beyond what the model sends (for example, negative expense amounts are not rejected). KiwiSaver tools do validate balance > 0, non-zero adjustment amounts, valid types, date formats, and future dates.
- The model is early access and may change or behave differently from stable models. Tool-calling behavior has not been fully tested.
- A reasoning model may pause to think before replying.
- The model accepts images, but this project uses text only.
