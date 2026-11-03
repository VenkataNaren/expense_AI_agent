"""
Expense Helper: a simple single-agent project.

The agent can add expenses, list them, and total them by category.
Uses an NVIDIA-hosted model through the OpenAI-compatible API.

Setup:
    pip install openai python-dotenv
    Create a .env file next to this script (see .env.example).
"""
import json
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ---------- 1. Connect to a model (the "brain") ----------
# Settings come from a .env file in the same folder (never put keys in code).
load_dotenv()

client = OpenAI(
    base_url=os.environ["NVIDIA_BASE_URL"],
    api_key=os.environ["NVIDIA_API_KEY"],
)
MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")

MAX_STEPS = 5  # safety limit so the loop can never run forever


# ---------- 2. Our tiny "database" ----------
DB_PATH = os.environ.get("EXPENSES_DB_PATH", "expenses.db")


def _connect():
    """Open a fresh SQLite connection for a single operation."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the table and run the safe migration. Called once at import."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                note TEXT DEFAULT '',
                spent_at TEXT
            )
            """
        )

        # Safe migration: add spent_at column to existing tables that lack it,
        # and backfill old rows with today's date at 00:00.
        columns = [row[1] for row in conn.execute("PRAGMA table_info(expenses)").fetchall()]
        if "spent_at" not in columns:
            conn.execute("ALTER TABLE expenses ADD COLUMN spent_at TEXT")
            today = datetime.now().strftime("%Y-%m-%d")
            conn.execute(
                "UPDATE expenses SET spent_at = ? WHERE spent_at IS NULL OR spent_at = ''",
                (f"{today} 00:00",),
            )
            conn.commit()
    finally:
        conn.close()


# Run once when the module is imported.
init_db()

# ---------- 3. Tools (plain Python functions the agent may call) ----------
def format_time_12h(stored_time):
    """Convert stored 'YYYY-MM-DD HH:MM' to 'YYYY-MM-DD hh:MM AM/PM'."""
    if not stored_time:
        return ""
    try:
        dt = datetime.strptime(stored_time, "%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%d %I:%M %p")
    except ValueError:
        return stored_time


def _parse_spent_at(value):
    """Parse accepted date/time formats and return (datetime, error_string)."""
    if value is None or value == "":
        return datetime.now(), None

    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %I:%M %p", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt), None
        except ValueError:
            continue

    return None, (
        f"Error: Invalid date/time '{value}'. "
        "Use 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD hh:MM AM/PM', or 'YYYY-MM-DD'."
    )


def add_expense(amount, category, note="", spent_at=None, confirm=None):
    """Record a new expense, with confirmation guardrail for future timestamps."""
    dt, error = _parse_spent_at(spent_at)
    if error:
        return error

    confirm = (confirm or "").strip().lower()
    now = datetime.now()

    if dt > now:
        details = {
            "amount": float(amount),
            "category": category.lower(),
            "note": note,
            "spent_at": dt.strftime("%Y-%m-%d %H:%M"),
            "display_amount": amount,
            "display_category": category,
        }

        if confirm in ("y", "yes"):
            _PENDING_ADD = {"details": None}
            return _perform_add(details)

        if confirm in ("n", "no"):
            _PENDING_ADD = {"details": None}
            return "Expense addition cancelled. No changes were made."

        _PENDING_ADD = {"details": details}
        return (
            f"Found a future-dated expense to add:\n"
            f"{details['display_amount']} - {details['category']} - {details['note']} - "
            f"{format_time_12h(details['spent_at'])}\n\n"
            "The date/time is in the future. Are you sure you want to add this record? (y/n)"
        )

    details = {
        "amount": float(amount),
        "category": category.lower(),
        "note": note,
        "spent_at": dt.strftime("%Y-%m-%d %H:%M"),
        "display_amount": amount,
        "display_category": category,
    }
    return _perform_add(details)


def _perform_add(details):
    """Insert an expense row into the database."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO expenses (amount, category, note, spent_at) VALUES (?, ?, ?, ?)",
            (details["amount"], details["category"], details["note"], details["spent_at"]),
        )
        conn.commit()
    finally:
        conn.close()
    return f"Added {details['display_amount']} to {details['display_category']}."


def list_expenses():
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT amount, category, note, spent_at FROM expenses ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No expenses recorded."
    lines = [
        f"{amount} - {category} - {note} - {format_time_12h(spent_at)}"
        for amount, category, note, spent_at in rows
    ]
    return "\n".join(lines)


def total_by_category(category):
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE category = ?",
            (category.lower(),),
        ).fetchone()
    finally:
        conn.close()
    total = row[0]
    return f"Total for {category}: {total}"


def delete_expense(id=None, note=None, amount=None, spent_at=None):
    """Delete expense(s) matching all provided criteria (id, note, amount, spent_at)."""
    conditions = []
    params = []
    if id is not None:
        conditions.append("id = ?")
        params.append(int(id))
    if note is not None:
        conditions.append("note = ?")
        params.append(note)
    if amount is not None:
        conditions.append("amount = ?")
        params.append(float(amount))
    if spent_at is not None:
        dt, error = _parse_spent_at(spent_at)
        if error:
            return error
        conditions.append("spent_at = ?")
        params.append(dt.strftime("%Y-%m-%d %H:%M"))

    if not conditions:
        return (
            "Error: Please specify at least one of id, note, amount, or spent_at "
            "to identify the expense to delete."
        )

    where = " AND ".join(conditions)
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT id, amount, category, note, spent_at FROM expenses WHERE {where}",
            params,
        ).fetchall()
        if not rows:
            return "No matching expense found."
        conn.execute(f"DELETE FROM expenses WHERE {where}", params)
        conn.commit()
    finally:
        conn.close()

    deleted = [
        f"id {r[0]}: {r[1]} - {r[2]} - {r[3]} - {format_time_12h(r[4])}" for r in rows
    ]
    return "Deleted:\n" + "\n".join(deleted)


# Pending deletion state for the y/n confirmation guardrail.
# Stores the matching rows until the user confirms with "y"/"yes".
_PENDING_DELETE = {"criteria": None, "rows": None}

# Pending add state for future-dated expenses requiring confirmation.
# Stores the expense details until the user confirms with "y"/"yes".
_PENDING_ADD = {"details": None}


def delete_expense(id=None, note=None, amount=None, spent_at=None, confirm=None):
    """Delete expense(s) matching all provided criteria, with y/n confirmation guardrail.

    Phase 1: Call without confirm (or confirm != "y"/"yes") to preview matches and
    request confirmation. No rows are deleted.
    Phase 2: Call with confirm="y" to actually delete the pending matches.
    Call with confirm="n" to cancel.
    """
    confirm = (confirm or "").strip().lower()

    conditions = []
    params = []
    if id is not None:
        conditions.append("id = ?")
        params.append(int(id))
    if note is not None:
        conditions.append("note = ?")
        params.append(note)
    if amount is not None:
        conditions.append("amount = ?")
        params.append(float(amount))
    if spent_at is not None:
        dt, error = _parse_spent_at(spent_at)
        if error:
            return error
        conditions.append("spent_at = ?")
        params.append(dt.strftime("%Y-%m-%d %H:%M"))

    if not conditions:
        return (
            "Error: Please specify at least one of id, note, amount, or spent_at "
            "to identify the expense to delete."
        )

    where = " AND ".join(conditions)
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT id, amount, category, note, spent_at FROM expenses WHERE {where}",
            params,
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No matching expense found."

    details = [
        f"ID {r[0]}: {r[1]} - {r[2]} - {r[3]} - {format_time_12h(r[4])}" for r in rows
    ]
    preview = "\n".join(details)

    if confirm in ("y", "yes"):
        conn = _connect()
        try:
            conn.execute(f"DELETE FROM expenses WHERE {where}", params)
            conn.commit()
        finally:
            conn.close()
        _PENDING_DELETE = {"criteria": None, "rows": None}
        return f"Deleted {len(rows)} record(s):\n{preview}"

    if confirm in ("n", "no"):
        _PENDING_DELETE = {"criteria": None, "rows": None}
        return "Deletion cancelled. No changes were made."

    # Pending confirmation: store state, do NOT delete.
    _PENDING_DELETE = {"criteria": (id, note, amount, spent_at), "rows": rows}
    return (
        f"Found {len(rows)} matching expense(s) to delete:\n{preview}\n\n"
        "Are you sure you want to delete this record? (y/n)"
    )


TOOL_FUNCTIONS = {
    "add_expense": add_expense,
    "list_expenses": list_expenses,
    "total_by_category": total_by_category,
    "delete_expense": delete_expense,
}

# The "menu" the model sees: name + description + inputs for each tool
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_expense",
            "description": "Record a new expense.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount spent"},
                    "category": {"type": "string", "description": "e.g. food, travel"},
                    "note": {"type": "string", "description": "Optional detail"},
                    "spent_at": {
                        "type": "string",
                        "description": (
                            "When the expense happened. Accepts 'YYYY-MM-DD HH:MM' "
                            "(24-hour), 'YYYY-MM-DD hh:MM AM/PM' (12-hour), or "
                            "'YYYY-MM-DD' (date only = midnight). Defaults to now."
                        ),
                    },
                    "confirm": {
                        "type": "string",
                        "description": (
                            'Confirmation flag. Set to "y" or "yes" to actually add a '
                            "future-dated expense after previewing it; 'n' or 'no' to cancel. "
                            "Omit to preview a future-dated expense and request confirmation."
                        ),
                    },
                },
                "required": ["amount", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_expenses",
            "description": "Show all recorded expenses.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "total_by_category",
            "description": "Add up all expenses in one category.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_expense",
            "description": "Delete an expense record by ID, note (description), amount, and/or timestamp. Provide at least one criterion; all provided criteria must match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The expense ID (shown in the expense list)",
                    },
                    "note": {
                        "type": "string",
                        "description": "Exact description/note of the expense",
                    },
                    "amount": {"type": "number", "description": "Amount spent"},
                    "spent_at": {
                        "type": "string",
                        "description": (
                            "When the expense happened. Accepts 'YYYY-MM-DD HH:MM' "
                            "(24-hour), 'YYYY-MM-DD hh:MM AM/PM' (12-hour), or "
                            "'YYYY-MM-DD' (date only = midnight)."
                        ),
                    },
                    "confirm": {
                        "type": "string",
                        "description": (
                            'Confirmation flag. Set to "y" or "yes" to actually perform '
                            'the deletion after previewing matches; "n" or "no" to cancel. '
                            "Omit to preview matches and request confirmation."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an expense helper. Use the tools to record and report expenses. "
    "Never invent numbers; if you need data, call a tool. "
    "A current local time is injected at the start of each user message. "
    "If the user mentions relative times like 'yesterday', 'this morning', "
    "'at 3pm', 'noon', or 'midnight', convert them to an exact date and time "
    "before calling add_expense: treat 'noon' as 12:00 PM and 'midnight' as "
    "12:00 AM. If the time is unclear, ask the user instead of guessing. "
    "If a user wants to correct or remove an expense (duplicate, wrong amount, wrong "
    "timestamp), call delete_expense WITHOUT the confirm argument first to preview "
    "the matching record(s) and request confirmation. Present the record details to "
    "the user and ask: 'Are you sure you want to delete this record? (y/n)'. "
    "Wait for the user's reply: if they say 'y'/'yes', call delete_expense again "
    'with confirm="y" to perform the deletion; if they say "n"/"no", call delete_expense '
    'with confirm="n" to cancel. Then, if deleting, use add_expense to add the corrected '
    "record with the right details and timestamp. Confirm to the user what was deleted "
    "and what was corrected. "
    "If add_expense is called with a future-dated timestamp, the tool will preview the "
    "expense and ask: 'Are you sure you want to add this record? (y/n)'. Wait for the "
    "user's reply: if they say 'y'/'yes', call add_expense again with confirm='y' to "
    "actually record it; if they say 'n'/'no', call add_expense with confirm='n' to cancel."
)

# ---------- 4. The agent loop ----------
def run_agent(user_message, history):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %I:%M %p (%A)")
    history.append({"role": "system", "content": f"Current local time: {current_time_str}"})
    history.append({"role": "user", "content": user_message})

    tool_calls_made = []

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=history, tools=TOOLS
        )
        message = response.choices[0].message
        history.append(message)

        # No tool requested -> the agent is done, return its answer
        if not message.tool_calls:
            return message.content, tool_calls_made

        # Otherwise run each requested tool and feed the result back
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as error:  # never crash on a bad tool call
                result = f"Error: {error}"
            print(f"  [tool] {name} -> {result}")  # simple logging
            tool_calls_made.append(
                {"name": name, "args": args, "result": str(result)}
            )
            history.append(
                {"role": "tool", "tool_call_id": call.id, "content": str(result)}
            )

    return "Stopped: too many steps.", tool_calls_made

# ---------- 5. Chat with it ----------
if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Expense Helper ready. Type 'quit' to exit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        reply, _ = run_agent(text, history)
        print("Agent:", reply)
