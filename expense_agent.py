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

conn = sqlite3.connect(DB_PATH)
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


def add_expense(amount, category, note="", spent_at=None):
    dt, error = _parse_spent_at(spent_at)
    if error:
        return error

    stored = dt.strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO expenses (amount, category, note, spent_at) VALUES (?, ?, ?, ?)",
        (float(amount), category.lower(), note, stored),
    )
    conn.commit()
    return f"Added {amount} to {category}."


def list_expenses():
    rows = conn.execute(
        "SELECT amount, category, note, spent_at FROM expenses ORDER BY id"
    ).fetchall()
    if not rows:
        return "No expenses recorded."
    lines = [
        f"{amount} - {category} - {note} - {format_time_12h(spent_at)}"
        for amount, category, note, spent_at in rows
    ]
    return "\n".join(lines)


def total_by_category(category):
    row = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE category = ?",
        (category.lower(),),
    ).fetchone()
    total = row[0]
    return f"Total for {category}: {total}"

TOOL_FUNCTIONS = {
    "add_expense": add_expense,
    "list_expenses": list_expenses,
    "total_by_category": total_by_category,
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
]

SYSTEM_PROMPT = (
    "You are an expense helper. Use the tools to record and report expenses. "
    "Never invent numbers; if you need data, call a tool. "
    "A current local time is injected at the start of each user message. "
    "If the user mentions relative times like 'yesterday', 'this morning', "
    "'at 3pm', 'noon', or 'midnight', convert them to an exact date and time "
    "before calling add_expense: treat 'noon' as 12:00 PM and 'midnight' as "
    "12:00 AM. If the time is unclear, ask the user instead of guessing."
)

# ---------- 4. The agent loop ----------
def run_agent(user_message, history):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %I:%M %p (%A)")
    history.append({"role": "system", "content": f"Current local time: {current_time_str}"})
    history.append({"role": "user", "content": user_message})

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=history, tools=TOOLS
        )
        message = response.choices[0].message
        history.append(message)

        # No tool requested -> the agent is done, return its answer
        if not message.tool_calls:
            return message.content

        # Otherwise run each requested tool and feed the result back
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as error:  # never crash on a bad tool call
                result = f"Error: {error}"
            print(f"  [tool] {name} -> {result}")  # simple logging
            history.append(
                {"role": "tool", "tool_call_id": call.id, "content": str(result)}
            )

    return "Stopped: too many steps."

# ---------- 5. Chat with it ----------
if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Expense Helper ready. Type 'quit' to exit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        print("Agent:", run_agent(text, history))
 