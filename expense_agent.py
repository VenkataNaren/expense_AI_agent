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
expenses = []


# ---------- 3. Tools (plain Python functions the agent may call) ----------
def add_expense(amount, category, note=""):
    expenses.append(
        {"amount": float(amount), "category": category.lower(), "note": note}
    )
    return f"Added {amount} to {category}."

def list_expenses():
    return json.dumps(expenses) if expenses else "No expenses recorded."

def total_by_category(category):
    total = sum(e["amount"] for e in expenses if e["category"] == category.lower())
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
    "Never invent numbers; if you need data, call a tool."
)

# ---------- 4. The agent loop ----------
def run_agent(user_message, history):
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
 