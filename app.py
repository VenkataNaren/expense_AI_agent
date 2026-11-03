"""Streamlit UI for the Expense Helper agent.

Run with: streamlit run app.py
"""
import sqlite3

import streamlit as st

# ---------- Helpers (no streamlit calls at module level so tests can import) ----------

from expense_agent import format_time_12h


def _connect():
    """Open a fresh SQLite connection for a single operation.

    Reads DB_PATH dynamically so reloading expense_agent with a new
    EXPENSES_DB_PATH (as tests do) is respected.
    """
    import expense_agent as ea

    return sqlite3.connect(ea.DB_PATH)


def load_expenses_for_table():
    """Load all expenses newest first, with 12-hour display time."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, amount, category, note, spent_at FROM expenses ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "ID": r[0],
            "Amount": r[1],
            "Category": r[2],
            "Note": r[3],
            "Date/Time": format_time_12h(r[4]) if r[4] else "",
        }
        for r in rows
    ]


def category_totals():
    """Return dict of category -> total amount."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT amount, category FROM expenses").fetchall()
    finally:
        conn.close()
    totals = {}
    for amount, category in rows:
        totals[category] = totals.get(category, 0) + amount
    return totals


def render_tool_calls(tool_calls):
    """Display tool call details inside an expander."""
    if not tool_calls:
        return
    with st.expander("Tool calls"):
        for tc in tool_calls:
            st.write(f"**{tc['name']}**")
            st.json(tc["args"])
            st.write(tc["result"])


# ---------- UI ----------

def main():
    st.set_page_config(page_title="Expense Helper")
    st.title("Expense Helper")

    if "history" not in st.session_state:
        from expense_agent import SYSTEM_PROMPT
        st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "messages" not in st.session_state:
        st.session_state.messages = []

    tab_chat, tab_expenses = st.tabs(["Chat", "Expenses"])

    # ---- Chat tab ----
    with tab_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    render_tool_calls(msg["tool_calls"])

        if prompt := st.chat_input("Ask about your expenses..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                from expense_agent import run_agent
                reply, tool_calls = run_agent(prompt, st.session_state.history)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "tool_calls": tool_calls,
                    }
                )
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    render_tool_calls(tool_calls)
            except Exception as exc:
                st.error(f"Sorry, something went wrong: {exc}")

    # ---- Expenses tab ----
    with tab_expenses:
        st.subheader("All Expenses")
        if st.button("Refresh"):
            st.rerun()

        try:
            data = load_expenses_for_table()
            if data:
                st.table(data)

                st.subheader("Total by Category")
                totals = category_totals()
                for category in sorted(totals):
                    st.write(f"**{category}**: {totals[category]:.2f}")
            else:
                st.info("No expenses recorded.")
        except Exception as exc:
            st.error(f"Could not load expenses: {exc}")


if __name__ == "__main__":
    main()
