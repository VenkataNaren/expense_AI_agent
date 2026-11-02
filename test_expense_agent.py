"""Tests for expense_agent.py"""

import importlib
import os
import sqlite3
from datetime import datetime
from unittest import mock

import pytest


def _reload_with_env(env, db_path=None):
    """Patch env + source modules, reload expense_agent, return it."""
    env = dict(env)
    if db_path is not None:
        env["EXPENSES_DB_PATH"] = str(db_path)
    with mock.patch.dict(os.environ, env, clear=True), mock.patch(
        "dotenv.load_dotenv"
    ), mock.patch("openai.OpenAI") as mock_openai:
        import expense_agent

        importlib.reload(expense_agent)
        return expense_agent, mock_openai


def test_client_requires_env_vars():
    """Client creation should fail when required env vars are missing."""
    with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
        "dotenv.load_dotenv"
    ):
        with pytest.raises(KeyError):
            import expense_agent
            importlib.reload(expense_agent)


def test_client_instantiated_with_env(tmp_path):
    """Client should be created using env vars."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )

    mock_openai.assert_called_with(
        base_url="https://test.example.com/v1",
        api_key="test-api-key",
    )
    assert expense_agent.MODEL == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    expense_agent.conn.close()


def test_model_override_from_env(tmp_path):
    """NVIDIA_MODEL env var should override the default model."""
    db_path = tmp_path / "expenses.db"
    expense_agent, _ = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
            "NVIDIA_MODEL": "custom/model",
        },
        db_path=db_path,
    )
    assert expense_agent.MODEL == "custom/model"
    expense_agent.conn.close()


@pytest.fixture
def agent(tmp_path):
    """Fresh agent with a temp database; closes connection after test."""
    db_path = tmp_path / "expenses.db"
    expense_agent, _ = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )
    yield expense_agent
    expense_agent.conn.close()


def _get_expenses(expense_agent):
    """Fetch all rows as a list of dicts (amount, category, note, spent_at)."""
    rows = expense_agent.conn.execute(
        "SELECT amount, category, note, spent_at FROM expenses ORDER BY id"
    ).fetchall()
    return [
        {
            "amount": amount,
            "category": category,
            "note": note,
            "spent_at": spent_at,
        }
        for amount, category, note, spent_at in rows
    ]


def _strip_spent_at(expenses):
    """Remove spent_at from each dict for exact-match assertions."""
    return [
        {k: v for k, v in e.items() if k != "spent_at"} for e in expenses
    ]


def test_add_expense_records_amount_category_note(agent):
    """add_expense should insert a row with amount, category, and note."""
    result = agent.add_expense("10.50", "Food", "lunch")
    assert result == "Added 10.50 to Food."
    expenses = _get_expenses(agent)
    assert _strip_spent_at(expenses) == [
        {"amount": 10.5, "category": "food", "note": "lunch"}
    ]
    assert expenses[0]["spent_at"] is not None
    # Verify stored format is YYYY-MM-DD HH:MM
    datetime.strptime(expenses[0]["spent_at"], "%Y-%m-%d %H:%M")


def test_add_expense_default_note(agent):
    """note should default to an empty string when not provided."""
    agent.add_expense(5, "Transport")
    expenses = _get_expenses(agent)
    assert _strip_spent_at(expenses) == [
        {"amount": 5.0, "category": "transport", "note": ""}
    ]
    assert expenses[0]["spent_at"] is not None


def test_add_expense_lowercases_category(agent):
    """Categories should be stored lowercase regardless of input case."""
    agent.add_expense(3, "FOOD")
    agent.add_expense(2, "Food")
    assert _strip_spent_at(_get_expenses(agent)) == [
        {"amount": 3.0, "category": "food", "note": ""},
        {"amount": 2.0, "category": "food", "note": ""},
    ]


def test_add_expense_default_spent_at_is_current(agent, monkeypatch):
    """When spent_at is omitted, it defaults to the current date/time."""
    fixed = datetime(2026, 11, 3, 14, 30)
    import expense_agent as ea_mod
    monkeypatch.setattr(ea_mod, "datetime", type("DummyDT", (), {"now": staticmethod(lambda: fixed)}))

    result = agent.add_expense("7", "Coffee")
    assert "Error" not in result
    expenses = _get_expenses(agent)
    assert expenses[0]["spent_at"] == "2026-11-03 14:30"


def test_add_expense_24_hour_input_stored_correctly(agent):
    """'YYYY-MM-DD HH:MM' is stored as-is in 24-hour format."""
    result = agent.add_expense("12", "Food", "dinner", "2026-11-03 18:45")
    assert "Error" not in result
    expenses = _get_expenses(agent)
    assert expenses[-1]["spent_at"] == "2026-11-03 18:45"


def test_add_expense_12_hour_input_stored_as_24_hour(agent):
    """'YYYY-MM-DD hh:MM AM/PM' is converted and stored in 24-hour format."""
    result = agent.add_expense("8", "Food", "snack", "2026-11-03 03:30 PM")
    assert "Error" not in result
    expenses = _get_expenses(agent)
    assert expenses[-1]["spent_at"] == "2026-11-03 15:30"


def test_add_expense_12am_stores_as_00_00(agent):
    """12:00 AM stores as 00:00."""
    result = agent.add_expense("1", "Food", "late", "2026-11-03 12:00 AM")
    assert "Error" not in result
    expenses = _get_expenses(agent)
    assert expenses[-1]["spent_at"] == "2026-11-03 00:00"


def test_add_expense_12pm_stores_as_12_00(agent):
    """12:00 PM stores as 12:00 (not 00:00)."""
    result = agent.add_expense("2", "Food", "noon", "2026-11-03 12:00 PM")
    assert "Error" not in result
    expenses = _get_expenses(agent)
    assert expenses[-1]["spent_at"] == "2026-11-03 12:00"


def test_add_expense_date_only_becomes_00_00(agent):
    """Just 'YYYY-MM-DD' is stored as that date at 00:00."""
    result = agent.add_expense("3", "Transport", "morning", "2026-11-03")
    assert "Error" not in result
    expenses = _get_expenses(agent)
    assert expenses[-1]["spent_at"] == "2026-11-03 00:00"


def test_add_expense_invalid_value_returns_error(agent):
    """Invalid spent_at returns an error string instead of crashing."""
    result = agent.add_expense("5", "Food", "bad", "not-a-date")
    assert result.startswith("Error:")

    result2 = agent.add_expense("5", "Food", "bad2", "2026-13-01 10:00")
    assert result2.startswith("Error:")


def test_format_time_12h_shows_ampm(agent):
    """format_time_12h converts 24-hour storage to 12-hour AM/PM display."""
    assert agent.format_time_12h("2026-11-03 14:30") == "2026-11-03 02:30 PM"
    assert agent.format_time_12h("2026-11-03 00:00") == "2026-11-03 12:00 AM"
    assert agent.format_time_12h("2026-11-03 12:00") == "2026-11-03 12:00 PM"
    assert agent.format_time_12h("2026-11-03 09:05") == "2026-11-03 09:05 AM"
    assert agent.format_time_12h("") == ""


def test_list_expenses_includes_12_hour_time(agent):
    """list_expenses shows each expense with its 12-hour display time."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    agent.add_expense(5, "Transport", "", "2026-11-03 08:15")
    result = agent.list_expenses()
    assert "02:30 PM" in result
    assert "08:15 AM" in result


def test_migration_adds_spent_at_column(tmp_path):
    """Migration adds spent_at to a pre-existing table without it and backfills."""
    db_path = tmp_path / "expenses.db"
    # Create a DB manually WITHOUT spent_at column
    import sqlite3 as _sqlite3
    raw = _sqlite3.connect(str(db_path))
    raw.execute(
        """
        CREATE TABLE expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            note TEXT DEFAULT ''
        )
        """
    )
    raw.execute("INSERT INTO expenses (amount, category, note) VALUES (10, 'Food', 'x')")
    raw.commit()
    raw.close()

    # Now reload the agent; migration should run
    expense_agent, _ = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )
    columns = [row[1] for row in expense_agent.conn.execute("PRAGMA table_info(expenses)").fetchall()]
    assert "spent_at" in columns
    rows = expense_agent.conn.execute("SELECT spent_at FROM expenses").fetchall()
    assert rows[0][0] is not None
    assert rows[0][0].endswith(" 00:00")
    expense_agent.conn.close()


def test_list_expenses_returns_all(agent):
    """list_expenses should return every recorded expense."""
    agent.add_expense("10.50", "Food", "lunch")
    agent.add_expense(5, "Transport")
    result = agent.list_expenses()
    assert "10.5" in result
    assert "food" in result
    assert "lunch" in result
    assert "5.0" in result
    assert "transport" in result


def test_list_expenses_empty(agent):
    """list_expenses should say no expenses when the table is empty."""
    assert agent.list_expenses() == "No expenses recorded."


def test_total_by_category_sums(agent):
    """total_by_category should add up amounts for one category."""
    agent.add_expense(10, "Food")
    agent.add_expense(20, "Food")
    agent.add_expense(5, "Transport")
    assert agent.total_by_category("Food") == "Total for Food: 30.0"
    assert agent.total_by_category("food") == "Total for food: 30.0"
    assert agent.total_by_category("Transport") == "Total for Transport: 5.0"


def test_total_by_category_unknown(agent):
    """total_by_category should return zero for a category with no expenses."""
    assert agent.total_by_category("Unknown") == "Total for Unknown: 0"
