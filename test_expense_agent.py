"""Tests for expense_agent.py"""

import importlib
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
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


@pytest.fixture
def agent(tmp_path):
    """Fresh agent with a temp database; connections are per-operation, so nothing to close."""
    db_path = tmp_path / "expenses.db"
    expense_agent, _ = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )
    yield expense_agent


def _get_expenses(expense_agent):
    """Fetch all rows as a list of dicts (amount, category, note, spent_at)."""
    import sqlite3

    conn = sqlite3.connect(expense_agent.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT amount, category, note, spent_at FROM expenses ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
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


def test_add_expense_blocks_future_timestamp(agent, monkeypatch):
    """add_expense should preview (not insert) a future-dated expense and request confirmation."""
    fixed_now = datetime(2026, 11, 3, 12, 00)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    # Phase 1: preview only, no insertion
    result = agent.add_expense("10", "Food", "lunch", "2026-11-03 14:30")
    assert "Are you sure you want to add this record? (y/n)" in result
    assert "future" in result.lower()
    assert len(_get_expenses(agent)) == 0  # nothing inserted yet
    # Phase 2: confirm insertion
    result = agent.add_expense("10", "Food", "lunch", "2026-11-03 14:30", confirm="y")
    assert "Added" in result
    assert len(_get_expenses(agent)) == 1


def test_add_expense_cancel_future_timestamp(agent, monkeypatch):
    """add_expense should cancel a future-dated expense when user replies 'n'."""
    fixed_now = datetime(2026, 11, 3, 12, 00)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    agent.add_expense("10", "Food", "lunch", "2026-11-03 14:30")  # preview
    result = agent.add_expense("10", "Food", "lunch", "2026-11-03 14:30", confirm="n")
    assert result == "Expense addition cancelled. No changes were made."
    assert len(_get_expenses(agent)) == 0


def test_add_expense_allows_past_timestamp(agent, monkeypatch):
    """add_expense should accept a spent_at that is in the past."""
    fixed_now = datetime(2026, 11, 3, 14, 30)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    result = agent.add_expense("10", "Food", "lunch", "2026-11-03 12:00")
    assert "Added" in result
    assert len(_get_expenses(agent)) == 1


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
    import sqlite3

    conn = sqlite3.connect(expense_agent.DB_PATH)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(expenses)").fetchall()]
        assert "spent_at" in columns
        rows = conn.execute("SELECT spent_at FROM expenses").fetchall()
        assert rows[0][0] is not None
        assert rows[0][0].endswith(" 00:00")
    finally:
        conn.close()


def test_kiwisaver_tables_created(agent):
    """init_db should create the kiwisaver_daily and kiwisaver_adjustments tables."""
    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert "kiwisaver_daily" in tables
    assert "kiwisaver_adjustments" in tables


def test_kiwisaver_daily_unique_date(agent):
    """kiwisaver_daily should reject a duplicate date (UNIQUE constraint)."""
    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        conn.execute(
            "INSERT INTO kiwisaver_daily (date, balance) VALUES (?, ?)",
            ("2026-11-03", 1000.0),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO kiwisaver_daily (date, balance) VALUES (?, ?)",
                ("2026-11-03", 2000.0),
            )
    finally:
        conn.close()


def test_add_kiwisaver_daily_records_balance(agent):
    """add_kiwisaver_daily should insert a row and INSERT OR REPLACE on duplicate date (with confirm)."""
    result = agent.add_kiwisaver_daily("2026-11-03", 1000.0, 50.0, 20.0)
    assert result == "Kiwisaver daily balance for 2026-11-03: 1000.0."

    # Duplicate date without confirm previews, does not overwrite.
    result = agent.add_kiwisaver_daily("2026-11-03", 1500.0, 75.0, 20.0)
    assert "Are you sure you want to overwrite? (y/n)" in result

    # Confirm overwrite.
    result = agent.add_kiwisaver_daily("2026-11-03", 1500.0, 75.0, 20.0, confirm="y")
    assert "overwrote" in result
    assert "1500.0" in result

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT date, balance, units, unit_price FROM kiwisaver_daily ORDER BY date"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0] == ("2026-11-03", 1500.0, 75.0, 20.0)


def test_list_kiwisaver_daily_empty(agent):
    """list_kiwisaver_daily should report no rows when empty."""
    assert agent.list_kiwisaver_daily() == "No Kiwisaver daily balances recorded."


def test_list_kiwisaver_daily_returns_all(agent):
    """list_kiwisaver_daily should return every recorded row."""
    agent.add_kiwisaver_daily("2026-11-03", 1000.0)
    agent.add_kiwisaver_daily("2026-11-04", 1100.0)
    result = agent.list_kiwisaver_daily()
    assert "2026-11-03" in result
    assert "2026-11-04" in result
    assert "1000.0" in result
    assert "1100.0" in result


def test_add_kiwisaver_adjustment_records(agent):
    """add_kiwisaver_adjustment should insert a row for a valid type."""
    result = agent.add_kiwisaver_adjustment("2026-11-03", 100.0, "employer")
    assert "employer" in result
    assert "100.0" in result

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT date, amount, type FROM kiwisaver_adjustments ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0] == ("2026-11-03", 100.0, "employer")


def test_add_kiwisaver_adjustment_invalid_type(agent):
    """add_kiwisaver_adjustment should reject an invalid type."""
    result = agent.add_kiwisaver_adjustment("2026-11-03", 50.0, "bonus")
    assert result.startswith("Error:")

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM kiwisaver_adjustments").fetchone()
    finally:
        conn.close()
    assert rows[0] == 0


def test_add_kiwisaver_adjustment_rejects_zero_amount(agent):
    """add_kiwisaver_adjustment should reject amount = 0."""
    result = agent.add_kiwisaver_adjustment("2026-11-03", 0, "employee")
    assert result.startswith("Error:")


def test_add_kiwisaver_adjustment_allows_negative_amount(agent):
    """add_kiwisaver_adjustment should allow negative amounts."""
    result = agent.add_kiwisaver_adjustment("2026-11-03", -50.0, "fee")
    assert "fee" in result
    assert "-50.0" in result


def test_add_kiwisaver_adjustment_records_valid(agent):
    """add_kiwisaver_adjustment should record valid adjustments for all allowed types."""
    for t in ("employee", "employer", "government", "tax", "fee"):
        result = agent.add_kiwisaver_adjustment("2026-11-03", 100.0, t)
        assert t in result

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT type FROM kiwisaver_adjustments ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == ["employee", "employer", "government", "tax", "fee"]


def test_add_kiwisaver_daily_rejects_zero_balance(agent):
    """add_kiwisaver_daily should reject balance <= 0."""
    result = agent.add_kiwisaver_daily("2026-11-03", 0)
    assert result.startswith("Error:")

    result = agent.add_kiwisaver_daily("2026-11-03", -50)
    assert result.startswith("Error:")


def test_add_kiwisaver_daily_rejects_future_date(agent, monkeypatch):
    """add_kiwisaver_daily should reject future dates."""
    fixed_now = datetime(2026, 11, 3, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    result = agent.add_kiwisaver_daily("2026-11-04", 1000.0)
    assert result.startswith("Error:")
    assert "future" in result.lower()


def test_add_kiwisaver_daily_rejects_bad_date_format(agent):
    """add_kiwisaver_daily should reject bad date formats."""
    result = agent.add_kiwisaver_daily("not-a-date", 1000.0)
    assert result.startswith("Error:")

    result = agent.add_kiwisaver_daily("2026-13-01", 1000.0)
    assert result.startswith("Error:")


def test_add_kiwisaver_daily_units_unit_price_warning(agent, monkeypatch):
    """add_kiwisaver_daily should warn when units*unit_price differs from balance by >1%."""
    fixed_now = datetime(2026, 11, 3, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    # units * unit_price = 100 * 10 = 1000, balance = 1000, within 1% -> no warning
    result = agent.add_kiwisaver_daily("2026-11-01", 1000.0, units=100.0, unit_price=10.0)
    assert "Warning" not in result

    # units * unit_price = 100 * 12 = 1200, balance = 1000, 20% diff -> warning
    result = agent.add_kiwisaver_daily("2026-11-02", 1000.0, units=100.0, unit_price=12.0)
    assert "Warning" in result
    assert "more than 1%" in result


def test_add_kiwisaver_daily_overwrite_confirm_flow(agent, monkeypatch):
    """add_kiwisaver_daily should preview overwrite and require confirmation."""
    fixed_now = datetime(2026, 11, 5, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    # Insert initial
    agent.add_kiwisaver_daily("2026-11-03", 1000.0)

    # Phase 1: preview overwrite, no confirm
    result = agent.add_kiwisaver_daily("2026-11-03", 2000.0)
    assert "Are you sure you want to overwrite? (y/n)" in result
    assert "1000.0" in result
    assert "2000.0" in result

    # Verify nothing changed yet
    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        row = conn.execute(
            "SELECT balance FROM kiwisaver_daily WHERE date = ?", ("2026-11-03",)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1000.0

    # Phase 2: confirm overwrite
    result = agent.add_kiwisaver_daily("2026-11-03", 2000.0, confirm="y")
    assert "overwrote" in result
    assert "2000.0" in result

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        row = conn.execute(
            "SELECT balance FROM kiwisaver_daily WHERE date = ?", ("2026-11-03",)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 2000.0


def test_add_kiwisaver_daily_overwrite_cancel(agent, monkeypatch):
    """add_kiwisaver_daily should cancel overwrite on confirm='n'."""
    fixed_now = datetime(2026, 11, 5, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    agent.add_kiwisaver_daily("2026-11-03", 1000.0)
    agent.add_kiwisaver_daily("2026-11-03", 2000.0)  # preview
    result = agent.add_kiwisaver_daily("2026-11-03", 2000.0, confirm="n")
    assert result == "Kiwisaver daily balance update cancelled. No changes were made."

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        row = conn.execute(
            "SELECT balance FROM kiwisaver_daily WHERE date = ?", ("2026-11-03",)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1000.0


def test_add_kiwisaver_daily_overwrite_with_warnings_and_confirm(agent, monkeypatch):
    """add_kiwisaver_daily overwrite preview includes warnings, and confirm='y' applies previewed values."""
    fixed_now = datetime(2026, 11, 5, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    # Day 1 balance 1000, day 2 balance 1000 (stale, 0% change)
    agent.add_kiwisaver_daily("2026-11-03", 1000.0)
    agent.add_kiwisaver_daily("2026-11-04", 1000.0)

    # Preview overwrite of day 4 with units*unit_price that differs >1% from balance
    result = agent.add_kiwisaver_daily("2026-11-04", 1000.0, units=100.0, unit_price=12.0)
    assert "Are you sure you want to overwrite? (y/n)" in result
    assert "Warning" in result
    assert "more than 1%" in result

    # Nothing changed yet
    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        row = conn.execute(
            "SELECT balance, units, unit_price FROM kiwisaver_daily WHERE date = ?",
            ("2026-11-04",),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1000.0  # balance still 1000

    # Confirm applies previewed values (1000, 100, 12) and includes warning
    result = agent.add_kiwisaver_daily("2026-11-04", 1000.0, units=100.0, unit_price=12.0, confirm="y")
    assert "Warning" in result
    assert "more than 1%" in result
    assert "overwrote 1000.0" in result

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        row = conn.execute(
            "SELECT balance, units, unit_price FROM kiwisaver_daily WHERE date = ?",
            ("2026-11-04",),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1000.0
    assert row[1] == 100.0
    assert row[2] == 12.0


def test_add_kiwisaver_daily_confirm_ignores_changed_args(agent, monkeypatch):
    """add_kiwisaver_daily confirm='y' must apply previewed args, not changed args."""
    fixed_now = datetime(2026, 11, 5, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    agent.add_kiwisaver_daily("2026-11-03", 1000.0)

    # Preview: change balance to 1500
    agent.add_kiwisaver_daily("2026-11-03", 1500.0)

    # Confirm with a DIFFERENT balance (2000) — should still apply 1500 from preview
    result = agent.add_kiwisaver_daily("2026-11-03", 2000.0, confirm="y")
    assert "overwrote 1000.0" in result
    assert "1500.0" in result

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        row = conn.execute(
            "SELECT balance FROM kiwisaver_daily WHERE date = ?", ("2026-11-03",)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1500.0  # previewed value applied, not 2000


def test_add_kiwisaver_daily_confirm_without_pending(agent, monkeypatch):
    """add_kiwisaver_daily confirm='y' with no prior preview returns an error."""
    fixed_now = datetime(2026, 11, 5, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    agent.add_kiwisaver_daily("2026-11-03", 1000.0)

    # Confirm without any pending preview
    result = agent.add_kiwisaver_daily("2026-11-03", 2000.0, confirm="y")
    assert result.startswith("Error:")
    assert "pending" in result.lower()


def test_delete_expense_confirm_uses_stored_criteria(agent):
    """delete_expense confirm='y' must apply previewed criteria, not changed args."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    agent.add_expense(5.00, "Food", "dinner", "2026-11-03 19:00")

    # Preview: delete note="lunch"
    result = agent.delete_expense(note="lunch")
    assert "Are you sure you want to delete this record? (y/n)" in result

    # Confirm with a DIFFERENT note — should still delete the previewed "lunch" row
    result = agent.delete_expense(note="dinner", confirm="y")
    assert "Deleted 1 record(s)" in result

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        rows = conn.execute("SELECT note FROM expenses ORDER BY id").fetchall()
    finally:
        conn.close()
    assert rows == [("dinner",)]  # "lunch" was deleted, "dinner" remains


def test_delete_expense_confirm_without_pending(agent):
    """delete_expense confirm='y' without prior preview still deletes immediately (backward compatible)."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")

    # Confirm without any prior preview — should delete immediately
    result = agent.delete_expense(note="lunch", confirm="y")
    assert "Deleted 1 record(s)" in result

    import sqlite3

    conn = sqlite3.connect(agent.DB_PATH)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM expenses").fetchone()
    finally:
        conn.close()
    assert rows[0] == 0


def test_add_kiwisaver_daily_warns_5pct_change(agent, monkeypatch):
    """add_kiwisaver_daily should warn when balance changes >5% from previous day."""
    fixed_now = datetime(2026, 11, 5, 12, 0)
    import expense_agent as ea_mod

    class DummyDT:
        now = staticmethod(lambda: fixed_now)
        strptime = staticmethod(datetime.strptime)
        strftime = staticmethod(datetime.strftime)

    monkeypatch.setattr(ea_mod, "datetime", DummyDT)

    # Insert previous day balance
    agent.add_kiwisaver_daily("2026-11-03", 1000.0)

    # New day with 10% increase -> warning
    result = agent.add_kiwisaver_daily("2026-11-04", 1100.0)
    assert "Warning" in result
    assert "more than 5%" in result

    # New day with ~4% change -> no warning
    result = agent.add_kiwisaver_daily("2026-11-05", 1055.0)
    assert "Warning" not in result


def test_list_kiwisaver_adjustments_empty(agent):
    """list_kiwisaver_adjustments should report no rows when empty."""
    assert agent.list_kiwisaver_adjustments() == "No Kiwisaver adjustments recorded."


def test_list_kiwisaver_empty(agent):
    """list_kiwisaver should report no rows when empty."""
    assert agent.list_kiwisaver() == "No Kiwisaver daily balances recorded."


def test_list_kiwisaver_single_row_na(agent):
    """list_kiwisaver should show N/A change for the first row."""
    agent.add_kiwisaver_daily("2026-11-03", 1000.0)
    result = agent.list_kiwisaver()
    assert "2026-11-03 - balance 1000.0 - change N/A" in result


def test_list_kiwisaver_multiple_rows_pct_change(agent):
    """list_kiwisaver should compute daily % change between consecutive rows."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 1100.0)  # +10%
    agent.add_kiwisaver_daily("2026-11-03", 1050.0)  # -4.55% (approx)
    result = agent.list_kiwisaver()
    assert "2026-11-01 - balance 1000.0 - change N/A" in result
    assert "2026-11-02 - balance 1100.0 - change +10.00%" in result
    assert "-4.55%" in result


def test_list_kiwisaver_start_date_filter(agent):
    """list_kiwisaver with start_date should exclude earlier rows."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 1100.0)
    agent.add_kiwisaver_daily("2026-11-03", 1200.0)
    result = agent.list_kiwisaver(start_date="2026-11-02")
    assert "2026-11-01" not in result
    assert "2026-11-02" in result
    assert "2026-11-03" in result


def test_list_kiwisaver_end_date_filter(agent):
    """list_kiwisaver with end_date should exclude later rows."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 1100.0)
    agent.add_kiwisaver_daily("2026-11-03", 1200.0)
    result = agent.list_kiwisaver(end_date="2026-11-02")
    assert "2026-11-01" in result
    assert "2026-11-02" in result
    assert "2026-11-03" not in result


def test_list_kiwisaver_both_filters(agent):
    """list_kiwisaver with both start_date and end_date should return rows in range."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 1100.0)
    agent.add_kiwisaver_daily("2026-11-03", 1200.0)
    agent.add_kiwisaver_daily("2026-11-04", 1300.0)
    result = agent.list_kiwisaver(start_date="2026-11-02", end_date="2026-11-03")
    assert "2026-11-01" not in result
    assert "2026-11-02" in result
    assert "2026-11-03" in result
    assert "2026-11-04" not in result


def test_list_kiwisaver_invalid_start_date(agent):
    """list_kiwisaver should reject an invalid start_date format."""
    result = agent.list_kiwisaver(start_date="not-a-date")
    assert result.startswith("Error:")


def test_list_kiwisaver_invalid_end_date(agent):
    """list_kiwisaver should reject an invalid end_date format."""
    result = agent.list_kiwisaver(end_date="2026-13-01")
    assert result.startswith("Error:")


def test_kiwisaver_performance_unit_price_basis(agent):
    """kiwisaver_performance with unit_price on both ends uses unit price (ignores contributions)."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0, 50.0, 20.0)
    agent.add_kiwisaver_daily("2026-11-02", 1100.0, 50.0, 22.0)  # +10%
    agent.add_kiwisaver_daily("2026-11-03", 1045.0, 50.0, 20.9)  # -5%

    result = agent.kiwisaver_performance("2026-11-01", "2026-11-03")

    assert "Kiwisaver performance from 2026-11-01 to 2026-11-03:" in result
    assert "Basis: unit price" in result
    assert "Start value: 20.00" in result
    assert "End value: 20.90" in result
    assert "Change: +4.50%" in result
    assert "Days: 3" in result
    assert "Best day: 2026-11-02 (+10.00%)" in result
    assert "Worst day: 2026-11-03 (-5.00%)" in result
    assert "Max drawdown: 5.00%" in result


def test_kiwisaver_performance_balance_basis_estimate(agent):
    """kiwisaver_performance without unit_price uses balance minus adjustments (estimate)."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 1100.0)
    agent.add_kiwisaver_adjustment("2026-11-02", 100.0, "employer")

    result = agent.kiwisaver_performance("2026-11-01", "2026-11-02")

    assert "Basis: balance (adjusted for contributions)" in result
    assert "Start value: 1,000.00" in result
    assert "End value: 1,000.00" in result  # 1100 - 100 adjustment
    assert "Change: +0.00%" in result
    assert "Note: Result is an estimate" in result


def test_kiwisaver_performance_stale_days(agent):
    """kiwisaver_performance flags days where balance is unchanged."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 1000.0)  # stale
    agent.add_kiwisaver_daily("2026-11-03", 1100.0)

    result = agent.kiwisaver_performance("2026-11-01", "2026-11-03")
    assert "Stale days (unchanged balance): 2026-11-02" in result


def test_kiwisaver_performance_single_row(agent):
    """kiwisaver_performance with a single row shows zero change and no best/worst day."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0, 50.0, 20.0)

    result = agent.kiwisaver_performance("2026-11-01", "2026-11-01")

    assert "Start value: 20.00" in result
    assert "End value: 20.00" in result
    assert "Change: +0.00%" in result
    assert "Days: 1" in result


def test_kiwisaver_performance_invalid_start_date(agent):
    """kiwisaver_performance rejects an invalid start_date."""
    result = agent.kiwisaver_performance("not-a-date", "2026-11-03")
    assert result.startswith("Error:")


def test_kiwisaver_performance_start_after_end(agent):
    """kiwisaver_performance rejects start_date after end_date."""
    result = agent.kiwisaver_performance("2026-11-03", "2026-11-01")
    assert result.startswith("Error:")


def test_kiwisaver_performance_no_data(agent):
    """kiwisaver_performance reports no data when the range is empty."""
    result = agent.kiwisaver_performance("2026-11-01", "2026-11-03")
    assert result == "No Kiwisaver data in the given date range."


def test_kiwisaver_performance_max_drawdown_with_adjustment(agent):
    """kiwisaver_performance max drawdown accounts for mid-period adjustment (cumulative).
    
    Day 1: balance 1000 (no adjustment) → effective 1000
    Day 2: balance 900 (+employer 100) → effective 800
    Day 3: balance 950 (+employer 100) → effective 850
    
    Start: 1000, End: 850, Change: -15%
    Peak: 1000, Trough: 800 → Max drawdown: 20% around day 2
    """
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 900.0)
    agent.add_kiwisaver_daily("2026-11-03", 950.0)
    agent.add_kiwisaver_adjustment("2026-11-02", 100.0, "employer")

    result = agent.kiwisaver_performance("2026-11-01", "2026-11-03")

    assert "Start value: 1,000.00" in result
    assert "End value: 850.00" in result
    assert "Change: -15.00%" in result
    assert "Max drawdown: 20.00%" in result
    assert "around 2026-11-02" in result


def test_kiwisaver_performance_stale_days(agent):
    """kiwisaver_performance flags days where balance is unchanged."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0)
    agent.add_kiwisaver_daily("2026-11-02", 1000.0)  # stale
    agent.add_kiwisaver_daily("2026-11-03", 1100.0)

    result = agent.kiwisaver_performance("2026-11-01", "2026-11-03")
    assert "Stale days (unchanged balance): 2026-11-02" in result


def test_list_kiwisaver_unit_price_basis_with_gap(agent):
    """list_kiwisaver uses unit price when available on both ends; warns about skipped None row."""
    agent.add_kiwisaver_daily("2026-11-01", 1000.0, 50.0, 20.0)
    agent.add_kiwisaver_daily("2026-11-02", 1000.0, 50.0, None)  # unit_price missing
    agent.add_kiwisaver_daily("2026-11-03", 1100.0, 50.0, 22.0)

    result = agent.list_kiwisaver("2026-11-01", "2026-11-03")

    assert "2026-11-01 - balance 20.0 - change N/A" in result
    assert "2026-11-02 - balance  - change N/A" in result  # skipped None row
    assert "2026-11-03 - balance 22.0 - change +10.00%" in result
    assert "Warning" in result or "skip" in result.lower()


def test_kiwisaver_performance_realistic_no_adjustments(agent):
    """Realistic 22-day Kiwisaver balance series with no adjustments."""
    dates = [
        "2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19",
        "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24",
        "2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29",
        "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03",
        "2026-11-04", "2026-11-05",
    ]
    balances = [
        10000.00, 10000.00, 9969.24, 9952.61, 10146.39,
        10130.53, 10115.32, 10115.32, 10115.32, 10115.32,
        10115.32, 10115.32, 10175.46, 10103.16, 10061.15,
        10119.52, 10119.52, 10119.52, 10182.75, 10182.75,
        10090.14, 10062.89,
    ]
    for d, b in zip(dates, balances):
        agent.add_kiwisaver_daily(d, b)

    # kiwisaver_performance over the full range
    result = agent.kiwisaver_performance("2026-10-15", "2026-11-05")
    assert "Days: 22" in result
    assert "Start value: 10,000.00" in result
    assert "End value: 10,062.89" in result
    assert "Change: +0.63%" in result
    assert "Best day: 2026-10-19 (+1.95%)" in result
    assert "Worst day: 2026-11-04 (-0.91%)" in result
    assert "Max drawdown: 1.18%" in result
    assert "Stale days (unchanged balance): 2026-10-16, 2026-10-22, 2026-10-23, 2026-10-24, 2026-10-25, 2026-10-26, 2026-10-31, 2026-11-01, 2026-11-03" in result

    # list_kiwisaver over the same range must agree
    list_result = agent.list_kiwisaver("2026-10-15", "2026-11-05")
    assert "Change: +0.63%" not in list_result  # list_kiwisaver doesn't print summary
    # Both show the same daily % changes
    assert "+1.95%" in list_result
    assert "-0.91%" in list_result


def test_kiwisaver_performance_unit_price_ignores_adjustment(agent):
    """With unit_price on both ends, basis is unit price and adjustments are ignored."""
    dates = [
        "2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19",
        "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24",
        "2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29",
        "2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03",
        "2026-11-04", "2026-11-05",
    ]
    balances = [
        10000.00, 10000.00, 9969.24, 9952.61, 10146.39,
        10130.53, 10115.32, 10115.32, 10115.32, 10115.32,
        10115.32, 10115.32, 10175.46, 10103.16, 10061.15,
        10119.52, 10119.52, 10119.52, 10182.75, 10182.75,
        10090.14, 10062.89,
    ]
    for d, b in zip(dates, balances):
        agent.add_kiwisaver_daily(d, b, unit_price=b / 1000.0)

    # Add a large employee adjustment mid-period — must be ignored on unit-price basis
    agent.add_kiwisaver_adjustment("2026-10-28", 500.0, "employee")

    result = agent.kiwisaver_performance("2026-10-15", "2026-11-05")
    assert "Basis: unit price" in result
    assert "Days: 22" in result
    assert "Start value: 10.00" in result
    assert "End value: 10.06" in result
    assert "Change: +0.63%" in result
    assert "500" not in result  # adjustment must not appear


def test_kiwisaver_stale_day_with_adjustment_still_flagged(agent):
    """A stale day with an adjustment is still flagged stale (raw balance, not adjusted)."""
    agent.add_kiwisaver_daily("2026-10-15", 10000.00)
    agent.add_kiwisaver_daily("2026-10-16", 10000.00)  # stale: same balance
    agent.add_kiwisaver_adjustment("2026-10-16", 500.0, "employee")  # adjustment on stale day

    result = agent.kiwisaver_performance("2026-10-15", "2026-10-16")
    assert "Stale days (unchanged balance): 2026-10-16" in result


def test_backfill_dry_run_writes_nothing(tmp_path, monkeypatch):
    """--dry-run (default) prints planned changes but writes nothing to the DB."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    csv_path = tmp_path / "kiwisaver.csv"
    csv_path.write_text(
        "Date,KIWI Saver\n"
        '"Thursday, 15 October 2026","10,000.00"\n'
        '"Friday, 16 October 2026","10,100.00"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "backfill_implied_unit_price.py", str(csv_path)],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0
    assert "dry-run" in result.stdout.lower() or "would" in result.stdout.lower()
    # DB unchanged: no rows inserted
    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM kiwisaver_daily").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_backfill_apply_idempotent(tmp_path, monkeypatch):
    """Running --apply twice gives the same result."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    csv_path = tmp_path / "kiwisaver.csv"
    csv_path.write_text(
        "Date,KIWI Saver\n"
        '"Thursday, 15 October 2026","10,000.00"\n'
        '"Friday, 16 October 2026","10,100.00"\n',
        encoding="utf-8",
    )

    # Apply twice
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "backfill_implied_unit_price.py", "--apply", str(csv_path)],
            capture_output=True, text=True, cwd="C:\\expense_AI_agent",
        )
        assert result.returncode == 0

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT date, unit_price, units, price_source "
            "FROM kiwisaver_daily ORDER BY date"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    # Second run should not duplicate rows
    for date, up, units, src in rows:
        assert src == "implied"
        assert abs(units - 6952.3989) < 0.0001


def test_backfill_never_overwrites_actual(tmp_path, monkeypatch):
    """A row with price_source='actual' is never overwritten."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    # Pre-seed a row with an actual unit price
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO kiwisaver_daily (date, balance, unit_price, units, price_source) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-10-15", 10000.00, 10.50, 100.0, "actual"),
        )
        conn.commit()
    finally:
        conn.close()

    csv_path = tmp_path / "kiwisaver.csv"
    csv_path.write_text(
        "Date,KIWI Saver\n"
        '"Thursday, 15 October 2026","10,000.00"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "backfill_implied_unit_price.py", "--apply", str(csv_path)],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0
    assert "skip" in result.stdout.lower()

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT unit_price, price_source FROM kiwisaver_daily WHERE date = ?",
            ("2026-10-15",),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 10.50  # unchanged
    assert row[1] == "actual"


def test_backfill_implied_price_rounded_4_decimals(tmp_path, monkeypatch):
    """implied_unit_price = balance / assumed_units, rounded to 4 decimals using Decimal."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    csv_path = tmp_path / "kiwisaver.csv"
    csv_path.write_text(
        "Date,KIWI Saver\n"
        '"Thursday, 15 October 2026","10,000.00"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "backfill_implied_unit_price.py",
         "--assumed-units", "6952.3989", "--apply", str(csv_path)],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0

    # 10000.00 / 6952.3989 = 1.438348... -> 1.4383 (rounded to 4 decimals)
    expected_price = Decimal("10000.00") / Decimal("6952.3989")
    expected_price = expected_price.quantize(Decimal("0.0001"))

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT unit_price FROM kiwisaver_daily WHERE date = ?",
            ("2026-10-15",),
        ).fetchone()
    finally:
        conn.close()
    assert Decimal(str(row[0])) == expected_price


def test_backfill_bad_csv_rows_reported(tmp_path, monkeypatch):
    """Bad CSV rows are reported in output, not silently dropped."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    csv_path = tmp_path / "kiwisaver.csv"
    csv_path.write_text(
        "Date,KIWI Saver\n"
        '"Not a Date","10,000.00"\n'
        '"Friday, 16 October 2026","not-a-number"\n'
        '"Saturday, 17 October 2026","10,100.00"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "backfill_implied_unit_price.py", "--apply", str(csv_path)],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0
    # Bad rows reported in output
    assert "Bad CSV rows" in result.stdout or "unparseable" in result.stdout.lower()
    # Only the valid row inserted
    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM kiwisaver_daily").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


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


def _get_expenses_with_id(expense_agent):
    """Fetch all rows including id, ordered by id."""
    import sqlite3

    conn = sqlite3.connect(expense_agent.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, amount, category, note, spent_at FROM expenses ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return rows


def test_delete_expense_by_note(agent):
    """delete_expense should preview matches without deleting, then delete on confirm."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    agent.add_expense(5, "Transport", "taxi", "2026-11-03 08:15")
    # Phase 1: preview only, no deletion
    result = agent.delete_expense(note="lunch")
    assert "Are you sure you want to delete this record? (y/n)" in result
    assert "lunch" in result
    assert len(_get_expenses(agent)) == 2  # nothing deleted yet
    # Phase 2: confirm deletion
    result = agent.delete_expense(note="lunch", confirm="y")
    assert "Deleted 1 record(s)" in result
    remaining = _get_expenses(agent)
    assert _strip_spent_at(remaining) == [
        {"amount": 5.0, "category": "transport", "note": "taxi"}
    ]


def test_delete_expense_by_id(agent):
    """delete_expense should delete the row matching an exact id on confirm."""
    agent.add_expense("10.50", "Food", "lunch")
    agent.add_expense(5, "Transport", "taxi")
    rows = _get_expenses_with_id(agent)
    target_id = rows[0][0]
    result = agent.delete_expense(id=target_id, confirm="y")
    assert "Deleted 1 record(s)" in result
    remaining = _get_expenses_with_id(agent)
    assert len(remaining) == 1
    assert remaining[0][0] != target_id


def test_delete_expense_by_amount(agent):
    """delete_expense should remove rows matching an exact amount on confirm."""
    agent.add_expense("10.50", "Food", "lunch")
    agent.add_expense(10.50, "Food", "dinner")
    agent.add_expense(5, "Transport", "taxi")
    result = agent.delete_expense(amount=10.50, confirm="y")
    assert "Deleted 2 record(s)" in result
    remaining = _strip_spent_at(_get_expenses(agent))
    assert remaining == [{"amount": 5.0, "category": "transport", "note": "taxi"}]


def test_delete_expense_by_spent_at(agent):
    """delete_expense should remove rows matching an exact timestamp on confirm."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    agent.add_expense(5, "Transport", "taxi", "2026-11-03 08:15")
    result = agent.delete_expense(spent_at="2026-11-03 14:30", confirm="y")
    assert "Deleted 1 record(s)" in result
    remaining = _strip_spent_at(_get_expenses(agent))
    assert remaining == [{"amount": 5.0, "category": "transport", "note": "taxi"}]


def test_delete_expense_12_hour_timestamp(agent):
    """delete_expense should accept a 12-hour AM/PM timestamp on confirm."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 02:30 PM")
    result = agent.delete_expense(spent_at="2026-11-03 02:30 PM", confirm="y")
    assert "Deleted 1 record(s)" in result
    assert agent.list_expenses() == "No expenses recorded."


def test_delete_expense_by_combination(agent):
    """delete_expense should require all provided criteria to match on confirm."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    agent.add_expense("10.50", "Food", "dinner", "2026-11-03 19:00")
    result = agent.delete_expense(note="lunch", amount=10.50, confirm="y")
    assert "Deleted 1 record(s)" in result
    remaining = _strip_spent_at(_get_expenses(agent))
    assert remaining == [
        {"amount": 10.5, "category": "food", "note": "dinner"}
    ]


def test_delete_expense_no_match(agent):
    """delete_expense should report when nothing matches."""
    agent.add_expense("10.50", "Food", "lunch")
    result = agent.delete_expense(note="dinner", confirm="y")
    assert result == "No matching expense found."
    assert len(_get_expenses(agent)) == 1


def test_delete_expense_no_criteria(agent):
    """delete_expense should error when no identifying criterion is given."""
    result = agent.delete_expense()
    assert result.startswith("Error:")


def test_delete_expense_invalid_timestamp(agent):
    """delete_expense should error on an invalid spent_at value."""
    result = agent.delete_expense(spent_at="not-a-date")
    assert result.startswith("Error:")


def test_delete_expense_cancel_with_n(agent):
    """delete_expense should cancel deletion when user replies 'n'."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    result = agent.delete_expense(note="lunch")
    assert "Are you sure you want to delete this record? (y/n)" in result
    assert len(_get_expenses(agent)) == 1
    result = agent.delete_expense(note="lunch", confirm="n")
    assert result == "Deletion cancelled. No changes were made."
    assert len(_get_expenses(agent)) == 1  # still present


def test_delete_expense_cancel_with_no(agent):
    """delete_expense should cancel deletion when user replies 'no'."""
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    agent.delete_expense(note="lunch")
    result = agent.delete_expense(note="lunch", confirm="no")
    assert result == "Deletion cancelled. No changes were made."
    assert len(_get_expenses(agent)) == 1


def test_correct_duplicate_entry(agent):
    """Correcting a duplicate: delete the wrong entry (with confirmation), then add the corrected one."""
    # Accidental entry with the wrong timestamp
    agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:00")
    # Phase 1: preview the wrong entry
    preview = agent.delete_expense(note="lunch", amount=10.50, spent_at="2026-11-03 14:00")
    assert "Are you sure you want to delete this record? (y/n)" in preview
    # Phase 2: confirm deletion
    deleted = agent.delete_expense(
        note="lunch", amount=10.50, spent_at="2026-11-03 14:00", confirm="y"
    )
    assert "Deleted 1 record(s)" in deleted
    # Add the corrected record with the proper timestamp
    added = agent.add_expense("10.50", "Food", "lunch", "2026-11-03 14:30")
    assert "Added" in added
    expenses = _strip_spent_at(_get_expenses(agent))
    assert expenses == [{"amount": 10.5, "category": "food", "note": "lunch"}]
    full = _get_expenses(agent)
    assert full[0]["spent_at"] == "2026-11-03 14:30"


def test_load_expenses_for_table(tmp_path, monkeypatch):
    """load_expenses_for_table returns rows newest first with 12-hour time."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))

    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    ea_mod.add_expense(10, "Food", "lunch", "2026-11-03 14:30")
    ea_mod.add_expense(5, "Transport", "", "2026-11-03 08:15")

    import app
    data = app.load_expenses_for_table()
    assert len(data) == 2
    # Newest first: Transport (08:15) then Food (14:30)
    assert data[0]["Category"] == "transport"
    assert data[0]["Amount"] == 5.0
    assert "08:15 AM" in data[0]["Date/Time"]
    assert data[1]["Category"] == "food"
    assert data[1]["Amount"] == 10.0
    assert "02:30 PM" in data[1]["Date/Time"]


def test_category_totals(tmp_path, monkeypatch):
    """category_totals sums amounts per category."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))

    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    ea_mod.add_expense(10, "Food", "lunch")
    ea_mod.add_expense(20, "Food", "dinner")
    ea_mod.add_expense(5, "Transport", "")

    import app
    totals = app.category_totals()
    assert totals == {"food": 30.0, "transport": 5.0}


def test_agent_kiwisaver_performance_via_mocked_model(tmp_path, monkeypatch):
    """'how is my KiwiSaver doing since 15 Oct?' triggers kiwisaver_performance tool."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )

    # Add some Kiwisaver data for the tool to read
    expense_agent.add_kiwisaver_daily("2026-10-15", 1000.0, 50.0, 20.0)
    expense_agent.add_kiwisaver_daily("2026-10-16", 1100.0, 50.0, 22.0)

    # Mock model: first response requests a tool call, second returns text
    mock_client = mock_openai.return_value

    tool_call = mock.Mock()
    tool_call.id = "call_123"
    tool_call.function.name = "kiwisaver_performance"
    tool_call.function.arguments = '{"start_date": "2026-10-15", "end_date": "2026-10-16"}'

    first_response = mock.Mock()
    first_response.choices = [mock.Mock()]
    first_response.choices[0].message = mock.Mock()
    first_response.choices[0].message.tool_calls = [tool_call]
    first_response.choices[0].message.content = None

    second_response = mock.Mock()
    second_response.choices = [mock.Mock()]
    second_response.choices[0].message = mock.Mock()
    second_response.choices[0].message.tool_calls = []
    second_response.choices[0].message.content = (
        "Your KiwiSaver is up 10%. (For tracking purposes only, not financial advice.)"
    )

    mock_client.chat.completions.create.side_effect = [first_response, second_response]

    history = [{"role": "system", "content": expense_agent.SYSTEM_PROMPT}]
    reply, tool_calls = expense_agent.run_agent(
        "how is my KiwiSaver doing since 15 Oct?", history
    )

    # Verify kiwisaver_performance was triggered
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "kiwisaver_performance"
    assert tool_calls[0]["args"]["start_date"] == "2026-10-15"
    assert tool_calls[0]["args"]["end_date"] == "2026-10-16"

    # Verify the model's final answer and the not-financial-advice disclaimer
    assert "up 10%" in reply
    assert "not financial advice" in reply.lower()


def test_llm_calls_rows_written(tmp_path, monkeypatch):
    """run_agent writes one row per model call to llm_calls with tokens and latency."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )

    mock_client = mock_openai.return_value

    response = mock.Mock()
    response.id = "resp_abc123"
    response.usage = mock.Mock()
    response.usage.prompt_tokens = 120
    response.usage.completion_tokens = 45
    response.choices = [mock.Mock()]
    response.choices[0].message = mock.Mock()
    response.choices[0].message.tool_calls = []
    response.choices[0].message.content = "Hello!"

    mock_client.chat.completions.create.return_value = response

    history = [{"role": "system", "content": expense_agent.SYSTEM_PROMPT}]
    expense_agent.run_agent("hello", history)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT request_id, step, model, prompt_tokens, completion_tokens, "
            "latency_s, tool_name FROM llm_calls ORDER BY step"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][0] == "resp_abc123"  # request_id
    assert rows[0][1] == 1  # step
    assert rows[0][2] == expense_agent.MODEL  # model
    assert rows[0][3] == 120  # prompt_tokens
    assert rows[0][4] == 45  # completion_tokens
    assert isinstance(rows[0][5], float)  # latency_s
    assert rows[0][5] >= 0
    assert rows[0][6] is None  # tool_name (no tool called)


def test_llm_calls_missing_usage_field(tmp_path, monkeypatch):
    """run_agent handles a missing usage field without crashing; stores NULL tokens."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )

    mock_client = mock_openai.return_value

    response = mock.Mock()
    response.id = "resp_no_usage"
    del response.usage  # no usage field
    response.choices = [mock.Mock()]
    response.choices[0].message = mock.Mock()
    response.choices[0].message.tool_calls = []
    response.choices[0].message.content = "Done"

    mock_client.chat.completions.create.return_value = response

    history = [{"role": "system", "content": expense_agent.SYSTEM_PROMPT}]
    reply, _ = expense_agent.run_agent("hello", history)

    assert reply == "Done"

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT prompt_tokens, completion_tokens FROM llm_calls"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][0] is None  # prompt_tokens NULL
    assert rows[0][1] is None  # completion_tokens NULL


def test_llm_calls_no_secrets_or_message_content(tmp_path, monkeypatch):
    """llm_calls stores no API key or user message content."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "super-secret-key-12345",
        },
        db_path=db_path,
    )

    mock_client = mock_openai.return_value

    response = mock.Mock()
    response.id = "resp_xyz"
    response.usage = mock.Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.choices = [mock.Mock()]
    response.choices[0].message = mock.Mock()
    response.choices[0].message.tool_calls = []
    response.choices[0].message.content = "Response"

    mock_client.chat.completions.create.return_value = response

    user_msg = "I spent $500 on a secret purchase at the mall"
    history = [{"role": "system", "content": expense_agent.SYSTEM_PROMPT}]
    expense_agent.run_agent(user_msg, history)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT request_id, step, model, prompt_tokens, completion_tokens, "
            "latency_s, tool_name FROM llm_calls"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    row_text = " ".join(str(c) for c in rows[0] if c is not None)
    assert "super-secret-key-12345" not in row_text
    assert "$500" not in row_text
    assert "secret purchase" not in row_text
    assert "mall" not in row_text


def test_request_cost_known_pricing(tmp_path, monkeypatch):
    """request_cost returns correct cost for a model with known pricing."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
            "NVIDIA_MODEL": "test/known-model",
        },
        db_path=db_path,
    )

    # Insert a row directly into llm_calls with known token counts
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO llm_calls
                (timestamp, request_id, step, model, prompt_tokens,
                 completion_tokens, latency_s, tool_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-11-05 10:00:00", "req_known", 1, "test/known-model",
             1_000_000, 500_000, 0.5, None),
        )
        conn.commit()
    finally:
        conn.close()

    # Set pricing: $1.00 per 1M input, $2.00 per 1M output
    expense_agent.PRICING["test/known-model"] = {
        "input_per_1m": 1.00,
        "output_per_1m": 2.00,
    }

    cost = expense_agent.request_cost("req_known")
    assert cost is not None
    expected = 1.00 + 1.00  # 1M input @ $1.00/1M + 500K output @ $2.00/1M
    assert abs(cost - expected) < 0.0001


def test_request_cost_unknown_model(tmp_path, monkeypatch):
    """request_cost returns None for a model not in PRICING."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO llm_calls
                (timestamp, request_id, step, model, prompt_tokens,
                 completion_tokens, latency_s, tool_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-11-05 10:00:00", "req_unknown", 1, "test/unknown-model",
             100, 50, 0.1, None),
        )
        conn.commit()
    finally:
        conn.close()

    cost = expense_agent.request_cost("req_unknown")
    assert cost is None


def test_request_cost_not_found(tmp_path, monkeypatch):
    """request_cost returns None for a request_id that does not exist."""
    db_path = tmp_path / "expenses.db"
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        },
        db_path=db_path,
    )

    cost = expense_agent.request_cost("req_does_not_exist")
    assert cost is None


def test_usage_report_output(tmp_path, monkeypatch):
    """usage_report.py prints per-request and totals with made-up prices."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO llm_calls
                (timestamp, request_id, step, model, prompt_tokens,
                 completion_tokens, latency_s, tool_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-11-05 10:00:00", "req_a", 1, "test/priced-model", 100, 50, 0.5, None),
        )
        conn.execute(
            """
            INSERT INTO llm_calls
                (timestamp, request_id, step, model, prompt_tokens,
                 completion_tokens, latency_s, tool_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-11-05 10:01:00", "req_a", 2, "test/priced-model", 200, 100, 1.0, None),
        )
        conn.commit()
    finally:
        conn.close()

    ea_mod.PRICING["test/priced-model"] = {
        "input_per_1m": 10.00,
        "output_per_1m": 20.00,
    }

    result = subprocess.run(
        [sys.executable, "usage_report.py"],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0


def test_benchmark_runs_and_creates_report(tmp_path, monkeypatch):
    """benchmark_agent.py runs 10 prompts and writes benchmark_baseline.json."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    result = subprocess.run(
        [sys.executable, "benchmark_agent.py"],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0

    report_path = tmp_path / "benchmark_baseline.json"
    assert report_path.exists()

    import json
    data = json.loads(report_path.read_text())
    assert "results" in data
    assert len(data["results"]) == 10
    assert "summary" in data

    for r in data["results"]:
        assert "prompt" in r
        assert "answer" in r
        assert "steps" in r
        assert "prompt_tokens" in r
        assert "completion_tokens" in r
        assert "total_tokens" in r
        assert "latency_s" in r
        assert "cost_usd" in r
        assert "tool_calls" in r


def test_benchmark_seeds_data(tmp_path, monkeypatch):
    """benchmark_agent.py seeds 5 expenses and 10 KiwiSaver rows."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    result = subprocess.run(
        [sys.executable, "benchmark_agent.py"],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0

    conn = sqlite3.connect(str(db_path))
    try:
        expense_count = conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]
        kiwisaver_count = conn.execute("SELECT COUNT(*) FROM kiwisaver_daily").fetchone()[0]
    finally:
        conn.close()

    assert expense_count == 5
    assert kiwisaver_count == 10


def test_benchmark_no_real_data(tmp_path, monkeypatch):
    """benchmark_agent.py uses made-up data only."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    result = subprocess.run(
        [sys.executable, "benchmark_agent.py"],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT category, note FROM expenses").fetchall()
    finally:
        conn.close()

    categories = {r[0] for r in rows}
    notes = {r[1] for r in rows}
    assert categories.issubset({"food", "transport", "coffee", "entertainment", "utilities"})
    for note in notes:
        assert note not in ("rent", "mortgage", "salary", "bonus")


def test_benchmark_agent_no_agent_changes(tmp_path, monkeypatch):
    """benchmark_agent.py does not modify expense_agent.py behavior."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    ea_mod.add_expense("10.00", "Test", "check")
    result = ea_mod.list_expenses()
    assert "10.0" in result
    assert "test" in result


def test_usage_report_unknown_pricing(tmp_path, monkeypatch):
    """usage_report.py handles unknown pricing without crashing."""
    db_path = tmp_path / "expenses.db"
    monkeypatch.setenv("EXPENSES_DB_PATH", str(db_path))
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO llm_calls
                (timestamp, request_id, step, model, prompt_tokens,
                 completion_tokens, latency_s, tool_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-11-05 10:00:00", "req_b", 1, "test/unpriced-model", 100, 50, 0.5, None),
        )
        conn.commit()
    finally:
        conn.close()

    result = subprocess.run(
        [sys.executable, "usage_report.py"],
        capture_output=True, text=True, cwd="C:\\expense_AI_agent",
    )
    assert result.returncode == 0

