"""Tests for expense_agent.py"""

import importlib
import os
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
    """Fetch all rows as a list of dicts (amount, category, note)."""
    rows = expense_agent.conn.execute(
        "SELECT amount, category, note FROM expenses ORDER BY id"
    ).fetchall()
    return [
        {"amount": amount, "category": category, "note": note}
        for amount, category, note in rows
    ]


def test_add_expense_records_amount_category_note(agent):
    """add_expense should insert a row with amount, category, and note."""
    result = agent.add_expense("10.50", "Food", "lunch")
    assert result == "Added 10.50 to Food."
    assert _get_expenses(agent) == [
        {"amount": 10.5, "category": "food", "note": "lunch"}
    ]


def test_add_expense_default_note(agent):
    """note should default to an empty string when not provided."""
    agent.add_expense(5, "Transport")
    assert _get_expenses(agent) == [
        {"amount": 5.0, "category": "transport", "note": ""}
    ]


def test_add_expense_lowercases_category(agent):
    """Categories should be stored lowercase regardless of input case."""
    agent.add_expense(3, "FOOD")
    agent.add_expense(2, "Food")
    assert _get_expenses(agent) == [
        {"amount": 3.0, "category": "food", "note": ""},
        {"amount": 2.0, "category": "food", "note": ""},
    ]


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
