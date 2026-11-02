"""Tests for expense_agent.py"""

import importlib
import os
from unittest import mock

import pytest


def _reload_with_env(env):
    """Patch env + source modules, reload expense_agent, return it."""
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


def test_client_instantiated_with_env():
    """Client should be created using env vars."""
    expense_agent, mock_openai = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        }
    )

    mock_openai.assert_called_with(
        base_url="https://test.example.com/v1",
        api_key="test-api-key",
    )
    assert expense_agent.MODEL == "nvidia/nemotron-3.5-super-vl-120b-a12b"


def test_model_override_from_env():
    """NVIDIA_MODEL env var should override the default model."""
    expense_agent, _ = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
            "NVIDIA_MODEL": "custom/model",
        }
    )
    assert expense_agent.MODEL == "custom/model"


def _fresh_agent():
    """Reload the module so the in-memory expenses list starts empty."""
    expense_agent, _ = _reload_with_env(
        {
            "NVIDIA_BASE_URL": "https://test.example.com/v1",
            "NVIDIA_API_KEY": "test-api-key",
        }
    )
    return expense_agent


def test_add_expense_records_amount_category_note():
    """add_expense should append a dict with amount, category, and note."""
    expense_agent = _fresh_agent()
    expense_agent.expenses.clear()
    result = expense_agent.add_expense("10.50", "Food", "lunch")
    assert result == "Added 10.50 to Food."
    assert expense_agent.expenses == [
        {"amount": 10.5, "category": "food", "note": "lunch"}
    ]


def test_add_expense_default_note():
    """note should default to an empty string when not provided."""
    expense_agent = _fresh_agent()
    expense_agent.expenses.clear()
    expense_agent.add_expense(5, "Transport")
    assert expense_agent.expenses == [
        {"amount": 5.0, "category": "transport", "note": ""}
    ]


def test_add_expense_lowercases_category():
    """Categories should be stored lowercase regardless of input case."""
    expense_agent = _fresh_agent()
    expense_agent.expenses.clear()
    expense_agent.add_expense(3, "FOOD")
    expense_agent.add_expense(2, "Food")
    assert expense_agent.expenses == [
        {"amount": 3.0, "category": "food", "note": ""},
        {"amount": 2.0, "category": "food", "note": ""},
    ]
