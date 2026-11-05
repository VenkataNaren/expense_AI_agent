"""Benchmark comparison: run the same set of prompts through two models.

Usage:
    python benchmark_compare.py MODEL_1 MODEL_2

Runs a fixed set of prompts through each model and prints a side-by-side
comparison table showing steps, tokens, latency, and cost per model.

Models are specified by name (e.g., nvidia/nemotron-3.5-super-vl-120b-a12b).
The NVIDIA_MODEL env var is overridden per-run for each model.
"""
import importlib
import json
import os
import sys
import tempfile
import time

import expense_agent as ea


PROMPTS = [
    "Add an expense: 12.50 on coffee this morning",
    "Add an expense: 45.00 on transport for a taxi ride",
    "List all my expenses",
    "What is my total for food?",
    "What is my total for transport?",
    "Delete the coffee expense I just added - confirm with y",
    "How is my KiwiSaver doing since 2026-10-15?",
    "List my KiwiSaver daily balances",
    "What is the weather like today?",
    "Delete the taxi expense - confirm with n",
]


def seed_data(expense_agent_mod):
    """Add 5 expenses and 10 KiwiSaver rows with made-up data."""
    expense_agent_mod.add_expense("12.50", "coffee", "morning")
    expense_agent_mod.add_expense("45.00", "transport", "taxi ride")
    expense_agent_mod.add_expense("8.75", "food", "lunch")
    expense_agent_mod.add_expense("32.00", "entertainment", "movie tickets")
    expense_agent_mod.add_expense("15.60", "utilities", "internet bill")

    kiwisaver_rows = [
        ("2026-10-15", 10000.00, 500.0, 20.0),
        ("2026-10-16", 10000.00, 500.0, 20.0),
        ("2026-10-17", 9969.24, 498.5, 20.0),
        ("2026-10-18", 9952.61, 497.6, 20.0),
        ("2026-10-19", 10146.39, 507.3, 20.0),
        ("2026-10-20", 10130.53, 506.5, 20.0),
        ("2026-10-21", 10115.32, 505.8, 20.0),
        ("2026-10-22", 10115.32, 505.8, 20.0),
        ("2026-10-23", 10115.32, 505.8, 20.0),
        ("2026-10-24", 10115.32, 505.8, 20.0),
    ]
    for date, balance, units, unit_price in kiwisaver_rows:
        expense_agent_mod.add_kiwisaver_daily(date, balance, units, unit_price)


def run_single_model(model_name, db_path):
    """Run all prompts against one model and return results list."""
    os.environ["NVIDIA_MODEL"] = model_name

    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    seed_data(ea_mod)

    results = []
    for prompt in PROMPTS:
        history = [{"role": "system", "content": ea_mod.SYSTEM_PROMPT}]

        start = time.perf_counter()
        try:
            answer, tool_calls = ea_mod.run_agent(prompt, history)
            elapsed = time.perf_counter() - start

            # Query the last llm_calls entry for token counts
            conn = sqlite3.connect(ea_mod.DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT prompt_tokens, completion_tokens FROM llm_calls "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()

            prompt_tokens = rows[0] if rows else None
            completion_tokens = rows[1] if rows else None
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            # Cost estimate
            cost_usd = None
            pricing = ea_mod.PRICING.get(model_name)
            if pricing and prompt_tokens is not None and completion_tokens is not None:
                cost_usd = (
                    prompt_tokens * pricing["input_per_1m"] / 1_000_000
                    + completion_tokens * pricing["output_per_1m"] / 1_000_000
                )

            tool_names = [tc["name"] for tc in tool_calls]

            results.append({
                "prompt": prompt,
                "answer": answer,
                "steps": len(tool_calls),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_s": round(elapsed, 4),
                "cost_usd": round(cost_usd, 6) if cost_usd is not None else None,
                "tool_calls": tool_names,
                "error": None,
            })
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append({
                "prompt": prompt,
                "answer": f"ERROR: {type(e).__name__}: {e}",
                "steps": 0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "latency_s": round(elapsed, 4),
                "cost_usd": None,
                "tool_calls": [],
                "error": str(e),
            })

    return results


def print_comparison(results_1, results_2, model_1, model_2):
    """Print side-by-side comparison table."""
    print("\nExpense Agent Benchmark Comparison")
    print("=" * 140)
    print(
        f"{'Prompt':<45} {'Model':<28} {'Steps':<6} {'Tokens':<8} "
        f"{'Latency(s)':<11} {'Cost(USD)':<10} {'Tools':<20}"
    )
    print("-" * 140)

    for i, (r1, r2) in enumerate(zip(results_1, results_2)):
        cost_1 = f"${r1['cost_usd']:.6f}" if r1["cost_usd"] is not None else "N/A"
        cost_2 = f"${r2['cost_usd']:.6f}" if r2["cost_usd"] is not None else "N/A"
        tokens_1 = str(r1["total_tokens"]) if r1["total_tokens"] is not None else "N/A"
        tokens_2 = str(r2["total_tokens"]) if r2["total_tokens"] is not None else "N/A"
        latency_1 = f"{r1['latency_s']:.4f}" if r1["latency_s"] is not None else "N/A"
        latency_2 = f"{r2['latency_s']:.4f}" if r2["latency_s"] is not None else "N/A"
        tools_1 = ",".join(r1["tool_calls"]) if r1["tool_calls"] else "-"
        tools_2 = ",".join(r2["tool_calls"]) if r2["tool_calls"] else "-"

        print(
            f"{r1['prompt'][:43]:<45} {model_1:<28} {r1['steps']:<6} "
            f"{tokens_1:<8} {latency_1:<11} "
            f"{cost_1:<10} {tools_1:<20}"
        )
        print(
            f"{r2['prompt'][:43]:<45} {model_2:<28} {r2['steps']:<6} "
            f"{tokens_2:<8} {latency_2:<11} "
            f"{cost_2:<10} {tools_2:<20}"
        )
        print("-" * 140)

    # Summary
    valid_t1 = [r["total_tokens"] for r in results_1 if r["total_tokens"] is not None]
    valid_t2 = [r["total_tokens"] for r in results_2 if r["total_tokens"] is not None]
    total_t1 = sum(valid_t1) if valid_t1 else None
    total_t2 = sum(valid_t2) if valid_t2 else None
    valid_lat_1 = [r["latency_s"] for r in results_1 if r["latency_s"] is not None]
    valid_lat_2 = [r["latency_s"] for r in results_2 if r["latency_s"] is not None]
    total_lat_1 = sum(valid_lat_1) if valid_lat_1 else None
    total_lat_2 = sum(valid_lat_2) if valid_lat_2 else None
    total_steps_1 = sum(r["steps"] for r in results_1)
    total_steps_2 = sum(r["steps"] for r in results_2)
    cost_1_vals = [r["cost_usd"] for r in results_1 if r["cost_usd"] is not None]
    cost_2_vals = [r["cost_usd"] for r in results_2 if r["cost_usd"] is not None]
    total_c_1 = sum(cost_1_vals) if cost_1_vals else None
    total_c_2 = sum(cost_2_vals) if cost_2_vals else None

    # Format totals safely
    t1_str = str(total_t1) if total_t1 is not None else "N/A"
    t2_str = str(total_t2) if total_t2 is not None else "N/A"
    lat_1_str = f"{total_lat_1:.4f}" if total_lat_1 is not None else "N/A"
    lat_2_str = f"{total_lat_2:.4f}" if total_lat_2 is not None else "N/A"
    cost_1_str = f"${total_c_1:.6f}" if total_c_1 is not None else "N/A"
    cost_2_str = f"${total_c_2:.6f}" if total_c_2 is not None else "N/A"

    print(f"\n{'TOTAL':<45} {model_1:<28} {total_steps_1:<6} "
          f"{t1_str:<8} {lat_1_str:<11} {cost_1_str:<10}")
    print(f"{'TOTAL':<45} {model_2:<28} {total_steps_2:<6} {t2_str:<8} "
          f"{lat_2_str:<11} {cost_2_str:<10}")
    print("=" * 140)


def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmark_compare.py MODEL_1 MODEL_2")
        sys.exit(1)

    model_1 = sys.argv[1]
    model_2 = sys.argv[2]

    # Use a temporary database for isolation
    tmp_dir = tempfile.mkdtemp(prefix="benchmark_compare_")
    db_path = os.path.join(tmp_dir, "expenses.db")
    os.environ["EXPENSES_DB_PATH"] = db_path

    # Import fresh to pick up env var
    import expense_agent as ea_mod
    importlib.reload(ea_mod)

    print(f"Running benchmark for model 1: {model_1}")
    results_1 = run_single_model(model_1, db_path)

    # Re-seed fresh database for model 2
    import sqlite3
    if os.path.exists(db_path):
        os.remove(db_path)
    importlib.reload(ea_mod)
    seed_data(ea_mod)

    print(f"Running benchmark for model 2: {model_2}")
    results_2 = run_single_model(model_2, db_path)

    print_comparison(results_1, results_2, model_1, model_2)


if __name__ == "__main__":
    main()
