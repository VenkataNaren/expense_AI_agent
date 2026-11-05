"""Run a fixed set of prompts through the expense agent and measure telemetry.

Seeds a temporary database with made-up data (5 expenses, 10 KiwiSaver rows),
then runs 10 benchmark prompts. For each prompt, records the answer, steps
used, tokens, latency, and estimated cost. Saves results to benchmark_baseline.json
and prints a summary table.

Does not modify agent behavior — only reads from run_agent, llm_calls, and
request_cost.

Run: python benchmark_agent.py
"""
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime

import expense_agent as ea


PROMPTS = [
    "Add an expense: 12.50 on coffee this morning",
    "Add an expense: 45.00 on transport for a taxi ride",
    "List all my expenses",
    "What is my total for food?",
    "What is my total for transport?",
    "Delete the coffee expense I just added — confirm with y",
    "How is my KiwiSaver doing since 2026-10-15?",
    "List my KiwiSaver daily balances",
    "What is the weather like today?",
    "Delete the taxi expense — confirm with n",
]


def seed_data():
    """Add 5 expenses and 10 KiwiSaver rows with made-up data."""
    # 5 expenses with made-up amounts
    ea.add_expense("12.50", "coffee", "morning")
    ea.add_expense("45.00", "transport", "taxi ride")
    ea.add_expense("8.75", "food", "lunch")
    ea.add_expense("32.00", "entertainment", "movie tickets")
    ea.add_expense("15.60", "utilities", "internet bill")

    # 10 KiwiSaver daily rows with made-up balances
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
        ea.add_kiwisaver_daily(date, balance, units, unit_price)


def _count_llm_calls(db_path):
    """Return the number of rows in llm_calls."""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    finally:
        conn.close()


def _max_llm_call_id(db_path):
    """Return the max id in llm_calls, or 0 if empty."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT MAX(id) FROM llm_calls").fetchone()
    finally:
        conn.close()
    return row[0] or 0


def _get_new_llm_calls(db_path, before_max_id):
    """Return llm_calls rows inserted after before_max_id."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT request_id, step, model, prompt_tokens, completion_tokens,
                   latency_s, tool_name
            FROM llm_calls
            WHERE id > ?
            ORDER BY step
            """,
            (before_max_id,),
        ).fetchall()
    finally:
        conn.close()
    return rows


def run_benchmark():
    """Run all benchmark prompts and return results list."""
    results = []

    for prompt in PROMPTS:
        history = [{"role": "system", "content": ea.SYSTEM_PROMPT}]

        # Measure before
        before_max_id = _max_llm_call_id(ea.DB_PATH)

        start = __import__("time").perf_counter()
        answer, tool_calls = ea.run_agent(prompt, history)
        elapsed = __import__("time").perf_counter() - start

        # Get new llm_calls from this run
        new_calls = _get_new_llm_calls(ea.DB_PATH, before_max_id)

        steps = len(new_calls)
        prompt_tokens = sum(r[3] or 0 for r in new_calls)
        completion_tokens = sum(r[4] or 0 for r in new_calls)
        total_tokens = prompt_tokens + completion_tokens
        latency_s = round(sum(r[5] or 0 for r in new_calls), 4)

        # Cost: use the last request_id in this run
        cost_usd = None
        if new_calls:
            last_request_id = new_calls[-1][0]
            cost_usd = ea.request_cost(last_request_id)
            if cost_usd is not None:
                cost_usd = round(cost_usd, 6)

        tool_names = [tc["name"] for tc in tool_calls]

        results.append({
            "prompt": prompt,
            "answer": answer,
            "steps": steps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_s": latency_s,
            "cost_usd": cost_usd,
            "tool_calls": tool_names,
        })

    return results


def print_summary(results):
    """Print a summary table."""
    print("\nBenchmark Summary")
    print("=" * 100)
    print(
        f"{'#':<3} {'Prompt':<45} {'Steps':<6} {'Tokens':<8} {'Cost (USD)':<12} "
        f"{'Latency (s)':<12} {'Tools':<20}"
    )
    print("-" * 100)
    for i, r in enumerate(results, 1):
        cost_str = f"${r['cost_usd']}" if r["cost_usd"] is not None else "unknown"
        print(
            f"{i:<3} {r['prompt']:<45} {r['steps']:<6} {r['total_tokens']:<8} "
            f"{cost_str:<12} {r['latency_s']:<12} {', '.join(r['tool_calls']):<20}"
        )
    print("-" * 100)

    total_tokens = sum(r["total_tokens"] for r in results)
    total_latency = sum(r["latency_s"] for r in results)
    known_costs = [r["cost_usd"] for r in results if r["cost_usd"] is not None]
    total_cost = sum(known_costs) if known_costs else None
    total_steps = sum(r["steps"] for r in results)

    print(f"Total steps: {total_steps}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total latency: {total_latency:.4f} s")
    if total_cost is not None:
        print(f"Total estimated cost: ${total_cost:.6f}")
    else:
        print("Total estimated cost: unknown (some models unpriced)")


def main():
    # Use EXPENSES_DB_PATH if already set (e.g., by tests), otherwise create a temp dir
    existing_db = os.environ.get("EXPENSES_DB_PATH")
    if existing_db:
        db_path = existing_db
        tmp_dir = os.path.dirname(db_path)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="benchmark_agent_")
        db_path = os.path.join(tmp_dir, "expenses.db")
        os.environ["EXPENSES_DB_PATH"] = db_path

    # Reload expense_agent so it picks up the new DB_PATH
    import importlib
    importlib.reload(ea)

    # Seed made-up data
    seed_data()

    # Run benchmark
    results = run_benchmark()

    # Build output
    output = {
        "results": results,
        "summary": {
            "total_prompts": len(results),
            "total_steps": sum(r["steps"] for r in results),
            "total_tokens": sum(r["total_tokens"] for r in results),
            "total_latency_s": round(sum(r["latency_s"] for r in results), 4),
        },
    }

    # Write results file
    report_path = os.path.join(tmp_dir, "benchmark_baseline.json")
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print_summary(results)
    print(f"\nResults written to: {report_path}")


if __name__ == "__main__":
    main()
