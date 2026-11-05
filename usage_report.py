"""Usage report for LLM calls.

Prints, per request and in total:
  - steps (number of model calls)
  - total tokens (prompt + completion)
  - estimated cost (USD, using PRICING from expense_agent)
  - average latency (seconds)
  - slowest call (max latency)

Models not in PRICING show "unknown" for cost instead of zero.

Run: python usage_report.py
"""
import sqlite3
from collections import defaultdict

import expense_agent as ea


def main():
    conn = sqlite3.connect(ea.DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT request_id, step, model, prompt_tokens, completion_tokens,
                   latency_s
            FROM llm_calls
            ORDER BY request_id, step
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No LLM calls recorded.")
        return

    # Group by request_id
    by_request = defaultdict(list)
    for request_id, step, model, prompt_tokens, completion_tokens, latency_s in rows:
        by_request[request_id].append(
            (step, model, prompt_tokens, completion_tokens, latency_s)
        )

    total_steps = 0
    total_tokens = 0
    total_cost = 0.0
    total_cost_known = True
    max_latency = 0.0
    slowest_request = None
    slowest_step = None

    print(f"{'Request ID':<20} {'Steps':<7} {'Tokens':<9} {'Cost (USD)':<12} {'Avg Latency (s)':<16} {'Slowest Call (s)':<17}")
    print("-" * 85)

    for request_id in sorted(by_request.keys()):
        calls = by_request[request_id]
        steps = len(calls)
        tokens = sum((pt or 0) + (ct or 0) for _, _, pt, ct, _ in calls)
        latencies = [ls or 0.0 for *_, ls in calls]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        slowest_in_request = max(latencies) if latencies else 0.0

        # Cost
        model = calls[0][1]
        pricing = ea.PRICING.get(model)
        if pricing is None:
            cost_str = "unknown"
            total_cost_known = False
        else:
            cost = sum((pt or 0) * pricing["input_per_1m"] / 1_000_000
                       + (ct or 0) * pricing["output_per_1m"] / 1_000_000
                       for _, _, pt, ct, _ in calls)
            cost_str = f"${cost:.6f}"
            total_cost += cost

        total_steps += steps
        total_tokens += tokens

        if slowest_in_request > max_latency:
            max_latency = slowest_in_request
            slowest_request = request_id
            slowest_step = max(c[0] for c in calls if c[4] == slowest_in_request)

        print(
            f"{request_id:<20} {steps:<7} {tokens:<9} {cost_str:<12} "
            f"{avg_latency:<16.4f} {slowest_in_request:<17.4f}"
        )

    print("-" * 85)
    print(f"{'TOTAL':<20} {total_steps:<7} {total_tokens:<9}", end="")
    if total_cost_known:
        print(f" ${total_cost:.6f}")
    else:
        print(" unknown")
    print(f"Average latency across all calls: {sum(ls or 0.0 for *_, ls in rows) / len(rows):.4f} s")
    if slowest_request is not None:
        print(f"Slowest call: request {slowest_request}, step {slowest_step}, {max_latency:.4f} s")


if __name__ == "__main__":
    main()
