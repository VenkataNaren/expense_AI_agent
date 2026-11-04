"""One-off script: import KiwiSaver daily balances from a CSV into kiwisaver_daily.

CSV columns (flexible names): a date column and a balance column.
- date format: "Thursday, 15 October 2026"
- balance format: "21,156.25" (commas allowed)

Skips dates that already exist in the database (no overwrite).
Prints a data-quality report: days where the balance is identical to the
previous day (probably not refreshed). Does NOT delete or modify any rows.

Run: python import_kiwisaver.py kiwisaver.csv
"""
import csv
import sqlite3
import sys
from datetime import datetime

from dotenv import load_dotenv

import expense_agent as ea


def _find_column(headers, candidates):
    """Return the first header that matches any candidate (case-insensitive, contains)."""
    lower = [h.strip().lower() for h in headers]
    for cand in candidates:
        cand = cand.lower()
        for i, h in enumerate(lower):
            if cand in h:
                return headers[i]
    return None


def parse_date(value):
    """Parse 'Thursday, 15 October 2026' -> 'YYYY-MM-DD'. Returns None on failure."""
    value = value.strip()
    try:
        return datetime.strptime(value, "%A, %d %B %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_balance(value):
    """Parse '21,156.25' -> float. Returns None on failure."""
    value = value.strip().replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def main(csv_path):
    load_dotenv()

    # Read CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        date_col = _find_column(headers, ["date"])
        balance_col = _find_column(headers, ["balance", "kiwi", "kivi"])
        if not date_col or not balance_col:
            print(f"Error: Could not find date and balance columns in {headers}")
            sys.exit(1)

        rows = []
        for row in reader:
            date_str = parse_date(row[date_col])
            balance = parse_balance(row[balance_col])
            if date_str is None:
                print(f"Skipping row with unparseable date: {row}")
                continue
            if balance is None:
                print(f"Skipping row with unparseable balance: {row}")
                continue
            rows.append((date_str, balance))

    rows.sort(key=lambda r: r[0])

    # Connect to the same DB as expense_agent
    conn = sqlite3.connect(ea.DB_PATH)
    try:
        existing_dates = {
            row[0] for row in conn.execute("SELECT date FROM kiwisaver_daily").fetchall()
        }
        latest = conn.execute(
            "SELECT date, balance FROM kiwisaver_daily ORDER BY date DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    # Insert new rows, skipping existing dates
    inserted = []
    skipped = 0
    for date_str, balance in rows:
        if date_str in existing_dates:
            print(f"Skipping existing date: {date_str}")
            skipped += 1
            continue
        conn = sqlite3.connect(ea.DB_PATH)
        try:
            conn.execute(
                "INSERT INTO kiwisaver_daily (date, balance) VALUES (?, ?)",
                (date_str, balance),
            )
            conn.commit()
        finally:
            conn.close()
        inserted.append((date_str, balance))

    # Data-quality report: balance identical to previous day
    # Build sequence: latest existing DB row + inserted rows, ordered by date
    sequence = []
    if latest:
        sequence.append((latest[0], latest[1]))
    sequence.extend(inserted)

    print("\nData-quality report: days where balance is identical to previous day")
    print("-" * 60)
    found = False
    for i in range(1, len(sequence)):
        prev_date, prev_balance = sequence[i - 1]
        curr_date, curr_balance = sequence[i]
        if curr_balance == prev_balance:
            print(
                f"{curr_date}: balance {curr_balance:,.2f} same as "
                f"{prev_date} ({prev_balance:,.2f})"
            )
            found = True
    if not found:
        print("(none)")

    print(f"\nImported {len(inserted)} new row(s). Skipped {skipped} existing date(s).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python import_kiwisaver.py <csv_file>")
        sys.exit(1)
    main(sys.argv[1])
