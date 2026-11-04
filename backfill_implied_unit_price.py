"""Backfill implied unit prices into kiwisaver_daily from a CSV.

For each row:
  implied_unit_price = balance / assumed_units, rounded to 4 decimals (Decimal)

Uses Decimal throughout for money precision.

Default mode is --dry-run (prints what would change, writes nothing).
Use --apply to write changes. Before writing, copies the database to a
timestamped backup (expenses.db.bak-YYYYMMDD-HHMMSS).

CSV columns: "Date" and "KIWI Saver" (exact match first, substring fallback).
Date format: "Thursday, 15 October 2026". Balance format: "21,156.25" (commas).

Rows already marked price_source='actual' are never overwritten.

Run:
  python backfill_implied_unit_price.py kiwisaver.csv          # dry-run
  python backfill_implied_unit_price.py --apply kiwisaver.csv  # write changes
  python backfill_implied_unit_price.py --assumed-units 1000.0 --apply kiwisaver.csv
"""
import argparse
import csv
import shutil
import sqlite3
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from dotenv import load_dotenv

import expense_agent as ea

ASSUMED_UNITS_DEFAULT = "6952.3989"
PRICE_QUANTIZE = Decimal("0.0001")


def _find_column(headers, candidates):
    """Return the first header that matches any candidate.

    Exact match (case-insensitive) first, then substring (case-insensitive
    contains) as a fallback.
    """
    lower = [h.strip().lower() for h in headers]
    # Exact match first
    for cand in candidates:
        cand = cand.lower()
        for i, h in enumerate(lower):
            if h == cand:
                return headers[i]
    # Substring fallback
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
    """Parse '21,156.25' -> Decimal. Returns None on failure."""
    value = value.strip().replace(",", "")
    try:
        return Decimal(value)
    except Exception:
        return None


def _ensure_price_source_column(conn):
    """Add nullable price_source column to kiwisaver_daily if missing."""
    columns = [row[1] for row in conn.execute("PRAGMA table_info(kiwisaver_daily)").fetchall()]
    if "price_source" not in columns:
        conn.execute("ALTER TABLE kiwisaver_daily ADD COLUMN price_source TEXT")
        conn.commit()


def _get_existing_rows(conn):
    """Return dict of date -> (unit_price, units, price_source)."""
    rows = conn.execute(
        "SELECT date, unit_price, units, price_source FROM kiwisaver_daily"
    ).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def _backup_database(db_path):
    """Copy the database file to a timestamped backup."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{db_path}.bak-{timestamp}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def _compute_implied_price(balance, assumed_units):
    """balance / assumed_units, rounded to 4 decimals using Decimal."""
    if assumed_units == 0:
        raise ValueError("assumed_units must not be zero")
    return (balance / assumed_units).quantize(PRICE_QUANTIZE, rounding=ROUND_HALF_UP)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backfill implied unit prices into kiwisaver_daily from a CSV."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "--assumed-units",
        default=ASSUMED_UNITS_DEFAULT,
        help=f"Assumed units for all rows (default: {ASSUMED_UNITS_DEFAULT})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to the database (default is dry-run, writes nothing)",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    # Parse assumed units as Decimal
    try:
        assumed_units = Decimal(args.assumed_units)
    except Exception:
        print(f"Error: --assumed-units must be a number, got '{args.assumed_units}'")
        sys.exit(1)
    if assumed_units == 0:
        print("Error: --assumed-units must not be zero")
        sys.exit(1)

    # Read CSV
    try:
        with open(args.csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            date_col = _find_column(headers, ["date"])
            balance_col = _find_column(headers, ["balance", "kiwi", "kivi"])
            if not date_col or not balance_col:
                print(f"Error: Could not find date and balance columns in {headers}")
                sys.exit(1)

            csv_rows = []
            bad_rows = []
            for row in reader:
                date_str = parse_date(row[date_col])
                balance = parse_balance(row[balance_col])
                if date_str is None:
                    bad_rows.append((row, "unparseable date"))
                    continue
                if balance is None:
                    bad_rows.append((row, "unparseable balance"))
                    continue
                csv_rows.append((date_str, balance))
    except FileNotFoundError:
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)

    csv_rows.sort(key=lambda r: r[0])

    # Connect to the same DB as expense_agent
    conn = sqlite3.connect(ea.DB_PATH)
    try:
        _ensure_price_source_column(conn)
        existing = _get_existing_rows(conn)
    finally:
        conn.close()

    # Plan actions
    planned = []  # (date, balance, implied_price, action, existing_info)
    for date_str, balance in csv_rows:
        implied_price = _compute_implied_price(balance, assumed_units)
        if date_str in existing:
            unit_price, units, price_source = existing[date_str]
            if price_source == "actual":
                planned.append(
                    (date_str, balance, implied_price, "skip", "price_source=actual")
                )
            elif unit_price is None:
                planned.append(
                    (date_str, balance, implied_price, "update", f"unit_price=NULL")
                )
            else:
                # unit_price present but price_source not 'actual' -> update
                planned.append(
                    (date_str, balance, implied_price, "update", f"unit_price={unit_price}")
                )
        else:
            planned.append(
                (date_str, balance, implied_price, "insert", "new row")
            )

    if args.apply:
        # Backup before writing
        backup_path = _backup_database(ea.DB_PATH)
        print(f"Backed up database to: {backup_path}")

        conn = sqlite3.connect(ea.DB_PATH)
        try:
            inserted = 0
            updated = 0
            skipped = 0
            for date_str, balance, implied_price, action, info in planned:
                if action == "skip":
                    skipped += 1
                    continue
                if action == "insert":
                    conn.execute(
                        """
                        INSERT INTO kiwisaver_daily
                            (date, balance, unit_price, units, price_source)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            date_str,
                            float(balance),
                            float(implied_price),
                            float(assumed_units),
                            "implied",
                        ),
                    )
                    inserted += 1
                elif action == "update":
                    conn.execute(
                        """
                        UPDATE kiwisaver_daily
                        SET unit_price = ?, units = ?, price_source = 'implied'
                        WHERE date = ?
                        """,
                        (float(implied_price), float(assumed_units), date_str),
                    )
                    updated += 1
            conn.commit()
        finally:
            conn.close()

        print(f"Inserted {inserted} new row(s).")
        print(f"Updated {updated} existing row(s).")
        print(f"Skipped {skipped} row(s) (price_source=actual).")
    else:
        print("DRY-RUN: no changes will be written. Use --apply to write.")
        for date_str, balance, implied_price, action, info in planned:
            print(
                f"  {date_str} | balance {balance} | implied price {implied_price} | "
                f"action: {action} ({info})"
            )

    # Report bad CSV rows (in both modes)
    if bad_rows:
        print("\nBad CSV rows (skipped, not inserted):")
        for row, reason in bad_rows:
            print(f"  {row} -- {reason}")

    # Data-quality report: stale days
    print("\nData-quality report: days where balance is identical to previous day")
    print("-" * 60)
    # Build sequence from all rows in DB (after potential write)
    conn = sqlite3.connect(ea.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT date, balance FROM kiwisaver_daily ORDER BY date"
        ).fetchall()
    finally:
        conn.close()

    found = False
    for i in range(1, len(rows)):
        prev_date, prev_balance = rows[i - 1]
        curr_date, curr_balance = rows[i]
        if curr_balance == prev_balance:
            print(
                f"{curr_date}: balance {curr_balance:,.2f} same as "
                f"{prev_date} ({prev_balance:,.2f})"
            )
            found = True
    if not found:
        print("(none)")


if __name__ == "__main__":
    main()
