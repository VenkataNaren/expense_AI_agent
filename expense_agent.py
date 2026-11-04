"""
Expense Helper: a simple single-agent project.

The agent can add expenses, list them, and total them by category.
Uses an NVIDIA-hosted model through the OpenAI-compatible API.

Setup:
    pip install openai python-dotenv
    Create a .env file next to this script (see .env.example).
"""
import json
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ---------- 1. Connect to a model (the "brain") ----------
# Settings come from a .env file in the same folder (never put keys in code).
load_dotenv()

client = OpenAI(
    base_url=os.environ["NVIDIA_BASE_URL"],
    api_key=os.environ["NVIDIA_API_KEY"],
)
MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")

MAX_STEPS = 5  # safety limit so the loop can never run forever


# ---------- 2. Our tiny "database" ----------
DB_PATH = os.environ.get("EXPENSES_DB_PATH", "expenses.db")


def _connect():
    """Open a fresh SQLite connection for a single operation."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the tables and run safe migrations. Called once at import."""
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                note TEXT DEFAULT '',
                spent_at TEXT
            )
            """
        )

        # Safe migration: add spent_at column to existing tables that lack it,
        # and backfill old rows with today's date at 00:00.
        columns = [row[1] for row in conn.execute("PRAGMA table_info(expenses)").fetchall()]
        if "spent_at" not in columns:
            conn.execute("ALTER TABLE expenses ADD COLUMN spent_at TEXT")
            today = datetime.now().strftime("%Y-%m-%d")
            conn.execute(
                "UPDATE expenses SET spent_at = ? WHERE spent_at IS NULL OR spent_at = ''",
                (f"{today} 00:00",),
            )

        # KiwiSaver daily balances (one row per date).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kiwisaver_daily (
                date TEXT UNIQUE,
                balance REAL NOT NULL,
                units REAL,
                unit_price REAL
            )
            """
        )

        # KiwiSaver adjustments (employee, employer, government, tax, fee).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kiwisaver_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                amount REAL,
                type TEXT
            )
            """
        )

        # Safe migration: add any missing columns to the KiwiSaver tables if an
        # older version of the table already exists.
        for table, expected in (
            ("kiwisaver_daily", {"date": "TEXT", "balance": "REAL NOT NULL", "units": "REAL", "unit_price": "REAL", "price_source": "TEXT"}),
            ("kiwisaver_adjustments", {"date": "TEXT", "amount": "REAL", "type": "TEXT"}),
        ):
            existing = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            for col, coltype in expected.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")

        conn.commit()
    finally:
        conn.close()


# Run once when the module is imported.
init_db()

# ---------- 3. Tools (plain Python functions the agent may call) ----------
def format_time_12h(stored_time):
    """Convert stored 'YYYY-MM-DD HH:MM' to 'YYYY-MM-DD hh:MM AM/PM'."""
    if not stored_time:
        return ""
    try:
        dt = datetime.strptime(stored_time, "%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%d %I:%M %p")
    except ValueError:
        return stored_time


def _parse_spent_at(value):
    """Parse accepted date/time formats and return (datetime, error_string)."""
    if value is None or value == "":
        return datetime.now(), None

    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %I:%M %p", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt), None
        except ValueError:
            continue

    return None, (
        f"Error: Invalid date/time '{value}'. "
        "Use 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD hh:MM AM/PM', or 'YYYY-MM-DD'."
    )


def add_expense(amount, category, note="", spent_at=None, confirm=None):
    """Record a new expense, with confirmation guardrail for future timestamps."""
    dt, error = _parse_spent_at(spent_at)
    if error:
        return error

    confirm = (confirm or "").strip().lower()
    now = datetime.now()

    if dt > now:
        details = {
            "amount": float(amount),
            "category": category.lower(),
            "note": note,
            "spent_at": dt.strftime("%Y-%m-%d %H:%M"),
            "display_amount": amount,
            "display_category": category,
        }

        if confirm in ("y", "yes"):
            _PENDING_ADD = {"details": None}
            return _perform_add(details)

        if confirm in ("n", "no"):
            _PENDING_ADD = {"details": None}
            return "Expense addition cancelled. No changes were made."

        _PENDING_ADD = {"details": details}
        return (
            f"Found a future-dated expense to add:\n"
            f"{details['display_amount']} - {details['category']} - {details['note']} - "
            f"{format_time_12h(details['spent_at'])}\n\n"
            "The date/time is in the future. Are you sure you want to add this record? (y/n)"
        )

    details = {
        "amount": float(amount),
        "category": category.lower(),
        "note": note,
        "spent_at": dt.strftime("%Y-%m-%d %H:%M"),
        "display_amount": amount,
        "display_category": category,
    }
    return _perform_add(details)


def _perform_add(details):
    """Insert an expense row into the database."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO expenses (amount, category, note, spent_at) VALUES (?, ?, ?, ?)",
            (details["amount"], details["category"], details["note"], details["spent_at"]),
        )
        conn.commit()
    finally:
        conn.close()
    return f"Added {details['display_amount']} to {details['display_category']}."


def list_expenses():
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT amount, category, note, spent_at FROM expenses ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No expenses recorded."
    lines = [
        f"{amount} - {category} - {note} - {format_time_12h(spent_at)}"
        for amount, category, note, spent_at in rows
    ]
    return "\n".join(lines)


def total_by_category(category):
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE category = ?",
            (category.lower(),),
        ).fetchone()
    finally:
        conn.close()
    total = row[0]
    return f"Total for {category}: {total}"


def delete_expense(id=None, note=None, amount=None, spent_at=None):
    """Delete expense(s) matching all provided criteria (id, note, amount, spent_at)."""
    conditions = []
    params = []
    if id is not None:
        conditions.append("id = ?")
        params.append(int(id))
    if note is not None:
        conditions.append("note = ?")
        params.append(note)
    if amount is not None:
        conditions.append("amount = ?")
        params.append(float(amount))
    if spent_at is not None:
        dt, error = _parse_spent_at(spent_at)
        if error:
            return error
        conditions.append("spent_at = ?")
        params.append(dt.strftime("%Y-%m-%d %H:%M"))

    if not conditions:
        return (
            "Error: Please specify at least one of id, note, amount, or spent_at "
            "to identify the expense to delete."
        )

    where = " AND ".join(conditions)
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT id, amount, category, note, spent_at FROM expenses WHERE {where}",
            params,
        ).fetchall()
        if not rows:
            return "No matching expense found."
        conn.execute(f"DELETE FROM expenses WHERE {where}", params)
        conn.commit()
    finally:
        conn.close()

    deleted = [
        f"id {r[0]}: {r[1]} - {r[2]} - {r[3]} - {format_time_12h(r[4])}" for r in rows
    ]
    return "Deleted:\n" + "\n".join(deleted)


# Pending deletion state for the y/n confirmation guardrail.
# Stores the matching rows until the user confirms with "y"/"yes".
_PENDING_DELETE = {"criteria": None, "rows": None}

# Pending add state for future-dated expenses requiring confirmation.
# Stores the expense details until the user confirms with "y"/"yes".
_PENDING_ADD = {"details": None}

# Pending kiwisaver daily overwrite state for the y/n confirmation guardrail.
# Stores the proposed daily balance details until the user confirms with "y"/"yes".
_PENDING_KIWISAVER = {"details": None}


def delete_expense(id=None, note=None, amount=None, spent_at=None, confirm=None):
    """Delete expense(s) matching all provided criteria, with y/n confirmation guardrail.

    Phase 1: Call without confirm (or confirm != "y"/"yes") to preview matches and
    request confirmation. No rows are deleted.
    Phase 2: Call with confirm="y" to actually delete the pending matches.
    Call with confirm="n" to cancel.
    """
    global _PENDING_DELETE
    confirm = (confirm or "").strip().lower()

    conditions = []
    params = []
    if id is not None:
        conditions.append("id = ?")
        params.append(int(id))
    if note is not None:
        conditions.append("note = ?")
        params.append(note)
    if amount is not None:
        conditions.append("amount = ?")
        params.append(float(amount))
    if spent_at is not None:
        dt, error = _parse_spent_at(spent_at)
        if error:
            return error
        conditions.append("spent_at = ?")
        params.append(dt.strftime("%Y-%m-%d %H:%M"))

    if not conditions:
        return (
            "Error: Please specify at least one of id, note, amount, or spent_at "
            "to identify the expense to delete."
        )

    where = " AND ".join(conditions)
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT id, amount, category, note, spent_at FROM expenses WHERE {where}",
            params,
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No matching expense found."

    details = [
        f"ID {r[0]}: {r[1]} - {r[2]} - {r[3]} - {format_time_12h(r[4])}" for r in rows
    ]
    preview = "\n".join(details)

    if confirm in ("y", "yes"):
        pending = _PENDING_DELETE
        if pending["criteria"] is not None:
            # Apply the previewed (stored) criteria, not the current call's args.
            where = pending["where"]
            params = pending["params"]
            rows = pending["rows"]
            preview = pending["preview"]
            _PENDING_DELETE = {"criteria": None, "rows": None}
            # Fall through to the delete block below
        # No pending preview: fall through to delete with current args below

    # Delete block — runs for both stored criteria and immediate deletion
    if confirm in ("y", "yes"):
        conn = _connect()
        try:
            conn.execute(f"DELETE FROM expenses WHERE {where}", params)
            conn.commit()
        finally:
            conn.close()
        return f"Deleted {len(rows)} record(s):\n{preview}"

    if confirm in ("n", "no"):
        _PENDING_DELETE = {"criteria": None, "rows": None}
        return "Deletion cancelled. No changes were made."

    # Pending confirmation: store state, do NOT delete.
    _PENDING_DELETE = {
        "criteria": (id, note, amount, spent_at),
        "rows": rows,
        "where": where,
        "params": params,
        "preview": preview,
    }
    return (
        f"Found {len(rows)} matching expense(s) to delete:\n{preview}\n\n"
        "Are you sure you want to delete this record? (y/n)"
    )


def _validate_kiwisaver_type(value):
    """Return (value, None) if valid, else (None, error_string)."""
    allowed = ("employee", "employer", "government", "tax", "fee")
    if value not in allowed:
        return None, (
            f"Error: Invalid kiwisaver type '{value}'. "
            f"Use one of: {', '.join(allowed)}."
        )
    return value, None


def _parse_kiwisaver_date(value):
    """Parse a strict YYYY-MM-DD date and return (date_str, error_string)."""
    if value is None or value == "":
        return None, "Error: Date is required."
    value = value.strip()
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None, "Error: Invalid date format. Use 'YYYY-MM-DD'."
    return dt.strftime("%Y-%m-%d"), None


def _get_previous_day_balance(date_str):
    """Return the balance for the most recent date before date_str, or None."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT balance FROM kiwisaver_daily WHERE date < ? ORDER BY date DESC LIMIT 1",
            (date_str,),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def add_kiwisaver_daily(date, balance, units=None, unit_price=None, confirm=None):
    """Insert or replace a Kiwisaver daily balance row with validation and confirm guardrail."""
    global _PENDING_KIWISAVER
    # Validate balance
    try:
        balance = float(balance)
    except (TypeError, ValueError):
        return "Error: balance must be a number."
    if balance <= 0:
        return "Error: balance must be greater than 0."

    # Validate date format (strict YYYY-MM-DD)
    date_str, error = _parse_kiwisaver_date(date)
    if error:
        return error

    # Reject future dates
    if datetime.strptime(date_str, "%Y-%m-%d").date() > datetime.now().date():
        return "Error: Date is in the future."

    # Validate units and unit_price are numbers
    if units is not None:
        try:
            units = float(units)
        except (TypeError, ValueError):
            return "Error: units must be a number."
    if unit_price is not None:
        try:
            unit_price = float(unit_price)
        except (TypeError, ValueError):
            return "Error: unit_price must be a number."

    warnings = []

    # Check units * unit_price within 1% of balance
    if units is not None and unit_price is not None:
        implied = units * unit_price
        if implied != balance:
            pct_diff = abs(implied - balance) / balance
            if pct_diff > 0.01:
                warnings.append(
                    f"Warning: units * unit_price ({implied:.2f}) differs from balance ({balance:.2f}) by more than 1%."
                )

    # Check if date already exists
    conn = _connect()
    try:
        existing = conn.execute(
            "SELECT balance, units, unit_price FROM kiwisaver_daily WHERE date = ?",
            (date_str,),
        ).fetchone()
    finally:
        conn.close()

    # 5% change warning from previous day's balance
    prev_balance = _get_previous_day_balance(date_str)
    if prev_balance is not None and prev_balance != 0:
        pct_change = abs(balance - prev_balance) / prev_balance
        if pct_change > 0.05:
            warnings.append(
                f"Warning: balance changed by more than 5% from previous day "
                f"({prev_balance:.2f} -> {balance:.2f})."
            )

    confirm = (confirm or "").strip().lower()

    if existing is not None:
        old_balance = existing[0]

        if confirm in ("y", "yes"):
            pending = _PENDING_KIWISAVER["details"]
            if pending is None:
                return (
                    "Error: No pending kiwisaver overwrite to confirm. "
                    "Call without confirm first to preview."
                )
            # Apply the previewed (stored) values, not the current call's args.
            _PENDING_KIWISAVER = {"details": None}
            _perform_kiwisaver_add(
                pending["date"], pending["balance"], pending["units"], pending["unit_price"]
            )
            stored_warnings = pending.get("_warnings", [])
            warning_text = " ".join(stored_warnings) + "\n" if stored_warnings else ""
            msg = (
                f"{warning_text}"
                f"Kiwisaver daily balance for {pending['date']}: "
                f"{pending['balance']} (overwrote {pending['old_balance']})."
            )
            return msg

        if confirm in ("n", "no"):
            _PENDING_KIWISAVER = {"details": None}
            return "Kiwisaver daily balance update cancelled. No changes were made."

        # Pending confirmation: store state, do NOT insert.
        # Capture warnings computed at preview time so confirm can replay them.
        preview_warnings = list(warnings)
        _PENDING_KIWISAVER = {
            "details": {
                "date": date_str,
                "balance": balance,
                "units": units,
                "unit_price": unit_price,
                "old_balance": old_balance,
                "old_units": existing[1],
                "old_unit_price": existing[2],
                "_warnings": preview_warnings,
            }
        }
        warning_text = " ".join(warnings) + "\n" if warnings else ""
        return (
            f"{warning_text}"
            f"Found existing Kiwisaver daily balance for {date_str}: {old_balance}.\n"
            f"New value: {balance}.\n"
            "Are you sure you want to overwrite? (y/n)"
        )

    # No existing date: insert
    _perform_kiwisaver_add(date_str, balance, units, unit_price)
    msg = f"Kiwisaver daily balance for {date_str}: {balance}."
    if warnings:
        msg = " ".join(warnings) + "\n" + msg
    return msg


def _perform_kiwisaver_add(date_str, balance, units, unit_price):
    """Insert or replace a Kiwisaver daily balance row."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO kiwisaver_daily (date, balance, units, unit_price)
            VALUES (?, ?, ?, ?)
            """,
            (date_str, balance, units, unit_price),
        )
        conn.commit()
    finally:
        conn.close()


def list_kiwisaver_daily():
    """Return all KiwiSaver daily balance rows."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT date, balance, units, unit_price FROM kiwisaver_daily ORDER BY date"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No Kiwisaver daily balances recorded."
    lines = [
        f"{date} - balance {balance} - units {units} - unit_price {unit_price}"
        for date, balance, units, unit_price in rows
    ]
    return "\n".join(lines)


def add_kiwisaver_adjustment(date, amount, type):
    """Insert a KiwiSaver adjustment row. type must be employee, employer, government, tax, or fee."""
    type, error = _validate_kiwisaver_type(type)
    if error:
        return error
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return "Error: amount must be a number."
    if amount == 0:
        return "Error: amount must not be zero."

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO kiwisaver_adjustments (date, amount, type)
            VALUES (?, ?, ?)
            """,
            (date, amount, type),
        )
        conn.commit()
    finally:
        conn.close()
    return f"Kiwisaver adjustment {type} {amount} on {date}."


def list_kiwisaver_adjustments():
    """Return all KiwiSaver adjustment rows."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, date, amount, type FROM kiwisaver_adjustments ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No Kiwisaver adjustments recorded."
    lines = [
        f"id {r[0]}: {r[1]} - {r[2]} - {r[3]}" for r in rows
    ]
    return "\n".join(lines)


def _build_kiwisaver_series(start_date=None, end_date=None):
    """Build the KiwiSaver series for a date range.

    Picks the basis: unit_price if present on both the first and last row
    (ignores contributions), otherwise balance minus cumulative adjustments
    up to each date. Returns (series, metadata) or (None, error_string).
    """
    # Validate dates if provided
    if start_date is not None:
        start_date, error = _parse_kiwisaver_date(start_date)
        if error:
            return None, error
    if end_date is not None:
        end_date, error = _parse_kiwisaver_date(end_date)
        if error:
            return None, error
    if start_date is not None and end_date is not None and start_date > end_date:
        return None, "Error: start_date must be on or before end_date."

    conditions = []
    params = []
    if start_date is not None:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date is not None:
        conditions.append("date <= ?")
        params.append(end_date)
    where = " AND ".join(conditions)
    if where:
        where = "WHERE " + where

    conn = _connect()
    try:
        rows = conn.execute(
            f"""
            SELECT date, balance, units, unit_price, price_source
            FROM kiwisaver_daily
            {where}
            ORDER BY date
            """,
            params,
        ).fetchall()
        adjustments = conn.execute(
            f"""
            SELECT date, amount
            FROM kiwisaver_adjustments
            {where}
            ORDER BY date
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        if start_date is None and end_date is None:
            return None, "No Kiwisaver daily balances recorded."
        return None, "No Kiwisaver data in the given date range."

    # Build cumulative adjustment sums per date (balance basis only).
    adj_by_date = {}
    for adj_date, amount in adjustments:
        adj_by_date[adj_date] = adj_by_date.get(adj_date, 0.0) + float(amount)

    sorted_adj_dates = sorted(adj_by_date.keys())
    cumulative_adj = 0.0
    adj_index = 0
    adj_cumulative_by_date = {}
    for date, _balance, _units, _unit_price, _price_source in rows:
        while adj_index < len(sorted_adj_dates) and sorted_adj_dates[adj_index] <= date:
            cumulative_adj += adj_by_date[sorted_adj_dates[adj_index]]
            adj_index += 1
        adj_cumulative_by_date[date] = cumulative_adj

    # Determine basis: unit_price if present on both ends.
    first_unit_price = rows[0][3]
    last_unit_price = rows[-1][3]
    use_unit_price = first_unit_price is not None and last_unit_price is not None

    series = []
    prev_basis_value = None
    prev_raw_balance = None
    stale_days = []
    daily_changes = []  # (date, pct)
    skipped_rows = []
    peak = None
    max_drawdown = 0.0
    max_drawdown_date = None
    any_implied = False

    for date, balance, units, unit_price, price_source in rows:
        if price_source == "implied":
            any_implied = True
        if use_unit_price:
            basis_value = unit_price
            if unit_price is None:
                skipped_rows.append(date)
                series.append(
                    {
                        "date": date,
                        "basis_value": None,
                        "raw_balance": balance,
                        "raw_unit_price": None,
                        "daily_pct_change": "N/A",
                        "is_stale": False,
                    }
                )
                # Keep the last valid basis_value so a later row can still
                # compute a % change across the gap.
                prev_raw_balance = balance
                continue
        else:
            basis_value = balance - adj_cumulative_by_date.get(date, 0.0)

        # Daily % change vs previous row.
        if (
            prev_basis_value is not None
            and basis_value is not None
            and prev_basis_value != 0
        ):
            pct = ((basis_value - prev_basis_value) / prev_basis_value) * 100
            daily_pct_change = f"{pct:+.2f}%"
            daily_changes.append((date, pct))
        else:
            daily_pct_change = "N/A"

        # Stale day: raw balance unchanged from previous row (independent of basis).
        is_stale = prev_raw_balance is not None and balance == prev_raw_balance
        if is_stale:
            stale_days.append(date)

        # Max drawdown: peak-to-trough decline on basis value.
        if basis_value is not None:
            if peak is None or basis_value > peak:
                peak = basis_value
            drawdown = ((peak - basis_value) / peak) * 100 if peak else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_date = date

        series.append(
            {
                "date": date,
                "basis_value": basis_value,
                "raw_balance": balance,
                "raw_unit_price": unit_price,
                "daily_pct_change": daily_pct_change,
                "is_stale": is_stale,
            }
        )
        prev_basis_value = basis_value
        prev_raw_balance = balance

    start_value = series[0]["basis_value"]
    end_value = series[-1]["basis_value"]

    if start_value is not None and start_value != 0 and end_value is not None:
        pct_change = ((end_value - start_value) / start_value) * 100
    else:
        pct_change = 0.0

    best_day = max(daily_changes, key=lambda x: x[1]) if daily_changes else None
    worst_day = min(daily_changes, key=lambda x: x[1]) if daily_changes else None

    basis_str = "unit price" if use_unit_price else "balance (adjusted for contributions)"

    metadata = {
        "start_value": start_value,
        "end_value": end_value,
        "pct_change": pct_change,
        "best_day": best_day,
        "worst_day": worst_day,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_drawdown_date,
        "num_days": len(series),
        "stale_days": stale_days,
        "skipped_rows": skipped_rows,
        "basis": basis_str,
        "use_unit_price": use_unit_price,
        "any_implied": any_implied,
    }

    return series, metadata


def list_kiwisaver(start_date=None, end_date=None):
    """List KiwiSaver daily balances in date order with daily % change."""
    series_list, meta = _build_kiwisaver_series(start_date, end_date)
    if series_list is None:
        # meta is the error string
        return meta

    lines = []
    for entry in series_list:
        basis_display = (
            f"{entry['basis_value']}" if entry["basis_value"] is not None else ""
        )
        lines.append(
            f"{entry['date']} - balance {basis_display} - change {entry['daily_pct_change']}"
        )

    if meta["skipped_rows"]:
        lines.append(
            f"Warning: skipped rows with missing unit price: {', '.join(meta['skipped_rows'])}"
        )

    if meta["any_implied"]:
        lines.append("Note: unit prices are estimated from balance and assumed units.")

    return "\n".join(lines)


def kiwisaver_performance(start_date, end_date):
    """Compute Kiwisaver performance between start_date and end_date inclusive.

    If unit_price exists on both ends, bases the return on unit price
    (ignores contributions). Otherwise uses balance minus cumulative
    adjustments up to each date and flags the result as an estimate.
    """
    series_list, meta = _build_kiwisaver_series(start_date, end_date)
    if series_list is None:
        # meta is the error string
        return meta

    lines = [
        f"Kiwisaver performance from {start_date} to {end_date}:",
        f"Basis: {meta['basis']}",
        f"Start value: {meta['start_value']:,.2f}",
        f"End value: {meta['end_value']:,.2f}",
        f"Change: {meta['pct_change']:+.2f}%",
        f"Days: {meta['num_days']}",
    ]
    if meta["best_day"]:
        lines.append(f"Best day: {meta['best_day'][0]} ({meta['best_day'][1]:+.2f}%)")
    if meta["worst_day"]:
        lines.append(f"Worst day: {meta['worst_day'][0]} ({meta['worst_day'][1]:+.2f}%)")
    if meta["max_drawdown_date"] is not None:
        lines.append(
            f"Max drawdown: {meta['max_drawdown']:.2f}% (around {meta['max_drawdown_date']})"
        )
    if meta["stale_days"]:
        lines.append(
            f"Stale days (unchanged balance): {', '.join(meta['stale_days'])}"
        )
    if meta["skipped_rows"]:
        lines.append(
            f"Warning: skipped rows with missing unit price: {', '.join(meta['skipped_rows'])}"
        )
    if not meta["use_unit_price"]:
        lines.append(
            "Note: Result is an estimate based on balance; unit price not available on both ends."
        )
    if meta["any_implied"]:
        lines.append(
            "Note: unit prices are estimated from balance and assumed units."
        )

    return "\n".join(lines)


TOOL_FUNCTIONS = {
    "add_expense": add_expense,
    "list_expenses": list_expenses,
    "total_by_category": total_by_category,
    "delete_expense": delete_expense,
    "add_kiwisaver_daily": add_kiwisaver_daily,
    "list_kiwisaver_daily": list_kiwisaver_daily,
    "add_kiwisaver_adjustment": add_kiwisaver_adjustment,
    "list_kiwisaver_adjustments": list_kiwisaver_adjustments,
    "list_kiwisaver": list_kiwisaver,
    "kiwisaver_performance": kiwisaver_performance,
}

# The "menu" the model sees: name + description + inputs for each tool
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_expense",
            "description": "Record a new expense.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount spent"},
                    "category": {"type": "string", "description": "e.g. food, travel"},
                    "note": {"type": "string", "description": "Optional detail"},
                    "spent_at": {
                        "type": "string",
                        "description": (
                            "When the expense happened. Accepts 'YYYY-MM-DD HH:MM' "
                            "(24-hour), 'YYYY-MM-DD hh:MM AM/PM' (12-hour), or "
                            "'YYYY-MM-DD' (date only = midnight). Defaults to now."
                        ),
                    },
                    "confirm": {
                        "type": "string",
                        "description": (
                            'Confirmation flag. Set to "y" or "yes" to actually add a '
                            "future-dated expense after previewing it; 'n' or 'no' to cancel. "
                            "Omit to preview a future-dated expense and request confirmation."
                        ),
                    },
                },
                "required": ["amount", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_expenses",
            "description": "Show all recorded expenses.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "total_by_category",
            "description": "Add up all expenses in one category.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_expense",
            "description": "Delete an expense record by ID, note (description), amount, and/or timestamp. Provide at least one criterion; all provided criteria must match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The expense ID (shown in the expense list)",
                    },
                    "note": {
                        "type": "string",
                        "description": "Exact description/note of the expense",
                    },
                    "amount": {"type": "number", "description": "Amount spent"},
                    "spent_at": {
                        "type": "string",
                        "description": (
                            "When the expense happened. Accepts 'YYYY-MM-DD HH:MM' "
                            "(24-hour), 'YYYY-MM-DD hh:MM AM/PM' (12-hour), or "
                            "'YYYY-MM-DD' (date only = midnight)."
                        ),
                    },
                    "confirm": {
                        "type": "string",
                        "description": (
                            'Confirmation flag. Set to "y" or "yes" to actually perform '
                            'the deletion after previewing matches; "n" or "no" to cancel. '
                            "Omit to preview matches and request confirmation."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_kiwisaver_daily",
            "description": "Insert or replace a KiwiSaver daily balance row for a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD date of the balance."},
                    "balance": {"type": "number", "description": "KiwiSaver balance."},
                    "units": {"type": "number", "description": "Number of units (optional)."},
                    "unit_price": {"type": "number", "description": "Price per unit (optional)."},
                },
                "required": ["date", "balance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_kiwisaver_daily",
            "description": "List all KiwiSaver daily balance rows.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_kiwisaver_adjustment",
            "description": "Add a KiwiSaver adjustment (employee, employer, government, tax, or fee).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD date of the adjustment."},
                    "amount": {"type": "number", "description": "Adjustment amount (can be negative)."},
                    "type": {"type": "string", "description": "One of: employee, employer, government, tax, fee."},
                },
                "required": ["date", "amount", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_kiwisaver_adjustments",
            "description": "List all KiwiSaver adjustment rows.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_kiwisaver",
            "description": "List KiwiSaver daily balances in date order with daily % change. Optionally filter by start_date and/or end_date (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD, inclusive)."},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD, inclusive)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kiwisaver_performance",
            "description": "Compute Kiwisaver performance between two dates (start/end value, % change, best/worst day, max drawdown, day count, stale days).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD, inclusive)."},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD, inclusive)."},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an expense helper. Use the tools to record and report expenses. "
    "Never invent numbers; if you need data, call a tool. "
    "A current local time is injected at the start of each user message. "
    "If the user mentions relative times like 'yesterday', 'this morning', "
    "'at 3pm', 'noon', or 'midnight', convert them to an exact date and time "
    "before calling add_expense: treat 'noon' as 12:00 PM and 'midnight' as "
    "12:00 AM. If the time is unclear, ask the user instead of guessing. "
    "If a user wants to correct or remove an expense (duplicate, wrong amount, wrong "
    "timestamp), call delete_expense WITHOUT the confirm argument first to preview "
    "the matching record(s) and request confirmation. Present the record details to "
    "the user and ask: 'Are you sure you want to delete this record? (y/n)'. "
    "Wait for the user's reply: if they say 'y'/'yes', call delete_expense again "
    'with confirm="y" to perform the deletion; if they say "n"/"no", call delete_expense '
    'with confirm="n" to cancel. Then, if deleting, use add_expense to add the corrected '
    "record with the right details and timestamp. Confirm to the user what was deleted "
    "and what was corrected. "
    "If add_expense is called with a future-dated timestamp, the tool will preview the "
    "expense and ask: 'Are you sure you want to add this record? (y/n)'. Wait for the "
    "user's reply: if they say 'y'/'yes', call add_expense again with confirm='y' to "
    "actually record it; if they say 'n'/'no', call add_expense with confirm='n' to cancel. "
    "If the user asks about KiwiSaver (balance, daily values, adjustments, or performance), "
    "use the kiwisaver tools: list_kiwisaver, kiwisaver_performance, add_kiwisaver_daily, "
    "list_kiwisaver_daily, add_kiwisaver_adjustment, list_kiwisaver_adjustments. "
    "Never estimate KiwiSaver numbers yourself; always call the appropriate tool with exact "
    "dates (YYYY-MM-DD). When reporting KiwiSaver results, state clearly that the figures are "
    "for tracking purposes only and are not financial advice."
)

# ---------- 4. The agent loop ----------
def run_agent(user_message, history):
    now = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %I:%M %p (%A)")
    history.append({"role": "system", "content": f"Current local time: {current_time_str}"})
    history.append({"role": "user", "content": user_message})

    tool_calls_made = []

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=history, tools=TOOLS
        )
        message = response.choices[0].message
        history.append(message)

        # No tool requested -> the agent is done, return its answer
        if not message.tool_calls:
            return message.content, tool_calls_made

        # Otherwise run each requested tool and feed the result back
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as error:  # never crash on a bad tool call
                result = f"Error: {error}"
            print(f"  [tool] {name} -> {result}")  # simple logging
            tool_calls_made.append(
                {"name": name, "args": args, "result": str(result)}
            )
            history.append(
                {"role": "tool", "tool_call_id": call.id, "content": str(result)}
            )

    return "Stopped: too many steps.", tool_calls_made

# ---------- 5. Chat with it ----------
if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Expense Helper ready. Type 'quit' to exit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        reply, _ = run_agent(text, history)
        print("Agent:", reply)
