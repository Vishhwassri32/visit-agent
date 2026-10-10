"""SQLite-backed visit slots and bookings."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from . import config


@contextmanager
def _conn(db_path: str | None = None):
    conn = sqlite3.connect(db_path or config.DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with _conn(db_path) as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL UNIQUE,
                end_time TEXT NOT NULL,
                capacity INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slot_id INTEGER NOT NULL REFERENCES slots(id),
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT,
                notes TEXT,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL
            );
            """
        )


def seed_slots(db_path: str | None = None, now: datetime | None = None) -> int:
    """Create hourly slots for the next N days. Safe to call repeatedly."""
    now = now or datetime.now()
    created = 0
    with _conn(db_path) as c:
        for d in range(config.SLOT_DAYS_AHEAD + 1):
            day = (now + timedelta(days=d)).replace(minute=0, second=0, microsecond=0)
            if day.weekday() in config.CLOSED_WEEKDAYS:
                continue
            for hour in range(config.SLOT_START_HOUR, config.SLOT_END_HOUR):
                start = day.replace(hour=hour)
                if start <= now:
                    continue
                end = start + timedelta(hours=1)
                cur = c.execute(
                    "INSERT OR IGNORE INTO slots (start_time, end_time, capacity) VALUES (?,?,?)",
                    (start.isoformat(timespec="minutes"), end.isoformat(timespec="minutes"), config.SLOT_CAPACITY),
                )
                created += cur.rowcount
    return created


def list_available_slots(date: str | None = None, limit: int = 10, db_path: str | None = None,
                         now: datetime | None = None) -> list[dict]:
    """Return upcoming slots with free capacity. `date` is YYYY-MM-DD (optional)."""
    now_iso = (now or datetime.now()).isoformat(timespec="minutes")
    sql = """
        SELECT s.id, s.start_time, s.end_time,
               s.capacity - COUNT(b.id) AS remaining
        FROM slots s
        LEFT JOIN bookings b ON b.slot_id = s.id AND b.status = 'confirmed'
        WHERE s.start_time > ?
    """
    params: list = [now_iso]
    if date:
        sql += " AND substr(s.start_time,1,10) = ?"
        params.append(date)
    sql += " GROUP BY s.id HAVING remaining > 0 ORDER BY s.start_time LIMIT ?"
    params.append(limit)
    with _conn(db_path) as c:
        return [dict(r) for r in c.execute(sql, params)]


class BookingError(Exception):
    pass


def book_slot(slot_id: int, name: str, phone: str, email: str | None = None,
              notes: str | None = None, db_path: str | None = None) -> dict:
    if not name.strip() or not phone.strip():
        raise BookingError("Name and phone are required.")
    with _conn(db_path) as c:
        c.execute("BEGIN IMMEDIATE")  # lock to avoid double-booking races
        try:
            slot = c.execute("SELECT * FROM slots WHERE id = ?", (slot_id,)).fetchone()
            if not slot:
                raise BookingError(f"Slot {slot_id} does not exist.")
            taken = c.execute(
                "SELECT COUNT(*) FROM bookings WHERE slot_id = ? AND status = 'confirmed'", (slot_id,)
            ).fetchone()[0]
            if taken >= slot["capacity"]:
                raise BookingError("That slot is already full. Please pick another.")
            cur = c.execute(
                "INSERT INTO bookings (slot_id, name, phone, email, notes, created_at) VALUES (?,?,?,?,?,?)",
                (slot_id, name.strip(), phone.strip(), email, notes, datetime.now().isoformat(timespec="seconds")),
            )
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        return {"booking_id": cur.lastrowid, "slot_id": slot_id, "start_time": slot["start_time"],
                "end_time": slot["end_time"], "name": name.strip(), "status": "confirmed"}


def cancel_booking(booking_id: int, phone: str, db_path: str | None = None) -> dict:
    """Cancel only if the phone matches the one used to book."""
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,)).fetchone()
        if not row or row["phone"] != phone.strip():
            raise BookingError("No matching booking found for that ID and phone number.")
        c.execute("UPDATE bookings SET status = 'cancelled' WHERE id = ?", (booking_id,))
        return {"booking_id": booking_id, "status": "cancelled"}
