from datetime import datetime

import pytest

from app import booking, config
from app.rag import Retriever

NOW = datetime(2026, 10, 12, 9, 0)  # a Monday morning


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "t.db")
    booking.init_db(path)
    booking.seed_slots(path, now=NOW)
    return path


def test_seed_skips_sunday_and_is_idempotent(db):
    assert booking.seed_slots(db, now=NOW) == 0
    slots = booking.list_available_slots(limit=500, db_path=db, now=NOW)
    assert slots and all(datetime.fromisoformat(s["start_time"]).weekday() != 6 for s in slots)


def test_book_then_slot_disappears(db):
    first = booking.list_available_slots(limit=1, db_path=db, now=NOW)[0]
    b = booking.book_slot(first["id"], "Asha", "9999999999", db_path=db)
    assert b["status"] == "confirmed"
    ids = [s["id"] for s in booking.list_available_slots(limit=500, db_path=db, now=NOW)]
    assert first["id"] not in ids


def test_double_booking_rejected(db):
    first = booking.list_available_slots(limit=1, db_path=db, now=NOW)[0]
    booking.book_slot(first["id"], "Asha", "9999999999", db_path=db)
    with pytest.raises(booking.BookingError):
        booking.book_slot(first["id"], "Ravi", "8888888888", db_path=db)


def test_cancel_requires_matching_phone_and_frees_slot(db):
    first = booking.list_available_slots(limit=1, db_path=db, now=NOW)[0]
    b = booking.book_slot(first["id"], "Asha", "9999999999", db_path=db)
    with pytest.raises(booking.BookingError):
        booking.cancel_booking(b["booking_id"], "0000000000", db_path=db)
    booking.cancel_booking(b["booking_id"], "9999999999", db_path=db)
    ids = [s["id"] for s in booking.list_available_slots(limit=500, db_path=db, now=NOW)]
    assert first["id"] in ids


def test_missing_contact_rejected(db):
    first = booking.list_available_slots(limit=1, db_path=db, now=NOW)[0]
    with pytest.raises(booking.BookingError):
        booking.book_slot(first["id"], "", "123", db_path=db)


def test_retrieval_finds_hours_and_cancellation():
    r = Retriever()
    assert r.load_dir(config.DOCS_DIR) > 0
    assert "Sunday" in r.search("what are your opening hours")[0]["text"]
    assert "booking ID" in r.search("how do I cancel my appointment")[0]["text"]
    assert r.search("quantum chromodynamics lagrangian") == []
