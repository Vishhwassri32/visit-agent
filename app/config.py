import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "bookings.db"))

# Visit slot generation
SLOT_DAYS_AHEAD = int(os.getenv("SLOT_DAYS_AHEAD", "7"))
SLOT_START_HOUR = int(os.getenv("SLOT_START_HOUR", "10"))
SLOT_END_HOUR = int(os.getenv("SLOT_END_HOUR", "17"))
SLOT_CAPACITY = int(os.getenv("SLOT_CAPACITY", "1"))
CLOSED_WEEKDAYS = {6}  # 0=Mon ... 6=Sun

BUSINESS_NAME = os.getenv("BUSINESS_NAME", "Arctechtire Showroom")
