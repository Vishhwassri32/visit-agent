import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import booking, config
from .agent import VisitAgent
from .rag import Retriever

retriever = Retriever()
sessions: dict[str, list[dict]] = {}  # swap for Redis/DB in production
_agent: VisitAgent | None = None


def get_agent() -> VisitAgent:
    global _agent
    if _agent is None:
        if not config.GEMINI_API_KEY:
            raise HTTPException(503, "GEMINI_API_KEY is not set.")
        _agent = VisitAgent(retriever)
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    booking.init_db()
    booking.seed_slots()
    retriever.load_dir()
    yield


app = FastAPI(title="Visit Booking Agent", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


class ChatOut(BaseModel):
    reply: str
    session_id: str


class BookIn(BaseModel):
    slot_id: int
    name: str
    phone: str
    email: str | None = None
    notes: str | None = None


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    sid = body.session_id or str(uuid.uuid4())
    reply, history = get_agent().chat(sessions.get(sid, []), body.message)
    sessions[sid] = history[-30:]  # cap history length
    return ChatOut(reply=reply, session_id=sid)


@app.get("/search")
def search(q: str, k: int = 4):
    """Debug: test retrieval alone, no LLM needed."""
    return {"results": retriever.search(q, k)}


@app.post("/ingest")
def ingest(docs: dict[str, str]):
    """Add documents: {"filename.md": "text..."}"""
    retriever.add_documents(docs)
    return {"chunks": len(retriever.chunks)}


@app.get("/slots")
def slots(date: str | None = None, limit: int = 10):
    return {"slots": booking.list_available_slots(date, limit)}


@app.post("/bookings")
def create_booking(body: BookIn):
    try:
        return booking.book_slot(body.slot_id, body.name, body.phone, body.email, body.notes)
    except booking.BookingError as e:
        raise HTTPException(409, str(e))


@app.delete("/bookings/{booking_id}")
def delete_booking(booking_id: int, phone: str):
    try:
        return booking.cancel_booking(booking_id, phone)
    except booking.BookingError as e:
        raise HTTPException(404, str(e))
