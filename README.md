# Visit Booking Agent (RAG + proactive slot booking)

A FastAPI service where customers chat with an assistant that:
1. **Answers questions with RAG**: retrieves from your docs (`data/docs/*.md`) and answers only from what it finds.
2. **Goes beyond the question**: after answering, it offers a visit, shows available slots, collects name and phone, confirms, then books.

## Architecture

```
Customer ──► POST /chat ──► VisitAgent (Gemini, tool-calling loop)
                               │
                               ├─ search_knowledge_base ─► Retriever (chunk + TF-IDF; swappable for vectors)
                               ├─ get_available_slots   ─┐
                               ├─ book_slot             ─┼► SQLite (slots, bookings)
                               └─ cancel_booking        ─┘
```

| File | Role |
|---|---|
| `app/rag.py` | Chunking and retrieval. `Retriever.search()` is the only interface the agent uses. |
| `app/booking.py` | Slot generation, availability, atomic booking (no double-booking), cancellation (phone-verified). |
| `app/agent.py` | System prompt, tool schemas, tool loop. The prompt enforces "confirm before booking". |
| `app/main.py` | REST API: `/chat`, `/search`, `/ingest`, `/slots`, `/bookings`. |
| `data/docs/faq.md` | Sample knowledge base. Replace with your own. |

## Run

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
uvicorn app.main:app --reload
pytest            # tests need no API key
```

Try it:
```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "What are your timings and can I come see the tyres tomorrow?"}'
# reuse the returned session_id in later calls to continue the conversation
```

`GET /search?q=parking` tests retrieval alone. `GET /slots` and `POST /bookings` work without the LLM.

## Configure (env vars)
`GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-flash`), `DOCS_DIR`, `DB_PATH`, `SLOT_DAYS_AHEAD`, `SLOT_START_HOUR`, `SLOT_END_HOUR`, `SLOT_CAPACITY`, `BUSINESS_NAME`. Closed weekdays are in `app/config.py`.

## Before production
- **Retrieval**: swap TF-IDF for embeddings plus a vector DB (pgvector, Qdrant, Chroma) behind the same `search()` method.
- **Sessions**: chat history is in memory; move to Redis or a DB.
- **Bookings**: SQLite is fine for a pilot; use Postgres for multiple instances. Add real business hours and holidays instead of fixed hourly slots.
- **Notifications**: send SMS, WhatsApp or email confirmations and reminders after `book_slot`.
- **Safety**: add auth and rate limiting on the API, validate phone numbers, and log tool calls for audit.
- **Timezone**: slots use server local time; set it to your business timezone (or store UTC).
