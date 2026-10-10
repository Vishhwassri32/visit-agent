"""Tool-calling agent: answers from the knowledge base (RAG) and, when the
customer shows visit intent (or the answer naturally leads to one), proposes
and books a visit slot."""
from google import genai
from google.genai import types

from . import booking, config
from .rag import Retriever

SYSTEM_PROMPT = f"""You are the customer assistant for {config.BUSINESS_NAME}.

How you work:
1. For any question about the business, ALWAYS call search_knowledge_base first and answer only from what it returns.
   If nothing relevant is found, say you don't have that information and offer to have a specialist follow up. Never invent prices, policies or hours.
2. Be proactive about visits: after answering a question where seeing the product or speaking to a specialist would help
   (pricing, product choice, fitting, etc.), briefly offer to book a visit. Do not push if the customer declines.
3. To book: call get_available_slots, offer 2-3 options in plain language, then collect the customer's name and phone
   (email optional). Confirm the chosen slot and details with the customer BEFORE calling book_slot.
4. Never call book_slot until the customer has clearly confirmed the slot and given their name and phone.
5. After booking, repeat the date/time and booking ID, and mention what to bring if the knowledge base says so.
6. For cancellations, require the booking ID and the phone number used to book.
Keep replies short and friendly. Times are local business time."""

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the business knowledge base (FAQ, products, policies, hours). Use for every factual question.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "get_available_slots",
        "description": "List upcoming visit slots that still have capacity. Optionally filter by a single date.",
        "input_schema": {"type": "object", "properties": {
            "date": {"type": "string", "description": "YYYY-MM-DD, optional"},
            "limit": {"type": "integer", "description": "max slots, default 6"}}},
    },
    {
        "name": "book_slot",
        "description": "Book a visit slot. Only call after the customer has confirmed the slot and provided name and phone.",
        "input_schema": {"type": "object", "properties": {
            "slot_id": {"type": "integer"}, "name": {"type": "string"}, "phone": {"type": "string"},
            "email": {"type": "string"}, "notes": {"type": "string", "description": "e.g. vehicle model, interest"}},
            "required": ["slot_id", "name", "phone"]},
    },
    {
        "name": "cancel_booking",
        "description": "Cancel an existing booking. Needs the booking ID and the phone number used to book.",
        "input_schema": {"type": "object", "properties": {
            "booking_id": {"type": "integer"}, "phone": {"type": "string"}}, "required": ["booking_id", "phone"]},
    },
]


class VisitAgent:
    def __init__(self, retriever: Retriever, client: genai.Client | None = None, max_steps: int = 6):
        self.retriever = retriever
        self.client = client or genai.Client(api_key=config.GEMINI_API_KEY)
        self.max_steps = max_steps

    def _run_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "search_knowledge_base":
                return {"results": self.retriever.search(args["query"])}
            if name == "get_available_slots":
                return {"slots": booking.list_available_slots(args.get("date"), args.get("limit", 6))}
            if name == "book_slot":
                return booking.book_slot(args["slot_id"], args["name"], args["phone"],
                                         args.get("email"), args.get("notes"))
            if name == "cancel_booking":
                return booking.cancel_booking(args["booking_id"], args["phone"])
            return {"error": f"unknown tool {name}"}
        except booking.BookingError as e:
            return {"error": str(e)}

    def chat(self, history: list[dict], user_message: str) -> tuple[str, list[dict]]:
        """Runs one customer turn. `history` is the prior message list; returns (reply, new_history)."""
        contents = history + [{"role": "user", "parts": [{"text": user_message}]}]
        tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters_json_schema=tool["input_schema"],
                )
                for tool in TOOLS
            ])
        ]
        for _ in range(self.max_steps):
            response = self.client.models.generate_content(
                model=config.MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=tools,
                    max_output_tokens=1024,
                ),
            )
            candidate_content = response.candidates[0].content
            contents.append(candidate_content.model_dump(exclude_none=True))
            function_calls = [part.function_call for part in candidate_content.parts or [] if part.function_call]
            if not function_calls:
                text = "".join(part.text or "" for part in candidate_content.parts or [])
                return text, contents
            function_responses = [
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": self._run_tool(call.name, dict(call.args or {}))},
                )
                for call in function_calls
            ]
            contents.append(types.Content(role="user", parts=function_responses).model_dump(exclude_none=True))
        return "Sorry, I couldn't complete that. Could you rephrase?", contents
