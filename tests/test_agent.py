from types import SimpleNamespace

from google.genai import types

from app.agent import VisitAgent


class FakeModels:
    def __init__(self):
        self.calls = []
        self.responses = [
            types.GenerateContentResponse(
                candidates=[types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(function_call=types.FunctionCall(
                            name="search_knowledge_base", args={"query": "opening hours"}
                        ))],
                    ),
                )],
            ),
            types.GenerateContentResponse(
                candidates=[types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="We open at 10.")]),
                )],
            ),
        ]

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_chat_runs_gemini_function_call_and_preserves_history():
    models = FakeModels()
    retriever = SimpleNamespace(search=lambda query: [{"text": "We open at 10."}])
    agent = VisitAgent(retriever, client=SimpleNamespace(models=models))

    reply, history = agent.chat([], "When do you open?")

    assert reply == "We open at 10."
    assert len(models.calls) == 2
    assert models.calls[0]["model"]
    assert models.calls[0]["contents"][0] == {
        "role": "user", "parts": [{"text": "When do you open?"}]
    }
    assert history[1]["role"] == "model"
    assert history[2]["role"] == "user"
    assert history[2]["parts"][0]["function_response"]["name"] == "search_knowledge_base"