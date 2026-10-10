"""Minimal RAG: chunk markdown/text docs, index with TF-IDF, retrieve top-k.

The Retriever interface (`search`) is deliberately small so you can swap in
embeddings + a vector DB (pgvector, Qdrant, Chroma) without touching the agent.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


@dataclass
class Chunk:
    text: str
    source: str


def chunk_text(text: str, source: str, max_chars: int = 700) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(Chunk(buf, source))
            buf = p
        else:
            buf = f"{buf}\n\n{p}".strip()
    if buf:
        chunks.append(Chunk(buf, source))
    return chunks


_STOP = set("a an the is are was be to of and or in on at for with do does did i my me you your we our it "
            "what how when where which who can could would should will this that there any".split())


def _tokenize(text: str) -> list[str]:
    """Lowercase words with a crude suffix stemmer so 'opening' matches 'open'."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 4:
            for suf in ("ing", "ed", "es", "s"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    w = w[: -len(suf)]
                    break
        out.append(w)
    return out


class Retriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vec = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, lowercase=False,
                                    stop_words=None, ngram_range=(1, 1), sublinear_tf=True)
        self._matrix = None

    def add_documents(self, docs: dict[str, str]) -> None:
        for source, text in docs.items():
            self.chunks.extend(chunk_text(text, source))
        self._rebuild()

    def load_dir(self, directory: Path | None = None) -> int:
        directory = Path(directory or config.DOCS_DIR)
        docs = {p.name: p.read_text(encoding="utf-8")
                for p in directory.glob("**/*") if p.suffix in {".md", ".txt"}}
        self.chunks = []
        self.add_documents(docs)
        return len(self.chunks)

    def _rebuild(self) -> None:
        if self.chunks:
            self._matrix = self._vec.fit_transform([c.text for c in self.chunks])
        else:
            self._matrix = None

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> list[dict]:
        if self._matrix is None:
            return []
        scores = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"text": self.chunks[i].text, "source": self.chunks[i].source, "score": round(float(scores[i]), 3)}
                for i in ranked if scores[i] >= min_score]
