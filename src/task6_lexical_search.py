"""Task 6: Unicode BM25 over the persisted dense corpus."""
import re
import unicodedata

from .task4_chunking_indexing import load_indexed_chunks

# Optional fixture/in-memory override. Normal execution reloads Chroma.
CORPUS: list[dict] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def build_bm25_index(corpus: list[dict]):
    from rank_bm25 import BM25Plus

    if not corpus:
        return None
    # Positive IDF even for one/two-document corpora; delta=0 excludes nonmatches.
    tokens = [_tokenize(item["content"]) for item in corpus]
    if not any(tokens):
        return None
    return BM25Plus(tokens, delta=0)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    tokens = _tokenize(query)
    if not tokens or top_k <= 0:
        return []
    corpus = list({item["id"]: item for item in (CORPUS or load_indexed_chunks())}.values())
    bm25 = build_bm25_index(corpus)
    if bm25 is None:
        return []
    scores = bm25.get_scores(tokens)
    results = [{"id": item["id"], "content": item["content"], "score": float(score),
                "metadata": {"url": None, **item["metadata"]}, "retrieval_method": "bm25"}
               for item, score in zip(corpus, scores) if score > 0]
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]
