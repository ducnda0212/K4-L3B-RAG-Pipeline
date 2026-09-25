"""Task 5: shared encoder and original cosine similarity for fallback."""
from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    if not query.strip() or top_k <= 0:
        return []
    collection = get_collection()
    count = collection.count() if hasattr(collection, "count") else top_k
    if count == 0:
        return []
    response = collection.query(
        query_embeddings=embed_texts([query]), n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    results = {}
    for item_id, content, metadata, distance in zip(
        response["ids"][0], response["documents"][0],
        response["metadatas"][0], response["distances"][0],
    ):
        item = {"id": item_id, "content": content, "score": 1.0 - float(distance),
                "metadata": {"url": None, **metadata}, "retrieval_method": "dense"}
        if item_id not in results or item["score"] > results[item_id]["score"]:
            results[item_id] = item
    return sorted(results.values(), key=lambda item: (-item["score"], item["id"]))[:top_k]
