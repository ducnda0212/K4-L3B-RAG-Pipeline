"""Task 4: deterministic chunks and a persistent cosine index."""
import math
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from .contracts import validate_document

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
STANDARDIZED_DIR = ROOT / "data" / "standardized"
CHROMA_DIR = ROOT / "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM") or "1024")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION") or "rag_documents"


@lru_cache(maxsize=2)
def _embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _validate_vectors(vectors, count: int) -> list[list[float]]:
    if len(vectors) != count:
        raise ValueError("Embedding provider returned the wrong vector count")
    output = [[float(value) for value in vector] for vector in vectors]
    if any(len(vector) != EMBEDDING_DIM for vector in output):
        raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}")
    if any(not all(math.isfinite(v) for v in vector) or not any(vector) for vector in output):
        raise ValueError("Embeddings must be finite, nonzero vectors")
    return output


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Shared index/query encoder; the group uses local BGE-M3 exclusively."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER != "sentence_transformers":
        raise ValueError("This pipeline supports EMBEDDING_PROVIDER=sentence_transformers")
    vectors = _embedding_model(EMBEDDING_MODEL).encode(
        texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False,
    ).tolist()
    return _validate_vectors(vectors, len(texts))


def get_collection():
    """Reject incompatible indexes instead of mixing vector spaces."""
    import chromadb

    expected = {
        "hnsw:space": "cosine", "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL, "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
    }
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata=expected, embedding_function=None,
    )
    if any((collection.metadata or {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Index configuration differs; choose a new CHROMA_COLLECTION and re-index")
    return collection


def load_documents() -> list[dict]:
    """Read Task 3 YAML front matter; index only the document body."""
    import yaml

    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", text, re.DOTALL)
        fields = yaml.safe_load(match.group(1)) if match else {}
        if fields is None:
            fields = {}
        if not isinstance(fields, dict):
            raise ValueError(f"Invalid front matter: {path}")
        body = text[match.end():].strip() if match else text.strip()
        if not body:
            continue
        relative = path.relative_to(STANDARDIZED_DIR)
        metadata = {key: value for key, value in fields.items()
                    if isinstance(value, (str, int, float, bool))}
        metadata.update({
            "source": fields.get("source") or relative.as_posix(),
            "title": fields.get("title") or path.stem,
            "doc_type": fields.get("doc_type") or relative.parts[0],
            "url": fields.get("url") or None,
        })
        item = {"id": relative.as_posix(), "content": body, "metadata": metadata}
        validate_document(item)
        documents.append(item)
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    seen = set()
    for document in documents:
        validate_document(document)
        if document["id"] in seen:
            raise ValueError(f"Duplicate document ID: {document['id']}")
        seen.add(document["id"])
        for index, content in enumerate(splitter.split_text(document["content"])):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}", "content": content,
                "metadata": {**document["metadata"], "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embedded = []
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        vectors = embed_texts([item["content"] for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count does not match chunks")
        embedded.extend({**item, "embedding": vector} for item, vector in zip(batch, vectors))
    return embedded


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Synchronize a WHOLE corpus snapshot, not an incremental document batch.

    Remove stale chunks after successful upserts. Empty input is a no-op.
    """
    if not chunks:
        return
    ids = [item["id"] for item in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs")
    for item in chunks:
        validate_document(item, require_chunk=True)
    vectors = _validate_vectors([item["embedding"] for item in chunks], len(chunks))
    collection = get_collection()
    for start in range(0, len(chunks), 128):
        batch = chunks[start:start + 128]
        collection.upsert(
            ids=ids[start:start + 128], documents=[item["content"] for item in batch],
            embeddings=vectors[start:start + 128],
            metadatas=[{key: value for key, value in item["metadata"].items()
                        if value is not None} for item in batch],
        )
    stale = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    for start in range(0, len(stale), 128):
        collection.delete(ids=stale[start:start + 128])


def load_indexed_chunks() -> list[dict]:
    """Reload the exact dense corpus for BM25 after a process restart."""
    collection = get_collection()
    chunks = []
    for offset in range(0, collection.count(), 1000):
        response = collection.get(limit=1000, offset=offset, include=["documents", "metadatas"])
        for item_id, content, metadata in zip(
            response["ids"], response["documents"], response["metadatas"],
        ):
            chunks.append({"id": item_id, "content": content,
                           "metadata": {"url": None, **metadata}})
    return sorted(chunks, key=lambda item: item["id"])


def run_pipeline() -> None:
    documents = load_documents()
    if not documents:
        raise ValueError(f"No Markdown documents in {STANDARDIZED_DIR}")
    chunks = embed_chunks(chunk_documents(documents))
    index_to_vectorstore(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
