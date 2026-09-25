"""Task 4–6 tests: no network or model download; real temporary Chroma DB."""
import unicodedata

import pytest

from src import task4_chunking_indexing as indexing
from src import task5_semantic_search as dense
from src import task6_lexical_search as lexical
from src.contracts import validate_search_results


def chunk(item_id, content, vector):
    return {"id": item_id, "content": content, "embedding": vector,
            "metadata": {"source": "test.md", "title": "Du lịch", "doc_type": "news",
                         "url": None, "chunk_index": 0}}


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(indexing, "EMBEDDING_DIM", 3)
    monkeypatch.setattr(indexing, "COLLECTION_NAME", "test_documents")
    monkeypatch.setattr(lexical, "CORPUS", [])
    return indexing.get_collection()


def test_metadata_and_deterministic_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(indexing, "STANDARDIZED_DIR", tmp_path)
    directory = tmp_path / "news"
    directory.mkdir()
    (directory / "hue.md").write_text(
        '---\ntitle: "Huế"\nsource: "hue.json"\ndoc_type: "news"\n'
        'url: "https://example.org/hue"\n---\n# Huế\n' + "Kinh thành Huế. " * 150,
        encoding="utf-8",
    )
    docs = indexing.load_documents()
    assert docs[0]["metadata"]["source"] == "hue.json"
    assert docs[0]["metadata"]["url"] == "https://example.org/hue"
    assert docs[0]["content"].startswith("# Huế")
    chunks = indexing.chunk_documents(docs)
    assert len(chunks) > 1
    assert chunks == indexing.chunk_documents(docs)
    assert all(len(item["content"]) <= indexing.CHUNK_SIZE for item in chunks)


def test_persistence_upsert_and_same_corpus(database, monkeypatch):
    chunks = [chunk("hue::chunk-0", "Kinh thành Huế", [1, 0, 0]),
              chunk("hoi-an::chunk-0", "Phố cổ Hội An", [0, 1, 0])]
    indexing.index_to_vectorstore(chunks)
    indexing.index_to_vectorstore(chunks)
    assert database.count() == 2
    # New collection handle reloads the stored corpus with no in-memory CORPUS.
    assert {item["id"] for item in indexing.load_indexed_chunks()} == {c["id"] for c in chunks}
    monkeypatch.setattr(dense, "embed_texts", lambda texts: [[1, 0, 0]])
    results = dense.semantic_search("di tích Huế", 10)
    validate_search_results(results, top_k=10, expected_method="dense")
    assert results[0]["id"] == "hue::chunk-0"
    assert results[0]["score"] == pytest.approx(1)
    assert results[0]["metadata"]["url"] is None
    sparse = lexical.lexical_search(unicodedata.normalize("NFD", "HUẾ!"), 10)
    validate_search_results(sparse, top_k=10, expected_method="bm25")
    assert [item["id"] for item in sparse] == ["hue::chunk-0"]
    indexing.index_to_vectorstore(chunks[:1])
    assert database.count() == 1
    assert lexical.lexical_search("Hội An") == []


def test_configuration_mismatch_rejected(database, monkeypatch):
    monkeypatch.setattr(indexing, "EMBEDDING_MODEL", "different-model")
    with pytest.raises(ValueError, match="configuration differs"):
        indexing.get_collection()


def test_dense_preserves_negative_cosine(database, monkeypatch):
    indexing.index_to_vectorstore([chunk("opposite", "Hội An", [-1, 0, 0])])
    monkeypatch.setattr(dense, "embed_texts", lambda texts: [[1, 0, 0]])
    assert dense.semantic_search("Huế")[0]["score"] == pytest.approx(-1)


def test_bm25_reload_in_fresh_process(database):
    import json
    import subprocess
    import sys

    indexing.index_to_vectorstore([chunk("persisted", "Huế", [1, 0, 0])])
    code = """
import json, sys
from pathlib import Path
from src import task4_chunking_indexing as indexing
indexing.CHROMA_DIR = Path(sys.argv[1])
indexing.EMBEDDING_DIM = 3
indexing.COLLECTION_NAME = 'test_documents'
from src.task6_lexical_search import lexical_search
print(json.dumps([item['id'] for item in lexical_search('Huế')]))
"""
    output = subprocess.run(
        [sys.executable, "-c", code, str(indexing.CHROMA_DIR)],
        cwd=indexing.ROOT, capture_output=True, text=True, timeout=60,
    )
    assert output.returncode == 0, output.stderr
    assert json.loads(output.stdout) == ["persisted"]


def test_empty_inputs_and_invalid_vectors(database):
    assert dense.semantic_search("query") == []
    assert lexical.lexical_search("query") == []
    assert indexing.embed_texts([]) == []
    for function in (dense.semantic_search, lexical.lexical_search):
        assert function(" ") == []
        assert function("Huế", 0) == []
        assert function("Huế", -1) == []
    for vector in ([1, 0], [0, 0, 0], [float("nan"), 1, 0]):
        with pytest.raises(ValueError):
            indexing.index_to_vectorstore([chunk("bad", "Huế", vector)])
    assert database.count() == 0


def test_embedding_nonmutating_and_batch_count(monkeypatch):
    chunks = [chunk("one", "Huế", [1, 0, 0])]
    chunks[0].pop("embedding")
    monkeypatch.setattr(indexing, "embed_texts", lambda texts: [[1, 0, 0]])
    output = indexing.embed_chunks(chunks)
    assert "embedding" not in chunks[0]
    assert output[0]["embedding"] == [1, 0, 0]
    monkeypatch.setattr(indexing, "embed_texts", lambda texts: [])
    with pytest.raises(ValueError, match="count"):
        indexing.embed_chunks(chunks)


def test_encoder_contract(monkeypatch):
    class Vectors:
        def tolist(self):
            return [[1, 0, 0]]

    class Model:
        def encode(self, texts, **kwargs):
            assert texts == ["Huế"]
            assert kwargs["normalize_embeddings"] is True
            return Vectors()

    monkeypatch.setattr(indexing, "EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setattr(indexing, "EMBEDDING_DIM", 3)
    monkeypatch.setattr(indexing, "_embedding_model", lambda name: Model())
    assert indexing.embed_texts(["Huế"]) == [[1.0, 0.0, 0.0]]
    monkeypatch.setattr(indexing, "EMBEDDING_PROVIDER", "unsupported")
    with pytest.raises(ValueError, match="supports"):
        indexing.embed_texts(["Huế"])


def test_real_corpus_and_golden_evidence():
    import json

    docs = indexing.load_documents()
    assert len(docs) >= 8
    chunks = indexing.chunk_documents(docs)
    assert len(chunks) > len(docs)
    assert len({item["id"] for item in chunks}) == len(chunks)
    assert all(item["metadata"]["url"] for item in docs)
    questions = json.loads((indexing.ROOT / "group_project/evaluation/cuong_questions.json")
                           .read_text(encoding="utf-8"))
    assert len(questions) == 4
    for question in questions:
        source = indexing.STANDARDIZED_DIR / "news" / question["source"]
        assert question["expected_context"] in source.read_text(encoding="utf-8")
