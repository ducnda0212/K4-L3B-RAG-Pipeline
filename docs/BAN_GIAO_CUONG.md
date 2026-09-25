# Bàn giao Task 4–6 — Cường

## Thiết lập và chạy

Từ thư mục gốc, dùng Python 3.10–3.13 và cài `python -m pip install -e ".[dev]"`.
Sao chép `.env.example` thành `.env` nếu chưa có. Cấu hình nhóm:

```dotenv
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
CHROMA_COLLECTION=rag_documents
```

```powershell
python -m src.task4_chunking_indexing
python -m pytest tests/test_index_search.py -q
```

Lần đầu embedding cần mạng để tải BGE-M3 và đủ bộ nhớ/ổ đĩa cho model.
Provider khác bị từ chối rõ ràng; chưa hỗ trợ embedding OpenAI/Gemini.
Không cần API key cho embedding local. Không ghi đè `.env` đang có.

Máy hiện tại có Python portable ở `.python/runtime/python.exe` để chạy test,
do lệnh `python` chưa có trong PATH. Runtime này và dependencies không được
commit. Có thể chạy `./.python/runtime/python.exe -m pytest tests/test_index_search.py -q`.
Đây mới là môi trường test Task 4–6; chưa cài toàn bộ dependencies của ứng dụng
hay tải trọng số BGE-M3. Môi trường demo cần thực hiện setup đầy đủ ở trên.

## Giao diện cho Hình

```python
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search

dense = semantic_search("Những di tích nổi bật tại Huế?", top_k=5)
sparse = lexical_search("Ngũ Hành Sơn", top_k=5)
```

Cả hai trả `list[SearchResult]`, score giảm dần, ID duy nhất, không quá `top_k`.
Dense dùng `1 - cosine_distance`, giữ cả score âm; Hình dùng score này để
hiệu chỉnh fallback, không dùng score BM25/RRF thay thế.
Query rỗng, `top_k <= 0`, hoặc index chưa có dữ liệu trả `[]`.
Lỗi cấu hình/provider được báo lên caller để tầng retrieval/UI xử lý.

`load_indexed_chunks()` đọc lại nội dung và metadata từ Chroma ở `chroma_db/`.
BM25 tự gọi hàm này mỗi lượt tìm kiếm, vì vậy không cần gán `CORPUS` sau restart
và không trộn Markdown mới sửa với dense index cũ. `CORPUS` chỉ là override
cho fixture/thử nghiệm; ứng dụng dùng mặc định rỗng. Với corpus lớn nên bổ sung
cache theo revision của index; bản hiện tại ưu tiên nhất quán dữ liệu.

## Quyết định kỹ thuật

- Recursive split: 500 **ký tự**, overlap tối đa 50 ký tự; ưu tiên đoạn, dòng,
  câu, khoảng trắng. Phù hợp làm baseline dễ chạy; chưa coi đây là cấu hình tối ưu.
- ID là đường dẫn Markdown tương đối + `::chunk-N`. Giữ source/title/doc_type/url
  và metadata scalar bổ sung từ YAML; không đưa front matter vào nội dung embedding.
- Chroma không lưu `None` trong metadata; khi đọc phục hồi `url=None` đúng contract.
- `index_to_vectorstore()` nhận **toàn bộ snapshot corpus**: upsert trước, xóa ID
  cũ không còn sau khi upsert thành công. Không dùng hàm này cho một phần corpus.
  Snapshot rỗng không xóa dữ liệu. Không chạy index đồng thời với phục vụ truy vấn:
  nhiều batch chưa có cơ chế atomic swap khi lỗi giữa chừng.
- Collection lưu provider/model/dimension/chunk config. Khi thay cấu hình, chọn
  `CHROMA_COLLECTION` mới rồi index lại. Cùng model/encoder cho index và query.
- BM25 dùng `BM25Plus(delta=0)` để có IDF dương cả với corpus rất nhỏ và không trả
  tài liệu không khớp từ. Tokenize Unicode NFC + casefold, giữ dấu tiếng Việt.
  Chưa có tách từ tiếng Việt hoặc mở rộng truy vấn không dấu.

## Đóng góp golden Q&A

Bộ golden hiện đã có các câu về Huế và Ngũ Hành Sơn. Bốn câu bổ sung có trích
nguyên văn corpus được lưu riêng trong `group_project/evaluation/cuong_questions.json`
để Vinh rà soát trước khi gộp, tránh sửa bộ đánh giá chung trong lúc người khác làm.

## Kết quả kiểm chứng ngày 25/09/2026

Môi trường: Python 3.12.10, ChromaDB 1.5.9, NumPy 1.26.4,
langchain-text-splitters 1.1.2, rank-bm25 0.2.2, PyYAML 6.0.3, pytest 9.1.1.

Lệnh đã chạy trên máy hiện tại:

```powershell
./.python/runtime/python.exe -m pytest -q --tb=short --basetemp=.cache/pytest-cuong-complete
git diff --check
```

- Toàn bộ suite: **23 passed, 6 failed**. Cả **9 test mới** trong
  `tests/test_index_search.py` đều qua; các contract test chunk/dense/BM25 cũng qua.
- 6 failure ngoài Task 4–6: RRF Task 7 (1), generation Task 10 (1), retrieval
  Task 9 (3), báo cáo evaluation còn `TODO` (1). Không đổi/xóa test để che lỗi.
- Corpus thật: đọc **9 Markdown** (4 legal, 5 news), tạo **892 chunks**.
  Các tài liệu giữ URL; 4 đoạn bằng chứng của Cường đều tồn tại nguyên văn.
- Chroma thật trong thư mục test: index lặp lại không tăng bản ghi; thu nhỏ
  snapshot loại chunks cũ; BM25 tìm được dữ liệu từ một tiến trình Python mới.
- Đã kiểm tra mismatch model/dimension, vector lỗi, query rỗng, top_k,
  cosine âm, chuẩn hóa Unicode, schema và thứ tự kết quả.
- `git diff --check` không phát hiện lỗi whitespace.

**Giới hạn kiểm chứng:** vector dùng trong test là fixture; chưa tải trọng số,
chưa chạy BGE-M3 trên 892 chunks và chưa tạo index demo tại `chroma_db/`.
Chưa xác nhận cài toàn bộ dependencies của ứng dụng; chưa có số đo chất lượng
retrieval hoặc evaluation A/B. Chạy pipeline embedding thật là bước cần làm
trước demo, không được xem các test fixture là bằng chứng chất lượng ngữ nghĩa.
