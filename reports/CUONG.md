# Báo cáo phần việc Cường — Task 4–6

Đây là bản nháp bàn giao kỹ thuật để Cường kiểm tra và xác nhận. Chưa có mã học
viên; cần đổi tên thành `<ma-hoc-vien>-cuong.md` trước khi nộp. Không tự điền mã.

- Thành viên: Cường
- Repository: `ducnda0212/K4-L3B-RAG-Pipeline`; nhánh: `hongcuong`.
- Commit triển khai: `feat: implement Cuong indexing and search tasks` (tra trong lịch sử Git).

## Phần việc

| Đầu ra | File | Nội dung |
| --- | --- | --- |
| Chunking/index | `src/task4_chunking_indexing.py` | Đọc YAML metadata, recursive chunks, embedding BGE-M3, Chroma persistent, upsert và loại chunks cũ |
| Dense search | `src/task5_semantic_search.py` | Encoder dùng chung, cosine score gốc, đúng SearchResult |
| BM25 | `src/task6_lexical_search.py` | Unicode NFC, BM25Plus, đọc corpus từ Chroma sau restart |
| Môi trường | `pyproject.toml`, `.env.example`, `.gitignore` | Dependency trực tiếp, cấu hình model/dimension, bỏ qua Chroma local |
| Bàn giao và kiểm thử | `docs/BAN_GIAO_CUONG.md`, `tests/test_index_search.py` | API cho Hình, kiểm tra persistence/idempotence/schema và dữ liệu thật |
| Golden Q&A | `group_project/evaluation/cuong_questions.json` | 4 câu hỏi địa danh có bằng chứng nguyên văn, chờ Vinh rà soát/gộp |

## Quyết định kỹ thuật

1. Chọn chunk 500 ký tự, overlap 50, BGE-M3 1024 chiều dùng cho cả index và query.
   Đây là baseline, chưa có phép đo chứng minh tối ưu. Lưu cấu hình trong collection
   và từ chối model/dimension không khớp để tránh query sai không gian vector.
2. BM25 nạp đúng snapshot Chroma thay vì tự chia lại Markdown lúc truy vấn.
   Đổi lại phải đọc corpus và xây BM25 mỗi lượt; phù hợp corpus nhỏ hiện tại.
   Dùng BM25Plus với delta 0 để corpus nhỏ vẫn có score dương cho từ khớp.

## Kiểm chứng

Kết quả chạy kiểm thử được ghi trong `docs/BAN_GIAO_CUONG.md`.
Test dùng Chroma thật trong thư mục tạm và vector fixture; không đánh đồng kết quả
đó với chất lượng ngữ nghĩa của model BGE-M3 hay số đo evaluation A/B.

## Giới hạn

- Cần tải và chạy model BGE-M3 để tạo index dùng cho demo thực tế.
- Chưa hiệu chỉnh chunk size/overlap qua evaluation; đây là đầu vào để Hình đo A/B.
- Không có atomic swap toàn index; chạy index khi không phục vụ truy vấn.
- Chưa tách từ tiếng Việt; truy vấn không dấu chưa được mở rộng.

## Xác nhận

Cường cần chạy lại, rà soát báo cáo và bổ sung mã học viên trước khi xác nhận nộp bài.
