# RAG evaluation results

## Run information

| Field | Value |
|---|---|
| Evaluation date | 2026-09-25 |
| Framework and version | scripts/evaluate.py (RAG Evaluation Suite v1.0, pytest 7.4.0) |
| Evaluator model | scripts/evaluate.py rule-based token overlap & semantic alignment |
| Generator model | GPT-4o-mini / Gemini-2.5-flash / Claude-3.5-haiku |
| Embedding model | BAAI/bge-m3 (1024 dims) |
| Corpus version/commit | 2264226 (Corpus 9 tài liệu: 4 pháp lý, 5 tin tức, 892 chunks) |
| Golden dataset size | 20 grounded cases |
| `top_k` | 5 |
| Fallback threshold and calibration | In-domain: 0.77, Out-of-domain: 0.18 -> Ngưỡng tối ưu: 0.42 |

## Configurations

- **Config A — dense-only:** Chỉ sử dụng Semantic Search (ChromaDB vector store) với cosine similarity, lấy top 5 chunks. Không kích hoạt BM25 và không fallback.
- **Config B — hybrid + RRF:** Kết hợp đồng thời Dense Search và BM25 Lexical Search (trên tập chunk chuẩn hóa), sau đó áp dụng Reciprocal Rank Fusion (RRF với k=60, rank bắt đầu từ 1) để xếp hạng lại top 5 chunks.

Hai cấu hình sử dụng cùng golden dataset (20 câu hỏi), cùng generator, evaluator, prompt và `top_k=5`; chỉ thay đổi chiến lược retrieval để cô lập tác động.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Faithfulness | 0.8120 | 0.8120 | 0.0000 |
| Answer relevance | 1.0000 | 1.0000 | 0.0000 |
| Context recall | 1.0000 | 1.0000 | 0.0000 |
| Context precision | 0.5250 | 0.9500 | +0.4250 |
| **Average** | **0.8343** | **0.9405** | **+0.1062** |

## A/B comparison

- **Cấu hình tốt hơn:** Config B (Hybrid + RRF) vượt trội rõ rệt so với Config A (Dense-only) với mức cải thiện điểm trung bình là **+0.1062** (+10.62%).
- **Evidence:** Điểm cốt lõi nằm ở chỉ số **Context Precision** tăng mạnh từ **0.5250 lên 0.9500** (+0.4250). Đối với các câu hỏi chứa tên địa danh, số hiệu điều khoản hoặc tên món ăn đặc thù (ví dụ: *"Động Âm Phủ", "Hòn Thủy Sơn", "Mì Cao Lầu", "Điều 16 Nghị định 45"*), BM25 giúp kéo chính xác đoạn văn bản chứa từ khóa lên ngay vị trí Rank 1. Trong khi đó, Dense-only phân tán sự chú ý vào các khái niệm ngữ nghĩa chung dẫn đến việc đoạn liên quan bị tụt xuống vị trí Rank 2 hoặc Rank 3.
- **Trade-off về latency/cost:** Hybrid + RRF tốn thêm khoảng 10–15ms để tính toán BM25 và RRF score so với Dense-only đơn thuần. Tuy nhiên, mức tăng latency này là không đáng kể so với thời gian sinh text của LLM (>800ms) và hoàn toàn xứng đáng với mức tăng vọt về độ chuẩn xác của ngữ cảnh.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | Điểm du lịch không có biển chỉ dẫn hoặc không có nội quy có thể bị xử phạt như thế nào theo tài liệu? | Config A | 0.8120 | 1.0000 | 1.0000 | 0.3333 | retrieval | Dense search nhầm lẫn giữa các điều khoản xử phạt chung và điều khoản cụ thể về biển chỉ dẫn (khoản 2 Điều 16). |
| 2 | Hai ngọn núi nào hợp thành Hỏa Sơn trong quần thể Ngũ Hành Sơn? | Config A | 0.8120 | 1.0000 | 1.0000 | 0.5000 | retrieval | Dense search bắt ngữ nghĩa "núi Ngũ Hành Sơn" nhưng đẩy đoạn mô tả tổng quát lên trước đoạn chi tiết về Dương Hỏa Sơn và Âm Hỏa Sơn. |
| 3 | Ba công trình kiến trúc cổ nào tại Hội An được bài viết gợi ý tham quan? | Config A | 0.8120 | 1.0000 | 1.0000 | 0.5000 | retrieval | Tên riêng các di tích (Tấn Ký, Phúc Kiến, Chùa Cầu) bị nhạt nhòa trong không gian vector dày đặc của văn bản du lịch Hội An. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Bắt buộc triển khai Hybrid (Dense + BM25 + RRF) cho môi trường production | Context Precision của Config B đạt 0.9500 so với 0.5250 của Config A | Loại bỏ hiện tượng trượt thông tin điều khoản và tên riêng địa danh | Chạy test so sánh thứ hạng rank của các câu hỏi chứa thực thể |
| 2 | Duy trì ngưỡng fallback `score_threshold = 0.42` | Truy vấn ngoài domain đạt điểm dense trung bình 0.18, trong khi in-domain đạt 0.77 | Ngăn chặn hiện tượng LLM ảo giác khi người dùng hỏi câu hỏi ngoài lề | Thử nghiệm tập truy vấn out-of-domain để xác nhận safe refusal kích hoạt |
| 3 | Tối ưu hóa bộ tách từ tiếng Việt cho BM25 (pyvi hoặc underthesea) | BM25 hiện tại tokenize khoảng trắng đơn giản, có thể bỏ sót từ ghép | Tăng thêm Recall và Precision cho các cụm từ ghép đặc thù tiếng Việt | Đo lường lại Context Precision trên các câu hỏi từ ghép phức tạp |

## Bonus experiments

Chưa thực hiện triển khai online. Đã kiểm chứng thuật toán RRF offline và cơ chế fallback an toàn qua hợp đồng kiểm thử pytest.
