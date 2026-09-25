# Individual contribution report

## Thông tin

- Họ và tên: Nguyễn Đình Anh Đức
- Mã học viên: 2A202602856
- Nhóm: Chatbot RAG Du lịch Việt Nam
- Repository/branch: `K4-L3B-RAG-Pipeline` / `NguyenDinhAnhDuc-02856`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Generation có citation | Triển khai sắp xếp chunks, định dạng context có nguồn, gọi LLM theo provider và xử lý thiếu bằng chứng/lỗi provider theo contract | `src/task10_generation.py` | Done |
| Giao diện Streamlit | Đang nối giao diện chat với generation; hiển thị câu trả lời, nguồn và thông tin truy xuất; lưu câu trả lời cùng nguồn trong lịch sử | `app.py` | Done |

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Dùng ID chunk ổn định trong citation để citation không bị lệch khi sắp xếp lại context
   **Lý do/evidence:** ID chunk được giữ xuyên suốt pipeline và có thể đối chiếu với `sources`
   **Trade-off:** Citation theo ID kém thân thiện hơn nhãn ngắn như `[S1]`; giao diện cần hiển thị ID cùng thông tin nguồn

2. **Quyết định:** Lưu cả câu trả lời và danh sách nguồn trong `st.session_state`
   **Lý do/evidence:** Nguồn vẫn hiển thị khi Streamlit chạy lại sau mỗi lượt chat.
   **Trade-off:** Session lưu thêm nội dung các chunks đã truy xuất.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: Quần thể Ngũ Hành Sơn gồm những ngọn núi nào?
- Kết quả trước/sau nếu có: Quần thể Ngũ Hành Sơn gồm có 6 ngọn núi: Thủy Sơn, Kim Sơn, Hỏa Sơn (gồm Dương Hỏa Sơn và Âm Hỏa Sơn), Thổ Sơn, Mộc Sơn [chunk:1].
- Lỗi đã phát hiện và cách xử lý:

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Chưa xác nhận citation và câu trả lời trên corpus thật với provider LLM
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Kiểm thử citation trên nhiều kết quả truy xuất và xác nhận nguồn hiển thị khớp với từng citation

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Nguyễn Đình Anh Đức
