import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Du lịch Việt Nam",
    page_icon="🧭",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Trợ lý Du lịch Việt Nam.")
    st.caption("Hỏi về Huế, Đà Nẵng và Hội An.")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("🧭 Trợ lý Du lịch Việt Nam.")
st.caption("Hỏi về Huế, Đà Nẵng và Hội An.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Nguồn tham khảo"):
                st.caption(f"Phương thức: {message['retrieval_source']}")

                for source in message["sources"]:
                    metadata = source["metadata"]
                    st.markdown(
                        f"**{metadata.get('title', 'Không rõ tiêu đề')}**"
                    )
                    st.caption(
                        f"chunk:{source['id']} · "
                        f"{source.get('retrieval_method', '')} · "
                        f"score: {source.get('score', '')}"
                    )

                    if metadata.get("url"):
                        st.markdown(f"[Mở nguồn]({metadata['url']})")

                    st.write(source["content"])

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm nguồn..."):
            try:
                from src.task10_generation import generate_with_citation
                result = generate_with_citation(query, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]
                retrieval_source = result["retrieval_source"]
            except Exception as error:
                answer = "Pipeline chưa chạy được. Hãy kiểm tra cấu hình và các Task trước."
                sources = []
                retrieval_source = "none"
                st.caption(str(error))

        st.markdown(answer)

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
        st.session_state.messages.append(assistant_message)

        if sources:
            with st.expander("Nguồn tham khảo"):
                st.caption(f"Phương thức: {retrieval_source}")

                for source in sources:
                    metadata = source["metadata"]
                    st.markdown(
                        f"**{metadata.get('title', 'Không rõ tiêu đề')}**"
                    )
                    st.caption(
                        f"chunk:{source['id']} · "
                        f"{source.get('retrieval_method', '')} · "
                        f"score: {source.get('score', '')}"
                    )

                    if metadata.get("url"):
                        st.markdown(f"[Mở nguồn]({metadata['url']})")

                    st.write(source["content"])
