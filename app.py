
import streamlit as st
from llm import generate_answer

st.set_page_config(
    page_title="Enterprise ML Knowledge Assistant",
    page_icon=":robot_face:",
    layout="wide"
)

st.title("Enterprise Data & ML Knowledge Assistant")
st.caption("GenAI | RAG | ChromaDB | Ollama | Llama 3.2")

question = st.text_input(
    "Ask a question about ML scoring or QA validation",
    placeholder="What validations are performed for Contact BP?"
)

if st.button("Generate Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Generating answer..."):
                result = generate_answer(question)

            st.subheader("Generated Answer")
            st.write(result["answer"])

            st.subheader("Performance Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Latency", f"{result['latency']} sec")
            col2.metric("Input Tokens", result["input_tokens"])
            col3.metric("Output Tokens", result["output_tokens"])

            st.subheader("Retrieved Evidence")

            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"Source {i}"):
                    st.write(source["text"])
                    st.caption(
                        f"Chunk ID: {source['metadata']['chunk_id']}"
                    )
                    st.caption(
                        f"Distance: {source['distance']:.4f}"
                    )

        except Exception as error:
            st.error(f"Error: {error}")
