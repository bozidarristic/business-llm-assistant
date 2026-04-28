import html

import pandas as pd
import streamlit as st

from app.services.rag_service import (
    build_assistant,
    build_embedder,
    build_llm,
    build_vectorstore,
    list_available_files,
    rebuild_index,
    save_uploaded_file,
)


st.set_page_config(
    page_title="Business LLM Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        background: #f7f8fb;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #5b6472;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .source-row {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.45rem;
        background: white;
    }
    .muted {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_embedder():
    return build_embedder()


@st.cache_resource(show_spinner=False)
def cached_vectorstore():
    return build_vectorstore()


@st.cache_resource(show_spinner=False)
def cached_llm():
    return build_llm()


def get_assistant():
    return build_assistant(
        embedder=cached_embedder(),
        vectorstore=cached_vectorstore(),
        llm=cached_llm(),
    )


def render_file_table(files: list[dict]) -> None:
    if not files:
        st.info("No files are available yet.")
        return

    df = pd.DataFrame(files)
    df = df.rename(
        columns={
            "name": "File",
            "category": "Source",
            "size_kb": "Size KB",
        }
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_indexed_sources() -> None:
    vectorstore = cached_vectorstore()
    count = vectorstore.count()
    st.metric("Indexed chunks", count)

    if count == 0:
        st.caption("No vector index found yet. Rebuild the index before asking questions.")
        return

    for source in vectorstore.list_sources():
        safe_source = html.escape(str(source["source"]))
        safe_type = html.escape(str(source["type"]))
        st.markdown(
            f"""
            <div class="source-row">
                <strong>{safe_source}</strong>
                <div class="muted">{safe_type} | {source["chunks"]} chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def rebuild_index_with_status() -> None:
    with st.spinner("Embedding documents and rebuilding the vector index..."):
        indexed_chunks = rebuild_index(
            embedder=cached_embedder(),
            vectorstore=cached_vectorstore(),
        )
    st.success(f"Index rebuilt successfully. {indexed_chunks} chunks are ready.")


st.markdown('<div class="app-title">Business LLM Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload business documents, rebuild the local index, and ask grounded questions.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Add files",
        type=["csv", "md", "txt"],
        accept_multiple_files=True,
        help="Supported formats: CSV, Markdown, and plain text.",
    )

    if st.button("Save and index files", use_container_width=True, disabled=not uploaded_files):
        saved_files = []
        for uploaded_file in uploaded_files:
            target = save_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
            saved_files.append(target.name)
        st.success(f"Saved {len(saved_files)} file(s).")
        rebuild_index_with_status()

    if st.button("Rebuild index", type="primary", use_container_width=True):
        rebuild_index_with_status()

    with st.expander("Show available files", expanded=False):
        render_file_table(list_available_files())

    st.divider()
    st.subheader("Indexed sources")
    render_indexed_sources()


left, right = st.columns([0.62, 0.38], gap="large")

with left:
    st.subheader("Ask a question")
    with st.form("question-form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder="Example: Draft a professional reply to Beta Manufacturing about the SLA issue.",
            height=112,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask assistant", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        elif cached_vectorstore().count() == 0:
            st.warning("Build the index first so the assistant has context.")
        else:
            with st.spinner("Retrieving context and generating an answer..."):
                answer = get_assistant().answer(question.strip())
            st.markdown("#### Answer")
            st.write(answer)

with right:
    st.subheader("Workflow")
    st.write(
        "1. Upload CSV, Markdown, or text files.\n\n"
        "2. Rebuild the local vector index.\n\n"
        "3. Ask questions grounded in the indexed business data."
    )

    st.subheader("Good test questions")
    st.code(
        "\n".join(
            [
                "What is the current status of Beta Manufacturing?",
                "Which leads should sales prioritize?",
                "What does the refund policy say?",
                "Draft a professional reply to Beta Manufacturing about the SLA issue.",
            ]
        ),
        language="text",
    )
