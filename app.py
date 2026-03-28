import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: white;
        border-radius: 10px;
        border: 1px solid #334155;
        padding: 10px;
    }

    .stTextArea textarea {
        background-color: #1e293b;
        color: white;
        border-radius: 10px;
    }

    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    .custom-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }

    .small-text {
        color: #cbd5e1;
        font-size: 14px;
    }

    .answer-box {
        background-color: #111827;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #22c55e;
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center;'>🤖 RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #cbd5e1;'>Ask questions from your PDF documents using FAISS + HuggingFace Embeddings + Ollama</p>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Choose Ollama Model", ["llama3", "gemma3:4b"], index=0)
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top K Chunks", 1, 5, 3, 1)

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "📂 Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs and building vector store..."):
        documents = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(docs)

            os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)

        llm = ChatOllama(model=model_name)

        st.success("✅ PDFs processed successfully!")

    # ---------------- STATS ----------------
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='custom-card'><h3>📄 Documents</h3><p class='small-text'>{len(uploaded_files)} uploaded</p></div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        f"<div class='custom-card'><h3>📑 Pages</h3><p class='small-text'>{len(documents)} total pages</p></div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        f"<div class='custom-card'><h3>✂️ Chunks</h3><p class='small-text'>{len(chunks)} chunks created</p></div>",
        unsafe_allow_html=True
    )

    # ---------------- QUESTION INPUT ----------------
    st.subheader("💬 Ask a question")
    user_question = st.text_input("Type your question here")

    if user_question:
        with st.spinner("Searching relevant content and generating answer..."):
            results = vectorstore.similarity_search(user_question, k=top_k)
            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below to answer the question.
- If the answer is not present in the context, say: "I could not find that in the documents."
- Keep the answer clear and concise.

Context:
{context}

Question:
{user_question}

Answer:
"""

            response = llm.invoke(prompt)

        # ---------------- ANSWER ----------------
        st.subheader("✅ Answer")
        st.markdown(
            f"<div class='answer-box'>{response.content}</div>",
            unsafe_allow_html=True
        )

        # ---------------- SOURCES ----------------
        with st.expander("📚 Retrieved Source Chunks"):
            for i, doc in enumerate(results, start=1):
                st.markdown(
                    f"""
                    <div class='custom-card'>
                        <h3>Chunk {i}</h3>
                        <p class='small-text'><b>Source:</b> {doc.metadata.get('source', 'Unknown')}</p>
                        <p class='small-text'><b>Page:</b> {doc.metadata.get('page', 'N/A')}</p>
                        <p>{doc.page_content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

else:
    st.info("Upload PDF files to start chatting with your documents.")