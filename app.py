import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

st.title("RAG Chatbot")

documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        documents.extend(loader.load())

st.write("Total pages loaded:", len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

st.write("Total chunks created:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

st.success("FAISS vector store created successfully!")

llm = ChatOllama(model="llama3")

user_question = st.text_input("Ask a question about your PDFs")

if user_question:
    results = vectorstore.similarity_search(user_question, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a helpful assistant.
Answer the question only from the provided context.
If the answer is not in the context, say: "I could not find that in the documents."

Context:
{context}

Question:
{user_question}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(results, start=1):
            st.write(f"### Chunk {i}")
            st.write(doc.page_content)
            st.write("---")