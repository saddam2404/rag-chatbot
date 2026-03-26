import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

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

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(chunks, embeddings)

st.success("FAISS vector store created successfully!")

if chunks:
    st.subheader("First chunk preview")
    st.write(chunks[0].page_content[:500])