# RAG Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents.

## Features
- Load multiple PDFs
- Chunk text using LangChain
- Generate embeddings using HuggingFace
- Store vectors using FAISS
- Retrieve relevant chunks
- Generate answers using Ollama (llama3)
- Displays source chunks and page numbers

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama (LLaMA3)

## How it works
PDF → Chunk → Embedding → FAISS → Retrieve → LLM → Answer