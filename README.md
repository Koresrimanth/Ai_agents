# GenAI RAG Agent

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on PDF and Excel files using LangChain, ChromaDB, and Streamlit.

## Features

- Upload and process PDF/XLSX files
- Store embeddings in ChromaDB
- Query using OpenAI GPT-4
- Interactive frontend with Streamlit

## Setup

1. Add your documents to the `data/` folder
2. Put your OpenAI key in `.env`
3. Run:
    - `python ingest.py`
    - `streamlit run app.py`
