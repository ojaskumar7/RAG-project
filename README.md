# RAG System using Google Gemini + LangChain + Pinecone

This project is a simple **Retrieval-Augmented Generation (RAG)** pipeline built with:

- **Google Gemini API** (via `langchain-google-genai`)
- **LangChain** for chaining + retrieval logic
- **Pinecone** for vector storage
- **Streamlit** for a small ingestion interface

It ingests documents, chunks them, embeds them using Gemini embeddings, stores them in Pinecone, and answers questions using a retrieval-aware Gemini model.

---

Folder Structure
.
├── .env
├── chains.py
├── data_ingestion.py
├── embeddings.py
├── exception.py
├── logger.py
├── storing_data.py
├── __init__.py
└── requirements.txt


