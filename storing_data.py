# storing_data.py
import os
import streamlit as st
from dotenv import load_dotenv
from logger import logging

load_dotenv()

# Read secrets / envs
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
PINECONE_ENV = st.secrets.get("PINECONE_ENV", os.getenv("PINECONE_ENV", None))

from data_ingestion import data_ingestion, data_chunking
from embeddings import create_vector_index, add_embeddings_to_db

def main():
    st.title("RAG ingestion (Gemini + LangChain)")

    uploaded_dir = st.text_input("Local directory with docs (leave blank to skip)", value="Data")
    url = st.text_input("Optional URL to ingest", value="https://callofduty.fandom.com/wiki/Call_of_Duty:_Mobile")
    index_name = st.text_input("Pinecone index name", value="aibot-gemini")
    chunk_size = st.number_input("Chunk size (words)", value=500, step=50)
    chunk_overlap = st.number_input("Chunk overlap (words)", value=50, step=10)

    if st.button("Run ingestion"):
        docs = data_ingestion(uploaded_dir if uploaded_dir else None, url if url else None)
        if not docs:
            st.warning("No documents found to ingest.")
            return

        chunks = data_chunking(docs, chunk_size, chunk_overlap)
        st.info(f"Created {len(chunks)} chunks")


        create_vector_index(index_name=index_name, dimension=3072, pinecone_api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

        # add embeddings
        store = add_embeddings_to_db(chunks, index_name=index_name, google_api_key=GOOGLE_API_KEY, pinecone_api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        st.success("Ingestion finished and embeddings added to Pinecone.")
        st.write("Index name:", index_name)
        st.write("Number of chunks added:", len(chunks))

if __name__ == "__main__":
    main()
