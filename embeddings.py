# embeddings.py  (Gemini / LangChain version)
import os
import sys
from logger import logging
from exception import CustomException

# Google Gemini embeddings wrapper (LangChain integration)
# NOTE: adjust import name if your environment uses a different package name.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LangChain-compatible Pinecone vector store use
import pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone

def init_pinecone(pinecone_api_key: str, environment: str = None):
    """
    Initialize the Pinecone client. Provide `environment` (e.g. 'us-east1-gcp') if your account requires it.
    """
    try:
        if environment:
            pinecone.init(api_key=pinecone_api_key, environment=environment)
        else:
            pinecone.init(api_key=pinecone_api_key)
        logging.info("Pinecone initialized")
    except Exception as e:
        raise CustomException(e, sys)


def create_vector_index(index_name: str, dimension: int = 3072, pinecone_api_key: str = None, environment: str = None):
    """
    Create a Pinecone index if it doesn't exist. Dimension should match your embedding model's output.
    """
    try:
        if pinecone_api_key:
            init_pinecone(pinecone_api_key, environment=environment)
        existing = pinecone.list_indexes()
        if index_name in existing:
            logging.info(f"Index '{index_name}' already exists")
        else:
            pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")
            logging.info(f"Created new index '{index_name}' with dim {dimension}")
        return index_name
    except Exception as e:
        raise CustomException(e, sys)


def add_embeddings_to_db(chunks, index_name, google_api_key=None, model_name="gemini-embedding-001", pinecone_api_key=None, environment=None):
    """
    chunks: list of {"chunk_text": ..., "source": ..., "page_number": ...}
    index_name: pinecone index name
    google_api_key: OPTIONAL override (or rely on env var GOOGLE_API_KEY)
    model_name: embedding model id to use; default uses a commonly named Gemini embedding model
    """
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=google_api_key)

        # ensure pinecone initialized
        if pinecone_api_key:
            init_pinecone(pinecone_api_key, environment=environment)

        texts = [c["chunk_text"] for c in chunks]
        metadatas = [{"source": c.get("source"), "page_number": c.get("page_number")} for c in chunks]
        ids = [f"doc-{i}" for i in range(len(texts))]

        # Use LangChain's Pinecone wrapper to upsert documents
        store = LangchainPinecone.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, index_name=index_name)
        logging.info("Added embeddings and metadata to Pinecone via LangChain wrapper")
        return store
    except Exception as e:
        logging.error(f"Error while adding embeddings: {e}")
        raise CustomException(e, sys)
