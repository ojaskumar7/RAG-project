# data_ingestion.py
import os
import glob
import requests
from logger import logging
from exception import CustomException

def read_local_text_files(directory_path):
    """
    Read .txt and .md files from directory and return list of dicts: {"text": ..., "source": path}
    """
    docs = []
    try:
        patterns = ["*.txt", "*.md"]
        for p in patterns:
            for path in glob.glob(os.path.join(directory_path, p)):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append({"text": text, "source": path})
                except Exception as e:
                    logging.error(f"Failed to read {path}: {e}")
        return docs
    except Exception as e:
        raise CustomException(e, None)


def fetch_url_text(url):
    """
    Simple fetcher for HTML pages (returns raw text / fallback).
    For production you might want a proper HTML parser (BeautifulSoup) and extraction.
    """
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        # return raw text. Users can replace with more sophisticated HTML extraction.
        return [{"text": r.text, "source": url}]
    except Exception as e:
        logging.error(f"Failed to fetch URL {url}: {e}")
        return []


def data_ingestion(directory_path: str = None, url: str = None):
    """
    Top-level ingestion function that returns a list of simple documents:
    [{"text": "...", "source": "..."}]
    """
    documents = []
    try:
        if directory_path and os.path.isdir(directory_path):
            logging.info(f"Reading local files from {directory_path}")
            documents.extend(read_local_text_files(directory_path))

        if url:
            logging.info(f"Fetching content from {url}")
            documents.extend(fetch_url_text(url))

        logging.info(f"Ingested {len(documents)} documents")
        return documents
    except Exception as e:
        raise CustomException(e, None)


def data_chunking(documents, chunk_size=500, chunk_overlap=50):
    """
    Very simple splitter: split by words into chunks (you can replace with token-based splitters).
    Returns a list of chunks: {"chunk_text": str, "source": original_source, "page_number": optional}
    """
    chunks = []
    try:
        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            words = text.split()
            i = 0
            page = 0
            while i < len(words):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                chunks.append({"chunk_text": chunk_text, "source": source, "page_number": page})
                i += chunk_size - chunk_overlap
                page += 1
        logging.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    except Exception as e:
        raise CustomException(e, None)
