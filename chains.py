# chains.py  (Gemini / LangChain version)
import sys
from logger import logging
from exception import CustomException

# LangChain Google GenAI chat model wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain imports (chain composition)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone as LangchainPinecone

def create_retriever(index_name, embeddings):
    """
    Create a retriever from an existing Pinecone index and an embeddings instance.
    embeddings: the same embeddings object (GoogleGenerativeAIEmbeddings) used when building the index.
    """
    try:
        docsearch = LangchainPinecone(index_name=index_name, embedding=embeddings)
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        logging.info("Retriever is created")
        return retriever
    except Exception as e:
        logging.error(f"Failed to create retriever: {e}")
        raise CustomException(e, sys)


def create_rag_chain(retriever, model_name="gemini-2.5-pro", temperature=0.2, max_tokens=512):
    """
    Create a retrieval-augmented QA chain using Gemini chat model via LangChain.
    Returns a RetrievalQA (callable) instance.
    """
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens)

        # system prompt: instruct the assistant how to use retrieved context
        system_prompt = (
            "You are an AI assistant. Use the provided context to answer the user's question. "
            "If the answer is not in the context, reply: 'The required information is not available.' "
            "Keep answers concise unless the user asks for more detail."
        )

        prompt_template = PromptTemplate(
            template="{system_message}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["system_message", "context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template, "verbose": False}
        )
        logging.info("RAG chain created (RetrievalQA)")
        return qa_chain
    except Exception as e:
        logging.error(f"Failed to create RAG chain: {e}")
        raise CustomException(e, sys)


def ask_question(question: str, qa_chain):
    """
    Ask the QA chain a question and return the answer and sources (if available).
    """
    try:
        result = qa_chain.run({"query": question}) if hasattr(qa_chain, "run") else qa_chain({"query": question})
        # RetrievalQA's run often returns a string answer; when using return_source_documents=True the call pattern differs.
        # To be resilient, check result type:
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or result.get("text")
        else:
            answer = result
        logging.info("Query answered by RAG chain")
        return answer
    except Exception as e:
        logging.error(f"Failed to get answer: {e}")
        raise CustomException(e, sys)


def clear_memory(memory):
    try:
        memory.clear()
        logging.info("Memory cleared")
    except Exception as e:
        logging.error(f"Failed to clear memory: {e}")
        raise CustomException(e, sys)
