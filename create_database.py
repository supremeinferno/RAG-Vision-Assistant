import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from config import CHROMA_DB_PATH

CHROMA_DB_PATH = os.path.join(
    tempfile.gettempdir(),
    "chroma_db"
)


def load_embeddings():
    """
    Load the embedding model.
    """

    return MistralAIEmbeddings()


def create_vector_database(pdf_path: str):
    """
    Create a new Chroma vector database from a PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the uploaded PDF.

    Returns
    -------
    tuple
        (pages, chunks)
    """

    # Remove previous database
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
    )

    chunks = splitter.split_documents(documents)

    # Create embeddings
    embedding_model = load_embeddings()

    # Create Chroma database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
    )

    # Persist database (older versions of Chroma require this)
    try:
        vectorstore.persist()
    except Exception:
        pass

    return len(documents), len(chunks)