from pathlib import Path

from langchain_chroma import Chroma
from src.vector_store import get_embedding_model

VECTOR_DB_PATH = Path(__file__).resolve().parent.parent / "vector_store"
def load_vector_store():

    embedding_model = get_embedding_model()

    db = Chroma(
        persist_directory= str(VECTOR_DB_PATH),
        embedding_function=embedding_model
    )

    return db


def retrieve_documents(question, k=5):

    db = load_vector_store()

    docs = db.similarity_search(
        question,
        k=k
    )

    return docs

evaluation_results = []




