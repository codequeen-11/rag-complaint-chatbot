
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vector_store(chunks):

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="vector_store"
    )

    return vector_store