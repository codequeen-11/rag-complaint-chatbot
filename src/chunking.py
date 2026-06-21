
from langchain_core.documents import Document

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
) 

def create_documents(df):
    """
    Convert dataframe rows to LangChain Documents.
    """

    documents = []

    for _, row in df.iterrows():

        doc = Document(
            page_content=row["clean_narrative"],
            metadata={
                "complaint_id": str(row["complaint_id"]),
                "product_category":
                    row["product_category"]
            }
        )

        documents.append(doc)

    return documents


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks