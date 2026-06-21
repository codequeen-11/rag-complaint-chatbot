
import pandas as pd

from sampling import create_stratified_sample

from chunking import (
    create_documents,
    chunk_documents
)

from vector_store import (
    build_vector_store
)

print("=" * 50)
print("Loading processed dataset...")
print("=" * 50)

df = pd.read_csv(
    "data/processed/filtered_complaints.csv"
)

print(df.shape)

print("\nCreating stratified sample...")

sample_df = create_stratified_sample(
    df,
    sample_size=12000
)

print(sample_df.shape)

print("\nCreating documents...")

documents = create_documents(
    sample_df
)

print(f"Documents: {len(documents)}")

print("\nChunking documents...")

chunks = chunk_documents(
    documents
)

print(f"Chunks: {len(chunks)}")

print("\nBuilding ChromaDB...")

vector_store = build_vector_store(
    chunks
)

print("\nVector database created successfully!")
