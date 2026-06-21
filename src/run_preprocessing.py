from data_processor import (
    load_dataset,
    filter_products,
    preprocess
)

RAW_DATA = "data/raw/complaints.csv"

OUTPUT = (
    "data/processed/"
    "filtered_complaints.csv"
)

df = load_dataset(RAW_DATA)

print("Loaded:", df.shape)

filtered_df = filter_products(df)

print("Filtered:", filtered_df.shape)

processed_df = preprocess(filtered_df)

print("Processed:", processed_df.shape)

processed_df.to_csv(
    OUTPUT,
    index=False
)

print("Saved successfully.")