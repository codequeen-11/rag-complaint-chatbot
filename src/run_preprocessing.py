import logging
import os
import pandas as pd
from data_processor import (
    load_dataset,
    filter_products,
    preprocess
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA = "data/raw/complaints.csv"
def main():

    try:

        if not os.path.exists(RAW_DATA):
            raise FileNotFoundError(
                f"Dataset not found: {RAW_DATA}"
            )

        logging.info("Loading dataset...")

        df = pd.read_csv(
            RAW_DATA,
            low_memory=False
        )


        if df.empty:
            raise ValueError(
                "Dataset is empty"
            )


        logging.info(
            f"Loaded dataset shape: {df.shape}"
        )


    except Exception as e:

        logging.error(
            f"Preprocessing failed: {e}"
        )

        raise


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

if __name__ == "__main__":
    main()
    