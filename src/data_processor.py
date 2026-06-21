"""
Task 1: EDA helpers and complaint text preprocessing
"""

import re
import pandas as pd


TARGET_PRODUCTS = {
    "credit card": "Credit Card",
    "credit card or prepaid card": "Credit Card",

    "personal loan": "Personal Loan",
    "personal loans": "Personal Loan",
    "payday loan, title loan, or personal loan": "Personal Loan",

    "checking or savings account": "Savings Account",

    "money transfers": "Money Transfer",
    "money transfer, virtual currency, or money service": "Money Transfer",
}


_BOILERPLATE = re.compile(
    r"(i am writing to (file|submit|report) a complaint.*?[.!]"
    r"|this is a complaint (about|regarding).*?[.!]"
    r"|i would like to (file|report|submit).*?[.!])",
    re.IGNORECASE,
)

_SPECIAL_CHARS = re.compile(r"[^a-z0-9\s.,!?;:()\-']")
_WHITESPACE = re.compile(r"\s+")

def load_dataset(path: str):
    """
    Load CFPB dataset and standardize column names.
    """

    df = pd.read_csv(
        path,
        # low_memory=False,
        engine="python"
    )

    df.columns = [
        c.strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    return df


def find_column(df, candidates):
    """
    Find first matching column.
    """

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find any of {candidates}"
    )


def filter_products(df):
    """
    Keep only target products and complaints with narratives.
    """

    product_col = find_column(
        df,
        ["product"]
    )

    narrative_col = find_column(
        df,
        [
            "consumer_complaint_narrative",
            "narrative"
        ]
    )

    df = df.copy()

    df["product_category"] = (
        df[product_col]
        .str.lower()
        .str.strip()
        .map(TARGET_PRODUCTS)
    )

    df = df[
        df["product_category"].notna()
    ]

    df = df[
        df[narrative_col].notna()
    ]

    df = df[
        df[narrative_col].str.strip() != ""
    ]

    df = df.rename(
        columns={
            narrative_col: "narrative"
        }
    )

    return df.reset_index(drop=True)


def clean_narrative(text):
    """
    Clean complaint text.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = _BOILERPLATE.sub("", text)

    text = re.sub(
        r"\bx{2,}\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = _SPECIAL_CHARS.sub(
        " ",
        text
    )

    text = _WHITESPACE.sub(
        " ",
        text
    )

    return text.strip()


def preprocess(df):
    """
    Apply preprocessing.
    """

    df = df.copy()

    df["clean_narrative"] = (
        df["narrative"]
        .apply(clean_narrative)
    )

    df = df[
        df["clean_narrative"].str.len() > 20
    ]

    df["word_count"] = (
        df["clean_narrative"]
        .str.split()
        .str.len()
    )
    df = df[
        df["word_count"] >= 3
    ]

    return df.reset_index(drop=True)


def narrative_statistics(df):
    """
    Narrative length statistics.
    """

    return {
        "count": len(df),
        "mean_words": df["word_count"].mean(),
        "median_words": df["word_count"].median(),
        "min_words": df["word_count"].min(),
        "max_words": df["word_count"].max()
    }


def narrative_availability(df):
    """
    Count missing narratives.
    """

    narrative_col = find_column(
        df,
        [
            "consumer_complaint_narrative",
            "narrative"
        ]
    )

    return {
        "with_narrative":
            df[narrative_col].notna().sum(),

        "without_narrative":
            df[narrative_col].isna().sum()
    }