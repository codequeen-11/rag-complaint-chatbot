import pandas as pd

def create_stratified_sample(
    df,
    sample_size=12000,
    random_state=42
):
    """
    Create a stratified sample while preserving
    product category proportions.
    """

    sampled_groups = []

    total_rows = len(df)

    for category, group in df.groupby("product_category"):

        proportion = len(group) / total_rows

        n_samples = round(
            proportion * sample_size
        )

        sampled_group = group.sample(
            n=n_samples,
            random_state=random_state
        )

        sampled_groups.append(sampled_group)

    sample_df = pd.concat(
        sampled_groups,
        ignore_index=True
    )

    return sample_df