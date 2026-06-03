import pandas as pd


def normalize_neigh_col(df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:

    # Computing the number of occurrencies of each neigh. in MI
    milan_neigh_counts = df[df["city"] == "MI"]["neighbourhood_cleansed"].value_counts()

    # Filtering only the neighborhoods with at least "threshold" occurrences
    major_milan_neighborhoods = milan_neigh_counts[
        milan_neigh_counts >= threshold
    ].index

    # Defining custom function to assign each neighborhood to a bucket
    def categorize_location(row):

        # For Bergamo, we bucket all towns into either Bergamo City or Province
        if row["city"] == "BG":

            if row["neighbourhood_cleansed"] == "Bergamo":
                return "Bergamo_City"

            else:
                return "Bergamo_Province"

        elif row["city"] == "MI":

            if row["neighbourhood_cleansed"] in major_milan_neighborhoods:
                return row["neighbourhood_cleansed"]

            else:
                return "Milan_Other"

        return "Unknown"

    df["neighbourhood"] = df.apply(categorize_location, axis=1)

    return df
