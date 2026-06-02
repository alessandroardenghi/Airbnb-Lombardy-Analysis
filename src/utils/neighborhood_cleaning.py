import pandas as pd


def normalize_neighborhood(df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
    milan_counts = df[df["city"] == "MI"]["neighbourhood_cleansed"].value_counts()
    major_milan_neighborhoods = milan_counts[milan_counts >= threshold].index

    def categorize_location(row):
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
