import pandas as pd


class NullAnalyzer:
    """
    This class takes as input a df and computes the number of NULLs per column, stratifying for a column.
    """

    def __init__(self, threshold: float, stratify_col: str = "city"):
        if threshold < 0 or threshold > 100:
            raise ValueError("The percentage threshold must be in [0, 100]")

        self.threshold = threshold
        self.strat_col = stratify_col

    def check_nulls_globally(self, df: pd.DataFrame):

        # COMPUTING GLOBAL NULL PCT
        global_null_pct = df.isna().mean() * 100.0

        # IDENTIFYING COLUMNS > THRESHOLD_PCT
        flagged_columns = global_null_pct[global_null_pct > self.threshold]

        print(f"OVERALL NULL PCT (Threshold > {self.threshold:.2f}%)")
        if flagged_columns.empty:
            print("\tNo columns exceeded the specified global null threshold.\n")
        else:
            for col, val in flagged_columns.items():
                print(f"\tColumn: {col:<30} | Global Nulls: {val:.2f}%")
            print()

        return

    def check_nulls_strat(
        self, df: pd.DataFrame, delta_threshold: float
    ) -> pd.DataFrame:
        if self.strat_col not in df.columns:
            raise KeyError(f"DF must include column {self.strat_col}")

        df_mi = df[df[self.strat_col].str.upper() == "MI"]
        df_bg = df[df[self.strat_col].str.upper() == "BG"]

        comparison = pd.DataFrame(
            {"mi": df_mi.isna().mean() * 100, "bg": df_bg.isna().mean() * 100}
        ).fillna(0)

        comparison["delta"] = (comparison["mi"] - comparison["bg"]).abs()
        above_thresh_cols = comparison[comparison["delta"] > delta_threshold]

        print("Data Integrity Difference between BG and MI:")
        if above_thresh_cols.empty:
            print("\tData integrity is consistent between cities")
        else:
            for col, row in above_thresh_cols.iterrows():
                print(
                    f"\tColumn: {col:<30} | "
                    f"\tMI Nulls: {row['mi_null_pct']:>6.2f}% | "
                    f"\tBG Nulls: {row['bg_null_pct']:>6.2f}% | "
                    f"\tDelta: {row['absolute_delta']:>6.2f}%"
                )
            print()
        return
