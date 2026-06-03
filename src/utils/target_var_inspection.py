import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_kde(
    df: pd.DataFrame,
    target_column: str = "availability_90",
    city_column: str = "city",
    rng: Tuple[int, int] = (0, 90),
):

    # VALIDATION ON COLUMNS
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is missing from the DataFrame."
        )

    if city_column not in df.columns:
        raise KeyError(f"City column '{city_column}' is missing from the DataFrame.")

    # ISOLATE TARGET VARIABLES FOR MI AND BG
    milan_series = df[df[city_column] == "MI"][target_column].dropna()
    bergamo_series = df[df[city_column] == "BG"][target_column].dropna()

    if milan_series.empty:
        raise ValueError("The Milan Listings contain no valid data for plotting")
    if bergamo_series.empty:
        raise ValueError("The Bergamo Listings contain no valid data for plotting")

    # INITIALIZE SEABORN
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # PLOT MILAN
    sns.kdeplot(
        data=milan_series,
        ax=axes[0],
        fill=True,
        color="blue",
        linewidth=1.0,
    )

    axes[0].set_title("City: Milan (MI)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(target_column)
    axes[0].set_ylabel("Density")

    # PLOT BG
    sns.kdeplot(
        data=bergamo_series,
        ax=axes[1],
        fill=True,
        color="orange",
        linewidth=1.0,
    )
    axes[1].set_title("City: Bergamo (BG)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(target_column)
    axes[1].set_ylabel("Density")

    # BOUNDARIES SET WITH MIN AND MAX AVAILABILITY
    for ax in axes:
        ax.set_xlim(rng[0], rng[1])

    fig.suptitle(
        f"Cross-City Kernel Density Estimation Comparison: {target_column}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return


def plot_barchart(df: pd.DataFrame, category_col: str) -> None:

    # VALIDATION ON COLUMN
    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not found in the DataFrame.")

    if df.empty:
        raise ValueError("The provided DataFrame contains no data.")

    # COMPUTING THE FREQUENCY OF EACH CATEGORY
    value_counts = df[category_col].value_counts()
    category_order = value_counts.index

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.countplot(
        data=df,
        x=category_col,
        order=category_order,
        palette="viridis",
        hue=category_col,
        legend=False,
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3, fmt="%d", fontsize=10)

    plot_title = f"Frequency Distribution of {category_col}"
    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(category_col.replace("_", " ").title(), fontsize=12, labelpad=10)
    ax.set_ylabel("Count", fontsize=12, labelpad=10)

    if len(category_order) > 4 or any(len(str(cat)) > 10 for cat in category_order):
        plt.xticks(rotation=45, ha="right")

    current_ymax = ax.get_ylim()[1]
    ax.set_ylim(0, current_ymax * 1.1)

    plt.tight_layout()

    return


def plot_log_transformed_distr(
    df: pd.DataFrame, variable_col: str, city_col: str = "city"
):

    plot_df = df[[variable_col, city_col]].copy()

    # APPLYING LOG(1 + X) TRANSFORMATION TO THE TARGET VARIABLE
    log_col_name = f"log_{variable_col}"
    plot_df[log_col_name] = np.log1p(plot_df[variable_col])

    g = sns.FacetGrid(
        plot_df,
        col=city_col,
        hue=city_col,
        palette="viridis",
        sharey=False,
        height=4,
        aspect=1.3,
    )
    g.map_dataframe(
        sns.histplot, x=log_col_name, kde=True, bins=30, alpha=0.6, stat="density"
    )
    g.set_titles(col_template="{col_name} Market")
    g.set_axis_labels(f"Log({variable_col} + 1)", "Density")
    plt.tight_layout()
    plt.show()
