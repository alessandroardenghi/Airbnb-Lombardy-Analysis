import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_kde_by_city(
    df: pd.DataFrame,
    target_column: str = "availability_90",
    city_column: str = "city",
    rng: Tuple[int, int] = (0, 90),
):

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is missing from the DataFrame."
        )

    if city_column not in df.columns:
        raise KeyError(f"City column '{city_column}' is missing from the DataFrame.")

    # ISOLATE TARGET VARIABLES FOR MI AND BG
    milan_series: pd.Series = df[df[city_column] == "MI"][target_column].dropna()
    bergamo_series: pd.Series = df[df[city_column] == "BG"][target_column].dropna()

    if milan_series.empty:
        raise ValueError(
            "The Milan (MI) market subset contains no valid data for plotting."
        )
    if bergamo_series.empty:
        raise ValueError(
            "The Bergamo (BG) market subset contains no valid data for plotting."
        )

    # INITIALIZE SEABORN
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # PLOT MILAN
    sns.kdeplot(
        data=milan_series,
        ax=axes[0],
        fill=True,
        color="#1f77b4",
        linewidth=1.0,
    )
    axes[0].set_title("Market Segment: Milan (MI)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(target_column)
    axes[0].set_ylabel("Density")

    # PLOT BG
    sns.kdeplot(
        data=bergamo_series,
        ax=axes[1],
        fill=True,
        color="#ff7f0e",
        linewidth=1.0,
    )
    axes[1].set_title("Market Segment: Bergamo (BG)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(target_column)
    axes[1].set_ylabel("Density")

    # BOUNDARIES SET WITH MIN AND MAX AVAILABILITY
    for ax in axes:
        ax.set_xlim(rng[0], rng[1])

    fig.suptitle(
        f"Cross-Market Kernel Density Estimation (KDE) Comparison: {target_column}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return


def generate_violin_plot_by_city(
    df: pd.DataFrame,
    target_column: str = "availability_90",
    city_column: str = "city",
    rng: Tuple[int, int] = (0, 90),
):

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is missing from the DataFrame schema."
        )
    if city_column not in df.columns:
        raise KeyError(
            f"City identifier column '{city_column}' is missing from the DataFrame schema."
        )

    # Filter strictly for the two target markets to ensure clean comparison
    df = df[df[city_column].isin(["MI", "BG"])].dropna(subset=[target_column])

    if df.empty:
        raise ValueError(
            "The filtered dataset contains zero valid observations for the specified cities."
        )

    # INITIALIZE SEABORN
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    market_palette = {"MI": "#1f77b4", "BG": "#ff7f0e"}

    # GENERATE VIOLIN PLOT
    sns.violinplot(
        data=df,
        x=city_column,
        y=target_column,
        ax=ax,
        palette=market_palette,
        hue=city_column,
        inner="box",
        linewidth=1.5,
        cut=0,
        legend=False,
    )

    # BOUNDARIES SET WITH MIN AND MAX OCCUPANCY PER YEAR
    ax.set_ylim(rng[0], rng[1])

    ax.set_title(
        f"Cross-Market Capacity Distribution Analysis: {target_column}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Market Segment", fontsize=12, labelpad=10)
    ax.set_ylabel("Nights Booked (Next 90 Days)", fontsize=12, labelpad=10)

    plt.tight_layout()

    return


def plot_categorical_barchart(
    df: pd.DataFrame,
    category_col: str,
    title: Optional[str] = None,
) -> None:

    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not found in the DataFrame.")

    if df.empty:
        raise ValueError("The provided DataFrame contains no data.")

    # Calculate exact order of categories by descending frequency
    value_counts = df[category_col].value_counts()
    category_order = value_counts.index

    # Initialize plotting environment
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the bar chart
    sns.countplot(
        data=df,
        x=category_col,
        order=category_order,
        palette="viridis",
        hue=category_col,  # Assigned to hue to suppress Seaborn future warnings with palettes
        legend=False,
        ax=ax,
    )

    # Automatically annotate bars with their exact count values
    for container in ax.containers:
        ax.bar_label(container, padding=3, fmt="%d", fontsize=10)

    # Apply formatting
    plot_title = title if title else f"Frequency Distribution of {category_col}"
    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(category_col.replace("_", " ").title(), fontsize=12, labelpad=10)
    ax.set_ylabel("Count", fontsize=12, labelpad=10)

    # Rotate x-axis labels if there are many categories or long strings
    if len(category_order) > 4 or any(len(str(cat)) > 10 for cat in category_order):
        plt.xticks(rotation=45, ha="right")

    # Expand the y-axis limit slightly to accommodate the bar annotations
    current_ymax = ax.get_ylim()[1]
    ax.set_ylim(0, current_ymax * 1.1)

    plt.tight_layout()

    return
