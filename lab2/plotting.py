import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates


def save_parallel_coordinates(
    df: pd.DataFrame,
    out_dir: str,
    *,
    metric_col: str = "metric",
    filename: str = "parallel_coordinates.png",
    include_mapping_file: bool = True,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    display_order: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
) -> str:
    """Create and save a parallel coordinates plot from grid search results.

    Returns the saved figure path.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Copy to avoid mutating caller df
    df_plot = df.copy()

    # Encode categorical columns to numeric for axes comparability
    categorical_cols: List[str] = []
    for col in df_plot.columns:
        if df_plot[col].dtype == object:
            categorical_cols.append(col)

    mappings: Dict[str, Dict[str, int]] = {}
    for col in categorical_cols:
        codes = df_plot[col].astype(str).astype("category").cat.codes
        categories = df_plot[col].astype(str).astype("category").cat.categories
        mapping: Dict[str, int] = {str(cat): int(i)
                                   for i, cat in enumerate(categories)}
        mappings[col] = mapping
        df_plot[col] = codes

    # Class column is based on metric bins (user-defined bins and labels)
    class_col = "metric_bin"
    if bins is None or labels is None:
        # Default bins consistent with user request; include 70-75 to avoid gaps
        bins = [-float("inf"), 70, 75, 80, 85, 87, float("inf")]
        labels = ["<70", "70-75", "75-80", "80-85", "85-87", ">=87"]

    df_plot[class_col] = pd.cut(
        df_plot[metric_col], bins=bins, labels=labels, right=False, include_lowest=True
    )

    if display_order is None:
        display_order = [">=87", "85-87", "80-85", "75-80", "<70", "70-75"]

    if palette is None:
        palette = {
            ">=87": "#1a9850",
            "85-87": "#66bd63",
            "80-85": "#fee08b",
            "75-80": "#fdae61",
            "<70": "#d73027",
            "70-75": "#f46d43",
        }

    present = [
        c for c in display_order if c in df_plot[class_col].astype(str).unique()]
    df_plot[class_col] = df_plot[class_col].astype("category")
    df_plot[class_col] = df_plot[class_col].cat.set_categories(
        present, ordered=True)

    # Select numeric columns (excluding class col)
    numeric_cols = [
        c for c in df_plot.columns if c != class_col and pd.api.types.is_numeric_dtype(df_plot[c])
    ]
    plot_df = df_plot[numeric_cols + [class_col]].dropna().copy()

    # Draw and save
    plt.figure(figsize=(12, 6))
    colors = [palette[c] for c in present]
    parallel_coordinates(plot_df, class_column=class_col,
                         color=colors, alpha=0.7)
    plt.title("Grid Search Parallel Coordinates")
    plt.xticks(rotation=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, filename)
    plt.savefig(plot_path)
    plt.close()

    # Save mappings for categorical inputs used on axes
    if include_mapping_file and mappings:
        mapping_path = os.path.join(out_dir, "categorical_mappings.txt")
        with open(mapping_path, "w") as f:
            for col, mapping in mappings.items():
                f.write(f"{col}:\n")
                for k, v in mapping.items():
                    f.write(f"  {k} -> {v}\n")
                f.write("\n")

    return plot_path
