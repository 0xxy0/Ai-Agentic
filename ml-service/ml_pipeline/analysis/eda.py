"""
analysis/eda.py  --  Exploratory Data Analysis
===============================================
Generates:
    * Class distribution bar chart
    * Churn rate by contract type
    * Monthly charges distribution (churned vs. retained)
    * Tenure boxplot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

_PALETTE = {"churned": "#e63946", "retained": "#2a9d8f"}
_OUT_DIR  = Path("outputs")
_OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
_SEP = "=" * 60


def _save(fig: plt.Figure, name: str) -> Path:
    path = _OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved -> {path}")
    return path


def plot_class_distribution(df: pd.DataFrame) -> Path:
    counts = df["churn"].value_counts()
    labels = ["Retained", "Churned"]
    colors = [_PALETTE["retained"], _PALETTE["churned"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200, f"{cnt:,}", ha="center", fontsize=11)
    ax.set_title("Churn Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Customers")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return _save(fig, "01_class_distribution.png")


def plot_churn_by_contract(df: pd.DataFrame) -> Path:
    rate = df.groupby("contract")["churn"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(rate.index, rate.values, color="#457b9d", edgecolor="white")
    for bar, val in zip(bars, rate.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold", pad=12)
    return _save(fig, "02_churn_by_contract.png")


def plot_monthly_charges_dist(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, _PALETTE["retained"], "Retained"),
                                (1, _PALETTE["churned"],  "Churned")]:
        subset = df.loc[df["churn"] == label, "monthlycharges"]
        sns.kdeplot(subset, ax=ax, color=color, fill=True, alpha=0.4, label=name, linewidth=2)
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Density")
    ax.set_title("Monthly Charges Distribution by Churn Status",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend()
    return _save(fig, "03_monthly_charges_dist.png")


def plot_tenure_churn(df: pd.DataFrame) -> Path:
    df_plot = df.copy()
    df_plot["Churn Status"] = df_plot["churn"].map({0: "Retained", 1: "Churned"})
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = {"Retained": _PALETTE["retained"], "Churned": _PALETTE["churned"]}
    sns.boxplot(data=df_plot, x="Churn Status", y="tenure",
                palette=palette, ax=ax, width=0.4, order=["Retained", "Churned"])
    ax.set_title("Tenure Distribution by Churn Status", fontsize=14, fontweight="bold", pad=12)
    return _save(fig, "04_tenure_churn.png")


def run_eda(df: pd.DataFrame) -> None:
    print(f"\n{_SEP}")
    print("  [Step 6] Exploratory Data Analysis")
    print(_SEP)
    
    # Sample for plotting if dataset is huge
    if len(df) > 100_000:
        print(f"[EDA] Large dataset detected ({len(df):,} rows). Sampling 100,000 for plots.")
        df_plot = df.sample(100_000, random_state=42)
    else:
        df_plot = df

    plot_class_distribution(df) # distribution should use full data for counts
    if "contract" in df_plot.columns:
        plot_churn_by_contract(df_plot)
    plot_monthly_charges_dist(df_plot)
    if "tenure" in df_plot.columns:
        plot_tenure_churn(df_plot)
    print("[EDA] All plots saved to outputs/\n")
