# """
# Exploratory Data Analysis for UAC Care Load Forecasting
# Generates summary statistics and key plots (saved to outputs/)
# """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_data, prepare_features

OUTPUT = Path(__file__).parent.parent / "outputs"
OUTPUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def main():
    df = load_data()
    print("=" * 60)
    print("UAC Dataset – Exploratory Data Analysis")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    print("\n--- Descriptive Statistics (primary series) ---")
    print(df[["Children_in_HHS_Care", "Children_discharged_HHS"]].describe().round(1))

    # 1. Full history of Care Load
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["Children_in_HHS_Care"], color="#2b6cb0", lw=1.2)
    ax.set_title("Children in HHS Care – Full History")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT / "eda_care_load_history.png", dpi=120)
    plt.close()

    # 2. Recent 365 days dual axis
    recent = df.tail(365)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(recent["Date"], recent["Children_in_HHS_Care"], color="#2b6cb0", label="In Care")
    ax1.set_ylabel("Children in HHS Care", color="#2b6cb0")
    ax2 = ax1.twinx()
    ax2.plot(recent["Date"], recent["Children_discharged_HHS"], color="#38a169", alpha=0.7, label="Discharges")
    ax2.set_ylabel("Daily Discharges", color="#38a169")
    ax1.set_title("Last 365 Days – Care Load vs Daily Discharges")
    fig.tight_layout()
    fig.savefig(OUTPUT / "eda_recent_dual.png", dpi=120)
    plt.close()

    # 3. Day-of-week seasonality
    df["dow"] = df["Date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="dow", y="Children_in_HHS_Care", order=order, ax=ax)
    ax.set_title("Care Load by Day of Week")
    ax.set_xlabel("")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(OUTPUT / "eda_dow_boxplot.png", dpi=120)
    plt.close()

    # 4. Monthly average trend
    monthly = df.set_index("Date")["Children_in_HHS_Care"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax, color="#2b6cb0", marker="o", ms=3)
    ax.set_title("Monthly Average Children in HHS Care")
    ax.set_ylabel("Average Count")
    fig.tight_layout()
    fig.savefig(OUTPUT / "eda_monthly_avg.png", dpi=120)
    plt.close()

    # 5. Distribution of daily discharges
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Children_discharged_HHS"].dropna(), bins=40, kde=True, ax=ax, color="#38a169")
    ax.set_title("Distribution of Daily Discharges")
    ax.set_xlabel("Discharges")
    fig.tight_layout()
    fig.savefig(OUTPUT / "eda_discharge_dist.png", dpi=120)
    plt.close()

    # 6. Correlation of engineered features (sample)
    feat = prepare_features(df)
    corr_cols = [c for c in feat.columns if "lag" in c or "rollmean" in c or c == "Children_in_HHS_Care"]
    if len(corr_cols) > 1:
        corr = feat[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Feature Correlation (Care Load lags & rolling means)")
        fig.tight_layout()
        fig.savefig(OUTPUT / "eda_feature_corr.png", dpi=120)
        plt.close()

    print(f"\nEDA plots saved to {OUTPUT}")
    print("Files:", [p.name for p in OUTPUT.glob("eda_*.png")])


if __name__ == "__main__":
    main()
