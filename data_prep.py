# """
# Data preparation & feature engineering for UAC Care Load Forecasting
# """

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "uac_daily_cleaned.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["Date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["quarter"] = df["Date"].dt.quarter
    return df


def add_lag_features(df: pd.DataFrame, target: str = "Children_in_HHS_Care",
                     lags: list[int] = [1, 7, 14, 30]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "Children_in_HHS_Care",
                         windows: list[int] = [7, 14]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{target}_rollmean{w}"] = df[target].rolling(w, min_periods=1).mean()
        df[f"{target}_rollstd{w}"] = df[target].rolling(w, min_periods=1).std()
    return df


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Net pressure = transfers − discharges (when available)"""
    df = df.copy()
    if "Children_transferred_to_HHS" in df.columns and "Children_discharged_HHS" in df.columns:
        df["net_pressure"] = (
            df["Children_transferred_to_HHS"].fillna(0) - df["Children_discharged_HHS"]
        )
        df["net_pressure_roll7"] = df["net_pressure"].rolling(7, min_periods=1).mean()
    return df


def prepare_features(df: pd.DataFrame,
                     target: str = "Children_in_HHS_Care") -> pd.DataFrame:
    """Full feature pipeline"""
    df = add_calendar_features(df)
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    df = add_flow_features(df)
    # Drop rows that still have NaNs from lag creation
    df = df.dropna(subset=[c for c in df.columns if "lag" in c]).reset_index(drop=True)
    return df


def train_test_split_time(df: pd.DataFrame, test_days: int = 60):
    """Strict chronological split"""
    split_idx = len(df) - test_days
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


if __name__ == "__main__":
    df = load_data()
    print("Raw shape:", df.shape)
    feat = prepare_features(df)
    print("Feature shape:", feat.shape)
    print("Columns:", feat.columns.tolist())
    train, test = train_test_split_time(feat)
    print(f"Train: {len(train)} | Test: {len(test)}")
