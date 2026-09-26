# """ Forecasting models for UAC Care Load & Discharge Demand """
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Optional statistical models
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }


# ---------- Baseline models ----------
class NaivePersistence:
    """Forecast = last observed value"""
    def fit(self, y):
        self.last = y.iloc[-1]
        return self

    def predict(self, horizon: int):
        return np.full(horizon, self.last)


class MovingAverageForecast:
    def __init__(self, window: int = 7):
        self.window = window

    def fit(self, y):
        self.mean = y.iloc[-self.window:].mean()
        return self

    def predict(self, horizon: int):
        return np.full(horizon, self.mean)


# ---------- Statistical ----------
class ExpSmoothingModel:
    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, y):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required")
        self.model = ExponentialSmoothing(
            y, trend="add", seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated"
        ).fit(optimized=True)
        return self

    def predict(self, horizon: int):
        return self.model.forecast(horizon).values


class SARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, y):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required")
        self.model = SARIMAX(
            y, order=self.order, seasonal_order=self.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        return self

    def predict(self, horizon: int):
        return self.model.forecast(horizon).values


# ---------- Machine Learning ----------
def get_feature_cols(target: str):
    return [
        "dow", "month", "day", "is_weekend", "quarter",
        f"{target}_lag1", f"{target}_lag7",
        f"{target}_lag14", f"{target}_lag30",
        f"{target}_rollmean7", f"{target}_rollmean14",
        f"{target}_rollstd7", f"{target}_rollstd14",
    ]


class MLForecaster:
    """Recursive multi-step forecasting with RF or GBM"""
    def __init__(self, model_type: str = "rf", target: str = "Children_in_HHS_Care"):
        self.target = target
        self.model_type = model_type
        if model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=3,
                random_state=42, n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                random_state=42
            )
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        candidates = get_feature_cols(self.target)
        self.feature_cols = [c for c in candidates if c in train_df.columns]
        if not self.feature_cols:
            raise ValueError(f"No feature columns found for target {self.target}")
        X = train_df[self.feature_cols]
        y = train_df[self.target]
        self.model.fit(X, y)
        self.last_row = train_df.iloc[-1].copy()
        self.history = train_df[self.target].tolist()
        return self

    def _make_features(self, last_vals: dict, date: pd.Timestamp) -> pd.DataFrame:
        row = {
            "dow": date.dayofweek,
            "month": date.month,
            "day": date.day,
            "is_weekend": int(date.dayofweek >= 5),
            "quarter": date.quarter,
        }
        hist = self.history
        for lag in [1, 7, 14, 30]:
            col = f"{self.target}_lag{lag}"
            if col in self.feature_cols:
                row[col] = hist[-lag] if len(hist) >= lag else hist[-1]
        for w in [7, 14]:
            mcol = f"{self.target}_rollmean{w}"
            scol = f"{self.target}_rollstd{w}"
            if mcol in self.feature_cols:
                window = hist[-w:] if len(hist) >= w else hist
                row[mcol] = np.mean(window)
            if scol in self.feature_cols:
                window = hist[-w:] if len(hist) >= w else hist
                row[scol] = np.std(window) if len(window) > 1 else 0.0
        return pd.DataFrame([row])[self.feature_cols]

    def predict(self, horizon: int, start_date: pd.Timestamp):
        preds = []
        current_date = start_date
        for _ in range(horizon):
            current_date += pd.Timedelta(days=1)
            X = self._make_features({}, current_date)
            pred = self.model.predict(X)[0]
            pred = max(0, pred)
            preds.append(pred)
            self.history.append(pred)
        return np.array(preds)


def run_all_models(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   target: str = "Children_in_HHS_Care") -> dict:
    """Train & evaluate all models on the same horizon"""
    y_train = train_df[target]
    y_test = test_df[target].values
    horizon = len(y_test)
    start_date = train_df["Date"].iloc[-1]

    results = {}

    # 1. Naïve
    naive = NaivePersistence().fit(y_train)
    pred = naive.predict(horizon)
    results["Naive"] = {"pred": pred, "metrics": evaluate(y_test, pred)}

    # 2. Moving Average
    ma = MovingAverageForecast(window=7).fit(y_train)
    pred = ma.predict(horizon)
    results["MovingAvg_7"] = {"pred": pred, "metrics": evaluate(y_test, pred)}

    # 3. Exponential Smoothing
    if HAS_STATSMODELS:
        try:
            es = ExpSmoothingModel(seasonal_periods=7).fit(y_train)
            pred = es.predict(horizon)
            results["ExpSmoothing"] = {"pred": pred, "metrics": evaluate(y_test, pred)}
        except Exception as e:
            print("ExpSmoothing failed:", e)

        # 4. SARIMA (light order for speed)
        try:
            sarima = SARIMAModel(order=(1, 1, 1), seasonal_order=(0, 1, 1, 7)).fit(y_train)
            pred = sarima.predict(horizon)
            results["SARIMA"] = {"pred": pred, "metrics": evaluate(y_test, pred)}
        except Exception as e:
            print("SARIMA failed:", e)

    # 5. Random Forest
    rf = MLForecaster(model_type="rf", target=target).fit(train_df)
    pred = rf.predict(horizon, start_date)
    results["RandomForest"] = {"pred": pred, "metrics": evaluate(y_test, pred)}

    # 6. Gradient Boosting
    gbm = MLForecaster(model_type="gbm", target=target).fit(train_df)
    pred = gbm.predict(horizon, start_date)
    results["GradientBoosting"] = {"pred": pred, "metrics": evaluate(y_test, pred)}

    return results
