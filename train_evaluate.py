# """Train all models, evaluate, save metrics & forecasts """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
import joblib
from data_prep import load_data, prepare_features, train_test_split_time
from models import run_all_models, MLForecaster

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def main(test_days: int = 60, target: str = "Children_in_HHS_Care"):
    print("=" * 60)
    print("UAC Care Load Forecasting – Training & Evaluation")
    print("=" * 60)

    # 1. Load & prepare
    df = load_data()
    feat = prepare_features(df, target=target)
    train, test = train_test_split_time(feat, test_days=test_days)

    print(f"\nTrain period : {train['Date'].min().date()} → {train['Date'].max().date()} ({len(train)} days)")
    print(f"Test period  : {test['Date'].min().date()} → {test['Date'].max().date()} ({len(test)} days)")
    print(f"Target       : {target}")

    # 2. Run models
    print("\nTraining models …")
    results = run_all_models(train, test, target=target)

    # 3. Metrics table
    metrics_rows = []
    for name, res in results.items():
        m = res["metrics"]
        metrics_rows.append({
            "Model": name,
            "MAE": round(m["MAE"], 1),
            "RMSE": round(m["RMSE"], 1),
            "MAPE_%": round(m["MAPE"], 2),
        })
    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
    print("\n--- Model Performance (sorted by MAE) ---")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # 4. Forecast vs Actual
    forecast_df = test[["Date", target]].copy()
    forecast_df = forecast_df.rename(columns={target: "Actual"})
    for name, res in results.items():
        forecast_df[name] = res["pred"]
    forecast_df.to_csv(OUTPUT_DIR / "forecast_vs_actual.csv", index=False)
    print(f"\nSaved forecast_vs_actual.csv ({len(forecast_df)} rows)")

    # 5. Retrain best model on full data for future forecasts
    best_name = metrics_df.iloc[0]["Model"]
    print(f"\nBest model by MAE: {best_name}")

    # Save a ready-to-use RF model (always useful for Streamlit)
    full_feat = prepare_features(df, target=target)
    rf_full = MLForecaster(model_type="rf", target=target).fit(full_feat)
    joblib.dump(rf_full, MODEL_DIR / "rf_care_load.joblib")
    print("Saved models/rf_care_load.joblib")

    # Also for discharge demand
    print("\n--- Training Discharge Demand model ---")
    feat_d = prepare_features(df, target="Children_discharged_HHS")
    # Adjust feature names for discharge target
    # (simple approach: reuse same lag structure but on discharge series)
    train_d, test_d = train_test_split_time(feat_d, test_days=test_days)
    results_d = run_all_models(train_d, test_d, target="Children_discharged_HHS")

    metrics_d = []
    for name, res in results_d.items():
        m = res["metrics"]
        metrics_d.append({
            "Model": name,
            "MAE": round(m["MAE"], 1),
            "RMSE": round(m["RMSE"], 1),
            "MAPE_%": round(m["MAPE"], 2),
        })
    metrics_d_df = pd.DataFrame(metrics_d).sort_values("MAE")
    print(metrics_d_df.to_string(index=False))
    metrics_d_df.to_csv(OUTPUT_DIR / "model_metrics_discharge.csv", index=False)

    # Save RF for discharge
    rf_d = MLForecaster(model_type="rf", target="Children_discharged_HHS").fit(feat_d)
    joblib.dump(rf_d, MODEL_DIR / "rf_discharge.joblib")
    print("Saved models/rf_discharge.joblib")

    # Summary JSON
    summary = {
        "best_care_load_model": best_name,
        "care_load_metrics": metrics_df.to_dict(orient="records"),
        "discharge_metrics": metrics_d_df.to_dict(orient="records"),
        "train_end": str(train["Date"].max().date()),
        "test_days": test_days,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Training complete. Outputs saved to /outputs")
    return metrics_df


if __name__ == "__main__":
    main(test_days=60)
