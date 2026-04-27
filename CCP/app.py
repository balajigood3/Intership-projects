# ========================================= #
# CUSTOMER CHURN PREDICTION - SINGLE FILE   #
# ========================================= #
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import streamlit as st

# ========================================= #
# CONFIG                                    #
# ========================================= #
DATA_PATH = "Telco_Customer_Churn.csv"
MODEL_PATH = "churn_model.pkl"

# ========================================= #
# DATA PREPROCESSING                        #
# ========================================= #
def preprocess(df):
    df = df.copy()
    # Drop ID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    
    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fix ChainedAssignmentError (Newer Pandas compatibility)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # Target encoding
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        
    # Feature Engineering
    df["tenure_group"] = df["tenure"] // 12
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    return df

# ========================================= #
# TRAIN MODEL                               #
# ========================================= #
def train_model():
    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    # Evaluation
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "auc": roc_auc_score(y_test, probs),
        "acc": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds)
    }
    
    return model, metrics

# ========================================= #
# LOAD MODEL                                #
# ========================================= #
def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model, None

# ========================================= #
# PREDICTION FUNCTION                       #
# ========================================= #
def predict(model, input_data):
    df = pd.DataFrame([input_data])
    # Feature Engineering
    df["tenure_group"] = df["tenure"] // 12
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df = pd.get_dummies(df)
    
    # Align columns with training data
    model_cols = model.feature_names_in_
    for col in model_cols:
        if col not in df:
            df[col] = 0
    df = df[model_cols]
    
    prob = model.predict_proba(df)[0][1] # Probability of Churn
    pred = model.predict(df)[0]
    return pred, prob, df

# ========================================= #
# SHAP EXPLANATION                          #
# ========================================= #
def explain_prediction(model, input_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    return shap_values

# ========================================= #
# STREAMLIT UI                              #
# ========================================= #
def main():
    st.set_page_config(page_title="Churn AI", layout="centered")
    st.title("📊 Customer Churn Prediction AI")
    st.write("Predict if a customer will leave the company")
    
    model, _ = load_model()

    # Inputs
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    daily_consumption = st.number_input("Daily Consumption (GB)", 0.0, 10.0, 1.0)
    if st.button("Predict Churn"):
        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "DailyConsumption": daily_consumption
        }
        
        # Run prediction
        pred, prob, processed_df = predict(model, input_data)
        
        st.subheader("Result:")
        if pred == 1:
            st.error(f"High Risk of Churn ({prob:.2%})")
            st.write("Suggestion: Offer discount / retention plan")
        else:
            st.success(f"Customer Likely to Stay ({1-prob:.2%})")
            st.write("Customer is stable")

        # --- Corrected SHAP Visualization ---
        st.markdown("---")
        st.subheader("Why this prediction?")
        with st.spinner("Calculating factor impacts..."):
            shap_values = explain_prediction(model, processed_df)
            
            # Handle Random Forest SHAP output (list of arrays)
            # We want index [1] for Churn and we .flatten() to make it 1D for Pandas
            if isinstance(shap_values, list):
                impact_values = shap_values[1].flatten()
            else:
                # Fallback for different SHAP versions
                impact_values = shap_values[:,:,1].flatten() if len(shap_values.shape) > 2 else shap_values.flatten()

            feature_names = processed_df.columns
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": impact_values
            }).sort_values(by="Impact", ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                st.write("### Increases Risk")
                st.dataframe(shap_df.head(5))
            with c2:
                st.write("### Reduces Risk")
                st.dataframe(shap_df.tail(5))

    # --- Evaluation Section ---
    st.markdown("---")
    st.write("### Model Evaluation")
    if st.checkbox("Show Model Training Performance"):
        # We re-run training to get the metrics dict
        _, metrics = train_model()
        colA, colB = st.columns(2)
        colA.metric("Accuracy", f"{metrics['acc']:.2%}")
        colB.metric("ROC-AUC", f"{metrics['auc']:.4f}")
        st.write("**Detailed Classification Report:**")
        st.code(metrics['report'])

if __name__ == "__main__":
    main()
