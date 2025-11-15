import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path

try:
    import joblib
except Exception:
    joblib = None

MODEL_PATH = Path("rf_model.pkl")
SCALER_PATH = Path("scaler.pkl")

FEATURES = ['year','month','day','Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COL = "prediction"


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    if joblib:
        return joblib.load(path)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler(path: Path):
    if not path.exists():
        return None
    if joblib:
        return joblib.load(path)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_df(model, scaler, df: pd.DataFrame):



    X = df.copy()
    # If scaler provided, apply to numeric columns only
    if scaler is not None:
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X[numeric_cols] = scaler.transform(X[numeric_cols])
        except Exception as e:
            st.warning(f"Scaler apply failed: {e}")
    preds = model.predict(X)
    # If model outputs probabilities and you want them, try predict_proba
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    res = pd.DataFrame({TARGET_COL: preds})
    if proba is not None:
        # append probabilities for classes
        for i in range(proba.shape[1]):
            res[f"prob_class_{i}"] = proba[:, i]
    return pd.concat([df.reset_index(drop=True), res], axis=1)

# APP LAYOUT
st.set_page_config(page_title="Model prediction - Streamlit", layout="centered")
st.title("🔮 Predict with your trained model (Streamlit)")
st.markdown("Upload a CSV with the same feature columns used during training, or use the Manual Input tab to enter values.")

# Load model & scaler
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

if model is None:
    st.warning(f"Model not found at `{MODEL_PATH}`. Please place your model file there or update MODEL_PATH in the script.")
else:
    st.success("Model loaded successfully.")

if FEATURES:
    st.info(f"Manual input mode: using FEATURES list with {len(FEATURES)} inputs.")

# Tabs: Upload CSV, Manual Input
tab1, tab2 = st.tabs(["Upload CSV", "Manual Input"])

with tab1:
    uploaded = st.file_uploader("Upload CSV file with features", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data (first 5 rows):")
            st.dataframe(df.head())

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df["year"] = df["Date"].dt.year
                df["month"] = df["Date"].dt.month
                df["day"] = df["Date"].dt.day


            try:
                df = df[FEATURES]
            except KeyError as e:
                st.error(f"CSV missing required columns: {e}")
                st.stop()

            if st.button("Run predictions on uploaded CSV"):
                if model is None:
                    st.error("No model loaded.")
                else:
                    out_df = predict_df(model, scaler, df)
                    st.write("Predictions:")
                    st.dataframe(out_df.head())

                    # allow download
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

with tab2:
    st.write("Enter values for features below and click Predict.")
    if FEATURES:
        input_data = {}
        for f in FEATURES:
            # Try to render number input by default; strings fallback
            input_data[f] = st.text_input(f, value="")
        if st.button("Predict for single sample"):
            try:
                # convert to numeric where possible
                sample = {k: (float(v) if v != "" else np.nan) for k, v in input_data.items()}
                sample_df = pd.DataFrame([sample])
                out_df = predict_df(model, scaler, sample_df)
                st.write(out_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("No FEATURES list configured. Please upload a CSV in the 'Upload CSV' tab or edit the FEATURES list in the script to enable manual inputs.")



# Optional: small diagnostics if model supports feature_importances_
if model is not None and hasattr(model, "feature_importances_"):
    try:
        fi = model.feature_importances_
        st.subheader("Model feature importances (top 10)")
        # Try to get feature names from FEATURES, else numeric cols of last uploaded df
        names = FEATURES if FEATURES else None
        if names is None:
            # try to inspect the model's expected input
            names = None
        if names and len(names) == len(fi):
            imp_df = pd.DataFrame({"feature": names, "importance": fi}).sort_values("importance", ascending=False).head(10)
            st.bar_chart(imp_df.set_index("feature"))
        else:
            st.info("Feature importances are available, but feature names were not configured or lengths mismatch. Configure FEATURES in the script to display them.")
    except Exception:
        pass

st.markdown("---")
st.caption("If you need this app adapted to your project's exact file layout (different model/scaler names or a specific input form), tell me and I will update the script.")
