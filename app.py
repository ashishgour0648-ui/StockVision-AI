
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ==========================================
# STOCKVISION AI - STOCK PRICE PREDICTION SYSTEM
# ==========================================

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide"
)

# Custom design
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #00d4ff;
}
.stMetric {
    background-color: #1b2430;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():
    rf_model = joblib.load("random_forest_model[1].pkl")
    lstm_model = load_model("lstm_model[1].keras")
    scaler = joblib.load("lstm_scaler[1].pkl")

    with open("rf_features[1].json", "r") as file:
        features = json.load(file)

    return rf_model, lstm_model, scaler, features


st.title("📈 StockVision AI")
st.subheader("Stock Price Prediction System Using Machine Learning")

st.write(
    "Predict stock prices using Random Forest and LSTM machine learning models."
)

st.info(
    "This project uses models trained on historical RELIANCE data. "
    "Predictions are experimental estimates, not guaranteed market forecasts."
)

# ==========================================
# LOAD MODELS SAFELY
# ==========================================

try:
    rf_model, lstm_model, scaler, features = load_models()
    st.success("Both trained models loaded successfully!")

except Exception as error:
    st.error("Model loading failed. Please check the model files and requirements.")
    st.code(str(error))
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Settings")
st.sidebar.write("StockVision AI Dashboard")

st.sidebar.markdown("---")
st.sidebar.write("**Available Models**")
st.sidebar.write("🌲 Random Forest")
st.sidebar.write("🧠 LSTM Neural Network")

st.sidebar.markdown("---")
st.sidebar.caption("Developed as a BCA Major Project")

# ==========================================
# UPLOAD STOCK DATA
# ==========================================

st.header("📂 Upload Stock Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel stock data",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.warning("Please upload a stock dataset to start prediction.")
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

except Exception as error:
    st.error("Unable to read the uploaded file.")
    st.code(str(error))
    st.stop()

# ==========================================
# DATA CLEANING
# ==========================================

df.columns = df.columns.str.strip()

required_columns = [
    "Date", "Open", "High", "Low", "Close", "Volume"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)
df = df.sort_values("Date").drop_duplicates(subset="Date")
df = df.reset_index(drop=True)

if len(df) < 210:
    st.error("Please upload at least 210 valid daily stock records.")
    st.stop()

# ==========================================
# FEATURE ENGINEERING
# ==========================================

data = df.copy()

data["Return"] = data["Close"].pct_change()

for lag in [1, 2, 3, 5, 10]:
    data[f"Return_Lag_{lag}"] = data["Return"].shift(lag)

for lag in [1, 2, 3, 5]:
    data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)

for window in [5, 20, 50]:
    moving_average = data["Close"].rolling(window).mean()
    data[f"MA_{window}_Ratio"] = data["Close"] / moving_average

for window in [5, 10, 20]:
    data[f"Volatility_{window}"] = (
        data["Return"].rolling(window).std()
    )

data["Volume_Change"] = data["Volume"].pct_change()

data["Volume_MA_5"] = data["Volume"].rolling(5).mean()

data["Daily_Range"] = (
    data["High"] - data["Low"]
) / data["Close"]

data["Open_Close_Ratio"] = (
    data["Open"] / data["Close"]
)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna().reset_index(drop=True)

if data.empty:
    st.error("Not enough valid data after feature calculation.")
    st.stop()

# ==========================================
# STOCK INFORMATION
# ==========================================

latest = data.iloc[-1]
latest_close = float(latest["Close"])
latest_date = latest["Date"]

st.header("📊 Stock Information")

col1, col2, col3 = st.columns(3)

col1.metric("Latest Closing Price", f"₹ {latest_close:,.2f}")
col2.metric("Latest Available Date", str(latest_date.date()))
col3.metric("Total Records", f"{len(data):,}")

# ==========================================
# STOCK PRICE CHART
# ==========================================

st.header("📉 Historical Stock Price")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(data["Date"], data["Close"], label="Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price (₹)")
ax.set_title("Historical Closing Price")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# ==========================================
# RANDOM FOREST PREDICTION
# ==========================================

st.header("🌲 Random Forest Prediction")

try:
    rf_input = data[features].iloc[[-1]]

    predicted_return = float(
        rf_model.predict(rf_input)[0]
    )

    rf_prediction = latest_close * np.exp(predicted_return)

    st.metric(
        "Random Forest Estimated Next Close",
        f"₹ {rf_prediction:,.2f}"
    )

except Exception as error:
    st.error("Random Forest prediction failed.")
    st.code(str(error))
    st.stop()

# ==========================================
# LSTM PREDICTION
# ==========================================

st.header("🧠 LSTM Prediction")

try:
    last_60_prices = data["Close"].values[-60:].reshape(-1, 1)

    scaled_prices = scaler.transform(last_60_prices)

    lstm_input = scaled_prices.reshape(1, 60, 1)

    lstm_scaled_prediction = lstm_model.predict(
        lstm_input, verbose=0
    )

    lstm_prediction = float(
        scaler.inverse_transform(lstm_scaled_prediction)[0][0]
    )

    st.metric(
        "LSTM Estimated Next Close",
        f"₹ {lstm_prediction:,.2f}"
    )

except Exception as error:
    st.error("LSTM prediction failed.")
    st.code(str(error))
    st.stop()

# ==========================================
# MODEL COMPARISON
# ==========================================

st.header("⚖️ Model Comparison")

difference = abs(rf_prediction - lstm_prediction)

col1, col2, col3 = st.columns(3)

col1.metric("Random Forest", f"₹ {rf_prediction:,.2f}")
col2.metric("LSTM", f"₹ {lstm_prediction:,.2f}")
col3.metric("Prediction Difference", f"₹ {difference:,.2f}")

comparison = pd.DataFrame({
    "Model": ["Random Forest", "LSTM"],
    "Estimated Next Close (₹)": [
        rf_prediction,
        lstm_prediction
    ]
})

st.dataframe(comparison, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    comparison["Model"],
    comparison["Estimated Next Close (₹)"]
)

ax.set_ylabel("Estimated Price (₹)")
ax.set_title("Model Prediction Comparison")
ax.grid(axis="y", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# ==========================================
# DOWNLOAD RESULTS
# ==========================================

st.header("📥 Download Prediction Report")

report = pd.DataFrame({
    "Model": ["Random Forest", "LSTM"],
    "Latest Dataset Date": [str(latest_date.date())] * 2,
    "Latest Closing Price": [latest_close] * 2,
    "Estimated Next Close": [
        rf_prediction,
        lstm_prediction
    ]
})

csv_data = report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction Report (CSV)",
    data=csv_data,
    file_name="StockVision_AI_Predictions.csv",
    mime="text/csv"
)

# ==========================================
# DISCLAIMER
# ==========================================

st.markdown("---")

st.caption(
    "Disclaimer: StockVision AI is an educational machine learning project. "
    "Predictions are experimental and should not be used as financial advice."
)

st.markdown(
    "**StockVision AI | Machine Learning Project**"
)
