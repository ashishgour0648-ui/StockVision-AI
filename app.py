import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os
import zipfile
import glob
import warnings

from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# ==========================================
# STOCKVISION AI - MULTI COMPANY SYSTEM
# ==========================================

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockVision AI")
st.subheader("Multi-Company Stock Price Prediction System")
st.markdown(
    "Machine Learning based stock analysis using Random Forest and LSTM."
)

PACKAGE_NAME = "StockVision_AI_MultiCompany_Package.zip"
ARTIFACT_DIR = "/tmp/stockvision_artifacts"

# ==========================================
# LOAD PACKAGE
# ==========================================

@st.cache_resource
def load_artifacts():

    zip_files = glob.glob(f"/mount/src/**/{PACKAGE_NAME}", recursive=True)

    if not zip_files:
        zip_files = glob.glob(f"/**/{PACKAGE_NAME}", recursive=True)

    if not zip_files:
        raise FileNotFoundError(
            f"{PACKAGE_NAME} repository mein nahi mili. "
            "Check karo ki ZIP GitHub par upload hui hai."
        )

    zip_path = zip_files[0]
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(ARTIFACT_DIR)

    rf_model = joblib.load(
        os.path.join(ARTIFACT_DIR, "multi_company_rf_model.pkl")
    )

    with open(
        os.path.join(ARTIFACT_DIR, "multi_company_rf_features.json"),
        "r"
    ) as f:
        feature_info = json.load(f)

    if isinstance(feature_info, list):
        rf_features = feature_info
    else:
        rf_features = (
            feature_info.get("features")
            or feature_info.get("feature_names")
            or feature_info.get("columns")
        )

    if not rf_features:
        raise ValueError("RF feature names JSON mein nahi mile.")

    lstm_model = load_model(
        os.path.join(ARTIFACT_DIR, "multi_company_lstm_model.keras"),
        compile=False
    )

    x_scaler = joblib.load(
        os.path.join(ARTIFACT_DIR, "multi_company_lstm_X_scaler.pkl")
    )

    y_scaler = joblib.load(
        os.path.join(ARTIFACT_DIR, "multi_company_lstm_y_scaler.pkl")
    )

    data = pd.read_csv(
        os.path.join(
            ARTIFACT_DIR,
            "StockVision_All_Companies_Cleaned.csv"
        )
    )

    comparison = pd.read_csv(
        os.path.join(
            ARTIFACT_DIR,
            "StockVision_Model_Comparison.csv"
        )
    )

    data["Date"] = pd.to_datetime(data["Date"])
    data["Symbol"] = data["Symbol"].astype(str)

    return (
        rf_model,
        rf_features,
        lstm_model,
        x_scaler,
        y_scaler,
        data,
        comparison
    )


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(company_data):

    df = company_data.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))

    for lag in [1, 2, 3, 5, 10]:
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    for window in [5, 10, 20, 50]:
        moving_average = df["Close"].rolling(window).mean()
        ratio = df["Close"] / moving_average
        df[f"MA_Ratio_{window}"] = ratio
        df[f"Close_MA_Ratio_{window}"] = ratio

    for window in [5, 10, 20]:
        volatility = df["Return"].rolling(window).std()
        df[f"Volatility_{window}"] = volatility

    df["Volume_Change"] = df["Volume"].pct_change()
    volume_average = df["Volume"].rolling(20).mean()
    df["Volume_MA_Ratio"] = df["Volume"] / volume_average

    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Ratio"] = df["Open"] / df["Close"]

    return df.replace([np.inf, -np.inf], np.nan)


# ==========================================
# LOAD DATA AND MODELS
# ==========================================

try:
    (
        rf_model,
        rf_features,
        lstm_model,
        x_scaler,
        y_scaler,
        all_data,
        comparison
    ) = load_artifacts()

except Exception as e:
    st.error(f"System loading error: {e}")
    st.stop()


# ==========================================
# COMPANY SELECTION
# ==========================================

companies = sorted(all_data["Symbol"].dropna().unique())

st.sidebar.header("⚙️ Stock Selection")

selected_company = st.sidebar.selectbox(
    "Select Company",
    companies
)

company_data = all_data[
    all_data["Symbol"] == selected_company
].copy()

company_data = company_data.sort_values("Date").reset_index(drop=True)

if company_data.empty:
    st.error("Selected company ka data nahi mila.")
    st.stop()


# ==========================================
# COMPANY INFORMATION
# ==========================================

latest_date = company_data["Date"].iloc[-1]
latest_close = float(company_data["Close"].iloc[-1])

st.markdown("---")
st.header(f"📊 {selected_company} Stock Analysis")

c1, c2, c3 = st.columns(3)

c1.metric("Company", selected_company)
c2.metric("Latest Historical Close", f"₹{latest_close:,.2f}")
c3.metric("Latest Data Date", latest_date.strftime("%d-%m-%Y"))

st.info(
    "Important: Is project ka historical dataset 01 October 2020 tak hai. "
    "Ye live market data nahi hai. Predictions educational and experimental "
    "estimates hain, investment advice nahi."
)


# ==========================================
# RANDOM FOREST PREDICTION
# ==========================================

feature_data = create_features(company_data)
valid_features = feature_data.dropna().copy()

if valid_features.empty:
    st.error("Prediction ke liye sufficient historical data nahi hai.")
    st.stop()

latest_row = valid_features.iloc[[-1]].copy()

# Create Symbol one-hot columns
symbol_data = pd.DataFrame({"Symbol": [selected_company]})
symbol_encoded = pd.get_dummies(symbol_data, columns=["Symbol"])

# Combine engineered values and company encoding
rf_input = pd.concat(
    [
        latest_row.drop(columns=["Date", "Symbol"], errors="ignore"),
        symbol_encoded
    ],
    axis=1
)

# Match the exact feature order used during training
rf_input = rf_input.reindex(
    columns=rf_features,
    fill_value=0
)

rf_return = float(rf_model.predict(rf_input)[0])
rf_prediction = latest_close * np.exp(rf_return)


# ==========================================
# LSTM PREDICTION
# ==========================================

lstm_prediction = None
lstm_error = None

try:
    returns = np.log(
        company_data["Close"] / company_data["Close"].shift(1)
    ).dropna().values

    if len(returns) < 30:
        raise ValueError("LSTM ke liye 30 days ka data required hai.")

    last_sequence = returns[-30:].reshape(-1, 1)

    scaled_sequence = x_scaler.transform(last_sequence)
    scaled_sequence = scaled_sequence.reshape(1, 30, 1)

    scaled_prediction = lstm_model.predict(
        scaled_sequence,
        verbose=0
    )

    predicted_return = y_scaler.inverse_transform(
        scaled_prediction.reshape(-1, 1)
    )[0][0]

    lstm_prediction = latest_close * np.exp(float(predicted_return))

except Exception as e:
    lstm_error = str(e)


# ==========================================
# PREDICTION RESULTS
# ==========================================

st.markdown("---")
st.header("🤖 Next Trading Day Prediction")

p1, p2, p3 = st.columns(3)

p1.metric(
    "Random Forest Prediction",
    f"₹{rf_prediction:,.2f}"
)

if lstm_prediction is not None:
    p2.metric(
        "LSTM Prediction",
        f"₹{lstm_prediction:,.2f}"
    )
else:
    p2.metric("LSTM Prediction", "Not Available")

p3.metric(
    "Baseline (No Change)",
    f"₹{latest_close:,.2f}"
)

if lstm_error:
    st.warning(f"LSTM prediction available nahi hui: {lstm_error}")

if lstm_prediction is not None:
    difference = abs(rf_prediction - lstm_prediction)
    st.write(f"**RF aur LSTM prediction difference:** ₹{difference:,.2f}")


# ==========================================
# HISTORICAL STOCK CHART
# ==========================================

st.markdown("---")
st.header("📈 Historical Stock Price Chart")

chart_data = company_data.tail(250)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(
    chart_data["Date"],
    chart_data["Close"],
    label="Historical Closing Price"
)
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (₹)")
ax.set_title(f"{selected_company} Historical Stock Price")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)


# ==========================================
# MODEL COMPARISON
# ==========================================

st.markdown("---")
st.header("🏆 Model Performance Comparison")

company_comparison = comparison[
    comparison["Company"].astype(str) == selected_company
]

if not company_comparison.empty:

    st.dataframe(
        company_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "MAE (Mean Absolute Error) jitna kam hota hai, "
        "test data par prediction error utna kam hota hai."
    )

else:
    st.info("Is company ka comparison record available nahi hai.")


# ==========================================
# HISTORICAL DATA TABLE
# ==========================================

st.markdown("---")
st.header("📋 Historical Data")

st.dataframe(
    company_data.tail(20).sort_values(
        "Date",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# DOWNLOAD REPORT
# ==========================================

st.markdown("---")
st.header("⬇️ Download Report")

report = pd.DataFrame({
    "Company": [selected_company],
    "Latest Historical Date": [latest_date.strftime("%Y-%m-%d")],
    "Latest Historical Close": [latest_close],
    "Random Forest Estimate": [rf_prediction],
    "LSTM Estimate": [
        lstm_prediction if lstm_prediction is not None else np.nan
    ],
    "Baseline Estimate": [latest_close],
    "Dataset Type": ["Historical data ending 2020-10-01"]
})

st.download_button(
    label="📥 Download Prediction Report (CSV)",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name=f"StockVision_{selected_company}_Report.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(
    "StockVision AI | BCA Major Project | "
    "Machine Learning: Random Forest + LSTM"
)
