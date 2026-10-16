
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

st.set_page_config(
    page_title="Smart Energy Detection",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Energy Waste & Anomaly Detection")
st.caption("Monitor electricity consumption and identify unusual usage patterns.")

load_dotenv()
uri = os.getenv("MONGODB_URI")
db = None

if uri:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client["smart_energy_db"]
    except Exception as e:
        st.warning(f"MongoDB is unavailable. Predictions will not be saved. Details: {e}")

@st.cache_data
def load_data():
    data = pd.read_csv("energy_with_anomalies.csv")
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    return data

@st.cache_resource
def load_models():
    detector = joblib.load("anomaly_model.pkl")
    regressor = joblib.load("energy_model.pkl")
    anomaly_features = joblib.load("anomaly_features.pkl")
    energy_features = joblib.load("energy_features.pkl")
    return detector, regressor, anomaly_features, energy_features

try:
    df = load_data()
    detector, regressor, anomaly_features, energy_features = load_models()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

required = (
    ["Global_active_power", "is_anomaly", "Datetime"]
    + anomaly_features
    + energy_features
)

missing = [col for col in set(required) if col not in df.columns]

if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

total = len(df)
anomalies = int(df["is_anomaly"].sum())
normal = total - anomalies
average_power = df["Global_active_power"].mean()

st.subheader("Energy Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Readings", f"{total:,}")
c2.metric("Normal Readings", f"{normal:,}")
c3.metric("Potential Anomalies", f"{anomalies:,}")
c4.metric("Average Power", f"{average_power:.2f} kW")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Energy Consumption Over Time")
    chart_data = (
        df[["Datetime", "Global_active_power"]]
        .dropna()
        .sort_values("Datetime")
        .head(2000)
        .set_index("Datetime")
    )
    st.line_chart(chart_data, y="Global_active_power")

with right:
    st.subheader("Normal vs Potential Anomalies")
    counts = pd.Series({
        "Normal": normal,
        "Potential Anomaly": anomalies
    })
    st.bar_chart(counts)

st.divider()
st.subheader("Anomaly Analysis")

anomaly_data = df[df["is_anomaly"] == 1]
normal_data = df[df["is_anomaly"] == 0]

fig_data = pd.DataFrame({
    "Normal": [normal_data["Global_active_power"].mean()],
    "Potential Anomaly": [
        anomaly_data["Global_active_power"].mean()
        if len(anomaly_data) else 0
    ]
})

st.write("Average active power by group (kW)")
st.bar_chart(fig_data.T)

st.divider()
st.subheader("🔎 Test a New Energy Reading")

with st.form("energy_form"):
    values = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(anomaly_features):
        median_value = pd.to_numeric(
            df[feature], errors="coerce"
        ).median()

        if pd.isna(median_value):
            median_value = 0.0

        with (col1 if i % 2 == 0 else col2):
            values[feature] = st.number_input(
                feature.replace("_", " "),
                min_value=0.0,
                value=float(median_value),
                format="%.4f",
                key=f"input_{feature}"
            )

    submitted = st.form_submit_button(
        "Detect Anomaly",
        use_container_width=True
    )

if submitted:
    reading = pd.DataFrame([values])

    try:
        result = detector.predict(reading[anomaly_features])[0]
        prediction = float(
            regressor.predict(reading[energy_features])[0]
        )
        is_anomaly = int(result == -1)

        if is_anomaly:
            st.error("⚠️ Potential energy anomaly detected!")
        else:
            st.success("✅ Energy consumption appears normal.")

        st.metric("Estimated Active Power", f"{prediction:.3f} kW")

        record = {
            "input_reading": {
                key: float(value)
                for key, value in values.items()
            },
            "is_anomaly": is_anomaly,
            "predicted_active_power_kw": prediction,
            "created_at": datetime.now(timezone.utc)
        }

        if db is not None:
            saved = db.predictions.insert_one(record)
            st.success("Prediction saved to MongoDB Atlas.")

            st.subheader("Saved Prediction Document")
            st.json({
                "_id": str(saved.inserted_id),
                **record,
                "created_at": record["created_at"].isoformat()
            })
        else:
            st.info(
                "MongoDB is not connected. "
                "This prediction was not saved."
            )
            st.json({
                **record,
                "created_at": record["created_at"].isoformat()
            })

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.subheader("🗄️ MongoDB Prediction Records")

if db is not None:
    try:
        records = list(
            db.predictions.find()
            .sort("_id", -1)
            .limit(20)
        )

        if records:
            for record in records:
                record["_id"] = str(record["_id"])

                if isinstance(record.get("created_at"), datetime):
                    record["created_at"] = record["created_at"].isoformat()

                with st.expander(
                    f"Prediction {record['_id']}"
                ):
                    st.json(record)

            st.caption("Showing the latest 20 saved predictions.")
        else:
            st.info("No predictions have been saved yet.")

    except Exception as e:
        st.error(f"Unable to load MongoDB records: {e}")
else:
    st.info(
        "Connect MongoDB Atlas using MONGODB_URI "
        "to view saved prediction records here."
    )

st.divider()
st.subheader("Recent Readings")

display_columns = [
    "Datetime",
    "Global_active_power",
    "Global_intensity",
    "Voltage",
    "is_anomaly"
]

st.dataframe(
    df[display_columns]
    .tail(20)
    .sort_values("Datetime", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Anomalies are potential outliers, not confirmed energy waste. "
    "Predicted active power is an estimate based on other measurements."
)
