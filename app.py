import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from groq import Groq

# --- PAGE SETUP ---
st.set_page_config(page_title="AI-NIDS Student Project", layout="wide")

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**: This system uses **Random Forest** to detect Network attacks and **Groq AI** to explain packets.
""")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("1. Settings")
groq_api_key = st.sidebar.text_input("Groq API Key (starts with gsk_)", type="password")
st.sidebar.caption("[Get a free key here](https://console.groq.com/keys)")

st.sidebar.header("2. Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CIC-IDS CSV", type=["csv"])

# Default file (if you still want to keep it)
DATA_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

st.sidebar.header("3. Model Controls")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, 0.05)
n_estimators = st.sidebar.slider("Trees (n_estimators)", 10, 300, 100, 10)
max_depth = st.sidebar.slider("Max depth", 2, 50, 10, 1)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# Features used
FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow Packets/s'
]
TARGET = "Label"


@st.cache_data
def load_data_from_path(filepath: str):
    df = pd.read_csv(filepath, nrows=15000)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


@st.cache_data
def load_data_from_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def train_model(df: pd.DataFrame):
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        return None

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=int(random_state),
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model": clf,
        "accuracy": acc,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "labels": sorted(y.unique()),
        "cm": cm,
        "report": report
    }


# ---------------------------
# LOAD DATA
# ---------------------------
df = None
if uploaded_file is not None:
    df = load_data_from_upload(uploaded_file)
    st.sidebar.success(f"Uploaded dataset loaded: {len(df)} rows")
else:
    # fallback to local file
    try:
        df = load_data_from_path(DATA_FILE)
        st.sidebar.success(f"Local dataset loaded: {len(df)} rows")
    except Exception:
        st.sidebar.warning("No upload detected and local CSV not found.")

if df is None:
    st.error("Please upload a CSV from the sidebar (or add the default CSV into the app folder).")
    st.stop()

# Show quick preview
with st.expander("📄 Dataset preview"):
    st.dataframe(df.head(50), use_container_width=True)

# ---------------------------
# TRAIN BUTTON
# ---------------------------
st.sidebar.header("4. Train")
if st.sidebar.button("Train Model Now"):
    with st.spinner("Training model..."):
        results = train_model(df)
        if results is not None:
            st.session_state["results"] = results
            st.session_state["alerts"] = []  # reset alerts log
            st.sidebar.success(f"Training Complete! Accuracy: {results['accuracy']:.2%}")

# ---------------------------
# DASHBOARD
# ---------------------------
st.header("Threat Analysis Dashboard")

if "results" not in st.session_state:
    st.info("Click **Train Model Now** in the sidebar to begin.")
    st.stop()

results = st.session_state["results"]
clf = results["model"]

tab1, tab2, tab3 = st.tabs(["🧪 Simulation", "📊 Model Evaluation", "✍️ Manual Packet Test"])

# ---------------------------
# TAB 1: SIMULATION
# ---------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Simulation")
        st.info("Pick a random packet from the test data to simulate live traffic.")

        if st.button("🎲 Capture Random Packet"):
            random_idx = np.random.randint(0, len(results["X_test"]))
            packet_data = results["X_test"].iloc[random_idx]
            actual_label = results["y_test"].iloc[random_idx]

            st.session_state["current_packet"] = packet_data
            st.session_state["actual_label"] = actual_label

    if "current_packet" in st.session_state:
        packet = st.session_state["current_packet"]

        with col1:
            st.write("**Packet Header Info:**")
            st.dataframe(packet.to_frame(name="value"), use_container_width=True)

        with col2:
            st.subheader("AI Detection Result")
            prediction = clf.predict([packet])[0]

            if prediction == "BENIGN":
                st.success("✅ STATUS: **SAFE (BENIGN)**")
            else:
                st.error(f"🚨 STATUS: **ATTACK DETECTED ({prediction})**")

            st.caption(f"Ground Truth Label: {st.session_state['actual_label']}")

            # Alerts log
            if prediction != "BENIGN":
                st.session_state["alerts"].append({
                    "prediction": prediction,
                    "ground_truth": st.session_state["actual_label"],
                    "packet_index": int(results["X_test"].index.get_loc(packet.name)) if packet.name in results["X_test"].index else None
                })

            st.markdown("---")
            st.subheader("Ask AI Analyst (Groq)")

            if st.button("Generate Explanation"):
                if not groq_api_key:
                    st.warning("Please enter your Groq API Key in the sidebar first.")
                else:
                    try:
                        client = Groq(api_key=groq_api_key)

                        prompt = f"""
You are a cybersecurity analyst.
A network packet was detected as: {prediction}.

Packet Technical Details:
{packet.to_string()}

Explain in simple student-friendly language:
1) Why these values could indicate {prediction}.
2) If BENIGN, why it looks normal.
3) One quick tip to reduce this type of attack (if ATTACK).
Keep it short.
"""

                        with st.spinner("Groq is analyzing the packet..."):
                            completion = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.6,
                            )
                            st.info(completion.choices[0].message.content)

                    except Exception as e:
                        st.error(f"API Error: {e}")

    # Recent Alerts Panel
    st.markdown("---")
    st.subheader("📌 Recent Alerts (this session)")
    if st.session_state.get("alerts"):
        st.dataframe(pd.DataFrame(st.session_state["alerts"]), use_container_width=True)
    else:
        st.caption("No alerts yet.")

# ---------------------------
# TAB 2: MODEL EVALUATION
# ---------------------------
with tab2:
    st.subheader("Model Performance")
    st.write(f"**Accuracy:** {results['accuracy']:.2%}")

    st.write("**Confusion Matrix:**")
    cm_df = pd.DataFrame(results["cm"], index=results["labels"], columns=results["labels"])
    st.dataframe(cm_df, use_container_width=True)

    st.write("**Classification Report:**")
    st.code(results["report"])

    # Download predictions
    st.markdown("---")
    st.subheader("Download Test Predictions")
    out = results["X_test"].copy()
    out["y_true"] = results["y_test"].values
    out["y_pred"] = results["y_pred"]
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download predictions CSV", data=csv_bytes, file_name="nids_predictions.csv", mime="text/csv")

# ---------------------------
# TAB 3: MANUAL PACKET TEST
# ---------------------------
with tab3:
    st.subheader("Manual Packet Input")
    st.caption("Try your own values to see how the model reacts (good for demos).")

    default_vals = results["X_test"].iloc[0].to_dict()
    manual = {}

    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            manual[feat] = st.number_input(
                feat,
                value=float(default_vals.get(feat, 0.0)),
                format="%.6f"
            )

    if st.button("Run Manual Prediction"):
        manual_df = pd.DataFrame([manual])
        pred = clf.predict(manual_df)[0]
        if pred == "BENIGN":
            st.success("✅ Manual Packet Result: **BENIGN**")
        else:
            st.error(f"🚨 Manual Packet Result: **ATTACK ({pred})**")
