import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Zero-Shot Router", layout="centered")

st.title("Zero-Shot Dynamic Routing System")
st.caption("Intent classification with runtime label updates — no retraining required.")
st.divider()

## section1--predict

st.subheader("Classify Intent")
text_input = st.text_input("Enter an utterance", placeholder="e.g. what's the weather like today?")

if st.button("Predict", type="primary"):
    if text_input.strip():
        with st.spinner("Classifying...."):
            try:
                response = requests.post(f"{API_BASE}/predict", json={"text":text_input})
                response.raise_for_status()
                result = response.json()

                intent = result["intent"]
                is_oos = result["is_oos"]

                if is_oos:
                    st.warning(f"Out of Scope")
                else:
                    st.success(f"{intent}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please enter some text")

st.divider()

## section 2 update labels

st.subheader("Update Intent Labels")
st.caption("Replace the active label set with new set at runtime. No retraining needed.")

labels_input = st.text_area(
    "Enter labels (one per line)",
    placeholder="weather\ntransfer\ntimer\nflight_status\ntranslate",
    height=150,
)

if st.button("Update Labels"):
    raw = [l.strip() for l in labels_input.strip().splitlines() if l.strip()]
    if raw:
        with st.spinner("Updating labels..."):
            try:
                response = requests.put(f"{API_BASE}/update-labels", json={"labels": raw})
                response.raise_for_status()
                result = response.json()
                st.success(f"Labels updated: {result['previous_count']} → {result['new_count']} intents")
                st.code("\n".join(result["labels"]), language=None)
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter at least one label.")