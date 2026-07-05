"""Streamlit web UI for Tokopedia review predictions."""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Tokopedia Review Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Tokopedia Review Predictor")
st.caption("Deep Learning classification for product review sentiment and category.")


def _render_probability_bars(probabilities: dict) -> None:
    """Render probability bars from highest to lowest."""
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for label, prob in sorted_probs:
        st.progress(float(prob), text=f"{label}: {prob:.1%}")


tabs = st.tabs(["Predict", "Batch", "Analytics"])

# ---------------------------------------------------------------------------
# Tab 1: Single Prediction
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Single Review Analysis")
    review_text = st.text_area(
        "Review text",
        height=150,
        placeholder="Enter an Indonesian product review...",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button(
        "Analyze Review",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:
        if not review_text.strip():
            st.error("Review text cannot be empty.")
        else:
            with st.spinner("Analyzing review..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"review_text": review_text},
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to API. Make sure the FastAPI server is running."
                    )
                    st.stop()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.stop()

            st.success("Analysis complete")

            left, right = st.columns(2)
            with left:
                with st.container(border=True):
                    st.markdown("### Sentiment")
                    sentiment = result["sentiment"]
                    st.metric(
                        label="Prediction",
                        value=sentiment["label"].upper(),
                        delta=f"{sentiment['confidence']:.1%} confidence",
                    )
                    st.markdown("**Probability Distribution**")
                    _render_probability_bars(sentiment["probabilities"])

            with right:
                with st.container(border=True):
                    st.markdown("### Category")
                    category = result["category"]
                    st.metric(
                        label="Prediction",
                        value=category["label"],
                        delta=f"{category['confidence']:.1%} confidence",
                    )
                    st.markdown("**Probability Distribution**")
                    _render_probability_bars(category["probabilities"])

            with st.expander("Cleaned text"):
                st.text(result["cleaned_text"])

# ---------------------------------------------------------------------------
# Tab 2: Batch Prediction
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel with a 'review_text' column",
        type=["csv", "xlsx"],
    )

    if uploaded_file is None:
        st.info("Upload a file to start batch prediction.")
    else:
        try:
            if uploaded_file.name.endswith(".csv"):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")
            st.stop()

        if "review_text" not in input_df.columns:
            st.error("The uploaded file must contain a 'review_text' column.")
        else:
            st.markdown("**Preview**")
            st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                texts = input_df["review_text"].dropna().astype(str).tolist()
                if not texts:
                    st.warning("No valid review texts found in the file.")
                else:
                    with st.spinner("Processing batch..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/predict/batch",
                                json={"texts": texts},
                                timeout=120,
                            )
                            response.raise_for_status()
                            results = response.json()["results"]
                        except requests.exceptions.ConnectionError:
                            st.error(
                                "Cannot connect to API. Make sure the FastAPI server is running."
                            )
                            st.stop()
                        except Exception as exc:
                            st.error(f"Batch prediction failed: {exc}")
                            st.stop()

                    result_df = pd.DataFrame(results)
                    output_df = pd.concat(
                        [input_df.reset_index(drop=True), result_df],
                        axis=1,
                    )

                    st.markdown("**Results**")
                    st.dataframe(output_df, use_container_width=True)

                    csv_bytes = output_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

# ---------------------------------------------------------------------------
# Tab 3: Analytics
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Model Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("status") == "ok":
            status_cols = st.columns(len(health["models"]))
            for idx, model in enumerate(health["models"]):
                with status_cols[idx]:
                    st.metric(
                        label=model["task"].upper(),
                        value="Ready" if model["loaded"] else "Not Loaded",
                        delta=f"{model['model_type']} | max_len={model['max_len']}",
                    )
        else:
            st.error("API is not responding correctly.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
    except Exception as exc:
        st.error(f"Failed to load model status: {exc}")

    st.divider()

    st.subheader("Model Metrics")
    metrics_cols = st.columns(2)
    task_folders = [
        ("sentiment", "sentiment_model"),
        ("category", "category_model"),
    ]
    for idx, (task, folder) in enumerate(task_folders):
        metrics_path = (
            Path(__file__).resolve().parents[2] / "models" / folder / "metrics.json"
        )
        with metrics_cols[idx]:
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                st.markdown(f"**{task.upper()}**")
                acc_col, f1_col = st.columns(2)
                acc_col.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                f1_col.metric("F1-Macro", f"{metrics['f1_macro']:.2%}")
            else:
                st.info(f"{task.capitalize()} metrics are not available yet.")

    st.divider()

    st.subheader("Stored Predictions")
    try:
        predictions = requests.get(
            f"{API_URL}/predictions",
            params={"page_size": 100},
            timeout=10,
        ).json()
        if predictions["total"] == 0:
            st.info("No predictions stored yet.")
        else:
            df_preds = pd.DataFrame(predictions["items"])

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.markdown("**Sentiment Distribution**")
                sentiment_counts = (
                    df_preds["sentiment_label"].value_counts().reset_index()
                )
                sentiment_counts.columns = ["label", "count"]
                fig = px.bar(
                    sentiment_counts,
                    x="label",
                    y="count",
                    color="label",
                    text="count",
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            with chart_right:
                st.markdown("**Category Distribution**")
                category_counts = (
                    df_preds["category_label"].value_counts().reset_index()
                )
                category_counts.columns = ["label", "count"]
                fig = px.bar(
                    category_counts,
                    x="label",
                    y="count",
                    color="label",
                    text="count",
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Recent Predictions**")
            st.dataframe(df_preds, use_container_width=True)
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
    except Exception as exc:
        st.error(f"Failed to load predictions: {exc}")
