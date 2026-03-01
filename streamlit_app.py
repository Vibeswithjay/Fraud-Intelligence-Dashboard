import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")

# Local path on your Mac
OUTPUT_DIR = Path("/Users/user/Downloads/Fraud") / "fraud_project_outputs"

st.title("Fraud Intelligence Dashboard")
st.caption("Interactive dashboard for fraud detection outputs")

# ──────── LOCAL OR CLOUD MODE ────────
if OUTPUT_DIR.exists():
    # Local mode (your Mac)
    def read_csv_if_exists(path):
        return pd.read_csv(path) if path.exists() else None
    
    results_df = read_csv_if_exists(OUTPUT_DIR / "model_comparison_results.csv")
    alerts_df = read_csv_if_exists(OUTPUT_DIR / "all_scored_test_transactions.csv")
    feature_importance_df = read_csv_if_exists(OUTPUT_DIR / "lightgbm_feature_importance.csv")
    missing_df = read_csv_if_exists(OUTPUT_DIR / "missing_values_report.csv")
    eda_summary_df = read_csv_if_exists(OUTPUT_DIR / "eda_summary_table.csv")
    
else:
    # Cloud mode (Streamlit Cloud)
    st.sidebar.warning("🌐 Running on Streamlit Cloud – Upload your 5 CSV files below")
    with st.sidebar.expander("📤 Upload Output Files (required on cloud)", expanded=True):
        results_upload = st.file_uploader("model_comparison_results.csv", type="csv", key="res")
        alerts_upload = st.file_uploader("all_scored_test_transactions.csv", type="csv", key="alt")
        feature_upload = st.file_uploader("lightgbm_feature_importance.csv", type="csv", key="feat")
        missing_upload = st.file_uploader("missing_values_report.csv", type="csv", key="mis")
        eda_upload = st.file_uploader("eda_summary_table.csv", type="csv", key="eda")
        
        results_df = pd.read_csv(results_upload) if results_upload is not None else None
        alerts_df = pd.read_csv(alerts_upload) if alerts_upload is not None else None
        feature_importance_df = pd.read_csv(feature_upload) if feature_upload is not None else None
        missing_df = pd.read_csv(missing_upload) if missing_upload is not None else None
        eda_summary_df = pd.read_csv(eda_upload) if eda_upload is not None else None

# ──────── TABS (same as before) ────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Data Quality", "Model Comparison", "Fraud Alerts", "Feature Importance"]
)

with tab1:
    st.subheader("Overview")
    if eda_summary_df is not None:
        st.dataframe(eda_summary_df, use_container_width=True)
    else:
        st.info("eda_summary_table.csv not found / not uploaded")

with tab2:
    st.subheader("Data Quality")
    if missing_df is not None:
        st.dataframe(missing_df, use_container_width=True)
        if "Missing Count" in missing_df.columns and (missing_df["Missing Count"] > 0).sum() == 0:
            st.success("No missing values detected.")
    else:
        st.info("missing_values_report.csv not found / not uploaded")

with tab3:
    st.subheader("Model Comparison")
    if results_df is not None:
        st.dataframe(results_df, use_container_width=True)
        metric_options = [c for c in ["pr_auc", "recall", "precision", "f1", "roc_auc"] if c in results_df.columns]
        if metric_options:
            metric = st.selectbox("Choose metric", metric_options)
            fig = px.bar(results_df.sort_values(metric, ascending=False), x="model", y=metric, color=metric, template="plotly_white", title=f"Model Comparison by {metric.upper()}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("model_comparison_results.csv not found / not uploaded")

with tab4:
    st.subheader("Fraud Alerts")
    if alerts_df is not None and "fraud_score" in alerts_df.columns:
        threshold = st.slider("Fraud score threshold", 0.0, 1.0, 0.5, 0.01)
        filtered = alerts_df[alerts_df["fraud_score"] >= threshold].copy()
        col1, col2 = st.columns(2)
        col1.metric("Rows above threshold", f"{len(filtered):,}")
        if "isFraud" in filtered.columns:
            col2.metric("Actual frauds above threshold", f"{int(filtered['isFraud'].sum()):,}")
        fig = px.histogram(alerts_df, x="fraud_score", color="isFraud" if "isFraud" in alerts_df.columns else None, nbins=50, barmode="overlay", opacity=0.7, template="plotly_white", title="Fraud Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered.sort_values("fraud_score", ascending=False).head(500), use_container_width=True)
    else:
        st.info("all_scored_test_transactions.csv not found / not uploaded")

with tab5:
    st.subheader("Feature Importance")
    if feature_importance_df is not None:
        st.dataframe(feature_importance_df.head(30), use_container_width=True)
        top_n = st.slider("Top N features", 5, 30, 15, 1)
        top_features = feature_importance_df.head(top_n).sort_values("importance", ascending=True)
        fig = px.bar(top_features, x="importance", y="feature", orientation="h", color="importance", template="plotly_white", title=f"Top {top_n} Features")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("lightgbm_feature_importance.csv not found / not uploaded")
