"""Aggregate research dashboard for the readmission benchmark.

This app intentionally does not accept patient inputs or expose individual risk
scores. It presents reproducible, aggregate held-out evaluation artifacts only.
"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"


def artifact(name):
    return OUTPUT / name


def load_csv(name):
    return pd.read_csv(artifact(name))


st.set_page_config(page_title="Readmission Benchmark | Research Dashboard", page_icon="🧪", layout="wide")
st.title("Patient Readmission Risk Prediction")
st.caption("Research dashboard · held-out aggregate evaluation · not a clinical decision-support tool")
st.warning(
    "This historical benchmark is for education only. It does not provide patient-level predictions and must not be used "
    "for treatment, triage, discharge, or any decision affecting an individual.",
    icon="⚠️",
)

required = ["test_metrics.csv", "validation_comparison.csv", "run_manifest.json", "permutation_importance.csv"]
missing = [name for name in required if not artifact(name).exists()]
if missing:
    st.info("Generate the reproducible artifacts first: `python run.py --bootstrap 100`.")
    st.stop()

metrics = load_csv("test_metrics.csv")
manifest = json.loads(artifact("run_manifest.json").read_text(encoding="utf-8"))
selected = manifest["selected_model"]
best = metrics.loc[metrics.model.eq(selected)].iloc[0]
prevalence = manifest["splits"]["test"]["positive_rate"]

st.subheader("Held-out performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model", selected.replace("_", " ").title(), "validation AP selection")
c2.metric("Average precision", f"{best.average_precision:.3f}", f"baseline prevalence {prevalence:.3f}")
c3.metric("ROC-AUC", f"{best.roc_auc:.3f}", "held-out test set")
c4.metric("Calibrated Brier score", f"{best.brier_score:.3f}", "lower is better")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Model comparison")
    display = metrics[["model", "average_precision", "roc_auc", "brier_score", "precision", "recall"]].copy()
    st.dataframe(display.style.format({column: "{:.3f}" for column in display.columns[1:]}), hide_index=True, use_container_width=True)
with right:
    st.subheader("Evaluation visual")
    image = artifact("evaluation.png")
    if image.exists():
        st.image(str(image), use_container_width=True)
    else:
        st.info("Run the pipeline to create the evaluation image.")

left, right = st.columns(2)
with left:
    st.subheader("Validation permutation importance")
    importance = load_csv("permutation_importance.csv").head(10).sort_values("mean_ap_decrease")
    st.bar_chart(importance.set_index("feature")["mean_ap_decrease"], horizontal=True)
    st.caption("Decrease in validation average precision after permuting one feature. This is association, not causation.")
with right:
    st.subheader("Exploratory subgroup audit")
    subgroup = load_csv("subgroup_audit.csv")
    summary = subgroup.pivot(index="group", columns="attribute", values="roc_auc")
    st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)
    st.caption("Groups with fewer than 100 encounters or a single label are omitted. These descriptive estimates do not establish fairness.")

with st.expander("Protocol, limits, and next steps"):
    st.markdown(
        """
        - Patient IDs are disjoint across train, calibration, validation, and test partitions.
        - Imputation, scaling, and encoding are fit only on training data.
        - The threshold maximizes validation F2 for an academic recall demonstration; it is not a clinical operating policy.
        - The dataset lacks reliable dates and hospital identifiers, so it has neither temporal nor external validation.
        - A real deployment would need prospective validation, site-specific calibration, clinical governance, privacy controls, drift monitoring, and an approved action pathway.
        """
    )
