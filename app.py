from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customers_demo.csv"
FEATURES = ["monthly_charges", "tenure_months", "support_tickets", "contract_type"]

st.set_page_config(page_title="ChurnScope | Customer Churn Intelligence", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp {background: #0a1020; color: #eaf2ff;}
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stMetric"] {background:#121d33; border:1px solid #263653; border-radius:14px; padding:16px;}
h1,h2,h3 {color:#f1f6ff !important;}
div.stButton > button {background:#19bfd1; color:#07121f; font-weight:700; border-radius:10px; border:0;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not DATA.exists():
        st.error("Missing data/customers_demo.csv. Run `py src/main.py` once to generate it.")
        st.stop()
    df = pd.read_csv(DATA)
    missing = set(FEATURES + ["churn"]) - set(df.columns)
    if missing:
        st.error(f"Dataset is missing columns: {sorted(missing)}")
        st.stop()
    return df

@st.cache_resource
def fit_model():
    df = load_data()
    X, y = df[FEATURES], df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), FEATURES[:3]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["contract_type"])
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    return pipeline, metrics, y_test.to_numpy(), pred, proba

st.title("📊 ChurnScope")
st.caption("Customer Churn Prediction • Interactive Machine Learning Dashboard")
st.warning("DEMO ONLY — All customer records and outcomes are synthetic. Predictions are not validated for real business decisions.")

df = load_data()
model, metrics, y_test, y_pred, y_score = fit_model()

page = st.sidebar.radio("Explore", ["Predict churn", "Model performance", "Explore dataset"])
st.sidebar.caption("Logistic regression · 75/25 stratified split · random_state=42")

if page == "Predict churn":
    st.subheader("Customer risk estimator")
    st.write("Enter a customer's details and select **Predict churn probability**.")
    with st.form("prediction"):
        a,b = st.columns(2)
        with a:
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=75.0, step=5.0)
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=240, value=18, step=1)
        with b:
            tickets = st.number_input("Support tickets", min_value=0, max_value=100, value=2, step=1)
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        submitted = st.form_submit_button("Predict churn probability", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{"monthly_charges": monthly, "tenure_months": tenure,
                             "support_tickets": tickets, "contract_type": contract}])
        probability = float(model.predict_proba(row)[0,1])
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric("Estimated churn probability", f"{probability:.1%}")
            st.progress(probability)
        with c2:
            label = "Higher estimated risk" if probability >= .5 else "Lower estimated risk"
            st.info(f"**{label}** using a 50% display threshold. This is a demonstration prediction, not a calibrated production risk score.")
            st.dataframe(row, hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("About this model")
    st.write("A logistic regression classifier learns relationships between monthly charges, tenure, support tickets, and contract type. The model is trained on synthetic demonstration records.")

elif page == "Model performance":
    st.subheader("Held-out test set metrics")
    cols = st.columns(5)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.1%}" if name != "ROC-AUC" else f"{value:.3f}")
    left,right = st.columns(2)
    with left:
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        fig, ax = plt.subplots(figsize=(5.4,4.2))
        im = ax.imshow(cm, cmap="Blues")
        for (i,j), val in np.ndenumerate(cm):
            ax.text(j,i,str(val),ha="center",va="center",color="white" if val>cm.max()/2 else "black",fontsize=13)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Stay","Churn"],yticklabels=["Stay","Churn"],xlabel="Predicted",ylabel="Actual",title="Confusion matrix")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        fpr,tpr,_=roc_curve(y_test,y_score)
        fig,ax=plt.subplots(figsize=(5.4,4.2))
        ax.plot(fpr,tpr,color="#0089b8",lw=2,label=f"AUC = {metrics['ROC-AUC']:.3f}")
        ax.plot([0,1],[0,1],"--",color="gray")
        ax.set(xlabel="False positive rate",ylabel="True positive rate",title="ROC curve")
        ax.legend();fig.tight_layout();st.pyplot(fig);plt.close(fig)
    st.caption("Evaluation uses a stratified 25% test split of the synthetic dataset. No real-world generalization is claimed.")

else:
    st.subheader("Explore synthetic customer records")
    c1,c2,c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['churn'].mean():.1%}")
    c3.metric("Average monthly charge", f"${df['monthly_charges'].mean():.2f}")
    st.write("#### Churn rate by contract")
    grouped = df.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    st.bar_chart(grouped)
    st.write("#### Monthly charges distribution")
    fig,ax=plt.subplots(figsize=(9,3))
    ax.hist(df["monthly_charges"],bins=25,color="#0089b8",edgecolor="white")
    ax.set(xlabel="Monthly charges ($)",ylabel="Customers")
    fig.tight_layout();st.pyplot(fig);plt.close(fig)
    st.write("#### Preview")
    st.dataframe(df.head(25),use_container_width=True,hide_index=True)
    st.download_button("Download demo dataset (CSV)",data=df.to_csv(index=False),file_name="customers_demo.csv",mime="text/csv")
