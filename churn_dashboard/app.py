
import io
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

st.set_page_config(page_title="ABC Corp — Churn Risk Dashboard", layout="wide")

st.title("ABC Corporation — Customer Churn Risk Dashboard")
st.write("Upload customer data, score attrition probabilities (0–1), tune a decision threshold, and view drivers.")

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_resource
def build_prep(cat_cols, num_cols):
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler())])
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', num_pipe, num_cols),
                              ('cat', cat_pipe, cat_cols)])

def split_X_y(df: pd.DataFrame):
    y = None
    if 'Attrition_Flag' in df.columns:
        y = (df['Attrition_Flag'].astype(str).str.strip().str.lower() == 'attrited customer').astype(int).values
        X = df.drop(columns=['Attrition_Flag', 'CLIENTNUM'], errors='ignore')
    else:
        X = df.drop(columns=['CLIENTNUM'], errors='ignore')
    return X, y

def small_bakeoff(prep, X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_split=2,
                                               max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                                        subsample=0.8, max_features='sqrt', random_state=42)
    }
    results = {}
    best_name, best_auc, best_pipe = None, -np.inf, None
    for name, clf in models.items():
        pipe = Pipeline([('prep', prep), ('clf', clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_valid)[:, 1] if hasattr(pipe.named_steps['clf'], "predict_proba") \
                else pipe.decision_function(X_valid)
        if proba.ndim == 1 and not hasattr(pipe.named_steps['clf'], "predict_proba"):
            smin, smax = proba.min(), proba.max()
            proba = (proba - smin) / (smax - smin + 1e-9)
        auc = roc_auc_score(y_valid, proba) if y_valid is not None else np.nan
        results[name] = auc
        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe
    return best_name, best_pipe, results

def evaluate(y_true, proba, threshold):
    yhat = (proba >= threshold).astype(int)
    metrics = {
        "ROC-AUC": roc_auc_score(y_true, proba) if y_true is not None else np.nan,
        "PR-AUC": average_precision_score(y_true, proba) if y_true is not None else np.nan,
        "Precision": precision_score(y_true, yhat, zero_division=0) if y_true is not None else np.nan,
        "Recall": recall_score(y_true, yhat, zero_division=0) if y_true is not None else np.nan,
        "F1": f1_score(y_true, yhat, zero_division=0) if y_true is not None else np.nan,
    }
    return metrics, yhat

def plot_roc(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0,1],[0,1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    return fig

def plot_pr(y_true, proba):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    return fig

def topk_capture(y_true, proba, frac=0.10):
    k = max(1, int(np.ceil(frac * len(proba))))
    idx = np.argsort(-proba)[:k]
    return (y_true[idx].sum() / max(1, y_true.sum())) if y_true is not None else np.nan

def feature_importance_from_pipe(pipe, X: pd.DataFrame):
    prep = pipe.named_steps['prep']
    clf = pipe.named_steps['clf']
    try:
        ohe = prep.named_transformers_['cat'].named_steps['ohe']
        num_list = prep.transformers_[0][2]
        cat_list = list(ohe.get_feature_names_out(prep.transformers_[1][2]))
        feat_names = list(num_list) + cat_list
    except Exception:
        feat_names = [f"f{i}" for i in range(pipe.named_steps['clf'].n_features_in_)]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        df = pd.DataFrame({"Feature": feat_names, "Importance": vals}).sort_values("Importance", ascending=False)
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()
        df = pd.DataFrame({"Feature": feat_names, "Importance": np.abs(coefs), "Signed_Coefficient": coefs}).sort_values("Importance", ascending=False)
    else:
        df = pd.DataFrame({"Feature": feat_names})
    return df

# Sidebar: data input
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload customer_data.csv", type=["csv"])
sample_path = st.sidebar.text_input("...or path to CSV (server-side)", value="")

df = None
if uploaded is not None:
    df = load_csv(uploaded)
elif sample_path and os.path.exists(sample_path):
    df = load_csv(sample_path)

if df is None:
    st.info("Upload a CSV to begin. Expect columns like Attrition_Flag, CLIENTNUM, transaction/balance features.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head())

# Split X/y
X, y = split_X_y(df)

# Identify column types
cat_cols = [c for c in X.columns if X[c].dtype == 'object']
num_cols = [c for c in X.columns if X[c].dtype != 'object']
prep = build_prep(cat_cols, num_cols)

# Optional: load existing model if present
model_path = os.path.join("models", "champion.pkl")
use_saved = False
pipe = None
if os.path.exists(model_path):
    try:
        pipe = joblib.load(model_path)
        use_saved = True
        st.success("Loaded saved champion model from /models/champion.pkl")
    except Exception as e:
        st.warning(f"Could not load saved model: {e}")

# If no saved model, quick split + bakeoff to pick a champion
global X_valid, y_valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y if y is not None else None, random_state=42)

if not use_saved:
    with st.spinner("Training quick models and selecting a champion..."):
        best_name, pipe, aucs = small_bakeoff(prep, X_train, y_train)
        st.write("Validation ROC-AUC (higher is better):", aucs)
        st.success(f"Selected champion: {best_name}")
        # Save champion
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, model_path)

# Predict probabilities on the full dataset
with st.spinner("Scoring probabilities..."):
    if hasattr(pipe.named_steps['clf'], "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    else:
        s = pipe.decision_function(X)
        smin, smax = s.min(), s.max()
        proba = (s - smin) / (smax - smin + 1e-9)

st.sidebar.header("Decision Policy")
threshold = st.sidebar.slider("Decision Threshold (τ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
top_frac = st.sidebar.slider("Top-K Fraction for Lift (e.g., 0.10 = top 10%)", min_value=0.01, max_value=0.5, value=0.10, step=0.01)

# Evaluate (requires y)
metrics = {}
yhat = None
if y is not None:
    metrics, yhat = evaluate(y, proba, threshold)
    cmat = confusion_matrix(y, yhat)
    st.subheader("Metrics")
    st.write(pd.DataFrame([metrics]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}" if not np.isnan(metrics['ROC-AUC']) else "—")
    with col2:
        st.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}" if not np.isnan(metrics['PR-AUC']) else "—")
    with col3:
        capture = topk_capture(y, proba, top_frac)
        st.metric(f"Top-{int(top_frac*100)}% Capture", f"{capture*100:.1f}%" if not np.isnan(capture) else "—")

    colA, colB = st.columns(2)
    with colA:
        st.pyplot(plot_roc(y, proba))
    with colB:
        st.pyplot(plot_pr(y, proba))

    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cmat, index=["Actual:0","Actual:1"], columns=["Pred:0","Pred:1"]))

# Feature importance
st.subheader("Feature Importance (Champion)")
fi_df = feature_importance_from_pipe(pipe, X)
st.dataframe(fi_df.head(20))

# Download scored file
st.subheader("Download Scored Data")
scored = df.copy()
scored["churn_probability"] = proba
if yhat is not None:
    scored["churn_prediction_at_tau"] = yhat
csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV with Scores", data=csv_bytes, file_name="scored_customers.csv", mime="text/csv")

st.caption("Note: When the target is not provided, the app skips evaluation metrics and only produces probabilities.")
