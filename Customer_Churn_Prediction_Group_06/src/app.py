import streamlit as st
import joblib
import json
import numpy as np
import os
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "churn_data.csv"))

MODEL_CANDIDATES = ["xgb_churn_model.pkl", "churn_adaboost_model.pkl", "SVM_model.pkl"]
LABEL_ENCODERS_FILE = "label_encoders.pkl"

# ---------------- UI STYLES ----------------
def add_styles():
    st.markdown(
        """
        <style>
        /* page */
        .app-header { display:flex; gap:16px; align-items:center; }
        .app-logo {
            width:64px;height:64px;border-radius:14px;
            background:linear-gradient(135deg,#0ea5e9,#60a5fa);
            display:flex;align-items:center;justify-content:center;color:white;font-weight:800;
            font-size:22px;box-shadow:0 8px 20px rgba(2,6,23,0.08);
        }
        .app-title { font-size:20px;margin:0;color:#0f172a;font-weight:700 }
        .app-sub { color:#475569;margin-top:4px;font-size:13px }

        /* model card */
        .model-card { padding:12px;border-radius:10px;background:#f8fafc;border:1px solid #e6eef8 }
        .meta { color:#334155;font-size:13px }

        /* result badge */
        .badge { display:inline-block;padding:8px 14px;border-radius:999px;color:white;font-weight:700 }
        .badge-green { background:linear-gradient(90deg,#10b981,#059669) }
        .badge-red { background:linear-gradient(90deg,#ef4444,#b91c1c) }

        /* compact table */
        .stDataFrame table { font-size:13px }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- core helpers ----------------
def load_models():
    models = {}
    for name in MODEL_CANDIDATES:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.error(f"Failed to load {name}: {e}")
        else:
            st.warning(f"Model file not found: {path}")
    # fallback: try to load any .pkl/.joblib in MODEL_DIR if none of the candidates found
    if not models and os.path.isdir(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith((".pkl", ".joblib")):
                p = os.path.join(MODEL_DIR, fname)
                try:
                    models[fname] = joblib.load(p)
                except Exception:
                    continue
    return models

def load_feature_names():
    path = os.path.join(MODEL_DIR, "feature_names.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def infer_features_from_dataset(dataset_path, exclude_cols=None, max_rows=500):
    exclude_cols = exclude_cols or ["customerID", "Churn"]
    if not os.path.exists(dataset_path):
        return None
    try:
        df = pd.read_csv(dataset_path, nrows=max_rows)
    except Exception:
        return None
    cols = [c for c in df.columns if c not in exclude_cols]
    info = []
    for c in cols:
        series = df[c].dropna()
        unique_vals = series.unique().tolist()
        # heuristic numeric detection
        if pd.api.types.is_numeric_dtype(series) or c.lower() in ("tenure", "monthlycharges", "totalcharges", "seniorcitizen"):
            ftype = "numeric"
            options = None
            # special-case seniorcitizen to be binary
            if c.lower() == "seniorcitizen":
                ftype = "categorical"
                options = [0, 1]
        elif len(unique_vals) <= 20:
            ftype = "categorical"
            options = [str(x) for x in unique_vals]
        else:
            ftype = "text"
            options = None
        info.append({"name": c, "type": ftype, "options": options})
    return info

def load_label_encoders():
    p = os.path.join(MODEL_DIR, LABEL_ENCODERS_FILE)
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except Exception as e:
            st.warning(f"Failed to load {LABEL_ENCODERS_FILE}: {e}")
    return None

def apply_label_encoders_to_df(df, label_encoders):
    if not label_encoders:
        return df
    df = df.copy()
    # accept either a dict {col: mapping_or_encoder} or an sklearn-like object (not common here)
    if not isinstance(label_encoders, dict):
        st.warning("label_encoders.pkl does not contain a mapping dict; skipping encoding step.")
        return df
    for col, enc in label_encoders.items():
        if col not in df.columns:
            continue
        val = df.at[0, col]
        if isinstance(enc, dict):
            mapped = enc.get(val)
            if mapped is None:
                st.warning(f"Value '{val}' for column '{col}' not found in label mapping — setting -1")
                df.at[0, col] = -1
            else:
                df.at[0, col] = mapped
        else:
            # attempt sklearn-style transform if provided
            try:
                transformed = enc.transform([val])
                df.at[0, col] = transformed[0]
            except Exception:
                if hasattr(enc, "classes_"):
                    try:
                        idx = list(enc.classes_).index(val)
                        df.at[0, col] = int(idx)
                    except ValueError:
                        st.warning(f"Value '{val}' for column '{col}' not in encoder classes — setting -1")
                        df.at[0, col] = -1
                else:
                    st.warning(f"Could not encode value '{val}' for column '{col}' — setting -1")
                    df.at[0, col] = -1
    return df

def predict_model(model, X_df):
    """
    Predict using the given model. If the model is a Pipeline it may accept a DataFrame
    directly; otherwise convert to numpy array.
    """
    # If model is a sklearn Pipeline or otherwise accepts DataFrame, try DataFrame first
    try:
        pred = model.predict(X_df)
    except Exception:
        try:
            X_arr = X_df.values
            pred = model.predict(X_arr)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
    out = {}
    try:
        out["prediction"] = int(pred[0]) if hasattr(pred[0], "__int__") else str(pred[0])
    except Exception:
        out["prediction"] = str(pred[0])
    try:
        # predict_proba may accept DataFrame or ndarray similarly
        try:
            proba = model.predict_proba(X_df)
        except Exception:
            proba = model.predict_proba(X_df.values)
        out["probabilities"] = proba.tolist()
    except Exception:
        out["probabilities"] = None
    return out

# ---------------- UI helpers ----------------
def render_header():
    st.markdown(
        """
        <div class="app-header">
          <div class="app-logo">CC</div>
          <div>
            <div class="app-title">Customer Churn — Predictor</div>
            <div class="app-sub">Fast, explainable predictions for production demos</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_model_card(model_name, model_obj):
    # enhanced sidebar card with file size and feature expectation
    p = os.path.join(MODEL_DIR, model_name)
    size = "unknown"
    try:
        size = f"{os.path.getsize(p)/1024:.1f} KB"
    except Exception:
        pass
    n_in = getattr(model_obj, "n_features_in_", None)
    classes = getattr(model_obj, "classes_", None)
    st.sidebar.markdown("<div class='model-card'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Model:** {model_name}")
    st.sidebar.markdown(f"<div class='meta'>File: {size} • expects: <b>{n_in if n_in is not None else 'unknown'}</b></div>", unsafe_allow_html=True)
    if classes is not None:
        st.sidebar.markdown(f"<div class='meta'>Classes: {', '.join(map(str, list(classes)[:8]))}</div>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

def render_dataset_preview(dataset_path):
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path, nrows=100)
        with st.expander("Dataset preview (first 100 rows)", expanded=False):
            st.dataframe(df.head(10))
            csv = df.to_csv(index=False)
            st.download_button("Download sample CSV", data=csv, file_name="churn_data_sample.csv", mime="text/csv")

def render_inputs_form(feature_info, label_encoders):
    """
    Render input widgets inside a form. Uses explicit session_state keys so the
    "Clear form" button (outside the form) can reset values.
    Returns a DataFrame (single row) when Predict is submitted, otherwise None.
    """
    form_key = "input_form"
    # prepare default values map for clearing
    defaults = {}
    for fi in feature_info:
        fname = fi["name"]
        ftype = fi.get("type", "numeric")
        opts = fi.get("options")
        if fname.lower() == "seniorcitizen":
            defaults[fname] = 0
        elif ftype == "numeric":
            defaults[fname] = 0.0
        elif ftype == "categorical" and opts:
            defaults[fname] = opts[0]
        else:
            defaults[fname] = ""

    with st.form(form_key, clear_on_submit=False):
        st.markdown("### Enter features")
        left, right = st.columns([2, 1])
        with left:
            for fi in feature_info:
                fname = fi["name"]
                ftype = fi.get("type", "numeric")
                options = fi.get("options")
                key = f"inp__{fname}"
                if fname.lower() == "seniorcitizen":
                    st.radio("Senior Citizen", options=[0, 1], format_func=lambda v: "Yes" if v == 1 else "No", index=0, horizontal=True, key=key)
                elif ftype == "numeric":
                    st.number_input(fname, value=0.0, format="%.4f", step=0.1, key=key)
                elif ftype == "categorical" and options is not None:
                    st.selectbox(fname, options=options, format_func=str, key=key)
                else:
                    st.text_input(fname, value="", key=key)
        with right:
            st.markdown("#### Tips")
            st.write("- Senior citizen: select Yes/No")
            st.write("- Use realistic MonthlyCharges / Tenure values")
            st.write("- Encoded categorical inputs will be applied before prediction")
            st.markdown("---")
            st.write("Loaded encoders:")
            st.write(", ".join(label_encoders.keys()) if label_encoders else "none found")
            st.markdown("---")
            st.markdown("Use the Clear button (right) to reset inputs")
        submitted = st.form_submit_button("Predict", help="Run prediction with selected model")

    # Clear button must be outside the form (st.button cannot be used inside st.form)
    if st.button("Clear form", help="Reset all input fields to defaults"):
        for fname, val in defaults.items():
            key = f"inp__{fname}"
            # reset session state key to default
            st.session_state[key] = val
        st.experimental_rerun()

    if submitted:
        # gather values from session_state keys
        inputs = {}
        for fi in feature_info:
            fname = fi["name"]
            key = f"inp__{fname}"
            inputs[fname] = st.session_state.get(key, defaults.get(fname))
        df_input = pd.DataFrame([inputs], columns=[f["name"] for f in feature_info])
        if label_encoders:
            df_input = apply_label_encoders_to_df(df_input, label_encoders)
        return df_input
    return None

def render_prediction_result(result, df_input):
    # smart label + color
    pred_raw = result.get("prediction")
    probs = result.get("probabilities")
    # determine churn label (try common conventions)
    churn_yes = False
    if isinstance(pred_raw, (int, np.integer)):
        churn_yes = bool(pred_raw == 1)
    elif isinstance(pred_raw, str):
        churn_yes = str(pred_raw).strip().lower() in ("yes", "true", "1", "churn")
    label = "Churn" if churn_yes else "No Churn"
    badge_class = "badge-red" if churn_yes else "badge-green"
    # top probability
    top_pct = None
    if probs:
        top = max(probs[0])
        top_pct = float(top) * 100.0
    # layout
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"<div class='badge {badge_class}'>{label}</div>", unsafe_allow_html=True)
        st.markdown(f"**Raw prediction:** `{pred_raw}`")
        if top_pct is not None:
            st.markdown(f"**Confidence:** {top_pct:.1f}%")
            # nice progress bar
            st.progress(min(int(top_pct), 100))
    with c2:
        st.markdown("#### Input (after encoding)")
        st.dataframe(df_input.T.rename(columns={0: "value"}))
        if probs is not None:
            st.markdown("#### Probabilities per class")
            prob_df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(len(probs[0]))])
            st.dataframe(prob_df.T)

# ---------------- Main ----------------
def main():
    st.set_page_config(page_title="Customer Churn Predictor", layout="wide", initial_sidebar_state="expanded")
    add_styles()
    render_header()

    models = load_models()
    if not models:
        st.error("No models loaded. Put your model files in src/models/ and restart.")
        return

    feature_names_from_file = load_feature_names()
    dataset_inf = infer_features_from_dataset(DATASET_PATH)
    label_encoders = load_label_encoders()

    # Model selector in sidebar
    model_choice = st.sidebar.selectbox("Choose model", options=list(models.keys()))
    model = models[model_choice]
    render_model_card(model_choice, model)
    render_dataset_preview(DATASET_PATH)

    expected_features = getattr(model, "n_features_in_", None)

    # decide feature list and types
    feature_info = None
    if feature_names_from_file:
        if expected_features is None or len(feature_names_from_file) == expected_features:
            feature_info = [{"name": n, "type": "numeric", "options": None} for n in feature_names_from_file]
    if feature_info is None and dataset_inf:
        if expected_features is None or len(dataset_inf) == expected_features:
            feature_info = dataset_inf
    if feature_info is None:
        n = expected_features or 3
        feature_info = [{"name": f"feature_{i}", "type": "numeric", "options": None} for i in range(n)]
        st.warning("Using generic feature names - update src/models/feature_names.json or provide a dataset header to infer features.")

    # enforce SeniorCitizen to be binary 0/1 (case-insensitive)
    for fi in feature_info:
        if fi["name"].lower() == "seniorcitizen":
            fi["type"] = "categorical"
            fi["options"] = [0, 1]

    # Render inputs form and handle prediction
    df_input = render_inputs_form(feature_info, label_encoders)

    if df_input is not None:
        with st.spinner("Running model..."):
            try:
                result = predict_model(model, df_input)
                st.success("Prediction complete")
                render_prediction_result(result, df_input)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()