import os
import io
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib

# Use non-GUI backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "nova_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "nova_model.pkl")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(STATIC_IMG_DIR, exist_ok=True)


def ensure_dataset_exists(rows: int = 300) -> None:
    if os.path.exists(DATA_PATH):
        return
    rng = np.random.RandomState(int(time.time()) % (2**32 - 1))
    n = rows
    genders = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    age = rng.normal(35, 10, size=n).clip(18, 70).round().astype(int)
    earnings = (rng.lognormal(mean=10.5, sigma=0.5, size=n)).clip(20000, 300000)
    trips = rng.poisson(lam=180, size=n).clip(10, 600)
    rating = np.clip(rng.normal(4.6, 0.3, size=n), 1.0, 5.0)
    tenure_months = rng.randint(1, 120, size=n)
    cancellation_rate = np.clip(rng.beta(2, 10, size=n), 0, 0.9)
    city_tier = rng.choice([1, 2, 3], size=n, p=[0.3, 0.5, 0.2])
    past_defaults = rng.binomial(3, 0.15, size=n)

    # Base score combining features
    base_score = (
        (earnings / 10000)
        + (trips / 50)
        + (rating * 2)
        + (tenure_months / 24)
        - (cancellation_rate * 5)
        - (past_defaults * 2.0)
        - (city_tier - 1) * 0.5
    )

    # Add slight demographic bias for realism (to be mitigated later)
    gender_bias = np.where(genders == "Male", 0.5, -0.3)
    score = base_score + gender_bias + rng.normal(0, 1.0, size=n)
    prob_good = 1 / (1 + np.exp(-(score - 5)))
    good_credit = rng.binomial(1, prob_good)

    df = pd.DataFrame(
        {
            "gender": genders,
            "age": age,
            "earnings": np.round(earnings, 2),
            "trips": trips,
            "rating": np.round(rating, 2),
            "tenure_months": tenure_months,
            "cancellation_rate": np.round(cancellation_rate, 3),
            "city_tier": city_tier,
            "past_defaults": past_defaults,
            "good_credit": good_credit,
        }
    )
    # Ensure both classes exist; if imbalanced to a single class, rebalance by score threshold
    if df["good_credit"].nunique() < 2:
        thr = np.median(score)
        df["good_credit"] = (score >= thr).astype(int)
        # If still single class (degenerate), randomly flip a few labels
        if df["good_credit"].nunique() < 2:
            flip_idx = rng.choice(df.index, size=max(1, len(df)//10), replace=False)
            df.loc[flip_idx, "good_credit"] = 1 - df.loc[flip_idx, "good_credit"].values
    df.to_csv(DATA_PATH, index=False)


def load_dataset() -> pd.DataFrame:
    ensure_dataset_exists()
    return pd.read_csv(DATA_PATH)


def append_rows_to_dataset(rows: int) -> None:
    """Append new synthetic rows to the existing dataset file."""
    ensure_dataset_exists()
    if rows <= 0:
        return
    rng = np.random.RandomState(int(time.time()) % (2**32 - 1))
    n = rows
    genders = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    age = rng.normal(35, 10, size=n).clip(18, 70).round().astype(int)
    earnings = (rng.lognormal(mean=10.5, sigma=0.5, size=n)).clip(20000, 300000)
    trips = rng.poisson(lam=180, size=n).clip(10, 600)
    rating = np.clip(rng.normal(4.6, 0.3, size=n), 1.0, 5.0)
    tenure_months = rng.randint(1, 120, size=n)
    cancellation_rate = np.clip(rng.beta(2, 10, size=n), 0, 0.9)
    city_tier = rng.choice([1, 2, 3], size=n, p=[0.3, 0.5, 0.2])
    past_defaults = rng.binomial(3, 0.15, size=n)

    base_score = (
        (earnings / 10000)
        + (trips / 50)
        + (rating * 2)
        + (tenure_months / 24)
        - (cancellation_rate * 5)
        - (past_defaults * 2.0)
        - (city_tier - 1) * 0.5
    )
    gender_bias = np.where(genders == "Male", 0.5, -0.3)
    score = base_score + gender_bias + rng.normal(0, 1.0, size=n)
    prob_good = 1 / (1 + np.exp(-(score - 5)))
    good_credit = rng.binomial(1, prob_good)

    df_new = pd.DataFrame(
        {
            "gender": genders,
            "age": age,
            "earnings": np.round(earnings, 2),
            "trips": trips,
            "rating": np.round(rating, 2),
            "tenure_months": tenure_months,
            "cancellation_rate": np.round(cancellation_rate, 3),
            "city_tier": city_tier,
            "past_defaults": past_defaults,
            "good_credit": good_credit,
        }
    )

    df_existing = load_dataset()
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out.to_csv(DATA_PATH, index=False)


def save_figure(fig, prefix: str) -> str:
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{ts}.png"
    filepath = os.path.join(STATIC_IMG_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"images/{filename}"


def get_feature_target(df: pd.DataFrame):
    target_col = "good_credit"
    feature_cols = [
        "age",
        "earnings",
        "trips",
        "rating",
        "tenure_months",
        "cancellation_rate",
        "city_tier",
        "past_defaults",
        "gender",
    ]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values
    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = [
        col
        for col in X.columns
        if X[col].dtype != "object" and col != "good_credit"
    ]
    categorical_features = [col for col in X.columns if X[col].dtype == "object"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", "passthrough", categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_features, categorical_features


def train_models(df: pd.DataFrame):
    X, y = get_feature_target(df)
    # Guard against single-class targets or too-few samples per class for stratified split
    def has_adequate_classes(labels: np.ndarray) -> bool:
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            return False
        # Need at least 2 per class to allow a stratified train/test split
        return counts.min() >= 2

    if not has_adequate_classes(y):
        # Try regenerating dataset with more rows
        if os.path.exists(DATA_PATH):
            os.remove(DATA_PATH)
        ensure_dataset_exists(rows=max(800, len(df) * 2))
        df = load_dataset()
        X, y = get_feature_target(df)
        if not has_adequate_classes(y):
            # As a fallback, avoid stratify and manually ensure train has both classes
            # Shuffle and split 80/20
            idx = np.arange(len(y))
            rng = np.random.RandomState(42)
            rng.shuffle(idx)
            split = int(0.8 * len(y))
            train_idx, test_idx = idx[:split], idx[split:]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # If train still single-class, raise clearer error
            if len(np.unique(y_train)) < 2:
                raise ValueError("Dataset is degenerate (one class). Please regenerate from Data page with more rows (e.g., 800+).")
        else:
            # Proceed to normal split below
            pass
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # If variables exist from fallback manual split, reuse them; else do stratified split
    if 'X_train' in locals():
        pass
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Prepare encodings for gender
    def encode(Xdf: pd.DataFrame):
        Xenc = Xdf.copy()
        if "gender" in Xenc.columns:
            Xenc["gender"] = Xenc["gender"].map({"Male": 1, "Female": 0}).fillna(0)
        return Xenc

    X_train_enc = encode(X_train)
    X_test_enc = encode(X_test)

    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=200, n_jobs=None)
    lr.fit(X_train_enc, y_train)
    y_pred_lr = lr.predict(X_test_enc)
    y_proba_lr = lr.predict_proba(X_test_enc)[:, 1]
    metrics_lr = {
        "accuracy": float(accuracy_score(y_test, y_pred_lr)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_lr)),
        "classification_report": classification_report(
            y_test, y_pred_lr, output_dict=False
        ),
    }
    models["Logistic Regression"] = (lr, metrics_lr)

    # XGBoost (optional)
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=4,
        )
        xgb.fit(X_train_enc, y_train)
        y_pred_xgb = xgb.predict(X_test_enc)
        y_proba_xgb = xgb.predict_proba(X_test_enc)[:, 1]
        metrics_xgb = {
            "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
            "roc_auc": float(roc_auc_score(y_test, y_proba_xgb)),
            "classification_report": classification_report(
                y_test, y_pred_xgb, output_dict=False
            ),
        }
        models["XGBoost"] = (xgb, metrics_xgb)

    # Select best by ROC AUC
    best_name, (best_model, best_metrics) = sorted(
        models.items(), key=lambda kv: kv[1][1]["roc_auc"], reverse=True
    )[0]

    # Save pipeline: encoder is trivial mapping, so store mapping info
    bundle = {
        "model_name": best_name,
        "model": best_model,
        "feature_order": list(X_train_enc.columns),
        "metadata": {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "metrics": best_metrics,
        },
    }
    joblib.dump(bundle, MODEL_PATH)

    return models, best_name, best_metrics


def load_model():
    if not os.path.exists(MODEL_PATH):
        # Train a fresh model if missing
        df = load_dataset()
        _, _, _ = train_models(df)
    return joblib.load(MODEL_PATH)


def predict_single(input_dict: dict):
    bundle = load_model()
    model = bundle["model"]
    feature_order = bundle["feature_order"]

    # Build single-row DataFrame matching training encodings
    df_input = pd.DataFrame([input_dict])
    if "gender" in df_input.columns:
        df_input["gender"] = df_input["gender"].map({"Male": 1, "Female": 0}).fillna(0)
    df_input = df_input.reindex(columns=feature_order, fill_value=0)

    proba = float(model.predict_proba(df_input)[0, 1])
    pred = int(proba >= 0.5)
    return pred, proba


def plot_target_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.countplot(x="good_credit", data=df, ax=ax)
    ax.set_title("Target Distribution (good_credit)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return save_figure(fig, "target_dist")


def plot_correlations(df: pd.DataFrame) -> str:
    corr = df.drop(columns=["gender"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Feature Correlations")
    return save_figure(fig, "correlations")


def plot_feature_histograms(df: pd.DataFrame) -> str:
    numeric_cols = [
        c
        for c in df.columns
        if c not in ["good_credit", "gender"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    n = len(numeric_cols)
    cols = 3
    rows = int(np.ceil(n / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Histograms", y=1.02)
    return save_figure(fig, "histograms")


def fairness_analysis(df: pd.DataFrame):
    # Train a model and get predictions
    models, best_name, _ = train_models(df)
    model, _ = models[best_name]

    X, y = get_feature_target(df)

    X_enc = X.copy()
    X_enc["gender"] = X_enc["gender"].map({"Male": 1, "Female": 0}).fillna(0)

    proba = model.predict_proba(X_enc)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Selection rate and accuracy by gender
    groups = df["gender"].values
    mask_m = groups == "Male"
    mask_f = groups == "Female"

    def metrics_for(mask):
        sel = float(pred[mask].mean()) if mask.sum() else 0.0
        acc = float((pred[mask] == y[mask]).mean()) if mask.sum() else 0.0
        return sel, acc

    sel_m, acc_m = metrics_for(mask_m)
    sel_f, acc_f = metrics_for(mask_f)

    before = {
        "selection_rate": {"Male": sel_m, "Female": sel_f},
        "accuracy": {"Male": acc_m, "Female": acc_f},
        "overall_accuracy": float((pred == y).mean()),
    }

    # Simple mitigation: group thresholds to equalize selection rates to global rate
    global_sel = float(pred.mean())
    def find_threshold_for_group(mask):
        # Choose threshold so that selection rate ~= global_sel
        ps = proba[mask]
        if len(ps) == 0:
            return 0.5
        thr = np.quantile(ps, 1 - global_sel)
        return float(thr)

    thr_m = find_threshold_for_group(mask_m)
    thr_f = find_threshold_for_group(mask_f)

    pred_post = pred.copy()
    pred_post[mask_m] = (proba[mask_m] >= thr_m).astype(int)
    pred_post[mask_f] = (proba[mask_f] >= thr_f).astype(int)

    sel_m2, acc_m2 = metrics_for(mask_m)
    sel_f2, acc_f2 = metrics_for(mask_f)
    # recompute with post preds
    def metrics_for_post(mask):
        sel = float(pred_post[mask].mean()) if mask.sum() else 0.0
        acc = float((pred_post[mask] == y[mask]).mean()) if mask.sum() else 0.0
        return sel, acc
    sel_m2, acc_m2 = metrics_for_post(mask_m)
    sel_f2, acc_f2 = metrics_for_post(mask_f)

    after = {
        "selection_rate": {"Male": sel_m2, "Female": sel_f2},
        "accuracy": {"Male": acc_m2, "Female": acc_f2},
        "overall_accuracy": float((pred_post == y).mean()),
    }

    # Plot bar chart before vs after for selection rate
    fig, ax = plt.subplots(figsize=(6, 4))
    index = np.arange(2)
    width = 0.35
    before_vals = [before["selection_rate"]["Male"], before["selection_rate"]["Female"]]
    after_vals = [after["selection_rate"]["Male"], after["selection_rate"]["Female"]]
    ax.bar(index - width / 2, before_vals, width, label="Before")
    ax.bar(index + width / 2, after_vals, width, label="After")
    ax.set_xticks(index)
    ax.set_xticklabels(["Male", "Female"])
    ax.set_ylabel("Selection Rate")
    ax.set_title("Fairness Mitigation: Selection Rate by Gender")
    ax.legend()
    fairness_img = save_figure(fig, "fairness")

    result = {
        "before": before,
        "after": after,
        "chart_path": fairness_img,
        "thresholds": {"Male": thr_m, "Female": thr_f},
    }
    return result


def shap_plot_for_input(input_dict: dict) -> str:
    if not SHAP_AVAILABLE:
        # Fallback: bar plot of standardized feature values as proxy
        features = list(input_dict.keys())
        values = [input_dict[k] if isinstance(input_dict[k], (int, float)) else 0 for k in features]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(features, values)
        ax.set_title("Feature Contribution (proxy)")
        return save_figure(fig, "shap_proxy")

    bundle = load_model()
    model = bundle["model"]
    feature_order = bundle["feature_order"]

    x_row = pd.DataFrame([input_dict])
    if "gender" in x_row.columns:
        x_row["gender"] = x_row["gender"].map({"Male": 1, "Female": 0}).fillna(0)
    x_row = x_row.reindex(columns=feature_order, fill_value=0)

    try:
        if hasattr(model, "get_booster"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_row)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, np.zeros((10, x_row.shape[1])))
            shap_values = explainer.shap_values(x_row)
        else:
            # KernelExplainer as last resort (slow) with tiny background
            background = x_row.copy()
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(x_row, nsamples=50)[1]

        # Force plot style bar summary for single row
        fig = plt.figure(figsize=(6, 3.5))
        shap.plots.bar(shap.Explanation(values=np.array(shap_values).reshape(-1),
                                        base_values=None,
                                        data=x_row.values.reshape(-1),
                                        feature_names=list(x_row.columns)), show=False)
        return save_figure(fig, "shap")
    except Exception:
        # Robust fallback
        features = list(x_row.columns)
        values = x_row.iloc[0].values
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(features, values)
        ax.set_title("Feature Contribution (proxy)")
        return save_figure(fig, "shap_proxy")


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/data", methods=["GET", "POST"])
def data_page():
    if request.method == "POST":
        action = (request.form.get("action", "generate") or "generate").strip()
        rows_raw = (request.form.get("rows", "") or "").strip()
        try:
            rows = int(rows_raw)
        except Exception:
            rows = 300
        if action == "append":
            if rows < 1:
                rows = 100
            append_rows_to_dataset(rows)
        else:
            # Generate new synthetic dataset
            if rows < 100:
                rows = 300
            if os.path.exists(DATA_PATH):
                os.remove(DATA_PATH)
            ensure_dataset_exists(rows)
            # remove stale model to force retrain
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
        return redirect(url_for("data_page"))

    df = load_dataset()
    rows = df.to_dict(orient="records")
    columns = list(df.columns)
    return render_template("data.html", columns=columns, rows=rows)


@app.route("/data/download", methods=["GET"]) 
def data_download():
    ensure_dataset_exists()
    return send_file(
        DATA_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="nova_dataset.csv",
    )


@app.route("/analysis")
def analysis_page():
    df = load_dataset()
    img1 = plot_target_distribution(df)
    img2 = plot_correlations(df)
    img3 = plot_feature_histograms(df)
    return render_template("analysis.html", img1=img1, img2=img2, img3=img3)


@app.route("/models", methods=["GET", "POST"])
def models_page():
    metrics = None
    all_models = None
    best_name = None
    if request.method == "POST":
        df = load_dataset()
        models, best_name, best_metrics = train_models(df)
        metrics = best_metrics
        all_models = {name: m[1] for name, m in models.items()}
    else:
        if os.path.exists(MODEL_PATH):
            bundle = load_model()
            metrics = bundle.get("metadata", {}).get("metrics")
            best_name = bundle.get("model_name")
    return render_template("models.html", metrics=metrics, best_name=best_name, all_models=all_models)


@app.route("/fairness", methods=["GET", "POST"])
def fairness_page():
    if request.method == "POST":
        df = load_dataset()
        result = fairness_analysis(df)
        return render_template(
            "fairness.html",
            before=result["before"],
            after=result["after"],
            chart_path=result["chart_path"],
            thresholds=result["thresholds"],
        )
    return render_template("fairness.html")


@app.route("/demo", methods=["GET", "POST"])
def demo_page():
    result = None
    shap_img = None
    # Defaults for initial render and to persist values after submit
    default_values = {
        "age": "30",
        "gender": "Male",
        "earnings": "50000",
        "trips": "200",
        "rating": "4.5",
        "tenure_months": "12",
        "cancellation_rate": "0.05",
        "city_tier": "2",
        "past_defaults": "0",
    }

    form_values = default_values.copy()

    if request.method == "POST":
        form = request.form
        # Keep string values for repopulating the form
        form_values = {
            "age": (form.get("age") or default_values["age"]),
            "gender": (form.get("gender") or default_values["gender"]),
            "earnings": (form.get("earnings") or default_values["earnings"]),
            "trips": (form.get("trips") or default_values["trips"]),
            "rating": (form.get("rating") or default_values["rating"]),
            "tenure_months": (form.get("tenure_months") or default_values["tenure_months"]),
            "cancellation_rate": (form.get("cancellation_rate") or default_values["cancellation_rate"]),
            "city_tier": (form.get("city_tier") or default_values["city_tier"]),
            "past_defaults": (form.get("past_defaults") or default_values["past_defaults"]),
        }

        # Convert to numeric types for prediction
        input_payload = {
            "age": float(form_values["age"]),
            "earnings": float(form_values["earnings"]),
            "trips": float(form_values["trips"]),
            "rating": float(form_values["rating"]),
            "tenure_months": float(form_values["tenure_months"]),
            "cancellation_rate": float(form_values["cancellation_rate"]),
            "city_tier": float(form_values["city_tier"]),
            "past_defaults": float(form_values["past_defaults"]),
            "gender": form_values["gender"],
        }
        pred, proba = predict_single(input_payload)
        shap_img = shap_plot_for_input(input_payload)
        result = {
            "prediction": "Good Credit" if pred == 1 else "Bad Credit",
            "probability": round(proba, 4),
        }
    return render_template("demo.html", result=result, shap_img=shap_img, form_values=form_values)


@app.route("/api/preview")
def api_preview():
    df = load_dataset()
    return jsonify({"columns": list(df.columns), "rows": df.head(10).to_dict(orient="records")})


@app.route("/api/train", methods=["POST"]) 
def api_train():
    df = load_dataset()
    _, best_name, best_metrics = train_models(df)
    return jsonify({"best_model": best_name, "metrics": best_metrics})


@app.route("/api/fairness", methods=["POST"]) 
def api_fairness():
    df = load_dataset()
    result = fairness_analysis(df)
    return jsonify(result)


@app.route("/api/predict", methods=["POST"]) 
def api_predict():
    payload = request.get_json(force=True)
    pred, proba = predict_single(payload)
    return jsonify({"prediction": int(pred), "probability": float(proba)})


if __name__ == "__main__":
    ensure_dataset_exists()
    # Disable the reloader to avoid signal issues in non-main threads (e.g., Streamlit)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


