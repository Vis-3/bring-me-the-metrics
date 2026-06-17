"""
Inference — scores all underground artists and outputs a ranked
breakout prediction table.

Loads the latest calibrated model from MLflow, applies the same
feature engineering as training, and produces:
  - breakout_probability: calibrated P(breakout)
  - shap_top_factor: the single feature pushing this artist toward breakout

Run:
    uv run python -m ml.predict
"""
import mlflow.sklearn
import pandas as pd
import numpy as np
import shap

from ml.data import load_features, engineer_features, ALL_FEATURES
from config import UNDERGROUND_LISTENER_CEILING

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def load_latest_model():
    """Load the most recent registered model version from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    versions = client.get_latest_versions("metal-breakout-predictor")
    if not versions:
        raise RuntimeError(
            "No registered model found. Run ml.train first."
        )
    latest = versions[-1]
    model_uri = f"models:/metal-breakout-predictor/{latest.version}"
    print(f"  Loading model version {latest.version} from {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def predict() -> pd.DataFrame:
    print("\n── Breakout Predictor: Inference ──\n")

    # ── Load all artists (not just training set) ─────────────────────────────
    raw = load_features()
    df, metadata = engineer_features(raw)

    # Score every artist — model outputs probability for all, not just underground
    X_all = df[ALL_FEATURES]

    model = load_latest_model()
    probs = model.predict_proba(X_all)[:, 1]

    # ── SHAP for individual explanations ─────────────────────────────────────
    # Extract the base XGBoost from the calibrated wrapper
    xgb_base = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(xgb_base)
    shap_values = explainer.shap_values(X_all)

    # Top SHAP factor per artist — the feature with the highest absolute contribution
    top_factor_idx = np.abs(shap_values).argmax(axis=1)
    top_factors = [ALL_FEATURES[i] for i in top_factor_idx]
    top_shap_values = shap_values[np.arange(len(shap_values)), top_factor_idx]

    # ── Build output table ───────────────────────────────────────────────────
    results = raw[["artist_name", "subgenre", "country", "formed_year",
                   "current_listeners", "total_albums"]].copy()
    results["breakout_probability"] = probs.round(3)
    results["top_breakout_factor"] = top_factors
    results["top_factor_shap"] = top_shap_values.round(3)

    # Current tier label
    results["current_tier"] = pd.cut(
        results["current_listeners"],
        bins=[0, 200_000, 1_000_000, float("inf")],
        labels=["Underground", "Rising", "Breakout"],
    )

    # ── Underground bands ranked by breakout probability ─────────────────────
    underground = (
        results[results["current_listeners"] < UNDERGROUND_LISTENER_CEILING]
        .sort_values("breakout_probability", ascending=False)
        .reset_index(drop=True)
    )
    underground.index += 1  # 1-based ranking

    print("\nTop 20 underground bands most likely to break out:\n")
    display_cols = [
        "artist_name", "subgenre", "current_listeners",
        "breakout_probability", "top_breakout_factor"
    ]
    print(underground[display_cols].head(20).to_string())

    # Save full predictions
    results.to_csv("breakout_predictions.csv", index=False)
    print("\n✓ Full predictions saved to breakout_predictions.csv")

    return underground


if __name__ == "__main__":
    predict()
