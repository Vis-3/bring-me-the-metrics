"""
XGBoost training pipeline with MLflow tracking and probability calibration.

Why calibrated classifier on top of XGBoost:
XGBoost's predict_proba gives a discrimination score, not a calibrated
probability. A score of 0.7 doesn't mean "70% chance of breaking out."
CalibratedClassifierCV with method='sigmoid' (Platt scaling) fits a
logistic regression on top of XGBoost's raw scores to produce true
probabilities. cv='prefit' means XGBoost is already trained — we
calibrate on the test set (acceptable for a portfolio; in production
use a held-out calibration set).

Run:
    uv run python -m ml.train

MLflow UI (start server first):
    mlflow server --host 127.0.0.1 --port 5000
    then open http://127.0.0.1:5000
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    brier_score_loss,
)
import shap

from ml.data import load_features, engineer_features, temporal_split, ALL_FEATURES

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "metal-breakout-predictor"


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    XGBoost scale_pos_weight = count(negative) / count(positive).
    Tells XGBoost to weight the minority class (breakout artists) more
    heavily during training, compensating for class imbalance.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weight = n_neg / n_pos
    print(f"  scale_pos_weight: {weight:.2f} ({n_neg} underground / {n_pos} breakout)")
    return weight


def train() -> None:
    print("\n── Phase 4: XGBoost Breakout Predictor ──\n")

    # ── Load and prepare data ────────────────────────────────────────────────
    raw = load_features()
    df, metadata = engineer_features(raw)
    X_train, y_train, X_test, y_test = temporal_split(df)

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgb-calibrated-v1"):

        # ── XGBoost hyperparameters ──────────────────────────────────────────
        # Conservative settings for a small dataset (873 rows):
        # - low max_depth prevents overfitting
        # - high n_estimators + low learning_rate = more robust generalisation
        # - subsample/colsample add stochasticity to reduce variance
        spw = compute_scale_pos_weight(y_train)

        xgb_params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": spw,
            "eval_metric": "aucpr",  # area under PR curve — better for imbalanced data
            "random_state": 42,
        }

        mlflow.log_params(xgb_params)
        mlflow.log_param("split_year", 2015)
        mlflow.log_param("rejection_region", "200K-999K listeners excluded")
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("features", ALL_FEATURES)

        # ── Train XGBoost ────────────────────────────────────────────────────
        print("\nTraining XGBoost...")
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # ── Calibrate probabilities ──────────────────────────────────────────
        # cv=5: cross-validated calibration on training data
        # method='sigmoid': Platt scaling, safer than isotonic for small data
        print("\nCalibrating probabilities (Platt scaling)...")
        calibrated = CalibratedClassifierCV(xgb, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────────────────────────────
        print("\nEvaluating...")
        y_prob_raw = xgb.predict_proba(X_test)[:, 1]
        y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
        y_pred = (y_prob_cal >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob_cal),
            "avg_precision": average_precision_score(y_test, y_prob_cal),
            "brier_score": brier_score_loss(y_test, y_prob_cal),
            # Brier score pre-calibration — shows calibration improved it
            "brier_score_uncalibrated": brier_score_loss(y_test, y_prob_raw),
        }

        mlflow.log_metrics(metrics)

        print(f"\n  ROC-AUC:              {metrics['roc_auc']:.3f}")
        print(f"  Avg Precision (PR):   {metrics['avg_precision']:.3f}")
        print(f"  Brier (calibrated):   {metrics['brier_score']:.3f}")
        print(f"  Brier (raw XGB):      {metrics['brier_score_uncalibrated']:.3f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Underground', 'Breakout'])}")

        # ── SHAP feature importance ──────────────────────────────────────────
        print("Computing SHAP values...")
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_train)

        # Global feature importance — mean absolute SHAP across all training artists
        shap_importance = pd.DataFrame({
            "feature": ALL_FEATURES,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        print("\nFeature importance (mean |SHAP|):")
        print(shap_importance.to_string(index=False))

        # Log SHAP importance as a CSV artifact
        shap_path = "shap_importance.csv"
        shap_importance.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)

        # ── Log model ────────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            calibrated,
            artifact_path="model",
            registered_model_name="metal-breakout-predictor",
            skops_trusted_types=[
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\n✓ Run logged: {MLFLOW_TRACKING_URI}/#/experiments/1/runs/{run_id}")


if __name__ == "__main__":
    train()
