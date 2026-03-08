"""
Training script for Credit Scoring MLOps project.
Trains 3 models, tracks experiments with MLflow, and registers the best one.
"""

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
TARGET_COLUMN = "SeriousDlqin2yrs"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "credit-scoring"
REGISTERED_MODEL_NAME = "credit-scoring-model"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=14,  # ~93/7 class ratio
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    ),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    # Gini coefficient: 2*AUC - 1
    gini = 2 * auc - 1

    # KS statistic
    from scipy.stats import ks_2samp
    scores_pos = y_prob[y_true == 1]
    scores_neg = y_prob[y_true == 0]
    ks_stat, _ = ks_2samp(scores_pos, scores_neg)

    return {
        "auc_roc": round(auc, 4),
        "average_precision": round(ap, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_and_track(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, str]:
    """Train one model, log everything to MLflow. Returns (auc, run_id)."""

    with mlflow.start_run(run_name=name) as run:
        logger.info(f"Training {name}...")

        # Log hyperparameters
        mlflow.log_params(model.get_params())

        # Train
        model.fit(X_train, y_train)

        # Predict probabilities
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

        # Metrics
        train_metrics = {f"train_{k}": v for k, v in compute_metrics(y_train, y_prob_train).items()}
        test_metrics = {f"test_{k}": v for k, v in compute_metrics(y_test, y_prob_test).items()}
        all_metrics = {**train_metrics, **test_metrics}

        mlflow.log_metrics(all_metrics)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log classification report as artifact
        y_pred = (y_prob_test >= 0.5).astype(int)
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(f"Model: {name}\n\n{report}")
        mlflow.log_artifact("classification_report.txt")

        auc = test_metrics["test_auc_roc"]
        run_id = run.info.run_id

        logger.info(
            f"{name} — AUC: {auc:.4f} | Gini: {test_metrics['test_gini']:.4f} | "
            f"KS: {test_metrics['test_ks_statistic']:.4f}"
        )

    return auc, run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Starting training ===")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN]).values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df.drop(columns=[TARGET_COLUMN]).values
    y_test = test_df[TARGET_COLUMN].values

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")

    # Train all models
    results = {}
    for name, model in MODELS.items():
        auc, run_id = train_and_track(name, model, X_train, y_train, X_test, y_test)
        results[name] = {"auc": auc, "run_id": run_id}

    # Find best model
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_run_id = results[best_name]["run_id"]
    best_auc = results[best_name]["auc"]

    logger.info(f"\n{'='*50}")
    logger.info(f"Best model: {best_name} | AUC: {best_auc:.4f}")
    logger.info(f"Run ID: {best_run_id}")

    # Register best model in MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    logger.info(f"Registered model version: {mv.version}")

    # Summary table
    logger.info("\n--- Results summary ---")
    for name, r in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
        logger.info(f"  {name:<25} AUC: {r['auc']:.4f}")

    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()