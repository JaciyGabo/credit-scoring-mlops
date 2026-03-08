"""
Feature engineering script for Credit Scoring MLOps project.
All transformations live inside a single sklearn Pipeline to avoid
data leakage and guarantee consistency between training and inference.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
PROCESSED_PATH = Path("data/processed/cs-training-clean.csv")
FEATURES_DIR = Path("data/processed")
PIPELINE_PATH = Path("data/processed/preprocessing_pipeline.joblib")

TARGET_COLUMN = "SeriousDlqin2yrs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

INCOME_IDX = FEATURE_COLS.index("MonthlyIncome")       # 4
DEPENDENTS_IDX = FEATURE_COLS.index("NumberOfDependents")  # 9


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------
class RestoreColumnOrder(BaseEstimator, TransformerMixin):
    """
    ColumnTransformer puts named-transform columns first, then remainder.
    This restores the original FEATURE_COLS order so FeatureEngineer
    can reference columns by their expected position.
    """

    def __init__(self, imputed_indices: list[int], total_cols: int):
        self.imputed_indices = imputed_indices
        self.total_cols = total_cols

    def fit(self, X, y=None):
        remainder = [i for i in range(self.total_cols) if i not in self.imputed_indices]
        shuffled = self.imputed_indices + remainder
        self.restore_order_ = [shuffled.index(i) for i in range(self.total_cols)]
        return self

    def transform(self, X):
        return X[:, self.restore_order_]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived features. Runs AFTER imputation so MonthlyIncome
    has no nulls — LogMonthlyIncome and the original imputed value
    are now fully consistent.
    """

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        df = pd.DataFrame(X, columns=FEATURE_COLS)

        # Sum of all delinquency buckets
        df["TotalLatePayments"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
        )

        # Log transform — consistent with imputed MonthlyIncome (no more fillna(0))
        df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])

        # Ordinal age bin — NOT scaled downstream
        df["AgeBin"] = pd.cut(
            df["age"],
            bins=[0, 30, 50, 70, 120],
            labels=[0, 1, 2, 3],
        ).astype(float)

        return df.values

    def get_feature_names_out(self, input_features=None):
        return FEATURE_COLS + ["TotalLatePayments", "LogMonthlyIncome", "AgeBin"]


class SelectiveScaler(BaseEstimator, TransformerMixin):
    """
    Scales only continuous features.
    AgeBin (ordinal, last column) is passed through unchanged to preserve
    interpretability in linear models and avoid distorting tree splits.
    """

    def __init__(self, skip_last_n: int = 1):
        self.skip_last_n = skip_last_n
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[:, : -self.skip_last_n])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy().astype(float)
        X[:, : -self.skip_last_n] = self.scaler.transform(X[:, : -self.skip_last_n])
        return X


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline() -> tuple[Pipeline, list[str]]:
    imputer = ColumnTransformer(
        transformers=[
            ("median_income", SimpleImputer(strategy="median"), [INCOME_IDX]),
            ("median_dependents", SimpleImputer(strategy="median"), [DEPENDENTS_IDX]),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("restore_order", RestoreColumnOrder(
                imputed_indices=[INCOME_IDX, DEPENDENTS_IDX],
                total_cols=len(FEATURE_COLS),
            )),
            ("feature_engineer", FeatureEngineer()),
            ("scaler", SelectiveScaler(skip_last_n=1)),  # skip AgeBin
        ]
    )

    all_features = FEATURE_COLS + ["TotalLatePayments", "LogMonthlyIncome", "AgeBin"]
    return pipeline, all_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Starting feature engineering ===")

    df = pd.read_csv(PROCESSED_PATH)
    logger.info(f"Loaded {len(df):,} rows")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COLUMN].values

    # Split BEFORE fitting — guarantees no data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"Train default rate: {y_train.mean():.2%}")
    logger.info(f"Test default rate : {y_test.mean():.2%}")

    # Fit on TRAIN only — no leakage from test set
    pipeline, feature_names = build_pipeline()
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)
    logger.info(f"Processed shape — train: {X_train_proc.shape} | test: {X_test_proc.shape}")

    # Save CSVs
    pd.DataFrame(X_train_proc, columns=feature_names).assign(
        **{TARGET_COLUMN: y_train}
    ).to_csv(FEATURES_DIR / "train.csv", index=False)

    pd.DataFrame(X_test_proc, columns=feature_names).assign(
        **{TARGET_COLUMN: y_test}
    ).to_csv(FEATURES_DIR / "test.csv", index=False)

    logger.info(f"Saved train.csv and test.csv to {FEATURES_DIR}")

    # One .joblib handles everything at inference time
    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info(f"Pipeline saved to {PIPELINE_PATH}")

    logger.info("=== Feature engineering complete ===")


if __name__ == "__main__":
    main()