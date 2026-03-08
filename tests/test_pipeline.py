"""
Unit tests for Credit Scoring MLOps project.
Covers: transformers, feature pipeline, and API endpoints.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.features.transformers import (
    DEPENDENTS_IDX,
    FEATURE_COLS,
    INCOME_IDX,
    FeatureEngineer,
    RestoreColumnOrder,
    SelectiveScaler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
N_ROWS = 100
N_COLS = len(FEATURE_COLS)  # 10


@pytest.fixture
def sample_array() -> np.ndarray:
    """Clean array with no nulls, realistic values."""
    rng = np.random.default_rng(42)
    X = np.zeros((N_ROWS, N_COLS))
    X[:, FEATURE_COLS.index("RevolvingUtilizationOfUnsecuredLines")] = rng.uniform(0, 1, N_ROWS)
    X[:, FEATURE_COLS.index("age")] = rng.integers(20, 80, N_ROWS)
    X[:, FEATURE_COLS.index("NumberOfTime30-59DaysPastDueNotWorse")] = rng.integers(0, 5, N_ROWS)
    X[:, FEATURE_COLS.index("DebtRatio")] = rng.uniform(0, 1, N_ROWS)
    X[:, FEATURE_COLS.index("MonthlyIncome")] = rng.uniform(1000, 10000, N_ROWS)
    X[:, FEATURE_COLS.index("NumberOfOpenCreditLinesAndLoans")] = rng.integers(1, 15, N_ROWS)
    X[:, FEATURE_COLS.index("NumberOfTimes90DaysLate")] = rng.integers(0, 3, N_ROWS)
    X[:, FEATURE_COLS.index("NumberRealEstateLoansOrLines")] = rng.integers(0, 4, N_ROWS)
    X[:, FEATURE_COLS.index("NumberOfTime60-89DaysPastDueNotWorse")] = rng.integers(0, 3, N_ROWS)
    X[:, FEATURE_COLS.index("NumberOfDependents")] = rng.integers(0, 5, N_ROWS)
    return X.astype(float)


@pytest.fixture
def array_with_nulls(sample_array) -> np.ndarray:
    """Array with nulls in MonthlyIncome and NumberOfDependents."""
    X = sample_array.copy()
    null_rows = np.random.choice(N_ROWS, size=20, replace=False)
    X[null_rows, INCOME_IDX] = np.nan
    X[null_rows[:5], DEPENDENTS_IDX] = np.nan
    return X


@pytest.fixture
def full_pipeline() -> Pipeline:
    """Build and fit the full preprocessing pipeline."""
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
                total_cols=N_COLS,
            )),
            ("feature_engineer", FeatureEngineer()),
            ("scaler", SelectiveScaler(skip_last_n=1)),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# RestoreColumnOrder tests
# ---------------------------------------------------------------------------
class TestRestoreColumnOrder:

    def test_output_shape_unchanged(self, sample_array):
        transformer = RestoreColumnOrder(
            imputed_indices=[INCOME_IDX, DEPENDENTS_IDX],
            total_cols=N_COLS,
        )
        transformer.fit(sample_array)
        out = transformer.transform(sample_array)
        assert out.shape == sample_array.shape

    def test_column_values_preserved(self, sample_array):
        """After shuffle + restore, each column should match the original."""
        # Simulate ColumnTransformer output: imputed cols first, then remainder
        income_col = sample_array[:, INCOME_IDX]
        dep_col = sample_array[:, DEPENDENTS_IDX]
        remainder = [i for i in range(N_COLS) if i not in (INCOME_IDX, DEPENDENTS_IDX)]
        shuffled = np.hstack([
            sample_array[:, [INCOME_IDX, DEPENDENTS_IDX]],
            sample_array[:, remainder],
        ])

        transformer = RestoreColumnOrder(
            imputed_indices=[INCOME_IDX, DEPENDENTS_IDX],
            total_cols=N_COLS,
        )
        transformer.fit(shuffled)
        restored = transformer.transform(shuffled)

        np.testing.assert_array_equal(restored[:, INCOME_IDX], income_col)
        np.testing.assert_array_equal(restored[:, DEPENDENTS_IDX], dep_col)


# ---------------------------------------------------------------------------
# FeatureEngineer tests
# ---------------------------------------------------------------------------
class TestFeatureEngineer:

    def test_output_has_13_columns(self, sample_array):
        fe = FeatureEngineer()
        fe.fit(sample_array)
        out = fe.transform(sample_array)
        assert out.shape[1] == 13  # 10 original + 3 engineered

    def test_total_late_payments_correct(self, sample_array):
        fe = FeatureEngineer()
        out = fe.transform(sample_array)
        df_in = pd.DataFrame(sample_array, columns=FEATURE_COLS)
        expected = (
            df_in["NumberOfTime30-59DaysPastDueNotWorse"]
            + df_in["NumberOfTimes90DaysLate"]
            + df_in["NumberOfTime60-89DaysPastDueNotWorse"]
        ).values
        total_late_idx = 10  # first engineered column
        np.testing.assert_array_almost_equal(out[:, total_late_idx], expected)

    def test_log_monthly_income_no_negatives(self, sample_array):
        fe = FeatureEngineer()
        out = fe.transform(sample_array)
        log_income_idx = 11
        assert np.all(out[:, log_income_idx] >= 0)

    def test_age_bin_values_in_range(self, sample_array):
        fe = FeatureEngineer()
        out = fe.transform(sample_array)
        age_bin_idx = 12
        valid_values = {0.0, 1.0, 2.0, 3.0}
        unique_vals = set(out[:, age_bin_idx])
        assert unique_vals.issubset(valid_values)

    def test_transform_is_stateless(self, sample_array):
        """fit() should return self without changing state."""
        fe = FeatureEngineer()
        result = fe.fit(sample_array)
        assert result is fe


# ---------------------------------------------------------------------------
# SelectiveScaler tests
# ---------------------------------------------------------------------------
class TestSelectiveScaler:

    def test_output_shape_unchanged(self, sample_array):
        fe = FeatureEngineer()
        X_eng = fe.transform(sample_array)
        scaler = SelectiveScaler(skip_last_n=1)
        scaler.fit(X_eng)
        out = scaler.transform(X_eng)
        assert out.shape == X_eng.shape

    def test_last_column_not_scaled(self, sample_array):
        """AgeBin (last column) must be identical before and after scaling."""
        fe = FeatureEngineer()
        X_eng = fe.transform(sample_array)
        scaler = SelectiveScaler(skip_last_n=1)
        scaler.fit(X_eng)
        out = scaler.transform(X_eng)
        np.testing.assert_array_equal(out[:, -1], X_eng[:, -1])

    def test_continuous_columns_are_scaled(self, sample_array):
        """Scaled continuous columns should have mean ≈ 0 and std ≈ 1."""
        fe = FeatureEngineer()
        X_eng = fe.transform(sample_array)
        scaler = SelectiveScaler(skip_last_n=1)
        scaler.fit(X_eng)
        out = scaler.transform(X_eng)
        means = out[:, :-1].mean(axis=0)
        stds = out[:, :-1].std(axis=0)
        np.testing.assert_array_almost_equal(means, np.zeros_like(means), decimal=10)
        np.testing.assert_array_almost_equal(stds, np.ones_like(stds), decimal=10)

    def test_is_fitted_after_fit(self, sample_array):
        fe = FeatureEngineer()
        X_eng = fe.transform(sample_array)
        scaler = SelectiveScaler(skip_last_n=1)
        scaler.fit(X_eng)
        assert hasattr(scaler, "n_features_in_")


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------
class TestFullPipeline:

    def test_pipeline_handles_nulls(self, array_with_nulls, full_pipeline):
        """Pipeline must not raise on nulls in MonthlyIncome or NumberOfDependents."""
        out = full_pipeline.fit_transform(array_with_nulls)
        assert not np.isnan(out).any()

    def test_pipeline_output_shape(self, sample_array, full_pipeline):
        out = full_pipeline.fit_transform(sample_array)
        assert out.shape == (N_ROWS, 13)

    def test_transform_consistency(self, sample_array, full_pipeline):
        """fit_transform and fit+transform must produce identical results."""
        out1 = full_pipeline.fit_transform(sample_array)
        full_pipeline2 = full_pipeline  # already fitted
        out2 = full_pipeline2.transform(sample_array)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_no_data_leakage(self, sample_array, full_pipeline):
        """Pipeline fitted on train must transform test without re-fitting."""
        train, test = sample_array[:80], sample_array[80:]
        full_pipeline.fit(train)
        out = full_pipeline.transform(test)
        assert out.shape == (20, 13)


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "revolving_utilization": 0.75,
    "age": 45,
    "times_30_59_days_late": 0,
    "debt_ratio": 0.35,
    "monthly_income": 5000.0,
    "open_credit_lines": 8,
    "times_90_days_late": 0,
    "real_estate_loans": 1,
    "times_60_89_days_late": 0,
    "number_of_dependents": 2.0,
}


class TestAPI:

    @pytest.fixture
    def client(self):
        """
        Test client with model/pipeline mocked so tests don't need
        MLflow running or real files on disk.
        """
        import api.main as app_module
        from unittest.mock import MagicMock
        import numpy as np

        mock_pipeline = MagicMock()
        mock_pipeline.transform.return_value = np.zeros((1, 13))

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

        app_module.pipeline = mock_pipeline
        app_module.model = mock_model

        with TestClient(app_module.app) as c:
            yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "default_probability" in data
        assert "risk_tier" in data
        assert "model_version" in data

    def test_predict_probability_range(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        prob = response.json()["default_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_tier_values(self, client):
        response = client.post("/predict", json=VALID_PAYLOAD)
        tier = response.json()["risk_tier"]
        assert tier in {"LOW", "MEDIUM", "HIGH"}

    def test_predict_invalid_age(self, client):
        payload = {**VALID_PAYLOAD, "age": -5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_missing_field(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422