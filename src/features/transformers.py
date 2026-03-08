"""
Shared custom sklearn transformers.
Imported by both src/features/features.py and api/main.py so that
joblib can serialize and deserialize the pipeline correctly.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

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

INCOME_IDX = FEATURE_COLS.index("MonthlyIncome")
DEPENDENTS_IDX = FEATURE_COLS.index("NumberOfDependents")


class RestoreColumnOrder(BaseEstimator, TransformerMixin):
    def __init__(self, imputed_indices: list, total_cols: int):
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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df["TotalLatePayments"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
        )
        df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])
        df["AgeBin"] = pd.cut(
            df["age"],
            bins=[0, 30, 50, 70, 120],
            labels=[0, 1, 2, 3],
        ).astype(float)
        return df.values

    def get_feature_names_out(self, input_features=None):
        return FEATURE_COLS + ["TotalLatePayments", "LogMonthlyIncome", "AgeBin"]


class SelectiveScaler(BaseEstimator, TransformerMixin):
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