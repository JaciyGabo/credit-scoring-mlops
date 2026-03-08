"""
Data ingestion script for Credit Scoring MLOps project.
Reads raw CSV, validates schema, and saves processed snapshot.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

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
RAW_PATH = Path("data/raw/cs-training.csv")
PROCESSED_PATH = Path("data/processed/cs-training-clean.csv")

EXPECTED_COLUMNS = {
    "SeriousDlqin2yrs",
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
}

TARGET_COLUMN = "SeriousDlqin2yrs"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    logger.info("Validating schema...")
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)
    logger.info("Schema OK")


def log_data_summary(df: pd.DataFrame) -> None:
    total = len(df)
    target_rate = df[TARGET_COLUMN].mean()
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    logger.info(f"Total records     : {total:,}")
    logger.info(f"Default rate      : {target_rate:.2%}")
    logger.info(f"Columns with nulls: {len(cols_with_nulls)}")
    for col, count in cols_with_nulls.items():
        logger.info(f"  {col}: {count:,} nulls ({count/total:.1%})")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying basic cleaning...")
    initial_len = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_len - len(df):,} duplicate rows")

    # Remove rows with age <= 0
    invalid_age = (df["age"] <= 0).sum()
    df = df[df["age"] > 0]
    logger.info(f"Removed {invalid_age:,} rows with invalid age")

    # Cap extreme values in utilization (>1 is technically possible but >10 is noise)
    extreme_util = (df["RevolvingUtilizationOfUnsecuredLines"] > 10).sum()
    df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 10]
    logger.info(f"Removed {extreme_util:,} rows with extreme utilization")

    logger.info(f"Final dataset: {len(df):,} rows")
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Starting data ingestion ===")

    df = load_raw(RAW_PATH)
    validate_schema(df)
    log_data_summary(df)
    df = basic_cleaning(df)
    log_data_summary(df)
    save_processed(df, PROCESSED_PATH)

    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    main()