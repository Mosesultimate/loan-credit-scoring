import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


RAW_DEFAULT = (
    Path(__file__).resolve().parent / "data" / "lending_club_credit_data.csv"
)
CLEAN_DEFAULT = (
    Path(__file__).resolve().parent / "data" / "lending_club_credit_data_cleaned.csv"
)


NUMERIC_COLUMNS: List[str] = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "credit_history_years",
    "loan_status",
]

STRING_COLUMNS: List[str] = [
    "grade",
    "emp_length",
    "home_ownership",
    "purpose",
    "addr_state",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean and wrangle the Lending Club credit data CSV "
            "for loan credit-scoring tasks."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DEFAULT),
        help="Path to raw CSV file "
        "(default: 'data/lending_club_credit_data.csv' next to this script).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CLEAN_DEFAULT),
        help="Path to save cleaned CSV "
        "(default: 'data/lending_club_credit_data_cleaned.csv').",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {path}")
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded dataframe with %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in STRING_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": pd.NA, "NA": pd.NA, "NaN": pd.NA, "null": pd.NA})
    return df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def report_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    total_rows = len(df)
    summary = (missing[missing > 0] / total_rows).sort_values(ascending=False)
    if summary.empty:
        logger.info("No missing values detected.")
        return
    logger.info("Columns with missing values (ratio):")
    for col, ratio in summary.items():
        logger.info("  %s: %.4f", col, ratio)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols_present = [c for c in NUMERIC_COLUMNS if c in df.columns]
    string_cols_present = [c for c in STRING_COLUMNS if c in df.columns]

    for col in numeric_cols_present:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in string_cols_present:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info("Removed %d duplicate rows.", removed)
    return df


def report_outliers(df: pd.DataFrame) -> None:
    numeric_cols_present = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if not numeric_cols_present:
        logger.info("No numeric columns available for outlier reporting.")
        return

    logger.info("Outlier check (using simple 1st/99th percentiles):")
    for col in numeric_cols_present:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.01)
        q99 = series.quantile(0.99)
        logger.info(
            "  %s: p1=%.3f, p99=%.3f, min=%.3f, max=%.3f",
            col,
            q1,
            q99,
            series.min(),
            series.max(),
        )


def save_cleaned_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving cleaned data to %s", path)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned dataframe with %d rows and %d columns", df.shape[0], df.shape[1])


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    logger.info("Input CSV path: %s", input_path)
    logger.info("Output CSV path: %s", output_path)

    df = load_data(input_path)

    # Tailored cleaning for this dataset
    df = clean_string_columns(df)
    df = coerce_numeric_columns(df)

    report_missing_values(df)
    df = impute_missing_values(df)
    df = remove_duplicates(df)
    report_outliers(df)

    save_cleaned_data(df, output_path)


if __name__ == "__main__":
    main()

