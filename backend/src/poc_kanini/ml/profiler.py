import numpy as np
import pandas as pd
from poc_kanini.ml.models import DatasetProfile


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """Analyze a pandas DataFrame and extract structured profiling metadata.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A DatasetProfile containing structural characteristics and statistics.
    """
    row_count = len(df)
    column_count = len(df.columns)
    columns = [str(col) for col in df.columns]
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    missing_counts = {str(col): int(count) for col, count in df.isnull().sum().items()}

    # Identify numeric and datetime columns, and classify remaining as categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [str(col) for col in numeric_cols]

    datetime_columns = []
    # Check pre-existing datetime columns
    dt_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    datetime_columns.extend([str(col) for col in dt_cols])

    for col in df.columns:
        col_str = str(col)
        if col_str in numeric_columns or col_str in datetime_columns:
            continue

        non_null_series = df[col].dropna()
        if non_null_series.empty:
            continue

        first_val = str(non_null_series.iloc[0]).strip()
        # Avoid checking if value is pure numeric string
        if first_val.replace('.', '', 1).isdigit():
            continue

        try:
            # Parse a sample of the series to detect datetimes.
            # Use warnings filter to suppress pandas infer_datetime_format warning.
            import warnings
            sample = non_null_series.head(1000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                coerced = pd.to_datetime(sample, errors='coerce', dayfirst=False)
            valid_ratio = coerced.notnull().sum() / len(sample)
            if valid_ratio > 0.8:
                datetime_columns.append(col_str)
                dtypes[col_str] = "datetime64[ns]"
        except Exception:
            pass

    categorical_columns = [
        str(col) for col in df.columns
        if str(col) not in numeric_columns and str(col) not in datetime_columns
    ]


    # Calculate summary statistics for numeric columns
    summary_stats: dict[str, dict[str, float]] = {}
    if row_count > 0:
        for col in numeric_columns:
            desc = df[col].describe()
            # Handle empty/all-null numeric columns where statistics are NaN
            stats = {}
            for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                val = desc.get(stat_name)
                # Cast to float, fallback to 0.0 if NaN/missing
                if val is None or pd.isna(val):
                    stats[stat_name] = 0.0
                else:
                    stats[stat_name] = float(val)
            summary_stats[col] = stats
    else:
        # Zero-row default stats
        for col in numeric_columns:
            summary_stats[col] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "25%": 0.0,
                "50%": 0.0,
                "75%": 0.0,
                "max": 0.0,
            }

    return DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        dtypes=dtypes,
        missing_counts=missing_counts,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        summary_statistics=summary_stats,
    )
