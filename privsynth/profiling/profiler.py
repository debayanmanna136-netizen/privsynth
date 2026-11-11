import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    profile = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "numerical_cols": list(df.select_dtypes(include=['number']).columns),
        "categorical_cols": list(df.select_dtypes(exclude=['number']).columns),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_col": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "unique_values_count": {col: df[col].nunique() for col in df.columns}
    }
    return profile
